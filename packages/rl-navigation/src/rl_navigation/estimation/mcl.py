"""Localization on a known map with a particle filter: Monte Carlo localization (MCL).

The filter follows Augmented_MCL of Thrun, Burgard and Fox, *Probabilistic Robotics* (Table 8.3),
with low-variance resampling (Table 4.4) and a cut-down beam model (Table 6.1). Unlike the EKF
it holds many pose hypotheses, so it can localize from anywhere on the map (global localization)
and find the robot again after it is carried elsewhere (a kidnapping).

- **Motion.** Each particle draws its own wheel travel from the encoders' noise model
  (:class:`~rl_navigation.sim.WheelOdometry`, variance ``k |travel|`` per wheel) and moves on the
  exact unicycle arc. A particle whose move would end within ``robot_radius`` of an obstacle stays
  where it was, as the simulated robot does when it collides.
- **Measurement.** A reading is used when it carries a range (:func:`usable_readings`: not a
  dropout, a NaN, no echo or a reading clipped at ``min_range``), at most ``max_beams`` of them
  spread evenly over the scan. Each used beam's likelihood is
  ``(1 - z_rand) N(z; z_hat, sigma^2) + z_rand / (max_range - min_range)`` with
  ``sigma^2 = noise_std^2 + model_std^2`` and ``z_hat`` the expected range capped at
  ``max_range``. ``model_std`` widens the sensor's own noise to the spacing of the particles: with
  ``sigma`` at the sensor's centimetre, a few thousand particles over a room rarely land close
  enough to the truth to be weighted, and the filter locks onto a wrong pose. Thrun's short and
  max-range terms are left out: the simulator has no unmapped obstacles, and a max-range reading is
  a dropout here.
- **Recovery.** ``w_avg``, the weighted mean over particles of each particle's per-beam
  geometric-mean likelihood, feeds a slow and a fast average. When the fast one falls below the
  slow one, each particle is redrawn uniformly over free space with probability
  ``p_inject = 1 - w_fast / w_slow``, but only when the *expected* count ``p_inject * P`` is at
  least 1. The draws are Bernoulli, so an update can resample with 0 particles injected.
  Averaging per beam keeps ``w_avg`` free of underflow and of the number of beams that happened
  to drop out.
- **Deviation from Table 8.3:** ``w_avg`` leaves out the particles injected on the previous scan.
  Counted in, as Thrun counts them, the random particles drag ``w_avg`` down, which holds
  ``w_fast`` under ``w_slow`` and injects more of them: a loop that runs for ``w_slow``'s memory
  (``1 / alpha_slow``, about 1000 scans) after the robot is found (in one measurement, 1892
  injections in 100 scans of 300 particles, against 282). Resetting the averages after an
  injection, as ROS AMCL does, ends the loop but stops injecting while the filter is still lost
  (kidnapping recovery fell from 9 to 5 of 10 runs). What remains: after a disturbance,
  ``w_slow`` remembers the better likelihood from before it, so about 1 % of the particles are
  still injected per scan, and the filter resamples on every scan, until ``w_slow`` catches up.
  Measured in a 5 x 5 m room with 300 particles after a 10 cm bump of a still robot: 504, 282,
  216, 231, 199, 158, 172 and 111 injections in successive 100-scan windows, none from scan 800
  on, and resampling on every scan until about scan 700, about 70 s at 10 Hz.
- **Resampling** happens when the effective sample size falls below
  ``resample_threshold * num_particles`` or particles are injected, and is followed by a small
  Gaussian jitter (``roughening``), which keeps a still robot's particles distinct.
- **Estimate.** :func:`heaviest_cluster`: the pose and covariance of the heaviest cluster of
  particles, and the cluster's share of the weight. The covariance is conservative: ``model_std``
  widens the likelihood, so the cloud, and with it the covariance, is wider than the pose error
  (``EST-2`` in TESTING.md: mean tracking NEES 0.57 to 0.94 over all steps of seeds 101 to 150,
  both sensors, their own and 30 % dropout, against 3 for a consistent filter).
  ``MclSettings.covariance_scale`` corrects it per axis, with fitted values per sensor; the default
  leaves it unscaled. A symmetric arena could give clusters of similar weight, but resampling
  collapses all but one within a few scans, so the share then ends near 1 whichever mirror pose
  survived.

Frames and units are the EKF's: the pose ``(x, y, theta)`` is the robot's centre in the map's
frame, metres and radians, ``theta`` in ``[-pi, pi)``, and a scan passed to
:meth:`MclLocalizer.update` is taken at the pose reached by the last :meth:`MclLocalizer.predict`.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.kalman import symmetrized
from rl_navigation.estimation.maps import ParticleMap, RangeMap, usable_readings
from rl_navigation.estimation.poses import (
    checked_axis_values,
    checked_start,
    checked_wheel_travel,
    pose_difference,
)
from rl_navigation.sim.geometry import FloatArray
from rl_navigation.sim.kinematics import unicycle_step, wrap_angle
from rl_navigation.sim.odometry import WheelOdometry

# The cluster estimate: a weighted histogram of positions in CLUSTER_BIN-metre cells, each cell
# summed with its CLUSTER_NEIGHBOURHOOD x CLUSTER_NEIGHBOURHOOD neighbours; in the heaviest
# neighbourhood, the heaviest of HEADING_BINS heading bins (each with its two neighbours); then
# MEAN_SHIFT_STEPS steps of a weighted mean over particles within CLUSTER_RADIUS metres and
# CLUSTER_HEADING_RADIUS radians.
CLUSTER_BIN = 0.25
CLUSTER_NEIGHBOURHOOD = 3
HEADING_BINS = 16
MEAN_SHIFT_STEPS = 3
CLUSTER_RADIUS = 0.5
CLUSTER_HEADING_RADIUS = 0.5

# Batches of candidate poses that sample_free_poses draws before it gives up, and redraws of the
# particles of a Gaussian start that are not in free space.
FREE_SAMPLE_ATTEMPTS = 100
GAUSSIAN_START_ATTEMPTS = 100


@dataclass(frozen=True)
class MclSettings:
    """The particle filter's settings.

    Args:
        num_particles: Particles ``P``.
        max_beams: Most beams used per scan, spread evenly over the usable ones.
        model_std: Standard deviation in metres added in quadrature to the sensor's noise in the
            beam likelihood (> 0).
        z_rand: Weight of the uniform term of the beam likelihood, in ``(0, 1)``: it keeps every
            log-likelihood finite and bounds the damage of one outlier.
        robot_radius: A particle must be more than this from every obstacle, in metres.
        resample_threshold: Resample when the effective sample size falls below this share of
            ``num_particles``, in ``(0, 1]``.
        alpha_slow: Rate of the slow average of the likelihood, ``0 < alpha_slow < alpha_fast``.
        alpha_fast: Rate of the fast average, at most 1.
        roughening: Standard deviations ``(x, y, theta)`` of the jitter after resampling, metres and
            radians, each > 0; also added to the reported covariance.
        covariance_scale: Per-axis factors ``d = (d_x, d_y, d_theta)``, each > 0, applied to the
            reported covariance as ``D^1/2 C D^1/2`` with ``D = diag(d)``: each variance times its
            factor, each covariance times the root of both. The cluster's covariance overstates the
            error (mean NEES 0.57 to 0.94 over all steps; a consistent estimate gives 3), and by a
            different amount per axis, so one scalar cannot correct it. The default ``(1, 1, 1)``
            reports the cluster's covariance unchanged (conservative, and one default would mislead
            the other sensor). Fitted on seeds 1 to 20 (tracking, first 50 steps skipped, ``d_i`` =
            mean ``e_i^2 / P_ii``): lidar ``(0.130, 0.124, 0.308)``, HC-SR04 ``(0.174, 0.180,
            0.525)``. Judged on seeds 101 to 150 (``EST-2`` in TESTING.md): per-axis ``e_i^2 / (d_i
            P_ii)`` within [0.5, 2.0] on all four rows (0.92 to 1.06); the share of steps whose
            50-run ANEES lies in [2.36, 3.72] passes the 85 % gate on three rows (lidar 93.1 % and
            91.3 %, HC-SR04 with 30 % dropout 87.1 %) and fails on HC-SR04 with its own dropout
            (82.5 %). A fitted scale holds only for the settings it was fitted with (2000 particles,
            16 beams, ``model_std`` 0.15) and only while the filter is localized: it means nothing
            while the filter is lost, and MCL's NEES has a heavier tail than chi-square(3).

    Raises:
        ValueError: A setting out of range.
    """

    num_particles: int = 2000
    max_beams: int = 16
    model_std: float = 0.15
    z_rand: float = 0.1
    robot_radius: float = 0.1
    resample_threshold: float = 0.5
    alpha_slow: float = 0.001
    alpha_fast: float = 0.1
    roughening: tuple[float, float, float] = (0.005, 0.005, 0.005)
    covariance_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        for name in ("num_particles", "max_beams"):
            value = getattr(self, name)
            if isinstance(value, bool) or not (isinstance(value, int | np.integer) and value >= 1):
                raise ValueError(
                    f"MclSettings {name} must be an integer of at least 1, got {value}."
                )
        checks = [
            ("model_std", self.model_std > 0.0, "positive"),
            ("z_rand", 0.0 < self.z_rand < 1.0, "in (0, 1)"),
            ("robot_radius", self.robot_radius >= 0.0, "at least 0"),
            ("resample_threshold", 0.0 < self.resample_threshold <= 1.0, "in (0, 1]"),
            ("alpha_slow", 0.0 < self.alpha_slow < self.alpha_fast, "in (0, alpha_fast)"),
            ("alpha_fast", self.alpha_fast <= 1.0, "at most 1"),
        ]
        for name, valid, wanted in checks:
            if not valid:
                raise ValueError(f"MclSettings {name} must be {wanted}, got {getattr(self, name)}.")
        for name in ("roughening", "covariance_scale"):
            checked_axis_values(getattr(self, name), "MclSettings", name)


@dataclass(frozen=True)
class MclUpdate:
    """What one :meth:`MclLocalizer.update` did.

    Attributes:
        used: ``(N,)`` beams in the likelihood. Unlike the EKF's, they weigh hypotheses rather
            than correct one pose, and there are at most ``max_beams`` of them.
        effective_size: Effective sample size of the weights after this scan, before any
            resampling (the previous one when no beam was usable).
        resampled: Whether the particles were resampled.
        injected: Particles redrawn uniformly over free space.
    """

    used: NDArray[np.bool_]
    effective_size: float
    resampled: bool
    injected: int


def _normalized(weights: ArrayLike) -> FloatArray:
    weights = np.asarray(weights, dtype=np.float64)
    normalized: FloatArray = weights / weights.sum()
    return normalized


def _settled_weights(weights: FloatArray, fresh: NDArray[np.bool_]) -> FloatArray:
    """The weights with the ``fresh`` (just injected) particles left out, renormalized; all of
    them when every particle is fresh."""
    settled = np.where(fresh, 0.0, weights)
    total = settled.sum()
    return settled / total if total > 0.0 else weights


def low_variance_resample(weights: ArrayLike, u0: float) -> NDArray[np.intp]:
    """Systematic resampling: ``P`` indices drawn at the evenly spaced positions ``(k + u0) / P``.

    The cumulative weights are divided by their total, so a trailing particle of zero weight has
    no sliver left by rounding to be drawn in.

    Args:
        weights: ``(P,)`` non-negative weights with a positive sum, normalized or not.
        u0: The single random offset, in ``[0, 1)`` and in units of ``1 / P``.

    Returns:
        ``(P,)`` indices into the particles.
    """
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64))
    cumulative /= cumulative[-1]
    count = len(cumulative)
    # Below 1 even when (P - 1 + u0) / P rounds up, so no position falls past the last edge.
    positions = np.minimum((np.arange(count) + u0) / count, np.nextafter(1.0, 0.0))
    return np.searchsorted(cumulative, positions, side="right")


def effective_sample_size(weights: ArrayLike) -> float:
    """``1 / sum(w^2)`` of the normalized weights: ``P`` when they are equal, 1 when one has all."""
    normalized = _normalized(weights)
    return float(1.0 / (normalized @ normalized))


def recovery_update(
    w_slow: float, w_fast: float, w_avg: float, alpha_slow: float, alpha_fast: float
) -> tuple[float, float, float]:
    """One step of Augmented_MCL's averages of the measurement likelihood.

    ``w_slow += alpha_slow (w_avg - w_slow)``, ``w_fast += alpha_fast (w_avg - w_fast)``, and the
    share of particles to redraw uniformly, ``max(0, 1 - w_fast / w_slow)``.

    Returns:
        ``(w_slow, w_fast, p_inject)``.
    """
    w_slow += alpha_slow * (w_avg - w_slow)
    w_fast += alpha_fast * (w_avg - w_fast)
    p_inject = max(0.0, 1.0 - w_fast / w_slow) if w_slow > 0.0 else 0.0
    return w_slow, w_fast, p_inject


def measurement_log_likelihood(
    readings: ArrayLike,
    expected: ArrayLike,
    range_map: RangeMap,
    model_std: float,
    z_rand: float,
) -> FloatArray:
    """Log-likelihood of ``M`` readings from each of ``P`` poses, summed over the beams.

    Per beam ``log((1 - z_rand) N(z; z_hat, sigma^2) + z_rand / (max_range - min_range))``, with
    ``sigma^2 = noise_std^2 + model_std^2`` and ``z_hat`` the expected range capped at
    ``max_range``, so a pose whose beam meets nothing (``inf``) reads as far as the sensor sees.

    Args:
        readings: ``(M,)`` usable readings.
        expected: ``(P, M)`` expected ranges from each pose.
        range_map: The map, for the sensor's limits and noise.
        model_std: Extra standard deviation in metres.
        z_rand: Weight of the uniform term.

    Returns:
        ``(P,)`` log-likelihoods, all finite.
    """
    capped = np.minimum(np.asarray(expected, dtype=np.float64), range_map.max_range)
    sigma = math.hypot(range_map.noise_std, model_std)
    z = (np.asarray(readings, dtype=np.float64) - capped) / sigma
    hit = (1.0 - z_rand) * np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma)
    uniform = z_rand / (range_map.max_range - range_map.min_range)
    log_likelihood: FloatArray = np.log(hit + uniform).sum(axis=-1)
    return log_likelihood


def _within(
    particles: FloatArray, mean: FloatArray, radius: float, heading_radius: float
) -> NDArray[np.bool_]:
    difference = pose_difference(particles, mean)
    within: NDArray[np.bool_] = (np.hypot(difference[:, 0], difference[:, 1]) < radius) & (
        np.abs(difference[:, 2]) < heading_radius
    )
    return within


def _weighted_mean(particles: FloatArray, weights: FloatArray, reference: FloatArray) -> FloatArray:
    """Weighted mean of poses, averaging headings as wrapped differences from ``reference``."""
    mean = reference + weights @ pose_difference(particles, reference) / weights.sum()
    mean[2] = wrap_angle(mean[2])
    return mean


def heaviest_cluster(
    particles: ArrayLike, weights: ArrayLike, roughening: ArrayLike
) -> tuple[FloatArray, FloatArray, float]:
    """The pose and covariance of the heaviest cluster of particles, and its share of the weight.

    Positions are binned in :data:`CLUSTER_BIN` cells and each cell summed with its neighbours;
    in the heaviest neighbourhood, the heaviest heading bin (of :data:`HEADING_BINS`, with its two
    neighbours) seeds the cluster. :data:`MEAN_SHIFT_STEPS` weighted means over the particles
    within :data:`CLUSTER_RADIUS` and :data:`CLUSTER_HEADING_RADIUS` of the last mean refine it.
    Headings are averaged as wrapped differences, so a cluster across ``+-pi`` reports ``+-pi``.

    Args:
        particles: ``(P, 3)`` poses.
        weights: ``(P,)`` non-negative weights with a positive sum.
        roughening: ``(3,)`` standard deviations of the filter's jitter: their squares are added
            to the covariance, the kernel the particles stand for, so it is positive-definite even
            when every particle of the cluster coincides.

    Returns:
        ``(pose, covariance, share)``: ``(3,)``, ``(3, 3)`` and the cluster's share of the weight.
    """
    particles = np.asarray(particles, dtype=np.float64)
    weights = _normalized(weights)
    xy = particles[:, :2]
    origin = xy.min(axis=0)
    cells = np.floor((xy - origin) / CLUSTER_BIN).astype(np.intp)
    shape = cells.max(axis=0) + 1
    histogram = np.zeros(shape)
    np.add.at(histogram, (cells[:, 0], cells[:, 1]), weights)
    reach = CLUSTER_NEIGHBOURHOOD // 2
    padded = np.pad(histogram, reach)
    summed = np.zeros(shape)
    for i in range(CLUSTER_NEIGHBOURHOOD):
        for j in range(CLUSTER_NEIGHBOURHOOD):
            summed += padded[i : i + shape[0], j : j + shape[1]]
    best = np.array(np.unravel_index(np.argmax(summed), summed.shape))
    near = np.all(np.abs(cells - best) <= reach, axis=1)  # holds the neighbourhood's weight

    width = 2.0 * math.pi / HEADING_BINS
    heading_bins = np.floor((particles[:, 2] + math.pi) / width).astype(np.intp) % HEADING_BINS
    by_heading = np.bincount(heading_bins[near], weights[near], minlength=HEADING_BINS)
    by_heading = by_heading + np.roll(by_heading, 1) + np.roll(by_heading, -1)
    heading = -math.pi + (int(np.argmax(by_heading)) + 0.5) * width
    seed_xy = weights[near] @ xy[near] / weights[near].sum()
    # The heaviest heading bin with its two neighbours: their weight is by_heading's maximum,
    # positive, so the seed always has members.
    offset = (heading_bins - int(np.argmax(by_heading))) % HEADING_BINS
    members = near & np.isin(offset, (0, 1, HEADING_BINS - 1))
    mean = _weighted_mean(particles[members], weights[members], np.array([*seed_xy, heading]))
    for _ in range(MEAN_SHIFT_STEPS):
        within = _within(particles, mean, CLUSTER_RADIUS, CLUSTER_HEADING_RADIUS)
        if not weights[within].sum() > 0.0:
            break
        members = within
        mean = _weighted_mean(particles[members], weights[members], mean)
    share = float(weights[members].sum())
    difference = pose_difference(particles[members], mean)
    member_weights = weights[members] / share
    covariance = (difference * member_weights[:, None]).T @ difference
    covariance += np.diag(np.asarray(roughening, dtype=np.float64) ** 2)
    return mean, symmetrized(covariance), share


def sample_free_poses(
    range_map: ParticleMap,
    n: int,
    margin: float,
    rng: np.random.Generator,
    max_attempts: int = FREE_SAMPLE_ATTEMPTS,
) -> FloatArray:
    """``n`` poses drawn uniformly over the map's free space and headings in ``[-pi, pi)``.

    Batches of ``n`` candidates are drawn over :attr:`ParticleMap.bounds` and those more than
    ``margin`` from every obstacle kept, until ``n`` are found.

    Raises:
        ValueError: ``max_attempts`` batches found fewer than ``n`` free poses.
    """
    x_min, y_min, x_max, y_max = range_map.bounds
    found: list[FloatArray] = []
    count = 0
    for _ in range(max_attempts):
        candidates = np.column_stack(
            [
                rng.uniform(x_min, x_max, n),
                rng.uniform(y_min, y_max, n),
                rng.uniform(-math.pi, math.pi, n),
            ]
        )
        free = candidates[range_map.free(candidates[:, :2], margin)]
        found.append(free)
        count += len(free)
        if count >= n:
            return np.concatenate(found)[:n]
    raise ValueError(
        f"sample_free_poses found {count} of {n} free poses in {max_attempts} batches of "
        f"{n} on the {type(range_map).__name__}: is any of it more than {margin} m from every "
        "obstacle?"
    )


class MclLocalizer:
    """The robot's pose from wheel odometry and range scans on a known map, as a particle cloud.

    Args:
        range_map: The known map seen through the robot's range sensor.
        odometry: The wheel encoders' model (wheel base and noise).
        pose: ``(3,)`` initial pose estimate, with ``covariance``; None localizes globally, from
            particles spread uniformly over the free space.
        covariance: ``(3, 3)`` symmetric positive-definite covariance of ``pose``.
        rng: The filter's own generator: motion noise, resampling and injection.
        settings: The filter's settings.

    Raises:
        ValueError: A covariance without a pose or a pose without one, an invalid pose or
            covariance, or a start whose particles cannot be drawn in free space.
    """

    def __init__(
        self,
        range_map: ParticleMap,
        odometry: WheelOdometry,
        pose: ArrayLike | None = None,
        covariance: ArrayLike | None = None,
        *,
        rng: np.random.Generator,
        settings: MclSettings = MclSettings(),  # noqa: B008 -- frozen, so sharing it is safe
    ) -> None:
        self.range_map = range_map
        self.odometry = odometry
        self.settings = settings
        self._rng = rng
        count = settings.num_particles
        if pose is None:
            if covariance is not None:
                raise ValueError(
                    "MclLocalizer got a covariance without a pose: pass both or neither."
                )
            particles = sample_free_poses(range_map, count, settings.robot_radius, rng)
        else:
            if covariance is None:
                raise ValueError(
                    "MclLocalizer got a pose without a covariance: pass both, or neither to "
                    "localize globally."
                )
            particles = self._gaussian_particles(*checked_start(pose, covariance, "MclLocalizer"))
        self._particles = particles
        self._log_weights = np.full(count, -math.log(count))
        self._w_slow: float | None = None
        # Particles injected on the last resampling, left out of the next scan's w_avg.
        self._fresh = np.zeros(count, dtype=bool)
        self._w_fast = 0.0
        self._estimate: tuple[FloatArray, FloatArray, float] | None = None

    def _gaussian_particles(self, pose: FloatArray, covariance: FloatArray) -> FloatArray:
        """Particles drawn from ``N(pose, covariance)``, those outside free space redrawn."""
        particles = np.empty((self.settings.num_particles, 3))
        missing = np.ones(len(particles), dtype=bool)
        for _ in range(GAUSSIAN_START_ATTEMPTS):
            draws = self._rng.multivariate_normal(pose, covariance, size=int(missing.sum()))
            draws[:, 2] = wrap_angle(draws[:, 2])
            particles[missing] = draws
            missing[missing] = ~self.range_map.free(draws[:, :2], self.settings.robot_radius)
            if not missing.any():
                return particles
        raise ValueError(
            f"MclLocalizer could not draw its particles in free space around {pose.tolist()} in "
            f"{GAUSSIAN_START_ATTEMPTS} attempts: is the start pose within robot_radius "
            f"{self.settings.robot_radius} m of an obstacle?"
        )

    @property
    def particles(self) -> FloatArray:
        """A copy of the ``(P, 3)`` particles."""
        return self._particles.copy()

    @property
    def weights(self) -> FloatArray:
        """The ``(P,)`` normalized weights."""
        return np.exp(self._log_weights)

    def _cluster(self) -> tuple[FloatArray, FloatArray, float]:
        if self._estimate is None:
            self._estimate = heaviest_cluster(
                self._particles, self.weights, self.settings.roughening
            )
        return self._estimate

    @property
    def pose(self) -> FloatArray:
        """The mean pose of the heaviest cluster of particles (a copy)."""
        return self._cluster()[0].copy()

    @property
    def covariance(self) -> FloatArray:
        """The ``(3, 3)`` covariance of the heaviest cluster, plus the roughening kernel, scaled by
        ``settings.covariance_scale`` (a copy)."""
        root = np.sqrt(np.asarray(self.settings.covariance_scale, dtype=np.float64))
        scaled: FloatArray = self._cluster()[1] * np.outer(root, root)
        return scaled

    @property
    def cluster_weight(self) -> float:
        """The heaviest cluster's share of the weight: near 1 when the filter is sure, about
        ``1 / k`` among ``k`` hypotheses that the map cannot tell apart."""
        return self._cluster()[2]

    def predict(self, wheel_travel: ArrayLike) -> None:
        """Move every particle by its own noisy draw of one encoder reading.

        Args:
            wheel_travel: ``(2,)`` finite ``(left, right)`` wheel travel in metres since the last
                call, per :class:`~rl_navigation.sim.WheelOdometry`'s reading contract.

        Raises:
            ValueError: ``wheel_travel`` is not two finite numbers.
        """
        travel = checked_wheel_travel(wheel_travel, "MclLocalizer.predict")
        count = len(self._particles)
        std = np.sqrt(self.odometry.noise_per_metre * np.abs(travel))
        noisy = travel + std * self._rng.standard_normal((count, 2))
        increments = noisy @ self.odometry.increments_jacobian().T  # (P, 2) distance and turn
        moved = unicycle_step(self._particles, increments[:, 0], increments[:, 1], 1.0)
        free = self.range_map.free(moved[:, :2], self.settings.robot_radius)
        self._particles = np.where(free[:, None], moved, self._particles)
        self._estimate = None

    def update(self, readings: ArrayLike) -> MclUpdate:
        """Weigh the particles by one scan, and resample or inject when due.

        With no usable reading nothing changes and nothing is drawn.

        Args:
            readings: ``(N,)`` scan in the sensor's mount order, metres.

        Returns:
            What the update did.

        Raises:
            ValueError: The scan's length is not the sensor's.
        """
        z = np.asarray(readings, dtype=np.float64)
        if z.shape != (self.range_map.num_readings,):
            raise ValueError(
                f"MclLocalizer.update got readings of shape {z.shape}; the sensor has "
                f"{self.range_map.num_readings} beams."
            )
        used = usable_readings(z, self.range_map)
        if not used.any():
            return MclUpdate(used, effective_sample_size(self.weights), False, 0)
        beams = np.flatnonzero(used)
        settings = self.settings
        if len(beams) > settings.max_beams:
            beams = beams[np.round(np.linspace(0, len(beams) - 1, settings.max_beams)).astype(int)]
            used = np.zeros_like(used)
            used[beams] = True
        expected = self.range_map.expected_beam_ranges(self._particles, beams)
        log_likelihood = measurement_log_likelihood(
            z[beams], expected, self.range_map, settings.model_std, settings.z_rand
        )
        w_avg = float(
            _settled_weights(self.weights, self._fresh) @ np.exp(log_likelihood / len(beams))
        )
        self._fresh = np.zeros(len(self._particles), dtype=bool)
        if self._w_slow is None:
            self._w_slow = self._w_fast = w_avg
        self._w_slow, self._w_fast, p_inject = recovery_update(
            self._w_slow, self._w_fast, w_avg, settings.alpha_slow, settings.alpha_fast
        )
        log_weights = self._log_weights + log_likelihood
        log_weights -= log_weights.max()
        log_weights -= math.log(np.exp(log_weights).sum())
        self._log_weights = log_weights
        self._estimate = None
        effective_size = effective_sample_size(np.exp(log_weights))
        count = len(self._particles)
        inject = p_inject * count >= 1.0
        if not (inject or effective_size < settings.resample_threshold * count):
            return MclUpdate(used, effective_size, False, 0)
        injected = self._resample(p_inject if inject else 0.0)
        return MclUpdate(used, effective_size, True, injected)

    def _resample(self, p_inject: float) -> int:
        """Low-variance resampling, roughening, then injection; returns the particles injected."""
        settings, rng = self.settings, self._rng
        count = len(self._particles)
        parents = self._particles[low_variance_resample(self.weights, float(rng.random()))]
        jittered = parents + np.asarray(settings.roughening) * rng.standard_normal((count, 3))
        free = self.range_map.free(jittered[:, :2], settings.robot_radius)
        particles = np.where(free[:, None], jittered, parents)
        injected = rng.random(count) < p_inject
        if injected.any():
            particles[injected] = sample_free_poses(
                self.range_map, int(injected.sum()), settings.robot_radius, rng
            )
        particles[:, 2] = wrap_angle(particles[:, 2])
        self._fresh = injected  # indexes the new particles, as the next scan reads it
        self._particles = particles
        self._log_weights = np.full(count, -math.log(count))
        return int(injected.sum())

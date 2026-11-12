"""Landmark EKF-SLAM on circle features: the robot's pose and a map of circles, in one Gaussian.

**The variant** (pinned): Thrun, Burgard and Fox, *Probabilistic Robotics*, Table 10.1 (known
correspondence) and Table 10.2 (maximum likelihood with a new-landmark threshold alpha), with
§10.3.3's provisional list, and the first-estimates Jacobian EKF of Huang, Mourikis and
Roumeliotis (ICRA 2008), which fixes the inconsistency Julier and Uhlmann (ICRA 2001) and Bailey
et al. (IROS 2006) describe. Differences from the tables:

1. The control is the measured wheel travel, propagated as ``EkfLocalizer.predict`` does it.
2. The measurement is a body-frame circle ``z = (c_x, c_y, rho)`` with its fit covariance plus a
   model-error floor ``(floor_factor sigma)^2 I``, sigma the range noise (no angle to wrap). The floor absorbs the fit's small
   per-circle bias, which hundreds of views would otherwise turn into an overconfident radius and
   duplicate landmarks.
3. The radius is in the state: each landmark is ``m_j = (m_x, m_y, rho)``.
4. FEJ Jacobians (below); ``jacobians="standard"`` is kept as a reference.
5. One batch Joseph update per scan (:func:`~rl_navigation.estimation.kalman.kalman_update`).
6. No iteration: relinearizing at the current estimate would bring back the spurious information
   FEJ removes. No hold: keeping ``P`` while moving the state can break FEJ's consistency.

**State.** ``x = (x, y, theta, m_1x, m_1y, rho_1, ...)`` of shape ``(3 + 3L,)``, ``P`` of shape
``(3 + 3L, 3 + 3L)``, each landmark's first estimate ``first`` ``(L, 3)``, and the pose after the
last predict, ``previous_prediction`` ``(3,)``.

**Jacobians.** ``R = R(theta)``, ``J = [[0, -1], [1, 0]]``, ``R' = RJ = JR``.

- Motion: ``F = [[1, 0, -(y' - y)], [0, 1, x' - x], [0, 0, 1]]``; FEJ takes the pose before the
  step from the previous prediction, ``F_k = [[1, 0, -(y_{k+1|k} - y_{k|k-1})], [0, 1,
  x_{k+1|k} - x_{k|k-1}], [0, 0, 1]]``. ``G`` is ``motion_jacobians``' at ``x_{k|k}``. Landmarks do
  not move: only the pose rows and columns of ``P`` change.
- Measurement: ``h_j = (R^T (m_j - p), rho_j)``; ``dh/dp = -R^T``, ``dh/dtheta = R'^T (m_j - p)``,
  ``dh/dm_xy = R^T``, ``dh_rho/drho_j = 1``. FEJ evaluates ``dh/dtheta`` at ``first_j - p``.
- Initialization: ``g = (p + R z_xy, z_rho)``, ``G_pose = [[I, R' z_xy], [0, 0]]``,
  ``G_z = blockdiag(R, 1)``.

**Why FEJ.** Global translation and rotation are unobservable. At ``(p, theta, m_j)`` they are
``N_t = [I; 0; I; 0; ...]`` and ``N_r = [Jp; 1; Jm_1; 0; ...]``. ``H_j N = 0`` for both, and FEJ's
``H_j`` keeps that with ``N_r`` built at the first estimates. FEJ's ``F`` carries ``N`` at
``x_{k|k-1}`` to ``N`` at ``x_{k+1|k}``; the standard ``F`` misses by ``J (p_{k|k} -
p_{k|k-1})``, which is where a plain EKF-SLAM gains information about the absolute heading that
nothing measures. Initialization adds ``G_pose N_pose``, the null-space form at ``first = m``.

**Order of operations.** ``predict``: the new pose from ``x_{k|k}``, ``F`` from the previous
prediction and the new pose, ``G`` at ``x_{k|k}``, ``P`` propagated, then the previous prediction
and the pose set to the new pose. ``update``: extract features; associate (``H`` at the current
pose and the first estimates); one batch update (the previous prediction is not changed); the
result guards; then the provisional list and the new landmarks, initialized at the updated pose,
``first_j = m_j``. An update with no features changes nothing but the candidates' clock.

**Guards** (the filter never raises on a scan's content): a feature whose covariance with the
floor is not finite and positive definite is skipped; a labelled feature whose own ``S`` is not
positive definite is skipped, and one whose NIS exceeds ``oracle_quantile`` is rejected; a stacked
``S`` that is not positive definite falls back to one feature at a time; an update leaving ``x``
or ``P`` non-finite, ``P`` indefinite or the pose outside ``bounds`` is undone.

Known limit: FEJ linearizes at each landmark's first estimate, so a landmark first seen while the
heading was poorly known stays linearized there. The gates keep such a filter from diverging, not
from being wrong (OC-EKF is the follow-up).

Shares ``kalman_update``, ``motion_jacobians`` and ``chi2_quantile`` with ``estimation``. When a
third package needs them, they move to a shared module (``sim`` for the motion model).

Frames and units: metres and radians, the world frame, ``theta`` in ``[-pi, pi)``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.ekf import chi2_quantile, motion_jacobians
from rl_navigation.estimation.kalman import kalman_update, symmetrized
from rl_navigation.estimation.poses import checked_start, checked_wheel_travel
from rl_navigation.sim.geometry import FloatArray
from rl_navigation.sim.kinematics import unicycle_step, wrap_angle
from rl_navigation.sim.odometry import WheelOdometry
from rl_navigation.sim.sensors import RangeSensor
from rl_navigation.slam.features import CircleFeature, FeatureSettings, extract_circles

# associate_ml's labels for a feature that matches no landmark: a new one, or neither.
NEW_LANDMARK = -1
AMBIGUOUS = -2

# An updated covariance whose smallest eigenvalue lies below -PSD_TOLERANCE is indefinite.
PSD_TOLERANCE = 1e-9

Associate = Callable[[Sequence[CircleFeature]], Sequence[int]]
"""Labels each feature of a scan with a landmark's label (any non-negative int), or a negative
number to leave it out: the known-correspondence path (an oracle in the desk tool)."""

_QUARTER_TURN = np.array([[0.0, -1.0], [1.0, 0.0]])


def _rotation(theta: float) -> FloatArray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def fej_motion_jacobian(previous_prediction: ArrayLike, prediction: ArrayLike) -> FloatArray:
    """``(3, 3)`` the first-estimates motion Jacobian: the pose before the step is the previous
    prediction, ``F = [[1, 0, -(y_{k+1|k} - y_{k|k-1})], [0, 1, x_{k+1|k} - x_{k|k-1}], [0, 0, 1]]``.
    Equal to ``motion_jacobians``' ``F`` when the previous prediction is the estimate."""
    previous = np.asarray(previous_prediction, dtype=np.float64)
    new = np.asarray(prediction, dtype=np.float64)
    F = np.eye(3)
    F[0, 2] = -(new[1] - previous[1])
    F[1, 2] = new[0] - previous[0]
    return F


def landmark_measurement(
    pose: ArrayLike, landmarks: ArrayLike, lever: ArrayLike | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The body-frame circle each landmark predicts, and its Jacobians.

    Args:
        pose: ``(3,)`` the robot's pose.
        landmarks: ``(3,)`` or ``(L, 3)`` landmarks ``(m_x, m_y, rho)``.
        lever: Positions ``(2,)`` or ``(L, 2)`` at which ``dh/dtheta = R'^T (lever - p)`` is
            evaluated: the first estimates for FEJ; None uses the landmarks' own.

    Returns:
        ``(h, H_pose, H_landmark)``: ``(..., 3)``, ``(..., 3, 3)`` and ``(..., 3, 3)``.
    """
    pose = np.asarray(pose, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.float64)
    arm = landmarks[..., :2] if lever is None else np.asarray(lever, dtype=np.float64)
    R = _rotation(float(pose[2]))
    body = (landmarks[..., :2] - pose[:2]) @ R  # R^T (m - p), row-wise
    h = np.concatenate([body, landmarks[..., 2:3]], axis=-1)
    lever_body = (arm - pose[:2]) @ R
    H_pose = np.zeros(landmarks.shape[:-1] + (3, 3))
    H_pose[..., :2, :2] = -R.T
    # R'^T v = -J R^T v: (u_1, -u_0) for u = R^T v.
    H_pose[..., 0, 2] = lever_body[..., 1]
    H_pose[..., 1, 2] = -lever_body[..., 0]
    H_landmark = np.zeros_like(H_pose)
    H_landmark[..., :2, :2] = R.T
    H_landmark[..., 2, 2] = 1.0
    return h, H_pose, H_landmark


def initialize_landmark(pose: ArrayLike, z: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """A landmark from a body-frame circle seen at ``pose``, and its Jacobians.

    Returns:
        ``(m, G_pose, G_z)``: ``m = (p + R z_xy, z_rho)``, ``G_pose = [[I, R' z_xy], [0, 0]]``
        ``(3, 3)`` and ``G_z = blockdiag(R, 1)`` ``(3, 3)``.
    """
    pose = np.asarray(pose, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    R = _rotation(float(pose[2]))
    world = R @ z[:2]
    m = np.array([pose[0] + world[0], pose[1] + world[1], z[2]])
    G_pose = np.zeros((3, 3))
    G_pose[:2, :2] = np.eye(2)
    G_pose[:2, 2] = _QUARTER_TURN @ world  # R' z = J R z
    G_z = np.eye(3)
    G_z[:2, :2] = R
    return m, G_pose, G_z


def associate_ml(nis: ArrayLike, gate: float, new_landmark_nis: float) -> NDArray[np.intp]:
    """Maximum-likelihood data association (Thrun's Table 10.2), one to one within a scan.

    Pairs are taken in increasing NIS while both are free, if within ``gate``. A feature left
    over is a new landmark when its NIS exceeds ``new_landmark_nis`` for every landmark (taken
    or not), otherwise ambiguous and discarded. A NaN NIS counts as no match (infinite).

    Args:
        nis: ``(F, L)`` each feature's NIS against each landmark.
        gate: The association gate.
        new_landmark_nis: Thrun's alpha.

    Returns:
        ``(F,)`` a landmark index, :data:`NEW_LANDMARK` or :data:`AMBIGUOUS` per feature.
    """
    nis = np.asarray(nis, dtype=np.float64)
    nis = np.where(np.isnan(nis), np.inf, nis)
    num_features, num_landmarks = nis.shape
    labels = np.full(num_features, AMBIGUOUS, dtype=np.intp)
    taken_feature = np.zeros(num_features, dtype=bool)
    taken_landmark = np.zeros(num_landmarks, dtype=bool)
    for flat in np.argsort(nis, axis=None, kind="stable"):
        f, j = divmod(int(flat), num_landmarks)
        if nis[f, j] > gate:
            break
        if taken_feature[f] or taken_landmark[j]:
            continue
        labels[f] = j
        taken_feature[f] = taken_landmark[j] = True
    nearest = nis.min(axis=1, initial=np.inf)
    labels[~taken_feature & (nearest > new_landmark_nis)] = NEW_LANDMARK
    return labels


@dataclass(frozen=True)
class SlamSettings:
    """How the landmark SLAM extracts, weighs and associates circles.

    Args:
        features: The circle extraction's settings.
        floor_factor: The model-error floor in units of the sensor's range noise: a standard
            deviation of ``floor_factor * noise_std`` metres is added to every feature's centre and
            radius (``floor^2 I``), once, before any gate (>= 0). 1.5 (15 mm at 1 cm) brings the
            radius NEES to about 1 on tuning seeds; at 2 cm noise, 30 mm cuts it from 12 to 3.
        gate_quantile: Chi-square (3 degrees of freedom) quantile of the ML gate, in (0, 1).
        new_landmark_nis: Thrun's alpha: a feature whose NIS exceeds it for every landmark is a
            new candidate. Not below the gate.
        confirm: Sightings within ``candidate_radius``, at most one a scan, that make a candidate a
            landmark (>= 1).
        candidate_radius: Metres within which a sighting agrees with a candidate, in the centre
            and in the radius (> 0).
        candidate_timeout: Scans after which an unseen candidate is dropped (>= 1).
        oracle_quantile: Chi-square quantile of the labelled path's gate, in (0, 1).
        jacobians: ``"fej"`` (first estimates) or ``"standard"`` (a reference that is
            overconfident in the heading).

    Raises:
        ValueError: A value out of its range.
    """

    features: FeatureSettings = field(default_factory=FeatureSettings)
    floor_factor: float = 1.5
    gate_quantile: float = 0.99
    new_landmark_nis: float = 30.0
    confirm: int = 3
    candidate_radius: float = 0.1
    candidate_timeout: int = 50
    oracle_quantile: float = 0.999
    jacobians: Literal["fej", "standard"] = "fej"

    def __post_init__(self) -> None:
        problems = [
            (
                not self.floor_factor >= 0.0,
                f"floor_factor must be >= 0, got {self.floor_factor}",
            ),
            (
                not 0.0 < self.gate_quantile < 1.0,
                f"gate_quantile must be in (0, 1), got {self.gate_quantile}",
            ),
            (
                not 0.0 < self.oracle_quantile < 1.0,
                f"oracle_quantile must be in (0, 1), got {self.oracle_quantile}",
            ),
            (self.confirm < 1, f"confirm must be >= 1, got {self.confirm}"),
            (
                not self.candidate_radius > 0.0,
                f"candidate_radius must be positive, got {self.candidate_radius}",
            ),
            (
                self.candidate_timeout < 1,
                f"candidate_timeout must be >= 1, got {self.candidate_timeout}",
            ),
            (
                self.jacobians not in ("fej", "standard"),
                f"jacobians must be 'fej' or 'standard', got {self.jacobians!r}",
            ),
        ]
        if 0.0 < self.gate_quantile < 1.0:
            gate = chi2_quantile(self.gate_quantile, 3)
            problems.append(
                (
                    not self.new_landmark_nis >= gate,
                    f"new_landmark_nis must be at least the gate {gate:.2f}, got "
                    f"{self.new_landmark_nis}",
                )
            )
        wrong = [message for failed, message in problems if failed]
        if wrong:
            raise ValueError("SlamSettings: " + "; ".join(wrong) + ".")


@dataclass(frozen=True)
class SlamUpdate:
    """What one :meth:`EkfSlam.update` did with a scan's features.

    Attributes:
        used: ``(N,)`` the beams of the features that corrected the state.
        associated: Features that corrected the state.
        new: Landmarks added.
        ambiguous: Features discarded: between the gate and the new-landmark threshold, or a
            label given twice.
        rejected: Labelled features beyond the oracle's gate.
        skipped: Features with an unsound covariance or innovation covariance.
        rejected_update: Whether the update was undone by a result guard.
        landmarks: ``(associated,)`` the landmarks those features corrected.
    """

    used: NDArray[np.bool_]
    associated: int = 0
    new: int = 0
    ambiguous: int = 0
    rejected: int = 0
    skipped: int = 0
    rejected_update: bool = False
    landmarks: NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))


@dataclass(eq=False)
class _Candidate:
    """A provisional landmark: where it was last seen, how often, and when."""

    world: FloatArray
    sightings: int
    last_scan: int


def positive_definite(matrices: FloatArray) -> NDArray[np.bool_]:
    """``(...)`` whether each square ``(..., n, n)`` matrix is finite and passes Cholesky: a
    covariance the filter can factor."""
    finite = np.isfinite(matrices).all(axis=(-2, -1))
    safe = np.where(finite[..., None, None], matrices, np.eye(matrices.shape[-1]))
    try:
        np.linalg.cholesky(safe)
        return finite
    except np.linalg.LinAlgError:
        flat = safe.reshape(-1, *safe.shape[-2:])
        ok = np.empty(len(flat), dtype=bool)
        for i, matrix in enumerate(flat):
            try:
                np.linalg.cholesky(matrix)
                ok[i] = True
            except np.linalg.LinAlgError:
                ok[i] = False
        return finite & ok.reshape(finite.shape)


class EkfSlam:
    """The robot's pose and a map of circle landmarks, from wheel odometry and lidar scans.

    It keeps the :class:`~rl_navigation.estimation.evaluation.Localizer` protocol (``pose``,
    ``covariance``, ``predict``, ``update``), so a trial drives it like the localizers. See the
    module docstring for the variant, the Jacobians and the guards.

    Args:
        sensor: The lidar; its ``noise_std`` is the fits' range noise (> 0).
        odometry: The wheel encoders' model.
        pose: ``(3,)`` initial pose estimate.
        covariance: ``(3, 3)`` symmetric positive-definite initial covariance.
        settings: The SLAM settings; None uses the defaults.
        bounds: ``(x_min, y_min, x_max, y_max)`` the pose must stay within after an update, or
            None for no bound.

    Raises:
        ValueError: A pose or covariance that is not valid, a sensor without noise, or bounds
            that are not four finite increasing values.
    """

    def __init__(
        self,
        sensor: RangeSensor,
        odometry: WheelOdometry,
        pose: ArrayLike,
        covariance: ArrayLike,
        settings: SlamSettings | None = None,
        *,
        bounds: ArrayLike | None = None,
    ) -> None:
        pose, covariance = checked_start(pose, covariance, "EkfSlam")
        self._setup(sensor, odometry, settings, bounds)
        self._x = pose
        self._P = covariance
        self._first = np.zeros((0, 3))
        self._labels = np.zeros(0, dtype=np.intp)
        self._previous_prediction = pose.copy()

    @classmethod
    def from_state(
        cls,
        sensor: RangeSensor,
        odometry: WheelOdometry,
        state: ArrayLike,
        state_covariance: ArrayLike,
        settings: SlamSettings | None = None,
        *,
        first_estimates: ArrayLike | None = None,
        labels: ArrayLike | None = None,
        bounds: ArrayLike | None = None,
    ) -> "EkfSlam":
        """A filter with a map already: a saved state, or a test's.

        The covariance is checked for shape, finiteness and symmetry only, not for definiteness,
        so a degenerate state can be restored (and the guards tested).

        Args:
            state: ``(3 + 3L,)`` the pose, then each landmark ``(m_x, m_y, rho)``.
            state_covariance: ``(3 + 3L, 3 + 3L)``.
            first_estimates: ``(L, 3)`` FEJ's first estimates; None uses the landmarks.
            labels: ``(L,)`` each landmark's label for the labelled path; None gives -1 (none).

        Raises:
            ValueError: Shapes that do not agree, non-finite values or an asymmetric covariance.
        """
        x = np.array(state, dtype=np.float64)
        P = np.array(state_covariance, dtype=np.float64)
        if x.ndim != 1 or len(x) < 3 or (len(x) - 3) % 3 or not np.isfinite(x).all():
            raise ValueError(f"EkfSlam state must be 3 + 3L finite values, got shape {x.shape}.")
        if P.shape != (len(x), len(x)) or not np.isfinite(P).all():
            raise ValueError(
                f"EkfSlam state_covariance must be a finite {len(x)} x {len(x)} matrix, got "
                f"{P.shape}."
            )
        if not np.allclose(P, P.T, rtol=1e-12, atol=0.0):
            raise ValueError("EkfSlam state_covariance must be symmetric.")
        count = (len(x) - 3) // 3
        landmarks = x[3:].reshape(count, 3)
        first = landmarks.copy() if first_estimates is None else np.array(first_estimates, float)
        names = np.full(count, -1) if labels is None else np.array(labels, dtype=np.intp)
        if first.shape != (count, 3) or names.shape != (count,):
            raise ValueError(
                f"EkfSlam first_estimates must be ({count}, 3) and labels ({count},), got "
                f"{first.shape} and {names.shape}."
            )
        x[2] = wrap_angle(x[2])
        slam = cls.__new__(cls)
        slam._setup(sensor, odometry, settings, bounds)
        slam._x, slam._P, slam._first = x, P, first
        slam._labels = names.astype(np.intp)
        slam._previous_prediction = x[:3].copy()
        return slam

    def _setup(
        self,
        sensor: RangeSensor,
        odometry: WheelOdometry,
        settings: SlamSettings | None,
        bounds: ArrayLike | None,
    ) -> None:
        if not sensor.noise_std > 0.0:
            raise ValueError("EkfSlam needs a sensor with positive noise_std: the fits use it.")
        self.sensor = sensor
        self.odometry = odometry
        self.settings = SlamSettings() if settings is None else settings
        if bounds is None:
            self.bounds = None
        else:
            box = np.array(bounds, dtype=np.float64)
            if box.shape != (4,) or not np.isfinite(box).all() or not (box[:2] < box[2:]).all():
                raise ValueError(
                    f"EkfSlam bounds must be (x_min, y_min, x_max, y_max), got {bounds}."
                )
            self.bounds = box
        self._floor = (self.settings.floor_factor * sensor.noise_std) ** 2 * np.eye(3)
        self._gate = chi2_quantile(self.settings.gate_quantile, 3)
        self._oracle_gate = chi2_quantile(self.settings.oracle_quantile, 3)
        self._candidates: list[_Candidate] = []
        self._scan = 0

    # The state -----------------------------------------------------------------------------------

    @property
    def pose(self) -> FloatArray:
        """A copy of the estimated pose ``(x, y, theta)``."""
        return self._x[:3].copy()

    @property
    def covariance(self) -> FloatArray:
        """A copy of the ``(3, 3)`` pose covariance."""
        return self._P[:3, :3].copy()

    @property
    def state(self) -> FloatArray:
        """A copy of the ``(3 + 3L,)`` state."""
        return self._x.copy()

    @property
    def state_covariance(self) -> FloatArray:
        """A copy of the ``(3 + 3L, 3 + 3L)`` covariance."""
        return self._P.copy()

    @property
    def num_landmarks(self) -> int:
        return len(self._first)

    @property
    def landmarks(self) -> FloatArray:
        """``(L, 3)`` each landmark ``(m_x, m_y, rho)``."""
        return self._x[3:].reshape(-1, 3).copy()

    @property
    def landmark_covariances(self) -> FloatArray:
        """``(L, 3, 3)`` each landmark's covariance block."""
        index = 3 + 3 * np.arange(self.num_landmarks)[:, None] + np.arange(3)
        return self._P[index[:, :, None], index[:, None, :]].copy()

    @property
    def first_estimates(self) -> FloatArray:
        """``(L, 3)`` each landmark's first estimate (FEJ's linearization point)."""
        return self._first.copy()

    @property
    def labels(self) -> NDArray[np.intp]:
        """``(L,)`` each landmark's label on the labelled path, -1 for one the ML rule added."""
        return self._labels.copy()

    @property
    def previous_prediction(self) -> FloatArray:
        """The pose after the last predict (FEJ's ``x_{k|k-1}``)."""
        return self._previous_prediction.copy()

    @property
    def num_candidates(self) -> int:
        """Provisional landmarks awaiting confirmation."""
        return len(self._candidates)

    # Predict -------------------------------------------------------------------------------------

    def predict(self, wheel_travel: ArrayLike) -> None:
        """Move the pose by one encoder reading; grow the pose rows and columns of ``P``.

        Raises:
            ValueError: ``wheel_travel`` is not two finite numbers.
        """
        travel = checked_wheel_travel(wheel_travel, "EkfSlam.predict")
        distance, turn = self.odometry.increments(travel)
        pose = self._x[:3]
        F, G = motion_jacobians(pose, distance, turn)
        new = unicycle_step(pose, distance, turn, 1.0)
        if self.settings.jacobians == "fej":
            F = fej_motion_jacobian(self._previous_prediction, new)
        GA = G @ self.odometry.increments_jacobian()
        P = self._P  # private: the properties hand out copies, so it is changed in place
        P[:3, :3] = symmetrized(
            F @ P[:3, :3] @ F.T + GA @ self.odometry.wheel_covariance(travel) @ GA.T
        )
        P[:3, 3:] = F @ P[:3, 3:]
        P[3:, :3] = P[:3, 3:].T  # an exact transpose: P stays symmetric
        self._previous_prediction = new.copy()
        self._x[:3] = new

    # Update --------------------------------------------------------------------------------------

    def update(self, readings: ArrayLike, associate: Associate | None = None) -> SlamUpdate:
        """Correct the state with one scan, then extend the map.

        Args:
            readings: ``(N,)`` scan in the sensor's mount order, metres.
            associate: Labels the scan's features (the known-correspondence path); None uses the
                maximum-likelihood rule.

        Raises:
            ValueError: The scan's length is not the sensor's, or ``associate`` does not label
                every feature.
        """
        z = np.asarray(readings, dtype=np.float64)
        if z.shape != (self.sensor.num_readings,):
            raise ValueError(
                f"EkfSlam.update needs readings of shape ({self.sensor.num_readings},), got "
                f"{z.shape}."
            )
        return self.update_features(
            extract_circles(self.sensor, z, self.settings.features), associate
        )

    def update_features(
        self, features: Sequence[CircleFeature], associate: Associate | None = None
    ) -> SlamUpdate:
        """Correct the state with one scan's features, then extend the map (see :meth:`update`)."""
        self._scan += 1
        labels = None
        if associate is not None:
            labels = [int(label) for label in associate(features)]
            if len(labels) != len(features):
                raise ValueError(
                    f"EkfSlam.update's associate must return one label per feature: got "
                    f"{len(labels)} for {len(features)}."
                )
        counts = {"ambiguous": 0, "rejected": 0, "skipped": 0}
        noises = [np.asarray(f.covariance, dtype=np.float64) + self._floor for f in features]
        sound = positive_definite(np.array(noises).reshape(-1, 3, 3)) & np.array(
            [np.isfinite(f.z).all() for f in features], dtype=bool
        )
        counts["skipped"] += int((~sound).sum())
        kept = [i for i in range(len(features)) if sound[i]]
        x0, P0 = self._x, self._P
        rows, new = self._associate(features, noises, kept, labels, counts)
        used = np.zeros(self.sensor.num_readings, dtype=bool)
        rejected_update = False
        applied: list[tuple[int, int]] = []
        if rows:
            outcome = self._correct(features, noises, rows, counts)
            if outcome is None:
                rejected_update = True
            else:
                (self._x, self._P), applied = outcome
                for f, _ in applied:
                    used[features[f].beams] = True
        if rejected_update:
            self._x, self._P = x0, P0
        added = self._extend(features, noises, new, labels)
        self._candidates = [
            c
            for c in self._candidates
            if self._scan - c.last_scan < self.settings.candidate_timeout
        ]
        return SlamUpdate(
            used=used,
            associated=len(applied),
            new=added,
            ambiguous=counts["ambiguous"],
            rejected=counts["rejected"],
            skipped=counts["skipped"],
            rejected_update=rejected_update,
            landmarks=np.array([j for _, j in applied], dtype=np.intp),
        )

    def _predicted(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Every landmark's prediction, its Jacobians and ``S`` without the feature's noise."""
        landmarks = self._x[3:].reshape(-1, 3)
        lever = self._first[:, :2] if self.settings.jacobians == "fej" else None
        h, H_pose, H_landmark = landmark_measurement(self._x[:3], landmarks, lever)
        count = len(landmarks)
        P = self._P
        P_pp = P[:3, :3]
        index = 3 + 3 * np.arange(count)[:, None] + np.arange(3)
        P_pm = P[:3][:, index].transpose(1, 0, 2)  # (L, 3, 3): cov(pose, m_j)
        P_mm = P[index[:, :, None], index[:, None, :]]
        cross = H_pose @ P_pm @ H_landmark.transpose(0, 2, 1)
        base = (
            H_pose @ P_pp @ H_pose.transpose(0, 2, 1)
            + cross
            + cross.transpose(0, 2, 1)
            + H_landmark @ P_mm @ H_landmark.transpose(0, 2, 1)
        )
        return h, H_pose, H_landmark, base

    def _associate(
        self,
        features: Sequence[CircleFeature],
        noises: list[FloatArray],
        kept: list[int],
        labels: list[int] | None,
        counts: dict[str, int],
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """The ``(feature, landmark)`` rows to correct with, and the features that are new."""
        rows: list[tuple[int, int]] = []
        new: list[int] = []
        if not kept:
            return rows, new
        count = self.num_landmarks
        h, _, _, base = self._predicted()
        if labels is None:
            if count:
                z = np.array([features[i].z for i in kept])
                innovation = z[:, None, :] - h[None]
                S = base[None] + np.array([noises[i] for i in kept])[:, None]
                ok = positive_definite(S)
                safe = np.where(ok[..., None, None], S, np.eye(3))
                solved = np.linalg.solve(safe, innovation[..., None])[..., 0]
                nis = np.where(ok, np.einsum("flk,flk->fl", innovation, solved), np.inf)
            else:
                nis = np.zeros((len(kept), 0))
            for i, label in zip(
                kept, associate_ml(nis, self._gate, self.settings.new_landmark_nis), strict=True
            ):
                if label >= 0:
                    rows.append((i, int(label)))
                elif label == NEW_LANDMARK:
                    new.append(i)
                else:
                    counts["ambiguous"] += 1
            return rows, new
        index = {int(label): j for j, label in enumerate(self._labels) if label >= 0}
        seen: set[int] = set()
        for i in sorted(kept, key=lambda i: -len(features[i].beams)):
            label = labels[i]
            if label < 0:
                continue
            if label in seen:
                counts["ambiguous"] += 1
                continue
            seen.add(label)
            if label not in index:
                new.append(i)
                continue
            j = index[label]
            S = base[j] + noises[i]
            if not positive_definite(S):
                counts["skipped"] += 1
                continue
            innovation = features[i].z - h[j]
            if float(innovation @ np.linalg.solve(S, innovation)) > self._oracle_gate:
                counts["rejected"] += 1
                continue
            rows.append((i, j))
        return rows, new

    def _correct(
        self,
        features: Sequence[CircleFeature],
        noises: list[FloatArray],
        rows: list[tuple[int, int]],
        counts: dict[str, int],
    ) -> tuple[tuple[FloatArray, FloatArray], list[tuple[int, int]]] | None:
        """One batch update over ``rows`` (feature by feature if the stacked ``S`` is not positive
        definite), checked; None when a result guard undoes it."""
        h, H_pose, H_landmark, _ = self._predicted()
        x0, P0 = self._x, self._P
        H = np.zeros((3 * len(rows), len(x0)))
        innovation = np.empty(3 * len(rows))
        R = np.zeros((3 * len(rows), 3 * len(rows)))
        for r, (f, j) in enumerate(rows):
            block = slice(3 * r, 3 * r + 3)
            H[block, :3] = H_pose[j]
            H[block, 3 + 3 * j : 6 + 3 * j] = H_landmark[j]
            innovation[block] = features[f].z - h[j]
            R[block, block] = noises[f]
        S = H @ P0 @ H.T + R
        applied = list(rows)
        if positive_definite(S[None])[0]:
            x, P = kalman_update(x0, P0, innovation, H, R)
        else:
            x, P, applied = x0, P0, []
            for r, row in enumerate(rows):
                block = slice(3 * r, 3 * r + 3)
                H_r = H[block]
                if not positive_definite((H_r @ P @ H_r.T + R[block, block])[None])[0]:
                    counts["skipped"] += 1
                    continue
                shifted = innovation[block] - H_r @ (x - x0)  # the linear model's innovation now
                x, P = kalman_update(x, P, shifted, H_r, R[block, block])
                applied.append(row)
            if not applied:
                return (x0, P0), []
        x = np.array(x, dtype=np.float64)
        if not (np.isfinite(x).all() and np.isfinite(P).all()):
            return None
        x[2] = wrap_angle(x[2])
        if self.bounds is not None and not (
            (self.bounds[:2] <= x[:2]).all() and (x[:2] <= self.bounds[2:]).all()
        ):
            return None
        try:
            np.linalg.cholesky(P + PSD_TOLERANCE * np.eye(len(P)))
        except np.linalg.LinAlgError:
            return None
        return (x, P), applied

    def _extend(
        self,
        features: Sequence[CircleFeature],
        noises: list[FloatArray],
        new: list[int],
        labels: list[int] | None,
    ) -> int:
        """The provisional list and the new landmarks, at the updated pose; how many were added."""
        added = 0
        settings = self.settings
        for i in new:
            z = features[i].z
            if labels is not None or settings.confirm == 1:
                added += self._add_landmark(z, noises[i], -1 if labels is None else labels[i])
                continue
            world = initialize_landmark(self._x[:3], z)[0]
            hit = None
            for candidate in self._candidates:
                if candidate.last_scan == self._scan:
                    continue  # a scan sights a candidate once, however many fits it has of it
                if (
                    np.hypot(*(candidate.world[:2] - world[:2])) < settings.candidate_radius
                    and abs(candidate.world[2] - world[2]) < settings.candidate_radius
                ):
                    hit = candidate
                    break
            if hit is None:
                self._candidates.append(_Candidate(world, 1, self._scan))
                continue
            hit.world, hit.sightings, hit.last_scan = world, hit.sightings + 1, self._scan
            if hit.sightings >= settings.confirm:
                self._candidates.remove(hit)
                added += self._add_landmark(z, noises[i], -1)
        return added

    def _add_landmark(self, z: FloatArray, noise: FloatArray, label: int) -> int:
        """Augment the state with the landmark ``z`` sees; 1, or 0 if the result is not finite."""
        m, G_pose, G_z = initialize_landmark(self._x[:3], z)
        P = self._P
        cross = G_pose @ P[:3, :]  # (3, D): cov(m, x)
        block = G_pose @ P[:3, :3] @ G_pose.T + G_z @ noise @ G_z.T
        if not (np.isfinite(m).all() and np.isfinite(cross).all() and np.isfinite(block).all()):
            return 0
        self._P = np.block([[P, cross.T], [cross, symmetrized(block)]])
        self._x = np.concatenate([self._x, m])
        self._first = np.vstack([self._first, m])
        self._labels = np.append(self._labels, label).astype(np.intp)
        return 1

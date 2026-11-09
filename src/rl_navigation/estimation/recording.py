"""One seeded localization trial as pictures: the EKF and MCL side by side on the same robot.

:func:`record_trials` runs :func:`~rl_navigation.estimation.evaluation.run_trial` twice on one
seed, the EKF from the noisy start and MCL from anywhere, each filter wrapped so that it keeps a
snapshot (pose, covariance, scan and a subsample of MCL's particles) as built and after every
update. The trajectory does not depend on the filter, so the two trials share one robot.
:func:`render_frames` paints a frame per snapshot with the painters of
:mod:`rl_navigation.sim.render`; :mod:`rl_navigation.localization_video` writes them to an MP4.

Frame ``k`` shows each filter after its ``k``-th update with the robot where it was after step
``k``. Frame 0 shows the filters as built (MCL's cloud over the whole arena, no MCL estimate yet)
with the robot after step 1: ``run_trial`` does not hand out the true start.

Imports NumPy and :mod:`rl_navigation.sim` only, like the rest of the package, which does not
import it: it is a tool for the desk, not for the robot.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from rl_navigation.estimation.evaluation import (
    Localizer,
    LocalizerFactory,
    ScanResult,
    Trial,
    ekf,
    mcl,
    run_trial,
)
from rl_navigation.estimation.maps import SensorMap
from rl_navigation.estimation.mcl import MclSettings
from rl_navigation.sim import render
from rl_navigation.sim.geometry import Arena, FloatArray
from rl_navigation.sim.odometry import WheelOdometry
from rl_navigation.sim.sensors import RangeSensor

# Particles kept per snapshot, at a fixed stride: after low-variance resampling the weights are
# equal, so every stride-th particle is a fair sample of the cloud.
MAX_PARTICLES = 600

# A 5 m arena at 160 pixels per metre is 800 x 800.
PIXELS_PER_METRE = 160
PARTICLE_SIZE = 2  # pixels
MARKER_SCALE = 0.6  # an estimate's disc, as a share of the robot's radius
ELLIPSE_SIGMA = 2.0
STRIP_ROWS = 12  # the progress strip under the arena
KIDNAP_MARK_COLS = 3

BEAM_COLOUR = (250, 205, 205)
TRUE_PATH_COLOUR = (40, 170, 70)
PARTICLE_COLOUR = (245, 165, 60)
ROBOT_COLOUR = render.ROBOT_COLOUR
HEADING_COLOUR = render.HEADING_COLOUR
MCL_COLOUR = (190, 80, 0)
EKF_COLOUR = (180, 40, 180)
ESTIMATE_HEADING_COLOUR = (255, 255, 255)
STRIP_COLOUR = (225, 225, 225)
PROGRESS_COLOUR = (140, 140, 140)
KIDNAP_COLOUR = (220, 0, 0)

LEGEND = (
    "blue disc, yellow heading: the true robot",
    "pale red lines: the robot's scan",
    "green line: the true path, broken at the kidnapping",
    "orange dots: MCL's particles (a fixed-stride subsample)",
    "brown disc and ellipse: MCL's estimate and 2-sigma position ellipse (from step 1)",
    "magenta disc and ellipse: the EKF's estimate and 2-sigma position ellipse",
    "grey strip: progress; red mark: the kidnapping",
)


class SnapshotLocalizer:
    """A :class:`~rl_navigation.estimation.evaluation.Localizer` that keeps snapshots of the one
    it wraps: as built, and at the end of each ``update``.

    A snapshot is the pose, the covariance and, when the filter has ``particles``, every
    ``stride``-th particle from the first, at most ``max_particles``. Reading ``pose`` or
    ``covariance`` records nothing. :attr:`scans` holds the readings of each update.
    """

    def __init__(self, inner: Localizer, max_particles: int = MAX_PARTICLES) -> None:
        self.inner = inner
        self.max_particles = max_particles
        self.poses: list[FloatArray] = []
        self.covariances: list[FloatArray] = []
        self.scans: list[FloatArray] = []
        self.particles: list[FloatArray] | None = (
            None if getattr(inner, "particles", None) is None else []
        )
        self._snapshot()

    def _snapshot(self) -> None:
        self.poses.append(np.array(self.inner.pose, dtype=np.float64))
        self.covariances.append(np.array(self.inner.covariance, dtype=np.float64))
        if self.particles is not None:
            cloud = np.asarray(getattr(self.inner, "particles", None), dtype=np.float64)
            stride = max(1, len(cloud) // self.max_particles)
            self.particles.append(cloud[::stride][: self.max_particles].copy())

    @property
    def pose(self) -> FloatArray:
        """The wrapped filter's pose."""
        return self.inner.pose

    @property
    def covariance(self) -> FloatArray:
        """The wrapped filter's covariance."""
        return self.inner.covariance

    def predict(self, wheel_travel: ArrayLike) -> None:
        """The wrapped filter's prediction."""
        self.inner.predict(wheel_travel)

    def update(self, readings: ArrayLike) -> ScanResult:
        """The wrapped filter's update, then a snapshot."""
        result = self.inner.update(readings)
        self.scans.append(np.array(readings, dtype=np.float64))
        self._snapshot()
        return result


class SnapshotFactory:
    """A :class:`~rl_navigation.estimation.evaluation.LocalizerFactory` that wraps the filter
    the factory it is given builds in a :class:`SnapshotLocalizer`, kept as :attr:`built`, and
    keeps the trial's arena, sensor and robot radius."""

    def __init__(self, factory: LocalizerFactory, max_particles: int = MAX_PARTICLES) -> None:
        self.factory = factory
        self.max_particles = max_particles
        self.built: SnapshotLocalizer | None = None
        self.arena: Arena | None = None
        self.sensor: RangeSensor | None = None
        self.robot_radius: float | None = None

    def __call__(
        self,
        *,
        range_map: SensorMap,
        odometry: WheelOdometry,
        start: FloatArray,
        covariance: FloatArray,
        rng: np.random.Generator,
        robot_radius: float,
    ) -> Localizer:
        self.arena, self.sensor, self.robot_radius = range_map.arena, range_map.sensor, robot_radius
        inner = self.factory(
            range_map=range_map,
            odometry=odometry,
            start=start,
            covariance=covariance,
            rng=rng,
            robot_radius=robot_radius,
        )
        self.built = SnapshotLocalizer(inner, self.max_particles)
        return self.built


@dataclass(frozen=True, eq=False)
class Snapshots:
    """One filter's snapshots: as built, then after each of ``T`` updates.

    Attributes:
        poses: ``(T + 1, 3)``.
        covariances: ``(T + 1, 3, 3)``.
        scans: ``(T, N)`` the readings of each update.
        particles: ``(T + 1, M, 3)`` the kept particles, or None for a filter without them.
    """

    poses: FloatArray
    covariances: FloatArray
    scans: FloatArray
    particles: FloatArray | None

    @classmethod
    def of(cls, built: SnapshotLocalizer) -> "Snapshots":
        """The snapshots a :class:`SnapshotLocalizer` kept, as arrays."""
        width = len(built.scans[0]) if built.scans else 0
        return cls(
            poses=np.array(built.poses, dtype=np.float64),
            covariances=np.array(built.covariances, dtype=np.float64),
            scans=np.array(built.scans, dtype=np.float64).reshape(-1, width),
            particles=None
            if built.particles is None
            else np.array(built.particles, dtype=np.float64),
        )


@dataclass(frozen=True, eq=False)
class Recording:
    """The EKF's and MCL's trials on one seed, with their snapshots and what to draw them on.

    Attributes:
        arena: The trial's arena.
        sensor: The robot's sensor (for the beams).
        robot_radius: The robot's radius in metres.
        ekf: The EKF's snapshots.
        mcl: MCL's snapshots.
        ekf_trial: The EKF's trial.
        mcl_trial: MCL's trial, on the same trajectory.
    """

    arena: Arena
    sensor: RangeSensor
    robot_radius: float
    ekf: Snapshots
    mcl: Snapshots
    ekf_trial: Trial
    mcl_trial: Trial

    @property
    def true_poses(self) -> FloatArray:
        """``(T, 3)`` the robot's pose after each step, the same in both trials."""
        return self.ekf_trial.true_poses

    @property
    def scans(self) -> FloatArray:
        """``(T, N)`` the scan of each step."""
        return self.ekf.scans

    @property
    def kidnap_at(self) -> int | None:
        """The step at which the robot was kidnapped, or None."""
        return self.ekf_trial.kidnap_at

    @property
    def collided(self) -> bool:
        """Whether the trials ended at a collision before their last step."""
        return self.ekf_trial.collided


def record_trials(
    seed: int,
    sensor: RangeSensor,
    *,
    steps: int = 600,
    kidnap_at: int | None = None,
    dropout: float | None = None,
    mcl_settings: MclSettings = MclSettings(),  # noqa: B008 -- frozen, so sharing it is safe
    max_particles: int = MAX_PARTICLES,
) -> Recording:
    """Run the EKF (from the noisy start) and MCL (from anywhere) on one seed, with snapshots.

    Args:
        seed: The trials' seed.
        sensor: The robot's range sensor.
        steps: Steps to run; a collision ends both trials early, at the same step.
        kidnap_at: Step at which to kidnap the robot, in ``[1, steps)``; None never does.
        dropout: Sensor dropout probability; None uses the sensor's own.
        mcl_settings: MCL's settings (its particles and covariance scale).
        max_particles: Most of MCL's particles kept per snapshot.

    Returns:
        The two trials and their snapshots.

    Raises:
        ValueError: ``kidnap_at`` outside ``[1, steps)``; a collision on the first step (nothing
            to draw); or trials whose true poses or scans differ, which would mean the trajectory
            depends on the filter and the frames would mix two robots.
    """
    factories = {
        "ekf": SnapshotFactory(ekf(), max_particles),
        "mcl": SnapshotFactory(mcl(mcl_settings, global_start=True), max_particles),
    }
    trials = {
        name: run_trial(
            seed,
            sensor,
            steps=steps,
            dropout_prob=dropout,
            make_localizer=factory,
            kidnap_at=kidnap_at,
        )
        for name, factory in factories.items()
    }
    ekf_built, mcl_built, factory = factories["ekf"].built, factories["mcl"].built, factories["ekf"]
    assert ekf_built is not None and mcl_built is not None  # run_trial built both
    assert factory.arena is not None and factory.sensor is not None
    assert factory.robot_radius is not None
    ekf_trial, mcl_trial = trials["ekf"], trials["mcl"]
    if ekf_trial.true_poses.shape != mcl_trial.true_poses.shape or not np.array_equal(
        ekf_trial.true_poses, mcl_trial.true_poses
    ):
        raise ValueError(
            f"record_trials: the EKF's and MCL's trials on seed {seed} have different true poses; "
            "the trajectory must not depend on the filter."
        )
    ekf_snapshots, mcl_snapshots = Snapshots.of(ekf_built), Snapshots.of(mcl_built)
    if not np.array_equal(ekf_snapshots.scans, mcl_snapshots.scans):
        raise ValueError(
            f"record_trials: the EKF's and MCL's trials on seed {seed} read different scans."
        )
    if len(ekf_trial.true_poses) == 0:
        raise ValueError(f"record_trials: seed {seed} collided on its first step; nothing to draw.")
    return Recording(
        arena=factory.arena,
        sensor=factory.sensor,
        robot_radius=factory.robot_radius,
        ekf=ekf_snapshots,
        mcl=mcl_snapshots,
        ekf_trial=ekf_trial,
        mcl_trial=mcl_trial,
    )


def covariance_ellipse(
    mean: ArrayLike, covariance: ArrayLike, n_sigma: float = ELLIPSE_SIGMA, points: int = 48
) -> FloatArray:
    """The ``n_sigma`` boundary of the position part of a pose covariance, as a closed outline.

    A point ``p`` on it satisfies ``d^T S^-1 d = n_sigma^2`` with ``d = p - mean[:2]`` and ``S``
    the covariance's top-left ``2 x 2``. Eigenvalues below 0 (round-off) are taken as 0, so a
    degenerate covariance gives a segment.

    Args:
        mean: The pose; its first two entries are the centre.
        covariance: ``(3, 3)`` or ``(2, 2)``.
        n_sigma: Standard deviations out.
        points: Points round the outline.

    Returns:
        ``(points + 1, 2)``, the last point equal to the first; all NaN for a covariance that is
        not finite.
    """
    position = np.asarray(covariance, dtype=np.float64)[:2, :2]
    if not np.isfinite(position).all():
        return np.full((points + 1, 2), np.nan)
    values, vectors = np.linalg.eigh(0.5 * (position + position.T))
    radii = n_sigma * np.sqrt(np.clip(values, 0.0, None))
    angles = 2.0 * np.pi * np.arange(points + 1) / points
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    unit[-1] = unit[0]
    centre = np.asarray(mean, dtype=np.float64)[:2]
    outline: FloatArray = centre + (unit * radii) @ vectors.T
    return outline


def _broken_path(true_poses: FloatArray, kidnap_at: int | None) -> FloatArray:
    """``(T, 2)`` true positions, with a NaN row before the first pose after the kidnapping."""
    path = true_poses[:, :2]
    if kidnap_at is not None and kidnap_at < len(path):
        path = np.insert(path, kidnap_at, np.nan, axis=0)
    return path


def _as_path(segments: FloatArray) -> FloatArray:
    """``(S, 2, 2)`` segments as one path, each followed by a NaN row (for ``paint_path``)."""
    gaps = np.full((len(segments), 1, 2), np.nan)
    return np.concatenate([segments, gaps], axis=1).reshape(-1, 2)


def _strip(cols: int, frame: int, frames: int, kidnap_frame: int | None) -> render.Image:
    """The progress strip: grey up to ``frame`` of ``frames``, a red mark at the kidnapping."""
    strip = np.empty((STRIP_ROWS, cols, 3), dtype=np.uint8)
    strip[...] = STRIP_COLOUR
    last = max(frames - 1, 1)
    strip[:, : int(round(cols * frame / last))] = PROGRESS_COLOUR
    if kidnap_frame is not None and kidnap_frame < frames:
        centre = int(round((cols - 1) * kidnap_frame / last))
        low = max(centre - KIDNAP_MARK_COLS // 2, 0)
        strip[:, low : low + KIDNAP_MARK_COLS] = KIDNAP_COLOUR
    return strip


def _paint_estimate(
    image: render.Image,
    pose: FloatArray,
    covariance: FloatArray,
    marker_radius: float,
    colour: render.Colour,
    pixels_per_metre: float,
) -> None:
    render.paint_path(
        image, covariance_ellipse(pose, covariance), colour, pixels_per_metre=pixels_per_metre
    )
    render.paint_pose(
        image,
        pose,
        marker_radius,
        colour,
        ESTIMATE_HEADING_COLOUR,
        pixels_per_metre=pixels_per_metre,
    )


def render_frames(
    recording: Recording, pixels_per_metre: float = PIXELS_PER_METRE
) -> Iterator[render.Image]:
    """One frame per snapshot, ``(rows + STRIP_ROWS, cols, 3)`` uint8, painted one at a time.

    Painted in order: the scan's beams, the true path (broken at the kidnapping), MCL's
    particles, the robot, then each filter's 2-sigma ellipse and a smaller disc for its estimate
    (the EKF's, then MCL's from frame 1), over the arena, with the progress strip below.
    """
    background = render.arena_image(recording.arena, pixels_per_metre)
    cols = background.shape[1]
    path = _broken_path(recording.true_poses, recording.kidnap_at)
    frames = len(recording.ekf.poses)
    kidnap_frame = None if recording.kidnap_at is None else recording.kidnap_at + 1
    marker = MARKER_SCALE * recording.robot_radius
    particles = recording.mcl.particles
    for frame in range(frames):
        step = max(frame - 1, 0)  # frame 0 shows the robot after step 1
        pose = recording.true_poses[step]
        image = background.copy()
        beams = recording.sensor.beams(pose, recording.scans[step])
        render.paint_path(image, _as_path(beams), BEAM_COLOUR, pixels_per_metre=pixels_per_metre)
        shown = step + 1 + int(recording.kidnap_at is not None and step >= recording.kidnap_at)
        render.paint_path(image, path[:shown], TRUE_PATH_COLOUR, pixels_per_metre=pixels_per_metre)
        if particles is not None:
            render.paint_points(
                image,
                particles[frame][:, :2],
                PARTICLE_COLOUR,
                size=PARTICLE_SIZE,
                pixels_per_metre=pixels_per_metre,
            )
        render.paint_pose(
            image,
            pose,
            recording.robot_radius,
            ROBOT_COLOUR,
            HEADING_COLOUR,
            pixels_per_metre=pixels_per_metre,
        )
        _paint_estimate(
            image,
            recording.ekf.poses[frame],
            recording.ekf.covariances[frame],
            marker,
            EKF_COLOUR,
            pixels_per_metre,
        )
        if frame > 0:  # MCL's global start has no estimate worth drawing
            _paint_estimate(
                image,
                recording.mcl.poses[frame],
                recording.mcl.covariances[frame],
                marker,
                MCL_COLOUR,
                pixels_per_metre,
            )
        yield np.concatenate([image, _strip(cols, frame, frames, kidnap_frame)])

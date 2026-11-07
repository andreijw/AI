"""Seeded localization trials in the simulator, scored against its ground truth.

A trial draws a layout, drives the robot with a simple wandering controller, feeds a filter (the
EKF by default, or MCL) the simulated encoders and range scans, and records the true pose
(:attr:`World.pose`), the estimate, its covariance and dead reckoning on the same encoder
readings. A trial can also kidnap the robot: carry it to another pose without telling the filter.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rl_navigation.estimation.ekf import EkfLocalizer
from rl_navigation.estimation.maps import SensorMap
from rl_navigation.estimation.mcl import MclLocalizer, MclSettings, sample_free_poses
from rl_navigation.estimation.poses import pose_difference
from rl_navigation.sim.geometry import FloatArray, segment_hits
from rl_navigation.sim.kinematics import unicycle_step, wrap_angle
from rl_navigation.sim.layouts import LayoutParams, generate_layout
from rl_navigation.sim.odometry import WheelOdometry
from rl_navigation.sim.sensors import RangeSensor
from rl_navigation.sim.world import World

# The wandering controller: cruise at CRUISE_SPEED with a random turn rate up to WANDER_TURN,
# and turn in place at TURN_RATE while the disc swept LOOK_AHEAD metres ahead would hit anything.
CRUISE_SPEED = 0.3
WANDER_TURN = 0.5
TURN_RATE = 2.0
LOOK_AHEAD = 0.6

# A trial is lost when its position error ever exceeds this, in metres.
LOST_ERROR = 0.5

# A filter is localized at a step when its position error is under LOCALIZED_POSITION metres and
# its heading error under LOCALIZED_HEADING radians. It localized when it is so at every step of
# the LOCALIZE_HOLD steps from LOCALIZE_DEADLINE steps after the start or the kidnapping: with
# 0.1 s steps, within 25 s, held for 5 s.
LOCALIZED_POSITION = 0.2
LOCALIZED_HEADING = 0.2
LOCALIZE_DEADLINE = 250
LOCALIZE_HOLD = 50

# A kidnapping carries the robot at least KIDNAP_DISTANCE metres, drawn in at most KIDNAP_ATTEMPTS
# tries, to a pose LayoutParams.margin clear of the robot's radius, like a layout's start.
KIDNAP_DISTANCE = 1.0
KIDNAP_ATTEMPTS = 1000

# Upper end of the two-sided 95 % interval of chi-square with 3 degrees of freedom: a consistent
# filter's NEES lies below it at 97.5 % of steps.
NEES_UPPER_95 = 9.35


@dataclass(frozen=True, eq=False)
class Trial:
    """One run's record, one row per step that completed.

    Attributes:
        true_poses: ``(T, 3)`` the simulator's pose after each step.
        estimates: ``(T, 3)`` the filter's pose after each update.
        covariances: ``(T, 3, 3)`` its covariance.
        dead_reckoning: ``(T, 3)`` the encoder readings integrated from the same start.
        beams_used: ``(T,)`` beams each update used. For the EKF, the beams that corrected the
            pose; for MCL, the beams in the likelihood (at most ``max_beams``): not like for like.
        collided: Whether the run ended at a collision before its last step.
        kidnap_at: The step at which the robot was kidnapped, or None.
    """

    true_poses: FloatArray
    estimates: FloatArray
    covariances: FloatArray
    dead_reckoning: FloatArray
    beams_used: NDArray[np.int_]
    collided: bool
    kidnap_at: int | None = None


class ScanResult(Protocol):
    """What a filter's update reports: which beams it used."""

    @property
    def used(self) -> NDArray[np.bool_]: ...


class Localizer(Protocol):
    """A pose filter as the trials drive it: :class:`EkfLocalizer` or :class:`MclLocalizer`."""

    @property
    def pose(self) -> FloatArray: ...

    @property
    def covariance(self) -> FloatArray: ...

    def predict(self, wheel_travel: ArrayLike) -> None: ...

    def update(self, readings: ArrayLike) -> ScanResult: ...


class LocalizerFactory(Protocol):
    """Builds a trial's filter; :func:`run_trial` calls it with keywords only."""

    def __call__(
        self,
        *,
        range_map: SensorMap,
        odometry: WheelOdometry,
        start: FloatArray,
        covariance: FloatArray,
        rng: np.random.Generator,
        robot_radius: float,
    ) -> Localizer: ...


def ekf(iterations: int = 3, linearity_tol: float | None = None) -> LocalizerFactory:
    """A factory of :class:`EkfLocalizer` started at the trial's noisy start and covariance."""

    def make(
        *,
        range_map: SensorMap,
        odometry: WheelOdometry,
        start: FloatArray,
        covariance: FloatArray,
        rng: np.random.Generator,
        robot_radius: float,
    ) -> Localizer:
        return EkfLocalizer(
            range_map,
            odometry,
            start,
            covariance,
            iterations=iterations,
            linearity_tol=linearity_tol,
        )

    return make


def mcl(
    settings: MclSettings = MclSettings(),  # noqa: B008 -- frozen, so sharing it is safe
    global_start: bool = False,
) -> LocalizerFactory:
    """A factory of :class:`MclLocalizer` with the filter's stream as its generator.

    It starts at the trial's noisy start and covariance, or from anywhere when ``global_start``.
    The trial's robot radius replaces ``settings.robot_radius``, so the filter's free space is the
    simulated robot's.
    """

    def make(
        *,
        range_map: SensorMap,
        odometry: WheelOdometry,
        start: FloatArray,
        covariance: FloatArray,
        rng: np.random.Generator,
        robot_radius: float,
    ) -> Localizer:
        return MclLocalizer(
            range_map,
            odometry,
            None if global_start else start,
            None if global_start else covariance,
            rng=rng,
            settings=replace(settings, robot_radius=robot_radius),
        )

    return make


def _wander(world: World, rng: np.random.Generator) -> tuple[float, float]:
    """A command that wanders without hitting anything: turn in place while blocked ahead."""
    pose = world.pose
    ahead = pose[:2] + LOOK_AHEAD * np.array([np.cos(pose[2]), np.sin(pose[2])])
    if segment_hits(world.arena, pose[:2], ahead, world.robot_radius):
        return 0.0, TURN_RATE
    return CRUISE_SPEED, float(rng.uniform(-WANDER_TURN, WANDER_TURN))


def run_trial(
    seed: int,
    sensor: RangeSensor,
    *,
    steps: int = 600,
    dropout_prob: float | None = None,
    odometry: WheelOdometry | None = None,
    filter_odometry: WheelOdometry | None = None,
    layout_params: LayoutParams | None = None,
    iterations: int | None = None,
    linearity_tol: float | None = None,
    initial_std: Sequence[float] = (0.05, 0.05, 0.05),
    make_localizer: LocalizerFactory | None = None,
    kidnap_at: int | None = None,
) -> Trial:
    """Localize a wandering robot for up to ``steps`` steps of a seeded layout.

    The seed's ``SeedSequence`` is split into independent streams for the layout, the
    controller, the world (slip and sensor noise), the encoders, the initial error, the filter's
    own draws and the kidnapping. The filter is given the true start plus
    ``N(0, diag(initial_std^2))`` with that covariance (MCL from :func:`mcl` with
    ``global_start`` ignores both).

    At step ``kidnap_at`` the robot is carried, before it moves, to a pose drawn uniformly over the
    free space at least :data:`KIDNAP_DISTANCE` from where it was; the filter is not told. Dead
    reckoning carries on from the old pose and means nothing afterwards.

    Args:
        seed: The trial's seed.
        sensor: The robot's range sensor, also the filter's measurement model.
        steps: Steps to run; a collision ends the run early.
        dropout_prob: Sensor dropout probability; None uses the sensor's own.
        odometry: The simulated encoders; None uses the defaults.
        filter_odometry: The filter's encoder model; None uses ``odometry`` (a matched model).
        layout_params: How the layout is drawn; None uses the defaults.
        iterations: The EKF's update iterations; None uses 3.
        linearity_tol: The EKF's linearity tolerance; None uses its default.
        initial_std: Standard deviations of the initial error in ``x``, ``y`` and ``theta``.
        make_localizer: Builds the filter (:func:`ekf` or :func:`mcl`); None is
            ``ekf(iterations, linearity_tol)``.
        kidnap_at: Step at which to kidnap the robot, in ``[1, steps)``; None never does.

    Returns:
        The trial's record.

    Raises:
        ValueError: ``make_localizer`` given together with ``iterations`` or ``linearity_tol``
            (which only the default EKF takes), or ``kidnap_at`` outside ``[1, steps)``.
    """
    if make_localizer is None:
        make_localizer = ekf(3 if iterations is None else iterations, linearity_tol)
    elif iterations is not None or linearity_tol is not None:
        raise ValueError(
            "run_trial takes iterations and linearity_tol for its default EKF only; with "
            "make_localizer, pass them to ekf(...) instead."
        )
    if kidnap_at is not None and not 1 <= kidnap_at < steps:
        raise ValueError(f"run_trial kidnap_at must be in [1, {steps}), got {kidnap_at}.")
    layout_rng, control_rng, world_rng, odometry_rng, initial_rng, filter_rng, kidnap_rng = (
        np.random.default_rng(stream) for stream in np.random.SeedSequence(seed).spawn(7)
    )
    params = LayoutParams() if layout_params is None else layout_params
    layout = generate_layout(layout_rng, params)

    def world_at(pose: FloatArray) -> World:
        """The trial's world with the robot at ``pose``: at the start, and after a kidnapping."""
        return World(layout.arena, sensor, pose, layout.goal, robot_radius=params.robot_radius)

    world = world_at(layout.start)
    range_map = SensorMap(layout.arena, sensor)
    odometry = WheelOdometry() if odometry is None else odometry
    model = odometry if filter_odometry is None else filter_odometry
    std = np.asarray(initial_std, dtype=np.float64)
    start = world.pose + std * initial_rng.standard_normal(3)
    start[2] = wrap_angle(start[2])
    localizer = make_localizer(
        range_map=range_map,
        odometry=model,
        start=start,
        covariance=np.diag(std**2),
        rng=filter_rng,
        robot_radius=params.robot_radius,
    )
    dead = start
    rows: list[tuple[FloatArray, FloatArray, FloatArray, FloatArray, int]] = []
    collided = False
    for step in range(steps):
        if step == kidnap_at:
            pose = _kidnap_pose(
                range_map, world.pose, params.robot_radius + params.margin, kidnap_rng
            )
            world = world_at(pose)
        v, omega = _wander(world, control_rng)
        motion = world.step(v, omega, world_rng)
        if motion.collided:
            collided = True
            break
        travel = odometry.read(motion.v, motion.omega, world.dt, odometry_rng)
        localizer.predict(travel)
        dead = unicycle_step(dead, *model.increments(travel), 1.0)
        readings, _ = world.read_sensors(world_rng, dropout_prob=dropout_prob)
        result = localizer.update(readings)
        rows.append(
            (world.pose, localizer.pose, localizer.covariance, dead, int(result.used.sum()))
        )
    return Trial(
        true_poses=np.array([row[0] for row in rows], dtype=np.float64).reshape(-1, 3),
        estimates=np.array([row[1] for row in rows], dtype=np.float64).reshape(-1, 3),
        covariances=np.array([row[2] for row in rows], dtype=np.float64).reshape(-1, 3, 3),
        dead_reckoning=np.array([row[3] for row in rows], dtype=np.float64).reshape(-1, 3),
        beams_used=np.array([row[4] for row in rows], dtype=np.int_),
        collided=collided,
        kidnap_at=kidnap_at,
    )


def _kidnap_pose(
    range_map: SensorMap, pose: FloatArray, margin: float, rng: np.random.Generator
) -> FloatArray:
    """A free pose at least :data:`KIDNAP_DISTANCE` from ``pose``."""
    for _ in range(KIDNAP_ATTEMPTS):
        candidate = sample_free_poses(range_map, 1, margin, rng)[0]
        if np.hypot(*(candidate[:2] - pose[:2])) >= KIDNAP_DISTANCE:
            return candidate
    raise ValueError(
        f"run_trial found no free pose {KIDNAP_DISTANCE} m from {pose.tolist()} to kidnap the "
        f"robot to in {KIDNAP_ATTEMPTS} draws."
    )


def _check_trials(trials: Sequence[Trial]) -> None:
    """Refuse an empty list and a trial without steps (a collision on its first step)."""
    if not trials:
        raise ValueError("Summarizing needs trials; got no trials.")
    empty = [index for index, trial in enumerate(trials) if len(trial.true_poses) == 0]
    if empty:
        raise ValueError(
            f"Trials {empty} have no steps (a collision on the first step): leave them out and "
            "count them as collisions."
        )


def _nees(trial: Trial) -> FloatArray:
    """``(T,)`` normalised estimation error squared, ``e^T P^-1 e``."""
    errors = pose_difference(trial.estimates, trial.true_poses)
    solved = np.linalg.solve(trial.covariances, errors[..., None])[..., 0]
    return np.asarray(np.einsum("ti,ti->t", errors, solved))


def average_nees(trials: Sequence[Trial]) -> FloatArray:
    """The NEES at each step averaged over the trials, cut to the shortest trial: ``(T,)``.

    Raises:
        ValueError: No trials, or a trial without steps.
    """
    _check_trials(trials)
    length = min(len(trial.true_poses) for trial in trials)
    return np.mean([_nees(trial)[:length] for trial in trials], axis=0)


def _error_summary(trials: Sequence[Trial], estimates: str, prefix: str) -> dict[str, float]:
    """Position and heading error of ``estimates`` (an attribute of each trial) against the truth."""
    errors = [pose_difference(getattr(trial, estimates), trial.true_poses) for trial in trials]
    position = [np.hypot(error[:, 0], error[:, 1]) for error in errors]
    every_position = np.concatenate(position)
    return {
        f"{prefix}position_rmse": float(np.sqrt(np.mean(every_position**2))),
        f"{prefix}heading_rmse": float(np.sqrt(np.mean(np.concatenate(errors)[:, 2] ** 2))),
        f"{prefix}final_position_error": float(np.mean([p[-1] for p in position])),
        f"{prefix}max_position_error": float(every_position.max(initial=0.0)),
        f"{prefix}lost": float(sum(p.max(initial=0.0) > LOST_ERROR for p in position)),
    }


def summarize(trials: Sequence[Trial]) -> dict[str, float]:
    """Pose error and consistency over trials, for the filter and for dead reckoning.

    Returns:
        ``position_rmse`` (m) and ``heading_rmse`` (rad) over every step;
        ``final_position_error`` (mean over trials) and ``max_position_error``; ``mean_nees``,
        ``median_nees`` and ``nees_below_upper_95`` (the share of steps under
        :data:`NEES_UPPER_95`); ``beams_used`` per update; ``lost`` (trials whose error ever
        exceeded :data:`LOST_ERROR`), ``steps`` per trial and ``collisions``; and the five error
        figures again for dead reckoning, prefixed ``dead_reckoning_``.

    Raises:
        ValueError: No trials, or a trial without steps.
    """
    _check_trials(trials)
    nees = np.concatenate([_nees(trial) for trial in trials])
    return {
        **_error_summary(trials, "estimates", ""),
        "mean_nees": float(nees.mean()),
        "median_nees": float(np.median(nees)),
        "nees_below_upper_95": float(np.mean(nees <= NEES_UPPER_95)),
        "beams_used": float(np.concatenate([trial.beams_used for trial in trials]).mean()),
        "steps": float(np.mean([len(trial.true_poses) for trial in trials])),
        "collisions": float(sum(trial.collided for trial in trials)),
        **_error_summary(trials, "dead_reckoning", "dead_reckoning_"),
    }


def localization_steps(
    trial: Trial,
    deadline: int = LOCALIZE_DEADLINE,
    hold: int = LOCALIZE_HOLD,
    position_tol: float = LOCALIZED_POSITION,
    heading_tol: float = LOCALIZED_HEADING,
) -> tuple[int | None, int | None]:
    """Steps the filter took to localize from the start, and again after the kidnapping.

    The filter localized when it is within ``position_tol`` metres and ``heading_tol`` radians at
    every one of the ``hold`` steps from ``deadline`` steps after the start (or after
    ``trial.kidnap_at``); a run that ends (a collision) before the window does has not. The steps
    it took count from the start (or the kidnapping) to the first step from which it stayed
    within tolerance to the window's end.

    Returns:
        ``(steps, recovery_steps)``, each None when the filter did not localize;
        ``recovery_steps`` is also None for a trial without a kidnapping.

    Raises:
        ValueError: ``deadline`` below 0, ``hold`` below 1, or a first window reaching past the
            kidnapping.
    """
    if deadline < 0 or hold < 1:
        raise ValueError(
            f"localization_steps needs deadline >= 0 and hold >= 1, got {deadline}, {hold}."
        )
    if trial.kidnap_at is not None and deadline + hold > trial.kidnap_at:
        raise ValueError(
            f"localization_steps's first window ends at step {deadline + hold}, past the "
            f"trial's kidnap_at {trial.kidnap_at}."
        )
    errors = pose_difference(trial.estimates, trial.true_poses)
    within = (np.hypot(errors[:, 0], errors[:, 1]) < position_tol) & (
        np.abs(errors[:, 2]) < heading_tol
    )

    def settled(start: int) -> int | None:
        end = start + deadline + hold
        if end > len(within) or not within[start + deadline : end].all():
            return None
        outside = np.flatnonzero(~within[start:end])
        return int(outside[-1]) + 1 if len(outside) else 0

    return settled(0), None if trial.kidnap_at is None else settled(trial.kidnap_at)


def localization_success(
    trial: Trial,
    deadline: int = LOCALIZE_DEADLINE,
    hold: int = LOCALIZE_HOLD,
    position_tol: float = LOCALIZED_POSITION,
    heading_tol: float = LOCALIZED_HEADING,
) -> tuple[bool, bool | None]:
    """Whether the filter localized from the start, and again after the kidnapping
    (:func:`localization_steps`).

    Returns:
        ``(localized, recovered)``; ``recovered`` is None for a trial without a kidnapping.

    Raises:
        ValueError: As :func:`localization_steps`.
    """
    steps, recovery = localization_steps(trial, deadline, hold, position_tol, heading_tol)
    return steps is not None, None if trial.kidnap_at is None else recovery is not None


def _median_steps(steps: Sequence[int | None]) -> float:
    """Median of the steps that are not None; NaN when none is."""
    found = [step for step in steps if step is not None]
    return float(np.median(found)) if found else float("nan")


def summarize_localization(
    trials: Sequence[Trial], deadline: int = LOCALIZE_DEADLINE, hold: int = LOCALIZE_HOLD
) -> dict[str, float]:
    """Global localization and kidnapping recovery over trials (:func:`localization_steps`).

    Returns:
        ``global_share`` (trials localized from the start) and ``median_steps_to_localize`` (over
        those trials; NaN when none localized); ``recovered_share`` and
        ``median_steps_to_recover`` the same for the trials with a kidnapping (absent when none
        has one); ``trials`` and ``collisions``.

    Raises:
        ValueError: No trials.
    """
    if not trials:
        raise ValueError("Summarizing needs trials; got no trials.")
    outcomes = [localization_steps(trial, deadline, hold) for trial in trials]
    summary = {
        "global_share": float(np.mean([steps is not None for steps, _ in outcomes])),
        "median_steps_to_localize": _median_steps([steps for steps, _ in outcomes]),
    }
    kidnapped = [
        recovery
        for trial, (_, recovery) in zip(trials, outcomes, strict=True)
        if trial.kidnap_at is not None
    ]
    if kidnapped:
        summary["recovered_share"] = float(np.mean([steps is not None for steps in kidnapped]))
        summary["median_steps_to_recover"] = _median_steps(kidnapped)
    summary["trials"] = float(len(trials))
    summary["collisions"] = float(sum(trial.collided for trial in trials))
    return summary

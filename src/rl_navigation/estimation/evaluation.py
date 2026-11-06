"""Seeded localization trials in the simulator, scored against its ground truth.

A trial draws a layout, drives the robot with a simple wandering controller, feeds the filter
the simulated encoders and range scans, and records the true pose (:attr:`World.pose`), the
estimate, its covariance and dead reckoning on the same encoder readings.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rl_navigation.estimation.ekf import EkfLocalizer, pose_difference
from rl_navigation.estimation.maps import SensorMap
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
        beams_used: ``(T,)`` beams each update used.
        collided: Whether the run ended at a collision before its last step.
    """

    true_poses: FloatArray
    estimates: FloatArray
    covariances: FloatArray
    dead_reckoning: FloatArray
    beams_used: NDArray[np.int_]
    collided: bool


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
    iterations: int = 3,
    linearity_tol: float | None = None,
    initial_std: Sequence[float] = (0.05, 0.05, 0.05),
) -> Trial:
    """Localize a wandering robot for up to ``steps`` steps of a seeded layout.

    The seed's ``SeedSequence`` is split into independent streams for the layout, the
    controller, the world (slip and sensor noise), the encoders and the initial error. The
    filter starts at the true start plus ``N(0, diag(initial_std^2))`` with that covariance.

    Args:
        seed: The trial's seed.
        sensor: The robot's range sensor, also the filter's measurement model.
        steps: Steps to run; a collision ends the run early.
        dropout_prob: Sensor dropout probability; None uses the sensor's own.
        odometry: The simulated encoders; None uses the defaults.
        filter_odometry: The filter's encoder model; None uses ``odometry`` (a matched model).
        layout_params: How the layout is drawn; None uses the defaults.
        iterations: The filter's update iterations.
        linearity_tol: The filter's linearity tolerance; None uses its default.
        initial_std: Standard deviations of the initial error in ``x``, ``y`` and ``theta``.

    Returns:
        The trial's record.
    """
    layout_rng, control_rng, world_rng, odometry_rng, initial_rng = (
        np.random.default_rng(stream) for stream in np.random.SeedSequence(seed).spawn(5)
    )
    params = LayoutParams() if layout_params is None else layout_params
    layout = generate_layout(layout_rng, params)
    world = World(layout.arena, sensor, layout.start, layout.goal, robot_radius=params.robot_radius)
    odometry = WheelOdometry() if odometry is None else odometry
    model = odometry if filter_odometry is None else filter_odometry
    std = np.asarray(initial_std, dtype=np.float64)
    start = world.pose + std * initial_rng.standard_normal(3)
    start[2] = wrap_angle(start[2])
    ekf = EkfLocalizer(
        SensorMap(layout.arena, sensor),
        model,
        start,
        np.diag(std**2),
        iterations=iterations,
        linearity_tol=linearity_tol,
    )
    dead = start
    rows: list[tuple[FloatArray, FloatArray, FloatArray, FloatArray, int]] = []
    collided = False
    for _ in range(steps):
        v, omega = _wander(world, control_rng)
        motion = world.step(v, omega, world_rng)
        if motion.collided:
            collided = True
            break
        travel = odometry.read(motion.v, motion.omega, world.dt, odometry_rng)
        ekf.predict(travel)
        dead = unicycle_step(dead, *model.increments(travel), 1.0)
        readings, _ = world.read_sensors(world_rng, dropout_prob=dropout_prob)
        result = ekf.update(readings)
        rows.append((world.pose, ekf.pose, ekf.covariance, dead, int(result.used.sum())))
    return Trial(
        true_poses=np.array([row[0] for row in rows], dtype=np.float64).reshape(-1, 3),
        estimates=np.array([row[1] for row in rows], dtype=np.float64).reshape(-1, 3),
        covariances=np.array([row[2] for row in rows], dtype=np.float64).reshape(-1, 3, 3),
        dead_reckoning=np.array([row[3] for row in rows], dtype=np.float64).reshape(-1, 3),
        beams_used=np.array([row[4] for row in rows], dtype=np.int_),
        collided=collided,
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

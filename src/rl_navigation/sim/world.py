"""The robot in its arena: pose, motion with wheel slip and swept collision, goal and sensors."""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from rl_navigation.sim.geometry import Arena, FloatArray, clearance, segment_hits
from rl_navigation.sim.kinematics import unicycle_step
from rl_navigation.sim.sensors import RangeSensor

# Largest turn in one step, in radians, that the collision check follows. It sweeps the chord of
# the arc; the arc bulges beyond it by its sagitta, (v dt / turn) (1 - cos(turn / 2)), about
# v dt turn / 8: at most 6.3 % of the step's length at this limit (3 mm at 0.5 m/s and dt 0.1 s).
MAX_TURN_PER_STEP = 0.5


class Motion(NamedTuple):
    """What one :meth:`World.step` did: the command executed after wheel slip, and a collision.

    ``v`` and ``omega`` are the executed command (what the wheels did) even when the move was
    blocked and the robot stayed put.
    """

    v: float
    omega: float
    collided: bool


class World:
    """A disc robot with range sensors in an arena, driving towards a goal.

    Args:
        arena: The arena.
        sensor: The robot's range sensors.
        pose: ``(3,)`` start pose ``(x, y, theta)``.
        goal: ``(2,)`` goal position.
        robot_radius: The robot's radius in metres.
        goal_radius: The goal is reached when the robot's centre is closer than this, in metres.
        dt: Step length in seconds.
        wheel_slip: Largest fraction of a command lost to slip, in ``[0, 1]``.

    Raises:
        ValueError: A non-positive radius or step, a slip outside ``[0, 1]``, or a start pose
            within ``robot_radius`` of an obstacle or a wall, inside a box or outside the arena
            (already in collision).
    """

    def __init__(
        self,
        arena: Arena,
        sensor: RangeSensor,
        pose: ArrayLike,
        goal: ArrayLike,
        robot_radius: float = 0.1,
        goal_radius: float = 0.2,
        dt: float = 0.1,
        wheel_slip: float = 0.0,
    ) -> None:
        if not robot_radius > 0.0:
            raise ValueError(f"World robot_radius must be positive, got {robot_radius}.")
        if not goal_radius > 0.0:
            raise ValueError(f"World goal_radius must be positive, got {goal_radius}.")
        if not dt > 0.0:
            raise ValueError(f"World dt must be positive, got {dt}.")
        if not 0.0 <= wheel_slip <= 1.0:
            raise ValueError(f"World wheel_slip must be in [0, 1], got {wheel_slip}.")
        self.arena = arena
        self.sensor = sensor
        self.goal = np.array(goal, dtype=np.float64).reshape(2)
        self.robot_radius = float(robot_radius)
        self.goal_radius = float(goal_radius)
        self.dt = float(dt)
        self.wheel_slip = float(wheel_slip)
        self._pose = np.array(pose, dtype=np.float64).reshape(3)
        if clearance(arena, self._pose[None, :2])[0] <= self.robot_radius:
            raise ValueError(
                f"World start pose {self._pose.tolist()} is in collision: it is "
                f"{float(clearance(arena, self._pose[None, :2])[0]):.3g} m from an obstacle, within "
                f"the robot's radius {self.robot_radius} m."
            )

    @property
    def pose(self) -> FloatArray:
        """A copy of the pose ``(x, y, theta)``."""
        return self._pose.copy()

    @property
    def goal_distance(self) -> float:
        """Straight-line distance from the robot's centre to the goal, in metres."""
        return float(np.linalg.norm(self.goal - self._pose[:2]))

    @property
    def at_goal(self) -> bool:
        """Whether the robot's centre is within ``goal_radius`` of the goal."""
        return self.goal_distance < self.goal_radius

    def step(self, v: float, omega: float, rng: np.random.Generator) -> Motion:
        """Drive for one step of ``dt`` with the commanded ``v`` (m/s) and ``omega`` (rad/s).

        Slip scales each command by ``1 - wheel_slip * u`` with ``u ~ U(0, 1)``, one draw for ``v``
        and one for ``omega``, drawn even when ``wheel_slip`` is 0 so the generator advances the
        same way. The robot then moves on the exact unicycle arc; if its disc, swept along the
        chord from the old position to the new one, touches anything, it stays at the old pose.
        Over a 0.1 s step at 0.5 m/s and 2 rad/s the chord is within 1.3 mm of the arc; a turn
        beyond :data:`MAX_TURN_PER_STEP` is refused, since the arc could then pass an obstacle
        the chord misses.

        Args:
            v: Commanded forward speed.
            omega: Commanded turn rate.
            rng: Generator for the slip.

        Returns:
            The executed command and whether the move collided.

        Raises:
            ValueError: ``|omega| dt`` exceeds :data:`MAX_TURN_PER_STEP`; nothing is drawn or moved.
        """
        if abs(omega) * self.dt > MAX_TURN_PER_STEP:
            raise ValueError(
                f"World.step turns {abs(omega) * self.dt:.3g} rad in one step, more than "
                f"MAX_TURN_PER_STEP={MAX_TURN_PER_STEP} rad that the chord collision check follows: "
                "use a shorter dt or a lower turn rate."
            )
        scale = 1.0 - self.wheel_slip * rng.random(2)
        executed_v, executed_omega = float(v * scale[0]), float(omega * scale[1])
        moved = unicycle_step(self._pose, executed_v, executed_omega, self.dt)
        collided = segment_hits(self.arena, self._pose[:2], moved[:2], self.robot_radius)
        if not collided:
            self._pose = moved
        return Motion(executed_v, executed_omega, collided)

    def read_sensors(
        self,
        rng: np.random.Generator,
        noise_std: float | None = None,
        dropout_prob: float | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """One scan from the current pose; see :meth:`RangeSensor.read`.

        Returns:
            ``(readings, true_ranges)``, each ``(N,)``.
        """
        return self.sensor.read(self.arena, self._pose, rng, noise_std, dropout_prob)

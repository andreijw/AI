"""The Gymnasium environment over ``rl_navigation.sim``: drive a disc robot to a goal through
random obstacles, seeing only range sensors.

It imports Gymnasium and ``rl_navigation.sim`` only, not the training framework, so another
project can ``import rl_navigation.gym_env`` and ``gym.make`` it. Importing it registers two ids:

- ``Navigation-v0``: five discrete actions, for DQN.
- ``NavigationContinuous-v0``: two actions in ``[-1, 1]``.

Both end an episode on reaching the goal or on a collision (a termination), or after 400 steps
(a truncation). ``rl_navigation.environments`` adds the framework's noise and domain
randomization on top.
"""

import dataclasses
import math
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from rl_navigation.sim import (
    Layout,
    LayoutParams,
    RangeSensor,
    World,
    generate_layout,
    hc_sr04,
    lidar,
    wrap_angle,
)
from rl_navigation.sim import render as scene
from rl_navigation.sim.world import MAX_TURN_PER_STEP

DISCRETE_ID = "Navigation-v0"
CONTINUOUS_ID = "NavigationContinuous-v0"
MAX_EPISODE_STEPS = 400
SENSORS = ("hc_sr04", "lidar")

# Each discrete action's (forward speed, turn rate) as fractions of (v_max, omega_max): straight
# ahead, forward arcs left and right at half speed, turns on the spot left and right.
DISCRETE_ACTIONS = np.array([(1.0, 0.0), (0.5, 1.0), (0.5, -1.0), (0.0, 1.0), (0.0, -1.0)])

# Observation entries before the readings: goal distance, cos and sin of its bearing, then the
# last executed forward speed and turn rate.
GOAL_AND_COMMAND = 5


def make_sensor(sensor: str, lidar_beams: int) -> RangeSensor:
    """The named sensor preset: five HC-SR04 sonars, or a full-circle lidar of ``lidar_beams``.

    Raises:
        ValueError: An unknown name, or fewer than one lidar beam.
    """
    if sensor == "hc_sr04":
        return hc_sr04()
    if sensor == "lidar":
        return lidar(num_beams=lidar_beams)
    raise ValueError(f"sensor must be one of {', '.join(SENSORS)}, got {sensor!r}")


class NavigationGymEnv(gym.Env[NDArray[np.float32], Any]):
    """A disc robot with range sensors drives to a goal in a randomly drawn arena.

    **Episode.** Each reset draws a layout (obstacles, start pose, goal) from the environment's
    generator with :func:`~rl_navigation.sim.generate_layout`, then draws one integer that seeds
    the episode's own generator for wheel slip and sensor noise. Layouts therefore follow one
    another in the same order however long each episode lasts. ``reset(options={"layout": ...})``
    plays a given :class:`~rl_navigation.sim.Layout` instead (tests, or planning on a known map).

    **Actions.** Continuous: ``a`` in ``[-1, 1]^2``, clipped to it first, gives forward speed
    ``v = v_max (a0 + 1) / 2`` in ``[0, v_max]`` (no reverse: the sonars face forward) and turn
    rate ``omega = omega_max a1``. Discrete: the rows of :data:`DISCRETE_ACTIONS`, as fractions
    of ``(v_max, omega_max)``.

    **Observation** (float32, every entry in ``[-1, 1]``): goal distance over the arena's
    diagonal; cos and sin of the goal's bearing from the heading (no jump at +-pi); the last
    executed command ``(v / v_max, omega / omega_max)``, slip included and zero after a reset;
    then each reading over the sensor's maximum range, in mount order. 10 values with the
    HC-SR04 sonars, 5 + ``lidar_beams`` with the lidar.

    **Reward**: ``progress_weight (d_prev - d) / d0 - step_cost - collision_penalty [collided]``,
    with ``d`` the distance to the goal and ``d0`` the episode's starting one. Progress sums to
    ``progress_weight`` over a whole trip whatever its length, so a success is worth about the
    same on every layout, and every collision costs more than a whole episode of step costs.
    A collision leaves the robot in place and ends the episode, as reaching the goal does.
    ``info`` holds ``is_success``, ``collision`` and ``goal_distance``.

    Args:
        render_mode: None or ``"rgb_array"``.
        continuous: Two continuous actions, or five discrete ones.
        sensor: ``"hc_sr04"`` or ``"lidar"``.
        lidar_beams: Beams of the lidar (unused with the sonars).
        num_obstacles: Obstacles per layout; None takes ``layout_params``'s.
        layout_params: How layouts are drawn, and the robot's radius; None for the defaults.
        wheel_slip: Largest fraction of a command lost to slip, in ``[0, 1]``.
        sensor_noise_std: Range noise in metres; None takes the sensor's own.
        dropout_prob: Probability that a reading is lost; None takes the sensor's own.
        v_max: Top forward speed in m/s.
        omega_max: Top turn rate in rad/s.
        dt: Step length in seconds.
        goal_radius: The goal is reached within this distance, in metres.
        progress_weight: Reward for covering the whole start-to-goal distance.
        step_cost: Reward taken off every step.
        collision_penalty: Reward taken off a collision.

    ``wheel_slip``, ``sensor_noise_std``, ``dropout_prob`` and ``num_obstacles`` are public
    attributes that the next reset reads, so a wrapper can randomize them between episodes.

    Raises:
        ValueError: An unknown sensor, a non-positive speed or step, a turn per step beyond
            :data:`~rl_navigation.sim.world.MAX_TURN_PER_STEP`, an obstacle count the layout
            parameters refuse, a ``wheel_slip`` or ``dropout_prob`` outside ``[0, 1]``, or a
            negative ``sensor_noise_std``. :meth:`reset` refuses a given layout whose start is
            already within ``goal_radius`` of the goal.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        continuous: bool = True,
        sensor: str = "hc_sr04",
        lidar_beams: int = 64,
        num_obstacles: int | None = None,
        layout_params: LayoutParams | None = None,
        wheel_slip: float = 0.0,
        sensor_noise_std: float | None = None,
        dropout_prob: float | None = None,
        v_max: float = 0.5,
        omega_max: float = 2.0,
        dt: float = 0.1,
        goal_radius: float = 0.2,
        progress_weight: float = 10.0,
        step_cost: float = 0.01,
        collision_penalty: float = 10.0,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be None or 'rgb_array', got {render_mode!r}")
        if not (v_max > 0.0 and omega_max > 0.0 and dt > 0.0):
            raise ValueError(
                f"v_max, omega_max and dt must be positive, got {v_max}, {omega_max} and {dt}"
            )
        if omega_max * dt > MAX_TURN_PER_STEP:
            raise ValueError(
                f"omega_max * dt = {omega_max * dt:.3g} rad per step exceeds MAX_TURN_PER_STEP="
                f"{MAX_TURN_PER_STEP}: lower omega_max or dt"
            )
        self.render_mode = render_mode
        self.continuous = continuous
        self.sensor = make_sensor(sensor, lidar_beams)
        self.layout_params = LayoutParams() if layout_params is None else layout_params
        self.num_obstacles = (
            self.layout_params.num_obstacles if num_obstacles is None else num_obstacles
        )
        self._episode_params()  # refuse an impossible obstacle count now, not at the first reset
        self.wheel_slip = wheel_slip
        self.sensor_noise_std = (
            self.sensor.noise_std if sensor_noise_std is None else sensor_noise_std
        )
        self.dropout_prob = self.sensor.dropout_prob if dropout_prob is None else dropout_prob
        for name, value, high in (
            ("wheel_slip", self.wheel_slip, 1.0),
            ("sensor_noise_std", self.sensor_noise_std, math.inf),
            ("dropout_prob", self.dropout_prob, 1.0),
        ):
            if not 0.0 <= value <= high:
                raise ValueError(f"{name} must be in [0, {high:g}], got {value!r}")
        self.v_max, self.omega_max, self.dt = float(v_max), float(omega_max), float(dt)
        self.goal_radius = float(goal_radius)
        self.progress_weight = float(progress_weight)
        self.step_cost = float(step_cost)
        self.collision_penalty = float(collision_penalty)

        size = GOAL_AND_COMMAND + self.sensor.num_readings
        low = np.zeros(size, dtype=np.float32)
        low[[1, 2, 4]] = -1.0  # the bearing's cos and sin, and the turn rate
        self._obs_low, self._obs_high = low, np.ones(size, dtype=np.float32)
        self.observation_space = gym.spaces.Box(self._obs_low, self._obs_high)
        self.action_space: gym.Space[Any] = (
            gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
            if continuous
            else gym.spaces.Discrete(len(DISCRETE_ACTIONS))
        )
        self._diagonal = math.hypot(self.layout_params.width, self.layout_params.height)
        self.layout: Layout | None = None
        self.world: World | None = None
        self._episode_rng = np.random.default_rng(0)  # replaced at every reset
        self._start_distance = 1.0
        self._distance = 1.0
        self._command = np.zeros(2)
        self._readings = np.zeros(self.sensor.num_readings)
        self._background: NDArray[np.uint8] | None = None

    def _episode_params(self) -> LayoutParams:
        """The layout parameters with this episode's obstacle count."""
        count = int(round(self.num_obstacles))
        if count == self.layout_params.num_obstacles:
            return self.layout_params
        return dataclasses.replace(self.layout_params, num_obstacles=count)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        layout = (options or {}).get("layout")
        if layout is None:
            layout = generate_layout(self.np_random, self._episode_params())
        self._episode_rng = np.random.default_rng(int(self.np_random.integers(2**63)))
        self.layout = layout
        self.world = World(
            layout.arena,
            self.sensor,
            layout.start,
            layout.goal,
            robot_radius=self.layout_params.robot_radius,
            goal_radius=self.goal_radius,
            dt=self.dt,
            wheel_slip=self.wheel_slip,
        )
        if self.world.at_goal:
            raise ValueError(
                f"the layout's start is {self.world.goal_distance:.3g} m from the goal, within "
                f"goal_radius={self.goal_radius} m: the episode would end before it starts"
            )
        self._start_distance = self.world.goal_distance
        self._distance = self.world.goal_distance
        self._command = np.zeros(2)
        self._background = None
        return self._observe(), self._info(collided=False)

    def _command_for(self, action: Any) -> tuple[float, float]:
        """The ``(v, omega)`` an action commands."""
        if self.continuous:
            a = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
            return self.v_max * (a[0] + 1.0) / 2.0, self.omega_max * a[1]
        if not self.action_space.contains(action):
            raise ValueError(
                f"action must be an integer in [0, {len(DISCRETE_ACTIONS)}), got {action!r}"
            )
        fraction_v, fraction_omega = DISCRETE_ACTIONS[int(action)]
        return self.v_max * fraction_v, self.omega_max * fraction_omega

    def step(self, action: Any) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.world is None:
            raise RuntimeError("NavigationGymEnv.step called before reset")
        v, omega = self._command_for(action)
        motion = self.world.step(v, omega, self._episode_rng)
        self._command = np.array([motion.v / self.v_max, motion.omega / self.omega_max])
        previous, self._distance = self._distance, self.world.goal_distance
        reward = (
            self.progress_weight * (previous - self._distance) / self._start_distance
            - self.step_cost
            - self.collision_penalty * motion.collided
        )
        terminated = motion.collided or self.world.at_goal
        return self._observe(), float(reward), bool(terminated), False, self._info(motion.collided)

    def _observe(self) -> NDArray[np.float32]:
        """Read the sensors and build the observation (see the class docstring)."""
        assert self.world is not None
        readings, _ = self.world.read_sensors(
            self._episode_rng, self.sensor_noise_std, self.dropout_prob
        )
        self._readings = readings
        pose = self.world.pose
        to_goal = self.world.goal - pose[:2]
        bearing = float(wrap_angle(math.atan2(to_goal[1], to_goal[0]) - pose[2]))
        obs = np.concatenate(
            [
                [self._distance / self._diagonal, math.cos(bearing), math.sin(bearing)],
                self._command,
                readings / self.sensor.max_range,
            ]
        )
        # Rounding can put a ratio a hair outside [-1, 1]: keep it inside the space.
        return np.clip(obs, self._obs_low, self._obs_high).astype(np.float32)

    def _info(self, collided: bool) -> dict[str, Any]:
        assert self.world is not None
        return {
            "is_success": bool(self.world.at_goal),
            "collision": bool(collided),
            "goal_distance": self._distance,
        }

    def _beams(self) -> NDArray[np.float64]:
        """``(N, 2, 2)``: each mount's position and the point it last read."""
        assert self.world is not None
        x, y, theta = self.world.pose
        angles = theta + self.sensor.mount_angles
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        mounts = np.array([x, y]) + self.sensor.mount_radius * directions
        return np.stack([mounts, mounts + self._readings[:, None] * directions], axis=1)

    def render(self) -> NDArray[np.uint8] | None:
        """The current frame as ``(rows, cols, 3)`` uint8 at 80 pixels per metre, or None
        without ``render_mode="rgb_array"``."""
        if self.render_mode != "rgb_array" or self.world is None:
            return None
        if self._background is None:  # once per layout, and only when rendering
            self._background = scene.arena_image(self.world.arena)
        return scene.draw_scene(
            self._background,
            self.world.pose,
            self.world.goal,
            self.world.robot_radius,
            self.world.goal_radius,
            self._beams(),
        )


for _env_id, _continuous in ((DISCRETE_ID, False), (CONTINUOUS_ID, True)):
    if _env_id not in gym.registry:
        gym.register(
            id=_env_id,
            entry_point="rl_navigation.gym_env:NavigationGymEnv",
            max_episode_steps=MAX_EPISODE_STEPS,
            kwargs={"continuous": _continuous},
        )

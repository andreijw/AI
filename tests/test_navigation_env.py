"""The navigation environment: the raw Gymnasium env over ``rl_navigation.sim``
(``rl_navigation.gym_env``), its rgb_array renderer (``rl_navigation.sim.render``), the framework
wrapper with noise and domain randomization (``rl_navigation.environments``) and the two
challenges (``rl_navigation.benchmark``)."""

import contextlib
import json
import math
import os
import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_core.robustness import (
    Perturbation,
    canonical_setting,
    perturbed_section,
    physics_defaults,
)
from rl_core.training.curriculum import curriculum_from_config
from rl_navigation.benchmark import NavigationChallenge, NavigationContinuousChallenge
from rl_navigation.environments import (
    NavigationEnv,
    make_continuous_env_from_config,
    make_env_from_config,
)
from rl_navigation.environments.navigation_env import obstacle_count
from rl_navigation.gym_env import NavigationGymEnv
from rl_navigation.sim import Arena, Layout
from rl_navigation.sim import render as nav_render
from tests.helpers import minimal_config, run_main

DISCRETE_ID, CONTINUOUS_ID = "Navigation-v0", "NavigationContinuous-v0"
IDS = (DISCRETE_ID, CONTINUOUS_ID)
SENSORS = ("hc_sr04", "lidar")
DIAGONAL = 5.0 * math.sqrt(2.0)  # the default 5 x 5 m arena
FORWARD = {DISCRETE_ID: 0, CONTINUOUS_ID: np.array([1.0, 0.0], dtype=np.float32)}
SPIN_LEFT = {DISCRETE_ID: 3, CONTINUOUS_ID: np.array([-1.0, 1.0], dtype=np.float32)}


def _layout(start, goal, boxes=()):
    """A fixed layout in the empty 5 x 5 m arena (or with the given boxes)."""
    arena = Arena(5.0, 5.0, boxes=np.array(boxes, dtype=np.float64).reshape(-1, 4))
    return Layout(arena, np.array(start, dtype=np.float64), np.array(goal, dtype=np.float64))


def _raw(continuous=True, **kwargs):
    return NavigationGymEnv(continuous=continuous, **kwargs)


def _obstacles(layout):
    return len(layout.arena.circle_radii) + len(layout.arena.boxes)


# ---------------------------------------------------------------------------
# The raw Gymnasium environment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("continuous", [False, True], ids=["discrete", "continuous"])
@pytest.mark.parametrize("sensor", SENSORS)
@pytest.mark.parametrize("render_mode", [None, "rgb_array"])
def test_the_raw_env_passes_gymnasiums_checker_without_a_warning(continuous, sensor, render_mode):
    """On the raw env as gym.make builds it, which gives it the spec the render check reads."""
    env = gym.make(
        CONTINUOUS_ID if continuous else DISCRETE_ID, sensor=sensor, render_mode=render_mode
    )
    try:
        assert isinstance(env.unwrapped, NavigationGymEnv)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_env(env.unwrapped, skip_render_check=render_mode is None)
    finally:
        env.close()


@pytest.mark.parametrize(("sensor", "size"), [("hc_sr04", 10), ("lidar", 69)])
@pytest.mark.parametrize("env_id", IDS)
def test_both_ids_are_registered_with_the_spaces_and_step_limit_of_the_plan(env_id, sensor, size):
    """Goal (3), last command (2), then one reading per sensor mount: 5 sonars, 64 lidar beams."""
    env = gym.make(env_id, sensor=sensor)
    try:
        assert env.observation_space.shape == (size,)
        assert env.observation_space.dtype == np.float32
        np.testing.assert_array_equal(
            env.observation_space.low[:5], [0.0, -1.0, -1.0, 0.0, -1.0], "goal, then command"
        )
        np.testing.assert_array_equal(env.observation_space.high, np.ones(size, np.float32))
        np.testing.assert_array_equal(env.observation_space.low[5:], np.zeros(size - 5))
        if env_id == DISCRETE_ID:
            assert env.action_space == gym.spaces.Discrete(5)
        else:
            assert env.action_space == gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        assert env.spec is not None and env.spec.max_episode_steps == 400
        assert env.metadata["render_modes"] == ["rgb_array"]
        assert env.metadata["render_fps"] == 10, "one frame per 0.1 s step"
    finally:
        env.close()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"render_mode": "human"}, "render_mode"),
        ({"v_max": 0.0}, "positive"),
        ({"dt": -0.1}, "positive"),
        ({"omega_max": 6.0}, "MAX_TURN_PER_STEP"),
    ],
)
def test_the_raw_env_refuses_settings_it_cannot_simulate(kwargs, match):
    """A human window comes from gym.make's HumanRendering over rgb_array, not from the env;
    6 rad/s for 0.1 s turns 0.6 rad, beyond what the swept collision check follows."""
    with pytest.raises(ValueError, match=match):
        _raw(**kwargs)


@pytest.mark.parametrize(
    ("setting", "value"),
    [("wheel_slip", 2.0), ("wheel_slip", -0.1), ("sensor_noise_std", -0.1), ("dropout_prob", 1.5)],
)
def test_gym_make_refuses_a_slip_or_sensor_setting_out_of_range(setting, value):
    """At construction, not at the first reset or reading."""
    with pytest.raises(ValueError, match=setting):
        gym.make(CONTINUOUS_ID, **{setting: value})


def test_a_given_layout_whose_start_is_already_at_the_goal_is_refused():
    """Progress is divided by the start distance: a start inside the goal radius would make it
    explode, and the episode would be over before its first step."""
    env = _raw()
    try:
        with pytest.raises(ValueError, match="goal_radius"):
            env.reset(options={"layout": _layout((1.0, 1.0, 0.0), (1.1, 1.0))})
    finally:
        env.close()


def test_the_raw_env_refuses_a_step_before_reset_and_an_unknown_discrete_action():
    env = _raw(continuous=False)
    try:
        with pytest.raises(RuntimeError, match="before reset"):
            env.step(0)
        env.reset(seed=0)
        with pytest.raises(ValueError, match="action must be an integer"):
            env.step(5)
    finally:
        env.close()


def test_the_wrapper_refuses_another_environment_id():
    with pytest.raises(ValueError, match="Navigation-v0"):
        NavigationEnv(env_name="CartPole-v1")


def test_an_unknown_sensor_is_refused_naming_the_known_ones():
    with pytest.raises(ValueError, match="hc_sr04.*lidar"):
        _raw(sensor="sonar")


@pytest.mark.parametrize("sensor", SENSORS)
def test_a_forward_step_gives_the_hand_computed_reward_and_observation(sensor):
    """Start (1, 1) facing the goal at (4, 1): d0 = 3. Full speed 0.5 m/s for 0.1 s moves 0.05 m,
    so the reward is 10 * 0.05 / 3 - 0.01 = 0.156667 and the goal is 2.95 m away."""
    env = _raw(sensor=sensor)
    try:
        obs, info = env.reset(seed=0, options={"layout": _layout((1.0, 1.0, 0.0), (4.0, 1.0))})
        np.testing.assert_allclose(obs[:5], [3.0 / DIAGONAL, 1.0, 0.0, 0.0, 0.0], atol=1e-6)
        assert info["goal_distance"] == pytest.approx(3.0)

        obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], np.float32))
    finally:
        env.close()
    assert reward == pytest.approx(10.0 * 0.05 / 3.0 - 0.01, abs=1e-9)
    assert reward == pytest.approx(0.156667, abs=1e-6)
    np.testing.assert_allclose(obs[:5], [2.95 / DIAGONAL, 1.0, 0.0, 1.0, 0.0], atol=1e-6)
    assert not terminated and not truncated
    assert info == {"is_success": False, "collision": False, "goal_distance": pytest.approx(2.95)}


@pytest.mark.parametrize(
    ("heading", "bearing"),
    [(0.0, (1.0, 0.0)), (-math.pi / 2, (0.0, 1.0)), (math.pi / 2, (0.0, -1.0))],
    ids=["ahead", "left", "right"],
)
def test_the_goal_bearing_is_in_the_robot_frame(heading, bearing):
    """The goal due east of a robot facing south lies 90 degrees to its left: (cos, sin) = (0, 1)."""
    env = _raw()
    try:
        obs, _ = env.reset(options={"layout": _layout((1.0, 1.0, heading), (4.0, 1.0))})
    finally:
        env.close()
    np.testing.assert_allclose(obs[1:3], bearing, atol=1e-6)


def test_the_bearing_has_no_jump_where_the_angle_wraps():
    """A goal straight behind is at +pi or -pi: the observation is the same on either side."""
    env = _raw()
    try:
        observed = [
            env.reset(options={"layout": _layout((2.0, 1.0, heading), (4.0, 1.0))})[0][1:3]
            for heading in (math.pi - 1e-6, -math.pi + 1e-6)
        ]
    finally:
        env.close()
    np.testing.assert_allclose(observed[0], [-1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(observed[0], observed[1], atol=1e-5)


@pytest.mark.parametrize(
    ("sensor", "index", "expected"), [("hc_sr04", 7, 0.725), ("lidar", 5, 0.75)]
)
def test_readings_are_ranges_over_the_maximum_range_in_mount_order(sensor, index, expected):
    """From (2, 1) facing the east wall at x = 5: the middle sonar (index 5 + 2) sits on the 0.1 m
    rim and reads 2.9 m of its 4 m; lidar beam 0 starts at the centre and reads 3 m."""
    env = _raw(sensor=sensor, sensor_noise_std=0.0, dropout_prob=0.0)
    try:
        obs, _ = env.reset(options={"layout": _layout((2.0, 1.0, 0.0), (4.0, 4.0))})
    finally:
        env.close()
    assert obs[index] == pytest.approx(expected, abs=1e-6)


def test_a_dropout_reads_the_maximum_range():
    env = _raw(dropout_prob=1.0)
    try:
        obs, _ = env.reset(seed=1)
    finally:
        env.close()
    np.testing.assert_array_equal(obs[5:], np.ones(5, np.float32))


def test_the_sensor_defaults_are_the_sensors_own_and_public():
    """physics_defaults reads these attributes: the HC-SR04 placeholders, no slip, 6 obstacles."""
    env = _raw()
    try:
        assert (env.wheel_slip, env.sensor_noise_std, env.dropout_prob) == (0.0, 0.01, 0.02)
        assert env.num_obstacles == 6
    finally:
        env.close()


@pytest.mark.parametrize(
    ("action", "command"),
    [(0, (1.0, 0.0)), (1, (0.5, 1.0)), (2, (0.5, -1.0)), (3, (0.0, 1.0)), (4, (0.0, -1.0))],
)
def test_each_discrete_action_drives_its_fraction_of_the_top_speeds(action, command):
    env = _raw(continuous=False)
    try:
        env.reset(options={"layout": _layout((2.5, 2.5, 0.0), (4.5, 4.5))})
        obs, *_ = env.step(action)
    finally:
        env.close()
    np.testing.assert_allclose(obs[3:5], command, atol=1e-6)


@pytest.mark.parametrize(
    ("action", "command"),
    [((1.0, 0.0), (1.0, 0.0)), ((0.0, -0.5), (0.5, -0.5)), ((-1.0, 1.0), (0.0, 1.0))],
)
def test_a_continuous_action_maps_to_forward_speed_and_turn_rate(action, command):
    """v = v_max (a0 + 1) / 2 in [0, v_max], never reverse; omega = omega_max a1."""
    env = _raw()
    try:
        env.reset(options={"layout": _layout((2.5, 2.5, 0.0), (4.5, 4.5))})
        obs, *_ = env.step(np.array(action, np.float32))
    finally:
        env.close()
    np.testing.assert_allclose(obs[3:5], command, atol=1e-6)


def test_an_action_outside_the_box_is_clipped_before_it_is_mapped():
    """A PPO sample of (3, 3) must drive as (1, 1), not turn 0.6 rad in a step (which World refuses)
    nor drive faster; (-3, 0) must stand still, not reverse."""
    layout = _layout((2.5, 2.5, 0.0), (4.5, 4.5))
    results = []
    for action in ((3.0, 3.0), (1.0, 1.0), (-3.0, 0.0)):
        env = _raw()
        try:
            env.reset(seed=4, options={"layout": layout})
            results.append(env.step(np.array(action, np.float32)))
        finally:
            env.close()
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
    np.testing.assert_allclose(results[2][0][3:5], [0.0, 0.0], atol=1e-6)
    assert results[2][4]["goal_distance"] == pytest.approx(math.hypot(2.0, 2.0)), "it stood still"


def test_driving_into_a_wall_ends_the_episode_with_the_collision_penalty():
    """0.13 m from the west wall facing it: the 0.05 m step would leave 0.08 m < 0.1 m radius.
    The robot stays put, so the reward is 0 progress - 0.01 - 10 = -10.01."""
    env = _raw()
    try:
        env.reset(options={"layout": _layout((0.13, 2.5, math.pi), (4.0, 2.5))})
        obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], np.float32))
    finally:
        env.close()
    assert reward == pytest.approx(-10.01)
    assert terminated and not truncated
    assert info["collision"] and not info["is_success"]
    assert info["goal_distance"] == pytest.approx(3.87), "a blocked move leaves the robot in place"


def test_reaching_the_goal_ends_the_episode_as_a_success():
    """0.27 m from the goal: 0.22 m after one step (outside the 0.2 m goal radius), 0.17 m after
    two. Each step earns 10 * 0.05 / 0.27 - 0.01."""
    env = _raw()
    try:
        env.reset(options={"layout": _layout((1.0, 1.0, 0.0), (1.27, 1.0))})
        first = env.step(np.array([1.0, 0.0], np.float32))
        second = env.step(np.array([1.0, 0.0], np.float32))
    finally:
        env.close()
    assert not first[2] and not first[4]["is_success"], "0.22 m is not yet at the goal"
    assert second[2] and not second[3]
    assert second[4]["is_success"] and not second[4]["collision"]
    assert first[1] + second[1] == pytest.approx(2 * (10.0 * 0.05 / 0.27 - 0.01))


@pytest.mark.parametrize("env_id", IDS)
def test_spinning_in_place_is_truncated_at_400_steps_costing_four(env_id):
    env = gym.make(env_id)
    try:
        env.reset(seed=0, options={"layout": _layout((2.5, 2.5, 0.0), (4.5, 4.5))})
        total, steps, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(SPIN_LEFT[env_id])
            total += reward
            steps += 1
    finally:
        env.close()
    assert (steps, terminated, truncated) == (400, False, True)
    assert total == pytest.approx(-4.0)


def test_the_last_command_is_the_executed_one_with_slip_and_zero_after_reset():
    env = _raw(wheel_slip=1.0)
    try:
        env.reset(seed=3, options={"layout": _layout((2.5, 2.5, 0.0), (4.5, 4.5))})
        obs, *_ = env.step(np.array([1.0, 1.0], np.float32))
        after_reset, _ = env.reset(seed=3)
    finally:
        env.close()
    assert 0.0 <= obs[3] < 1.0 and 0.0 <= obs[4] < 1.0, "slip takes a random share off each"
    np.testing.assert_array_equal(after_reset[3:5], [0.0, 0.0])


def _play_raw(seed, actions):
    env = _raw()
    try:
        seen = [env.reset(seed=seed)[0]]
        for action in actions:
            obs, reward, terminated, truncated, _ = env.step(action)
            seen.append(np.append(obs, reward))
            if terminated or truncated:
                seen.append(env.reset()[0])
        return seen
    finally:
        env.close()


def test_the_same_seed_replays_everything_and_another_seed_gives_another_layout():
    actions = [np.array([0.5, 0.3 * (-1) ** i], np.float32) for i in range(40)]
    first, second = _play_raw(11, actions), _play_raw(11, actions)
    for index, (a, b) in enumerate(zip(first, second, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"same-seed runs diverged at {index}")
    starts = []
    for seed in (11, 12):
        env = _raw()
        try:
            env.reset(seed=seed)
            starts.append(np.concatenate([env.layout.start, env.layout.goal]))
        finally:
            env.close()
    assert not np.array_equal(starts[0], starts[1]), "a different seed should draw another layout"


def test_the_next_layout_does_not_depend_on_how_long_the_episode_ran():
    """Steps draw from the episode's own generator, so episode k's layout is the same whether
    episode k - 1 lasted one step or thirty."""
    layouts = []
    for steps in (1, 30):
        env = _raw(wheel_slip=0.5)
        try:
            env.reset(seed=21)
            for _ in range(steps):
                if env.step(np.array([-1.0, 1.0], np.float32))[2]:
                    break
            env.reset()
            layouts.append(env.layout)
        finally:
            env.close()
    np.testing.assert_array_equal(layouts[0].start, layouts[1].start)
    np.testing.assert_array_equal(layouts[0].goal, layouts[1].goal)
    np.testing.assert_array_equal(layouts[0].arena.segments, layouts[1].arena.segments)


def test_a_drawn_layout_keeps_the_raw_envs_obstacle_count():
    env = _raw(num_obstacles=3)
    try:
        env.reset(seed=2)
        assert _obstacles(env.layout) == 3
        env.num_obstacles = 0
        env.reset()
        assert _obstacles(env.layout) == 0, "a count set between episodes applies at the reset"
    finally:
        env.close()


def test_an_obstacle_count_above_the_layouts_capacity_is_refused():
    """The largest obstacle covers 1 m^2, so the 25 m^2 arena takes at most 12 at max_fill 0.5."""
    with pytest.raises(ValueError, match="num_obstacles"):
        _raw(num_obstacles=13)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pixel(frame, x, y):
    """The colour at arena point (x, y): 80 px/m, y up, so row 0 is the top wall."""
    return tuple(frame[frame.shape[0] - 1 - int(y * 80), int(x * 80)])


def test_the_rgb_array_frame_draws_obstacles_goal_robot_and_heading_y_up():
    env = _raw(render_mode="rgb_array")
    try:
        env.reset(
            options={"layout": _layout((1.0, 1.0, 0.0), (4.0, 1.0), boxes=[(3.0, 3.0, 4.0, 4.0)])}
        )
        frame = env.render()
    finally:
        env.close()
    assert frame.shape == (400, 400, 3) and frame.dtype == np.uint8
    assert _pixel(frame, 3.5, 3.5) == nav_render.OBSTACLE_COLOUR, "box in the upper right"
    assert _pixel(frame, 3.5, 1.5) == nav_render.FREE_COLOUR, "free below the box"
    assert _pixel(frame, 0.5, 4.5) == nav_render.FREE_COLOUR
    assert _pixel(frame, 4.0, 1.0) == nav_render.GOAL_COLOUR
    assert _pixel(frame, 1.0, 0.95) == nav_render.ROBOT_COLOUR
    assert _pixel(frame, 1.07, 1.0) == nav_render.HEADING_COLOUR, "the heading points along +x"
    assert _pixel(frame, 0.93, 1.0) == nav_render.ROBOT_COLOUR, "and not behind"
    assert _pixel(frame, 2.5, 1.0) == nav_render.BEAM_COLOUR, "the forward sonar's beam"


def test_without_a_render_mode_render_returns_nothing():
    env = _raw()
    try:
        env.reset(seed=0)
        assert env.render() is None
    finally:
        env.close()


# ---------------------------------------------------------------------------
# The framework wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", IDS)
def test_the_wrapper_builds_both_ids_with_the_sensor_it_is_given(env_id):
    with contextlib.closing(NavigationEnv(env_name=env_id, sensor="lidar", lidar_beams=8)) as env:
        assert env.observation_space.shape == (13,)
        assert env.max_episode_steps == 400


@pytest.mark.parametrize(
    ("env_id", "setting"),
    [(DISCRETE_ID, "action_noise_std"), (CONTINUOUS_ID, "action_noise_prob")],
)
def test_each_id_refuses_the_other_action_spaces_noise(env_id, setting):
    with pytest.raises(ValueError, match=setting):
        NavigationEnv(env_name=env_id, **{setting: 0.1})


def _executed_commands(env, action, episodes=20):
    """The executed command of one step from the same pose, ``episodes`` times."""
    layout = _layout((2.5, 2.5, 0.0), (4.5, 4.5))
    commands = []
    for _ in range(episodes):
        env.reset(options={"layout": layout})
        commands.append(tuple(env.step(action)[0][3:5]))
    return np.array(commands)


def test_continuous_action_noise_perturbs_the_clipped_action_and_stays_in_bounds():
    """(5, 0) is clipped to (1, 0) first, so the noise lowers v about half the time; noise added
    before clipping would leave v at full speed every time."""
    with contextlib.closing(
        NavigationEnv(env_name=CONTINUOUS_ID, seed=1, action_noise_std=0.5)
    ) as env:
        commands = _executed_commands(env, np.array([5.0, 0.0], np.float32))
    assert (commands[:, 0] < 0.999).sum() >= 4, "the noise should lower the speed some steps"
    assert (commands[:, 0] >= 0.0).all() and (commands[:, 0] <= 1.0).all()
    assert (np.abs(commands[:, 1]) <= 1.0).all() and np.ptp(commands[:, 1]) > 0.1


def test_discrete_action_noise_replaces_the_action_with_another_one():
    with contextlib.closing(
        NavigationEnv(env_name=DISCRETE_ID, seed=1, action_noise_prob=1.0)
    ) as env:
        commands = _executed_commands(env, 0)
    chosen = (1.0, 0.0)
    assert not any(np.allclose(c, chosen) for c in commands), "every action should be replaced"
    assert len({tuple(np.round(c, 6)) for c in commands}) >= 3, "by one of the others, at random"


def test_obstacle_count_draws_integers_between_the_rounded_ends():
    """(0.6, 2.4) rounds to 1 and 2: both are drawn, and nothing else."""
    rng = np.random.default_rng(0)
    draws = [obstacle_count(rng, 0.6, 2.4) for _ in range(400)]
    assert set(draws) == {1, 2}
    assert all(type(n) is int for n in draws)
    assert {obstacle_count(rng, 3.0, 3.0) for _ in range(20)} == {3}


def test_a_pinned_obstacle_count_holds_on_every_reset():
    with contextlib.closing(
        NavigationEnv(seed=4, domain_randomization={"num_obstacles": (2.0, 2.0)})
    ) as env:
        for _ in range(4):
            env.reset()
            assert _obstacles(env.env.unwrapped.layout) == 2
            assert env.env.unwrapped.num_obstacles == 2


def test_randomized_sensor_settings_reach_the_readings_and_slip_reaches_the_world():
    randomization = {
        "sensor_noise_std": (0.0, 0.0),
        "dropout_prob": (1.0, 1.0),
        "wheel_slip": (0.3, 0.3),
    }
    with contextlib.closing(NavigationEnv(seed=4, domain_randomization=randomization)) as env:
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs[5:], np.ones(5, np.float32), "every reading dropped")
        assert env.env.unwrapped.world.wheel_slip == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("randomization", "match"),
    [
        ({"wheel_slip": (0.5, 1.5)}, "wheel_slip"),
        ({"dropout_prob": (-0.1, 0.5)}, "dropout_prob"),
        ({"sensor_noise_std": (-0.1, 0.0)}, "sensor_noise_std"),
        ({"num_obstacles": (-1.0, 2.0)}, "num_obstacles"),
        ({"num_obstacles": (0.0, 13.0)}, "num_obstacles"),
        ({"gravity": (9.0, 10.0)}, "Unsupported"),
    ],
)
def test_a_randomization_outside_a_parameters_bounds_is_refused(randomization, match):
    with pytest.raises(ValueError, match=match):
        NavigationEnv(domain_randomization=randomization)


def test_a_scheduled_obstacle_count_above_capacity_is_refused_and_nothing_changes():
    with contextlib.closing(
        NavigationEnv(domain_randomization={"num_obstacles": (0.0, 6.0)})
    ) as env:
        with pytest.raises(ValueError, match="num_obstacles"):
            env.apply_settings({"domain_randomization": {"num_obstacles": (0.0, 13.0)}})
        assert env.domain_randomization == {"num_obstacles": (0.0, 6.0)}
        env.apply_settings({"domain_randomization": {"num_obstacles": (0.0, 12.0)}})
        assert env.domain_randomization == {"num_obstacles": (0.0, 12.0)}


def test_the_obstacle_capacity_is_part_of_the_settings_rule():
    """check_setting is what the robustness sweep and the curriculum consult: it must refuse more
    obstacles than the layout holds, not only the constructor."""
    with pytest.raises(ValueError, match="num_obstacles"):
        NavigationEnv.check_setting("domain_randomization", {"num_obstacles": (0.0, 13.0)})
    assert NavigationEnv.check_setting("domain_randomization", {"num_obstacles": (0.0, 12.0)}) == {
        "num_obstacles": (0.0, 12.0)
    }


def test_a_robustness_perturbation_above_the_obstacle_capacity_is_refused_before_any_episode():
    with (
        contextlib.closing(NavigationEnv()) as env,
        pytest.raises(ValueError, match="num_obstacles"),
    ):
        perturbed_section({}, env, Perturbation("num_obstacles", 13.0))


def test_a_curriculum_starting_above_the_obstacle_capacity_is_refused_when_it_is_built():
    with contextlib.closing(
        NavigationEnv(domain_randomization={"num_obstacles": (0.0, 6.0)})
    ) as env:
        section = {
            "mode": "linear",
            "episodes": 2,
            "start": {"domain_randomization": {"num_obstacles": (13.0, 13.0)}},
        }
        with pytest.raises(ValueError, match="num_obstacles"):
            curriculum_from_config({"curriculum": section}, env)


@pytest.mark.parametrize(
    "name", ["wheel_slip", "sensor_noise_std", "dropout_prob", "num_obstacles"]
)
def test_the_robustness_sweep_can_pin_each_parameter(name):
    with contextlib.closing(NavigationEnv()) as env:
        assert canonical_setting(env, name) == name


@pytest.mark.parametrize("challenge", [NavigationChallenge, NavigationContinuousChallenge])
def test_physics_defaults_are_the_unrandomized_settings(challenge):
    section = {"domain_randomization": {"num_obstacles": [1, 2]}, "num_obstacles": 4}
    assert physics_defaults(challenge(), section) == pytest.approx(
        {"wheel_slip": 0.0, "sensor_noise_std": 0.01, "dropout_prob": 0.02, "num_obstacles": 4.0}
    )


def test_a_curriculum_can_raise_the_obstacle_count_from_none_to_six():
    with contextlib.closing(
        NavigationEnv(seed=2, domain_randomization={"num_obstacles": (6, 6)})
    ) as env:
        section = {
            "mode": "linear",
            "episodes": 2,
            "start": {"domain_randomization": {"num_obstacles": (0, 0)}},
        }
        scheduler = curriculum_from_config({"curriculum": section}, env)
        assert scheduler is not None
        env.apply_settings(scheduler.settings())
        env.reset()
        assert _obstacles(env.env.unwrapped.layout) == 0
        scheduler.record(0.0)
        assert scheduler.settings() == {"domain_randomization": {"num_obstacles": (3.0, 3.0)}}
        env.apply_settings(scheduler.settings())
        env.reset()
        assert _obstacles(env.env.unwrapped.layout) == 3


def test_a_restored_rng_state_draws_the_same_next_episode():
    """A resumed run continues with the layout, randomization and readings it would have had."""
    randomization = {"num_obstacles": (2.0, 6.0), "wheel_slip": (0.0, 0.5)}
    with (
        contextlib.closing(NavigationEnv(seed=5, domain_randomization=randomization)) as first,
        contextlib.closing(NavigationEnv(seed=99, domain_randomization=randomization)) as second,
    ):
        first.reset()
        first.step(0)
        second.set_rng_state(first.rng_state())
        expected, actual = first.reset()[0], second.reset()[0]
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(
            first.env.unwrapped.layout.goal, second.env.unwrapped.layout.goal
        )
        np.testing.assert_array_equal(first.step(1)[0], second.step(1)[0])


def test_frame_stack_stacks_the_navigation_observation():
    with contextlib.closing(NavigationEnv(seed=1, frame_stack=3)) as env:
        obs, _ = env.reset()
        assert obs.shape == (30,)
        assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# Config factories and the challenges
# ---------------------------------------------------------------------------


def test_the_factories_build_each_id_with_its_settings():
    env = make_env_from_config({"seed": 1, "sensor": "lidar", "lidar_beams": 4, "num_obstacles": 2})
    try:
        assert env.env_name == DISCRETE_ID and env.observation_space.shape == (9,)
        assert env.env.unwrapped.num_obstacles == 2
    finally:
        env.close()
    env = make_continuous_env_from_config({"seed": 1, "action_noise_std": 0.2})
    try:
        assert env.env_name == CONTINUOUS_ID and env.action_noise_std == 0.2
    finally:
        env.close()


@pytest.mark.parametrize(
    ("factory", "setting"),
    [
        (make_env_from_config, "action_noise_std"),
        (make_continuous_env_from_config, "action_noise_prob"),
    ],
)
def test_each_factory_refuses_the_other_ids_action_noise(factory, setting):
    with pytest.raises(ValueError, match=f"does not support .*{setting}"):
        factory({setting: 0.1})


def test_both_challenges_are_solved_at_a_training_average_of_seven():
    """The maintainer's threshold (2026-11-07): about 90 % of training episodes at the goal."""
    for challenge in (NavigationChallenge(), NavigationContinuousChallenge()):
        assert challenge.solve_threshold == 7.0, f"{challenge.name} is solved at 7.0"


def test_the_continuous_random_agent_gets_the_action_bounds():
    challenge = NavigationContinuousChallenge()
    with contextlib.closing(challenge.make_env({"seed": 0}, render_mode=None)) as env:
        agent = challenge.make_agent("random", {"seed": 0}, env)
    np.testing.assert_allclose(agent.action_low, [-1.0, -1.0])
    np.testing.assert_allclose(agent.action_high, [1.0, 1.0])


# ---------------------------------------------------------------------------
# Through the scripts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", IDS)
def test_train_py_runs_a_random_agent_for_two_episodes(env_id, tmp_path, monkeypatch):
    config = minimal_config(
        tmp_path,
        environment={"name": env_id, "max_episode_steps": 30},
        training={"num_episodes": 2, "max_steps_per_episode": 30},
        agent={"config": {"seed": 0}},
    )
    run_main(["--config", config], monkeypatch)
    logs = list((tmp_path / "logs").rglob("metrics_*.jsonl"))
    assert logs, "train.py should write its metrics log under the config's log_dir"
    records = [json.loads(line) for line in logs[0].read_text().splitlines()]
    evaluations = [r["evaluation"] for r in records if "evaluation" in r]
    assert evaluations, f"the run should log an evaluation, logged {records}"
    assert all(math.isfinite(e["mean_reward"]) for e in evaluations), evaluations


@pytest.mark.usefixtures("clear_sdl")
def test_record_py_writes_a_navigation_video(tmp_path, monkeypatch):
    pytest.importorskip("moviepy")
    challenge = NavigationContinuousChallenge()
    with contextlib.closing(challenge.make_env({"seed": 0}, render_mode=None)) as env:
        agent = challenge.make_agent("random", {"seed": 0}, env)
    checkpoint = str(tmp_path / "agent.pt")
    agent.save(checkpoint)
    config = minimal_config(tmp_path, environment={"name": CONTINUOUS_ID, "max_episode_steps": 10})
    video_dir = tmp_path / "videos"
    run_main(
        ["--config", config, "--agent-type", "random", "--checkpoint", checkpoint]
        + ["--episodes", "1", "--video-dir", str(video_dir)],
        monkeypatch,
        module="record",
    )
    assert sorted(os.listdir(video_dir)) == ["navigation_continuous-playback-full.mp4"]


@pytest.mark.parametrize("key", ["navigation", "navigation_continuous"])
def test_the_benchmark_runs_the_random_agent_for_one_seed(key, tmp_path):
    from benchmark import BenchmarkSettings, run_benchmark

    result = run_benchmark(
        challenge_name=key,
        settings=BenchmarkSettings(
            agents=("random",),
            seeds=(1,),
            episodes=2,
            eval_episodes=1,
            output_dir=str(tmp_path / "out"),
        ),
        report_path=str(tmp_path / "report.md"),
        save_plot=False,
    )
    assert result["summary"]["random"]["runs"] == 1
    assert os.path.isfile(tmp_path / "report.md")

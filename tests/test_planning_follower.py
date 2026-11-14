"""rl_navigation.planning.follower: the pure-pursuit follower, its command, and the discrete-action
choice with its short safety check."""

import math

import numpy as np
import pytest

from rl_navigation.planning.follower import (
    Command,
    FollowerParams,
    PurePursuit,
    continuous_action,
    safe_discrete_action,
)
from rl_navigation.planning.grid import PlanningGrid
from rl_navigation.sim import Arena, World, hc_sr04

V_MAX, OMEGA_MAX, DT = 0.5, 2.0, 0.1
PARAMS = FollowerParams(v_max=V_MAX, omega_max=OMEGA_MAX, dt=DT, lookahead=0.25)
# The env's five rows as fractions of (v_max, omega_max): ahead, arcs left and right, spins.
ROWS = np.array([(1.0, 0.0), (0.5, 1.0), (0.5, -1.0), (0.0, 1.0), (0.0, -1.0)])


def straight_path(length=3.0):
    return np.array([[0.0, 0.0], [length, 0.0]])


def test_on_a_straight_path_at_heading_zero_it_drives_straight_at_full_speed():
    command = PurePursuit(straight_path(), PARAMS).command([0.0, 0.0, 0.0])
    assert command == Command(V_MAX, 0.0)


@pytest.mark.parametrize(
    "heading, omega",
    [
        (math.pi / 2, -OMEGA_MAX),  # facing +y, the path ahead along +x is to the right
        (-math.pi / 2, OMEGA_MAX),  # facing -y: to the left
        (math.pi, -OMEGA_MAX),  # wrap_angle maps the error -pi to -pi: clockwise
    ],
)
def test_facing_away_from_the_path_it_turns_on_the_spot_towards_it(heading, omega):
    command = PurePursuit(straight_path(), PARAMS).command([0.0, 0.0, heading])
    assert command == Command(0.0, omega)


def test_below_the_turn_threshold_the_curvature_is_two_sin_alpha_over_the_distance():
    # Robot 0.1 m beside the path, heading along it: the target is L = 0.25 m away on the path,
    # at x = sqrt(0.25^2 - 0.1^2) = 0.229; alpha = atan2(-0.1, 0.229) = -0.4115 rad (23.6 deg),
    # sin(alpha) = -0.4, kappa = 2 (-0.4) / 0.25 = -3.2 /m, omega = -3.2 x 0.5 = -1.6 rad/s.
    command = PurePursuit(straight_path(), PARAMS).command([0.0, 0.1, 0.0])
    assert command.v == V_MAX
    assert command.omega == pytest.approx(-1.6)


def test_above_omega_max_speed_and_turn_rate_scale_together():
    # With L = 0.15 m and the robot 0.06 m off the path, alpha is again -0.4115 rad (the same
    # ratio), kappa = 2 (-0.4) / 0.15 = -5.33 /m and kappa v_max = -2.67 rad/s > omega_max: both
    # scale by 2 / 2.67, to omega = -2 and v = 2 / 5.33 = 0.375 m/s, the curvature unchanged.
    params = FollowerParams(v_max=V_MAX, omega_max=OMEGA_MAX, dt=DT, lookahead=0.15)
    command = PurePursuit(straight_path(), params).command([0.0, 0.06, 0.0])
    assert command.omega == pytest.approx(-OMEGA_MAX)
    assert command.v == pytest.approx(OMEGA_MAX / (0.8 / 0.15))
    assert command.omega / command.v == pytest.approx(-0.8 / 0.15)


def test_a_small_offset_steers_at_full_speed():
    follower = PurePursuit(straight_path(), PARAMS)
    command = follower.command([0.0, 0.02, 0.0])
    alpha = math.atan2(-0.02, math.sqrt(0.25**2 - 0.02**2))
    assert command.v == V_MAX
    assert command.omega == pytest.approx(2.0 * math.sin(alpha) / 0.25 * V_MAX)


def test_at_the_goal_the_command_is_zero_with_no_division_by_zero():
    follower = PurePursuit(straight_path(1.0), PARAMS)
    for x in (0.0, 0.25, 0.5, 0.75):  # drive along it, so the projection reaches the end
        follower.command([x, 0.0, 0.0])
    with np.errstate(all="raise"):
        command = follower.command([1.0, 0.0, 0.3])
    assert command == Command(0.0, 0.0)


@pytest.mark.parametrize("seed", range(20))
def test_commands_stay_within_the_env_limits(seed):
    rng = np.random.default_rng(seed)
    path = np.cumsum(rng.uniform(-1.0, 1.0, (6, 2)), axis=0)
    follower = PurePursuit(path, PARAMS)
    for pose in np.column_stack([rng.uniform(-3, 3, (50, 2)), rng.uniform(-math.pi, math.pi, 50)]):
        command = follower.command(pose)
        assert 0.0 <= command.v <= V_MAX and abs(command.omega) <= OMEGA_MAX + 1e-12


def test_progress_never_runs_backwards_and_cannot_jump_across_a_u():
    """A U-shaped path whose two legs are 0.3 m apart: a robot on the first leg is nearer the
    second leg's far end than to some of its own path, but the projection window
    (s_prev + L + v_max dt = 0.3 m ahead) cannot reach across the 0.3 + 2 m of path between."""
    u = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.3], [0.0, 0.3]])
    follower = PurePursuit(u, PARAMS)
    seen = []
    for x in np.linspace(0.0, 1.0, 11):
        follower.command([x, 0.16, 0.0])  # nearer the return leg (0.14 m) than the first (0.16 m)
        seen.append(follower.progress)
    assert seen == sorted(seen), "progress must never decrease"
    assert max(seen) < 1.5, f"the projection jumped to the return leg: progress {max(seen):.2f} m"


def test_the_target_is_the_goal_within_lookahead_of_it():
    follower = PurePursuit(straight_path(1.0), PARAMS)
    command = follower.command([0.9, 0.05, 0.0])
    alpha = math.atan2(-0.05, 0.1)
    assert abs(alpha) > PARAMS.turn_in_place, "close to the goal, slightly off: turn first"
    assert command.v == 0.0 and command.omega == -OMEGA_MAX


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"lookahead": 0.0}, "lookahead"),
        ({"turn_in_place": 0.0}, "turn_in_place"),
        ({"turn_in_place": 4.0}, "turn_in_place"),
        ({"v_max": 0.0}, "v_max"),
        ({"omega_max": -1.0}, "omega_max"),
        ({"dt": math.nan}, "dt"),
    ],
)
def test_follower_params_refuse_bad_values(kwargs, name):
    base = {"v_max": V_MAX, "omega_max": OMEGA_MAX, "dt": DT}
    with pytest.raises(ValueError, match=name):
        FollowerParams(**{**base, **kwargs})


def test_a_path_needs_two_finite_points():
    for bad in ([[0.0, 0.0]], [[0.0, 0.0], [np.nan, 1.0]], np.zeros((3, 3))):
        with pytest.raises(ValueError, match="PurePursuit"):
            PurePursuit(bad, PARAMS)


def test_an_l_shaped_path_in_the_simulator_reaches_the_goal_close_to_the_path():
    arena = Arena(5.0, 5.0)
    path = np.array([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0]])
    world = World(arena, hc_sr04(), [1.0, 1.0, math.pi / 2], path[-1], dt=DT)
    follower = PurePursuit(path, PARAMS)
    rng = np.random.default_rng(0)
    worst, taken = 0.0, 0
    for _ in range(200):
        taken += 1
        command = follower.command(world.pose)
        motion = world.step(command.v, command.omega, rng)
        assert not motion.collided
        p = world.pose[:2]
        worst = max(worst, min(abs(p[1] - 1.0) if p[0] <= 4.0 else 9.0, abs(p[0] - 4.0)))
        if world.at_goal:
            break
    assert world.at_goal, "the follower should reach the goal of an L path in an empty arena"
    # 6 m at 0.5 m/s is 120 steps, plus the first quarter turn and the corner.
    assert taken < 150, f"took {taken} steps"
    assert worst < 0.08, f"cross-track error {worst:.3f} m"


def test_continuous_action_is_the_inverse_of_the_env_mapping():
    for v, omega in ((0.0, 0.0), (V_MAX, OMEGA_MAX), (0.2, -1.3)):
        action = continuous_action(Command(v, omega), V_MAX, OMEGA_MAX)
        # gym_env: v = v_max (a0 + 1) / 2, omega = omega_max a1.
        assert V_MAX * (action[0] + 1.0) / 2.0 == pytest.approx(v)
        assert OMEGA_MAX * action[1] == pytest.approx(omega)
        assert np.all(np.abs(action) <= 1.0)


def free_grid(shape=(40, 40), resolution=0.05):
    return PlanningGrid(np.zeros(shape, dtype=bool), resolution)


@pytest.mark.parametrize(
    "command, row",
    [
        (Command(V_MAX, 0.0), 0),
        (Command(V_MAX, 0.5 * OMEGA_MAX), 0),  # (1, 0.5): 0.5 from ahead, 0.707 from the left arc
        (Command(0.3 * V_MAX, 0.9 * OMEGA_MAX), 1),
        (Command(0.3 * V_MAX, -0.9 * OMEGA_MAX), 2),
        (Command(0.0, OMEGA_MAX), 3),
        (Command(0.0, -OMEGA_MAX), 4),
    ],
)
def test_the_nearest_row_is_chosen_where_the_way_is_free(command, row):
    pose = np.array([1.0, 1.0, 0.0])
    got = safe_discrete_action(command, ROWS, V_MAX, OMEGA_MAX, pose, free_grid(), DT)
    assert got == row


def test_a_straight_row_into_a_wall_is_refused_for_the_next_safe_one():
    """The robot 6 cm short of a blocked band (x >= 1.30 m), facing it, wants (1, 0.4). Rows by
    distance: ahead (0.40), left arc (0.78), left spin (1.17), ... Ahead reaches x = 1.39 within
    3 steps of 5 cm and the left arc x = 1.31 (3 steps of 2.5 cm while turning 0.2 rad each), both
    in the band; a spin never translates, so the left spin is chosen."""
    blocked = np.zeros((40, 40), dtype=bool)
    blocked[:, 26:] = True  # x >= 1.3 m blocked
    grid = PlanningGrid(blocked, 0.05)
    pose = np.array([1.24, 1.0, 0.0])
    command = Command(V_MAX, 0.4 * OMEGA_MAX)
    assert safe_discrete_action(command, ROWS, V_MAX, OMEGA_MAX, pose, grid, DT) == 3
    assert safe_discrete_action(command, ROWS, V_MAX, OMEGA_MAX, [1.19, 1.0, 0.0], grid, DT) == 1, (
        "at x = 1.19 the left arc stops at 1.26, short of the band: it is the next safe row"
    )
    assert safe_discrete_action(command, ROWS, V_MAX, OMEGA_MAX, [0.5, 1.0, 0.0], grid, DT) == 0


def test_a_robot_on_a_blocked_cell_can_still_spin():
    blocked = np.ones((10, 10), dtype=bool)
    grid = PlanningGrid(blocked, 0.05)
    got = safe_discrete_action(
        Command(V_MAX, -0.1), ROWS, V_MAX, OMEGA_MAX, [0.2, 0.2, 0.0], grid, DT
    )
    assert got == 4, "every moving row is refused; the spin towards the command's turn is kept"


def test_both_mappings_command_what_the_env_executes():
    """Through NavigationGymEnv with no slip: the observation's last executed command
    ``(v / v_max, omega / omega_max)`` is the follower's command (continuous) or the chosen row's
    (discrete)."""
    from rl_navigation.gym_env import DISCRETE_ACTIONS, GOAL_AND_COMMAND, NavigationGymEnv

    for continuous in (True, False):
        env = NavigationGymEnv(continuous=continuous, wheel_slip=0.0)
        try:
            env.reset(seed=10000)
            command = Command(0.3, -0.7)
            if continuous:
                action = continuous_action(command, env.v_max, env.omega_max)
                expected = (command.v / env.v_max, command.omega / env.omega_max)
            else:
                grid = PlanningGrid(np.zeros((100, 100), dtype=bool), 0.05)
                action = safe_discrete_action(
                    command,
                    DISCRETE_ACTIONS,
                    env.v_max,
                    env.omega_max,
                    env.world.pose,
                    grid,
                    env.dt,
                )
                expected = tuple(DISCRETE_ACTIONS[action])
            obs, *_ = env.step(action)
            np.testing.assert_allclose(
                obs[GOAL_AND_COMMAND - 2 : GOAL_AND_COMMAND], expected, atol=1e-6
            )
        finally:
            env.close()


def test_a_bearing_error_just_above_the_threshold_turns_on_the_spot():
    """Heading 0.6 rad (34 deg) to the left of a straight path: the target is 0.25 m ahead on it,
    alpha = -0.6 rad, beyond 25 deg: a spin clockwise, not an arc."""
    command = PurePursuit(straight_path(), PARAMS).command([0.0, 0.0, 0.6])
    assert command == Command(0.0, -OMEGA_MAX)
    assert PurePursuit(straight_path(), PARAMS).command([0.0, 0.0, 0.4]).v > 0.0, "23 deg: an arc"


def test_near_the_end_the_curvature_uses_the_distance_to_the_goal_not_the_lookahead():
    # Driven to x = 0.9, 0.02 m off a 1 m path: the target is the end (1, 0), d = 0.102 m,
    # alpha = atan2(-0.02, 0.1) = -0.197 rad (11 deg), omega = 2 sin(alpha) / d x 0.5 = -1.92.
    follower = PurePursuit(straight_path(1.0), PARAMS)
    for x in (0.0, 0.25, 0.5, 0.75):
        follower.command([x, 0.0, 0.0])
    command = follower.command([0.9, 0.02, 0.0])
    d, alpha = math.hypot(0.1, 0.02), math.atan2(-0.02, 0.1)
    assert command.v == V_MAX
    assert command.omega == pytest.approx(2.0 * math.sin(alpha) / d * V_MAX)


def test_a_robot_on_a_blocked_cell_keeps_the_nearest_spin_before_a_free_move():
    """Only the robot's own cell is blocked. Spinning cannot collide, so the wanted spin (row 3)
    is kept, although the left arc (row 1, next nearest) would also leave onto free cells."""
    blocked = np.zeros((40, 40), dtype=bool)
    blocked[20, 20] = True
    grid = PlanningGrid(blocked, 0.05)
    pose = [1.025, 1.025, 0.0]
    assert (
        safe_discrete_action(Command(0.0, OMEGA_MAX), ROWS, V_MAX, OMEGA_MAX, pose, grid, DT) == 3
    )


def test_a_repeated_point_in_the_path_is_skipped():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    follower = PurePursuit(path, PARAMS)
    for x in np.arange(0.0, 2.0, 0.05):
        command = follower.command([x, 0.0, 0.0])
        assert command == Command(V_MAX, 0.0) or x > 1.75
    assert follower.progress == pytest.approx(1.95)


def test_a_row_that_would_leave_the_grid_is_refused():
    grid = PlanningGrid(np.zeros((10, 10), dtype=bool), 0.05)  # 0.5 m square
    pose = [0.45, 0.25, 0.0]  # 5 cm from the right edge, facing it
    assert safe_discrete_action(Command(V_MAX, 0.0), ROWS, V_MAX, OMEGA_MAX, pose, grid, DT) == 3


def test_with_no_spin_among_the_rows_an_unsafe_choice_is_an_error():
    blocked = np.ones((10, 10), dtype=bool)
    grid = PlanningGrid(blocked, 0.05)
    moving_only = ROWS[:3]
    with pytest.raises(ValueError, match="no row"):
        safe_discrete_action(
            Command(V_MAX, 0.0), moving_only, V_MAX, OMEGA_MAX, [0.2, 0.2, 0.0], grid, DT
        )

"""rl_navigation.planning.navigator: plan, follow, replan; the margin ladder, the escape bubble,
the substitute goal, retries after "no_path", and the two maps."""

import math

import numpy as np
import pytest

from rl_navigation.mapping.grid import HC_SR04_LOG_ODDS, OccupancyGrid
from rl_navigation.planning.follower import Command, FollowerParams
from rl_navigation.planning.grid import PlanningGrid, from_occupied
from rl_navigation.planning.navigator import (
    NO_PATH,
    Navigator,
    NavigatorParams,
    OnlineMap,
    TrueMap,
)
from rl_navigation.sim import Arena, World, hc_sr04

V_MAX, OMEGA_MAX, DT, RES = 0.5, 2.0, 0.1, 0.05
FOLLOWER = FollowerParams(v_max=V_MAX, omega_max=OMEGA_MAX, dt=DT)
BOX_ARENA = Arena(5.0, 5.0, boxes=[[2.0, 1.0, 3.0, 4.0]])  # a box between start and goal


class FakeMap:
    """A map whose grid (at every inflation) and version the test sets by hand."""

    def __init__(self, blocked, occupied=None):
        self.version = 0
        self.resolution = RES
        self.set(blocked, occupied)

    def set(self, blocked, occupied=None):
        self._grid = PlanningGrid(blocked, RES, occupied=occupied)

    def grid(self, inflation):
        del inflation
        return self._grid


def drive(world, navigator, world_map, steps=400, online=False):
    rng = np.random.default_rng(0)
    for _ in range(steps):
        if online:
            readings, _ = world.read_sensors(rng)
            world_map.update(world.pose, readings)
        command = navigator.command(world.pose, world_map)
        motion = world.step(command.v, command.omega, rng)
        if motion.collided:
            return "collision"
        if world.at_goal:
            return "goal"
    return "timeout"


def test_on_the_true_map_it_drives_around_a_box_to_the_goal():
    world = World(BOX_ARENA, hc_sr04(), [1.0, 2.5, 0.0], [4.0, 2.5], dt=DT)
    navigator = Navigator(world.goal, NavigatorParams(margin=0.08), FOLLOWER)
    assert drive(world, navigator, TrueMap(BOX_ARENA, RES)) == "goal"
    assert navigator.inflation == pytest.approx(0.1 + RES / 2 + 0.08), "the full margin fits"


def test_on_an_online_map_it_finds_the_box_and_drives_around_it():
    world = World(BOX_ARENA, hc_sr04(), [1.0, 2.5, 0.0], [4.0, 2.5], dt=DT)
    occupancy = OccupancyGrid(hc_sr04(), (100, 100), RES, HC_SR04_LOG_ODDS)
    online = OnlineMap(occupancy)
    navigator = Navigator(world.goal, NavigatorParams(), FOLLOWER)
    assert drive(world, navigator, online, online=True) == "goal"
    assert online.version > 0 and navigator.plans > 1, "the box was seen and planned around"


def test_it_plans_once_then_waits_replan_every_steps():
    navigator = Navigator([2.0, 2.0], NavigatorParams(replan_every=4), FOLLOWER)
    world_map = FakeMap(np.zeros((60, 60), dtype=bool))
    for _ in range(4):
        navigator.command([0.5, 0.5, 0.0], world_map)
    assert navigator.plans == 1
    navigator.command([0.5, 0.5, 0.0], world_map)
    assert navigator.plans == 2, "the fifth step is due a new plan"


def test_a_new_map_replans_only_when_the_rest_of_the_path_is_blocked():
    navigator = Navigator([2.5, 0.5], NavigatorParams(replan_every=100), FOLLOWER)
    blocked = np.zeros((60, 60), dtype=bool)
    world_map = FakeMap(blocked)
    navigator.command([0.5, 0.5, 0.0], world_map)
    away = blocked.copy()
    away[50, 50] = True  # far from the straight path along y = 0.5
    world_map.set(away)
    world_map.version = 1
    navigator.command([0.5, 0.5, 0.0], world_map)
    assert navigator.plans == 1, "the path is still free: no replan"
    across = away.copy()
    across[5:15, 30] = True  # a wall across y = 0.5 at x = 1.5
    world_map.set(across)
    world_map.version = 2
    navigator.command([0.5, 0.5, 0.0], world_map)
    assert navigator.plans == 2, "the path is blocked: one replan"
    assert navigator.status == "following"


def test_after_no_path_it_spins_and_retries_only_on_a_new_map_or_when_due():
    blocked = np.zeros((60, 60), dtype=bool)
    blocked[:, 30] = True  # a wall across the arena
    world_map = FakeMap(blocked)
    navigator = Navigator([2.5, 0.5], NavigatorParams(replan_every=5), FOLLOWER)
    command = navigator.command([0.5, 0.5, math.pi / 2], world_map)
    assert navigator.status == NO_PATH and navigator.plans == 1 and navigator.searches == 0
    assert command == Command(0.0, -OMEGA_MAX), "the goal is to the right: spin clockwise"
    for _ in range(4):
        navigator.command([0.5, 0.5, math.pi / 2], world_map)
    assert navigator.plans == 1, "no retry before replan_every steps on an unchanged map"
    world_map.version = 1
    navigator.command([0.5, 0.5, math.pi / 2], world_map)
    assert navigator.plans == 2, "a new map is retried at once"
    open_wall = blocked.copy()
    open_wall[40:45, 30] = False
    world_map.set(open_wall)
    navigator.command([0.5, 0.5, math.pi / 2], world_map)
    assert navigator.plans == 2, "the version did not change, so the map is not looked at"
    world_map.version = 2
    navigator.command([0.5, 0.5, math.pi / 2], world_map)
    assert navigator.status == "following" and navigator.plans == 3


def test_the_escape_bubble_frees_inflation_but_never_an_obstacle():
    occupied = np.zeros((60, 60), dtype=bool)
    occupied[10, 12] = True  # one obstacle cell 0.1 m right of the robot
    inflation = 0.1 + RES / 2 + 0.08
    grid = from_occupied(occupied, RES, inflation)
    world_map = FakeMap(grid.blocked, grid.occupied)
    navigator = Navigator([2.0, 2.0], NavigatorParams(), FOLLOWER)
    navigator.command([0.525, 0.525, 0.0], world_map)  # the centre of cell (10, 10), in the band
    assert navigator.status == "following", "the robot inside the band may leave it"
    unknown = FakeMap(grid.blocked)  # no occupied mask: nothing can be told apart
    stuck = Navigator([2.0, 2.0], NavigatorParams(), FOLLOWER)
    stuck.command([0.525, 0.525, 0.0], unknown)
    assert stuck.status == NO_PATH


def test_a_blocked_goal_is_replaced_by_a_free_cell_within_tolerance():
    blocked = np.zeros((60, 60), dtype=bool)
    blocked[40:43, 40:43] = True  # the goal (2.075, 2.075) is in the middle cell (41, 41)
    navigator = Navigator([2.075, 2.075], NavigatorParams(), FOLLOWER)
    navigator.command([0.5, 0.5, 0.0], FakeMap(blocked))
    assert navigator.status == "following"
    end = navigator.path[-1]
    assert np.hypot(*(end - [2.075, 2.075])) == pytest.approx(2 * RES), (
        "the nearest free centres are two cells straight from the goal's (0.1 m; a diagonal is 0.14)"
    )
    blocked[36:47, 36:47] = True  # nothing free within 0.15 m
    far = Navigator([2.075, 2.075], NavigatorParams(), FOLLOWER)
    far.command([0.5, 0.5, 0.0], FakeMap(blocked))
    assert far.status == NO_PATH


def test_the_margin_steps_down_through_a_narrow_gap():
    # Two boxes leave a 0.4 m gap: with r + res/2 + 0.08 = 0.205 m on each side it is shut
    # (0.41 > 0.4); at r + res/2 + 0.04 = 0.165 m it is open.
    arena = Arena(5.0, 5.0, boxes=[[2.4, 0.0, 2.6, 2.3], [2.4, 2.7, 2.6, 5.0]])
    navigator = Navigator([4.0, 2.5], NavigatorParams(margin=0.08), FOLLOWER)
    navigator.command([1.0, 2.5, 0.0], TrueMap(arena, RES))
    assert navigator.inflation == pytest.approx(0.1 + RES / 2 + 0.04)


def test_the_online_map_ignores_readings_beyond_its_range():
    sensor = hc_sr04()
    occupancy = OccupancyGrid(sensor, (100, 100), RES, HC_SR04_LOG_ODDS)
    online = OnlineMap(occupancy, map_range=1.5)
    pose = [2.5, 2.5, 0.0]
    online.update(pose, np.full(sensor.num_readings, 2.0))
    assert online.version == 0 and not occupancy.observed.any(), "2 m readings are left out"
    online.update(pose, np.full(sensor.num_readings, 1.0))
    assert online.version == 1 and occupancy.occupied().any()
    first = online.grid(0.2)
    assert online.grid(0.2) is first, "a grid is kept until the map changes"
    online.update(pose, np.full(sensor.num_readings, np.nan))
    assert online.version == 1 and online.grid(0.2) is first


def test_true_map_grids_are_the_arena_inflated_and_kept():
    world_map = TrueMap(BOX_ARENA, RES)
    grid = world_map.grid(0.2)
    assert world_map.grid(0.2) is grid and world_map.version == 0
    assert grid.blocked[50, 50] and not grid.blocked[50, 10]


def test_the_safety_grid_is_the_smallest_inflation():
    navigator = Navigator([1.0, 1.0], NavigatorParams(margin=0.1), FOLLOWER)
    world_map = TrueMap(BOX_ARENA, RES)
    assert navigator.safety_grid(world_map) is world_map.grid(0.1 + RES / 2)


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"robot_radius": 0.0}, "robot_radius"),
        ({"margin": -0.01}, "margin"),
        ({"replan_every": 0}, "replan_every"),
        ({"replan_every": 2.0}, "replan_every"),
        ({"goal_tolerance": math.nan}, "goal_tolerance"),
    ],
)
def test_navigator_params_refuse_bad_values(kwargs, name):
    with pytest.raises(ValueError, match=name):
        NavigatorParams(**kwargs)


def test_online_map_refuses_a_bad_range():
    occupancy = OccupancyGrid(hc_sr04(), (10, 10), RES, HC_SR04_LOG_ODDS)
    with pytest.raises(ValueError, match="map_range"):
        OnlineMap(occupancy, map_range=0.0)


def test_a_pose_or_goal_outside_the_map_has_no_path():
    world_map = FakeMap(np.zeros((20, 20), dtype=bool))
    navigator = Navigator([5.0, 5.0], NavigatorParams(), FOLLOWER)
    navigator.command([0.5, 0.5, 0.0], world_map)
    assert navigator.status == NO_PATH and navigator.path is None


def test_prepare_builds_every_level_of_the_ladder_before_the_first_step():
    world_map = TrueMap(BOX_ARENA, RES)
    navigator = Navigator([4.0, 2.5], NavigatorParams(margin=0.06), FOLLOWER)
    grids = navigator.prepare(world_map)
    inflations = (0.1 + RES / 2 + 0.06, 0.1 + RES / 2 + 0.03, 0.1 + RES / 2)
    assert len(grids) == 3
    for grid, inflation in zip(grids, inflations, strict=True):
        assert world_map.grid(inflation) is grid, "the map keeps the prepared grid"


def test_a_retry_after_no_path_looks_at_the_smallest_inflation_first_and_stops_there():
    """While no path exists the map changes at almost every scan; a retry that fails must cost one
    grid and one flood fill, not the whole ladder."""
    blocked = np.zeros((60, 60), dtype=bool)
    blocked[:, 30] = True
    world_map = FakeMap(blocked)
    asked = []
    original = world_map.grid
    world_map.grid = lambda inflation: asked.append(round(inflation, 4)) or original(inflation)
    navigator = Navigator([2.5, 0.5], NavigatorParams(margin=0.06), FOLLOWER)
    navigator.command([0.5, 0.5, 0.0], world_map)
    assert navigator.status == NO_PATH and len(asked) == 3, "the first plan tries the ladder"
    asked.clear()
    world_map.version = 1
    navigator.command([0.5, 0.5, 0.0], world_map)
    assert asked == [round(0.1 + RES / 2, 4)], f"a failing retry asked for {asked}"
    world_map.set(np.zeros((60, 60), dtype=bool))
    world_map.version = 2
    asked.clear()
    navigator.command([0.5, 0.5, 0.0], world_map)
    assert navigator.status == "following"
    assert navigator.inflation == pytest.approx(0.1 + RES / 2 + 0.06), "then the full ladder"


def test_with_no_path_it_turns_a_full_circle_in_one_direction_to_rescan():
    """With no path the robot must sweep its sensors all round. Turning towards the goal's
    bearing dithers about it (the sign flips as the heading passes it), so the direction is
    chosen once, when "no_path" begins, and held until a path is found: two full turns take
    2 x 2 pi / (omega_max dt) = 62.8, so 63 steps of 0.2 rad, all one way."""
    blocked = np.zeros((60, 60), dtype=bool)
    blocked[:, 30] = True
    world_map = FakeMap(blocked)
    navigator = Navigator([2.5, 0.5], NavigatorParams(replan_every=10), FOLLOWER)
    heading, turned = math.pi / 2, 0.0
    for _ in range(63):
        command = navigator.command([0.5, 0.5, heading], world_map)
        assert navigator.status == NO_PATH and command.v == 0.0
        heading += command.omega * DT
        turned += command.omega * DT
    assert abs(turned) >= 4 * math.pi - 1e-9, f"turned {turned:.2f} rad, not two full circles"


def test_discrete_actions_through_the_safety_check_reach_the_goal_round_a_box():
    from rl_navigation.planning.follower import safe_discrete_action

    rows = np.array([(1.0, 0.0), (0.5, 1.0), (0.5, -1.0), (0.0, 1.0), (0.0, -1.0)])
    follower = FollowerParams(V_MAX, OMEGA_MAX, DT, turn_in_place=math.radians(45.0))
    world = World(BOX_ARENA, hc_sr04(), [1.0, 2.5, 0.0], [4.0, 2.5], dt=DT)
    navigator = Navigator(world.goal, NavigatorParams(margin=0.06), follower)
    world_map = TrueMap(BOX_ARENA, RES)
    rng = np.random.default_rng(0)
    for _ in range(400):
        command = navigator.command(world.pose, world_map)
        row = safe_discrete_action(
            command, rows, V_MAX, OMEGA_MAX, world.pose, navigator.safety_grid(world_map), DT
        )
        motion = world.step(rows[row, 0] * V_MAX, rows[row, 1] * OMEGA_MAX, rng)
        assert not motion.collided
        if world.at_goal:
            break
    assert world.at_goal


def test_a_robot_inside_the_band_checks_its_path_from_where_it_leaves_the_band():
    """The robot's own cell is blocked by inflation only. On a new map version the rest of the
    path is checked from the band's edge on: a change that leaves it free causes no replan, one
    that blocks it beyond the band does."""
    occupied = np.zeros((60, 60), dtype=bool)
    occupied[12, 10] = True  # 0.1 m above the robot at (0.525, 0.525)
    inflation = 0.1 + RES / 2 + 0.05
    grid = from_occupied(occupied, RES, inflation)
    world_map = FakeMap(grid.blocked, grid.occupied)
    navigator = Navigator([2.525, 0.525], NavigatorParams(margin=0.05, replan_every=100), FOLLOWER)
    navigator.command([0.525, 0.525, 0.0], world_map)
    assert navigator.status == "following" and navigator.plans == 1
    world_map.version = 1
    navigator.command([0.525, 0.525, 0.0], world_map)
    assert navigator.plans == 1, "the path from the band's edge on is free: no replan"
    beyond = occupied.copy()
    beyond[8:13, 30] = True  # across the path at x = 1.5 m
    grid = from_occupied(beyond, RES, inflation)
    world_map.set(grid.blocked, grid.occupied)
    world_map.version = 2
    navigator.command([0.525, 0.525, 0.0], world_map)
    assert navigator.plans == 2, "blocked beyond the band: a replan"


def test_plans_made_below_the_full_margin_are_counted():
    arena = Arena(5.0, 5.0, boxes=[[2.4, 0.0, 2.6, 2.3], [2.4, 2.7, 2.6, 5.0]])
    navigator = Navigator([4.0, 2.5], NavigatorParams(margin=0.08, replan_every=2), FOLLOWER)
    world_map = TrueMap(arena, RES)
    assert navigator.reduced_margin_plans == 0
    for _ in range(5):  # plans at steps 1, 3 and 5, each through the 0.4 m gap
        navigator.command([1.0, 2.5, 0.0], world_map)
    assert navigator.plans == 3 and navigator.reduced_margin_plans == 3
    open_map = TrueMap(Arena(5.0, 5.0), RES)
    roomy = Navigator([4.0, 2.5], NavigatorParams(margin=0.08), FOLLOWER)
    roomy.command([1.0, 2.5, 0.0], open_map)
    assert roomy.reduced_margin_plans == 0


def test_a_successful_retry_searches_the_smallest_level_once():
    blocked = np.zeros((60, 60), dtype=bool)
    blocked[:, 30] = True
    world_map = FakeMap(blocked)
    navigator = Navigator([2.5, 0.5], NavigatorParams(margin=0.06), FOLLOWER)
    navigator.command([0.5, 0.5, 0.0], world_map)
    assert navigator.status == NO_PATH
    # Open the wall only for the smallest inflation: larger levels still have no path.
    narrow = FakeMap(blocked)
    small = round(0.1 + RES / 2, 6)
    opened = blocked.copy()
    opened[40:45, 30] = False
    grids = {small: PlanningGrid(opened, RES)}
    narrow.grid = lambda inflation: grids.get(round(inflation, 6), PlanningGrid(blocked, RES))
    narrow.version = 1
    navigator.command([0.5, 0.5, 0.0], narrow)
    assert navigator.status == "following" and navigator.inflation == pytest.approx(small)
    assert navigator.searches == 1, f"{navigator.searches} A* searches: the retry's result is kept"

"""rl_navigation.planning_comparison: the desk harness that plays planners and policies on the same
seeded layouts (controllers, episodes), its statistics (metrics), its JSON records and its report.

Every layout here is a tuning seed (10000 to 10199); 30000 to 30499 are the planning item's judged
layouts and are played once, by the judged run only.
"""

import json
import math

import numpy as np
import pytest

from rl_navigation.gym_env import DISCRETE_ACTIONS, NavigationGymEnv
from rl_navigation.planning import DEFAULT_MAP_RANGE, TUNED_SETTINGS
from rl_navigation.planning_comparison import (
    G4_SCOPE,
    GATES,
    PLANNER_ROWS,
    EpisodeRecord,
    PlannerController,
    PolicyController,
    Row,
    common_successes,
    gate_verdicts,
    load_records,
    mcnemar,
    merge_records,
    play,
    render_report,
    save_records,
    shortest_length,
    sliding_means,
    speed_verdict,
    spl,
    summarize,
    wilson_interval,
)
from rl_navigation.planning_comparison import report as report_module
from rl_navigation.sim import Arena, Layout, clearance

SEED = 10000  # a tuning layout


def _record(seed, success=True, collision=False, **fields):
    values = {
        "steps": 10,
        "reward": 1.0,
        "time": 1.0,
        "path_length": 1.0,
        "shortest_length": 1.0,
        "min_clearance": 0.2,
        "step_ms": (1.0,),
        "ladder_level": None,
    }
    values.update(fields)
    return EpisodeRecord(
        seed=seed,
        success=success,
        collision=collision,
        timeout=not success and not collision,
        **values,
    )


def _records(outcomes, first_seed=0, **fields):
    """One record per outcome ("s" success, "c" collision, "t" timeout), seeds counting up."""
    return [
        _record(first_seed + i, success=o == "s", collision=o == "c", **fields)
        for i, o in enumerate(outcomes)
    ]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("successes", "n", "expected"),
    [
        # Published Wilson 95% intervals: 9 of 10 is 0.5958 to 0.9821; 0 of 10 is 0 to 0.2775,
        # whose upper end is z^2 / (n + z^2) = 3.8415 / 13.8415.
        (9, 10, (0.59584, 0.98212)),
        (0, 10, (0.0, 0.27753)),
        (10, 10, (0.72247, 1.0)),
    ],
)
def test_wilson_intervals_match_their_published_values(successes, n, expected):
    np.testing.assert_allclose(wilson_interval(successes, n), expected, atol=1e-5)


@pytest.mark.parametrize(("successes", "n"), [(0, 0), (-1, 5), (6, 5)])
def test_wilson_refuses_counts_that_are_not_a_share(successes, n):
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(successes, n)


def test_spl_weights_each_success_by_shortest_over_the_longer_of_the_two():
    records = [
        _record(0, path_length=2.0, shortest_length=1.0),  # 1 / 2
        _record(1, path_length=0.8, shortest_length=1.0),  # stopped inside the goal radius: 1
        _record(2, success=False, path_length=0.5, shortest_length=1.0),  # 0
        _record(3, path_length=4.0, shortest_length=2.0),  # 2 / 4
    ]
    assert spl(records) == pytest.approx((0.5 + 1.0 + 0.0 + 0.5) / 4)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        # b = 1 first only, c = 6 second only: 2 (C(7,0) + C(7,1)) / 2^7 = 16 / 128.
        ("sssss" + "s" + "tttttt", "sssss" + "t" + "ssssss", (1, 6, 0.125)),
        ("ccccc", "sssss", (0, 5, 0.0625)),  # 2 / 2^5
        ("ssstttc", "tttsssc", (3, 3, 1.0)),  # two-sided, capped at 1
        ("ssss", "ssss", (0, 0, 1.0)),  # no discordant pair
    ],
)
def test_mcnemar_is_the_exact_binomial_test_on_the_discordant_pairs(first, second, expected):
    result = mcnemar(_records(first), _records(second))
    assert (result.first_only, result.second_only) == expected[:2]
    assert result.p_value == pytest.approx(expected[2])


def test_pairings_refuse_rows_played_on_different_layouts():
    with pytest.raises(ValueError, match="same seeds"):
        mcnemar(_records("ss"), _records("ss", first_seed=1))
    with pytest.raises(ValueError, match="same seeds"):
        common_successes(_records("ss"), _records("sss"))


def test_medians_are_taken_on_the_layouts_both_rows_reached():
    first = [
        _record(0, path_length=1.0, time=4.0),
        _record(1, path_length=3.0, time=8.0),
        _record(2, path_length=9.0, time=99.0),  # the second row failed here: left out
        _record(3, success=False, path_length=0.1, time=0.1),
    ]
    second = [
        _record(0, path_length=2.0, time=5.0),
        _record(1, path_length=2.0, time=7.0),
        _record(2, success=False),
        _record(3, path_length=5.0, time=50.0),  # the first row failed here: left out
    ]
    common = common_successes(first, second)
    assert common.count == 2
    assert (common.first_path, common.second_path) == (2.0, 2.0)
    assert (common.first_time, common.second_time) == (6.0, 6.0)


def test_with_no_common_success_the_medians_are_nan():
    common = common_successes(_records("st"), _records("ts"))
    assert common.count == 0 and math.isnan(common.first_path) and math.isnan(common.second_time)


def test_sliding_means_average_every_window_of_one_replan_period():
    np.testing.assert_allclose(sliding_means([1.0, 2.0, 3.0, 4.0, 8.0], 3), [2.0, 3.0, 5.0])
    np.testing.assert_allclose(sliding_means([1.0, 2.0], 3), [1.5]), "shorter: one, of all"


def test_a_summary_counts_outcomes_and_pools_the_step_times():
    records = [
        *_records("sss", min_clearance=0.3, step_ms=(1.0, 2.0, 3.0)),
        _record(3, success=False, collision=True, min_clearance=0.1, step_ms=(9.0,)),
        _record(4, success=False, min_clearance=0.2, ladder_level=1, step_ms=(1.0,)),
    ]
    summary = summarize(records, period=2)
    assert (summary.episodes, summary.successes, summary.collisions, summary.timeouts) == (
        5,
        3,
        1,
        1,
    )
    assert summary.success_rate == pytest.approx(0.6)
    assert summary.min_clearance == pytest.approx(0.1)
    assert summary.reduced_margin == 1, "one episode planned below the full margin"
    assert summary.step_ms_max == 9.0
    # Windows of 2 per episode: (1.5, 2.5) three times, then 9 and 1 (one step each).
    assert summary.sliding_ms_p99 == pytest.approx(np.quantile([1.5, 2.5] * 3 + [9.0, 1.0], 0.99))


def _row(name, outcomes, setting=None, value=None, kind=None, actions=None, radius=0.1, **fields):
    """A row of hand-made records; a planner row's kind and action kind follow its name."""
    kind = kind or ("planner" if name.startswith("P-") else "policy")
    actions = actions or (
        "discrete" if name.endswith("-d") or name.startswith("DQN") else "continuous"
    )
    return Row(name, kind, actions, setting, value, tuple(_records(outcomes, **fields)), radius)


def _gate_row(name, successes, collisions=0, clearance_value=0.2, n=500, radius=0.1):
    outcomes = "s" * successes + "c" * collisions + "t" * (n - successes - collisions)
    return {name: _row(name, outcomes, radius=radius, min_clearance=clearance_value)}


@pytest.mark.parametrize(
    ("row", "successes", "collisions", "clearance_value", "passed"),
    [
        # G1: success >= 98 % (490 of 500), no collision, clearance >= r + 0.01.
        ("P-true-c", 490, 0, 0.11, True),
        ("P-true-c", 489, 0, 0.2, False),
        ("P-true-c", 499, 1, 0.2, False),
        ("P-true-c", 500, 0, 0.1099, False),
        # G2: success >= 97 % (485), collisions <= 0.5 % (2.5 of 500), clearance >= r + 0.005.
        ("P-true-d", 485, 2, 0.105, True),
        ("P-true-d", 484, 0, 0.2, False),
        ("P-true-d", 497, 3, 0.2, False),
        ("P-true-d", 500, 0, 0.1049, False),
        # G3c and G3d: success >= 93 % (465), collisions <= 2 % (10); no clearance bound.
        ("P-online-c", 465, 10, 0.0, True),
        ("P-online-c", 464, 0, 0.2, False),
        ("P-online-c", 470, 11, 0.2, False),
        ("P-online-d", 465, 10, 0.0, True),
        ("P-online-d", 464, 0, 0.2, False),
        ("P-online-d", 470, 11, 0.2, False),
    ],
)
def test_each_gate_holds_on_its_own_boundaries(row, successes, collisions, clearance_value, passed):
    (verdict,) = gate_verdicts(_gate_row(row, successes, collisions, clearance_value))
    assert verdict.gate.row == row
    assert verdict.passed is passed, (
        f"{verdict.gate.name}: {successes} / {collisions} / {clearance_value}"
    )


def test_the_clearance_gate_uses_the_rows_robot_radius():
    """0.13 m clears r + 0.01 for a 0.1 m robot, not for a 0.15 m one."""
    assert gate_verdicts(_gate_row("P-true-c", 500, clearance_value=0.13))[0].passed
    assert not gate_verdicts(_gate_row("P-true-c", 500, clearance_value=0.13, radius=0.15))[
        0
    ].passed


def test_the_registered_gates_are_the_plans():
    assert [
        (g.name, g.row, g.min_success, g.max_collision_rate, g.min_clearance) for g in GATES
    ] == [
        ("G1", "P-true-c", 0.98, 0.0, 0.01),
        ("G2", "P-true-d", 0.97, 0.005, 0.005),
        ("G3c", "P-online-c", 0.93, 0.02, None),
        ("G3d", "P-online-d", 0.93, 0.02, None),
    ]


@pytest.mark.parametrize(
    ("step_ms", "passed"),
    [
        ((15.0,) * 10, True),  # sliding p99 exactly 15 ms
        ((15.1,) * 10, False),
        ((1.0,) * 9 + (100.5,), False),  # one step over 100 ms
    ],
)
def test_the_speed_gate_g4_bounds_the_sliding_mean_and_every_step(step_ms, passed):
    assert speed_verdict(summarize([_record(0, step_ms=step_ms)])).passed is passed


# ---------------------------------------------------------------------------
# controllers and episodes, in the navigation environment
# ---------------------------------------------------------------------------


class Recording:
    """A controller that hands each action on and keeps it, with the pose it was chosen at."""

    def __init__(self, inner):
        self.inner = inner
        self.actions = []

    def reset(self, env):
        self.env = env
        self.inner.reset(env)

    def act(self, observation):
        action = self.inner.act(observation)
        self.actions.append(action)
        return action

    @property
    def ladder_level(self):
        return self.inner.ladder_level


def _replay(actions, seed, continuous):
    """The observations, rewards and positions of a plain run of ``actions`` from ``seed``."""
    env = NavigationGymEnv(continuous=continuous)
    try:
        obs, _ = env.reset(seed=seed)
        observations, rewards, positions = [obs], [], [env.world.pose[:2].copy()]
        for action in actions:
            obs, reward, _, _, _ = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            positions.append(env.world.pose[:2].copy())
        return observations, rewards, np.array(positions), env.layout
    finally:
        env.close()


class ObservationLog(Recording):
    def __init__(self, inner):
        super().__init__(inner)
        self.observations = []

    def act(self, observation):
        self.observations.append(np.array(observation))
        return super().act(observation)


@pytest.mark.parametrize(
    ("map_source", "continuous", "seed"),
    [("online", True, 10003), ("online", False, 10003), ("true", True, SEED)],
)
def test_an_episode_record_is_what_a_plain_run_of_its_actions_gives(map_source, continuous, seed):
    """The online controller builds its map from the readings in the observation and reads no
    scan of its own: a second read would draw the episode generator's noise and dropout, and a
    plain run of the same actions would then see other observations."""
    controller = ObservationLog(PlannerController(map_source))
    env = NavigationGymEnv(continuous=continuous)
    try:
        record = play(env, controller, seed)
    finally:
        env.close()
    observations, rewards, positions, layout = _replay(controller.actions, seed, continuous)
    for i, (seen, plain) in enumerate(zip(controller.observations, observations, strict=False)):
        np.testing.assert_array_equal(
            seen, plain, err_msg=f"observation {i} differs from a plain run"
        )
    assert record.seed == seed and record.steps == len(controller.actions) == len(record.step_ms)
    assert record.reward == pytest.approx(sum(rewards))
    assert record.path_length == pytest.approx(np.hypot(*np.diff(positions, axis=0).T).sum())
    assert record.min_clearance == pytest.approx(clearance(layout.arena, positions).min())
    assert record.time == pytest.approx(0.1 * record.steps)
    assert record.success and not record.collision and not record.timeout, "a tuning layout reached"
    assert record.ladder_level in (0, 1, 2)
    assert record.plan_ms_max == max(record.step_ms)


def test_shortest_length_is_the_smoothed_plan_at_no_margin_and_at_least_the_straight_line():
    env = NavigationGymEnv()
    try:
        env.reset(seed=SEED)
        layout = env.layout
    finally:
        env.close()
    length = shortest_length(layout, robot_radius=0.1)
    straight = float(np.hypot(*(layout.goal - layout.start[:2])))
    assert straight <= length < 1.5 * straight


def test_play_keeps_the_shortest_length_of_each_seed_in_the_cache_it_is_given(monkeypatch):
    cache = {}
    env = NavigationGymEnv()
    try:
        first = play(env, PlannerController("true"), SEED, shortest_cache=cache, max_steps=2)
        assert cache == {SEED: first.shortest_length}
        cache[SEED] = 123.0
        again = play(env, PlannerController("true"), SEED, shortest_cache=cache, max_steps=2)
    finally:
        env.close()
    assert again.shortest_length == 123.0, "the cached length is used, not recomputed"


def test_an_episode_cut_at_max_steps_is_a_timeout():
    env = NavigationGymEnv()
    try:
        record = play(env, PlannerController("true"), SEED, max_steps=3)
    finally:
        env.close()
    assert record.steps == 3 and record.timeout and not record.success and not record.collision


class Straight:
    def __init__(self):
        self.training = []

    def select_action(self, observation, training=False):
        self.training.append(training)
        return 0  # DISCRETE_ACTIONS[0]: straight ahead at full speed


def test_a_policy_is_played_greedily_until_it_collides():
    policy = Straight()
    env = NavigationGymEnv(continuous=False)
    try:
        record = play(env, PolicyController(policy), SEED)
    finally:
        env.close()
    assert set(policy.training) == {False}, "every action is chosen with training=False"
    assert record.collision and not record.success and not record.timeout
    assert record.ladder_level is None, "a policy has no margin ladder"


@pytest.mark.parametrize("continuous", [True, False])
def test_the_planner_takes_the_tuned_settings_of_the_action_kind(continuous):
    controller = PlannerController("true")
    env = NavigationGymEnv(continuous=continuous)
    try:
        env.reset(seed=SEED)
        controller.reset(env)
        action = controller.act(None)
    finally:
        env.close()
    settings = TUNED_SETTINGS["continuous" if continuous else "discrete"]
    navigator = controller.navigator
    assert navigator.params.margin == settings.margin
    assert navigator.follower_params.turn_in_place == settings.turn_in_place
    assert (navigator.follower_params.v_max, navigator.follower_params.dt) == (env.v_max, env.dt)
    if continuous:
        assert np.shape(action) == (2,)
    else:
        assert action in range(len(DISCRETE_ACTIONS))


def test_the_tuned_settings_are_pr_2s_and_live_on_the_robot_side():
    """rl_navigation.planning holds them, so the robot imports the settings the comparison judged."""
    import rl_navigation.planning as planning

    assert {"TUNED_SETTINGS", "PlannerSettings", "DEFAULT_MAP_RANGE"} <= set(planning.__all__)
    assert DEFAULT_MAP_RANGE == 1.0
    assert TUNED_SETTINGS["continuous"].margin == 0.05
    assert TUNED_SETTINGS["continuous"].turn_in_place == pytest.approx(math.radians(25))
    assert TUNED_SETTINGS["discrete"].margin == 0.06
    assert TUNED_SETTINGS["discrete"].turn_in_place == pytest.approx(math.radians(45))


def test_the_planner_rows_name_their_map_and_action_kind():
    assert PLANNER_ROWS == {
        "P-true-c": ("true", "continuous"),
        "P-true-d": ("true", "discrete"),
        "P-online-c": ("online", "continuous"),
        "P-online-d": ("online", "discrete"),
    }


def test_the_ladder_level_is_the_margin_the_plan_needed():
    """A 0.33 m gap centred on a cell row: shut at the continuous margin (0.175 m on each side of
    the centre line), open at half of it (0.15 m)."""
    gap = Arena(5.0, 5.0, boxes=[[2.4, 0.0, 2.6, 2.36], [2.4, 2.69, 2.6, 5.0]])
    controller = PlannerController("true")
    env = NavigationGymEnv()
    try:
        obs, _ = env.reset(
            options={"layout": Layout(gap, np.array([1.0, 2.525, 0.0]), [4.0, 2.525])}
        )
        controller.reset(env)
        assert controller.ladder_level is None, "no plan before the first step"
        controller.act(obs)
    finally:
        env.close()
    assert controller.ladder_level == 1


# A goal walled in by four boxes: no path at any margin.
WALLED_GOAL = Layout(
    Arena(
        5.0,
        5.0,
        boxes=[
            [3.0, 3.0, 4.0, 3.1],
            [3.0, 3.9, 4.0, 4.0],
            [3.0, 3.0, 3.1, 4.0],
            [3.9, 3.0, 4.0, 4.0],
        ],
    ),
    np.array([1.0, 1.0, 0.0]),
    np.array([3.5, 3.5]),
)


def test_with_no_path_ever_found_the_ladder_level_is_none():
    controller = PlannerController("true")
    env = NavigationGymEnv()
    try:
        obs, _ = env.reset(options={"layout": WALLED_GOAL})
        controller.reset(env)
        for _ in range(3):
            obs, *_ = env.step(controller.act(obs))
    finally:
        env.close()
    assert controller.navigator.status == "no_path"
    assert controller.ladder_level is None


def test_with_no_path_the_shortest_length_is_the_straight_line():
    assert shortest_length(WALLED_GOAL, robot_radius=0.1) == pytest.approx(np.hypot(2.5, 2.5))


def test_an_observation_of_dropouts_writes_no_cell_of_the_online_map():
    """A dropout reads the sensor's maximum range (1 in the observation): not a hit, and beyond
    the map range, so the online map stays unknown."""
    controller = PlannerController("online")
    env = NavigationGymEnv()
    try:
        obs, _ = env.reset(seed=SEED)
        controller.reset(env)
        dropped = np.array(obs)
        dropped[5:] = 1.0
        controller.act(dropped)
    finally:
        env.close()
    assert controller.map_range == DEFAULT_MAP_RANGE
    assert not controller.world_map.occupancy.log_odds.any(), "no cell written"


def test_a_planner_refuses_an_unknown_map_source():
    with pytest.raises(ValueError, match="map_source"):
        PlannerController("lidar")


def test_the_online_planner_refuses_an_observation_that_is_not_the_raw_one():
    controller = PlannerController("online")
    env = NavigationGymEnv()
    try:
        obs, _ = env.reset(seed=SEED)
        controller.reset(env)
        with pytest.raises(ValueError, match="observation"):
            controller.act(np.concatenate([obs, obs]))  # a frame stack, say
    finally:
        env.close()


def test_a_controller_needs_a_navigation_environment():
    with pytest.raises(TypeError, match="NavigationGymEnv"):
        play(object(), PolicyController(Straight()), SEED)


# ---------------------------------------------------------------------------
# records and report
# ---------------------------------------------------------------------------


def _rows():
    return [
        _row("P-true-c", "ssss"),
        _row("P-online-c", "ssst"),
        _row("P-online-c", "sstt", "dropout_prob", 0.4),
        # Two planner rows no policy row matches: another value, and the value on another setting.
        _row("P-online-c", "ssss", "dropout_prob", 0.1),
        _row("P-online-c", "tttt", "sensor_noise_std", 0.4),
        _row("TD3 s1", "sctt"),
        _row("TD3 s1", "ssss", "dropout_prob", 0.4),
    ]


def test_records_round_trip_through_json(tmp_path):
    path = tmp_path / "out" / "records.json"
    header = [("Seeds", "0 to 3"), ("Config", "`a.yaml`")]
    rows = [*_rows(), _row("x", "s", kind="planner", radius=0.125, ladder_level=2)]
    save_records(path, rows, header)
    assert load_records(path) == (rows, header)


def test_loading_refuses_another_format(tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"format": 99, "header": [], "rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        load_records(path)


def test_merging_joins_the_rows_of_files_played_on_the_same_seeds(tmp_path):
    rows = _rows()
    save_records(tmp_path / "a.json", rows[:2], [("Seeds", "0 to 3"), ("Job", "a")])
    save_records(tmp_path / "b.json", rows[2:], [("Seeds", "0 to 3"), ("Job", "b")])
    merged, header = merge_records([tmp_path / "a.json", tmp_path / "b.json"])
    assert merged == rows
    assert header == [("Seeds", "0 to 3"), ("Job", "a"), ("Job", "b")], "repeated lines once"


def test_merging_refuses_other_seeds_or_a_row_twice(tmp_path):
    rows = _rows()
    save_records(tmp_path / "a.json", rows[:1], [])
    other = _row("P-true-d", "ss", first_seed=7)
    save_records(tmp_path / "b.json", [other], [])
    with pytest.raises(ValueError, match="seeds"):
        merge_records([tmp_path / "a.json", tmp_path / "b.json"])
    with pytest.raises(ValueError, match="twice"):
        merge_records([tmp_path / "a.json", tmp_path / "a.json"])


def test_the_report_states_the_gates_and_pairs_each_policy_with_each_planner():
    report = render_report(_rows(), title="Planning comparison", header=[("Seeds", "0 to 3")])
    assert report.startswith("# Planning comparison\n")
    assert "- **Seeds**: 0 to 3" in report
    # G1 holds on 4 of 4; G3c fails at 3 of 4 (75 % < 93 %); G2 and G3d were not played.
    assert "| G1 | P-true-c | 100.0% (4/4) | 0 (0.0%) | 0.200 | pass |" in report
    assert "| G3c | P-online-c | 75.0% (3/4) | 0 (0.0%) | 0.200 | FAIL |" in report
    assert "| G2 | P-true-d | – | – | – | not played |" in report
    assert "| G3d | P-online-d | – | – | – | not played |" in report
    assert "r = 0.1 m" in report, "the radius is the rows' own"
    # TD3 against P-online-c at dropout 0.4: TD3 only on seeds 2 and 3, McNemar 2 / 2^2 * 2 = 0.5.
    assert "| TD3 s1 | P-online-c | dropout_prob | 0.4 | 2 | 2 | 0 | 0.500 |" in report
    # Nominal: against both planners; never across a perturbation.
    assert "| TD3 s1 | P-true-c | nominal | – | 1 | 0 | 3 | 0.250 |" in report
    assert report.count("| TD3 s1 | P-") == 3
    # Like for like: the continuous policy against the online continuous planner.
    assert report.count("| primary |") == 2
    assert "| TD3 s1 | P-online-c | nominal | – | 1 | 0 | 2 | 0.500 |" in report
    nominal_true = next(line for line in report.splitlines() if "| TD3 s1 | P-true-c |" in line)
    assert nominal_true.endswith("| – |"), "the oracle pairing is not primary"
    # The SPL biases are stated.
    assert "overestimates" in report and "goal radius" in report
    assert "| P-online-c | dropout_prob | 0.4 | 4 | 50.0% (15.0–85.0) |" in report


def test_the_report_gives_the_spread_across_an_agents_checkpoints_and_no_pooled_claim():
    rows = [
        _row("P-online-c", "ssst"),
        _row("TD3 s1", "ssss"),
        _row("TD3 s2", "sstt"),
        _row("DQN s42", "tttt"),
    ]
    report = render_report(rows, title="t", header=[])
    # TD3: 100 % and 50 % against the planner's 75 %, so +25 and -25 points.
    assert "| TD3 | P-online-c | nominal | – | 2 | 50.0 to 100.0% | -25.0 to +25.0 |" in report
    assert "| DQN | P-online-c |" not in report, "one checkpoint has no spread"


def test_a_report_with_one_checkpoint_per_agent_has_no_spread_section():
    assert "Spread" not in render_report(_rows(), title="t", header=[])


def test_a_discrete_policy_is_primary_against_the_online_discrete_planner():
    rows = [_row("P-online-d", "ss"), _row("P-online-c", "ss"), _row("DQN s42", "ss")]
    lines = render_report(rows, title="t", header=[]).splitlines()
    assert next(x for x in lines if x.startswith("| DQN s42 | P-online-d |")).endswith(
        "| primary |"
    )
    assert next(x for x in lines if x.startswith("| DQN s42 | P-online-c |")).endswith("| – |")


def test_g4_judges_the_nominal_planner_rows_by_the_maintainers_decision():
    """The maintainer's decision, 2026-11-15: the four planner rows at nominal sensors only."""
    assert G4_SCOPE == "nominal"


# _rows() has P-true-c and P-online-c nominal (P-true-d and P-online-d not played) and three
# perturbed P-online-c rows.
@pytest.mark.parametrize(("scope", "rows_timed"), [("nominal", 4), ("every", 7)])
def test_the_g4_scope_chooses_the_rows_timed_and_the_wording(monkeypatch, scope, rows_timed):
    monkeypatch.setattr(report_module, "G4_SCOPE", scope)
    report = render_report(_rows(), title="t", header=[])
    g4 = report[report.index("| G4 Row |") :].split("\n\n")[0].splitlines()[2:]
    assert len(g4) == rows_timed
    wording = "each planner's nominal row" if scope == "nominal" else "every planner row"
    assert f"**G4** ({wording})" in report


def test_a_report_needs_one_robot_radius():
    rows = [_row("P-true-c", "s"), _row("P-true-d", "s", radius=0.2)]
    with pytest.raises(ValueError, match="robot radius"):
        render_report(rows, title="t", header=[])


def test_g4_is_judged_from_the_solo_timing_rows_when_given():
    timing = [
        _row("P-true-c", "s", step_ms=(50.0,) * 10),  # sliding p99 50 ms: fails G4
        _row("P-online-c", "s", step_ms=(2.0,) * 10),
    ]
    report = render_report(
        _rows(), title="t", header=[], timing_rows=timing, timing_source="`g4-solo.json`"
    )
    g4 = report[report.index("| G4 Row |") :].split("\n\n")[0].splitlines()[2:]
    assert g4 == [
        "| P-true-c | nominal | – | 50.00 | 50.0 | FAIL |",
        "| P-true-d | nominal | – | – | – | not played |",
        "| P-online-c | nominal | – | 2.00 | 2.0 | pass |",
        "| P-online-d | nominal | – | – | – | not played |",
    ], "the solo run's rows, not the comparison's, every planner named"
    assert "G4 is judged from the solo timing run, `g4-solo.json`" in report


def test_without_solo_timing_the_report_says_g4_is_not_judged_from_a_solo_run():
    report = render_report(_rows(), title="t", header=[])
    assert "not the solo timing run G4 is judged from" in report


def test_a_nominal_planner_row_missing_from_the_run_is_not_played_for_g4():
    report = render_report([_row("P-online-d", "s")], title="t", header=[])
    g4 = report[report.index("| G4 Row |") :].split("\n\n")[0].splitlines()[2:]
    assert [line.split(" | ")[0] for line in g4] == [
        "| P-true-c",
        "| P-true-d",
        "| P-online-c",
        "| P-online-d",
    ]
    assert sum(line.endswith("| not played |") for line in g4) == 3

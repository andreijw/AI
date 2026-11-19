"""scripts/planning_timing: the planning timing rows' seed ranges (TESTING.md PLAN-1)."""

import pytest

from tests.helpers import load_script


@pytest.fixture(scope="module")
def time_planner():
    return load_script("planning_timing/time_planner")


def test_plan_1_times_the_tuning_layouts_by_default(time_planner):
    assert time_planner.parse_seeds([]) == range(10000, 10200)


def test_a_seed_range_can_be_given(time_planner):
    assert time_planner.parse_seeds(["--first-seed", "30000", "--count", "500"]) == range(
        30000, 30500
    )


def test_an_empty_range_is_refused(time_planner, capsys):
    with pytest.raises(SystemExit):
        time_planner.parse_seeds(["--count", "0"])
    assert "--count" in capsys.readouterr().err


def test_the_script_runs_on_two_tuning_layouts_and_names_its_code_and_seeds(time_planner, capsys):
    import run_provenance

    time_planner.main(["--count", "2"])
    first, timings = capsys.readouterr().out.splitlines()[:2]
    root = time_planner.ROOT
    assert first == (
        f"commit {run_provenance.commit(root)}; working tree {run_provenance.working_tree(root)}; "
        "seeds 10000 to 10001"
    )
    out = timings.split()
    assert out[0] == "astar" and "from_occupied" in out

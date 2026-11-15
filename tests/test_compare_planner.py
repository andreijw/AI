"""compare_planner.py: the planning comparison's command line, which holds every rl_core and
train.py use (configs, checkpoints, the robustness sweep) and plays the rows through
rl_navigation.planning_comparison.

Every run here plays tuning layouts (10000 to 10199) for one or two episodes; 30000 to 30499 are
the judged layouts and are played once, by the judged run only.
"""

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from rl_core.robustness import Perturbation, perturbed_section, robustness_rows
from rl_navigation.benchmark import NavigationChallenge, NavigationContinuousChallenge
from rl_navigation.planning_comparison import PolicyController, load_records
from tests.helpers import run_main

FIRST = 10000  # tuning layouts
REPO_ROOT = Path(__file__).resolve().parent.parent


def _module():
    return importlib.import_module("compare_planner")


def _config(tmp_path, name="NavigationContinuous-v0", agent="random", file="config.yaml"):
    config = {
        "environment": {"name": name, "max_episode_steps": 400, "seed": 1, "num_obstacles": 6},
        "agent": {"type": agent, "config": {"seed": 3}},
        "training": {"num_episodes": 1, "max_steps_per_episode": 400},
    }
    path = tmp_path / file
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, str(path)


def _agent(challenge, config, seed=3):
    env = challenge.make_env(config["environment"], render_mode=None)
    try:
        return challenge.make_agent(config["agent"]["type"], {"seed": seed}, env)
    finally:
        env.close()


@pytest.mark.parametrize(
    "challenge", [NavigationContinuousChallenge(), NavigationChallenge()], ids=["c", "d"]
)
def test_a_policy_row_plays_what_the_robustness_benchmark_plays(tmp_path, challenge):
    """On 5 layouts, nominal and perturbed, a random policy played by the harness gives the same
    rewards, lengths and successes as rl_core.robustness_rows (two agents seeded alike, since
    each draws its evaluation actions from its own generator)."""
    config, _ = _config(tmp_path, name=challenge.name)
    dropout = Perturbation("dropout_prob", 0.1)
    expected = robustness_rows(
        challenge, config, _agent(challenge, config), [dropout], episodes=5, eval_seed=FIRST
    )
    agent = _agent(challenge, config)
    section = config["environment"]
    env = challenge.make_env(section, render_mode=None)
    try:
        perturbed = perturbed_section(section, env, dropout)[1]
    finally:
        env.close()
    module = _module()
    for row, played_section in zip(expected, (section, perturbed), strict=True):
        records, radius = module.play_row(
            challenge,
            played_section,
            PolicyController(module.GreedyAgent(agent)),
            range(FIRST, FIRST + 5),
            max_steps=400,
            shortest_cache={},
        )
        assert tuple(r.reward for r in records) == pytest.approx(row.rewards, abs=1e-9)
        assert tuple(r.steps for r in records) == row.lengths
        assert tuple(r.success for r in records) == row.successes
        assert radius == 0.1, "the environment's robot radius"
    assert len(set(expected[0].lengths)) > 1, "the five layouts play differently"


class Doubling:
    """An agent whose normalisation doubles the observation."""

    def __init__(self):
        self.calls = []

    def normalize_observation(self, observation, training=True):
        self.calls.append(("normalize", training))
        return 2 * observation

    def select_action(self, observation, training=True):
        self.calls.append(("select", training))
        return observation


def test_an_agent_acts_greedily_on_its_normalised_observation():
    agent = Doubling()
    assert _module().GreedyAgent(agent).select_action(np.array([1.0, 2.0])).tolist() == [2.0, 4.0]
    assert agent.calls == [("normalize", False), ("select", False)], "neither call trains"


def _random_checkpoint(tmp_path, name="NavigationContinuous-v0", label="run-a"):
    config, config_path = _config(tmp_path, name=name, file=f"{label}.yaml")
    challenge = (
        NavigationContinuousChallenge()
        if name == "NavigationContinuous-v0"
        else NavigationChallenge()
    )
    folder = tmp_path / label
    folder.mkdir()
    checkpoint = str(folder / "agent_episode_1.pt")
    _agent(challenge, config).save(checkpoint)
    return config_path, checkpoint


def _files(root):
    return sorted(
        os.path.relpath(os.path.join(d, f), root) for d, _, files in os.walk(root) for f in files
    )


def test_cli_plays_planners_nominal_and_sweeps_the_online_and_policy_rows(tmp_path, monkeypatch):
    config, checkpoint = _random_checkpoint(tmp_path)
    records, report = tmp_path / "out" / "records.json", tmp_path / "out" / "report.md"
    before = _files(tmp_path)
    run_main(
        [
            "--first-seed", str(FIRST), "--episodes", "2",
            "--planners", "P-true-d", "P-online-c",
            "--agent", config, checkpoint,
            "--sweep", "dropout_prob=0,0.4",
            "--records", str(records), "--report", str(report),
        ],
        monkeypatch,
        module="compare_planner",
    )  # fmt: skip
    assert _files(tmp_path) == sorted([*before, "out/records.json", "out/report.md"])
    rows, header = load_records(records)
    assert [(r.name, r.kind, r.actions, r.setting, r.value) for r in rows] == [
        ("P-true-d", "planner", "discrete", None, None),
        ("P-online-c", "planner", "continuous", None, None),
        ("P-online-c", "planner", "continuous", "dropout_prob", 0.0),
        ("P-online-c", "planner", "continuous", "dropout_prob", 0.4),
        ("RANDOM run-a", "policy", "continuous", None, None),
        ("RANDOM run-a", "policy", "continuous", "dropout_prob", 0.0),
        ("RANDOM run-a", "policy", "continuous", "dropout_prob", 0.4),
    ], "the true map nominal only; the online map and the policy at every sweep value"
    assert all([e.seed for e in r.episodes] == [FIRST, FIRST + 1] for r in rows)
    assert dict(header)["Seeds"] == f"{FIRST} to {FIRST + 1} (2 layouts)"
    assert {r.robot_radius for r in rows} == {0.1}
    md = report.read_text(encoding="utf-8")
    with open(checkpoint, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() in md, "the report names the checkpoint"
    assert "| RANDOM run-a | P-online-c | dropout_prob | 0.4 |" in md


def test_cli_without_output_paths_prints_and_writes_nothing(tmp_path, monkeypatch, capsys):
    before = _files(tmp_path)
    run_main(
        ["--first-seed", str(FIRST), "--episodes", "1", "--planners", "P-true-c"],
        monkeypatch,
        module="compare_planner",
    )
    assert "| P-true-c | nominal |" in capsys.readouterr().out
    assert _files(tmp_path) == before


def test_cli_merges_the_records_of_several_jobs_into_one_report(tmp_path, monkeypatch):
    config, checkpoint = _random_checkpoint(tmp_path, name="Navigation-v0", label="dqn-like")
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    common = ["--first-seed", str(FIRST), "--episodes", "1"]
    run_main(
        [*common, "--planners", "P-true-d", "--records", str(first)],
        monkeypatch,
        module="compare_planner",
    )
    run_main(
        [*common, "--planners", "--agent", config, checkpoint, "--records", str(second)],
        monkeypatch,
        module="compare_planner",
    )
    report = tmp_path / "report.md"
    run_main(
        ["--from-records", str(first), str(second), "--report", str(report)],
        monkeypatch,
        module="compare_planner",
    )
    md = report.read_text(encoding="utf-8")
    assert "| P-true-d | nominal |" in md and "| RANDOM dqn-like | nominal |" in md
    assert "| RANDOM dqn-like | P-true-d | nominal |" in md, "the jobs' rows are paired"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--episodes", "0"], "--episodes"),
        (["--sweep", "dropout_prob"], "SETTING=V1,V2"),
        (["--sweep", "dropout_prob=0.1", "--sweep", "dropout_prob=0.2"], "one --sweep"),
        (["--sweep", "friction=1.0"], "friction"),
        (["--sweep", "dropout_prob=2"], "dropout_prob"),
        (["--planners"], "nothing to play"),
        (["--from-records", "x.json"], "--from-records"),
    ],
)
def test_cli_refuses_bad_arguments_before_playing(tmp_path, monkeypatch, capsys, args, message):
    module = _module()
    played = []
    monkeypatch.setattr(module, "play", lambda *a, **k: played.append(a))
    base = ["--first-seed", str(FIRST), "--episodes", "1"]
    if args[:1] == ["--episodes"]:
        base = base[:2]
    monkeypatch.setattr("sys.argv", ["compare_planner.py", *base, *args])
    with pytest.raises(SystemExit):
        module.main()
    assert message in capsys.readouterr().err
    assert not played, "nothing is played after a bad argument"


def test_cli_needs_seeds_unless_it_merges(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["compare_planner.py", "--planners", "P-true-c"])
    with pytest.raises(SystemExit):
        _module().main()
    assert "--first-seed" in capsys.readouterr().err


def test_cli_runs_in_a_fresh_process(tmp_path):
    """A fresh interpreter has discovered no challenge yet: the script discovers them itself."""
    report = tmp_path / "report.md"
    result = subprocess.run(
        [
            sys.executable, "compare_planner.py", "--first-seed", str(FIRST), "--episodes", "1",
            "--planners", "P-true-c", "--report", str(report),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "| P-true-c | nominal |" in report.read_text(encoding="utf-8")


def _git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _planner_run(tmp_path, monkeypatch, records, *extra, planners=("P-true-c", "P-true-d")):
    run_main(
        [
            "--first-seed", str(FIRST), "--episodes", "1", "--planners", *planners,
            "--records", str(records), *extra,
        ],
        monkeypatch,
        module="compare_planner",
    )  # fmt: skip


def test_the_header_records_the_code_the_map_range_the_radius_and_the_load(tmp_path, monkeypatch):
    records = tmp_path / "records.json"
    _planner_run(tmp_path, monkeypatch, records, planners=("P-true-c",))
    header = dict(load_records(records)[1])
    assert header["Commit"] == _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    assert header["Working tree"].startswith("dirty; index tree " if dirty else "clean")
    assert header["Online map range"] == "1 m"
    assert header["Robot radius"] == "0.1 m"
    for when in ("start", "end"):
        values = header[f"Load average at {when} (records.json)"].split()
        assert len(values) == 3 and all(float(v) >= 0 for v in values)


def test_records_are_written_after_each_row_and_a_resume_plays_only_the_rest(tmp_path, monkeypatch):
    module = importlib.import_module("compare_planner")
    real, played = module.play_row, []

    def failing_second(*args, **kwargs):
        if played:
            raise RuntimeError("killed")
        played.append(args[2])
        return real(*args, **kwargs)

    records = tmp_path / "records.json"
    monkeypatch.setattr(sys, "argv", ["compare_planner.py"])
    monkeypatch.setattr(module, "play_row", failing_second)
    argv = ["--first-seed", str(FIRST), "--episodes", "1", "--planners", "P-true-c", "P-true-d"]
    monkeypatch.setattr(sys, "argv", ["compare_planner.py", *argv, "--records", str(records)])
    with pytest.raises(RuntimeError, match="killed"):
        module.main()
    rows, _ = load_records(records)
    assert [r.name for r in rows] == ["P-true-c"], "the first row was saved before the crash"

    calls = []

    def counting(*args, **kwargs):
        calls.append(args[1]["name"])
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "play_row", counting)
    monkeypatch.setattr(
        sys, "argv", ["compare_planner.py", *argv, "--records", str(records), "--resume"]
    )
    module.main()
    assert calls == ["Navigation-v0"], "only P-true-d is played"
    resumed, _ = load_records(records)
    assert [r.name for r in resumed] == ["P-true-c", "P-true-d"]
    assert resumed[0] == rows[0], "the saved row is kept, not replayed"


def test_existing_records_are_refused_without_resume(tmp_path, monkeypatch, capsys):
    records = tmp_path / "records.json"
    _planner_run(tmp_path, monkeypatch, records, planners=("P-true-c",))
    with pytest.raises(SystemExit):
        _planner_run(tmp_path, monkeypatch, records, planners=("P-true-c",))
    assert "--resume" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "message"),
    [("Commit", "commit"), ("Seeds", "seeds")],
)
def test_a_resume_refuses_records_of_other_code_or_seeds(
    tmp_path, monkeypatch, capsys, change, message
):
    records = tmp_path / "records.json"
    _planner_run(tmp_path, monkeypatch, records, planners=("P-true-c",))
    data = json.loads(records.read_text(encoding="utf-8"))
    data["header"] = [[k, "other" if k == change else v] for k, v in data["header"]]
    records.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit):
        _planner_run(tmp_path, monkeypatch, records, "--resume")
    assert message in capsys.readouterr().err


def test_merging_refuses_records_of_other_code(tmp_path, monkeypatch, capsys):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    _planner_run(tmp_path, monkeypatch, first, planners=("P-true-c",))
    _planner_run(tmp_path, monkeypatch, second, planners=("P-true-d",))
    data = json.loads(second.read_text(encoding="utf-8"))
    data["header"] = [[k, "other" if k == "Commit" else v] for k, v in data["header"]]
    second.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["compare_planner.py", "--from-records", str(first), str(second)]
    )
    with pytest.raises(SystemExit):
        _module().main()
    assert "commit" in capsys.readouterr().err


def test_cli_takes_g4_from_the_solo_timing_records(tmp_path, monkeypatch):
    comparison, timing = tmp_path / "a.json", tmp_path / "g4-solo.json"
    _planner_run(tmp_path, monkeypatch, comparison, planners=("P-true-c",))
    _planner_run(tmp_path, monkeypatch, timing, planners=("P-true-c",))
    report = tmp_path / "report.md"
    run_main(
        [
            "--from-records", str(comparison), "--timing-records", str(timing),
            "--report", str(report),
        ],
        monkeypatch,
        module="compare_planner",
    )  # fmt: skip
    md = report.read_text(encoding="utf-8")
    assert f"G4 is judged from the solo timing run, `{timing}`" in md
    assert "Load average at start (g4-solo.json)" in md, "the solo run's header is kept"


def test_timing_records_go_with_from_records_only(monkeypatch, capsys):
    module = _module()
    played = []
    monkeypatch.setattr(module, "play", lambda *a, **k: played.append(a))
    argv = ["--first-seed", str(FIRST), "--episodes", "1", "--timing-records", "x"]
    monkeypatch.setattr(sys, "argv", ["compare_planner.py", *argv])
    with pytest.raises(SystemExit):
        module.main()
    assert "--timing-records goes with --from-records" in capsys.readouterr().err
    assert not played, "refused before anything is played"

"""Characterization of BenchmarkRunner.run: its exact console output, metrics JSON and report.

The expected files in tests/data/benchmark_characterization/ were recorded from the runner before
it was split into helpers, with a scripted trainer and a fixed clock, so they hold on every
platform. A refactor of the runner must leave them unchanged. A deliberate change to what the
benchmark writes regenerates them: run this file with RECORD_BENCHMARK_CHARACTERIZATION=1, check the
diff of tests/data/benchmark_characterization/, then run it again without the variable (a
recording run fails on purpose, so it can never pass by accepting any output).
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import benchmark
from benchmark import BenchmarkRunner, BenchmarkSettings
from tests.test_benchmark import PerAgentBudgetChallenge, ScriptedTrainer

EXPECTED_DIR = Path(__file__).parent / "data" / "benchmark_characterization"
RECORD = os.environ.get("RECORD_BENCHMARK_CHARACTERIZATION") == "1"


class _SeparateEvalChallenge(PerAgentBudgetChallenge):
    """Every run evaluates on its own environment, so the report says so."""

    def setup_eval_env(self, agent_type, seed):
        return self.make_env({"seed": seed}, render_mode=None)


class _FixedClock:
    """perf_counter whose step grows by 0.25 s per call, so every run gets its own exact duration
    and a duration reported against the wrong seed shows."""

    def __init__(self) -> None:
        self.now = 0.0
        self.step = 1.0

    def perf_counter(self) -> float:
        self.step += 0.25
        self.now += self.step
        return self.now


SCENARIOS = {
    "own_budgets_training_env_eval": (
        PerAgentBudgetChallenge,
        {"agents": ("long", "short"), "seeds": (1, 2), "window": 1, "eval_episodes": 1},
    ),
    "episode_override_separate_eval": (
        _SeparateEvalChallenge,
        {"agents": ("short", "long"), "seeds": (3, 2, 1), "episodes": 4, "window": 2},
    ),
}


def _run(scenario, tmp_path, monkeypatch, capsys):
    challenge_cls, kwargs = SCENARIOS[scenario]
    monkeypatch.setattr(benchmark, "Trainer", ScriptedTrainer)
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=_FixedClock().perf_counter))
    out_dir = tmp_path / "out"
    report = tmp_path / "report.md"
    result = BenchmarkRunner(challenge_cls()).run(
        BenchmarkSettings(output_dir=str(out_dir), **kwargs),
        report_path=str(report),
        save_plot=False,
    )
    console = capsys.readouterr().out

    def portable(text: str) -> str:
        return text.replace(str(tmp_path), "<tmp>").replace(os.sep, "/")

    return result, {
        "console.txt": portable(console),
        "metrics.json": portable(Path(result["metrics_json"]).read_text(encoding="utf-8")),
        "report.txt": portable(report.read_text(encoding="utf-8")),
    }


def _record(scenario: str, outputs: dict[str, str]) -> None:
    """Overwrite the expected files of one scenario with this run's outputs."""
    target = EXPECTED_DIR / scenario
    target.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (target / name).write_text(text, encoding="utf-8")


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("output", ["console.txt", "metrics.json", "report.txt"])
def test_runner_output_is_unchanged(scenario, output, tmp_path, monkeypatch, capsys):
    _, outputs = _run(scenario, tmp_path, monkeypatch, capsys)
    if RECORD:
        _record(scenario, outputs)
        pytest.fail(
            f"Recorded {scenario}/{output}; rerun without RECORD_BENCHMARK_CHARACTERIZATION."
        )
    expected = (EXPECTED_DIR / scenario / output).read_text(encoding="utf-8")
    assert outputs[output] == expected, f"{scenario}/{output} differs from the recorded output."


def test_the_recorded_outputs_cover_both_evaluation_modes():
    """The two scenarios differ where the runner branches: own budgets against an override,
    evaluation on the training environment against a separate one."""
    own = (EXPECTED_DIR / "own_budgets_training_env_eval" / "report.txt").read_text("utf-8")
    separate = (EXPECTED_DIR / "episode_override_separate_eval" / "report.txt").read_text("utf-8")
    assert own != separate
    assert "4 episodes for every agent" in separate
    assert "4 episodes for every agent" not in own
    assert "Evaluation runs on a separate environment" in separate
    assert "Evaluation runs on a separate environment" not in own

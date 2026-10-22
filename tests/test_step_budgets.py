"""Training budgets in environment steps, and the step-based solve measures (CHANGELOG.md,
2026-10-16)."""

import warnings

import numpy as np
import pytest

from benchmark import classify_run
from rl_core.agents.base_agent import BaseAgent
from rl_core.training.budget import apply_budget_override
from rl_core.training.trainer import Trainer
from rl_core.utils.visualization import plot_benchmark_comparison


class _FixedLengthEnv:
    """Every episode lasts exactly `length` steps; reward 1 per step."""

    def __init__(self, length: int):
        self.length, self.t = length, 0

    def reset(self, **_kwargs):
        self.t = 0
        return np.zeros(2), {}

    def step(self, _action):
        self.t += 1
        return np.zeros(2), 1.0, self.t >= self.length, False, {}


class _Idle(BaseAgent):
    def __init__(self, anneal_steps=None):
        super().__init__(2, 2, {})
        self.anneal_steps = anneal_steps

    def select_action(self, observation, training=True):
        return 0

    def update(self, batch):
        return {}

    def _checkpoint_arrays(self):
        return {}

    def _load_checkpoint_arrays(self, data):
        pass


def _trainer(length=30, anneal_steps=None, **config):
    settings = {"num_episodes": 1000, "save_frequency": 10_000, "eval_frequency": 10_000, **config}
    return Trainer(agent=_Idle(anneal_steps), env=_FixedLengthEnv(length), config=settings)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def test_step_budget_stops_after_the_episode_that_crosses_it():
    """30-step episodes, budget 50: the second episode crosses it and is finished (60 steps)."""
    trainer = _trainer(total_steps=50)
    summary = trainer.train()
    assert len(trainer.episode_rewards) == 2
    assert trainer.total_steps_trained == 60
    assert (summary["total_episodes"], summary["total_steps"]) == (2, 60)


def test_without_a_step_budget_the_episode_count_rules():
    trainer = _trainer(num_episodes=3)
    summary = trainer.train()
    assert (summary["total_episodes"], summary["total_steps"]) == (3, 90)


def test_evaluation_episodes_do_not_count_as_training_steps():
    trainer = _trainer(total_steps=50, eval_frequency=1)
    trainer.train()
    assert trainer.total_steps_trained == 60


@pytest.mark.parametrize("bad", [0, -5, 2.5, "many"])
def test_step_budget_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="total_steps must be a positive integer"):
        _trainer(total_steps=bad)


def test_a_learning_schedule_longer_than_the_budget_is_warned_about():
    with pytest.warns(UserWarning, match="anneal_steps 200000 exceeds the step budget 100000"):
        _trainer(total_steps=100_000, anneal_steps=200_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _trainer(total_steps=100_000, anneal_steps=100_000)


# ---------------------------------------------------------------------------
# One rule for budget overrides
# ---------------------------------------------------------------------------


def test_an_episode_override_replaces_a_step_budget():
    training = {"num_episodes": 10, "total_steps": 5000}
    apply_budget_override(training, episodes=3, steps=None)
    assert training == {"num_episodes": 3}


def test_a_step_override_sets_the_step_budget():
    training = {"num_episodes": 10}
    apply_budget_override(training, episodes=None, steps=4000)
    assert training == {"num_episodes": 10, "total_steps": 4000}


def test_no_override_leaves_the_budget_alone():
    training = {"num_episodes": 10, "total_steps": 5000}
    apply_budget_override(training, episodes=None, steps=None)
    assert training == {"num_episodes": 10, "total_steps": 5000}


def test_episode_and_step_overrides_together_are_rejected():
    with pytest.raises(ValueError, match="either an episode or a step budget"):
        apply_budget_override({}, episodes=3, steps=4000)


# ---------------------------------------------------------------------------
# Steps to solve and held
# ---------------------------------------------------------------------------


def test_steps_to_solve_counts_every_step_up_to_the_solve_episode():
    rewards, lengths = [10.0, 10.0, 500.0, 500.0], [10, 10, 500, 500]
    result = classify_run(rewards, 0.0, threshold=475.0, window=2, episode_lengths=lengths)
    assert result["solved_episode"] == 4
    assert result["solved_step"] == 1020


def test_a_run_shorter_than_the_window_is_not_held():
    result = classify_run([500.0] * 3, 500.0, threshold=475.0, window=10, episode_lengths=[500] * 3)
    assert result["held"] is False
    assert result["is_solved"] is False


def test_held_means_the_last_full_window_is_at_the_threshold():
    rewards = [500.0] * 10 + [100.0] * 5 + [500.0] * 5
    assert classify_run(rewards, 0.0, 475.0, window=5)["held"] is True
    assert classify_run(rewards, 0.0, 475.0, window=10)["held"] is False


def test_held_implies_solved():
    rewards = [100.0] * 20 + [500.0] * 10
    result = classify_run(rewards, 0.0, 475.0, window=10)
    assert result["held"] and result["is_solved"]


def test_without_a_threshold_held_is_none():
    assert classify_run([1.0] * 20, 0.0, threshold=None, window=10)["held"] is None


def test_steps_to_solve_is_none_without_episode_lengths_or_a_solve():
    assert classify_run([500.0] * 10, 0.0, 475.0, window=5)["solved_step"] is None
    assert (
        classify_run([1.0] * 10, 0.0, 475.0, window=5, episode_lengths=[1] * 10)["solved_step"]
        is None
    )


# ---------------------------------------------------------------------------
# Comparison plot against environment steps
# ---------------------------------------------------------------------------


def test_plot_against_steps_keeps_runs_with_different_episode_counts(tmp_path, monkeypatch):
    """A strong agent finishes its budget in fewer, longer episodes; its curve must not be cut short."""
    import matplotlib.pyplot as plt

    figures = []
    monkeypatch.setattr(plt, "close", lambda fig=None: figures.append(fig))
    curves = {"strong": [[100.0] * 5, [120.0] * 4], "weak": [[20.0] * 25, [10.0] * 50]}
    lengths = {"strong": [[100] * 5, [120] * 4], "weak": [[20] * 25, [10] * 50]}
    plot_benchmark_comparison(
        curves, window=2, save_path=str(tmp_path / "p.png"), episode_lengths=lengths
    )
    ax = figures[0].axes[0]
    assert ax.get_xlabel() == "Environment steps"
    strong, weak = ax.lines[0].get_xdata(), ax.lines[1].get_xdata()
    # Each agent's curve runs to its shortest run's total steps, from where every run has data.
    assert (strong[0], strong[-1]) == (120.0, 480.0)
    assert (weak[0], weak[-1]) == (20.0, 500.0)


def test_runner_plots_against_steps_only_when_every_budget_is_in_steps(tmp_path, monkeypatch):
    import benchmark

    seen = []
    monkeypatch.setattr(
        benchmark,
        "plot_benchmark_comparison",
        lambda *a, **k: seen.append(k.get("episode_lengths")),
    )
    from rl_cartpole.benchmark import CartPoleChallenge

    runner = benchmark.BenchmarkRunner(CartPoleChallenge())
    kwargs = {"agents": ["random"], "seeds": [1], "eval_episodes": 1, "save_plot": True}
    runner.run(steps=60, output_dir=str(tmp_path / "a"), **kwargs)
    runner.run(episodes=3, output_dir=str(tmp_path / "b"), **kwargs)
    assert seen[0] is not None and seen[1] is None


def test_budget_of_reads_steps_first():
    from rl_core.training.budget import budget_of

    assert budget_of({"num_episodes": 10, "total_steps": 500}) == (500, "steps")
    assert budget_of({"num_episodes": 10}) == (10, "episodes")


def test_a_budget_met_exactly_starts_no_further_episode():
    trainer = _trainer(total_steps=60)
    trainer.train()
    assert (len(trainer.episode_rewards), trainer.total_steps_trained) == (2, 60)

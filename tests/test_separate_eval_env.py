"""Evaluation on its own environment with a fixed set of start states (roadmap: "Separate Evaluation Environment").

Before this change evaluation ran on the training environment, so each evaluation consumed start states (and,
for the random agents, generator draws) that training would otherwise have used: how often a run evaluated
changed its training episodes.
"""

import glob
import json

import numpy as np
import pytest
import yaml

from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import ContinuousRandomAgent, RandomAgent, ReinforceAgent
from rl_core.seeding import eval_seed_for
from rl_core.training import Trainer
from tests.helpers import minimal_config, run_main

# ---------------------------------------------------------------------------
# The acceptance criterion, through train.py
# ---------------------------------------------------------------------------


def _training_averages(tmp_path, monkeypatch, agent_type: str, eval_frequency: int) -> list[float]:
    tmp_path.mkdir(parents=True)
    path = minimal_config(tmp_path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["agent"]["type"] = agent_type
    config["agent"]["config"]["seed"] = 7
    config["training"].update(
        {"num_episodes": 30, "eval_frequency": eval_frequency, "save_frequency": 100}
    )
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    run_main(["--config", path], monkeypatch)
    (metrics,) = glob.glob(str(tmp_path / "logs" / "cartpole" / agent_type / "metrics_*.jsonl"))
    with open(metrics, encoding="utf-8") as f:
        return [row["avg_reward"] for row in map(json.loads, f) if "avg_reward" in row]


@pytest.mark.parametrize("agent_type", ["random", "reinforce"])
def test_how_often_a_run_evaluates_does_not_change_its_training(tmp_path, monkeypatch, agent_type):
    often = _training_averages(tmp_path / "often", monkeypatch, agent_type, eval_frequency=1)
    never = _training_averages(tmp_path / "never", monkeypatch, agent_type, eval_frequency=1000)
    assert len(often) == 3
    assert often == never, (
        "evaluation must not consume the training episodes' start states or draws"
    )


def test_train_builds_and_closes_a_separate_evaluation_environment(tmp_path, monkeypatch):
    closed = []
    original_close = CartPoleEnv.close
    monkeypatch.setattr(
        CartPoleEnv, "close", lambda self: closed.append(id(self)) or original_close(self)
    )
    run_main(["--config", minimal_config(tmp_path)], monkeypatch)
    assert len(set(closed)) == 2, "the training and the evaluation environment are both closed"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class _RecordingEnv(CartPoleEnv):
    """CartPoleEnv that records the seed of every reset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_seeds = []

    def reset(self, seed=None, options=None):
        self.reset_seeds.append(seed)
        return super().reset(seed=seed, options=options)


def _trainer(eval_env=None, eval_seed=None, **config):
    env = _RecordingEnv(seed=1, max_episode_steps=20)
    agent = ReinforceAgent(4, 2, {"seed": 0})
    settings = {
        "num_episodes": 3,
        "max_steps_per_episode": 20,
        "eval_frequency": 100,
        "save_frequency": 100,
    }
    trainer = Trainer(
        env=env, agent=agent, config={**settings, **config}, eval_env=eval_env, eval_seed=eval_seed
    )
    return trainer, env


def test_every_evaluation_plays_the_same_start_states():
    eval_env = _RecordingEnv(seed=1, max_episode_steps=20)
    trainer, env = _trainer(eval_env=eval_env, eval_seed=1000)
    first = trainer.evaluate(num_episodes=3)
    second = trainer.evaluate(num_episodes=3)
    assert eval_env.reset_seeds == [1000, 1001, 1002] * 2
    assert first == second, (
        "a greedy policy that did not change scores the same on the same start states"
    )
    assert env.reset_seeds == [], "evaluation does not touch the training environment"
    eval_env.close()
    env.close()


def test_without_an_evaluation_environment_evaluation_stays_on_the_training_environment():
    trainer, env = _trainer()
    trainer.evaluate(num_episodes=2)
    assert env.reset_seeds == [None, None], "the fallback for direct Trainer callers and tests"
    env.close()


def test_an_evaluation_environment_needs_an_evaluation_seed():
    eval_env = CartPoleEnv(seed=1)
    with pytest.raises(ValueError, match="eval_seed"):
        _trainer(eval_env=eval_env)
    eval_env.close()


@pytest.mark.parametrize("bad", [-1, True, 2.5])
def test_the_evaluation_seed_must_be_a_non_negative_integer(bad):
    eval_env = CartPoleEnv(seed=1)
    with pytest.raises(ValueError, match="eval_seed"):
        _trainer(eval_env=eval_env, eval_seed=bad)
    eval_env.close()


# ---------------------------------------------------------------------------
# The evaluation seed
# ---------------------------------------------------------------------------


def test_the_evaluation_seed_is_reproducible_and_apart_from_training_seeds():
    assert eval_seed_for(42) == eval_seed_for(42)
    assert eval_seed_for(42) != eval_seed_for(43)
    for seed in (0, 1, 42, 123):
        assert 2**31 <= eval_seed_for(seed) < 2**32, "far above the small seeds runs train with"


def test_a_run_without_a_seed_still_gets_one_fixed_evaluation_seed():
    value = eval_seed_for(None)
    assert isinstance(value, int) and 2**31 <= value < 2**32


# ---------------------------------------------------------------------------
# Random agents: evaluation draws from its own generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_agent",
    [lambda: RandomAgent(4, 2, {"seed": 5}), lambda: ContinuousRandomAgent(3, 1, {"seed": 5})],
)
def test_a_random_agents_evaluation_actions_do_not_shift_its_training_actions(make_agent):
    plain, evaluated = make_agent(), make_agent()
    obs = np.zeros(plain.observation_dim)
    for _ in range(5):
        evaluated.select_action(obs, training=False)
    plain_actions = [plain.select_action(obs, training=True) for _ in range(50)]
    evaluated_actions = [evaluated.select_action(obs, training=True) for _ in range(50)]
    np.testing.assert_array_equal(np.array(plain_actions), np.array(evaluated_actions))


# ---------------------------------------------------------------------------
# Challenges
# ---------------------------------------------------------------------------


def test_a_config_challenge_builds_a_fresh_evaluation_environment():
    from rl_cartpole.benchmark import CartPoleChallenge

    env, _, _ = CartPoleChallenge().setup_run("random", seed=3, episodes=1)
    eval_env = CartPoleChallenge().setup_eval_env("random", seed=3)
    try:
        assert isinstance(eval_env, CartPoleEnv) and eval_env is not env
    finally:
        env.close()
        eval_env.close()


def test_a_challenge_without_configs_has_no_evaluation_environment():
    from tests.test_benchmark import CustomGenericChallenge

    assert CustomGenericChallenge().setup_eval_env("mock_agent", seed=1) is None


def test_the_benchmark_evaluates_on_a_separate_environment(tmp_path, monkeypatch):
    import benchmark

    seen = []
    original = benchmark.Trainer

    def spy(*args, **kwargs):
        seen.append((kwargs.get("eval_env"), kwargs.get("eval_seed")))
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark, "Trainer", spy)
    benchmark.run_benchmark(
        challenge_name="cartpole", agents=["random"], seeds=[2], episodes=2,
        output_dir=str(tmp_path / "out"), report_path=str(tmp_path / "report.md"), save_plot=False,
        eval_episodes=1,
    )  # fmt: skip
    (eval_env, eval_seed) = seen[0]
    assert isinstance(eval_env, CartPoleEnv)
    assert eval_seed == eval_seed_for(2)
    assert SEPARATE_EVAL_SENTENCE in (tmp_path / "report.md").read_text(encoding="utf-8")


SEPARATE_EVAL_SENTENCE = "Evaluation runs on a separate environment"


def test_the_report_does_not_claim_separate_evaluation_for_a_challenge_without_configs(tmp_path):
    from benchmark import BenchmarkRunner
    from tests.test_benchmark import CustomGenericChallenge

    report = tmp_path / "report.md"
    BenchmarkRunner(CustomGenericChallenge()).run(
        agents=["mock_agent"], episodes=2, seeds=[11], output_dir=str(tmp_path / "out"),
        report_path=str(report), save_plot=False, eval_episodes=1,
    )  # fmt: skip
    assert SEPARATE_EVAL_SENTENCE not in report.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "make_agent",
    [
        lambda seed: RandomAgent(4, 2, {"seed": seed}),
        lambda seed: ContinuousRandomAgent(3, 1, {"seed": seed}),
    ],
)
def test_a_random_agents_evaluation_generator_survives_a_checkpoint(tmp_path, make_agent):
    agent = make_agent(5)
    obs = np.zeros(agent.observation_dim)
    for _ in range(3):
        agent.select_action(obs, training=False)
    path = str(tmp_path / "random.pt")
    agent.save(path)
    restored = make_agent(99)  # another seed: the evaluation stream must come from the file
    restored.load(path)
    expected = [agent.select_action(obs, training=False) for _ in range(20)]
    actual = [restored.select_action(obs, training=False) for _ in range(20)]
    np.testing.assert_array_equal(np.array(actual), np.array(expected))

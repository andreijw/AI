"""The success-rate metric: ``info["is_success"]`` at the step that ends an episode, counted by
``Trainer`` for training and evaluation, kept in the trainer file and absent for an environment
that never reports it (``rl_core.training.trainer``, ``rl_core.training.checkpoints``)."""

import gymnasium as gym
import numpy as np
import pytest

from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import RandomAgent
from rl_core.training import Trainer


class GoalEnv(gym.Env):
    """Episodes of ``length`` steps whose outcome is scripted: episode k ends with
    ``is_success`` = ``outcomes[k % len(outcomes)]``, or without the key where that is None.
    With ``terminal_only`` the key is set on the ending step only, else on every step (False
    before the end), as the navigation environment does."""

    observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, outcomes, length=3, terminal_only=False):
        self.outcomes, self.length, self.terminal_only = list(outcomes), length, terminal_only
        self.episode, self.t = -1, 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode, self.t = self.episode + 1, 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.t += 1
        terminated = self.t >= self.length
        outcome = self.outcomes[self.episode % len(self.outcomes)]
        info = {}
        if outcome is not None and (terminated or not self.terminal_only):
            info["is_success"] = outcome if terminated else False
        return np.zeros(1, np.float32), 1.0, terminated, False, info


class RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, metrics):
        self.records.append(metrics)

    def info(self, message):
        pass

    def warning(self, message):
        pass


def _trainer(env, directory, logger=None, **training):
    config = {"num_episodes": 4, "max_steps_per_episode": 10, "eval_frequency": 1000,
              "save_frequency": 1000, "checkpoint_dir": str(directory), **training}  # fmt: skip
    return Trainer(env=env, agent=RandomAgent(1, 2, {"seed": 0}), config=config, logger=logger)


def test_evaluation_reports_the_share_of_episodes_that_reached_the_goal(tmp_path):
    trainer = _trainer(GoalEnv([True, False, True, True]), tmp_path)
    stats = trainer.evaluate(num_episodes=4)
    assert stats["success_rate"] == pytest.approx(0.75), "three of the four episodes succeeded"
    assert [episode.success for episode in trainer.evaluate_episodes(4)] == [
        True, False, True, True,
    ], "each evaluated episode carries its own outcome"  # fmt: skip


def test_training_records_each_episodes_outcome_and_reports_its_rates(tmp_path):
    logger = RecordingLogger()
    # 110 one-step episodes: the first ten fail, then 7 of every 10 succeed.
    outcomes = [False] * 10 + ([True] * 7 + [False] * 3) * 10
    env = GoalEnv(outcomes, length=1)
    trainer = _trainer(env, tmp_path, logger, num_episodes=len(outcomes))
    stats = trainer.train()
    assert trainer.episode_successes == outcomes, "one outcome per training episode, in order"
    assert stats["final_success_rate"] == pytest.approx(0.7), (
        "the final rate covers the last 100 episodes, not the whole run (70 of 110)"
    )
    periodic = [r for r in logger.records if "episode" in r]
    assert [r["success_rate"] for r in periodic[:2]] == pytest.approx([0.0, 0.7]), (
        "each periodic record gives the success rate of its own ten episodes"
    )


@pytest.mark.parametrize(
    "terminal_only, expected",
    [(True, None), (False, False)],
    ids=["key-on-terminal-steps-only", "key-on-every-step"],
)
def test_an_episode_the_trainer_cuts_takes_the_last_steps_outcome(
    tmp_path, terminal_only, expected
):
    """An episode cut by max_steps_per_episode before the environment ends it has the outcome its
    last step reported: unknown when the key comes only with the ending step."""
    env = GoalEnv([True], length=10, terminal_only=terminal_only)
    trainer = _trainer(env, tmp_path, max_steps_per_episode=3, num_episodes=2)
    stats = trainer.train()
    assert trainer.episode_successes == [expected, expected]
    assert ("final_success_rate" in stats) is (expected is not None), (
        "a rate is reported only when every episode's outcome is known"
    )
    assert ("success_rate" in trainer.evaluate(2)) is (expected is not None)


def test_a_run_mixing_known_and_unknown_outcomes_reports_no_rate(tmp_path):
    trainer = _trainer(GoalEnv([True, None]), tmp_path, num_episodes=4, save_frequency=4)
    stats = trainer.train()
    assert trainer.episode_successes == [True, None, True, None]
    assert "final_success_rate" not in stats, "an unknown outcome leaves the rate unknown"
    assert "success_rate" not in trainer.evaluate(2)
    with np.load(tmp_path / "trainer_episode_4.npz") as data:
        assert "episode_successes" not in data.files, "an incomplete history is not saved"


# ---------------------------------------------------------------------------
# An environment that never reports success: every output exactly as before
# ---------------------------------------------------------------------------


def test_without_is_success_the_trainer_outputs_have_exactly_their_old_keys(tmp_path):
    logger = RecordingLogger()
    env = CartPoleEnv(seed=1, max_episode_steps=20)
    try:
        trainer = _cartpole_trainer(env, tmp_path, logger)
        stats = trainer.train()
        evaluation = trainer.evaluate(num_episodes=2)
    finally:
        env.close()
    assert set(evaluation) == {"mean_reward", "std_reward", "mean_length"}
    assert set(stats) == {
        "total_episodes", "total_steps", "avg_reward", "avg_length", "final_avg_reward",
    }  # fmt: skip
    assert set(next(r for r in logger.records if "episode" in r)) == {
        "episode", "avg_reward", "avg_length",
    }, "the periodic metrics record gains no key"  # fmt: skip
    assert trainer.episode_successes == [None] * 10
    with np.load(tmp_path / "trainer_episode_10.npz") as data:
        assert sorted(data.files) == sorted(CARTPOLE_TRAINER_FILE_KEYS)


def _cartpole_trainer(env, directory, logger):
    config = {"num_episodes": 10, "max_steps_per_episode": 20, "eval_frequency": 1000,
              "save_frequency": 10, "checkpoint_dir": str(directory)}  # fmt: skip
    return Trainer(env=env, agent=RandomAgent(4, 2, {"seed": 0}), config=config, logger=logger)


# A CartPole run's trainer file without an evaluation environment or curriculum, as on develop.
CARTPOLE_TRAINER_FILE_KEYS = [
    "agent_checkpoint", "agent_sha256", "agent_class", "env_id", "episode_rewards",
    "episode_lengths", "total_steps_trained", "update_metric_window", "non_finite_metrics_seen",
    "env_has_reset", "env_start_rng", "env_wrapper_rng",
]  # fmt: skip


# ---------------------------------------------------------------------------
# The trainer file
# ---------------------------------------------------------------------------

OUTCOMES = [True, False, False, True, True, False]


def _checkpointed(directory, episodes=4):
    """A GoalEnv run saved every two episodes; returns its trainer."""
    trainer = _trainer(GoalEnv(OUTCOMES), directory, num_episodes=episodes, save_frequency=2)
    trainer.train()
    return trainer


def test_the_trainer_file_keeps_the_outcomes_and_a_resume_continues_them(tmp_path):
    uninterrupted = _checkpointed(tmp_path / "full", episodes=6)
    _checkpointed(tmp_path / "stopped", episodes=4)
    with np.load(tmp_path / "stopped" / "trainer_episode_4.npz") as data:
        saved = data["episode_successes"]
        assert saved.dtype == np.bool_, "saved as a bool array, which loads without pickle"
        assert saved.tolist() == OUTCOMES[:4]

    env = GoalEnv(OUTCOMES)
    env.episode = 3  # the environment continues where the stopped run's left off
    resumed = _trainer(env, tmp_path / "stopped", num_episodes=6, save_frequency=2)
    resumed.resume(str(tmp_path / "stopped" / "agent_episode_4.pt"))
    assert resumed.episode_successes == OUTCOMES[:4], "the resume restores the outcomes"
    stats = resumed.train()
    assert resumed.episode_successes == uninterrupted.episode_successes
    assert stats["final_success_rate"] == pytest.approx(0.5)


def _rewrite(path, change):
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    change(arrays)
    np.savez(path, **arrays)


@pytest.mark.parametrize(
    "value, match",
    [
        (np.array([[True, False], [False, True]]), "one value per episode"),
        (np.array([1, 0, 0, 1]), "bool"),
        (np.array([True, False, False]), "3 episodes"),
    ],
    ids=["two-dimensional", "integers", "wrong-length"],
)
def test_a_malformed_outcome_history_is_refused(tmp_path, value, match):
    _checkpointed(tmp_path)
    _rewrite(
        tmp_path / "trainer_episode_4.npz", lambda a: a.__setitem__("episode_successes", value)
    )
    trainer = _trainer(GoalEnv(OUTCOMES), tmp_path, num_episodes=6)
    with pytest.raises(ValueError, match=f"episode_successes.*{match}|{match}.*episode_successes"):
        trainer.resume(str(tmp_path / "agent_episode_4.pt"))


def test_a_trainer_file_from_before_the_metric_resumes_with_unknown_outcomes(tmp_path):
    _checkpointed(tmp_path)
    _rewrite(tmp_path / "trainer_episode_4.npz", lambda a: a.pop("episode_successes"))
    trainer = _trainer(GoalEnv(OUTCOMES), tmp_path, num_episodes=6)
    trainer.resume(str(tmp_path / "agent_episode_4.pt"))
    stats = trainer.train()
    assert trainer.episode_successes[:4] == [None] * 4, "the earlier outcomes are unknown"
    assert all(isinstance(s, bool) for s in trainer.episode_successes[4:])
    assert "final_success_rate" not in stats, "the last-100 window holds unknown outcomes"

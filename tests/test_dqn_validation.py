"""DQN rejects invalid settings at construction, and warm-up reports no loss (CHANGELOG.md, 2026-10-19)."""

import numpy as np
import pytest

from rl_core.agents import DQNAgent
from tests.agent_internals import (
    DQN_ONLINE_WEIGHTS,
    assert_weights_unchanged,
    live_weights,
    weights,
)


def _agent(**config) -> DQNAgent:
    return DQNAgent(4, 2, config={"seed": 3, "hidden_dim": 8, **config})


# ---------------------------------------------------------------------------
# Settings rejected at construction, each naming its key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["target_update_frequency", "batch_size", "buffer_capacity", "min_buffer_size", "hidden_dim",
            "max_updates_per_call"],
)  # fmt: skip
@pytest.mark.parametrize("bad", [0, -2, 2.5, True, "8"])
def test_integer_settings_must_be_positive_integers(key, bad):
    """A float or a boolean is not silently truncated (2.5 used to become 2, True became 1)."""
    with pytest.raises(ValueError, match=key):
        _agent(**{key: bad})


@pytest.mark.parametrize("bad", ["false", 1, None])
def test_double_q_must_be_a_boolean(bad):
    with pytest.raises(ValueError, match="double_q must be true or false"):
        _agent(double_q=bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, True, float("nan"), "abc"])
def test_tau_must_lie_in_zero_one(bad):
    with pytest.raises(ValueError, match="tau"):
        _agent(tau=bad)


@pytest.mark.parametrize("min_buffer_size", [16, 31])
def test_min_buffer_size_below_batch_size_is_rejected(min_buffer_size):
    """It used to fail only at the first sample, mid-run."""
    with pytest.raises(ValueError, match="min_buffer_size"):
        _agent(batch_size=32, min_buffer_size=min_buffer_size)


def test_buffer_capacity_below_min_buffer_size_is_rejected():
    """The buffer could never warm up, so the agent silently never trained."""
    with pytest.raises(ValueError, match="buffer_capacity"):
        _agent(batch_size=16, min_buffer_size=64, buffer_capacity=32)


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, True, float("nan"), "1"])
def test_updates_per_step_must_be_positive(bad):
    """0 used to mean one update per call; leaving the key unset still does."""
    with pytest.raises(ValueError, match="updates_per_step"):
        _agent(updates_per_step=bad)


def test_valid_numpy_integers_are_accepted():
    agent = _agent(
        batch_size=np.int64(16), min_buffer_size=np.int64(16), target_update_frequency=np.int32(5)
    )
    assert (agent.batch_size, agent.target_update_frequency) == (16, 5)
    assert isinstance(agent.batch_size, int)


def test_the_committed_config_is_valid():
    from rl_cartpole.benchmark import CartPoleChallenge

    env, agent, _ = CartPoleChallenge().setup_run("dqn", seed=1, episodes=None)
    env.close()
    assert agent.min_buffer_size >= agent.batch_size


# ---------------------------------------------------------------------------
# tau = 1 copies the target every gradient step
# ---------------------------------------------------------------------------


def _batch(n: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "observations": rng.standard_normal((n, 4)),
        "actions": rng.integers(0, 2, n),
        "rewards": np.ones(n),
        "next_observations": rng.standard_normal((n, 4)),
        "dones": np.array([False] * (n - 1) + [True]),
    }


def test_tau_one_copies_the_online_network_after_every_gradient_step():
    agent = _agent(
        tau=1.0, target_update_frequency=1000, batch_size=4, min_buffer_size=4, updates_per_step=0.5
    )
    agent.update(_batch())
    arrays = live_weights(agent)
    for name in DQN_ONLINE_WEIGHTS:
        online, target = arrays[name], arrays[f"target_{name}"]
        np.testing.assert_array_equal(target, online)
        assert not np.shares_memory(target, online), f"target_{name} aliases {name}."


# ---------------------------------------------------------------------------
# Warm-up reports only epsilon
# ---------------------------------------------------------------------------


def test_warm_up_reports_epsilon_but_no_loss_or_q_value():
    agent = _agent(batch_size=8, min_buffer_size=50)
    before = weights(agent)
    metrics = agent.update(_batch())
    assert metrics == {"epsilon": agent.epsilon}
    assert_weights_unchanged(agent, before, "in warm-up")


def test_the_training_log_averages_loss_only_over_real_updates(tmp_path):
    """A logged window spent in warm-up carries epsilon only, and later windows average no zeros."""
    import json

    from rl_cartpole.environments import CartPoleEnv
    from rl_core.training import Trainer
    from rl_core.utils.logger import Logger

    env = CartPoleEnv(seed=0, max_episode_steps=20)
    agent = DQNAgent(4, 2, {"seed": 0, "hidden_dim": 8, "batch_size": 8, "min_buffer_size": 200})
    # Episodes last at most 20 steps, so the buffer warms within episodes 10 to 19: the second
    # window mixes warm-up calls with real updates.
    returned: list[dict[str, float]] = []
    real_update = agent.update

    def recording_update(batch):
        metrics = real_update(batch)
        returned.append(metrics)
        return metrics

    agent.update = recording_update  # type: ignore[method-assign]
    logger = Logger(name="test_dqn_warmup_log", log_dir=str(tmp_path / "logs"))
    config = {"num_episodes": 20, "max_steps_per_episode": 20, "eval_frequency": 100, "save_frequency": 100,
              "checkpoint_dir": str(tmp_path / "ckpt")}  # fmt: skip
    try:
        Trainer(env=env, agent=agent, config=config, logger=logger).train()
        with open(logger.metrics_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    finally:
        logger.close()
        env.close()
    first = next(row for row in rows if "episode" in row)
    assert set(first["update"]) == {"epsilon"}, (
        "the first 10 episodes (at most 200 steps) are all warm-up"
    )
    second_window = returned[10:20]
    real_losses = [m["loss"] for m in second_window if "loss" in m]
    assert 0 < len(real_losses) < len(second_window), "the second window should mix both kinds"
    second = [row for row in rows if "episode" in row][1]
    assert second["update"]["loss"] == pytest.approx(np.mean(real_losses))


# ---------------------------------------------------------------------------
# A batch the replay buffer does not store trains nothing
# ---------------------------------------------------------------------------


def _without(batch: dict[str, np.ndarray], *keys: str) -> dict[str, np.ndarray]:
    return {k: v for k, v in batch.items() if k not in keys}


@pytest.mark.parametrize(
    "unstored",
    [
        pytest.param(lambda b: _without(b, "dones"), id="no-terminal-flags"),
        pytest.param(lambda b: _without(b, "next_observations"), id="no-next-observations"),
        pytest.param(lambda b: {k: v[:0] for k, v in b.items()}, id="empty"),
    ],
)
def test_a_batch_that_stores_no_transition_runs_no_gradient_step(unstored):
    """After warm-up, a batch the buffer cannot store used to run a gradient step on old samples."""
    agent = _agent(batch_size=4, min_buffer_size=4, updates_per_step=1.0, epsilon_decay=0.9)
    agent.update(_batch())  # warms the buffer
    before, epsilon = weights(agent), agent.epsilon

    metrics = agent.update(unstored(_batch()))

    assert metrics == {"epsilon": epsilon}, "nothing was trained, so no loss or Q-value to report"
    assert agent.epsilon == epsilon, "epsilon decays once per update that trained, not per call"
    assert_weights_unchanged(agent, before, "after a batch that stored no transition")

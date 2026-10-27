"""DQN with prioritised replay (``prioritized_replay: true``): the importance-weighted TD loss, the
priority write-back after each gradient step, the beta schedule, the config and checkpoints.

The loss is mean(w * ½(Q(s, a) - y)²) with the weights w and targets y held constant; with every
weight one it is the uniform loss exactly. Beta rises linearly from ``priority_beta_start`` to 1
over ``priority_beta_steps`` transitions (``anneal_steps`` when unset), on the counter the
learning-rate decay reads.
"""

import numpy as np
import pytest

from rl_core.agents import DQNAgent
from rl_core.agents.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from tests.agent_internals import gradient_check_parameters, loss_and_grads, set_weights, weights
from tests.helpers import assert_gradients_match, assert_same_checkpoint

EPS = 1e-5
N_STEPS, OBS_DIM, HIDDEN_DIM, ACTION_DIM = 6, 4, 6, 3
PRIORITIZED = {"prioritized_replay": True, "priority_beta_steps": 50}


def _agent(**config) -> DQNAgent:
    # Seed 31 puts every hidden unit at least 0.2 from the ReLU kink for the batch below
    # (tests/test_dqn_maths.py).
    settings = {"hidden_dim": HIDDEN_DIM, "seed": 31}
    return DQNAgent(observation_dim=OBS_DIM, action_dim=ACTION_DIM, config={**settings, **config})


def _batch(seed: int = 3, n: int = N_STEPS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    terminated = np.zeros(n, dtype=bool)
    terminated[n // 2] = True  # an episode end mid-batch
    return {
        "observations": rng.standard_normal((n, OBS_DIM)),
        "actions": rng.integers(0, ACTION_DIM, size=n),
        "rewards": rng.normal(size=n),
        "next_observations": rng.standard_normal((n, OBS_DIM)),
        "terminated": terminated,
    }


# ---------------------------------------------------------------------------
# The weighted loss
# ---------------------------------------------------------------------------


def test_weighted_loss_matches_hand_computation():
    agent = _agent()
    current = weights(agent)
    set_weights(  # Q(s, .) = [1, 3, -5] for every state
        agent,
        W1=np.zeros_like(current["W1"]),
        b1=np.zeros_like(current["b1"]),
        b2=np.array([1.0, 3.0, -5.0]),
    )
    observations = _batch()["observations"][:3]
    actions, targets = np.array([0, 1, 1]), np.array([2.0, 2.5, 4.0])

    components, _ = loss_and_grads(
        agent, observations, actions, targets, weights=np.array([1.0, 0.5, 0.25])
    )

    # TD errors [-1, 0.5, -1]; w * ½ δ² = [0.5, 0.0625, 0.125]; mean = 0.6875 / 3
    assert components["loss"] == pytest.approx(0.6875 / 3.0, rel=1e-12)


def test_weighted_gradients_match_finite_differences():
    agent = _agent()
    set_weights(agent, b1=np.linspace(-0.05, 0.05, HIDDEN_DIM), b2=np.array([0.3, -0.2, 0.1]))
    batch = _batch()
    observations, actions = batch["observations"], batch["actions"]
    targets = np.array([1.5, -0.5, 2.0, 0.0, 1.0, -1.0])  # held constant
    sample_weights = np.array([1.0, 0.3, 0.7, 0.1, 0.5, 0.9])  # held constant, all different
    pre_activation = observations @ weights(agent)["W1"] + weights(agent)["b1"]
    assert np.abs(pre_activation).min() > 100 * EPS * np.abs(observations).max()

    _, grads = loss_and_grads(agent, observations, actions, targets, weights=sample_weights)

    assert_gradients_match(
        lambda: loss_and_grads(agent, observations, actions, targets, weights=sample_weights)[0][
            "loss"
        ],
        gradient_check_parameters(agent)[0],
        grads,
    )


def test_a_zero_weight_removes_its_sample_from_the_gradient():
    """Ignoring the weights in both the loss and its gradient would pass a gradient check; this
    cannot pass that way. A zero-weighted row is the same as dropping it, rescaled by B / (B - 1)
    because the mean still divides by the full batch."""
    agent = _agent()
    batch = _batch()
    observations, actions = batch["observations"], batch["actions"]
    targets = np.array([1.5, -0.5, 2.0, 0.0, 1.0, -1.0])
    sample_weights = np.ones(N_STEPS)
    sample_weights[1] = 0.0
    keep = np.arange(N_STEPS) != 1

    _, weighted = loss_and_grads(agent, observations, actions, targets, weights=sample_weights)
    _, dropped = loss_and_grads(agent, observations[keep], actions[keep], targets[keep])

    for name, grad in dropped.items():
        np.testing.assert_allclose(
            weighted[name] * N_STEPS / (N_STEPS - 1), grad, rtol=1e-12, err_msg=name
        )


def test_weights_of_one_give_exactly_the_uniform_loss_and_gradients():
    agent = _agent()
    batch = _batch()
    args = (batch["observations"], batch["actions"], np.linspace(-1.0, 1.0, N_STEPS))
    uniform_components, uniform = loss_and_grads(agent, *args)
    ones_components, ones = loss_and_grads(agent, *args, weights=np.ones(N_STEPS))
    assert ones_components == uniform_components
    for name in uniform:
        np.testing.assert_array_equal(ones[name], uniform[name], err_msg=name)


# ---------------------------------------------------------------------------
# update(): write-back, beta and metrics
# ---------------------------------------------------------------------------


def _spy_on_buffer(monkeypatch, buffer: PrioritizedReplayBuffer) -> list[tuple]:
    """Record each sample and priority write-back, in order, with their arguments."""
    calls: list[tuple] = []
    sample, write = buffer.sample, buffer.update_priorities

    def spy_sample(batch_size, *, beta=1.0):
        result = sample(batch_size, beta=beta)
        calls.append(("sample", beta, result))
        return result

    def spy_write(indices, td_errors):
        calls.append(("write", np.copy(indices), np.copy(td_errors)))
        write(indices, td_errors)

    monkeypatch.setattr(buffer, "sample", spy_sample)
    monkeypatch.setattr(buffer, "update_priorities", spy_write)
    return calls


def test_each_gradient_step_writes_back_its_own_td_errors_before_the_next_sample(monkeypatch):
    agent = _agent(
        **PRIORITIZED, batch_size=4, min_buffer_size=4, updates_per_step=0.5, priority_alpha=0.6
    )
    assert isinstance(agent.replay_buffer, PrioritizedReplayBuffer)
    calls = _spy_on_buffer(monkeypatch, agent.replay_buffer)
    batch = _batch()
    before = weights(agent)

    agent.update(batch)

    assert [c[0] for c in calls] == ["sample", "write"] * 3, "6 transitions * 0.5 = 3 steps"
    first_sample, first_write = calls[0][2], calls[1]
    np.testing.assert_array_equal(first_write[1], first_sample["indices"])
    # The first step's TD errors, from the weights before any step and the initial target
    w = before
    q = np.maximum(0.0, first_sample["observations"] @ w["W1"] + w["b1"]) @ w["W2"] + w["b2"]
    next_q = (
        np.maximum(0.0, first_sample["next_observations"] @ w["target_W1"] + w["target_b1"])
        @ w["target_W2"]
        + w["target_b2"]
    )
    targets = first_sample["rewards"] + (1.0 - first_sample["dones"]) * agent.gamma * next_q.max(1)
    expected = q[np.arange(4), first_sample["actions"]] - targets
    np.testing.assert_allclose(first_write[2], expected, rtol=1e-10)
    last_indices, last_errors = calls[-1][1], calls[-1][2]
    np.testing.assert_allclose(
        agent.replay_buffer.priorities[last_indices],
        (np.abs(last_errors) + 1e-6) ** 0.6,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "config,expected_betas",
    [
        # anneal_steps 50, beta from 0.4: counter before each call 0, 6, 12 -> fraction 1, .88, .76
        ({"anneal_steps": 50}, [0.4, 0.4 + 0.6 * 0.12, 0.4 + 0.6 * 0.24]),
        # priority_beta_steps 12 overrides anneal_steps 50: fraction 1, 0.5, 0 -> 0.4, 0.7, 1.0
        ({"anneal_steps": 50, "priority_beta_steps": 12}, [0.4, 0.7, 1.0]),
    ],
)
def test_beta_rises_on_the_transitions_counter_before_the_batch_is_stored(
    monkeypatch, config, expected_betas
):
    agent = _agent(
        prioritized_replay=True, batch_size=4, min_buffer_size=4, priority_beta_start=0.4, **config
    )
    calls = _spy_on_buffer(monkeypatch, agent.replay_buffer)
    reported = []
    for seed in range(3):
        reported.append(agent.update(_batch(seed))["priority_beta"])
    sampled = [c[1] for c in calls if c[0] == "sample"]
    np.testing.assert_allclose(reported, expected_betas, rtol=1e-12)
    np.testing.assert_allclose(sampled, expected_betas, rtol=1e-12)


def test_metrics_report_beta_and_the_mean_importance_weight():
    agent = _agent(**PRIORITIZED, batch_size=4, min_buffer_size=4)
    metrics = agent.update(_batch())
    assert {"loss", "epsilon", "mean_q", "priority_beta", "mean_is_weight"} <= set(metrics)
    assert 0.0 < metrics["mean_is_weight"] <= 1.0


def test_without_the_flag_the_buffer_is_uniform_and_no_priority_metric_is_reported():
    agent = _agent(batch_size=4, min_buffer_size=4)
    assert type(agent.replay_buffer) is ReplayBuffer
    assert not {"priority_beta", "mean_is_weight"} & set(agent.update(_batch()))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config,key",
    [
        ({"prioritized_replay": True}, "priority_beta_steps"),  # no beta schedule at all
        ({"priority_alpha": 0.6}, "priority_alpha"),  # set while the flag is off
        ({"priority_beta_steps": 10}, "priority_beta_steps"),  # set while the flag is off
        ({**PRIORITIZED, "priority_alpha": 1.5}, "priority_alpha"),
        ({**PRIORITIZED, "priority_epsilon": 0.0}, "priority_epsilon"),
        ({**PRIORITIZED, "priority_beta_start": 1.2}, "priority_beta_start"),
        ({**PRIORITIZED, "priority_beta_steps": 0}, "priority_beta_steps"),
    ],
)
def test_invalid_priority_settings_are_refused_naming_the_key(config, key):
    with pytest.raises(ValueError, match=key):
        _agent(**config)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_a_loaded_prioritized_agent_continues_exactly_as_the_saved_one(tmp_path):
    config = {**PRIORITIZED, "batch_size": 4, "min_buffer_size": 4, "optimizer": "adam"}
    agent = _agent(**config)
    for seed in range(3):
        agent.update(_batch(seed))
    path = str(tmp_path / "dqn.npz")
    agent.save(path)
    loaded = _agent(**{**config, "seed": 7})
    loaded.update(_batch(9))  # trained once, so nothing passes by being still at its initial value
    loaded.load(path)

    assert agent.update(_batch(5)) == loaded.update(_batch(5))
    assert_same_checkpoint(agent, loaded, tmp_path)


def test_a_uniform_agent_loads_a_prioritized_checkpoint_and_drops_the_priorities(tmp_path):
    prioritized = _agent(**PRIORITIZED, batch_size=4, min_buffer_size=4)
    prioritized.update(_batch())
    path = str(tmp_path / "dqn.npz")
    prioritized.save(path)
    uniform = _agent(batch_size=4, min_buffer_size=4)
    uniform.load(path)
    assert len(uniform.replay_buffer) == N_STEPS
    assert "loss" in uniform.update(_batch(1))

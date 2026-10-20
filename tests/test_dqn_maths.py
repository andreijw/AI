"""The hand-written maths of the DQN agent.

Completes the set started in tests/test_policy_gradient_maths.py and
tests/test_continuous_policy_gradient_maths.py:

* the analytic gradients agree with finite differences of the TD loss;
* the loss and the Bellman targets equal values worked out by hand;
* computing a loss changes nothing: weights, target network, epsilon, counter or replay buffer;
* the apply step clips each gradient element and descends;
* update() samples, builds targets from the target network, steps, and syncs the target network
  as a replay written out in the test says it should.

Some tests pin behaviour as it is today (epsilon decays once per update() call; a buffer that is
not warm returns zeros). They guard the refactor; they do not claim that behaviour is ideal.
"""

import copy

import numpy as np
import pytest

from rl_core.agents import DQNAgent
from tests.agent_internals import (
    DQN_ONLINE_WEIGHTS,
    apply_gradients,
    assert_same_weights,
    bellman_targets,
    counters,
    gradient_check_parameters,
    live_weights,
    loss_and_grads,
    replay_generator,
    set_counters,
    set_weights,
    weights,
)
from tests.helpers import assert_gradients_match

EPS = 1e-5  # perturbation used by tests.helpers.numerical_grad
N_STEPS, OBS_DIM, HIDDEN_DIM, ACTION_DIM = 6, 4, 6, 3


def _agent(**config) -> DQNAgent:
    # Seed 31 puts every hidden unit at least 0.2 from the ReLU kink for the batch below.
    settings = {"hidden_dim": HIDDEN_DIM, "seed": 31}
    return DQNAgent(observation_dim=OBS_DIM, action_dim=ACTION_DIM, config={**settings, **config})


def _batch() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    observations = rng.standard_normal((N_STEPS, OBS_DIM))
    return {
        "observations": observations,
        "actions": rng.integers(0, ACTION_DIM, size=N_STEPS),
        "rewards": np.array([1.0, 0.0, 2.0, -1.0, 0.5, 3.0]),
        "next_observations": rng.standard_normal((N_STEPS, OBS_DIM)),  # not the observations
        "dones": np.array([False, False, True, False, False, True]),
    }


def _state(agent: DQNAgent) -> dict[str, object]:
    """Everything an update can change, copied."""
    snapshot: dict[str, object] = dict(weights(agent))
    snapshot["epsilon"] = agent.epsilon
    snapshot["counter"] = counters(agent)["update_counter"]
    snapshot["buffer_size"] = len(agent.replay_buffer)
    snapshot["buffer_generator"] = replay_generator(agent).bit_generator.state
    return snapshot


def _assert_state_unchanged(agent: DQNAgent, before: dict[str, object]) -> None:
    after = _state(agent)
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(after[name], value, err_msg=f"{name} was modified.")
        else:
            assert after[name] == value, f"{name} was modified."


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------


def test_gradients_match_finite_differences():
    agent = _agent()
    set_weights(agent, b1=np.linspace(-0.05, 0.05, HIDDEN_DIM), b2=np.array([0.3, -0.2, 0.1]))
    batch = _batch()
    observations, actions = batch["observations"], batch["actions"]
    targets = np.array([1.5, -0.5, 2.0, 0.0, 1.0, -1.0])  # held constant
    current = weights(agent)
    pre_activation = observations @ current["W1"] + current["b1"]
    margin = 100 * EPS * max(1.0, float(np.abs(observations).max()))
    assert np.abs(pre_activation).min() > margin, "A hidden unit sits on the ReLU kink."
    assert (pre_activation > 0).any() and not (pre_activation > 0).all()

    _, grads = loss_and_grads(agent, observations, actions, targets)

    assert_gradients_match(
        lambda: loss_and_grads(agent, observations, actions, targets)[0]["loss"],
        gradient_check_parameters(agent)[0],
        grads,
    )


# ---------------------------------------------------------------------------
# Values worked out by hand
# ---------------------------------------------------------------------------


def _zero_trunk(agent: DQNAgent, online_bias, target_bias) -> DQNAgent:
    """Make Q(s, .) the output bias for every state, separately for the two networks."""
    current = weights(agent)
    set_weights(
        agent,
        W1=np.zeros_like(current["W1"]),
        b1=np.zeros_like(current["b1"]),
        b2=online_bias,
        target_W1=np.zeros_like(current["target_W1"]),
        target_b1=np.zeros_like(current["target_b1"]),
        target_b2=target_bias,
    )
    return agent


def test_loss_and_mean_q_match_hand_computation():
    agent = _zero_trunk(_agent(), online_bias=[1.0, 3.0, -5.0], target_bias=[0.0, 0.0, 0.0])
    observations = _batch()["observations"][:3]
    actions = np.array([0, 1, 1])
    targets = np.array([2.0, 2.5, 4.0])

    components, _ = loss_and_grads(agent, observations, actions, targets)

    # Q(s, a) = [1, 3, 3]; TD errors = Q - y = [-1, 0.5, -1]
    # loss = mean(0.5 * [1, 0.25, 1]) = 0.375;  mean_q = (1 + 3 + 3) / 3
    assert components["loss"] == pytest.approx(0.375, rel=1e-12)
    assert components["mean_q"] == pytest.approx(7.0 / 3.0, rel=1e-12)


def test_bellman_targets_match_hand_computation_and_use_the_target_network():
    agent = _zero_trunk(
        _agent(gamma=0.5), online_bias=[10.0, 20.0, 30.0], target_bias=[1.0, 3.0, 2.0]
    )
    next_observations = _batch()["next_observations"][:3]
    rewards = np.array([1.0, 0.0, 2.0])
    dones = np.array([False, False, True])

    targets = bellman_targets(agent, rewards, next_observations, dones)

    # max over the TARGET network's Q-values is 3 (the online network would give 30):
    #   y = r + (1 - done) * 0.5 * 3 = [1 + 1.5, 0 + 1.5, 2 + 0]
    np.testing.assert_allclose(targets, [2.5, 1.5, 2.0], rtol=1e-12)


# ---------------------------------------------------------------------------
# Computing a loss changes nothing; the apply step clips and descends
# ---------------------------------------------------------------------------


def test_loss_and_grads_changes_no_state():
    agent = _agent()
    batch = _batch()
    agent.replay_buffer.add_batch(
        batch["observations"],
        batch["actions"],
        batch["rewards"],
        batch["next_observations"],
        batch["dones"],
    )
    targets = np.array([1.5, -0.5, 2.0, 0.0, 1.0, -1.0])
    before = _state(agent)

    first = loss_and_grads(agent, batch["observations"], batch["actions"], targets)
    second = loss_and_grads(agent, batch["observations"], batch["actions"], targets)

    _assert_state_unchanged(agent, before)
    assert first[0] == second[0]


def test_apply_gradients_clips_each_element_then_descends():
    agent = _agent(learning_rate=0.1, grad_clip_value=0.5)
    rng = np.random.default_rng(9)
    before = weights(agent)
    grads = {name: rng.standard_normal(before[name].shape) for name in DQN_ONLINE_WEIGHTS}
    grads["W1"] *= 5.0  # most of these lie outside the clip range; the others mostly inside
    kept = {name: gradient.copy() for name, gradient in grads.items()}
    outside = np.concatenate([np.abs(g).ravel() > 0.5 for g in grads.values()])
    assert outside.any() and not outside.all(), "Need gradients on both sides of the clip."

    apply_gradients(agent, grads)

    after = weights(agent)
    for name in DQN_ONLINE_WEIGHTS:
        expected = before[name] - 0.1 * np.clip(kept[name], -0.5, 0.5)
        np.testing.assert_allclose(after[name], expected, rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(
            grads[name], kept[name], err_msg="The caller's gradients changed."
        )
        target = f"target_{name}"
        np.testing.assert_array_equal(after[target], before[target], err_msg=f"{target} moved.")


# ---------------------------------------------------------------------------
# update(): sample, build targets, step, sync
# ---------------------------------------------------------------------------


def _replay(reference: DQNAgent, batch: dict[str, np.ndarray], updates: int) -> dict[str, float]:
    """Do what update() should do, written out with plain arithmetic on a copy of the agent.

    The reference is a deep copy made before the update, so its replay buffer draws the same
    samples; it must add the batch first and draw exactly one sample per gradient step.
    """
    reference.replay_buffer.add_batch(
        batch["observations"],
        batch["actions"],
        batch["rewards"],
        batch["next_observations"],
        batch["dones"],
    )
    losses, mean_qs = [], []
    any_clipped, any_unclipped = False, False
    counter = counters(reference)["update_counter"]
    for _ in range(updates):
        sample = reference.replay_buffer.sample(reference.batch_size)
        w = weights(reference)
        hidden = np.maximum(0.0, sample["next_observations"] @ w["target_W1"] + w["target_b1"])
        next_q = hidden @ w["target_W2"] + w["target_b2"]
        not_done = 1.0 - sample["dones"].astype(np.float64)
        targets = sample["rewards"] + not_done * reference.gamma * next_q.max(axis=-1)

        components, grads = loss_and_grads(
            reference, sample["observations"], sample["actions"], targets
        )
        losses.append(components["loss"])
        mean_qs.append(components["mean_q"])
        limit = reference.grad_clip_value
        for name in DQN_ONLINE_WEIGHTS:
            any_clipped |= bool((np.abs(grads[name]) > limit).any())
            any_unclipped |= bool((np.abs(grads[name]) <= limit).any())
            w[name] = w[name] - reference.learning_rate * np.clip(grads[name], -limit, limit)

        counter += 1
        for name in DQN_ONLINE_WEIGHTS:
            target = f"target_{name}"
            if reference.tau is not None:
                w[target] = reference.tau * w[name] + (1.0 - reference.tau) * w[target]
            elif counter % reference.target_update_frequency == 0:
                w[target] = w[name].copy()
        set_weights(reference, **w)
    set_counters(reference, update_counter=counter)
    return {
        "loss": float(np.mean(losses)),
        "mean_q": float(np.mean(mean_qs)),
        "any_clipped": any_clipped,
        "any_unclipped": any_unclipped,
    }


# min_buffer_size differs from batch_size so that sampling by the wrong one of the two is visible.
BASE = {"batch_size": 4, "min_buffer_size": 5, "learning_rate": 0.05, "gamma": 0.9}


@pytest.mark.parametrize(
    "config,expected_updates",
    [
        # 6 transitions * 0.5 = 3 updates; hard sync after the 2nd, so the run ends between syncs.
        # The tight clip makes some gradient elements clipped and leaves others alone.
        ({"updates_per_step": 0.5, "target_update_frequency": 2, "grad_clip_value": 0.05}, 3),
        # 6 transitions * 1.0 = 6 updates, capped at 2; soft target update (tau is not 0.5, where
        # swapping tau and 1 - tau would change nothing).
        ({"updates_per_step": 1.0, "max_updates_per_call": 2, "tau": 0.3}, 2),
        # updates_per_step unset: exactly one update per call.
        ({"target_update_frequency": 5}, 1),
        # 6 transitions * 0.1 rounds down to 0, which is raised to 1.
        ({"updates_per_step": 0.1, "target_update_frequency": 5}, 1),
        # 6 transitions * 0.25 = 1.5 rounds DOWN to 1 (rounding to nearest or up would give 2).
        ({"updates_per_step": 0.25, "target_update_frequency": 5}, 1),
        # tau of exactly 1.0 copies the online network after every one of the 3 steps; the
        # frequency (5) is unused, so a fallback to hard sync would leave the target unmoved.
        ({"updates_per_step": 0.5, "tau": 1.0, "target_update_frequency": 5}, 3),
    ],
)
def test_update_replays_as_sample_target_step_sync(config, expected_updates):
    agent = _agent(**BASE, **config)
    set_weights(
        agent,
        b1=np.linspace(-0.05, 0.05, HIDDEN_DIM),
        b2=np.array([0.3, -0.2, 0.1]),
        target_b2=np.array([-0.1, 0.4, 0.2]),  # the target network differs from the online one
    )
    reference = copy.deepcopy(agent)
    batch = _batch()

    metrics = agent.update(batch)

    expected = _replay(reference, batch, expected_updates)
    assert_same_weights(agent, reference)
    assert counters(agent)["update_counter"] == expected_updates
    assert metrics["loss"] == pytest.approx(expected["loss"], rel=1e-12)
    assert metrics["mean_q"] == pytest.approx(expected["mean_q"], rel=1e-12)
    assert metrics["epsilon"] == pytest.approx(agent.epsilon_start * agent.epsilon_decay, rel=1e-12)
    assert agent.epsilon == metrics["epsilon"]
    if "grad_clip_value" in config:
        assert expected["any_clipped"] and expected["any_unclipped"], "The clip must bite partly."


def test_hard_sync_copies_the_online_network_and_then_leaves_it_alone():
    """Right after a sync the target equals the online network but shares no memory with it."""
    config = {"updates_per_step": 0.5, "target_update_frequency": 3}
    agent = _agent(**BASE, **config)
    set_weights(agent, target_b2=np.array([-0.1, 0.4, 0.2]))

    agent.update(_batch())  # 3 updates: the sync happens on the last one

    assert counters(agent)["update_counter"] == 3
    arrays = live_weights(agent)
    for name in DQN_ONLINE_WEIGHTS:
        online, target = arrays[name], arrays[f"target_{name}"]
        np.testing.assert_array_equal(target, online, err_msg=f"{name} was not synced.")
        assert not np.shares_memory(target, online), f"Target {name} aliases the online array."


def test_buffer_that_is_not_warm_reports_epsilon_and_changes_nothing_but_the_buffer():
    """The batch is stored, nothing is learned, epsilon does not decay, and no loss is reported."""
    agent = _agent(batch_size=4, min_buffer_size=10)
    before = _state(agent)

    metrics = agent.update(_batch())

    assert metrics == {"epsilon": agent.epsilon_start}
    assert len(agent.replay_buffer) == N_STEPS
    after = _state(agent)
    for name, value in before.items():
        if name == "buffer_size":
            continue
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(after[name], value, err_msg=f"{name} was modified.")
        else:
            assert after[name] == value, f"{name} was modified."


def test_epsilon_decays_once_per_update_call_down_to_its_floor():
    """Current behaviour: one decay per update() call, however many gradient steps it takes."""
    agent = _agent(**BASE, updates_per_step=0.5, epsilon_decay=0.5, epsilon_end=0.2)

    seen = [agent.update(_batch())["epsilon"] for _ in range(3)]

    assert counters(agent)["update_counter"] == 9, "Three calls of three gradient steps each."
    assert seen == pytest.approx([0.5, 0.25, 0.2], rel=1e-12)

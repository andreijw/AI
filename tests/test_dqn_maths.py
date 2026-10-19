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
from tests.helpers import assert_gradients_match

EPS = 1e-5  # perturbation used by tests.helpers.numerical_grad
N_STEPS, OBS_DIM, HIDDEN_DIM, ACTION_DIM = 6, 4, 6, 3
ONLINE = {"W1": "_W1", "b1": "_b1", "W2": "_W2", "b2": "_b2"}
TARGET = {"W1": "_target_W1", "b1": "_target_b1", "W2": "_target_W2", "b2": "_target_b2"}


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
    snapshot: dict[str, object] = {
        attribute: np.array(getattr(agent, attribute))
        for attribute in [*ONLINE.values(), *TARGET.values()]
    }
    snapshot["epsilon"] = agent.epsilon
    snapshot["counter"] = agent._update_counter
    snapshot["buffer_size"] = len(agent.replay_buffer)
    snapshot["buffer_generator"] = copy.deepcopy(agent.replay_buffer._rng.bit_generator.state)
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
    agent._b1 = np.linspace(-0.05, 0.05, HIDDEN_DIM)
    agent._b2 = np.array([0.3, -0.2, 0.1])
    batch = _batch()
    observations, actions = batch["observations"], batch["actions"]
    targets = np.array([1.5, -0.5, 2.0, 0.0, 1.0, -1.0])  # held constant
    pre_activation = observations @ agent._W1 + agent._b1
    margin = 100 * EPS * max(1.0, float(np.abs(observations).max()))
    assert np.abs(pre_activation).min() > margin, "A hidden unit sits on the ReLU kink."
    assert (pre_activation > 0).any() and not (pre_activation > 0).all()

    _, grads = agent._loss_and_grads(observations, actions, targets)

    assert_gradients_match(
        lambda: agent._loss_and_grads(observations, actions, targets)[0]["loss"],
        {name: getattr(agent, attribute) for name, attribute in ONLINE.items()},
        grads,
    )


# ---------------------------------------------------------------------------
# Values worked out by hand
# ---------------------------------------------------------------------------


def _zero_trunk(agent: DQNAgent, online_bias, target_bias) -> DQNAgent:
    """Make Q(s, .) the output bias for every state, separately for the two networks."""
    agent._W1 = np.zeros_like(agent._W1)
    agent._b1 = np.zeros_like(agent._b1)
    agent._b2 = np.array(online_bias, dtype=np.float64)
    agent._target_W1 = np.zeros_like(agent._target_W1)
    agent._target_b1 = np.zeros_like(agent._target_b1)
    agent._target_b2 = np.array(target_bias, dtype=np.float64)
    return agent


def test_loss_and_mean_q_match_hand_computation():
    agent = _zero_trunk(_agent(), online_bias=[1.0, 3.0, -5.0], target_bias=[0.0, 0.0, 0.0])
    observations = _batch()["observations"][:3]
    actions = np.array([0, 1, 1])
    targets = np.array([2.0, 2.5, 4.0])

    components, _ = agent._loss_and_grads(observations, actions, targets)

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

    targets = agent._compute_bellman_targets(rewards, next_observations, dones)

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

    first = agent._loss_and_grads(batch["observations"], batch["actions"], targets)
    second = agent._loss_and_grads(batch["observations"], batch["actions"], targets)

    _assert_state_unchanged(agent, before)
    assert first[0] == second[0]


def test_apply_gradients_clips_each_element_then_descends():
    agent = _agent(learning_rate=0.1, grad_clip_value=0.5)
    rng = np.random.default_rng(9)
    grads = {name: rng.standard_normal(getattr(agent, a).shape) for name, a in ONLINE.items()}
    grads["W1"] *= 5.0  # most of these lie outside the clip range; the others mostly inside
    kept = {name: gradient.copy() for name, gradient in grads.items()}
    before = {name: getattr(agent, attribute).copy() for name, attribute in ONLINE.items()}
    targets_before = {name: getattr(agent, attribute).copy() for name, attribute in TARGET.items()}
    outside = np.concatenate([np.abs(g).ravel() > 0.5 for g in grads.values()])
    assert outside.any() and not outside.all(), "Need gradients on both sides of the clip."

    agent._apply_gradients(grads)

    for name, attribute in ONLINE.items():
        expected = before[name] - 0.1 * np.clip(kept[name], -0.5, 0.5)
        np.testing.assert_allclose(getattr(agent, attribute), expected, rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(
            grads[name], kept[name], err_msg="The caller's gradients changed."
        )
    for name, attribute in TARGET.items():
        np.testing.assert_array_equal(getattr(agent, attribute), targets_before[name])


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
    counter = reference._update_counter
    for _ in range(updates):
        sample = reference.replay_buffer.sample(reference.batch_size)
        hidden = np.maximum(
            0.0, sample["next_observations"] @ reference._target_W1 + reference._target_b1
        )
        next_q = hidden @ reference._target_W2 + reference._target_b2
        not_done = 1.0 - sample["dones"].astype(np.float64)
        targets = sample["rewards"] + not_done * reference.gamma * next_q.max(axis=-1)

        components, grads = reference._loss_and_grads(
            sample["observations"], sample["actions"], targets
        )
        losses.append(components["loss"])
        mean_qs.append(components["mean_q"])
        limit = reference.grad_clip_value
        for name, attribute in ONLINE.items():
            any_clipped |= bool((np.abs(grads[name]) > limit).any())
            any_unclipped |= bool((np.abs(grads[name]) <= limit).any())
            stepped = getattr(reference, attribute) - reference.learning_rate * np.clip(
                grads[name], -limit, limit
            )
            setattr(reference, attribute, stepped)

        counter += 1
        for name, attribute in TARGET.items():
            online = getattr(reference, ONLINE[name])
            if reference.tau is not None:
                blended = reference.tau * online + (1.0 - reference.tau) * getattr(
                    reference, attribute
                )
                setattr(reference, attribute, blended)
            elif counter % reference.target_update_frequency == 0:
                setattr(reference, attribute, online.copy())
    reference._update_counter = counter
    return {
        "loss": float(np.mean(losses)),
        "mean_q": float(np.mean(mean_qs)),
        "any_clipped": any_clipped,
        "any_unclipped": any_unclipped,
    }


def _assert_same_networks(agent: DQNAgent, reference: DQNAgent) -> None:
    for attribute in [*ONLINE.values(), *TARGET.values()]:
        np.testing.assert_allclose(
            getattr(agent, attribute),
            getattr(reference, attribute),
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"{attribute} differs.",
        )


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
    agent._b1 = np.linspace(-0.05, 0.05, HIDDEN_DIM)
    agent._b2 = np.array([0.3, -0.2, 0.1])
    agent._target_b2 = np.array([-0.1, 0.4, 0.2])  # the target network differs from the online one
    reference = copy.deepcopy(agent)
    batch = _batch()

    metrics = agent.update(batch)

    expected = _replay(reference, batch, expected_updates)
    _assert_same_networks(agent, reference)
    assert agent._update_counter == expected_updates
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
    agent._target_b2 = np.array([-0.1, 0.4, 0.2])

    agent.update(_batch())  # 3 updates: the sync happens on the last one

    assert agent._update_counter == 3
    for name in ONLINE:
        online, target = getattr(agent, ONLINE[name]), getattr(agent, TARGET[name])
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

    assert agent._update_counter == 9, "Three calls of three gradient steps each."
    assert seen == pytest.approx([0.5, 0.25, 0.2], rel=1e-12)

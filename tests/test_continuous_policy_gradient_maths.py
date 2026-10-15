"""The hand-written maths of the continuous (Gaussian-policy) agents.

Mirrors tests/test_policy_gradient_maths.py for ContinuousActorCriticAgent and
ContinuousPPOAgent:

* analytic gradients agree with finite differences of the loss, parameter by parameter;
* losses, and PPO's generalised advantage estimates, equal values worked out by hand;
* computing a loss changes neither the weights nor the optimiser state;
* the Adam step and update() do with the gradients what they should.

Unlike the discrete agents, these average over the batch inside the loss and its gradients, and
the step is Adam with global gradient-norm clipping.
"""

import copy
import math

import numpy as np
import pytest

from rl_core.agents.continuous_actor_critic import ContinuousActorCriticAgent
from rl_core.agents.continuous_ppo import ContinuousPPOAgent
from tests.helpers import ReferenceAdam, assert_gradients_match

EPS = 1e-5  # perturbation used by tests.helpers.numerical_grad
N_STEPS, OBS_DIM, HIDDEN_DIM, ACTION_DIM = 6, 3, 6, 2
PARAMETER_ATTRIBUTES = {
    "W1": "_W1",
    "b1": "_b1",
    "W_mu": "_W_mu",
    "b_mu": "_b_mu",
    "log_std": "_log_std",
    "W_v": "_W_v",
    "b_v": "_b_v",
}
LOG_2PI = math.log(2.0 * math.pi)

# ---------------------------------------------------------------------------
# Fixtures for the gradient checks
# ---------------------------------------------------------------------------


def _batch(action_dim: int = ACTION_DIM, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = rng.standard_normal((N_STEPS, OBS_DIM))
    actions = rng.uniform(-1.5, 1.5, size=(N_STEPS, action_dim))
    return states, actions


def _agent(agent_cls, action_dim: int = ACTION_DIM, **config):
    # value_coef and entropy_coef are set away from 1 and from their tiny defaults so that
    # dropping either from a gradient changes it by far more than the test's tolerance.
    # log_std stays at its default of -0.5: far below about -5 the 1e-8 the agents add to the
    # variance makes their log_std gradient differ measurably from the exact derivative.
    # Seed 24 puts every hidden unit at least 0.1 from the ReLU kink for this batch.
    settings = {"hidden_dim": HIDDEN_DIM, "seed": 24, "value_coef": 0.5, "entropy_coef": 0.3}
    return agent_cls(observation_dim=OBS_DIM, action_dim=action_dim, config={**settings, **config})


def _assert_away_from_kinks(agent, states: np.ndarray) -> None:
    pre_activation = states @ agent._W1 + agent._b1
    margin = 100 * EPS * max(1.0, float(np.abs(states).max()))
    assert np.abs(pre_activation).min() > margin, "A hidden unit sits on the ReLU kink."
    active = pre_activation > 0
    assert active.any() and not active.all(), "Need both active and inactive hidden units."
    assert np.all((agent._log_std > -2.0) & (agent._log_std < 1.0)), "log_std outside safe range."


def _parameters(agent) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Live parameter arrays for the gradient check, with the float value bias boxed."""
    boxed_value_bias = np.array([agent._b_v], dtype=np.float64)
    parameters = {
        name: getattr(agent, attribute)
        for name, attribute in PARAMETER_ATTRIBUTES.items()
        if name != "b_v"
    }
    parameters["b_v"] = boxed_value_bias
    return parameters, boxed_value_bias


def _boxed(grads: dict) -> dict[str, np.ndarray]:
    return {
        name: np.atleast_1d(np.asarray(value, dtype=np.float64)) for name, value in grads.items()
    }


def _state(agent) -> dict[str, np.ndarray]:
    """Every weight and every piece of optimiser state, copied."""
    snapshot = {
        name: np.array(getattr(agent, attribute), dtype=np.float64)
        for name, attribute in PARAMETER_ATTRIBUTES.items()
    }
    for name in PARAMETER_ATTRIBUTES:
        snapshot[f"adam_m:{name}"] = np.array(agent._optimizer.m[name], dtype=np.float64)
        snapshot[f"adam_v:{name}"] = np.array(agent._optimizer.v[name], dtype=np.float64)
    snapshot["adam_t"] = np.array(agent._optimizer.t)
    return snapshot


def _assert_state_unchanged(agent, before: dict[str, np.ndarray]) -> None:
    after = _state(agent)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name], err_msg=f"{name} was modified.")


def _normalised(values: np.ndarray) -> np.ndarray:
    """Advantage normalisation as the continuous agents do it."""
    return (values - np.mean(values)) / (np.std(values) + 1e-8)


FIXED_RETURNS = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5])
FIXED_ADVANTAGES = np.array([0.8, -1.1, 0.5, 1.3, -0.6, 0.2])


def _ppo_old_log_probs(agent, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Old log-probabilities that put the ratios on both sides of the clip range."""
    mu, log_std, _, _ = agent._forward(states)
    current = agent._log_prob(actions, mu, log_std)
    # ratio = exp(current - old) = exp(-offset): 0.61, 1.65, 0.95, 1.05, 0.74, 1.35
    return current + np.array([0.5, -0.5, 0.05, -0.05, 0.3, -0.3])


# ---------------------------------------------------------------------------
# Gradient checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action_dim", [1, 2])
def test_actor_critic_gradients_match_finite_differences(action_dim):
    agent = _agent(ContinuousActorCriticAgent, action_dim)
    agent._b_v = 0.2
    states, actions = _batch(action_dim)
    _assert_away_from_kinks(agent, states)
    parameters, boxed_value_bias = _parameters(agent)
    inputs = (states, actions, FIXED_RETURNS, FIXED_ADVANTAGES)

    def loss() -> float:
        agent._b_v = float(boxed_value_bias[0])
        return agent._loss_and_grads(*inputs)[0]["total"]

    _, grads = agent._loss_and_grads(*inputs)

    assert_gradients_match(loss, parameters, _boxed(grads))


@pytest.mark.parametrize("action_dim", [1, 2])
def test_ppo_gradients_match_finite_differences(action_dim):
    agent = _agent(ContinuousPPOAgent, action_dim)
    agent._b_v = 0.2
    states, actions = _batch(action_dim)
    _assert_away_from_kinks(agent, states)
    old_log_probs = _ppo_old_log_probs(agent, states, actions)
    parameters, boxed_value_bias = _parameters(agent)
    inputs = (states, actions, FIXED_RETURNS, FIXED_ADVANTAGES, old_log_probs)

    def loss() -> float:
        agent._b_v = float(boxed_value_bias[0])
        return agent._minibatch_loss_and_grads(*inputs)[0]["total"]

    components, grads = agent._minibatch_loss_and_grads(*inputs)

    # ratios 0.61 and 0.74 with a positive advantage, 1.65 and 1.35 with a negative one, are the
    # unclipped side of the minimum; the batch has both kinds of step.
    ratios = np.exp(-np.array([0.5, -0.5, 0.05, -0.05, 0.3, -0.3]))
    clipped = np.clip(ratios, 0.8, 1.2) * FIXED_ADVANTAGES < ratios * FIXED_ADVANTAGES
    assert clipped.any() and not clipped.all(), "The batch must exercise both branches of the clip."
    assert math.isfinite(components["total"])
    assert_gradients_match(loss, parameters, _boxed(grads))


# ---------------------------------------------------------------------------
# Losses worked out by hand
# ---------------------------------------------------------------------------
#
# With the trunk zeroed the hidden layer is zero, so the policy mean is the mean bias and the
# value is the value bias at every step. One action dimension, mean 0.5, log_std 0 (variance 1):
#
#   log p(a) = -0.5 * ((a - 0.5)^2 + log 2 pi)
#   actions [1.5, 0.0, -1.0] -> residuals [1, -0.5, -1.5] -> squares [1, 0.25, 2.25]
#
# Rewards [-1, 0, -3] with gamma 0.5:
#
#   G[2] = -3        G[1] = 0 + 0.5 * -3 = -1.5        G[0] = -1 + 0.5 * -1.5 = -1.75
#
# The actions are all different and the rewards uneven, so a return paired with the wrong step
# changes the policy loss.

HAND_ACTIONS = np.array([[1.5], [0.0], [-1.0]])
HAND_RESIDUAL_SQUARES = np.array([1.0, 0.25, 2.25])
HAND_LOG_PROBS = -0.5 * (HAND_RESIDUAL_SQUARES + LOG_2PI)
HAND_RETURNS = np.array([-1.75, -1.5, -3.0])
HAND_VALUE = -2.0
HAND_ENTROPY = 0.5 * (LOG_2PI + 1.0)  # log_std + 0.5 * log(2 pi e) with log_std = 0


def _hand_agent(agent_cls, **config):
    """An agent whose policy is N(0.5, 1) and whose value is -2 for every state."""
    agent = agent_cls(
        observation_dim=3,
        action_dim=1,
        config={"hidden_dim": 5, "seed": 1, "gamma": 0.5, "init_log_std": 0.0, **config},
    )
    agent._W1 = np.zeros_like(agent._W1)
    agent._b1 = np.zeros_like(agent._b1)
    agent._b_mu = np.array([0.5])
    agent._b_v = HAND_VALUE
    return agent


def _hand_batch(dones: tuple[bool, bool, bool] = (False, False, True)) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "observations": rng.standard_normal((3, 3)),  # irrelevant: the trunk is zeroed
        "actions": HAND_ACTIONS.copy(),
        "rewards": np.array([-1.0, 0.0, -3.0]),
        "next_observations": rng.standard_normal((3, 3)),
        "dones": np.array(dones),
    }


def test_actor_critic_losses_match_hand_computation():
    agent = _hand_agent(ContinuousActorCriticAgent)

    metrics = agent.update(_hand_batch())

    # advantages = normalised(G - V) = normalised([0.25, 0.5, -1.0]) = [0.5080, 0.8890, -1.3970]
    # log p = [-1.4189, -1.0439, -2.0439]
    # policy = -mean(log p * A) = -(-0.7208 - 0.9281 + 2.8554) / 3 = -0.4022
    # value  = 0.5 * mean((V - G)^2) = 0.5 * (0.0625 + 0.25 + 1.0) / 3 = 0.21875
    advantages = _normalised(HAND_RETURNS - HAND_VALUE)
    expected_policy = float(-np.mean(HAND_LOG_PROBS * advantages))
    assert expected_policy == pytest.approx(-0.4022, abs=5e-5)  # the hand value, to four places
    assert metrics["policy_loss"] == pytest.approx(expected_policy, rel=1e-6)
    assert metrics["value_loss"] == pytest.approx(0.21875, rel=1e-12)
    assert metrics["entropy"] == pytest.approx(HAND_ENTROPY, rel=1e-12)
    expected_loss = expected_policy + agent.value_coef * 0.21875 - agent.entropy_coef * HAND_ENTROPY
    assert metrics["loss"] == pytest.approx(expected_loss, rel=1e-6)


def test_actor_critic_one_step_episode_uses_its_raw_advantage():
    """A single step cannot be normalised (its spread is zero), so its advantage is used as is."""
    agent = _hand_agent(ContinuousActorCriticAgent)
    batch = {key: value[:1] for key, value in _hand_batch().items()}

    metrics = agent.update(batch)

    # One step: G = -1, V = -2, A = G - V = 1, log p(1.5) = -0.5 * (1 + log 2 pi) = -1.4189
    # policy = -(log p * A) = 1.4189; normalising a single value would make it 0.
    assert metrics["policy_loss"] == pytest.approx(0.5 * (1.0 + LOG_2PI), rel=1e-6)
    assert metrics["value_loss"] == pytest.approx(0.5, rel=1e-12)


def test_ppo_minibatch_loss_matches_hand_computation():
    agent = _hand_agent(ContinuousPPOAgent)  # clip_ratio defaults to 0.2
    states = _hand_batch()["observations"]
    advantages = np.array([1.0, 1.0, -0.5])
    ratios = np.array([1.0, 2.0, 0.5])
    old_log_probs = HAND_LOG_PROBS - np.log(ratios)

    components, _ = agent._minibatch_loss_and_grads(
        states, HAND_ACTIONS, HAND_RETURNS, advantages, old_log_probs
    )

    # step 0: ratio 1.0, A  1.0 -> min(1.0, 1.0)    =  1.0
    # step 1: ratio 2.0, A  1.0 -> min(2.0, 1.2)    =  1.2
    # step 2: ratio 0.5, A -0.5 -> min(-0.25, -0.4) = -0.4
    # policy = -mean = -(1.0 + 1.2 - 0.4) / 3 = -0.6
    assert components["policy"] == pytest.approx(-0.6, rel=1e-6)
    assert components["value"] == pytest.approx(0.21875, rel=1e-12)
    assert components["entropy"] == pytest.approx(HAND_ENTROPY, rel=1e-12)
    expected_total = -0.6 + agent.value_coef * 0.21875 - agent.entropy_coef * HAND_ENTROPY
    assert components["total"] == pytest.approx(expected_total, rel=1e-6)


def test_entropy_sums_over_action_dimensions():
    """Two action dimensions carry twice the entropy of one, not the same amount."""
    agent = _agent(ContinuousActorCriticAgent, action_dim=2, init_log_std=0.0)
    states, actions = _batch(2)

    components, _ = agent._loss_and_grads(states, actions, FIXED_RETURNS, FIXED_ADVANTAGES)

    assert components["entropy"] == pytest.approx(2 * HAND_ENTROPY, rel=1e-12)


def _record_minibatch_inputs(agent) -> list[tuple[np.ndarray, np.ndarray]]:
    """Make the agent record the returns and advantages each mini-batch is given."""
    recorded: list[tuple[np.ndarray, np.ndarray]] = []
    original = agent._minibatch_loss_and_grads

    def recording(states, actions, returns, advantages, old_log_probs, **kwargs):
        recorded.append((np.array(returns), np.array(advantages)))
        return original(states, actions, returns, advantages, old_log_probs, **kwargs)

    agent._minibatch_loss_and_grads = recording
    return recorded


@pytest.mark.parametrize(
    "dones,expected_advantages",
    [
        # V = -2 everywhere, gamma = 0.5, lambda = 0.5, so gamma * lambda = 0.25.
        # Last step done (no bootstrap):
        #   d2 = -3 - (-2) = -1                         A2 = -1
        #   d1 = 0 + 0.5 * -2 + 2 = 1                   A1 = 1 + 0.25 * -1   = 0.75
        #   d0 = -1 + 0.5 * -2 + 2 = 0                  A0 = 0 + 0.25 * 0.75 = 0.1875
        ((False, False, True), [0.1875, 0.75, -1.0]),
        # Last step not done (bootstrap from V(next state) = -2):
        #   d2 = -3 + 0.5 * -2 + 2 = -2                 A2 = -2
        #   d1 = 1                                      A1 = 1 + 0.25 * -2  = 0.5
        #   d0 = 0                                      A0 = 0 + 0.25 * 0.5 = 0.125
        ((False, False, False), [0.125, 0.5, -2.0]),
        # An episode ends in the middle of the batch (step 1), the last step is not done:
        #   d2 = -2                                     A2 = -2
        #   d1 = 0 + 0 - (-2) = 2   (no bootstrap)      A1 = 2             (nothing carried back)
        #   d0 = 0                                      A0 = 0 + 0.25 * 2 = 0.5
        ((False, True, False), [0.5, 2.0, -2.0]),
    ],
)
def test_ppo_update_feeds_hand_computed_gae_to_the_minibatch(dones, expected_advantages):
    agent = _hand_agent(ContinuousPPOAgent, gae_lambda=0.5, ppo_epochs=1, batch_size=8)
    recorded = _record_minibatch_inputs(agent)

    agent.update(_hand_batch(dones))

    assert len(recorded) == 1, "One epoch over three steps with batch_size 8 is one mini-batch."
    returns, advantages = recorded[0]
    order = np.argsort(returns)  # the mini-batch is shuffled; returns are distinct
    raw = np.array(expected_advantages)
    expected_returns = raw + HAND_VALUE
    expected_order = np.argsort(expected_returns)
    np.testing.assert_allclose(returns[order], expected_returns[expected_order], rtol=1e-12)
    np.testing.assert_allclose(
        advantages[order], _normalised(raw)[expected_order], rtol=1e-9, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Computing a loss changes nothing
# ---------------------------------------------------------------------------


def test_actor_critic_loss_and_grads_changes_no_state():
    agent = _agent(ContinuousActorCriticAgent)
    states, actions = _batch()
    before = _state(agent)

    first = agent._loss_and_grads(states, actions, FIXED_RETURNS)
    second = agent._loss_and_grads(states, actions, FIXED_RETURNS)

    _assert_state_unchanged(agent, before)
    assert first[0] == second[0]


def test_ppo_minibatch_loss_and_grads_changes_no_state():
    agent = _agent(ContinuousPPOAgent)
    states, actions = _batch()
    inputs = (
        states,
        actions,
        FIXED_RETURNS,
        FIXED_ADVANTAGES,
        _ppo_old_log_probs(agent, states, actions),
    )
    before = _state(agent)

    first = agent._minibatch_loss_and_grads(*inputs)
    second = agent._minibatch_loss_and_grads(*inputs)

    _assert_state_unchanged(agent, before)
    assert first[0] == second[0]


def test_actor_critic_default_advantages_are_normalised_returns_minus_values():
    """Leaving advantages out is the same as passing normalised(G - V) for the current weights."""
    agent = _agent(ContinuousActorCriticAgent)
    agent._b_v = 0.2
    states, actions = _batch()
    _, _, values, _ = agent._forward(states)

    default_components, default_grads = agent._loss_and_grads(states, actions, FIXED_RETURNS)
    explicit_components, explicit_grads = agent._loss_and_grads(
        states, actions, FIXED_RETURNS, _normalised(FIXED_RETURNS - values)
    )

    assert default_components == explicit_components
    for name in default_grads:
        np.testing.assert_array_equal(default_grads[name], explicit_grads[name])


def test_actor_critic_rejects_advantages_of_the_wrong_length():
    agent = _agent(ContinuousActorCriticAgent)
    states, actions = _batch()

    with pytest.raises(ValueError, match="advantages"):
        agent._loss_and_grads(states, actions, FIXED_RETURNS, np.ones(N_STEPS + 1))


# ---------------------------------------------------------------------------
# The Adam step, and what update() does with the gradients
# ---------------------------------------------------------------------------


class _ReferenceAdam(ReferenceAdam):
    """The shared reference Adam, applying its steps to a continuous agent's parameters."""

    def step(self, agent, grads: dict) -> None:  # type: ignore[override]
        for name, step in super().step(grads).items():
            attribute = PARAMETER_ATTRIBUTES[name]
            current = getattr(agent, attribute)
            setattr(agent, attribute, float(current - step) if name == "b_v" else current - step)


def _assert_same_weights(agent, reference) -> None:
    for name, attribute in PARAMETER_ATTRIBUTES.items():
        np.testing.assert_allclose(
            getattr(agent, attribute),
            getattr(reference, attribute),
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"{name} differs.",
        )


def test_adam_step_matches_an_independent_implementation():
    agent, reference = _agent(ContinuousActorCriticAgent), _agent(ContinuousActorCriticAgent)
    adam = _ReferenceAdam(learning_rate=0.01, max_grad_norm=agent.max_grad_norm)
    rng = np.random.default_rng(5)
    untouched = _state(agent)

    for scale in (0.01, 50.0, 0.3):  # below the clip threshold, far above it, and above again
        grads = {
            name: scale * rng.standard_normal(np.shape(getattr(agent, attribute)))
            for name, attribute in PARAMETER_ATTRIBUTES.items()
        }
        grads["b_v"] = float(grads["b_v"])
        agent._adam_step(dict(grads), 0.01)
        adam.step(reference, grads)

    _assert_same_weights(agent, reference)
    assert agent._optimizer.t == 3
    moved = _state(agent)
    assert all(not np.array_equal(moved[name], untouched[name]) for name in PARAMETER_ATTRIBUTES)


STEP_REWARDS = np.array([1.0, 0.0, 2.0, 1.0, 0.0, 3.0])
# Discounted returns of STEP_REWARDS for gamma 0.5, from the last step back:
#   3;  0 + 1.5 = 1.5;  1 + 0.75 = 1.75;  2 + 0.875 = 2.875;  0 + 1.4375;  1 + 0.71875
STEP_RETURNS = np.array([1.71875, 1.4375, 2.875, 1.75, 1.5, 3.0])


def _step_batch(last_done: bool = True) -> dict[str, np.ndarray]:
    states, actions = _batch()
    rng = np.random.default_rng(21)
    return {
        "observations": states,
        "actions": actions,
        "rewards": STEP_REWARDS.copy(),
        "next_observations": rng.standard_normal((N_STEPS, OBS_DIM)),
        "dones": np.array([False] * (N_STEPS - 1) + [last_done]),
    }


def test_actor_critic_update_takes_one_adam_step_on_its_gradients():
    config = {"gamma": 0.5, "learning_rate": 0.01}
    agent = _agent(ContinuousActorCriticAgent, **config)
    reference = _agent(ContinuousActorCriticAgent, **config)
    batch = _step_batch()

    metrics = agent.update(batch)

    components, grads = reference._loss_and_grads(
        batch["observations"], batch["actions"], STEP_RETURNS
    )
    _ReferenceAdam(0.01, reference.max_grad_norm).step(reference, grads)
    _assert_same_weights(agent, reference)
    assert agent._optimizer.t == 1
    assert metrics["policy_loss"] == pytest.approx(components["policy"], rel=1e-12)
    assert metrics["value_loss"] == pytest.approx(components["value"], rel=1e-12)
    assert metrics["entropy"] == pytest.approx(components["entropy"], rel=1e-12)
    assert metrics["loss"] == pytest.approx(components["total"], rel=1e-12)


def _with_biases(agent):
    """Give every bias a non-zero value, so a forward pass that drops one is visible."""
    agent._b1 = np.linspace(-0.3, 0.3, HIDDEN_DIM)
    agent._b_mu = np.array([0.2, -0.1])
    agent._b_v = 0.4
    return agent


def _independent_forward(agent, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Policy means and values for a batch of states, written out from the weights."""
    hidden = np.maximum(0.0, states @ agent._W1 + agent._b1)
    return hidden @ agent._W_mu + agent._b_mu, hidden @ agent._W_v + agent._b_v


def _independent_log_prob(agent, actions: np.ndarray, mu: np.ndarray) -> np.ndarray:
    variance = np.exp(2.0 * agent._log_std) + 1e-8  # the agents add 1e-8 to the variance
    per_dimension = -0.5 * ((actions - mu) ** 2 / variance + 2.0 * agent._log_std + LOG_2PI)
    return np.sum(per_dimension, axis=-1)


def test_forward_matches_the_network_written_out_from_its_weights():
    """The forward pass used to choose actions and to record values includes every bias."""
    agent = _with_biases(_agent(ContinuousPPOAgent))
    states, actions = _batch()
    expected_mu, expected_values = _independent_forward(agent, states)

    mu, log_std, values, _ = agent._forward(states)
    single_mu, _, single_value, _ = agent._forward(states[0])

    np.testing.assert_allclose(mu, expected_mu, rtol=1e-12)
    np.testing.assert_allclose(values, expected_values, rtol=1e-12)
    np.testing.assert_allclose(single_mu, expected_mu[0], rtol=1e-12)
    assert single_value == pytest.approx(expected_values[0], rel=1e-12)
    np.testing.assert_allclose(
        agent._log_prob(actions, mu, log_std),
        _independent_log_prob(agent, actions, expected_mu),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "last_done,batch_size",
    [
        (True, 4),  # mini-batches of 4 and 2
        (False, 4),  # the same, bootstrapping from the state after the last step
        (True, 5),  # mini-batches of 5 and 1: the trailing single step must still be used
    ],
)
def test_ppo_update_replays_as_minibatch_steps_on_fixed_inputs(last_done, batch_size):
    """Two epochs of shuffled mini-batches, against inputs recorded before the first.

    The reference agent is built identically, so its generator produces the same shuffles; it
    must not draw from that generator for anything else. Its values, log-probabilities and
    advantages come from a forward pass and a GAE written out here, and its steps from the
    independent Adam above.
    """
    config = {
        "gamma": 0.5,
        "gae_lambda": 0.5,
        "learning_rate": 0.05,
        "ppo_epochs": 2,
        "batch_size": batch_size,
    }
    agent = _with_biases(_agent(ContinuousPPOAgent, **config))
    reference = _with_biases(_agent(ContinuousPPOAgent, **config))
    batch = _step_batch(last_done)
    states, actions = batch["observations"], batch["actions"]

    metrics = agent.update(batch)

    mu, values = _independent_forward(reference, states)
    old_log_probs = _independent_log_prob(reference, actions, mu)
    next_value = (
        0.0
        if last_done
        else float(_independent_forward(reference, batch["next_observations"][-1:])[1][0])
    )
    raw_advantages = np.zeros(N_STEPS)
    running = 0.0
    for step in reversed(range(N_STEPS)):
        following = values[step + 1] if step + 1 < N_STEPS else next_value
        not_done = 0.0 if batch["dones"][step] else 1.0
        delta = STEP_REWARDS[step] + 0.5 * following * not_done - values[step]
        running = delta + 0.5 * 0.5 * not_done * running
        raw_advantages[step] = running
    returns = raw_advantages + values
    advantages = _normalised(raw_advantages)

    adam = _ReferenceAdam(0.05, reference.max_grad_norm)
    indices = np.arange(N_STEPS)
    seen: list[dict[str, float]] = []
    for _ in range(2):
        reference._rng.shuffle(indices)
        for start in range(0, N_STEPS, batch_size):
            chosen = indices[start : start + batch_size]
            components, grads = reference._minibatch_loss_and_grads(
                states[chosen],
                actions[chosen],
                returns[chosen],
                advantages[chosen],
                old_log_probs[chosen],
            )
            adam.step(reference, grads)
            seen.append(components)

    assert len(seen) == 4, "Two epochs of two mini-batches each."
    assert len(chosen) == N_STEPS % batch_size, "The last mini-batch is the remainder."
    _assert_same_weights(agent, reference)
    assert agent._optimizer.t == 4
    for metric, component in (
        ("policy_loss", "policy"),
        ("value_loss", "value"),
        ("entropy", "entropy"),
    ):
        expected = float(np.mean([components[component] for components in seen]))
        assert metrics[metric] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("agent_cls", [ContinuousActorCriticAgent, ContinuousPPOAgent])
def test_checkpoint_resumes_with_the_same_adam_state(agent_cls, tmp_path):
    """A reloaded agent continues exactly: the checkpoint holds the Adam moments, not only t."""
    agent = _agent(agent_cls, gamma=0.5)
    batch = _step_batch()
    agent.update(batch)
    path = str(tmp_path / "agent.npz")
    agent.save(path)

    resumed = _agent(agent_cls, gamma=0.5, seed=99)  # different initial weights
    resumed.load(path)
    assert resumed._optimizer.t == agent._optimizer.t > 0
    # Checkpoints do not hold the sampling generator (PPO's shuffle); share it so only the
    # optimiser state is under test.
    resumed._rng = copy.deepcopy(agent._rng)

    agent.update(batch)
    resumed.update(batch)
    after, expected = _state(resumed), _state(agent)
    for name in expected:
        np.testing.assert_array_equal(after[name], expected[name], err_msg=f"{name} differs.")

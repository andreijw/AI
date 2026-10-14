"""The hand-written maths of the discrete policy-gradient agents.

For REINFORCE, Actor-Critic and PPO this checks three things the behaviour tests cannot:

* the analytic gradients agree with finite differences of the loss, parameter by parameter;
* the loss equals a value worked out by hand, so a return paired with the wrong step is caught;
* computing a loss and its gradients changes no weight.

Each agent exposes one method that computes the loss and gradients for given inputs. Whatever
the algorithm holds constant while differentiating (standardised returns, advantages, old
log-probabilities) is an argument of that method, so the numerical gradient holds it constant
too. Losses and gradients are summed over the steps of the batch, as the agents accumulate them.
"""

import numpy as np
import pytest

from rl_core.agents import ActorCriticAgent, PPOAgent, ReinforceAgent
from tests.helpers import ReferenceAdam, assert_gradients_match

EPS = 1e-5  # perturbation used by tests.helpers.numerical_grad
N_STEPS, OBS_DIM, HIDDEN_DIM, ACTION_DIM = 6, 4, 6, 3

# ---------------------------------------------------------------------------
# Fixtures for the gradient checks: a tiny network and a fixed batch
# ---------------------------------------------------------------------------


def _batch(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    observations = rng.standard_normal((N_STEPS, OBS_DIM))
    actions = rng.integers(0, ACTION_DIM, size=N_STEPS)
    return observations, actions


def _agent(agent_cls, **config):
    # value_coef and entropy_coef are set away from 1 and from their tiny defaults so that
    # dropping either from a gradient changes it by far more than the test's tolerance.
    settings = {"hidden_dim": HIDDEN_DIM, "seed": 11, "value_coef": 0.5, "entropy_coef": 0.3}
    return agent_cls(observation_dim=OBS_DIM, action_dim=ACTION_DIM, config={**settings, **config})


def _assert_away_from_kinks(agent, observations: np.ndarray) -> None:
    """A finite difference is only meaningful where the loss is smooth.

    The hidden units must not sit on the ReLU kink, some must be active and some inactive (or a
    missing ReLU mask in backpropagation would go unnoticed), and no probability may be near the
    1e-8 floor the agents clip at before taking a logarithm.
    """
    pre_activation = observations @ agent._W1 + agent._b1
    margin = 100 * EPS * max(1.0, float(np.abs(observations).max()))
    assert np.abs(pre_activation).min() > margin, "A hidden unit sits on the ReLU kink."
    active = pre_activation > 0
    assert active.any() and not active.all(), "Need both active and inactive hidden units."
    forward = agent._policy if isinstance(agent, ReinforceAgent) else agent._forward
    smallest = min(float(np.min(forward(obs)[0])) for obs in observations)
    assert smallest > 1e-6, "A probability is close to the clip floor."


def _weights(agent) -> dict[str, np.ndarray]:
    return {
        name: np.array(value, dtype=np.float64)
        for name, value in vars(agent).items()
        if name.startswith(("_W", "_b")) and isinstance(value, (np.ndarray, float))
    }


def _assert_weights_unchanged(agent, before: dict[str, np.ndarray]) -> None:
    after = _weights(agent)
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(after[name], before[name], err_msg=f"{name} was modified.")


def _actor_critic_parameters(agent) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Live parameter arrays for the gradient check, with the float value bias boxed.

    ``_b_v`` is a Python float, which cannot be perturbed in place. The returned one-element
    array stands in for it; the loss closure copies it back onto the agent before each call.
    """
    boxed_value_bias = np.array([agent._b_v], dtype=np.float64)
    parameters = {
        "W1": agent._W1,
        "b1": agent._b1,
        "W_pi": agent._W_pi,
        "b_pi": agent._b_pi,
        "W_v": agent._W_v,
        "b_v": boxed_value_bias,
    }
    return parameters, boxed_value_bias


def _boxed(grads: dict) -> dict[str, np.ndarray]:
    return {
        name: np.atleast_1d(np.asarray(value, dtype=np.float64)) for name, value in grads.items()
    }


# ---------------------------------------------------------------------------
# Gradient checks
# ---------------------------------------------------------------------------


def test_reinforce_gradients_match_finite_differences():
    agent = _agent(ReinforceAgent)
    observations, actions = _batch()
    return_weights = np.array([0.7, -1.2, 0.4, 1.5, -0.9, 0.3])
    _assert_away_from_kinks(agent, observations)

    _, grads = agent._loss_and_grads(observations, actions, return_weights)

    assert_gradients_match(
        lambda: agent._loss_and_grads(observations, actions, return_weights)[0],
        {"W1": agent._W1, "b1": agent._b1, "W2": agent._W2, "b2": agent._b2},
        grads,
    )


def test_actor_critic_gradients_match_finite_differences():
    agent = _agent(ActorCriticAgent)
    agent._b_v = 0.2
    observations, actions = _batch()
    returns = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5])
    advantages = np.array([0.8, -1.1, 0.5, 1.3, -0.6, 0.2])  # held constant
    _assert_away_from_kinks(agent, observations)
    parameters, boxed_value_bias = _actor_critic_parameters(agent)

    def loss() -> float:
        agent._b_v = float(boxed_value_bias[0])
        return agent._loss_and_grads(observations, actions, returns, advantages)[0]["total"]

    _, grads = agent._loss_and_grads(observations, actions, returns, advantages)

    assert_gradients_match(loss, parameters, _boxed(grads))


def _ppo_inputs(agent, observations, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns, advantages and old log-probabilities that put ratios on both sides of the clip."""
    returns = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5])
    advantages = np.array([0.8, -1.1, 0.5, 1.3, -0.6, 0.2])
    probabilities = [agent._forward(obs)[0] for obs in observations]
    current = np.log([probs[a] for probs, a in zip(probabilities, actions, strict=True)])
    # ratio = exp(current - old) = exp(-offset): 0.61, 1.65, 0.95, 1.05, 0.74, 1.35
    old_log_probs = current + np.array([0.5, -0.5, 0.05, -0.05, 0.3, -0.3])
    return returns, advantages, old_log_probs


def test_ppo_gradients_match_finite_differences():
    agent = _agent(PPOAgent)
    agent._b_v = 0.2
    observations, actions = _batch()
    returns, advantages, old_log_probs = _ppo_inputs(agent, observations, actions)
    _assert_away_from_kinks(agent, observations)
    parameters, boxed_value_bias = _actor_critic_parameters(agent)
    inputs = (observations, actions, returns, advantages, old_log_probs)

    def loss() -> float:
        agent._b_v = float(boxed_value_bias[0])
        return agent._epoch_loss_and_grads(*inputs)[0]["total"]

    _, grads, clipped_count = agent._epoch_loss_and_grads(*inputs)

    assert 0 < clipped_count < N_STEPS, "The batch must exercise both branches of the clip."
    assert_gradients_match(loss, parameters, _boxed(grads))


# ---------------------------------------------------------------------------
# Losses worked out by hand
# ---------------------------------------------------------------------------
#
# With the trunk zeroed the hidden layer is zero for every observation, so the policy is
# softmax(policy bias) and the value is the value bias at every step. The batch below is three
# steps with rewards [1, 0, 3] and gamma 0.5:
#
#   G[2] = 3        G[1] = 0 + 0.5 * 3 = 1.5        G[0] = 1 + 0.5 * 1.5 = 1.75
#
# The rewards are deliberately uneven: reversing them, reversing the returns or squaring the
# discount each gives a different loss, so a return paired with the wrong step is caught.

HAND_RETURNS = np.array([1.75, 1.5, 3.0])
HAND_ACTIONS = np.array([0, 1, 1])
HAND_PROBS = np.array([0.8, 0.2])
HAND_LOG_PROBS = np.log(HAND_PROBS)[HAND_ACTIONS]  # log pi(a_t) for each step
HAND_VALUE = 0.5
HAND_ENTROPY = -float(np.sum(HAND_PROBS * np.log(HAND_PROBS)))  # 0.5004...


def _hand_batch() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "observations": rng.standard_normal((3, 4)),  # irrelevant: the trunk is zeroed
        "actions": HAND_ACTIONS.copy(),
        "rewards": np.array([1.0, 0.0, 3.0]),
        "next_observations": rng.standard_normal((3, 4)),
        "dones": np.array([False, False, True]),
    }


def _hand_agent(agent_cls, **config):
    """An agent whose policy is [0.8, 0.2] and whose value is 0.5 for every observation."""
    agent = agent_cls(
        observation_dim=4, action_dim=2, config={"hidden_dim": 5, "seed": 1, "gamma": 0.5, **config}
    )
    agent._W1 = np.zeros_like(agent._W1)
    agent._b1 = np.zeros_like(agent._b1)
    if isinstance(agent, ReinforceAgent):
        agent._b2 = np.log(HAND_PROBS)
    else:
        agent._b_pi = np.log(HAND_PROBS)
        agent._b_v = HAND_VALUE
    return agent


def _standardised(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std()


def test_reinforce_policy_loss_matches_hand_computation():
    agent = _hand_agent(ReinforceAgent)

    metrics = agent.update(_hand_batch())

    # weights = standardised returns = [-0.5080, -0.8890, 1.3970]
    # loss = mean(-weight * log pi(a)) = 0.2347
    expected = float(np.mean(-_standardised(HAND_RETURNS) * HAND_LOG_PROBS))
    assert expected == pytest.approx(0.2347, abs=5e-5)  # the hand value, to four places
    assert metrics["policy_loss"] == pytest.approx(expected, rel=1e-12)


def test_actor_critic_losses_match_hand_computation():
    agent = _hand_agent(ActorCriticAgent)

    metrics = agent.update(_hand_batch())

    # advantages = G - V = [1.25, 1.0, 2.5]
    # policy = mean(-log pi(a) * A) = (0.2231 * 1.25 + 1.6094 * 1.0 + 1.6094 * 2.5) / 3 = 1.9707
    # value  = mean(0.5 * (V - G)^2) = 0.5 * (1.5625 + 1.0 + 6.25) / 3 = 1.46875
    advantages = HAND_RETURNS - HAND_VALUE
    expected_policy = float(np.mean(-HAND_LOG_PROBS * advantages))
    assert expected_policy == pytest.approx(1.9707, abs=5e-5)
    assert metrics["policy_loss"] == pytest.approx(expected_policy, rel=1e-12)
    assert metrics["value_loss"] == pytest.approx(1.46875, rel=1e-12)
    assert metrics["entropy_bonus"] == pytest.approx(HAND_ENTROPY, rel=1e-12)


def test_ppo_epoch_loss_matches_hand_computation():
    agent = _hand_agent(PPOAgent)  # clip_epsilon defaults to 0.2
    observations = _hand_batch()["observations"]
    advantages = np.array([1.0, 1.0, -0.5])
    ratios = np.array([1.0, 2.0, 0.5])
    old_log_probs = HAND_LOG_PROBS - np.log(ratios)

    components, _, clipped_count = agent._epoch_loss_and_grads(
        observations, HAND_ACTIONS, HAND_RETURNS, advantages, old_log_probs
    )

    # step 0: ratio 1.0, A  1.0 -> min(1.0, 1.0)    =  1.0   not clipped
    # step 1: ratio 2.0, A  1.0 -> min(2.0, 1.2)    =  1.2   clipped
    # step 2: ratio 0.5, A -0.5 -> min(-0.25, -0.4) = -0.4   clipped
    # policy = -(1.0 + 1.2 - 0.4) = -1.8
    assert clipped_count == 2
    assert components["policy"] == pytest.approx(-1.8, rel=1e-12)
    # value = sum(0.5 * (V - G)^2) = 0.5 * (1.5625 + 1.0 + 6.25) = 4.40625
    assert components["value"] == pytest.approx(4.40625, rel=1e-12)
    assert components["entropy"] == pytest.approx(3 * HAND_ENTROPY, rel=1e-12)
    expected_total = -1.8 + agent.value_coef * 4.40625 - agent.entropy_coef * 3 * HAND_ENTROPY
    assert components["total"] == pytest.approx(expected_total, rel=1e-12)


def test_ppo_update_applies_one_epoch_built_from_hand_computed_returns():
    """update() must feed the epoch the returns, advantages and log-probabilities it should.

    PPO's reported losses cannot show a return paired with the wrong step: the first-epoch policy
    loss is zero for any normalised advantages, and the value loss does not change when returns
    are permuted. The weights do, so this compares them with what one epoch gives when the test
    builds the inputs itself from the hand-written returns.
    """
    agent = _hand_agent(PPOAgent, ppo_epochs=1, learning_rate=0.1)
    reference = _hand_agent(PPOAgent, ppo_epochs=1, learning_rate=0.1)
    batch = _hand_batch()

    metrics = agent.update(batch)

    advantages = _standardised(HAND_RETURNS - HAND_VALUE)
    _, grads, _ = reference._epoch_loss_and_grads(
        batch["observations"], HAND_ACTIONS, HAND_RETURNS, advantages, HAND_LOG_PROBS
    )
    scale = reference.learning_rate / 3
    np.testing.assert_allclose(
        agent._b_pi, reference._b_pi - scale * grads["b_pi"], rtol=1e-12, atol=0
    )
    assert agent._b_v == pytest.approx(reference._b_v - scale * grads["b_v"], rel=1e-12)
    assert not np.array_equal(agent._b_pi, reference._b_pi), "The update did not move the policy."
    assert metrics["value_loss"] == pytest.approx(1.46875, rel=1e-12)


# ---------------------------------------------------------------------------
# Computing a loss changes nothing
# ---------------------------------------------------------------------------


def test_reinforce_loss_and_grads_changes_no_weight():
    agent = _agent(ReinforceAgent)
    observations, actions = _batch()
    return_weights = np.array([0.7, -1.2, 0.4, 1.5, -0.9, 0.3])
    before = _weights(agent)

    first = agent._loss_and_grads(observations, actions, return_weights)
    second = agent._loss_and_grads(observations, actions, return_weights)

    _assert_weights_unchanged(agent, before)
    assert first[0] == second[0]


def test_actor_critic_loss_and_grads_changes_no_weight():
    agent = _agent(ActorCriticAgent)
    observations, actions = _batch()
    returns = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5])
    before = _weights(agent)

    first = agent._loss_and_grads(observations, actions, returns)
    second = agent._loss_and_grads(observations, actions, returns)

    _assert_weights_unchanged(agent, before)
    assert first[0] == second[0]


def test_ppo_epoch_loss_and_grads_changes_no_weight():
    agent = _agent(PPOAgent)
    observations, actions = _batch()
    inputs = (observations, actions, *_ppo_inputs(agent, observations, actions))
    before = _weights(agent)

    first = agent._epoch_loss_and_grads(*inputs)
    second = agent._epoch_loss_and_grads(*inputs)

    _assert_weights_unchanged(agent, before)
    assert first[0] == second[0]


def test_actor_critic_default_advantages_are_returns_minus_current_values():
    """Leaving advantages out is the same as passing G - V(s) for the current weights.

    update() relies on the default so that it needs only one forward pass per step; the gradient
    check passes the array explicitly. This ties the two paths together.
    """
    agent = _agent(ActorCriticAgent)
    agent._b_v = 0.2
    observations, actions = _batch()
    returns = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5])
    values = np.array([agent._forward(obs)[1] for obs in observations])

    default_components, default_grads = agent._loss_and_grads(observations, actions, returns)
    explicit_components, explicit_grads = agent._loss_and_grads(
        observations, actions, returns, returns - values
    )

    assert default_components == explicit_components
    for name in default_grads:
        np.testing.assert_array_equal(default_grads[name], explicit_grads[name])


def test_actor_critic_rejects_advantages_of_the_wrong_length():
    agent = _agent(ActorCriticAgent)
    observations, actions = _batch()
    returns = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5])

    with pytest.raises(ValueError, match="advantages"):
        agent._loss_and_grads(observations, actions, returns, np.ones(N_STEPS + 1))


# ---------------------------------------------------------------------------
# update() turns the gradients into the right step
# ---------------------------------------------------------------------------
#
# The checks above pin the loss and its gradients. These pin what update() does with them:
# descend (not ascend), scale by learning_rate / n_steps, touch every parameter, and for PPO
# repeat for each epoch against the log-probabilities recorded before the first one. The
# expected weights are built in the test with plain arithmetic, on a network whose trunk is not
# zeroed, so every parameter has a non-zero gradient.

STEP_REWARDS = np.array([1.0, 0.0, 2.0, 1.0, 0.0, 3.0])
# Discounted returns of STEP_REWARDS for gamma 0.5, from the last step back:
#   3;  0 + 1.5 = 1.5;  1 + 0.75 = 1.75;  2 + 0.875 = 2.875;  0 + 1.4375;  1 + 0.71875
STEP_RETURNS = np.array([1.71875, 1.4375, 2.875, 1.75, 1.5, 3.0])
PARAMETER_ATTRIBUTES = {
    "W1": "_W1",
    "b1": "_b1",
    "W2": "_W2",
    "b2": "_b2",
    "W_pi": "_W_pi",
    "b_pi": "_b_pi",
    "W_v": "_W_v",
    "b_v": "_b_v",
}


def _step_batch() -> dict[str, np.ndarray]:
    observations, actions = _batch()
    return {
        "observations": observations,
        "actions": actions,
        "rewards": STEP_REWARDS.copy(),
        "next_observations": observations.copy(),
        "dones": np.array([False] * (N_STEPS - 1) + [True]),
    }


def _descend(agent, grads: dict, scale: float) -> None:
    """Apply one plain gradient-descent step to an agent, written out independently of it."""
    for name, gradient in grads.items():
        attribute = PARAMETER_ATTRIBUTES[name]
        setattr(agent, attribute, getattr(agent, attribute) - scale * gradient)


def _assert_same_weights(agent, reference) -> None:
    actual, expected = _weights(agent), _weights(reference)
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=1e-12, atol=1e-15, err_msg=f"{name} differs."
        )


def test_reinforce_update_descends_the_gradient_scaled_by_learning_rate_over_steps():
    config = {"gamma": 0.5, "learning_rate": 0.05}
    agent, reference = _agent(ReinforceAgent, **config), _agent(ReinforceAgent, **config)
    batch = _step_batch()
    untouched = _weights(agent)

    metrics = agent.update(batch)

    loss, grads = reference._loss_and_grads(
        batch["observations"], batch["actions"], _standardised(STEP_RETURNS)
    )
    _descend(reference, grads, scale=0.05 / N_STEPS)
    _assert_same_weights(agent, reference)
    assert metrics["policy_loss"] == pytest.approx(loss / N_STEPS, rel=1e-12)
    moved = _weights(agent)
    assert all(not np.array_equal(moved[name], untouched[name]) for name in moved)


def test_actor_critic_update_descends_the_gradient_scaled_by_learning_rate_over_steps():
    config = {"gamma": 0.5, "learning_rate": 0.05}
    agent, reference = _agent(ActorCriticAgent, **config), _agent(ActorCriticAgent, **config)
    batch = _step_batch()
    untouched = _weights(agent)

    metrics = agent.update(batch)

    components, grads = reference._loss_and_grads(
        batch["observations"], batch["actions"], STEP_RETURNS
    )
    _descend(reference, grads, scale=0.05 / N_STEPS)
    _assert_same_weights(agent, reference)
    assert metrics["policy_loss"] == pytest.approx(components["policy"] / N_STEPS, rel=1e-12)
    assert metrics["value_loss"] == pytest.approx(components["value"] / N_STEPS, rel=1e-12)
    assert metrics["entropy_bonus"] == pytest.approx(components["entropy"] / N_STEPS, rel=1e-12)
    moved = _weights(agent)
    assert all(not np.array_equal(moved[name], untouched[name]) for name in moved)


def test_ppo_update_runs_every_epoch_against_the_log_probabilities_recorded_first():
    """Two epochs: the second must use the first's weights but the original log-probabilities.

    The learning rate is large so that the first epoch moves the policy far enough for the
    second to clip some steps. The reference repeats one epoch at a time with the inputs held
    fixed, which is what makes the ratio differ from one and the clip matter.
    """
    config = {"gamma": 0.5, "learning_rate": 2.0, "ppo_epochs": 2}
    agent, reference = _agent(PPOAgent, **config), _agent(PPOAgent, **config)
    batch = _step_batch()
    observations, actions = batch["observations"], batch["actions"]

    metrics = agent.update(batch)

    forward = [reference._forward(obs) for obs in observations]
    old_log_probs = np.log([probs[a] for (probs, _, _), a in zip(forward, actions, strict=True)])
    old_values = np.array([value for _, value, _ in forward])
    assert np.ptp(old_values) > 1e-3, "The value function must not be constant for this test."
    advantages = _standardised(STEP_RETURNS - old_values)
    for _ in range(2):
        components, grads, clipped_count = reference._epoch_loss_and_grads(
            observations, actions, STEP_RETURNS, advantages, old_log_probs
        )
        _descend(reference, grads, scale=2.0 / N_STEPS)

    _assert_same_weights(agent, reference)
    assert 0 < clipped_count < N_STEPS, "The second epoch must clip some steps but not all."
    assert metrics["policy_loss"] == pytest.approx(components["policy"] / N_STEPS, rel=1e-12)
    assert metrics["value_loss"] == pytest.approx(components["value"] / N_STEPS, rel=1e-12)
    assert metrics["entropy_bonus"] == pytest.approx(components["entropy"] / N_STEPS, rel=1e-12)
    assert metrics["clip_fraction"] == pytest.approx(clipped_count / N_STEPS, rel=1e-12)
    assert metrics["policy_loss"] != pytest.approx(components["total"] / N_STEPS, rel=1e-6)


# ---------------------------------------------------------------------------
# The Adam option: `optimizer: adam` steps the mean gradient with Adam instead of SGD
# ---------------------------------------------------------------------------
# The clip of 0.05 is below the mean gradient's norm for this batch, so it scales every step.

ADAM = {"gamma": 0.5, "learning_rate": 0.05, "optimizer": "adam", "max_grad_norm": 0.05}


def _adam_descend(agent, adam: ReferenceAdam, grads: dict) -> None:
    """Step an agent with the reference Adam on the episode-mean gradient."""
    mean = {
        name: np.asarray(gradient, dtype=np.float64) / N_STEPS for name, gradient in grads.items()
    }
    norm = np.sqrt(sum(float(np.sum(np.square(g))) for g in mean.values()))
    assert norm > ADAM["max_grad_norm"], "The clip must bite for this test to cover it."
    for name, step in adam.step(mean).items():
        attribute = PARAMETER_ATTRIBUTES[name]
        current = getattr(agent, attribute)
        setattr(agent, attribute, float(current - step) if name == "b_v" else current - step)


def test_reinforce_with_adam_steps_the_mean_gradient_with_adam():
    agent, reference = _agent(ReinforceAgent, **ADAM), _agent(ReinforceAgent, **ADAM)
    adam = ReferenceAdam(ADAM["learning_rate"], ADAM["max_grad_norm"])
    batch = _step_batch()

    for _ in range(2):  # the second step uses the moments of the first
        agent.update(batch)
        _, grads = reference._loss_and_grads(
            batch["observations"], batch["actions"], _standardised(STEP_RETURNS)
        )
        _adam_descend(reference, adam, grads)

    _assert_same_weights(agent, reference)


def test_actor_critic_with_adam_steps_the_mean_gradient_with_adam():
    agent, reference = _agent(ActorCriticAgent, **ADAM), _agent(ActorCriticAgent, **ADAM)
    adam = ReferenceAdam(ADAM["learning_rate"], ADAM["max_grad_norm"])
    batch = _step_batch()

    for _ in range(2):
        agent.update(batch)
        _, grads = reference._loss_and_grads(batch["observations"], batch["actions"], STEP_RETURNS)
        _adam_descend(reference, adam, grads)

    _assert_same_weights(agent, reference)


def test_ppo_with_adam_takes_one_adam_step_per_epoch():
    config = {**ADAM, "ppo_epochs": 2}
    agent, reference = _agent(PPOAgent, **config), _agent(PPOAgent, **config)
    adam = ReferenceAdam(ADAM["learning_rate"], ADAM["max_grad_norm"])
    batch = _step_batch()
    observations, actions = batch["observations"], batch["actions"]

    agent.update(batch)

    forward = [reference._forward(obs) for obs in observations]
    old_log_probs = np.log([probs[a] for (probs, _, _), a in zip(forward, actions, strict=True)])
    advantages = _standardised(STEP_RETURNS - np.array([value for _, value, _ in forward]))
    for _ in range(2):
        _, grads, _ = reference._epoch_loss_and_grads(
            observations, actions, STEP_RETURNS, advantages, old_log_probs
        )
        _adam_descend(reference, adam, grads)

    _assert_same_weights(agent, reference)
    assert agent._optimizer.t == 2


DISCRETE_AGENTS = [ReinforceAgent, ActorCriticAgent, PPOAgent]


@pytest.mark.parametrize("agent_cls", DISCRETE_AGENTS)
def test_default_optimizer_is_plain_sgd(agent_cls):
    assert _agent(agent_cls)._optimizer is None


@pytest.mark.parametrize("agent_cls", DISCRETE_AGENTS)
def test_unknown_optimizer_is_rejected_at_construction(agent_cls):
    with pytest.raises(ValueError, match="optimizer must be one of"):
        _agent(agent_cls, optimizer="rmsprop")


@pytest.mark.parametrize("agent_cls", DISCRETE_AGENTS)
def test_adam_checkpoint_resumes_exactly(agent_cls, tmp_path):
    """A reloaded Adam agent continues with the same moments and step count as the original."""
    agent = _agent(agent_cls, **ADAM)
    batch = _step_batch()
    agent.update(batch)
    path = str(tmp_path / "agent.npz")
    agent.save(path)

    resumed = _agent(agent_cls, **{**ADAM, "seed": 99})  # different initial weights
    resumed.load(path)
    assert resumed._optimizer.t == agent._optimizer.t

    agent.update(batch)
    resumed.update(batch)
    actual, expected = _weights(resumed), _weights(agent)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=f"{name} differs.")


@pytest.mark.parametrize("agent_cls", DISCRETE_AGENTS)
def test_sgd_checkpoint_loads_into_an_adam_agent_with_fresh_optimizer_state(agent_cls, tmp_path):
    sgd = _agent(agent_cls, gamma=0.5)
    path = str(tmp_path / "sgd.npz")
    sgd.save(path)
    with np.load(path) as data:
        assert not any(key.startswith("adam_") for key in data.files)

    adam_agent = _agent(agent_cls, **ADAM)
    adam_agent.update(_step_batch())
    adam_agent.load(path)

    assert adam_agent._optimizer.t == 0
    for moment in (*adam_agent._optimizer.m.values(), *adam_agent._optimizer.v.values()):
        np.testing.assert_array_equal(moment, 0.0)
    _assert_same_weights(adam_agent, sgd)

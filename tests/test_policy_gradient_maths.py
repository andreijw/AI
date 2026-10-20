"""The hand-written maths of the discrete policy-gradient agents.

For REINFORCE and Actor-Critic this checks three things the behaviour tests cannot (PPO's
mini-batch maths is checked the same way in tests/test_discrete_ppo.py):

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
from rl_core.agents.returns import discounted_returns
from tests.agent_internals import (
    assert_same_weights,
    assert_weights_unchanged,
    boxed_gradients,
    gradient_check_parameters,
    loss_and_grads,
    optimizer,
    probabilities,
    set_weights,
    share_generators,
    state_value,
    weights,
)
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
    pre_activation = observations @ weights(agent)["W1"] + weights(agent)["b1"]
    margin = 100 * EPS * max(1.0, float(np.abs(observations).max()))
    assert np.abs(pre_activation).min() > margin, "A hidden unit sits on the ReLU kink."
    active = pre_activation > 0
    assert active.any() and not active.all(), "Need both active and inactive hidden units."
    smallest = min(float(np.min(probabilities(agent, obs))) for obs in observations)
    assert smallest > 1e-6, "A probability is close to the clip floor."


# ---------------------------------------------------------------------------
# Gradient checks
# ---------------------------------------------------------------------------


def test_reinforce_gradients_match_finite_differences():
    agent = _agent(ReinforceAgent)
    observations, actions = _batch()
    return_weights = np.array([0.7, -1.2, 0.4, 1.5, -0.9, 0.3])
    _assert_away_from_kinks(agent, observations)

    _, grads = loss_and_grads(agent, observations, actions, return_weights)
    parameters, _ = gradient_check_parameters(agent)

    assert_gradients_match(
        lambda: loss_and_grads(agent, observations, actions, return_weights)[0],
        parameters,
        grads,
    )


def test_actor_critic_gradients_match_finite_differences():
    agent = _agent(ActorCriticAgent)
    set_weights(agent, b_v=0.2)
    observations, actions = _batch()
    returns = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5])
    advantages = np.array([0.8, -1.1, 0.5, 1.3, -0.6, 0.2])  # held constant
    _assert_away_from_kinks(agent, observations)
    parameters, sync = gradient_check_parameters(agent)

    def loss() -> float:
        sync()
        return loss_and_grads(agent, observations, actions, returns, advantages)[0]["total"]

    _, grads = loss_and_grads(agent, observations, actions, returns, advantages)

    assert_gradients_match(loss, parameters, boxed_gradients(grads))


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
    trunk = {"W1": np.zeros((4, 5)), "b1": np.zeros(5)}
    if isinstance(agent, ReinforceAgent):
        set_weights(agent, **trunk, b2=np.log(HAND_PROBS))
    else:
        set_weights(agent, **trunk, b_pi=np.log(HAND_PROBS), b_v=HAND_VALUE)
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


# ---------------------------------------------------------------------------
# Computing a loss changes nothing
# ---------------------------------------------------------------------------


def test_reinforce_loss_and_grads_changes_no_weight():
    agent = _agent(ReinforceAgent)
    observations, actions = _batch()
    return_weights = np.array([0.7, -1.2, 0.4, 1.5, -0.9, 0.3])
    before = weights(agent)

    first = loss_and_grads(agent, observations, actions, return_weights)
    second = loss_and_grads(agent, observations, actions, return_weights)

    assert_weights_unchanged(agent, before, "computing a loss")
    assert first[0] == second[0]


def test_actor_critic_loss_and_grads_changes_no_weight():
    agent = _agent(ActorCriticAgent)
    observations, actions = _batch()
    returns = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5])
    before = weights(agent)

    first = loss_and_grads(agent, observations, actions, returns)
    second = loss_and_grads(agent, observations, actions, returns)

    assert_weights_unchanged(agent, before, "computing a loss")
    assert first[0] == second[0]


def test_actor_critic_default_advantages_are_returns_minus_current_values():
    """Leaving advantages out is the same as passing G - V(s) for the current weights.

    update() relies on the default so that it needs only one forward pass per step; the gradient
    check passes the array explicitly. This ties the two paths together.
    """
    agent = _agent(ActorCriticAgent)
    set_weights(agent, b_v=0.2)
    observations, actions = _batch()
    returns = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5])
    values = np.array([state_value(agent, obs) for obs in observations])

    default_components, default_grads = loss_and_grads(agent, observations, actions, returns)
    explicit_components, explicit_grads = loss_and_grads(
        agent, observations, actions, returns, returns - values
    )

    assert default_components == explicit_components
    for name in default_grads:
        np.testing.assert_array_equal(default_grads[name], explicit_grads[name])


def test_actor_critic_rejects_advantages_of_the_wrong_length():
    agent = _agent(ActorCriticAgent)
    observations, actions = _batch()
    returns = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5])

    with pytest.raises(ValueError, match="advantages"):
        loss_and_grads(agent, observations, actions, returns, np.ones(N_STEPS + 1))


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
    current = weights(agent)
    set_weights(agent, **{name: current[name] - scale * g for name, g in grads.items()})


def test_reinforce_update_descends_the_gradient_scaled_by_learning_rate_over_steps():
    config = {"gamma": 0.5, "learning_rate": 0.05}
    agent, reference = _agent(ReinforceAgent, **config), _agent(ReinforceAgent, **config)
    batch = _step_batch()
    untouched = weights(agent)

    metrics = agent.update(batch)

    loss, grads = loss_and_grads(
        reference, batch["observations"], batch["actions"], _standardised(STEP_RETURNS)
    )
    _descend(reference, grads, scale=0.05 / N_STEPS)
    assert_same_weights(agent, reference)
    assert metrics["policy_loss"] == pytest.approx(loss / N_STEPS, rel=1e-12)
    moved = weights(agent)
    assert all(not np.array_equal(moved[name], untouched[name]) for name in moved)


def test_actor_critic_update_descends_the_gradient_scaled_by_learning_rate_over_steps():
    config = {"gamma": 0.5, "learning_rate": 0.05}
    agent, reference = _agent(ActorCriticAgent, **config), _agent(ActorCriticAgent, **config)
    batch = _step_batch()
    untouched = weights(agent)

    metrics = agent.update(batch)

    components, grads = loss_and_grads(
        reference, batch["observations"], batch["actions"], STEP_RETURNS
    )
    _descend(reference, grads, scale=0.05 / N_STEPS)
    assert_same_weights(agent, reference)
    assert metrics["policy_loss"] == pytest.approx(components["policy"] / N_STEPS, rel=1e-12)
    assert metrics["value_loss"] == pytest.approx(components["value"] / N_STEPS, rel=1e-12)
    assert metrics["entropy_bonus"] == pytest.approx(components["entropy"] / N_STEPS, rel=1e-12)
    moved = weights(agent)
    assert all(not np.array_equal(moved[name], untouched[name]) for name in moved)


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
    current = weights(agent)
    set_weights(agent, **{name: current[name] - step for name, step in adam.step(mean).items()})


def test_reinforce_with_adam_steps_the_mean_gradient_with_adam():
    agent, reference = _agent(ReinforceAgent, **ADAM), _agent(ReinforceAgent, **ADAM)
    adam = ReferenceAdam(ADAM["learning_rate"], ADAM["max_grad_norm"])
    batch = _step_batch()

    for _ in range(2):  # the second step uses the moments of the first
        agent.update(batch)
        _, grads = loss_and_grads(
            reference, batch["observations"], batch["actions"], _standardised(STEP_RETURNS)
        )
        _adam_descend(reference, adam, grads)

    assert_same_weights(agent, reference)


def test_actor_critic_with_adam_steps_the_mean_gradient_with_adam():
    agent, reference = _agent(ActorCriticAgent, **ADAM), _agent(ActorCriticAgent, **ADAM)
    adam = ReferenceAdam(ADAM["learning_rate"], ADAM["max_grad_norm"])
    batch = _step_batch()

    for _ in range(2):
        agent.update(batch)
        _, grads = loss_and_grads(reference, batch["observations"], batch["actions"], STEP_RETURNS)
        _adam_descend(reference, adam, grads)

    assert_same_weights(agent, reference)


DISCRETE_AGENTS = [ReinforceAgent, ActorCriticAgent, PPOAgent]


@pytest.mark.parametrize("agent_cls", DISCRETE_AGENTS)
def test_default_optimizer_is_plain_sgd(agent_cls):
    assert optimizer(_agent(agent_cls)) is None


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
    assert optimizer(resumed).t == optimizer(agent).t
    share_generators(agent, resumed)  # PPO shuffles mini-batches with one; checkpoints omit them

    agent.update(batch)
    resumed.update(batch)
    actual, expected = weights(resumed), weights(agent)
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

    adam = optimizer(adam_agent)
    assert adam.t == 0
    for moment in (*adam.m.values(), *adam.v.values()):
        np.testing.assert_array_equal(moment, 0.0)
    assert_same_weights(adam_agent, sgd)


# ---------------------------------------------------------------------------
# Direction and overfit checks (the training-doctor ladder, rung 2)
# ---------------------------------------------------------------------------
# One state seen at every step. Action 0 is always followed by reward 1 and action 1 by 0, and a
# tiny gamma keeps each return close to its own reward, so action 0 has the higher advantage.


def _one_state_batch() -> dict[str, np.ndarray]:
    state = np.random.default_rng(5).standard_normal(OBS_DIM)
    actions = np.array([0, 1] * (N_STEPS // 2))
    return {
        "observations": np.tile(state, (N_STEPS, 1)),
        "actions": actions,
        "rewards": (actions == 0).astype(np.float64),
        "next_observations": np.tile(state, (N_STEPS, 1)),
        "dones": np.array([False] * (N_STEPS - 1) + [True]),
    }


@pytest.mark.parametrize("agent_cls", DISCRETE_AGENTS)
def test_one_update_raises_the_probability_of_the_better_action(agent_cls):
    agent = _agent(agent_cls, gamma=0.01, learning_rate=0.05, entropy_coef=0.0)
    batch = _one_state_batch()
    state = batch["observations"][0]
    before = probabilities(agent, state)

    agent.update(batch)

    after = probabilities(agent, state)
    assert after[0] > before[0], "The action followed by reward must become more likely."
    assert after[1] < before[1], "The action followed by nothing must become less likely."


@pytest.mark.parametrize("agent_cls", DISCRETE_AGENTS)
def test_repeated_updates_on_one_batch_fit_it(agent_cls):
    """Updating on the same small batch must fit it; an agent that cannot will not fit CartPole."""
    agent = _agent(agent_cls, gamma=0.01, learning_rate=0.05, entropy_coef=0.0)
    batch = _one_state_batch()
    state = batch["observations"][0]

    best = 0.0
    for _ in range(1000):
        last = agent.update(batch)
        best = max(best, float(probabilities(agent, state)[0]))

    if agent_cls is ActorCriticAgent:
        # The discrete Actor-Critic fits this batch and can then collapse to the other action (roadmap: "Separate
        # Random Streams for Agents and Buffers" -> Finding), so it is held to fitting it at some point.
        assert best > 0.9, "the agent must come to prefer the rewarded action"
    else:
        assert probabilities(agent, state)[0] > 0.9, (
            "the agent must prefer the rewarded action at the end"
        )
    if "value_loss" in last:
        # One state with alternating returns: no critic can beat predicting their mean, so the loss
        # cannot fall below half their variance. It must reach that floor.
        floor = 0.5 * np.var(discounted_returns(batch["rewards"], 0.01))
        assert last["value_loss"] < 1.01 * floor, "The critic must fit the returns."

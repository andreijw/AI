"""Numerical behaviour shared by the discrete policy-gradient agents.

Two things every softmax-policy agent must get right, checked for REINFORCE, Actor-Critic and
PPO alike: the softmax stays finite for extreme logits, and the update really uses the agent's
own discount factor.
"""

import warnings

import numpy as np
import pytest

from rl_core.agents import ActorCriticAgent, PPOAgent, ReinforceAgent

POLICY_AGENTS = [ReinforceAgent, ActorCriticAgent, PPOAgent]
OBSERVATION = np.array([0.05, -0.1, 0.02, 0.08])


def _set_logits(agent, logits) -> None:
    """Make the policy head output exactly these logits for any observation.

    Zeroing the head's weights and setting its bias is the one reliable way to force a chosen
    logit vector: scaling the weights depends on the sign of whatever the seed produced, and a
    hidden layer that outputs zero would make the weights irrelevant.
    """
    weight, bias = ("_W2", "_b2") if isinstance(agent, ReinforceAgent) else ("_W_pi", "_b_pi")
    setattr(agent, weight, np.zeros_like(getattr(agent, weight)))
    setattr(agent, bias, np.asarray(logits, dtype=np.float64))


def _probabilities(agent, observation=OBSERVATION) -> np.ndarray:
    forward = agent._policy if isinstance(agent, ReinforceAgent) else agent._forward
    return np.asarray(forward(observation)[0])


def _make(agent_cls, action_dim: int = 2, **config):
    return agent_cls(observation_dim=4, action_dim=action_dim, config={"seed": 42, **config})


# ---------------------------------------------------------------------------
# Softmax stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent_cls", POLICY_AGENTS)
@pytest.mark.parametrize(
    "logits,expected",
    [
        ([1000.0, -1000.0], [1.0, 0.0]),  # exp(1000) overflows without the max subtraction
        ([1000.0, 1000.0], [0.5, 0.5]),  # inf / inf without it
        ([-1000.0, -1000.0], [0.5, 0.5]),  # 0 / 0 without it
        ([0.0, 0.0], [0.5, 0.5]),  # equal logits: uniform, and it must be normalised
    ],
)
def test_softmax_is_finite_and_normalised_for_extreme_logits(agent_cls, logits, expected):
    agent = _make(agent_cls)
    _set_logits(agent, logits)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an overflow or invalid-value warning is a failure
        probs = _probabilities(agent)
        actions = [agent.select_action(OBSERVATION, training=mode) for mode in (True, False)]

    assert np.all(np.isfinite(probs)), f"Probabilities are not finite: {probs}"
    np.testing.assert_allclose(probs, expected, rtol=0, atol=1e-12)
    assert all(0 <= action < 2 for action in actions)


@pytest.mark.parametrize("agent_cls", POLICY_AGENTS)
def test_one_logit_far_above_the_rest_takes_all_the_probability(agent_cls):
    agent = _make(agent_cls, action_dim=3)
    _set_logits(agent, [-5.0, 800.0, 3.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        probs = _probabilities(agent)
        greedy_action = agent.select_action(OBSERVATION, training=False)

    np.testing.assert_allclose(probs, [0.0, 1.0, 0.0], rtol=0, atol=1e-12)
    assert greedy_action == 1


# ---------------------------------------------------------------------------
# The update uses the agent's own discount factor
# ---------------------------------------------------------------------------


def _late_reward_batch() -> dict[str, np.ndarray]:
    """Six fixed steps with uneven rewards, the largest last.

    The discount cancels out of the update for a one-step episode, for all-zero rewards and
    for a reward on the first step only, so the batch has to avoid those shapes.
    """
    rng = np.random.default_rng(7)
    return {
        "observations": rng.standard_normal((6, 4)),
        "actions": np.array([0, 1, 1, 0, 1, 0]),
        "rewards": np.array([1.0, 0.0, 2.0, 1.0, 0.0, 3.0]),
        "next_observations": rng.standard_normal((6, 4)),
        "dones": np.array([False, False, False, False, False, True]),
    }


def _weights_after_update(agent_cls, gamma: float) -> dict[str, np.ndarray]:
    agent = _make(agent_cls, gamma=gamma)
    agent.update(_late_reward_batch())
    return {
        name: np.array(value, dtype=np.float64)
        for name, value in vars(agent).items()
        if name.startswith(("_W", "_b")) and isinstance(value, (np.ndarray, float))
    }


def _largest_difference(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> float:
    assert first.keys() == second.keys()
    return max(float(np.max(np.abs(first[name] - second[name]))) for name in first)


@pytest.mark.parametrize("agent_cls", POLICY_AGENTS)
def test_update_depends_on_the_agents_discount_factor(agent_cls):
    """A different gamma must lead to different weights, well beyond rounding noise."""
    short_sighted = _weights_after_update(agent_cls, gamma=0.5)
    far_sighted = _weights_after_update(agent_cls, gamma=0.99)

    difference = _largest_difference(short_sighted, far_sighted)
    assert difference > 1e-6, (
        f"{agent_cls.__name__} gave the same weights for gamma 0.5 and 0.99 "
        f"(largest difference {difference:.2e}): its update is ignoring gamma."
    )


@pytest.mark.parametrize("agent_cls", POLICY_AGENTS)
def test_update_is_identical_for_the_same_discount_factor(agent_cls):
    """Guards the test above: with gamma held fixed, nothing else makes the weights differ."""
    first = _weights_after_update(agent_cls, gamma=0.9)
    second = _weights_after_update(agent_cls, gamma=0.9)

    assert _largest_difference(first, second) == 0.0

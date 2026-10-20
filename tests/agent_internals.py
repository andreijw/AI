"""The one place an agent test may touch an agent's private state.

Behaviour tests assert through ``select_action``, the metrics ``update`` returns and ``save``/``load``.
Tests that check the maths directly (gradient checks, losses worked out by hand, an update replayed
against an independent reference) also need the weights, the forward pass and the loss-and-gradient
methods. They reach them only through the functions here, so a refactor that renames an attribute
changes this file and no test. ``tests/test_source_conventions.py`` holds the migrated test files to
that (roadmap: "Test Agents Through Their Public Behaviour").

Weights are keyed by the names the agent's checkpoint uses (``W1``, ``b1``, ``W2`` … for REINFORCE;
``W1``, ``b1``, ``W_pi``, ``b_pi``, ``W_v``, ``b_v`` for the discrete actor-critic agents; ``W1``, ``b1``,
``W_mu``, ``b_mu``, ``log_std``, ``W_v``, ``b_v`` for the Gaussian ones).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from rl_core.agents import (
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    PPOAgent,
    ReinforceAgent,
)
from rl_core.agents.actor_critic_base import ActorCriticBase
from rl_core.agents.continuous_actor_critic import ContinuousActorCriticBase
from rl_core.agents.optim import Adam

# ---------------------------------------------------------------------------
# Policy-gradient family: ReinforceAgent, ActorCriticAgent, PPOAgent (softmax policies) and
# ContinuousActorCriticAgent, ContinuousPPOAgent (Gaussian policies)
# ---------------------------------------------------------------------------

type PolicyAgent = ReinforceAgent | ActorCriticBase | ContinuousActorCriticBase
type DiscretePolicyAgent = ReinforceAgent | ActorCriticBase
type CriticAgent = ActorCriticBase | ContinuousActorCriticBase
type AnyPPOAgent = PPOAgent | ContinuousPPOAgent

REINFORCE_WEIGHTS = ("W1", "b1", "W2", "b2")
ACTOR_CRITIC_WEIGHTS = ("W1", "b1", "W_pi", "b_pi", "W_v", "b_v")
GAUSSIAN_WEIGHTS = ("W1", "b1", "W_mu", "b_mu", "log_std", "W_v", "b_v")
_FLOAT_WEIGHTS = {"b_v"}  # stored as a Python float, not an array


def weight_names(agent: PolicyAgent) -> tuple[str, ...]:
    """The agent's weight names, in checkpoint order."""
    if isinstance(agent, ReinforceAgent):
        return REINFORCE_WEIGHTS
    if isinstance(agent, ContinuousActorCriticBase):
        return GAUSSIAN_WEIGHTS
    return ACTOR_CRITIC_WEIGHTS


def weights(agent: PolicyAgent) -> dict[str, np.ndarray]:
    """A float64 copy of every weight, by name; ``b_v`` comes back as a 0-d array."""
    return {
        name: np.array(getattr(agent, f"_{name}"), dtype=np.float64, copy=True)
        for name in weight_names(agent)
    }


def set_weights(agent: PolicyAgent, **arrays: Any) -> None:
    """
    Overwrite named weights, for a test that needs a specific weight state.

    Raises:
        KeyError: For a name the agent has no weight of.
        ValueError: For a value whose shape differs from the weight it replaces.
    """
    names = weight_names(agent)
    for name, value in arrays.items():
        if name not in names:
            raise KeyError(f"{type(agent).__name__} has no weight {name!r}; it has {names}")
        if name in _FLOAT_WEIGHTS:
            if np.shape(value) != ():
                raise ValueError(f"{name} is a scalar, got shape {np.shape(value)}")
            setattr(agent, f"_{name}", float(value))
            continue
        current = getattr(agent, f"_{name}")
        replacement = np.array(value, dtype=np.float64, copy=True)
        if replacement.shape != current.shape:
            raise ValueError(f"{name} has shape {current.shape}, got {replacement.shape}")
        setattr(agent, f"_{name}", replacement)


def assert_same_weights(actual: Any, expected: Any, rtol: float = 1e-12) -> None:
    """Assert two agents hold the same weights, to ``rtol`` (plus 1e-15 absolute for zeros);
    ``rtol=0`` asks for bit-identical weights."""
    got, want = weights(actual), weights(expected)
    assert got.keys() == want.keys(), "The agents have different weights."
    atol = 0.0 if rtol == 0.0 else 1e-15
    for name in want:
        np.testing.assert_allclose(
            got[name], want[name], rtol=rtol, atol=atol, equal_nan=False, err_msg=f"{name} differs."
        )


def assert_weights_unchanged(agent: Any, before: dict[str, np.ndarray], when: str = "") -> None:
    """Assert every weight equals ``before`` (from :func:`weights`) exactly."""
    after = weights(agent)
    assert after.keys() == before.keys()
    for name in before:
        np.testing.assert_array_equal(after[name], before[name], err_msg=f"{name} moved {when}.")


def set_constant_critic(agent: CriticAgent, value: float) -> None:
    """Make the critic return ``value`` for every state: zero its output weights, set its bias."""
    set_weights(agent, W_v=np.zeros_like(weights(agent)["W_v"]), b_v=value)


def gradient_check_parameters(
    agent: PolicyAgent,
) -> tuple[dict[str, np.ndarray], Callable[[], None]]:
    """
    The live weight arrays for ``tests.helpers.assert_gradients_match``, and a ``sync`` function.

    ``b_v`` is a Python float, which cannot be perturbed in place, so a one-element array stands
    in for it. The loss closure calls ``sync()`` first to copy that array back onto the agent.
    """
    names = weight_names(agent)
    boxed = {name: np.array([getattr(agent, f"_{name}")]) for name in _FLOAT_WEIGHTS & set(names)}
    params = {name: boxed.get(name, getattr(agent, f"_{name}")) for name in names}

    def sync() -> None:
        for name, box in boxed.items():
            setattr(agent, f"_{name}", float(box[0]))

    return params, sync


def boxed_gradients(grads: dict[str, Any]) -> dict[str, np.ndarray]:
    """Gradients shaped like :func:`gradient_check_parameters`: a float's becomes a 1-element array."""
    return {name: np.atleast_1d(np.asarray(g, dtype=np.float64)) for name, g in grads.items()}


def probabilities(agent: DiscretePolicyAgent, observation: np.ndarray) -> np.ndarray:
    """The policy's action probabilities for one observation."""
    if isinstance(agent, ReinforceAgent):
        return agent._policy(observation)[0]
    return agent._forward(observation)[0]


def hidden_activations(agent: PolicyAgent, observation: np.ndarray) -> np.ndarray:
    """The trunk's ReLU output for one observation."""
    if isinstance(agent, ReinforceAgent):
        return agent._policy(observation)[1]
    if isinstance(agent, ContinuousActorCriticBase):
        return agent._forward(observation)[3]
    return agent._forward(observation)[2]


def state_value(agent: CriticAgent, observation: np.ndarray) -> Any:
    """The critic's V(s), as the forward pass returns it (not converted): a float for one
    observation, and for a Gaussian agent an array for a batch."""
    if isinstance(agent, ContinuousActorCriticBase):
        return agent._forward(observation)[2]
    return agent._forward(observation)[1]


def gaussian_forward(
    agent: ContinuousActorCriticBase, states: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Any, np.ndarray]:
    """``(mean, log_std, value, hidden activations)`` for one state or a batch."""
    return agent._forward(states)


def log_prob(
    agent: ContinuousActorCriticBase, actions: np.ndarray, mu: np.ndarray, log_std: np.ndarray
) -> np.ndarray:
    """The Gaussian log-density of ``actions``, summed over action dimensions."""
    return agent._log_prob(actions, mu, log_std)


def chunk_targets(
    agent: ContinuousActorCriticAgent,
    states: np.ndarray,
    rewards: np.ndarray,
    bootstrap_obs: np.ndarray | None,
    terminal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """The continuous Actor-Critic's ``(advantages, returns)`` for one n-step chunk."""
    return agent._chunk_targets(states, rewards, bootstrap_obs, terminal)


def rollout_targets(
    agent: ContinuousPPOAgent, states: np.ndarray, rewards: np.ndarray, dones: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous PPO's ``(advantages, returns)`` for a rollout, before normalising."""
    return agent._rollout_targets(states, rewards, dones)


def forward_batch(
    agent: ActorCriticBase, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(probabilities, values, hidden activations)`` for a batch of observations."""
    return agent._forward_batch(observations)


def loss_and_grads(
    agent: ReinforceAgent | ActorCriticAgent | ContinuousActorCriticAgent,
    observations: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    advantages: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    One episode's loss and gradients, summed over steps, with nothing applied.

    For REINFORCE ``returns`` are the return weights it scales log-probabilities by and the loss
    is a float; for the actor-critic agents the loss is a dict of components and ``advantages``
    are optional.
    """
    if isinstance(agent, ReinforceAgent):
        if advantages is not None:
            raise TypeError("REINFORCE takes no advantages")
        return agent._loss_and_grads(observations, actions, returns)
    return agent._loss_and_grads(observations, actions, returns, advantages)


def ppo_minibatch_loss_and_grads(
    agent: AnyPPOAgent,
    observations: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    advantages: np.ndarray,
    old_log_probs: np.ndarray,
    clip_epsilon: float | None = None,
) -> tuple[Any, ...]:
    """PPO's mean loss components and gradients for one mini-batch; the discrete agent also
    returns its clipped-step count."""
    return agent._minibatch_loss_and_grads(
        observations, actions, returns, advantages, old_log_probs, clip_epsilon
    )


def adam_step(agent: ContinuousActorCriticBase, grads: dict[str, Any], lr: float) -> None:
    """Apply one step of a Gaussian agent's own Adam to its weights."""
    agent._adam_step(grads, lr)


def _spy_on(monkeypatch: Any, agent: Any, method: str, spy: Callable) -> None:
    original = getattr(agent, method)
    monkeypatch.setattr(agent, method, lambda *a, **kw: spy(original, *a, **kw))


def spy_on_ppo_minibatches(monkeypatch: Any, agent: AnyPPOAgent, spy: Callable) -> None:
    """Replace the agent's mini-batch loss with ``spy(original, *args, **kwargs)`` for one test."""
    _spy_on(monkeypatch, agent, "_minibatch_loss_and_grads", spy)


def spy_on_adam_step(monkeypatch: Any, agent: ContinuousActorCriticBase, spy: Callable) -> None:
    """Replace a Gaussian agent's Adam step with ``spy(original, grads, lr)`` for one test."""
    _spy_on(monkeypatch, agent, "_adam_step", spy)


def spy_on_chunk_targets(
    monkeypatch: Any, agent: ContinuousActorCriticAgent, spy: Callable
) -> None:
    """Replace the continuous Actor-Critic's chunk targets with
    ``spy(original, states, rewards, bootstrap_obs, terminal)`` for one test."""
    _spy_on(monkeypatch, agent, "_chunk_targets", spy)


def spy_on_loss_and_grads(
    monkeypatch: Any, agent: ActorCriticAgent | ContinuousActorCriticAgent, spy: Callable
) -> None:
    """Replace the agent's loss with ``spy(original, states, actions, returns, advantages=None)``."""
    _spy_on(monkeypatch, agent, "_loss_and_grads", spy)


def ppo_rollout_episode(
    agent: AnyPPOAgent, batch: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """One episode as PPO stores it in its rollout: ``(observations, actions, rewards, dones)``."""
    return agent._rollout_episode(batch)


def optimizer(agent: PolicyAgent) -> Adam | None:
    """The agent's Adam, or None with plain SGD."""
    return agent._optimizer


def shuffle_generator(agent: AnyPPOAgent | ContinuousActorCriticAgent) -> np.random.Generator:
    """A copy of the agent's mini-batch shuffle generator in its current state."""
    return copy.deepcopy(agent._shuffle_rng)


def replace_shuffle_generator(
    agent: AnyPPOAgent | ContinuousActorCriticAgent, generator: np.random.Generator
) -> None:
    """Make the agent shuffle mini-batches with ``generator`` from now on."""
    agent._shuffle_rng = generator


def share_generators(source: Any, target: Any) -> None:
    """
    Give ``target`` copies of every generator ``source`` holds, in their current state.

    Checkpoints do not store generators, so an exact-resume test shares them to put only what
    the checkpoint holds under test.
    """
    for name, value in vars(source).items():
        if isinstance(value, np.random.Generator):
            setattr(target, name, copy.deepcopy(value))

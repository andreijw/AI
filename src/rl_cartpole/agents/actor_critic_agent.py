"""Actor-Critic (A2C) agent for CartPole."""

from typing import Any, Dict

import numpy as np

from .actor_critic_base import ActorCriticBase


class ActorCriticAgent(ActorCriticBase):
    """
    Actor-Critic agent (A2C style) for CartPole.

    Uses a two-headed MLP with a shared trunk: the *actor* head outputs a
    softmax policy over discrete actions while the *critic* head estimates
    the state-value function V(s).  Policy parameters are updated with a
    REINFORCE-style gradient scaled by the advantage

        A_t = G_t - V(s_t)

    where G_t is the discounted Monte-Carlo return and V(s_t) is the
    current critic estimate.  The critic is trained via mean-squared-error
    regression towards G_t.  An optional entropy bonus encourages exploration.

    Compared with the plain REINFORCE agent this approach:

    * Replaces the scalar mean-return baseline with a *learned*, state-
      dependent baseline, substantially reducing gradient variance.
    * Trains a value function that can be reused for downstream extensions
      (e.g. GAE, PPO).

    References:
        Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement
        Learning. ICML 2016.

    Config keys (all optional):
        learning_rate (float): SGD step size. Default: 1e-3.
        gamma         (float): Discount factor ∈ (0, 1]. Default: 0.99.
        hidden_dim    (int):   Shared hidden layer width. Default: 128.
        value_coef    (float): Critic-loss weight. Default: 0.5.
        entropy_coef  (float): Entropy-bonus weight (encourages exploration).
                               Default: 0.01.
        seed          (int):   RNG seed for weight initialisation. Default: None.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialise the Actor-Critic agent.

        Args:
            observation_dim: Dimension of the observation vector.
            action_dim:      Number of discrete actions.
            config:          Hyperparameter dictionary (see class docstring).
        """
        super().__init__(observation_dim, action_dim, config)

        self.learning_rate: float = float(config.get("learning_rate", 1e-3))

        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update actor and critic parameters using the collected trajectory.

        The actor is updated with advantage-weighted policy gradients plus an
        entropy bonus; the critic is updated via MSE regression towards
        Monte-Carlo returns, scaled by ``value_coef``.

        The returned metrics report each component of the objective separately
        so that callers can monitor them independently:

        * ``policy_loss``   — mean ``-log π(a|s) · A_t`` (advantage term only,
                              before entropy regularisation).
        * ``value_loss``    — mean unscaled critic MSE ``0.5·(V(s)−G)²``
                              (before ``value_coef`` scaling).
        * ``entropy_bonus`` — mean policy entropy ``H(π(·|s))`` (higher is
                              more exploratory; weighted by ``entropy_coef``
                              in the actual parameter update).

        Args:
            batch: Dictionary with keys:
                   - "observations": array of shape (T, observation_dim)
                   - "actions":      integer array of shape (T,)
                   - "rewards":      float array of shape (T,)

        Returns:
            Dictionary with keys ``"policy_loss"``, ``"value_loss"``, and
            ``"entropy_bonus"``.
        """
        observations, actions, rewards, n_steps = self._parse_trajectory(batch)
        if n_steps == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_bonus": 0.0}

        # Monte-Carlo returns (same as REINFORCE, no normalisation here so the
        # value function retains scale information)
        returns = self._compute_returns(rewards)

        grads = self._zero_ac_gradients()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for obs, action, g in zip(observations, actions, returns):
            probs, value, h = self._forward(obs)

            # Advantage estimate: how much better this action was than expected
            advantage = g - value

            # Actor loss: -log π(a|s) * advantage  (reported separately)
            log_prob_a = np.log(np.clip(probs[action], 1e-8, 1.0))
            total_policy_loss += -log_prob_a * advantage

            # Critic loss: 0.5 * (V(s) - G)^2  (reported unscaled)
            total_value_loss += 0.5 * (value - g) ** 2

            # Entropy: H = -Σ_a π(a) log π(a)  (reported separately)
            log_probs = np.log(np.clip(probs, 1e-8, 1.0))
            entropy = -float(np.sum(probs * log_probs))
            total_entropy += entropy

            # Gradient of total loss w.r.t. policy logits (gradient-descent direction):
            #   d(policy_loss)/d(z_j) = advantage * (π(j) - 1[j==a])
            #   d(-entropy_coef * H)/d(z_j) = entropy_coef * π(j) * (log π(j) + H)
            d_pi_logits = advantage * probs.copy()
            d_pi_logits[action] -= advantage
            d_pi_logits += self.entropy_coef * probs * (log_probs + entropy)

            # Gradient of total loss w.r.t. critic output:
            #   d(value_coef * 0.5*(V-G)^2)/d(V) = value_coef * (V - G)
            d_v_out = self.value_coef * (value - g)

            self._accumulate_ac_gradients(grads, obs, h, d_pi_logits, d_v_out)

        # SGD update averaged over episode length
        self._apply_ac_gradients(grads, self.learning_rate / n_steps)

        return {
            "policy_loss": float(total_policy_loss / n_steps),
            "value_loss": float(total_value_loss / n_steps),
            "entropy_bonus": float(total_entropy / n_steps),
        }

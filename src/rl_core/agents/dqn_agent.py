"""Deep Q-Network (DQN) agent for CartPole."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from rl_core.seeding import AGENT_EXPLORATION_STREAM, AGENT_WEIGHTS_STREAM, derive_rng

from .base_agent import BaseAgent
from .config_values import optional_positive_int, positive_int
from .optim import Adam, anneal_fraction, anneal_steps_from_config, optimizer_from_config
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

# Settings of prioritised replay, refused unless prioritized_replay is on.
PRIORITY_KEYS = ("priority_alpha", "priority_beta_start", "priority_epsilon", "priority_beta_steps")


def _optional_real(config: dict[str, Any], key: str) -> float | None:
    """``config[key]`` as a float, or None if unset, rejecting booleans, strings and NaN."""
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | np.number) or np.isnan(value):
        raise ValueError(f"{key} must be a number, got {value!r}")
    return float(value)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent with experience replay and target network.

    Approximates state-action values Q(s, a) using a multi-layer perceptron.
    Trained via off-policy Bellman temporal-difference error minimization:
        L = 0.5 * (Q_online(s, a) - (r + (1 - done) * γ * max_a' Q_target(s', a')))^2

    Features:
    * Separate online and target Q-networks for stable gradient updates.
    * Epsilon-greedy action selection, decayed once per update call (``epsilon_decay``) or
      linearly per training action (``exploration_steps``).
    * Bootstrapping through time-limit cuts: only a true termination ends the Bellman target.
    * Plain SGD with an element-wise clip (``grad_clip_value``), or Adam with an optional
      global-norm clip (``optimizer: adam``, ``max_grad_norm``).
    * An optional linear learning-rate decay to zero over ``anneal_steps`` transitions, as PPO's:
      each update's step size is ``learning_rate`` times ``anneal_fraction`` of the transitions
      given to ``update`` before it, warm-up included. Only the step is scaled; Adam's moments see
      the full gradient. With an episode budget the decay runs over ``anneal_steps`` as set, and a
      checkpoint saved before the option existed resumes the decay from the start.
    * Ring-buffered experience replay for uncorrelated mini-batch updates, uniform by default or
      proportional prioritised replay (Schaul et al. 2016) with ``prioritized_replay: true``:
      transitions are drawn in proportion to ``(|δ| + priority_epsilon)^priority_alpha`` from
      their last TD error δ, and the loss weights each sample by its importance weight, whose
      exponent beta rises linearly from ``priority_beta_start`` to 1 over
      ``priority_beta_steps`` transitions (``anneal_steps`` when unset). Each gradient step writes
      its TD errors back as the sampled transitions' priorities.
    * Hard target syncs every ``target_update_frequency`` gradient steps, or Polyak averaging
      with ``tau`` in (0, 1] after every step (``tau`` 1 copies the network; the frequency is then
      unused).
    * Settings validated at construction: positive integer sizes and frequencies, ``tau`` in
      (0, 1], ``min_buffer_size >= batch_size`` and ``buffer_capacity >= min_buffer_size``.
    """

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        """
        Initialise the DQN agent.

        Args:
            observation_dim: Dimension of observation vectors.
            action_dim: Number of discrete actions.
            config: Hyperparameter dictionary.
        """
        super().__init__(observation_dim, action_dim, config)

        self.learning_rate: float = float(config.get("learning_rate", 1e-3))
        self.gamma: float = float(config.get("gamma", 0.99))
        self.hidden_dim: int = positive_int(config, "hidden_dim", 128)
        self.buffer_capacity: int = positive_int(config, "buffer_capacity", 10000)
        self.batch_size: int = positive_int(config, "batch_size", 64)
        self.min_buffer_size: int = positive_int(config, "min_buffer_size", self.batch_size)
        self.epsilon_start: float = float(config.get("epsilon_start", 1.0))
        self.epsilon_end: float = float(config.get("epsilon_end", 0.02))
        self.epsilon_decay: float = float(config.get("epsilon_decay", 0.995))
        self.target_update_frequency: int = positive_int(config, "target_update_frequency", 100)
        self.tau: float | None = _optional_real(config, "tau")
        self.grad_clip_value: float = float(config.get("grad_clip_value", 10.0))
        self.exploration_steps: int | None = self._read_exploration_steps(config)
        self.updates_per_step: float | None = _optional_real(config, "updates_per_step")
        self.max_updates_per_call: int = positive_int(config, "max_updates_per_call", 64)
        self.anneal_steps: int | None = anneal_steps_from_config(config)
        self._validate_dims(observation_dim, action_dim)
        self._validate_rates()
        self._validate_buffer_and_optimizer(config)

        self.epsilon: float = self.epsilon_start
        self._update_counter: int = 0
        self._exploration_step: int = 0
        self._steps_trained: int = 0
        self._explore_rng: np.random.Generator = derive_rng(
            config.get("seed"), AGENT_EXPLORATION_STREAM
        )
        self._init_q_networks(derive_rng(config.get("seed"), AGENT_WEIGHTS_STREAM))
        # The same object as replay_buffer when prioritised replay is on, typed for its methods.
        self._prioritized: PrioritizedReplayBuffer | None = self._read_prioritized_buffer(config)
        self.replay_buffer: ReplayBuffer = (
            self._prioritized
            if self._prioritized is not None
            else ReplayBuffer(
                capacity=self.buffer_capacity,
                observation_dim=self.observation_dim,
                seed=config.get("seed"),
            )
        )
        self._optimizer: Adam | None = optimizer_from_config(config, self._parameters())

    def _validate_rates(self) -> None:
        """
        Check the rates and coefficients read in ``__init__``, each error naming its key.

        Raises:
            ValueError: For the first one out of range.
        """
        if self.tau is not None and not 0.0 < self.tau <= 1.0:
            raise ValueError(
                f"tau must be in (0, 1] (1 copies the target every gradient step; for periodic hard "
                f"syncs leave tau unset and use target_update_frequency), got {self.tau!r}"
            )
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma!r}")
        if not (0.0 <= self.epsilon_start <= 1.0):
            raise ValueError(f"epsilon_start must be in [0, 1], got {self.epsilon_start!r}")
        if not (0.0 <= self.epsilon_end <= 1.0):
            raise ValueError(f"epsilon_end must be in [0, 1], got {self.epsilon_end!r}")
        if not (0.0 < self.epsilon_decay <= 1.0):
            raise ValueError(f"epsilon_decay must be in (0, 1], got {self.epsilon_decay!r}")
        if self.grad_clip_value <= 0.0:
            raise ValueError(f"grad_clip_value must be positive, got {self.grad_clip_value!r}")
        if self.updates_per_step is not None and self.updates_per_step <= 0:
            raise ValueError(
                f"updates_per_step must be positive (leave it unset for one update per call), "
                f"got {self.updates_per_step!r}"
            )

    def _validate_buffer_and_optimizer(self, config: dict[str, Any]) -> None:
        """
        Check that the replay buffer can warm up and fill a batch, and that a clip setting suits
        the optimizer.

        Raises:
            ValueError: For the first inconsistent pair of settings.
        """
        if self.min_buffer_size < self.batch_size:
            raise ValueError(
                f"min_buffer_size ({self.min_buffer_size}) must be at least batch_size "
                f"({self.batch_size}): the first sample needs a full batch"
            )
        if self.buffer_capacity < self.min_buffer_size:
            raise ValueError(
                f"buffer_capacity ({self.buffer_capacity}) must be at least min_buffer_size "
                f"({self.min_buffer_size}), or the buffer never warms up and the agent never trains"
            )
        if str(config.get("optimizer", "sgd")).lower() != "sgd" and "grad_clip_value" in config:
            raise ValueError(
                "grad_clip_value is the element-wise clip of plain SGD; "
                "with Adam, clip the global norm with max_grad_norm"
            )

    def _read_prioritized_buffer(self, config: dict[str, Any]) -> PrioritizedReplayBuffer | None:
        """
        The prioritised buffer ``prioritized_replay`` asks for, or None; sets
        ``priority_beta_start`` and ``priority_beta_steps``.

        Raises:
            ValueError: Naming the key, for a ``priority_*`` setting without
                ``prioritized_replay``, one out of range, or no beta schedule (neither
                ``priority_beta_steps`` nor ``anneal_steps``).
        """
        enabled = config.get("prioritized_replay", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"prioritized_replay must be true or false, got {enabled!r}")
        self.priority_beta_start: float | None = None
        self.priority_beta_steps: int | None = None
        if not enabled:
            for key in PRIORITY_KEYS:
                if config.get(key) is not None:
                    raise ValueError(f"{key} is set, but prioritized_replay is not")
            return None
        alpha = _optional_real(config, "priority_alpha")
        alpha = 0.6 if alpha is None else alpha
        epsilon = _optional_real(config, "priority_epsilon")
        epsilon = 1e-6 if epsilon is None else epsilon
        beta_start = _optional_real(config, "priority_beta_start")
        beta_start = 0.4 if beta_start is None else beta_start
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"priority_alpha must be in [0, 1], got {alpha!r}")
        if not epsilon > 0.0:
            raise ValueError(f"priority_epsilon must be positive, got {epsilon!r}")
        if not 0.0 <= beta_start <= 1.0:
            raise ValueError(f"priority_beta_start must be in [0, 1], got {beta_start!r}")
        beta_steps = optional_positive_int(config, "priority_beta_steps") or self.anneal_steps
        if beta_steps is None:
            raise ValueError(
                "priority_beta_steps must be set when anneal_steps is not: the transitions over "
                "which the importance-sampling exponent rises to 1"
            )
        self.priority_beta_start, self.priority_beta_steps = beta_start, beta_steps
        return PrioritizedReplayBuffer(
            capacity=self.buffer_capacity,
            observation_dim=self.observation_dim,
            seed=config.get("seed"),
            alpha=alpha,
            epsilon=epsilon,
        )

    def _init_q_networks(self, weights_rng: np.random.Generator) -> None:
        """He-initialised online Q-network (W1 drawn before W2) and an identical target copy."""
        self._W1: npt.NDArray[np.float64] = weights_rng.standard_normal(
            (self.observation_dim, self.hidden_dim)
        ) * np.sqrt(2.0 / self.observation_dim)
        self._b1: npt.NDArray[np.float64] = np.zeros(self.hidden_dim)
        self._W2: npt.NDArray[np.float64] = weights_rng.standard_normal(
            (self.hidden_dim, self.action_dim)
        ) * np.sqrt(2.0 / self.hidden_dim)
        self._b2: npt.NDArray[np.float64] = np.zeros(self.action_dim)
        self._sync_target()

    def _sync_target(self) -> None:
        """Hard sync: the target network becomes a copy (sharing no memory) of the online one."""
        self._target_W1: npt.NDArray[np.float64] = np.copy(self._W1)
        self._target_b1: npt.NDArray[np.float64] = np.copy(self._b1)
        self._target_W2: npt.NDArray[np.float64] = np.copy(self._W2)
        self._target_b2: npt.NDArray[np.float64] = np.copy(self._b2)

    @staticmethod
    def _read_exploration_steps(config: dict[str, Any]) -> int | None:
        """
        Read ``exploration_steps``: decay epsilon linearly over this many training actions.

        Raises:
            ValueError: If it is not a positive integer, or if ``epsilon_decay`` is set too.
        """
        if config.get("exploration_steps") is None:
            return None
        value = positive_int(config, "exploration_steps", 0)
        if config.get("epsilon_decay") is not None:
            raise ValueError(
                "Set epsilon_decay (once per update call) or exploration_steps "
                "(linear per training action), not both"
            )
        return value

    def _parameters(self) -> dict[str, npt.NDArray[np.float64]]:
        """The online network's parameters by gradient name."""
        return {"W1": self._W1, "b1": self._b1, "W2": self._W2, "b2": self._b2}

    def _scheduled_epsilon(self) -> float:
        """Epsilon after ``_exploration_step`` training actions on the linear schedule."""
        remaining = anneal_fraction(self._exploration_step, self.exploration_steps)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * remaining

    def _forward_q(self, obs: np.ndarray, target: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute forward pass Q-values for an observation or batch of observations.

        Args:
            obs: Observation vector (observation_dim,) or batch (N, observation_dim).
            target: If True, uses the target network parameters; otherwise online network.

        Returns:
            Tuple of (q_values, hidden_activations).
        """
        w1 = self._target_W1 if target else self._W1
        b1 = self._target_b1 if target else self._b1
        w2 = self._target_W2 if target else self._W2
        b2 = self._target_b2 if target else self._b2

        h = np.maximum(0.0, obs @ w1 + b1)
        q = h @ w2 + b2
        return q, h

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action given an observation.

        During training, applies epsilon-greedy action selection; with ``exploration_steps``
        set, each training action then advances the linear epsilon schedule by one step.
        During evaluation, acts greedily (argmax Q) and leaves epsilon alone.

        Args:
            observation: Current observation vector.
            training: Whether in training mode.

        Returns:
            Selected action index.
        """
        if training and self._explore_rng.random() < self.epsilon:
            action = int(self._explore_rng.integers(0, self.action_dim))
        else:
            q_vals, _ = self._forward_q(observation, target=False)
            action = int(np.argmax(q_vals))

        if training and self.exploration_steps is not None:
            self._exploration_step += 1
            self.epsilon = self._scheduled_epsilon()
        return action

    def _compute_bellman_targets(
        self,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Bellman temporal-difference target values:
            y = r + (1 - done) * γ * max_a' Q_target(s', a')

        Args:
            rewards: Batch of scalar rewards.
            next_obs: Batch of next observation vectors.
            dones: Batch of termination booleans.

        Returns:
            Array of target values of shape (batch_size,).
        """
        next_q, _ = self._forward_q(next_obs, target=True)
        max_next_q = np.max(next_q, axis=-1)
        not_done = 1.0 - dones.astype(np.float64)
        return rewards + not_done * self.gamma * max_next_q

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Update the Q-network using collected transitions.

        Ingests transitions into the replay buffer, samples a mini-batch,
        computes the Bellman loss, and applies gradient descent.

        A transition is stored as terminal only if it ended the episode for real: the batch's
        ``terminated`` flags when present (the trainer sends them), else its ``dones``. A cut at
        the time limit therefore still bootstraps from the next state.

        Args:
            batch: Trajectory dictionary with 'observations', 'actions', 'rewards',
                   and optionally 'next_observations', 'terminated' and 'dones'.

        Returns:
            Dictionary with training metrics: 'loss', 'epsilon', 'mean_q'; only 'epsilon' when no
            update happened: during warm-up (the buffer below ``min_buffer_size``), or when the
            batch stored no transition (empty, or without ``next_observations`` or either flag).
            With ``anneal_steps`` set, 'anneal_fraction' too: the share of ``learning_rate`` this
            call's steps used. With prioritised replay, 'priority_beta' (this call's exponent) and
            'mean_is_weight' (the mean importance weight of its samples) too.
        """
        fraction = anneal_fraction(self._steps_trained, self.anneal_steps)
        beta = self._priority_beta()
        n_stored = self.replay_buffer.add_transitions(batch)
        self._steps_trained += n_stored
        annealing = {} if self.anneal_steps is None else {"anneal_fraction": float(fraction)}

        if n_stored == 0 or not self.replay_buffer.can_sample(self.min_buffer_size):
            # No update, so no loss or Q-value for the log to average in, and no epsilon decay.
            return {"epsilon": self.epsilon, **annealing}

        if self.updates_per_step is not None:
            num_updates = min(
                self.max_updates_per_call, max(1, int(n_stored * self.updates_per_step))
            )
        else:
            num_updates = 1

        total_loss = 0.0
        total_q = 0.0
        total_weight = 0.0

        for _ in range(num_updates):
            # Sample mini-batch from experience replay, by priority when prioritised
            if self._prioritized is not None:
                samples = self._prioritized.sample(self.batch_size, beta=beta)
                sample_weights = samples["weights"]
            else:
                samples = self.replay_buffer.sample(self.batch_size)
                sample_weights = None
            obs = samples["observations"]
            actions = samples["actions"]
            rewards = samples["rewards"]
            next_obs = samples["next_observations"]
            dones = samples["dones"]

            # 1. Compute Bellman targets
            targets = self._compute_bellman_targets(rewards, next_obs, dones)

            # 2. TD loss and its gradients for the online network
            components, grads, td_error = self._loss_and_grads(
                obs, actions, targets, sample_weights
            )

            # 3. Clipped gradient descent step; the step's TD errors become the priorities
            self._apply_gradients(grads, self.learning_rate * fraction)
            if self._prioritized is not None:
                self._prioritized.update_priorities(samples["indices"], td_error)
                total_weight += float(np.mean(samples["weights"]))

            # 4. Target network synchronization
            self._update_counter += 1
            if self.tau is not None:
                self._target_W1 = self.tau * self._W1 + (1.0 - self.tau) * self._target_W1
                self._target_b1 = self.tau * self._b1 + (1.0 - self.tau) * self._target_b1
                self._target_W2 = self.tau * self._W2 + (1.0 - self.tau) * self._target_W2
                self._target_b2 = self.tau * self._b2 + (1.0 - self.tau) * self._target_b2
            elif self._update_counter % self.target_update_frequency == 0:
                self._sync_target()

            total_loss += components["loss"]
            total_q += components["mean_q"]

        # 5. Epsilon decay (the linear schedule advances in select_action instead)
        if self.exploration_steps is None:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        prioritized = (
            {}
            if self._prioritized is None
            else {"priority_beta": beta, "mean_is_weight": total_weight / num_updates}
        )
        return {
            "loss": total_loss / num_updates,
            "epsilon": float(self.epsilon),
            "mean_q": total_q / num_updates,
            **annealing,
            **prioritized,
        }

    def _priority_beta(self) -> float:
        """The importance-sampling exponent after ``_steps_trained`` transitions: linear from
        ``priority_beta_start`` to 1 over ``priority_beta_steps`` (1 without prioritised replay)."""
        if self.priority_beta_start is None:
            return 1.0
        remaining = anneal_fraction(self._steps_trained, self.priority_beta_steps)
        return 1.0 - (1.0 - self.priority_beta_start) * remaining

    def _loss_and_grads(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> tuple[dict[str, float], dict[str, np.ndarray], np.ndarray]:
        """
        Compute the TD loss and its gradients for one mini-batch, changing no state.

        The loss is ``mean(w·½·(Q(s, a) − y)²)`` over the mini-batch, for the online network,
        with ``w`` the importance weights (1 when ``weights`` is None). The targets ``y`` and the
        weights are held constant when differentiating.

        Args:
            obs:     Array of shape (B, observation_dim).
            actions: Integer array of shape (B,): the action taken in each state.
            targets: Array of shape (B,): the Bellman target of each transition.
            weights: Array of shape (B,): each sample's importance weight, or None.

        Returns:
            Tuple of (components, gradients, TD errors). Components holds ``loss`` and
            ``mean_q``, the mean of the predicted Q-values. Gradients are those of ``loss``, keyed
            ``W1``, ``b1``, ``W2``, ``b2``, and are not clipped; :meth:`_apply_gradients` clips
            them. The TD errors ``Q(s, a) − y`` have shape (B,).
        """
        b_size = len(actions)

        # Forward pass online network
        q_vals, h = self._forward_q(obs, target=False)
        pred_q = q_vals[np.arange(b_size), actions]

        # TD error and MSE loss: 0.5 * (pred_q - targets)^2
        td_error = pred_q - targets
        weighted = td_error if weights is None else weights * td_error
        loss = float(np.mean(0.5 * (weighted * td_error)))

        # Backward pass
        d_q = np.zeros_like(q_vals)
        d_q[np.arange(b_size), actions] = weighted

        grad_w2 = (h.T @ d_q) / b_size
        grad_b2 = np.mean(d_q, axis=0)

        d_h = d_q @ self._W2.T
        d_pre_h = d_h * (h > 0.0)

        grad_w1 = (obs.T @ d_pre_h) / b_size
        grad_b1 = np.mean(d_pre_h, axis=0)

        components = {"loss": loss, "mean_q": float(np.mean(pred_q))}
        grads = {"W1": grad_w1, "b1": grad_b1, "W2": grad_w2, "b2": grad_b2}
        return components, grads, td_error

    def _apply_gradients(
        self, grads: dict[str, np.ndarray], learning_rate: float | None = None
    ) -> None:
        """
        Take one optimizer step on the online network, of size ``learning_rate`` (the configured
        ``learning_rate`` when None).

        With Adam, the step comes from the shared optimizer (global-norm clip ``max_grad_norm``
        if set). With plain SGD, each gradient element is clipped to
        ``[-grad_clip_value, grad_clip_value]`` before the step. The arrays in ``grads`` are not
        modified.

        Args:
            grads: Gradients keyed ``W1``, ``b1``, ``W2``, ``b2``, as returned by
                   :meth:`_loss_and_grads`.
            learning_rate: Step size of this step (the decayed rate when annealing).
        """
        lr = self.learning_rate if learning_rate is None else learning_rate
        if self._optimizer is not None:
            steps = self._optimizer.step(grads, lr)
            for name, parameter in self._parameters().items():
                parameter -= steps[name]  # in place: the attribute holds this same array
            return
        limit = self.grad_clip_value
        self._W1 -= lr * np.clip(grads["W1"], -limit, limit)
        self._b1 -= lr * np.clip(grads["b1"], -limit, limit)
        self._W2 -= lr * np.clip(grads["W2"], -limit, limit)
        self._b2 -= lr * np.clip(grads["b2"], -limit, limit)

    def _checkpoint_arrays(self) -> dict[str, Any]:
        """
        Everything training needs to resume beyond the generators: both networks, epsilon, the
        exploration, target-sync and transitions-trained counters, the optimizer's state and the
        replay buffer.
        """
        arrays: dict[str, Any] = {
            "W1": self._W1,
            "b1": self._b1,
            "W2": self._W2,
            "b2": self._b2,
            "target_W1": self._target_W1,
            "target_b1": self._target_b1,
            "target_W2": self._target_W2,
            "target_b2": self._target_b2,
            "epsilon": np.array(self.epsilon),
            "exploration_step": np.array(self._exploration_step),
            "update_counter": np.array(self._update_counter),
            "steps_trained": np.array(self._steps_trained),
            **self.replay_buffer.checkpoint_state(),
        }
        if self._optimizer is not None:
            arrays.update(self._optimizer.state())
        return arrays

    def _load_checkpoint_arrays(self, data: Mapping[str, Any]) -> None:
        """
        Restore a checkpoint, or an older one.

        An older checkpoint has no counters, optimizer state or replay buffer: the target-sync
        counter is left as it is, Adam starts from zero moments, the buffer is emptied, and with
        ``exploration_steps`` the exploration step is rebuilt from the saved epsilon so that
        exploration does not restart. Without the transitions-trained counter the learning-rate
        decay starts again from the full rate.
        """
        self.replay_buffer.restore(data)
        self._W1 = data["W1"]
        self._b1 = data["b1"]
        self._W2 = data["W2"]
        self._b2 = data["b2"]
        self._target_W1 = data["target_W1"]
        self._target_b1 = data["target_b1"]
        self._target_W2 = data["target_W2"]
        self._target_b2 = data["target_b2"]
        if "epsilon" in data:
            self.epsilon = float(data["epsilon"])
        if "update_counter" in data:
            self._update_counter = int(data["update_counter"])
        if "exploration_step" in data:
            self._exploration_step = int(data["exploration_step"])
        else:
            self._exploration_step = self._exploration_step_for(self.epsilon)
        self._steps_trained = int(data["steps_trained"]) if "steps_trained" in data else 0
        if self._optimizer is not None:
            self._optimizer.load_state(data)

    def _exploration_step_for(self, epsilon: float) -> int:
        """The step of the linear schedule at which epsilon equals ``epsilon`` (0 without one)."""
        if self.exploration_steps is None or self.epsilon_start == self.epsilon_end:
            return 0
        share = (self.epsilon_start - epsilon) / (self.epsilon_start - self.epsilon_end)
        return int(np.clip(round(share * self.exploration_steps), 0, self.exploration_steps))

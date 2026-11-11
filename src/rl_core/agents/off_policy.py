"""What the replay-buffer agents for continuous actions share: DDPG, TD3 and SAC."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from functools import reduce
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from rl_core.seeding import (
    AGENT_EXPLORATION_STREAM,
    AGENT_WEIGHTS_STREAM,
    derive_rng,
)
from rl_core.utils.config_values import (
    check_positive_int,
    non_negative_int,
    optional_positive_int,
    positive_int,
)

from .base_agent import BaseAgent
from .bounds import action_bounds
from .inference import DenseLayer, InferencePolicy
from .mid_episode import TRANSITIONS_KEY, MidEpisodeUpdates
from .mlp import Params, layer_count, mlp_backward, mlp_forward
from .optim import Adam
from .replay_buffer import ReplayBuffer

Array = npt.NDArray[np.float64]


# The shared config keys every subclass's DEFAULTS must hold.
SHARED_KEYS = (
    "hidden_sizes",
    "learning_rate",
    "gamma",
    "tau",
    "batch_size",
    "buffer_capacity",
    "learning_starts",
    "updates_per_step",
)


def global_norm(grads: Params) -> float:
    """The L2 norm of all gradients together."""
    return float(np.sqrt(sum(float(np.sum(g**2)) for g in grads.values())))


def add_metrics(totals: dict[str, float], step: Mapping[str, float]) -> None:
    """Add one gradient step's metrics to the running ``totals``, in place."""
    for name, value in step.items():
        totals[name] = totals.get(name, 0.0) + value


def mean_metrics(totals: Mapping[str, float], count: int) -> dict[str, float]:
    """Each summed metric divided by the ``count`` of steps that added to it."""
    return {name: total / count for name, total in totals.items()}


class OffPolicyAgent(MidEpisodeUpdates, BaseAgent):
    """
    An agent for bounded continuous actions that learns from a replay buffer with critics
    ``Q(s, a) = mlp([s, a])``.

    It holds the config keys every such agent reads, with a subclass's defaults (``DEFAULTS``):
    ``hidden_sizes``, ``learning_rate``, ``gamma``, ``tau``, ``batch_size``, ``buffer_capacity``,
    ``learning_starts``, ``updates_per_step``, plus ``max_updates_per_call`` and ``max_grad_norm``
    (default none). Around a subclass's networks it supplies:

    * The action bounds, finite, with ``scale = (high - low) / 2`` and ``bias = (high + low) / 2``.
    * ``select_action``: the greedy action when not training; otherwise uniform in the bounds for
      the first ``learning_starts`` training actions, then the subclass's ``_training_action``.
    * ``update``: stores the batch, then once warm-up is over and the buffer can fill a batch,
      ``updates_per_step`` gradient steps per transition (``MidEpisodeUpdates``), at most
      ``max_updates_per_call``, which the subclass's ``_train_steps`` takes.
    * The critic loss ``mean ½ (Q(s, a) - y)²`` and a critic step over the critics
      ``_critic_names`` lists; one Adam step (``_apply``); Polyak averaging of the target pairs
      ``_target_names`` lists.
    * The export policy of a tanh-squashed actor (``_tanh_scale_policy``).
    * Checkpoints of every network ``_networks`` returns and every Adam of
      ``_optimizers_by_net``, the counters and the replay buffer; loading refuses a file without
      one of the networks (another agent's) or with other shapes.

    Weights, exploration and replay sampling each draw from their own stream of the seed
    (``rl_core.seeding``). The base draws no weights: ``_build_networks`` gets the weights
    generator and decides the order.

    Init order: :meth:`_validate` and then :meth:`_build_networks` run inside this class's
    ``__init__``, after the shared keys are read and before the subclass's ``__init__`` continues
    past ``super().__init__``. An override of either may use the shared attributes and must read
    anything else from ``config``, never from an attribute the subclass sets after ``super()``.
    A concrete subclass sets ``SHORT_NAME`` and a ``DEFAULTS`` holding every key of
    ``SHARED_KEYS`` (or inherits them); ``__init__`` raises TypeError otherwise.
    """

    # The defaults of the shared config keys above, per agent.
    DEFAULTS: ClassVar[Mapping[str, Any]]
    # The agent's name in messages: "not a TD3 checkpoint".
    SHORT_NAME: ClassVar[str]
    # The replay buffer's columns hold its filled part; ReplayBuffer.restore checks them.
    _UNSIZED_CHECKPOINT_PREFIXES: ClassVar[tuple[str, ...]] = (ReplayBuffer.KEY_PREFIX,)
    # Every network, its Adam state and the counters are required (they were always saved); a
    # checkpoint from before the buffer was saved has none, and one saved without mid-episode
    # updates has no transition count.
    _OPTIONAL_CHECKPOINT_PREFIXES: ClassVar[tuple[str, ...]] = (
        ReplayBuffer.KEY_PREFIX,
        TRANSITIONS_KEY,
    )

    def __init__(self, observation_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        """
        Args:
            observation_dim: Length of an observation vector.
            action_dim: Length of an action vector.
            config: Hyperparameters; see the class docstring and the subclass's.

        Raises:
            ValueError: For a non-positive dimension or an invalid hyperparameter.
        """
        self._check_class_constants()
        super().__init__(observation_dim, action_dim, config)
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError(
                f"observation_dim and action_dim must be positive, got {observation_dim!r} "
                f"and {action_dim!r}"
            )
        defaults = self.DEFAULTS
        raw_sizes = config.get("hidden_sizes", defaults["hidden_sizes"])
        if np.ndim(raw_sizes) != 1:
            raise ValueError(f"hidden_sizes must be a list of widths, got {raw_sizes!r}")
        if len(raw_sizes) == 0:
            raise ValueError(f"hidden_sizes must hold at least one width, got {raw_sizes!r}")
        self.hidden_sizes: list[int] = [
            check_positive_int(f"hidden_sizes[{i}]", width) for i, width in enumerate(raw_sizes)
        ]
        self.learning_rate = float(config.get("learning_rate", defaults["learning_rate"]))
        self.gamma = float(config.get("gamma", defaults["gamma"]))
        self.tau = float(config.get("tau", defaults["tau"]))
        self.batch_size = positive_int(config, "batch_size", defaults["batch_size"])
        self.buffer_capacity = positive_int(config, "buffer_capacity", defaults["buffer_capacity"])
        self.learning_starts = non_negative_int(
            config, "learning_starts", defaults["learning_starts"]
        )
        self.updates_per_step = float(config.get("updates_per_step", defaults["updates_per_step"]))
        self.max_updates_per_call: int | None = optional_positive_int(
            config, "max_updates_per_call"
        )
        raw_clip = config.get("max_grad_norm")
        self.max_grad_norm: float | None = None if raw_clip is None else float(raw_clip)
        self._validate(config)

        self.action_low, self.action_high = action_bounds(config, action_dim, require_finite=True)
        low, high = self.action_low.astype(np.float64), self.action_high.astype(np.float64)
        self.scale: Array = (high - low) / 2.0
        self.bias: Array = (high + low) / 2.0

        seed = config.get("seed")
        self._build_networks(derive_rng(seed, AGENT_WEIGHTS_STREAM))

        self._explore_rng = derive_rng(seed, AGENT_EXPLORATION_STREAM)
        # The buffer samples on the replay stream of the seed itself, and stores raw observations
        # when normalising (ReplayBuffer).
        self.replay_buffer = ReplayBuffer(
            self.buffer_capacity,
            observation_dim,
            seed=seed,
            action_dim=action_dim,
            normalizer=self.observation_normalizer,
        )
        self._training_actions = 0
        self._gradient_steps = 0

    @classmethod
    def _check_class_constants(cls) -> None:
        """Raise TypeError if the class lacks ``SHORT_NAME`` or a ``DEFAULTS`` with every shared
        key: a subclass that forgot them would otherwise fail with a bare AttributeError."""
        missing = [name for name in ("SHORT_NAME", "DEFAULTS") if not hasattr(cls, name)]
        if missing:
            raise TypeError(f"{cls.__name__} must set {' and '.join(missing)} (OffPolicyAgent)")
        lacking = [key for key in SHARED_KEYS if key not in cls.DEFAULTS]
        if lacking:
            raise TypeError(f"{cls.__name__}.DEFAULTS lacks {', '.join(lacking)}")

    def _validate(self, config: Mapping[str, Any]) -> None:
        """
        Raise ValueError for any shared hyperparameter outside its range.

        Runs inside :meth:`__init__` once the shared keys are read, before any network is built.
        A subclass may extend it (call ``super()._validate(config)`` first); it must read its own
        keys from ``config``, as its ``__init__`` has not yet run past ``super().__init__``.
        """
        optimizer = str(config.get("optimizer", "adam")).lower()
        if optimizer != "adam":
            raise ValueError(
                f"{self.SHORT_NAME} trains with Adam; optimizer must be 'adam', got {optimizer!r}"
            )
        checks = [
            ("learning_rate", self.learning_rate > 0.0),
            ("gamma", 0.0 < self.gamma <= 1.0),
            ("tau", 0.0 < self.tau <= 1.0),
            ("updates_per_step", self.updates_per_step > 0.0),
            ("max_grad_norm", self.max_grad_norm is None or self.max_grad_norm > 0.0),
        ]
        for name, ok in checks:
            if not ok:
                raise ValueError(f"{name} is out of range: {getattr(self, name)!r}")

    # ------------------------------------------------------------------
    # What a subclass declares
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_networks(self, weights_rng: np.random.Generator) -> None:
        """Draw every network from ``weights_rng`` in the agent's order, copy the targets and
        make an Adam for each trained parameter group.

        Runs inside :meth:`__init__`, after :meth:`_validate` and before the subclass's
        ``__init__`` continues: it may use the shared attributes (``hidden_sizes``,
        ``max_grad_norm`` …) and must read anything else from ``self.config``."""

    @abstractmethod
    def _networks(self) -> dict[str, Params]:
        """Every trained parameter group and target by name, in checkpoint order."""

    @abstractmethod
    def _optimizers_by_net(self) -> dict[str, Adam]:
        """The Adam of every trained parameter group, by its name in :meth:`_networks`. Another
        backend trains these same groups and mirrors its optimizer state into these."""

    @abstractmethod
    def _critic_names(self) -> tuple[str, ...]:
        """The critics in :meth:`_networks` that :meth:`_critic_step` trains: one or two."""

    @abstractmethod
    def _target_names(self) -> list[tuple[str, str]]:
        """(online, target) names in :meth:`_networks` for every network with a target, in the
        order they are averaged."""

    @abstractmethod
    def _greedy_action(self, obs: Array) -> Array:
        """The evaluation action for one observation (1, O), shape (A,)."""

    @abstractmethod
    def _training_action(self, obs: Array) -> Array:
        """A training action after warm-up for one observation (1, O), shape (A,), in the bounds;
        it may draw from ``_explore_rng``."""

    @abstractmethod
    def _train_steps(self, num_updates: int) -> dict[str, float]:
        """
        Take ``num_updates`` gradient steps, each on a fresh sample from the replay buffer, and
        return the means of their metrics. The step's backend: another backend overrides only
        this.
        """

    # ------------------------------------------------------------------
    # Critics and targets
    # ------------------------------------------------------------------

    def _critic_forward(
        self, obs: Array, actions: Array, params: Params
    ) -> tuple[Array, dict[str, list[Array]]]:
        """``(Q, cache)``: Q of shape (B,) for observations (B, O) and actions (B, A)."""
        q, cache = mlp_forward(params, np.concatenate([obs, actions], axis=1))
        return q[:, 0], cache

    def _critic_loss_and_grads(
        self, obs: Array, actions: Array, targets: Array, params: Params | None = None
    ) -> tuple[dict[str, float], Params]:
        """
        Critic loss ``mean ½ (Q(s, a) - y)²`` and its gradients, with ``targets`` constant, for
        ``params`` (the first critic by default).

        Changes no state.
        """
        params = self._networks()[self._critic_names()[0]] if params is None else params
        q, cache = self._critic_forward(obs, actions, params)
        td_error = q - targets
        d_q = (td_error / len(q))[:, None]
        grads, _ = mlp_backward(params, cache, d_q)
        components = {
            "critic_loss": float(np.mean(0.5 * td_error**2)),
            "mean_q": float(np.mean(q)),
            "critic_grad_norm": global_norm(grads),
        }
        return components, grads

    def _critic_step(self, obs: Array, actions: Array, targets: Array) -> dict[str, float]:
        """
        One Adam step for each critic of :meth:`_critic_names` towards the same ``targets``, every
        gradient taken before any step. Returns the critic's metrics; for twin critics
        ``critic_loss`` is the mean of the two losses, ``mean_q`` the first critic's and
        ``critic_grad_norm`` the norm of both critics' gradients together.
        """
        networks, optimizers = self._networks(), self._optimizers_by_net()
        names = self._critic_names()
        results = [
            self._critic_loss_and_grads(obs, actions, targets, networks[name]) for name in names
        ]
        for name, (_, grads) in zip(names, results, strict=True):
            self._apply(networks[name], optimizers[name], grads)
        if len(results) == 1:
            return results[0][0]
        (first, _), (second, _) = results
        return {
            "critic_loss": (first["critic_loss"] + second["critic_loss"]) / 2.0,
            "mean_q": first["mean_q"],
            "critic_grad_norm": float(
                np.hypot(first["critic_grad_norm"], second["critic_grad_norm"])
            ),
        }

    def _min_target_q(self, next_obs: Array, actions: Array) -> Array:
        """The target critics' Q of ``actions`` at ``next_obs``, (B,): ``Q'`` for one critic,
        ``min(Q1', Q2')`` for twins (each critic of :meth:`_critic_names` through its target)."""
        networks, target_of = self._networks(), dict(self._target_names())
        values = [
            self._critic_forward(next_obs, actions, networks[target_of[name]])[0]
            for name in self._critic_names()
        ]
        return reduce(np.minimum, values)

    def _bootstrapped(self, rewards: Array, terminated: Array, next_values: Array) -> Array:
        """``r + gamma (1 - terminated) next_values``, (B,): no bootstrap past a terminal state."""
        not_done = 1.0 - np.asarray(terminated, dtype=np.float64)
        return np.asarray(rewards, dtype=np.float64) + self.gamma * not_done * next_values

    def _apply(self, params: Params, optimizer: Adam, grads: Params) -> None:
        """One Adam step on ``params`` in place."""
        for name, step in optimizer.step(grads, self.learning_rate).items():
            params[name] -= step

    def _target_pairs(self) -> list[tuple[Params, Params]]:
        """(online, target) parameters for every network with a target, in the order they are
        averaged."""
        networks = self._networks()
        return [(networks[online], networks[target]) for online, target in self._target_names()]

    def _polyak(self) -> None:
        """Move each target parameter a fraction ``tau`` towards its online parameter."""
        for online, target in self._target_pairs():
            for name in online:
                target[name] = self.tau * online[name] + (1.0 - self.tau) * target[name]

    def _tanh_scale_policy(self, actor: Params) -> InferencePolicy:
        """The greedy policy of an actor MLP: ReLU on every layer but the last, the last cut to
        its first ``action_dim`` outputs (all of a deterministic actor's; a squashed Gaussian's
        mean), then ``bias + scale · tanh``."""
        count, act_dim = layer_count(actor), self.action_dim
        last = count - 1
        layers = [DenseLayer(actor[f"W{i}"], actor[f"b{i}"], relu=True) for i in range(last)]
        layers.append(
            DenseLayer(actor[f"W{last}"][:, :act_dim], actor[f"b{last}"][:act_dim], relu=False)
        )
        return InferencePolicy(
            layers=tuple(layers),
            head="tanh_scale",
            action_low=self.action_low,
            action_high=self.action_high,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        A float32 action of shape (action_dim,) inside the bounds.

        Evaluation (``training=False``) returns the greedy action and changes nothing. A training
        action counts towards ``learning_starts``: uniform in the bounds during warm-up, then the
        agent's exploring action.
        """
        obs = np.asarray(observation, dtype=np.float64).reshape(1, -1)
        if not training:
            return self._greedy_action(obs).astype(np.float32)

        self._training_actions += 1
        if self._training_actions <= self.learning_starts:
            action = self._explore_rng.uniform(self.action_low, self.action_high)
        else:
            action = self._training_action(obs)
        return np.asarray(action, dtype=np.float32)

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Store the episode's transitions, then train once warm-up is over.

        Runs ``updates_per_step`` gradient steps per transition (at least one, at most
        ``max_updates_per_call``). Updating mid-episode (:meth:`enable_mid_episode_updates`), the
        steps follow the running count of transitions instead, so a call may take none.

        Returns:
            The means of the agent's metrics over the gradient steps (see the subclass); empty
            during warm-up or when the batch stored nothing.
        """
        n = self.replay_buffer.add_transitions(batch)
        if (
            n == 0
            or self._training_actions < self.learning_starts
            or not self.replay_buffer.can_sample(self.batch_size)
        ):
            return {}

        num_updates = self._updates_for(n, self.updates_per_step)
        if num_updates == 0:
            return {}
        if self.max_updates_per_call is not None:
            num_updates = min(num_updates, self.max_updates_per_call)
        return self._train_steps(num_updates)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _checkpoint_arrays(self) -> dict[str, Any]:
        """Everything training needs to resume beyond the generators: every network, every Adam
        state (``<net>_adam_``), the action and gradient-step counters and the replay buffer."""
        arrays: dict[str, Any] = {
            f"{net}_{name}": value
            for net, params in self._networks().items()
            for name, value in params.items()
        }
        for net, optimizer in self._optimizers_by_net().items():
            arrays.update(optimizer.state(prefix=f"{net}_adam_"))
        arrays["training_actions"] = np.array(self._training_actions)
        arrays["gradient_steps"] = np.array(self._gradient_steps)
        arrays.update(self._mid_episode_checkpoint_arrays())
        arrays.update(self.replay_buffer.checkpoint_state())
        return arrays

    def _checkpoint_kind(self) -> str:
        return self.SHORT_NAME

    def _load_checkpoint_arrays(self, data: Mapping[str, Any]) -> None:
        """
        Restore a checkpoint; one saved before the replay buffer was saved empties the buffer.
        Entries of networks this agent does not have (a TD3 file's second critic, loaded by
        DDPG) are ignored.

        Raises:
            ValueError: If its replay buffer does not fit this one. :meth:`BaseAgent.load` has
                refused a missing network (another agent's checkpoint, "not a TD3 checkpoint") or
                one of another shape (other ``hidden_sizes``) before this is called.
        """
        networks = self._networks()
        self.replay_buffer.restore(data)
        for net, params in networks.items():
            for name in params:
                params[name] = np.array(data[f"{net}_{name}"], dtype=np.float64)
        for net, optimizer in self._optimizers_by_net().items():
            optimizer.load_state(data, prefix=f"{net}_adam_")
        self._training_actions = int(data["training_actions"])
        self._gradient_steps = int(data["gradient_steps"])
        self._load_mid_episode_count(data)

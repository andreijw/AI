"""Training pipeline for RL agents."""

import logging
import math
import os
import warnings
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from ..agents.base_agent import BaseAgent
from ..environments import ResumableEnv
from ..utils.logger import json_safe
from .checkpoints import (
    TrainerState,
    checkpoint_paths,
    file_sha256,
    trainer_state_path,
    write_atomically,
)

_logger = logging.getLogger(__name__)

# Episodes between periodic metric records; also the window their averages cover.
_LOG_INTERVAL = 10


class Trainer:
    """
    Training pipeline for reinforcement learning agents.

    Handles the training loop, episode collection, and metric logging.
    """

    def __init__(
        self,
        env: gym.Env | Any,
        agent: BaseAgent,
        config: dict[str, Any],
        logger: Any | None = None,
        eval_env: gym.Env | Any | None = None,
        eval_seed: int | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            env: Environment to train in
            agent: Agent to train
            config: Training configuration
            logger: Optional logger for training metrics
            eval_env: Environment to evaluate in, apart from the training environment, so that
                evaluating does not consume training start states. Without it evaluation runs on
                ``env`` and continues its sequence.
            eval_seed: Required with ``eval_env``: episode i of every evaluation resets ``eval_env``
                with ``eval_seed + i``, so all evaluations in a run play the same start states
                (see ``rl_core.seeding.eval_seed_for``).

        Raises:
            ValueError: If ``eval_env`` is given without a non-negative integer ``eval_seed``.
        """
        if eval_env is not None and (
            eval_seed is None
            or isinstance(eval_seed, bool)
            or not isinstance(eval_seed, int | np.integer)
            or eval_seed < 0
        ):
            raise ValueError(f"eval_env needs eval_seed, a non-negative integer; got {eval_seed!r}")
        self.env = env
        self.eval_env = eval_env
        self.eval_seed: int | None = (
            int(eval_seed) if eval_env is not None and eval_seed is not None else None
        )
        self.agent = agent
        self.config = config
        self.logger = logger

        # Training parameters
        self.num_episodes = self._get_int_config("num_episodes", 1000)
        self.max_steps_per_episode = self._get_int_config("max_steps_per_episode", 500)
        self.eval_frequency = self._get_int_config("eval_frequency", 100)
        self.save_frequency = self._get_int_config("save_frequency", 100)
        self.checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
        # Step budget: when set, train until this many environment steps (finishing the episode that
        # crosses it) and ignore num_episodes. Evaluation episodes do not count.
        raw_steps = config.get("total_steps")
        self.total_steps: int | None = None
        if raw_steps is not None:
            if isinstance(raw_steps, bool) or not isinstance(raw_steps, int) or raw_steps <= 0:
                raise ValueError(f"total_steps must be a positive integer, got {raw_steps!r}")
            self.total_steps = raw_steps
            anneal = getattr(agent, "anneal_steps", None)
            if anneal is not None and anneal > raw_steps:
                warnings.warn(
                    f"anneal_steps {anneal} exceeds the step budget {raw_steps}: training stops "
                    "before the learning rate and clip range finish decaying",
                    stacklevel=2,
                )
        self.total_steps_trained = 0

        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be a positive integer, got {self.num_episodes!r}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                "max_steps_per_episode must be a positive integer, "
                f"got {self.max_steps_per_episode!r}"
            )
        if self.eval_frequency <= 0:
            raise ValueError(
                f"eval_frequency must be a positive integer, got {self.eval_frequency!r}"
            )
        if self.save_frequency <= 0:
            raise ValueError(
                f"save_frequency must be a positive integer, got {self.save_frequency!r}"
            )

        # Metrics tracking
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

        # Training progress, read by env wrappers such as TrainingVideoRecorder.
        # current_episode is the 1-based training episode in progress (0 before training).
        self.current_episode = 0
        self.evaluating = False

        # Values agent.update() has returned since the last periodic record, by metric name.
        self._update_metric_window: dict[str, list[float]] = {}
        self._non_finite_metrics_seen: set[str] = set()
        # 1-based training episode in which an update metric first stopped being finite.
        self.first_non_finite_episode: int | None = None

    def _get_int_config(self, name: str, default: int) -> int:
        """Get an integer config value without lossy coercion."""
        raw_value = self.config.get(name, default)

        if isinstance(raw_value, bool):
            raise ValueError(
                f"Configuration value for '{name}' must be a positive integer, "
                f"got boolean {raw_value!r}"
            )

        if isinstance(raw_value, int):
            return raw_value

        if isinstance(raw_value, float):
            if not raw_value.is_integer():
                raise ValueError(
                    f"Configuration value for '{name}' must be a positive integer, "
                    f"got non-integer float {raw_value!r}"
                )
            return int(raw_value)

        if isinstance(raw_value, str):
            value = raw_value.strip()
            if not value:
                raise ValueError(
                    f"Configuration value for '{name}' must be a positive integer, got empty string"
                )
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(
                    f"Configuration value for '{name}' must be a positive integer, "
                    f"got string {raw_value!r}"
                ) from exc

        raise ValueError(
            f"Configuration value for '{name}' must be a positive integer, "
            f"got {type(raw_value).__name__}: {raw_value!r}"
        )

    def _log_info(self, message: str) -> None:
        """Log an informational message using the logger if available, otherwise the module logger."""
        if self.logger is not None:
            info_method = getattr(self.logger, "info", None)
            if callable(info_method):
                info_method(message)
                return
        _logger.info(message)

    def _log_warning(self, message: str) -> None:
        """Log a warning using the logger if available, otherwise the module logger."""
        if self.logger is not None:
            warning_method = getattr(self.logger, "warning", None)
            if callable(warning_method):
                warning_method(message)
                return
        _logger.warning(message)

    def _record_update_metrics(self, metrics: Any) -> None:
        """
        Keep the numeric values one agent.update() call returned for the next periodic record.

        Values that are not real numbers are skipped. The first non-finite value of each metric
        is reported at WARNING level with the episode it appeared in; training continues.

        Args:
            metrics: Whatever agent.update() returned. Anything that is not a mapping is ignored.
        """
        if not isinstance(metrics, Mapping):
            return
        for name, raw_value in metrics.items():
            if isinstance(raw_value, bool) or not isinstance(
                raw_value, (int, float, np.integer, np.floating)
            ):
                continue
            value = float(raw_value)
            self._update_metric_window.setdefault(name, []).append(value)
            if not math.isfinite(value) and name not in self._non_finite_metrics_seen:
                self._non_finite_metrics_seen.add(name)
                if self.first_non_finite_episode is None:
                    self.first_non_finite_episode = self.current_episode
                self._log_warning(
                    f"Non-finite update metric '{name}' = {value} first appeared at "
                    f"episode {self.current_episode}."
                )

    def _drain_update_metrics(self) -> dict[str, Any]:
        """
        Summarize the update metrics gathered since the last periodic record, then clear them.

        Returns:
            Fields to merge into the periodic record: ``update`` maps each metric to its mean,
            and ``non_finite`` lists the metrics whose mean is not finite. Empty when no update
            metrics were gathered; ``non_finite`` is omitted when every mean is finite.
        """
        window, self._update_metric_window = self._update_metric_window, {}
        if not window:
            return {}
        # A mean is non-finite if any value was, or if finite values overflowed when summed.
        # _log_metrics writes such a mean as None; here it is also named so it cannot pass silently.
        with np.errstate(over="ignore", invalid="ignore"):
            means = {name: float(np.mean(values)) for name, values in window.items()}
        non_finite = [name for name, mean in means.items() if not math.isfinite(mean)]
        fields: dict[str, Any] = {"update": means}
        if non_finite:
            fields["non_finite"] = non_finite
        return fields

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        """
        Log training or evaluation metrics in a logger-agnostic way.

        The metrics pass through :func:`json_safe` first (non-finite numbers become None), so
        every record is strict JSON. Supports:
        - Custom loggers exposing `log(metrics: dict)`
        - Standard `logging.Logger`-like objects (using `.info(...)`)
        - Fallback to module logger when no compatible logger is provided
        """
        normalized_metrics = json_safe(metrics)
        if isinstance(self.logger, logging.Logger):
            # Use standard logging formatting; avoid calling logger.log with a dict
            self.logger.info("Metrics: %s", normalized_metrics)
        elif self.logger is None or not self._log_to_custom_logger(normalized_metrics):
            _logger.info("METRICS: %s", normalized_metrics)

    def _log_to_custom_logger(self, normalized_metrics: dict[str, Any]) -> bool:
        """
        Send metrics to a custom logger's `log(metrics: dict)`, else to its `.info(...)`.

        Returns:
            False if the logger has neither in a usable form.
        """
        log_method = getattr(self.logger, "log", None)
        if callable(log_method):
            try:
                log_method(normalized_metrics)
                return True
            except TypeError:
                # Likely a standard logging.Logger.log(level, msg, *args, **kwargs)
                pass
        info_method = getattr(self.logger, "info", None)
        if callable(info_method):
            info_method(f"Metrics: {normalized_metrics}")
            return True
        return False

    def train(self) -> dict[str, Any]:
        """
        Run the training loop.

        Returns:
            Dictionary of training statistics
        """
        budget = (
            f"{self.total_steps} environment steps"
            if self.total_steps is not None
            else f"{self.num_episodes} episodes"
        )
        self._log_info(f"Starting training for {budget}...")

        if self.episode_rewards and not self._budget_left(len(self.episode_rewards)):
            self._log_info(
                f"The budget is already met after {len(self.episode_rewards)} episodes; "
                "nothing to train."
            )
        while self._budget_left(len(self.episode_rewards)):
            episode = len(self.episode_rewards)  # done so far, restored by resume()
            self.current_episode = episode + 1
            episode_reward, episode_length = self._run_episode(training=True)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.total_steps_trained += episode_length

            # Log progress
            if (episode + 1) % _LOG_INTERVAL == 0:
                avg_reward = np.mean(self.episode_rewards[-_LOG_INTERVAL:])
                avg_length = np.mean(self.episode_lengths[-_LOG_INTERVAL:])
                self._log_info(
                    f"Episode {episode + 1}{self._progress_suffix()} | "
                    f"Avg Reward (last {_LOG_INTERVAL}): {avg_reward:.2f} | "
                    f"Avg Length (last {_LOG_INTERVAL}): {avg_length:.2f}"
                )

                self._log_metrics(
                    {
                        "episode": episode + 1,
                        "avg_reward": avg_reward,
                        "avg_length": avg_length,
                        **self._drain_update_metrics(),
                    }
                )

            # Evaluation
            if (episode + 1) % self.eval_frequency == 0:
                eval_stats = self.evaluate()
                self._log_info(f"Evaluation at episode {episode + 1}: {eval_stats}")

                self._log_metrics({"evaluation": eval_stats})

            # Save checkpoint
            if (episode + 1) % self.save_frequency == 0:
                self._save_checkpoint(episode + 1)

        self._log_info("Training complete!")

        return {
            "total_episodes": len(self.episode_rewards),
            "total_steps": self.total_steps_trained,
            "avg_reward": np.mean(self.episode_rewards),
            "avg_length": np.mean(self.episode_lengths),
            "final_avg_reward": np.mean(self.episode_rewards[-100:]),
        }

    def _budget_left(self, episodes_done: int) -> bool:
        """Whether another training episode starts: steps when budgeted in steps, else episodes."""
        if self.total_steps is not None:
            return self.total_steps_trained < self.total_steps
        return episodes_done < self.num_episodes

    def _progress_suffix(self) -> str:
        if self.total_steps is not None:
            return f" | steps {self.total_steps_trained}/{self.total_steps}"
        return f"/{self.num_episodes}"

    def _run_episode(
        self, training: bool = True, env: Any | None = None, reset_seed: int | None = None
    ) -> tuple[float, int]:
        """
        Run a single episode.

        Args:
            training: Whether to train during the episode
            env: Environment to run in; the training environment by default
            reset_seed: Seed for the episode's reset, or None to continue the env's sequence

        Returns:
            Tuple of (episode_reward, episode_length)
        """
        env = self.env if env is None else env
        obs, _ = env.reset(seed=reset_seed) if reset_seed is not None else env.reset()
        episode_reward = 0.0
        episode_length = 0

        # Collect trajectory transitions
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        terminateds = []

        done = False
        while not done and episode_length < self.max_steps_per_episode:
            # Select action
            action = self.agent.select_action(obs, training=training)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store trajectory transitions
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            dones.append(done)
            terminateds.append(bool(terminated))

            episode_reward += float(reward)
            episode_length += 1
            obs = next_obs

        # Agent update
        if training:
            batch = {
                "observations": np.array(observations),
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "next_observations": np.array(next_observations),
                "dones": np.array(dones),
                "terminated": np.array(terminateds, dtype=bool),
            }
            self._record_update_metrics(self.agent.update(batch))

        return episode_reward, episode_length

    def evaluate(self, num_episodes: int = 10) -> dict[str, float]:
        """
        Evaluate the agent greedily.

        With an evaluation environment, episode i resets it with ``eval_seed + i``, so every
        evaluation in a run plays the same start states and leaves the training environment alone.
        Without one, evaluation runs on the training environment and continues its sequence.

        Args:
            num_episodes: Number of episodes to evaluate over

        Returns:
            Dictionary of evaluation metrics
        """
        eval_rewards = []
        eval_lengths = []

        self.evaluating = True
        try:
            for i in range(num_episodes):
                if self.eval_env is not None and self.eval_seed is not None:
                    episode_reward, episode_length = self._run_episode(
                        training=False, env=self.eval_env, reset_seed=self.eval_seed + i
                    )
                else:
                    episode_reward, episode_length = self._run_episode(training=False)
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
        finally:
            self.evaluating = False

        return {
            "mean_reward": float(np.mean(eval_rewards)),
            "std_reward": float(np.std(eval_rewards)),
            "mean_length": float(np.mean(eval_lengths)),
        }

    def _save_checkpoint(self, episode: int) -> None:
        """
        Save a training checkpoint: the agent to ``agent_episode_<n>.pt`` and, beside it, what
        :meth:`resume` needs to continue the run to ``trainer_episode_<n>.npz`` (a
        :class:`TrainerState`, which records the agent file's hash). Each file is written to a
        temporary name and moved into place, the agent's first.

        Args:
            episode: Training episodes done
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        agent_path, trainer_path = checkpoint_paths(self.checkpoint_dir, episode)
        write_atomically(agent_path, self.agent.save)
        arrays = self._trainer_state(agent_path).to_arrays()

        def write_state(path: str) -> None:
            with open(path, "wb") as f:
                np.savez_compressed(f, **arrays)

        write_atomically(trainer_path, write_state)
        self._log_info(f"Checkpoint saved to {agent_path}")

    def _env_id(self) -> str:
        name = getattr(self.env, "env_name", None) or getattr(
            getattr(self.env, "spec", None), "id", None
        )
        return str(name or type(self.env).__name__)

    def _trainer_state(self, agent_path: str) -> TrainerState:
        return TrainerState(
            agent_checkpoint=os.path.basename(agent_path),
            agent_sha256=file_sha256(agent_path),
            agent_class=type(self.agent).__name__,
            env_id=self._env_id(),
            episode_rewards=list(self.episode_rewards),
            episode_lengths=list(self.episode_lengths),
            total_steps_trained=self.total_steps_trained,
            update_metric_window=self._update_metric_window,
            non_finite_metrics_seen=sorted(self._non_finite_metrics_seen),
            first_non_finite_episode=self.first_non_finite_episode,
            eval_seed=self.eval_seed,
            env_state=self.env.rng_state() if isinstance(self.env, ResumableEnv) else {},
        )

    def resume(self, agent_checkpoint: str) -> None:
        """
        Continue a run from a checkpoint this trainer's ``_save_checkpoint`` wrote: load the agent,
        then the trainer's histories, counters, metric window, evaluation seed and the training
        environment's generators. :meth:`train` then trains up to the budget, which counts the
        episodes and steps done before the stop, and numbering continues.

        The whole trainer file is read and checked first: that it belongs to this agent file,
        agent class and environment, and that its fields agree with each other. If the agent
        checkpoint itself then fails to load, build a new trainer.

        Raises:
            FileNotFoundError: If the trainer file beside the checkpoint is missing (checkpoints
                from before 2026-10-21 hold only the agent).
            ValueError: If the checkpoint is not named ``agent_episode_<n>``, the trainer file is
                malformed or belongs to another agent file, agent class or environment, or the run
                had environment generators or an evaluation environment this trainer lacks (or
                the other way round).
        """
        trainer_path = trainer_state_path(agent_checkpoint)
        if not os.path.exists(trainer_path):
            raise FileNotFoundError(
                f"{os.path.basename(trainer_path)} is missing beside {agent_checkpoint}: only "
                "checkpoints saved with their trainer state can be resumed"
            )
        with np.load(trainer_path) as data:
            state = TrainerState.from_arrays(data, agent_checkpoint)
        self._check_resumable(state)
        self.agent.load(agent_checkpoint)
        if isinstance(self.env, ResumableEnv):
            self.env.set_rng_state(state.env_state)
        self.episode_rewards = list(state.episode_rewards)
        self.episode_lengths = list(state.episode_lengths)
        self.total_steps_trained = state.total_steps_trained
        self.current_episode = len(self.episode_rewards)
        self._update_metric_window = {k: list(v) for k, v in state.update_metric_window.items()}
        self._non_finite_metrics_seen = set(state.non_finite_metrics_seen)
        self.first_non_finite_episode = state.first_non_finite_episode
        if state.eval_seed is not None:
            self.eval_seed = state.eval_seed
        self._log_info(
            f"Resumed from {agent_checkpoint} after {self.current_episode} episodes "
            f"({self.total_steps_trained} steps)"
        )

    def _check_resumable(self, state: TrainerState) -> None:
        """Raise ValueError unless the saved run can continue on this agent and environments."""
        for what, saved, actual in (
            ("agent", state.agent_class, type(self.agent).__name__),
            ("environment", state.env_id, self._env_id()),
        ):
            if saved != actual:
                raise ValueError(f"The checkpoint's {what} is {saved}; this trainer's is {actual}")
        if bool(state.env_state) != isinstance(self.env, ResumableEnv):
            raise ValueError(
                "The checkpoint holds environment generators this environment cannot restore"
                if state.env_state
                else "The checkpoint holds no environment generators for this environment"
            )
        if (state.eval_seed is not None) != (self.eval_env is not None):
            raise ValueError(
                "The run evaluated on its own evaluation environment; give this trainer one"
                if state.eval_seed is not None
                else "The run evaluated on its training environment; this trainer has an "
                "evaluation environment"
            )
        if isinstance(self.env, ResumableEnv):
            expected = set(self.env.rng_state())
            unexpected, missing = set(state.env_state) - expected, expected - set(state.env_state)
            if unexpected or missing:
                raise ValueError(
                    f"The checkpoint's environment state has keys {sorted(unexpected)} this "
                    f"environment does not, and lacks {sorted(missing)}"
                )
            self.env.check_rng_state(state.env_state)

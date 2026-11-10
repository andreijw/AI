"""Training pipeline for RL agents."""

import logging
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from ..agents.base_agent import BaseAgent
from ..environments import ResumableEnv
from ..utils.config_values import optional_positive_int
from ..utils.logger import json_safe
from .checkpoints import (
    TrainerState,
    checkpoint_paths,
    file_sha256,
    read_trainer_state,
    write_atomically,
)
from .curriculum import CurriculumScheduler, require_schedulable
from .parallel import InlinePool, num_envs_of, rng_states, split_rng_states

_logger = logging.getLogger(__name__)

# Episodes between periodic metric records; also the window their averages cover.
_LOG_INTERVAL = 10


@dataclass(frozen=True)
class EpisodeResult:
    """
    One played episode: its total reward, its steps, and whether it succeeded.

    ``success`` is ``info["is_success"]`` from the episode's last step, the Gymnasium and SB3
    convention, or None when that step's ``info`` has no such key: an environment that never
    reports success, or one that reports it only on the ending step for an episode
    ``max_steps_per_episode`` cut short.
    """

    reward: float
    length: int
    success: bool | None


# One step as the trainer keeps it: (observation, action, reward, next observation, done,
# terminated, raw observation, raw next observation), the observations as normalised for the agent.
_Transition = tuple[Any, Any, Any, Any, bool, bool, Any, Any]


@dataclass
class _Playing:
    """An episode in progress in one of several training environments."""

    observation: Any  # as normalised for the agent
    raw_observation: Any
    reward: float = 0.0
    length: int = 0
    success: bool | None = None
    # Steps not yet handed to the agent: the episode so far, or with update_every its steps
    # since the last window.
    transitions: list[_Transition] = field(default_factory=list)


def success_rate(successes: Sequence[bool | None]) -> float | None:
    """The share of ``successes`` that are True; None if there are none or any is unknown."""
    if not successes or any(success is None for success in successes):
        return None
    return sum(bool(success) for success in successes) / len(successes)


def known_successes(successes: Sequence[bool | None]) -> list[bool] | None:
    """``successes`` as bools when every one is known (see :func:`success_rate`), else None."""
    if success_rate(successes) is None:
        return None
    return [bool(success) for success in successes]


def _success_field(name: str, successes: Sequence[bool | None]) -> dict[str, float]:
    """``{name: rate}`` when the success rate of ``successes`` is known, else nothing."""
    rate = success_rate(successes)
    return {} if rate is None else {name: rate}


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
        curriculum: CurriculumScheduler | None = None,
        *,
        save_final: bool = False,
        extra_envs: Sequence[Any] = (),
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
            curriculum: Settings for ``env`` before each training episode, from easy to the
                target ``eval_env`` was built with (see ``rl_core.training.curriculum``). Without
                one the training environment's settings are never touched.
            save_final: Whether ``train()`` saves a checkpoint after its last episode (see
                ``train``). Only ``train.py`` asks for it; the benchmark, ``record.py`` and other
                callers of a config that names a ``checkpoint_dir`` do not write checkpoints they
                never asked for.
            extra_envs: The training environments after ``env`` when ``training.num_envs`` is
                above 1, built like it (``BenchmarkChallenge.make_extra_envs``). They are stepped
                in lockstep with it, each episode handed to the agent and recorded as it finishes
                (see :meth:`train`).

        Raises:
            ValueError: If ``eval_env`` is given without a non-negative integer ``eval_seed``, or
                ``curriculum`` without ``eval_env`` or with an ``env`` that cannot change its
                settings between episodes; if ``training.num_envs`` is not a positive integer or
                is not the number of training environments given; or, with several, if there is
                no ``eval_env``, an environment renders, ``update_every`` is set, or the agent
                refuses them (``enable_parallel_envs``).
        """
        if eval_env is not None and (
            eval_seed is None
            or isinstance(eval_seed, bool)
            or not isinstance(eval_seed, int | np.integer)
            or eval_seed < 0
        ):
            raise ValueError(f"eval_env needs eval_seed, a non-negative integer; got {eval_seed!r}")
        if curriculum is not None:
            # Evaluation and the solve check of greedy episodes must play the target, not the
            # training environment's current difficulty.
            if eval_env is None:
                raise ValueError(
                    "A curriculum needs an evaluation environment built at the target settings"
                )
            for training_env in (env, *extra_envs):
                require_schedulable(training_env)
        self.curriculum = curriculum
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
        if not isinstance(save_final, bool):
            raise ValueError(f"save_final must be a bool, got {save_final!r}")
        self.save_final = save_final
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
                    "before the annealed learning rate (and clip range) finish decaying",
                    stacklevel=2,
                )
        self.total_steps_trained = 0
        # Hand the agent the episode so far every update_every steps instead of the whole episode
        # at its end (None); with several environments, every environment's steps of each window
        # of update_every vector steps at once. Validated here, before the agent, which may refuse
        # it, is told: of the environments first, then of update_every.
        self.update_every: int | None = optional_positive_int(config, "update_every")
        self._check_parallel_envs(env, extra_envs, agent, eval_env)
        if self.update_every is not None:
            agent.enable_mid_episode_updates(self.update_every)

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
        # Each training episode's EpisodeResult.success.
        self.episode_successes: list[bool | None] = []

        # Training progress, read by env wrappers such as TrainingVideoRecorder.
        # current_episode is the 1-based training episode in progress (0 before training).
        self.current_episode = 0
        self.evaluating = False

        # Values agent.update() has returned since the last periodic record, by metric name.
        self._update_metric_window: dict[str, list[float]] = {}
        self._non_finite_metrics_seen: set[str] = set()
        # 1-based training episode in which an update metric first stopped being finite.
        self.first_non_finite_episode: int | None = None
        # The episode count of the last checkpoint this trainer saved.
        self._last_saved_episode: int | None = None

    def _check_parallel_envs(
        self, env: Any, extra_envs: Sequence[Any], agent: BaseAgent, eval_env: Any
    ) -> None:
        """Read ``num_envs``, keep the training environments, and refuse what several cannot do."""
        self.num_envs = num_envs_of(self.config)
        self.envs: tuple[Any, ...] = (env, *extra_envs)
        if len(self.envs) != self.num_envs:
            raise ValueError(
                f"training.num_envs is {self.num_envs}, but the trainer was given "
                f"{len(self.envs)} training environment(s)"
            )
        # Several environments train through the lockstep collector; tests force it for one.
        self._lockstep = self.num_envs > 1
        if self.num_envs == 1:
            return
        if eval_env is None:
            raise ValueError(
                f"num_envs {self.num_envs} needs an eval_env: evaluating on a training "
                "environment would cut the episode it is playing"
            )
        rendering = [i for i, e in enumerate(self.envs) if getattr(e, "render_mode", None)]
        if rendering:
            raise ValueError(
                f"num_envs {self.num_envs} cannot render or record training: environment(s) "
                f"{rendering} have a render_mode; train with one environment to watch it"
            )
        agent.enable_parallel_envs(self.num_envs, self.update_every)

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

        Saves a checkpoint every ``save_frequency`` episodes and, when the trainer was built with
        ``save_final=True`` (as ``train.py`` does), once more after the last episode so the newest
        files hold the final weights, unless that episode was just saved or this call trained none
        (a resume with the budget met). An interrupted or failed run saves nothing more.

        Returns:
            Dictionary of training statistics
        """
        start = len(self.episode_rewards)
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
        if self._lockstep:
            self._train_lockstep()
        while not self._lockstep and self._budget_left(len(self.episode_rewards)):
            episode = len(self.episode_rewards)  # done so far, restored by resume()
            self.current_episode = episode + 1
            if self.curriculum is not None:
                self._start_curriculum_episode()
            self._record_episode(self._run_episode(training=True))
            if (episode + 1) % self.eval_frequency == 0:
                self._evaluate_and_log(episode + 1)
            if (episode + 1) % self.save_frequency == 0:
                self._save_checkpoint(episode + 1)

        self._save_final_checkpoint(start)
        self._log_info("Training complete!")

        stats = {
            "total_episodes": len(self.episode_rewards),
            "total_steps": self.total_steps_trained,
            "avg_reward": np.mean(self.episode_rewards),
            "avg_length": np.mean(self.episode_lengths),
            "final_avg_reward": np.mean(self.episode_rewards[-100:]),
            **_success_field("final_success_rate", self.episode_successes[-100:]),
        }
        if self.curriculum is not None:
            stats["curriculum_difficulty"] = self._end_curriculum()
        return stats

    def _record_episode(self, played: EpisodeResult) -> None:
        """Record one finished training episode: its histories, the step count, the curriculum,
        and the periodic log record every ``_LOG_INTERVAL`` episodes."""
        self.episode_rewards.append(played.reward)
        self.episode_lengths.append(played.length)
        self.episode_successes.append(played.success)
        self.total_steps_trained += played.length
        done = len(self.episode_rewards)
        if self.curriculum is not None and self.curriculum.record(played.reward):
            self._log_info(
                f"Curriculum stage {self.curriculum.stage}/{self.curriculum.stages} "
                f"(difficulty {self.curriculum.difficulty:.2f}) from episode {done + 1}"
            )
        if done % _LOG_INTERVAL == 0:
            avg_reward = np.mean(self.episode_rewards[-_LOG_INTERVAL:])
            avg_length = np.mean(self.episode_lengths[-_LOG_INTERVAL:])
            self._log_info(
                f"Episode {done}{self._progress_suffix()} | "
                f"Avg Reward (last {_LOG_INTERVAL}): {avg_reward:.2f} | "
                f"Avg Length (last {_LOG_INTERVAL}): {avg_length:.2f}"
            )
            self._log_metrics(
                {
                    "episode": done,
                    "avg_reward": avg_reward,
                    "avg_length": avg_length,
                    **_success_field("success_rate", self.episode_successes[-_LOG_INTERVAL:]),
                    **((self.curriculum and self.curriculum.played_metrics()) or {}),
                    **self._drain_update_metrics(),
                }
            )

    def _evaluate_and_log(self, episodes_done: int) -> None:
        eval_stats = self.evaluate()
        self._log_info(f"Evaluation at episode {episodes_done}: {eval_stats}")
        self._log_metrics({"evaluation": eval_stats})

    def _may_start(self, in_flight_episodes: int, in_flight_steps: int) -> bool:
        """Whether a training environment may start another episode while these are in flight:
        the budget counts them as done. With nothing in flight this is :meth:`_budget_left`."""
        if self.total_steps is not None:
            return self.total_steps_trained + in_flight_steps < self.total_steps
        return len(self.episode_rewards) + in_flight_episodes < self.num_episodes

    def _train_lockstep(self) -> None:
        """
        Train from every training environment in lockstep, one vector step at a time.

        Each vector step starts an episode in every idle environment the budget allows (each
        given the curriculum's settings first), chooses every playing environment's action in
        environment order, and steps them all. Without ``update_every`` an episode that ends
        (terminated, truncated, or cut at ``max_steps_per_episode``) goes to the agent as one
        batch. With it, every ``update_every`` vector steps the agent gets every environment's
        steps of that window in one :meth:`BaseAgent.update_segments` call: per environment, the
        end of an episode that finished in the window, then the episode still playing, cut at the
        window's end (so it bootstraps); what is left when training ends goes in a last call.
        Finished episodes are then recorded in environment order, and evaluation and a checkpoint
        follow once, at the end of a vector step in which their frequency was crossed, named by the
        episodes recorded then. Once the budget is met no episode starts, and those in flight
        finish and are trained on.

        With ``update_every``, an episode that ends off a window boundary is recorded in the vector
        step it ends, before its last steps (up to ``update_every - 1``) are trained on at the
        window's end: an evaluation or checkpoint of that vector step sees the agent before them,
        and a resume from such a checkpoint never trains on them. The single-environment loop
        updates before it records.

        A non-finite update metric is attributed to ``current_episode``: without ``update_every``
        the episode the update was on; with it, the next episode to be recorded, since a window
        mixes several. With one environment this plays exactly the episodes, updates and records of
        the single-environment loop, and with ``update_every`` too while every episode's length is
        a multiple of it (the single loop restarts its chunks with each episode).
        """
        pool = InlinePool(self.envs)
        playing: dict[int, _Playing] = {}
        # With update_every: the segments of episodes that ended in this window, by environment,
        # the vector steps the window has run, and how many episodes ended in it.
        closed: dict[int, list[list[_Transition]]] = {}
        window_steps = episodes_ended = 0
        while True:
            self.current_episode = len(self.episode_rewards) + 1
            starting: list[int] = []
            for index in range(self.num_envs):
                in_flight_steps = sum(episode.length for episode in playing.values())
                if index not in playing and self._may_start(
                    len(playing) + len(starting), in_flight_steps
                ):
                    starting.append(index)
            if starting:
                settings = None if self.curriculum is None else self.curriculum.settings()
                for index, raw, _ in pool.reset(starting, settings):
                    obs = self.agent.normalize_observation(raw, training=True)
                    playing[index] = _Playing(obs, raw)
            if not playing:
                if closed:
                    self._hand_over_window(playing, closed, episodes_ended)
                self.current_episode = len(self.episode_rewards)
                return

            actions = {
                index: self.agent.select_action(playing[index].observation, training=True)
                for index in sorted(playing)
            }
            ended = []
            for index, raw_next, reward, terminated, truncated, info in pool.step(actions):
                episode = playing[index]
                next_obs = self.agent.normalize_observation(raw_next, training=True)
                done = terminated or truncated
                episode.transitions.append(
                    (
                        episode.observation,
                        actions[index],
                        reward,
                        next_obs,
                        done,
                        bool(terminated),
                        episode.raw_observation,
                        raw_next,
                    )
                )
                episode.reward += float(reward)
                episode.length += 1
                episode.success = bool(info["is_success"]) if "is_success" in info else None
                episode.observation, episode.raw_observation = next_obs, raw_next
                if done or episode.length >= self.max_steps_per_episode:
                    ended.append(index)

            before = len(self.episode_rewards)
            if self.update_every is None:
                for position, index in enumerate(ended):
                    # The number this episode is recorded as, so a non-finite update metric names
                    # it (first_non_finite_episode) even when several end in one vector step.
                    self.current_episode = before + position + 1
                    self._update_agent(playing[index].transitions)
                    self.agent.end_episode()
            else:
                for index in ended:
                    closed.setdefault(index, []).append(playing[index].transitions)
                    playing[index].transitions = []
                episodes_ended += len(ended)
                window_steps += 1
                if window_steps == self.update_every:
                    self._hand_over_window(playing, closed, episodes_ended)
                    window_steps = episodes_ended = 0
            for index in ended:
                episode = playing.pop(index)
                self._record_episode(EpisodeResult(episode.reward, episode.length, episode.success))
            after = len(self.episode_rewards)
            if after // self.eval_frequency > before // self.eval_frequency:
                self._evaluate_and_log(after)
            if after // self.save_frequency > before // self.save_frequency:
                self._save_checkpoint(after)

    def _hand_over_window(
        self,
        playing: dict[int, _Playing],
        closed: dict[int, list[list[_Transition]]],
        episodes_ended: int,
    ) -> None:
        """Give the agent one window: per environment in order, the segments of its episodes that
        ended in the window, then that of the episode it is playing; then ``end_episode`` once for
        each episode that ended. Empties the window."""
        segments = []
        for index in range(self.num_envs):
            segments += [self._batch(transitions) for transitions in closed.get(index, [])]
            if index in playing and playing[index].transitions:
                segments.append(self._batch(playing[index].transitions))
                playing[index].transitions = []
        closed.clear()
        if segments:
            for metrics in self.agent.update_segments(segments):
                self._record_update_metrics(metrics)
        for _ in range(episodes_ended):
            self.agent.end_episode()

    def _save_final_checkpoint(self, start: int) -> None:
        """Save the last episode when ``save_final`` is set, this call trained (``start`` is the
        episodes done before it) and the last checkpoint saved is not of that episode."""
        done = len(self.episode_rewards)
        if self.save_final and done > start and done != self._last_saved_episode:
            self._save_checkpoint(done)

    def _start_curriculum_episode(self) -> None:
        """Give the training environment the curriculum's settings for the next episode."""
        assert self.curriculum is not None
        require_schedulable(self.env).apply_settings(self.curriculum.settings())

    def _end_curriculum(self) -> float:
        """The difficulty the last training episode played, warning if it is below the target."""
        assert self.curriculum is not None
        played = (self.curriculum.played_metrics() or self.curriculum.metrics())[
            "curriculum_difficulty"
        ]
        if played < 1.0:
            self._log_warning(
                f"Training ended at curriculum difficulty {played:.2f}, below the target its "
                "evaluation played"
            )
        return float(played)

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
    ) -> EpisodeResult:
        """
        Run a single episode.

        Args:
            training: Whether to train during the episode
            env: Environment to run in; the training environment by default
            reset_seed: Seed for the episode's reset, or None to continue the env's sequence

        Returns:
            The episode's reward, length and success (see :class:`EpisodeResult`)
        """
        env = self.env if env is None else env
        raw_obs, _ = env.reset(seed=reset_seed) if reset_seed is not None else env.reset()
        # Each observation passes through the agent's normalisation once (identity when off), so
        # the action, the stored transition and the next step's bootstrap share one input.
        obs = self.agent.normalize_observation(raw_obs, training=training)
        episode_reward = 0.0
        episode_length = 0
        success: bool | None = None

        # Transitions not yet handed to the agent: the whole episode, or with update_every the
        # steps since the last update. Each keeps the raw observations too, for an agent whose
        # replay buffer stores them and normalises when it samples.
        transitions: list[_Transition] = []
        update_every = self.update_every if training else None

        done = False
        while not done and episode_length < self.max_steps_per_episode:
            action = self.agent.select_action(obs, training=training)
            raw_next, reward, terminated, truncated, info = env.step(action)
            next_obs = self.agent.normalize_observation(raw_next, training=training)
            done = terminated or truncated
            transitions.append(
                (obs, action, reward, next_obs, done, bool(terminated), raw_obs, raw_next)
            )

            episode_reward += float(reward)
            episode_length += 1
            success = bool(info["is_success"]) if "is_success" in info else None
            obs, raw_obs = next_obs, raw_next
            # The step that ends the episode goes with the final update below.
            last = done or episode_length >= self.max_steps_per_episode
            if update_every is not None and episode_length % update_every == 0 and not last:
                self._update_agent(transitions)
                transitions = []

        if training:
            self._update_agent(transitions)
            self.agent.end_episode()

        return EpisodeResult(episode_reward, episode_length, success)

    def _update_agent(self, transitions: list[_Transition]) -> None:
        """Hand the agent these transitions, in order, as one batch (:meth:`_batch`), and keep its
        metrics."""
        self._record_update_metrics(self.agent.update(self._batch(transitions)))

    def _batch(self, transitions: list[_Transition]) -> dict[str, np.ndarray]:
        """
        These transitions, in order, as one batch. A normalising agent's batch also carries
        ``raw_observations`` and ``raw_next_observations``, the environment's observations before
        normalisation, which a replay buffer stores; an on-policy agent ignores them.
        """
        (observations, actions, rewards, next_observations, dones, terminateds, raws,
         raw_nexts) = zip(*transitions, strict=True)  # fmt: skip
        batch = {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_observations": np.array(next_observations),
            "dones": np.array(dones),
            "terminated": np.array(terminateds, dtype=bool),
        }
        if self.agent.observation_normalizer is not None:
            batch["raw_observations"] = np.array(raws)
            batch["raw_next_observations"] = np.array(raw_nexts)
        return batch

    def evaluate(self, num_episodes: int = 10) -> dict[str, float]:
        """
        Evaluate the agent greedily.

        With an evaluation environment, episode i resets it with ``eval_seed + i``, so every
        evaluation in a run plays the same start states and leaves the training environment alone.
        Without one, evaluation runs on the training environment and continues its sequence.

        Args:
            num_episodes: Number of episodes to evaluate over

        Returns:
            Dictionary of evaluation metrics, with ``success_rate`` when every episode reported
            its success (see :class:`EpisodeResult`)
        """
        episodes = self.evaluate_episodes(num_episodes)
        eval_rewards = [episode.reward for episode in episodes]
        eval_lengths = [episode.length for episode in episodes]
        return {
            "mean_reward": float(np.mean(eval_rewards)),
            "std_reward": float(np.std(eval_rewards)),
            "mean_length": float(np.mean(eval_lengths)),
            **_success_field("success_rate", [episode.success for episode in episodes]),
        }

    def evaluate_episodes(self, num_episodes: int) -> list[EpisodeResult]:
        """
        Each greedy episode's reward, length and success, played as :meth:`evaluate` plays them.

        Args:
            num_episodes: Number of episodes to play
        """
        episodes = []
        self.evaluating = True
        try:
            for i in range(num_episodes):
                if self.eval_env is not None and self.eval_seed is not None:
                    episodes.append(
                        self._run_episode(
                            training=False, env=self.eval_env, reset_seed=self.eval_seed + i
                        )
                    )
                else:
                    episodes.append(self._run_episode(training=False))
        finally:
            self.evaluating = False
        return episodes

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
        self._last_saved_episode = episode
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
            env_state=rng_states(self.envs),
            curriculum=None if self.curriculum is None else self.curriculum.spec(),
            episode_successes=known_successes(self.episode_successes),
            num_envs=self.num_envs,
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
        state = read_trainer_state(agent_checkpoint)
        self._check_resumable(state)
        self.agent.load(agent_checkpoint)
        for env, env_state in zip(
            self.envs, split_rng_states(self.envs, state.env_state), strict=True
        ):
            if isinstance(env, ResumableEnv):
                env.set_rng_state(env_state)
        self.episode_rewards = list(state.episode_rewards)
        self.episode_lengths = list(state.episode_lengths)
        # A run that did not save its successes (not all known, or from before the metric) resumes
        # with them unknown, so no rate covers those episodes.
        self.episode_successes = (
            list(state.episode_successes)
            if state.episode_successes is not None
            else [None] * len(self.episode_rewards)
        )
        self.total_steps_trained = state.total_steps_trained
        self.current_episode = len(self.episode_rewards)
        self._update_metric_window = {k: list(v) for k, v in state.update_metric_window.items()}
        self._non_finite_metrics_seen = set(state.non_finite_metrics_seen)
        self.first_non_finite_episode = state.first_non_finite_episode
        if state.eval_seed is not None:
            self.eval_seed = state.eval_seed
        if self.curriculum is not None:
            self.curriculum.replay(self.episode_rewards)
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
        if state.num_envs != self.num_envs:
            raise ValueError(
                f"The run trained with num_envs {state.num_envs}; this trainer has "
                f"{self.num_envs}: resume with the same number of environments"
            )
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
        own = None if self.curriculum is None else self.curriculum.spec()
        if state.curriculum != own:
            raise ValueError(
                "The run trained without a curriculum; this trainer has one"
                if state.curriculum is None
                else "The run trained with a curriculum; give this trainer the same one"
                if own is None
                else "The run trained with a different curriculum (mode, pacing, start or target) "
                "from this trainer's"
            )
        if isinstance(self.env, ResumableEnv):
            expected = set(rng_states(self.envs))
            unexpected, missing = set(state.env_state) - expected, expected - set(state.env_state)
            if unexpected or missing:
                raise ValueError(
                    f"The checkpoint's environment state has keys {sorted(unexpected)} this "
                    f"environment does not, and lacks {sorted(missing)}"
                )
            for env, env_state in zip(
                self.envs, split_rng_states(self.envs, state.env_state), strict=True
            ):
                if isinstance(env, ResumableEnv):
                    env.check_rng_state(env_state)

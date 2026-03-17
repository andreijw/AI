"""Training pipeline for RL agents."""

import os
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..agents.base_agent import BaseAgent
from ..environments.cartpole_env import CartPoleEnv


class Trainer:
    """
    Training pipeline for reinforcement learning agents.

    Handles the training loop, episode collection, and metric logging.
    """

    def __init__(
        self,
        env: CartPoleEnv,
        agent: BaseAgent,
        config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        """
        Initialize the trainer.

        Args:
            env: Environment to train in
            agent: Agent to train
            config: Training configuration
            logger: Optional logger for training metrics
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.logger = logger

        # Training parameters
        self.num_episodes = config.get("num_episodes", 1000)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 500)
        self.eval_frequency = config.get("eval_frequency", 100)
        self.save_frequency = config.get("save_frequency", 100)
        self.checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")

        # Metrics tracking
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _log_info(self, message: str) -> None:
        """Log an informational message using the logger if available, otherwise print."""
        if self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info(message)
        else:
            print(message)

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log training or evaluation metrics in a logger-agnostic way.

        Supports:
        - Custom loggers exposing `log(metrics: dict)`
        - Standard `logging.Logger`-like objects (using `.info(...)`)
        - Fallback to printing when no compatible logger is provided
        """
        def _normalize_value(value: Any) -> Any:
            """
            Recursively convert values to JSON-serializable Python built-ins.

            - NumPy scalars (np.generic) -> corresponding Python scalars via .item()
            - Containers (dict, list, tuple) -> same structure with normalized contents
            """
            if isinstance(value, np.generic):
                # Includes np.floating, np.integer, etc.
                return value.item()
            if isinstance(value, dict):
                return {k: _normalize_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_normalize_value(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_normalize_value(v) for v in value)
            return value

        normalized_metrics = _normalize_value(metrics)

        # No logger configured: print metrics
        if self.logger is None:
            print(f"METRICS: {normalized_metrics}")
            return

        # Special handling for standard logging.Logger instances
        if isinstance(self.logger, logging.Logger):
            # Use standard logging formatting; avoid calling logger.log with a dict
            self.logger.info("Metrics: %s", normalized_metrics)
            return

        # Prefer a custom `log(metrics: dict)` method if available
        log_method = getattr(self.logger, "log", None)
        if callable(log_method):
            try:
                # Custom logger expected to accept a single dict argument
                log_method(normalized_metrics)
                return
            except TypeError:
                # Likely a standard logging.Logger.log(level, msg, *args, **kwargs)
                pass

        # Fallback: use `.info(...)` if available
        info_method = getattr(self.logger, "info", None)
        if callable(info_method):
            info_method(f"Metrics: {normalized_metrics}")
        else:
            # Last resort: print metrics
            print(f"METRICS: {normalized_metrics}")

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.

        Returns:
            Dictionary of training statistics
        """
        self._log_info(f"Starting training for {self.num_episodes} episodes...")

        for episode in range(self.num_episodes):
            episode_reward, episode_length = self._run_episode(training=True)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                self._log_info(
                    f"Episode {episode + 1}/{self.num_episodes} | "
                    f"Avg Reward (last 10): {avg_reward:.2f} | "
                    f"Avg Length (last 10): {avg_length:.2f}"
                )

                self._log_metrics(
                    {
                        "episode": episode + 1,
                        "avg_reward": avg_reward,
                        "avg_length": avg_length,
                    }
                )

            # Evaluation
            if (episode + 1) % self.eval_frequency == 0:
                eval_stats = self._evaluate()
                self._log_info(f"Evaluation at episode {episode + 1}: {eval_stats}")

                self._log_metrics({"evaluation": eval_stats})

            # Save checkpoint
            if (episode + 1) % self.save_frequency == 0:
                self._save_checkpoint(episode + 1)

        self._log_info("Training complete!")

        return {
            "total_episodes": self.num_episodes,
            "avg_reward": np.mean(self.episode_rewards),
            "avg_length": np.mean(self.episode_lengths),
            "final_avg_reward": np.mean(self.episode_rewards[-100:]),
        }

    def _run_episode(self, training: bool = True) -> Tuple[float, int]:
        """
        Run a single episode.

        Args:
            training: Whether to train during the episode

        Returns:
            Tuple of (episode_reward, episode_length)
        """
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        # Placeholder for trajectory collection (for future PPO implementation)
        observations = []
        actions = []
        rewards = []

        done = False
        while not done and episode_length < self.max_steps_per_episode:
            # Select action
            action = self.agent.select_action(obs, training=training)

            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store trajectory (for future policy updates)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

            episode_reward += reward
            episode_length += 1
            obs = next_obs

        # Agent update (placeholder - actual implementation depends on algorithm)
        if training:
            batch = {
                "observations": np.array(observations),
                "actions": np.array(actions),
                "rewards": np.array(rewards),
            }
            self.agent.update(batch)

        return episode_reward, episode_length

    def _evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent.

        Args:
            num_episodes: Number of episodes to evaluate over

        Returns:
            Dictionary of evaluation metrics
        """
        eval_rewards = []
        eval_lengths = []

        for _ in range(num_episodes):
            episode_reward, episode_length = self._run_episode(training=False)
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

        return {
            "mean_reward": float(np.mean(eval_rewards)),
            "std_reward": float(np.std(eval_rewards)),
            "mean_length": float(np.mean(eval_lengths)),
        }

    def _save_checkpoint(self, episode: int) -> None:
        """
        Save a training checkpoint.

        Args:
            episode: Current episode number
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(self.checkpoint_dir, f"agent_episode_{episode}.pt")
        self.agent.save(checkpoint_path)
        self._log_info(f"Checkpoint saved to {checkpoint_path}")

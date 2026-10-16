"""Main training script for CartPole RL agent."""

import argparse
import os
import sys
from contextlib import contextmanager

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_cartpole.environments import CartPoleEnv
from rl_core.agents import (
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
)
from rl_core.training import Trainer
from rl_core.training.budget import apply_budget_override
from rl_core.utils import load_config, plot_training_metrics, setup_logger
from rl_core.utils.video import TrainingVideoRecorder

AGENT_CLASSES = {
    "random": RandomAgent,
    "reinforce": ReinforceAgent,
    "actor_critic": ActorCriticAgent,
    "ppo": PPOAgent,
    "dqn": DQNAgent,
}

CONTINUOUS_AGENT_CLASSES = {
    "random": ContinuousRandomAgent,
    "actor_critic": ContinuousActorCriticAgent,
    "ppo": ContinuousPPOAgent,
}


def load_run_config(config_path, agent_type=None, num_episodes=None, total_steps=None):
    """Load and validate a run config, applying CLI overrides. Returns (config, agent_type)."""
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    for section in ("environment", "agent", "training"):
        section_value = config.get(section)
        if not isinstance(section_value, dict):
            raise ValueError(
                f"Config file '{config_path}' must define a '{section}' mapping section."
            )
    if config["agent"].get("config") is None:
        config["agent"]["config"] = {}
    elif not isinstance(config["agent"].get("config"), dict):
        raise ValueError(
            f"Config file '{config_path}' must define 'agent.config' as a mapping section."
        )

    if agent_type is not None:
        config["agent"]["type"] = agent_type
    apply_budget_override(config["training"], num_episodes, total_steps)
    agent_type = config["agent"].get("type")
    if not isinstance(agent_type, str):
        raise ValueError(
            f"Config file '{config_path}' must define 'agent.type' as a string "
            f"in {sorted(AGENT_CLASSES)}."
        )
    if agent_type not in AGENT_CLASSES:
        raise ValueError(
            f"Config file '{config_path}' has unsupported 'agent.type': {agent_type!r}. "
            f"Expected one of {sorted(AGENT_CLASSES)}."
        )
    return config, agent_type


def agent_output_dir(base_dir, agent_type, challenge=None):
    """Per-agent (and optionally per-challenge) output folder."""
    if challenge:
        return os.path.join(base_dir, challenge, agent_type)
    return os.path.join(base_dir, agent_type)


def require_moviepy():
    """Raise ImportError with install instructions if moviepy (needed to write videos) is missing."""
    try:
        import moviepy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'moviepy' package is required for --record-video. "
            "Install it with: pip install moviepy  "
            "or: pip install -e '.[video]'"
        ) from exc


@contextmanager
def headless_rendering(enabled=True):
    """Default SDL to offscreen drivers so rgb_array rendering needs no display; restore on exit."""
    sdl_env_backup = {}
    if enabled:
        sdl_defaults = {
            "SDL_VIDEODRIVER": "offscreen",
            "SDL_AUDIODRIVER": "dummy",
        }
        for key, value in sdl_defaults.items():
            sdl_env_backup[key] = os.environ.get(key)
            os.environ.setdefault(key, value)
    try:
        yield
    finally:
        for key, previous in sdl_env_backup.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def create_env(config, render_mode):
    """Create the CartPoleEnv or PendulumEnv described by the config's environment section."""
    env_config = config["environment"]
    env_name = env_config.get("name") or "CartPole-v1"
    if "pendulum" in env_name.lower():
        from rl_pendulum.environments import PendulumEnv

        return PendulumEnv(
            env_name=env_name,
            render_mode=render_mode,
            max_episode_steps=env_config.get("max_episode_steps", 200),
            seed=env_config.get("seed"),
            obs_noise_std=env_config.get("obs_noise_std", 0.0),
            action_noise_std=env_config.get("action_noise_std", 0.0),
            domain_randomization=env_config.get("domain_randomization"),
        )

    obs_noise_std = env_config.get("obs_noise_std")
    action_noise_prob = env_config.get("action_noise_prob")
    return CartPoleEnv(
        env_name="CartPole-v1" if env_name is None else env_name,
        render_mode=render_mode,
        max_episode_steps=env_config.get("max_episode_steps", 500),
        seed=env_config.get("seed"),
        obs_noise_std=0.0 if obs_noise_std is None else obs_noise_std,
        action_noise_prob=0.0 if action_noise_prob is None else action_noise_prob,
        domain_randomization=env_config.get("domain_randomization"),
    )


def create_agent(agent_type, config, env):
    """Create an agent of agent_type sized for env, using the config's agent section."""
    import gymnasium as gym

    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    if is_continuous:
        if agent_type not in CONTINUOUS_AGENT_CLASSES:
            raise ValueError(
                f"Agent type '{agent_type}' is not supported for continuous action spaces. "
                f"Supported: {sorted(CONTINUOUS_AGENT_CLASSES)}"
            )
        agent_cfg = dict(config["agent"]["config"])
        if hasattr(env.action_space, "low") and "action_low" not in agent_cfg:
            agent_cfg["action_low"] = env.action_space.low
        if hasattr(env.action_space, "high") and "action_high" not in agent_cfg:
            agent_cfg["action_high"] = env.action_space.high

        return CONTINUOUS_AGENT_CLASSES[agent_type](
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=agent_cfg,
        )

    return AGENT_CLASSES[agent_type](
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config["agent"]["config"],
    )


def close_env(env, logger):
    """Close env, logging instead of raising if closing fails."""
    try:
        env.close()
    except Exception:
        logger.exception("Failed to close environment")
    else:
        logger.info("Environment closed")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CartPole RL agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cartpole_default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during training (requires a display; mutually exclusive with --record-video)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help=(
            "Record videos of training episodes to disk. "
            "Works on headless machines (e.g. NVIDIA Orin Nano via SSH) "
            "because no display is required. Mutually exclusive with --render."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        help="Directory to save recorded videos (default: ./videos/<agent_type>)",
    )
    parser.add_argument(
        "--video-every",
        type=int,
        help=(
            "With --record-video, record every Nth training episode (must be > 0). "
            "Default: training.eval_frequency."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a training-metrics plot (rewards and episode lengths) after training.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        help="Directory to save the training metrics plot (default: ./plots/<agent_type>)",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=sorted(AGENT_CLASSES),
        help="Override the agent type from config (random, reinforce, actor_critic, ppo, dqn).",
    )
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--num-episodes",
        type=int,
        help="Train for this many episodes (must be > 0); replaces any training.total_steps.",
    )
    budget.add_argument(
        "--total-steps",
        type=int,
        help="Train for this many environment steps (must be > 0); sets training.total_steps.",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        choices=["cartpole", "pendulum"],
        default=None,
        help="Override challenge type (cartpole or pendulum).",
    )
    args = parser.parse_args()

    if args.render and args.record_video:
        parser.error("--render and --record-video are mutually exclusive. Use one or the other.")
    if args.num_episodes is not None and args.num_episodes <= 0:
        parser.error("--num-episodes must be a positive integer.")
    if args.total_steps is not None and args.total_steps <= 0:
        parser.error("--total-steps must be a positive integer.")
    if args.video_every is not None and not args.record_video:
        parser.error("--video-every requires --record-video.")
    if args.video_every is not None and args.video_every <= 0:
        parser.error("--video-every must be a positive integer.")

    config, agent_type = load_run_config(
        args.config, args.agent_type, args.num_episodes, args.total_steps
    )
    env_name = config["environment"].get("name") or "CartPole-v1"
    challenge = args.challenge or ("pendulum" if "pendulum" in env_name.lower() else "cartpole")

    config["training"]["checkpoint_dir"] = agent_output_dir(
        config["training"].get("checkpoint_dir", "./checkpoints"), agent_type, challenge=challenge
    )
    video_dir = args.video_dir or agent_output_dir("./videos", agent_type, challenge=challenge)
    plot_dir = args.plot_dir or agent_output_dir("./plots", agent_type, challenge=challenge)
    config["training"]["log_dir"] = agent_output_dir(
        config["training"].get("log_dir", "./logs"), agent_type, challenge=challenge
    )

    # Setup logger
    logger = setup_logger(
        name=f"rl_{challenge}",
        log_dir=config["training"]["log_dir"],
    )
    logger.info(f"Starting {challenge.capitalize()} RL training")
    logger.info(f"Configuration: {config}")

    # --record-video uses rgb_array (no display needed); --render uses human (requires display).
    if args.record_video:
        render_mode = "rgb_array"
    elif args.render:
        render_mode = "human"
    else:
        render_mode = config["environment"].get("render_mode")

    with headless_rendering(enabled=args.record_video):
        env = None
        try:
            env = create_env(config, render_mode)
            if args.record_video:
                require_moviepy()
            logger.info("Environment created")

            agent = create_agent(agent_type, config, env)
            logger.info(f"Agent created: {agent_type}")

            trainer = Trainer(
                env=env,
                agent=agent,
                config=config["training"],
                logger=logger,
            )
            logger.info("Trainer created")

            # Wrap with TrainingVideoRecorder when --record-video is set. It captures
            # rgb_array frames and saves them as mp4 files, so it works on headless
            # machines where no display is available. It follows the trainer's episode
            # counter, so it must be created after the trainer. CartPoleEnv.wrap_env()
            # injects the wrapper through the proper interface rather than bypassing it
            # via direct attribute assignment.
            if args.record_video:
                video_every = (
                    args.video_every if args.video_every is not None else trainer.eval_frequency
                )
                os.makedirs(video_dir, exist_ok=True)
                env.wrap_env(
                    TrainingVideoRecorder(
                        env.env,
                        video_folder=video_dir,
                        progress=trainer,
                        every=video_every,
                    )
                )
                logger.info(f"Recording videos every {video_every} episodes to '{video_dir}'")

            # Train
            try:
                stats = trainer.train()
                logger.info(f"Training complete. Final stats: {stats}")

                if args.plot:
                    plot_path = plot_training_metrics(
                        episode_rewards=trainer.episode_rewards,
                        episode_lengths=trainer.episode_lengths,
                        save_dir=plot_dir,
                        title=f"{challenge.capitalize()} {agent_type.capitalize()} Agent – Training Metrics",
                    )
                    logger.info(f"Training metrics plot saved to '{plot_path}'")
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
        finally:
            if env is not None:
                close_env(env, logger)


if __name__ == "__main__":
    main()

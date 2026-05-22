"""Main training script for CartPole RL agent."""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_cartpole.agents import ActorCriticAgent, PPOAgent, RandomAgent, ReinforceAgent
from rl_cartpole.environments import CartPoleEnv
from rl_cartpole.training import Trainer
from rl_cartpole.utils import load_config, plot_training_metrics, setup_logger

AGENT_CLASSES = {
    "random": RandomAgent,
    "reinforce": ReinforceAgent,
    "actor_critic": ActorCriticAgent,
    "ppo": PPOAgent,
}


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
        default="./videos",
        help="Directory to save recorded videos (default: ./videos)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a training-metrics plot (rewards and episode lengths) after training.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="./plots",
        help="Directory to save the training metrics plot (default: ./plots)",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=sorted(AGENT_CLASSES),
        help="Override the agent type from config (random, reinforce, actor_critic, ppo).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        help="Override training.num_episodes from config (must be > 0).",
    )
    args = parser.parse_args()

    if args.render and args.record_video:
        parser.error("--render and --record-video are mutually exclusive. Use one or the other.")
    if args.num_episodes is not None and args.num_episodes <= 0:
        parser.error("--num-episodes must be a positive integer.")

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    for section in ("environment", "agent", "training"):
        section_value = config.get(section)
        if not isinstance(section_value, dict):
            raise ValueError(
                f"Config file '{args.config}' must define a '{section}' mapping section."
            )
    if config["agent"].get("config") is None:
        config["agent"]["config"] = {}
    elif not isinstance(config["agent"].get("config"), dict):
        raise ValueError(
            f"Config file '{args.config}' must define 'agent.config' as a mapping section."
        )

    if args.agent_type is not None:
        config["agent"]["type"] = args.agent_type
    if args.num_episodes is not None:
        config["training"]["num_episodes"] = args.num_episodes

    # Setup logger
    logger = setup_logger(
        name="rl_cartpole",
        log_dir=config["training"]["log_dir"],
    )
    logger.info("Starting CartPole RL training")
    logger.info(f"Configuration: {config}")

    sdl_env_backup = {}
    if args.record_video:
        sdl_defaults = {
            "SDL_VIDEODRIVER": "offscreen",
            "SDL_AUDIODRIVER": "dummy",
        }
        for key, value in sdl_defaults.items():
            sdl_env_backup[key] = os.environ.get(key)
            os.environ.setdefault(key, value)

    env = None
    try:
        # Determine render mode.
        # --record-video uses rgb_array (no display needed); --render uses human (requires display).
        if args.record_video:
            render_mode = "rgb_array"
        elif args.render:
            render_mode = "human"
        else:
            render_mode = config["environment"].get("render_mode")

        env_config = config["environment"]
        env_name = env_config.get("name")
        obs_noise_std = env_config.get("obs_noise_std")
        action_noise_prob = env_config.get("action_noise_prob")

        # Create environment
        env = CartPoleEnv(
            env_name="CartPole-v1" if env_name is None else env_name,
            render_mode=render_mode,
            max_episode_steps=env_config["max_episode_steps"],
            seed=env_config["seed"],
            obs_noise_std=0.0 if obs_noise_std is None else obs_noise_std,
            action_noise_prob=0.0 if action_noise_prob is None else action_noise_prob,
            domain_randomization=env_config.get("domain_randomization"),
        )

        # Wrap with RecordVideo when --record-video is set.
        # RecordVideo captures rgb_array frames and saves them as mp4 files, so
        # it works on headless machines where no display is available.
        # CartPoleEnv.wrap_env() is used to inject the wrapper through the proper
        # interface rather than bypassing it via direct attribute assignment.
        if args.record_video:
            try:
                import moviepy  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "The 'moviepy' package is required for --record-video. "
                    "Install it with: pip install moviepy  "
                    "or: pip install -e '.[video]'"
                ) from exc
            from gymnasium.wrappers import RecordVideo

            eval_frequency = config["training"].get("eval_frequency", 100)
            os.makedirs(args.video_dir, exist_ok=True)
            env.wrap_env(
                RecordVideo(
                    env.env,
                    video_folder=args.video_dir,
                    episode_trigger=lambda ep: ep > 0 and ep % eval_frequency == 0,
                    name_prefix="cartpole-training",
                    disable_logger=True,
                )
            )
            logger.info(f"Recording videos every {eval_frequency} episodes to '{args.video_dir}'")
        logger.info("Environment created")

        # Create agent
        agent_type = config["agent"]["type"]
        agent_cls = AGENT_CLASSES.get(agent_type)
        if agent_cls is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        agent = agent_cls(
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=config["agent"]["config"],
        )

        logger.info(f"Agent created: {agent_type}")

        # Create trainer
        trainer = Trainer(
            env=env,
            agent=agent,
            config=config["training"],
            logger=logger,
        )
        logger.info("Trainer created")

        # Train
        try:
            stats = trainer.train()
            logger.info(f"Training complete. Final stats: {stats}")

            if args.plot:
                plot_path = plot_training_metrics(
                    episode_rewards=trainer.episode_rewards,
                    episode_lengths=trainer.episode_lengths,
                    save_dir=args.plot_dir,
                    title=f"CartPole {agent_type.capitalize()} Agent – Training Metrics",
                )
                logger.info(f"Training metrics plot saved to '{plot_path}'")
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
    finally:
        try:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    logger.exception("Failed to close environment")
                else:
                    logger.info("Environment closed")
        finally:
            for key, previous in sdl_env_backup.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous


if __name__ == "__main__":
    main()

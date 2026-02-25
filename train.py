"""Main training script for CartPole RL agent."""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gymnasium.wrappers import RecordVideo

from rl_cartpole.agents import RandomAgent
from rl_cartpole.environments import CartPoleEnv
from rl_cartpole.training import Trainer
from rl_cartpole.utils import load_config, setup_logger


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
        help="Render the environment during training (requires a display)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help=(
            "Record videos of training episodes to disk. "
            "Works on headless machines (e.g. NVIDIA Orin Nano via SSH) "
            "because no display is required."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./videos",
        help="Directory to save recorded videos (default: ./videos)",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Setup logger
    logger = setup_logger(
        name="rl_cartpole",
        log_dir=config["training"]["log_dir"],
    )
    logger.info("Starting CartPole RL training")
    logger.info(f"Configuration: {config}")

    # Determine render mode.
    # --record-video uses rgb_array (no display needed); --render uses human (requires display).
    if args.record_video:
        render_mode = "rgb_array"
    elif args.render:
        render_mode = "human"
    else:
        render_mode = config["environment"].get("render_mode")

    # Create environment
    env = CartPoleEnv(
        render_mode=render_mode,
        max_episode_steps=config["environment"]["max_episode_steps"],
        seed=config["environment"]["seed"],
    )

    # Wrap with RecordVideo when --record-video is set.
    # RecordVideo captures rgb_array frames and saves them as mp4 files, so
    # it works on headless machines where no display is available.
    if args.record_video:
        try:
            import moviepy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'moviepy' package is required for --record-video. "
                "Install it with: pip install moviepy  "
                "or: pip install -e '.[video]'"
            ) from exc
        # Configure pygame to render off-screen so no display is required.
        # This is needed on headless machines (e.g. SSH into NVIDIA Orin Nano).
        os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        eval_frequency = config["training"].get("eval_frequency", 100)
        os.makedirs(args.video_dir, exist_ok=True)
        env.env = RecordVideo(
            env.env,
            video_folder=args.video_dir,
            episode_trigger=lambda ep: ep > 0 and ep % eval_frequency == 0,
            name_prefix="cartpole-training",
            disable_logger=True,
        )
        logger.info(f"Recording videos every {eval_frequency} episodes to '{args.video_dir}'")
    logger.info("Environment created")

    # Create agent
    agent_type = config["agent"]["type"]
    if agent_type == "random":
        agent = RandomAgent(
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=config["agent"]["config"],
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

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
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        env.close()
        logger.info("Environment closed")


if __name__ == "__main__":
    main()

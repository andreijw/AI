"""Record a trained CartPole agent from a checkpoint into one video."""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gymnasium.wrappers import RecordVideo

from rl_cartpole.training import Trainer
from rl_cartpole.utils import setup_logger
from rl_cartpole.utils.video import merge_episode_clips
from train import (
    AGENT_CLASSES,
    agent_output_dir,
    close_env,
    create_agent,
    create_env,
    headless_rendering,
    load_run_config,
    require_moviepy,
)

PLAYBACK_NAME_PREFIX = "cartpole-playback"


def main():
    """Play evaluation episodes with a checkpointed agent and save them as one MP4."""
    parser = argparse.ArgumentParser(description="Record a trained CartPole agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cartpole_default.yaml",
        help="Configuration the checkpoint was trained with",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Agent checkpoint to load (e.g. checkpoints/ppo/agent_episode_1400.pt)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to record (default: 5)",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        help="Directory to save the video (default: ./videos/<agent_type>)",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=sorted(AGENT_CLASSES),
        help="Override the agent type from config (random, reinforce, actor_critic, ppo).",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer.")

    require_moviepy()
    config, agent_type = load_run_config(args.config, args.agent_type)
    video_dir = args.video_dir or agent_output_dir("./videos", agent_type)
    logger = setup_logger(name="rl_cartpole", log_dir=config["training"]["log_dir"])

    with headless_rendering():
        env = create_env(config, render_mode="rgb_array")
        try:
            agent = create_agent(agent_type, config, env)
            agent.load(args.checkpoint)
            logger.info(f"Loaded {agent_type} agent from '{args.checkpoint}'")

            env.wrap_env(
                RecordVideo(
                    env.env,
                    video_folder=video_dir,
                    episode_trigger=lambda _: True,
                    name_prefix=PLAYBACK_NAME_PREFIX,
                    disable_logger=True,
                )
            )
            trainer = Trainer(env=env, agent=agent, config=config["training"], logger=logger)
            stats = trainer.evaluate(num_episodes=args.episodes)
            logger.info(f"Recorded {args.episodes} episodes: {stats}")
        finally:
            # Closing writes the last episode's clip, so merging happens afterwards.
            close_env(env, logger)

    video_path = merge_episode_clips(video_dir, name_prefix=PLAYBACK_NAME_PREFIX)
    logger.info(f"Playback video saved to '{video_path}'")


if __name__ == "__main__":
    main()

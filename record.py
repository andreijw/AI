"""Record a trained agent from a checkpoint into one video, on any registered challenge."""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gymnasium.wrappers import RecordVideo

from rl_core.training import Trainer
from rl_core.utils import setup_logger
from rl_core.utils.video import merge_episode_clips
from train import (
    add_challenge_arguments,
    agent_output_dir,
    close_env,
    headless_rendering,
    load_run_config,
    require_moviepy,
    select_run_challenge,
)


def main():
    """Play evaluation episodes with a checkpointed agent and save them as one MP4."""
    parser = argparse.ArgumentParser(description="Record a trained RL agent")
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
    add_challenge_arguments(parser)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer.")

    require_moviepy()
    config, agent_type = load_run_config(args.config, args.agent_type)
    run_challenge = select_run_challenge(config, agent_type, args.challenge)
    challenge = run_challenge.key
    video_dir = args.video_dir or agent_output_dir("./videos", agent_type, challenge=challenge)
    playback_prefix = f"{challenge}-playback"
    logger = setup_logger(name=f"rl_{challenge}", log_dir=config["training"]["log_dir"])

    with headless_rendering():
        env = run_challenge.make_env(config["environment"], render_mode="rgb_array")
        try:
            agent = run_challenge.make_agent(agent_type, config["agent"]["config"], env)
            agent.load(args.checkpoint)
            logger.info(f"Loaded {agent_type} agent from '{args.checkpoint}'")

            env.wrap_env(
                RecordVideo(
                    env.env,
                    video_folder=video_dir,
                    episode_trigger=lambda _: True,
                    name_prefix=playback_prefix,
                    disable_logger=True,
                )
            )
            trainer = Trainer(env=env, agent=agent, config=config["training"], logger=logger)
            stats = trainer.evaluate(num_episodes=args.episodes)
            logger.info(f"Recorded {args.episodes} episodes: {stats}")
        finally:
            # Closing writes the last episode's clip, so merging happens afterwards.
            close_env(env, logger)

    video_path = merge_episode_clips(video_dir, name_prefix=playback_prefix)
    logger.info(f"Playback video saved to '{video_path}'")


if __name__ == "__main__":
    main()

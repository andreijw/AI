"""Train an agent on any registered challenge, chosen from the config's environment name."""

import argparse
import os
import sys
from contextlib import contextmanager

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rl_core.challenges import (
    CHALLENGES,
    BenchmarkChallenge,
    discover_challenges,
    import_failures_text,
    select_challenge,
)
from rl_core.training import Trainer
from rl_core.training.budget import apply_budget_override
from rl_core.utils import load_config, plot_training_metrics, setup_logger
from rl_core.utils.video import TrainingVideoRecorder


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
        raise ValueError(f"Config file '{config_path}' must define 'agent.type' as a string.")
    return config, agent_type


def add_challenge_arguments(parser):
    """Discover the challenges and add --agent-type and --challenge with their choices to parser."""
    discover_challenges()
    agent_types = sorted({agent for cls in CHALLENGES.values() for agent in cls().supported_agents})
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=agent_types,
        help=f"Override the agent type from config ({', '.join(agent_types)}).",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        choices=sorted(CHALLENGES),
        default=None,
        help=(
            "Challenge to run (default: the one whose environment the config names; "
            "required when the config has no environment.name)."
        ),
    )
    parser.epilog = import_failures_text() or None


def select_run_challenge(config, agent_type, key=None) -> BenchmarkChallenge:
    """The challenge for config's environment.name and key, checked to support agent_type."""
    challenge = select_challenge(config["environment"].get("name"), key)()
    if agent_type not in challenge.supported_agents:
        raise challenge.unsupported_agent_error(agent_type)
    return challenge


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
    parser = argparse.ArgumentParser(description="Train an RL agent on a registered challenge")
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
    add_challenge_arguments(parser)
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
    run_challenge = select_run_challenge(config, agent_type, args.challenge)
    challenge = run_challenge.key

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
            env = run_challenge.make_env(config["environment"], render_mode)
            if args.record_video:
                require_moviepy()
            logger.info("Environment created")

            agent = run_challenge.make_agent(agent_type, config["agent"]["config"], env)
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
            # counter, so it must be created after the trainer. The wrapper's wrap_env()
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

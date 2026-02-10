"""Main training script for CartPole RL agent."""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

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
        help="Render the environment during training",
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
    
    # Create environment
    render_mode = "human" if args.render else config["environment"].get("render_mode")
    
    # Prepare domain randomization config if enabled
    domain_randomization = None
    dr_config = config["environment"].get("domain_randomization", {})
    if dr_config.get("enabled", False):
        domain_randomization = {}
        if "gravity" in dr_config:
            domain_randomization["gravity"] = (
                dr_config["gravity"]["min"],
                dr_config["gravity"]["max"]
            )
        if "pole_length" in dr_config:
            domain_randomization["pole_length"] = (
                dr_config["pole_length"]["min"],
                dr_config["pole_length"]["max"]
            )
        if "cart_mass" in dr_config:
            domain_randomization["cart_mass"] = (
                dr_config["cart_mass"]["min"],
                dr_config["cart_mass"]["max"]
            )
    
    env = CartPoleEnv(
        render_mode=render_mode,
        max_episode_steps=config["environment"]["max_episode_steps"],
        seed=config["environment"]["seed"],
        observation_noise_std=config["environment"].get("observation_noise_std", 0.0),
        action_noise_std=config["environment"].get("action_noise_std", 0.0),
        domain_randomization=domain_randomization,
    )
    logger.info("Environment created")
    logger.info(f"Environment augmentation: obs_noise={config['environment'].get('observation_noise_std', 0.0)}, "
                f"action_noise={config['environment'].get('action_noise_std', 0.0)}, "
                f"domain_rand={'enabled' if domain_randomization else 'disabled'}")
    
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

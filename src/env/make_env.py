"""
Factory functions for creating CartPole environments (single and vectorized).
"""

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from typing import Optional, Dict, Tuple, Callable
import yaml

from .cartpole_env import CartPoleEnv


def make_single_env(
    env_name: str = "CartPole-v1",
    render_mode: Optional[str] = None,
    observation_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
    domain_randomization: Optional[Dict[str, Tuple[float, float]]] = None,
) -> gym.Env:
    """
    Create a single CartPole environment with optional noise and domain randomization.
    
    Args:
        env_name: Name of the Gymnasium environment
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        observation_noise_std: Standard deviation for observation noise
        action_noise_std: Standard deviation for action noise
        domain_randomization: Dictionary with physics parameter ranges
        
    Returns:
        Configured CartPole environment
    """
    # Create base environment
    env = gym.make(env_name, render_mode=render_mode)
    
    # Wrap with our custom wrapper if any noise or randomization is enabled
    if observation_noise_std > 0 or action_noise_std > 0 or domain_randomization:
        env = CartPoleEnv(
            env,
            observation_noise_std=observation_noise_std,
            action_noise_std=action_noise_std,
            domain_randomization=domain_randomization,
        )
    
    return env


def make_vec_env(
    num_envs: int = 4,
    env_name: str = "CartPole-v1",
    render_mode: Optional[str] = None,
    observation_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
    domain_randomization: Optional[Dict[str, Tuple[float, float]]] = None,
    async_envs: bool = False,
) -> gym.vector.VectorEnv:
    """
    Create vectorized CartPole environments for parallel training.
    
    Args:
        num_envs: Number of parallel environments
        env_name: Name of the Gymnasium environment
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        observation_noise_std: Standard deviation for observation noise
        action_noise_std: Standard deviation for action noise
        domain_randomization: Dictionary with physics parameter ranges
        async_envs: Whether to use asynchronous vectorization (multiprocessing)
        
    Returns:
        Vectorized environment
    """
    def env_fn() -> gym.Env:
        """Factory function for creating individual environments."""
        return make_single_env(
            env_name=env_name,
            render_mode=render_mode,
            observation_noise_std=observation_noise_std,
            action_noise_std=action_noise_std,
            domain_randomization=domain_randomization,
        )
    
    # Create list of environment factory functions
    env_fns = [env_fn for _ in range(num_envs)]
    
    # Create vectorized environment
    if async_envs:
        vec_env = AsyncVectorEnv(env_fns)
    else:
        vec_env = SyncVectorEnv(env_fns)
    
    return vec_env


def make_env_from_config(config_path: str) -> gym.Env:
    """
    Create environment from YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configured environment (single or vectorized)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract configuration
    env_name = config.get('env_name', 'CartPole-v1')
    render_mode = config.get('render_mode', None)
    num_envs = config.get('num_envs', 1)
    
    # Observation noise
    obs_noise_config = config.get('observation_noise', {})
    obs_noise_std = obs_noise_config.get('std', 0.0) if obs_noise_config.get('enabled', False) else 0.0
    
    # Action noise
    action_noise_config = config.get('action_noise', {})
    action_noise_std = action_noise_config.get('std', 0.0) if action_noise_config.get('enabled', False) else 0.0
    
    # Domain randomization
    dr_config = config.get('domain_randomization', {})
    domain_randomization = None
    if dr_config.get('enabled', False):
        domain_randomization = {}
        if 'gravity' in dr_config:
            domain_randomization['gravity'] = (dr_config['gravity']['min'], dr_config['gravity']['max'])
        if 'pole_length' in dr_config:
            domain_randomization['pole_length'] = (dr_config['pole_length']['min'], dr_config['pole_length']['max'])
        if 'cart_mass' in dr_config:
            domain_randomization['cart_mass'] = (dr_config['cart_mass']['min'], dr_config['cart_mass']['max'])
    
    # Create environment
    if num_envs > 1:
        return make_vec_env(
            num_envs=num_envs,
            env_name=env_name,
            render_mode=render_mode,
            observation_noise_std=obs_noise_std,
            action_noise_std=action_noise_std,
            domain_randomization=domain_randomization,
        )
    else:
        return make_single_env(
            env_name=env_name,
            render_mode=render_mode,
            observation_noise_std=obs_noise_std,
            action_noise_std=action_noise_std,
            domain_randomization=domain_randomization,
        )

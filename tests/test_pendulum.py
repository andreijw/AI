"""Unit tests for the Pendulum-v1 environment wrapper, vectorization, and challenge."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from benchmark import CHALLENGES
from rl_pendulum.benchmark import PendulumChallenge
from rl_pendulum.environments import (
    PendulumEnv,
    make_env,
    make_env_from_config,
    make_vec_env,
    make_vec_env_from_config,
)


def test_pendulum_env_basic_step_and_reset():
    """Verify standard initialization, reset, and step transitions for PendulumEnv."""
    env = PendulumEnv(seed=42)
    obs, info = env.reset()

    assert obs.shape == (3,)
    assert isinstance(info, dict)
    assert env.episode_steps == 0
    assert env.episode_reward == 0.0

    # Step with continuous torque action
    action = np.array([1.5], dtype=np.float32)
    obs_next, reward, terminated, truncated, info_step = env.step(action)

    assert obs_next.shape == (3,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.episode_steps == 1
    assert env.episode_reward == reward

    # Step with scalar float action
    obs_next2, reward2, _, _, _ = env.step(0.0)
    assert obs_next2.shape == (3,)
    assert env.episode_steps == 2
    assert env.episode_reward == reward + reward2

    env.close()


def test_pendulum_env_observation_noise():
    """Verify Gaussian observation noise injection."""
    seed = 100
    env_clean = PendulumEnv(seed=seed, obs_noise_std=0.0)
    env_noisy = PendulumEnv(seed=seed, obs_noise_std=0.5)

    obs_clean, _ = env_clean.reset(seed=seed)
    obs_noisy, _ = env_noisy.reset(seed=seed)

    # Clean and noisy should differ because noise is added
    assert not np.allclose(obs_clean, obs_noisy)

    env_clean.close()
    env_noisy.close()


def test_pendulum_env_action_noise_and_clipping():
    """Verify action noise perturbation and clipping within bounds [-2.0, 2.0]."""
    env = PendulumEnv(seed=42, action_noise_std=10.0)
    env.reset(seed=42)

    # Extreme action should be clipped after noise
    noisy_action = env._apply_action_noise(np.array([2.0], dtype=np.float32))
    assert -2.0 <= float(noisy_action[0]) <= 2.0

    env.close()


def test_pendulum_domain_randomization_validation():
    """Verify validation of domain randomization configuration."""
    with pytest.raises(ValueError, match="domain_randomization must be a mapping"):
        PendulumEnv(domain_randomization="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported domain randomization parameter"):
        PendulumEnv(domain_randomization={"unknown_param": (1.0, 2.0)})

    with pytest.raises(ValueError, match="must be a 2-element sequence"):
        PendulumEnv(domain_randomization={"g": (1.0, 2.0, 3.0)})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="must contain numeric values"):
        PendulumEnv(domain_randomization={"g": ("min", "max")})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="has min > max"):
        PendulumEnv(domain_randomization={"g": (12.0, 8.0)})

    with pytest.raises(ValueError, match="must be strictly positive"):
        PendulumEnv(domain_randomization={"g": (-1.0, 5.0)})


def test_pendulum_domain_randomization_application():
    """Verify domain randomization successfully updates physics attributes on reset."""
    dr_config = {
        "gravity": (5.0, 15.0),
        "mass": (0.5, 2.0),
        "length": (0.5, 2.0),
    }
    env = PendulumEnv(seed=42, domain_randomization=dr_config)
    env.reset(seed=42)

    unwrapped = env.env.unwrapped
    assert 5.0 <= unwrapped.g <= 15.0
    assert 0.5 <= unwrapped.m <= 2.0
    assert 0.5 <= unwrapped.l <= 2.0

    env.close()


def test_pendulum_wrap_env_and_metadata():
    """Verify wrap_env and metadata exposure."""
    env = PendulumEnv()
    assert env.metadata is not None
    assert env.observation_space.shape == (3,)

    # Wrap dummy
    dummy = gym.Wrapper(env.env)
    env.wrap_env(dummy)
    assert env.env == dummy

    env.close()


def test_pendulum_factory_make_env_and_configs():
    """Verify make_env and make_env_from_config factory functions."""
    config = {
        "name": "Pendulum-v1",
        "max_episode_steps": 150,
        "seed": 42,
        "obs_noise_std": 0.05,
        "action_noise_std": 0.02,
        "domain_randomization": {"g": (9.0, 11.0)},
    }
    env = make_env_from_config(config)
    obs, _ = env.reset()
    assert obs.shape == (3,)
    assert env.max_episode_steps == 150
    assert env.obs_noise_std == 0.05
    assert env.action_noise_std == 0.02
    assert env.domain_randomization == {"g": (9.0, 11.0)}
    env.close()

    env_direct = make_env(env_name="Pendulum-v1", seed=123)
    assert env_direct.seed == 123
    env_direct.close()


def test_pendulum_vectorized_environments():
    """Verify make_vec_env and make_vec_env_from_config create functional sync/async vector envs."""
    # Sync Vector Env
    vec_sync = make_vec_env(num_envs=2, seed=42, async_envs=False)
    obs_sync, _ = vec_sync.reset()
    assert obs_sync.shape == (2, 3)

    actions = np.zeros((2, 1), dtype=np.float32)
    next_obs, rewards, dones, truncs, _ = vec_sync.step(actions)
    assert next_obs.shape == (2, 3)
    assert rewards.shape == (2,)
    vec_sync.close()

    # Async Vector Env via config
    cfg = {"name": "Pendulum-v1", "seed": 100, "async_envs": False}
    vec_cfg = make_vec_env_from_config(cfg, num_envs=2)
    obs_cfg, _ = vec_cfg.reset()
    assert obs_cfg.shape == (2, 3)
    vec_cfg.close()


def test_pendulum_benchmark_challenge_registration_and_setup():
    """Verify PendulumChallenge is registered and properly sets up agents."""
    assert "pendulum" in CHALLENGES
    challenge = PendulumChallenge(config_dir="configs")

    assert challenge.name == "Pendulum-v1"
    assert challenge.key == "pendulum"
    assert challenge.solve_threshold == -200.0
    assert challenge.default_episodes == 200
    assert set(challenge.supported_agents) == {"actor_critic", "ppo", "random"}

    # Setup run for PPO
    env_ppo, agent_ppo, train_cfg_ppo = challenge.setup_run("ppo", seed=42, episodes=10)
    assert isinstance(env_ppo, PendulumEnv)
    assert train_cfg_ppo["num_episodes"] == 10
    env_ppo.close()

    # Setup run for Actor-Critic
    env_ac, agent_ac, _ = challenge.setup_run("actor_critic", seed=42, episodes=5)
    assert isinstance(env_ac, PendulumEnv)
    env_ac.close()

    # Setup run for Random
    env_rand, agent_rand, _ = challenge.setup_run("random", seed=42, episodes=5)
    assert isinstance(env_rand, PendulumEnv)
    env_rand.close()


def test_pendulum_env_passes_the_inner_env_a_bounded_action():
    """The wrapper clips before stepping; Gymnasium's own clip must not be what keeps torque bounded."""
    env = PendulumEnv(seed=1)
    env.reset(seed=1)
    sent = []
    inner_step = env.env.step
    env.env.step = lambda action: (sent.append(np.array(action)), inner_step(action))[1]
    env.step(np.array([5.0], dtype=np.float32))
    env.step(np.array([-7.5], dtype=np.float32))
    env.close()
    np.testing.assert_allclose(np.concatenate(sent), [2.0, -2.0])


def test_pendulum_env_clips_before_adding_action_noise():
    """An out-of-range action still gets noise: clipping only after the noise would swallow it."""
    env = PendulumEnv(seed=1, action_noise_std=0.5)
    env.reset(seed=1)
    torques = []
    for _ in range(20):
        env.step(np.array([5.0], dtype=np.float32))
        torques.append(float(env.env.unwrapped.last_u))
    env.close()
    assert all(-2.0 <= t <= 2.0 for t in torques)
    assert min(torques) < 1.9, "Noise must still move the torque below the limit sometimes."

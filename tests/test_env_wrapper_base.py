"""The shared environment wrapper base (rl_core.environments.BaseEnvWrapper).

Seeding, the challenge-specific noise and domain randomization are covered for both challenges in
``tests/test_env_seeding.py``, ``tests/test_environment.py`` and ``tests/test_pendulum.py``. This file
checks the contract a challenge wrapper builds on: the action hook, the order of the reset steps,
observation noise and the defaults a subclass need not override.
"""

import gymnasium as gym
import numpy as np
import pytest

from rl_core.environments import BaseEnvWrapper, checked_range


class _ActionLog(gym.Wrapper):
    """Records the actions the inner environment receives."""

    def __init__(self, env):
        super().__init__(env)
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return self.env.step(action)


class _Plain(BaseEnvWrapper):
    """A wrapper that overrides nothing but its constructor."""

    def __init__(self, seed=None, obs_noise_std=0.0, max_episode_steps=5):
        super().__init__("CartPole-v1", None, max_episode_steps, seed, obs_noise_std)
        self._make_env()


class _Hooked(_Plain):
    """Flips every action and records the order of the hooks."""

    def __init__(self, **kwargs):
        self.calls = []
        super().__init__(**kwargs)

    def _prepare_action(self, action):
        self.calls.append(("prepare", action))
        return 1 - action

    def _apply_domain_randomization(self):
        self.calls.append("randomize")

    def _add_observation_noise(self, obs):
        self.calls.append("noise")
        return super()._add_observation_noise(obs)


def test_the_prepared_action_reaches_the_inner_environment():
    env = _Hooked(seed=0)
    log = _ActionLog(env.env)
    env.wrap_env(log)
    env.reset()
    env.step(0)
    env.step(1)
    assert log.actions == [1, 0]
    assert ("prepare", 0) in env.calls and ("prepare", 1) in env.calls
    env.close()


def test_reset_randomizes_before_adding_observation_noise():
    env = _Hooked(seed=0)
    env.reset()
    assert env.calls == ["randomize", "noise"]
    env.close()


def test_observation_noise_changes_the_observation_and_keeps_its_dtype():
    clean, noisy = _Plain(seed=3), _Plain(seed=3, obs_noise_std=0.1)
    clean_obs, _ = clean.reset()
    noisy_obs, _ = noisy.reset()
    assert noisy_obs.dtype == clean_obs.dtype == np.float32
    assert not np.array_equal(noisy_obs, clean_obs)
    clean.close()
    noisy.close()


def test_a_wrapper_without_overrides_plays_an_episode_and_reports_it():
    env = _Plain(seed=1, max_episode_steps=3)
    env.reset()
    total, done, info = 0.0, False, {}
    while not done:
        _, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)
        total += reward
        done = terminated or truncated
    # Pushing left from a seeded start lasts longer than three steps, so the time limit ends it.
    assert info["episode"] == {"steps": 3, "reward": total}
    assert env.episode_steps == 3
    env.close()


def test_spaces_and_metadata_are_the_inner_environments():
    env = _Plain()
    assert env.metadata is env.env.metadata  # vector environments write into this dict
    assert env.action_space is env.env.action_space
    assert env.observation_space is env.env.observation_space
    env.close()


def test_the_constructor_seed_seeds_the_action_space():
    env = _Plain(seed=5)
    reference = gym.spaces.Discrete(2, seed=5)
    assert [env.action_space.sample() for _ in range(32)] == [reference.sample() for _ in range(32)]
    env.close()


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_observation_noise_must_be_finite_and_non_negative(bad):
    with pytest.raises(ValueError, match="obs_noise_std must be a finite, non-negative number"):
        _Plain(obs_noise_std=bad)


def test_checked_range_returns_floats():
    assert checked_range("gravity", 8, 12.5) == (8.0, 12.5)
    assert checked_range("gravity", 9.8, 9.8) == (9.8, 9.8)  # a range of one value pins it
    assert all(type(bound) is float for bound in checked_range("gravity", 8, 12))


@pytest.mark.parametrize(
    "low,high,message",
    [
        ("8", 12.0, "must contain numeric values"),
        (8.0, float("inf"), "must use finite values"),
        (12.0, 8.0, "has min > max"),
    ],
)
def test_checked_range_rejects_bad_ranges_naming_the_parameter(low, high, message):
    with pytest.raises(ValueError, match=f"range for 'gravity' {message}"):
        checked_range("gravity", low, high)

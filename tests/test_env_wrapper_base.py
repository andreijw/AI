"""The shared environment wrapper base (rl_core.environments.BaseEnvWrapper).

Seeding, the challenge-specific noise and domain randomization are covered for both challenges in
``tests/test_env_seeding.py``, ``tests/test_environment.py`` and ``tests/test_pendulum.py``. This file
checks the contract a challenge wrapper builds on: the action hook, the order of the reset steps,
observation noise and the defaults a subclass need not override.
"""

import sys

import gymnasium as gym
import numpy as np
import pytest

from rl_core.environments import BaseEnvWrapper, checked_range, uses_box2d
from rl_core.seeding import WRAPPER_STREAM, derive_rng


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


class _ProbeEnv(gym.Env):
    """Reads ``gain`` in ``reset`` and keeps what it saw; ``option`` is a make() keyword."""

    observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, render_mode=None, option="default"):
        self.option = option
        self.gain = -1.0
        self.gains_seen = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.gains_seen.append(self.gain)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        return np.zeros(1, np.float32), 0.0, False, False, {}


gym.register("ProbeEnv-v0", entry_point=_ProbeEnv)


class _Randomized(BaseEnvWrapper):
    """Draws ``gain`` from the wrapper generator at the start of each episode."""

    def __init__(self, seed=None, **make_kwargs):
        super().__init__("ProbeEnv-v0", None, 5, seed, 0.0)
        self._make_env(**make_kwargs)

    def _apply_domain_randomization(self):
        self.env.unwrapped.gain = float(self._np_random.uniform(1.0, 2.0))


def test_the_inner_reset_sees_the_gain_drawn_for_its_own_episode():
    env = _Randomized(seed=7)
    for _ in range(3):
        env.reset()
    rng = derive_rng(7, WRAPPER_STREAM)
    drawn = [float(rng.uniform(1.0, 2.0)) for _ in range(3)]
    assert env.env.unwrapped.gains_seen == drawn
    env.close()


def test_a_reset_seed_restarts_the_draws_the_inner_reset_sees():
    env = _Randomized(seed=7)
    env.reset()
    env.reset(seed=3)
    expected = float(derive_rng(3, WRAPPER_STREAM).uniform(1.0, 2.0))
    assert env.env.unwrapped.gains_seen[-1] == expected
    env.close()


def test_extra_make_keywords_reach_gym_make():
    custom, default = _Randomized(option="custom"), _Randomized()
    assert custom.env.unwrapped.option == "custom"
    assert default.env.unwrapped.option == "default"
    custom.close()
    default.close()


class _Box2DPlain(BaseEnvWrapper):
    """A wrapper on one of Gymnasium's Box2D environments that overrides nothing else."""

    def __init__(self, env_name):
        super().__init__(env_name, None, 5, 0, 0.0)
        self._make_env()


@pytest.mark.parametrize("env_name", ["LunarLander-v3", "BipedalWalker-v3", "CarRacing-v3"])
def test_without_box2d_a_box2d_environment_names_it_and_this_projects_install(
    monkeypatch, env_name
):
    """Any wrapper on a Box2D environment, by its registered entry point, gets the hint that
    works on the Jetson (./setup.sh builds box2d-py), not Gymnasium's own."""
    monkeypatch.setitem(sys.modules, "Box2D", None)  # import Box2D now fails, installed or not
    with pytest.raises(ImportError, match=rf"^{env_name} needs Box2D: run \./setup\.sh.*box2d-py"):
        _Box2DPlain(env_name)


def test_an_id_the_registry_does_not_know_is_left_to_gym_make():
    """``gym.make`` also takes "module:Env-vN" ids; the Box2D check must not refuse what it would
    accept, nor turn its own error into a different one."""
    assert uses_box2d("gymnasium.envs.box2d:LunarLander-v3") is False
    assert uses_box2d("NoSuchEnv-v0") is False


def test_without_box2d_an_environment_that_does_not_use_it_still_builds(monkeypatch):
    monkeypatch.setitem(sys.modules, "Box2D", None)
    env = _Plain(seed=0)
    try:
        assert env.reset()[0].shape == (4,), "CartPole needs no Box2D"
    finally:
        env.close()


def test_the_box2d_install_hint_setup_and_the_extra_name_one_box2d_py_version():
    """rl-challenges' ``box2d`` extra, the version ./setup.sh builds and the hint's manual command must
    agree: setup.sh installs the extras after building Box2D, so a different pin there makes pip
    replace the build. (A Dependabot bump of the extra alone left them apart, 2026-11.)"""
    import re
    import tomllib
    from pathlib import Path

    from rl_core.environments.base_env import BOX2D_INSTALL_HINT

    root = Path(__file__).resolve().parents[1]
    pyproject = root / "packages" / "rl-challenges" / "pyproject.toml"
    extra = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    (pin,) = [
        dep
        for dep in extra["project"]["optional-dependencies"]["box2d"]
        if dep.startswith("box2d-py")
    ]
    assert re.fullmatch(r"box2d-py==[\d.]+", pin), "an exact pin"
    built = re.findall(r"box2d-py==[\d.]+", (root / "setup.sh").read_text(encoding="utf-8"))
    assert built == [pin], "setup.sh builds exactly the extra's version"
    assert re.findall(r"box2d-py==[\d.]+", BOX2D_INSTALL_HINT) == [pin]

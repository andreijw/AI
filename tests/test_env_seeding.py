"""Seeding contract shared by the challenge environment wrappers.

A seeded wrapper yields a reproducible sequence of *different* episodes: the constructor seed
is applied on the first reset, later unseeded resets continue the generators, and an explicit
``reset(seed=...)`` restarts the sequence. The wrapper's noise and domain-randomization stream
is separate from the base environment's.
"""

import gymnasium as gym
import numpy as np
import pytest

from rl_acrobot.environments import AcrobotEnv
from rl_cartpole.environments import CartPoleEnv
from rl_cartpole.environments import make_vec_env as make_cartpole_vec_env
from rl_core.agents import BaseAgent
from rl_core.seeding import derive_rng
from rl_core.training import Trainer
from rl_lunarlander.environments import LunarLanderEnv
from rl_mountaincar.environments import MountainCarEnv
from rl_pendulum.environments import PendulumEnv
from tests.helpers import env_marks


def lunarlander_continuous_env(**kwargs) -> LunarLanderEnv:
    """The LunarLander wrapper on LunarLanderContinuous-v3."""
    return LunarLanderEnv(env_name="LunarLanderContinuous-v3", **kwargs)


# Each wrapper's constructor (or a factory taking the same keywords), marked as its environment
# needs: the Box2D ones build only where box2d-py is installed.
ENV_CLASSES = [
    CartPoleEnv,
    PendulumEnv,
    AcrobotEnv,
    MountainCarEnv,
    pytest.param(LunarLanderEnv, marks=env_marks("LunarLander-v3"), id="LunarLanderEnv"),
    pytest.param(
        lunarlander_continuous_env,
        marks=env_marks("LunarLanderContinuous-v3"),
        id="LunarLanderContinuous",
    ),
]
GRAVITY_RANGE = {CartPoleEnv: ("gravity", (5.0, 15.0)), PendulumEnv: ("g", (5.0, 15.0))}
SEED = 42
NOISE_STD = 0.1


def _action(env, step: int):
    """A fixed, valid action for the given step of an episode, from the environment's action space:
    the actions in turn when discrete, else 0.5 and -1.0 in turn in every dimension."""
    if isinstance(env.action_space, gym.spaces.Discrete):
        return step % int(env.action_space.n)
    return np.full(env.action_space.shape, 0.5 if step % 2 == 0 else -1.0, dtype=np.float32)


def _play(env, episodes: int = 3, steps: int = 4) -> list[np.ndarray]:
    """Reset and step through several episodes; return every observation and reward seen."""
    seen: list[np.ndarray] = []
    for _ in range(episodes):
        obs, _ = env.reset()
        seen.append(np.asarray(obs, dtype=np.float64))
        for step in range(steps):
            obs, reward, terminated, truncated, _ = env.step(_action(env, step))
            seen.append(np.append(np.asarray(obs, dtype=np.float64), reward))
            if terminated or truncated:
                break
    return seen


def _reset_observations(env, count: int = 3) -> list[np.ndarray]:
    return [env.reset()[0].copy() for _ in range(count)]


def _all_differ(arrays: list[np.ndarray]) -> bool:
    return all(
        not np.array_equal(arrays[i], arrays[j])
        for i in range(len(arrays))
        for j in range(i + 1, len(arrays))
    )


def _base_gravity(env) -> float:
    base = env.env.unwrapped
    return float(base.gravity if hasattr(base, "gravity") else base.g)


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_seeded_env_starts_each_episode_from_a_different_state(env_cls):
    """Consecutive resets of a seeded environment give different start states."""
    env = env_cls(seed=SEED)
    try:
        observations = _reset_observations(env)
    finally:
        env.close()

    assert _all_differ(observations), f"A seeded {env_cls.__name__} replayed a start state."


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_two_envs_with_the_same_seed_agree_episode_by_episode(env_cls):
    """The sequence of episodes is reproducible from the constructor seed."""
    first, second = env_cls(seed=SEED), env_cls(seed=SEED)
    try:
        seen_first, seen_second = _play(first), _play(second)
    finally:
        first.close()
        second.close()

    assert len(seen_first) == len(seen_second)
    for index, (a, b) in enumerate(zip(seen_first, seen_second, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"Same-seed runs diverged at step {index}.")


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_seed_zero_is_a_seed_like_any_other(env_cls):
    """Zero is falsy but is a real seed, both in the constructor and passed to reset()."""
    first, second = env_cls(seed=0), env_cls(seed=0)
    other = env_cls(seed=SEED)
    try:
        seen_first, seen_second = _reset_observations(first), _reset_observations(second)
        restarted = other.reset(seed=0)[0].copy()
    finally:
        first.close()
        second.close()
        other.close()

    assert _all_differ(seen_first)
    for a, b in zip(seen_first, seen_second, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(restarted, seen_first[0])


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_invalid_seed_is_rejected_at_construction(env_cls):
    """A seed that is not a non-negative integer fails before any environment is created."""
    with pytest.raises(ValueError, match="seed"):
        env_cls(seed=-1)


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_different_seeds_give_different_first_states(env_cls):
    """The constructor seed is actually applied on the first reset."""
    first, second = env_cls(seed=1), env_cls(seed=2)
    try:
        obs_first, obs_second = first.reset()[0], second.reset()[0]
    finally:
        first.close()
        second.close()

    assert not np.array_equal(obs_first, obs_second)


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_explicit_seed_restarts_the_sequence_including_noise(env_cls):
    """reset(seed=s) restarts the start states and the wrapper's noise stream, at any time."""
    env = env_cls(seed=SEED, obs_noise_std=NOISE_STD)
    try:
        first_start = env.reset(seed=5)[0].copy()
        first_follow_up = env.reset()[0].copy()
        env.reset()  # move further along before restarting
        second_start = env.reset(seed=5)[0].copy()
        second_follow_up = env.reset()[0].copy()
    finally:
        env.close()

    np.testing.assert_array_equal(first_start, second_start)
    np.testing.assert_array_equal(
        first_follow_up, second_follow_up, err_msg="The episode after a reseed was not replayed."
    )
    assert not np.array_equal(first_start, first_follow_up)


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_observation_noise_continues_across_episodes(env_cls):
    """Noise is drawn afresh each episode, and reproducibly for the same seed."""
    clean = env_cls(seed=SEED)
    noisy, noisy_twin = (env_cls(seed=SEED, obs_noise_std=NOISE_STD) for _ in range(2))
    try:
        clean_obs = _reset_observations(clean)
        noisy_obs = _reset_observations(noisy)
        twin_obs = _reset_observations(noisy_twin)
    finally:
        clean.close()
        noisy.close()
        noisy_twin.close()

    noise = [
        n.astype(np.float64) - c.astype(np.float64)
        for n, c in zip(noisy_obs, clean_obs, strict=True)
    ]
    assert all(np.any(np.abs(n) > 1e-6) for n in noise), "No noise was added."
    assert all(n.dtype == c.dtype for n, c in zip(noisy_obs, clean_obs, strict=True)), (
        "Noise should keep the observation's dtype."
    )
    assert not np.allclose(noise[0], noise[1], atol=1e-6), "The same noise was added twice."
    for a, b in zip(noisy_obs, twin_obs, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_wrapper_noise_stream_is_separate_from_the_base_environment(env_cls):
    """The wrapper must not reuse the very numbers that produced the start state."""
    clean = env_cls(seed=SEED)
    noisy = env_cls(seed=SEED, obs_noise_std=NOISE_STD)
    try:
        clean_obs, noisy_obs = clean.reset()[0], noisy.reset()[0]
    finally:
        clean.close()
        noisy.close()

    noise = noisy_obs.astype(np.float64) - clean_obs.astype(np.float64)
    base_stream_noise = np.random.default_rng(SEED).normal(0, NOISE_STD, clean_obs.shape)
    assert not np.allclose(noise, base_stream_noise, atol=1e-5), (
        "Observation noise came from the same stream as the base environment's start state."
    )


@pytest.mark.parametrize("env_cls", sorted(GRAVITY_RANGE, key=lambda cls: cls.__name__))
def test_domain_randomization_resamples_each_episode(env_cls):
    """A seeded environment samples different physics per episode, reproducibly."""
    key, value_range = GRAVITY_RANGE[env_cls]
    first, second = (env_cls(seed=SEED, domain_randomization={key: value_range}) for _ in range(2))
    try:
        gravity_first, gravity_second = [], []
        for _ in range(3):
            first.reset()
            second.reset()
            gravity_first.append(_base_gravity(first))
            gravity_second.append(_base_gravity(second))
    finally:
        first.close()
        second.close()

    assert len(set(gravity_first)) == 3, f"Gravity was resampled to the same value: {gravity_first}"
    assert gravity_first == gravity_second


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_unseeded_env_still_gives_different_start_states(env_cls):
    """Without a seed, consecutive resets differ, as they always have."""
    env = env_cls(seed=None)
    try:
        observations = _reset_observations(env)
    finally:
        env.close()

    assert _all_differ(observations)


class _ConstantActionAgent(BaseAgent):
    """Deterministic policy: always pushes the cart the same way."""

    def __init__(self):
        super().__init__(observation_dim=4, action_dim=2, config={})

    def select_action(self, observation, training=True):
        return 1

    def update(self, batch):
        return {}

    def _checkpoint_arrays(self):
        return {}

    def _load_checkpoint_arrays(self, data):
        pass


def test_evaluation_episodes_are_not_one_episode_repeated(tmp_path):
    """With a deterministic policy, seeded evaluation episodes still differ from one another."""
    env = CartPoleEnv(seed=SEED, max_episode_steps=50)
    config = {"num_episodes": 1, "checkpoint_dir": str(tmp_path / "ckpt")}
    try:
        stats = Trainer(env=env, agent=_ConstantActionAgent(), config=config).evaluate(
            num_episodes=8
        )
    finally:
        env.close()

    assert stats["std_reward"] > 0.0, "Every evaluation episode had the same reward."


def test_vectorized_sub_environment_gets_a_new_state_after_autoreset():
    """A seeded sub-environment does not return to its first state when it resets itself."""
    vec_env = make_cartpole_vec_env(num_envs=2, seed=SEED, max_episode_steps=5, async_envs=False)
    try:
        first_obs, _ = vec_env.reset()
        actions = np.zeros(2, dtype=np.int64)
        later_obs = []
        episodes_finished = 0
        for _ in range(16):
            obs, _, terminated, truncated, _ = vec_env.step(actions)
            later_obs.append(obs[0].copy())
            episodes_finished += int(terminated[0] or truncated[0])
    finally:
        vec_env.close()

    # Whichever autoreset mode Gymnasium uses, a replayed start state would show up again here.
    assert episodes_finished >= 2, "Sub-environment 0 did not reset itself during the test."
    assert not any(np.array_equal(obs, first_obs[0]) for obs in later_obs), (
        "The sub-environment returned to its first state after an autoreset."
    )


# ---------------------------------------------------------------------------
# derive_rng
# ---------------------------------------------------------------------------


def test_derive_rng_is_reproducible_for_a_seed_and_stream():
    a, b = derive_rng(7, stream=1), derive_rng(7, stream=1)

    np.testing.assert_array_equal(a.random(5), b.random(5))


def test_derive_rng_differs_from_the_plain_seeded_generator_and_between_streams():
    plain = np.random.default_rng(7).random(5)
    stream_one = derive_rng(7, stream=1).random(5)
    stream_two = derive_rng(7, stream=2).random(5)

    assert not np.array_equal(stream_one, plain)
    assert not np.array_equal(stream_one, stream_two)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_derive_rng_never_equals_a_plainly_seeded_generator(seed):
    """No (seed, stream) pair may reproduce default_rng(n) for a nearby n, seed 0 included."""
    derived = derive_rng(seed, stream=1).random(5)

    for plain_seed in (0, 1, seed, seed + 1):
        assert not np.array_equal(derived, np.random.default_rng(plain_seed).random(5)), (
            f"derive_rng({seed}, 1) draws the same numbers as default_rng({plain_seed})."
        )


@pytest.mark.parametrize("bad_stream", [-1, 1.5, "1", True, None])
def test_derive_rng_rejects_invalid_streams(bad_stream):
    with pytest.raises(ValueError, match="stream"):
        derive_rng(7, stream=bad_stream)


def test_derive_rng_differs_between_seeds():
    assert not np.array_equal(derive_rng(7, stream=1).random(5), derive_rng(8, stream=1).random(5))


def test_derive_rng_without_a_seed_is_not_reproducible():
    assert not np.array_equal(
        derive_rng(None, stream=1).random(5), derive_rng(None, stream=1).random(5)
    )


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "7", True])
def test_derive_rng_rejects_invalid_seeds(bad_seed):
    with pytest.raises(ValueError, match="seed"):
        derive_rng(bad_seed, stream=1)

"""The noise and domain-randomization wrappers' seeded episodes, as recorded on 2026-11-03.

The wrapper tests elsewhere compare two runs of the same code, so a change to the order in which a
wrapper draws its noise or its physics would agree with itself there. This file compares against
arrays recorded before the wrappers' settings became changeable between episodes (the curriculum
work): CartPole with observation noise, action flips and all four randomized parameters (CartPole
draws them in a fixed order whatever the mapping's order); Pendulum with observation and action
noise and its randomization keys out of order and under their long names (Pendulum draws in the
mapping's order, so this case guards the order of draws); and the continuous navigation
environment with action noise and randomized slip, sensor noise and dropout (recorded on 2026-11-06
before the state-estimation work, which must leave the simulator's draws alone). Each case plays
three seeded episodes with a fixed action sequence and records every observation, reward and the
physics drawn for each episode.

Numbers agree to a relative 1e-12, as in tests/test_dqn_golden.py: the file was recorded on one
machine (aarch64). On the recording machine, ENV_GOLDEN_EXACT=1 demands every array bit for bit.

A deliberate change regenerates the file: run this module with RECORD_ENV_GOLDEN=1, check why every
changed array changed, then run it again without the variable (a recording run fails on purpose).
"""

import os
from operator import attrgetter
from pathlib import Path

import numpy as np
import pytest

from rl_cartpole.environments.cartpole_env import CartPoleEnv
from rl_navigation.environments.navigation_env import NavigationEnv
from rl_pendulum.environments.pendulum_env import PendulumEnv

GOLDEN = Path(__file__).parent / "data" / "env_wrapper_trajectories.npz"
RECORD = os.environ.get("RECORD_ENV_GOLDEN") == "1"
EXACT = os.environ.get("ENV_GOLDEN_EXACT") == "1"
RTOL, ATOL = 1e-12, 1e-14
EPISODES, STEPS = 3, 40

CARTPOLE_PHYSICS = ("gravity", "masscart", "masspole", "length", "total_mass", "polemass_length")
PENDULUM_PHYSICS = ("g", "m", "l")
NAVIGATION_PHYSICS = ("wheel_slip", "sensor_noise_std", "dropout_prob")


def _cartpole():
    env = CartPoleEnv(
        seed=3,
        obs_noise_std=0.05,
        action_noise_prob=0.2,
        domain_randomization={
            "length": [0.3, 0.9],
            "masspole": [0.05, 0.2],
            "gravity": [9.0, 10.5],
            "masscart": [0.8, 1.2],
        },
    )
    return env, (lambda t: t % 2), CARTPOLE_PHYSICS


def _pendulum():
    env = PendulumEnv(
        seed=3,
        obs_noise_std=0.02,
        action_noise_std=0.3,
        domain_randomization={"length": [0.8, 1.2], "g": [9.0, 10.5], "Mass": [0.8, 1.2]},
    )
    return env, (lambda t: np.array([np.sin(t / 3.0) * 1.5], dtype=np.float32)), PENDULUM_PHYSICS


def _navigation():
    env = NavigationEnv(
        "NavigationContinuous-v0",
        seed=3,
        action_noise_std=0.1,
        domain_randomization={
            "wheel_slip": [0.0, 0.3],
            "sensor_noise_std": [0.0, 0.03],
            "dropout_prob": [0.0, 0.1],
        },
    )
    return (
        env,
        (lambda t: np.array([0.8, np.sin(t / 4.0)], dtype=np.float32)),
        NAVIGATION_PHYSICS,
    )


CASES = {"cartpole": _cartpole, "pendulum": _pendulum, "navigation": _navigation}


def _outputs(name):
    env, action, physics = CASES[name]()
    observations, rewards, drawn, lengths = [], [], [], []
    try:
        for _ in range(EPISODES):
            obs, _ = env.reset()
            drawn.append(attrgetter(*physics)(env.env.unwrapped))
            steps = 0
            for t in range(STEPS):
                observations.append(np.asarray(obs))
                obs, reward, terminated, truncated, _ = env.step(action(t))
                rewards.append(float(reward))
                steps += 1
                if terminated or truncated:
                    break
            lengths.append(steps)
        state = env.rng_state()
    finally:
        env.close()
    return {
        "observations": np.array(observations),
        "rewards": np.array(rewards),
        "physics": np.array(drawn, dtype=np.float64),
        "episode_lengths": np.array(lengths),
        **{f"rng/{key}": value for key, value in state.items()},
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_wrapper_episodes_match_the_recorded_arrays(name):
    outputs = _outputs(name)
    prefix = name + "/"
    if RECORD:
        recorded = dict(np.load(GOLDEN)) if GOLDEN.exists() else {}
        recorded.update({prefix + k: v for k, v in outputs.items()})
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN, **recorded)
        pytest.fail(f"Recorded {prefix}; rerun without RECORD_ENV_GOLDEN.")
    with np.load(GOLDEN) as golden:
        expected = {k[len(prefix) :]: golden[k] for k in golden.files if k.startswith(prefix)}
    assert sorted(outputs) == sorted(expected), f"{name} records other arrays than the file"
    for key, value in outputs.items():
        assert value.dtype == expected[key].dtype, f"{prefix}{key} changed dtype"
        if EXACT or value.dtype.kind not in "fc":  # lengths and generator states: exact
            np.testing.assert_array_equal(value, expected[key], err_msg=f"{prefix}{key} differs")
        else:
            np.testing.assert_allclose(
                value, expected[key], rtol=RTOL, atol=ATOL, err_msg=f"{prefix}{key} differs"
            )


def test_the_recorded_cases_draw_new_physics_each_episode():
    """Without this the file could not notice randomization that stopped drawing."""
    with np.load(GOLDEN) as golden:
        assert {key.split("/")[0] for key in golden.files} == set(CASES)
        for name, randomized in (
            ("cartpole", slice(0, 4)),
            ("pendulum", slice(0, 3)),
            ("navigation", slice(0, 3)),
        ):
            physics = golden[f"{name}/physics"][:, randomized]
            assert len(np.unique(physics, axis=0)) == EPISODES, f"{name} repeated its physics"

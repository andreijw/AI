"""Unit tests for the MountainCarContinuous-v0 challenge, its factory and DDPG's OU noise through
the Trainer."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from rl_core.agents import ContinuousRandomAgent, DDPGAgent
from rl_core.challenges import discover_challenges, select_challenge, with_box_action_bounds
from rl_core.training import Trainer
from rl_mountaincar.benchmark import MountainCarContinuousChallenge
from rl_mountaincar.environments import (
    MountainCarEnv,
    make_continuous_env_from_config,
    make_env_from_config,
)
from tests.agent_internals import exploration_generator, set_counters
from tests.helpers import minimal_config, needs_torch, run_main


def test_mountaincar_continuous_env_takes_a_bounded_force_and_charges_for_it():
    env = make_continuous_env_from_config({"seed": 0})
    try:
        obs, _ = env.reset()
        assert obs.shape == (2,), "the car's position and velocity"
        assert env.action_space == gym.spaces.Box(-1.0, 1.0, (1,), np.float32), "one force"
        assert env.env.spec.max_episode_steps == 999, "MountainCarContinuous-v0's own limit"
        _, reward, _, _, _ = env.step(np.array([0.5], dtype=np.float32))
        np.testing.assert_allclose(reward, -0.1 * 0.5**2, err_msg="the force costs -0.1·a²")
    finally:
        env.close()


def test_mountaincar_continuous_factory_builds_the_shared_wrapper():
    env = make_continuous_env_from_config({"seed": 1, "max_episode_steps": 50})
    try:
        assert isinstance(env, MountainCarEnv), "both MountainCar ids share one wrapper"
        assert env.env.spec.id == "MountainCarContinuous-v0"
        assert env.env.spec.max_episode_steps == 50, "the config's limit applies"
    finally:
        env.close()


@pytest.mark.parametrize("unsupported", ["action_noise_std", "domain_randomization"])
def test_mountaincar_continuous_config_rejects_settings_it_does_not_support(unsupported):
    with pytest.raises(
        ValueError, match=f"MountainCarContinuous-v0 does not support .*{unsupported}"
    ):
        make_continuous_env_from_config({unsupported: 0.1})


def test_mountaincar_continuous_challenge_is_discovered_and_selected_by_its_environment():
    found = discover_challenges()
    assert found["mountaincar_continuous"] is MountainCarContinuousChallenge
    assert select_challenge("MountainCarContinuous-v0", None) is MountainCarContinuousChallenge
    challenge = MountainCarContinuousChallenge()
    assert challenge.solve_threshold == 90.0, "Gymnasium's reward_threshold"
    assert challenge.supported_agents == ["ddpg", "random", "td3"]


def _torch_td3_agent() -> type:
    from rl_core.agents.torch_td3_agent import TorchTD3Agent  # here, not above: torch is optional

    return TorchTD3Agent


@pytest.mark.parametrize(
    ("agent_type", "expected_class"),
    [
        ("random", lambda: ContinuousRandomAgent),
        ("ddpg", lambda: DDPGAgent),
        pytest.param("td3", _torch_td3_agent, marks=needs_torch),
    ],
    ids=["random", "ddpg", "td3"],
)
def test_mountaincar_continuous_setup_run_builds_each_agent_from_its_committed_config(
    agent_type, expected_class
):
    env, agent, training = MountainCarContinuousChallenge().setup_run(agent_type, 1, episodes=2)
    try:
        expected = expected_class()
        assert type(agent) is expected, f"{agent_type} should build {expected.__name__}"
        assert (agent.observation_dim, agent.action_dim) == (2, 1), "sized from the spaces"
        assert training["num_episodes"] == 2, "the episode override should apply"
        action = agent.select_action(env.reset()[0], training=False)
        assert action.shape == (1,) and -1.0 <= action[0] <= 1.0, "a force inside the bounds"
    finally:
        env.close()


def test_mountaincar_continuous_agents_get_the_environments_bounds():
    challenge = MountainCarContinuousChallenge()
    env = challenge.make_env({"seed": 0}, render_mode=None)
    try:
        config = challenge.agent_config({}, env)
        np.testing.assert_array_equal(config["action_low"], [-1.0])
        np.testing.assert_array_equal(config["action_high"], [1.0])
        assert challenge.agent_config({"action_low": -0.5}, env)["action_low"] == -0.5, (
            "a config's own bound wins"
        )
    finally:
        env.close()


def test_box_bounds_are_refused_for_a_discrete_environment():
    env = make_env_from_config({"seed": 0})
    try:
        with pytest.raises(TypeError, match="MountainCar-v0 needs a Box action space"):
            with_box_action_bounds({}, env, "MountainCar-v0")
    finally:
        env.close()


def test_the_committed_ddpg_config_explores_with_correlated_noise():
    """Past warm-up, consecutive training actions of the committed config share their noise:
    MountainCarContinuous needs pushes held for many steps, which white noise does not give."""
    env, agent, _ = MountainCarContinuousChallenge().setup_run("ddpg", 1, episodes=1)
    env.close()
    set_counters(agent, training_actions=agent.learning_starts)
    obs = np.array([-0.5, 0.0])
    greedy = agent.select_action(obs, training=False)
    noise = np.array([agent.select_action(obs, training=True) - greedy for _ in range(300)])[:, 0]
    assert np.corrcoef(noise[:-1], noise[1:])[0, 1] > 0.9, "the noise should be OU, not white"


def _ddpg_ou(seed: int = 3) -> DDPGAgent:
    return DDPGAgent(2, 1, {"seed": seed, "hidden_sizes": [8, 8], "batch_size": 16,
                            "learning_starts": 0, "noise_type": "ornstein_uhlenbeck",
                            "noise_std": 0.5, "ou_theta": 0.5, "ou_dt": 0.1})  # fmt: skip


def _first_noise(agent: DDPGAgent, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(noise the next training action carries, the noise one OU step from zero would be)."""
    expected = 0.5 * np.sqrt(0.1) * exploration_generator(agent).standard_normal(1)
    action = agent.select_action(obs, training=True)
    return action - agent.select_action(obs, training=False), expected


def test_the_trainer_starts_every_episode_with_zero_ou_noise_and_saves_it_so(tmp_path):
    """Through the real Trainer: after its episodes, and in its checkpoint, the OU process is back at
    zero, so a run or a resumed run starts the next episode one step from zero."""
    env = make_continuous_env_from_config({"seed": 3, "max_episode_steps": 30})
    try:
        agent = _ddpg_ou()
        config = {"num_episodes": 3, "max_steps_per_episode": 30, "eval_frequency": 100,
                  "save_frequency": 3, "checkpoint_dir": str(tmp_path)}  # fmt: skip
        Trainer(env=env, agent=agent, config=config).train()
        obs = np.array([-0.5, 0.0], dtype=np.float32)

        resumed = _ddpg_ou(seed=11)
        resumed.load(str(tmp_path / "agent_episode_3.pt"))
        for run in (agent, resumed):
            noise, expected = _first_noise(run, obs)
            np.testing.assert_allclose(noise, expected, atol=1e-6, err_msg="one step from zero")
    finally:
        env.close()


def test_train_py_trains_ddpg_on_mountaincar_continuous(tmp_path, monkeypatch):
    path = minimal_config(
        tmp_path,
        environment={"name": "MountainCarContinuous-v0", "max_episode_steps": 20},
        training={"num_episodes": 2},
    )

    run_main(["--config", path, "--agent-type", "ddpg"], monkeypatch)

    assert list((tmp_path / "logs" / "mountaincar_continuous" / "ddpg").glob("metrics_*.jsonl")), (
        "train.py should log the run under logs/mountaincar_continuous/ddpg"
    )

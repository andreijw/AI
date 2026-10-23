"""Unit tests for the MountainCar-v0 environment wrapper, its factories and its challenge."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from rl_core.agents import DQNAgent, RandomAgent
from rl_core.challenges import discover_challenges, select_challenge
from rl_mountaincar.benchmark import MountainCarChallenge
from rl_mountaincar.environments import MountainCarEnv, make_env, make_env_from_config
from tests.helpers import minimal_config, run_main


def test_mountaincar_env_observes_position_and_velocity_and_has_three_pushes():
    env = MountainCarEnv(seed=0)
    try:
        obs, info = env.reset()
        assert obs.shape == (2,), "MountainCar observes the car's position and velocity"
        assert obs.dtype == np.float32, "observations should stay float32"
        assert env.action_space == gym.spaces.Discrete(3), "push left, no push or push right"
        assert isinstance(info, dict), "reset should return an info dict"
    finally:
        env.close()


@pytest.mark.parametrize("action", [0, 1, 2])
def test_mountaincar_env_steps_every_action_and_tracks_the_episode(action):
    env = MountainCarEnv(seed=0)
    try:
        env.reset()
        _, reward, terminated, truncated, _ = env.step(action)
        assert reward == -1.0, "every step costs -1"
        assert not (terminated or truncated), "one step from the valley cannot end the episode"
        assert (env.episode_steps, env.episode_reward) == (1, -1.0), "the wrapper counts the step"
    finally:
        env.close()


def test_mountaincar_env_truncates_at_its_step_limit_and_reports_the_episode():
    env = MountainCarEnv(seed=0, max_episode_steps=9)
    try:
        env.reset()
        for _ in range(8):
            _, _, terminated, truncated, _ = env.step(1)
            assert not (terminated or truncated), "the limit cuts the episode at step 9, not before"
        _, _, terminated, truncated, info = env.step(1)
        assert truncated and not terminated, (
            "a time-limit cut is a truncation, not reaching the flag"
        )
        assert info["episode"] == {"steps": 9, "reward": -9.0}, "the finished episode is reported"
    finally:
        env.close()


def test_mountaincar_env_terminates_when_the_car_reaches_the_flag():
    """Rocking back and forth reaches the flag well within 200 steps; that end is a termination."""
    env = MountainCarEnv(seed=0)
    try:
        obs, _ = env.reset()
        for _ in range(200):
            push_with_velocity = 2 if obs[1] >= 0 else 0
            obs, reward, terminated, truncated, _ = env.step(push_with_velocity)
            if terminated or truncated:
                break
        assert terminated and not truncated, "reaching the flag should terminate the episode"
        assert reward == -1.0, "the goal step costs -1 too"
        assert obs[0] >= 0.5, "the car should be at the flag"
    finally:
        env.close()


def test_mountaincar_factories_build_the_wrapper_with_defaults():
    from_config, direct = make_env_from_config({"seed": 1}), make_env(seed=1)
    try:
        for env in (from_config, direct):
            assert isinstance(env, MountainCarEnv), "both factories build the MountainCar wrapper"
            assert env.env.spec.max_episode_steps == 200, (
                "the default limit is MountainCar-v0's 200"
            )
        np.testing.assert_array_equal(
            from_config.reset()[0], direct.reset()[0], "both factories should seed alike"
        )
    finally:
        from_config.close()
        direct.close()


@pytest.mark.parametrize("unsupported", ["action_noise_prob", "domain_randomization"])
def test_mountaincar_config_rejects_settings_it_does_not_support(unsupported):
    with pytest.raises(ValueError, match=f"MountainCar-v0 does not support .*{unsupported}"):
        make_env_from_config({unsupported: 0.1})


def test_mountaincar_challenge_is_discovered_and_selected_by_its_environment():
    assert discover_challenges()["mountaincar"] is MountainCarChallenge, "found without naming it"
    assert select_challenge("MountainCar-v0", None) is MountainCarChallenge
    challenge = MountainCarChallenge()
    assert challenge.solve_threshold == -110.0, "Gymnasium's reward_threshold for MountainCar-v0"
    assert challenge.supported_agents == ["dqn", "random"], (
        "PPO never found the flag (CHANGELOG.md, 2026-10-23), so it has no MountainCar config"
    )


@pytest.mark.parametrize(
    ("agent_type", "agent_class"),
    [("random", RandomAgent), ("dqn", DQNAgent)],
)
def test_mountaincar_setup_run_builds_each_agent_from_its_committed_config(agent_type, agent_class):
    env, agent, training = MountainCarChallenge().setup_run(agent_type, seed=1, episodes=3)
    try:
        assert isinstance(env, MountainCarEnv), "the run should train on the MountainCar wrapper"
        assert type(agent) is agent_class, f"{agent_type} should build {agent_class.__name__}"
        assert (agent.observation_dim, agent.action_dim) == (2, 3), "sized from the spaces"
        assert training["num_episodes"] == 3, "the episode override should apply"
        obs, _ = env.reset()
        action = int(agent.select_action(obs, training=False))
        assert action in {0, 1, 2}, "the greedy action should be one of the three pushes"
    finally:
        env.close()


def test_train_py_trains_on_mountaincar(tmp_path, monkeypatch):
    path = minimal_config(
        tmp_path, environment={"name": "MountainCar-v0"}, training={"num_episodes": 2}
    )

    run_main(["--config", path, "--agent-type", "dqn"], monkeypatch)

    assert list((tmp_path / "logs" / "mountaincar" / "dqn").glob("metrics_*.jsonl")), (
        "train.py should log the MountainCar run under logs/mountaincar/dqn"
    )

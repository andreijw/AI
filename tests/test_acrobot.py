"""Unit tests for the Acrobot-v1 environment wrapper, its factories and its challenge."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from rl_acrobot.benchmark import AcrobotChallenge
from rl_acrobot.environments import AcrobotEnv, make_env, make_env_from_config
from rl_core.agents import DQNAgent, PPOAgent, RandomAgent
from rl_core.challenges import discover_challenges, select_challenge
from tests.helpers import minimal_config, run_main


def test_acrobot_env_has_six_observations_and_three_actions():
    env = AcrobotEnv(seed=0)
    try:
        obs, info = env.reset()
        assert obs.shape == (6,), "Acrobot observes cos/sin of both joints and both velocities"
        assert obs.dtype == np.float32
        assert env.action_space == gym.spaces.Discrete(3), "torque -1, 0 or +1"
        assert isinstance(info, dict)
    finally:
        env.close()


@pytest.mark.parametrize("action", [0, 1, 2])
def test_acrobot_env_steps_every_action_and_tracks_the_episode(action):
    env = AcrobotEnv(seed=0)
    try:
        env.reset()
        _, reward, terminated, truncated, _ = env.step(action)
        assert reward == -1.0, "every step before the goal costs -1"
        assert not terminated and not truncated
        assert env.episode_steps == 1
        assert env.episode_reward == -1.0
    finally:
        env.close()


def test_acrobot_env_truncates_at_its_step_limit_and_reports_the_episode():
    env = AcrobotEnv(seed=0, max_episode_steps=7)
    try:
        env.reset()
        for _ in range(6):
            _, _, terminated, truncated, _ = env.step(1)
            assert not (terminated or truncated), "the limit cuts the episode at step 7, not before"
        _, _, terminated, truncated, info = env.step(1)
        assert truncated and not terminated, "a time-limit cut is a truncation, not a goal"
        assert info["episode"]["steps"] == 7
        assert info["episode"]["reward"] == -7.0
    finally:
        env.close()


def test_acrobot_factories_build_the_wrapper_with_defaults():
    from_config = make_env_from_config({"name": "Acrobot-v1", "seed": 1})
    direct = make_env(seed=1)
    try:
        for env in (from_config, direct):
            assert isinstance(env, AcrobotEnv)
            assert env.env.spec.max_episode_steps == 500, "the default limit is Acrobot-v1's 500"
        np.testing.assert_array_equal(from_config.reset()[0], direct.reset()[0])
    finally:
        from_config.close()
        direct.close()


def test_acrobot_config_treats_a_null_step_limit_as_the_default():
    env = make_env_from_config({"name": "Acrobot-v1", "max_episode_steps": None})
    try:
        assert env.env.spec.max_episode_steps == 500, "null in YAML should mean Acrobot-v1's 500"
    finally:
        env.close()


@pytest.mark.parametrize(
    "unsupported",
    [
        {"action_noise_prob": 0.1},
        {"action_noise_std": 0.1},
        {"domain_randomization": {"LINK_MASS_1": [0.9, 1.1]}},
    ],
)
def test_acrobot_config_rejects_settings_it_does_not_support(unsupported):
    """A noise or randomization setting must not be silently ignored."""
    with pytest.raises(ValueError, match=next(iter(unsupported))):
        make_env_from_config({"name": "Acrobot-v1", **unsupported})


@pytest.mark.parametrize(
    "neutral",
    [{"action_noise_prob": 0.0}, {"action_noise_std": None}, {"domain_randomization": None}],
)
def test_acrobot_config_accepts_unsupported_settings_left_off(neutral):
    env = make_env_from_config({"name": "Acrobot-v1", **neutral})
    env.close()


def test_acrobot_challenge_is_discovered_and_selected_by_its_environment():
    found = discover_challenges()
    assert found["acrobot"] is AcrobotChallenge
    assert select_challenge("Acrobot-v1", None) is AcrobotChallenge
    challenge = AcrobotChallenge()
    assert challenge.name == "Acrobot-v1"
    assert challenge.solve_threshold == -100.0, "Gymnasium's reward_threshold for Acrobot-v1"
    assert challenge.supported_agents == ["dqn", "ppo", "random"]


@pytest.mark.parametrize(
    ("agent_type", "agent_class"),
    [("random", RandomAgent), ("dqn", DQNAgent), ("ppo", PPOAgent)],
)
def test_acrobot_setup_run_builds_each_agent_from_its_committed_config(agent_type, agent_class):
    env, agent, training = AcrobotChallenge().setup_run(agent_type, seed=1, episodes=3)
    try:
        assert isinstance(env, AcrobotEnv)
        assert type(agent) is agent_class
        assert (agent.observation_dim, agent.action_dim) == (6, 3)
        assert training["num_episodes"] == 3
        obs, _ = env.reset()
        actions = {int(agent.select_action(obs, training=True)) for _ in range(30)}
        assert actions <= {0, 1, 2}, "every action should be one of Acrobot's three"
    finally:
        env.close()


def test_acrobot_challenge_rejects_an_agent_without_a_config():
    env = AcrobotChallenge().make_env({}, render_mode=None)
    try:
        with pytest.raises(ValueError, match="not supported by challenge 'acrobot'"):
            AcrobotChallenge().make_agent("ddpg", {}, env)
    finally:
        env.close()


def test_train_py_trains_on_acrobot(tmp_path, monkeypatch):
    path = minimal_config(
        tmp_path, environment={"name": "Acrobot-v1"}, training={"num_episodes": 2}
    )

    run_main(["--config", path, "--agent-type", "dqn"], monkeypatch)

    assert list((tmp_path / "logs" / "acrobot" / "dqn").glob("metrics_*.jsonl")), (
        "train.py should log the Acrobot run under logs/acrobot/dqn"
    )

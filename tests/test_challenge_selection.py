"""Choosing a challenge from a run config, and building its environment and agent through it."""

import numpy as np
import pytest

import rl_cartpole.environments.cartpole_env as cartpole_env_module
import rl_pendulum.environments.pendulum_env as pendulum_env_module
from rl_core.agents import (
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    ContinuousRandomAgent,
    DQNAgent,
    PPOAgent,
    RandomAgent,
    ReinforceAgent,
)
from rl_core.challenges import CHALLENGES, discover_challenges, select_challenge

discover_challenges()

EXPECTED_AGENTS = {
    "cartpole": {
        "random": RandomAgent,
        "reinforce": ReinforceAgent,
        "actor_critic": ActorCriticAgent,
        "ppo": PPOAgent,
        "dqn": DQNAgent,
    },
    "pendulum": {
        "random": ContinuousRandomAgent,
        "actor_critic": ContinuousActorCriticAgent,
        "ppo": ContinuousPPOAgent,
    },
}

ENV_CLASSES = {
    "cartpole": cartpole_env_module.CartPoleEnv,
    "pendulum": pendulum_env_module.PendulumEnv,
}

# ---------------------------------------------------------------------------
# select_challenge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_name", "key"), [("CartPole-v1", "cartpole"), ("Pendulum-v1", "pendulum")]
)
def test_environment_name_selects_its_challenge(env_name, key):
    assert select_challenge(env_name, None) is CHALLENGES[key]
    assert select_challenge(env_name, key) is CHALLENGES[key], "a matching --challenge is accepted"


def test_challenge_key_alone_selects_that_challenge():
    assert select_challenge(None, "pendulum") is CHALLENGES["pendulum"]


@pytest.mark.parametrize("env_name", ["Acrobot-v1", "cartpole-v1", "CartPole-v0", "MyPendulum-v1"])
def test_unknown_environment_name_is_an_error_naming_the_options(env_name):
    """Ids match exactly: no lower-casing and no substring match picks a challenge."""
    with pytest.raises(ValueError, match=f"Unknown environment '{env_name}'") as error:
        select_challenge(env_name, None)
    assert "CartPole-v1 (cartpole)" in str(error.value), "the message should list every option"
    assert "Pendulum-v1 (pendulum)" in str(error.value), "the message should list every option"


def test_challenge_key_that_disagrees_with_the_environment_is_an_error():
    with pytest.raises(ValueError, match="Pendulum-v1") as error:
        select_challenge("Pendulum-v1", "cartpole")
    assert "cartpole" in str(error.value) and "pendulum" in str(error.value)


def test_unknown_challenge_key_is_an_error():
    with pytest.raises(ValueError, match="Unknown challenge 'nope'"):
        select_challenge(None, "nope")


def test_no_environment_name_and_no_key_is_an_error():
    """A config without environment.name no longer falls back to CartPole."""
    with pytest.raises(ValueError, match="environment.name"):
        select_challenge(None, None)


# ---------------------------------------------------------------------------
# make_env
# ---------------------------------------------------------------------------


class _CapturedError(Exception):
    pass


def _capture_env_kwargs(monkeypatch, key):
    """Stop the challenge's wrapper at construction and return the keyword arguments it got."""
    captured = {}

    def init(self, *args, **kwargs):
        captured.update(kwargs)
        raise _CapturedError

    monkeypatch.setattr(ENV_CLASSES[key], "__init__", init)
    return captured


@pytest.mark.parametrize(
    ("key", "env_config", "expected"),
    [
        (
            "cartpole",
            {"name": "CartPole-v1", "max_episode_steps": 20, "seed": 7, "obs_noise_std": 0.05,
             "action_noise_prob": 0.1, "domain_randomization": {"gravity": [8.0, 12.0]}},
            {"env_name": "CartPole-v1", "max_episode_steps": 20, "seed": 7, "obs_noise_std": 0.05,
             "action_noise_prob": 0.1, "domain_randomization": {"gravity": (8.0, 12.0)},
             "render_mode": "rgb_array"},
        ),
        (
            "pendulum",
            {"name": "Pendulum-v1", "max_episode_steps": 30, "seed": 9, "obs_noise_std": 0.02,
             "action_noise_std": 0.3, "domain_randomization": {"g": [9.0, 11.0]}},
            {"env_name": "Pendulum-v1", "max_episode_steps": 30, "seed": 9, "obs_noise_std": 0.02,
             "action_noise_std": 0.3, "domain_randomization": {"g": [9.0, 11.0]},
             "render_mode": "rgb_array"},
        ),
    ],
)  # fmt: skip
def test_make_env_forwards_the_environment_section(monkeypatch, key, env_config, expected):
    captured = _capture_env_kwargs(monkeypatch, key)
    with pytest.raises(_CapturedError):
        CHALLENGES[key]().make_env(env_config, render_mode="rgb_array")
    assert captured == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("cartpole", {"env_name": "CartPole-v1", "max_episode_steps": 500, "obs_noise_std": 0.0,
                      "action_noise_prob": 0.0}),
        ("pendulum", {"env_name": "Pendulum-v1", "max_episode_steps": 200, "obs_noise_std": 0.0,
                      "action_noise_std": 0.0}),
    ],
)  # fmt: skip
def test_make_env_treats_null_values_as_missing(monkeypatch, key, expected):
    """A YAML ``null`` for a field with a default gets the default, not None."""
    nulls = {"name": None, "max_episode_steps": None, "obs_noise_std": None}
    nulls["action_noise_prob" if key == "cartpole" else "action_noise_std"] = None
    captured = _capture_env_kwargs(monkeypatch, key)
    with pytest.raises(_CapturedError):
        CHALLENGES[key]().make_env(nulls, render_mode=None)
    assert {name: captured[name] for name in expected} == expected
    assert captured["seed"] is None


@pytest.mark.parametrize("key", sorted(ENV_CLASSES))
def test_make_env_builds_a_wrapper_the_scripts_can_wrap(key):
    env = CHALLENGES[key]().make_env({"seed": 3, "max_episode_steps": 5}, render_mode=None)
    try:
        assert isinstance(env, ENV_CLASSES[key])
        assert callable(env.wrap_env), "train.py and record.py install video recorders through it"
    finally:
        env.close()


# ---------------------------------------------------------------------------
# make_agent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(EXPECTED_AGENTS))
def test_supported_agents_are_exactly_the_agents_the_challenge_builds(key):
    assert sorted(CHALLENGES[key]().supported_agents) == sorted(EXPECTED_AGENTS[key])


@pytest.mark.parametrize(
    ("key", "agent_type"),
    [(key, agent) for key, agents in sorted(EXPECTED_AGENTS.items()) for agent in sorted(agents)],
)
def test_make_agent_builds_the_agent_sized_for_the_environment(key, agent_type):
    challenge = CHALLENGES[key]()
    env = challenge.make_env({"seed": 0, "max_episode_steps": 5}, render_mode=None)
    try:
        agent = challenge.make_agent(agent_type, {"seed": 0}, env)
    finally:
        env.close()
    assert type(agent) is EXPECTED_AGENTS[key][agent_type]
    assert agent.observation_dim == {"cartpole": 4, "pendulum": 3}[key]
    assert agent.action_dim == {"cartpole": 2, "pendulum": 1}[key]


@pytest.mark.parametrize("agent_type", sorted(EXPECTED_AGENTS["pendulum"]))
def test_pendulum_agents_get_the_environment_torque_bounds(agent_type):
    challenge = CHALLENGES["pendulum"]()
    env = challenge.make_env({"seed": 0}, render_mode=None)
    agent_config = {"seed": 0}
    try:
        agent = challenge.make_agent(agent_type, agent_config, env)
    finally:
        env.close()
    np.testing.assert_allclose(agent.action_low, [-2.0])
    np.testing.assert_allclose(agent.action_high, [2.0])
    assert agent_config == {"seed": 0}, (
        "make_agent must not write the bounds into the caller's dict"
    )


def test_configured_pendulum_bounds_win_over_the_environment():
    challenge = CHALLENGES["pendulum"]()
    env = challenge.make_env({"seed": 0}, render_mode=None)
    try:
        agent = challenge.make_agent("random", {"action_low": -1.0, "action_high": 1.0}, env)
    finally:
        env.close()
    np.testing.assert_allclose(agent.action_low, [-1.0])
    np.testing.assert_allclose(agent.action_high, [1.0])


@pytest.mark.parametrize(("key", "agent_type"), [("cartpole", "sac"), ("pendulum", "dqn")])
def test_unsupported_agent_is_an_error_naming_the_supported_ones(key, agent_type):
    challenge = CHALLENGES[key]()
    env = challenge.make_env({"seed": 0}, render_mode=None)
    try:
        with pytest.raises(ValueError, match=f"'{agent_type}'") as error:
            challenge.make_agent(agent_type, {}, env)
    finally:
        env.close()
    assert key in str(error.value)
    assert str(sorted(EXPECTED_AGENTS[key])) in str(error.value)


TWIN_CHALLENGE = """
from rl_core.challenges import BenchmarkChallenge, register_challenge


@register_challenge
class Twin(BenchmarkChallenge):
    name = "CartPole-v1"
    key = "{key}"
    solve_threshold = None
    default_episodes = 1
    supported_agents = ["random"]

    def make_env(self, env_config, render_mode):
        raise NotImplementedError

    def make_agent(self, agent_type, agent_config, env):
        raise NotImplementedError
"""


def test_two_challenges_on_one_environment_need_a_key(isolated_registry):
    """A second challenge on CartPole-v1 must not silently replace the first."""
    from tests.helpers import unique_name, write_package

    key = unique_name("key")
    write_package(isolated_registry, unique_name(), TWIN_CHALLENGE.format(key=key))
    discover_challenges(isolated_registry)

    with pytest.raises(ValueError, match="CartPole-v1.*--challenge") as error:
        select_challenge("CartPole-v1", None)
    assert "cartpole" in str(error.value) and key in str(error.value)
    assert select_challenge("CartPole-v1", "cartpole") is CHALLENGES["cartpole"]
    assert select_challenge("CartPole-v1", key) is CHALLENGES[key]

"""Unit tests for the LunarLander-v3 challenges, discrete and continuous, and their factory.

Discovery, selection and config reading need no Box2D, so they run everywhere (CI included). Building
an environment needs box2d-py (./setup.sh): those tests are marked ``needs_box2d``, and without
Box2D building one fails at once with this project's install hint.
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym
import numpy as np
import pytest

from benchmark import BenchmarkSettings, run_benchmark
from rl_core.agents import ContinuousRandomAgent, RandomAgent
from rl_core.challenges import discover_challenges, select_challenge
from rl_lunarlander.benchmark import LunarLanderChallenge, LunarLanderContinuousChallenge
from rl_lunarlander.environments import (
    LunarLanderEnv,
    make_continuous_env_from_config,
    make_env_from_config,
)
from tests.helpers import minimal_config, needs_box2d, run_main

CHALLENGES = {
    "lunarlander": (LunarLanderChallenge, "LunarLander-v3"),
    "lunarlander_continuous": (LunarLanderContinuousChallenge, "LunarLanderContinuous-v3"),
}
FACTORIES = {
    "lunarlander": make_env_from_config,
    "lunarlander_continuous": make_continuous_env_from_config,
}


# ---------------------------------------------------------------------------
# Without Box2D: discovery, selection, config reading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(CHALLENGES))
def test_lunarlander_challenges_are_discovered_and_selected_by_their_environment(key):
    challenge_cls, environment_id = CHALLENGES[key]
    assert discover_challenges()[key] is challenge_cls
    assert select_challenge(environment_id, None) is challenge_cls
    challenge = challenge_cls()
    assert (challenge.key, challenge.name) == (key, environment_id)
    assert challenge.solve_threshold == 200.0, "Gymnasium's reward_threshold"
    assert challenge.default_episodes == 500
    assert challenge.supported_agents == ["random"]


@pytest.mark.parametrize("key", sorted(FACTORIES))
def test_an_unsupported_environment_setting_is_refused_before_anything_is_built(key):
    with pytest.raises(ValueError, match="action_noise_std"):
        FACTORIES[key]({"action_noise_std": 0.1})


def test_without_box2d_building_an_environment_names_this_projects_install(monkeypatch):
    """Gymnasium's own hint (pip install "gymnasium[box2d]") fails on the Jetson."""
    monkeypatch.setitem(sys.modules, "Box2D", None)  # import Box2D now fails, installed or not
    with pytest.raises(ImportError, match=r"setup\.sh.*box2d-py"):
        LunarLanderEnv(seed=0)


# ---------------------------------------------------------------------------
# With Box2D: the environments
# ---------------------------------------------------------------------------


@needs_box2d
@pytest.mark.parametrize(
    ("key", "action_space"),
    [
        ("lunarlander", gym.spaces.Discrete(4)),
        ("lunarlander_continuous", gym.spaces.Box(-1.0, 1.0, (2,), np.float32)),
    ],
)
def test_the_environments_observe_eight_values_and_take_their_actions(key, action_space):
    env = FACTORIES[key]({"seed": 1})
    try:
        assert env.observation_space.shape == (8,)
        assert env.action_space == action_space
        obs, _ = env.reset()
        assert obs.shape == (8,) and np.isfinite(obs).all()
    finally:
        env.close()


@needs_box2d
def test_firing_nothing_ends_in_a_crash_terminated_with_minus_100():
    env = make_env_from_config({"seed": 2})
    try:
        env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(0)  # no engine: free fall
        assert terminated and not truncated, "a fall ends in a crash, not the time limit"
        assert reward == -100.0, "Gymnasium's crash reward"
    finally:
        env.close()


@needs_box2d
def test_the_step_limit_truncates_the_episode():
    env = make_env_from_config({"seed": 3, "max_episode_steps": 5})
    try:
        env.reset()
        flags = [env.step(0)[2:4] for _ in range(5)]
        assert flags[-1] == (False, True), "the fifth step is cut at the limit"
        assert not any(any(flag) for flag in flags[:-1])
    finally:
        env.close()


@needs_box2d
@pytest.mark.parametrize("key", sorted(FACTORIES))
def test_observation_noise_is_applied(key):
    """Noise on top of the same start state: every value moves, by much less than a different
    start would (std 0.1, so beyond 1 is a ten-sigma draw)."""
    clean, noisy = FACTORIES[key]({"seed": 4}), FACTORIES[key]({"seed": 4, "obs_noise_std": 0.1})
    try:
        difference = np.abs(noisy.reset()[0] - clean.reset()[0])
        assert (difference > 0.0).all(), "every observed value should carry noise"
        assert (difference < 1.0).all(), "the noise should sit on the same seeded state"
    finally:
        clean.close()
        noisy.close()


# ---------------------------------------------------------------------------
# With Box2D: the challenges' agents and the scripts
# ---------------------------------------------------------------------------


@needs_box2d
@pytest.mark.parametrize(
    ("key", "agent_cls", "action_dim"),
    [("lunarlander", RandomAgent, 4), ("lunarlander_continuous", ContinuousRandomAgent, 2)],
)
def test_setup_run_builds_the_random_agent_from_its_committed_config(key, agent_cls, action_dim):
    env, agent, training = CHALLENGES[key][0]().setup_run("random", 1, episodes=2)
    try:
        assert type(agent) is agent_cls
        assert (agent.observation_dim, agent.action_dim) == (8, action_dim)
        assert training["num_episodes"] == 2, "the episode override should apply"
        action = agent.select_action(env.reset()[0], training=True)
        assert env.action_space.contains(np.asarray(action, dtype=env.action_space.dtype))
    finally:
        env.close()


@needs_box2d
def test_the_continuous_agent_gets_the_environments_bounds_without_changing_the_callers_dict():
    """Through agent_config, since the random agent's default bounds are already [-1, 1] and could
    not show a challenge that drops them; a learning agent's might not be."""
    challenge = LunarLanderContinuousChallenge()
    env = challenge.make_env({"seed": 0}, render_mode=None)
    agent_config = {"seed": 0}
    try:
        built = challenge.agent_config(agent_config, env)
        agent = challenge.make_agent("random", agent_config, env)
    finally:
        env.close()
    np.testing.assert_allclose(built["action_low"], [-1.0, -1.0])
    np.testing.assert_allclose(built["action_high"], [1.0, 1.0])
    np.testing.assert_allclose(agent.action_low, [-1.0, -1.0])
    assert agent_config == {"seed": 0}, (
        "make_agent must not write the bounds into the caller's dict"
    )


@needs_box2d
@pytest.mark.usefixtures("clear_sdl")
def test_record_renders_a_lunarlander_episode_to_video(tmp_path, monkeypatch):
    """rgb_array rendering goes through pygame-ce: a plain pygame installed over it breaks this."""
    pytest.importorskip("moviepy")
    checkpoint = str(tmp_path / "agent.pt")
    RandomAgent(observation_dim=8, action_dim=4, config={"seed": 0}).save(checkpoint)
    config = minimal_config(
        tmp_path,
        environment={"name": "LunarLander-v3", "max_episode_steps": 30},
    )
    video_dir = tmp_path / "videos"
    run_main(
        ["--config", config, "--agent-type", "random", "--checkpoint", checkpoint]
        + ["--episodes", "1", "--video-dir", str(video_dir)],
        monkeypatch,
        module="record",
    )
    assert os.listdir(video_dir) == ["lunarlander-playback-full.mp4"]


@needs_box2d
@pytest.mark.parametrize("key", sorted(CHALLENGES))
def test_the_benchmark_runs_the_random_agent_and_reports_the_threshold(key, tmp_path):
    report = tmp_path / "report.md"
    result = run_benchmark(
        challenge_name=key,
        settings=BenchmarkSettings(
            agents=("random",), seeds=(1,), episodes=3, eval_episodes=2, output_dir=str(tmp_path)
        ),
        report_path=str(report),
        save_plot=False,
    )
    assert result["challenge"] == CHALLENGES[key][1]
    assert result["summary"]["random"]["runs"] == 1
    assert "average reward >= 200" in report.read_text(encoding="utf-8")

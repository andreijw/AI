"""Unit tests for the BipedalWalker-v3 challenge and its factory.

Discovery, selection and config reading need no Box2D, so they run everywhere (CI included). Building
the environment needs box2d-py (./setup.sh): those tests are marked ``needs_box2d``, and without
Box2D building one fails at once with this project's install hint.
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym
import numpy as np
import pytest

from benchmark import BenchmarkSettings, run_benchmark
from rl_bipedalwalker.benchmark import BipedalWalkerChallenge
from rl_bipedalwalker.environments import BipedalWalkerEnv, make_env_from_config
from rl_core.agents import ContinuousRandomAgent, SACAgent
from rl_core.challenges import discover_challenges, select_challenge
from tests.helpers import (
    minimal_config,
    needs_box2d,
    needs_torch,
    run_main,
    torch_sac_agent,
    torch_td3_agent,
)

ZERO_ACTION = np.zeros(4, dtype=np.float32)

# ---------------------------------------------------------------------------
# Without Box2D: discovery, selection, config reading
# ---------------------------------------------------------------------------


def test_bipedalwalker_is_discovered_and_selected_by_its_environment():
    assert discover_challenges()["bipedalwalker"] is BipedalWalkerChallenge
    assert select_challenge("BipedalWalker-v3", None) is BipedalWalkerChallenge
    challenge = BipedalWalkerChallenge()
    assert (challenge.key, challenge.name) == ("bipedalwalker", "BipedalWalker-v3")
    assert challenge.solve_threshold == 300.0, "Gymnasium's reward_threshold"
    assert challenge.default_episodes == 500
    assert challenge.supported_agents == ["random", "sac", "td3"]


@pytest.mark.parametrize(
    "setting",
    [
        {"action_noise_std": 0.1},
        {"action_noise_prob": 0.1},
        {"domain_randomization": {"g": [9, 10]}},
    ],
)
def test_an_unsupported_environment_setting_is_refused_before_anything_is_built(setting):
    (name,) = setting
    with pytest.raises(ValueError, match=name):
        make_env_from_config(setting)


def test_mask_velocity_is_refused_because_no_velocity_entries_are_listed():
    """Hull and joint speeds are mixed with lidar and contacts; nothing defines which to hide."""
    with pytest.raises(ValueError, match="mask_velocity is not supported for BipedalWalker-v3"):
        make_env_from_config({"mask_velocity": True})


def test_without_box2d_building_the_environment_names_this_projects_install(monkeypatch):
    """Gymnasium's own hint (pip install "gymnasium[box2d]") fails on the Jetson."""
    monkeypatch.setitem(sys.modules, "Box2D", None)  # import Box2D now fails, installed or not
    with pytest.raises(ImportError, match=r"BipedalWalker-v3 needs Box2D.*setup\.sh.*box2d-py"):
        BipedalWalkerEnv(seed=0)


# ---------------------------------------------------------------------------
# With Box2D: the environment
# ---------------------------------------------------------------------------


@needs_box2d
def test_the_walker_observes_24_values_and_takes_four_joint_torques():
    env = make_env_from_config({"seed": 1})
    try:
        assert env.observation_space.shape == (24,)
        assert env.action_space == gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        obs, _ = env.reset()
        assert obs.shape == (24,) and np.isfinite(obs).all()
        assert env.max_episode_steps == 1600, "Gymnasium's limit is the default"
    finally:
        env.close()


@needs_box2d
def test_the_wrapper_built_directly_also_defaults_to_gymnasiums_1600_steps():
    env = BipedalWalkerEnv()
    try:
        assert env.max_episode_steps == 1600
    finally:
        env.close()


@needs_box2d
def test_a_walker_that_applies_no_torque_falls_terminated_with_minus_100():
    """With seed 0 the hull touches the ground at about step 115 (Gymnasium 1.4): a fall terminates,
    so the replay buffer stores it as an end without bootstrapping, unlike a cut at the limit."""
    env = make_env_from_config({"seed": 0})
    try:
        env.reset()
        steps, terminated, truncated = 0, False, False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(ZERO_ACTION)
            steps += 1
        assert (terminated, truncated) == (True, False), "a fall ends the episode, not the limit"
        assert reward == -100.0, "Gymnasium's fall reward"
        assert steps < 1600, "the fall, well before the step limit"
    finally:
        env.close()


@needs_box2d
def test_the_step_limit_truncates_the_episode():
    env = make_env_from_config({"seed": 3, "max_episode_steps": 5})
    try:
        env.reset()
        flags = [env.step(ZERO_ACTION)[2:4] for _ in range(5)]
        assert flags[-1] == (False, True), "the fifth step is cut at the limit"
        assert not any(any(flag) for flag in flags[:-1])
    finally:
        env.close()


@needs_box2d
def test_observation_noise_is_applied():
    """Noise on top of the same start state: every value moves, by much less than a different
    start would (std 0.1, so beyond 1 is a ten-sigma draw)."""
    clean = make_env_from_config({"seed": 4})
    noisy = make_env_from_config({"seed": 4, "obs_noise_std": 0.1})
    try:
        difference = np.abs(noisy.reset()[0] - clean.reset()[0])
        assert (difference > 0.0).all(), "every observed value should carry noise"
        assert (difference < 1.0).all(), "the noise should sit on the same seeded state"
    finally:
        clean.close()
        noisy.close()


# ---------------------------------------------------------------------------
# With Box2D: the challenge's agents and the scripts
# ---------------------------------------------------------------------------


@needs_box2d
@pytest.mark.parametrize(
    ("agent_type", "expected_class"),
    [
        ("random", lambda: ContinuousRandomAgent),
        pytest.param("td3", torch_td3_agent, marks=needs_torch),
        pytest.param("sac", torch_sac_agent, marks=needs_torch),
    ],
    ids=["random", "td3", "sac"],
)
def test_setup_run_builds_each_agent_from_its_committed_config(agent_type, expected_class):
    env, agent, training = BipedalWalkerChallenge().setup_run(agent_type, 1, episodes=2)
    try:
        assert type(agent) is expected_class(), f"{agent_type} should build its committed class"
        assert (agent.observation_dim, agent.action_dim) == (24, 4)
        assert training["num_episodes"] == 2, "the episode override should apply"
        assert training["max_steps_per_episode"] == 1600, "the trainer must not cut at 500"
        assert env.max_episode_steps == 1600
        action = agent.select_action(env.reset()[0], training=True)
        assert env.action_space.contains(np.asarray(action, dtype=env.action_space.dtype))
    finally:
        env.close()


@needs_box2d
def test_the_agent_gets_the_environments_bounds_without_changing_the_callers_dict():
    """Checked on agent_config's output, since BipedalWalker's [-1, 1] bounds equal every agent's
    defaults and an agent's own bounds could not show a challenge that drops them. One agent is
    enough: agent_config does not depend on the agent type. SAC here is the NumPy agent."""
    agent_type = "sac"
    challenge = BipedalWalkerChallenge()
    env = challenge.make_env({"seed": 0}, render_mode=None)
    agent_config = {"seed": 0}
    try:
        built = challenge.agent_config(agent_config, env)
        agent = challenge.make_agent(agent_type, agent_config, env)
    finally:
        env.close()
    np.testing.assert_allclose(built["action_low"], [-1.0] * 4)
    np.testing.assert_allclose(built["action_high"], [1.0] * 4)
    np.testing.assert_allclose(agent.action_low, [-1.0] * 4)
    assert agent_config == {"seed": 0}, (
        "make_agent must not write the bounds into the caller's dict"
    )


# A SAC actor small enough to save and play in a test (the NumPy backend; the torch one saves the
# same format).
SMALL_SAC = {"seed": 0, "hidden_sizes": [8]}


@needs_box2d
@pytest.mark.usefixtures("clear_sdl")
@pytest.mark.parametrize(
    ("agent_type", "agent"),
    [
        (
            "random",
            lambda: ContinuousRandomAgent(observation_dim=24, action_dim=4, config={"seed": 0}),
        ),
        ("sac", lambda: SACAgent(observation_dim=24, action_dim=4, config=dict(SMALL_SAC))),
    ],
)
def test_record_renders_a_bipedalwalker_episode_to_video(tmp_path, monkeypatch, agent_type, agent):
    """rgb_array rendering goes through pygame-ce, as LunarLander's does; a SAC checkpoint loads
    and plays its deterministic action."""
    pytest.importorskip("moviepy")
    checkpoint = str(tmp_path / "agent.pt")
    agent().save(checkpoint)
    config = minimal_config(
        tmp_path,
        environment={"name": "BipedalWalker-v3", "max_episode_steps": 30},
        agent={"type": agent_type, "config": dict(SMALL_SAC)},
    )
    video_dir = tmp_path / "videos"
    run_main(
        ["--config", config, "--agent-type", agent_type, "--checkpoint", checkpoint]
        + ["--episodes", "1", "--video-dir", str(video_dir)],
        monkeypatch,
        module="record",
    )
    assert os.listdir(video_dir) == ["bipedalwalker-playback-full.mp4"]


@needs_box2d
def test_the_benchmark_runs_the_random_agent_and_reports_the_threshold(tmp_path):
    report = tmp_path / "report.md"
    result = run_benchmark(
        challenge_name="bipedalwalker",
        settings=BenchmarkSettings(
            agents=("random",), seeds=(1,), episodes=2, eval_episodes=1, output_dir=str(tmp_path)
        ),
        report_path=str(report),
        save_plot=False,
    )
    assert result["challenge"] == "BipedalWalker-v3"
    assert result["summary"]["random"]["runs"] == 1
    assert "average reward >= 300" in report.read_text(encoding="utf-8")

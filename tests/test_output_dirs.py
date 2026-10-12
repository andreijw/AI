"""Unit tests for agent and challenge output directory scoping."""

import os

from train import agent_output_dir


def test_agent_output_dir_without_challenge():
    """Verify legacy backward-compatible folder path when challenge is None."""
    path = agent_output_dir("./checkpoints", "ppo")
    assert path == os.path.join("./checkpoints", "ppo")


def test_agent_output_dir_with_challenge():
    """Verify challenge-scoped folder path <base>/<challenge>/<agent>."""
    path_cartpole = agent_output_dir("./checkpoints", "ppo", challenge="cartpole")
    assert path_cartpole == os.path.join("./checkpoints", "cartpole", "ppo")

    path_pendulum = agent_output_dir("./videos", "actor_critic", challenge="pendulum")
    assert path_pendulum == os.path.join("./videos", "pendulum", "actor_critic")

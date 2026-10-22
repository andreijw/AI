"""Both PPO agents save their partly filled rollout in checkpoints and restore it on load
(CHANGELOG.md, 2026-10-21). A partly filled rollout is never trained on when training ends."""

import numpy as np
import pytest

from rl_core.agents.continuous_ppo import ContinuousPPOAgent
from rl_core.agents.ppo_agent import PPOAgent
from tests.agent_internals import assert_same_weights
from tests.helpers import assert_same_checkpoint

ROLLOUT_STEPS = 12


def _agent(agent_cls, seed: int = 3):
    config = {"hidden_dim": 6, "seed": seed, "rollout_steps": ROLLOUT_STEPS, "minibatch_size": 4}
    if agent_cls is ContinuousPPOAgent:
        return agent_cls(3, 1, {**config, "action_low": -2.0, "action_high": 2.0})
    return agent_cls(4, 2, config)


def _episode(agent, length: int, seed: int, terminated: bool = True) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((length, agent.observation_dim))
    if isinstance(agent, ContinuousPPOAgent):
        actions = rng.uniform(-2.0, 2.0, size=(length, 1))
    else:
        actions = rng.integers(0, 2, size=length)
    return {
        "observations": obs,
        "actions": actions,
        "rewards": rng.uniform(-1.0, 1.0, size=length),
        "next_observations": np.roll(obs, -1, axis=0),
        "terminated": (np.arange(length) == length - 1) & terminated,
    }


AGENTS = [PPOAgent, ContinuousPPOAgent]


@pytest.mark.parametrize("agent_cls", AGENTS)
def test_a_loaded_agent_resumes_a_partly_filled_rollout_exactly(agent_cls, tmp_path):
    """The resumed agent has other weights, optimizer state and generators before the load."""
    agent = _agent(agent_cls)
    assert agent.update(_episode(agent, 12, seed=1)), "the first rollout trains"
    # Cut, not terminated: its last buffered reward carries gamma * V(next state).
    assert agent.update(_episode(agent, 7, seed=2, terminated=False)) == {}, "7 of 12 buffered"
    path = str(tmp_path / "ppo.npz")
    agent.save(path)
    resumed = _agent(agent_cls, seed=99)
    resumed.update(_episode(resumed, 12, seed=50))  # different weights and a trained optimizer
    resumed.load(path)

    for run in (agent, resumed):
        metrics = run.update(_episode(run, 6, seed=3))
        assert metrics.get("rollout_steps") == 13.0, (
            "7 restored steps and 6 new ones train together"
        )

    assert_same_weights(resumed, agent, rtol=0.0)


@pytest.mark.parametrize("agent_cls", AGENTS)
@pytest.mark.parametrize("saved_steps", [0, 2])
def test_a_load_replaces_the_buffered_steps_with_the_checkpoints(agent_cls, saved_steps, tmp_path):
    """Buffered steps belong to the policy that collected them, so they travel with its weights;
    a checkpoint saved with nothing buffered empties the buffer."""
    agent = _agent(agent_cls)
    if saved_steps:
        agent.update(_episode(agent, saved_steps, seed=1))
    path = str(tmp_path / "ppo.npz")
    agent.save(path)
    agent.update(_episode(agent, 8, seed=2))  # buffered after the save, dropped by the load
    agent.load(path)
    missing = ROLLOUT_STEPS - saved_steps
    assert agent.update(_episode(agent, missing - 1, seed=3)) == {}, "one step short: buffering"
    metrics = agent.update(_episode(agent, 1, seed=4))
    assert metrics.get("rollout_steps") == float(ROLLOUT_STEPS), "the checkpoint's steps train too"


@pytest.mark.parametrize("agent_cls", AGENTS)
def test_a_checkpoint_with_a_malformed_rollout_changes_nothing(agent_cls, tmp_path):
    """The rollout is checked before any weight is replaced, so a failed load leaves the agent
    as it was."""
    saved = _agent(agent_cls)
    saved.update(_episode(saved, 12, seed=1))
    saved.update(_episode(saved, 5, seed=2))
    saved.save(str(tmp_path / "saved.npz"))
    with np.load(tmp_path / "saved.npz") as data:
        bad = {k: data[k] for k in data.files}
    bad["ppo_rollout_rewards"] = bad["ppo_rollout_rewards"][:-1]
    np.savez(tmp_path / "bad.npz", **bad)

    target = _agent(agent_cls, seed=99)
    target.update(_episode(target, 12, seed=50))
    target.save(str(tmp_path / "before.npz"))
    with pytest.raises(ValueError, match="ppo_rollout_rewards"):
        target.load(str(tmp_path / "bad.npz"))
    untouched = _agent(agent_cls, seed=99)
    untouched.load(str(tmp_path / "before.npz"))
    assert_same_checkpoint(untouched, target, tmp_path)

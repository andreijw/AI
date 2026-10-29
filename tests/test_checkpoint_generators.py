"""A checkpoint holds every generator an agent draws from, and DQN's and DDPG's replay buffers, so
a loaded agent continues exactly as the saved one would have (CHANGELOG.md, 2026-10-21)."""

import json

import numpy as np
import pytest

from rl_core.agents import BaseAgent, ContinuousRandomAgent, RandomAgent
from rl_core.seeding import generator_state, restore_generator
from tests.helpers import assert_same_checkpoint
from tests.test_agent_harness import ALL_AGENTS, CONTINUOUS_AGENTS, agent_id

AGENTS = [
    pytest.param(cls, config, 4, 2, id=agent_id(cls, config)) for cls, config in ALL_AGENTS
] + [pytest.param(cls, config, 3, 1, id=cls.__name__) for cls, config in CONTINUOUS_AGENTS]


def _run(agent: BaseAgent, episodes: int, seed: int, obs_dim: int) -> list[np.ndarray]:
    """Act and update for ``episodes`` short episodes on generated observations; the actions."""
    rng = np.random.default_rng(seed)
    taken = []
    for _ in range(episodes):
        length = 6
        observations = rng.standard_normal((length + 1, obs_dim))
        actions = [agent.select_action(obs, training=True) for obs in observations[:-1]]
        taken.extend(np.atleast_1d(np.asarray(a, dtype=np.float64)) for a in actions)
        agent.update(
            {
                "observations": observations[:-1],
                "actions": np.array(actions),
                "rewards": rng.uniform(-1.0, 1.0, size=length),
                "next_observations": observations[1:],
                "dones": np.arange(length) == length - 1,
                "terminated": np.zeros(length, dtype=bool),
            }
        )
    return taken


@pytest.mark.parametrize(("agent_cls", "config", "obs_dim", "action_dim"), AGENTS)
def test_a_loaded_agent_continues_exactly_like_the_saved_one(
    agent_cls, config, obs_dim, action_dim, tmp_path
):
    """Loaded into an agent with another seed that has trained too, so any generator or buffer
    the checkpoint misses keeps the wrong state and the two runs part."""
    saved = agent_cls(obs_dim, action_dim, dict(config))
    _run(saved, episodes=3, seed=1, obs_dim=obs_dim)
    path = str(tmp_path / "agent.npz")
    saved.save(path)
    loaded = agent_cls(obs_dim, action_dim, {**config, "seed": 7})
    _run(loaded, episodes=2, seed=2, obs_dim=obs_dim)
    loaded.load(path)

    expected = _run(saved, episodes=3, seed=3, obs_dim=obs_dim)
    actual = _run(loaded, episodes=3, seed=3, obs_dim=obs_dim)

    np.testing.assert_array_equal(np.array(actual), np.array(expected), "training actions differ")
    assert_same_checkpoint(saved, loaded, tmp_path)


@pytest.mark.parametrize(("agent_cls", "config", "obs_dim", "action_dim"), AGENTS)
def test_generator_keys_are_saved_in_sorted_order(agent_cls, config, obs_dim, action_dim, tmp_path):
    """The file's bytes then do not depend on the order the generators were created in."""
    agent = agent_cls(obs_dim, action_dim, dict(config))
    agent.save(str(tmp_path / "agent.npz"))
    with np.load(tmp_path / "agent.npz") as data:
        keys = [key for key in data.files if key.startswith("rng_")]
    assert keys, "every agent draws from at least one generator"
    assert keys == sorted(keys)


@pytest.mark.parametrize(("agent_cls", "config", "obs_dim", "action_dim"), AGENTS)
def test_an_old_checkpoint_leaves_generators_seeded_and_empties_a_replay_buffer(
    agent_cls, config, obs_dim, action_dim, tmp_path
):
    """Checkpoints saved before generators and buffers were saved still load: the generators keep
    the state the new agent was built with, and a replay buffer is emptied."""
    agent = agent_cls(obs_dim, action_dim, dict(config))
    agent.save(str(tmp_path / "agent.npz"))
    with np.load(tmp_path / "agent.npz") as data:
        old = {k: data[k] for k in data.files if not k.startswith(("rng_", "replay_"))}
    np.savez(tmp_path / "old.npz", **old)

    fresh = agent_cls(obs_dim, action_dim, dict(config))
    loaded = agent_cls(obs_dim, action_dim, dict(config))
    _run(loaded, episodes=3, seed=1, obs_dim=obs_dim)  # fills any replay buffer
    loaded.load(str(tmp_path / "old.npz"))
    if hasattr(loaded, "replay_buffer"):
        assert len(loaded.replay_buffer) == 0, "an old checkpoint holds no buffer to keep"
    _run(fresh, episodes=1, seed=4, obs_dim=obs_dim)
    expected = _run(fresh, episodes=1, seed=5, obs_dim=obs_dim)
    fresh_again = agent_cls(obs_dim, action_dim, dict(config))
    fresh_again.load(str(tmp_path / "old.npz"))
    _run(fresh_again, episodes=1, seed=4, obs_dim=obs_dim)
    actual = _run(fresh_again, episodes=1, seed=5, obs_dim=obs_dim)
    np.testing.assert_array_equal(np.array(actual), np.array(expected), "seeded generators")


@pytest.mark.parametrize(
    "bad_state",
    [
        pytest.param(generator_state(np.random.Generator(np.random.PCG64DXSM(1))), id="other-kind"),
        pytest.param(np.array("not json"), id="not-json"),
        pytest.param(np.array('{"state": 1}'), id="no-bit-generator"),
    ],
)
@pytest.mark.parametrize(("agent_cls", "config", "obs_dim", "action_dim"), AGENTS)
def test_a_generator_state_that_cannot_be_restored_leaves_the_agent_unchanged(
    agent_cls, config, obs_dim, action_dim, bad_state, tmp_path
):
    agent = agent_cls(obs_dim, action_dim, dict(config))
    agent.save(str(tmp_path / "agent.npz"))
    with np.load(tmp_path / "agent.npz") as data:
        bad = {k: data[k] for k in data.files}
    key = next(k for k in bad if k.startswith("rng_"))
    bad[key] = bad_state
    np.savez(tmp_path / "bad.npz", **bad)

    target = agent_cls(obs_dim, action_dim, {**config, "seed": 7})
    _run(target, episodes=2, seed=2, obs_dim=obs_dim)
    target.save(str(tmp_path / "before.npz"))
    with pytest.raises(ValueError, match=key):
        target.load(str(tmp_path / "bad.npz"))
    untouched = agent_cls(obs_dim, action_dim, {**config, "seed": 7})
    untouched.load(str(tmp_path / "before.npz"))
    assert_same_checkpoint(untouched, target, tmp_path)


class _TwoNamesOneKey(RandomAgent):
    def __init__(self):
        super().__init__(4, 2, {"seed": 1})
        self.rng = np.random.default_rng(3)  # "rng_rng", like _rng


def test_two_generators_that_map_to_one_key_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="rng_rng"):
        _TwoNamesOneKey().save(str(tmp_path / "agent.npz"))


class _ArrayNamedLikeAGenerator(RandomAgent):
    def _checkpoint_arrays(self):
        return {**super()._checkpoint_arrays(), "rng_rng": np.zeros(1)}


def test_an_array_that_would_overwrite_another_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="rng_rng"):
        _ArrayNamedLikeAGenerator(4, 2, {"seed": 1}).save(str(tmp_path / "agent.npz"))


# ---------------------------------------------------------------------------
# The random agents' checkpoints from before 2026-10-21
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("agent_cls", "obs_dim", "action_dim"), [(RandomAgent, 4, 2), (ContinuousRandomAgent, 3, 1)]
)
def test_a_random_agents_legacy_generator_keys_still_restore(
    agent_cls, obs_dim, action_dim, tmp_path
):
    saved = agent_cls(obs_dim, action_dim, {"seed": 4})
    for _ in range(5):
        saved.select_action(np.zeros(obs_dim), training=True)
        saved.select_action(np.zeros(obs_dim), training=False)
    saved.save(str(tmp_path / "agent.npz"))
    with np.load(tmp_path / "agent.npz") as data:
        legacy = {k: data[k] for k in data.files if not k.startswith("rng_")}
        legacy["rng_state"] = data["rng_rng"]
        legacy["eval_rng_state"] = data["rng_eval_rng"]
    np.savez(tmp_path / "legacy.npz", **legacy)

    loaded = agent_cls(obs_dim, action_dim, {"seed": 9})
    loaded.load(str(tmp_path / "legacy.npz"))
    assert_same_checkpoint(saved, loaded, tmp_path)


# ---------------------------------------------------------------------------
# rl_core.seeding: one generator's state as a checkpoint array
# ---------------------------------------------------------------------------


def test_a_generator_state_restores_in_place():
    source, target = np.random.default_rng(1), np.random.default_rng(2)
    source.random(3)
    alias = target
    restore_generator(target, generator_state(source))
    assert alias.random() == source.random(), "the same object continues the saved sequence"


def test_a_generator_state_is_a_string_array_that_needs_no_pickle(tmp_path):
    state = generator_state(np.random.default_rng(1))
    assert state.dtype.kind == "U" and state.shape == ()
    np.savez(tmp_path / "g.npz", g=state)
    with np.load(tmp_path / "g.npz", allow_pickle=False) as data:
        assert json.loads(str(data["g"]))["bit_generator"] == "PCG64"


def test_a_state_from_another_bit_generator_is_rejected():
    other = np.random.Generator(np.random.PCG64DXSM(1))
    with pytest.raises(ValueError):
        restore_generator(np.random.default_rng(1), generator_state(other))

"""EpisodeRollout: whole episodes buffered for PPO until a rollout is full, and saved in checkpoints
(CHANGELOG.md, 2026-10-21)."""

import numpy as np
import pytest

from rl_core.agents.rollout import EpisodeRollout, RolloutColumns

OBS_DIM = 3
COLUMNS = ("observations", "actions", "rewards", "dones")
ACTION_SHAPE = {None: (), 2: (2,)}


def _episode(length: int, start: float = 0.0, action_dim: int | None = None) -> RolloutColumns:
    """Observations numbered from ``start`` so the order of steps can be read back."""
    steps = start + np.arange(length, dtype=np.float64)
    actions = (
        np.arange(length, dtype=np.int64)
        if action_dim is None
        else np.tile(steps[:, None], (1, action_dim))
    )
    return RolloutColumns(
        observations=np.tile(steps[:, None], (1, OBS_DIM)),
        actions=actions,
        rewards=-steps,
        dones=np.arange(length) == length - 1,
    )


def _assert_columns_equal(actual: RolloutColumns, expected: RolloutColumns) -> None:
    for name, a, e in zip(COLUMNS, actual, expected, strict=True):
        np.testing.assert_array_equal(a, e, err_msg=name)
        assert a.dtype == e.dtype, f"{name} changed dtype from {e.dtype} to {a.dtype}"


def _joined(*episodes: RolloutColumns) -> RolloutColumns:
    return RolloutColumns(*(np.concatenate(c) for c in zip(*episodes, strict=True)))


def test_episodes_are_buffered_until_the_rollout_holds_enough_steps():
    rollout = EpisodeRollout(rollout_steps=10)
    first, second = _episode(4), _episode(7, start=4)
    assert rollout.add(first) is None, "4 of 10 steps: still buffering"
    assert len(rollout) == 4
    full = rollout.add(second)
    assert full is not None, "11 steps reach the 10 needed; whole episodes, so it may overshoot"
    _assert_columns_equal(full, _joined(first, second))
    assert len(rollout) == 0, "a full rollout is handed over and the buffer starts again"


def test_a_rollout_is_full_at_exactly_rollout_steps():
    rollout = EpisodeRollout(rollout_steps=10)
    rollout.add(_episode(4))
    assert rollout.add(_episode(6, start=4)) is not None, "4 + 6 = 10 steps fill the rollout"


def test_rollout_steps_of_one_hands_over_every_episode():
    episode = _episode(3)
    _assert_columns_equal(EpisodeRollout(rollout_steps=1).add(episode), episode)


def test_an_empty_buffer_saves_nothing():
    """No zero-length arrays: a continuous (0,) actions column would not join (n, action_dim)."""
    assert EpisodeRollout(rollout_steps=5).checkpoint_state() == {}


@pytest.mark.parametrize("action_dim", [None, 2], ids=["discrete", "continuous"])
def test_a_restored_buffer_continues_as_if_never_saved(action_dim):
    first, second = _episode(3, action_dim=action_dim), _episode(4, 3, action_dim=action_dim)
    third = _episode(5, 7, action_dim=action_dim)
    saved = EpisodeRollout(rollout_steps=10)
    saved.add(first)
    saved.add(second)
    state = saved.checkpoint_state()
    assert sorted(state) == sorted(f"ppo_rollout_{name}" for name in COLUMNS)

    restored = EpisodeRollout(rollout_steps=10)
    restored.restore(state, observation_dim=OBS_DIM, action_shape=ACTION_SHAPE[action_dim])
    assert len(restored) == 7
    _assert_columns_equal(restored.add(third), _joined(first, second, third))


def test_restoring_replaces_what_was_buffered():
    restored = EpisodeRollout(rollout_steps=100)
    restored.add(_episode(6, start=50))
    source = EpisodeRollout(rollout_steps=100)
    source.add(_episode(2))
    restored.restore(source.checkpoint_state(), observation_dim=OBS_DIM, action_shape=())
    assert len(restored) == 2, "the checkpoint's buffer, not the one held before the load"


def test_a_checkpoint_without_a_rollout_restores_an_empty_buffer():
    """Checkpoints saved before rollouts were saved, or with an empty buffer, hold no keys."""
    rollout = EpisodeRollout(rollout_steps=100)
    rollout.add(_episode(6))
    rollout.restore({"W1": np.zeros((3, 4))}, observation_dim=OBS_DIM, action_shape=())
    assert len(rollout) == 0


def _state(length: int = 4) -> dict[str, np.ndarray]:
    rollout = EpisodeRollout(rollout_steps=100)
    rollout.add(_episode(length))
    return rollout.checkpoint_state()


def test_restoring_part_of_a_rollout_is_rejected():
    state = _state()
    del state["ppo_rollout_dones"]
    with pytest.raises(ValueError, match="ppo_rollout_dones"):
        EpisodeRollout(rollout_steps=10).restore(state, observation_dim=OBS_DIM, action_shape=())


def test_restoring_columns_of_different_lengths_is_rejected():
    state = _state()
    state["ppo_rollout_rewards"] = state["ppo_rollout_rewards"][:-1]
    with pytest.raises(ValueError, match="ppo_rollout_rewards"):
        EpisodeRollout(rollout_steps=10).restore(state, observation_dim=OBS_DIM, action_shape=())


@pytest.mark.parametrize("observation_dim", [OBS_DIM - 1, OBS_DIM + 1])
def test_restoring_observations_of_another_width_is_rejected(observation_dim):
    with pytest.raises(ValueError, match="ppo_rollout_observations"):
        EpisodeRollout(rollout_steps=10).restore(
            _state(), observation_dim=observation_dim, action_shape=()
        )


@pytest.mark.parametrize(
    ("key", "shape"),
    [
        ("ppo_rollout_observations", (4,)),
        ("ppo_rollout_actions", (4, 1)),
        ("ppo_rollout_rewards", (4, 1)),
        ("ppo_rollout_dones", (4, 1)),
    ],
)
def test_restoring_a_column_of_the_wrong_shape_is_rejected(key, shape):
    """A (n, 1) rewards column would broadcast inside GAE and give quietly wrong advantages."""
    state = _state()
    state[key] = np.zeros(shape, dtype=state[key].dtype)
    with pytest.raises(ValueError, match=key):
        EpisodeRollout(rollout_steps=10).restore(state, observation_dim=OBS_DIM, action_shape=())


def test_a_rejected_restore_leaves_the_buffer_as_it_was():
    rollout = EpisodeRollout(rollout_steps=100)
    rollout.add(_episode(6))
    state = _state()
    state["ppo_rollout_rewards"] = state["ppo_rollout_rewards"][:-1]
    with pytest.raises(ValueError):
        rollout.restore(state, observation_dim=OBS_DIM, action_shape=())
    assert len(rollout) == 6, "a checkpoint that fails validation changes nothing"


def test_a_zero_length_rollout_restores_an_empty_buffer():
    state = {k: v[:0] for k, v in _state().items()}
    rollout = EpisodeRollout(rollout_steps=10)
    rollout.restore(state, observation_dim=OBS_DIM, action_shape=())
    assert len(rollout) == 0
    assert rollout.checkpoint_state() == {}

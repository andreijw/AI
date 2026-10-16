"""ReplayBuffer with continuous actions (``action_dim``), and the discrete default left as it was."""

import numpy as np
import pytest

from rl_core.agents import ReplayBuffer

OBS_DIM, ACTION_DIM = 3, 2


def _transitions(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "obs": rng.standard_normal((n, OBS_DIM)),
        "actions": rng.uniform(-2.0, 2.0, size=(n, ACTION_DIM)),
        "rewards": rng.standard_normal(n),
        "next_obs": rng.standard_normal((n, OBS_DIM)),
        "dones": np.zeros(n, dtype=bool),
    }


def _continuous(capacity: int = 10, seed: int = 4) -> ReplayBuffer:
    return ReplayBuffer(capacity, OBS_DIM, seed=seed, action_dim=ACTION_DIM)


def test_continuous_actions_are_stored_as_float_vectors():
    buffer = _continuous()
    t = _transitions(4)
    buffer.add_batch(t["obs"], t["actions"], t["rewards"], t["next_obs"], t["dones"])
    assert buffer.actions.shape == (10, ACTION_DIM) and buffer.actions.dtype == np.float64
    np.testing.assert_array_equal(buffer.actions[:4], t["actions"])


def test_a_single_continuous_transition_keeps_its_action_vector():
    buffer = _continuous()
    buffer.add(np.zeros(OBS_DIM), np.array([0.25, -1.5]), 1.0, np.ones(OBS_DIM), False)
    np.testing.assert_array_equal(buffer.actions[0], [0.25, -1.5])


def test_continuous_samples_pair_each_action_with_its_transition():
    buffer = _continuous(capacity=8)
    t = _transitions(8)
    t["rewards"] = np.arange(8.0)  # the reward names the row
    buffer.add_batch(t["obs"], t["actions"], t["rewards"], t["next_obs"], t["dones"])
    sample = buffer.sample(5)
    assert sample["actions"].shape == (5, ACTION_DIM)
    rows = sample["rewards"].astype(int)
    np.testing.assert_array_equal(sample["actions"], t["actions"][rows])
    np.testing.assert_array_equal(sample["observations"], t["obs"][rows])


def test_continuous_batches_wrap_around_the_ring():
    buffer = _continuous(capacity=5)
    first, second = _transitions(3, seed=1), _transitions(4, seed=2)
    for t in (first, second):
        buffer.add_batch(t["obs"], t["actions"], t["rewards"], t["next_obs"], t["dones"])
    # 7 transitions into 5 slots: slots 0-1 hold second[2:], slot 2 holds first[2], 3-4 hold second[:2]
    np.testing.assert_array_equal(buffer.actions[:2], second["actions"][2:])
    np.testing.assert_array_equal(buffer.actions[2], first["actions"][2])
    np.testing.assert_array_equal(buffer.actions[3:], second["actions"][:2])


def test_a_continuous_batch_larger_than_the_buffer_keeps_its_newest_actions():
    buffer = _continuous(capacity=5)
    t = _transitions(7)
    buffer.add_batch(t["obs"], t["actions"], t["rewards"], t["next_obs"], t["dones"])
    np.testing.assert_array_equal(buffer.actions, t["actions"][-5:])


@pytest.mark.parametrize("bad_shape", [(), (4,), (4, 1), (4, 3), (4, ACTION_DIM, 1)])
def test_continuous_actions_of_the_wrong_shape_are_rejected(bad_shape):
    """The trainer sends (T, A); anything else is a caller's mistake, not something to reshape."""
    buffer = _continuous()
    t = _transitions(4)
    with pytest.raises(ValueError, match="actions"):
        buffer.add_batch(t["obs"], np.zeros(bad_shape), t["rewards"], t["next_obs"], t["dones"])


def test_a_single_continuous_action_of_the_wrong_shape_is_rejected():
    buffer = _continuous()
    with pytest.raises(ValueError, match="action"):
        buffer.add(np.zeros(OBS_DIM), np.array([0.5]), 1.0, np.zeros(OBS_DIM), False)


@pytest.mark.parametrize("bad", [0, -1])
def test_action_dim_must_be_positive(bad):
    with pytest.raises(ValueError, match="action_dim"):
        ReplayBuffer(10, OBS_DIM, action_dim=bad)


def test_the_discrete_default_is_unchanged():
    """Without action_dim, actions are int64 scalars and sampling draws the same as before."""
    buffer = ReplayBuffer(10, OBS_DIM, seed=7)
    assert buffer.actions.shape == (10,) and buffer.actions.dtype == np.int64
    t = _transitions(10)
    actions = np.arange(10) % 2
    buffer.add_batch(t["obs"], actions, t["rewards"], t["next_obs"], t["dones"])
    expected_indices = np.random.default_rng(7).choice(10, size=4, replace=False)
    np.testing.assert_array_equal(buffer.sample(4)["actions"], actions[expected_indices])

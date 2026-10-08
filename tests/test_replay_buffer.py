"""Unit tests for ReplayBuffer component."""

import numpy as np
import pytest

from rl_cartpole.agents.replay_buffer import ReplayBuffer


def test_init_valid():
    """Verify initialization with valid parameters."""
    buffer = ReplayBuffer(capacity=100, observation_dim=4, seed=42)
    assert len(buffer) == 0
    assert buffer.capacity == 100
    assert buffer.observation_dim == 4
    assert not buffer.can_sample(10)


@pytest.mark.parametrize("invalid_capacity", [0, -10, -1])
def test_init_invalid_capacity(invalid_capacity):
    """Verify ReplayBuffer rejects non-positive capacities."""
    with pytest.raises(ValueError, match="capacity must be a positive integer"):
        ReplayBuffer(capacity=invalid_capacity, observation_dim=4)


@pytest.mark.parametrize("invalid_dim", [0, -4, -1])
def test_init_invalid_observation_dim(invalid_dim):
    """Verify ReplayBuffer rejects non-positive observation dimensions."""
    with pytest.raises(ValueError, match="observation_dim must be a positive integer"):
        ReplayBuffer(capacity=100, observation_dim=invalid_dim)


def test_add_single_transition():
    """Verify adding a single transition updates size and stores data."""
    buffer = ReplayBuffer(capacity=10, observation_dim=4)
    obs = np.array([0.1, 0.2, 0.3, 0.4])
    next_obs = np.array([0.15, 0.25, 0.35, 0.45])

    buffer.add(obs, action=1, reward=1.0, next_obs=next_obs, done=False)

    assert len(buffer) == 1
    sample = buffer.sample(batch_size=1)
    np.testing.assert_allclose(sample["observations"][0], obs)
    assert sample["actions"][0] == 1
    assert sample["rewards"][0] == 1.0
    np.testing.assert_allclose(sample["next_observations"][0], next_obs)
    assert not sample["dones"][0]


def test_ring_buffer_wraparound():
    """Verify older transitions are overwritten when capacity is exceeded."""
    capacity = 5
    buffer = ReplayBuffer(capacity=capacity, observation_dim=1)

    for i in range(8):
        buffer.add(
            obs=np.array([float(i)]),
            action=i % 2,
            reward=float(i),
            next_obs=np.array([float(i + 1)]),
            done=bool(i % 3 == 0),
        )

    assert len(buffer) == capacity
    # Items 0, 1, 2 should have been overwritten by 5, 6, 7
    stored_obs = set(buffer.observations.flatten())
    expected_obs = {3.0, 4.0, 5.0, 6.0, 7.0}
    assert stored_obs == expected_obs


def test_sample_shapes_and_types():
    """Verify sampled batch contains correct dictionary keys, shapes, and dtypes."""
    buffer = ReplayBuffer(capacity=50, observation_dim=4, seed=123)
    for i in range(20):
        buffer.add(
            obs=np.ones(4) * i,
            action=i % 2,
            reward=1.0,
            next_obs=np.ones(4) * (i + 1),
            done=(i == 19),
        )

    batch_size = 16
    assert buffer.can_sample(batch_size)
    batch = buffer.sample(batch_size=batch_size)

    assert isinstance(batch, dict)
    expected_keys = {"observations", "actions", "rewards", "next_observations", "dones"}
    assert set(batch.keys()) == expected_keys

    assert batch["observations"].shape == (batch_size, 4)
    assert batch["actions"].shape == (batch_size,)
    assert batch["rewards"].shape == (batch_size,)
    assert batch["next_observations"].shape == (batch_size, 4)
    assert batch["dones"].shape == (batch_size,)

    assert np.issubdtype(batch["observations"].dtype, np.floating)
    assert np.issubdtype(batch["actions"].dtype, np.integer)
    assert np.issubdtype(batch["rewards"].dtype, np.floating)
    assert np.issubdtype(batch["next_observations"].dtype, np.floating)
    assert batch["dones"].dtype == bool


def test_sample_reproducibility():
    """Verify RNG seed produces deterministic sampling."""
    buf1 = ReplayBuffer(capacity=20, observation_dim=2, seed=42)
    buf2 = ReplayBuffer(capacity=20, observation_dim=2, seed=42)

    for i in range(15):
        obs = np.array([float(i), float(i * 2)])
        next_obs = obs + 0.1
        buf1.add(obs, action=i % 2, reward=float(i), next_obs=next_obs, done=False)
        buf2.add(obs, action=i % 2, reward=float(i), next_obs=next_obs, done=False)

    sample1 = buf1.sample(8)
    sample2 = buf2.sample(8)

    np.testing.assert_allclose(sample1["observations"], sample2["observations"])
    np.testing.assert_array_equal(sample1["actions"], sample2["actions"])


def test_sample_insufficient_data():
    """Verify sampling raises ValueError if buffer has fewer items than batch_size."""
    buffer = ReplayBuffer(capacity=50, observation_dim=2)
    buffer.add(np.zeros(2), 0, 1.0, np.ones(2), False)

    with pytest.raises(ValueError, match="Cannot sample 10 samples from buffer with size 1"):
        buffer.sample(10)


@pytest.mark.parametrize("invalid_batch_size", [0, -1, -5])
def test_sample_invalid_batch_size(invalid_batch_size):
    """Verify sampling raises ValueError for non-positive batch sizes."""
    buffer = ReplayBuffer(capacity=50, observation_dim=2)
    buffer.add(np.zeros(2), 0, 1.0, np.ones(2), False)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        buffer.sample(invalid_batch_size)


def test_add_batch():
    """Verify vectorized batch ingestion via add_batch."""
    buffer = ReplayBuffer(capacity=10, observation_dim=2)
    n = 4
    obs = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    actions = np.array([0, 1, 0, 1], dtype=int)
    rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    next_obs = obs + 1.0
    dones = np.array([False, False, False, True], dtype=bool)

    buffer.add_batch(obs, actions, rewards, next_obs, dones)
    assert len(buffer) == 4

    sample = buffer.sample(4)
    assert len(sample["actions"]) == 4


def test_add_batch_wraparound():
    """Verify add_batch handles wraparound correctly across buffer boundary."""
    capacity = 5
    buffer = ReplayBuffer(capacity=capacity, observation_dim=1)

    obs = np.arange(8, dtype=np.float32).reshape(8, 1)
    actions = np.zeros(8, dtype=int)
    rewards = np.arange(8, dtype=np.float32)
    next_obs = obs + 1.0
    dones = np.zeros(8, dtype=bool)

    buffer.add_batch(obs, actions, rewards, next_obs, dones)
    assert len(buffer) == capacity

    stored = set(buffer.observations.flatten())
    assert stored == {3.0, 4.0, 5.0, 6.0, 7.0}


def test_add_batch_mismatched_lengths():
    """Verify add_batch raises ValueError if inputs have inconsistent lengths."""
    buffer = ReplayBuffer(capacity=10, observation_dim=2)
    obs = np.zeros((3, 2))
    actions = np.zeros(2, dtype=int)  # length 2 vs 3
    rewards = np.zeros(3)
    next_obs = np.zeros((3, 2))
    dones = np.zeros(3, dtype=bool)

    with pytest.raises(ValueError, match="Inconsistent batch lengths"):
        buffer.add_batch(obs, actions, rewards, next_obs, dones)


def test_add_batch_empty():
    """Verify add_batch with empty arrays does not change buffer state."""
    buffer = ReplayBuffer(capacity=5, observation_dim=2)
    obs = np.empty((0, 2))
    actions = np.empty((0,), dtype=int)
    rewards = np.empty((0,))
    next_obs = np.empty((0, 2))
    dones = np.empty((0,), dtype=bool)

    buffer.add_batch(obs, actions, rewards, next_obs, dones)
    assert len(buffer) == 0


def test_add_batch_wraparound_with_offset():
    """Verify add_batch wraparound when starting from a non-zero index."""
    capacity = 5
    buffer = ReplayBuffer(capacity=capacity, observation_dim=1)

    # Insert 3 elements: indices 0, 1, 2 filled, _idx = 3
    for i in range(3):
        buffer.add(np.array([float(i)]), i, float(i), np.array([float(i + 1)]), False)
    assert len(buffer) == 3

    # Now add batch of 3 elements: should fill indices 3, 4, then wrap around to 0
    obs = np.array([[10.0], [11.0], [12.0]])
    actions = np.array([0, 1, 0])
    rewards = np.array([10.0, 11.0, 12.0])
    next_obs = obs + 1.0
    dones = np.array([False, False, True])

    buffer.add_batch(obs, actions, rewards, next_obs, dones)
    assert len(buffer) == 5
    assert buffer._idx == 1  # (3 + 3) % 5 = 1

    # Check that index 0 was overwritten with 12.0
    assert buffer.observations[0, 0] == 12.0
    assert buffer.observations[3, 0] == 10.0
    assert buffer.observations[4, 0] == 11.0


def test_clear():
    """Verify clear empties the buffer."""
    buffer = ReplayBuffer(capacity=10, observation_dim=2)
    buffer.add(np.zeros(2), 0, 1.0, np.ones(2), False)
    assert len(buffer) == 1

    buffer.clear()
    assert len(buffer) == 0
    assert not buffer.can_sample(1)

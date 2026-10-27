"""Unit tests for the proportional prioritised replay buffer (Schaul et al. 2016, section 3.3).

The buffer stores each priority already raised to alpha, gives a new transition the largest
priority so far, samples by stratified proportional draws, and weights each sample by
(N * P(i))^-beta over the buffer's largest such weight.
"""

import numpy as np
import pytest

from rl_core.agents.replay_buffer import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    importance_weights,
)

OBS_DIM = 2


def _buffer(capacity=8, alpha=1.0, epsilon=1e-6, seed=0) -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(capacity, OBS_DIM, seed, alpha=alpha, epsilon=epsilon)


def _fill(buffer: ReplayBuffer, n: int, start: int = 0) -> None:
    rows = np.arange(start, start + n, dtype=np.float64)
    buffer.add_batch(
        np.column_stack([rows, -rows]),
        np.arange(n) % 3,
        rows,
        np.column_stack([rows + 1, -rows]),
        np.zeros(n, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Importance weights, worked by hand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "beta,expected",
    [
        # P = [0.25, 0.75], N = 2: (N P)^-1 = [2, 2/3], over the largest (2): [1, 1/3]
        (1.0, [1.0, 1.0 / 3.0]),
        # (N P)^-0.5 = [sqrt 2, sqrt(2/3)], over sqrt 2: [1, sqrt(1/3)]
        (0.5, [1.0, np.sqrt(1.0 / 3.0)]),
        # beta 0: no correction
        (0.0, [1.0, 1.0]),
    ],
)
def test_importance_weights_match_hand_computation(beta, expected):
    weights = importance_weights(np.array([0.25, 0.75]), p_min=0.25, beta=beta)
    np.testing.assert_allclose(weights, expected, rtol=1e-12)


def test_importance_weights_are_scaled_by_the_buffers_least_likely_item_not_the_batchs():
    # The batch holds only P = 0.5, but the buffer's least likely item has P = 0.25:
    # (2 * 0.5)^-1 / (2 * 0.25)^-1 = 1 / 2. Scaling by the batch's largest weight would give 1.
    weights = importance_weights(np.array([0.5]), p_min=0.25, beta=1.0)
    np.testing.assert_allclose(weights, [0.5], rtol=1e-12)


# ---------------------------------------------------------------------------
# Priorities
# ---------------------------------------------------------------------------


def test_new_transitions_start_at_one_and_then_at_the_largest_priority_so_far_across_a_wrap():
    buffer = _buffer(capacity=5)
    _fill(buffer, 3)
    np.testing.assert_array_equal(buffer.priorities[:3], 1.0, "before any update, p_max is 1")
    buffer.update_priorities(np.array([0]), np.array([2.0]))  # alpha 1: p = 2 + 1e-6
    _fill(buffer, 4, start=3)  # writes slots 3, 4, then wraps to 0, 1
    for slot in (3, 4, 0, 1):
        assert buffer.priorities[slot] == pytest.approx(2.0 + 1e-6, rel=1e-12), f"slot {slot}"
    assert buffer.priorities[2] == 1.0, "slot 2 was not written and keeps its priority"


def test_update_priorities_stores_the_error_plus_epsilon_raised_to_alpha_and_raises_the_maximum():
    buffer = _buffer(alpha=0.5, epsilon=1e-6)
    _fill(buffer, 4)
    buffer.update_priorities(np.array([1, 3]), np.array([-4.0, 0.25]))
    np.testing.assert_allclose(
        buffer.priorities[[1, 3]], [np.sqrt(4.0 + 1e-6), np.sqrt(0.25 + 1e-6)], rtol=1e-12
    )
    assert buffer.max_priority == pytest.approx(np.sqrt(4.0 + 1e-6), rel=1e-12)


def test_alpha_zero_makes_every_priority_one_and_every_weight_one():
    buffer = _buffer(alpha=0.0)
    _fill(buffer, 6)
    buffer.update_priorities(np.arange(6), np.array([0.0, 1.0, 5.0, 0.1, 9.0, 2.0]))
    np.testing.assert_array_equal(buffer.priorities[:6], 1.0)
    np.testing.assert_array_equal(buffer.sample(4, beta=1.0)["weights"], 1.0)


def test_clear_empties_the_buffer_and_resets_the_largest_priority():
    buffer = _buffer()
    _fill(buffer, 4)
    buffer.update_priorities(np.array([2]), np.array([7.0]))
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.max_priority == 1.0, "a stale maximum would follow into the next fill"
    _fill(buffer, 2)
    np.testing.assert_array_equal(buffer.priorities[:2], 1.0)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_samples_are_drawn_in_proportion_to_priority():
    buffer = _buffer(capacity=4, seed=11)
    _fill(buffer, 4)
    buffer.update_priorities(np.arange(4), np.array([1.0, 2.0, 3.0, 4.0]))  # P = [.1 .2 .3 .4]
    drawn = np.concatenate([buffer.sample(4)["indices"] for _ in range(5000)])
    shares = np.bincount(drawn, minlength=4) / drawn.size
    np.testing.assert_allclose(shares, [0.1, 0.2, 0.3, 0.4], atol=0.01)


def test_equal_priorities_with_a_batch_of_the_whole_buffer_draw_each_item_once():
    """Stratified: one draw from each equal slice of the cumulative priority. Independent draws
    would repeat an index in all but 0.24% of batches of 8 from 8."""
    buffer = _buffer(capacity=8, seed=2)
    _fill(buffer, 8)
    for _ in range(5):
        np.testing.assert_array_equal(np.sort(buffer.sample(8)["indices"]), np.arange(8))


def test_a_dominant_priority_is_drawn_several_times_in_one_batch():
    buffer = _buffer(capacity=4, seed=3)
    _fill(buffer, 4)
    buffer.update_priorities(np.array([0]), np.array([1000.0]))  # 1000 of a total mass of 1003
    indices = buffer.sample(4)["indices"]
    assert np.sum(indices == 0) >= 3, "samples are with replacement and must not be de-duplicated"


def test_sample_returns_the_rows_of_its_indices_with_their_weights():
    buffer = _buffer(capacity=2, seed=4)
    _fill(buffer, 2)
    buffer.update_priorities(np.array([0, 1]), np.array([1.0, 3.0]))  # P ~ [0.25, 0.75]
    sample = buffer.sample(2, beta=0.5)
    idx = sample["indices"]
    np.testing.assert_array_equal(sample["observations"], buffer.observations[idx])
    np.testing.assert_array_equal(sample["rewards"], buffer.rewards[idx])
    expected = np.where(idx == 0, 1.0, np.sqrt(1.0 / 3.0))
    np.testing.assert_allclose(sample["weights"], expected, rtol=1e-5)  # epsilon shifts P a hair


def test_sample_without_beta_is_a_drop_in_for_the_uniform_buffer_and_corrects_fully():
    buffer = _buffer(capacity=2, seed=5)
    _fill(buffer, 2)
    buffer.update_priorities(np.array([0, 1]), np.array([1.0, 3.0]))
    sample = buffer.sample(2)
    expected = np.where(sample["indices"] == 0, 1.0, 1.0 / 3.0)
    np.testing.assert_allclose(sample["weights"], expected, rtol=1e-5)


def test_sampling_more_than_the_buffer_holds_is_refused_as_for_the_uniform_buffer():
    buffer = _buffer()
    _fill(buffer, 3)
    with pytest.raises(ValueError, match="Cannot sample 4"):
        buffer.sample(4)


@pytest.mark.parametrize("beta", [-0.1, 1.5])
def test_beta_outside_zero_to_one_is_refused(beta):
    buffer = _buffer()
    _fill(buffer, 3)
    with pytest.raises(ValueError, match="beta"):
        buffer.sample(2, beta=beta)


@pytest.mark.parametrize(
    "kwargs,name",
    [({"alpha": -0.1}, "alpha"), ({"alpha": 1.5}, "alpha"), ({"epsilon": 0.0}, "epsilon")],
)
def test_invalid_alpha_or_epsilon_is_refused(kwargs, name):
    with pytest.raises(ValueError, match=name):
        PrioritizedReplayBuffer(8, OBS_DIM, 0, **{"alpha": 0.6, "epsilon": 1e-6, **kwargs})


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_a_restored_buffer_has_the_same_priorities_and_draws_the_same_samples():
    buffer = _buffer(seed=6)
    _fill(buffer, 6)
    buffer.update_priorities(np.array([1, 4]), np.array([3.0, 0.5]))
    state = {k: np.copy(v) for k, v in buffer.checkpoint_state().items()}
    restored = _buffer(seed=99)
    restored.restore(state)
    np.testing.assert_array_equal(restored.priorities[:6], buffer.priorities[:6])
    assert restored.max_priority == buffer.max_priority
    np.testing.assert_array_equal(restored.sample(4)["indices"], buffer.sample(4)["indices"])


def test_a_uniform_buffers_checkpoint_restores_with_every_priority_at_one():
    uniform = ReplayBuffer(8, OBS_DIM, 0)
    _fill(uniform, 5)
    restored = _buffer()
    restored.restore(uniform.checkpoint_state())
    assert len(restored) == 5
    np.testing.assert_array_equal(restored.priorities[:5], 1.0)
    assert restored.max_priority == 1.0


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.pop("replay_max_priority"),  # one of the two keys only
        lambda s: s.__setitem__("replay_priorities", s["replay_priorities"][:-1]),  # wrong shape
        lambda s: s.__setitem__("replay_priorities", -s["replay_priorities"]),  # not positive
    ],
    ids=["partial", "shape", "negative"],
)
def test_a_partial_or_malformed_priority_checkpoint_is_refused_and_changes_nothing(change):
    buffer = _buffer()
    _fill(buffer, 4)
    state = {k: np.copy(v) for k, v in buffer.checkpoint_state().items()}
    change(state)
    target = _buffer()
    _fill(target, 2)
    with pytest.raises(ValueError, match="replay_"):
        target.restore(state)
    assert len(target) == 2, "a refused restore leaves the buffer as it was"

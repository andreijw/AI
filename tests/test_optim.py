"""The shared Adam optimiser and the ``optimizer`` config option (rl_core.agents.optim)."""

import numpy as np
import pytest

from rl_core.agents.optim import Adam, optimizer_from_config
from tests.helpers import ReferenceAdam

SHAPES = {"W": (3, 2), "b": (2,), "s": ()}


def _grads(rng: np.random.Generator, scale: float) -> dict[str, np.ndarray | float]:
    grads: dict[str, np.ndarray | float] = {
        name: scale * rng.standard_normal(shape) for name, shape in SHAPES.items() if shape
    }
    grads["s"] = float(scale * rng.standard_normal())  # a scalar parameter, like a value bias
    return grads


def _zeros() -> dict[str, np.ndarray | float]:
    return {name: (np.zeros(shape) if shape else 0.0) for name, shape in SHAPES.items()}


@pytest.mark.parametrize("max_grad_norm", [None, 0.5])
def test_adam_steps_match_an_independent_implementation(max_grad_norm):
    """Three steps, the second far above the clip threshold, agree with the reference."""
    adam = Adam(_zeros(), max_grad_norm=max_grad_norm)
    reference = ReferenceAdam(learning_rate=0.01, max_grad_norm=max_grad_norm)
    rng = np.random.default_rng(7)

    for scale in (0.01, 50.0, 0.3):
        grads = _grads(rng, scale)
        steps = adam.step(dict(grads), 0.01)
        expected = reference.step(grads)
        assert steps.keys() == expected.keys()
        for name in expected:
            np.testing.assert_allclose(steps[name], expected[name], rtol=1e-12, atol=1e-15)

    assert adam.t == 3


def test_adam_without_a_clip_does_not_clip_large_gradients():
    """With no max_grad_norm, a huge gradient is not scaled down, unlike with a clip."""
    grads = {"W": np.full((3, 2), 100.0), "b": np.full(2, 100.0), "s": 100.0}
    unclipped, clipped = Adam(_zeros()), Adam(_zeros(), max_grad_norm=0.5)
    unclipped.step(grads, 0.1)
    clipped.step(grads, 0.1)
    # The first moment is (1 - beta1) * the gradient Adam actually used.
    np.testing.assert_allclose(unclipped.m["W"], 0.1 * 100.0)
    norm = np.sqrt(9 * 100.0**2)  # nine elements of 100
    np.testing.assert_allclose(clipped.m["W"], 0.1 * 100.0 * 0.5 / (norm + 1e-6))


def test_adam_step_leaves_the_gradients_unchanged():
    grads = _grads(np.random.default_rng(1), 2.0)
    before = {name: np.array(value, copy=True) for name, value in grads.items()}
    Adam(_zeros(), max_grad_norm=0.5).step(grads, 0.01)
    for name, value in grads.items():
        np.testing.assert_array_equal(value, before[name])


def test_adam_state_round_trips():
    """Saved state restores the moments and the step count exactly."""
    adam = Adam(_zeros(), max_grad_norm=0.5)
    rng = np.random.default_rng(2)
    for _ in range(2):
        adam.step(_grads(rng, 1.0), 0.01)

    restored = Adam(_zeros(), max_grad_norm=0.5)
    restored.load_state(adam.state())

    assert restored.t == 2
    grads = _grads(rng, 1.0)
    expected, actual = adam.step(dict(grads), 0.01), restored.step(dict(grads), 0.01)
    for name, step in expected.items():
        np.testing.assert_array_equal(actual[name], step)


def test_adam_state_without_saved_moments_starts_fresh():
    """A checkpoint written before the optimiser existed loads with zero moments and step 0."""
    adam = Adam(_zeros())
    adam.step(_grads(np.random.default_rng(3), 1.0), 0.01)
    adam.load_state({"W": np.ones((3, 2))})  # weights only
    assert adam.t == 0
    for name in SHAPES:
        np.testing.assert_array_equal(adam.m[name], 0.0)
        np.testing.assert_array_equal(adam.v[name], 0.0)


def test_adam_state_keys_do_not_collide_with_parameter_names():
    assert all(key.startswith("adam_") for key in Adam(_zeros()).state())


def test_optimizer_from_config_defaults_to_plain_sgd():
    assert optimizer_from_config({}, _zeros()) is None
    assert optimizer_from_config({"optimizer": "sgd"}, _zeros()) is None


def test_optimizer_from_config_builds_adam_with_the_configured_clip():
    adam = optimizer_from_config({"optimizer": "Adam", "max_grad_norm": 0.5}, _zeros())
    assert isinstance(adam, Adam)
    assert adam.max_grad_norm == 0.5
    unclipped = optimizer_from_config({"optimizer": "adam"}, _zeros())
    assert isinstance(unclipped, Adam)
    assert unclipped.max_grad_norm is None


def test_optimizer_from_config_rejects_an_unknown_optimizer():
    with pytest.raises(ValueError, match="optimizer must be one of"):
        optimizer_from_config({"optimizer": "rmsprop"}, _zeros())


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_optimizer_from_config_rejects_a_non_positive_clip(bad):
    with pytest.raises(ValueError, match="max_grad_norm must be positive"):
        optimizer_from_config({"optimizer": "adam", "max_grad_norm": bad}, _zeros())


def test_optimizer_from_config_rejects_a_clip_without_adam():
    """A clip with plain SGD would be silently ignored, so it is an error instead."""
    with pytest.raises(ValueError, match='max_grad_norm requires optimizer "adam"'):
        optimizer_from_config({"max_grad_norm": 0.5}, _zeros())


def test_adam_state_with_only_a_step_count_keeps_it():
    """Old continuous checkpoints stored only adam_t: it is restored, the moments start at zero."""
    adam = Adam(_zeros())
    adam.load_state({"adam_t": np.array(7)})
    assert adam.t == 7
    np.testing.assert_array_equal(adam.m["W"], 0.0)


def test_adam_state_with_a_wrongly_shaped_moment_is_rejected():
    state = Adam(_zeros()).state()
    state["adam_m_W"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match=r"adam_m_W has shape \(2, 3\), expected \(3, 2\)"):
        Adam(_zeros()).load_state(state)


def test_two_optimizers_saved_under_different_prefixes_keep_their_own_state():
    """An agent with an actor and a critic optimizer saves both into one checkpoint."""
    params = {"W": np.zeros(3)}
    actor, critic = Adam(params), Adam(params)
    actor.step({"W": np.ones(3)}, 0.1)
    for _ in range(3):
        critic.step({"W": np.full(3, 2.0)}, 0.1)
    data = {**actor.state(prefix="actor_adam_"), **critic.state(prefix="critic_adam_")}

    restored_actor, restored_critic = Adam(params), Adam(params)
    restored_actor.load_state(data, prefix="actor_adam_")
    restored_critic.load_state(data, prefix="critic_adam_")

    assert (restored_actor.t, restored_critic.t) == (1, 3)
    np.testing.assert_array_equal(restored_actor.m["W"], actor.m["W"])
    np.testing.assert_array_equal(restored_critic.v["W"], critic.v["W"])


def test_the_default_prefix_is_adam_so_existing_checkpoints_still_load():
    params = {"W": np.zeros(2)}
    adam = Adam(params)
    adam.step({"W": np.ones(2)}, 0.1)
    assert set(adam.state()) == {"adam_t", "adam_m_W", "adam_v_W"}
    restored = Adam(params)
    restored.load_state(adam.state())
    assert restored.t == 1

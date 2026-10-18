"""The shared multi-layer perceptron: initialisation, forward pass and hand-written backward pass."""

import numpy as np
import pytest

from rl_core.agents.mlp import init_mlp, mlp_backward, mlp_forward
from tests.helpers import assert_gradients_match, numerical_grad


def test_init_has_one_weight_and_bias_per_layer_in_torch_default_range():
    params = init_mlp([3, 5, 4, 2], np.random.default_rng(0))
    assert sorted(params) == ["W0", "W1", "W2", "b0", "b1", "b2"]
    for i, (fan_in, fan_out) in enumerate([(3, 5), (5, 4), (4, 2)]):
        assert params[f"W{i}"].shape == (fan_in, fan_out) and params[f"b{i}"].shape == (fan_out,)
        bound = 1.0 / np.sqrt(fan_in)  # PyTorch's nn.Linear default, which SB3 uses
        for name in (f"W{i}", f"b{i}"):
            assert params[name].dtype == np.float64
            assert np.abs(params[name]).max() <= bound
            if params[name].size >= 10:  # too few draws to judge otherwise
                assert np.abs(params[name]).max() > 0.5 * bound, "draws should fill the range"


def test_init_is_reproducible_per_generator_state():
    a = init_mlp([3, 4, 1], np.random.default_rng(5))
    b = init_mlp([3, 4, 1], np.random.default_rng(5))
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


@pytest.mark.parametrize("sizes", [[3], []])
def test_init_needs_at_least_one_layer_of_positive_sizes(sizes):
    with pytest.raises(ValueError, match="sizes"):
        init_mlp(sizes, np.random.default_rng(0))


def test_init_rejects_a_non_positive_size():
    with pytest.raises(ValueError, match="sizes"):
        init_mlp([3, 0, 1], np.random.default_rng(0))


def test_forward_matches_a_hand_computation():
    params = {
        "W0": np.array([[1.0, -1.0], [2.0, 0.5]]),
        "b0": np.array([0.0, 0.25]),
        "W1": np.array([[2.0], [-3.0]]),
        "b1": np.array([1.0]),
    }
    x = np.array([[1.0, 1.0], [-1.0, 0.0]])
    out, _ = mlp_forward(params, x)
    # row 1: pre = [3, -0.25] -> relu [3, 0] -> 2*3 + 1 = 7
    # row 2: pre = [-1, 1.25] -> relu [0, 1.25] -> -3*1.25 + 1 = -2.75 (no ReLU on the output)
    np.testing.assert_allclose(out, [[7.0], [-2.75]])


def _fixture():
    """A batch whose hidden pre-activations sit well clear of the ReLU kink for a 1e-5 nudge."""
    params = init_mlp([4, 6, 5, 2], np.random.default_rng(3))
    x = np.random.default_rng(4).standard_normal((7, 4))
    _, cache = mlp_forward(params, x)
    for pre in cache["pre"][:-1]:
        assert np.abs(pre).min() > 1e-3, "fixture sits on a ReLU kink"
    return params, x


def test_backward_matches_finite_differences_for_every_parameter():
    params, x = _fixture()
    weights = np.random.default_rng(9).standard_normal((7, 2))  # loss = sum(weights * out)

    def loss():
        return float(np.sum(weights * mlp_forward(params, x)[0]))

    _, cache = mlp_forward(params, x)
    grads, _ = mlp_backward(params, cache, weights)
    assert_gradients_match(loss, params, grads)


def test_backward_returns_the_gradient_with_respect_to_the_input():
    params, x = _fixture()
    weights = np.random.default_rng(9).standard_normal((7, 2))
    _, cache = mlp_forward(params, x)
    _, d_x = mlp_backward(params, cache, weights)
    expected = numerical_grad(lambda: float(np.sum(weights * mlp_forward(params, x)[0])), x)
    np.testing.assert_allclose(d_x, expected, rtol=1e-6, atol=1e-8)


def test_backward_changes_neither_the_parameters_nor_the_upstream_gradient():
    params, x = _fixture()
    d_out = np.ones((7, 2))
    before = {name: value.copy() for name, value in params.items()}
    _, cache = mlp_forward(params, x)
    mlp_backward(params, cache, d_out)
    for name, value in params.items():
        np.testing.assert_array_equal(value, before[name])
    np.testing.assert_array_equal(d_out, np.ones((7, 2)))

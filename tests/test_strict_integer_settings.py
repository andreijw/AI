"""Every agent rejects a non-integer for its integer settings at construction, naming the key.

Roadmap: "Strict Integer Settings for Every Agent". Before, ``int(...)`` turned 2.5 into 2, 4.0 into 4,
``True`` into 1 and ``"4"`` into 4 without a word. DQN and the PPO-specific settings are covered in
``tests/test_dqn_validation.py``, ``tests/test_discrete_ppo.py`` and ``tests/test_continuous_ppo_rollouts.py``.
"""

import numpy as np
import pytest

from rl_core.agents import (
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    DDPGAgent,
    PPOAgent,
    ReinforceAgent,
)

BAD_INTEGERS = [2.5, 4.0, True, "4"]
BOUNDS = {"action_low": -2.0, "action_high": 2.0}


@pytest.mark.parametrize(
    "agent_cls,extra",
    [
        (ReinforceAgent, {}),
        (ActorCriticAgent, {}),
        (PPOAgent, {}),
        (ContinuousActorCriticAgent, BOUNDS),
        (ContinuousPPOAgent, BOUNDS),
    ],
)
@pytest.mark.parametrize("bad", [*BAD_INTEGERS, 0, -3])
def test_hidden_dim_must_be_a_positive_integer(agent_cls, extra, bad):
    with pytest.raises(ValueError, match="hidden_dim must be a positive integer"):
        agent_cls(3, 2, {"seed": 0, "hidden_dim": bad, **extra})


def _ddpg(**config) -> DDPGAgent:
    settings = {"seed": 0, "hidden_sizes": [6, 5], "batch_size": 4, "learning_starts": 0, **BOUNDS}
    return DDPGAgent(3, 2, {**settings, **config})


DDPG_CASES = [
    (key, value, kind)
    for key, out_of_range, kind in [
        ("batch_size", 0, "positive"),
        ("buffer_capacity", 0, "positive"),
        ("max_updates_per_call", 0, "positive"),
        ("learning_starts", -1, "non-negative"),
    ]
    for value in [*BAD_INTEGERS, out_of_range]
]


@pytest.mark.parametrize("key,value,kind", DDPG_CASES)
def test_ddpg_integer_settings_must_be_integers_in_range(key, value, kind):
    with pytest.raises(ValueError, match=f"{key} must be a {kind} integer"):
        _ddpg(**{key: value})


def test_ddpg_accepts_zero_learning_starts_and_an_unset_update_cap():
    agent = _ddpg(learning_starts=0, max_updates_per_call=None)
    assert agent.learning_starts == 0
    assert agent.max_updates_per_call is None


@pytest.mark.parametrize(
    "widths,message",
    [
        ([64, 2.5], r"hidden_sizes\[1\] must be a positive integer, got 2.5"),
        (np.array([64.0, 64.0]), r"hidden_sizes\[0\] must be a positive integer"),
        (["64", 64], r"hidden_sizes\[0\] must be a positive integer"),
        ([4, 0], r"hidden_sizes\[1\] must be a positive integer"),
    ],
)
def test_ddpg_hidden_sizes_must_be_positive_integer_widths(widths, message):
    with pytest.raises(ValueError, match=message):
        _ddpg(hidden_sizes=widths)


@pytest.mark.parametrize("widths", [[], np.array([], dtype=int)])
def test_ddpg_hidden_sizes_must_hold_at_least_one_width(widths):
    with pytest.raises(ValueError, match="hidden_sizes must hold at least one width"):
        _ddpg(hidden_sizes=widths)


def test_ddpg_accepts_numpy_integer_widths():
    agent = _ddpg(hidden_sizes=np.array([6, 5]))
    assert agent.hidden_sizes == [6, 5]
    assert all(type(width) is int for width in agent.hidden_sizes)

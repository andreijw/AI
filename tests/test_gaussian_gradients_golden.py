"""The Gaussian agents' gradients, bit for bit, as recorded before their backward passes were shared.

The other maths tests build their expected values from the agent's own loss_and_grads, so a
rewritten backward pass would agree with itself there. This file compares against arrays recorded
from the code before ContinuousActorCriticAgent and ContinuousPPOAgent shared one, for both
agents, bounded and unbounded means and one and two action dimensions: the gradients, the loss
components, the weights after one Adam step and the checkpoint's keys in order.

The names, their order and every dtype must match exactly. The numbers must agree to a relative
1e-12: the file was recorded on one machine (aarch64, OpenBLAS), and other BLAS builds and SIMD
paths may sum in another order and differ in the last bits, while a wrong sign, a dropped factor
or a missing mask changes them by many orders of magnitude more. On the recording machine the
refactor was bit-identical (the sign of an exact zero aside); the pull request records that run.

A deliberate change to the maths regenerates the file: run this module with
RECORD_GAUSSIAN_GOLDEN=1, check why every changed array changed, then run it again without the
variable (a recording run fails on purpose, so it can never pass by accepting any output).
"""

import itertools
import os
from pathlib import Path

import numpy as np
import pytest

from rl_core.agents.continuous_actor_critic import ContinuousActorCriticAgent
from rl_core.agents.continuous_ppo import ContinuousPPOAgent
from tests.agent_internals import (
    adam_step,
    gaussian_forward,
    log_prob,
    loss_and_grads,
    ppo_minibatch_loss_and_grads,
    weights,
)

GOLDEN = Path(__file__).parent / "data" / "gaussian_gradients.npz"
RECORD = os.environ.get("RECORD_GAUSSIAN_GOLDEN") == "1"
N_STEPS, OBS_DIM, HIDDEN_DIM = 7, 3, 5
BOUNDS = {"action_low": -2.0, "action_high": 3.0}
RTOL, ATOL = 1e-12, 1e-14

# PPO's steps, chosen by hand so every case reaches each branch of its policy gradient. The old
# log-probability is the current one minus LOG_RATIO, so ratio = exp(LOG_RATIO); with the clip
# range 0.2 a step is clipped (no gradient) when its ratio is outside [0.8, 1.2] on the side its
# advantage pushes, and stale (no gradient) when |log-ratio| > 20.
CLIP = 0.2
LOG_RATIO = np.array([0.0, -0.05, 0.4, -0.4, 0.4, -0.4, 25.0])
PPO_ADVANTAGES = np.array([0.8, -1.2, 1.5, -0.7, -0.9, 1.1, -2.0])
ACTIVE, CLIPPED, STALE = [0, 1, 4, 5], [2, 3], [6]

CASES = [
    (agent, bounded, action_dim, advantages)
    for agent, bounded, action_dim in itertools.product(("ac", "ppo"), (False, True), (1, 2))
    for advantages in (("normalised", "given") if agent == "ac" else ("given",))
]


def _case_id(case) -> str:
    agent, bounded, action_dim, advantages = case
    return f"{agent}-{'bounded' if bounded else 'unbounded'}-{action_dim}d-{advantages}"


def _outputs(case, tmp_path) -> dict[str, np.ndarray]:
    agent_name, bounded, action_dim, advantage_mode = case
    agent_cls = ContinuousActorCriticAgent if agent_name == "ac" else ContinuousPPOAgent
    config = {
        "hidden_dim": HIDDEN_DIM,
        "seed": 11,
        "value_coef": 0.7,
        "entropy_coef": 0.05,
        "clip_epsilon": CLIP,
    }
    if bounded:
        config.update(bounded_mean=True, **BOUNDS)
    agent = agent_cls(observation_dim=OBS_DIM, action_dim=action_dim, config=config)

    rng = np.random.default_rng(5 + action_dim)
    states = rng.standard_normal((N_STEPS, OBS_DIM))
    actions = rng.uniform(-2.5, 3.5, size=(N_STEPS, action_dim))
    returns = rng.normal(0.0, 2.0, size=N_STEPS)
    advantages = rng.normal(0.0, 1.0, size=N_STEPS)

    if agent_name == "ac":
        given = advantages if advantage_mode == "given" else None
        components, grads = loss_and_grads(agent, states, actions, returns, given)
    else:
        mu, log_std, _, _ = gaussian_forward(agent, states)
        current = log_prob(agent, actions, mu, log_std)
        components, grads = ppo_minibatch_loss_and_grads(
            agent, states, actions, returns, PPO_ADVANTAGES, current - LOG_RATIO
        )

    out = {f"grad_{k}": np.asarray(v) for k, v in grads.items()}
    out.update({f"loss_{k}": np.asarray(v) for k, v in components.items()})
    out["grad_order"] = np.array(list(grads))
    adam_step(agent, grads, 0.01)
    out.update({f"after_{k}": v for k, v in weights(agent).items()})
    path = str(tmp_path / "agent.npz")
    agent.save(path)
    with np.load(path) as data:
        out["checkpoint_keys"] = np.array(data.files)
    return out


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_gradients_match_the_recorded_arrays_exactly(case, tmp_path):
    outputs = _outputs(case, tmp_path)
    prefix = _case_id(case) + "/"
    if RECORD:
        recorded = dict(np.load(GOLDEN)) if GOLDEN.exists() else {}
        recorded.update({prefix + k: v for k, v in outputs.items()})
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN, **recorded)
        pytest.fail(f"Recorded {prefix}; rerun without RECORD_GAUSSIAN_GOLDEN.")
    with np.load(GOLDEN) as golden:
        expected = {k[len(prefix) :]: golden[k] for k in golden.files if k.startswith(prefix)}
    assert sorted(outputs) == sorted(expected)
    for name, value in outputs.items():
        assert value.dtype == expected[name].dtype, f"{prefix}{name} changed dtype"
        if value.dtype.kind in "US":  # key orders: exact
            np.testing.assert_array_equal(value, expected[name], err_msg=f"{prefix}{name} differs")
        else:
            np.testing.assert_allclose(
                value, expected[name], rtol=RTOL, atol=ATOL, err_msg=f"{prefix}{name} differs"
            )


def test_the_recorded_cases_cover_every_agent_mean_and_action_size():
    with np.load(GOLDEN) as golden:
        cases = {key.split("/")[0] for key in golden.files}
    assert cases == {_case_id(case) for case in CASES}
    assert len(cases) == 12


def test_the_ppo_steps_reach_active_clipped_and_stale_branches():
    """The surrogate's minimum is the clipped term exactly on the CLIPPED steps, and only the
    STALE step has a log-ratio beyond 20, so each kind of step is in every PPO case."""
    ratio = np.exp(np.clip(LOG_RATIO, -20.0, 20.0))
    unclipped = ratio * PPO_ADVANTAGES
    clipped = np.clip(ratio, 1.0 - CLIP, 1.0 + CLIP) * PPO_ADVANTAGES
    stale = np.abs(LOG_RATIO) > 20.0
    gradient = (unclipped <= clipped) & ~stale
    assert np.flatnonzero(gradient).tolist() == ACTIVE
    assert np.flatnonzero(~gradient & ~stale).tolist() == CLIPPED
    assert np.flatnonzero(stale).tolist() == STALE
    assert (ratio[CLIPPED] < 1.0 - CLIP).any() and (ratio[CLIPPED] > 1.0 + CLIP).any()

"""DQN's seeded training with uniform replay, as recorded on 2026-10-27.

The other DQN tests replay an update through the agent's own loss method, so a reordered draw or a
changed sampling path would agree with itself there. This file compares against arrays recorded
from the code of 2026-10-27, before a change to the replay buffer or the update could alter them
(the parked branch ``feature/dqn-prioritized-replay`` was checked against it): plain SGD with its
element clip; Adam with a
global-norm clip, a learning-rate decay, a linear epsilon schedule and several gradient steps per
call; and Polyak targets. Each case plays seeded episodes through ``select_action`` and ``update``,
then records every metric, every checkpoint array (weights, Adam state, counters, the replay buffer,
every generator) with the checkpoint's keys, and a greedy action.

Names and every dtype must match exactly; the checkpoint's keys are compared as a set, since
loading goes by name. Numbers agree to a relative 1e-12, as in tests/test_ddpg_td3_golden.py: the
file was recorded on one machine (aarch64, OpenBLAS). On the recording machine, DQN_GOLDEN_EXACT=1
demands every array bit for bit.

A deliberate change to the training regenerates the file: run this module with
RECORD_DQN_GOLDEN=1, check why every changed array changed, then run it again without the variable
(a recording run fails on purpose, so it can never pass by accepting any output).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from rl_core.agents import DQNAgent

GOLDEN = Path(__file__).parent / "data" / "dqn_training.npz"
RECORD = os.environ.get("RECORD_DQN_GOLDEN") == "1"
EXACT = os.environ.get("DQN_GOLDEN_EXACT") == "1"
RTOL, ATOL = 1e-12, 1e-14

OBS_DIM, ACT_DIM, EPISODE, EPISODES = 2, 3, 12, 6
BASE = {
    "seed": 17,
    "hidden_dim": 5,
    "learning_rate": 0.05,
    "gamma": 0.9,
    "batch_size": 8,
    "buffer_capacity": 40,  # wraps in the fourth episode
    "min_buffer_size": 20,  # the second episode ends warm-up
    "target_update_frequency": 3,
}
CASES = {
    "sgd": {"grad_clip_value": 0.5},
    "adam-decay": {
        "optimizer": "adam",
        "max_grad_norm": 0.5,
        "anneal_steps": 60,
        "exploration_steps": 30,
        "updates_per_step": 0.5,
        "max_updates_per_call": 5,
    },
    "polyak": {"optimizer": "adam", "tau": 0.2, "updates_per_step": 0.25},
}


def _outputs(name: str, tmp_path) -> dict[str, np.ndarray]:
    agent = DQNAgent(OBS_DIM, ACT_DIM, {**BASE, **CASES[name]})
    rng = np.random.default_rng(5)
    out: dict[str, np.ndarray] = {}
    for episode in range(EPISODES):
        obs = rng.standard_normal((EPISODE + 1, OBS_DIM))
        actions = np.array([agent.select_action(o, training=True) for o in obs[:-1]])
        terminated = np.zeros(EPISODE, dtype=bool)
        terminated[-1] = episode % 2 == 1
        metrics = agent.update(
            {
                "observations": obs[:-1],
                "actions": actions,
                "rewards": rng.normal(size=EPISODE),
                "next_observations": obs[1:],
                "terminated": terminated,
            }
        )
        out[f"actions_{episode}"] = actions
        out[f"metric_names/{episode}"] = np.array(list(metrics), dtype=str)
        out.update({f"metric_{episode}/{k}": np.asarray(v) for k, v in metrics.items()})
    out["greedy_action"] = np.asarray(agent.select_action(np.array([0.4, -0.3]), training=False))
    path = tmp_path / "agent.npz"
    agent.save(str(path))
    with np.load(path) as data:
        out["checkpoint_keys"] = np.array(data.files)
        out.update({f"checkpoint_{k}": data[k] for k in data.files})
    return out


@pytest.mark.parametrize("name", list(CASES))
def test_training_matches_the_recorded_arrays(name, tmp_path):
    outputs = _outputs(name, tmp_path)
    prefix = name + "/"
    if RECORD:
        recorded = dict(np.load(GOLDEN)) if GOLDEN.exists() else {}
        recorded.update({prefix + k: v for k, v in outputs.items()})
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN, **recorded)
        pytest.fail(f"Recorded {prefix}; rerun without RECORD_DQN_GOLDEN.")
    with np.load(GOLDEN) as golden:
        expected = {k[len(prefix) :]: golden[k] for k in golden.files if k.startswith(prefix)}
    assert sorted(outputs) == sorted(expected), f"{name} records other arrays than the file"
    for key, value in outputs.items():
        assert value.dtype == expected[key].dtype, f"{prefix}{key} changed dtype"
        if key == "checkpoint_keys":  # loading goes by name, so the order is not a contract
            value, expected[key] = np.sort(value), np.sort(expected[key])
        if EXACT or value.dtype.kind not in "fc":  # names, counts and generator states: exact
            np.testing.assert_array_equal(value, expected[key], err_msg=f"{prefix}{key} differs")
        else:
            np.testing.assert_allclose(
                value, expected[key], rtol=RTOL, atol=ATOL, err_msg=f"{prefix}{key} differs"
            )


def test_the_recorded_cases_warm_up_train_and_wrap_the_buffer():
    """Without these the file could not notice a broken warm-up, a skipped update or a buffer that
    never wraps."""
    with np.load(GOLDEN) as golden:
        assert {key.split("/")[0] for key in golden.files} == set(CASES)
        assert set(golden["sgd/metric_names/0"]) == {"epsilon"}, "episode 0 should only warm up"
        assert "loss" in set(golden["sgd/metric_names/1"]), "episode 1 should train"
        assert int(golden["sgd/checkpoint_replay_size"]) == BASE["buffer_capacity"]
        assert int(golden["sgd/checkpoint_replay_idx"]) == (EPISODE * EPISODES) % 40

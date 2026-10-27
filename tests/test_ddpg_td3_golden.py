"""DDPG's and TD3's seeded training, as recorded before the PyTorch backend read its layout from hooks.

The other DDPG and TD3 tests replay an update through the agent's own loss methods, so a reordered
step or a renamed Adam key would agree with itself there. This file compares against arrays
recorded from the code before the networks, optimizers and target pairs became declared hooks: for
Gaussian and Ornstein-Uhlenbeck exploration, with and without gradient clipping, and for TD3 with
a ``policy_delay`` boundary falling across two ``update`` calls. Each case plays seeded episodes
through ``select_action``, ``update`` and ``end_episode``, then records every metric, every
checkpoint array (weights, Adam states, counters, the replay buffer, every generator) with the
checkpoint's keys, and a greedy action.

Names and every dtype must match exactly; the checkpoint's keys are compared as a set, since
loading goes by name. Numbers agree to a relative 1e-12, as in
tests/test_gaussian_gradients_golden.py: the file was recorded on one machine (aarch64, OpenBLAS)
and other BLAS builds may differ in the last bits. On the recording machine, DDPG_GOLDEN_EXACT=1
demands every array bit for bit; the pull request records that run.

A deliberate change to the training regenerates the file: run this module with
RECORD_DDPG_GOLDEN=1, check why every changed array changed, then run it again without the
variable (a recording run fails on purpose, so it can never pass by accepting any output).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from rl_core.agents import DDPGAgent, TD3Agent

GOLDEN = Path(__file__).parent / "data" / "ddpg_td3_training.npz"
RECORD = os.environ.get("RECORD_DDPG_GOLDEN") == "1"
EXACT = os.environ.get("DDPG_GOLDEN_EXACT") == "1"
RTOL, ATOL = 1e-12, 1e-14

OBS_DIM, ACT_DIM, EPISODE, EPISODES = 3, 2, 10, 5
BASE = {
    "seed": 13,
    "hidden_sizes": [6, 5],
    "learning_rate": 0.01,
    "gamma": 0.9,
    "tau": 0.05,
    "batch_size": 8,
    "buffer_capacity": 100,
    "learning_starts": 15,  # the second episode ends warm-up part-way through
    "noise_std": 0.3,
    # Ten transitions, capped at five steps: an odd count, so TD3's every-second actor step falls
    # on the first step of one call and the second of the next.
    "max_updates_per_call": 5,
    "action_low": [-2.0, -1.0],
    "action_high": [1.0, 3.0],
}
# Clips every critic step here; the actor's norm sits near 0.05, so some actor steps clip and some
# do not.
CLIP = {"max_grad_norm": 0.05}
CASES = {
    "ddpg-gaussian": (DDPGAgent, {}),
    "ddpg-ou": (DDPGAgent, {"noise_type": "ornstein_uhlenbeck"}),
    "ddpg-clipped": (DDPGAgent, CLIP),
    "td3": (TD3Agent, {}),
    "td3-clipped": (TD3Agent, CLIP),
}


def _outputs(name: str, tmp_path) -> dict[str, np.ndarray]:
    agent_cls, overrides = CASES[name]
    agent = agent_cls(OBS_DIM, ACT_DIM, {**BASE, **overrides})
    rng = np.random.default_rng(3)
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
        agent.end_episode()
        out[f"actions_{episode}"] = actions
        out[f"metric_names/{episode}"] = np.array(list(metrics), dtype=str)
        out.update({f"metric_{episode}/{k}": np.asarray(v) for k, v in metrics.items()})
    out["greedy_action"] = agent.select_action(np.array([0.4, -0.3, 0.2]), training=False)
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
        pytest.fail(f"Recorded {prefix}; rerun without RECORD_DDPG_GOLDEN.")
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


def test_the_recorded_cases_reach_warm_up_training_and_both_td3_step_kinds():
    """Without these the file could not notice a broken warm-up, a skipped actor step or a policy
    delay counted per call instead of per gradient step."""
    with np.load(GOLDEN) as golden:
        assert {key.split("/")[0] for key in golden.files} == set(CASES)
        assert golden["ddpg-gaussian/metric_names/0"].size == 0, "episode 0 should only warm up"
        assert golden["ddpg-gaussian/metric_names/1"].size > 0, "episode 1 should train"
        # Five steps per call from gradient step 1: actor steps 2 and 4, then 6, 8 and 10.
        actor_updates = [float(golden[f"td3/metric_{e}/actor_updates"]) for e in (1, 2, 3)]
    assert actor_updates == [2.0, 3.0, 2.0]

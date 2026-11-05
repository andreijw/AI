"""DDPG's, TD3's and SAC's seeded training, DDPG's and TD3's as recorded before the PyTorch backend read its layout from hooks
and before the two moved onto a shared off-policy base.

The other DDPG and TD3 tests replay an update through the agent's own loss methods, so a reordered
step or a renamed Adam key would agree with itself there. This file compares against arrays
recorded from the code before the networks, optimizers and target pairs became declared hooks: for
Gaussian and Ornstein-Uhlenbeck exploration, with and without gradient clipping, and for TD3 with
a ``policy_delay`` boundary falling across two ``update`` calls. The cases recorded before
``OffPolicyAgent`` (rl_core.agents.off_policy) took over the shared code add mid-episode updates
(``update_every``, a running ``update_transitions`` count that a call can owe no step on), a save
that a differently seeded agent loads and trains on from, and DDPG resuming from a TD3 checkpoint.
Each case plays seeded episodes through ``select_action``, ``update`` and ``end_episode``, then
records every metric, every checkpoint array (weights, Adam states, counters, the replay buffer,
every generator) with the checkpoint's keys, and a greedy action. The SAC cases, in their own file,
were recorded from the first NumPy SAC (2026-11-04), so the PyTorch SAC and any later sharing of
its code must reproduce it: with a learned temperature, with a fixed one and gradient clipping, and
updating mid-episode across a save and load.

Names and every dtype must match exactly; the checkpoint's keys are compared as a set, since
loading goes by name. Numbers agree to a relative 1e-12, as in
tests/test_gaussian_gradients_golden.py: the file was recorded on one machine (aarch64, OpenBLAS)
and other BLAS builds may differ in the last bits. On the recording machine, DDPG_GOLDEN_EXACT=1
demands every array bit for bit; the pull request records that run.

A deliberate change to the training regenerates the file: run this module (``-k`` picks the cases)
with RECORD_DDPG_GOLDEN=1, check why every changed array changed, then run it again without the
variable (a recording run fails on purpose, so it can never pass by accepting any output).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rl_core.agents import DDPGAgent, SACAgent, TD3Agent
from rl_core.agents.off_policy import OffPolicyAgent

GOLDEN = Path(__file__).parent / "data" / "ddpg_td3_training.npz"
SAC_GOLDEN = Path(__file__).parent / "data" / "sac_training.npz"
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


@dataclass(frozen=True)
class Case:
    """One recorded run: the agent, its config overrides, and how the episodes reach it."""

    agent: type[OffPolicyAgent]
    overrides: dict[str, Any] = field(default_factory=dict)
    # Updating mid-episode: each episode reaches ``update`` in calls of this many transitions.
    update_every: int | None = None
    # Save after RESUME_AFTER episodes, load into a fresh agent of this class seeded otherwise (so
    # anything the load skips shows), and play the remaining episodes on it.
    resume_as: type[OffPolicyAgent] | None = None

    @property
    def golden(self) -> Path:
        """The file the case is recorded in."""
        return SAC_GOLDEN if self.agent is SACAgent else GOLDEN


RESUME_AFTER, RESUME_SEED = 3, 99
# A quarter step per transition, in calls of three, three, three and one: once training starts the
# calls owe 0, 1, 1, 0, 1, 1, 0, 1 ... steps, so some calls past warm-up train not at all.
MID_EPISODE = {"updates_per_step": 0.25}
CASES = {
    "ddpg-gaussian": Case(DDPGAgent),
    "ddpg-ou": Case(DDPGAgent, {"noise_type": "ornstein_uhlenbeck"}),
    "ddpg-clipped": Case(DDPGAgent, CLIP),
    "td3": Case(TD3Agent),
    "td3-clipped": Case(TD3Agent, CLIP),
    "ddpg-mid-episode": Case(DDPGAgent, MID_EPISODE, update_every=3),
    "td3-mid-episode": Case(TD3Agent, MID_EPISODE, update_every=3),
    "ddpg-resume": Case(DDPGAgent, {"noise_type": "ornstein_uhlenbeck"}, resume_as=DDPGAgent),
    "td3-mid-episode-resume": Case(TD3Agent, MID_EPISODE, update_every=3, resume_as=TD3Agent),
    "ddpg-from-td3": Case(TD3Agent, CLIP, resume_as=DDPGAgent),
    "sac": Case(SACAgent),
    "sac-fixed-clipped": Case(SACAgent, {"ent_coef": 0.2, **CLIP}),
    "sac-mid-episode": Case(SACAgent, MID_EPISODE, update_every=3),
    "sac-mid-episode-resume": Case(SACAgent, MID_EPISODE, update_every=3, resume_as=SACAgent),
}


def _play(
    agent: OffPolicyAgent,
    case: Case,
    rng: np.random.Generator,
    episodes: range,
    out: dict[str, np.ndarray],
) -> None:
    """Play ``episodes`` through the agent, recording its actions and every call's metrics."""
    for episode in episodes:
        obs = rng.standard_normal((EPISODE + 1, OBS_DIM))
        actions = np.array([agent.select_action(o, training=True) for o in obs[:-1]])
        terminated = np.zeros(EPISODE, dtype=bool)
        terminated[-1] = episode % 2 == 1
        rewards = rng.normal(size=EPISODE)
        out[f"actions_{episode}"] = actions
        step = case.update_every or EPISODE
        for call, start in enumerate(range(0, EPISODE, step)):
            cut = slice(start, start + step)
            metrics = agent.update(
                {
                    "observations": obs[:-1][cut],
                    "actions": actions[cut],
                    "rewards": rewards[cut],
                    "next_observations": obs[1:][cut],
                    "terminated": terminated[cut],
                }
            )
            label = str(episode) if case.update_every is None else f"{episode}.{call}"
            out[f"metric_names/{label}"] = np.array(list(metrics), dtype=str)
            out.update({f"metric_{label}/{k}": np.asarray(v) for k, v in metrics.items()})
        agent.end_episode()


def _build(agent_cls: type[OffPolicyAgent], case: Case, **config: Any) -> OffPolicyAgent:
    agent = agent_cls(OBS_DIM, ACT_DIM, {**BASE, **case.overrides, **config})
    if case.update_every is not None:
        agent.enable_mid_episode_updates(case.update_every)
    return agent


def _outputs(name: str, tmp_path) -> dict[str, np.ndarray]:
    case = CASES[name]
    agent = _build(case.agent, case)
    rng = np.random.default_rng(3)
    out: dict[str, np.ndarray] = {}
    if case.resume_as is None:
        _play(agent, case, rng, range(EPISODES), out)
    else:
        _play(agent, case, rng, range(RESUME_AFTER), out)
        agent.save(str(tmp_path / "before.npz"))
        agent = _build(case.resume_as, case, seed=RESUME_SEED)
        agent.load(str(tmp_path / "before.npz"))
        _play(agent, case, rng, range(RESUME_AFTER, EPISODES), out)
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
    prefix, path = name + "/", CASES[name].golden
    if RECORD:
        recorded = dict(np.load(path)) if path.exists() else {}
        recorded.update({prefix + k: v for k, v in outputs.items()})
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **recorded)
        pytest.fail(f"Recorded {prefix}; rerun without RECORD_DDPG_GOLDEN.")
    with np.load(path) as golden:
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
    for path in (GOLDEN, SAC_GOLDEN):
        with np.load(path) as golden:
            recorded = {key.split("/")[0] for key in golden.files}
        assert recorded == {name for name, case in CASES.items() if case.golden == path}
    with np.load(GOLDEN) as golden:
        assert golden["ddpg-gaussian/metric_names/0"].size == 0, "episode 0 should only warm up"
        assert golden["ddpg-gaussian/metric_names/1"].size > 0, "episode 1 should train"
        # Five steps per call from gradient step 1: actor steps 2 and 4, then 6, 8 and 10.
        actor_updates = [float(golden[f"td3/metric_{e}/actor_updates"]) for e in (1, 2, 3)]
    assert actor_updates == [2.0, 3.0, 2.0]


def test_the_new_cases_skip_calls_carry_the_delay_and_resume_seamlessly():
    """Without these the mid-episode cases could pass with a count that never owes zero steps or a
    policy delay reset per call, and a resumed run could differ from an uninterrupted one."""
    with np.load(GOLDEN) as golden:
        # A quarter step per transition: calls past warm-up owe no step and return no metrics.
        assert golden["ddpg-mid-episode/metric_names/1.0"].size == 0, "1.0 owes no step"
        assert golden["ddpg-mid-episode/metric_names/1.1"].size > 0, "1.1 should train"
        assert int(golden["ddpg-mid-episode/checkpoint_update_transitions"]) == 40
        # One step a call, so TD3's actor steps (every second) alternate across calls.
        actor_updates = [
            float(golden[f"td3-mid-episode/metric_{c}/actor_updates"])
            for c in ("1.1", "1.2", "2.0", "2.1")
        ]
        assert actor_updates == [0.0, 1.0, 0.0, 1.0]
        # A save loaded by an agent of another seed trains on exactly as if never interrupted.
        for whole, resumed in (
            ("ddpg-ou", "ddpg-resume"),
            ("td3-mid-episode", "td3-mid-episode-resume"),
        ):
            keys = sorted(k.split("/", 1)[1] for k in golden.files if k.startswith(whole + "/"))
            for key in keys:
                np.testing.assert_array_equal(
                    golden[f"{resumed}/{key}"],
                    golden[f"{whole}/{key}"],
                    err_msg=f"{resumed}/{key} should equal the uninterrupted {whole}/{key}",
                )
        # DDPG reads its own networks from a TD3 file and saves no second critic.
        assert not any("critic_2" in k for k in golden["ddpg-from-td3/checkpoint_keys"])


def test_the_sac_cases_train_learn_or_hold_alpha_and_resume_seamlessly():
    """Without these the SAC file could pin a run that never trains, a temperature that never
    moves (or a fixed one that does), or a resume that differs from the uninterrupted run."""
    with np.load(SAC_GOLDEN) as golden:
        assert golden["sac/metric_names/0"].size == 0, "episode 0 should only warm up"
        assert "ent_coef_loss" in golden["sac/metric_names/1"], "episode 1 trains alpha"
        learned = [float(golden[f"sac/metric_{e}/ent_coef"]) for e in (1, 4)]
        assert learned[1] < learned[0], "alpha should fall from 1 on these episodes"
        fixed = [float(golden[f"sac-fixed-clipped/metric_{e}/ent_coef"]) for e in (1, 4)]
        np.testing.assert_allclose(fixed, [0.2, 0.2], rtol=1e-12)
        assert "ent_coef_loss" not in golden["sac-fixed-clipped/metric_names/4"]
        whole, resumed = "sac-mid-episode", "sac-mid-episode-resume"
        keys = sorted(k.split("/", 1)[1] for k in golden.files if k.startswith(whole + "/"))
        for key in keys:
            np.testing.assert_array_equal(
                golden[f"{resumed}/{key}"],
                golden[f"{whole}/{key}"],
                err_msg=f"{resumed}/{key} should equal the uninterrupted {whole}/{key}",
            )

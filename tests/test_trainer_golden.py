"""Seeded training through ``Trainer.train()``, as recorded on 2026-11-09 from develop (``c85e3d7``).

Every other golden file drives an agent or a wrapper directly, so a change to the trainer's loop (the
order of its draws, when it updates, cuts, bootstraps, evaluates, saves or advances a curriculum) would
agree with itself everywhere else. This file records whole training runs from the code before the
parallel-environments work (docs/roadmap.md, Training) and holds every later trainer to them: with
``training.num_envs`` absent or 1, training must stay what it was.

Each case trains a real seeded wrapper for a few short episodes with a tiny network, and records the
episode histories, every metric record the trainer logs (update means, evaluations, curriculum
fields), the checkpoint files it wrote, the arrays of its last agent checkpoint and trainer file, and
the training environment's generators at the end. Together the cases cover REINFORCE, both Actor-Critics
(Monte Carlo, chunked, and with ``update_every``), both PPOs, observation normalisation, a time-limit
truncation and a ``max_steps_per_episode`` cut, evaluation with and without an evaluation environment,
``save_frequency`` with ``save_final``, a step budget crossed mid-episode, both curriculum modes that
advance on their own (linear and reward_threshold), an environment that reports ``is_success``, and DQN
with and without ``update_every`` (the loop is shared).

Arms: each case runs with no ``num_envs`` key and with ``num_envs: 1``; both must match the one
recording. Before the key existed the trainer ignored it, so the second arm only pins that it still
means "today's training". The parallel-environments work adds a third arm: the lockstep collector
forced at one environment (``tests/trainer_internals.py``).

Numbers agree to a relative 1e-12, as in tests/test_dqn_golden.py: the file was recorded on one machine
(aarch64, OpenBLAS) and CI checks it on x86, where matrix products can round differently in the last
bit. Cases are kept tiny so such a difference cannot grow into a different sampled action. Names,
counts, dtypes and generator states must match exactly; on the recording machine,
TRAINER_GOLDEN_EXACT=1 demands every array bit for bit. The agent checkpoint's hash in the trainer file
is not recorded (it hashes the float bytes), and the trainer file's metric window, which it stores as
JSON text, is compared as numbers.

A deliberate change to training regenerates the file: run this module with RECORD_TRAINER_GOLDEN=1,
check why every changed array changed, then run it again without the variable (a recording run fails on
purpose, so it can never pass by accepting any output).
"""

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rl_cartpole.environments.cartpole_env import CartPoleEnv
from rl_core.agents import (
    ActorCriticAgent,
    ContinuousActorCriticAgent,
    ContinuousPPOAgent,
    DQNAgent,
    PPOAgent,
    ReinforceAgent,
)
from rl_core.agents.base_agent import BaseAgent
from rl_core.training import Trainer
from rl_core.training.curriculum import curriculum_from_config
from rl_navigation.environments.navigation_env import NavigationEnv
from rl_pendulum.environments.pendulum_env import PendulumEnv

GOLDEN = Path(__file__).parent / "data" / "trainer_training.npz"
RECORD = os.environ.get("RECORD_TRAINER_GOLDEN") == "1"
EXACT = os.environ.get("TRAINER_GOLDEN_EXACT") == "1"
RTOL, ATOL = 1e-12, 1e-14
EVAL_SEED = 5000
PENDULUM_STEPS = 24  # 3 x 8: the update_every case's chunks line up with its episodes
TORQUE = {"action_low": -2.0, "action_high": 2.0}

# The training-config arms every case runs under. "num_envs: 1" must mean today's training.
ARMS: dict[str, dict[str, Any]] = {"no-key": {}, "num_envs-1": {"num_envs": 1}}


@dataclass(frozen=True)
class Case:
    """One recorded run: its environments, its agent and its training section."""

    env: Callable[[], Any]
    agent: Callable[[Any], BaseAgent]
    training: dict[str, Any]
    eval_env: Callable[[], Any] | None = None
    curriculum: dict[str, Any] | None = None
    save_final: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _dims(env: Any) -> tuple[int, int]:
    space = env.action_space
    action_dim = int(space.n) if hasattr(space, "n") else int(space.shape[0])
    return int(env.observation_space.shape[0]), action_dim


def _cartpole(**settings: Any) -> Callable[[], CartPoleEnv]:
    return lambda: CartPoleEnv(**{"seed": 3, **settings})


def _pendulum(**settings: Any) -> Callable[[], PendulumEnv]:
    return lambda: PendulumEnv(**{"seed": 4, "max_episode_steps": PENDULUM_STEPS, **settings})


def _agent(cls: type[BaseAgent], **config: Any) -> Callable[[Any], BaseAgent]:
    return lambda env: cls(*_dims(env), {"seed": 11, "hidden_dim": 8, **config})


PPO = {
    "rollout_steps": 24,
    "minibatch_size": 8,
    "ppo_epochs": 2,
    "optimizer": "adam",
    "learning_rate": 0.01,
    "anneal_steps": 120,
    "max_grad_norm": 0.5,
}
CONTINUOUS_AC = {
    "n_steps": 8,
    "gae_lambda": 0.9,
    "learning_rate": 0.01,
    "bounded_mean": True,
    "normalize_observations": True,
    "anneal_steps": 200,
    **TORQUE,
}
DQN = {
    "learning_rate": 0.05,
    "gamma": 0.9,
    "batch_size": 8,
    "buffer_capacity": 40,
    "min_buffer_size": 16,
    "target_update_frequency": 3,
}
NOISE_CURRICULUM_START = {"obs_noise_std": 0.0}

CASES: dict[str, Case] = {
    # Evaluation on the training environment, which continues its start-state sequence.
    "reinforce-eval-on-train-env": Case(
        _cartpole(),
        _agent(ReinforceAgent, learning_rate=0.02),
        {"num_episodes": 10, "eval_frequency": 5, "save_frequency": 100},
    ),
    # The trainer cuts at 12 steps, and saves at 4 and 8 and (save_final) at 10.
    "actor-critic-cut-save-final": Case(
        _cartpole(),
        _agent(ActorCriticAgent, learning_rate=0.02, entropy_coef=0.05),
        {
            "num_episodes": 10,
            "max_steps_per_episode": 12,
            "eval_frequency": 100,
            "save_frequency": 4,
        },
        save_final=True,
    ),
    # Time limit at 15: the cut episodes bootstrap; observation normalisation; an evaluation
    # environment with fixed start states, evaluated twice.
    "ppo-normalised-truncated-eval-env": Case(
        _cartpole(max_episode_steps=15),
        _agent(PPOAgent, normalize_observations=True, **PPO),
        {"num_episodes": 10, "eval_frequency": 4, "save_frequency": 100},
        eval_env=_cartpole(max_episode_steps=15, seed=9),
    ),
    "continuous-ac-chunks": Case(
        _pendulum(),
        _agent(ContinuousActorCriticAgent, **CONTINUOUS_AC),
        {
            "num_episodes": 10,
            "max_steps_per_episode": PENDULUM_STEPS,
            "eval_frequency": 100,
            "save_frequency": 100,
        },
    ),
    "continuous-ac-monte-carlo": Case(
        _pendulum(),
        _agent(ContinuousActorCriticAgent, learning_rate=0.01, **TORQUE),
        {
            "num_episodes": 10,
            "max_steps_per_episode": PENDULUM_STEPS,
            "eval_frequency": 100,
            "save_frequency": 100,
        },
    ),
    # True A2C: each 8-step chunk handed over as it is collected.
    "continuous-ac-update-every": Case(
        _pendulum(),
        _agent(ContinuousActorCriticAgent, **CONTINUOUS_AC),
        {
            "num_episodes": 10,
            "max_steps_per_episode": PENDULUM_STEPS,
            "update_every": 8,
            "eval_frequency": 100,
            "save_frequency": 100,
        },
    ),
    "continuous-ppo-truncated": Case(
        _pendulum(),
        _agent(
            ContinuousPPOAgent,
            rollout_steps=32,
            batch_size=16,
            ppo_epochs=2,
            learning_rate=0.01,
            gae_lambda=0.9,
            **TORQUE,
        ),
        {
            "num_episodes": 10,
            "max_steps_per_episode": PENDULUM_STEPS,
            "eval_frequency": 100,
            "save_frequency": 100,
        },
    ),
    # Agent seed 17: every PPO ratio stays at least 0.04 from the clip edges, so a last-bit
    # difference on another machine cannot flip a clipped step (seed 11 failed on CI's x86).
    "ppo-curriculum-linear": Case(
        _cartpole(obs_noise_std=0.05),
        _agent(PPOAgent, **{**PPO, "seed": 17}),
        {"num_episodes": 10, "eval_frequency": 6, "save_frequency": 100},
        eval_env=_cartpole(obs_noise_std=0.05, seed=9),
        curriculum={"mode": "linear", "start": NOISE_CURRICULUM_START, "episodes": 4},
    ),
    "reinforce-curriculum-reward-threshold": Case(
        _cartpole(obs_noise_std=0.05),
        _agent(ReinforceAgent, learning_rate=0.02),
        {"num_episodes": 20, "eval_frequency": 100, "save_frequency": 100},
        eval_env=_cartpole(obs_noise_std=0.05, seed=9),
        curriculum={
            "mode": "reward_threshold",
            "start": NOISE_CURRICULUM_START,
            "stages": 4,
            "threshold": 20.0,
            "window": 2,
            "min_episodes_per_stage": 2,
        },
    ),
    # 50 steps, crossed partway through an episode, which then finishes.
    "actor-critic-step-budget": Case(
        _cartpole(),
        _agent(ActorCriticAgent, learning_rate=0.02),
        {"total_steps": 50, "eval_frequency": 100, "save_frequency": 100},
    ),
    "ppo-navigation-success": Case(
        lambda: NavigationEnv(seed=6, max_episode_steps=20),
        _agent(PPOAgent, **PPO),
        {"num_episodes": 10, "eval_frequency": 100, "save_frequency": 100},
    ),
    "dqn": Case(
        _cartpole(),
        _agent(DQNAgent, **DQN),
        {"num_episodes": 10, "eval_frequency": 100, "save_frequency": 100},
    ),
    "dqn-update-every": Case(
        _cartpole(),
        _agent(DQNAgent, updates_per_step=0.5, exploration_steps=30, **DQN),
        {"num_episodes": 10, "update_every": 4, "eval_frequency": 100, "save_frequency": 100},
    ),
}


class _Capture:
    """A custom logger: keeps every metric record the trainer logs, and drops its text lines."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any]) -> None:
        self.records.append(metrics)

    def info(self, message: str) -> None:
        return None

    def warning(self, message: str) -> None:
        return None


def _flatten(prefix: str, value: Any, out: dict[str, np.ndarray]) -> None:
    """``value``'s leaves as arrays under ``prefix/...``: numbers as float64 or int64, names as str."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}/{key}", item, out)
    elif isinstance(value, list | tuple) and all(isinstance(item, str) for item in value):
        out[prefix] = np.array(value, dtype=str)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _flatten(f"{prefix}/{index}", item, out)
    elif value is None:
        out[prefix] = np.array("None")
    elif isinstance(value, bool | np.bool_):
        out[prefix] = np.array(bool(value))
    elif isinstance(value, int | np.integer):
        out[prefix] = np.array(int(value), dtype=np.int64)
    else:
        out[prefix] = np.array(float(value), dtype=np.float64)


def _trainer_file(path: Path, out: dict[str, np.ndarray]) -> None:
    with np.load(path) as data:
        out["trainer_keys"] = np.array(sorted(data.files))
        for key in data.files:
            if key == "agent_sha256":  # a hash of the agent file's float bytes: machine-dependent
                continue
            if key == "update_metric_window":  # JSON text of floats: compared as numbers
                _flatten("trainer/update_metric_window", json.loads(str(data[key])), out)
                continue
            out[f"trainer/{key}"] = data[key]


def _outputs(name: str, arm: str, tmp_path: Path) -> dict[str, np.ndarray]:
    case = CASES[name]
    directory = tmp_path / "checkpoints"
    training = {**case.training, "checkpoint_dir": str(directory), **ARMS[arm]}
    env = case.env()
    eval_env = None if case.eval_env is None else case.eval_env()
    logger = _Capture()
    try:
        curriculum = (
            None
            if case.curriculum is None
            else curriculum_from_config({"curriculum": case.curriculum}, env)
        )
        trainer = Trainer(
            env=env,
            agent=case.agent(env),
            config=training,
            logger=logger,
            eval_env=eval_env,
            eval_seed=None if eval_env is None else EVAL_SEED,
            curriculum=curriculum,
            save_final=case.save_final,
        )
        stats = trainer.train()
        env_state_end = env.rng_state()
        final_agent = tmp_path / "final_agent.pt"
        trainer.agent.save(str(final_agent))
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()

    out: dict[str, np.ndarray] = {
        "episode_rewards": np.array(trainer.episode_rewards, dtype=np.float64),
        "episode_lengths": np.array(trainer.episode_lengths, dtype=np.int64),
        # -1 for an episode whose success is unknown.
        "episode_successes": np.array(
            [-1 if s is None else int(s) for s in trainer.episode_successes], dtype=np.int64
        ),
    }
    _flatten("stats", stats, out)
    out["record_count"] = np.array(len(logger.records), dtype=np.int64)
    for index, record in enumerate(logger.records):
        _flatten(f"record/{index}", record, out)
    out.update({f"env_state_end/{k}": v for k, v in env_state_end.items()})
    # The agent as training left it, whether or not the run saved a checkpoint.
    with np.load(final_agent) as data:
        out["final_agent_keys"] = np.array(sorted(data.files))
        out.update({f"final_agent/{k}": data[k] for k in data.files})

    files = sorted(p.name for p in directory.iterdir()) if directory.exists() else []
    out["checkpoint_files"] = np.array(files, dtype=str)
    agents = sorted(directory.glob("agent_episode_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if agents:
        last = agents[-1]
        with np.load(last) as data:
            out["agent_keys"] = np.array(sorted(data.files))
            out.update({f"agent/{k}": data[k] for k in data.files})
        _trainer_file(
            last.with_name(last.name.replace("agent_", "trainer_")).with_suffix(".npz"), out
        )
    return out


def _recorded(name: str) -> dict[str, np.ndarray]:
    prefix = name + "/"
    with np.load(GOLDEN) as golden:
        return {k[len(prefix) :]: golden[k] for k in golden.files if k.startswith(prefix)}


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("name", list(CASES))
def test_training_matches_the_recorded_arrays(name, arm, tmp_path):
    outputs = _outputs(name, arm, tmp_path)
    if RECORD:
        if arm != next(iter(ARMS)):
            pytest.skip("the file is recorded from the first arm only")
        recorded = dict(np.load(GOLDEN)) if GOLDEN.exists() else {}
        recorded = {k: v for k, v in recorded.items() if not k.startswith(name + "/")}
        recorded.update({f"{name}/{k}": v for k, v in outputs.items()})
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN, **recorded)
        pytest.fail(f"Recorded {name}; rerun without RECORD_TRAINER_GOLDEN.")
    expected = _recorded(name)
    missing, extra = sorted(set(expected) - set(outputs)), sorted(set(outputs) - set(expected))
    assert not missing and not extra, (
        f"{name} ({arm}) should record the same arrays as the file; missing {missing[:5]}, "
        f"new {extra[:5]}"
    )
    # Every differing array is reported, not just the first, so a failure on another machine shows
    # where the runs part (an update metric only, or the weights and episodes after it).
    failures = []
    for key, value in outputs.items():
        want = expected[key]
        try:
            assert value.dtype == want.dtype, f"{name}/{key} ({arm}) changed dtype"
            if EXACT or value.dtype.kind not in "fc":  # names, counts and generator states: exact
                np.testing.assert_array_equal(value, want, err_msg=f"{name}/{key} ({arm}) differs")
            else:
                np.testing.assert_allclose(
                    value, want, rtol=RTOL, atol=ATOL, err_msg=f"{name}/{key} ({arm}) differs"
                )
        except AssertionError as error:
            failures.append(str(error).strip())
    assert not failures, f"{len(failures)} array(s) differ:\n\n" + "\n\n".join(failures)


def _record_keys(recorded: dict[str, np.ndarray], suffix: str) -> list[str]:
    return [k for k in recorded if k.startswith("record/") and k.endswith(suffix)]


def test_the_recorded_cases_exercise_what_they_are_for():
    """Without these the file could pass while no case cut, bootstrapped, evaluated, saved, crossed
    a budget, advanced a curriculum or reported success."""
    with np.load(GOLDEN) as golden:
        assert {key.split("/")[0] for key in golden.files} == set(CASES), (
            "the file should hold exactly the cases this module defines"
        )
    cut = _recorded("actor-critic-cut-save-final")
    assert 12 in cut["episode_lengths"], "the trainer's 12-step cut should end some episode"
    assert list(cut["checkpoint_files"]) == [
        "agent_episode_10.pt",
        "agent_episode_4.pt",
        "agent_episode_8.pt",
        "trainer_episode_10.npz",
        "trainer_episode_4.npz",
        "trainer_episode_8.npz",
    ], "save_frequency 4 and save_final should write episodes 4, 8 and 10"

    truncated = _recorded("ppo-normalised-truncated-eval-env")
    assert 15 in truncated["episode_lengths"], "the 15-step time limit should truncate an episode"
    assert len(_record_keys(truncated, "/evaluation/mean_reward")) == 2, "two evaluations"
    assert int(truncated["final_agent/obs_norm_count"]) > sum(truncated["episode_lengths"]), (
        "the PPO case should normalise every training observation, start states included"
    )
    assert (
        len(_record_keys(_recorded("reinforce-eval-on-train-env"), "/evaluation/mean_reward")) == 2
    )
    for name in CASES:
        if name != "actor-critic-step-budget":
            assert _record_keys(
                _recorded(name),
                "/update/"
                + ("loss" if "dqn" in name or "continuous-ac" in name else "policy_loss"),
            ), f"{name} should log update metrics"

    pendulum = _recorded("continuous-ppo-truncated")
    assert set(pendulum["episode_lengths"]) == {PENDULUM_STEPS}, "Pendulum is cut by its limit"

    budget = _recorded("actor-critic-step-budget")
    lengths = budget["episode_lengths"]
    assert lengths[:-1].sum() < 50 <= lengths.sum(), "the last episode should cross 50 steps"

    threshold = _recorded("reinforce-curriculum-reward-threshold")
    stages = [threshold[k] for k in _record_keys(threshold, "/curriculum_stage")]
    assert stages and max(stages) >= 1, "the reward_threshold curriculum should advance a stage"

    success = _recorded("ppo-navigation-success")
    assert (success["episode_successes"] >= 0).all(), "navigation should report every success"

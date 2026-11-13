#!/usr/bin/env python3
"""Write the reference checkpoints the tests load: one small one per agent, after a few seeded updates.

``tests/test_reference_checkpoints.py`` loads each file and checks its entries (key, dtype, shape
and a digest of the values) against ``tests/data/checkpoints/manifest.json``, and that loading it
into a fresh agent and saving again gives the same entries. A refactor of the checkpoint code must
keep both. The files are not compared byte for byte: ``np.savez_compressed`` stamps zip times.

Run it from the repository root, then rerun the tests; rewrite the files only when the checkpoint
format changes on purpose, and say so in the pull request:

    python scripts/write_reference_checkpoints.py

The PyTorch backends (``torch_*``) are written only where PyTorch is installed, with the gpu
profile's environment and the packages of this tree on the path::

    PYTHONPATH=packages/rl-core/src .venv-gpu/bin/python scripts/write_reference_checkpoints.py

Without PyTorch, or with ``--numpy-only``, their manifest entries are kept as they are.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from rl_core.environments import OBSERVATION_DEFAULTS

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "tests" / "data" / "checkpoints"
MANIFEST = "manifest.json"
SEED = 3
STEPS = 12  # transitions per update
UPDATES = 3

_DISCRETE = {"observation_dim": 4, "action_dim": 3, "actions": "discrete"}
_CONTINUOUS = {"observation_dim": 3, "action_dim": 2, "actions": "continuous"}
_POLICY = {"hidden_dim": 8, "learning_rate": 0.01, "optimizer": "adam", "seed": SEED}
_OFF_POLICY = {
    "hidden_sizes": [8, 6],
    "batch_size": 4,
    "buffer_capacity": 64,
    "learning_starts": 8,
    "seed": SEED,
}
_TORCH = {"backend": "torch", "device": "cpu", "dtype": "float64"}

# name: (agent class as "module:Class", dimensions and action kind, agent config)
CASES: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "random": ("rl_core.agents:RandomAgent", _DISCRETE, {"seed": SEED}),
    "continuous_random": ("rl_core.agents:ContinuousRandomAgent", _CONTINUOUS, {"seed": SEED}),
    "reinforce": ("rl_core.agents:ReinforceAgent", _DISCRETE, _POLICY),
    "actor_critic": ("rl_core.agents:ActorCriticAgent", _DISCRETE, _POLICY),
    "ppo": (
        "rl_core.agents:PPOAgent",
        _DISCRETE,
        {**_POLICY, "minibatch_size": 4, "normalize_observations": True},
    ),
    "dqn": (
        "rl_core.agents:DQNAgent",
        _DISCRETE,
        {**_POLICY, "batch_size": 4, "min_buffer_size": 8, "buffer_capacity": 64},
    ),
    "dqn_dueling": (
        "rl_core.agents:DQNAgent",
        _DISCRETE,
        {
            **_POLICY,
            "batch_size": 4,
            "min_buffer_size": 8,
            "buffer_capacity": 64,
            "dueling": True,
            "double_q": True,
        },
    ),
    "continuous_actor_critic": (
        "rl_core.agents:ContinuousActorCriticAgent",
        _CONTINUOUS,
        {"hidden_dim": 8, "seed": SEED},
    ),
    "continuous_ppo": (
        "rl_core.agents:ContinuousPPOAgent",
        _CONTINUOUS,
        {"hidden_dim": 8, "batch_size": 4, "seed": SEED},
    ),
    "ddpg": ("rl_core.agents:DDPGAgent", _CONTINUOUS, _OFF_POLICY),
    "td3": ("rl_core.agents:TD3Agent", _CONTINUOUS, _OFF_POLICY),
    "sac": ("rl_core.agents:SACAgent", _CONTINUOUS, _OFF_POLICY),
    "torch_ddpg": ("rl_core.agents:DDPGAgent", _CONTINUOUS, {**_OFF_POLICY, **_TORCH}),
    "torch_td3": ("rl_core.agents:TD3Agent", _CONTINUOUS, {**_OFF_POLICY, **_TORCH}),
    "torch_sac": ("rl_core.agents:SACAgent", _CONTINUOUS, {**_OFF_POLICY, **_TORCH}),
}


def needs_torch(name: str) -> bool:
    """True for a case built with the PyTorch backend."""
    return CASES[name][2].get("backend") == "torch"


def build(name: str) -> Any:
    """A fresh agent for case ``name``, with the default observation settings recorded."""
    target, dims, config = CASES[name]
    module, _, class_name = target.partition(":")
    base = getattr(importlib.import_module(module), class_name)
    agent = base.class_for_config(config)(dims["observation_dim"], dims["action_dim"], config)
    agent.record_observation_settings(OBSERVATION_DEFAULTS, OBSERVATION_DEFAULTS)
    return agent


def _batch(rng: np.random.Generator, dims: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """One seeded episode of ``STEPS`` transitions that ends in a terminal state."""
    obs = rng.standard_normal((STEPS + 1, dims["observation_dim"]))
    actions: np.ndarray
    if dims["actions"] == "discrete":
        actions = rng.integers(0, dims["action_dim"], size=STEPS)
    else:
        actions = rng.uniform(-1.0, 1.0, size=(STEPS, dims["action_dim"]))
    ends = np.zeros(STEPS, dtype=bool)
    ends[-1] = True
    return {
        "observations": obs[:-1],
        "actions": actions,
        "rewards": rng.standard_normal(STEPS),
        "next_observations": obs[1:],
        "dones": ends,
        "terminated": ends.copy(),
    }


def trained(name: str) -> Any:
    """Case ``name``'s agent after ``UPDATES`` seeded updates and some acting, so its optimizer
    state, counters, buffers and generators have all moved from their start."""
    agent = build(name)
    dims = CASES[name][1]
    rng = np.random.default_rng(SEED)
    for _ in range(UPDATES):
        batch = _batch(rng, dims)
        for observation in batch["observations"][:3]:
            agent.select_action(observation, training=True)
        agent.update(batch)
        agent.end_episode()
    return agent


def describe(array: np.ndarray) -> dict[str, Any]:
    """An entry's dtype, shape and a SHA-256 digest of its values (C order)."""
    values = np.ascontiguousarray(array)
    return {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def describe_file(path: Path) -> dict[str, dict[str, Any]]:
    """:func:`describe` for every entry of the checkpoint at ``path``, by key, in sorted order."""
    with np.load(path) as data:
        return {key: describe(data[key]) for key in sorted(data.files)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--numpy-only",
        action="store_true",
        help="skip the PyTorch cases even where it is installed",
    )
    args = parser.parse_args(argv)
    has_torch = not args.numpy_only and importlib.util.find_spec("torch") is not None
    manifest_path = args.output_dir / MANIFEST
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest: dict[str, Any] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (target, dims, config) in CASES.items():
        if needs_torch(name) and not has_torch:
            if name in old:
                manifest[name] = old[name]
            print(f"{name}: skipped, PyTorch not used (manifest entry kept)")
            continue
        path = args.output_dir / f"{name}.npz"
        trained(name).save(str(path))
        manifest[name] = {
            "agent": target,
            **{key: dims[key] for key in ("observation_dim", "action_dim")},
            "config": config,
            "entries": describe_file(path),
        }
        print(f"{name}: {path.stat().st_size:,} bytes, {len(manifest[name]['entries'])} entries")
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    total = sum(p.stat().st_size for p in args.output_dir.iterdir())
    print(f"{args.output_dir}: {total:,} bytes in all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

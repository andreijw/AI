"""Shared helpers for tests that run the train.py / record.py scripts."""

import importlib
import sys


def run_main(args, monkeypatch, module="train"):
    """Import a script module fresh and run its main() with the given sys.argv."""
    monkeypatch.setattr(sys, "argv", [f"{module}.py"] + args)
    # Force re-import so module-level imports pick up anything the test patched.
    sys.modules.pop(module, None)
    importlib.import_module(module).main()


def minimal_config(tmp_path):
    """Write a minimal YAML config to tmp_path and return its path."""
    import yaml

    config = {
        "environment": {
            "name": "CartPole-v1",
            "render_mode": None,
            "max_episode_steps": 20,
            "seed": 42,
        },
        "agent": {
            "type": "random",
            "config": {"learning_rate": 0.0003, "gamma": 0.99},
        },
        "training": {
            "num_episodes": 4,
            "max_steps_per_episode": 20,
            "eval_frequency": 2,
            "save_frequency": 10,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
        },
        "logging": {"level": "INFO", "log_metrics": True},
    }
    config_path = str(tmp_path / "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

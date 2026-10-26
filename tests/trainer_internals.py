"""The one place tests name a Trainer's private members, as tests/agent_internals.py is for agents.

Everything else goes through ``train()``, ``evaluate()`` and the files and log lines they produce
(tests/test_source_conventions.py holds the test files to that).
"""

from collections.abc import Mapping
from typing import Any

from rl_core.training.trainer import Trainer


def log_metrics(trainer: Trainer, metrics: Mapping[str, Any]) -> None:
    """Send one metrics record through the trainer's logger dispatch, as train() does every
    ten episodes; its five logger shapes are easier to pin one record at a time."""
    trainer._log_metrics(dict(metrics))

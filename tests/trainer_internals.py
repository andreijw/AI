"""The one place tests name a Trainer's private members, as tests/agent_internals.py is for agents.

Everything else goes through ``train()``, ``evaluate()`` and the files and log lines they produce
(tests/test_source_conventions.py holds the test files to that).
"""

from collections.abc import Mapping
from typing import Any

from rl_core.training import trainer as trainer_module
from rl_core.training.trainer import Trainer


def log_metrics(trainer: Trainer, metrics: Mapping[str, Any]) -> None:
    """Send one metrics record through the trainer's logger dispatch, as train() does every
    ten episodes; its five logger shapes are easier to pin one record at a time."""
    trainer._log_metrics(dict(metrics))


def force_lockstep(trainer: Trainer) -> None:
    """Train through the lockstep collector even with one environment, which training otherwise
    plays through the single-environment loop; tests/test_trainer_golden.py holds the two to the
    same recording."""
    trainer._lockstep = True


def record_step(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], bool]:
    """The trainer's ``_record_step``: one environment step as the transition both training loops
    store, and whether it ends the episode."""
    return trainer_module._record_step(*args, **kwargs)

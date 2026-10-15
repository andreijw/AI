"""The registry of benchmark challenges, and discovery of the packages that provide them.

A challenge package is a top-level package named ``rl_<something>`` with a ``benchmark`` module
that defines a :class:`~rl_core.challenges.BenchmarkChallenge` subclass decorated with
:func:`register_challenge`. :func:`discover_challenges` imports every such module it finds, so
adding a challenge needs no edit anywhere else.
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import rl_core

from .challenge import BenchmarkChallenge

#: Registered challenge classes by key (``"cartpole"``, ``"pendulum"``, ...).
CHALLENGES: dict[str, type[BenchmarkChallenge]] = {}

#: Challenge packages whose ``benchmark`` module failed to import, with the error.
IMPORT_FAILURES: dict[str, str] = {}

# The directory that holds rl_core and the challenge packages beside it (``src/``).
_DEFAULT_SEARCH_DIR = Path(rl_core.__file__).resolve().parent.parent


def register_challenge(cls: type[BenchmarkChallenge]) -> type[BenchmarkChallenge]:
    """
    Class decorator that adds a challenge to :data:`CHALLENGES` under its ``key``.

    The class is instantiated once with no arguments to read ``key``, so a challenge's
    ``__init__`` must accept being called with ``config_dir`` alone or with nothing.

    Raises:
        ValueError: If a different class is already registered under the same key.
    """
    key = cls().key
    existing = CHALLENGES.get(key)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Challenge key {key!r} already registered by {existing.__module__}.{existing.__name__}"
        )
    CHALLENGES[key] = cls
    return cls


def _challenge_packages(search_dir: Path) -> list[str]:
    """Names of the challenge packages directly inside ``search_dir``, sorted."""
    names = []
    for path in sorted(search_dir.iterdir()):
        is_package = path.is_dir() and (path / "__init__.py").is_file()
        if (
            is_package
            and path.name.isidentifier()
            and path.name.startswith("rl_")
            and path.name != "rl_core"
            and (path / "benchmark.py").is_file()
        ):
            names.append(path.name)
    return names


def discover_challenges(search_dir: Path | None = None) -> dict[str, type[BenchmarkChallenge]]:
    """
    Import every challenge package's ``benchmark`` module so that its challenges register.

    Modules are imported by their dotted name (``rl_cartpole.benchmark``), so ``search_dir``
    must be on ``sys.path`` and a module imported twice is not registered twice. A package
    whose module raises is skipped with a warning and recorded in :data:`IMPORT_FAILURES`.
    With a non-editable install the default directory is ``site-packages``, where any
    third-party ``rl_*`` package with a ``benchmark`` module would also be imported.

    Args:
        search_dir: Directory to scan. Defaults to the one that holds ``rl_core``.

    Returns:
        :data:`CHALLENGES`, after discovery.
    """
    for name in _challenge_packages(search_dir or _DEFAULT_SEARCH_DIR):
        try:
            importlib.import_module(f"{name}.benchmark")
        except Exception as error:  # a broken challenge must not hide the others
            reason = f"{type(error).__name__}: {error}"
            IMPORT_FAILURES[name] = reason
            warnings.warn(f"Challenge package {name} failed to import ({reason})", stacklevel=2)
        else:
            IMPORT_FAILURES.pop(name, None)
    return CHALLENGES


def import_failures_text() -> str:
    """One line per challenge package that failed to import, or an empty string."""
    if not IMPORT_FAILURES:
        return ""
    failed = ", ".join(f"{pkg} ({reason})" for pkg, reason in sorted(IMPORT_FAILURES.items()))
    return f"Failed to import: {failed}"


def unknown_challenge_message(name: str) -> str:
    """The error text for a challenge that is not registered, naming any import failures."""
    message = f"Unknown challenge '{name}'. Supported: {sorted(CHALLENGES)}"
    failures = import_failures_text()
    return f"{message}. {failures}" if failures else message

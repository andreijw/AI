"""The registry of benchmark challenges, and discovery of the packages that provide them.

A challenge package defines a :class:`~rl_core.challenges.BenchmarkChallenge` subclass decorated
with :func:`register_challenge` in a module that its distribution declares as an entry point in the
:data:`ENTRY_POINT_GROUP` group, named after the package::

    [project.entry-points."rl_core.challenges"]
    rl_cartpole = "rl_cartpole.benchmark"

:func:`discover_challenges` imports every such module, so another project (a game) adds a challenge
by declaring one entry point, with no edit here.
"""

from __future__ import annotations

import importlib
import re
import warnings
from collections.abc import Iterable
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path

from .challenge import BenchmarkChallenge

#: The entry-point group a distribution declares its challenge modules in.
ENTRY_POINT_GROUP = "rl_core.challenges"

#: This repository's distributions: their challenges load before any other project's, so a plugin
#: can never take an in-repo challenge's key.
FIRST_PARTY_DISTS = frozenset({"rl-core", "rl-navigation", "rl-challenges"})

#: How to make the challenges visible when none is found: entry points are installed metadata.
INSTALL_HINT = (
    "No challenge package is installed: run `pip install -r requirements-dev.txt` in the checkout "
    "(in a git worktree, scripts/worktree_venv.sh); an environment from before the split into "
    "packages/ needs `pip uninstall rl-cartpole` first"
)

#: Registered challenge classes by key (``"cartpole"``, ``"pendulum"``, ...).
CHALLENGES: dict[str, type[BenchmarkChallenge]] = {}

#: Challenge packages whose module failed to import, with the error, and entry points that clashed
#: with an earlier one (keyed ``<distribution>:<entry name>``).
IMPORT_FAILURES: dict[str, str] = {}


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


def dist_name(entry_point: EntryPoint) -> str:
    """The normalized name (PEP 503) of the distribution that declares ``entry_point``, or ``""``."""
    dist = entry_point.dist
    return re.sub(r"[-_.]+", "-", dist.name).lower() if dist is not None else ""


def challenge_entry_points() -> Iterable[EntryPoint]:
    """
    Every installed entry point in :data:`ENTRY_POINT_GROUP`: the source
    :func:`discover_challenges` reads.

    It is looked up on this module at each call, so a caller may replace it, as the test suite
    does to see this repository's distributions only (``tests/conftest.py``).
    """
    return entry_points(group=ENTRY_POINT_GROUP)


def _select_entry_points(
    found: Iterable[EntryPoint],
) -> tuple[list[EntryPoint], dict[str, str]]:
    """
    The entry points to load, in load order, and the clashes to record as failures.

    First-party distributions come first, then the rest by ``(distribution, entry name)``, so
    whichever class registers a key first does not depend on the order of ``sys.path``. An entry
    whose module an earlier one already names is dropped (two distributions shipping one package, as
    an old monolithic install beside the split ones). An entry
    that reuses an earlier entry's name for a different module is not loaded: it is a clash, keyed
    ``<distribution>:<name>``.
    """
    ordered = sorted(
        found, key=lambda ep: (dist_name(ep) not in FIRST_PARTY_DISTS, dist_name(ep), ep.name)
    )
    selected: list[EntryPoint] = []
    by_name: dict[str, EntryPoint] = {}
    clashes: dict[str, str] = {}
    for ep in ordered:
        if any(ep.value == chosen.value for chosen in selected):
            continue
        first = by_name.get(ep.name)
        if first is not None:
            clashes[f"{dist_name(ep)}:{ep.name}"] = (
                f"entry point {ep.name} = {ep.value} clashes with "
                f"{dist_name(first)}'s {ep.name} = {first.value}"
            )
            continue
        by_name[ep.name] = ep
        selected.append(ep)
    return selected, clashes


def _record_failure(name: str, reason: str, *, clash: bool = False) -> None:
    """Record and warn about a challenge package that failed to import, or (``clash``) an entry
    point that was not loaded; the warning names :func:`discover_challenges`'s caller."""
    IMPORT_FAILURES[name] = reason
    what = "entry point {} was not loaded" if clash else "package {} failed to import"
    warnings.warn(f"Challenge {what.format(name)} ({reason})", stacklevel=3)


def discover_challenges(search_dir: Path | None = None) -> dict[str, type[BenchmarkChallenge]]:
    """
    Import every challenge module so that its challenges register.

    With no ``search_dir``, the modules are the installed entry points in
    :data:`ENTRY_POINT_GROUP`, in :func:`_select_entry_points`'s order. With ``search_dir``, they
    are the ``benchmark`` modules of the ``rl_*`` packages directly inside it (``search_dir`` must
    be on ``sys.path``). A module imported twice is not registered twice. A module that raises,
    or exits (``SystemExit``), is skipped with a warning and recorded in :data:`IMPORT_FAILURES`
    under its top-level package.

    Args:
        search_dir: Directory to scan instead of the entry points (the tests' fake packages).

    Returns:
        :data:`CHALLENGES`, after discovery.
    """
    if search_dir is not None:
        modules = [(name, f"{name}.benchmark") for name in _challenge_packages(search_dir)]
    else:
        selected, clashes = _select_entry_points(challenge_entry_points())
        for name, reason in clashes.items():
            _record_failure(name, reason, clash=True)
        modules = [(ep.module.partition(".")[0], ep.module) for ep in selected]
    failed: set[str] = set()
    for package, module in modules:
        try:
            importlib.import_module(module)
        # A broken challenge must not hide the others, nor a plugin that calls sys.exit end the run.
        except (Exception, SystemExit) as error:
            _record_failure(package, f"{type(error).__name__}: {error}")
            failed.add(package)
        else:
            if package not in failed:  # another module of the package failed in this pass
                IMPORT_FAILURES.pop(package, None)
    return CHALLENGES


def import_failures_text() -> str:
    """One line per challenge package that failed to import, or an empty string."""
    if not IMPORT_FAILURES:
        return ""
    failed = ", ".join(f"{pkg} ({reason})" for pkg, reason in sorted(IMPORT_FAILURES.items()))
    return f"Failed to import: {failed}"


def _with_import_failures(message: str) -> str:
    """``message``, followed by the challenge packages that failed to import, if any, or by
    :data:`INSTALL_HINT` when no challenge is registered at all."""
    failures = import_failures_text()
    if failures:
        return f"{message}. {failures}"
    return message if CHALLENGES else f"{message}. {INSTALL_HINT}"


def unknown_challenge_message(name: str) -> str:
    """The error text for a challenge that is not registered, naming any import failures."""
    return _with_import_failures(f"Unknown challenge '{name}'. Supported: {sorted(CHALLENGES)}")


def select_challenge(env_name: str | None, key: str | None) -> type[BenchmarkChallenge]:
    """
    The registered challenge for a run, from its config's ``environment.name`` and a chosen key.

    The environment name must equal a challenge's ``name`` (its Gymnasium id) exactly. A key,
    when given as well, must name a challenge on that environment, and is required when several
    challenges share it; given alone, it decides. Call
    :func:`discover_challenges` first.

    Args:
        env_name: The config's ``environment.name``, or None if it has none.
        key: The challenge key chosen on the command line (``--challenge``), or None.

    Raises:
        ValueError: If the name or key is unknown, if they disagree, if the name is
            ambiguous without a key, or if both are None.
    """
    if key is not None and key not in CHALLENGES:
        raise ValueError(unknown_challenge_message(key))
    if env_name is None:
        if key is None:
            raise ValueError(
                "Cannot tell which challenge to run: set environment.name in the config "
                f"or choose one of {sorted(CHALLENGES)} with --challenge."
            )
        return CHALLENGES[key]

    names = {key_: cls().name for key_, cls in CHALLENGES.items()}
    matches = sorted(key_ for key_, name in names.items() if name == env_name)
    if not matches:
        options = ", ".join(f"{name} ({key_})" for key_, name in sorted(names.items()))
        raise ValueError(
            _with_import_failures(f"Unknown environment '{env_name}'. Supported: {options}")
        )
    if key is not None:
        if key not in matches:
            raise ValueError(
                f"--challenge {key} does not match the config's environment '{env_name}', "
                f"which belongs to challenge {' or '.join(repr(m) for m in matches)}."
            )
        return CHALLENGES[key]
    if len(matches) > 1:
        raise ValueError(
            f"Environment '{env_name}' belongs to several challenges; choose one of "
            f"{matches} with --challenge."
        )
    return CHALLENGES[matches[0]]

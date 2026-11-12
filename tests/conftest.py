"""Fixtures shared across test files."""

import sys

import pytest

import rl_core
from rl_core.challenges import CHALLENGES, registry
from tests.helpers import (
    FAKE_PACKAGE_PREFIX,
    first_party_entry_points,
    foreign_checkout_message,
    uses_onnx,
)


def pytest_configure(config):
    """Refuse to run against another checkout's code (a worktree using the main environment), then
    let discovery see this repository's challenges only, never a plugin installed beside them (a
    game). Before collection, because some test modules run discovery when they are imported."""
    message = foreign_checkout_message(config.rootpath, rl_core.__file__)
    if message is not None:
        pytest.exit(message, returncode=pytest.ExitCode.USAGE_ERROR)
    registry.challenge_entry_points = first_party_entry_points(registry.challenge_entry_points)


def pytest_collection_modifyitems(items):
    """Run the tests that start a thread for the rest of the process after all the others: those
    that render through pygame (they use ``clear_sdl``; SDL starts a native thread), then those that
    need onnxruntime (importing it starts one). A test that forks afterwards (an async vector
    environment) could deadlock in the child (Python warns "This process ... is multi-threaded, use
    of fork() may lead to deadlocks")."""
    items.sort(key=lambda item: (uses_onnx(item), "clear_sdl" in item.fixturenames))  # stable sort


@pytest.fixture(autouse=True)
def onnxruntime_only_in_onnx_tests(request):
    """Fail a test not marked needs_onnx that leaves onnxruntime imported: it would run before the
    tests that fork. A test that hides the onnx extra hides onnxruntime too (sys.modules[...] = None)."""
    yield
    if not uses_onnx(request.node) and sys.modules.get("onnxruntime") is not None:
        pytest.fail(
            "onnxruntime is imported, but this test is not marked needs_onnx (tests/helpers.py), so it "
            "runs before tests that fork: mark it, or hide onnxruntime as well as onnx"
        )


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    """Put tmp_path on the import path and undo every registration and import afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    challenges, failures = dict(CHALLENGES), dict(registry.IMPORT_FAILURES)
    # Read before the test can replace the entry-point source.
    first_party = {ep.module.partition(".")[0] for ep in registry.challenge_entry_points()}
    modules = set(sys.modules)
    yield tmp_path
    # A real challenge first imported during the test stays in sys.modules, so it would never
    # register again: keep it, and drop what the test registered. Real means a module of this
    # repository's entry points, whatever its package is called.
    real = {
        key: cls
        for key, cls in CHALLENGES.items()
        if cls.__module__.partition(".")[0] in first_party
    }
    CHALLENGES.clear()
    CHALLENGES.update({**real, **challenges})
    registry.IMPORT_FAILURES.clear()
    registry.IMPORT_FAILURES.update(failures)
    for name in set(sys.modules) - modules:
        if name.startswith(FAKE_PACKAGE_PREFIX):
            del sys.modules[name]


@pytest.fixture
def clear_sdl(monkeypatch):
    """Unset the SDL drivers so a test sees what the script's headless rendering sets."""
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)

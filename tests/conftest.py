"""Fixtures shared across test files."""

import sys

import pytest

from rl_core.challenges import CHALLENGES, registry
from tests.helpers import FAKE_PACKAGE_PREFIX, uses_onnx


def pytest_collection_modifyitems(items):
    """Run the tests that need onnxruntime after all the others. Importing it starts a thread that
    lives as long as the process, and a test that forks afterwards (an async vector environment)
    could deadlock in the child (Python warns "This process ... is multi-threaded, use of fork() may
    lead to deadlocks")."""
    items.sort(key=uses_onnx)  # stable: the order within each group is kept


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
    modules = set(sys.modules)
    yield tmp_path
    # A real challenge first imported during the test stays in sys.modules, so it would never
    # register again: keep it, and drop only what the test's fake packages registered (fake packages
    # are named by tests.helpers.unique_name, with FAKE_PACKAGE_PREFIX).
    real = {
        key: cls
        for key, cls in CHALLENGES.items()
        if cls.__module__.startswith("rl_") and not cls.__module__.startswith(FAKE_PACKAGE_PREFIX)
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

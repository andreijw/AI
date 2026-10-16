"""Fixtures shared across test files."""

import sys

import pytest

from rl_core.challenges import CHALLENGES, registry


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    """Put tmp_path on the import path and undo every registration and import afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    challenges, failures = dict(CHALLENGES), dict(registry.IMPORT_FAILURES)
    modules = set(sys.modules)
    yield tmp_path
    CHALLENGES.clear()
    CHALLENGES.update(challenges)
    registry.IMPORT_FAILURES.clear()
    registry.IMPORT_FAILURES.update(failures)
    for name in set(sys.modules) - modules:
        if name.startswith("rl_fake"):
            del sys.modules[name]


@pytest.fixture
def clear_sdl(monkeypatch):
    """Unset the SDL drivers so a test sees what the script's headless rendering sets."""
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)

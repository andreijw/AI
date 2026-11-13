"""rl_core.training.robustness, the robustness benchmark's 0.1.0 path: from 0.2.0 it re-exports
rl_core.robustness and warns on every name it hands out (PEP 562). The file goes with the shim in 0.3.0."""

import importlib
import warnings

import pytest

import rl_core.robustness as robustness

NAMES = [
    "Perturbation",
    "RobustnessRow",
    "canonical_setting",
    "difficulty_reached",
    "parse_sweep",
    "perturbed_section",
    "physics_defaults",
    "robustness_rows",
    "training_range",
]


def _deprecations(statement: str) -> list[warnings.WarningMessage]:
    """The DeprecationWarnings of running ``statement`` in a fresh namespace."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        exec(statement, {})  # a statement written in this file
    return [w for w in record if issubclass(w.category, DeprecationWarning)]


def test_the_new_module_declares_the_nine_names():
    assert sorted(robustness.__all__) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_each_old_name_warns_and_is_the_new_object(name):
    old = importlib.import_module("rl_core.training.robustness")
    with pytest.deprecated_call(match=rf"import {name} from rl_core\.robustness"):
        value = getattr(old, name)
    assert value is getattr(robustness, name)


def test_the_warning_points_at_the_importing_line():
    (warning,) = _deprecations("from rl_core.training.robustness import parse_sweep")
    assert "is deprecated since rl-core 0.2.0 and is removed in 0.3.0" in str(warning.message)
    assert warning.filename == "<string>", "attributed to the shim, not to the code importing it"


@pytest.mark.parametrize(
    ("statement", "count"),
    [
        ("from rl_core.training.robustness import *", 9),
        ("import inspect, rl_core.training.robustness as m; inspect.getmembers(m)", 9),
        ("import rl_core.training.robustness as m; dir(m)", 0),
        ("import rl_core.training.robustness", 0),
        ("from rl_core.training import robustness", 0),
    ],
    ids=["star", "getmembers", "dir", "import", "from-package"],
)
def test_how_many_warnings_each_use_gives(statement, count):
    assert len(_deprecations(statement)) == count


def test_a_star_import_gives_the_nine_names():
    namespace: dict[str, object] = {}
    with pytest.deprecated_call():
        exec("from rl_core.training.robustness import *", namespace)  # fixed statement
    assert sorted(k for k in namespace if not k.startswith("__")) == NAMES


def test_dir_lists_the_nine_names():
    old = importlib.import_module("rl_core.training.robustness")
    assert set(NAMES) <= set(dir(old))


def test_an_unknown_name_raises_without_a_warning():
    old = importlib.import_module("rl_core.training.robustness")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError, match="no_such_name"):
            old.no_such_name  # noqa: B018 - the attribute access is the test

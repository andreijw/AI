"""The one way rl-core and the distributions on it deprecate a name (rl_core.utils.deprecation)."""

import inspect
import warnings

import pytest

from rl_core.utils.deprecation import deprecated_function, warn_deprecated

MESSAGE = (
    "rl_core.old_name is deprecated since rl-core 0.2.0 and is removed in 0.3.0: "
    "import new_name from rl_core.new_home"
)


def _deprecated_function():
    warn_deprecated(
        "rl_core.old_name",
        "import new_name from rl_core.new_home",
        dist="rl-core",
        since="0.2.0",
        removed_in="0.3.0",
    )


def test_the_warning_names_the_version_the_removal_and_the_replacement():
    with pytest.warns(DeprecationWarning) as record:
        _deprecated_function()
    assert [str(w.message) for w in record] == [MESSAGE]


def test_the_warning_points_at_the_caller_of_the_deprecated_name():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        _deprecated_function()
    assert [(w.filename, w.lineno) for w in record] == [(__file__, line)]


def test_an_unexpected_deprecation_fails_the_suite():
    """pyproject.toml's filterwarnings makes the message an error unless a test expects it, so no
    in-tree code keeps using a deprecated name unnoticed."""
    with pytest.raises(DeprecationWarning, match="is removed in 0.3.0"):
        _deprecated_function()


def test_another_projects_deprecation_is_not_made_an_error():
    """The filter names our three distributions: warning filters ignore case, so a looser pattern
    such as ``rl-[a-z]+`` also matched another project's "RL-Lib"."""
    with warnings.catch_warnings(record=True) as record:
        warnings.warn(
            "x is deprecated since RL-Lib 1.0 and is removed in 2.0",
            DeprecationWarning,
            stacklevel=1,
        )
    assert len(record) == 1


def _builder(a, *, b=2):
    """Build something."""
    return a + b


def test_a_deprecated_function_warns_then_calls_the_builder():
    old = deprecated_function(
        _builder, "pkg.old", "use pkg.new", dist="rl-challenges", since="0.2.0", removed_in="0.3.0"
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        assert old(1, b=3) == 4
    assert [str(w.message) for w in record] == [
        "pkg.old is deprecated since rl-challenges 0.2.0 and is removed in 0.3.0: use pkg.new"
    ]
    assert (record[0].filename, record[0].lineno) == (__file__, line)


def test_a_deprecated_function_keeps_the_signature_and_says_it_is_deprecated():
    old = deprecated_function(
        _builder, "pkg.old", "use pkg.new", dist="rl-challenges", since="0.2.0", removed_in="0.3.0"
    )
    assert inspect.signature(old) == inspect.signature(_builder)
    assert old.__name__ == "old"
    assert old.__doc__.startswith(
        "Deprecated since rl-challenges 0.2.0, removed in 0.3.0: use pkg.new."
    )
    assert "Build something." in old.__doc__

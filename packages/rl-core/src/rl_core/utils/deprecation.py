"""How rl-core, and the distributions built on it, deprecate a name.

One wording everywhere: the repository's pytest ``filterwarnings`` (pyproject.toml) matches it and makes it an
error wherever a test does not expect the warning, so no in-tree code keeps using a deprecated name unnoticed.
"""

import functools
import warnings
from collections.abc import Callable


def warn_deprecated(
    old: str, replacement: str, *, dist: str, since: str, removed_in: str, stacklevel: int = 3
) -> None:
    """Warn that ``old`` is deprecated: "<old> is deprecated since <dist> <since> and is removed in
    <removed_in>: <replacement>".

    ``stacklevel`` 3 attributes the warning to the caller of the deprecated function that calls this.
    """
    warnings.warn(
        f"{old} is deprecated since {dist} {since} and is removed in {removed_in}: {replacement}",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated_function[**P, R](
    builder: Callable[P, R], old: str, replacement: str, *, dist: str, since: str, removed_in: str
) -> Callable[P, R]:
    """``builder`` under its deprecated public name: each call warns (:func:`warn_deprecated`, at the
    caller's line), then calls ``builder``. The wrapper keeps ``builder``'s signature, takes the last part
    of ``old`` as its name and opens its docstring with the deprecation."""

    @functools.wraps(builder)
    def deprecated(*args: P.args, **kwargs: P.kwargs) -> R:
        warn_deprecated(old, replacement, dist=dist, since=since, removed_in=removed_in)
        return builder(*args, **kwargs)

    deprecated.__name__ = deprecated.__qualname__ = old.rpartition(".")[2]
    notice = f"Deprecated since {dist} {since}, removed in {removed_in}: {replacement}."
    deprecated.__doc__ = f"{notice}\n\n{builder.__doc__ or ''}"
    return deprecated

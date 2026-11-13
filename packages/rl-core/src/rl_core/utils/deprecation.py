"""How rl-core, and the distributions built on it, deprecate a name.

One wording everywhere: the repository's pytest ``filterwarnings`` (pyproject.toml) matches it and makes it an
error wherever a test does not expect the warning, so no in-tree code keeps using a deprecated name unnoticed.
"""

import warnings


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

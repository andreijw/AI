#!/usr/bin/env python3
"""Print ``true`` when every changed path read from stdin is documentation, else ``false``.

CI pipes ``git diff --name-only`` of a pull request into this script and skips the Python checks and
tests on docs-only changes, while still reporting the required checks. Anything unexpected (an empty
list, code, tests, CI files) prints ``false`` so everything runs.

It runs on the runner's own ``python3`` before setup-python: keep it stdlib-only.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

CODE_DIRS = ("src/", "tests/")


def is_docs_only(paths: Iterable[str]) -> bool:
    """True when the list is non-empty and each path is under docs/ or a Markdown file outside code."""
    paths = list(paths)
    return bool(paths) and all(
        not path.startswith(CODE_DIRS) and (path.startswith("docs/") or path.endswith(".md"))
        for path in paths
    )


def main() -> int:
    paths = [line.strip() for line in sys.stdin if line.strip()]
    print("true" if is_docs_only(paths) else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The commit and working-tree state a desk run records (standard library and git only).

``compare_planner.py`` and ``scripts/planning_timing/time_planner.py`` write them into their
output, so that records of different code are never merged, resumed or reported together. A dirty
tree is named by a hash of its content against HEAD, unstaged and untracked changes included; where
git cannot say, the value is :data:`UNKNOWN`, which :func:`same` never matches.
"""

import hashlib
import os
import subprocess  # nosec B404 - git, with fixed arguments
from pathlib import Path

UNKNOWN = "unknown"


def git_output(root: str | Path, *args: str) -> bytes | None:
    """git's output in ``root``, or None where git cannot say."""
    try:
        result = subprocess.run(  # nosec B603 B607 - fixed git arguments
            ["git", "-C", str(root), *args], capture_output=True, timeout=60, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def commit(root: str | Path) -> str:
    """HEAD's commit hash, or :data:`UNKNOWN`."""
    out = git_output(root, "rev-parse", "HEAD")
    return UNKNOWN if out is None else out.decode().strip()


def working_tree(root: str | Path) -> str:
    """``"clean"``; ``"dirty; content sha256 <hex>"``, the hash of ``git diff HEAD`` and of every
    untracked, not ignored file's path and bytes; or :data:`UNKNOWN`."""
    status = git_output(root, "status", "--porcelain")
    if status is None:
        return UNKNOWN
    if not status.strip():
        return "clean"
    diff = git_output(root, "diff", "HEAD", "--binary")
    untracked = git_output(root, "ls-files", "--others", "--exclude-standard", "-z")
    if diff is None or untracked is None:
        return UNKNOWN
    digest = hashlib.sha256(diff)
    for name in sorted(n for n in untracked.split(b"\0") if n):
        digest.update(b"\0" + name + b"\0")
        try:
            with open(os.path.join(os.fsencode(str(root)), name), "rb") as f:
                digest.update(f.read())
        except OSError:
            return UNKNOWN
    return f"dirty; content sha256 {digest.hexdigest()}"


def same(a: str | None, b: str | None) -> bool:
    """Whether two recorded values agree; :data:`UNKNOWN` (or a missing value) never does."""
    return a is not None and a == b and a != UNKNOWN

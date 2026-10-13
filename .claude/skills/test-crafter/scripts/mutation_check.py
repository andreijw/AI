#!/usr/bin/env python3
"""Apply small breaks to production code one at a time and report which tests notice.

Usage (from the repository root, with the virtual environment's Python):

    python .claude/skills/test-crafter/scripts/mutation_check.py mutations.json

``mutations.json`` is a list of objects:

    [
      {
        "file": "src/rl_core/agents/dqn_agent.py",
        "old": "(1.0 - dones)",
        "new": "1.0",
        "tests": ["tests/test_dqn_agent.py"],
        "why": "bootstrap is no longer masked on terminal steps"
      }
    ]

``old`` must occur exactly once in ``file``. ``tests`` are pytest targets (files or node ids).
Each mutation is applied, the tests are run, and the original bytes are written back even if
the run is interrupted. A mutation is KILLED when at least one test fails and SURVIVED when
every test still passes; a survivor means the behaviour is not locked in.

Exit code: 0 if every mutation was killed, 1 if any survived, 2 on a usage or setup error.
"""

from __future__ import annotations

import json
import re
import subprocess  # nosec B404
import sys
from pathlib import Path

FAILED_LINE = re.compile(r"^(?:FAILED|ERROR) (\S+)", flags=re.MULTILINE)


def _run_tests(tests: list[str]) -> tuple[bool, list[str]]:
    """Run pytest on the targets; return (all passed, names of failing tests)."""
    result = subprocess.run(  # nosec B603
        [sys.executable, "-m", "pytest", *tests, "--no-cov", "-q", "-x", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        check=False,
    )
    failing = FAILED_LINE.findall(result.stdout)
    if result.returncode not in (0, 1):
        # Collection or usage error: pytest never ran the tests, so nothing was proven.
        failing.append(f"pytest exited with code {result.returncode}")
    return result.returncode == 0, failing


def _load(path: Path) -> list[dict]:
    mutations = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mutations, list) or not mutations:
        raise ValueError("The mutation file must be a non-empty JSON list.")
    for index, mutation in enumerate(mutations):
        missing = {"file", "old", "new", "tests"} - set(mutation)
        if missing:
            raise ValueError(f"Mutation {index} is missing {sorted(missing)}.")
        if mutation["old"] == mutation["new"]:
            raise ValueError(f"Mutation {index}: 'old' and 'new' are identical.")
        source = Path(mutation["file"]).read_text(encoding="utf-8")
        count = source.count(mutation["old"])
        if count != 1:
            raise ValueError(
                f"Mutation {index}: 'old' occurs {count} times in {mutation['file']}; "
                "it must occur exactly once."
            )
    return mutations


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    try:
        mutations = _load(Path(sys.argv[1]))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    all_tests = sorted({test for mutation in mutations for test in mutation["tests"]})
    passed, failing = _run_tests(all_tests)
    if not passed:
        print(f"error: the tests must pass before mutating. Failing: {failing}")
        return 2

    survivors = 0
    for index, mutation in enumerate(mutations):
        path = Path(mutation["file"])
        original = path.read_bytes()
        try:
            text = original.decode("utf-8").replace(mutation["old"], mutation["new"])
            path.write_bytes(text.encode("utf-8"))
            passed, failing = _run_tests(mutation["tests"])
        finally:
            path.write_bytes(original)
        label = mutation.get("why", f"{mutation['old']!r} -> {mutation['new']!r}")
        if passed:
            survivors += 1
            print(f"SURVIVED  [{index}] {path}: {label}")
        else:
            print(f"KILLED    [{index}] {path}: {label}")
            for name in failing:
                print(f"            by {name}")

    print(f"\n{len(mutations) - survivors} killed, {survivors} survived.")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())

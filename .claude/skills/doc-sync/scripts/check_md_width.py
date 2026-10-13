#!/usr/bin/env python3
"""Report Markdown prose lines longer than the limit in .markdownlint.json (rule MD013).

A stand-in for the one markdownlint rule that most often fails CI, for machines without Node.
It is not a replacement for markdownlint: it checks line length only. Code blocks and tables
are skipped when the config exempts them, as markdownlint does.

Usage (from the repository root):

    python .claude/skills/doc-sync/scripts/check_md_width.py                # every tracked *.md
    python .claude/skills/doc-sync/scripts/check_md_width.py README.md ...  # the files named

Exit code: 0 if no line is too long, 1 otherwise.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404
import sys
from pathlib import Path

CONFIG_PATH = Path(".markdownlint.json")
FENCES = ("```", "~~~")


def _rule() -> tuple[int, bool, bool]:
    """Return (line length, check code blocks, check tables) from the markdownlint config."""
    rule = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("MD013", {})
    if not isinstance(rule, dict):
        rule = {}
    return rule.get("line_length", 80), rule.get("code_blocks", True), rule.get("tables", True)


def _tracked_markdown() -> list[str]:
    result = subprocess.run(  # nosec B603 B607
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        capture_output=True,
        text=True,
        check=True,
    )
    # markdownlint's glob skips dot-directories unless told otherwise; match that.
    return [
        name
        for name in result.stdout.splitlines()
        if not any(part.startswith(".") for part in Path(name).parts)
    ]


def long_lines(path: Path, limit: int, code_blocks: bool, tables: bool) -> list[tuple[int, int]]:
    """Return (line number, length) for each line of the file that exceeds the limit."""
    found = []
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(FENCES):
            in_fence = not in_fence
            continue
        if in_fence and not code_blocks:
            continue
        if stripped.startswith("|") and not tables:
            continue
        if len(line) > limit:
            found.append((number, len(line)))
    return found


def main() -> int:
    limit, code_blocks, tables = _rule()
    names = sys.argv[1:] or _tracked_markdown()
    total = 0
    for name in names:
        for number, length in long_lines(Path(name), limit, code_blocks, tables):
            total += 1
            print(f"{name}:{number} is {length} characters (limit {limit})")
    print(f"{total} line(s) over {limit} characters in {len(names)} file(s).")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())

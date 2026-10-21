#!/usr/bin/env python3
"""Synchronize the README workflow list with files in .github/workflows.

``--check`` writes nothing and exits 1 when the list is stale; CI runs it on every pull request with
the runner's own ``python3``, so keep this script stdlib-only.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
README_PATH = REPO_ROOT / "README.md"
START_MARKER = "<!-- WORKFLOWS:START -->"
END_MARKER = "<!-- WORKFLOWS:END -->"


def _extract_name(workflow_text: str, fallback: str) -> str:
    match = re.search(r"^name:\s*[\"']?(.+?)[\"']?\s*$", workflow_text, flags=re.MULTILINE)
    return match.group(1) if match else fallback


def build_workflow_block(workflow_dir: Path) -> str:
    workflow_files = sorted(workflow_dir.glob("*.yml")) + sorted(workflow_dir.glob("*.yaml"))
    lines = [START_MARKER, ""]
    for workflow_file in workflow_files:
        workflow_text = workflow_file.read_text(encoding="utf-8")
        workflow_name = _extract_name(workflow_text, workflow_file.name)
        lines.append(f"- **{workflow_file.name}** - {workflow_name}")
    lines.extend(["", END_MARKER])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if README.md is stale")
    args = parser.parse_args(argv)

    readme = README_PATH.read_text(encoding="utf-8")
    if START_MARKER not in readme or END_MARKER not in readme:
        raise ValueError("README.md is missing workflow markers for automated sync.")

    updated_block = build_workflow_block(WORKFLOW_DIR)
    pattern = re.compile(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        flags=re.DOTALL,
    )
    updated_readme = pattern.sub(lambda _: updated_block, readme)

    if updated_readme == readme:
        print("README.md already up to date.")
        return 0
    if args.check:
        print("README.md workflow list is stale: run `python scripts/sync_workflow_docs.py`.")
        return 1
    README_PATH.write_text(updated_readme, encoding="utf-8")
    print("README.md updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

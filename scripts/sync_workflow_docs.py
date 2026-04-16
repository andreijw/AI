#!/usr/bin/env python3
"""Synchronize the README workflow list with files in .github/workflows."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW_DIR = Path(".github/workflows")
README_PATH = Path("README.md")
START_MARKER = "<!-- WORKFLOWS:START -->"
END_MARKER = "<!-- WORKFLOWS:END -->"


def _extract_name(workflow_text: str, fallback: str) -> str:
    match = re.search(r"^name:\s*[\"']?(.+?)[\"']?\s*$", workflow_text, flags=re.MULTILINE)
    return match.group(1) if match else fallback


def _build_workflow_block() -> str:
    workflow_files = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    lines = [START_MARKER, ""]
    for workflow_file in workflow_files:
        workflow_text = workflow_file.read_text(encoding="utf-8")
        workflow_name = _extract_name(workflow_text, workflow_file.name)
        lines.append(f"- **{workflow_file.name}** - {workflow_name}")
    lines.extend(["", END_MARKER])
    return "\n".join(lines)


def main() -> int:
    readme = README_PATH.read_text(encoding="utf-8")
    if START_MARKER not in readme or END_MARKER not in readme:
        raise ValueError("README.md is missing workflow markers for automated sync.")

    updated_block = _build_workflow_block()
    pattern = re.compile(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        flags=re.DOTALL,
    )
    updated_readme = pattern.sub(updated_block, readme)

    if updated_readme != readme:
        README_PATH.write_text(updated_readme, encoding="utf-8")
        print("README.md updated.")
    else:
        print("README.md already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

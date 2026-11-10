#!/usr/bin/env python3
"""Check that every relative link in the repository's Markdown resolves.

A link counts when its target is a file or folder (resolved from the folder of the file that holds
it) and, if it carries an anchor into a Markdown file, when that file has a heading with the
GitHub slug. Fenced code, inline code and URLs (``http:``, ``https:``, ``mailto:``) are skipped.

One exception: a missing target inside ``plots/``, ``logs/``, ``checkpoints/`` or ``videos/`` passes
when git ignores it, since those folders hold generated output (a benchmark report links its plot)
that a fresh clone does not have. A typo in a link into one of them therefore passes too.

CI runs this on every pull request, docs-only ones included, with the runner's own ``python3``, so
keep it stdlib-only.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIPPED_DIRS = {".git", "node_modules", "__pycache__"}
GENERATED_DIRS = {"plots", "logs", "checkpoints", "videos"}

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
INLINE_CODE = re.compile(r"(`+)(?:(?!\1).)+?\1", re.DOTALL)
LINK = re.compile(
    r"\]\(\s*(<[^>]*>|[^()\s]*(?:\([^()\s]*\)[^()\s]*)*)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
HTML_ANCHOR = re.compile(r"""<a\s[^>]*?(?:id|name)=["']([^"']+)["']""", re.IGNORECASE)
URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def markdown_files(root: Path) -> list[Path]:
    """Tracked and not-yet-tracked (but not ignored) Markdown files; a walk when git is absent."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "*.md"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8")
        names = [name for name in listed.split("\0") if name]
        return sorted(root / name for name in names if (root / name).is_file())
    except (OSError, subprocess.CalledProcessError):
        return sorted(
            path
            for path in root.rglob("*.md")
            if not SKIPPED_DIRS & set(path.relative_to(root).parts)
        )


def _blank(match: re.Match[str]) -> str:
    """The same text with every character but newlines replaced, so line numbers survive."""
    return re.sub(r"[^\n]", " ", match.group(0))


def outside_fences(text: str) -> str:
    """``text`` with fenced code blocks blanked out (line numbers kept)."""
    kept: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        opener = FENCE.match(line)
        if fence is None and opener:
            fence = opener.group(1)[0] * len(opener.group(1))
            kept.append("")
        elif fence is not None:
            closes = opener and line.strip() == opener.group(1)[0] * len(line.strip())
            if closes and len(line.strip()) >= len(fence) and line.strip()[0] == fence[0]:
                fence = None
            kept.append("")
        else:
            kept.append(line)
    return "\n".join(kept)


def prose(text: str) -> str:
    """``text`` with fenced code and inline code blanked out."""
    return INLINE_CODE.sub(_blank, outside_fences(text))


def slug(heading: str) -> str:
    """GitHub's anchor for a heading's text: lower case, markup and punctuation dropped."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # a link keeps its text
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("`", "").replace("*", "")
    text = re.sub(r"(?<!\w)_|_(?!\w)", "", text)  # emphasis underscores, not snake_case
    text = text.strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors(text: str) -> set[str]:
    """Every anchor a Markdown file offers: its heading slugs (repeats get -1, -2) and HTML ids."""
    found: set[str] = set()
    seen: dict[str, int] = {}
    for line in outside_fences(text).split("\n"):
        match = HEADING.match(line)
        if match:
            base = slug(match.group(1))
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f"{base}-{count}")
    found.update(HTML_ANCHOR.findall(text))
    return found


def _git_ignored_output(root: Path, path: Path) -> bool:
    """True for a path inside a generated-output folder that git ignores (a link to a plot)."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    if not relative.parts or relative.parts[0] not in GENERATED_DIRS:
        return False
    try:
        done = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", str(relative)],
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return done.returncode == 0


def check_links(root: Path) -> tuple[list[str], int]:
    """Broken links under ``root`` as ``file:line: message`` strings, and how many were checked."""
    texts = {path: path.read_text(encoding="utf-8") for path in markdown_files(root)}
    anchor_cache: dict[Path, set[str]] = {}

    def anchors_of(path: Path) -> set[str]:
        if path not in anchor_cache:
            anchor_cache[path] = {a.lower() for a in anchors(path.read_text(encoding="utf-8"))}
        return anchor_cache[path]

    problems: list[str] = []
    checked = 0
    for path, text in texts.items():
        where = path.relative_to(root).as_posix()
        body = prose(text)
        for match in LINK.finditer(body):
            target = match.group(1).strip("<>")
            if not target or URL_SCHEME.match(target):
                continue
            checked += 1
            line = body.count("\n", 0, match.start()) + 1
            file_part, _, anchor = target.partition("#")
            file_part, anchor = unquote(file_part), unquote(anchor)
            if file_part.startswith("/"):
                destination = root / file_part.lstrip("/")
            else:
                destination = path.parent / file_part if file_part else path
            if file_part and not destination.exists():
                if not _git_ignored_output(root, destination):
                    problems.append(f"{where}:{line}: no such file or folder: {target}")
                continue
            is_markdown = destination.is_file() and destination.suffix.lower() == ".md"
            if anchor and is_markdown and anchor.lower() not in anchors_of(destination):
                problems.append(f"{where}:{line}: no heading for the anchor: {target}")
    return problems, checked


def main() -> int:
    problems, checked = check_links(REPO_ROOT)
    for problem in problems:
        print(problem)
    print(f"{len(problems)} broken links over {checked} checked.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

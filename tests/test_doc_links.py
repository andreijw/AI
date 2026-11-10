"""Every relative link, cross-file anchor and file path in the repository's Markdown resolves."""

import subprocess
from pathlib import Path

from tests.helpers import load_script

check_doc_links = load_script("check_doc_links")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _problems(root: Path) -> list[str]:
    return check_doc_links.check_links(root)[0]


def test_the_repository_has_no_broken_link():
    problems, checked = check_doc_links.check_links(REPO_ROOT)
    assert checked > 50, f"only {checked} links found: the scan lost the Markdown files"
    assert problems == [], "broken links:\n" + "\n".join(problems)


def test_a_link_to_a_missing_file_is_reported_with_its_file_and_line(tmp_path):
    _write(tmp_path, "README.md", "# Title\n\nSee [the guide](docs/guide.md).\n")
    problems = _problems(tmp_path)
    assert len(problems) == 1, f"expected one broken link, got {problems}"
    assert "README.md:3" in problems[0] and "docs/guide.md" in problems[0]


def test_a_link_to_an_existing_file_or_folder_passes(tmp_path):
    _write(tmp_path, "docs/guide.md", "# Guide\n")
    _write(tmp_path, "README.md", "[g](docs/guide.md) and [d](docs/)\n")
    assert _problems(tmp_path) == []


def test_a_path_resolves_from_the_folder_of_the_file_that_holds_it(tmp_path):
    _write(tmp_path, "README.md", "# Top\n")
    _write(tmp_path, "docs/guide.md", "[up](../README.md) [wrong](docs/guide.md)\n")
    problems = _problems(tmp_path)
    assert len(problems) == 1, (
        f"only the link written from the repository root should fail: {problems}"
    )
    assert problems[0].startswith("docs/guide.md:1:"), problems[0]
    assert problems[0].endswith("no such file or folder: docs/guide.md"), problems[0]


def test_a_cross_file_anchor_must_match_a_heading_by_github_slug_rules(tmp_path):
    _write(
        tmp_path,
        "docs/guide.md",
        "# Guide\n\n## AI Assistant Skills\n\n## `Foo` (bar, baz)\n\n## Same\n\n## Same\n",
    )
    _write(
        tmp_path,
        "README.md",
        "[a](docs/guide.md#ai-assistant-skills) [b](docs/guide.md#foo-bar-baz) "
        "[c](docs/guide.md#same-1) [d](docs/guide.md#missing)\n",
    )
    problems = _problems(tmp_path)
    assert len(problems) == 1 and "#missing" in problems[0], f"only #missing is wrong: {problems}"


def test_a_same_file_anchor_is_checked_too(tmp_path):
    _write(tmp_path, "README.md", "# Top\n\n[ok](#top) [bad](#nowhere)\n")
    problems = _problems(tmp_path)
    assert len(problems) == 1 and "#nowhere" in problems[0], f"got {problems}"


def test_code_and_urls_are_not_checked(tmp_path):
    _write(
        tmp_path,
        "README.md",
        "# Top\n\n```\n[x](missing.md)\n```\n\nInline `[y](missing.md)` and "
        "[z](https://example.com/missing.md) and [m](mailto:a@b.c).\n",
    )
    assert _problems(tmp_path) == []


def test_non_ascii_headings_and_files_are_read_as_utf_8(tmp_path):
    _write(tmp_path, "docs/guide.md", "## Résumé → what's next\n")
    _write(tmp_path, "README.md", "[r](docs/guide.md#résumé--whats-next)\n")
    assert _problems(tmp_path) == []


def _git_repo(root: Path, ignore: str) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text(ignore, encoding="utf-8")


def test_a_missing_file_in_an_ignored_output_folder_passes(tmp_path):
    _git_repo(tmp_path, "plots/\n")
    _write(tmp_path, "README.md", "![plot](plots/x.png)\n")
    assert _problems(tmp_path) == [], "a generated plot is absent from a fresh clone, not broken"


def test_a_missing_tracked_path_is_still_reported_in_a_git_repo(tmp_path):
    _git_repo(tmp_path, "plots/\n")
    _write(tmp_path, "README.md", "[a](docs/missing.md) [b](other/x.png)\n")
    problems = _problems(tmp_path)
    assert len(problems) == 2, f"neither path is an ignored output folder: {problems}"


def test_an_ignored_path_outside_the_output_folders_is_still_reported(tmp_path):
    _git_repo(tmp_path, "build/\n")
    _write(tmp_path, "README.md", "[a](build/x.png)\n")
    assert len(_problems(tmp_path)) == 1, "only plots, logs, checkpoints and videos are exempt"

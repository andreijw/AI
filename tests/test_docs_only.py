"""Which pull requests count as docs-only, so CI skips the Python work but still reports its checks."""

import io

import pytest

from tests.helpers import load_script

docs_only = load_script("docs_only")


@pytest.mark.parametrize(
    "paths",
    [
        ["docs/roadmap.md"],
        ["README.md", "CHANGELOG.md"],
        ["docs/benchmarks/cartpole_results.md", ".claude/skills/dev-loop/SKILL.md"],
        ["docs/images/curve.png"],
        ["docs/conf.py"],  # anything under docs/ is documentation; no test or build reads it
    ],
)
def test_markdown_and_docs_changes_are_docs_only(paths):
    assert docs_only.is_docs_only(paths), (
        f"{paths} touch only documentation and should skip the work"
    )


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["src/rl_core/agents/dqn.py"],
        ["README.md", "pyproject.toml"],
        ["src/rl_core/NOTES.md"],
        ["tests/data/fixture.md"],
        [".github/workflows/ci.yml"],
        ["docs/roadmap.md", "scripts/docs_only.py"],
    ],
)
def test_anything_else_runs_everything(paths):
    assert not docs_only.is_docs_only(paths), (
        f"{paths} is not docs-only (an empty list or a code/test/CI path must run every check)"
    )


@pytest.mark.parametrize(
    ("stdin", "expected"),
    [("docs/a.md\nREADME.md\n", "true"), ("README.md\nsrc/x.py\n", "false"), ("\n", "false")],
)
def test_main_reads_paths_from_stdin_and_prints_the_verdict(monkeypatch, capsys, stdin, expected):
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    assert docs_only.main() == 0
    assert capsys.readouterr().out.strip() == expected, f"stdin {stdin!r} should print {expected}"

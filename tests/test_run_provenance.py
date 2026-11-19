"""run_provenance.py: the commit and working-tree state a desk run records, so that records of
different code are never merged, resumed or reported together."""

from pathlib import Path

import pytest

import run_provenance
from run_provenance import UNKNOWN, commit, same, working_tree

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeGit:
    """git's answers in a folder, set by the test: no repository is made or written."""

    def __init__(self, status=b"", diff=b"", untracked=b""):
        self.answers = {
            "status": status,
            "diff": diff,
            "ls-files": untracked,
            "rev-parse": b"c0ffee\n",
        }

    def __call__(self, root, *args):
        return self.answers[args[0]]


@pytest.fixture
def fake(monkeypatch):
    git = FakeGit()
    monkeypatch.setattr(run_provenance, "git_output", git)
    return git


def test_a_clean_tree_is_clean_at_its_commit(fake, tmp_path):
    assert working_tree(tmp_path) == "clean"
    assert commit(tmp_path) == "c0ffee"


def test_a_dirty_tree_is_hashed_with_its_unstaged_and_untracked_changes(fake, tmp_path):
    fake.answers.update(status=b" M a.txt\n", diff=b"-one\n+two\n")
    first = working_tree(tmp_path)
    assert first.startswith("dirty; content sha256 ")
    fake.answers["diff"] = b"-one\n+three\n"  # an unstaged edit shows in git diff HEAD
    second = working_tree(tmp_path)
    assert second != first, "an unstaged edit changes the hash"
    (tmp_path / "new.txt").write_text("x\n", encoding="utf-8")
    fake.answers["ls-files"] = b"new.txt\0"
    third = working_tree(tmp_path)
    assert third != second, "an untracked file changes the hash"
    (tmp_path / "new.txt").write_text("y\n", encoding="utf-8")
    assert working_tree(tmp_path) != third, "and so does its content"
    assert working_tree(tmp_path) == working_tree(tmp_path), "the same tree hashes alike"


def test_this_checkout_has_a_commit_and_a_known_tree():
    assert len(commit(REPO_ROOT)) == 40
    assert working_tree(REPO_ROOT) == "clean" or working_tree(REPO_ROOT).startswith("dirty; ")


def test_outside_a_repository_both_are_unknown(tmp_path):
    assert commit(tmp_path) == UNKNOWN and working_tree(tmp_path) == UNKNOWN


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("x", "x", True), ("x", "y", False), (UNKNOWN, UNKNOWN, False), (None, None, False)],
)
def test_unknown_is_never_the_same(a, b, expected):
    assert same(a, b) is expected

"""The README workflow list sync, and the --check mode CI runs on every pull request."""

import pytest

from tests.helpers import load_script

sync = load_script("sync_workflow_docs")

README = """# Title

<!-- WORKFLOWS:START -->

- **old.yml** - Old

<!-- WORKFLOWS:END -->

Tail.
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repository with two workflows and a stale README, wired into the script."""
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("name: CI\non: push\n", encoding="utf-8")
    (workflows / "b.yaml").write_text("on: push\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    monkeypatch.setattr(sync, "WORKFLOW_DIR", workflows)
    monkeypatch.setattr(sync, "README_PATH", readme)
    return readme


def test_block_lists_every_workflow_with_its_name_or_file_name(repo):
    block = sync.build_workflow_block(sync.WORKFLOW_DIR)
    assert "- **ci.yml** - CI" in block, "a workflow's `name:` should be listed"
    assert "- **b.yaml** - b.yaml" in block, (
        "a workflow without `name:` falls back to its file name"
    )


def test_check_fails_on_a_stale_readme_without_writing_it(repo, capsys):
    assert sync.main(["--check"]) == 1, "--check must fail when the README list is stale"
    assert repo.read_text(encoding="utf-8") == README, "--check must never rewrite the README"
    assert "sync_workflow_docs.py" in capsys.readouterr().out, (
        "the failure should say how to fix it"
    )


def test_sync_rewrites_only_the_block_and_then_check_passes(repo):
    assert sync.main([]) == 0
    text = repo.read_text(encoding="utf-8")
    assert "old.yml" not in text and "- **ci.yml** - CI" in text, "the block should be regenerated"
    assert text.startswith("# Title") and text.endswith("Tail.\n"), "text outside the block is kept"
    assert sync.main(["--check"]) == 0, "a freshly synced README should pass --check"

"""scripts/release.py: what makes a tag releasable, and which modules a plain install can import."""

import ast
import sys
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

from tests.helpers import load_script

release = load_script("release")

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = REPO_ROOT / "packages"

CHANGELOG = """# Changelog

## [Unreleased]

## [{version}] - 2026-11-11

### Added

- {note}

## [0.0.9] - 2026-11-01

- older
"""


def _dist(packages, name, version="0.1.0", dependencies=(), extras=None, note="First release."):
    """A fake distribution under ``packages`` with a static version and a CHANGELOG section."""
    root = packages / name
    (root / "src").mkdir(parents=True)
    lines = [
        "[project]",
        f'name = "{name}"',
        f'version = "{version}"',
        f"dependencies = {list(dependencies)!r}".replace("'", '"'),
    ]
    if extras:
        lines.append("[project.optional-dependencies]")
        lines += [f"{key} = {value!r}".replace("'", '"') for key, value in extras.items()]
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "CHANGELOG.md").write_text(
        CHANGELOG.format(version=version, note=note), encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------- tags


@pytest.mark.parametrize(
    ("tag", "dist", "version"),
    [
        ("rl-navigation-v0.1.0", "rl-navigation", "0.1.0"),
        ("rl-core-v0.2.0rc1", "rl-core", "0.2.0rc1"),
        ("r2-voice-v1.0.0", "r2-voice", "1.0.0"),
    ],
)
def test_a_tag_names_a_distribution_and_its_version(tag, dist, version):
    assert release.parse_tag(tag) == (dist, Version(version))


@pytest.mark.parametrize(
    "tag", ["rl-core-0.1.0", "v0.1.0", "rl-core-vnext", "rl-core-v01.0", "RL-core-v0.1.0"]
)
def test_a_malformed_tag_is_refused(tag):
    with pytest.raises(release.ReleaseError):
        release.parse_tag(tag)


# ---------------------------------------------------------------- check


def test_a_tag_matching_the_version_and_changelog_is_releasable(tmp_path):
    _dist(tmp_path, "rl-core", note="Split out.")
    dist, version, notes = release.check("rl-core-v0.1.0", [], tmp_path)
    assert (dist, version) == ("rl-core", Version("0.1.0"))
    assert notes == "### Added\n\n- Split out.\n", "the section's body only, not the next one"


def test_a_tag_must_spell_out_the_version(tmp_path):
    """Version("0.1") equals Version("0.1.0"), but the tag names the release users install."""
    _dist(tmp_path, "rl-core")
    with pytest.raises(release.ReleaseError, match="does not match rl-core's version 0.1.0"):
        release.check("rl-core-v0.1", [], tmp_path)


@pytest.mark.parametrize(
    ("version", "prerelease"),
    [("0.1.0", "false"), ("0.2.0rc1", "true"), ("0.2.0.dev1", "true"), ("1.0.0b2", "true")],
)
def test_the_workflow_is_told_whether_a_version_is_a_prerelease(version, prerelease):
    outputs = release.github_outputs("rl-core", Version(version))
    assert outputs == f"dist=rl-core\nversion={version}\nprerelease={prerelease}\n"


def test_a_prerelease_dependency_inside_the_range_is_admitted(tmp_path):
    """A release candidate of rl-core is installed by its URL, so ~=0.2.0 accepts 0.2.1rc1, which
    a plain specifier check (no prereleases) would refuse."""
    _dist(tmp_path, "rl-core", version="0.2.1rc1")
    _dist(tmp_path, "rl-challenges", dependencies=["rl-core~=0.2.0"])
    release.check("rl-challenges-v0.1.0", ["rl-core-v0.2.1rc1"], tmp_path)


def test_a_tag_that_does_not_match_the_version_is_refused(tmp_path):
    _dist(tmp_path, "rl-core", version="0.1.1")
    with pytest.raises(release.ReleaseError, match="does not match rl-core's version 0.1.1"):
        release.check("rl-core-v0.1.0", [], tmp_path)


def test_a_tag_for_no_distribution_is_refused(tmp_path):
    with pytest.raises(release.ReleaseError, match="no distribution 'rl-missing'"):
        release.check("rl-missing-v0.1.0", [], tmp_path)


def test_a_version_without_a_changelog_section_is_refused(tmp_path):
    root = _dist(tmp_path, "rl-core")
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n", encoding="utf-8")
    with pytest.raises(release.ReleaseError, match=r"no '## \[0.1.0\]' section"):
        release.check("rl-core-v0.1.0", [], tmp_path)


def test_an_empty_changelog_section_is_refused():
    with pytest.raises(release.ReleaseError, match="is empty"):
        release.changelog_section("## [0.1.0] - 2026-11-11\n\n## [0.0.9]\n- x\n", Version("0.1.0"))


def test_a_dependent_needs_its_dependency_released_first(tmp_path):
    _dist(tmp_path, "rl-core")
    _dist(tmp_path, "rl-challenges", dependencies=["rl-core~=0.1.0"])
    with pytest.raises(release.ReleaseError, match="has no tag: push rl-core-v0.1.0"):
        release.check("rl-challenges-v0.1.0", ["rl-challenges-v0.1.0"], tmp_path)
    release.check("rl-challenges-v0.1.0", ["rl-core-v0.1.0", "rl-challenges-v0.1.0"], tmp_path)


def test_a_requirement_in_an_extra_counts_too(tmp_path):
    _dist(tmp_path, "rl-core")
    _dist(tmp_path, "rl-navigation", extras={"challenge": ["rl-core~=0.1.0"]})
    with pytest.raises(release.ReleaseError, match="has no tag: push rl-core-v0.1.0"):
        release.check("rl-navigation-v0.1.0", [], tmp_path)
    assert release.dependency_tags("rl-navigation", tmp_path) == [], (
        "a plain install of rl-navigation pulls in no in-repo distribution"
    )


def test_a_requirement_that_excludes_the_in_tree_dependency_is_refused(tmp_path):
    """A breaking rl-core bump (0.2.0) with a dependent still on ~=0.1.0."""
    _dist(tmp_path, "rl-core", version="0.2.0")
    _dist(tmp_path, "rl-challenges", dependencies=["rl-core~=0.1.0"])
    with pytest.raises(release.ReleaseError, match="does not admit rl-core's version 0.2.0"):
        release.check("rl-challenges-v0.1.0", ["rl-core-v0.2.0"], tmp_path)


def test_a_dynamic_version_is_read_from_its_attribute(tmp_path):
    root = tmp_path / "rl-thing"
    (root / "src" / "rl_thing").mkdir(parents=True)
    (root / "src" / "rl_thing" / "__init__.py").write_text(
        '"""Doc."""\n\n__version__ = "0.3.1"\n', encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "rl-thing"\ndynamic = ["version"]\n'
        '[tool.setuptools.dynamic]\nversion = {attr = "rl_thing.__version__"}\n',
        encoding="utf-8",
    )
    assert release.project_version("rl-thing", tmp_path) == Version("0.3.1")


def test_the_real_distributions_are_releasable_once_their_dependencies_are_tagged():
    tags = [
        f"{path.parent.name}-v{release.project_version(path.parent.name)}"
        for path in PACKAGES.glob("*/pyproject.toml")
    ]
    for tag in tags:
        release.check(tag, tags)


# ---------------------------------------------------------------- smoke and sums


def test_only_the_promised_footprint_passes():
    packages = ["rl_navigation"]
    assert release.footprint_violations("rl-navigation", ["numpy", "rl_navigation"], packages) == []
    assert release.footprint_violations(
        "rl-navigation", ["numpy", "rl_navigation", "gymnasium"], packages
    ) == ["gymnasium"]
    assert release.footprint_violations("rl-core", ["gymnasium", "yaml"], ["rl_core"]) == []


# Import name -> distribution name, for the base requirements of the real distributions.
IMPORT_DISTS = {
    "numpy": "numpy",
    "gymnasium": "gymnasium",
    "yaml": "pyyaml",
    "matplotlib": "matplotlib",
    "rl_core": "rl-core",
}


def _module_level_imports(path):
    """Top-level names a module imports when it is imported (not inside a function)."""
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        for inner in ast.walk(node) if isinstance(node, ast.If | ast.Try) else [node]:
            if isinstance(inner, ast.Import):
                names.update(alias.name.split(".")[0] for alias in inner.names)
            elif isinstance(inner, ast.ImportFrom) and inner.level == 0 and inner.module:
                names.add(inner.module.split(".")[0])
    return names


def _module_name(src, path):
    parts = path.relative_to(src).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@pytest.mark.parametrize("dist", sorted(release.EXTRA_ONLY))
def test_the_extra_only_modules_are_exactly_those_that_import_an_extra(dist):
    """``smoke`` skips the modules that need an extra: a module importing anything beyond the
    distribution's base requirements must be listed, and each listed prefix must match one."""
    src = PACKAGES / dist / "src"
    base = {Requirement(spec).name for spec in release.project(dist)["dependencies"]}
    ours = set(release.import_packages(dist))
    stdlib = set(sys.stdlib_module_names)
    needs_extra = set()
    for path in src.rglob("*.py"):
        third_party = _module_level_imports(path) - stdlib - ours - {"__future__"}
        unknown = third_party - set(IMPORT_DISTS) - {"torch"}
        assert not unknown, f"{path}: add {unknown} to IMPORT_DISTS"
        if {IMPORT_DISTS.get(name, name) for name in third_party} - base:
            needs_extra.add(_module_name(src, path))
    skipped = release.EXTRA_ONLY[dist]
    assert all(name.startswith(skipped) for name in needs_extra), (
        f"{sorted(n for n in needs_extra if not n.startswith(skipped))} need an extra: "
        "add them to EXTRA_ONLY"
    )
    modules = {_module_name(src, path) for path in src.rglob("*.py")}
    assert all(any(m.startswith(prefix) for m in modules) for prefix in skipped), "a stale prefix"


def test_the_checksums_are_sha256sum_format(tmp_path):
    (tmp_path / "a.whl").write_bytes(b"abc")
    (tmp_path / "b.tar.gz").write_bytes(b"")
    target = release.write_sums(tmp_path)
    assert target.read_text(encoding="utf-8").splitlines() == [
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  a.whl",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  b.tar.gz",
    ]


def test_the_command_line_reports_a_failed_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(release, "git_tags", lambda: [])
    assert release.main(["check", "not-a-tag"]) == 1
    assert "not <distribution>-v<version>" in capsys.readouterr().err


def test_each_wheel_is_found_by_its_distribution_name(tmp_path):
    (tmp_path / "rl_core-0.1.0-py3-none-any.whl").write_bytes(b"")
    assert release.wheel_of("rl-core", tmp_path).name == "rl_core-0.1.0-py3-none-any.whl"
    with pytest.raises(release.ReleaseError, match="expected one rl-navigation wheel"):
        release.wheel_of("rl-navigation", tmp_path)


def test_smoke_all_gives_each_distribution_its_dependencies_wheels(tmp_path, monkeypatch):
    packages = tmp_path / "packages"
    _dist(packages, "rl-core")
    _dist(packages, "rl-challenges", dependencies=["rl-core~=0.1.0"])
    _dist(packages, "rl-navigation", extras={"challenge": ["rl-core~=0.1.0"]})
    wheels = tmp_path / "dist"
    wheels.mkdir()
    for name in ("rl_core", "rl_challenges", "rl_navigation"):
        (wheels / f"{name}-0.1.0-py3-none-any.whl").write_bytes(b"")
    calls = {}
    monkeypatch.setattr(release, "smoke", lambda dist, files, _: calls.setdefault(dist, files))
    release.smoke_all(wheels, packages)
    assert [w.name for w in calls["rl-challenges"]] == [
        "rl_challenges-0.1.0-py3-none-any.whl",
        "rl_core-0.1.0-py3-none-any.whl",
    ]
    assert [w.name for w in calls["rl-core"]] == ["rl_core-0.1.0-py3-none-any.whl"]
    assert [w.name for w in calls["rl-navigation"]] == ["rl_navigation-0.1.0-py3-none-any.whl"], (
        "an extra's requirement is not installed: the smoke test is of a plain install"
    )


def test_a_distribution_without_a_changelog_is_refused_cleanly(tmp_path):
    root = _dist(tmp_path, "rl-core")
    (root / "CHANGELOG.md").unlink()
    with pytest.raises(release.ReleaseError, match="rl-core has no CHANGELOG.md"):
        release.check("rl-core-v0.1.0", [], tmp_path)


def test_link_references_at_the_end_of_a_changelog_stay_out_of_the_notes():
    text = (
        "## [0.1.0] - 2026-11-11\n\n- First.\n\n"
        "[0.1.0]: https://github.com/andreijw/AI/releases/tag/rl-core-v0.1.0\n"
    )
    assert release.changelog_section(text, Version("0.1.0")) == "- First.\n"


PUBLISH = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "release.yml").read_text())


def _run_of(job, name):
    (step,) = [s for s in PUBLISH["jobs"][job]["steps"] if s.get("name") == name]
    return step["run"]


def test_a_rerun_refreshes_the_draft_and_never_touches_a_published_release():
    """A re-run after a fix replaces the draft's notes, title, prerelease flag and files (removing
    files this run did not build); a published release is refused. Read from the workflow, since
    only a tag run can exercise gh itself."""
    run = _run_of("publish", "Create or refresh the draft release")
    assert "isDraft" in run and "already published" in run
    assert 'gh release edit "$TAG" --title "$TAG" --notes-file notes.md' in run
    assert "delete-asset" in run and "--clobber" in run
    assert "release not found" in run, "only a missing release is created; other gh errors stop"


def test_a_dependency_lookup_tells_missing_from_failing():
    run = _run_of("build", "Download the released dependencies")
    assert "release not found" in run and "has no published release" in run
    assert "could not look up" in run, "a network or auth failure prints gh's own error"


# A stand-in for gh: logs its arguments, answers `release view` as GH_STATE says.
STUB_GH = """#!/usr/bin/env bash
echo "gh $*" >> "$GH_LOG"
if [[ "$1 $2" == "release view" ]]; then
  case "$GH_STATE" in
    draft) if [[ "$*" == *assets* ]]; then echo stale.whl; else echo true; fi ;;
    draft-assets-fail) if [[ "$*" == *assets* ]]; then echo "HTTP 502" >&2; exit 1; fi; echo true ;;
    published) echo false ;;
    missing) echo "release not found" >&2; exit 1 ;;
    broken) echo "HTTP 502: Bad Gateway" >&2; exit 1 ;;
  esac
fi
"""


def _publish(tmp_path, state):
    """Run the publish step's script with a stub gh; return (exit code, gh calls)."""
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(STUB_GH, encoding="utf-8")
    (bin_dir / "gh").chmod(0o755)
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "rl_core-0.1.0-py3-none-any.whl").write_bytes(b"")
    (tmp_path / "notes.md").write_text("- First.\n", encoding="utf-8")
    log = tmp_path / "gh.log"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "GH_STATE": state,
        "GH_LOG": str(log),
        "TAG": "rl-core-v0.1.0",
        "PRERELEASE": "false",
    }
    script = _run_of("publish", "Create or refresh the draft release")
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script], cwd=tmp_path, env=env, check=False
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result.returncode, calls


def test_publish_refreshes_a_draft_and_removes_files_this_run_did_not_build(tmp_path):
    code, calls = _publish(tmp_path, "draft")
    assert code == 0
    assert any(c.startswith("gh release edit rl-core-v0.1.0 --title") for c in calls)
    assert "gh release delete-asset rl-core-v0.1.0 stale.whl --yes" in calls
    assert any(c.startswith("gh release upload") and "--clobber" in c for c in calls)


@pytest.mark.parametrize("state", ["published", "broken", "draft-assets-fail"])
def test_publish_stops_without_touching_the_release(tmp_path, state):
    """A published release is never changed; a gh failure, even listing the draft's files,
    stops the step before anything is uploaded."""
    code, calls = _publish(tmp_path, state)
    assert code != 0
    assert not any(c.startswith(("gh release upload", "gh release create")) for c in calls)


def test_publish_creates_a_draft_only_when_none_exists(tmp_path):
    code, calls = _publish(tmp_path, "missing")
    assert code == 0
    assert any(c.startswith("gh release create rl-core-v0.1.0") and "--draft" in c for c in calls)

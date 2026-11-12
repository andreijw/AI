"""Every version is defined once.

- Third-party packages: pinned exactly in ``requirements-lock.txt``, which every development, CI and
  release install uses as constraints. The distributions' ``pyproject.toml`` files hold floors, one per
  package across all three, so a robot or a game installing a release is not tied to these pins.
- This repository's distributions: ``__version__`` in one module each (``tests/test_packages.py``).
- Python: ``.python-version``. markdownlint-cli: ``MARKDOWNLINT_CLI_VERSION`` in ``ci.yml``.
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK = REPO_ROOT / "requirements-lock.txt"
PYPROJECTS = sorted((REPO_ROOT / "packages").glob("*/pyproject.toml"))
WORKFLOWS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
OURS = {canonicalize_name(path.parent.name) for path in PYPROJECTS}
# Outside the lock on purpose: PyTorch comes from PyTorch's CUDA index (./setup.sh --profile gpu).
UNLOCKED = {"torch"}


def _lock():
    pins = {}
    for number, line in enumerate(LOCK.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        assert requirement.marker is None or "sys_platform" in str(requirement.marker), line
        (spec,) = requirement.specifier
        assert spec.operator == "==" and not requirement.extras, f"lock line {number}: {line}"
        name = canonicalize_name(requirement.name)
        assert name not in pins, f"lock line {number}: {name} is pinned twice"
        pins[name] = Version(spec.version)
    return pins


def _pyproject_requirements():
    """(pyproject path, Requirement) for every requirement the distributions declare, build ones too."""
    found = []
    for path in PYPROJECTS:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        specs = list(data["build-system"]["requires"]) + list(
            data["project"].get("dependencies", [])
        )
        for extra in data["project"].get("optional-dependencies", {}).values():
            specs.extend(extra)
        found.extend((path, Requirement(spec)) for spec in specs)
    return found


def _third_party(requirements):
    return [(p, r) for p, r in requirements if canonicalize_name(r.name) not in OURS]


def test_the_lock_pins_each_package_once_and_exactly():
    assert len(_lock()) > 50, "the lock holds the whole development environment"


def test_each_third_party_requirement_has_one_floor_across_the_distributions():
    floors = {}
    for path, requirement in _third_party(_pyproject_requirements()):
        floors.setdefault(canonicalize_name(requirement.name), {})[str(requirement.specifier)] = (
            path
        )
    disagreeing = {name: sorted(specs) for name, specs in floors.items() if len(specs) > 1}
    assert disagreeing == {}, (
        f"one floor per package across packages/*/pyproject.toml: {disagreeing}"
    )
    assert all(
        not any(spec.operator == "==" for spec in r.specifier)
        for _, r in _third_party(_pyproject_requirements())
    ), "a distribution holds floors, never an exact pin (the lock pins)"


def test_every_lock_pin_satisfies_every_floor():
    pins = _lock()
    for path, requirement in _third_party(_pyproject_requirements()):
        name = canonicalize_name(requirement.name)
        if name in UNLOCKED:
            continue
        assert name in pins, f"{path.parent.name} requires {name}, which the lock does not pin"
        assert requirement.specifier.contains(pins[name], prereleases=True), (
            f"{path.parent.name} requires {requirement}, but the lock pins {name}=={pins[name]}"
        )


def test_the_development_requirements_name_no_version_and_take_the_lock():
    """Its `-c requirements-lock.txt` line makes every `pip install -r requirements-dev.txt` pinned,
    extras on the same command line included."""
    lines = [
        line.strip()
        for line in (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "-c requirements-lock.txt", "the lock is the file's first requirement line"
    for line in lines[1:]:
        if line.startswith("-e "):
            continue
        assert not Requirement(line).specifier, f"requirements-dev.txt: {line} (the lock pins it)"
        assert canonicalize_name(Requirement(line).name) in _lock(), f"{line} has no lock pin"


def _install_commands(text):
    """Each `pip install` command in shell text, continuation lines joined."""
    joined = re.sub(r"\\\n\s*", " ", text)
    return [
        line
        for line in joined.splitlines()
        if "pip install" in line and not line.strip().startswith("#")
    ]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_ci_install_uses_the_lock(path):
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            for command in _install_commands(step.get("run", "")):
                assert any(
                    lock in command
                    for lock in (
                        "requirements-lock.txt",
                        "-r requirements-dev.txt",
                        "lock-without-numpy",
                    )
                ), (
                    f"{path.name}: {command.strip()} installs without the lock (-c requirements-lock.txt)"
                )


def test_every_setup_install_uses_the_lock():
    for command in _install_commands((REPO_ROOT / "setup.sh").read_text(encoding="utf-8")):
        assert '"${lock[@]}"' in command or "-r requirements-dev.txt" in command, command.strip()
    ps1 = (REPO_ROOT / "setup.ps1").read_text(encoding="utf-8")
    installs = [
        line for line in ps1.splitlines() if re.search(r'-m pip install|"pip", "install"', line)
    ]
    assert installs and all("LockArgs" in line for line in installs), installs


# A requirement written with a version: name==1.2, name>=1, name~=0.1 (no spaces, as pip reads them).
_PINNED = re.compile(r"(?<![\w.\-/])([A-Za-z][\w.\-]*(?:\[[\w,\-]+\])?)(==|~=|>=|<=|<|>)\d")
# Files that install or check software, and the only versions each may write.
_ALLOWED = {
    # Gymnasium's own box2d pin, named to say why the box2d extra does not use it.
    "setup.sh": {"box2d==2"},
    "packages/rl-core/src/rl_core/environments/base_env.py": {"box2d==2"},
}


def _version_files():
    paths = [
        REPO_ROOT / "setup.sh",
        REPO_ROOT / "setup.ps1",
        REPO_ROOT / ".pre-commit-config.yaml",
        REPO_ROOT / "requirements-dev.txt",
        *WORKFLOWS,
        *(REPO_ROOT / "scripts").glob("*.py"),
        *(REPO_ROOT / "scripts").glob("*.sh"),
        *(REPO_ROOT / "packages").glob("*/src/**/*.py"),
        *REPO_ROOT.glob("*.py"),
    ]
    return sorted(path for path in paths if path.is_file())


# A tool fetched by version on a command line (npx markdownlint-cli@0.39.0), not an action's `uses:` pin.
_AT_VERSION = re.compile(r"(?<![\w.\-/])([a-z][\w.\-]*)@\d+\.\d+")


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_workflow_fetches_a_tool_by_a_written_version(path):
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            found = _AT_VERSION.findall(step.get("run", ""))
            assert not found, (
                f"{path.name}: {found}: name the version in an env (MARKDOWNLINT_CLI_VERSION)"
            )


def test_no_version_is_written_outside_the_lock_and_the_floors():
    """A second definition of a version cannot creep into a script, a workflow or the code."""
    found = []
    for path in _version_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        for match in _PINNED.finditer(path.read_text(encoding="utf-8")):
            if f"{match[1]}{match[2]}{match[0][-1]}" not in _ALLOWED.get(relative, set()):
                found.append(f"{relative}: {match[0]}")
    assert found == [], f"versions belong in requirements-lock.txt or a pyproject floor: {found}"


def _markdownlint_version():
    ci = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    return str(ci["env"]["MARKDOWNLINT_CLI_VERSION"])


def test_markdownlint_is_one_version():
    version = _markdownlint_version()
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    (repo,) = [r for r in config["repos"] if r["repo"].endswith("/markdownlint-cli")]
    assert repo["rev"] == f"v{version}", (
        "pre-commit's markdownlint rev is ci.yml's MARKDOWNLINT_CLI_VERSION"
    )
    docs = [
        *(REPO_ROOT / ".claude" / "skills").glob("*/SKILL.md"),
        *(REPO_ROOT / "docs").glob("*.md"),
        *[
            REPO_ROOT / name
            for name in ("CONTRIBUTING.md", "GEMINI.md", "CLAUDE.md", "README.md", "TESTING.md")
        ],
    ]
    for path in sorted(docs):
        text = path.read_text(encoding="utf-8")
        named = set(re.findall(r"markdownlint-cli@([\d.]+)", text))
        named |= set(re.findall(r"`markdownlint-cli` ([\d.]+)", text))
        assert named <= {version}, f"{path.relative_to(REPO_ROOT)} names markdownlint-cli {named}"


def _python_version():
    return (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()


def test_python_is_one_version():
    version = _python_version()
    assert re.fullmatch(r"3\.\d+", version), ".python-version holds major.minor"
    for path in PYPROJECTS:
        with open(path, "rb") as f:
            project = tomllib.load(f)["project"]
        assert project["requires-python"] == f">={version}", path
        assert f"Programming Language :: Python :: {version}" in project["classifiers"], path
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        tool = tomllib.load(f)["tool"]
    assert tool["mypy"]["python_version"] == version
    assert tool["ruff"]["target-version"] == "py" + version.replace(".", "")
    for path in WORKFLOWS:
        for job in yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs", {}).values():
            for step in job.get("steps", []):
                if str(step.get("uses", "")).startswith("actions/setup-python@"):
                    assert step["with"].get("python-version-file") == ".python-version", path.name
                    assert "python-version" not in step["with"], path.name
    for name in ("setup.sh", "setup.ps1"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert ".python-version" in text and version not in text, f"{name} reads .python-version"


def _development_roots():
    """(distribution, extras) the dev install starts from: requirements-dev.txt's lines."""
    roots = []
    for line in (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-c ")):
            continue
        if line.startswith("-e "):
            path, _, extras = line.split()[1].partition("[")
            roots.append((Path(path).name, set(filter(None, extras.rstrip("]").split(",")))))
        else:
            requirement = Requirement(line)
            roots.append((requirement.name, set(requirement.extras)))
    return roots


def test_every_package_the_dev_install_pulls_in_is_locked():
    """Walk the installed metadata from requirements-dev.txt (as CI installs it): a dependency that is
    installed but not pinned would float. Offline; packages of extras CI does not install (video,
    box2d) and Windows-only ones are not reachable here, so stale entries are not detected."""
    from importlib.metadata import PackageNotFoundError, distribution

    pins = _lock()
    seen, missing = set(), set()
    todo = _development_roots()
    while todo:
        name, extras = todo.pop()
        key = (canonicalize_name(name), frozenset(extras))
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            continue
        if canonicalize_name(name) not in OURS | UNLOCKED and canonicalize_name(name) not in pins:
            missing.add(canonicalize_name(name))
        for spec in dist.requires or []:
            requirement = Requirement(spec)
            environments = [{"extra": extra} for extra in extras] or [{"extra": ""}]
            if requirement.marker is None or any(
                requirement.marker.evaluate(env) for env in environments
            ):
                todo.append((requirement.name, set(requirement.extras)))
    assert missing == set(), f"installed by the dev install but not pinned: {sorted(missing)}"

"""The distributions under packages/: what each must have, so a new library is a folder copied and checked.

A new distribution (a voice library, another R2D2 part) is started from an existing folder under packages/;
these tests then name whatever it still lacks.
"""

import ast
import subprocess
import sys
import tomllib
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
DISTS = sorted(path.parent.name for path in (REPO_ROOT / "packages").glob("*/pyproject.toml"))


def _pyproject(dist):
    with open(REPO_ROOT / "packages" / dist / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _in_tree_version(dist):
    data = _pyproject(dist)
    if "version" in data["project"]:
        return Version(data["project"]["version"])
    attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    module, _, name = attr.rpartition(".")
    path = REPO_ROOT / "packages" / dist / "src" / Path(*module.split(".")) / "__init__.py"
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if isinstance(node, ast.Assign | ast.AnnAssign) and any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            return Version(ast.literal_eval(node.value))
    raise AssertionError(f"{path} does not set {name}")


def test_the_three_distributions_are_found():
    """A new distribution adds its name here, so a folder missing its pyproject.toml is noticed."""
    assert DISTS == ["rl-challenges", "rl-core", "rl-navigation"]


@pytest.mark.parametrize("dist", DISTS)
def test_a_distribution_is_named_after_its_folder(dist):
    assert _pyproject(dist)["project"]["name"] == dist


@pytest.mark.parametrize("dist", DISTS)
def test_a_distribution_has_its_readme_changelog_and_license(dist):
    root = REPO_ROOT / "packages" / dist
    project = _pyproject(dist)["project"]
    assert project["readme"] == "README.md" and (root / "README.md").is_file()
    assert project["license"] == "MIT" and project["license-files"] == ["LICENSE"]
    assert (root / "LICENSE").read_bytes() == (REPO_ROOT / "LICENSE").read_bytes(), (
        "an sdist must hold its own LICENSE: copy the repository's"
    )
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{_in_tree_version(dist)}]" in changelog, (
        "CHANGELOG.md needs a section for the current version: a release's notes are that section"
    )


@pytest.mark.parametrize("dist", DISTS)
def test_a_distribution_does_not_configure_ruff(dist):
    """Ruff uses the closest pyproject.toml with a [tool.ruff] table and merges nothing, so one here
    would replace the repository's settings for the whole package."""
    assert "ruff" not in _pyproject(dist).get("tool", {})


@pytest.mark.parametrize("dist", DISTS)
def test_the_installed_metadata_has_the_in_tree_version(dist):
    """A version bump needs the editable install re-run (metadata is written at install time)."""
    assert Version(installed_version(dist)) == _in_tree_version(dist), (
        f"re-run the editable install of {dist} (pip install -r requirements-dev.txt)"
    )


def test_the_development_requirements_install_every_distribution():
    lines = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    installed = {
        line.split()[1].split("[")[0].removeprefix("./packages/")
        for line in lines
        if line.startswith("-e ./packages/")
    }
    assert installed == set(DISTS)


def test_no_two_distributions_ship_one_import_package():
    owners: dict[str, list[str]] = {}
    for dist in DISTS:
        for init in (REPO_ROOT / "packages" / dist / "src").glob("*/__init__.py"):
            owners.setdefault(init.parent.name, []).append(dist)
    assert {name: dists for name, dists in owners.items() if len(dists) > 1} == {}


@pytest.mark.parametrize("dist", DISTS)
def test_a_requirement_on_another_distribution_admits_its_in_tree_version(dist):
    """A breaking bump of rl-core (0.1 to 0.2) must move its dependents' ~=0.1.0 requirements too."""
    project = _pyproject(dist)["project"]
    specs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)
    for requirement in map(Requirement, specs):
        name = canonicalize_name(requirement.name)
        if name in DISTS:
            assert requirement.specifier.contains(_in_tree_version(name), prereleases=True), (
                f"{dist} requires {requirement}, which excludes {name} {_in_tree_version(name)}"
            )


def test_no_tracked_file_is_left_under_src():
    """The code moved to packages/<dist>/src/ (2026-11-11); a file under src/ is in no distribution."""
    tracked = subprocess.run(
        ["git", "ls-files", "src"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert tracked == []


# Import name -> distribution name, for every third-party import under packages/. A new import
# fails test_every_import_is_a_declared_requirement until it is listed here and declared.
IMPORT_DISTS = {
    "gymnasium": "gymnasium",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "rl_core": "rl-core",
    "torch": "torch",
    "yaml": "pyyaml",
}

# The modules of rl-navigation a robot does not need, each with the extra that provides what it
# imports beyond NumPy; every other module of it imports NumPy only.
NAVIGATION_EXTRA_MODULES = {
    "rl_navigation.benchmark": "challenge",
    "rl_navigation.environments": "challenge",
    "rl_navigation.gym_env": "gym",
    "rl_navigation.localization_video": "video",
}


def _third_party_imports(dist):
    """{module name: third-party distributions it imports, anywhere in the file}."""
    src = REPO_ROOT / "packages" / dist / "src"
    ours = {init.parent.name for init in src.glob("*/__init__.py")}
    found = {}
    for path in sorted(src.rglob("*.py")):
        parts = path.relative_to(src).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        names = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        names -= ours | set(sys.stdlib_module_names) | {"__future__"}
        unknown = names - set(IMPORT_DISTS)
        assert not unknown, f"{path}: add {sorted(unknown)} to IMPORT_DISTS (and declare it)"
        found[module] = {IMPORT_DISTS[name] for name in names}
    return found


def _requirement_names(specs):
    return {canonicalize_name(Requirement(spec).name) for spec in specs}


@pytest.mark.parametrize("dist", DISTS)
def test_every_import_is_a_declared_requirement(dist):
    """What a module imports, at module level or lazily, is a requirement or in an extra."""
    project = _pyproject(dist)["project"]
    declared = _requirement_names(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared |= _requirement_names(extra)
    for module, imported in _third_party_imports(dist).items():
        assert imported <= declared, f"{module} imports {sorted(imported - declared)}, undeclared"


def test_the_robot_side_of_rl_navigation_imports_numpy_only():
    """A robot installs rl-navigation alone: outside the modules its extras serve, NumPy only."""
    extras = _pyproject("rl-navigation")["project"]["optional-dependencies"]
    for module, imported in _third_party_imports("rl-navigation").items():
        owner = next(
            (
                name
                for name in NAVIGATION_EXTRA_MODULES
                if module == name or module.startswith(name + ".")
            ),
            None,
        )
        if owner is None:
            assert imported <= {"numpy"}, (
                f"{module} imports {sorted(imported)}: the robot gets NumPy only"
            )
        else:
            allowed = {"numpy"} | _requirement_names(extras[NAVIGATION_EXTRA_MODULES[owner]])
            if NAVIGATION_EXTRA_MODULES[owner] == "challenge":
                allowed |= _requirement_names(
                    extras["gym"]
                )  # the challenge runs the gym environment
            assert imported <= allowed, f"{module} imports {sorted(imported - allowed)}"


@pytest.mark.parametrize("dist", DISTS)
def test_every_import_package_is_marked_typed(dist):
    """PEP 561: a project type-checking against the distribution sees its annotations."""
    src = REPO_ROOT / "packages" / dist / "src"
    assert all((init.parent / "py.typed").is_file() for init in src.glob("*/__init__.py"))
    package_data = _pyproject(dist)["tool"]["setuptools"]["package-data"]
    assert "py.typed" in package_data.get("*", []), "package-data must ship py.typed in the wheel"

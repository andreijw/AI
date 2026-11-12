#!/usr/bin/env python3
"""Check, smoke-test and checksum a release of one distribution under packages/.

A release is a pushed tag ``<dist>-v<version>`` (``rl-navigation-v0.1.0``); .github/workflows/release.yml
runs these steps for it, and the ``Packages`` CI job runs ``smoke-all`` on every pull request.

    python scripts/release.py check <tag> [--notes notes.md] [--github-output FILE]
        The tag names a distribution under packages/ whose version it equals, the package's CHANGELOG.md
        has a section for that version (written to --notes), and every in-repo distribution it requires is
        already released (its in-tree version tagged) and admitted by the requirement.
    python scripts/release.py dependency-tags <dist>
        The tags of the in-repo distributions <dist> requires, one a line (for `gh release download`).
    python scripts/release.py smoke <dist> <wheel> [<dependency wheel> ...]
        Install the wheels in a fresh virtual environment (dependencies from PyPI) and import every module of
        the distribution except those its extras need; for rl-navigation, assert only NumPy is loaded.
    python scripts/release.py smoke-all <wheel dir>
        ``smoke`` for every distribution under packages/, each with the wheels of the in-repo
        distributions it requires from the same directory (the CI job, on code not yet released).
    python scripts/release.py sums <dir>
        Write <dir>/SHA256SUMS for the files in <dir>.

Exit code: 0 on success, 1 when a check fails, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess  # nosec B404
import sys
import tempfile
import tomllib
import venv
from collections.abc import Iterable
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = REPO_ROOT / "packages"
LINK_REFERENCE = re.compile(r"^\[[^\]]+\]:\s")
TAG = re.compile(r"^(?P<dist>[a-z0-9][a-z0-9-]*)-v(?P<version>[0-9][0-9a-z.]*)$")

#: Modules that need one of the distribution's extras, by name prefix: ``smoke`` does not import them
#: from a plain install. tests/test_release.py checks the list against the modules' own imports.
EXTRA_ONLY: dict[str, tuple[str, ...]] = {
    "rl-core": ("rl_core.agents.torch_",),
    "rl-navigation": (
        "rl_navigation.benchmark",
        "rl_navigation.environments",
        "rl_navigation.gym_env",
    ),
    "rl-challenges": (),
}

#: Third-party top-level modules a plain install may load, where the footprint is a promise.
FOOTPRINT: dict[str, frozenset[str]] = {"rl-navigation": frozenset({"numpy"})}


class ReleaseError(Exception):
    """A release check that failed, with the reason to print."""


def parse_tag(tag: str) -> tuple[str, Version]:
    """``("rl-navigation", Version("0.1.0"))`` for ``rl-navigation-v0.1.0``."""
    match = TAG.fullmatch(tag)
    if match is None:
        raise ReleaseError(f"tag {tag!r} is not <distribution>-v<version>")
    try:
        version = Version(match["version"])
    except InvalidVersion as error:
        raise ReleaseError(f"tag {tag!r}: {error}") from None
    if str(version) != match["version"]:
        raise ReleaseError(f"tag {tag!r}: write the version as {version}")
    return match["dist"], version


def project(dist: str, packages: Path = PACKAGES) -> dict:
    """The ``[project]`` table of ``packages/<dist>/pyproject.toml``."""
    path = packages / dist / "pyproject.toml"
    if not path.is_file():
        raise ReleaseError(f"no distribution {dist!r}: {path} does not exist")
    with open(path, "rb") as f:
        return tomllib.load(f)["project"]


def project_version(dist: str, packages: Path = PACKAGES) -> Version:
    """The version in the pyproject, or the ``__version__`` its ``version = {attr}`` names."""
    table = project(dist, packages)
    if "version" in table:
        return Version(table["version"])
    with open(packages / dist / "pyproject.toml", "rb") as f:
        attr = tomllib.load(f)["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    module, _, name = attr.rpartition(".")
    source = packages / dist / "src" / Path(*module.split("."))
    path = source / "__init__.py" if source.is_dir() else source.with_suffix(".py")
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return Version(ast.literal_eval(node.value))
    raise ReleaseError(f"{path} does not set {name}")


def changelog_section(text: str, version: Version) -> str:
    """The body of the ``## [<version>]`` section of a Keep a Changelog file, without the link
    references (``[0.1.0]: https://...``) such a file may keep at its end."""
    lines = text.splitlines()
    heading = re.compile(rf"^## \[{re.escape(str(version))}\]")
    starts = [i for i, line in enumerate(lines) if heading.match(line)]
    if not starts:
        raise ReleaseError(f"CHANGELOG.md has no '## [{version}]' section")
    body = []
    for line in lines[starts[0] + 1 :]:
        if line.startswith("## "):
            break
        if not LINK_REFERENCE.match(line):
            body.append(line)
    notes = "\n".join(body).strip()
    if not notes:
        raise ReleaseError(f"CHANGELOG.md's '## [{version}]' section is empty")
    return notes + "\n"


def in_repo_requirements(
    dist: str, packages: Path = PACKAGES, extras: bool = True
) -> list[Requirement]:
    """The requirements of ``dist`` on other distributions in packages/: with ``extras``, its
    extras' too (what must be released first), without, only what a plain install pulls in."""
    ours = {canonicalize_name(path.parent.name) for path in packages.glob("*/pyproject.toml")}
    table = project(dist, packages)
    specs = list(table.get("dependencies", []))
    if extras:
        for extra in table.get("optional-dependencies", {}).values():
            specs.extend(extra)
    return [req for req in map(Requirement, specs) if canonicalize_name(req.name) in ours]


def dependency_tags(dist: str, packages: Path = PACKAGES) -> list[str]:
    """``<dependency>-v<in-tree version>`` for each in-repo distribution a plain install of ``dist``
    requires: the releases whose wheels ``smoke`` installs beside it."""
    return sorted(
        {
            f"{canonicalize_name(req.name)}-v{project_version(canonicalize_name(req.name), packages)}"
            for req in in_repo_requirements(dist, packages, extras=False)
        }
    )


def check_dependencies(dist: str, tags: Iterable[str], packages: Path = PACKAGES) -> None:
    """Every in-repo requirement is released (its in-tree version tagged) and admits that version."""
    released = set(tags)
    for req in in_repo_requirements(dist, packages):
        name = canonicalize_name(req.name)
        version = project_version(name, packages)
        if not req.specifier.contains(version, prereleases=True):
            raise ReleaseError(
                f"{dist} requires {req}, which does not admit {name}'s version {version}: "
                "update the requirement (docs/releasing.md, A breaking rl-core change)"
            )
        if f"{name}-v{version}" not in released:
            raise ReleaseError(
                f"{dist} requires {name} {version}, which has no tag: push {name}-v{version} and "
                "publish its draft first (docs/releasing.md, Order)"
            )


def check(tag: str, tags: Iterable[str], packages: Path = PACKAGES) -> tuple[str, Version, str]:
    """``(dist, version, notes)`` for a releasable tag; raises :class:`ReleaseError` otherwise."""
    dist, version = parse_tag(tag)
    in_tree = project_version(dist, packages)
    # Version("0.1") == Version("0.1.0"): the tag must spell the version out.
    if str(in_tree) != str(version):
        raise ReleaseError(f"tag {tag} does not match {dist}'s version {in_tree}")
    changelog = packages / dist / "CHANGELOG.md"
    if not changelog.is_file():
        raise ReleaseError(f"{dist} has no CHANGELOG.md: its release notes come from it")
    notes = changelog_section(changelog.read_text(encoding="utf-8"), version)
    check_dependencies(dist, tags, packages)
    return dist, version, notes


def github_outputs(dist: str, version: Version) -> str:
    """The step outputs the Release workflow reads: the distribution, its version and whether the
    version is a PEP 440 prerelease (``a``, ``b``, ``rc`` or ``.dev``)."""
    return f"dist={dist}\nversion={version}\nprerelease={str(version.is_prerelease).lower()}\n"


def git_tags() -> list[str]:
    """The repository's tags (`git tag --list`)."""
    result = subprocess.run(  # nosec B603 B607
        ["git", "tag", "--list"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.split()


def import_packages(dist: str, packages: Path = PACKAGES) -> list[str]:
    """The top-level import packages ``dist`` ships (the directories under its src/)."""
    src = packages / dist / "src"
    return sorted(path.parent.name for path in src.glob("*/__init__.py"))


# Run in the fresh environment: import every module of the packages except the skipped prefixes, then
# print the loaded third-party top-level modules as JSON.
_SMOKE_CHILD = """
import importlib, importlib.resources, json, pkgutil, sys
packages, skipped = json.loads(sys.argv[1]), tuple(json.loads(sys.argv[2]))
imported = []
def visit(name):
    # Not pkgutil.walk_packages: it imports every subpackage to walk it, the skipped ones too.
    module = importlib.import_module(name)
    imported.append(name)
    for info in pkgutil.iter_modules(getattr(module, "__path__", []), name + "."):
        if not info.name.startswith(skipped):
            visit(info.name)
for name in packages:
    visit(name)
stdlib = set(sys.stdlib_module_names) | set(sys.builtin_module_names)
loaded = sorted({m.split(".")[0] for m in sys.modules} - stdlib - {"__main__"})
untyped = [name for name in packages if not (importlib.resources.files(name) / "py.typed").is_file()]
print(json.dumps({"imported": imported, "loaded": loaded, "untyped": untyped}))
"""


def footprint_violations(dist: str, loaded: Iterable[str], packages: list[str]) -> list[str]:
    """Top-level modules outside ``dist``'s promised footprint (none where it promises nothing).
    Names starting with an underscore or ``cython`` are compiled helpers of the allowed modules."""
    allowed = FOOTPRINT.get(dist)
    if allowed is None:
        return []
    return sorted(
        name
        for name in loaded
        if name not in allowed
        and name not in packages
        and not name.startswith(("_", "cython"))
        and name != "sitecustomize"
    )


def smoke(dist: str, wheels: list[Path], packages: Path = PACKAGES) -> list[str]:
    """Install ``wheels`` in a fresh environment and import ``dist``; return the modules imported."""
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / "venv"
        venv.create(env, with_pip=True)
        python = env / ("Scripts" if sys.platform == "win32" else "bin") / "python"
        subprocess.run(  # nosec B603
            [str(python), "-m", "pip", "install", "--quiet", *map(str, wheels)], check=True
        )
        names = import_packages(dist, packages)
        result = subprocess.run(  # nosec B603
            [
                str(python),
                "-I",
                "-c",
                _SMOKE_CHILD,
                json.dumps(names),
                json.dumps(EXTRA_ONLY[dist]),
            ],
            cwd=tmp,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise ReleaseError(f"importing {dist} failed:\n{result.stderr}")
    report = json.loads(result.stdout)
    if report["untyped"]:
        raise ReleaseError(f"{dist}'s wheel has no py.typed in {report['untyped']}")
    bad = footprint_violations(dist, report["loaded"], names)
    if bad:
        raise ReleaseError(f"{dist} loaded {bad} beyond its footprint {sorted(FOOTPRINT[dist])}")
    return report["imported"]


def wheel_of(dist: str, directory: Path) -> Path:
    """The one wheel of ``dist`` in ``directory``."""
    wheels = sorted(directory.glob(f"{dist.replace('-', '_')}-*.whl"))
    if len(wheels) != 1:
        raise ReleaseError(f"expected one {dist} wheel in {directory}, found {len(wheels)}")
    return wheels[0]


def smoke_all(directory: Path, packages: Path = PACKAGES) -> dict[str, int]:
    """``smoke`` every distribution with its in-repo requirements' wheels from ``directory``."""
    counts = {}
    for dist in sorted(path.parent.name for path in packages.glob("*/pyproject.toml")):
        needs = {
            canonicalize_name(req.name)
            for req in in_repo_requirements(dist, packages, extras=False)
        }
        wheels = [wheel_of(dist, directory)] + [wheel_of(name, directory) for name in sorted(needs)]
        counts[dist] = len(smoke(dist, wheels, packages))
    return counts


def write_sums(directory: Path) -> Path:
    """Write ``SHA256SUMS`` (``sha256sum`` format) for the files in ``directory``."""
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    target = directory / "SHA256SUMS"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("tag")
    check_parser.add_argument("--notes", type=Path)
    check_parser.add_argument("--github-output", type=Path)
    commands.add_parser("dependency-tags").add_argument("dist")
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("dist")
    smoke_parser.add_argument("wheels", nargs="+", type=Path)
    commands.add_parser("smoke-all").add_argument("directory", type=Path)
    commands.add_parser("sums").add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            dist, version, notes = check(args.tag, git_tags())
            if args.notes:
                args.notes.write_text(notes, encoding="utf-8")
            if args.github_output:
                with open(args.github_output, "a", encoding="utf-8") as f:
                    f.write(github_outputs(dist, version))
            print(f"{args.tag}: {dist} {version} is releasable")
        elif args.command == "dependency-tags":
            print("\n".join(dependency_tags(args.dist)))
        elif args.command == "smoke":
            imported = smoke(args.dist, args.wheels)
            print(f"{args.dist}: imported {len(imported)} modules from the installed wheel")
        elif args.command == "smoke-all":
            for dist, count in smoke_all(args.directory).items():
                print(f"{dist}: imported {count} modules from the installed wheel")
        else:
            print(write_sums(args.directory))
    except ReleaseError as error:
        print(f"release: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

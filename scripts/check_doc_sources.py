#!/usr/bin/env python3
"""Check that the teaching pages point at real code, cite their numbers, and that docs media is listed.

Three rules:

1. **Code pointers.** In ``docs/learn/``, inline code written as a dotted name whose first part is
   a package under ``packages/*/src`` (``rl_core.agents.returns.discounted_returns``) must resolve:
   to a module, or to a class, function, constant or attribute in it. Modules are parsed with
   ``ast``, never imported, so this runs without NumPy. A name re-exported with ``from ... import``
   is followed, and so is a method a class inherits from a base class named in the same package.
2. **Numbers have a source.** In ``docs/learn/``, every paragraph, list item or table row with a
   numeral must also hold a link to a report under ``docs/benchmarks/`` or a citation
   ``(CHANGELOG.md, <date>)``, and every date in such a citation must be a ``## <date>`` heading of
   ``CHANGELOG.md``. Headings, code and link targets are not read for numerals. A numeral is a
   number standing alone (``475``, ``-110``, ``0.99``, ``200k``), not a digit inside a name
   (``CartPole-v1``, ``Box2D``, ``SB3``) or written as a word.
3. **Docs media.** Every file under ``docs/images/`` must be listed in ``scripts/doc_media.json``
   with how it was made, linked from some Markdown page, a ``.png`` or ``.gif``, and within the
   manifest's budget: per file by type, and in total (one regeneration of every image).

CI runs this on every pull request, docs-only ones included, with the runner's own ``python3 -I``,
so keep it stdlib-only. It reads Markdown the way ``check_doc_links.py`` does, which it loads by path.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess  # nosec B404: git, with fixed arguments
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
LEARN_DIR = Path("docs/learn")
IMAGES_DIR = Path("docs/images")
BENCHMARKS_DIR = Path("docs/benchmarks")
MANIFEST = Path("scripts/doc_media.json")
CHANGELOG = Path("CHANGELOG.md")
MEDIA_SUFFIXES = {".png", ".gif"}


def _load_link_checker() -> ModuleType:
    """``check_doc_links.py`` from beside this file (``python3 -I`` keeps its folder off the path)."""
    path = Path(__file__).resolve().with_name("check_doc_links.py")
    spec = importlib.util.spec_from_file_location("check_doc_links", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


links = _load_link_checker()

DOTTED = re.compile(r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?:\(\))?`")
CITATION = re.compile(r"\(CHANGELOG\.md,([^)]*)\)")
CITATION_LINK = re.compile(r"\[CHANGELOG\.md,([^\]]*)\]\([^)]*\)")  # the clickable form
DATE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
CHANGELOG_HEADING = re.compile(r"^## (\d{4}-\d{2}(?:-\d{2})?)\s*$", re.MULTILINE)
# A number standing alone: not after a letter, digit, dot or "word-" (v1, Box2D, x86-64), not before a
# letter except an exponent (1e-3) or a k or M multiplier (200k).
NUMERAL = re.compile(
    r"(?<![\w.])(?<!\w-)-?\d(?:[\d,.]*\d)?(?:e-?\d+)?(?:[kM](?![A-Za-z]))?(?![A-Za-z_\d])"
)
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAG = re.compile(r"<[^>]+>")
IMAGE_ALT = re.compile(r"!\[[^\]]*\]")
REFERENCE_USE = re.compile(r"\]\[[^\]]*\]")
REFERENCE_DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:[ \t]*<?([^\s>]+)>?.*$", re.MULTILINE)
IMG_SRC = re.compile(r"""<img\s[^>]*?src=["']([^"']+)["']""", re.IGNORECASE)


# --- code pointers ---------------------------------------------------------------------------


class CodeIndex:
    """The packages under ``packages/*/src`` and their parsed modules.

    A plain class, not a dataclass: the tests load this script by path without registering it in
    ``sys.modules``, which a dataclass with postponed annotations needs.
    """

    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = roots  # top-level package name -> its src/ folder
        self._trees: dict[Path, ast.Module] = {}

    @classmethod
    def of(cls, root: Path) -> CodeIndex:
        roots = {
            init.parent.name: init.parent.parent
            for init in sorted(root.glob("packages/*/src/*/__init__.py"))
        }
        return cls(roots)

    def tree(self, path: Path) -> ast.Module:
        if path not in self._trees:
            self._trees[path] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return self._trees[path]

    def module_file(self, dotted: list[str]) -> Path | None:
        """The file of a dotted module name: ``pkg/__init__.py`` or ``pkg/mod.py``, or None."""
        if not dotted or dotted[0] not in self.roots:
            return None
        folder = self.roots[dotted[0]] / dotted[0]
        for index, part in enumerate(dotted[1:], start=1):
            if (folder / part / "__init__.py").is_file():
                folder = folder / part
            elif index == len(dotted) - 1 and (folder / f"{part}.py").is_file():
                return folder / f"{part}.py"
            else:
                return None
        return folder / "__init__.py"

    def module_name(self, path: Path) -> list[str]:
        for name, src in self.roots.items():
            try:
                parts = list(path.relative_to(src).with_suffix("").parts)
            except ValueError:
                continue
            if parts and parts[0] == name:
                return parts[:-1] if parts[-1] == "__init__" else parts
        raise ValueError(f"{path} is not in a known package")

    def resolves(self, dotted: str) -> bool:
        """Whether a dotted name is a module, or a name defined in one (followed through imports)."""
        parts = dotted.split(".")
        for split in range(len(parts), 0, -1):
            path = self.module_file(parts[:split])
            if path is not None:
                return self._in_module(path, parts[split:], depth=0)
        return False

    def _in_module(self, path: Path, attrs: list[str], depth: int) -> bool:
        if not attrs:
            return True
        found = self._definition(path, attrs[0], depth)
        if found is None:
            return False
        where, node = found
        if isinstance(node, Path):
            return self._in_module(node, attrs[1:], depth + 1)
        if isinstance(node, ast.ClassDef):
            return self._in_class(where, node, attrs[1:], depth)
        return len(attrs) == 1

    def _definition(self, path: Path, name: str, depth: int) -> tuple[Path, ast.AST | Path] | None:
        """Where ``name`` is defined at ``path``'s top level: a node, or a submodule's file."""
        if depth > 20:
            return None
        if path.name == "__init__.py":
            sub = self.module_file([*self.module_name(path), name])
            if sub is not None:
                return path, sub
        for node in _top_level(self.tree(path)):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name == name:
                    return path, node
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                if _assigns(node, name):
                    return path, node
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        source = self.module_file(self._from_module(path, node))
                        found = (
                            None if source is None else self._definition(source, name, depth + 1)
                        )
                        if found is not None:
                            return found
                    elif (alias.asname or alias.name) == name:
                        return self._imported(path, node, alias.name, depth)
            elif isinstance(node, ast.Import) and any(
                _bound_name(alias) == name for alias in node.names
            ):
                return path, node
        return None

    def _from_module(self, path: Path, node: ast.ImportFrom) -> list[str]:
        """The absolute dotted module of ``from <module> import ...`` written in ``path``."""
        if node.level:
            package = self.module_name(path)
            if path.name != "__init__.py":
                package = package[:-1]
            base = package[: len(package) - (node.level - 1)]
        else:
            base = []
        return [*base, *(node.module.split(".") if node.module else [])]

    def _module_of(self, path: Path, expr: ast.expr, depth: int) -> Path | None:
        """The module file an expression such as ``pkg.base`` or an import alias names, if any."""
        if isinstance(expr, ast.Attribute):
            parent = self._module_of(path, expr.value, depth + 1)
            return (
                None if parent is None else self.module_file([*self.module_name(parent), expr.attr])
            )
        if not isinstance(expr, ast.Name):
            return None
        found = self._definition(path, expr.id, depth + 1)
        if found is None:
            return None
        if isinstance(found[1], Path):
            return found[1]
        if isinstance(found[1], ast.Import):
            for alias in found[1].names:
                if _bound_name(alias) == expr.id:
                    return self.module_file((alias.name if alias.asname else expr.id).split("."))
        return None

    def _base_class(
        self, path: Path, base: ast.expr, depth: int
    ) -> tuple[Path, ast.ClassDef] | None:
        """The class a base-class expression (``Base`` or ``module.Base``) names, if in the packages."""
        if isinstance(base, ast.Name):
            found = self._definition(path, base.id, depth + 1)
        elif isinstance(base, ast.Attribute):
            module = self._module_of(path, base.value, depth + 1)
            found = None if module is None else self._definition(module, base.attr, depth + 1)
        else:
            found = None
        if found is not None and isinstance(found[1], ast.ClassDef):
            return found[0], found[1]
        return None

    def _imported(
        self, path: Path, node: ast.ImportFrom, name: str, depth: int
    ) -> tuple[Path, ast.AST | Path] | None:
        module = self._from_module(path, node)
        submodule = self.module_file([*module, name])
        if submodule is not None:
            return path, submodule
        source = self.module_file(module)
        if source is None:
            return path, node  # outside the packages (numpy, typing): a name, nothing to follow
        return self._definition(source, name, depth + 1)

    def _in_class(self, path: Path, cls: ast.ClassDef, attrs: list[str], depth: int) -> bool:
        if not attrs:
            return True
        name = attrs[0]
        for node in cls.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return self._in_class(path, node, attrs[1:], depth)
            if _defines(node, name):
                return len(attrs) == 1
        if len(attrs) == 1 and _assigned_on_self(cls, name):
            return True
        if depth > 20:
            return False
        for base in cls.bases:
            found = self._base_class(path, base, depth)
            if found is not None and self._in_class(found[0], found[1], attrs, depth + 1):
                return True
        return False


def _top_level(tree: ast.Module) -> list[ast.stmt]:
    """The module's statements, with those inside a top-level ``if`` or ``try`` (TYPE_CHECKING)."""
    statements: list[ast.stmt] = []
    for node in tree.body:
        statements.append(node)
        if isinstance(node, ast.If):
            statements.extend(node.body + node.orelse)
        elif isinstance(node, ast.Try):
            statements.extend(node.body + node.orelse + node.finalbody)
            for handler in node.handlers:
                statements.extend(handler.body)
    return statements


def _bound_name(alias: ast.alias) -> str:
    """The name an ``import`` binds: its alias, or the first part (``import os.path`` binds os)."""
    return alias.asname or alias.name.split(".")[0]


def _leaves(target: ast.expr) -> list[ast.expr]:
    """An assignment target's single targets, unpacking tuples and lists (``a, [b, *c] = ...``)."""
    if isinstance(target, ast.Tuple | ast.List):
        return [leaf for element in target.elts for leaf in _leaves(element)]
    if isinstance(target, ast.Starred):
        return _leaves(target.value)
    return [target]


def _targets(node: ast.AST) -> list[ast.expr]:
    """Every single target an assignment writes."""
    if isinstance(node, ast.Assign):
        return [leaf for target in node.targets for leaf in _leaves(target)]
    if isinstance(node, ast.AnnAssign | ast.AugAssign):
        return _leaves(node.target)
    return []


def _assigns(node: ast.AST, name: str) -> bool:
    """Whether an assignment binds the plain name ``name``."""
    return any(isinstance(leaf, ast.Name) and leaf.id == name for leaf in _targets(node))


def _defines(node: ast.stmt, name: str) -> bool:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return node.name == name
    return isinstance(node, ast.Assign | ast.AnnAssign) and _assigns(node, name)


def _assigned_on_self(cls: ast.ClassDef, name: str) -> bool:
    """Whether a method of the class assigns ``self.<name>``."""
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(method):
            for target in _targets(node):
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == name
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    return True
    return False


# --- numbers and citations ---------------------------------------------------------------------


def learn_pages(root: Path) -> list[Path]:
    return sorted((root / LEARN_DIR).rglob("*.md")) if (root / LEARN_DIR).is_dir() else []


def changelog_dates(root: Path) -> set[str]:
    path = root / CHANGELOG
    if not path.is_file():
        return set()
    return set(CHANGELOG_HEADING.findall(path.read_text(encoding="utf-8")))


def units(body: str) -> list[tuple[int, str]]:
    """Paragraphs, list items and table rows of prose, each with its first line number."""
    found: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0

    def close() -> None:
        if current:
            found.append((start, "\n".join(current)))
            current.clear()

    for number, line in enumerate(body.split("\n"), start=1):
        if not line.strip() or links.HEADING.match(line):
            close()
            continue
        if line.lstrip().startswith("|") or LIST_ITEM.match(line):
            close()
        if not current:
            start = number
        current.append(line)
        if line.lstrip().startswith("|"):
            close()
    close()
    return found


def _has_benchmark_link(page: Path, root: Path, text: str) -> bool:
    for match in links.LINK.finditer(text):
        target = match.group(1).strip("<>").partition("#")[0]
        if not target or links.URL_SCHEME.match(target):
            continue
        resolved = (page.parent / target).resolve()
        benchmarks = (root / BENCHMARKS_DIR).resolve()
        if resolved.suffix == ".md" and resolved.parent == benchmarks:
            return True
    return False


def numerals(text: str) -> list[str]:
    """The numerals of a unit's text, without citations, link targets, alt text, HTML or comments."""
    text = HTML_COMMENT.sub(" ", text)
    text = HTML_TAG.sub(" ", text)
    text = REFERENCE_DEFINITION.sub(" ", text)
    text = CITATION.sub(" ", text)
    text = CITATION_LINK.sub(" ", text)
    text = IMAGE_ALT.sub("!", text)
    text = REFERENCE_USE.sub("] ", text)
    text = links.LINK.sub("] ", text)
    text = LIST_ITEM.sub(" ", text, count=1)
    return NUMERAL.findall(text)


def check_page(page: Path, root: Path, index: CodeIndex, dates: set[str]) -> list[str]:
    where = page.relative_to(root).as_posix()
    raw = links.outside_fences(page.read_text(encoding="utf-8"))
    problems: list[str] = []
    for match in DOTTED.finditer(raw):
        dotted = match.group(1)
        package = dotted.split(".")[0]
        line = raw.count("\n", 0, match.start()) + 1
        if package in index.roots and not index.resolves(dotted):
            problems.append(f"{where}:{line}: no such module, class or name in the code: {dotted}")
        elif package.startswith("rl_") and package not in index.roots:
            problems.append(
                f"{where}:{line}: no package of this repository is named {package}: {dotted}"
            )
    body = links.prose(raw)
    for line, text in units(body):
        cited = CITATION.findall(text) + CITATION_LINK.findall(text)
        for citation in cited:
            for date in DATE.findall(citation) or ["(no date)"]:
                if date not in dates:
                    problems.append(f"{where}:{line}: CHANGELOG.md has no '## {date}' section")
        found = numerals(text)
        if found and not cited and not _has_benchmark_link(page, root, text):
            shown = ", ".join(found[:3]) + (", ..." if len(found) > 3 else "")
            problems.append(
                f"{where}:{line}: numbers without a source ({shown}): cite "
                "(CHANGELOG.md, <date>) or link a docs/benchmarks/ report in the same paragraph"
            )
    return problems


# --- media ------------------------------------------------------------------------------------


def _manifest(root: Path) -> tuple[dict[str, object], list[str]]:
    path = root / MANIFEST
    if not path.is_file():
        return {}, [f"{MANIFEST}: missing"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {}, [f"{MANIFEST}: not valid JSON: {error}"]
    if not isinstance(data, dict):
        return {}, [f"{MANIFEST}: must be a JSON object"]
    return data, []


def _referenced_files(root: Path) -> set[Path]:
    """Every file a relative link, image, ``<img src>`` or link reference in the Markdown names."""
    found: set[Path] = set()
    for page in links.markdown_files(root):
        body = links.prose(page.read_text(encoding="utf-8"))
        targets = [match.group(1).strip("<>") for match in links.LINK.finditer(body)]
        targets += IMG_SRC.findall(body) + REFERENCE_DEFINITION.findall(body)
        for target in (target.partition("#")[0] for target in targets):
            if target and not links.URL_SCHEME.match(target):
                base = root if target.startswith("/") else page.parent
                found.add((base / target.lstrip("/")).resolve())
    return found


def check_media(root: Path) -> tuple[list[str], int]:
    """Media problems, and how many files under ``docs/images/`` were checked."""
    folder = root / IMAGES_DIR
    files = sorted(p for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []
    data, problems = _manifest(root)
    if problems:
        return (problems if files or (root / MANIFEST).exists() else []), len(files)
    budget = data.get("budget")
    media = data.get("media")
    if not isinstance(budget, dict) or not isinstance(media, list):
        return [f"{MANIFEST}: needs a 'budget' object and a 'media' list"], len(files)
    limits = {key: budget.get(key) for key in ("png_kib", "gif_kib", "total_kib")}
    if not all(isinstance(value, int) and value > 0 for value in limits.values()):
        return [f"{MANIFEST}: budget needs positive integers png_kib, gif_kib, total_kib"], len(
            files
        )
    kib = {name: int(str(value)) for name, value in limits.items()}
    listed: dict[str, dict[str, object]] = {}
    for number, entry in enumerate(media):
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(key), str) and entry.get(key) for key in ("path", "made_by")
        ):
            problems.append(f"{MANIFEST}: media[{number}] needs a 'path' and a 'made_by' string")
            continue
        path = str(entry["path"])
        if path in listed:
            problems.append(f"{MANIFEST}: {path} is listed twice")
        listed[path] = entry
        if not (root / path).is_file():
            problems.append(f"{MANIFEST}: {path} is listed but does not exist")
    referenced = _referenced_files(root)
    total = 0
    for file in files:
        relative = file.relative_to(root).as_posix()
        size = file.stat().st_size
        total += size
        limit = kib.get(f"{file.suffix.lower().lstrip('.')}_kib")
        if file.suffix.lower() not in MEDIA_SUFFIXES or limit is None:
            problems.append(f"{relative}: only .png and .gif belong in {IMAGES_DIR}/")
        elif size > limit * 1024:
            problems.append(f"{relative}: {size // 1024} KiB, over the {limit} KiB budget")
        if relative not in listed:
            problems.append(f"{relative}: not listed in {MANIFEST}")
        if file.resolve() not in referenced:
            problems.append(f"{relative}: no Markdown page links it")
    if total > kib["total_kib"] * 1024:
        problems.append(
            f"{IMAGES_DIR}/: {total // 1024} KiB in all, over the {kib['total_kib']} KiB budget"
        )
    for stray in _media_outside_images(root):
        problems.append(f"{stray}: documentation media belongs in {IMAGES_DIR}/")
    return problems, len(files)


def _media_outside_images(root: Path) -> list[str]:
    """``.png`` and ``.gif`` files under ``docs/`` or ``packages/`` but not ``docs/images/``.

    Tracked and not-yet-tracked files when git is there (git-ignored output is not a doc's), else
    every file.
    """
    folders = ["docs", "packages"]
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", *folders],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8")
        names = [name for name in listed.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        names = [
            path.relative_to(root).as_posix()
            for folder in folders
            if (root / folder).is_dir()
            for path in (root / folder).rglob("*")
            if path.is_file()
        ]
    images = IMAGES_DIR.as_posix() + "/"
    return sorted(
        name
        for name in names
        if Path(name).suffix.lower() in MEDIA_SUFFIXES and not name.startswith(images)
    )


# --- entry point ------------------------------------------------------------------------------


def check_sources(root: Path) -> tuple[list[str], int, int]:
    """All problems, the number of teaching pages read and of media files checked."""
    index = CodeIndex.of(root)
    dates = changelog_dates(root)
    pages = learn_pages(root)
    problems = [problem for page in pages for problem in check_page(page, root, index, dates)]
    media_problems, media_count = check_media(root)
    return problems + media_problems, len(pages), media_count


def main() -> int:
    problems, pages, media = check_sources(REPO_ROOT)
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problems over {pages} teaching pages and {media} media files.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

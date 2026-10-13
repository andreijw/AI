"""Tests that scan the source tree for conventions no single unit test can enforce."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = sorted((REPO_ROOT / "src").rglob("*.py")) + [
    REPO_ROOT / "train.py",
    REPO_ROOT / "record.py",
]

# Constructors for an owned generator. Everything else on numpy.random uses global state.
ALLOWED_NUMPY_RANDOM_NAMES = {"default_rng", "Generator", "SeedSequence", "PCG64"}


def _dotted_name(node: ast.AST) -> str:
    """Return 'np.random.shuffle' for the matching attribute chain, or '' if it is not one."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def global_numpy_random_uses(source: str) -> list[tuple[int, str]]:
    """
    Return (line, description) for each use of a global random generator in source.

    Flags any ``np.random.<name>`` / ``numpy.random.<name>`` reference outside the allow-list
    (calls and bare references alike), imports that bring numpy.random in under another name,
    and the standard library's ``random`` module. It does not follow aliases assigned at
    runtime (``r = np.random``) or NumPy imported under a name other than ``np``.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            name = _dotted_name(node)
            module, _, member = name.rpartition(".")
            if module in {"np.random", "numpy.random"} and member not in ALLOWED_NUMPY_RANDOM_NAMES:
                found.append((node.lineno, name))
        elif isinstance(node, ast.ImportFrom):
            names = sorted(alias.name for alias in node.names)
            if node.module == "numpy.random" and set(names) - ALLOWED_NUMPY_RANDOM_NAMES:
                found.append((node.lineno, f"from numpy.random import {', '.join(names)}"))
            elif node.module == "numpy" and "random" in names:
                found.append((node.lineno, "from numpy import random"))
            elif node.module == "random":
                found.append((node.lineno, f"from random import {', '.join(names)}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy.random" and alias.asname is not None:
                    found.append((node.lineno, f"import numpy.random as {alias.asname}"))
                elif alias.name == "random":
                    found.append((node.lineno, "import random"))
    return found


def test_no_bare_global_numpy_random_calls_in_source():
    """Every random draw comes from an owned, seeded generator, never NumPy's global one."""
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line} {what}"
        for path in SOURCE_FILES
        for line, what in global_numpy_random_uses(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, (
        "Use the object's own np.random.default_rng(seed) instead of the global generator:\n"
        + "\n".join(offenders)
    )


def test_scan_covers_the_source_tree():
    """The scan is looking at real files, so an empty result means something."""
    names = {path.name for path in SOURCE_FILES}

    assert {"continuous_ppo.py", "trainer.py", "train.py", "record.py"} <= names
    assert all(path.exists() for path in SOURCE_FILES)


def test_scan_flags_global_calls_and_imports():
    """The scanner recognises each spelling of a global generator it claims to catch."""
    source = (
        "import numpy as np\n"
        "import numpy.random as npr\n"
        "from numpy.random import shuffle\n"
        "np.random.shuffle(x)\n"
        "numpy.random.seed(0)\n"
        "from numpy import random\n"
        "import random\n"
        "from random import choice\n"
        "sampler = np.random.rand\n"
    )

    assert sorted(global_numpy_random_uses(source)) == [
        (2, "import numpy.random as npr"),
        (3, "from numpy.random import shuffle"),
        (4, "np.random.shuffle"),
        (5, "numpy.random.seed"),
        (6, "from numpy import random"),
        (7, "import random"),
        (8, "from random import choice"),
        (9, "np.random.rand"),
    ]


def test_scan_allows_owned_generators():
    """Creating or annotating an owned generator is not a use of global state."""
    source = (
        "import numpy as np\n"
        "from numpy.random import Generator, default_rng\n"
        "rng: np.random.Generator = np.random.default_rng(0)\n"
        "other = np.random.Generator(np.random.PCG64(1))\n"
        "rng.shuffle(x)\n"
        "self._rng.standard_normal(3)\n"
    )

    assert global_numpy_random_uses(source) == []

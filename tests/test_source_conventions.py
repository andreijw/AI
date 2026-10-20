"""Tests that scan the source tree for conventions no single unit test can enforce."""

import ast
import re
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
        "Use the object's own generator (rl_core.seeding.derive_rng) instead of the global one:\n"
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


# ---------------------------------------------------------------------------
# Package dependencies: rl_core never depends on a challenge, and nothing aliases modules
# ---------------------------------------------------------------------------


def _imported_modules(source: str) -> list[tuple[int, str]]:
    """(line, module) for every static import statement in source."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
    return found


def _source_files_under(package_glob: str) -> list[Path]:
    return sorted(p for p in (REPO_ROOT / "src").glob(f"{package_glob}/**/*.py"))


def test_challenge_packages_do_not_import_the_benchmark_script():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in _source_files_under("rl_*")
        for line, module in _imported_modules(path.read_text(encoding="utf-8"))
        if module == "benchmark" or module.startswith("benchmark.")
    ]
    assert offenders == [], f"Import the challenge API from rl_core.challenges: {offenders}"


def test_rl_core_does_not_import_a_challenge_package():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line} imports {module}"
        for path in _source_files_under("rl_core")
        for line, module in _imported_modules(path.read_text(encoding="utf-8"))
        if module.split(".")[0].startswith("rl_") and module.split(".")[0] != "rl_core"
    ]
    assert offenders == [], f"rl_core must not depend on a challenge: {offenders}"


def test_no_source_file_registers_modules_under_another_name():
    offenders = []
    for path in SOURCE_FILES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and _dotted_name(node) == "sys.modules":
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == [], f"sys.modules aliasing hides import cycles: {offenders}"


def test_import_scan_sees_both_import_forms():
    source = "import benchmark\nfrom rl_cartpole.benchmark import X\nfrom . import y\n"
    assert _imported_modules(source) == [(1, "benchmark"), (2, "rl_cartpole.benchmark")]


# Building a generator from a seed with any of these equals default_rng(seed).
SEEDED_GENERATOR_CONSTRUCTORS = ("default_rng", "Generator", "PCG64", "SeedSequence")


def seeded_default_rng_calls(source: str) -> list[int]:
    """Lines that build a generator from an argument (a seed) directly, instead of with ``derive_rng``."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        is_call = isinstance(node, ast.Call) and _dotted_name(node.func).endswith(
            SEEDED_GENERATOR_CONSTRUCTORS
        )
        if is_call and (node.args or node.keywords):
            lines.append(node.lineno)
    return lines


def test_agents_seed_generators_through_named_streams():
    """An agent's generators come from rl_core.seeding.derive_rng(seed, <stream>): default_rng(seed) would repeat
    the environment's start-state numbers (roadmap: "Separate Random Streams for Agents and Buffers")."""
    agent_files = sorted((REPO_ROOT / "src" / "rl_core" / "agents").glob("*.py"))
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in agent_files
        for line in seeded_default_rng_calls(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "Use derive_rng(seed, <stream>) instead of default_rng(seed):\n" + "\n".join(offenders)
    )


def test_the_seeded_default_rng_scan_flags_what_it_claims():
    source = (
        "import numpy as np\nnp.random.default_rng(3)\nnp.random.default_rng()\ndefault_rng(seed=1)\n"
        "np.random.Generator(np.random.PCG64(5))\n"
    )
    assert seeded_default_rng_calls(source) == [2, 4, 5, 5]


# ---------------------------------------------------------------------------
# Agent tests go through the public interface; private access lives in tests/agent_internals.py
# ---------------------------------------------------------------------------

# Test files moved onto the public interface (roadmap: "Test Agents Through Their Public Behaviour").
# Each later family adds its files here.
PUBLIC_BEHAVIOUR_TEST_FILES = [
    REPO_ROOT / "tests" / name
    for name in (
        "test_agents_extensive.py",
        "test_reinforce_agent.py",
        "test_actor_critic_agent.py",
        "test_ppo_agent.py",
        "test_discrete_ppo.py",
        "test_policy_gradient_maths.py",
        "test_discrete_policy_numerics.py",
    )
]

_PRIVATE_NAME = re.compile(r"_[A-Za-z]\w*")


def private_accesses(source: str) -> list[tuple[int, str]]:
    """
    Return (line, description) for each way source reaches an object's private member.

    Flags ``obj._name`` on anything but ``self`` or ``cls``; ``obj.__dict__``; a string literal
    that is a private name (as passed to ``getattr``, ``setattr`` or ``monkeypatch.setattr``, or
    listed for a loop over them); ``getattr``, ``setattr``, ``hasattr`` or ``delattr`` with a
    computed name; ``vars(...)``, ``dir(...)`` and ``inspect.getmembers(...)``, which list private
    members wholesale; and importing a private name. Other dunder names are allowed.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            owner = node.value.id if isinstance(node.value, ast.Name) else ""
            private = node.attr.startswith("_") and not node.attr.startswith("__")
            if (private and owner not in {"self", "cls"}) or node.attr == "__dict__":
                found.append((node.lineno, f"{_dotted_name(node) or '...'}"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _PRIVATE_NAME.fullmatch(node.value) and not node.value.startswith("__"):
                found.append((node.lineno, repr(node.value)))
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name in {"vars", "dir", "inspect.getmembers"}:
                found.append((node.lineno, f"{name}(...)"))
            elif name in {"getattr", "setattr", "hasattr", "delattr"} and len(node.args) > 1:
                literal = isinstance(node.args[1], ast.Constant)
                if not literal:
                    found.append((node.lineno, f"{name} with a computed name"))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name.startswith("_") and not alias.name.startswith("__"):
                    found.append((node.lineno, f"import {alias.name}"))
    return found


def test_migrated_agent_tests_use_only_the_public_interface():
    """Private agent state is read only through tests/agent_internals.py, so a refactor with no
    behaviour change touches one helper instead of every test."""
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line} {what}"
        for path in PUBLIC_BEHAVIOUR_TEST_FILES
        for line, what in private_accesses(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, (
        "Assert through select_action, update's metrics or save/load; for maths, go through "
        "tests/agent_internals.py:\n" + "\n".join(offenders)
    )


def test_private_access_scan_flags_each_spelling():
    """The scanner recognises each way a test can reach a private member that it claims to catch."""
    source = (
        "agent._W1\n"
        "run.agent._forward(x)\n"
        "getattr(agent, '_b1')\n"
        "monkeypatch.setattr(agent, '_loss_and_grads', spy)\n"
        "names = ('_W_pi', '_W_v')\n"
        "vars(agent)\n"
        "agent.__dict__\n"
        "getattr(agent, f'_{name}')\n"
        "setattr(agent, name, value)\n"
        "dir(agent)\n"
        "inspect.getmembers(agent)\n"
        "from rl_core.agents.returns import _helper\n"
    )

    assert sorted(private_accesses(source)) == [
        (1, "agent._W1"),
        (2, "run.agent._forward"),
        (3, "'_b1'"),
        (4, "'_loss_and_grads'"),
        (5, "'_W_pi'"),
        (5, "'_W_v'"),
        (6, "vars(...)"),
        (7, "agent.__dict__"),
        (8, "getattr with a computed name"),
        (9, "setattr with a computed name"),
        (10, "dir(...)"),
        (11, "inspect.getmembers(...)"),
        (12, "import _helper"),
    ]


def test_private_access_scan_allows_self_dunders_and_plain_strings():
    source = (
        "self._W1 = 0\n"
        "cls._registry\n"
        "agent.__class__.__name__\n"
        "'__init__'\n"
        "match = 'weights_change'\n"
        "agent.select_action(obs)\n"
        "_local_helper(agent)\n"
    )

    assert private_accesses(source) == []

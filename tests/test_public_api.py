"""rl-core's public API: each public module's ``__all__``. Every other module path is private.

The distributions that depend on rl-core, the examples and the external-games guide import only public names;
the root scripts and scripts/ ship with this checkout, so their deep imports are allowed, but listed: a new one is
a visible diff. A change to any list here is an API change, made on purpose.
"""

import importlib
import re
from pathlib import Path

import pytest

from tests.helpers import package_of, resolved_imports

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The modules whose ``__all__`` is the API. The top level lists only ``__version__``: listing submodules it does
#: not import would make ``from rl_core import *`` import them all.
PUBLIC_MODULES = (
    "rl_core",
    "rl_core.agents",
    "rl_core.challenges",
    "rl_core.environments",
    "rl_core.export",
    "rl_core.export.c_header",
    "rl_core.export.onnx_model",
    "rl_core.export.rollout",
    "rl_core.export.verify",
    "rl_core.seeding",
    "rl_core.training",
    "rl_core.utils",
)

#: What the package ``__init__`` files exported in the released rl-core 0.1.0. A name here never leaves an
#: ``__all__`` without a deprecation.
RELEASED_0_1_0 = {
    "rl_core.agents": {
        "ActorCriticAgent",
        "ActorCriticBase",
        "BaseAgent",
        "ContinuousActorCriticAgent",
        "ContinuousActorCriticBase",
        "ContinuousPPOAgent",
        "ContinuousRandomAgent",
        "DDPGAgent",
        "DQNAgent",
        "PPOAgent",
        "RandomAgent",
        "ReinforceAgent",
        "ReplayBuffer",
        "SACAgent",
        "TD3Agent",
    },
    "rl_core.challenges": {
        "BenchmarkChallenge",
        "CHALLENGES",
        "ChallengeEnv",
        "discover_challenges",
        "import_failures_text",
        "register_challenge",
        "select_challenge",
        "unknown_challenge_message",
        "with_box_action_bounds",
    },
    "rl_core.environments": {
        "OBSERVATION_DEFAULTS",
        "POSITIVE",
        "BaseEnvWrapper",
        "ObservationOptions",
        "ParameterBounds",
        "ResumableEnv",
        "SchedulableEnv",
        "checked_randomization",
        "checked_range",
        "env_from_config",
        "non_negative_setting",
        "uses_box2d",
    },
    "rl_core.training": {"Trainer"},
    "rl_core.utils": {"load_config", "setup_logger"},
}

#: The API now: the released names and what 0.2.0 adds.
PUBLIC_API = {
    **RELEASED_0_1_0,
    "rl_core": {"__version__"},
    "rl_core.export": {"c_header", "onnx_model", "rollout", "verify"},
    "rl_core.export.c_header": {"render_c_header"},
    "rl_core.export.onnx_model": {
        "EXTRA",
        "missing_packages",
        "render_onnx_model",
        "verify_onnx_model",
    },
    "rl_core.export.rollout": {
        "Agreement",
        "GreedyRollout",
        "greedy_rollout",
        "policy_disagreements",
    },
    "rl_core.export.verify": {"VerificationReport", "find_c_compiler", "verify_c_header"},
    "rl_core.seeding": {"derive_rng", "env_seed", "eval_seed_for"},
    "rl_core.utils": {
        *RELEASED_0_1_0["rl_core.utils"],
        "check_positive_int",
        "non_negative_int",
        "read_env_settings",
    },
}

#: Private names a dependent may import, each with its reason. Its ``rl-core~=0.2.0`` pin guards them.
DEPENDENT_ALLOWLIST = {
    # The legacy vector factories' async_envs check: private, as they are deprecated (removed in 0.3.0).
    "packages/rl-challenges/src/rl_cartpole/environments/make_env.py: "
    "rl_core.utils.config_values.async_envs_setting",
    "packages/rl-challenges/src/rl_pendulum/environments/make_env.py: "
    "rl_core.utils.config_values.async_envs_setting",
}

#: The private rl_core names the checkout's own scripts import.
SCRIPT_DEEP_IMPORTS = {
    "benchmark.py: rl_core.training.budget.apply_budget_override",
    "benchmark.py: rl_core.training.budget.budget_of",
    "benchmark.py: rl_core.training.parallel.num_envs_of",
    "benchmark.py: rl_core.training.trainer.Trainer",
    "benchmark.py: rl_core.training.trainer.success_rate",
    "benchmark.py: rl_core.utils.smoothing.rolling_average",
    "benchmark.py: rl_core.utils.visualization.plot_benchmark_comparison",
    "export.py: rl_core.training.checkpoints.file_sha256",
    "record.py: rl_core.training.parallel.single_env_training",
    "record.py: rl_core.utils.video.merge_episode_clips",
    "robustness.py: rl_core.training.checkpoints.file_sha256",
    "robustness.py: rl_core.training.curriculum.curriculum_from_config",
    "robustness.py: rl_core.training.robustness.Perturbation",
    "robustness.py: rl_core.training.robustness.RobustnessRow",
    "robustness.py: rl_core.training.robustness.difficulty_reached",
    "robustness.py: rl_core.training.robustness.parse_sweep",
    "robustness.py: rl_core.training.robustness.perturbed_section",
    "robustness.py: rl_core.training.robustness.robustness_rows",
    "train.py: rl_core.training.budget.apply_budget_override",
    "train.py: rl_core.training.curriculum.curriculum_from_config",
    "train.py: rl_core.training.parallel.num_envs_of",
    "train.py: rl_core.utils.video.TrainingVideoRecorder",
    "train.py: rl_core.utils.visualization.plot_training_metrics",
}

DEPENDENT_SOURCES = [
    *sorted((REPO_ROOT / "packages" / "rl-challenges" / "src").rglob("*.py")),
    *sorted((REPO_ROOT / "packages" / "rl-navigation" / "src").rglob("*.py")),
    *sorted((REPO_ROOT / "examples").rglob("*.py")),
    REPO_ROOT / "example.py",
]
SCRIPT_SOURCES = [
    *(path for path in sorted(REPO_ROOT.glob("*.py")) if path.name != "example.py"),
    *sorted((REPO_ROOT / "scripts").rglob("*.py")),
]
GUIDE = REPO_ROOT / "docs" / "external-challenges.md"
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
IMPORT_LINE = re.compile(r"^(?:from|import) [^(\n]+\n", re.MULTILINE)


def _declared(module: str) -> set[str]:
    return set(importlib.import_module(module).__all__)


def is_public(dotted: str) -> bool:
    """A public module, or a name of a public module's API (PUBLIC_API, which the modules' ``__all__``
    must equal)."""
    if dotted in PUBLIC_MODULES:
        return True
    parent, _, name = dotted.rpartition(".")
    return name in PUBLIC_API.get(parent, ())


def private_imports(source: str, package: list[str]) -> list[str]:
    """The private rl_core modules and names ``source`` imports, each statement reported by its deepest
    name (``from rl_core.utils.config import x`` is ``rl_core.utils.config.x``)."""
    found = [
        (line, d) for line, d in resolved_imports(source, package) if d.split(".")[0] == "rl_core"
    ]
    deepest = [
        dotted
        for line, dotted in found
        if not any(
            other_line == line and other.startswith(dotted + ".") for other_line, other in found
        )
    ]
    return sorted({dotted for dotted in deepest if not is_public(dotted)})


def _private_imports_of(paths: list[Path]) -> set[str]:
    return {
        f"{path.relative_to(REPO_ROOT).as_posix()}: {dotted}"
        for path in paths
        for dotted in private_imports(path.read_text(encoding="utf-8"), package_of(path))
    }


@pytest.mark.parametrize("module", sorted(PUBLIC_API))
def test_each_public_module_declares_exactly_its_api(module):
    assert _declared(module) == PUBLIC_API[module], (
        f"{module}.__all__ changed: update PUBLIC_API here on purpose, and the CHANGELOG"
    )


@pytest.mark.parametrize("module", sorted(RELEASED_0_1_0))
def test_no_released_name_leaves_the_api(module):
    missing = RELEASED_0_1_0[module] - _declared(module)
    assert not missing, (
        f"{module} dropped {sorted(missing)}, released in 0.1.0: deprecate them first"
    )


def test_every_public_module_has_an_api():
    assert set(PUBLIC_API) == set(PUBLIC_MODULES)


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_every_declared_name_exists_and_star_imports(module):
    namespace: dict[str, object] = {}
    exec(f"from {module} import *", namespace)  # a module name from PUBLIC_MODULES
    assert _declared(module) <= set(namespace), "a name in __all__ is not defined"


def test_the_dependents_import_only_the_public_api():
    found = _private_imports_of(DEPENDENT_SOURCES)
    assert found - DEPENDENT_ALLOWLIST == set(), (
        "import from a package (rl_core.utils, rl_core.environments, ...), not its modules"
    )
    assert DEPENDENT_ALLOWLIST - found == set(), "a stale allowlist entry: remove it"


def test_the_external_games_guide_imports_only_the_public_api():
    """A game copies these imports. Some blocks are fragments of a class, so only their unindented
    import lines are parsed."""
    blocks = PYTHON_BLOCK.findall(GUIDE.read_text(encoding="utf-8"))
    imports = "".join(line for block in blocks for line in IMPORT_LINE.findall(block))
    assert "rl_core" in imports, "the guide's imports were not found"
    assert private_imports(imports, []) == []


def test_the_scripts_deep_imports_are_the_listed_ones():
    assert _private_imports_of(SCRIPT_SOURCES) == SCRIPT_DEEP_IMPORTS, (
        "a script's private rl_core import changed: prefer the public API, else update SCRIPT_DEEP_IMPORTS"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from rl_core.utils import load_config\n", []),
        ("from rl_core.utils.config import load_config\n", ["rl_core.utils.config.load_config"]),
        ("import rl_core.utils.config\n", ["rl_core.utils.config"]),
        ("from rl_core import agents, utils\n", []),
        ("from rl_core.export import c_header\n", []),
        ("from rl_core.agents import no_such_name\n", ["rl_core.agents.no_such_name"]),
        ("from .utils.config_values import flag\n", ["rl_core.utils.config_values.flag"]),
        ("import numpy\nfrom rl_cartpole.environments import make_env\n", []),
    ],
)
def test_the_private_import_scan(source, expected):
    assert private_imports(source, ["rl_core"]) == expected

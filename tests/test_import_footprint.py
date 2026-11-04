"""Importing the trainer and the challenges must not load the plotting stack."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests.export_cases import make_checkpoint  # noqa: E402
from tests.helpers import minimal_config  # noqa: E402


def test_trainer_and_challenges_do_not_import_matplotlib():
    """Training and benchmarking without plots need no matplotlib; a fresh process shows it.

    A fresh process, because this pytest session has already imported matplotlib.
    """
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "import rl_core.training, rl_core.utils, rl_core.utils.video; "
        "from rl_core.challenges import CHALLENGES, discover_challenges, import_failures_text; "
        "discover_challenges(); "
        "assert CHALLENGES and not import_failures_text(), import_failures_text(); "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib')[:5])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    # A challenge that fails to import is only a warning to discovery, so a failure there would
    # let this test pass without loading that package: the subprocess asserts none failed.
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        "Importing rl_core.training, rl_core.utils and the challenges (with discovery) should not "
        f"load matplotlib; loaded {result.stdout.strip()}, so something on that path imports "
        "rl_core.utils.visualization."
    )


@pytest.mark.parametrize("module", ["train", "record", "export", "benchmark"])
def test_entry_points_do_not_import_matplotlib(module):
    """train.py, record.py, export.py and src/benchmark.py load matplotlib only when they plot: pyplot costs
    about 0.8 s of start-up on the Jetson. A fresh process, as users get the modules."""
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        f"import {module}; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib')[:5])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"Importing {module} loaded {result.stdout.strip()}: import rl_core.utils.visualization "
        "inside the plotting function, not at module level."
    )


def test_discovering_the_challenges_does_not_import_box2d():
    """LunarLander's challenges register without Box2D, which CI does not install: only building
    the environment needs it. A fresh process, because this session may have built one."""
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "from rl_core.challenges import CHALLENGES, discover_challenges, import_failures_text; "
        "discover_challenges(); "
        "assert 'lunarlander' in CHALLENGES, sorted(CHALLENGES); "
        "assert not import_failures_text(), import_failures_text(); "
        "print(sorted(m for m in sys.modules "
        "if m.split('.')[0] == 'Box2D' or m.startswith('gymnasium.envs.box2d')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"Discovering the challenges loaded {result.stdout.strip()}: something imports Box2D at "
        "module level, so the challenges would not register where it is not installed."
    )


ONNX_PACKAGES = ("onnx", "onnxruntime")


@pytest.mark.parametrize("module", ["train", "record", "export", "benchmark"])
def test_entry_points_do_not_import_onnx(module):
    """onnx and onnxruntime are an optional extra, loaded only when an ONNX export asks for them."""
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        f"import {module}; "
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in {ONNX_PACKAGES!r})[:5])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"Importing {module} loaded {result.stdout.strip()}: import onnx and onnxruntime inside the "
        "functions of rl_core/export/onnx_model.py, not at module level."
    )


def test_a_c_export_does_not_import_onnx(tmp_path):
    """export.py --format c runs to the end without loading the ONNX packages: a fresh process."""
    checkpoint = make_checkpoint(tmp_path)
    argv = ["export.py", "--config", minimal_config(tmp_path), "--agent-type", "ppo"]
    argv += ["--format", "c", "--checkpoint", checkpoint, "--output", str(tmp_path / "p.h")]
    argv += ["--verify-episodes", "0"]
    code = (
        f"import sys; sys.path.insert(0, 'src'); sys.argv = {argv!r}; "
        "import export; export.main(); "
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in {ONNX_PACKAGES!r})[:5])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]", (
        f"export.py --format c loaded {result.stdout.strip()}"
    )

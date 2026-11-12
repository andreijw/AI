"""Importing the trainer and the challenges must not load the plotting stack."""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.export_cases import make_checkpoint
from tests.helpers import FIRST_PARTY_DISCOVERY, minimal_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_trainer_and_challenges_do_not_import_matplotlib():
    """Training and benchmarking without plots need no matplotlib; a fresh process shows it.

    A fresh process, because this pytest session has already imported matplotlib.
    """
    code = (
        "import sys; "
        "import rl_core.training, rl_core.utils, rl_core.utils.video; "
        + FIRST_PARTY_DISCOVERY
        + "from rl_core.challenges import CHALLENGES, discover_challenges, import_failures_text; "
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
        "import sys; "
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
        "import sys; "
        + FIRST_PARTY_DISCOVERY
        + "from rl_core.challenges import CHALLENGES, discover_challenges, import_failures_text; "
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
        "import sys; "
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
        f"import sys; sys.argv = {argv!r}; "
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


@pytest.mark.parametrize(
    "module",
    [
        "rl_navigation",
        "rl_navigation.sim",
        "rl_navigation.sim.geometry",
        "rl_navigation.sim.kinematics",
        "rl_navigation.sim.sensors",
        "rl_navigation.sim.layouts",
        "rl_navigation.sim.world",
        "rl_navigation.sim.render",
        "rl_navigation.sim.odometry",
        "rl_navigation.estimation",
        "rl_navigation.estimation.maps",
        "rl_navigation.estimation.kalman",
        "rl_navigation.estimation.ekf",
        "rl_navigation.estimation.evaluation",
        "rl_navigation.estimation.mcl",
        "rl_navigation.estimation.comparison",
        "rl_navigation.estimation.recording",
        "rl_navigation.localization_video",
        "rl_navigation.mapping",
        "rl_navigation.mapping.beams",
        "rl_navigation.mapping.grid",
        "rl_navigation.mapping.evaluation",
        "rl_navigation.slam",
        "rl_navigation.slam.features",
        "rl_navigation.slam.feature_evaluation",
        "rl_navigation.slam.ekf_slam",
        "rl_navigation.slam.evaluation",
    ],
)
def test_navigation_sim_imports_neither_gymnasium_nor_rl_core(module):
    """The simulator is a plain NumPy library that a robot imports: Gymnasium and the training
    framework stay out of it. A fresh process, because this session has imported both."""
    code = (
        "import sys; "
        f"import {module}; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('gymnasium', 'rl_core'))[:5])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"Importing {module} loaded {result.stdout.strip()}: rl_navigation.sim may import only "
        "NumPy and the standard library, and rl_navigation/__init__.py nothing."
    )


def test_navigation_estimation_imports_only_the_simulator():
    """The estimators run on the robot beside the simulator's geometry and nothing else of
    rl_navigation: not the Gymnasium layer, not the challenge's environments. A fresh process."""
    code = (
        "import sys; "
        "import rl_navigation.estimation, rl_navigation.estimation.evaluation, "
        "rl_navigation.estimation.mcl, rl_navigation.estimation.comparison, "
        "rl_navigation.estimation.recording; "
        "print(sorted(m for m in sys.modules if m.startswith('rl_navigation.') "
        "and m.split('.')[1] not in ('sim', 'estimation')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"rl_navigation.estimation loaded {result.stdout.strip()}: it may import NumPy, the "
        "standard library and rl_navigation.sim only."
    )


def _loaded_navigation_packages(imports: str) -> list[str]:
    """The ``rl_navigation`` subpackages a fresh process has loaded after ``imports``."""
    code = (
        "import sys; "
        f"import {imports}; "
        "print(' '.join(sorted({m.split('.')[1] for m in sys.modules "
        "if m.startswith('rl_navigation.')})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def test_the_mapping_core_imports_only_the_simulator():
    """The mapper runs on the robot beside the simulator's geometry: not the estimators, not the
    Gymnasium layer. It is a peer of rl_navigation.estimation, not a layer above it."""
    loaded = _loaded_navigation_packages(
        "rl_navigation.mapping, rl_navigation.mapping.beams, rl_navigation.mapping.grid"
    )
    assert loaded == ["mapping", "sim"], f"the mapping core loaded rl_navigation.{loaded}"


def test_the_mapping_evaluation_desk_tool_imports_only_sim_and_estimation():
    loaded = _loaded_navigation_packages("rl_navigation.mapping.evaluation")
    assert set(loaded) <= {"estimation", "mapping", "sim"}, f"loaded rl_navigation.{loaded}"


def test_the_mapping_package_does_not_import_its_evaluation():
    code = (
        "import sys; import rl_navigation.mapping; "
        "print('rl_navigation.mapping.evaluation' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "rl_navigation/mapping/__init__.py imported evaluation"


def test_the_slam_core_imports_only_the_simulator_and_the_estimation_core():
    """SLAM runs on the robot: a layer above the estimators (whose Kalman update, motion model and
    chi-square quantile it shares), with neither the mapper, the Gymnasium layer nor the
    estimators' desk tools. A fresh process."""
    code = (
        "import sys; "
        "import rl_navigation.slam, rl_navigation.slam.features, rl_navigation.slam.ekf_slam; "
        "print(' '.join(sorted({m.split('.')[1] for m in sys.modules "
        "if m.startswith('rl_navigation.')})), "
        "[m for m in ('rl_navigation.estimation.evaluation', 'rl_navigation.estimation.comparison', "
        "'rl_navigation.estimation.recording') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "estimation sim slam []", (
        f"the slam core loaded {result.stdout.strip()}: it may import NumPy, rl_navigation.sim "
        "and the estimation core only"
    )


@pytest.mark.parametrize(
    "tool", ["rl_navigation.slam.feature_evaluation", "rl_navigation.slam.evaluation"]
)
def test_the_slam_desk_tools_stay_inside_the_navigation_library(tool):
    loaded = _loaded_navigation_packages(tool)
    assert set(loaded) <= {"estimation", "mapping", "sim", "slam"}, f"loaded rl_navigation.{loaded}"


def test_the_slam_package_does_not_import_its_desk_tools():
    code = (
        "import sys; import rl_navigation.slam; "
        "print(any(m in sys.modules for m in ('rl_navigation.slam.feature_evaluation', "
        "'rl_navigation.slam.evaluation', 'rl_navigation.mapping')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "rl_navigation/slam/__init__.py imported a desk tool"


def test_the_slam_package_exports_the_filter():
    import rl_navigation.slam as slam

    assert {"EkfSlam", "SlamSettings", "SlamUpdate"} <= set(slam.__all__)


@pytest.mark.parametrize(
    "module",
    [
        "rl_navigation.estimation",
        "rl_navigation.estimation.evaluation",
        "rl_navigation.mapping",
        "rl_navigation.mapping.evaluation",
        "rl_navigation.gym_env",
        "rl_navigation.environments",
        "rl_navigation.benchmark",
    ],
)
def test_nothing_below_slam_loads_it(module):
    assert "slam" not in _loaded_navigation_packages(module), f"{module} loaded rl_navigation.slam"


@pytest.mark.parametrize(
    "module",
    [
        "rl_navigation.estimation",
        "rl_navigation.estimation.evaluation",
        "rl_navigation.environments",
        "rl_navigation.benchmark",
    ],
)
def test_the_estimators_and_the_challenge_do_not_load_the_mapper(module):
    assert "mapping" not in _loaded_navigation_packages(module), f"{module} loaded the mapper"


@pytest.mark.parametrize("module", ["rl_navigation.environments", "rl_navigation.benchmark"])
def test_the_navigation_challenge_does_not_load_the_estimators(module):
    """The estimators are the robot's library: the challenge (its environments and benchmark)
    trains a policy without them. A fresh process."""
    code = (
        "import sys; "
        f"import {module}; "
        "print(sorted(m for m in sys.modules if m.startswith('rl_navigation.estimation')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"Importing {module} loaded {result.stdout.strip()}."


def test_the_navigation_gymnasium_env_registers_its_ids_without_rl_core():
    """rl_navigation.gym_env is the Gymnasium layer another project can use without the trainer:
    importing it registers both ids and loads no rl_core module. A fresh process."""
    code = (
        "import sys; "
        "import gymnasium as gym; import rl_navigation.gym_env; "
        "env = gym.make('NavigationContinuous-v0'); env.reset(seed=0); env.close(); "
        "assert 'Navigation-v0' in gym.registry, sorted(gym.registry)[-5:]; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'rl_core')[:5])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", (
        f"rl_navigation.gym_env loaded {result.stdout.strip()}: it may import Gymnasium and "
        "rl_navigation.sim only."
    )


def test_importing_rl_navigation_loads_none_of_its_subpackages():
    """rl_navigation/__init__.py imports nothing, so the robot's code pays only for what it uses."""
    code = (
        "import sys; import rl_navigation; "
        "print(sorted(m for m in sys.modules if m.startswith('rl_navigation.')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"import rl_navigation loaded {result.stdout.strip()}."


@pytest.mark.parametrize("tool", ["comparison", "recording"])
def test_the_estimation_package_does_not_import_its_desk_tools(tool):
    """The comparison report and the recording are tools, not part of the library a robot
    imports. A fresh process."""
    code = (
        "import sys; import rl_navigation.estimation; "
        f"print('rl_navigation.estimation.{tool}' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        f"rl_navigation/estimation/__init__.py imported the {tool} module"
    )


VIDEO_STACK = ("matplotlib", "imageio_ffmpeg")


@pytest.mark.parametrize(
    "module", ["rl_navigation.estimation.recording", "rl_navigation.localization_video"]
)
def test_the_localization_recording_loads_no_plotting_or_video_stack(module):
    """The recording paints with NumPy; the command line loads matplotlib and imageio-ffmpeg only
    when it writes stills or the MP4. A fresh process."""
    code = (
        "import sys; "
        f"import {module}; "
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in {VIDEO_STACK!r})[:5])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"Importing {module} loaded {result.stdout.strip()}."

"""Importing the trainer and the challenges must not load the plotting stack."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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

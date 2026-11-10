"""The README names every benchmark challenge and links every published benchmark report."""

import re
from pathlib import Path

import pytest

from rl_core.challenges import discover_challenges

README = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
BENCHMARKS = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"

# The name the front page uses for each registry key. Explicit, since a key is not always the
# name: "mountaincar_continuous" appears as "MountainCarContinuous-v0".
README_NAMES = {
    "acrobot": "Acrobot-v1",
    "bipedalwalker": "BipedalWalker-v3",
    "cartpole": "CartPole-v1",
    "lunarlander": "LunarLander-v3",
    "lunarlander_continuous": "LunarLanderContinuous-v3",
    "mountaincar": "MountainCar-v0",
    "mountaincar_continuous": "MountainCarContinuous-v0",
    "navigation": "Navigation-v0",
    "navigation_continuous": "NavigationContinuous-v0",
    "pendulum": "Pendulum-v1",
}


def test_the_name_map_lists_exactly_the_registered_challenges():
    assert set(README_NAMES) == set(discover_challenges()), (
        "README_NAMES should list every registered challenge and nothing else: a new challenge "
        "needs its row in the README table and its name here"
    )


@pytest.mark.parametrize("key", sorted(README_NAMES))
def test_the_front_page_names_each_challenge(key):
    name = README_NAMES[key]
    # A name counts as a whole word: "Navigation-v0" must not be satisfied by "NavigationContinuous-v0".
    assert re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", README), (
        f"README.md does not name {name} ({key})"
    )


@pytest.mark.parametrize("report", sorted(p.name for p in BENCHMARKS.glob("*.md")))
def test_the_front_page_links_each_benchmark_report(report):
    assert f"docs/benchmarks/{report})" in README, (
        f"README.md does not link docs/benchmarks/{report}"
    )

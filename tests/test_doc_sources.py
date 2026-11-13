"""The teaching pages point at real code and cite their numbers; docs media is listed, linked and small."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import load_script

check_doc_sources = load_script("check_doc_sources")

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET = {"png_kib": 1, "gif_kib": 2, "total_kib": 3}


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _bytes(root: Path, name: str, size: int) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def _manifest(root: Path, media: list, budget: dict | None = None) -> None:
    text = json.dumps({"budget": budget or BUDGET, "media": media})
    _write(root, "scripts/doc_media.json", text)


def _entry(path: str) -> dict:
    return {"path": path, "made_by": "python make.py"}


def _code(root: Path) -> None:
    """A package with a submodule, a subpackage, a re-export and an inherited method."""
    src = "packages/demo/src/pkg"
    _write(
        root,
        f"{src}/__init__.py",
        "from .mod import Child\nfrom . import sub\nfrom .stars import *\n",
    )
    _write(
        root,
        f"{src}/mod.py",
        "import numpy as np\n"
        "from pkg.base import Base\n\n"
        "LIMIT = 3\n"
        "LOW, (HIGH, [MID]) = 0, (2, [1])\n"
        "FIRST, *REST = 1, 2, 3\n"
        "WIDTH: int = 4\n\n"
        "def helper():\n    pass\n\n"
        "class Child(Base):\n"
        "    rate = 0.1\n"
        "    first, second = 1, 2\n\n"
        "    class Inner:\n        depth = 1\n\n"
        "    def __init__(self):\n        self.weights = np.zeros(2)\n"
        "        self.low, [self.high, other] = 0, [1, 2]\n\n"
        "    def act(self):\n        pass\n",
    )
    _write(root, f"{src}/base.py", "class Base:\n    def inherited(self):\n        pass\n")
    _write(root, f"{src}/sub/__init__.py", "")
    _write(root, f"{src}/sub/leaf.py", "def leaf():\n    pass\n")
    _write(root, f"{src}/stars.py", "def starred():\n    pass\n")
    _write(
        root,
        f"{src}/attr.py",
        "import os.path\n"
        "import pkg.base\n"
        "from pkg import base as b\n\n"
        "class Dotted(pkg.base.Base):\n    pass\n\n"
        "class Aliased(b.Base):\n    pass\n",
    )


def _page_problems(root: Path, text: str, changelog: str = "## 2026-10-15\n") -> list[str]:
    _write(root, "CHANGELOG.md", "# Changelog\n\n" + changelog)
    _write(root, "docs/learn/page.md", text)
    return check_doc_sources.check_sources(root)[0]


# --- the repository ----------------------------------------------------------------------------


def test_the_repository_passes():
    problems, pages, media = check_doc_sources.check_sources(REPO_ROOT)
    assert pages > 0 and media > 0, (
        f"read {pages} pages and {media} media files: the scan lost them"
    )
    assert problems == [], "\n".join(problems)


def test_it_runs_isolated_with_the_standard_library_only():
    script = REPO_ROOT / "scripts" / "check_doc_sources.py"
    done = subprocess.run(
        [sys.executable, "-I", str(script)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip().endswith("media files."), done.stdout


def test_main_prints_each_problem_and_fails(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "CHANGELOG.md", "## 2026-10-15\n")
    _write(tmp_path, "docs/learn/page.md", "# P\n\nIt solved 5 of 5.\n")
    monkeypatch.setattr(check_doc_sources, "REPO_ROOT", tmp_path)
    assert check_doc_sources.main() == 1
    out = capsys.readouterr().out
    assert "docs/learn/page.md:3: numbers without a source" in out, out
    assert out.strip().endswith("1 problems over 1 teaching pages and 0 media files."), out


# --- code pointers -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "pkg",
        "pkg.mod",
        "pkg.mod.helper",
        "pkg.mod.LIMIT",
        "pkg.mod.WIDTH",
        "pkg.mod.Child",
        "pkg.mod.Child.act",
        "pkg.mod.Child.rate",
        "pkg.mod.Child.weights",
        "pkg.mod.Child.Inner.depth",
        "pkg.mod.Child.inherited",
        "pkg.Child.act",
        "pkg.sub.leaf.leaf",
        "pkg.sub.leaf",
        "pkg.mod.np",
        "pkg.mod.LOW",
        "pkg.mod.HIGH",
        "pkg.mod.MID",
        "pkg.mod.REST",
        "pkg.mod.Child.first",
        "pkg.mod.Child.second",
        "pkg.mod.Child.low",
        "pkg.mod.Child.high",
        "pkg.starred",
        "pkg.attr.os",
        "pkg.attr.Dotted.inherited",
        "pkg.attr.Aliased.inherited",
    ],
)
def test_a_pointer_to_real_code_passes(tmp_path, name):
    _code(tmp_path)
    assert _page_problems(tmp_path, f"# P\n\nSee `{name}`.\n") == []


@pytest.mark.parametrize(
    "name",
    [
        "pkg.missing",
        "pkg.mod.nothing",
        "pkg.mod.Child.nothing",
        "pkg.mod.helper.inner",
        "pkg.mod.LIMIT.real",
        "pkg.sub.leaf.other",
        "pkg.mod.Child.Inner.nothing",
        "pkg.mod.Child.other",
        "pkg.attr.Dotted.nothing",
        "pkg.unstarred",
    ],
)
def test_a_pointer_to_missing_code_is_reported_with_its_line(tmp_path, name):
    _code(tmp_path)
    problems = _page_problems(tmp_path, f"# P\n\nIntro.\n\nSee `{name}()`.\n")
    assert problems == [
        f"docs/learn/page.md:5: no such module, class or name in the code: {name}"
    ], problems


def test_only_names_of_the_repositorys_packages_are_pointers(tmp_path):
    _code(tmp_path)
    text = (
        "# P\n\n`numpy.nothing`, `training.update_every` and `pkg` alone.\n\n```\n`pkg.gone`\n```\n"
    )
    assert _page_problems(tmp_path, text) == []


@pytest.mark.parametrize("name", ["rl_typo.agents", "rl_cor.agents.returns"])
def test_a_dotted_rl_name_that_is_no_package_is_reported(tmp_path, name):
    _code(tmp_path)
    assert _page_problems(tmp_path, f"# P\n\nSee `{name}`.\n") == [
        f"docs/learn/page.md:3: no package of this repository is named {name.split('.')[0]}: {name}"
    ]


def test_pointers_outside_the_teaching_pages_are_not_checked(tmp_path):
    _code(tmp_path)
    _write(tmp_path, "docs/other.md", "See `pkg.missing`, solved 5 of 5.\n")
    assert check_doc_sources.check_sources(tmp_path)[0] == []


def test_the_real_code_resolves():
    index = check_doc_sources.CodeIndex.of(REPO_ROOT)
    assert index.resolves("rl_core.agents.td3_agent.TD3Agent.save"), "inherited from a mixin"
    assert index.resolves("rl_core.agents.off_policy.OffPolicyAgent.action_low"), "a tuple target"
    assert index.resolves("rl_core.agents.returns.discounted_returns")
    assert not index.resolves("rl_core.agents.returns.not_a_function")


# --- numbers and citations ---------------------------------------------------------------------


def test_a_number_without_a_source_is_reported(tmp_path):
    problems = _page_problems(tmp_path, "# P\n\nPPO solved\n5 of 5 seeds at -110.\n")
    assert len(problems) == 1 and problems[0].startswith("docs/learn/page.md:3: numbers"), problems
    assert "(5, 5, -110)" in problems[0], problems


def test_a_changelog_citation_with_a_real_section_is_a_source(tmp_path):
    text = "# P\n\nPPO solved 5 of 5 (CHANGELOG.md, 2026-10-15).\n"
    assert _page_problems(tmp_path, text) == []


def test_a_month_section_counts_as_a_date(tmp_path):
    text = "# P\n\nREINFORCE came first, 2 agents (CHANGELOG.md, 2026-03).\n"
    assert _page_problems(tmp_path, text, changelog="## 2026-03\n") == []


def test_every_date_of_a_citation_needs_its_section(tmp_path):
    text = "# P\n\nHeld 3 of 5 (CHANGELOG.md, 2026-10-15 and 2026-10-16).\n"
    problems = _page_problems(tmp_path, text)
    assert problems == ["docs/learn/page.md:3: CHANGELOG.md has no '## 2026-10-16' section"]


def test_a_citation_without_a_date_is_reported_even_without_numbers(tmp_path):
    problems = _page_problems(tmp_path, "# P\n\nAs logged (CHANGELOG.md, the PPO entry).\n")
    assert problems == ["docs/learn/page.md:3: CHANGELOG.md has no '## (no date)' section"]


def test_the_clickable_citation_form_is_a_source_and_its_date_is_checked(tmp_path):
    good = "Held 3 of 5 ([CHANGELOG.md, 2026-10-15](../../CHANGELOG.md#2026-10-15))."
    bad = "Held 2 of 5 ([CHANGELOG.md, 2026-10-16](../../CHANGELOG.md#2026-10-16))."
    problems = _page_problems(tmp_path, f"# P\n\n{good}\n\n{bad}\n")
    assert problems == ["docs/learn/page.md:5: CHANGELOG.md has no '## 2026-10-16' section"]


def test_shown_numbers_keep_their_sign_and_lose_trailing_punctuation(tmp_path):
    problems = _page_problems(tmp_path, "# P\n\nFrom -200 to 5, then 0.5.\n")
    assert len(problems) == 1 and "(-200, 5, 0.5)" in problems[0], problems


def test_a_link_to_a_benchmark_report_is_a_source(tmp_path):
    _write(tmp_path, "docs/benchmarks/cartpole_results.md", "# R\n")
    text = "# P\n\nSolved 5 of 5 ([report](../benchmarks/cartpole_results.md#x)).\n"
    assert _page_problems(tmp_path, text) == []


def test_a_link_to_another_page_is_not_a_source(tmp_path):
    _write(tmp_path, "docs/results.md", "# R\n")
    problems = _page_problems(tmp_path, "# P\n\nSolved 5 of 5 ([results](../results.md)).\n")
    assert len(problems) == 1 and "numbers without a source" in problems[0], problems


def test_each_list_item_and_table_row_needs_its_own_source(tmp_path):
    text = (
        "# P\n\n"
        "- held 4 of 5 (CHANGELOG.md, 2026-10-15)\n"
        "- held 3 of 5\n"
        "  over two lines\n\n"
        "| Agent | Seeds |\n"
        "| :--- | :--- |\n"
        "| PPO | 5 (CHANGELOG.md, 2026-10-15) |\n"
        "| DQN | 4 |\n"
    )
    problems = _page_problems(tmp_path, text)
    lines = [problem.split(":")[1] for problem in problems]
    assert lines == ["4", "10"], problems


@pytest.mark.parametrize(
    "text",
    [
        "CartPole-v1, LunarLander-v3, Box2D, SB3, x86-64, rl-zoo and a 3D view",
        "the code `gamma = 0.99` says so",
        "a link [here](runs/500.md) and <!-- note 42 -->",
        "four of five seeds",
        '<img src="../images/f1.png" alt="frame 381" width="300">',
        "![frame 381](../images/f1.png) and ![seed 101][still]\n\n[still]: ../images/s2.png",
    ],
)
def test_names_code_links_and_words_are_not_numerals(tmp_path, text):
    assert _page_problems(tmp_path, f"# P {7}\n\n## Phase 8\n\n{text}.\n") == []


@pytest.mark.parametrize("number", ["475", "-110", "0.99", "1e-3", "200k", "1,000", "2026-11-12"])
def test_numbers_in_their_usual_forms_are_numerals(tmp_path, number):
    problems = _page_problems(tmp_path, f"# P\n\nIt reached {number} here.\n")
    assert len(problems) == 1, f"{number} should need a source: {problems}"


def test_an_ordered_list_marker_is_not_a_numeral(tmp_path):
    assert _page_problems(tmp_path, "# P\n\n1. Collect.\n2. Update.\n") == []


# --- media -------------------------------------------------------------------------------------


def _media_problems(root: Path) -> list[str]:
    return check_doc_sources.check_media(root)[0]


def test_a_listed_linked_image_within_budget_passes(tmp_path):
    _bytes(tmp_path, "docs/images/a.png", 1024)
    _bytes(tmp_path, "docs/images/cartpole/b.gif", 2048)
    _manifest(tmp_path, [_entry("docs/images/a.png"), _entry("docs/images/cartpole/b.gif")])
    _write(tmp_path, "README.md", "![a](docs/images/a.png)\n")
    _write(tmp_path, "docs/learn/x.md", "![b](../images/cartpole/b.gif)\n")
    assert check_doc_sources.check_media(tmp_path) == ([], 2)


def test_an_unlisted_unlinked_image_is_reported_twice(tmp_path):
    _bytes(tmp_path, "docs/images/a.png", 10)
    _manifest(tmp_path, [])
    assert _media_problems(tmp_path) == [
        "docs/images/a.png: not listed in scripts/doc_media.json",
        "docs/images/a.png: no Markdown page links it",
    ]


def test_a_listed_file_that_does_not_exist_is_reported(tmp_path):
    _manifest(tmp_path, [_entry("docs/images/gone.png")])
    assert _media_problems(tmp_path) == [
        "scripts/doc_media.json: docs/images/gone.png is listed but does not exist"
    ]


@pytest.mark.parametrize(
    ("name", "size", "message"),
    [
        ("a.png", 1025, "docs/images/a.png: 1 KiB, over the 1 KiB budget"),
        ("a.gif", 2049, "docs/images/a.gif: 2 KiB, over the 2 KiB budget"),
        ("a.mp4", 10, "docs/images/a.mp4: only .png and .gif belong in docs/images/"),
    ],
)
def test_a_file_over_its_budget_or_of_another_type_is_reported(tmp_path, name, size, message):
    _bytes(tmp_path, f"docs/images/{name}", size)
    _manifest(tmp_path, [_entry(f"docs/images/{name}")])
    _write(tmp_path, "README.md", f"[v](docs/images/{name})\n")
    assert _media_problems(tmp_path) == [message]


def test_the_total_budget_covers_every_file(tmp_path):
    names = ["a.gif", "b.gif"]
    for name in names:
        _bytes(tmp_path, f"docs/images/{name}", 2000)
    _manifest(tmp_path, [_entry(f"docs/images/{name}") for name in names])
    _write(tmp_path, "README.md", "".join(f"![x](docs/images/{name})\n" for name in names))
    assert _media_problems(tmp_path) == ["docs/images/: 3 KiB in all, over the 3 KiB budget"]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            "{not json",
            [
                "scripts/doc_media.json: not valid JSON: Expecting property name enclosed in "
                "double quotes: line 1 column 2 (char 1)"
            ],
        ),
        ("[]", ["scripts/doc_media.json: must be a JSON object"]),
        ('{"media": []}', ["scripts/doc_media.json: needs a 'budget' object and a 'media' list"]),
        (
            '{"budget": {"png_kib": 1}, "media": []}',
            ["scripts/doc_media.json: budget needs positive integers png_kib, gif_kib, total_kib"],
        ),
        (
            json.dumps({"budget": BUDGET, "media": [{"path": "docs/images/a.png"}]}),
            [
                "scripts/doc_media.json: media[0] needs a 'path' and a 'made_by' string",
                "docs/images/a.png: not listed in scripts/doc_media.json",
            ],
        ),
        (
            json.dumps({"budget": BUDGET, "media": [_entry("docs/images/a.png")] * 2}),
            ["scripts/doc_media.json: docs/images/a.png is listed twice"],
        ),
    ],
)
def test_a_malformed_manifest_is_reported(tmp_path, manifest, message):
    _bytes(tmp_path, "docs/images/a.png", 10)
    _write(tmp_path, "README.md", "![a](docs/images/a.png)\n")
    _write(tmp_path, "scripts/doc_media.json", manifest)
    assert _media_problems(tmp_path) == message


def test_an_image_linked_by_an_html_tag_or_a_reference_counts_as_linked(tmp_path):
    for name in ("a.png", "b.png"):
        _bytes(tmp_path, f"docs/images/{name}", 10)
    _manifest(tmp_path, [_entry("docs/images/a.png"), _entry("docs/images/b.png")])
    _write(tmp_path, "README.md", '<img src="docs/images/a.png" width="300">\n')
    _write(tmp_path, "docs/x.md", "![b][still]\n\n[still]: images/b.png\n")
    assert _media_problems(tmp_path) == []


@pytest.mark.parametrize("name", ["docs/a.png", "docs/learn/b.gif", "packages/x/src/c.png"])
def test_media_outside_docs_images_is_reported(tmp_path, name):
    _bytes(tmp_path, name, 10)
    _manifest(tmp_path, [])
    _write(tmp_path, "README.md", f"![x]({name})\n")
    assert _media_problems(tmp_path) == [f"{name}: documentation media belongs in docs/images/"]


def test_images_without_a_manifest_are_reported(tmp_path):
    _bytes(tmp_path, "docs/images/a.png", 10)
    assert _media_problems(tmp_path) == ["scripts/doc_media.json: missing"]


def test_no_images_and_no_manifest_is_fine(tmp_path):
    assert check_doc_sources.check_media(tmp_path) == ([], 0)

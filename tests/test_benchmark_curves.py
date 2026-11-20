"""Where benchmark.py writes its learning-curve PNG, and that a committed one fits the docs budget.

The tracked report (``docs/benchmarks/<challenge>_results.md`` under the repository root) links a
curve PNG committed under ``docs/images/benchmarks/`` and listed in ``scripts/doc_media.json``;
every other report (a scratch or smoke run) keeps its plot in the output folder, so it can never
rewrite a tracked image. Each test points ``benchmark.REPO_ROOT`` at a temporary folder with its
own copy of ``configs/``: the real ``docs/`` and ``scripts/`` are never written.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

import benchmark
from benchmark import BenchmarkSettings, main, run_benchmark, write_curves_png
from tests.helpers import load_script, needs_matplotlib

REAL_ROOT = Path(__file__).resolve().parents[1]
TRACKED_REPORT = Path("docs/benchmarks/cartpole_results.md")
TRACKED_CURVES = Path("docs/images/benchmarks/cartpole_learning_curves.png")
CURVES_LINK = "../images/benchmarks/cartpole_learning_curves.png"
MANIFEST = Path("scripts/doc_media.json")
OTHER_ENTRY = {"path": "docs/images/other.png", "made_by": "by hand", "added_in": "abc1234"}
LAST_ENTRY = {"path": "docs/images/last.png", "made_by": "by hand"}


def _png_budget() -> int:
    """The repository's PNG budget in bytes, as the real scripts/doc_media.json sets it."""
    manifest = json.loads((REAL_ROOT / MANIFEST).read_text(encoding="utf-8"))
    return int(manifest["budget"]["png_kib"]) * 1024


def _write_manifest(root: Path, media: list[dict[str, str]], png_kib: int = 60) -> None:
    (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    manifest = {"budget": {"png_kib": png_kib, "gif_kib": 600, "total_kib": 6144}, "media": media}
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _media(root: Path) -> list[dict[str, str]]:
    return json.loads((root / MANIFEST).read_text(encoding="utf-8"))["media"]


def _files_under(folder: Path) -> list[str]:
    return sorted(p.as_posix() for p in folder.rglob("*") if p.is_file()) if folder.exists() else []


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A temporary repository root with configs and a docs media manifest, as the current folder."""
    root = tmp_path.resolve()
    shutil.copytree(REAL_ROOT / "configs", root / "configs")
    _write_manifest(root, [OTHER_ENTRY])
    monkeypatch.setattr(benchmark, "REPO_ROOT", root)
    monkeypatch.chdir(root)
    return root


@pytest.fixture()
def cli(repo, monkeypatch):
    """Run main() as one random-agent episode on CartPole, with extra flags."""

    def run(*flags: str) -> None:
        argv = ["benchmark", "--agents", "random", "--episodes", "1", "--seeds", "1"]
        argv += ["--eval-episodes", "1", "--config-dir", str(repo / "configs"), *flags]
        monkeypatch.setattr(sys, "argv", argv)
        main()

    return run


def _assert_tracked(repo: Path) -> None:
    assert (repo / TRACKED_CURVES).is_file()
    report = (repo / TRACKED_REPORT).read_text(encoding="utf-8")
    assert report.endswith(f"![Benchmark Learning Curves]({CURVES_LINK})\n")
    assert ((repo / TRACKED_REPORT).parent / CURVES_LINK).resolve() == (repo / TRACKED_CURVES)
    assert not list((repo / "plots").rglob("*.png")), "A tracked run plots once, under docs/."


@needs_matplotlib
@pytest.mark.parametrize("report", [None, "./docs/benchmarks/cartpole_results.md", "absolute"])
def test_the_tracked_report_links_a_curve_png_under_docs_images(repo, cli, report):
    """The default report, or the same file named relatively or absolutely, is the tracked one."""
    if report == "absolute":
        report = str(repo / TRACKED_REPORT)
    cli(*([] if report is None else ["--report", report]))
    _assert_tracked(repo)


@needs_matplotlib
def test_a_symlinked_report_path_to_the_tracked_report_is_tracked(repo, cli):
    (repo / "docs" / "benchmarks").mkdir(parents=True)
    (repo / "alias").symlink_to(repo / "docs", target_is_directory=True)
    cli("--report", "alias/benchmarks/cartpole_results.md")
    _assert_tracked(repo)


@needs_matplotlib
def test_a_run_from_a_subfolder_writes_the_tracked_report_at_the_repository_root(
    repo, cli, monkeypatch
):
    (repo / "sub").mkdir()
    monkeypatch.chdir(repo / "sub")
    cli()
    _assert_tracked(repo)
    assert not (repo / "sub" / "docs").exists()


@needs_matplotlib
def test_the_tracked_run_appends_its_manifest_entry_once(repo, cli):
    """A new curve PNG is listed last, naming the command; a rerun does not list it again."""
    cli()
    cli()

    media = _media(repo)
    assert [entry["path"] for entry in media] == [OTHER_ENTRY["path"], TRACKED_CURVES.as_posix()]
    assert media[0] == OTHER_ENTRY, "Other entries stay as they were."
    made_by = media[1]["made_by"]
    assert made_by == (
        "python benchmark.py --challenge cartpole --agents random --seeds 1 --episodes 1"
        f" --window 100 --eval-episodes 1 (the curves of {TRACKED_REPORT.as_posix()})"
    ), "A config folder inside the repository named 'configs' is the default and not written."


@needs_matplotlib
def test_a_rerun_updates_the_entry_in_place_and_keeps_its_other_keys(repo, cli):
    old = {"path": TRACKED_CURVES.as_posix(), "made_by": "an older run", "added_in": "def5678"}
    _write_manifest(repo, [OTHER_ENTRY, old, LAST_ENTRY])
    cli()

    media = _media(repo)
    assert media[0] == OTHER_ENTRY and media[2] == LAST_ENTRY
    assert media[1]["path"] == old["path"] and media[1]["added_in"] == "def5678"
    assert media[1]["made_by"].startswith("python benchmark.py --challenge cartpole")


@needs_matplotlib
def test_a_config_folder_inside_the_repository_is_named_relative_to_it(repo, cli):
    shutil.copytree(repo / "configs", repo / "my_configs")
    cli("--config-dir", str(repo / "my_configs"))
    made_by = _media(repo)[-1]["made_by"]
    assert " --config-dir my_configs (the curves of" in made_by


@needs_matplotlib
def test_a_config_folder_outside_the_repository_warns_of_a_machine_path(
    repo, cli, tmp_path_factory, capsys
):
    outside = tmp_path_factory.mktemp("elsewhere") / "configs"
    shutil.copytree(repo / "configs", outside)
    cli("--config-dir", str(outside))
    assert f"--config-dir {outside.resolve()} (the curves of" in _media(repo)[-1]["made_by"]
    assert "outside the repository" in capsys.readouterr().out


@needs_matplotlib
def test_the_manifest_is_written_only_after_the_report(repo, cli):
    """A run that cannot write its report leaves the manifest as it was."""
    before = (repo / MANIFEST).read_bytes()
    (repo / TRACKED_REPORT).mkdir(parents=True)  # a folder where the report should go
    with pytest.raises(IsADirectoryError):
        cli()
    assert (repo / MANIFEST).read_bytes() == before


@needs_matplotlib
def test_the_manifest_is_replaced_whole_or_not_at_all(repo, cli, monkeypatch):
    """Written to a temporary file and moved over the manifest, so a failed write corrupts none."""
    before = (repo / MANIFEST).read_bytes()

    def failing_replace(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(benchmark.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        cli()
    assert (repo / MANIFEST).read_bytes() == before
    assert sorted(p.name for p in (repo / "scripts").iterdir()) == ["doc_media.json"]


@needs_matplotlib
def test_a_scratch_report_writes_nothing_under_docs(repo, cli):
    """A smoke run with --report elsewhere keeps its plot in the output folder and links it there."""
    before = (repo / MANIFEST).read_bytes()
    cli("--output-dir", "out", "--report", "s/report.md")

    assert _files_under(repo / "docs") == []
    assert (repo / MANIFEST).read_bytes() == before
    assert (repo / "out" / "benchmark_comparison.png").is_file()
    report = (repo / "s" / "report.md").read_text(encoding="utf-8")
    assert report.endswith("![Benchmark Learning Curves](../out/benchmark_comparison.png)\n")


@needs_matplotlib
def test_a_scratch_report_through_the_python_api_writes_nothing_under_docs(repo):
    run_benchmark(
        settings=BenchmarkSettings(
            agents=("random",), seeds=(1,), episodes=1, eval_episodes=1, output_dir="out"
        ),
        config_dir=str(repo / "configs"),
        report_path="report.md",
    )
    assert _files_under(repo / "docs") == []
    assert (repo / "out" / "benchmark_comparison.png").is_file()


def test_no_plot_writes_no_png_and_leaves_the_manifest(repo, cli):
    before = (repo / MANIFEST).read_bytes()
    cli("--no-plot")

    assert _files_under(repo / "docs") == [(repo / TRACKED_REPORT).as_posix()]
    assert "Learning Trajectories" not in (repo / TRACKED_REPORT).read_text(encoding="utf-8")
    assert (repo / MANIFEST).read_bytes() == before
    assert not list(repo.rglob("*.png"))


def test_no_plot_keeps_linking_curves_an_earlier_tracked_run_committed(repo, cli):
    """The report still shows the committed curves, says they are older, and nothing is redrawn."""
    (repo / TRACKED_CURVES).parent.mkdir(parents=True)
    (repo / TRACKED_CURVES).write_bytes(b"earlier curves")
    _write_manifest(repo, [OTHER_ENTRY, {"path": TRACKED_CURVES.as_posix(), "made_by": "earlier"}])
    before = (repo / MANIFEST).read_bytes()
    cli("--no-plot")

    report = (repo / TRACKED_REPORT).read_text(encoding="utf-8")
    assert report.endswith(
        "\n## Learning Trajectories\n\n"
        "The curves are from an earlier run: this one was run with `--no-plot`.\n\n"
        f"![Benchmark Learning Curves]({CURVES_LINK})\n"
    )
    assert (repo / TRACKED_CURVES).read_bytes() == b"earlier curves"
    assert (repo / MANIFEST).read_bytes() == before


@needs_matplotlib
def test_without_a_manifest_the_tracked_run_still_plots_and_says_so(repo, cli, capsys):
    (repo / MANIFEST).unlink()
    cli()

    assert (repo / TRACKED_CURVES).is_file()
    assert not (repo / MANIFEST).exists()
    assert f"{MANIFEST.as_posix()} not found" in capsys.readouterr().out


@needs_matplotlib
def test_the_benchmark_png_is_a_small_palette_image(repo, cli):
    """The 1-episode run's own PNG: palette colours at the docs size, within the budget."""
    from PIL import Image

    cli()
    path = repo / TRACKED_CURVES
    assert path.stat().st_size <= _png_budget()
    with Image.open(path) as image:
        assert image.mode == "P", "Quantised to a palette, which keeps line plots small."
        assert image.size == (1100, 600), "11 x 6 inches at the docs resolution of 100 dpi."
        assert len(image.getcolors() or []) <= 64


@needs_matplotlib
def test_a_full_size_benchmark_plot_fits_the_png_budget(tmp_path):
    """Seven agents, five seeds, 3,000 noisy episodes each: a real report's worst case."""
    rng = np.random.default_rng(0)
    curves = {
        f"agent_{i}": [list(rng.normal(np.linspace(-500, 300, 3000), 300)) for _ in range(5)]
        for i in range(7)
    }
    path = write_curves_png(
        str(tmp_path / "curves.png"), curves, window=100, title="Worst case", target_score=200.0
    )
    assert os.path.getsize(path) <= _png_budget()


class SizedByResolution:
    """Stands in for write_curves_png: writes ``dpi * colours`` bytes and records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(
        self, path, curves, *, dpi=benchmark.CURVE_DPI, colours=benchmark.CURVE_COLOURS, **options
    ) -> str:
        self.calls.append((dpi, colours))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"x" * (dpi * colours))
        return os.path.abspath(path)


def test_an_over_budget_tracked_png_is_redrawn_smaller_until_it_fits(repo, cli, monkeypatch):
    """Fewer colours first, then a lower dpi: 6,400 and 3,200 bytes are over 3 KiB, 2,560 fits."""
    fake = SizedByResolution()
    monkeypatch.setattr(benchmark, "write_curves_png", fake)
    _write_manifest(repo, [OTHER_ENTRY], png_kib=3)
    cli()

    assert fake.calls == [(100, 64), (100, 32), (80, 32)]
    assert (repo / TRACKED_CURVES).stat().st_size == 2560
    assert (repo / TRACKED_REPORT).is_file()


def test_a_scratch_png_is_drawn_once_whatever_the_budget(repo, cli, monkeypatch):
    fake = SizedByResolution()
    monkeypatch.setattr(benchmark, "write_curves_png", fake)
    _write_manifest(repo, [OTHER_ENTRY], png_kib=1)
    cli("--output-dir", "out", "--report", "s/report.md")
    assert fake.calls == [(100, 64)]


@needs_matplotlib
def test_a_tracked_png_that_cannot_fit_stops_before_the_report_and_manifest(repo, cli, capsys):
    """A real 1-episode plot is over a 1 KiB budget even at 60 dpi and 32 colours."""
    _write_manifest(repo, [OTHER_ENTRY], png_kib=1)
    before = (repo / MANIFEST).read_bytes()
    with pytest.raises(SystemExit) as stopped:
        cli()

    assert stopped.value.code == 1
    err = capsys.readouterr().err
    assert TRACKED_CURVES.as_posix() in err and "over the 1 KiB budget" in err
    assert "60 dpi and 32 colours" in err
    assert not (repo / TRACKED_REPORT).exists()
    assert (repo / MANIFEST).read_bytes() == before
    assert (repo / "plots" / "benchmarks" / "cartpole" / "benchmark_metrics.json").is_file(), (
        "The run's numbers are kept in the metrics JSON."
    )


@needs_matplotlib
def test_a_tracked_run_leaves_the_docs_media_check_passing(repo, cli):
    """The PNG is linked by the report, listed in the manifest and within budget, as CI checks."""
    check_doc_sources = load_script("check_doc_sources")
    _write_manifest(repo, [])
    cli()

    assert check_doc_sources.check_media(repo) == ([], 1)

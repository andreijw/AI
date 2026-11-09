"""rl_navigation.localization_video: the command line that writes a localization trial's MP4 and
stills, and its video writer."""

import sys

import numpy as np
import pytest

from rl_navigation import localization_video as video_module
from rl_navigation.estimation import recording as recording_module
from tests.helpers import needs_video

# A short, cheap run: 12 steps kidnapped at 6, 100 particles, 20 px/m (a 100 x 100 arena). A flag
# given again later replaces its value here.
SMALL = ["--seed", "7", "--steps", "12", "--kidnap-at", "6", "--particles", "100"]
SMALL += ["--pixels-per-metre", "20"]


@pytest.fixture
def frames_written(monkeypatch):
    """Replace the MP4 writer: keep the frames it was given and the file it would have written."""
    written = {}

    def fake_write_video(path, frames, fps):
        written["path"], written["frames"], written["fps"] = path, list(frames), fps
        return len(written["frames"])

    monkeypatch.setattr(video_module, "write_video", fake_write_video)
    # CI has no imageio-ffmpeg: main() checks for it before running, and nothing here writes an MP4.
    monkeypatch.setattr(video_module, "require_imageio_ffmpeg", lambda: None)
    return written


@pytest.fixture
def trials_run(monkeypatch):
    """Record record_trials' arguments, still running it."""
    calls = []
    real = video_module.record_trials

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(video_module, "record_trials", spy)
    return calls


def test_the_cli_needs_an_output(monkeypatch, capsys):
    monkeypatch.setattr(video_module, "record_trials", None)  # nothing may run
    with pytest.raises(SystemExit):
        video_module.main(SMALL)
    assert "--output" in capsys.readouterr().err


def test_the_cli_refuses_to_overwrite_a_file(tmp_path, monkeypatch, capsys):
    existing = tmp_path / "taken.mp4"
    existing.write_bytes(b"keep me")
    monkeypatch.setattr(video_module, "record_trials", None)
    with pytest.raises(SystemExit):
        video_module.main([*SMALL, "--output", str(existing)])
    assert "exists" in capsys.readouterr().err
    assert existing.read_bytes() == b"keep me"


def test_the_cli_refuses_a_dangling_symlink_at_the_output(tmp_path, monkeypatch, capsys):
    """A link to a missing file does not exist(), but writing through it would create its target."""
    link = tmp_path / "link.mp4"
    link.symlink_to(tmp_path / "missing" / "target.mp4")
    monkeypatch.setattr(video_module, "record_trials", None)
    with pytest.raises(SystemExit):
        video_module.main([*SMALL, "--output", str(link)])
    assert "exists" in capsys.readouterr().err
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("which", ["output", "stills"])
def test_the_cli_refuses_a_file_where_a_directory_must_go(tmp_path, monkeypatch, capsys, which):
    """Checked before the trial runs, not when the frames are written after it."""
    blocker = tmp_path / "a_file"
    blocker.write_text("")
    output = blocker / "run.mp4" if which == "output" else tmp_path / "run.mp4"
    argv = [*SMALL, "--output", str(output)]
    if which == "stills":
        argv += ["--stills", "1", "--stills-dir", str(blocker)]
    monkeypatch.setattr(video_module, "record_trials", None)
    with pytest.raises(SystemExit):
        video_module.main(argv)
    assert "not a directory" in capsys.readouterr().err


def test_the_cli_refuses_a_still_that_is_a_dangling_symlink(tmp_path, monkeypatch, capsys):
    stills = tmp_path / "stills"
    stills.mkdir()
    output = tmp_path / "run.mp4"
    video_module.still_path(stills, output, 3).symlink_to(tmp_path / "elsewhere.png")
    monkeypatch.setattr(video_module, "record_trials", None)
    with pytest.raises(SystemExit):
        video_module.main(
            [*SMALL, "--output", str(output), "--stills", "3", "--stills-dir", str(stills)]
        )
    assert "overwrite" in capsys.readouterr().err
    assert not (tmp_path / "elsewhere.png").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--stills", "13", "--stills-dir", "s"], "--stills"),
        (["--stills", "-1", "--stills-dir", "s"], "--stills"),
        (["--stills", "3"], "--stills-dir"),
        (["--kidnap-at", "12"], "kidnap"),
        (["--kidnap-at", "0"], "kidnap"),
        (["--mcl-scale", "1", "0", "1"], "scale"),
        (["--dropout", "1.5"], "dropout"),
        (["--steps", "0"], "--steps"),
        (["--pixels-per-metre", "0"], "--pixels-per-metre"),
        (["--pixels-per-metre", "nan"], "--pixels-per-metre"),
        (["--fps", "-5"], "--fps"),
        (["--fps", "0"], "--fps"),
    ],
)
def test_the_cli_refuses_bad_arguments_before_running(
    tmp_path, monkeypatch, capsys, extra, message
):
    monkeypatch.setattr(video_module, "record_trials", None)  # nothing may run
    with pytest.raises(SystemExit):
        video_module.main([*SMALL, "--output", str(tmp_path / "out.mp4"), *extra])
    assert message in capsys.readouterr().err


def test_the_cli_writes_one_frame_per_snapshot_and_prints_the_legend_and_summary(
    tmp_path, frames_written, trials_run, capsys
):
    output = tmp_path / "run.mp4"
    video_module.main([*SMALL, "--output", str(output), "--fps", "5"])
    ((args, kwargs),) = trials_run
    assert args[0] == 7 and kwargs["steps"] == 12 and kwargs["kidnap_at"] == 6
    assert kwargs["mcl_settings"].num_particles == 100
    assert frames_written["path"] == output and frames_written["fps"] == 5
    assert len(frames_written["frames"]) == 13
    out = capsys.readouterr().out
    for line in recording_module.LEGEND:
        assert line in out
    assert "12 steps" in out and "kidnapped at step 6" in out


def test_the_cli_runs_without_a_kidnapping_and_passes_the_mcl_scale(
    tmp_path, frames_written, trials_run, capsys
):
    argv = [*SMALL, "--kidnap-at", "none", "--mcl-scale", "0.13", "0.124", "0.308"]
    argv += ["--dropout", "0.3", "--sensor", "hc_sr04", "--output", str(tmp_path / "a.mp4")]
    video_module.main(argv)
    ((args, kwargs),) = trials_run
    assert kwargs["kidnap_at"] is None and kwargs["dropout"] == 0.3
    assert kwargs["mcl_settings"].covariance_scale == (0.13, 0.124, 0.308)
    assert args[1].num_readings == 5, "the HC-SR04's five sonars"
    assert "no kidnapping" in capsys.readouterr().out


def test_the_cli_writes_stills_into_a_new_directory(tmp_path, frames_written):
    from matplotlib.image import imread

    stills = tmp_path / "new" / "stills"
    argv = [*SMALL, "--output", str(tmp_path / "run.mp4")]
    video_module.main([*argv, "--stills", "0", "7", "12", "--stills-dir", str(stills)])
    names = sorted(path.name for path in stills.iterdir())
    assert names == ["run_step0000.png", "run_step0007.png", "run_step0012.png"]
    image = imread(stills / "run_step0007.png")
    expected = frames_written["frames"][7]
    assert image.shape[:2] == expected.shape[:2]
    np.testing.assert_array_equal(np.round(image[..., :3] * 255).astype(np.uint8), expected)


def test_write_video_names_the_video_extra_when_imageio_ffmpeg_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # import fails
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ImportError, match=r"\.\[video\]"):
        video_module.write_video(tmp_path / "a.mp4", iter([frame]), fps=10)
    assert not (tmp_path / "a.mp4").exists()


def test_the_cli_checks_for_imageio_ffmpeg_before_running(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(video_module, "record_trials", None)  # nothing may run
    with pytest.raises(SystemExit):
        video_module.main([*SMALL, "--output", str(tmp_path / "a.mp4")])
    assert ".[video]" in capsys.readouterr().err


def test_pad_to_even_pads_odd_sides_with_the_edge_and_keeps_even_ones():
    frame = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    padded = video_module.pad_to_even(frame)
    assert padded.shape == (6, 8, 3)
    np.testing.assert_array_equal(padded[:5, :7], frame)
    np.testing.assert_array_equal(padded[5, :7], frame[4])
    np.testing.assert_array_equal(padded[:5, 7], frame[:, 6])
    even = np.zeros((4, 6, 3), dtype=np.uint8)
    assert video_module.pad_to_even(even) is even


@needs_video
def test_write_video_writes_an_h264_mp4_padded_to_even_sides(tmp_path):
    import imageio_ffmpeg

    frames = [np.full((21, 33, 3), 40 * k, dtype=np.uint8) for k in range(5)]
    path = tmp_path / "odd.mp4"
    assert video_module.write_video(path, iter(frames), fps=10) == 5
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()
    count = sum(1 for _ in reader)
    assert tuple(meta["size"]) == (34, 22)
    assert count == 5


@needs_video
def test_the_cli_writes_an_mp4(tmp_path):
    output = tmp_path / "run.mp4"
    video_module.main([*SMALL, "--output", str(output)])
    assert output.stat().st_size > 0

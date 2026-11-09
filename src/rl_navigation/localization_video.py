"""Write one seeded localization trial, the EKF against MCL, to an MP4 and optional PNG stills.

The frames come from :mod:`rl_navigation.estimation.recording`; this module adds the files::

    python -m rl_navigation.localization_video --seed 101 --sensor lidar \\
        --mcl-scale 0.130 0.124 0.308 --output scratch/seed101.mp4 \\
        --stills 0 301 320 380 --stills-dir scratch

``--output`` is required and never overwritten. The MP4 needs imageio-ffmpeg (the ``video`` extra)
and the stills matplotlib; both are imported only when used. Nothing in ``rl_navigation.sim`` or
``rl_navigation.estimation`` imports this module.
"""

import argparse
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np

from rl_navigation.estimation.comparison import SENSORS, STEP_SECONDS
from rl_navigation.estimation.evaluation import LOCALIZE_DEADLINE, LOCALIZE_HOLD, localization_steps
from rl_navigation.estimation.mcl import MclSettings
from rl_navigation.estimation.poses import checked_axis_values
from rl_navigation.estimation.recording import (
    LEGEND,
    PIXELS_PER_METRE,
    Recording,
    record_trials,
    render_frames,
)
from rl_navigation.sim.render import Image

VIDEO_EXTRA = 'pip install -e ".[video]"'


def require_imageio_ffmpeg() -> ModuleType:
    """The imageio_ffmpeg module, or an ImportError that names the extra to install."""
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise ImportError(
            f"Writing an MP4 needs imageio-ffmpeg, from the video extra: {VIDEO_EXTRA}."
        ) from error
    return imageio_ffmpeg


def pad_to_even(frame: Image) -> Image:
    """``frame`` with its last row and column repeated where a side is odd (H.264 in yuv420p
    needs even sides); ``frame`` itself when both are even."""
    rows, cols = frame.shape[:2]
    if rows % 2 == 0 and cols % 2 == 0:
        return frame
    return np.pad(frame, ((0, rows % 2), (0, cols % 2), (0, 0)), mode="edge")


def write_video(path: Path, frames: Iterable[Image], fps: float) -> int:
    """Write RGB uint8 frames of one size to an H.264 MP4 at ``fps``, padded to even sides.

    Returns:
        The number of frames written (none writes no file).

    Raises:
        ImportError: imageio-ffmpeg is not installed.
        ValueError: A frame of a different size from the first.
    """
    ffmpeg = require_imageio_ffmpeg()
    writer = None
    size = (0, 0)  # width, height of the first frame
    count = 0
    try:
        for frame in frames:
            padded = pad_to_even(frame)
            if writer is None:
                size = (padded.shape[1], padded.shape[0])
                # macro_block_size=1: the frame is already even, so ffmpeg must not rescale it.
                writer = ffmpeg.write_frames(
                    str(path),
                    size,
                    fps=fps,
                    macro_block_size=1,
                    pix_fmt_in="rgb24",
                    pix_fmt_out="yuv420p",
                    ffmpeg_log_level="error",
                )
                writer.send(None)  # start the generator
            elif (padded.shape[1], padded.shape[0]) != size:
                raise ValueError(
                    f"write_video: frame {count} is {padded.shape[1]} x {padded.shape[0]}, "
                    f"not {size[0]} x {size[1]} as the first."
                )
            writer.send(np.ascontiguousarray(padded))
            count += 1
    finally:
        if writer is not None:
            writer.close()
    return count


def still_path(directory: Path, output: Path, frame: int) -> Path:
    """Where frame ``frame``'s still goes: ``<directory>/<output stem>_step<frame:04d>.png``."""
    return directory / f"{output.stem}_step{frame:04d}.png"


def _taken(path: Path) -> bool:
    """Whether writing ``path`` would replace something: a file, or a link (even a dangling one,
    whose target the write would create)."""
    return path.exists() or path.is_symlink()


def _blocked(directory: Path) -> bool:
    """Whether ``directory`` or a parent of it is something other than a directory, so it cannot be
    created or written into."""
    return any(path.exists() and not path.is_dir() for path in (directory, *directory.parents))


def _saving_stills(
    frames: Iterator[Image], stills: Sequence[int], directory: Path, output: Path
) -> Iterator[Image]:
    """``frames`` unchanged, saving those numbered in ``stills`` as PNGs on the way."""
    wanted = set(stills)
    if wanted:
        from matplotlib.image import imsave

        directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        if index in wanted:
            imsave(still_path(directory, output, index), frame)
        yield frame


def _dropout(text: str) -> float | None:
    """``own`` for the sensor's own dropout, else a probability in ``[0, 1]``."""
    if text == "own":
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"dropout must be 'own' or a number: {text!r}") from error
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"dropout must be in [0, 1], got {value}")
    return value


def _kidnap_at(text: str) -> int | None:
    """``none`` for no kidnapping, else a step."""
    return None if text == "none" else int(text)


def summary(recording: Recording) -> str:
    """One line: the steps run, a collision, and how each filter localized or recovered
    (:func:`~rl_navigation.estimation.evaluation.localization_steps`)."""
    steps = len(recording.true_poses)
    head = f"{steps} steps" + (", ended by a collision" if recording.collided else "")
    kidnap_at = recording.kidnap_at
    if kidnap_at is None:
        mcl_steps, _ = localization_steps(recording.mcl_trial)
        ekf_steps, _ = localization_steps(recording.ekf_trial)
        return (
            f"{head}; no kidnapping; MCL from anywhere: {_steps_text(mcl_steps)}; "
            f"the EKF from the noisy start: {_steps_text(ekf_steps)}."
        )
    head += f", kidnapped at step {kidnap_at}"
    if kidnap_at < LOCALIZE_DEADLINE + LOCALIZE_HOLD:
        return (
            f"{head}; recovery not scored (the kidnapping comes before step "
            f"{LOCALIZE_DEADLINE + LOCALIZE_HOLD})."
        )
    _, mcl_recovery = localization_steps(recording.mcl_trial)
    _, ekf_recovery = localization_steps(recording.ekf_trial)
    return (
        f"{head}; MCL recovery: {_steps_text(mcl_recovery)}; "
        f"the EKF's recovery: {_steps_text(ekf_recovery)}."
    )


def _steps_text(steps: int | None) -> str:
    if steps is None:
        return "did not localize"
    return f"localized in {steps} steps ({steps * STEP_SECONDS:.1f} s)"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_navigation.localization_video",
        description="Write one seeded localization trial (the EKF and MCL) to an MP4.",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--sensor", choices=sorted(SENSORS), default="lidar")
    parser.add_argument(
        "--dropout", type=_dropout, default=None, help="'own' (default) or a probability"
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument(
        "--kidnap-at", type=_kidnap_at, default=300, help="a step in [1, steps), or 'none'"
    )
    parser.add_argument(
        "--mcl-scale",
        nargs=3,
        type=float,
        metavar=("SX", "SY", "STH"),
        help="MCL's covariance scale; (1, 1, 1) when not given",
    )
    parser.add_argument("--particles", type=int, default=MclSettings().num_particles)
    parser.add_argument("--fps", type=float, default=1.0 / STEP_SECONDS, help="default: real time")
    parser.add_argument("--pixels-per-metre", type=float, default=PIXELS_PER_METRE)
    parser.add_argument(
        "--output", type=Path, required=True, help="the MP4 to write; never overwritten"
    )
    parser.add_argument("--stills", nargs="+", type=int, default=[], metavar="FRAME")
    parser.add_argument("--stills-dir", type=Path, help="where the stills go (created)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Record a trial, write its MP4 (and stills), and print the legend and a summary."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error(f"--steps must be at least 1, got {args.steps}.")
    if args.kidnap_at is not None and not 1 <= args.kidnap_at < args.steps:
        parser.error(f"--kidnap-at must be in [1, {args.steps}) or none, got {args.kidnap_at}.")
    if not args.fps > 0:
        parser.error(f"--fps must be above 0, got {args.fps}.")
    if not args.pixels_per_metre > 0:
        parser.error(f"--pixels-per-metre must be above 0, got {args.pixels_per_metre}.")
    if _taken(args.output):
        parser.error(f"--output {args.output} exists; choose a new file.")
    for name, directory in (
        ("--output's folder", args.output.parent),
        ("--stills-dir", args.stills_dir),
    ):
        if directory is not None and _blocked(directory):
            parser.error(f"{name} {directory} is not a directory.")
    if args.stills and args.stills_dir is None:
        parser.error("--stills needs --stills-dir.")
    outside = [frame for frame in args.stills if not 0 <= frame <= args.steps]
    if outside:
        parser.error(f"--stills must be frames in [0, {args.steps}], got {outside}.")
    taken = [
        str(still_path(args.stills_dir, args.output, frame))
        for frame in args.stills
        if _taken(still_path(args.stills_dir, args.output, frame))
    ]
    if taken:
        parser.error(f"--stills would overwrite {', '.join(taken)}.")
    try:
        sx, sy, stheta = checked_axis_values(
            (1.0, 1.0, 1.0) if args.mcl_scale is None else args.mcl_scale,
            "localization_video",
            "--mcl-scale",
        )
        settings = MclSettings(
            num_particles=args.particles, covariance_scale=(float(sx), float(sy), float(stheta))
        )
        require_imageio_ffmpeg()
    except (ValueError, ImportError) as error:
        parser.error(str(error))
    print("Legend:")
    for line in LEGEND:
        print(f"  {line}")
    recording = record_trials(
        args.seed,
        SENSORS[args.sensor](),
        steps=args.steps,
        kidnap_at=args.kidnap_at,
        dropout=args.dropout,
        mcl_settings=settings,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = render_frames(recording, args.pixels_per_metre)
    if args.stills:
        frames = _saving_stills(frames, args.stills, args.stills_dir, args.output)
    written = write_video(args.output, frames, args.fps)
    print(f"Seed {args.seed}, {args.sensor}: {summary(recording)}")
    missed = sorted(frame for frame in args.stills if frame >= written)
    if missed:
        print(f"Stills {missed} were past the last frame (the trial ended early) and not written.")
    print(f"Wrote {written} frames to {args.output}.")


if __name__ == "__main__":
    main()

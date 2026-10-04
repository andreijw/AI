"""Training video recording and clip merging.

``RecordVideo`` keeps every frame of a recording in memory until the recording ends, so a
multi-episode video is not recorded continuously. Instead, each episode is written to its own
clip, and :func:`merge_episode_clips` joins the clips on disk afterwards.
"""

import logging
import os
import re
import subprocess  # nosec B404
import tempfile
from typing import Any, Protocol

from gymnasium.wrappers import RecordVideo

_logger = logging.getLogger(__name__)

TRAINING_NAME_PREFIX = "cartpole-training"


class TrainingProgress(Protocol):
    """Training state followed by :class:`TrainingVideoRecorder` (implemented by ``Trainer``)."""

    current_episode: int
    evaluating: bool


class TrainingVideoRecorder(RecordVideo):
    """``RecordVideo`` that records every Nth training episode, named by training episode.

    ``RecordVideo`` counts every environment reset, including evaluation episodes run on the
    same environment, so its own episode counter drifts away from the training episode
    number. This recorder follows ``progress`` instead: it never records evaluation episodes
    or resets before training starts, and writes ``cartpole-training-episode-<N>.mp4`` where N
    is the 1-based training episode.
    """

    def __init__(
        self,
        env: Any,
        video_folder: str,
        progress: TrainingProgress,
        every: int,
    ):
        """
        Args:
            env: Gymnasium environment created with ``render_mode="rgb_array"``
            video_folder: Directory the episode clips are written to
            progress: Source of the current training episode and evaluation state
            every: Record every ``every``-th training episode (must be > 0)
        """
        if every <= 0:
            raise ValueError(f"every must be a positive integer, got {every!r}")
        self._progress = progress

        # A plain closure, not a bound method: RecordVideo deep-copies its constructor
        # arguments, and copying a bound method would copy the trainer behind ``progress``.
        def should_record(_env_episode: int) -> bool:
            episode = progress.current_episode
            return not progress.evaluating and episode > 0 and episode % every == 0

        super().__init__(
            env,
            video_folder=video_folder,
            episode_trigger=should_record,
            name_prefix=TRAINING_NAME_PREFIX,
            disable_logger=True,
        )

    def start_recording(self, video_name: str) -> None:
        """Start a recording named after the training episode instead of the reset count."""
        super().start_recording(f"{self.name_prefix}-episode-{self._progress.current_episode}")


def merge_episode_clips(video_dir: str, name_prefix: str) -> str | None:
    """
    Join the episode clips in ``video_dir`` into ``<name_prefix>-full.mp4`` and delete them.

    Clips are joined in episode order with ffmpeg's concat demuxer, which copies the encoded
    streams instead of re-encoding them, so memory use stays constant and merging is fast.
    All clips come from the same recorder and therefore share codec, frame size and frame
    rate, as the concat demuxer requires. The clips are only deleted once the merge succeeds.

    Args:
        video_dir: Directory containing ``<name_prefix>-episode-<N>.mp4`` clips
        name_prefix: Clip file name prefix

    Returns:
        Path of the merged video, or None if there were no clips to merge
    """
    pattern = re.compile(rf"^{re.escape(name_prefix)}-episode-(\d+)\.mp4$")
    clips = sorted(
        (int(match.group(1)), name)
        for name in os.listdir(video_dir)
        if (match := pattern.match(name))
    )
    if not clips:
        _logger.warning(
            "No episode clips matching '%s-episode-*.mp4' in '%s'; nothing to merge.",
            name_prefix,
            video_dir,
        )
        return None

    import imageio_ffmpeg

    clip_paths = [os.path.abspath(os.path.join(video_dir, name)) for _, name in clips]
    output_path = os.path.join(video_dir, f"{name_prefix}-full.mp4")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as list_file:
        for path in clip_paths:
            escaped = path.replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")
    try:
        subprocess.run(  # nosec B603
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file.name,
                "-c",
                "copy",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to merge episode clips: {stderr}") from exc
    finally:
        os.remove(list_file.name)

    for path in clip_paths:
        os.remove(path)
    return output_path

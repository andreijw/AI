#!/usr/bin/env python3
"""Make the documentation's frames and clips from ``scripts/doc_media.json``.

Each manifest entry with a ``recipe`` is made here; one without (a still copied from another
command's output) says in ``made_by`` how it was made, and is left alone. A recipe is one of:

- ``still``: frame ``frame`` (0 is the reset's) of one episode, as a PNG.
- ``clip``: one episode, or its first 10 s, as a GIF: 300 px wide, 15 to 25 fps, 32 to 64 colours,
  every ``frame_step``-th rendered frame.
- ``mapping``: occupancy grids of one wandering robot's scans (``mapping.evaluation``'s
  ``record_run`` and ``build_map``), from the true poses (left) and from dead reckoning (right).

A still or clip builds the environment and agent from ``config`` as ``record.py`` does (with the
recipe's ``environment`` settings over the config's, and the agent loaded from ``checkpoint`` when
one is named, with the ``release_asset`` it is published as: a missing checkpoint stops the run,
before anything is written, naming the asset to download into its folder), resets with the recipe's
``seed`` and captures ``env.render()``. Images are written with Pillow, only under ``docs/images/``,
and only within the manifest's budget. Each entry made gets ``made_by`` (the command that remakes
it) and ``made_with``: the config's SHA-256, the checkpoint's, the commit, the platform and the
versions of Python, NumPy, Gymnasium, Pillow and pygame-ce.

Run from the repository's own environment (``.venv/bin/python``; Box2D items need ``./setup.sh``):

    python scripts/make_doc_media.py --all
    python scripts/make_doc_media.py docs/images/cartpole_random_still.png
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import subprocess  # nosec B404: git, with fixed arguments
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # train.py's helpers, which record.py builds with

from rl_core.challenges import discover_challenges  # noqa: E402
from rl_core.training.checkpoints import file_sha256  # noqa: E402
from train import (  # noqa: E402
    headless_rendering,
    load_run_config,
    numpy_agent_config,
    select_run_challenge,
)

IMAGES_DIR = Path("docs/images")
MANIFEST = Path("scripts/doc_media.json")
SUFFIX = {"still": ".png", "mapping": ".png", "clip": ".gif"}
BUDGET_KEY = {".png": "png_kib", ".gif": "gif_kib"}
CLIP_WIDTH = 300
CLIP_FPS = (15, 25)
CLIP_COLOURS = (32, 64)
CLIP_SECONDS = 10
PALETTE_SAMPLES = 16  # frames a clip's one palette is fitted to
MAP_GAP = 8  # px of white between the two maps
SENSORS = ("lidar", "hc_sr04")
# What the tree check leaves out: the script's outputs, its own code (``script_sha256`` records it)
# and the tests and Markdown, which no image is made from. Anything else changed since the commit
# makes ``made_with.tree`` "modified".
NOT_INPUTS = (
    f":!{IMAGES_DIR.as_posix()}",
    f":!{MANIFEST.as_posix()}",
    ":!scripts/make_doc_media.py",
    ":!tests",
    ":!*.md",
)


class MediaError(ValueError):
    """A recipe, path or result that breaks the docs media rules (CONTRIBUTING.md §5)."""


class Episode:
    """The frames kept from one episode, with every observation and action played.

    A plain class, not a dataclass: the tests load this script by path without registering it in
    ``sys.modules``, which a dataclass with postponed annotations needs.
    """

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.observations: list[np.ndarray] = []
        self.actions: list[Any] = []
        self.render_fps: int | None = None


# --- rules ---------------------------------------------------------------------------------------


def output_path(root: Path, path: str) -> Path:
    """Where ``path`` (relative to ``root``) is written; refused unless a PNG or GIF in docs/images/."""
    images = root.resolve() / IMAGES_DIR
    target = (root / path).resolve()
    if Path(path).is_absolute() or images not in target.parents:
        raise MediaError(f"{path}: docs media is written only under {IMAGES_DIR.as_posix()}/")
    if target.suffix not in BUDGET_KEY:
        raise MediaError(f"{path}: only .png and .gif belong in {IMAGES_DIR.as_posix()}/")
    return target


def _int_in(recipe: Mapping[str, Any], key: str, low: int, high: float, path: str) -> None:
    value = recipe.get(key)
    if not (isinstance(value, int) and not isinstance(value, bool) and low <= value <= high):
        raise MediaError(f"{path}: '{key}' must be an integer from {low} to {high}, got {value!r}")


def check_recipe(path: str, recipe: Mapping[str, Any]) -> None:
    """Refuse a recipe of an unknown kind, for the wrong file type, or outside the format rules."""
    kind = recipe.get("kind")
    if kind not in SUFFIX:
        raise MediaError(f"{path}: 'kind' must be one of {sorted(SUFFIX)}, got {kind!r}")
    if Path(path).suffix != SUFFIX[kind]:
        raise MediaError(f"{path}: a {kind} is written as {SUFFIX[kind]}")
    _int_in(recipe, "seed", 0, float("inf"), path)
    if kind == "mapping":
        if recipe.get("sensor") not in SENSORS:
            raise MediaError(f"{path}: 'sensor' must be one of {SENSORS}")
        _int_in(recipe, "steps", 1, float("inf"), path)
        _int_in(recipe, "scale", 1, 16, path)
        return
    if not isinstance(recipe.get("config"), str):
        raise MediaError(f"{path}: a {kind} needs the 'config' it is played from")
    if recipe.get("checkpoint") is not None and not isinstance(recipe.get("release_asset"), str):
        raise MediaError(f"{path}: a checkpoint needs the 'release_asset' it is published as")
    crop = recipe.get("crop")
    if crop is not None and not (
        isinstance(crop, list)
        and len(crop) == 4
        and all(isinstance(v, int) for v in crop)
        and 0 <= crop[0] < crop[2]
        and 0 <= crop[1] < crop[3]
    ):
        raise MediaError(
            f"{path}: 'crop' must be [left, top, right, bottom] in pixels, left < right and top < bottom"
        )
    if kind == "still":
        _int_in(recipe, "frame", 0, float("inf"), path)
        if recipe.get("width") is not None:
            _int_in(recipe, "width", 1, float("inf"), path)
        return
    if recipe.get("width") != CLIP_WIDTH:
        raise MediaError(f"{path}: a clip is {CLIP_WIDTH} px wide")
    _int_in(recipe, "fps", *CLIP_FPS, path)
    played = 1000 / _delay_ms(recipe["fps"])
    if not CLIP_FPS[0] <= played <= CLIP_FPS[1]:
        raise MediaError(
            f"{path}: {recipe['fps']} fps is stored as {_delay_ms(recipe['fps'])} ms a frame "
            f"(whole hundredths of a second) and plays at {played:.1f} fps, outside {CLIP_FPS}"
        )
    _int_in(recipe, "colours", *CLIP_COLOURS, path)
    _int_in({"frame_step": 1, **recipe}, "frame_step", 1, float("inf"), path)


# --- playing an episode --------------------------------------------------------------------------


def build(root: Path, recipe: Mapping[str, Any]) -> tuple[Any, Any]:
    """The environment (rendering ``rgb_array``) and agent of the recipe, as ``record.py`` builds them."""
    discover_challenges()
    config, agent_type = load_run_config(str(root / recipe["config"]))
    environment = {**config["environment"], **recipe.get("environment", {})}
    challenge = select_run_challenge(config, agent_type)
    env = challenge.make_env(environment, render_mode="rgb_array")
    try:
        agent = challenge.make_agent(agent_type, numpy_agent_config(config), env)
        if recipe.get("checkpoint") is not None:
            agent.load(str(root / recipe["checkpoint"]))
    except BaseException:
        env.close()
        raise
    return env, agent


def play(
    root: Path, recipe: Mapping[str, Any], frame_step: int = 1, max_frames: int | None = None
) -> Episode:
    """One greedy episode from ``reset(seed=recipe["seed"])``, keeping every ``frame_step``-th frame
    until the episode ends or ``max_frames`` are kept."""
    episode = Episode()
    with headless_rendering():
        env, agent = build(root, recipe)
        try:
            episode.render_fps = env.env.metadata.get("render_fps")
            observation, _ = env.reset(seed=recipe["seed"])
            step, done = 0, False
            while True:
                episode.observations.append(observation)
                if step % frame_step == 0:
                    episode.frames.append(np.asarray(env.render()))
                if done or (max_frames is not None and len(episode.frames) >= max_frames):
                    break
                action = agent.select_action(
                    agent.normalize_observation(observation, training=False), training=False
                )
                observation, _, terminated, truncated, _ = env.step(action)
                episode.actions.append(action)
                step, done = step + 1, terminated or truncated
        finally:
            env.close()
    return episode


# --- images --------------------------------------------------------------------------------------


def _image(frame: np.ndarray, recipe: Mapping[str, Any]) -> Image.Image:
    """A frame cropped to the recipe's ``crop``, then scaled to its ``width``."""
    image = Image.fromarray(frame)
    if recipe.get("crop") is not None:
        left, top, right, bottom = recipe["crop"]
        if right > image.width or bottom > image.height:
            raise MediaError(
                f"crop {recipe['crop']} reaches outside the {image.width} x {image.height} frame"
            )
        image = image.crop((left, top, right, bottom))
    width = recipe.get("width")
    if width is not None and width != image.width:
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _gif(images: list[Image.Image], fps: int, colours: int) -> bytes:
    """The frames with one palette of ``colours``, fitted to a sample of them, at ``fps``."""
    picks = np.linspace(0, len(images) - 1, min(PALETTE_SAMPLES, len(images))).round()
    sample = Image.new("RGB", (images[0].width, images[0].height * len(picks)))
    for row, index in enumerate(picks.astype(int)):
        sample.paste(images[index], (0, row * images[0].height))
    palette = sample.quantize(colors=colours, method=Image.Quantize.MEDIANCUT)
    frames = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=_delay_ms(fps),
        loop=0,
        optimize=True,
    )
    return buffer.getvalue()


def _delay_ms(fps: int) -> int:
    """A GIF frame's delay: whole hundredths of a second, the nearest to 1/fps."""
    return 10 * round(100 / fps)


def _still(root: Path, path: str, recipe: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    frame = recipe["frame"]
    episode = play(root, recipe, max_frames=frame + 1)
    if len(episode.frames) <= frame:
        raise MediaError(
            f"{path}: the episode ends at frame {len(episode.frames) - 1}, before frame {frame}"
        )
    return _png(_image(episode.frames[frame], recipe)), {"render_fps": episode.render_fps}


def _clip(root: Path, recipe: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    max_frames = CLIP_SECONDS * 1000 // _delay_ms(recipe["fps"])
    episode = play(root, recipe, frame_step=recipe.get("frame_step", 1), max_frames=max_frames)
    images = [_image(frame, recipe) for frame in episode.frames]
    data = _gif(images, recipe["fps"], recipe["colours"])
    extra = {"render_fps": episode.render_fps, "frames": len(images)}
    return data, {**extra, "delay_ms": _delay_ms(recipe["fps"])}


def _mapping(recipe: Mapping[str, Any]) -> bytes:
    from rl_navigation.mapping import HC_SR04_LOG_ODDS, LIDAR_LOG_ODDS
    from rl_navigation.mapping.evaluation import build_map, record_run
    from rl_navigation.sim import hc_sr04, lidar

    sensor, model = (lidar(), LIDAR_LOG_ODDS)
    if recipe["sensor"] == "hc_sr04":
        sensor, model = (hc_sr04(), HC_SR04_LOG_ODDS)
    run = record_run(recipe["seed"], sensor, steps=recipe["steps"])
    scale = np.ones((recipe["scale"], recipe["scale"]), dtype=np.uint8)
    tiles = []
    for poses in ("true", "dead_reckoning"):
        occupied = build_map(run, model, poses=poses).probabilities()
        grey = np.round(255.0 * (1.0 - occupied)).astype(np.uint8)
        tiles.append(np.kron(np.flipud(grey), scale))  # row 0 is y = 0: the bottom of the picture
    gap = np.full((tiles[0].shape[0], MAP_GAP), 255, dtype=np.uint8)
    return _png(Image.fromarray(np.hstack([tiles[0], gap, tiles[1]])))


# --- provenance ----------------------------------------------------------------------------------


#: What ``made_with`` records when git cannot answer (no git, not a repository, refused).
UNKNOWN = "unknown"


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(  # nosec B603 B607: git, with fixed arguments
            ["git", *args], cwd=root, capture_output=True, check=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def made_with(root: Path, recipe: Mapping[str, Any]) -> dict[str, Any]:
    """What a regeneration needs to know besides the recipe: hashes, commit, platform, versions."""
    made: dict[str, Any] = {}
    if "config" in recipe:
        made["config"] = recipe["config"]
        made["config_sha256"] = file_sha256(str(root / recipe["config"]))
    if recipe.get("checkpoint") is not None:
        made["checkpoint_sha256"] = file_sha256(str(root / recipe["checkpoint"]))
        made["release_asset"] = recipe["release_asset"]
    changed = _git(root, "status", "--porcelain", "--untracked-files=no", "--", ".", *NOT_INPUTS)
    commit = _git(root, "rev-parse", "HEAD")
    tree = UNKNOWN if changed is None else ("modified" if changed else "clean")
    made.update(
        commit=commit or UNKNOWN,
        tree=tree,
        script_sha256=file_sha256(__file__),
        platform=f"{platform.system()} {platform.machine()}",
        python=platform.python_version(),
        numpy=_version("numpy"),
        gymnasium=_version("gymnasium"),
        pillow=_version("pillow"),
        pygame_ce=_version("pygame-ce"),
    )
    return made


def render_item(root: Path, path: str, recipe: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """The encoded image of one recipe and its ``made_with`` record; nothing is written."""
    check_recipe(path, recipe)
    extra: dict[str, Any] = {}
    if recipe["kind"] == "mapping":
        data = _mapping(recipe)
    elif recipe["kind"] == "still":
        data, extra = _still(root, path, recipe)
    else:
        data, extra = _clip(root, recipe)
    return data, {**made_with(root, recipe), **extra}


# --- the manifest --------------------------------------------------------------------------------


def make(root: Path, paths: Sequence[str] | None) -> list[str]:
    """Make the manifest's items at ``paths`` (every item with a recipe when None), write them and
    record them in the manifest; return the paths written."""
    manifest_path = root / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in manifest["media"]}
    if paths is None:
        paths = [path for path, entry in entries.items() if "recipe" in entry]
    for path in paths:
        if path not in entries:
            raise MediaError(f"{path}: not listed in {MANIFEST.as_posix()}")
        if "recipe" not in entries[path]:
            raise MediaError(f"{path}: made by another command ({entries[path]['made_by']})")
        output_path(root, path)
        recipe = entries[path]["recipe"]
        check_recipe(path, recipe)
        checkpoint = recipe.get("checkpoint")
        if checkpoint is not None and not (root / checkpoint).is_file():
            raise MediaError(
                f"{path}: no checkpoint at {checkpoint}: download the release asset "
                f"{recipe['release_asset']} into {Path(checkpoint).parent.as_posix()}/"
            )
    written = []
    for path in paths:
        entry = entries[path]
        target = output_path(root, path)
        data, made = render_item(root, path, entry["recipe"])
        limit = manifest["budget"][BUDGET_KEY[target.suffix]]
        if len(data) > limit * 1024:
            raise MediaError(f"{path}: {len(data) / 1024:.1f} KiB, over the {limit} KiB budget")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entry["made_by"] = f"python scripts/make_doc_media.py {path}"
        entry["made_with"] = made
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{path}: {len(data) / 1024:.1f} KiB")
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("paths", nargs="*", help="manifest paths to make (docs/images/...)")
    parser.add_argument("--all", action="store_true", help="make every item that has a recipe")
    args = parser.parse_args(argv)
    if bool(args.paths) == args.all:
        parser.error("name the paths to make, or give --all")
    make(REPO_ROOT, None if args.all else args.paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())

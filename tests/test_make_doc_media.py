"""scripts/make_doc_media.py: the docs' frames and clips, made as the manifest says and only in docs/images/.

Images are compared by their decoded pixels within a tolerance, never byte for byte: zlib, Pillow and
the platform's rendering may change the bytes of the same picture.
"""

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.helpers import load_script

make_doc_media = load_script("make_doc_media")

REPO_ROOT = Path(__file__).resolve().parents[1]
CARTPOLE = "configs/cartpole_default.yaml"
NAVIGATION = "configs/navigation_default.yaml"
CARTPOLE_STILL = "docs/images/cartpole_random_still.png"
STILL = {"kind": "still", "config": CARTPOLE, "seed": 42, "frame": 8}
CLIP = {"kind": "clip", "config": CARTPOLE, "seed": 42, "fps": 25, "colours": 32, "width": 300}
MAPPING = {"kind": "mapping", "seed": 101, "steps": 40, "sensor": "lidar", "scale": 2}

# CartPole-v1 reset with seed 42 and played by configs/cartpole_default.yaml's random agent (its own
# seed 42): the observations after reset and after each step, and the actions. The pole falls past
# 12 degrees (0.2094 rad) on the tenth step.
CARTPOLE_OBSERVATIONS = [
    [0.0273956, -0.00611216, 0.03585979, 0.0197368],
    [0.02727336, -0.20172954, 0.03625453, 0.32351476],
    [0.02323877, -0.39734846, 0.04272482, 0.62740684],
    [0.0152918, -0.20284806, 0.05527296, 0.34847975],
    [0.01123484, -0.39871082, 0.06224256, 0.65806717],
    [0.00326062, -0.5946414, 0.0754039, 0.9696814],
    [-0.0086322, -0.7906901, 0.09479753, 1.2850666],
    [-0.02444601, -0.9868823, 0.12049886, 1.6058624],
    [-0.04418365, -1.1832057, 0.15261611, 1.9335531],
    [-0.06784777, -1.379596, 0.19128717, 2.2694097],
    [-0.09543969, -1.1867058, 0.23667537, 2.0412478],
]
CARTPOLE_ACTIONS = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]


def _decode(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.int16)


def _gif_frames(data: bytes) -> tuple[int, int, list[int], int]:
    """Width, frame count, each frame's delay in ms, and the most colours any frame uses."""
    image = Image.open(io.BytesIO(data))
    delays, colours = [], 0
    for index in range(image.n_frames):
        image.seek(index)
        delays.append(image.info["duration"])
        colours = max(colours, len(image.convert("RGB").getcolors(maxcolors=1 << 16) or []))
    return image.width, image.n_frames, delays, colours


def _root_with(tmp_path: Path, media: list, budget: dict | None = None) -> Path:
    """A repository root in tmp_path with the CartPole config and a manifest of ``media``."""
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO_ROOT / CARTPOLE, tmp_path / CARTPOLE)
    (tmp_path / "scripts").mkdir()
    budget = budget or {"png_kib": 60, "gif_kib": 600, "total_kib": 6144}
    manifest = {"budget": budget, "media": media}
    (tmp_path / "scripts" / "doc_media.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _manifest(root: Path) -> dict:
    return json.loads((root / "scripts" / "doc_media.json").read_text(encoding="utf-8"))


# --- playing an episode as record.py builds it ---------------------------------------------------


def test_a_seeded_cartpole_episode_plays_the_pinned_trajectory(clear_sdl):
    episode = make_doc_media.play(REPO_ROOT, STILL)

    np.testing.assert_allclose(
        episode.observations,
        CARTPOLE_OBSERVATIONS,
        atol=1e-6,
        err_msg="the config's random agent from reset(seed=42) should play the pinned observations",
    )
    assert [int(action) for action in episode.actions] == CARTPOLE_ACTIONS, (
        "the config's random agent (seed 42) should choose the pinned actions"
    )
    assert len(episode.frames) == len(CARTPOLE_OBSERVATIONS), (
        "every observation, the reset's included, should come with its rendered frame"
    )
    assert episode.render_fps == 50, "CartPole-v1 renders at 50 fps"


def test_the_episode_seed_is_the_recipes(clear_sdl):
    other = make_doc_media.play(REPO_ROOT, {**STILL, "seed": 7})

    assert not np.allclose(other.observations[0], CARTPOLE_OBSERVATIONS[0]), (
        "reset(seed=7) should start CartPole elsewhere than reset(seed=42)"
    )


def test_environment_settings_in_a_recipe_reach_the_environment():
    hc_sr04 = make_doc_media.play(REPO_ROOT, {"config": NAVIGATION, "seed": 42}, max_frames=1)
    lidar = make_doc_media.play(
        REPO_ROOT,
        {"config": NAVIGATION, "seed": 42, "environment": {"sensor": "lidar"}},
        max_frames=1,
    )

    assert len(lidar.observations[0]) > len(hc_sr04.observations[0]), (
        "sensor: lidar should give the robot more range readings than the five HC-SR04 sonars"
    )


def test_frame_step_keeps_every_nth_frame(clear_sdl):
    every = make_doc_media.play(REPO_ROOT, STILL)
    second = make_doc_media.play(REPO_ROOT, STILL, frame_step=2)

    assert len(second.frames) == 6, "frames 0, 2, 4, 6, 8 and 10 of the 11 should be kept"
    np.testing.assert_array_equal(
        second.frames[3], every.frames[6], err_msg="kept frame 3 is frame 6"
    )


# --- stills -------------------------------------------------------------------------------------


def test_a_cartpole_still_decodes_to_the_committed_pixels(clear_sdl):
    data, _ = make_doc_media.render_item(REPO_ROOT, CARTPOLE_STILL, STILL)

    made = _decode(data)
    committed = _decode((REPO_ROOT / CARTPOLE_STILL).read_bytes())
    assert made.shape == committed.shape == (400, 600, 3), "the still should be CartPole's 600x400"
    difference = np.abs(made - committed)
    assert difference.mean() <= 1.0, "the regenerated still should look like the committed one"
    assert (difference.max(axis=2) <= 16).mean() >= 0.99, (
        "at most 1 % of the pixels may differ by more than 16 levels (antialiasing)"
    )


def test_a_still_is_the_recipes_frame(clear_sdl):
    data, _ = make_doc_media.render_item(REPO_ROOT, CARTPOLE_STILL, STILL)
    frames = make_doc_media.play(REPO_ROOT, STILL).frames

    np.testing.assert_array_equal(
        _decode(data), frames[8], err_msg="a still without crop or width is frame 8 as rendered"
    )


def test_a_still_is_cropped_then_scaled(clear_sdl):
    recipe = {**STILL, "crop": [100, 50, 500, 350], "width": 200}
    data, _ = make_doc_media.render_item(REPO_ROOT, CARTPOLE_STILL, recipe)

    assert _decode(data).shape == (150, 200, 3), "a 400x300 crop scaled to 200 px wide is 200x150"


# --- clips --------------------------------------------------------------------------------------


def test_a_clip_is_a_gif_in_the_format(clear_sdl):
    data, made = make_doc_media.render_item(REPO_ROOT, "docs/images/clip.gif", CLIP)

    width, count, delays, colours = _gif_frames(data)
    assert width == 300, "a clip is 300 px wide"
    assert made["frames"] == len(CARTPOLE_OBSERVATIONS), (
        "the whole 10-step episode, reset frame included, is recorded as 11 frames"
    )
    # Pillow merges a frame identical to the one before into it, adding up their delays.
    assert count <= 11 and sum(delays) == 11 * 40, "25 fps is 40 ms a frame, 11 frames in all"
    assert colours <= 32, "the clip's palette has at most its 32 colours"


def test_a_clip_stops_after_ten_seconds():
    recipe = {**CLIP, "config": NAVIGATION, "fps": 16}
    length = len(make_doc_media.play(REPO_ROOT, recipe, frame_step=10**6).actions)
    assert length + 1 > 166, "the test needs an episode longer than the clip's 10 s"

    data, made = make_doc_media.render_item(REPO_ROOT, "docs/images/clip.gif", recipe)

    _, _, delays, _ = _gif_frames(data)
    # A GIF delay is whole hundredths of a second: 16 fps is 6 cs (60 ms, 16.7 fps played), and
    # 10 s holds 10000 // 60 = 166 such frames.
    assert set(delays) == {60}, "16 fps is stored as 60 ms a frame"
    assert made["delay_ms"] == 60, "the manifest records the delay actually stored"
    assert made["frames"] == 166, "the clip is cut at the last whole frame within 10 s"
    assert sum(delays) == 9_960, "the clip plays for its first 10 s and no longer"


@pytest.mark.parametrize("fps, accepted", [(15, False), (16, True), (25, True), (26, False)])
def test_a_clip_is_refused_when_its_stored_delay_plays_outside_15_to_25_fps(fps, accepted):
    # 15 fps: round(100 / 15) = 7 cs, played at 14.3 fps; 16 fps: 6 cs, 16.7 fps; 25 fps: 4 cs;
    # 26 fps: 4 cs as well, but the recipe asks outside the range.
    recipe = {**CLIP, "fps": fps}
    if accepted:
        make_doc_media.check_recipe("docs/images/a.gif", recipe)
    else:
        with pytest.raises(make_doc_media.MediaError, match="fps"):
            make_doc_media.check_recipe("docs/images/a.gif", recipe)


# --- the mapping figure -------------------------------------------------------------------------


def test_the_mapping_pair_shows_true_poses_beside_dead_reckoning():
    from rl_navigation.mapping import LIDAR_LOG_ODDS
    from rl_navigation.mapping.evaluation import build_map, record_run
    from rl_navigation.sim import lidar

    data, _ = make_doc_media.render_item(REPO_ROOT, "docs/images/map.png", MAPPING)

    image = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.int16)
    run = record_run(MAPPING["seed"], lidar(), steps=MAPPING["steps"])
    for poses, tile in (("true", image[:, :200]), ("dead_reckoning", image[:, -200:])):
        occupied = build_map(run, LIDAR_LOG_ODDS, poses=poses).probabilities()
        # Dark is occupied; row 0 of the grid (y = 0) is the bottom row; each cell is 2 x 2 px.
        expected = np.kron(np.flipud(255.0 * (1.0 - occupied)), np.ones((2, 2)))
        np.testing.assert_allclose(
            tile, expected, atol=0.5, err_msg=f"the {poses} map, scaled 2x, should be drawn as is"
        )
    assert (image[:, 200:-200] == 255).all(), "a white gap should part the two maps"


# --- the rules the script refuses to break -------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "README.png",
        "docs/clip.png",
        "docs/images/../clip.png",
        "docs/imagesx/clip.png",
        "/tmp/clip.png",
        "docs/images/clip.jpg",
        "docs/images/clip.mp4",
    ],
)
def test_output_outside_docs_images_is_refused(tmp_path, path):
    with pytest.raises(make_doc_media.MediaError, match="docs/images"):
        make_doc_media.output_path(tmp_path, path)


def test_a_link_out_of_docs_images_is_refused(tmp_path):
    (tmp_path / "docs" / "images").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "docs" / "images" / "out").symlink_to(tmp_path / "elsewhere")

    with pytest.raises(make_doc_media.MediaError, match="docs/images"):
        make_doc_media.output_path(tmp_path, "docs/images/out/clip.png")


@pytest.mark.parametrize("path", ["docs/images/clip.png", "docs/images/sub/clip.gif"])
def test_output_inside_docs_images_is_allowed(tmp_path, path):
    assert make_doc_media.output_path(tmp_path, path) == (tmp_path / path).resolve()


def test_make_refuses_an_entry_outside_docs_images_and_writes_nothing(tmp_path):
    root = _root_with(tmp_path, [{"path": "docs/clip.png", "made_by": "x", "recipe": STILL}])

    with pytest.raises(make_doc_media.MediaError, match="docs/images"):
        make_doc_media.make(root, ["docs/clip.png"])
    assert not (root / "docs").exists(), "nothing should be written outside docs/images/"


@pytest.mark.parametrize(
    "path, change, message",
    [
        ("a.gif", {"fps": 30}, "fps"),
        ("a.gif", {"fps": 10}, "fps"),
        ("a.gif", {"colours": 16}, "colours"),
        ("a.gif", {"colours": 128}, "colours"),
        ("a.gif", {"width": 400}, "300 px"),
        ("a.gif", {"frame_step": 0}, "frame_step"),
        ("a.png", {}, ".gif"),
        ("a.gif", {"kind": "video"}, "kind"),
        ("a.gif", {"config": None}, "config"),
        ("a.gif", {"checkpoint": "agent.pt"}, "release_asset"),
    ],
)
def test_a_clip_recipe_outside_the_format_is_refused(path, change, message):
    recipe = {key: value for key, value in {**CLIP, **change}.items() if value is not None}

    with pytest.raises(make_doc_media.MediaError, match=message):
        make_doc_media.check_recipe(f"docs/images/{path}", recipe)


@pytest.mark.parametrize(
    "path, recipe, message",
    [
        ("a.gif", STILL, ".png"),
        ("a.png", {**STILL, "frame": -1}, "frame"),
        ("a.png", {**STILL, "crop": [0, 0, 10]}, "crop"),
        ("a.png", {**STILL, "crop": [100, 0, 50, 10]}, "crop"),
        ("a.png", {**STILL, "crop": [0, 10, 10, 10]}, "crop"),
        ("a.png", {**STILL, "crop": [-1, 0, 10, 10]}, "crop"),
        ("a.png", {**STILL, "width": 0}, "width"),
        ("a.gif", MAPPING, ".png"),
        ("a.png", {**MAPPING, "sensor": "radar"}, "sensor"),
    ],
)
def test_a_still_or_mapping_recipe_outside_the_rules_is_refused(path, recipe, message):
    with pytest.raises(make_doc_media.MediaError, match=message):
        make_doc_media.check_recipe(f"docs/images/{path}", recipe)


def test_the_manifests_recipes_follow_the_rules():
    manifest = _manifest(REPO_ROOT)
    made = [entry for entry in manifest["media"] if "recipe" in entry]

    assert made, "the manifest should list the media this script makes"
    for entry in made:
        make_doc_media.check_recipe(entry["path"], entry["recipe"])
        assert entry["made_by"] == f"python scripts/make_doc_media.py {entry['path']}", (
            f"{entry['path']} should say the command that remakes it"
        )
        assert entry["made_with"]["commit"], (
            f"{entry['path']} should record the commit it came from"
        )


def test_a_crop_outside_the_frame_is_refused(clear_sdl):
    with pytest.raises(make_doc_media.MediaError, match="600 x 400"):
        make_doc_media.render_item(REPO_ROOT, CARTPOLE_STILL, {**STILL, "crop": [0, 0, 601, 400]})


def test_an_episode_that_ends_before_the_frame_is_refused(clear_sdl):
    with pytest.raises(make_doc_media.MediaError, match="frame 50"):
        make_doc_media.render_item(REPO_ROOT, CARTPOLE_STILL, {**STILL, "frame": 50})


def test_an_item_over_its_budget_is_refused_and_not_written(tmp_path, clear_sdl):
    entry = {"path": CARTPOLE_STILL, "made_by": "x", "recipe": STILL}
    root = _root_with(tmp_path, [entry], budget={"png_kib": 1, "gif_kib": 1, "total_kib": 2})

    with pytest.raises(make_doc_media.MediaError, match="1 KiB"):
        make_doc_media.make(root, [CARTPOLE_STILL])
    assert not (root / CARTPOLE_STILL).exists(), "an image over budget should not be written"


# --- what the manifest records ------------------------------------------------------------------


def test_an_item_records_how_it_was_made(clear_sdl):
    _, made = make_doc_media.render_item(REPO_ROOT, CARTPOLE_STILL, STILL)

    config_hash = hashlib.sha256((REPO_ROOT / CARTPOLE).read_bytes()).hexdigest()
    assert made["config_sha256"] == config_hash, "the config's SHA-256 should be recorded"
    assert len(made["commit"]) == 40, "the commit the script ran at should be recorded"
    assert made["pillow"] == metadata.version("pillow"), "the Pillow version should be recorded"
    assert made["pygame_ce"] == metadata.version("pygame-ce"), "and pygame-ce's"
    assert made["platform"], "and the platform"
    assert made["numpy"] == metadata.version("numpy"), "and NumPy's"
    assert made["gymnasium"] == metadata.version("gymnasium"), "and Gymnasium's"
    assert made["python"] == platform.python_version(), "and Python's"
    script = REPO_ROOT / "scripts" / "make_doc_media.py"
    assert made["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest(), (
        "and the SHA-256 of the script that made it"
    )
    assert "checkpoint_sha256" not in made, "a random agent has no checkpoint to record"


def test_a_checkpoint_is_loaded_and_its_hash_recorded(tmp_path, clear_sdl):
    from train import load_run_config, numpy_agent_config, select_run_challenge

    config = "configs/cartpole_dqn.yaml"
    env, _ = make_doc_media.build(REPO_ROOT, {"config": config})
    settings, agent_type = load_run_config(str(REPO_ROOT / config))
    challenge = select_run_challenge(settings, agent_type)
    # Seed 9's untrained network pushes left from this start; the config's own (seed 42) pushes right.
    other = challenge.make_agent(agent_type, {**numpy_agent_config(settings), "seed": 9}, env)
    env.close()
    checkpoint = tmp_path / "agent.pt"
    other.save(str(checkpoint))
    recipe = {**STILL, "config": config, "checkpoint": str(checkpoint), "release_asset": "dqn.pt"}

    _, made = make_doc_media.render_item(REPO_ROOT, CARTPOLE_STILL, {**recipe, "frame": 1})
    loaded = make_doc_media.play(REPO_ROOT, recipe, frame_step=10**6)
    fresh = make_doc_media.play(REPO_ROOT, {**recipe, "checkpoint": None}, frame_step=10**6)

    assert loaded.actions != fresh.actions, "the checkpoint's weights, not the config's, should act"
    assert made["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert made["release_asset"] == "dqn.pt", "the release asset to fetch it from is recorded"
    checkpoint.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="pickled"):  # NumPy refuses a file that is no archive
        make_doc_media.render_item(REPO_ROOT, CARTPOLE_STILL, recipe)


def test_make_writes_one_item_and_records_it_leaving_the_rest(tmp_path, clear_sdl):
    other = {"path": "docs/images/other.png", "made_by": "another command"}
    root = _root_with(tmp_path, [{"path": CARTPOLE_STILL, "made_by": "x", "recipe": STILL}, other])

    written = make_doc_media.make(root, [CARTPOLE_STILL])

    assert written == [CARTPOLE_STILL], "only the item asked for is made"
    assert (root / CARTPOLE_STILL).is_file(), "the still is written under docs/images/"
    manifest = _manifest(root)
    entry, kept = manifest["media"]
    assert entry["made_by"] == f"python scripts/make_doc_media.py {CARTPOLE_STILL}"
    assert entry["made_with"]["config"] == CARTPOLE, "the entry records its config"
    assert kept == other, "an entry the script did not make is left as it was"
    assert manifest["budget"]["png_kib"] == 60, "the budget is kept"


def _git(root: Path, *args: str) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}  # never the real repo
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        env=env,
    )


def test_a_git_failure_is_recorded_as_unknown_not_clean(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(make_doc_media.subprocess, "run", no_git)
    made = make_doc_media.made_with(tmp_path, {})
    assert made["commit"] == "unknown", "no commit is known without git"
    assert made["tree"] == "unknown", "a tree that could not be checked is not clean"


def test_the_tree_is_clean_when_only_outputs_changed_and_modified_when_an_input_did(
    tmp_path, clear_sdl
):
    second = "docs/images/cartpole_second.png"
    media = [
        {"path": CARTPOLE_STILL, "made_by": "x", "recipe": STILL},
        {"path": second, "made_by": "x", "recipe": {**STILL, "frame": 4}},
    ]
    root = _root_with(tmp_path, media)
    (root / "packages").mkdir()
    (root / "packages" / "code.py").write_text("X = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "start")

    make_doc_media.make(root, None)
    entries = _manifest(root)["media"]
    assert [entry["made_with"]["tree"] for entry in entries] == ["clean", "clean"], (
        "the script's own outputs (images, manifest) should not make the tree count as modified"
    )
    _git(root, "add", "-A")  # staged outputs, as in a pull request before its commit
    (root / "README.md").write_text("# Demo, edited\n", encoding="utf-8")
    make_doc_media.make(root, [second])
    assert _manifest(root)["media"][1]["made_with"]["tree"] == "clean", "Markdown is no input"

    (root / "packages" / "code.py").write_text("X = 2\n", encoding="utf-8")
    make_doc_media.make(root, [second])
    assert _manifest(root)["media"][1]["made_with"]["tree"] == "modified", (
        "an edited package file is an input the commit does not hold"
    )


def test_make_all_skips_entries_made_by_other_commands(tmp_path, clear_sdl):
    other = {"path": "docs/images/other.png", "made_by": "another command"}
    root = _root_with(tmp_path, [{"path": CARTPOLE_STILL, "made_by": "x", "recipe": STILL}, other])

    assert make_doc_media.make(root, None) == [CARTPOLE_STILL], "--all makes what has a recipe"


@pytest.mark.parametrize("path", ["docs/images/other.png", "docs/images/missing.png"])
def test_make_refuses_an_item_it_cannot_make(tmp_path, path):
    root = _root_with(tmp_path, [{"path": "docs/images/other.png", "made_by": "another command"}])

    with pytest.raises(make_doc_media.MediaError, match=path):
        make_doc_media.make(root, [path])


def test_main_needs_paths_or_all(capsys):
    with pytest.raises(SystemExit):
        make_doc_media.main([])
    assert "--all" in capsys.readouterr().err, "the usage error should name --all"

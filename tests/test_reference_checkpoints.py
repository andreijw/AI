"""Checkpoints written by an earlier version of the code still load, and save back the same.

``tests/data/checkpoints/`` holds one small checkpoint per agent, written after a few seeded updates
by ``scripts/write_reference_checkpoints.py``; ``manifest.json`` records each entry's key, dtype,
shape and a digest of its values. A change to the checkpoint code must keep every file loading and
every entry, read back or saved again, as recorded. The files are not compared byte for byte:
``np.savez_compressed`` stamps zip times.

The ``torch_*`` checkpoints need PyTorch and skip without it, as the other PyTorch tests do (CI).
"""

import json
import zipfile

import numpy as np
import pytest

from tests.agent_internals import device_arrays
from tests.helpers import load_script, needs_torch

writer = load_script("write_reference_checkpoints")
MANIFEST = json.loads((writer.OUTPUT_DIR / writer.MANIFEST).read_text(encoding="utf-8"))


def _cases():
    return [
        pytest.param(name, marks=[needs_torch] if writer.needs_torch(name) else [], id=name)
        for name in writer.CASES
    ]


def test_every_case_has_a_reference_checkpoint():
    """A case added to the writer without its file (or a file left behind) fails here."""
    assert set(MANIFEST) == set(writer.CASES)
    on_disk = {path.stem for path in writer.OUTPUT_DIR.glob("*.npz")}
    assert on_disk == set(writer.CASES)


@pytest.mark.parametrize("name", _cases())
def test_reference_checkpoint_holds_the_recorded_entries(name):
    """Every key, dtype, shape and value of the stored file is as the manifest records it."""
    recorded = MANIFEST[name]
    assert recorded["agent"] == writer.CASES[name][0]
    assert recorded["config"] == writer.CASES[name][2]

    entries = writer.describe_file(writer.OUTPUT_DIR / f"{name}.npz")

    assert sorted(entries) == sorted(recorded["entries"]), f"{name}: the keys differ"
    assert entries == recorded["entries"], f"{name}: a dtype, shape or value differs"


@pytest.mark.parametrize("name", _cases())
def test_reference_checkpoint_loads_and_saves_back_the_same(name, tmp_path):
    """Loaded into a fresh, untrained agent and saved again, the checkpoint gives back every
    entry it held, unchanged, and no other. A PyTorch backend also holds every loaded network and
    Adam state on its device: saving reads its NumPy mirror, which would hide a missed copy."""
    agent = writer.build(name)
    path = writer.OUTPUT_DIR / f"{name}.npz"
    agent.load(str(path))
    if writer.needs_torch(name):
        with np.load(path) as data:
            for key, value in device_arrays(agent).items():
                np.testing.assert_array_equal(value, data[key], err_msg=f"{name}: device {key}")
    agent.save(str(tmp_path / "again.npz"))

    entries = writer.describe_file(tmp_path / "again.npz")

    recorded = MANIFEST[name]["entries"]
    assert sorted(entries) == sorted(recorded), f"{name}: saving again changed the keys"
    differing = [key for key in recorded if entries[key] != recorded[key]]
    assert not differing, f"{name}: saving again changed {differing}"


def test_writer_still_writes_the_recorded_layout(tmp_path):
    """Run now, the writer gives every NumPy case the keys, dtypes and shapes the manifest
    records; values are not compared, as BLAS builds differ in the last bit."""
    assert writer.main(["--output-dir", str(tmp_path), "--numpy-only"]) == 0
    written = json.loads((tmp_path / writer.MANIFEST).read_text(encoding="utf-8"))
    for name in (n for n in writer.CASES if not writer.needs_torch(n)):
        layout = {key: (e["dtype"], e["shape"]) for key, e in written[name]["entries"].items()}
        recorded = MANIFEST[name]["entries"]
        assert layout == {key: (e["dtype"], e["shape"]) for key, e in recorded.items()}, name


@pytest.mark.parametrize("name", ["reinforce", "dqn"])
def test_a_saved_checkpoint_is_a_compressed_npz(name, tmp_path):
    """``save`` writes ``np.savez_compressed``'s format, every member deflated (CONTRIBUTING.md §5):
    the entries compared above would read the same from an uncompressed archive."""
    agent = writer.build(name)
    agent.load(str(writer.OUTPUT_DIR / f"{name}.npz"))
    agent.save(str(tmp_path / "again.npz"))

    with zipfile.ZipFile(tmp_path / "again.npz") as archive:
        members = archive.infolist()
    assert members, f"{name}: the checkpoint is empty"
    assert {member.compress_type for member in members} == {zipfile.ZIP_DEFLATED}


def test_the_released_checkpoint_names_still_import_from_base_agent():
    """rl-core 0.1.0 defined these in ``rl_core.agents.base_agent``; the move to ``checkpointing`` keeps them there."""
    from rl_core.agents import base_agent, checkpointing

    assert base_agent.GENERATOR_KEY_PREFIX is checkpointing.GENERATOR_KEY_PREFIX
    assert base_agent.OBSERVATION_KEY_PREFIX is checkpointing.OBSERVATION_KEY_PREFIX
    assert base_agent.OBS_NORM_KEY_PREFIX is checkpointing.OBS_NORM_KEY_PREFIX
    assert base_agent.check_checkpoint_shapes is checkpointing.check_checkpoint_shapes

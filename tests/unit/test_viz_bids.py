"""What the viewer learns from a file's place in the dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.viz import bids as B


@pytest.mark.parametrize("name, ext, stem", [
    ("sub-01_T1w.nii.gz", ".nii.gz", "sub-01_T1w"),
    ("sub-01_T1w.nii", ".nii", "sub-01_T1w"),
    ("sub-01_task-a_physio.tsv.gz", ".tsv.gz", "sub-01_task-a_physio"),
    ("sub-01_meg.fif", ".fif", "sub-01_meg"),
    ("x.mgz", ".mgz", "x"),
])
def test_extensions_and_stems(name, ext, stem) -> None:
    assert B.full_ext(name) == ext
    assert B.stem_of(Path(name)) == stem


def test_entities_and_suffix() -> None:
    name = "sub-01_ses-pre_task-rest_run-02_bold.nii.gz"
    assert B.parse_entities(name) == {"sub": "01", "ses": "pre", "task": "rest", "run": "02"}
    assert B.suffix_of(name) == "bold"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_sidecar_inheritance_nearest_file_wins(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    func = root / "sub-01" / "func"
    func.mkdir(parents=True)
    bold = func / "sub-01_task-rest_bold.nii.gz"
    bold.write_bytes(b"")
    _write(root / "task-rest_bold.json", {"RepetitionTime": 2.0, "TaskName": "rest"})
    _write(root / "task-other_bold.json", {"RepetitionTime": 9.0})
    _write(func / "sub-01_task-rest_bold.json", {"RepetitionTime": 1.5})
    sc = B.inherited_sidecar(bold, root)
    assert sc["RepetitionTime"] == 1.5
    assert sc["TaskName"] == "rest"


def test_context_of_a_bold_run(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    func = root / "sub-01" / "func"
    func.mkdir(parents=True)
    bold = func / "sub-01_task-rest_run-1_bold.nii.gz"
    bold.write_bytes(b"")
    _write(func / "sub-01_task-rest_run-1_bold.json", {"RepetitionTime": 2.0})
    (func / "sub-01_task-rest_run-1_events.tsv").write_text("onset\tduration\n0\t1\n")
    for rec in ("cardiac", "respiratory"):
        (func / f"sub-01_task-rest_run-1_recording-{rec}_physio.tsv.gz").write_bytes(b"")
    ctx = B.bids_context(bold, root)
    assert ctx.datatype == "func" and ctx.suffix == "bold"
    assert ctx.entities["run"] == "1"
    assert ctx.tr == 2.0
    assert ctx.frame_times(4).tolist() == [0.0, 2.0, 4.0, 6.0]
    assert ctx.events_path is not None and ctx.events_path.name.endswith("_events.tsv")
    assert len(ctx.physio_paths) == 2
    assert ctx.rel == "sub-01/func/sub-01_task-rest_run-1_bold.nii.gz"


def test_pet_frame_times_are_mid_frame(tmp_path: Path) -> None:
    pet = tmp_path / "sub-01" / "pet" / "sub-01_pet.nii.gz"
    pet.parent.mkdir(parents=True)
    pet.write_bytes(b"")
    _write(pet.with_name("sub-01_pet.json"),
           {"FrameTimesStart": [0, 10, 30], "FrameDuration": [10, 20, 60]})
    ctx = B.bids_context(pet, tmp_path)
    assert ctx.frame_times(3).tolist() == [5.0, 20.0, 60.0]
    # A frame count that disagrees with the sidecar is not guessed at.
    assert ctx.frame_times(5) is None


def test_frame_times_start_is_read_only_for_pet(tmp_path: Path) -> None:
    """``FrameTimesStart`` is a PET field: a BOLD run carrying one stray key
    keeps its own clock."""
    bold = tmp_path / "sub-01_task-a_bold.nii.gz"
    bold.write_bytes(b"")
    _write(bold.with_name("sub-01_task-a_bold.json"), {"FrameTimesStart": [0, 10, 30]})
    assert B.bids_context(bold).frame_times(3) is None


def test_sparse_bold_uses_volume_timing(tmp_path: Path) -> None:
    bold = tmp_path / "sub-01_task-a_bold.nii.gz"
    bold.write_bytes(b"")
    _write(bold.with_name("sub-01_task-a_bold.json"), {"VolumeTiming": [0, 5, 12]})
    assert B.bids_context(bold).frame_times(3).tolist() == [0.0, 5.0, 12.0]


def test_starts_without_durations_are_used_as_they_are(tmp_path: Path) -> None:
    pet = tmp_path / "sub-01_pet.nii.gz"
    pet.write_bytes(b"")
    _write(pet.with_name("sub-01_pet.json"), {"FrameTimesStart": [0, 10, 30]})
    assert B.bids_context(pet).frame_times(3).tolist() == [0.0, 10.0, 30.0]


@pytest.mark.parametrize("sidecar", [
    {},
    {"FrameTimesStart": "not a list"},
    {"FrameTimesStart": [0, "x", 2]},
    {"RepetitionTime": -1},
])
def test_unusable_timing_gives_no_times(tmp_path: Path, sidecar: dict) -> None:
    img = tmp_path / "sub-01_pet.nii.gz"
    img.write_bytes(b"")
    _write(img.with_name("sub-01_pet.json"), sidecar)
    assert B.bids_context(img).frame_times(3) is None


def test_a_broken_sidecar_is_not_an_error(tmp_path: Path) -> None:
    img = tmp_path / "sub-01_T1w.nii.gz"
    img.write_bytes(b"")
    img.with_name("sub-01_T1w.json").write_text("{ not json", encoding="utf-8")
    ctx = B.bids_context(img)
    assert ctx.sidecar == {} and ctx.tr is None


def test_b_values(tmp_path: Path) -> None:
    dwi = tmp_path / "sub-01" / "dwi" / "sub-01_dwi.nii.gz"
    dwi.parent.mkdir(parents=True)
    dwi.write_bytes(b"")
    dwi.with_name("sub-01_dwi.bval").write_text("0 1000 1000 2000\n")
    ctx = B.bids_context(dwi, tmp_path)
    assert ctx.bvals.tolist() == [0.0, 1000.0, 1000.0, 2000.0]
    assert ctx.datatype == "dwi"


def test_intended_for_in_both_spellings(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    fmap = root / "sub-01" / "fmap" / "sub-01_phasediff.nii.gz"
    fmap.parent.mkdir(parents=True)
    fmap.write_bytes(b"")
    _write(fmap.with_name("sub-01_phasediff.json"), {"IntendedFor": [
        "bids::sub-01/func/sub-01_task-a_bold.nii.gz",
        "func/sub-01_task-b_bold.nii.gz",
    ]})
    ctx = B.bids_context(fmap, root)
    rels = [p.relative_to(root).as_posix() for p in ctx.intended_for]
    assert rels == ["sub-01/func/sub-01_task-a_bold.nii.gz",
                    "sub-01/func/sub-01_task-b_bold.nii.gz"]


def test_outside_a_dataset_the_name_is_the_relative_path(tmp_path: Path) -> None:
    img = tmp_path / "loose.nii.gz"
    img.write_bytes(b"")
    ctx = B.bids_context(img)
    assert ctx.rel == "loose.nii.gz"
    assert ctx.datatype == ""
    assert isinstance(ctx.frame_times(1), (type(None), np.ndarray))

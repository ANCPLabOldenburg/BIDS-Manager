"""Defacing and skull stripping the images under ``derivatives/``.

A preprocessed T1w carries the face of the scan it came from, so a dataset
shared with its derivatives needs them done too. Off unless asked for; when
asked, only images of the head (no masks, segmentations or maps), at the
subjects and sessions the scope names, and a skull strip of one never lands
inside the other pipeline's folder or on top of another output.
"""

from __future__ import annotations

import gzip
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from bidsmgr.deface import engines, run, status
from bidsmgr.deface.apply import deface_dataset
from bidsmgr.deface.derivatives import (
    PIPELINE,
    derivative_desc,
    derivative_path,
)
from bidsmgr.deface.engines import TEMPLATE
from bidsmgr.deface.select import (
    Skip,
    derivative_pipelines,
    has_derivatives,
    targets_in_derivatives,
    walk,
)
from bidsmgr.project.operations import undo_last

T1 = "sub-01/anat/sub-01_T1w.nii.gz"
PREPROC = "derivatives/fmriprep/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz"
MNI = ("derivatives/fmriprep/sub-01/anat/"
       "sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")
MASK = "derivatives/fmriprep/sub-01/anat/sub-01_desc-brain_mask.nii.gz"
DSEG = "derivatives/fmriprep/sub-01/anat/sub-01_dseg.nii.gz"
BOLD = "derivatives/fmriprep/sub-01/func/sub-01_task-rest_desc-preproc_bold.nii.gz"
SUB2 = "derivatives/fmriprep/sub-02/anat/sub-02_desc-preproc_T1w.nii.gz"


def _header(path: Path, dim) -> Path:
    """A NIfTI-1 header with dim[] set; enough for the selector."""
    raw = bytearray(352)
    struct.pack_into("<i", raw, 0, 348)
    struct.pack_into("<8h", raw, 40, *dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(bytes(raw)))
    return path


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"}))
    for rel in (T1, PREPROC, MNI, MASK, DSEG, SUB2):
        _header(root / rel, (3, 8, 8, 8, 1, 1, 1, 1))
    _header(root / BOLD, (4, 8, 8, 8, 20, 1, 1, 1))
    (root / "derivatives/fmriprep/dataset_description.json").write_text(
        json.dumps({"Name": "fMRIPrep", "DatasetType": "derivative"}))
    return root


# ---------------------------------------------------------------- selecting

def test_derivatives_are_left_out_unless_asked_and_say_how_to_include_them(tmp_path):
    root = _dataset(tmp_path)
    sel = walk(root)
    assert [c.relative for c in sel.candidates] == [T1]
    reasons = {s.relative: s.reason for s in sel.skipped}
    assert reasons[PREPROC] is Skip.DERIVATIVE
    assert "Include derivatives" in Skip.DERIVATIVE.value


def test_asked_for_only_images_of_the_head_are_candidates(tmp_path):
    root = _dataset(tmp_path)
    sel = walk(root, include_derivatives=True)
    got = [c.relative for c in sel.candidates]
    assert got == [T1, PREPROC, MNI, SUB2]
    reasons = {s.relative: s.reason for s in sel.skipped}
    assert reasons[MASK] is Skip.NOT_HEAD
    assert reasons[DSEG] is Skip.NOT_HEAD
    assert reasons[BOLD] is Skip.DATATYPE


def test_each_candidate_says_which_pipeline_it_belongs_to(tmp_path):
    root = _dataset(tmp_path)
    by_rel = {c.relative: c.pipeline for c in walk(root, include_derivatives=True).candidates}
    assert by_rel[T1] == ""
    assert by_rel[PREPROC] == "fmriprep"


def test_a_subject_scope_takes_the_same_subject_in_every_pipeline(tmp_path):
    root = _dataset(tmp_path)
    sel = walk(root, [root / "sub-01"], include_derivatives=True)
    got = {c.relative for c in sel.candidates}
    assert got == {T1, PREPROC, MNI}, "sub-02's derivative is outside the scope"


def test_a_datatype_scope_takes_the_same_folder_in_every_pipeline(tmp_path):
    root = _dataset(tmp_path)
    sel = walk(root, [root / "sub-01" / "anat"], include_derivatives=True)
    assert {c.relative for c in sel.candidates} == {T1, PREPROC, MNI}


def test_a_derivative_picked_in_the_tree_is_a_candidate_on_its_own(tmp_path):
    root = _dataset(tmp_path)
    sel = walk(root, [root / PREPROC], include_derivatives=True)
    assert [c.relative for c in sel.candidates] == [PREPROC]
    assert targets_in_derivatives(root, [root / PREPROC])
    assert not targets_in_derivatives(root, [root / T1])
    assert not targets_in_derivatives(root, None)


def test_a_raw_defacemask_is_not_defaced(tmp_path):
    root = _dataset(tmp_path)
    _header(root / "sub-01/anat/sub-01_defacemask.nii.gz", (3, 8, 8, 8, 1, 1, 1, 1))
    reasons = {s.relative: s.reason for s in walk(root).skipped}
    assert reasons["sub-01/anat/sub-01_defacemask.nii.gz"] is Skip.NOT_HEAD


def test_our_own_skull_strip_is_never_offered_again(tmp_path):
    root = _dataset(tmp_path)
    out = root / f"derivatives/{PIPELINE}/sub-01/anat/sub-01_desc-brain_T1w.nii.gz"
    _header(out, (3, 8, 8, 8, 1, 1, 1, 1))
    status.sidecar_for(out).write_text(json.dumps(
        status.record({}, engines.engine("strip-atlas"))))
    reasons = {s.relative: s.reason
               for s in walk(root, include_derivatives=True).skipped}
    assert reasons[out.relative_to(root).as_posix()] is Skip.STRIPPED


def test_pipelines_are_the_folders_with_subject_data(tmp_path):
    root = _dataset(tmp_path)
    (root / "derivatives" / "empty").mkdir()
    (root / "derivatives" / ".cache" / "sub-01").mkdir(parents=True)
    assert [p.name for p in derivative_pipelines(root)] == ["fmriprep"]
    assert has_derivatives(root)
    bare = tmp_path / "bare"
    (bare / "sub-01").mkdir(parents=True)
    assert not has_derivatives(bare)


# ------------------------------------------------------- where a strip goes

def test_a_raw_image_strips_to_desc_brain(tmp_path):
    root = _dataset(tmp_path)
    assert derivative_path(root, root / T1) == (
        root / f"derivatives/{PIPELINE}/sub-01/anat/sub-01_desc-brain_T1w.nii.gz")


def test_a_derivative_strips_into_our_pipeline_with_its_own_desc(tmp_path):
    root = _dataset(tmp_path)
    assert derivative_desc(root, root / PREPROC) == "preprocbrain"
    assert derivative_path(root, root / PREPROC) == (
        root / f"derivatives/{PIPELINE}/sub-01/anat/sub-01_desc-preprocbrain_T1w.nii.gz")


def test_other_entities_of_a_derivative_survive(tmp_path):
    root = _dataset(tmp_path)
    target = derivative_path(root, root / MNI)
    assert target.name == ("sub-01_space-MNI152NLin2009cAsym_desc-preprocbrain_"
                           "T1w.nii.gz")


def test_a_derivative_without_desc_is_named_after_its_pipeline(tmp_path):
    root = _dataset(tmp_path)
    img = _header(root / "derivatives/anat-defaced/sub-01/anat/sub-01_T1w.nii.gz",
                  (3, 8, 8, 8, 1, 1, 1, 1))
    assert derivative_path(root, img).name == "sub-01_desc-anatdefacedbrain_T1w.nii.gz"


def test_nothing_is_written_inside_another_pipeline_or_from_our_own(tmp_path):
    root = _dataset(tmp_path)
    target = derivative_path(root, root / PREPROC)
    assert "fmriprep" not in target.relative_to(root).parts
    ours = root / f"derivatives/{PIPELINE}/sub-01/anat/sub-01_desc-brain_T1w.nii.gz"
    assert derivative_path(root, ours) is None
    assert derivative_path(root, root / "derivatives/fmriprep/figures/x.nii.gz") is None


# ----------------------------------------------------------------- applying

needs_niimath = pytest.mark.skipif(
    not run.available(), reason="needs the niimath wheel for this Python")
needs_strip = pytest.mark.skipif(
    not run.available("strip-atlas"), reason="needs the niimath wheel for this Python")


def _real(root: Path, rel: str) -> Path:
    """The atlas template as a head: it registers to itself."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, path)
    status.sidecar_for(path).write_text(json.dumps({"SkullStripped": False}))
    return path


@needs_niimath
def test_a_derivative_is_defaced_in_place_and_undone_byte_for_byte(tmp_path):
    root = tmp_path / "ds"
    (root / "sub-01").mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "t"}))
    img = _real(root, PREPROC)
    before = img.read_bytes()
    sidecar_before = status.sidecar_for(img).read_text()

    out = deface_dataset(root, include_derivatives=True)
    assert out.ok, out.failed
    assert out.defaced == [PREPROC]
    assert img.read_bytes() != before
    assert status.defaced_by_us(status.read_sidecar(status.sidecar_for(img)))
    assert not (root / "sourcedata").exists()

    assert undo_last(root)
    assert img.read_bytes() == before
    assert status.sidecar_for(img).read_text() == sidecar_before


@needs_niimath
def test_keeping_originals_in_sourcedata_never_mirrors_a_derivative(tmp_path):
    root = tmp_path / "ds"
    (root / "sub-01").mkdir(parents=True)
    _real(root, T1)
    _real(root, PREPROC)
    out = deface_dataset(root, include_derivatives=True,
                         keep_original_in_sourcedata=True)
    assert out.ok, out.failed
    assert (root / "sourcedata" / T1).is_file()
    assert not (root / "sourcedata" / "derivatives").exists()


@needs_strip
def test_a_derivative_is_stripped_into_our_pipeline_naming_its_source(tmp_path):
    root = tmp_path / "ds"
    (root / "sub-01").mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "t"}))
    img = _real(root, PREPROC)
    before = img.read_bytes()

    out = deface_dataset(root, engine_id="strip-atlas", include_derivatives=True)
    assert out.ok, out.failed
    assert img.read_bytes() == before, "the source derivative was overwritten"
    want = f"derivatives/{PIPELINE}/sub-01/anat/sub-01_desc-preprocbrain_T1w.nii.gz"
    assert out.produced == [want]
    doc = json.loads(status.sidecar_for(root / want).read_text())
    assert doc["Sources"] == [f"bids::{PREPROC}"]
    assert doc["SkullStripped"] is False or "SkullStripped" in doc


@needs_strip
def test_two_sources_that_would_share_a_name_do_not_overwrite_each_other(tmp_path):
    root = tmp_path / "ds"
    (root / "sub-01").mkdir(parents=True)
    a = "derivatives/pipea/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz"
    b = "derivatives/pipeb/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz"
    _real(root, a)
    _real(root, b)
    out = deface_dataset(root, engine_id="strip-atlas", include_derivatives=True)
    assert len(out.produced) == 1
    skipped = dict(out.skipped)
    assert b in skipped and a in skipped[b]


# ------------------------------------------------------------------ the CLI

def test_the_command_line_lists_derivatives_only_when_asked(tmp_path):
    root = _dataset(tmp_path)
    plain = subprocess.run(
        [sys.executable, "-m", "bidsmgr.cli.deface", str(root), "--dry-run"],
        capture_output=True, text=True, check=True).stdout
    asked = subprocess.run(
        [sys.executable, "-m", "bidsmgr.cli.deface", str(root), "--dry-run",
         "--include-derivatives"],
        capture_output=True, text=True, check=True).stdout
    def listed(text: str) -> bool:
        return any(line.strip() == PREPROC for line in text.splitlines())

    assert not listed(plain) and f"{PREPROC}: in derivatives/" in plain
    assert listed(asked)

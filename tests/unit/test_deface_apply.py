"""Defacing a dataset, with the real engine, on an image small enough for CI.

Nothing here asserts that the *right* voxels were removed. What is asserted is
everything around that, which is where the dataset-level mistakes live: the
history entry, the sidecar, the part-ticked selection, the staging cleanup, and
whether a face-bearing copy was left inside the dataset.

The one that matters most is the round trip. Defacing is destructive and
cannot be undone by hand, so "deface, undo, byte-identical" is the test that
justifies letting a user press the button.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from bidsmgr.deface import run, status
from bidsmgr.deface.apply import deface_dataset, sourcedata_mirror
from bidsmgr.deface.select import walk
from bidsmgr.project.operations import read_log, undo_last

pytestmark = pytest.mark.skipif(
    not run.available(),
    reason="needs the niimath wheel, which has no build for this Python",
)


def _volume(path: Path) -> Path:
    """A head-shaped test subject: the atlas template itself.

    A synthetic blob does not work here, and the reason is worth recording.
    niimath registers the subject to the template with a 12-DOF affine; a
    24-cubed cube with an identity affine is 24 mm across where the template is
    180 mm, so the fit is degenerate and the engine correctly refuses it with
    "no valid overlap". Building a synthetic head convincing enough to register
    would be a test fixture with its own bugs.

    The template registers to itself trivially and is 567 KB, so this stays
    fast. Nothing below asserts which voxels were removed, only what happens to
    the dataset around them, so a circular subject costs nothing.
    """
    from bidsmgr.deface.engines import TEMPLATE

    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, path)
    return path


def _dataset(tmp_path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"})
    )
    img = _volume(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    status.sidecar_for(img).write_text(json.dumps({"Manufacturer": "Siemens"}))
    return root


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_deface_then_undo_is_byte_identical(tmp_path):
    root = _dataset(tmp_path)
    img = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    sidecar = status.sidecar_for(img)
    before_img, before_json = _sha(img), sidecar.read_text()

    out = deface_dataset(root)
    assert out.ok, out.failed
    assert _sha(img) != before_img, "the image was not changed"

    undo_last(root)
    assert _sha(img) == before_img
    assert sidecar.read_text() == before_json


def test_the_sidecar_records_both_bids_fields(tmp_path):
    root = _dataset(tmp_path)
    deface_dataset(root, engine_id="allineate")
    doc = json.loads(status.sidecar_for(
        root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    ).read_text())

    assert doc["Manufacturer"] == "Siemens", "an unrelated field was lost"
    assert status.defaced_by_us(doc) is not None
    assert doc[status.METHOD][0].startswith("BIDS Manager")


def test_the_whole_run_is_one_entry_in_the_history(tmp_path):
    root = _dataset(tmp_path)
    _volume(root / "sub-01" / "anat" / "sub-01_T2w.nii.gz")

    out = deface_dataset(root)
    assert len(out.defaced) == 2
    log = read_log(root)
    assert len(log) == 1
    assert "2 images" in log[0]["label"]


def test_only_narrows_the_run_to_a_part_ticked_preview(tmp_path):
    root = _dataset(tmp_path)
    other = _volume(root / "sub-01" / "anat" / "sub-01_T2w.nii.gz")
    keep = _sha(other)

    sel = walk(root)
    assert len(sel.candidates) == 2
    out = deface_dataset(
        root, selection=sel, only=["sub-01/anat/sub-01_T1w.nii.gz"],
    )

    assert out.defaced == ["sub-01/anat/sub-01_T1w.nii.gz"]
    assert _sha(other) == keep, "an unticked file was defaced anyway"


def test_nothing_is_left_behind_in_staging(tmp_path):
    root = _dataset(tmp_path)
    deface_dataset(root)
    leftovers = list((root / ".bidsmgr").rglob("deface-staging/*"))
    assert not leftovers, f"temporary files survived: {leftovers}"


def test_the_sourcedata_mirror_is_opt_in(tmp_path):
    root = _dataset(tmp_path)
    deface_dataset(root)
    assert not (root / "sourcedata").exists(), (
        "the pristine image was written into the dataset without being asked "
        "for; it still contains the face"
    )


def test_the_sourcedata_mirror_holds_the_original_when_asked(tmp_path):
    root = _dataset(tmp_path)
    img = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    before = _sha(img)

    deface_dataset(root, keep_original_in_sourcedata=True)

    mirror = root / "sourcedata" / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    assert mirror.is_file()
    assert _sha(mirror) == before
    assert _sha(img) != before


def test_a_second_deface_does_not_overwrite_the_mirror(tmp_path):
    """Or the pristine copy becomes a defaced one and is worth nothing."""
    root = _dataset(tmp_path)
    img = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    before = _sha(img)

    deface_dataset(root, keep_original_in_sourcedata=True)
    deface_dataset(root, keep_original_in_sourcedata=True,
                   engine_id="allineate-robust")

    mirror = root / "sourcedata" / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    assert _sha(mirror) == before


def test_undo_also_removes_the_mirror_it_created(tmp_path):
    root = _dataset(tmp_path)
    deface_dataset(root, keep_original_in_sourcedata=True)
    mirror = root / "sourcedata" / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    assert mirror.is_file()

    undo_last(root)
    assert not mirror.exists(), (
        "undo left a face-bearing copy inside the dataset"
    )


def test_a_time_series_in_anat_is_skipped_with_its_reason(tmp_path):
    root = _dataset(tmp_path)
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    four_d = np.zeros((8, 8, 8, 5), dtype="float32")
    nib.save(
        nib.Nifti1Image(four_d, np.eye(4)),
        str(root / "sub-01" / "anat" / "sub-01_MEGRE.nii.gz"),
    )

    out = deface_dataset(root)
    assert out.defaced == ["sub-01/anat/sub-01_T1w.nii.gz"]
    assert any("MEGRE" in rel for rel, _ in out.skipped)


def test_sourcedata_mirror_path_refuses_anything_but_subject_data(tmp_path):
    root = tmp_path / "ds"
    assert sourcedata_mirror(root, root / "derivatives" / "x.nii.gz") is None
    assert sourcedata_mirror(root, root / "dataset_description.json") is None
    assert sourcedata_mirror(root, Path("/elsewhere/x.nii.gz")) is None
    assert sourcedata_mirror(root, root / "sub-01" / "anat" / "a.nii.gz") == (
        root / "sourcedata" / "sub-01" / "anat" / "a.nii.gz"
    )


def test_the_neck_cropping_engine_is_allowed_to_change_the_dimensions(tmp_path):
    """The regression for the first version of the shape guard.

    ``-robustfov`` crops the neck, so the output legitimately has fewer slices
    than the input. A guard that demanded an identical shape rejected every
    run of the one engine whose purpose is to change it, on a real T1w, with
    a message telling the user to report a bug that was ours.
    """
    root = _dataset(tmp_path)
    img = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"

    out = deface_dataset(root, engine_id="allineate-robust")
    assert out.ok, out.failed

    nib = pytest.importorskip("nibabel")
    from bidsmgr.deface.engines import TEMPLATE
    assert nib.load(str(img)).shape != nib.load(str(TEMPLATE)).shape, (
        "the image was not cropped, so this no longer tests the guard"
    )

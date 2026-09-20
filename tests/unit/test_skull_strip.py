"""Skull stripping, and where its output belongs.

A defaced image is the same scan with some voxels blanked, so it replaces the
original. A skull-stripped one is not: everything outside the brain has been
discarded by an algorithm that can be wrong, and BIDS has a word for a file an
algorithm produced from raw data. Writing it over the raw image leaves a
dataset whose "raw" data has been processed, and nothing in the tree says so.

So most of what is tested here is the destination, not the voxels.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bidsmgr.deface import engines, run
from bidsmgr.deface.apply import deface_dataset
from bidsmgr.deface.derivatives import (
    PIPELINE,
    dataset_description,
    derivative_name,
    derivative_path,
)
from bidsmgr.deface.engines import TEMPLATE
from bidsmgr.project.operations import read_log, undo_last

REL = "sub-01/anat/sub-01_T1w.nii.gz"


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Study A", "BIDSVersion": "1.11.1"})
    )
    shutil.copyfile(TEMPLATE, root / REL)
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"Manufacturer": "Siemens"})
    )
    return root


# ---------------------------------------------------------------- naming


@pytest.mark.parametrize("name,expected", [
    ("sub-01_T1w.nii.gz", "sub-01_desc-brain_T1w.nii.gz"),
    ("sub-01_ses-a_acq-mprage_T1w.nii",
     "sub-01_ses-a_acq-mprage_desc-brain_T1w.nii"),
    # An existing desc is REPLACED: two of them is not a BIDS name.
    ("sub-01_desc-preproc_T1w.nii.gz", "sub-01_desc-brain_T1w.nii.gz"),
])
def test_desc_brain_lands_immediately_before_the_suffix(name, expected):
    assert derivative_name(name) == expected


def test_the_derivative_mirrors_the_subject_path(tmp_path):
    root = tmp_path / "ds"
    got = derivative_path(root, root / "sub-01" / "ses-a" / "anat" / "sub-01_T1w.nii.gz")
    assert got == (
        root / "derivatives" / PIPELINE / "sub-01" / "ses-a" / "anat"
        / "sub-01_desc-brain_T1w.nii.gz"
    )


def test_there_is_no_derivative_of_a_derivative(tmp_path):
    """Which pipeline would it belong to? Nobody can answer that for the user."""
    root = tmp_path / "ds"
    assert derivative_path(root, root / "derivatives" / "x" / "sub-01" / "a.nii.gz") is None
    assert derivative_path(root, root / "dataset_description.json") is None
    assert derivative_path(root, Path("/elsewhere/a.nii.gz")) is None


def test_the_pipeline_description_says_what_made_it(tmp_path):
    root = _dataset(tmp_path)
    doc = dataset_description(root, engine_label="mindgrab", version="9.9.9")

    assert doc["DatasetType"] == "derivative"
    assert doc["BIDSVersion"] == "1.11.1", "disagrees with the raw dataset"
    assert doc["GeneratedBy"][0]["Name"] == "BIDS Manager"
    assert doc["GeneratedBy"][0]["Version"] == "9.9.9"
    assert "mindgrab" in doc["GeneratedBy"][0]["Description"]
    assert doc["SourceDatasets"][0]["Name"] == "Study A"


# ---------------------------------------------------------------- applying

strip = pytest.mark.skipif(
    not run.available("strip-atlas"),
    reason="needs the niimath wheel, which has no build for this Python",
)


@strip
def test_stripping_leaves_the_raw_image_alone(tmp_path):
    root = _dataset(tmp_path)
    before = (root / REL).read_bytes()

    out = deface_dataset(root, engine_id="strip-atlas")

    assert out.ok, out.failed
    assert (root / REL).read_bytes() == before, "the raw image was overwritten"
    assert out.produced == [
        f"derivatives/{PIPELINE}/sub-01/anat/sub-01_desc-brain_T1w.nii.gz"
    ]
    assert (root / out.produced[0]).is_file()


@strip
def test_the_derivative_folder_is_a_valid_derivative_dataset(tmp_path):
    """Without its dataset_description it is a folder, not a dataset."""
    root = _dataset(tmp_path)
    deface_dataset(root, engine_id="strip-atlas")

    doc = json.loads(
        (root / "derivatives" / PIPELINE / "dataset_description.json").read_text()
    )
    assert doc["DatasetType"] == "derivative"


@strip
def test_the_derivative_sidecar_keeps_the_source_fields_and_names_its_source(
    tmp_path,
):
    root = _dataset(tmp_path)
    out = deface_dataset(root, engine_id="strip-atlas")

    doc = json.loads(
        (root / out.produced[0]).with_suffix("").with_suffix(".json").read_text()
    )
    assert doc["Manufacturer"] == "Siemens"
    assert doc["Sources"] == [f"bids::{REL}"]
    from bidsmgr.deface import status
    assert status.defaced_by_us(doc) is not None


@strip
def test_the_raw_sidecar_is_not_told_the_image_was_stripped(tmp_path):
    """It was not. Claiming so would stop anyone defacing it later."""
    root = _dataset(tmp_path)
    deface_dataset(root, engine_id="strip-atlas")

    from bidsmgr.deface import status
    doc = json.loads((root / "sub-01" / "anat" / "sub-01_T1w.json").read_text())
    assert status.defaced_by_us_at_all(doc) is False


@strip
def test_in_place_is_available_and_is_not_the_default(tmp_path):
    root = _dataset(tmp_path)
    before = (root / REL).read_bytes()

    out = deface_dataset(root, engine_id="strip-atlas", in_place=True)

    assert out.ok
    assert (root / REL).read_bytes() != before
    assert not (root / "derivatives").exists()
    assert out.produced == []


@strip
def test_undo_removes_the_whole_derivative(tmp_path):
    root = _dataset(tmp_path)
    out = deface_dataset(root, engine_id="strip-atlas")
    produced = root / out.produced[0]
    assert produced.is_file()

    undo_last(root)

    assert not produced.exists()
    assert not (root / "derivatives" / PIPELINE / "dataset_description.json").exists()


@strip
def test_it_is_one_entry_in_the_history_and_says_strip(tmp_path):
    root = _dataset(tmp_path)
    deface_dataset(root, engine_id="strip-atlas")

    log = read_log(root)
    assert len(log) == 1
    assert "Skull strip" in log[0]["label"], (
        "a strip logged as a deface is a history that lies"
    )


@strip
def test_defacing_still_writes_in_place(tmp_path):
    """The whole point of the split: a defaced image IS raw data."""
    root = _dataset(tmp_path)
    before = (root / REL).read_bytes()

    out = deface_dataset(root, engine_id="allineate")

    assert out.ok
    assert (root / REL).read_bytes() != before
    assert not (root / "derivatives").exists()


# ---------------------------------------------------------------- mindgrab

mindgrab = pytest.mark.skipif(
    not run.available("mindgrab"),
    reason="needs brainchop, which ships with BIDS Manager",
)


@mindgrab
def test_mindgrab_keeps_the_images_own_grid(tmp_path):
    """A derivative on a different grid is not a derivative OF that scan.

    Nothing computed from the original would apply to it, which defeats the
    reason for producing one.
    """
    nib = pytest.importorskip("nibabel")
    import numpy as np

    root = _dataset(tmp_path)
    out = deface_dataset(root, engine_id="mindgrab")
    assert out.ok, out.failed

    src = nib.load(str(root / REL))
    got = nib.load(str(root / out.produced[0]))
    assert got.shape == src.shape
    assert np.allclose(got.affine, src.affine)


def test_an_engine_that_cannot_run_says_how_to_install_it():
    reason = run.unavailable_reason("mindgrab")
    if reason is None:
        pytest.skip("brainchop is installed here")
    assert "brainchop" in reason
    assert "atlas" in reason, "does not mention the engine that needs nothing"


def test_availability_is_per_engine_not_global():
    """Otherwise a missing optional extra greys out the engine that works."""
    assert run.available("strip-atlas") == (
        run.unavailable_reason("strip-atlas") is None
    )
    assert engines.MINDGRAB.backend == engines.BACKEND_BRAINCHOP
    assert engines.STRIP_ATLAS.backend == engines.BACKEND_NIIMATH


# ---------------------------------------------------------------------------
# The orientation regression.
#
# Both engines once produced a brain-shaped region in the WRONG PLACE: the
# mask was built from brainchop's in-memory array paired with its header, and
# those are in opposite axis orders. The output had a plausible volume, a
# plausible intensity profile and a plausible filename, and the brain was cut
# at an angle. Nothing in the old tests could tell the difference, because
# they all measured how MUCH was kept and never WHERE.


def _symmetry(path) -> float:
    """Dice of the kept region against its own left-right mirror."""
    nib = pytest.importorskip("nibabel")
    import numpy as np

    data = np.asanyarray(nib.load(str(path)).dataobj)
    kept = data > data.min()
    return float(2 * (kept & kept[::-1]).sum() / (2 * max(kept.sum(), 1)))


@pytest.mark.parametrize("engine_id", ["strip-atlas", "mindgrab"])
def test_the_kept_region_is_left_right_symmetric(tmp_path, engine_id):
    """A head is symmetric, so a correctly placed brain mask is too.

    A mask rotated with respect to the image is not, which is what makes this
    catch the failure that volume and intensity checks sail past.
    """
    if not run.available(engine_id):
        pytest.skip(f"{engine_id} is not available here")

    root = _dataset(tmp_path)
    out = deface_dataset(root, engine_id=engine_id)
    assert out.ok, out.failed

    produced = root / out.produced[0]
    dice = _symmetry(produced)
    assert dice > 0.85, f"{engine_id} kept an asymmetric region (Dice {dice:.3f})"


@pytest.mark.parametrize("engine_id", ["strip-atlas", "mindgrab"])
def test_the_kept_region_sits_on_the_bright_tissue(tmp_path, engine_id):
    """Where, not how much: a shifted mask keeps air and drops brain."""
    if not run.available(engine_id):
        pytest.skip(f"{engine_id} is not available here")
    nib = pytest.importorskip("nibabel")
    import numpy as np

    root = _dataset(tmp_path)
    out = deface_dataset(root, engine_id=engine_id)
    assert out.ok, out.failed

    source = np.asanyarray(nib.load(str(root / REL)).dataobj).astype(float)
    got = np.asanyarray(nib.load(str(root / out.produced[0])).dataobj).astype(float)
    kept = got > got.min()
    # The template is an average brain, so almost everything bright in it
    # belongs to the brain and should survive.
    bright = source > np.percentile(source[source > 0], 50)
    precision = float((kept & bright).sum() / max(kept.sum(), 1))
    assert precision > 0.80, (
        f"{engine_id} kept {1 - precision:.0%} background or skull"
    )


@mindgrab
def test_the_parent_process_never_loads_tinygrad(tmp_path):
    """The fix for a crash that happened AFTER the work succeeded.

    brainchop runs on tinygrad, which sets up a GPU runtime on first use.
    Doing that on a QThread and then letting the thread go took the whole
    application down once the strip had finished and the file was written: a
    correct result, and then no window. So the model runs in a process of its
    own and the parent must never import it.
    """
    import sys

    root = _dataset(tmp_path)
    out = deface_dataset(root, engine_id="mindgrab")
    assert out.ok, out.failed

    leaked = [m for m in ("tinygrad", "brainchop") if m in sys.modules]
    assert not leaked, (
        f"{leaked} was imported into the parent process; the crash this "
        "guards against comes back the moment it is"
    )


# ---------------------------------------------------------------------------
# Orientation independence.
#
# The rotation bug was found on one image, so "it works now" could have meant
# "it happens to work on that one". Nothing in either engine is allowed to
# assume a storage order: niimath registers through the affine, and the
# mindgrab mask is mapped back through world coordinates. This checks that
# claim instead of trusting it.


def _reoriented(image: Path, axcodes: tuple, out: Path) -> Path:
    """The SAME image, written in a different axis order.

    World content is untouched: only how the voxels are stored changes, which
    is exactly the thing an engine must not care about.
    """
    nib = pytest.importorskip("nibabel")

    img = nib.load(str(image))
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(img.affine),
        nib.orientations.axcodes2ornt(axcodes),
    )
    nib.save(img.as_reoriented(transform), str(out))
    return out


def _lr_axis(affine) -> int:
    nib = pytest.importorskip("nibabel")

    codes = nib.orientations.aff2axcodes(affine)
    return next(i for i, c in enumerate(codes) if c in "LR")


@pytest.mark.parametrize("axcodes", [
    ("L", "A", "S"),   # the other handedness
    ("P", "I", "R"),   # every axis somewhere else
])
@pytest.mark.parametrize("engine_id", ["strip-atlas", "mindgrab"])
def test_the_strip_does_not_care_how_the_image_is_stored(
    tmp_path, engine_id, axcodes,
):
    if not run.available(engine_id):
        pytest.skip(f"{engine_id} is not available here")
    nib = pytest.importorskip("nibabel")
    import numpy as np

    root = _dataset(tmp_path)
    source = _reoriented(
        root / REL, axcodes, root / "sub-01" / "anat" / "sub-01_T1w.nii.gz",
    )

    out = deface_dataset(root, engine_id=engine_id)
    assert out.ok, out.failed

    got = nib.load(str(root / out.produced[0]))
    src = nib.load(str(source))
    assert got.shape == src.shape
    assert np.allclose(got.affine, src.affine)

    data = np.asanyarray(got.dataobj)
    kept = data > data.min()
    axis = _lr_axis(src.affine)
    dice = 2 * (kept & np.flip(kept, axis=axis)).sum() / (2 * max(kept.sum(), 1))
    assert dice > 0.85, (
        f"{engine_id} on {''.join(axcodes)} kept an asymmetric region "
        f"(Dice {dice:.3f}), so the mask did not follow the storage order"
    )

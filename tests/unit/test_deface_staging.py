"""Defacing during conversion, in the staging directory.

The reason this exists separately from the Editor path is a privacy property,
not a convenience: conversion stages each subject in
``<bids_root>/.tmp_bidsmgr/sub-XXX/`` and commits it atomically, so defacing
there means the identifiable image **never enters the dataset**. There is no
window in which a sync client, a backup or a colleague could have seen it.

That also means there is nothing to undo, which is why this path does not use
the operations log and these tests do not check for one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bidsmgr.deface import run, status
from bidsmgr.deface.engines import TEMPLATE
from bidsmgr.fixups.deface import _staged_candidates, deface_staged

pytestmark = pytest.mark.skipif(
    not run.available(),
    reason="needs the niimath wheel, which has no build for this Python",
)


def _staged(tmp_path: Path) -> Path:
    """A staged subject, laid out the way convert lays one out."""
    staging = tmp_path / ".tmp_bidsmgr" / "sub-01"
    for datatype in ("anat", "func", "pet"):
        (staging / datatype).mkdir(parents=True)
    shutil.copyfile(TEMPLATE, staging / "anat" / "sub-01_T1w.nii.gz")
    (staging / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"Manufacturer": "Siemens"})
    )
    shutil.copyfile(TEMPLATE, staging / "pet" / "sub-01_pet.nii.gz")
    return staging


def test_it_defaces_the_staged_anatomy_and_pet(tmp_path):
    staging = _staged(tmp_path)
    before = (staging / "anat" / "sub-01_T1w.nii.gz").read_bytes()

    assert deface_staged(staging) == 2
    assert (staging / "anat" / "sub-01_T1w.nii.gz").read_bytes() != before


def test_it_records_the_method_in_the_staged_sidecar(tmp_path):
    staging = _staged(tmp_path)
    deface_staged(staging)
    doc = json.loads((staging / "anat" / "sub-01_T1w.json").read_text())
    assert status.defaced_by_us(doc) is not None
    assert doc["Manufacturer"] == "Siemens"


def test_a_sidecar_that_did_not_exist_is_created(tmp_path):
    """The PET image here has none, and the record still has to land."""
    staging = _staged(tmp_path)
    deface_staged(staging)
    doc = json.loads((staging / "pet" / "sub-01_pet.json").read_text())
    assert status.defaced_by_us(doc) is not None


def test_it_leaves_functional_data_alone(tmp_path):
    staging = _staged(tmp_path)
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    bold = staging / "func" / "sub-01_task-x_bold.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 10), "float32"), np.eye(4)),
             str(bold))
    before = bold.read_bytes()

    deface_staged(staging)
    assert bold.read_bytes() == before


def test_a_time_series_in_anat_is_not_a_candidate(tmp_path):
    staging = _staged(tmp_path)
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    nib.save(
        nib.Nifti1Image(np.zeros((8, 8, 8, 6), "float32"), np.eye(4)),
        str(staging / "anat" / "sub-01_MEGRE.nii.gz"),
    )
    names = [p.name for p in _staged_candidates(staging)]
    assert "sub-01_MEGRE.nii.gz" not in names
    assert "sub-01_T1w.nii.gz" in names


def test_nothing_temporary_survives(tmp_path):
    staging = _staged(tmp_path)
    deface_staged(staging)
    leftovers = [
        p.name for p in staging.rglob("*")
        if p.is_file() and "bidsmgr-deface-" in p.name
    ]
    assert not leftovers, leftovers


def test_a_failure_on_one_image_does_not_lose_the_others(tmp_path, monkeypatch):
    """A subject whose face could not be removed is still a converted subject.

    Refusing to write it would cost the user the conversion as well as the
    defacing, which is the wrong trade to make on their behalf.
    """
    from bidsmgr.fixups import deface as mod

    staging = _staged(tmp_path)
    real = mod.deface_to_temp
    calls = {"n": 0}

    def flaky(image, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise mod.DefaceFailed("engine said no")
        return real(image, **kw)

    monkeypatch.setattr(mod, "deface_to_temp", flaky)
    assert deface_staged(staging) == 1


def test_it_says_so_and_does_nothing_when_the_engine_is_missing(
    tmp_path, monkeypatch, caplog,
):
    from bidsmgr.fixups import deface as mod

    staging = _staged(tmp_path)
    before = (staging / "anat" / "sub-01_T1w.nii.gz").read_bytes()
    monkeypatch.setattr(mod, "available", lambda: False)

    with caplog.at_level("WARNING"):
        assert deface_staged(staging) == 0
    assert "niimath" in caplog.text
    assert (staging / "anat" / "sub-01_T1w.nii.gz").read_bytes() == before


# ---------------------------------------------------------------------------
# Defacing must never be able to fail a conversion.
#
# The module says so in its docstring, and it was not true: only DefaceFailed
# was caught, and the engine lookup happened OUTSIDE the loop. A stale engine
# id in QSettings raised a KeyError inside the subject commit and the whole
# conversion produced nothing. These pin the contract rather than that one
# exception.


def test_an_unknown_engine_is_refused_quietly_not_raised(tmp_path, caplog):
    """The real failure. It cost a user an entire conversion."""
    staging = _staged(tmp_path)
    before = (staging / "anat" / "sub-01_T1w.nii.gz").read_bytes()

    assert deface_staged(staging, engine_id="1") == 0
    assert (staging / "anat" / "sub-01_T1w.nii.gz").read_bytes() == before
    assert any("unknown deface engine" in r.message for r in caplog.records)


def test_an_engine_that_throws_anything_does_not_stop_the_others(
    tmp_path, monkeypatch, caplog,
):
    """One bad image must not take the rest of the subject with it."""
    from bidsmgr.fixups import deface as mod

    staging = _staged(tmp_path)
    real = mod.deface_to_temp
    calls = {"n": 0}

    def _flaky(image, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("something no one predicted")
        return real(image, **kwargs)

    monkeypatch.setattr(mod, "deface_to_temp", _flaky)

    assert deface_staged(staging) == 1, "the second image was not attempted"
    assert any("no one predicted" in r.message for r in caplog.records)


def test_an_unreadable_image_is_skipped_not_fatal(tmp_path):
    staging = _staged(tmp_path)
    (staging / "anat" / "sub-01_truncated_T1w.nii.gz").write_bytes(b"not a nifti")

    # The good ones still go; the broken one is simply not a candidate.
    assert deface_staged(staging) == 2
    assert "sub-01_truncated_T1w.nii.gz" not in [
        p.name for p in _staged_candidates(staging)
    ]

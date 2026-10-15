"""The header inspector: an image's header beside its sidecar.

Each disagreement it reports is one a tool would otherwise resolve silently
in one direction or the other, so each is built here on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from bidsmgr.viz.bids import bids_context  # noqa: E402
from bidsmgr.viz.data.volume import open_volume  # noqa: E402
from bidsmgr.viz.inspect import (  # noqa: E402
    HEADER_RULES, inspect, obliquity_degrees, rows_for_rule, worst,
)


def _bold(root: Path, *, tr_header: float = 2.0, sidecar: dict | None = None,
          shape=(4, 4, 6, 5), affine=None, sform: int = 1, qform: int = 1,
          name="sub-01_task-x_bold") -> Path:
    func = root / "sub-01" / "func"
    func.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32),
                          np.eye(4) if affine is None else affine)
    img.header.set_xyzt_units("mm", "sec")
    zooms = list(img.header.get_zooms())
    if len(zooms) >= 4:
        zooms[3] = tr_header
    img.header.set_zooms(zooms)
    img.set_sform(img.affine, code=sform)
    img.set_qform(img.affine, code=qform)
    path = func / f"{name}.nii.gz"
    nib.save(img, str(path))
    if sidecar is not None:
        (func / f"{name}.json").write_text(json.dumps(sidecar))
    return path


def _rows(path: Path, root: Path):
    return inspect(open_volume(path), bids_context(path, root))


def _row(rows, field):
    return next(r for r in rows if r.field == field)


class TestAgreement:
    def test_a_consistent_run_has_nothing_to_say(self, tmp_path):
        path = _bold(tmp_path, sidecar={"RepetitionTime": 2.0,
                                        "SliceTiming": [0, 1, 0.33, 1.33, 0.66, 1.66]})
        rows = _rows(path, tmp_path)
        assert worst(rows) is None
        assert _row(rows, "Repetition time").sidecar == "2 s"
        assert _row(rows, "Slice timing").header == "6 slices along z"


class TestDisagreements:
    def test_the_repetition_time(self, tmp_path):
        path = _bold(tmp_path, tr_header=2.5, sidecar={"RepetitionTime": 2.0})
        row = _row(_rows(path, tmp_path), "Repetition time")
        assert row.status == "error"
        assert (row.header, row.sidecar) == ("2.5 s", "2 s")
        assert "REPETITION_TIME_MISMATCH" in row.rules

    def test_a_rounding_is_not_a_disagreement(self, tmp_path):
        path = _bold(tmp_path, tr_header=2.0004, sidecar={"RepetitionTime": 2.0})
        assert _row(_rows(path, tmp_path), "Repetition time").status == "ok"

    def test_more_slice_times_than_slices(self, tmp_path):
        path = _bold(tmp_path, sidecar={"RepetitionTime": 2.0,
                                        "SliceTiming": [0.0] * 8})
        row = _row(_rows(path, tmp_path), "Slice timing")
        assert row.status == "error"
        assert "8 slice times for 6 slices" in row.note
        assert row.rules == ("SLICETIMING_ELEMENTS",)

    def test_a_slice_timed_after_the_next_volume(self, tmp_path):
        path = _bold(tmp_path, sidecar={"RepetitionTime": 2.0,
                                        "SliceTiming": [0, 0.5, 1, 1.5, 2.0, 2.5]})
        row = _row(_rows(path, tmp_path), "Slice timing")
        assert row.status == "error"
        assert row.rules == ("SLICETIMING_VALUES_GREATER_THAN_REPETITION_TIME",)

    def test_b_values_for_volumes_that_are_not_there(self, tmp_path):
        dwi = tmp_path / "sub-01" / "dwi"
        dwi.mkdir(parents=True)
        path = dwi / "sub-01_dwi.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 5), np.float32), np.eye(4)), str(path))
        (dwi / "sub-01_dwi.bval").write_text("0 1000 1000 2000 2000 0\n")
        row = _row(_rows(path, tmp_path), "b-values")
        assert row.status == "error"
        assert "6 b-values for 5 volumes" in row.note
        assert "shells 0, 1000, 2000" in row.sidecar

    def test_no_orientation_at_all(self, tmp_path):
        path = _bold(tmp_path, sform=0, qform=0, shape=(4, 4, 4))
        row = _row(_rows(path, tmp_path), "Orientation")
        assert row.status == "error"
        assert "SFORM_AND_QFORM_IN_IMAGE_HEADER_ARE_ZERO" in row.rules

    def test_no_units(self, tmp_path):
        path = tmp_path / "x.nii.gz"
        img = nib.Nifti1Image(np.zeros((3, 3, 3), np.float32), np.eye(4))
        img.header.set_xyzt_units(0, 0)
        nib.save(img, str(path))
        row = _row(_rows(path, None), "Units")
        assert row.status == "warning"
        assert "millimetres are assumed" in row.note


class TestGeometry:
    def test_obliquity(self):
        a = np.eye(4)
        assert obliquity_degrees(a) == pytest.approx(0.0)
        t = np.radians(20.0)
        a[:3, :3] = [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        assert obliquity_degrees(a) == pytest.approx(20.0, abs=1e-6)

    def test_an_oblique_image_says_so(self, tmp_path):
        t = np.radians(15.0)
        a = np.eye(4)
        a[:3, :3] = [[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]
        path = _bold(tmp_path, affine=a, shape=(4, 4, 4))
        row = _row(_rows(path, tmp_path), "Obliquity")
        assert row.status == "info" and row.header == "15.00 degrees"

    def test_axis_codes_and_codes(self, tmp_path):
        a = np.diag([-2.0, 2.0, 2.0, 1.0])
        path = _bold(tmp_path, affine=a, shape=(4, 4, 4))
        row = _row(_rows(path, tmp_path), "Orientation")
        assert row.header.startswith("LAS; sform 1 (scanner)")


class TestRules:
    def test_every_rule_has_a_row_that_can_answer_it(self, tmp_path):
        """A finding offering "Show in the header" must land on something."""
        path = _bold(tmp_path, tr_header=2.5,
                     sidecar={"RepetitionTime": 2.0, "SliceTiming": [0.0] * 6})
        rows = _rows(path, tmp_path)
        answered = {rule for r in rows for rule in r.rules}
        for rule in ("REPETITION_TIME_MISMATCH", "SLICETIMING_ELEMENTS", "NIFTI_UNIT",
                     "NIFTI_DIMENSION", "SFORM_AND_QFORM_IN_IMAGE_HEADER_ARE_ZERO"):
            assert rule in HEADER_RULES
            assert rule in answered, rule
        assert rows_for_rule(rows, "REPETITION_TIME_MISMATCH")[0].field == "Repetition time"

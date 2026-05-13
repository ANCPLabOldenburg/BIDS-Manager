"""Unit tests for ``fixups/intended_for.py``.

Two scopes:

* ``rename_for_fmap_token`` analogue here is the ``_format_bids_uri`` /
  ``_is_bold`` / ``_fieldmap_group_key`` triplet — covered with focused
  tests.
* ``populate_intended_for`` end-to-end on a synthetic staging tree
  (no DICOMs / dcm2niix needed; we hand-write JSON sidecars).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.fixups.intended_for import (
    _fieldmap_group_key,
    _format_bids_uri,
    _is_bold,
    populate_intended_for,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestFormatBidsUri:
    def test_no_session(self) -> None:
        uri = _format_bids_uri("001", None, "func", "sub-001_task-rest_bold.nii.gz")
        assert uri == "bids::sub-001/func/sub-001_task-rest_bold.nii.gz"

    def test_with_session(self) -> None:
        uri = _format_bids_uri("001", "pre", "func", "sub-001_ses-pre_task-rest_bold.nii.gz")
        assert uri == "bids::sub-001/ses-pre/func/sub-001_ses-pre_task-rest_bold.nii.gz"


class TestIsBold:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("sub-001_task-rest_bold.nii.gz", True),
            ("sub-001_task-rest_bold.nii", True),
            ("sub-001_task-rest_run-1_bold.nii.gz", True),
            ("sub-001_task-rest_sbref.nii.gz", False),
            ("sub-001_task-rest_bold_aborted.nii.gz", False),
            ("sub-001_T1w.nii.gz", False),
            ("sub-001_dwi.nii.gz", False),
        ],
    )
    def test_recognises_bold(self, name: str, expected: bool) -> None:
        assert _is_bold(name) is expected


class TestFieldmapGroupKey:
    @pytest.mark.parametrize(
        "filename,expected_key",
        [
            ("sub-001_magnitude1.json", "sub-001"),
            ("sub-001_magnitude2.json", "sub-001"),
            ("sub-001_phasediff.json", "sub-001"),
            ("sub-001_phase1.json", "sub-001"),
            ("sub-001_phase2.json", "sub-001"),
            ("sub-001_acq-fm2_run-1_magnitude1.json", "sub-001_acq-fm2_run-1"),
            ("sub-001_epi.json", "sub-001"),
        ],
    )
    def test_groups_fmap_components_together(
        self, tmp_path: Path, filename: str, expected_key: str,
    ) -> None:
        assert _fieldmap_group_key(tmp_path / filename) == expected_key


# ---------------------------------------------------------------------------
# populate_intended_for — synthetic staging trees
# ---------------------------------------------------------------------------


def _write_pair(folder: Path, basename: str, *, acq_time: str = "120000") -> None:
    """Create ``<basename>.nii.gz`` + ``<basename>.json`` in ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{basename}.nii.gz").write_bytes(b"\x1f\x8b")
    sidecar = folder / f"{basename}.json"
    sidecar.write_text(json.dumps({"AcquisitionTime": acq_time}, indent=4))


def _read(json_path: Path) -> dict:
    return json.loads(json_path.read_text())


class TestPopulateIntendedForTimeBased:
    def test_two_fmaps_three_runs_correct_binding(self, tmp_path: Path) -> None:
        """fmap acquired at 12:00 binds to runs at 12:05 and 12:10;
        fmap at 12:15 binds to run at 12:20."""
        sub = tmp_path
        _write_pair(sub / "fmap", "sub-001_magnitude1", acq_time="120000")
        _write_pair(sub / "fmap", "sub-001_magnitude2", acq_time="120000")
        _write_pair(sub / "fmap", "sub-001_phasediff",  acq_time="120000")
        _write_pair(sub / "fmap", "sub-001_acq-2_magnitude1", acq_time="121500")
        _write_pair(sub / "fmap", "sub-001_acq-2_phasediff",  acq_time="121500")

        _write_pair(sub / "func", "sub-001_task-rest_run-1_bold", acq_time="120500")
        _write_pair(sub / "func", "sub-001_task-rest_run-2_bold", acq_time="121000")
        _write_pair(sub / "func", "sub-001_task-rest_run-3_bold", acq_time="122000")

        n = populate_intended_for(sub, subject="001")
        assert n == 5  # five fmap JSONs updated

        # First fmap group → runs 1 and 2.
        first = _read(sub / "fmap" / "sub-001_magnitude1.json")["IntendedFor"]
        assert first == [
            "bids::sub-001/func/sub-001_task-rest_run-1_bold.nii.gz",
            "bids::sub-001/func/sub-001_task-rest_run-2_bold.nii.gz",
        ]
        # All three members of the first group share the list.
        assert _read(sub / "fmap" / "sub-001_magnitude2.json")["IntendedFor"] == first
        assert _read(sub / "fmap" / "sub-001_phasediff.json")["IntendedFor"] == first

        # Second fmap group → run 3.
        second = _read(sub / "fmap" / "sub-001_acq-2_magnitude1.json")["IntendedFor"]
        assert second == [
            "bids::sub-001/func/sub-001_task-rest_run-3_bold.nii.gz",
        ]
        assert _read(sub / "fmap" / "sub-001_acq-2_phasediff.json")["IntendedFor"] == second


class TestPopulateIntendedForFallback:
    def test_missing_run_time_links_all_to_all(self, tmp_path: Path) -> None:
        sub = tmp_path
        _write_pair(sub / "fmap", "sub-001_magnitude1", acq_time="120000")
        _write_pair(sub / "fmap", "sub-001_phasediff",  acq_time="120000")

        # One run has no AcquisitionTime — triggers fallback.
        _write_pair(sub / "func", "sub-001_task-rest_run-1_bold", acq_time="120500")
        no_time = sub / "func" / "sub-001_task-rest_run-2_bold.json"
        (sub / "func" / "sub-001_task-rest_run-2_bold.nii.gz").write_bytes(b"\x1f\x8b")
        no_time.write_text("{}")

        n = populate_intended_for(sub, subject="001")
        assert n == 2

        intended = _read(sub / "fmap" / "sub-001_magnitude1.json")["IntendedFor"]
        # Both runs included regardless of timing.
        assert sorted(intended) == sorted([
            "bids::sub-001/func/sub-001_task-rest_run-1_bold.nii.gz",
            "bids::sub-001/func/sub-001_task-rest_run-2_bold.nii.gz",
        ])

    def test_missing_fmap_time_links_all_to_all(self, tmp_path: Path) -> None:
        sub = tmp_path
        # fmap has no AcquisitionTime.
        (sub / "fmap").mkdir()
        (sub / "fmap" / "sub-001_magnitude1.nii.gz").write_bytes(b"\x1f\x8b")
        (sub / "fmap" / "sub-001_magnitude1.json").write_text("{}")

        _write_pair(sub / "func", "sub-001_task-rest_run-1_bold", acq_time="120500")
        _write_pair(sub / "func", "sub-001_task-rest_run-2_bold", acq_time="121000")

        n = populate_intended_for(sub, subject="001")
        assert n == 1
        intended = _read(sub / "fmap" / "sub-001_magnitude1.json")["IntendedFor"]
        assert sorted(intended) == sorted([
            "bids::sub-001/func/sub-001_task-rest_run-1_bold.nii.gz",
            "bids::sub-001/func/sub-001_task-rest_run-2_bold.nii.gz",
        ])


class TestPopulateIntendedForSkipCases:
    def test_skips_when_no_func_dir(self, tmp_path: Path) -> None:
        _write_pair(tmp_path / "fmap", "sub-001_magnitude1", acq_time="120000")
        n = populate_intended_for(tmp_path, subject="001")
        assert n == 0
        assert "IntendedFor" not in _read(tmp_path / "fmap" / "sub-001_magnitude1.json")

    def test_skips_when_no_fmap_dir(self, tmp_path: Path) -> None:
        _write_pair(tmp_path / "func", "sub-001_task-rest_bold", acq_time="120500")
        n = populate_intended_for(tmp_path, subject="001")
        assert n == 0

    def test_skips_when_subject_dir_missing(self, tmp_path: Path) -> None:
        n = populate_intended_for(tmp_path / "no-such-dir", subject="001")
        assert n == 0

    def test_excludes_sbref_and_aborted_runs(self, tmp_path: Path) -> None:
        """SBRef and aborted runs should not appear in IntendedFor lists."""
        sub = tmp_path
        _write_pair(sub / "fmap", "sub-001_magnitude1", acq_time="120000")
        _write_pair(sub / "fmap", "sub-001_phasediff",  acq_time="120000")
        _write_pair(sub / "func", "sub-001_task-rest_bold", acq_time="120500")
        _write_pair(sub / "func", "sub-001_task-rest_sbref", acq_time="120100")
        _write_pair(sub / "func", "sub-001_task-rest_bold_aborted", acq_time="120200")

        populate_intended_for(sub, subject="001")
        intended = _read(sub / "fmap" / "sub-001_magnitude1.json")["IntendedFor"]
        # Only the real bold run survives.
        assert intended == [
            "bids::sub-001/func/sub-001_task-rest_bold.nii.gz",
        ]


class TestPopulateIntendedForSessionLayout:
    def test_session_subdir_isolation(self, tmp_path: Path) -> None:
        """fmaps in ses-pre only bind to funcs in ses-pre, never ses-post."""
        sub = tmp_path
        ses_pre = sub / "ses-pre"
        ses_post = sub / "ses-post"

        _write_pair(ses_pre / "fmap", "sub-001_ses-pre_magnitude1", acq_time="120000")
        _write_pair(ses_pre / "fmap", "sub-001_ses-pre_phasediff",  acq_time="120000")
        _write_pair(ses_pre / "func", "sub-001_ses-pre_task-rest_bold", acq_time="120500")

        _write_pair(ses_post / "fmap", "sub-001_ses-post_magnitude1", acq_time="130000")
        _write_pair(ses_post / "fmap", "sub-001_ses-post_phasediff",  acq_time="130000")
        _write_pair(ses_post / "func", "sub-001_ses-post_task-rest_bold", acq_time="130500")

        n = populate_intended_for(sub, subject="001")
        # 2 fmap JSONs per session × 2 sessions = 4.
        assert n == 4

        pre = _read(ses_pre / "fmap" / "sub-001_ses-pre_magnitude1.json")["IntendedFor"]
        assert pre == [
            "bids::sub-001/ses-pre/func/sub-001_ses-pre_task-rest_bold.nii.gz",
        ]

        post = _read(ses_post / "fmap" / "sub-001_ses-post_magnitude1.json")["IntendedFor"]
        assert post == [
            "bids::sub-001/ses-post/func/sub-001_ses-post_task-rest_bold.nii.gz",
        ]

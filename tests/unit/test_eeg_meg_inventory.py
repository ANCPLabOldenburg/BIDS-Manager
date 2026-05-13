"""Unit tests for ``bidsmgr.inventory.eeg_meg``.

Most tests work on synthetic trees and patch the lazy ``mne`` import so
the suite runs anywhere mne is missing or slow to import. The
candidate-path discovery and subject-identity heuristics are pure logic
and don't need a real mne.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from bidsmgr.inventory import eeg_meg as eeg_meg_mod
from bidsmgr.inventory.eeg_meg import (
    EEG_MEG_COLUMNS,
    candidate_paths,
    guess_subject_session_task,
    scan_eeg_meg,
)


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------


def test_eeg_meg_columns_exact_order() -> None:
    """The 12 EEG/MEG-specific columns are in the locked order.

    User-editable: ``task``, ``run``, ``line_freq``, ``montage``.
    These four condition how the BIDS basename is built and what
    metadata mne-bids writes.
    """
    assert EEG_MEG_COLUMNS == (
        "task", "run", "format", "source_file",
        "n_channels", "sfreq", "duration_sec", "n_times",
        "recording_time", "has_positions",
        "line_freq", "montage",
    )


# ---------------------------------------------------------------------------
# candidate_paths walking + collapsing
# ---------------------------------------------------------------------------


class TestCandidatePaths:
    def test_collects_recognised_extensions(self, tmp_path: Path) -> None:
        for name in ["a.edf", "b.bdf", "c.fif", "d.cnt", "e.set", "f.gdf"]:
            (tmp_path / name).write_bytes(b"x")
        # Noise: should not be collected.
        (tmp_path / "notes.txt").write_text("hi")
        (tmp_path / "log.csv").write_text("hi")

        out = candidate_paths(tmp_path)
        names = sorted(p.name for p in out)
        assert names == ["a.edf", "b.bdf", "c.fif", "d.cnt", "e.set", "f.gdf"]

    def test_collapses_brainvision_triplet(self, tmp_path: Path) -> None:
        """Triplets keep only the .vhdr."""
        for name in ["rec.vhdr", "rec.vmrk", "rec.eeg"]:
            (tmp_path / name).write_bytes(b"x")
        out = candidate_paths(tmp_path)
        assert len(out) == 1
        assert out[0].name == "rec.vhdr"

    def test_orphan_eeg_is_kept_when_no_vhdr(self, tmp_path: Path) -> None:
        (tmp_path / "stray.eeg").write_bytes(b"x")
        out = candidate_paths(tmp_path)
        assert any(p.name == "stray.eeg" for p in out)

    def test_ds_directory_is_a_single_candidate(self, tmp_path: Path) -> None:
        """``.ds`` (CTF MEG) directories collapse to one entry; we don't
        descend into them."""
        ds_dir = tmp_path / "rec.ds"
        ds_dir.mkdir()
        (ds_dir / "data.meg4").write_bytes(b"x")
        (ds_dir / "header.xml").write_text("")
        # A sibling EDF should still be picked up.
        (tmp_path / "other.edf").write_bytes(b"x")
        out = candidate_paths(tmp_path)
        assert ds_dir in out
        assert any(p.name == "other.edf" for p in out)
        # Nothing inside the .ds dir is independently collected.
        assert not any(p.name == "data.meg4" for p in out)

    def test_mff_directory_is_a_single_candidate(self, tmp_path: Path) -> None:
        mff_dir = tmp_path / "rec.mff"
        mff_dir.mkdir()
        (mff_dir / "info.xml").write_text("")
        out = candidate_paths(tmp_path)
        assert mff_dir in out

    def test_walks_subdirs(self, tmp_path: Path) -> None:
        sub = tmp_path / "S001"
        sub.mkdir()
        (sub / "S001R01.edf").write_bytes(b"x")
        (sub / "S001R02.edf").write_bytes(b"x")
        out = candidate_paths(tmp_path)
        assert len(out) == 2
        assert all(p.parent.name == "S001" for p in out)


# ---------------------------------------------------------------------------
# Subject / session / task heuristic
# ---------------------------------------------------------------------------


class TestGuessSubjectSessionTask:
    def test_bids_form_path(self, tmp_path: Path) -> None:
        path = tmp_path / "sub-007" / "ses-pre" / "eeg" / "sub-007_ses-pre_task-rest_eeg.edf"
        sub, ses, task, run = guess_subject_session_task(path, tmp_path)
        assert sub == "007"
        assert ses == "pre"
        assert task == "rest"
        assert run == ""

    def test_bids_task_with_run(self, tmp_path: Path) -> None:
        path = tmp_path / "sub-007_ses-pre_task-rest_run-2_eeg.edf"
        _, _, task, run = guess_subject_session_task(path, tmp_path)
        assert task == "rest"
        assert run == "2"

    def test_klingelbach_underscore_form(self, tmp_path: Path) -> None:
        """``task_driving_run_05_00.fif`` → task=driving, run=05."""
        path = tmp_path / "sub_us04rt22" / "220215" / "task_driving_run_05_00.fif"
        sub, _, task, run = guess_subject_session_task(path, tmp_path)
        assert sub == "us04rt22"
        assert task == "driving"
        assert run == "05"

    def test_klingelbach_task_only_no_run(self, tmp_path: Path) -> None:
        path = tmp_path / "task_rest.fif"
        _, _, task, run = guess_subject_session_task(path, tmp_path)
        assert task == "rest"
        assert run == ""

    def test_klingelbach_task_with_underscore_artifact(
        self, tmp_path: Path,
    ) -> None:
        """``task_emptypost_00.fif`` keeps task=emptypost, drops the _00."""
        path = tmp_path / "task_emptypost_00.fif"
        _, _, task, run = guess_subject_session_task(path, tmp_path)
        assert task == "emptypost"
        assert run == ""

    def test_physiobank_form_eegmmidb(self, tmp_path: Path) -> None:
        """``S001/S001R03.edf`` → subject=S001, run=03; task gets the
        sanitised filename stem as fallback (mne-bids requires a
        non-empty task; user can rename in the TSV before convert)."""
        path = tmp_path / "S001" / "S001R03.edf"
        sub, _, task, run = guess_subject_session_task(path, tmp_path)
        assert sub == "S001"
        assert task == "S001R03"
        assert run == "03"

    def test_falls_back_to_topmost_folder_when_no_sub_token(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "S001" / "S001R01.edf"
        sub, ses, task, run = guess_subject_session_task(path, tmp_path)
        assert sub == "S001"
        assert ses == ""

    def test_filename_stem_when_path_is_flat(self, tmp_path: Path) -> None:
        """Flat layout, no recognizable task/run pattern → fallback uses
        sanitised stem as task; no run extracted (``_1`` is too
        ambiguous without an explicit ``run_`` token to count)."""
        path = tmp_path / "Subject20_1.edf"
        sub, ses, task, run = guess_subject_session_task(path, tmp_path)
        assert sub == "Subject201"
        assert ses == ""
        assert task == "Subject201"
        assert run == ""

    def test_explicit_task_token_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "Klingelbach driving" / "sub_x" / "task-rest_run-01.fif"
        _, _, task, run = guess_subject_session_task(path, tmp_path)
        assert task == "rest"
        assert run == "01"


# ---------------------------------------------------------------------------
# scan_eeg_meg with patched probe (avoids a real mne dependency in tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StubProbe:
    """Stand-in for ``ProbeResult`` returned by a patched ``_probe``."""

    source: Path
    sfreq: float = 500.0
    n_channels: int = 32
    n_times: int = 50_000
    duration_sec: float = 100.0
    recording_time: str = "2026-01-01T12:00:00"
    datatype: str = "eeg"
    has_positions: bool = False
    fmt: str = "EDF"


def _patch_probe(monkeypatch, *, datatype: str = "eeg") -> None:
    """Force ``_probe`` to succeed for any path it's called on."""
    monkeypatch.setattr(
        eeg_meg_mod, "_HAS_MNE", True, raising=False,
    )

    def fake_probe(path: Path):
        return _StubProbe(source=path, datatype=datatype)

    monkeypatch.setattr(eeg_meg_mod, "_probe", fake_probe)


class TestScanEegMeg:
    def test_returns_empty_on_unscannable_tree(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _patch_probe(monkeypatch)
        # Tree has no recognised files.
        (tmp_path / "notes.txt").write_text("hi")
        df = scan_eeg_meg(tmp_path)
        assert df.empty
        # Empty frame still carries the EEG/MEG columns.
        for col in EEG_MEG_COLUMNS:
            assert col in df.columns

    def test_scans_flat_edf_layout(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _patch_probe(monkeypatch)
        for name in ["S00.edf", "S01.edf", "S02.edf"]:
            (tmp_path / name).write_bytes(b"x")
        df = scan_eeg_meg(tmp_path, dataset="study")
        assert len(df) == 3
        # Each file with a unique stem becomes a unique subject in flat layout.
        assert df["BIDS_name"].nunique() == 3
        # All rows carry the dataset slug.
        assert (df["dataset"] == "study").all()
        # Every row's modality and proposed_datatype are set.
        assert (df["modality"] == "eeg").all()
        assert (df["proposed_datatype"] == "eeg").all()

    def test_eeg_meg_rows_carry_bids_guess_suffix(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The inspector's ``suffix`` column reads ``bids_guess_suffix``.

        EEG/MEG rows must populate that column so the inspector shows
        ``eeg`` / ``meg`` instead of a blank cell.
        """
        _patch_probe(monkeypatch)
        (tmp_path / "S00.edf").write_bytes(b"x")
        df = scan_eeg_meg(tmp_path)
        assert (df["bids_guess_suffix"] == "eeg").all()
        assert (df["bids_guess_datatype"] == "eeg").all()
        assert (df["bids_guess_classifier"] == "eeg_meg_scanner").all()

    def test_scans_hierarchical_subject_layout(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _patch_probe(monkeypatch)
        # Two subjects, two recordings each.
        for sub in ["S001", "S002"]:
            sub_dir = tmp_path / sub
            sub_dir.mkdir()
            for rec in ["R01.edf", "R02.edf"]:
                (sub_dir / rec).write_bytes(b"x")
        df = scan_eeg_meg(tmp_path)
        # 4 rows, 2 subjects.
        assert len(df) == 4
        assert df["BIDS_name"].nunique() == 2
        assert set(df["subject"]) == {"S001", "S002"}

    def test_proposed_basename_is_schema_built(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _patch_probe(monkeypatch)
        # BIDS-form path → schema-built basename should be canonical.
        sub_dir = tmp_path / "sub-007" / "ses-pre" / "eeg"
        sub_dir.mkdir(parents=True)
        (sub_dir / "sub-007_ses-pre_task-rest_eeg.edf").write_bytes(b"x")
        df = scan_eeg_meg(tmp_path)
        assert len(df) == 1
        bn = df.iloc[0]["proposed_basename"]
        assert bn.startswith("sub-")
        assert "_eeg" in bn
        assert "_task-rest" in bn

    def test_meg_datatype_routes_to_meg_suffix(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _patch_probe(monkeypatch, datatype="meg")
        sub_dir = tmp_path / "sub_us04rt22" / "220215"
        sub_dir.mkdir(parents=True)
        (sub_dir / "task_rest.fif").write_bytes(b"x")
        df = scan_eeg_meg(tmp_path)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["modality"] == "meg"
        assert row["proposed_datatype"] == "meg"
        assert "_meg" in row["proposed_basename"]

    def test_brainvision_triplet_yields_one_row(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _patch_probe(monkeypatch)
        for name in ["rec.vhdr", "rec.vmrk", "rec.eeg"]:
            (tmp_path / name).write_bytes(b"x")
        df = scan_eeg_meg(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["source_file"].endswith(".vhdr")

    def test_includes_dataset_slug_in_every_row(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _patch_probe(monkeypatch)
        for name in ["a.edf", "b.edf"]:
            (tmp_path / name).write_bytes(b"x")
        df = scan_eeg_meg(tmp_path, dataset="my_study")
        assert (df["dataset"] == "my_study").all()

    def test_line_freq_and_montage_stamped_into_every_row(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When the user passes scan-time defaults, every EEG/MEG row
        carries them — auditable in the TSV, editable per-row before
        convert."""
        _patch_probe(monkeypatch)
        for name in ["a.edf", "b.edf", "c.edf"]:
            (tmp_path / name).write_bytes(b"x")
        df = scan_eeg_meg(
            tmp_path, dataset="study", line_freq=60.0, montage="standard_1005",
        )
        assert (df["line_freq"].astype(str) == "60.0").all()
        assert (df["montage"] == "standard_1005").all()

    def test_line_freq_blank_when_not_supplied(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        _patch_probe(monkeypatch)
        (tmp_path / "a.edf").write_bytes(b"x")
        df = scan_eeg_meg(tmp_path, dataset="study")
        assert df.iloc[0]["line_freq"] == ""
        assert df.iloc[0]["montage"] == ""

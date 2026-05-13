"""Unit tests for ``cli/scan.py``'s auto-dispatch logic.

The unified TSV concatenates rows from the MRI scanner and the EEG/MEG
scanner. These tests patch both probes so the suite doesn't need a real
DICOM tree or mne installation.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.cli import scan as scan_mod
from bidsmgr.cli.scan import (
    BIDS_GUESS_COLUMNS,
    DATASET_COLUMNS,
    PROBE_COLUMNS,
    _empty_unified_dataframe,
    _finalize_unified_dataframe,
    _unified_column_order,
    run_scan,
)
from bidsmgr.inventory import eeg_meg as eeg_meg_mod
from bidsmgr.inventory.eeg_meg import EEG_MEG_COLUMNS
from bidsmgr.inventory.mri_dicom import (
    EXTENDED_COLUMNS,
    TSV_COLUMNS,
)


# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------


class TestUnifiedColumnContract:
    def test_full_unified_column_layout(self) -> None:
        """Locked schema: TSV(22) + BIDS_GUESS(8) + ENTITIES(1) +
        DATASET(1) + PROBE(4) + EXTENDED(3) + EEG_MEG(12) = 51.

        The new ``entities`` JSON column is the canonical source of
        truth for the BIDS basename; ``proposed_basename`` and the
        ``task``/``run``/``session`` mirror cells are derived from it
        by ``bidsmgr-rebuild``.
        """
        df = _empty_unified_dataframe()
        cols = _unified_column_order(df)
        assert len(cols) == 22 + 8 + 1 + 1 + 4 + 3 + 12
        # ``dataset`` comes after BidsGuess + the new ``entities`` column.
        ds_idx = cols.index("dataset")
        assert ds_idx == len(TSV_COLUMNS) + len(BIDS_GUESS_COLUMNS) + 1
        # ``entities`` lives between BidsGuess and dataset.
        entities_idx = cols.index("entities")
        assert entities_idx == len(TSV_COLUMNS) + len(BIDS_GUESS_COLUMNS)
        # Order: MRI groups first, EEG/MEG last.
        assert cols[: len(TSV_COLUMNS)] == list(TSV_COLUMNS)
        assert cols[-len(EEG_MEG_COLUMNS):] == list(EEG_MEG_COLUMNS)

    def test_finalize_fills_missing_with_empty_string(self) -> None:
        """``concat`` introduces NaN; finalize replaces with ``""``."""
        df = pd.DataFrame([
            {"BIDS_name": "sub-001", "series_uid": "1.2.3"},
            {"BIDS_name": "sub-002", "source_file": "rec.edf"},
        ])
        out = _finalize_unified_dataframe(df)
        # No NaN cells anywhere.
        assert not out.isna().any().any()
        # Unified columns all present.
        for col in EEG_MEG_COLUMNS:
            assert col in out.columns
        for col in TSV_COLUMNS:
            assert col in out.columns

    def test_empty_unified_has_all_columns(self) -> None:
        df = _empty_unified_dataframe()
        # Every column from every group is present even with 0 rows.
        for group in (TSV_COLUMNS, BIDS_GUESS_COLUMNS, DATASET_COLUMNS,
                       PROBE_COLUMNS, EXTENDED_COLUMNS, EEG_MEG_COLUMNS):
            for col in group:
                assert col in df.columns


# ---------------------------------------------------------------------------
# run_scan with both branches mocked
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StubProbe:
    source: Path
    sfreq: float = 500.0
    n_channels: int = 32
    n_times: int = 50_000
    duration_sec: float = 100.0
    recording_time: str = ""
    datatype: str = "eeg"
    has_positions: bool = False
    fmt: str = "EDF"


def _patch_eeg_probe(monkeypatch, *, datatype: str = "eeg") -> None:
    """Force the EEG/MEG probe to succeed for any path."""
    monkeypatch.setattr(eeg_meg_mod, "_HAS_MNE", True, raising=False)
    monkeypatch.setattr(
        eeg_meg_mod, "_probe",
        lambda path: _StubProbe(source=path, datatype=datatype),
    )


class TestRunScanDispatch:
    def test_eeg_only_tree_writes_unified_tsv(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """EEG/MEG-only scan: no DICOMs, but unified TSV is written
        with EEG rows in the locked column order."""
        _patch_eeg_probe(monkeypatch)
        # Synthesise an EEG-only tree.
        for name in ["a.edf", "b.edf"]:
            (tmp_path / name).write_bytes(b"x")

        out_tsv = tmp_path / "inventory.tsv"
        df = run_scan(tmp_path, out_tsv, n_jobs=1)

        assert out_tsv.exists()
        assert len(df) == 2
        # Unified column count.
        assert len(df.columns) >= 47
        # No files_by_uid sidecar (EEG-only inventory).
        sidecar = out_tsv.with_suffix(out_tsv.suffix + ".files_by_uid.json.gz")
        assert not sidecar.exists()

    def test_empty_tree_writes_empty_unified_tsv(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Tree with neither DICOMs nor EEG/MEG: scan produces a
        valid (empty) unified TSV."""
        _patch_eeg_probe(monkeypatch)
        # Just a stray text file — neither MRI nor EEG/MEG.
        (tmp_path / "notes.txt").write_text("hi")

        out_tsv = tmp_path / "inventory.tsv"
        df = run_scan(tmp_path, out_tsv, n_jobs=1)

        assert out_tsv.exists()
        assert len(df) == 0
        # Empty TSV still has the unified header.
        header = out_tsv.read_text().splitlines()[0]
        cols = header.split("\t")
        assert "task" in cols  # EEG/MEG column
        assert "BIDS_name" in cols  # MRI column
        assert "dataset" in cols

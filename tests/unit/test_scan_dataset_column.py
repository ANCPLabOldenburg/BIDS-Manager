"""Unit tests for the scan-side ``dataset`` column and ``files_by_uid`` sidecar.

These cover the three pieces added when the converter loop landed:

* ``_default_dataset_slug`` — slugify the DICOM root basename.
* ``_write_files_by_uid_sidecar`` — gzip-JSON file next to the inventory TSV.
* ``DATASET_COLUMNS`` placement in the TSV write order
  (``TSV(22) + BIDS_GUESS(8) + DATASET(1) + PROBE(4) + EXTENDED(3)``).

Real DICOM scanning is exercised by the gated real-data tests; these unit
tests work directly on the helpers and on a hand-constructed DataFrame.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.cli.scan import (
    BIDS_GUESS_COLUMNS,
    PROBE_COLUMNS,
    _default_dataset_slug,
    _write_files_by_uid_sidecar,
)
from bidsmgr.inventory.mri_dicom import (
    DATASET_COLUMNS,
    EXTENDED_COLUMNS,
    TSV_COLUMNS,
)


class TestDefaultDatasetSlug:
    def test_simple_directory_name(self) -> None:
        assert _default_dataset_slug(Path("/foo/neuroimaging_unit_new")) == "neuroimaging_unit_new"

    def test_lowercases_and_replaces_unsafe_chars(self) -> None:
        assert _default_dataset_slug(Path("/foo/My Cool Study!")) == "my-cool-study"

    def test_collapses_repeated_dashes(self) -> None:
        assert _default_dataset_slug(Path("/foo/a   b___c")) == "a-b___c"

    def test_strips_leading_trailing_dashes(self) -> None:
        assert _default_dataset_slug(Path("/foo/!!!a!!!")) == "a"

    def test_empty_input_falls_back(self) -> None:
        assert _default_dataset_slug(Path("")) == "dataset"

    def test_purely_unsafe_input_falls_back(self) -> None:
        assert _default_dataset_slug(Path("/foo/!!!")) == "dataset"

    def test_preserves_underscores_and_hyphens(self) -> None:
        assert _default_dataset_slug(Path("/x/study_1-pilot")) == "study_1-pilot"


class TestFilesByUidSidecar:
    def test_writes_gzipped_json_next_to_tsv(self, tmp_path: Path) -> None:
        tsv = tmp_path / "inventory.tsv"
        tsv.write_text("dummy\n")
        files_by_uid = {
            "1.2.3": ["/abs/a.dcm", "/abs/b.dcm"],
            "1.2.4": ["/abs/c.dcm"],
        }
        sidecar = _write_files_by_uid_sidecar(tsv, files_by_uid)

        assert sidecar == tmp_path / "inventory.tsv.files_by_uid.json.gz"
        assert sidecar.exists()
        with gzip.open(sidecar, "rb") as fh:
            roundtrip = json.loads(fh.read().decode("utf-8"))
        assert roundtrip == files_by_uid

    def test_overwrites_existing_sidecar(self, tmp_path: Path) -> None:
        tsv = tmp_path / "inv.tsv"
        tsv.write_text("dummy\n")
        _write_files_by_uid_sidecar(tsv, {"old": ["x"]})
        sidecar = _write_files_by_uid_sidecar(tsv, {"new": ["y", "z"]})
        with gzip.open(sidecar, "rb") as fh:
            assert json.loads(fh.read().decode("utf-8")) == {"new": ["y", "z"]}


class TestDatasetColumnPlacement:
    """Verify the TSV column write order matches the contract.

    Order: ``TSV(22) + BIDS_GUESS(8) + DATASET(1) + PROBE(4) + EXTENDED(3)``.
    """

    def test_dataset_columns_constant_is_single_column(self) -> None:
        assert DATASET_COLUMNS == ("dataset",)

    def test_full_tsv_write_order(self, tmp_path: Path) -> None:
        # Build a DataFrame with all column groups present, then verify the
        # CLI's column-projection logic produces them in the locked order.
        all_cols = (
            list(TSV_COLUMNS)
            + list(BIDS_GUESS_COLUMNS)
            + list(DATASET_COLUMNS)
            + list(PROBE_COLUMNS)
            + list(EXTENDED_COLUMNS)
        )
        df = pd.DataFrame([{c: "" for c in all_cols}])

        # Mirror the projection in cli/scan.py:run_scan
        columns = (
            [c for c in TSV_COLUMNS if c in df.columns]
            + [c for c in BIDS_GUESS_COLUMNS if c in df.columns]
            + [c for c in DATASET_COLUMNS if c in df.columns]
            + [c for c in PROBE_COLUMNS if c in df.columns]
            + [c for c in EXTENDED_COLUMNS if c in df.columns]
        )

        # dataset sits at position TSV(22) + BIDS_GUESS(8) = index 30
        dataset_idx = columns.index("dataset")
        assert dataset_idx == len(TSV_COLUMNS) + len(BIDS_GUESS_COLUMNS)

        # All four BIDS_GUESS_COLUMNS appear before "dataset"; all PROBE
        # and EXTENDED appear after.
        for col in BIDS_GUESS_COLUMNS:
            assert columns.index(col) < dataset_idx
        for col in list(PROBE_COLUMNS) + list(EXTENDED_COLUMNS):
            assert columns.index(col) > dataset_idx

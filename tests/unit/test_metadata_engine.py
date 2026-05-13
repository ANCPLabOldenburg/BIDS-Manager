"""Unit tests for ``bidsmgr.metadata.engine``.

Synthetic BIDS trees (no real DICOMs / no dcm2niix) — we hand-create
``sub-XXX/<datatype>/`` folders with trivial NIfTI/JSON pairs and run
``run_metadata`` against them, asserting on the produced files and the
returned ``MetadataReport``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.metadata import DatasetMetadata, MetadataReport, run_metadata
from bidsmgr.metadata.engine import _infer_datatype_suffix


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_pair(
    folder: Path,
    basename: str,
    *,
    sidecar: dict | None = None,
) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    nii = folder / f"{basename}.nii.gz"
    json_path = folder / f"{basename}.json"
    nii.write_bytes(b"\x1f\x8b")
    json_path.write_text(json.dumps(sidecar or {}, indent=2))
    return nii, json_path


def _make_minimal_bids(tmp_path: Path) -> Path:
    """Create a one-subject BIDS root with anat + func + fmap data."""
    root = tmp_path / "study"
    root.mkdir()
    # Pre-existing dataset_description.json from converter (has GeneratedBy).
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "study",
        "BIDSVersion": "1.10.0",
        "DatasetType": "raw",
        "GeneratedBy": [{
            "Name": "bidsmgr",
            "Version": "0.0.1",
            "Description": "dcm2niix-direct backend",
        }],
    }, indent=2))

    sub = root / "sub-001"
    _write_pair(sub / "anat", "sub-001_T1w",
                sidecar={"AcquisitionTime": "120000"})
    _write_pair(sub / "func", "sub-001_task-rest_bold",
                sidecar={"AcquisitionTime": "120500", "RepetitionTime": 2.0})
    _write_pair(sub / "fmap", "sub-001_phasediff",
                sidecar={"AcquisitionTime": "120100",
                         "EchoTime1": 0.00492,
                         "EchoTime2": 0.00738})
    return root


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class TestRunMetadataMinimal:
    def test_writes_required_files(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        report = run_metadata(root)
        assert isinstance(report, MetadataReport)

        # Every required dataset-level file is present.
        assert (root / "dataset_description.json").exists()
        assert (root / "participants.tsv").exists()
        assert (root / "participants.json").exists()
        assert (root / "README").exists()
        assert (root / "CHANGES").exists()
        assert (root / "sub-001" / "sub-001_scans.tsv").exists()

    def test_raises_when_root_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run_metadata(tmp_path / "no-such-dir")


# ---------------------------------------------------------------------------
# dataset_description.json merge semantics
# ---------------------------------------------------------------------------


class TestDatasetDescription:
    def test_preserves_existing_generated_by_and_appends(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        # Pre-existing has one GeneratedBy entry from the converter.
        run_metadata(root)
        dd = json.loads((root / "dataset_description.json").read_text())
        names = [entry.get("Name") for entry in dd["GeneratedBy"]]
        # Converter entry preserved, metadata entry appended.
        assert names == ["bidsmgr", "bidsmgr"]
        descs = [entry.get("Description") for entry in dd["GeneratedBy"]]
        assert "dcm2niix-direct backend" in descs
        assert "metadata engine" in descs

    def test_caller_supplied_fields_win(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        meta = DatasetMetadata(
            name="My Cool Study",
            license="CC0",
            authors=["Alice", "Bob"],
            funding=["NIH-12345"],
            dataset_doi="10.1234/x",
        )
        run_metadata(root, dataset_meta=meta)
        dd = json.loads((root / "dataset_description.json").read_text())
        assert dd["Name"] == "My Cool Study"
        assert dd["License"] == "CC0"
        assert dd["Authors"] == ["Alice", "Bob"]
        assert dd["Funding"] == ["NIH-12345"]
        assert dd["DatasetDOI"] == "10.1234/x"

    def test_user_edits_not_clobbered(self, tmp_path: Path) -> None:
        """A field added by the user is preserved across a metadata rerun."""
        root = _make_minimal_bids(tmp_path)
        # Simulate user editing the file and adding a custom key.
        existing = json.loads((root / "dataset_description.json").read_text())
        existing["Acknowledgements"] = "we thank the participants"
        existing["CustomField"] = "preserve me"
        (root / "dataset_description.json").write_text(json.dumps(existing, indent=2))

        run_metadata(root)
        dd = json.loads((root / "dataset_description.json").read_text())
        assert dd["Acknowledgements"] == "we thank the participants"
        assert dd["CustomField"] == "preserve me"

    def test_dedupes_generated_by_on_rerun(self, tmp_path: Path) -> None:
        """Multiple metadata runs must not pile up duplicate
        ``GeneratedBy`` entries — dedupe by (Name, Version, Description).
        """
        root = _make_minimal_bids(tmp_path)
        run_metadata(root)
        run_metadata(root)
        run_metadata(root)
        dd = json.loads((root / "dataset_description.json").read_text())
        meta_entries = [
            e for e in dd["GeneratedBy"] if e.get("Description") == "metadata engine"
        ]
        # Three runs but only one metadata entry persists.
        assert len(meta_entries) == 1
        # Converter entry is also still there (one entry, not duplicated).
        conv_entries = [
            e for e in dd["GeneratedBy"]
            if e.get("Description") == "dcm2niix-direct backend"
        ]
        assert len(conv_entries) == 1


# ---------------------------------------------------------------------------
# participants.tsv + participants.json
# ---------------------------------------------------------------------------


class TestParticipants:
    def test_default_to_na_without_inventory(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        run_metadata(root)

        df = pd.read_csv(
            root / "participants.tsv", sep="\t", dtype=str, keep_default_na=False,
        )
        assert list(df["participant_id"]) == ["sub-001"]
        # Empty-everywhere columns get pruned when building from scratch:
        # only 'participant_id' should remain.
        assert df.columns.tolist() == ["participant_id"]

    def test_enriches_from_inventory(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        # Synthesise a tiny bidsmgr-shaped inventory TSV.
        inv = tmp_path / "inv.tsv"
        pd.DataFrame([{
            "BIDS_name": "sub-001",
            "GivenName": "Alice",
            "FamilyName": "Smith",
            "PatientID": "PID42",
            "PatientAge": "030Y",
            "PatientSex": "F",
        }]).to_csv(inv, sep="\t", index=False)

        run_metadata(root, inventory_tsv=inv)

        df = pd.read_csv(
            root / "participants.tsv", sep="\t", dtype=str, keep_default_na=False,
        )
        row = df.iloc[0].to_dict()
        assert row["participant_id"] == "sub-001"
        assert row["given_name"] == "Alice"
        assert row["family_name"] == "Smith"
        assert row["patient_id"] == "PID42"
        assert row["age"] == "030Y"
        assert row["sex"] == "F"

    def test_participants_json_describes_only_present_columns(
        self, tmp_path: Path,
    ) -> None:
        root = _make_minimal_bids(tmp_path)
        run_metadata(root)
        data = json.loads((root / "participants.json").read_text())
        assert set(data) == {"participant_id"}

    def test_merge_preserves_user_columns(self, tmp_path: Path) -> None:
        """If participants.tsv already has a custom column, it survives a rerun."""
        root = _make_minimal_bids(tmp_path)
        # Pre-write a participants.tsv with an extra column the user added.
        df_pre = pd.DataFrame([{
            "participant_id": "sub-001",
            "handedness": "right",
            "age": "n/a",
            "sex": "n/a",
        }])
        df_pre.to_csv(root / "participants.tsv", sep="\t", index=False)

        # Run with inventory enrichment.
        inv = tmp_path / "inv.tsv"
        pd.DataFrame([{
            "BIDS_name": "sub-001",
            "GivenName": "Alice",
            "FamilyName": "Smith",
            "PatientID": "PID42",
            "PatientAge": "030Y",
            "PatientSex": "F",
        }]).to_csv(inv, sep="\t", index=False)
        run_metadata(root, inventory_tsv=inv)

        df = pd.read_csv(
            root / "participants.tsv", sep="\t", dtype=str, keep_default_na=False,
        )
        row = df.iloc[0].to_dict()
        # User column preserved.
        assert row["handedness"] == "right"
        # Empty cells got filled by the inventory.
        assert row["age"] == "030Y"
        assert row["sex"] == "F"


# ---------------------------------------------------------------------------
# README + CHANGES
# ---------------------------------------------------------------------------


class TestReadmeChanges:
    def test_readme_seeded_first_run(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        run_metadata(root)
        body = (root / "README").read_text()
        assert body.startswith(f"# {root.name}")

    def test_readme_never_overwrites(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        (root / "README").write_text("hand-written content")
        run_metadata(root)
        assert (root / "README").read_text() == "hand-written content"

    def test_changes_seeded_first_run(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        run_metadata(root)
        assert (root / "CHANGES").read_text().startswith("1.0.0 ")

    def test_changes_never_overwrites(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        (root / "CHANGES").write_text("0.1.0 2025-01-01\n  - the truth\n")
        run_metadata(root)
        assert (root / "CHANGES").read_text() == "0.1.0 2025-01-01\n  - the truth\n"


# ---------------------------------------------------------------------------
# *_scans.tsv
# ---------------------------------------------------------------------------


class TestScansTsv:
    def test_one_per_subject_no_session(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        run_metadata(root)
        scans = root / "sub-001" / "sub-001_scans.tsv"
        assert scans.exists()
        df = pd.read_csv(scans, sep="\t", dtype=str, keep_default_na=False)
        assert "filename" in df.columns and "acq_time" in df.columns
        # Three NIfTIs in the fixture: anat, func, fmap.
        assert len(df) == 3
        # Every entry has an acq_time (no n/a since fixture wrote AcquisitionTime).
        assert all(t == "120000" or t == "120500" or t == "120100"
                   for t in df["acq_time"])

    def test_per_session_when_sessions_exist(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        sub = root / "sub-001"
        _write_pair(sub / "ses-pre" / "anat", "sub-001_ses-pre_T1w",
                    sidecar={"AcquisitionTime": "120000"})
        _write_pair(sub / "ses-post" / "anat", "sub-001_ses-post_T1w",
                    sidecar={"AcquisitionTime": "130000"})

        run_metadata(root)
        assert (sub / "ses-pre" / "sub-001_ses-pre_scans.tsv").exists()
        assert (sub / "ses-post" / "sub-001_ses-post_scans.tsv").exists()
        assert not (sub / "sub-001_scans.tsv").exists()

    def test_acq_time_falls_back_to_na(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w", sidecar={})
        run_metadata(root)
        df = pd.read_csv(
            root / "sub-001" / "sub-001_scans.tsv",
            sep="\t", dtype=str, keep_default_na=False,
        )
        assert df.iloc[0]["acq_time"] == "n/a"


# ---------------------------------------------------------------------------
# Sidecar fill + audit
# ---------------------------------------------------------------------------


class TestSidecarFillAndAudit:
    def test_fills_taskname_from_filename(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        run_metadata(root)
        sidecar = root / "sub-001" / "func" / "sub-001_task-rest_bold.json"
        data = json.loads(sidecar.read_text())
        assert data["TaskName"] == "rest"

    def test_does_not_overwrite_existing_taskname(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        sidecar = root / "sub-001" / "func" / "sub-001_task-rest_bold.json"
        existing = json.loads(sidecar.read_text())
        existing["TaskName"] = "user-set-this"
        sidecar.write_text(json.dumps(existing, indent=2))

        run_metadata(root)
        data = json.loads(sidecar.read_text())
        assert data["TaskName"] == "user-set-this"

    def test_missing_required_reported(self, tmp_path: Path) -> None:
        """func/bold without RepetitionTime is flagged as missing required."""
        root = tmp_path / "study"
        root.mkdir()
        # bold sidecar without RepetitionTime → schema flags it.
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"AcquisitionTime": "120500"})
        report = run_metadata(root)
        assert any(
            "RepetitionTime" in msg for msg in report.missing_required
        ), report.missing_required

    def test_repetition_time_satisfies_volume_timing_alternative(
        self, tmp_path: Path,
    ) -> None:
        """func/bold: RepetitionTime present → VolumeTiming not flagged.

        bidsschematools lists both as required, but the BIDS spec marks
        them mutually exclusive. Our engine encodes that exclusivity.
        """
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest", "RepetitionTime": 2.0})
        report = run_metadata(root)
        assert all("VolumeTiming" not in msg for msg in report.missing_required), \
            report.missing_required

    def test_strips_double_underscore_rule_suffix(self, tmp_path: Path) -> None:
        """fmap/phase2 requires `EchoTime` (the schema reports it as
        ``EchoTime__fmap`` for internal disambiguation). The audit
        message should use the canonical BIDS field name.
        """
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "fmap", "sub-001_phase2",
                    sidecar={})  # no EchoTime
        report = run_metadata(root)
        # The missing-field message should mention the clean name.
        assert any("'EchoTime'" in msg for msg in report.missing_required), \
            report.missing_required
        # And NOT the rule-decorated form.
        assert all("EchoTime__fmap" not in msg for msg in report.missing_required)

    def test_missing_recommended_reported_separately(self, tmp_path: Path) -> None:
        """dwi has no required fields per the schema, but does have
        recommended ones (PhaseEncodingDirection, TotalReadoutTime).
        Missing recommended fields land in ``missing_recommended``, not
        ``missing_required``.
        """
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "dwi", "sub-001_dwi", sidecar={})
        report = run_metadata(root)
        assert report.missing_required == []
        assert any(
            "PhaseEncodingDirection" in msg
            for msg in report.missing_recommended
        ), report.missing_recommended


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestJsonReport:
    def test_report_written_by_default(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        run_metadata(root)
        report_path = root / ".bidsmgr" / "metadata_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        # Required top-level keys.
        assert data["bidsmgr_version"]
        assert data["bids_root"]
        assert data["generated_at"]
        # Lists are JSON-encoded with relative-to-root paths preserved.
        assert isinstance(data["files_written"], list)
        assert isinstance(data["sidecar_fills"], list)
        assert isinstance(data["todo_fills"], list)
        assert isinstance(data["missing_required"], list)
        assert isinstance(data["missing_recommended"], list)

    def test_report_includes_self_in_files_written(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        run_metadata(root)
        data = json.loads((root / ".bidsmgr" / "metadata_report.json").read_text())
        # The report records itself.
        assert any(
            "metadata_report.json" in str(p) for p in data["files_written"]
        )

    def test_no_report_flag_suppresses(self, tmp_path: Path) -> None:
        root = _make_minimal_bids(tmp_path)
        run_metadata(root, write_report=False)
        assert not (root / ".bidsmgr" / "metadata_report.json").exists()


class TestFillTodos:
    def test_writes_todo_for_missing_required_sidecar_fields(
        self, tmp_path: Path,
    ) -> None:
        """fmap/phase2 needs EchoTime; without --fill-todos it's missing,
        with --fill-todos the field appears with value ``"TODO"``.
        """
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "fmap", "sub-001_phase2", sidecar={})

        report = run_metadata(root, fill_todos=True)
        sidecar = root / "sub-001" / "fmap" / "sub-001_phase2.json"
        data = json.loads(sidecar.read_text())
        assert data["EchoTime"] == "TODO"

        # And the report records the TODO insertion.
        todo = next(t for t in report.todo_fills if t.sidecar == sidecar)
        assert "EchoTime" in todo.fields

    def test_writes_todo_for_missing_recommended_sidecar_fields(
        self, tmp_path: Path,
    ) -> None:
        """dwi has no required, but does have recommended fields."""
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "dwi", "sub-001_dwi", sidecar={})

        run_metadata(root, fill_todos=True)
        data = json.loads(
            (root / "sub-001" / "dwi" / "sub-001_dwi.json").read_text()
        )
        assert data["PhaseEncodingDirection"] == "TODO"
        assert data["TotalReadoutTime"] == "TODO"

    def test_never_overwrites_existing_value(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "fmap", "sub-001_phase2",
                    sidecar={"EchoTime": 0.005})

        run_metadata(root, fill_todos=True)
        data = json.loads(
            (root / "sub-001" / "fmap" / "sub-001_phase2.json").read_text()
        )
        # Existing value preserved verbatim — not clobbered with "TODO".
        assert data["EchoTime"] == 0.005

    def test_off_by_default(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "fmap", "sub-001_phase2", sidecar={})

        run_metadata(root)  # no fill_todos
        data = json.loads(
            (root / "sub-001" / "fmap" / "sub-001_phase2.json").read_text()
        )
        assert "EchoTime" not in data

    def test_fills_dataset_description_recommended(
        self, tmp_path: Path,
    ) -> None:
        """dataset_description.json gets License/Authors/etc. as TODO."""
        root = _make_minimal_bids(tmp_path)
        report = run_metadata(root, fill_todos=True)
        dd = json.loads((root / "dataset_description.json").read_text())
        for field in ("License", "Authors", "Acknowledgements",
                      "HowToAcknowledge", "Funding", "EthicsApprovals",
                      "ReferencesAndLinks", "DatasetDOI"):
            assert dd[field] == "TODO", f"{field!r} not filled"

        # And dataset_description.json appears in the todo_fills list.
        assert any(
            t.sidecar.name == "dataset_description.json"
            for t in report.todo_fills
        )

    def test_dataset_description_existing_values_preserved(
        self, tmp_path: Path,
    ) -> None:
        """Caller-supplied License is preserved; other recommended fields
        get TODO."""
        root = _make_minimal_bids(tmp_path)
        meta = DatasetMetadata(name="study", license="CC0",
                               authors=["Alice"])
        run_metadata(root, fill_todos=True, dataset_meta=meta)
        dd = json.loads((root / "dataset_description.json").read_text())
        assert dd["License"] == "CC0"
        assert dd["Authors"] == ["Alice"]
        # Others got TODO.
        assert dd["Funding"] == "TODO"
        assert dd["EthicsApprovals"] == "TODO"

    def test_idempotent_rerun_does_not_pile_up_todos(
        self, tmp_path: Path,
    ) -> None:
        """A second run doesn't re-add TODO entries (the field exists)."""
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "fmap", "sub-001_phase2", sidecar={})

        run_metadata(root, fill_todos=True)
        report2 = run_metadata(root, fill_todos=True)

        # Second run reports zero new TODO insertions for that sidecar.
        sidecar = root / "sub-001" / "fmap" / "sub-001_phase2.json"
        for t in report2.todo_fills:
            if t.sidecar == sidecar:
                assert t.fields == [], "second run should not re-add TODOs"


class TestInferDatatypeSuffix:
    @pytest.mark.parametrize(
        "rel_path,expected",
        [
            ("sub-001/anat/sub-001_T1w.json", ("anat", "T1w")),
            ("sub-001/func/sub-001_task-rest_bold.json", ("func", "bold")),
            ("sub-001/ses-pre/anat/sub-001_ses-pre_T1w.json", ("anat", "T1w")),
            ("sub-001/fmap/sub-001_phasediff.json", ("fmap", "phasediff")),
            ("sub-001/dwi/sub-001_run-1_dwi.json", ("dwi", "dwi")),
            ("sub-001/eeg/sub-001_task-rest_eeg.json", ("eeg", "eeg")),
            # Path without a recognised datatype dir → empty datatype.
            ("sub-001/sub-001_T1w.json", ("", "T1w")),
        ],
    )
    def test_path_parsing(
        self, tmp_path: Path, rel_path: str, expected: tuple[str, str],
    ) -> None:
        root = tmp_path / "ds"
        root.mkdir()
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
        assert _infer_datatype_suffix(target, root) == expected

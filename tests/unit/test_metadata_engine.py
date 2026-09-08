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

    def test_metadata_preserves_existing_name_without_explicit_name(
        self, tmp_path: Path,
    ) -> None:
        # A Name set by `bidsmgr create` / hand-edit must survive a metadata run
        # that did not pass an explicit name (no clobber to the folder name).
        root = _make_minimal_bids(tmp_path)  # folder is named "study"
        p = root / "dataset_description.json"
        data = json.loads(p.read_text())
        data["Name"] = "Human Chosen Name"
        p.write_text(json.dumps(data))

        run_metadata(root)  # no dataset_meta, no name

        assert json.loads(p.read_text())["Name"] == "Human Chosen Name"

    def test_metadata_falls_back_to_folder_name_when_none_on_disk(
        self, tmp_path: Path,
    ) -> None:
        # Fresh dataset with no Name on disk + no explicit name => folder name.
        root = tmp_path / "my_dataset"
        (root / "sub-001" / "anat").mkdir(parents=True)
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w", sidecar={})
        run_metadata(root)
        dd = json.loads((root / "dataset_description.json").read_text())
        assert dd["Name"] == "my_dataset"

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
        # Normalised out of DICOM's "030Y" into the number BIDS asks for.
        assert row["age"] == "30"
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
        # Normalised out of DICOM's "030Y" into the number BIDS asks for.
        assert row["age"] == "30"
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
    def test_writes_todo_for_a_string_typed_required_field(
        self, tmp_path: Path,
    ) -> None:
        """pet needs TracerName, a string, so the marker fits."""
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "pet", "sub-001_pet", sidecar={})

        report = run_metadata(root, fill_todos=True)
        sidecar = root / "sub-001" / "pet" / "sub-001_pet.json"
        data = json.loads(sidecar.read_text())
        assert data["TracerName"] == "TODO"

        # And the report records the TODO insertion.
        todo = next(t for t in report.todo_fills if t.sidecar == sidecar)
        assert "TracerName" in todo.fields

    def test_no_placeholder_for_a_field_a_string_would_not_fit(
        self, tmp_path: Path,
    ) -> None:
        """fmap/phase2 needs EchoTime, which is a NUMBER.

        Writing ``"EchoTime": "TODO"`` swaps a missing field for an invalid
        one, and the validator rightly rejects the type. The field is still
        reported as missing; it simply gets no placeholder to mark it with.
        """
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "fmap", "sub-001_phase2", sidecar={})

        report = run_metadata(root, fill_todos=True)
        data = json.loads(
            (root / "sub-001" / "fmap" / "sub-001_phase2.json").read_text()
        )
        assert "EchoTime" not in data
        assert any("EchoTime" in m for m in report.missing_required)

    def test_writes_todo_for_missing_recommended_sidecar_fields(
        self, tmp_path: Path,
    ) -> None:
        """dwi has no required, but does have recommended fields.

        Three outcomes, one per reason a field can refuse the marker.
        InstitutionName is free text and takes it. TotalReadoutTime is a
        number. PhaseEncodingDirection is a string but a CONTROLLED one, and
        "TODO" is not among the six axis codes it admits. All three stay in the
        missing-field report either way.
        """
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "dwi", "sub-001_dwi", sidecar={})

        report = run_metadata(root, fill_todos=True)
        data = json.loads(
            (root / "sub-001" / "dwi" / "sub-001_dwi.json").read_text()
        )
        assert data["InstitutionName"] == "TODO"
        assert "TotalReadoutTime" not in data
        assert "PhaseEncodingDirection" not in data
        for field in ("TotalReadoutTime", "PhaseEncodingDirection"):
            assert any(field in m for m in report.missing_recommended), field

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
        """dataset_description.json gets License/Authors/etc. as TODO.

        The marker takes the field's own shape: a bare string where the schema
        wants a string, and a one-element list where it wants an array of them.
        ``["TODO"]`` is as greppable as ``"TODO"`` and, unlike it, valid.
        """
        root = _make_minimal_bids(tmp_path)
        report = run_metadata(root, fill_todos=True)
        dd = json.loads((root / "dataset_description.json").read_text())
        for field in ("License", "Acknowledgements",
                      "HowToAcknowledge", "DatasetDOI"):
            assert dd[field] == "TODO", f"{field!r} not filled"
        for field in ("Authors", "Funding", "EthicsApprovals",
                      "ReferencesAndLinks"):
            assert dd[field] == ["TODO"], f"{field!r} not filled as an array"

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
        # Others got TODO, in the shape their type calls for.
        assert dd["Funding"] == ["TODO"]
        assert dd["EthicsApprovals"] == ["TODO"]

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


class TestParticipantsExtraColumns:
    def test_spreadsheet_extra_columns_carried_with_codebook(self, tmp_path):
        """Columns beyond the known demographics flow into participants.tsv and
        are described in participants.json (rs-bidsify-style), using a sibling
        codebook when present."""
        root = _make_minimal_bids(tmp_path)
        participants = tmp_path / "subjects.csv"
        participants.write_text(
            "participant_id,age,sex,group,iq\nsub-001,30,female,patient,118\n",
            encoding="utf-8",
        )
        (tmp_path / "subjects.json").write_text(json.dumps({
            "group": {"Description": "Study group", "Levels": {"patient": "clinical"}},
            "iq": {"Description": "Full-scale IQ", "Units": "points"},
        }), encoding="utf-8")

        run_metadata(root, participants_file=participants)

        tsv = (root / "participants.tsv").read_text()
        header = tsv.splitlines()[0].split("\t")
        assert "group" in header and "iq" in header
        assert "patient" in tsv and "118" in tsv

        pj = json.loads((root / "participants.json").read_text())
        assert pj["group"]["Levels"]["patient"] == "clinical"
        assert pj["iq"]["Units"] == "points"
        # Known demographic columns keep their built-in descriptions.
        assert pj["sex"]["Levels"]["F"] == "female"

    def test_extra_column_without_codebook_gets_default_description(self, tmp_path):
        root = _make_minimal_bids(tmp_path)
        participants = tmp_path / "subjects.tsv"
        participants.write_text(
            "participant_id\tcohort\nsub-001\twave1\n", encoding="utf-8",
        )
        run_metadata(root, participants_file=participants)
        pj = json.loads((root / "participants.json").read_text())
        assert pj["cohort"] == {"Description": "cohort"}
        assert "cohort" in (root / "participants.tsv").read_text().splitlines()[0]


class TestDatasetDescriptionFromScaffold:
    """The dataset's own description, stated once and kept.

    Every field here was already writable through ``bidsmgr-metadata`` flags
    and already modelled. What did not exist was anywhere to KEEP the answers,
    so the metadata run that happens by itself after a conversion never had
    them and every dataset came out reporting NO_AUTHORS.
    """

    @staticmethod
    def _scaffold(tmp_path: Path, **fields) -> Path:
        from bidsmgr.recording_meta import (
            RecordingMetaSpec, dump_spec, scaffold_sidecar_path,
        )
        inv = tmp_path / "inv.tsv"
        inv.write_text("BIDS_name\nsub-001\n")
        spec = RecordingMetaSpec()
        for key, value in fields.items():
            setattr(spec.dataset_description, key, value)
        scaffold_sidecar_path(inv).write_text(dump_spec(spec))
        return inv

    def test_authors_reach_dataset_description(self, tmp_path: Path) -> None:
        inv = self._scaffold(
            tmp_path, authors=["Lopez, Karel"], license="CC0-1.0",
            funding=["DFG 12345"],
        )
        root = _make_minimal_bids(tmp_path)
        run_metadata(root, inventory_tsv=inv, write_report=False)
        dd = json.loads((root / "dataset_description.json").read_text())
        assert dd["Authors"] == ["Lopez, Karel"]
        assert dd["License"] == "CC0-1.0"
        assert dd["Funding"] == ["DFG 12345"]

    def test_this_is_what_clears_no_authors(self, tmp_path: Path) -> None:
        """The defect in one assertion: the validator warned on every dataset
        we produced, because nothing carried the answer this far."""
        from bidsmgr.editor.validator import validate

        root = _make_minimal_bids(tmp_path)
        inv = self._scaffold(tmp_path, authors=["Lopez, Karel"])
        run_metadata(root, inventory_tsv=inv, write_report=False)
        report = validate(root)
        found = [
            i.rule_id for i in report.dataset_issues if i.rule_id == "NO_AUTHORS"
        ] + [
            i.rule_id for f in report.files for i in f.issues
            if i.rule_id == "NO_AUTHORS"
        ]
        assert found == []

    def test_an_explicit_argument_beats_the_stored_one(self, tmp_path: Path) -> None:
        """Precedence: what this run was told, then what was stored, then what
        is already on disk. A --author on the command line must not be quietly
        replaced by last month's answer."""
        inv = self._scaffold(tmp_path, authors=["Stored"], license="CC0-1.0")
        root = _make_minimal_bids(tmp_path)
        run_metadata(
            root, inventory_tsv=inv, write_report=False,
            dataset_meta=DatasetMetadata(authors=["Explicit"]),
        )
        dd = json.loads((root / "dataset_description.json").read_text())
        assert dd["Authors"] == ["Explicit"]
        # and the fields the caller did not mention still come from the scaffold
        assert dd["License"] == "CC0-1.0"

    def test_a_scaffold_without_the_block_changes_nothing(self, tmp_path: Path) -> None:
        """Older scaffolds have no such block, and must load unchanged."""
        inv = self._scaffold(tmp_path)  # every field left blank
        root = _make_minimal_bids(tmp_path)
        run_metadata(root, inventory_tsv=inv, write_report=False)
        dd = json.loads((root / "dataset_description.json").read_text())
        assert "Authors" not in dd or dd["Authors"] == []


class TestAgeNormalisation:
    """DICOM writes an age as three digits and a unit; BIDS wants a number.

    The raw ``PatientAge`` string was copied into participants.tsv unchanged,
    so a 65-year-old appeared as ``065Y`` and the validator rejected the whole
    column as not numeric. It only showed up once the inventory was passed to
    the metadata step, which is what populates the age column at all.
    """

    @pytest.mark.parametrize("raw, expected", [
        ("065Y", "65"),      # the ordinary case
        ("065y", "65"),      # case does not matter
        ("018M", "1.5"),     # an infant, in months
        ("65", "65"),        # already a number
        ("65.5", "65.5"),
        ("", ""),
        ("n/a", ""),         # not a guess
        ("abc", ""),
        (None, ""),
    ])
    def test_normalize_age(self, raw, expected) -> None:
        from bidsmgr.metadata.demographics import normalize_age
        assert normalize_age(raw) == expected

    def test_participants_tsv_age_is_numeric(self, tmp_path: Path) -> None:
        """End to end: the column the validator reads must be a number."""
        import pandas as pd

        root = _make_minimal_bids(tmp_path)
        inv = tmp_path / "inv.tsv"
        pd.DataFrame([{
            "BIDS_name": "sub-001", "PatientAge": "065Y", "PatientSex": "M",
            "include": 1,
        }]).to_csv(inv, sep="\t", index=False)

        run_metadata(root, inventory_tsv=inv, write_report=False)
        table = pd.read_csv(root / "participants.tsv", sep="\t", dtype=str)
        age = table.loc[table["participant_id"] == "sub-001", "age"].iloc[0]
        assert age == "65"
        float(age)  # must parse as a number, which is what BIDS asks for

    def test_an_age_left_by_an_earlier_run_is_repaired(self, tmp_path: Path) -> None:
        """The merge prefers what is already on disk, so fixing only the fresh
        path would leave every existing dataset broken for ever. A rerun has to
        repair the file it finds. "n/a" is a real answer and survives."""
        import pandas as pd

        root = _make_minimal_bids(tmp_path)
        (root / "participants.tsv").write_text(
            "participant_id\tage\tsex\n"
            "sub-001\t065Y\tM\n"
            "sub-002\tn/a\tF\n"
        )
        run_metadata(root, write_report=False)
        # keep_default_na: the literal "n/a" IS the BIDS answer for unknown,
        # and pandas would otherwise read it back as a missing value.
        table = pd.read_csv(
            root / "participants.tsv", sep="\t", dtype=str, keep_default_na=False,
        )
        ages = dict(zip(table["participant_id"], table["age"]))
        assert ages["sub-001"] == "65"
        assert ages["sub-002"] == "n/a"

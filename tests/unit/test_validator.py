"""Unit tests for ``bidsmgr.editor.validator``.

Validation is delegated to the standalone ``bidsval`` engine; this module is a
thin adapter (``bidsmgr.editor._bidsval_adapter``) that maps bidsval's results
into the GUI's :class:`bidsmgr.editor.types` shapes and adds the two BIDS
Manager-native supplements bidsval does not produce: the sidecar form data
(``sidecar_fields``) and TODO-placeholder findings.

These tests assert on *behaviour* (severities, the report shape the GUI consumes,
the supplements) rather than on bidsval's exact rule codes, so they stay robust
across bidsval releases. Synthetic BIDS trees, no real DICOMs.

Note on attribution: bidsval (and the BIDS spec) attach a data file's metadata
findings to the data file itself (``*_bold.nii.gz``), not to its ``.json``
sidecar. The sidecar form still shows which fields are missing because the
adapter populates ``sidecar_fields`` on the ``.json`` verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import (
    FieldLevel,
    Severity,
    ValidationReport,
    rollup_severity,
    validate,
)
from bidsmgr.editor.bidsmgr_checks import infer_datatype_suffix, value_kind


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_pair(folder: Path, basename: str, *, sidecar: dict | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    # A tiny non-empty payload so the data file is not flagged EMPTY_FILE.
    (folder / f"{basename}.nii.gz").write_bytes(b"\x1f\x8b\x08\x00stub")
    (folder / f"{basename}.json").write_text(json.dumps(sidecar or {}, indent=2))


def _make_dd(root: Path, *, name: str | None = None, extra: dict | None = None) -> Path:
    body = {"Name": name or root.name, "BIDSVersion": "1.10.0"}
    if extra:
        body.update(extra)
    p = root / "dataset_description.json"
    p.write_text(json.dumps(body, indent=2))
    return p


def _make_minimal_bids(tmp_path: Path) -> Path:
    root = tmp_path / "study"
    root.mkdir()
    _make_dd(root)
    sub = root / "sub-001"
    _write_pair(sub / "anat", "sub-001_T1w")
    _write_pair(sub / "func", "sub-001_task-rest_bold",
                sidecar={"TaskName": "rest", "RepetitionTime": 2.0})
    (sub / "func" / "sub-001_task-rest_events.tsv").write_text("onset\tduration\n0\t1\n")
    return root


# ---------------------------------------------------------------------------
# Top-level shape (the contract the GUI binds to)
# ---------------------------------------------------------------------------


class TestValidateBasics:
    def test_returns_pydantic_report(self, tmp_path: Path) -> None:
        report = validate(_make_minimal_bids(tmp_path))
        assert isinstance(report, ValidationReport)
        assert report.bidsmgr_version
        assert report.bids_version          # filled from bidsval's schema
        assert report.generated_at
        assert report.bids_root == (tmp_path / "study").resolve()

    def test_counts_have_the_three_keys(self, tmp_path: Path) -> None:
        report = validate(_make_minimal_bids(tmp_path))
        assert set(report.counts) >= {"ok", "warn", "err"}
        assert all(isinstance(v, int) for v in report.counts.values())

    def test_files_use_relative_paths(self, tmp_path: Path) -> None:
        report = validate(_make_minimal_bids(tmp_path))
        assert report.files  # at least one file walked
        for f in report.files:
            assert not f.path.is_absolute()
            assert isinstance(f.severity, Severity)

    def test_raises_when_root_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate(tmp_path / "no-such-dir")


# ---------------------------------------------------------------------------
# Dataset-level checks
# ---------------------------------------------------------------------------


class TestDatasetLevel:
    def test_missing_dataset_description_is_error(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w")
        report = validate(root)
        assert report.severity is Severity.ERR
        assert any(
            "dataset_description" in i.message.lower()
            for i in report.dataset_issues
        ), [(i.rule_id, i.message) for i in report.dataset_issues]

    def test_no_subjects_emits_dataset_warning(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        report = validate(root)
        assert any(
            "sub-" in i.message or "subject" in i.message.lower()
            for i in report.dataset_issues
        ), [(i.rule_id, i.message) for i in report.dataset_issues]


# ---------------------------------------------------------------------------
# Per-file findings + sidecar_fields supplement
# ---------------------------------------------------------------------------


class TestSidecarBehaviour:
    def test_missing_required_field_flags_the_data_file(self, tmp_path: Path) -> None:
        """A bold without RepetitionTime is an error on the data file, with the
        offending field surfaced so the Editor's fix button can jump to it."""
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest"})  # no RepetitionTime
        (root / "sub-001" / "func" / "sub-001_task-rest_events.tsv").write_text(
            "onset\tduration\n0\t1\n"
        )
        report = validate(root)
        data_file = next(
            f for f in report.files if f.path.name.endswith("_bold.nii.gz")
        )
        assert data_file.severity is Severity.ERR
        assert any(i.field == "RepetitionTime" for i in data_file.issues), \
            [(i.rule_id, i.field) for i in data_file.issues]

    def test_sidecar_fields_populated_with_levels(self, tmp_path: Path) -> None:
        """The .json verdict carries the schema-aware form rows (present/missing
        flagged per level) even though the metadata findings sit on the data
        file."""
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest", "RepetitionTime": 2.0,
                             "MyCustom": 42})
        (root / "sub-001" / "func" / "sub-001_task-rest_events.tsv").write_text(
            "onset\tduration\n0\t1\n"
        )
        report = validate(root)
        bold_json = next(
            f for f in report.files if f.path.name.endswith("_bold.json")
        )
        names = {sf.name: sf for sf in bold_json.sidecar_fields}
        assert names, "expected schema-aware sidecar_fields on the .json verdict"
        assert names["TaskName"].level is FieldLevel.REQUIRED
        assert names["TaskName"].present is True
        assert names["TaskName"].value == "rest"
        # A custom key the schema doesn't know appears as OPTIONAL.
        assert names["MyCustom"].level is FieldLevel.OPTIONAL
        assert names["MyCustom"].value_kind == "number"
        # At least one schema field is missing (a recommended row, not present).
        assert any(not sf.present for sf in bold_json.sidecar_fields)

    def test_requirement_levels_cover_the_whole_dataset(self, tmp_path: Path) -> None:
        """Every kind of JSON gets its levels from the schema, not just the one
        in a datatype folder.

        Three cases, each of which used to come back with no schema rows at all:
        an INHERITED sidecar at the dataset root, one at the subject level, and
        ``dataset_description.json``. The inherited ones sit ABOVE the datatype
        directories, so nothing in their path says ``func``; the suffix settles
        it, since only ``func`` carries ``bold``.
        """
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest", "RepetitionTime": 2.0})
        (root / "sub-001" / "func" / "sub-001_task-rest_events.tsv").write_text(
            "onset\tduration\n0\t1\n"
        )
        (root / "task-rest_bold.json").write_text(json.dumps({"RepetitionTime": 2.0}))
        (root / "sub-001" / "sub-001_task-rest_bold.json").write_text(
            json.dumps({"TaskName": "rest"})
        )

        report = validate(root)
        by_name = {str(f.path): f for f in report.files}

        for rel in ("task-rest_bold.json", "sub-001/sub-001_task-rest_bold.json"):
            fields = by_name[rel].sidecar_fields
            assert fields, f"{rel} got no schema rows"
            levels = {sf.name: sf.level for sf in fields}
            assert levels["RepetitionTime"] is FieldLevel.REQUIRED, rel
            assert FieldLevel.RECOMMENDED in levels.values(), rel

        dd_levels = {
            sf.name: sf.level
            for sf in by_name["dataset_description.json"].sidecar_fields
        }
        # A rule with a second selector: Genetics is required only of a dataset
        # that ships genetic_info.json. This one does not, and the form reads
        # the actual tree, so the field is not offered at all. That is the same
        # answer the validator gives, which is the point.
        assert "Genetics" not in dd_levels
        assert dd_levels["Name"] is FieldLevel.REQUIRED
        assert dd_levels["License"] is FieldLevel.RECOMMENDED
        # The badge that was wrong: guessing a level from a finding's severity
        # made these required. They are optional.
        for field in ("Funding", "EthicsApprovals", "ReferencesAndLinks"):
            assert dd_levels[field] is FieldLevel.OPTIONAL, field
        # Authors is recommended only while the dataset has no CITATION.cff,
        # which this one does not. Pinned again against the validator below.
        assert dd_levels["Authors"] is FieldLevel.RECOMMENDED


    def test_a_conditional_dataset_rule_fires_only_when_its_file_exists(
        self, tmp_path: Path,
    ) -> None:
        """``Genetics`` is required by a rule with TWO selectors: the file must
        be dataset_description.json AND the dataset must ship genetic_info.json.

        Both halves matter. Ignoring the second would demand Genetics of every
        dataset in existence; ignoring the rule would let a genetics dataset
        omit it.
        """
        def build(root: Path, *, with_genetics: bool) -> None:
            root.mkdir()
            _make_dd(root)
            _write_pair(root / "sub-001" / "anat", "sub-001_T1w")
            if with_genetics:
                (root / "genetic_info.json").write_text(
                    json.dumps({"GeneticLevel": "Genotype"})
                )

        def genetics_findings(root: Path) -> list[str]:
            report = validate(root)
            return [
                i.rule_id
                for f in report.files for i in f.issues if i.field == "Genetics"
            ] + [
                i.rule_id
                for i in report.dataset_issues if i.field == "Genetics"
            ]

        plain = tmp_path / "plain"
        build(plain, with_genetics=False)
        assert genetics_findings(plain) == []

        genetic = tmp_path / "genetic"
        build(genetic, with_genetics=True)
        assert genetics_findings(genetic), "Genetics must be required once genetic_info.json exists"

        # And the FORM must agree with the validator, in both directions. A
        # form that says optional while the validator raises an error sends the
        # user looking for a field the tool has already decided is mandatory.
        def dd_level(root: Path):
            report = validate(root)
            dd = next(f for f in report.files if f.path.name == "dataset_description.json")
            return {sf.name: sf.level for sf in dd.sidecar_fields}.get("Genetics")

        assert dd_level(plain) is None, "not offered when no genetic_info.json"
        assert dd_level(genetic) is FieldLevel.REQUIRED, "required once it exists"


    @pytest.mark.parametrize("present", [False, True])
    def test_form_and_validator_agree_on_conditional_dataset_fields(
        self, tmp_path: Path, present: bool,
    ) -> None:
        """The form and the validator must not disagree about one dataset.

        Both of these fields hinge on whether some OTHER file exists, which
        only a real tree can answer:

        * ``Genetics`` is required once ``genetic_info.json`` is present.
        * ``Authors`` is recommended only while ``CITATION.cff`` is ABSENT.

        A form that shrugs while the validator raises an error sends the user
        hunting for a field the tool has already decided about, and a form that
        demands one the validator does not want sends them adding data BIDS
        never asked for.
        """
        root = tmp_path / ("with" if present else "without")
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w")
        if present:
            (root / "genetic_info.json").write_text(
                json.dumps({"GeneticLevel": "Genotype"})
            )
            (root / "CITATION.cff").write_text(
                "cff-version: 1.2.0\nmessage: cite\ntitle: t\nauthors:\n  - name: A\n"
            )

        report = validate(root)
        dd = next(f for f in report.files if f.path.name == "dataset_description.json")
        levels = {sf.name: sf.level for sf in dd.sidecar_fields}
        flagged = {
            i.field
            for f in report.files for i in f.issues if i.field
        } | {i.field for i in report.dataset_issues if i.field}

        if present:
            # genetic_info.json is there, so Genetics is a real requirement and
            # the validator says so. CITATION.cff is there, so Authors is not.
            assert levels["Genetics"] is FieldLevel.REQUIRED
            assert "Genetics" in flagged
            assert levels["Authors"] is FieldLevel.OPTIONAL
            assert "Authors" not in flagged
        else:
            assert "Genetics" not in levels
            assert "Genetics" not in flagged
            assert levels["Authors"] is FieldLevel.RECOMMENDED
            assert "Authors" in flagged


class TestTodoPlaceholders:
    def test_todo_in_sidecar_is_warning_on_the_json(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root)
        _write_pair(root / "sub-001" / "func", "sub-001_task-rest_bold",
                    sidecar={"TaskName": "rest", "RepetitionTime": 2.0,
                             "Instructions": "TODO"})
        (root / "sub-001" / "func" / "sub-001_task-rest_events.tsv").write_text(
            "onset\tduration\n0\t1\n"
        )
        report = validate(root)
        bold_json = next(
            f for f in report.files if f.path.name.endswith("_bold.json")
        )
        todos = [
            i for i in bold_json.issues if i.rule_id == "bidsmgr.todo_placeholder"
        ]
        assert len(todos) == 1
        assert todos[0].field == "Instructions"
        assert todos[0].severity is Severity.WARN
        assert bold_json.severity is Severity.WARN

    def test_todo_in_dataset_description(self, tmp_path: Path) -> None:
        root = tmp_path / "study"
        root.mkdir()
        _make_dd(root, extra={"License": "TODO"})
        _write_pair(root / "sub-001" / "anat", "sub-001_T1w")
        report = validate(root)
        dd = next(
            f for f in report.files if f.path.name == "dataset_description.json"
        )
        assert any(
            i.rule_id == "bidsmgr.todo_placeholder" and i.field == "License"
            for i in dd.issues
        )


# ---------------------------------------------------------------------------
# Deep checks (the repurposed ``strict`` flag -> read_headers)
# ---------------------------------------------------------------------------


class TestDeepChecks:
    def test_strict_runs_without_crashing(self, tmp_path: Path) -> None:
        report = validate(_make_minimal_bids(tmp_path), strict=True)
        assert isinstance(report, ValidationReport)
        assert set(report.counts) >= {"ok", "warn", "err"}

    def test_schema_version_is_threaded(self, tmp_path: Path) -> None:
        """An explicit schema selection is honoured (the report records the
        BIDS version validated against)."""
        report = validate(_make_minimal_bids(tmp_path), schema="1.11.0")
        assert report.bids_version == "1.11.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestRollupSeverity:
    def test_empty_is_ok(self) -> None:
        assert rollup_severity([]) is Severity.OK

    def test_one_warn(self) -> None:
        assert rollup_severity([Severity.OK, Severity.WARN]) is Severity.WARN

    def test_err_dominates(self) -> None:
        assert (
            rollup_severity([Severity.OK, Severity.WARN, Severity.ERR])
            is Severity.ERR
        )


class TestValueKind:
    @pytest.mark.parametrize("value,expected", [
        (None, "null"),
        (True, "bool"),
        (False, "bool"),
        (42, "number"),
        (3.14, "number"),
        ("rest", "string"),
        ("TODO", "todo"),
        ([1, 2], "array"),
        ({"a": 1}, "object"),
    ])
    def test_classification(self, value, expected) -> None:
        assert value_kind(value) == expected


class TestInferDatatypeSuffix:
    def test_anat_t1w(self, tmp_path: Path) -> None:
        root = tmp_path / "ds"
        fp = root / "sub-001" / "anat" / "sub-001_T1w.nii.gz"
        assert infer_datatype_suffix(fp, root) == ("anat", "T1w")

    def test_func_bold_json(self, tmp_path: Path) -> None:
        root = tmp_path / "ds"
        fp = root / "sub-001" / "func" / "sub-001_task-rest_bold.json"
        assert infer_datatype_suffix(fp, root) == ("func", "bold")

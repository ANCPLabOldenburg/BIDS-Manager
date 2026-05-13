"""Tests for the Editor's right pane — grouped validation panel (M6 Step 5).

The pane renders three stacked sections (Dataset / Folder / File). Each
has a title + a severity-coloured count chip + zero-or-more
:class:`ValMessage` rows. Empty sections show a muted "no issues"
hint so the layout doesn't jump on file selection.

The pane is QSS-driven for all colours (object names ``val-*``); a
theme swap runs :meth:`repaint_for_palette` which forces Qt's
unpolish/polish cycle so the cached per-widget styles are dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtWidgets import QLabel

from bidsmgr.editor.types import (
    FieldLevel,
    FileVerdict,
    Issue,
    Severity,
    SidecarField,
    ValidationReport,
)
from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.widgets.val_message import ValMessage
from bidsmgr.gui.widgets.validation_pane import (
    ValidationPane,
    _count_chip_object_name,
    _folder_key_for,
)


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bids_root(tmp_path: Path) -> Path:
    root = tmp_path / "Studyname"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-01_T1w.nii.gz").write_bytes(b"")
    (anat / "sub-01_ses-01_T1w.json").write_text("{}")
    (root / "dataset_description.json").write_text(
        '{"Name": "Test", "BIDSVersion": "1.10.0"}'
    )
    (root / "participants.tsv").write_text("participant_id\nsub-01\n")
    return root


def _make_issue(sev: Severity, rule: str, msg: str) -> Issue:
    return Issue(severity=sev, rule_id=rule, message=msg)


def _report_with_three_levels(bids_root: Path) -> ValidationReport:
    """Hand-crafted report with one finding in each scope."""
    rel = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    ).relative_to(bids_root)
    return ValidationReport(
        bids_root=bids_root,
        bids_version="1.10.0",
        severity=Severity.ERR,
        counts={"ok": 0, "warn": 1, "err": 1},
        files=[
            FileVerdict(
                path=rel,
                severity=Severity.ERR,
                datatype="anat",
                suffix="T1w",
                issues=[
                    _make_issue(Severity.ERR, "FILE_MISSING_FIELD",
                                "Missing required field <code>MagneticFieldStrength</code>."),
                ],
            ),
        ],
        folder_issues={
            "sub-01/ses-01/anat": [
                _make_issue(Severity.WARN, "FOLDER_HAS_TODOS",
                            "1 file contains TODO placeholders."),
            ],
        },
        dataset_issues=[
            _make_issue(Severity.ERR, "DATASET_DESCRIPTION_NAME_EMPTY",
                        "dataset_description.json has an empty Name field."),
        ],
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_count_chip_picks_worst_severity() -> None:
    assert _count_chip_object_name([]) == "val-count-ok"
    assert _count_chip_object_name(
        [_make_issue(Severity.WARN, "r", "m")]
    ) == "val-count-warn"
    assert _count_chip_object_name([
        _make_issue(Severity.WARN, "r", "m"),
        _make_issue(Severity.ERR, "r", "m"),
    ]) == "val-count-err"


def test_folder_key_relative_to_root(tmp_path: Path) -> None:
    root = tmp_path / "DS"
    folder = root / "sub-01" / "ses-01" / "anat"
    folder.mkdir(parents=True)
    file_ = folder / "x.json"
    file_.write_text("{}")
    assert _folder_key_for(root, file_) == "sub-01/ses-01/anat"


def test_folder_key_for_root_level_file_is_empty(tmp_path: Path) -> None:
    root = tmp_path / "DS"
    root.mkdir()
    file_ = root / "README"
    file_.write_text("x")
    assert _folder_key_for(root, file_) == ""


def test_folder_key_returns_none_for_missing_inputs(tmp_path: Path) -> None:
    assert _folder_key_for(None, tmp_path) is None
    assert _folder_key_for(tmp_path, None) is None


# ---------------------------------------------------------------------------
# ValidationPane: empty / report / file selection
# ---------------------------------------------------------------------------


def test_starts_with_pre_validation_hint(qapp) -> None:
    pane = ValidationPane()
    hints = [
        lbl for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() == "pane-hint"
    ]
    assert hints, "expected a pane-hint label before a report is set"
    assert any("Validate dataset" in lbl.text() for lbl in hints)
    assert pane.findChildren(ValMessage) == []


def test_set_report_builds_three_sections(qapp, bids_root: Path) -> None:
    pane = ValidationPane()
    report = _report_with_three_levels(bids_root)
    pane.set_report(report)

    # Three section frames are inserted, regardless of file selection.
    section_titles = [
        lbl.text() for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() == "val-section-title"
    ]
    assert len(section_titles) == 3
    assert section_titles[0] == "Dataset"
    assert section_titles[1].startswith("Folder")
    assert section_titles[2].startswith("File")


def test_dataset_section_renders_dataset_issues(qapp, bids_root: Path) -> None:
    pane = ValidationPane()
    pane.set_report(_report_with_three_levels(bids_root))

    msgs = pane.findChildren(ValMessage)
    # Only one ValMessage so far — the dataset-level finding. The
    # folder/file sections are empty since no file is selected.
    assert len(msgs) == 1


def test_file_selection_surfaces_folder_and_file_issues(
    qapp, bids_root: Path,
) -> None:
    pane = ValidationPane()
    pane.set_report(_report_with_three_levels(bids_root))

    target = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    pane.set_current_file(target, bids_root)

    msgs = pane.findChildren(ValMessage)
    # 1 dataset + 1 folder + 1 file = 3 messages now.
    assert len(msgs) == 3

    # The "Folder" section title carries the folder key.
    folder_titles = [
        lbl.text() for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() == "val-section-title"
        and lbl.text().startswith("Folder")
    ]
    assert folder_titles == ["Folder · sub-01/ses-01/anat"]
    # The "File" section title carries the filename.
    file_titles = [
        lbl.text() for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() == "val-section-title"
        and lbl.text().startswith("File")
    ]
    assert file_titles == ["File · sub-01_ses-01_T1w.json"]


def test_empty_section_shows_no_issues_hint(qapp, bids_root: Path) -> None:
    """Empty file-section shows the muted hint (rather than just
    nothing) so the layout doesn't jump."""
    pane = ValidationPane()
    report = ValidationReport(
        bids_root=bids_root,
        dataset_issues=[],
        folder_issues={},
        files=[],
    )
    pane.set_report(report)
    target = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    pane.set_current_file(target, bids_root)

    # Three pane-hint labels (one per empty section).
    hints = [
        lbl for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() == "pane-hint"
    ]
    assert len(hints) == 3
    assert any("dataset-level issues" in lbl.text() for lbl in hints)
    assert any("folder-level issues" in lbl.text() for lbl in hints)
    assert any("file-level issues" in lbl.text() for lbl in hints)


def test_count_chip_object_names_match_severity(
    qapp, bids_root: Path,
) -> None:
    pane = ValidationPane()
    pane.set_report(_report_with_three_levels(bids_root))
    target = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    pane.set_current_file(target, bids_root)

    chips = [
        lbl for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() in (
            "val-count-ok", "val-count-warn", "val-count-err",
        )
    ]
    assert len(chips) == 3
    object_names = [c.objectName() for c in chips]
    # Dataset section has one ERR → err chip.
    assert object_names[0] == "val-count-err"
    # Folder section has one WARN → warn chip.
    assert object_names[1] == "val-count-warn"
    # File section has one ERR → err chip.
    assert object_names[2] == "val-count-err"


def test_repaint_for_palette_forces_qss_re_evaluation(
    qapp, bids_root: Path, monkeypatch,
) -> None:
    """Same unpolish/polish dance as :class:`SidecarFormPane` — without
    it Qt's per-widget style cache holds stale colors for custom
    ``QFrame#val-msg-*`` widgets on theme toggle.
    """
    pane = ValidationPane()
    pane.set_report(_report_with_three_levels(bids_root))

    unpolished: list = []
    polished: list = []
    real_unpolish = pane.style().unpolish
    real_polish = pane.style().polish

    def _spy_unpolish(w):
        unpolished.append(w)
        return real_unpolish(w)

    def _spy_polish(w):
        polished.append(w)
        return real_polish(w)

    monkeypatch.setattr(pane.style(), "unpolish", _spy_unpolish)
    monkeypatch.setattr(pane.style(), "polish", _spy_polish)

    from bidsmgr.gui.theme_manager import LIGHT
    pane.repaint_for_palette(LIGHT)

    assert pane in unpolished and pane in polished
    # At least the dataset-issue ValMessage got the cycle too.
    msgs = pane.findChildren(ValMessage)
    assert msgs
    assert any(m in unpolished for m in msgs)


# ---------------------------------------------------------------------------
# EditorPanel integration
# ---------------------------------------------------------------------------


def test_editor_panel_validate_populates_validation_pane(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    """End-to-end: running validate fills the right pane via the
    in-memory ReportWorker → editor → ValidationPane chain."""
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)

    with qtbot.waitSignal(
        panel.log_message, timeout=5000,
        check_params_cb=lambda msg: "Validation done" in msg,
    ):
        panel.start_dataset_validation()
    qtbot.waitUntil(lambda: panel._report_worker is None, timeout=5000)

    # The pane was bound to the report.
    titles = [
        lbl.text() for lbl in panel._validation_pane._body.findChildren(QLabel)
        if lbl.objectName() == "val-section-title"
    ]
    assert len(titles) == 3
    assert titles[0] == "Dataset"


def test_editor_panel_tree_click_updates_validation_pane(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    """Clicking a file in the tree shifts the Folder/File sections."""
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    with qtbot.waitSignal(
        panel.log_message, timeout=5000,
        check_params_cb=lambda msg: "Validation done" in msg,
    ):
        panel.start_dataset_validation()
    qtbot.waitUntil(lambda: panel._report_worker is None, timeout=5000)

    target = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    panel._on_file_selected(target)

    # File-section title now carries the filename.
    file_titles = [
        lbl.text() for lbl in panel._validation_pane._body.findChildren(QLabel)
        if lbl.objectName() == "val-section-title"
        and lbl.text().startswith("File")
    ]
    assert file_titles == ["File · sub-01_ses-01_T1w.json"]


def test_val_message_renders_field_chip_when_present(
    qapp, bids_root: Path,
) -> None:
    """Issues that name a JSON ``field`` now surface it as a chip
    (object name ``val-field``) on the ValMessage header — matching
    the HTML report's ``.field`` span."""
    pane = ValidationPane()
    rel = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    ).relative_to(bids_root)
    report = ValidationReport(
        bids_root=bids_root,
        files=[
            FileVerdict(
                path=rel,
                severity=Severity.WARN,
                issues=[
                    Issue(
                        severity=Severity.WARN,
                        rule_id="bidsmgr.todo_placeholder",
                        message="field 'License' contains TODO placeholder",
                        field="License",
                    ),
                    Issue(
                        severity=Severity.WARN,
                        rule_id="bidsmgr.layout",
                        message="(generic layout warning)",
                        # No field — no chip rendered.
                    ),
                ],
            ),
        ],
    )
    pane.set_report(report)
    pane.set_current_file(bids_root / rel, bids_root)

    field_chips = [
        lbl for lbl in pane.findChildren(QLabel)
        if lbl.objectName() == "val-field"
    ]
    # Exactly one chip rendered (for the License issue); the
    # field-less issue stays chip-free.
    assert len(field_chips) == 1
    assert field_chips[0].text() == "License"


def test_schema_audit_section_appears_for_json_with_fields(
    qapp, bids_root: Path,
) -> None:
    """When the selected file has ``sidecar_fields``, the pane grows a
    fourth section that summarises the schema audit."""
    pane = ValidationPane()
    rel = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    ).relative_to(bids_root)
    report = ValidationReport(
        bids_root=bids_root,
        files=[
            FileVerdict(
                path=rel,
                severity=Severity.ERR,
                datatype="anat", suffix="T1w",
                sidecar_fields=[
                    SidecarField(
                        level=FieldLevel.REQUIRED,
                        name="MagneticFieldStrength",
                        value=None, present=False, value_kind="missing",
                    ),
                    SidecarField(
                        level=FieldLevel.REQUIRED,
                        name="Manufacturer",
                        value="Siemens", present=True, value_kind="string",
                    ),
                    SidecarField(
                        level=FieldLevel.RECOMMENDED,
                        name="RepetitionTime",
                        value=2.0, present=True, value_kind="number",
                    ),
                ],
            ),
        ],
    )
    pane.set_report(report)
    pane.set_current_file(bids_root / rel, bids_root)

    titles = [
        lbl.text() for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() == "val-section-title"
    ]
    assert "Schema audit" in titles
    # Required-missing rolls the chip up to err.
    chips = [
        lbl for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() in ("val-count-ok", "val-count-warn", "val-count-err")
    ]
    audit_chip = chips[-1]
    assert audit_chip.objectName() == "val-count-err"
    assert "missing required" in audit_chip.text()
    # Missing field name surfaces in the audit body.
    miss_labels = [
        lbl.text() for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() == "val-audit-missing"
    ]
    assert any("MagneticFieldStrength" in t for t in miss_labels)


def test_schema_audit_chip_is_ok_when_nothing_missing(
    qapp, bids_root: Path,
) -> None:
    pane = ValidationPane()
    rel = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    ).relative_to(bids_root)
    report = ValidationReport(
        bids_root=bids_root,
        files=[
            FileVerdict(
                path=rel,
                severity=Severity.OK,
                datatype="anat", suffix="T1w",
                sidecar_fields=[
                    SidecarField(
                        level=FieldLevel.REQUIRED,
                        name="Manufacturer",
                        value="Siemens", present=True, value_kind="string",
                    ),
                ],
            ),
        ],
    )
    pane.set_report(report)
    pane.set_current_file(bids_root / rel, bids_root)
    chips = [
        lbl for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() in ("val-count-ok", "val-count-warn", "val-count-err")
    ]
    audit_chip = chips[-1]
    assert audit_chip.objectName() == "val-count-ok"


def test_schema_audit_section_omitted_when_no_fields(
    qapp, bids_root: Path,
) -> None:
    """Files without ``sidecar_fields`` (NIfTI / TSV / non-validated)
    don't get a Schema audit section."""
    pane = ValidationPane()
    nii_rel = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
    ).relative_to(bids_root)
    report = ValidationReport(
        bids_root=bids_root,
        files=[
            FileVerdict(
                path=nii_rel,
                severity=Severity.OK,
                datatype="anat", suffix="T1w",
                # No sidecar_fields.
            ),
        ],
    )
    pane.set_report(report)
    pane.set_current_file(bids_root / nii_rel, bids_root)
    titles = [
        lbl.text() for lbl in pane._body.findChildren(QLabel)
        if lbl.objectName() == "val-section-title"
    ]
    assert "Schema audit" not in titles


def test_root_swap_clears_validation_pane(
    qapp, qtbot, isolated_settings, bids_root: Path, tmp_path: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    with qtbot.waitSignal(
        panel.log_message, timeout=5000,
        check_params_cb=lambda msg: "Validation done" in msg,
    ):
        panel.start_dataset_validation()
    qtbot.waitUntil(lambda: panel._report_worker is None, timeout=5000)

    other = tmp_path / "OtherDataset"
    other.mkdir()
    panel._set_root(other, persist=False)

    # Pane is back to the pre-validation hint state.
    assert panel._validation_pane.findChildren(ValMessage) == []

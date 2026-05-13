"""Tests for Editor interactivity: clickable status chips + fix-button jumps.

* Toolbar status chips open an :class:`EditorIssuesDialog` filtered by
  severity. Activating a card selects the matching file in the BIDS
  tree (which cascades to the sidecar + validation panes).
* The fix-button on a :class:`ValMessage` propagates through
  :class:`ValidationPane.fix_requested` to
  :class:`EditorPanel._on_fix_requested`, which lands on the right
  file and asks the sidecar pane to focus the field named in the
  issue.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtWidgets import QLineEdit, QPushButton

from bidsmgr.editor.types import (
    FieldLevel,
    FileVerdict,
    Issue,
    Severity,
    SidecarField,
    ValidationReport,
)
from bidsmgr.gui.editor_issues_dialog import EditorIssuesDialog
from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.widgets.bids_tree_pane import PATH_ROLE
from bidsmgr.gui.widgets.val_message import ValMessage


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bids_root(tmp_path: Path) -> Path:
    """A dataset shaped like the user's neuroimging_old example:
    one TODO-laden dataset_description.json + a clean T1w sidecar."""
    root = tmp_path / "DS"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-01_T1w.nii.gz").write_bytes(b"")
    (anat / "sub-01_ses-01_T1w.json").write_text(
        '{"Manufacturer": "Siemens", "RepetitionTime": 2.0}'
    )
    (root / "dataset_description.json").write_text(
        '{"Name": "Test", "BIDSVersion": "1.10.0", '
        '"License": "TODO", "Authors": "TODO"}'
    )
    return root


def _make_report(bids_root: Path) -> ValidationReport:
    dd_rel = (bids_root / "dataset_description.json").relative_to(bids_root)
    t1w_rel = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    ).relative_to(bids_root)
    return ValidationReport(
        bids_root=bids_root,
        bids_version="1.10.0",
        severity=Severity.WARN,
        counts={"ok": 1, "warn": 2, "err": 0},
        files=[
            FileVerdict(
                path=dd_rel,
                severity=Severity.WARN,
                issues=[
                    Issue(
                        severity=Severity.WARN,
                        rule_id="bidsmgr.todo_placeholder",
                        message="field 'License' contains TODO placeholder",
                        field="License",
                        fix_label="Set a real value",
                        fix_action="set_field",
                    ),
                    Issue(
                        severity=Severity.WARN,
                        rule_id="bidsmgr.todo_placeholder",
                        message="field 'Authors' contains TODO placeholder",
                        field="Authors",
                        fix_label="Set a real value",
                        fix_action="set_field",
                    ),
                ],
            ),
            FileVerdict(
                path=t1w_rel,
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


# ---------------------------------------------------------------------------
# EditorIssuesDialog
# ---------------------------------------------------------------------------


def test_issues_dialog_lists_files_for_given_severity(
    qapp, bids_root: Path,
) -> None:
    report = _make_report(bids_root)
    dlg = EditorIssuesDialog(report, "warn", bids_root)
    # The one warn file in the report has a card.
    cards = dlg.findChildren(QPushButton, "issue-card-title")
    assert len(cards) == 1
    assert "dataset_description.json" in cards[0].text()


def test_issues_dialog_card_activation_emits_absolute_path(
    qapp, qtbot, bids_root: Path,
) -> None:
    report = _make_report(bids_root)
    dlg = EditorIssuesDialog(report, "warn", bids_root)
    card_btn = dlg.findChildren(QPushButton, "issue-card-title")[0]

    with qtbot.waitSignal(dlg.file_selected, timeout=500) as sig:
        card_btn.click()

    emitted = sig.args[0]
    assert isinstance(emitted, Path)
    assert emitted.is_absolute()
    assert emitted.name == "dataset_description.json"


def test_issues_dialog_ok_chip_lists_ok_files(qapp, bids_root: Path) -> None:
    report = _make_report(bids_root)
    dlg = EditorIssuesDialog(report, "ok", bids_root)
    cards = dlg.findChildren(QPushButton, "issue-card-title")
    assert len(cards) == 1
    assert "sub-01_ses-01_T1w.json" in cards[0].text()


def test_issues_dialog_err_section_shows_empty_state(
    qapp, bids_root: Path,
) -> None:
    """No err files in this report → empty hint, no cards."""
    report = _make_report(bids_root)
    dlg = EditorIssuesDialog(report, "err", bids_root)
    cards = dlg.findChildren(QPushButton, "issue-card-title")
    assert cards == []


# ---------------------------------------------------------------------------
# Chip click → dialog
# ---------------------------------------------------------------------------


def test_warn_chip_opens_dialog_when_report_loaded(
    qapp, qtbot, isolated_settings, bids_root: Path, monkeypatch,
) -> None:
    """Clicking the warn chip pops the issues dialog filtered by warn."""
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    panel._report = _make_report(bids_root)
    # Make the chip "visible" enough that its click handler fires
    # — chips start hidden, but the panel un-hides them in
    # ``_update_chips`` once a report lands.
    panel._update_chips(panel._report)

    captured: dict = {}
    real_init = EditorIssuesDialog.__init__

    def _spy_init(self, report, severity, root, parent=None):
        captured["severity"] = severity
        return real_init(self, report, severity, root, parent=parent)

    monkeypatch.setattr(EditorIssuesDialog, "__init__", _spy_init)
    # Stop the dialog from popping up modally during the test.
    monkeypatch.setattr(EditorIssuesDialog, "show", lambda self: None)

    panel._chip_warn.clicked.emit()

    assert captured.get("severity") == "warn"


def test_chip_click_with_no_report_is_noop(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    """If there's no report yet, clicking a chip must not crash."""
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    assert panel._report is None
    # Just make sure the method doesn't raise.
    panel._open_issues_dialog("warn")


# ---------------------------------------------------------------------------
# select_file_in_tree (used by dialog flow + fix-button flow)
# ---------------------------------------------------------------------------


def test_select_file_in_tree_updates_tree_selection(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    target = bids_root / "dataset_description.json"

    panel.select_file_in_tree(target)
    qapp.processEvents()

    selected = panel._tree_pane._tree.selectedItems()
    assert selected
    assert selected[0].data(0, PATH_ROLE) == str(target)


# ---------------------------------------------------------------------------
# Fix-button → focus field
# ---------------------------------------------------------------------------


def test_val_message_fix_request_carries_field(
    qapp, qtbot,
) -> None:
    """Clicking the fix button on a ValMessage emits ``fix_requested``
    with the field name baked in at construction."""
    msg = ValMessage(
        severity="warn",
        rule="bidsmgr.todo_placeholder",
        body_html="field 'License' contains TODO placeholder",
        fix_label="Set a real value",
        field="License",
    )
    fix_btn = msg.findChild(QPushButton, "val-fix")
    assert fix_btn is not None

    with qtbot.waitSignal(msg.fix_requested, timeout=500) as sig:
        fix_btn.click()
    assert sig.args == ["License"]


def test_fix_request_focuses_field_in_sidecar_form(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    """End-to-end: clicking the fix button on a dataset_description.json
    TODO issue lands the user on that file and focuses the editor for
    the named field."""
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    panel._report = _make_report(bids_root)
    panel._update_chips(panel._report)
    # Make sure the sidecar pane is bound to dataset_description.json
    # — that's the file whose form has the License editor.
    dd_path = bids_root / "dataset_description.json"
    panel._on_file_selected(dd_path)
    qapp.processEvents()
    # Push the report into the validation pane so it builds its rows.
    panel._validation_pane.set_report(panel._report)
    panel._validation_pane.set_current_file(dd_path, bids_root)
    qapp.processEvents()

    # Find one of the fix buttons in the rendered ValMessages.
    fix_buttons = panel._validation_pane.findChildren(QPushButton, "val-fix")
    assert fix_buttons, "expected fix buttons for the TODO issues"
    fix_buttons[0].click()
    qapp.processEvents()

    # The License editor in the sidecar form is now focused.
    rows = panel._sidecar_form._rows
    license_row = next((r for r in rows if r.key == "License"), None)
    assert license_row is not None
    editor = license_row.editor()
    assert isinstance(editor, QLineEdit)
    # ``hasFocus`` is unreliable under offscreen Qt (no active window).
    # ``focusWidget`` returns the widget that would be focused once
    # the toplevel window is activated — sufficient for the assertion
    # we care about: ``setFocus`` was routed to the License editor.
    assert panel._sidecar_form.focusWidget() is editor
    # ``selectAll`` was called as part of focus_field — the editor
    # shows the existing value pre-selected for easy overwrite.
    assert editor.selectedText() == editor.text()

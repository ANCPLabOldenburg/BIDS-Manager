"""The findings must fit in a narrow pane, and say which is which.

Four separate complaints, one theme: a validation message took far more room
than its content, and the parts that mattered were the ones that got pushed
out.

* A finding with nothing but a sentence (the Converter's scanner notes) was
  rendered with the full structured layout: an empty rule slot, a "WHAT IS
  WRONG" caption over a single line, and the padding for both.
* A long BIDS basename or file path in an issue card set the dialog's width
  floor, so it opened wide and could not be narrowed, and what it did show was
  clipped mid-word with no ellipsis.
* Every group chip in the whole-dataset view was painted amber, errors
  included, because the code compared the severity against ``"error"`` and
  the value is ``"err"``.
* There was no way to list errors alone, or to put them above warnings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtWidgets import QLabel, QPushButton

from bidsmgr.editor.types import FileVerdict, Issue, Severity, ValidationReport
from bidsmgr.gui.issues_dialog import IssuesDialog, _RowCard
from bidsmgr.gui.models import InventoryTableModel
from bidsmgr.gui.widgets import Chip, ElidedPushButton
from bidsmgr.gui.widgets.val_message import ValMessage
from bidsmgr.gui.widgets.validation_pane import ValidationPane


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _captions(widget) -> list[str]:
    return [
        lbl.text() for lbl in widget.findChildren(QLabel)
        if lbl.objectName() == "val-caption"
    ]


def _inventory_row(**overrides) -> dict:
    base = {
        "BIDS_name": "sub-001",
        "session": "ses-pre",
        "include": 1,
        "modality": "mri",
        "proposed_datatype": "func",
        "proposed_basename": (
            "sub-001_ses-pre_task-restingstate_acq-highres_run-01_bold"
        ),
        "bids_guess_classifier": "dcm2niix_bidsguess",
        "bids_guess_datatype": "func",
        "bids_guess_suffix": "bold",
        "bids_guess_confidence": "0.97",
        "bids_guess_skip": False,
        "proposed_issues": "",
        "entities": json.dumps(
            {"subject": "001", "session": "pre", "task": "restingstate"},
            sort_keys=True,
        ),
        "task": "restingstate",
        "run": "",
        "source_file": "",
        "series_uid": "1.1",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ValMessage density
# ---------------------------------------------------------------------------


def test_a_bare_sentence_gets_no_captions(qapp) -> None:
    """Nothing to tell apart means nothing to label."""
    msg = ValMessage("warn", "", "rerouted to fmap/epi: smaller than peer")
    assert _captions(msg) == []


def test_a_bare_sentence_still_shows_its_text(qapp) -> None:
    msg = ValMessage("warn", "", "rerouted to fmap/epi: smaller than peer")
    bodies = [
        lbl.text() for lbl in msg.findChildren(QLabel)
        if lbl.objectName() == "val-body"
    ]
    assert bodies == ["rerouted to fmap/epi: smaller than peer"]


def test_a_bare_sentence_keeps_its_fix_button(qapp) -> None:
    """Compact is not the same as stripped."""
    msg = ValMessage("warn", "", "something to repair", fix_label="Fix")
    buttons = [
        b for b in msg.findChildren(QPushButton)
        if b.objectName() == "val-fix"
    ]
    assert len(buttons) == 1
    captured: list[str] = []
    msg.fix_requested.connect(captured.append)
    buttons[0].click()
    assert captured == [""]


def test_a_structured_finding_keeps_its_captions(qapp) -> None:
    """The compact path must not swallow the labelled one."""
    msg = ValMessage(
        "err",
        "SIDECAR_KEY_REQUIRED",
        "missing required field 'RepetitionTime'",
        field="RepetitionTime",
        schema_rule="rules.sidecars.func.MRIFuncRepetitionTime",
    )
    captions = _captions(msg)
    assert "Field" in captions
    assert "What is wrong" in captions
    assert "Defined in" in captions


def test_a_bare_sentence_offers_no_accept_menu(qapp) -> None:
    """An acceptance is filed against a rule id, and there is none here."""
    from PyQt6.QtCore import Qt

    plain = ValMessage("warn", "", "a note with no rule behind it")
    assert plain.contextMenuPolicy() != Qt.ContextMenuPolicy.CustomContextMenu

    ruled = ValMessage("warn", "SOME_RULE", "a note with a rule behind it")
    assert ruled.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu


def test_the_compact_form_is_shorter(qapp) -> None:
    """The whole point, measured rather than asserted by eye."""
    text = "rerouted to fmap/epi: smaller than its DWI peer"
    plain = ValMessage("warn", "", text)
    structured = ValMessage("warn", "SOME_RULE", text, field="Something")
    plain.resize(320, plain.sizeHint().height())
    structured.resize(320, structured.sizeHint().height())
    assert plain.sizeHint().height() < structured.sizeHint().height()


# ---------------------------------------------------------------------------
# ElidedPushButton
# ---------------------------------------------------------------------------


def test_the_elided_button_keeps_the_whole_string(qapp) -> None:
    """Only the pixels shorten. ``text()`` is still the identifier."""
    full = "sub-001_ses-pre_task-restingstate_acq-highres_run-01_bold"
    btn = ElidedPushButton(full)
    btn.resize(60, 24)
    btn.grab()  # force a paint, which is where the elision happens
    assert btn.text() == full


def test_the_elided_button_claims_no_width(qapp) -> None:
    btn = ElidedPushButton("a very long label indeed, going on and on")
    assert btn.minimumSizeHint().width() == 0


def test_the_elided_button_recovers_the_text_in_its_tooltip(qapp) -> None:
    full = "sub-001_ses-pre_task-restingstate_bold"
    btn = ElidedPushButton(full)
    assert btn.toolTip() == full
    btn.setText("something else entirely")
    assert btn.toolTip() == "something else entirely"


# ---------------------------------------------------------------------------
# IssuesDialog (the Converter's chip pop-ups)
# ---------------------------------------------------------------------------


def _warn_dialog(qtbot) -> IssuesDialog:
    df = pd.DataFrame([
        _inventory_row(
            proposed_issues=(
                "rerouted to fmap/epi: smaller than its DWI peer | "
                "fmap multi-output: more than one file for this series"
            ),
            series_uid="2.2",
            BIDS_name="sub-002",
            entities=json.dumps(
                {"subject": "002", "session": "pre", "task": "restingstate"},
                sort_keys=True,
            ),
        ),
    ])
    dlg = IssuesDialog(InventoryTableModel(df), severity="warn")
    qtbot.addWidget(dlg)
    return dlg


def test_a_long_basename_does_not_set_the_dialog_width(qtbot) -> None:
    """This is why the pop-up used to open wider than the screen allowed."""
    dlg = _warn_dialog(qtbot)
    # The dialog's own floor, not the layout's: setMinimumWidth is deliberate
    # and small, and nothing inside may exceed it.
    assert dlg.layout().minimumSize().width() <= dlg.minimumWidth()


def test_the_row_title_is_elided_not_clipped(qtbot) -> None:
    dlg = _warn_dialog(qtbot)
    cards = dlg.findChildren(_RowCard)
    assert len(cards) == 1
    assert isinstance(cards[0]._title_btn, ElidedPushButton)
    # Still readable by anything that asks, tests included.
    assert "sub-002" in cards[0]._title_btn.text()


def test_the_scanner_notes_render_compactly(qtbot) -> None:
    """A scanner note carries no rule id, so it takes the compact path."""
    dlg = _warn_dialog(qtbot)
    messages = dlg.findChildren(ValMessage)
    assert len(messages) == 2
    for msg in messages:
        assert _captions(msg) == []


def test_the_editor_dialog_elides_its_paths_too(qtbot, tmp_path: Path) -> None:
    from bidsmgr.gui.editor_issues_dialog import EditorIssuesDialog, _FileCard

    root = tmp_path / "DS"
    (root / "sub-01" / "ses-01" / "func").mkdir(parents=True)
    report = ValidationReport(
        bids_root=root,
        bids_version="1.10.0",
        severity=Severity.ERR,
        counts={"ok": 0, "warn": 0, "err": 1},
        files=[
            FileVerdict(
                path=Path(
                    "sub-01/ses-01/func/"
                    "sub-01_ses-01_task-restingstate_run-01_bold.json"
                ),
                severity=Severity.ERR,
                datatype="func",
                suffix="bold",
                issues=[Issue(
                    severity=Severity.ERR,
                    rule_id="SIDECAR_KEY_REQUIRED",
                    message="missing required field 'RepetitionTime'",
                    field="RepetitionTime",
                )],
            ),
        ],
    )
    dlg = EditorIssuesDialog(report, "err", root)
    qtbot.addWidget(dlg)
    cards = dlg.findChildren(_FileCard)
    assert len(cards) == 1
    assert isinstance(cards[0]._title_btn, ElidedPushButton)
    assert dlg.layout().minimumSize().width() <= dlg.minimumWidth()


# ---------------------------------------------------------------------------
# Validation pane: severity filter, ordering, and the chip colour
# ---------------------------------------------------------------------------


def _mixed_report(root: Path) -> ValidationReport:
    rel = Path("sub-01/ses-01/anat/sub-01_ses-01_T1w.json")
    return ValidationReport(
        bids_root=root,
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
                    # Warning FIRST in source order, so a pane that does not
                    # sort shows it above the error.
                    Issue(
                        severity=Severity.WARN,
                        rule_id="SIDECAR_KEY_RECOMMENDED",
                        message="recommended field 'InstitutionName' missing",
                        field="InstitutionName",
                    ),
                    Issue(
                        severity=Severity.ERR,
                        rule_id="SIDECAR_KEY_REQUIRED",
                        message="missing required field 'RepetitionTime'",
                        field="RepetitionTime",
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def mixed_pane(qtbot, tmp_path: Path) -> ValidationPane:
    root = tmp_path / "DS"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    target = anat / "sub-01_ses-01_T1w.json"
    target.write_text("{}")
    pane = ValidationPane()
    qtbot.addWidget(pane)
    pane.set_report(_mixed_report(root))
    pane.set_current_file(target, root)
    return pane


def _severities(pane: ValidationPane) -> list[str]:
    return [
        msg.objectName().replace("val-msg-", "")
        for msg in pane.findChildren(ValMessage)
    ]


def test_worst_first_puts_the_error_above_the_warning(mixed_pane) -> None:
    assert mixed_pane._sev_sort.isChecked()
    order = _severities(mixed_pane)
    assert order.index("err") < order.index("warn")


def test_turning_the_sort_off_restores_the_validator_order(
    mixed_pane,
) -> None:
    mixed_pane._sev_sort.setChecked(False)
    order = _severities(mixed_pane)
    assert order.index("warn") < order.index("err")


def test_errors_only_hides_the_warning(mixed_pane) -> None:
    idx = mixed_pane._sev_filter.findData("err")
    assert idx >= 0
    mixed_pane._sev_filter.setCurrentIndex(idx)
    assert "warn" not in _severities(mixed_pane)
    assert "err" in _severities(mixed_pane)


def test_warnings_only_hides_the_error(mixed_pane) -> None:
    idx = mixed_pane._sev_filter.findData("warn")
    mixed_pane._sev_filter.setCurrentIndex(idx)
    assert "err" not in _severities(mixed_pane)
    assert "warn" in _severities(mixed_pane)


def test_the_filter_does_not_touch_the_report(mixed_pane) -> None:
    """Narrowing the list must not change what the pane knows."""
    idx = mixed_pane._sev_filter.findData("err")
    mixed_pane._sev_filter.setCurrentIndex(idx)
    issues = mixed_pane._report.files[0].issues
    assert len(issues) == 2


def test_a_mixed_section_shows_one_chip_per_severity(mixed_pane) -> None:
    """"5" in amber, for three warnings and two errors, is wrong twice.

    It hides that there are errors at all, and it hides how many.
    """
    from bidsmgr.gui.widgets.validation_pane import _count_chips

    issues = mixed_pane._report.files[0].issues
    kinds = [(c.text(), c.property("chipKind")) for c in _count_chips(issues)]
    assert kinds == [("1", "err"), ("1", "warn")]


def test_a_single_severity_section_still_shows_one_chip(qapp) -> None:
    """Nothing gets noisier than it was when there is only one kind."""
    from bidsmgr.gui.widgets.validation_pane import _count_chips

    only_warnings = [
        Issue(severity=Severity.WARN, rule_id="R", message="m"),
        Issue(severity=Severity.WARN, rule_id="R", message="m"),
    ]
    kinds = [(c.text(), c.property("chipKind")) for c in _count_chips(only_warnings)]
    assert kinds == [("2", "warn")]


def test_an_empty_section_keeps_a_neutral_zero(qapp) -> None:
    """Or the header jumps sideways as findings appear and disappear."""
    from bidsmgr.gui.widgets.validation_pane import _count_chips

    chips = _count_chips([])
    assert len(chips) == 1
    assert chips[0].text() == "0"
    assert chips[0].property("chipKind") == "default"


def test_the_file_section_renders_both_chips(mixed_pane) -> None:
    """End to end, not just the helper."""
    kinds = [c.property("chipKind") for c in mixed_pane.findChildren(Chip)]
    assert "err" in kinds and "warn" in kinds


def test_an_error_group_chip_is_red_not_amber(qtbot, tmp_path: Path) -> None:
    """The bug: ``severity.value`` is "err", and the code asked for "error"."""
    root = tmp_path / "DS"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    target = anat / "sub-01_ses-01_T1w.json"
    target.write_text("{}")
    pane = ValidationPane()
    qtbot.addWidget(pane)
    pane.set_report(_mixed_report(root))
    pane.set_current_file(target, root)
    # Switch to the whole-dataset (grouped) view, where the chips live.
    pane._on_mode_clicked(1)

    # A list, not a dict: both groups fired on one file so both chips read
    # "1 file", and keying by text would hide one of them.
    kinds = [chip.property("chipKind") for chip in pane.findChildren(Chip)]
    assert kinds, "expected group chips in the grouped view"
    assert "err" in kinds, (
        "an error group must be painted as an error, not a warning"
    )
    assert "warn" in kinds

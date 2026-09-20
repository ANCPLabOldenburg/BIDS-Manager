"""The Fix button in the chips popup has to go somewhere.

The report: "the fix Button in the warning messages do not take you to the
problematic field, it just do nothing."

It was literally connected to nothing. ``EditorIssuesDialog`` built a
``ValMessage`` with a ``fix_label`` whenever the finding carried one, so the
button was DRAWN, and its ``fix_requested`` signal had no receiver. The
validation pane's identical button was wired; this one never had been.

The dialog lists files other than the one on screen, so a fix from here has to
select the file first and focus the field second. Doing it the other way round
focuses a field in whatever was already open.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor.types import (
    FileVerdict,
    Issue,
    Severity,
    ValidationReport,
)
from bidsmgr.gui.editor_issues_dialog import EditorIssuesDialog
from bidsmgr.gui.widgets.val_message import ValMessage

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for sub in ("sub-01", "sub-02"):
        folder = root / sub / "anat"
        folder.mkdir(parents=True)
        (folder / f"{sub}_T1w.nii.gz").write_bytes(b"\0" * 16)
        (folder / f"{sub}_T1w.json").write_text(json.dumps({"EchoTime": 0.03}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


def _report(root: Path) -> ValidationReport:
    return ValidationReport(
        bids_root=root,
        files=[
            FileVerdict(
                path=Path("sub-01/anat/sub-01_T1w.json"),
                severity=Severity.WARN,
                datatype="anat", suffix="T1w",
                issues=[Issue(
                    severity=Severity.WARN,
                    rule_id="SIDECAR_KEY_RECOMMENDED",
                    message="missing recommended field",
                    field="InstitutionName",
                    fix_label="Fix",
                )],
            ),
            FileVerdict(
                path=Path("sub-02/anat/sub-02_T1w.json"),
                severity=Severity.WARN,
                datatype="anat", suffix="T1w",
                issues=[Issue(
                    severity=Severity.WARN,
                    rule_id="SIDECAR_KEY_RECOMMENDED",
                    message="missing recommended field",
                    field="Manufacturer",
                    fix_label="Fix",
                )],
            ),
        ],
    )


def _fix_buttons(dialog) -> list:
    from PyQt6.QtWidgets import QPushButton

    return [
        b for b in dialog.findChildren(QPushButton)
        if b.objectName() == "val-fix"
    ]


def test_the_button_is_drawn_when_the_finding_offers_one(
    qtbot, dataset: Path,
) -> None:
    dialog = EditorIssuesDialog(_report(dataset), "warn", dataset)
    qtbot.addWidget(dialog)
    assert len(_fix_buttons(dialog)) == 2


def test_pressing_it_announces_the_file_and_the_field(
    qtbot, dataset: Path,
) -> None:
    """The defect: this signal had no receiver, so the button did nothing."""
    dialog = EditorIssuesDialog(_report(dataset), "warn", dataset)
    qtbot.addWidget(dialog)
    seen: list[tuple] = []
    dialog.fix_requested.connect(lambda p, f: seen.append((p, f)))

    _fix_buttons(dialog)[0].click()
    assert seen, "the Fix button must emit something"
    path, field = seen[0]
    assert Path(path).name == "sub-01_T1w.json"
    assert field == "InstitutionName"


def test_each_button_carries_its_own_file(qtbot, dataset: Path) -> None:
    """The dialog lists many files, so the button alone cannot say which."""
    dialog = EditorIssuesDialog(_report(dataset), "warn", dataset)
    qtbot.addWidget(dialog)
    seen: list[tuple] = []
    dialog.fix_requested.connect(lambda p, f: seen.append((Path(p).name, f)))

    for button in _fix_buttons(dialog):
        button.click()
    assert seen == [
        ("sub-01_T1w.json", "InstitutionName"),
        ("sub-02_T1w.json", "Manufacturer"),
    ]


def test_the_panel_selects_the_file_before_focusing_the_field(
    qtbot, dataset: Path,
) -> None:
    """Order matters. Focusing first would focus a field in whatever file was
    already open, which is how a button can look like it does nothing."""
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._set_root(dataset, persist=False)

    order: list[str] = []
    panel.select_file_in_tree = lambda p: order.append(f"select:{Path(p).name}")
    panel._on_fix_requested = lambda p, f: order.append(f"focus:{f}")

    target = dataset / "sub-02" / "anat" / "sub-02_T1w.json"
    panel._on_fix_from_dialog(target, "Manufacturer")
    assert order == ["select:sub-02_T1w.json", "focus:Manufacturer"]


def test_a_finding_with_no_fix_label_gets_no_button(
    qtbot, dataset: Path,
) -> None:
    report = ValidationReport(
        bids_root=dataset,
        files=[FileVerdict(
            path=Path("sub-01/anat/sub-01_T1w.json"),
            severity=Severity.WARN,
            issues=[Issue(
                severity=Severity.WARN, rule_id="R", message="no fix here",
            )],
        )],
    )
    dialog = EditorIssuesDialog(report, "warn", dataset)
    qtbot.addWidget(dialog)
    assert not _fix_buttons(dialog)


def test_every_drawn_button_has_a_receiver(qtbot, dataset: Path) -> None:
    """The general form of the bug: a button that is drawn and connected to
    nothing is worse than no button, because it reads as broken."""
    dialog = EditorIssuesDialog(_report(dataset), "warn", dataset)
    qtbot.addWidget(dialog)
    for message in dialog.findChildren(ValMessage):
        if not _fix_buttons(message):
            continue
        assert message.receivers(message.fix_requested) > 0

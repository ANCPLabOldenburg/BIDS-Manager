"""No dialog may paint one widget's text on top of another's.

Reported from a screenshot: the scope bar's summary line was drawn nine
pixels inside its own combo boxes, so the two sets of letters overlapped.
The cause is a Qt rule that is easy to walk into and invisible on the
machine the code was written on.

**A word-wrapped QLabel answers with a height that depends on its width, and
Qt does not carry that answer out through a nested widget into a
QFormLayout.** The form sizes the row for one line, the label asks for two,
and the second lands on the widget above. It only shows once the font is
large enough for the text to need the second line, which is why it appeared
on a user's desktop and not here.

So this checks GEOMETRY rather than a screenshot, at font scales from 1.0 to
2.0, and it compares SIBLINGS only, because a child inside its parent is not
an overlap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from bidsmgr.gui.theme_manager import ThemeManager

pytestmark = pytest.mark.gui

#: 1.0 is the default, 1.15 is what the app ships, and the rest are what a
#: desktop accessibility setting produces.
SCALES = (1.0, 1.15, 1.3, 1.6, 2.0)

#: Widgets that draw text. Frames and containers may sit inside one another.
_TEXTY = (QLabel, QComboBox, QLineEdit, QPushButton)


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    func = root / "sub-001" / "func"
    anat = root / "sub-001" / "anat"
    func.mkdir(parents=True)
    anat.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    (root / "participants.tsv").write_text("participant_id\nsub-001\n")
    for name in (
        "sub-001_task-driving_run-01_recording-trigger_physio.tsv.gz",
        "sub-001_task-driving_run-01_recording-trigger_physio.json",
        "sub-001_task-driving_run-01_bold.nii.gz",
        "sub-001_task-driving_run-01_bold.json",
    ):
        (func / name).write_bytes(b"")
    (anat / "sub-001_T1w.nii.gz").write_bytes(b"")
    (anat / "sub-001_T1w.json").write_text("{}")
    return root


def _overlaps(top: QWidget) -> list[str]:
    """Every pair of sibling text widgets whose rectangles intersect."""
    found: list[str] = []

    def describe(w: QWidget) -> str:
        for getter in ("text", "currentText", "placeholderText"):
            if hasattr(w, getter):
                try:
                    value = getattr(w, getter)() or ""
                except TypeError:
                    continue
                if value:
                    return f"{type(w).__name__}({value[:40]!r})"
        return type(w).__name__

    def visit(parent: QWidget) -> None:
        kids = [
            c for c in parent.children()
            if isinstance(c, QWidget) and c.isVisibleTo(top)
        ]
        texty = [c for c in kids if isinstance(c, _TEXTY)]
        for i, a in enumerate(texty):
            for b in texty[i + 1:]:
                hit = a.geometry().intersected(b.geometry())
                # One pixel of touching is a rounding artefact; two is text
                # on top of text.
                if hit.width() > 1 and hit.height() > 1:
                    found.append(
                        f"{describe(a)} over {describe(b)}: "
                        f"{hit.width()}x{hit.height()}px"
                    )
        for child in kids:
            visit(child)

    visit(top)
    return found


def _check(qtbot, dialog, scale: float) -> None:
    qtbot.addWidget(dialog)
    dialog.resize(760, 720)
    dialog.show()
    qtbot.waitExposed(dialog)
    bad = _overlaps(dialog)
    assert not bad, (
        f"at font scale {scale}, {type(dialog).__name__} paints text on "
        "text:\n  " + "\n  ".join(bad)
    )


@pytest.fixture(params=SCALES)
def scaled(request, qapp):
    """Apply a font scale for one test, and put the default back after."""
    theme = ThemeManager(qapp, font_scale=request.param)
    theme.apply("dark")
    yield request.param
    ThemeManager(qapp, font_scale=1.0).apply("dark")


def test_sessions(qtbot, dataset, scaled):
    from bidsmgr.gui.edit_entities_dialog import EditEntitiesDialog

    physio = (
        dataset / "sub-001" / "func"
        / "sub-001_task-driving_run-01_recording-trigger_physio.tsv.gz"
    )
    _check(qtbot, EditEntitiesDialog(dataset, [physio], session_mode=True),
           scaled)


def test_add_or_remove_an_entity(qtbot, dataset, scaled):
    from bidsmgr.gui.edit_entities_dialog import EditEntitiesDialog

    physio = (
        dataset / "sub-001" / "func"
        / "sub-001_task-driving_run-01_recording-trigger_physio.tsv.gz"
    )
    _check(qtbot, EditEntitiesDialog(dataset, [physio]), scaled)


def test_delete(qtbot, dataset, scaled):
    from bidsmgr.gui.delete_dialog import DeleteDialog

    physio = (
        dataset / "sub-001" / "func"
        / "sub-001_task-driving_run-01_recording-trigger_physio.tsv.gz"
    )
    _check(qtbot, DeleteDialog(dataset, [physio]), scaled)


def test_index_widths(qtbot, dataset, scaled):
    from bidsmgr.gui.pad_values_dialog import PadValuesDialog

    _check(qtbot, PadValuesDialog(dataset), scaled)


def test_references(qtbot, dataset, scaled):
    from bidsmgr.gui.linkage_dialog import LinkageDialog

    _check(qtbot, LinkageDialog(dataset), scaled)


def test_deface(qtbot, dataset, scaled):
    from bidsmgr.gui.deface_dialog import DefaceDialog

    _check(qtbot, DefaceDialog(dataset, None), scaled)


def test_the_check_would_have_caught_the_reported_one(qtbot, dataset, qapp):
    """The fix was to stop word-wrapping a label inside a nested widget.

    Putting the wrap back has to make the check fail, or the check is
    decoration.
    """
    from bidsmgr.gui.edit_entities_dialog import EditEntitiesDialog

    theme = ThemeManager(qapp, font_scale=1.6)
    theme.apply("dark")
    try:
        physio = (
            dataset / "sub-001" / "func"
            / "sub-001_task-driving_run-01_recording-trigger_physio.tsv.gz"
        )
        dialog = EditEntitiesDialog(dataset, [physio])
        qtbot.addWidget(dialog)
        bar = dialog._scope_bar
        bar._summary.setVisible(True)
        bar._summary.setWordWrap(True)
        bar._summary.setText(
            "a summary long enough to need a second line at this font scale, "
            "which is what the reported screenshot had"
        )
        dialog.resize(760, 720)
        dialog.show()
        qtbot.waitExposed(dialog)
        assert _overlaps(dialog), "the check no longer detects the bug"
    finally:
        ThemeManager(qapp, font_scale=1.0).apply("dark")

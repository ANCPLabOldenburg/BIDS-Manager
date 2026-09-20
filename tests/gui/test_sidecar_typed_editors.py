"""The Editor fills a field the way its TYPE says to, not with a text box.

The metadata templates already ask each question with the control the field
deserves: a controlled vocabulary is a dropdown, a boolean is true/false, a
list is an add-and-remove list, a number is a numeric box with its unit
beside it. The Editor asked all of them with a line edit, so the same field
was filled two different ways depending on where you met it, and nothing
stopped a value the schema forbids.

Both now build from one description and one builder, so they cannot drift.
The tests below check the CONTROL matches what the schema declares, and that
what comes back out of it is the typed value rather than the text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QComboBox, QLineEdit

from bidsmgr import schema as sc
from bidsmgr.editor.types import FileVerdict, Severity, ValidationReport
from bidsmgr.gui.widgets.sidecar_form_pane import SCOPE_PRESENT, SidecarFormPane
from bidsmgr.gui.widgets.sidecar_row import SidecarRow

pytestmark = pytest.mark.gui


def _dataset(tmp_path: Path, payload: dict, datatype: str = "eeg",
             suffix: str = "eeg") -> tuple[Path, Path, ValidationReport]:
    root = tmp_path / "ds"
    folder = root / "sub-01" / datatype
    folder.mkdir(parents=True)
    name = f"sub-01_task-rest_{suffix}.json"
    target = folder / name
    target.write_text(json.dumps(payload, indent=2))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    report = ValidationReport(
        bids_root=root,
        files=[FileVerdict(
            path=Path(f"sub-01/{datatype}/{name}"),
            severity=Severity.WARN, datatype=datatype, suffix=suffix,
        )],
    )
    return root, target, report


def _pane(qtbot, root: Path, target: Path, report) -> SidecarFormPane:
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_autosave(False)
    pane.set_field_scope(SCOPE_PRESENT)
    pane.set_file(target, root, report)
    return pane


def _row(pane: SidecarFormPane, key: str) -> SidecarRow:
    return next(r for r in pane.findChildren(SidecarRow) if r.key == key)


# ---------------------------------------------------------------------------
# The control matches the declared type
# ---------------------------------------------------------------------------


def test_a_controlled_vocabulary_becomes_a_dropdown(
    qtbot, tmp_path: Path,
) -> None:
    root, target, report = _dataset(
        tmp_path, {"TaskName": "rest", "RecordingType": "continuous"},
    )
    pane = _pane(qtbot, root, target, report)
    editor = _row(pane, "RecordingType").editor()
    assert isinstance(editor, QComboBox)
    assert editor.property("template_kind") == "enum"

    offered = {editor.itemText(i) for i in range(editor.count())} - {""}
    declared = {
        str(v) for v in sc.field_metadata("RecordingType").enum
    } or offered
    assert offered == declared, "the list is the schema's, not ours"
    assert editor.currentText() == "continuous"


def test_a_number_gets_a_numeric_box_with_its_unit(
    qtbot, tmp_path: Path,
) -> None:
    root, target, report = _dataset(
        tmp_path, {"TaskName": "rest", "SamplingFrequency": 1000},
    )
    from PyQt6.QtWidgets import QLabel

    pane = _pane(qtbot, root, target, report)
    row = _row(pane, "SamplingFrequency")
    assert row.editor().property("template_kind") == "number"
    assert row.editor().validator() is not None
    # The unit is in the row's key label, so a dose box cannot be filled in
    # the wrong unit for want of saying which.
    texts = " ".join(lbl.text() for lbl in row.findChildren(QLabel))
    # From the per-datatype spec, not from ``field_metadata``: the units live
    # on the sidecar rules, and the bare object entry has none.
    unit = next(
        f.unit for f in sc.sidecar_fields("eeg", "eeg")
        if f.name == "SamplingFrequency"
    )
    assert unit == "Hz"
    assert unit in texts


def test_a_field_the_standard_never_heard_of_still_gets_a_box(
    qtbot, tmp_path: Path,
) -> None:
    """A user is entitled to keep a key of their own."""
    root, target, report = _dataset(
        tmp_path, {"TaskName": "rest", "LabInternalCode": "XY-7"},
    )
    pane = _pane(qtbot, root, target, report)
    editor = _row(pane, "LabInternalCode").editor()
    assert isinstance(editor, QLineEdit)
    # The untyped fallback seeds JSON, so a string arrives quoted and
    # round-trips through the JSON-first parse on commit.
    assert editor.text() == '"XY-7"'
    assert editor.property("template_kind") is None


def test_dataset_description_fields_are_typed_too(
    qtbot, tmp_path: Path,
) -> None:
    """A top-level file has no datatype, so it needs its own lookup, and
    Authors is the field most worth getting right."""
    root = tmp_path / "ds"
    root.mkdir()
    target = root / "dataset_description.json"
    target.write_text(json.dumps({
        "Name": "ds", "BIDSVersion": "1.10.0",
        "Authors": ["A Person", "Another Person"],
    }))
    pane = _pane(qtbot, root, target, None)
    editor = _row(pane, "Authors").editor()
    assert editor.property("template_kind") in ("people", "list")
    assert editor.value() == ["A Person", "Another Person"]


# ---------------------------------------------------------------------------
# What comes back out is the typed value
# ---------------------------------------------------------------------------


def test_a_dropdown_commits_the_value_not_the_index(
    qtbot, tmp_path: Path,
) -> None:
    root, target, report = _dataset(
        tmp_path, {"TaskName": "rest", "RecordingType": "continuous"},
    )
    pane = _pane(qtbot, root, target, report)
    row = _row(pane, "RecordingType")
    seen: list = []
    row.value_committed.connect(lambda k, v, kind: seen.append((k, v)))
    editor = row.editor()
    editor.setCurrentIndex(editor.findText("epoched"))
    # ``activated`` is what a user click emits; a programmatic
    # ``setCurrentIndex`` deliberately does not, so seeding a form cannot
    # look like an edit.
    editor.activated.emit(editor.currentIndex())
    assert seen and seen[-1] == ("RecordingType", "epoched")


def test_a_numeric_field_commits_a_number(qtbot, tmp_path: Path) -> None:
    """A number captured as text fails validation for a field the user
    answered correctly."""
    root, target, report = _dataset(
        tmp_path, {"TaskName": "rest", "SamplingFrequency": 1000},
    )
    pane = _pane(qtbot, root, target, report)
    row = _row(pane, "SamplingFrequency")
    seen: list = []
    row.value_committed.connect(lambda k, v, kind: seen.append(v))
    row.editor().setText("2048")
    row.editor().editingFinished.emit()
    assert seen and seen[-1] == 2048
    assert not isinstance(seen[-1], str)


def test_editing_a_typed_control_marks_the_file_dirty_at_once(
    qtbot, tmp_path: Path,
) -> None:
    """The unsaved indicator has to appear as you type, not when you click
    away. It has to do so whatever kind of control the field got."""
    root, target, report = _dataset(
        tmp_path, {"TaskName": "rest", "SamplingFrequency": 1000},
    )
    pane = _pane(qtbot, root, target, report)
    row = _row(pane, "SamplingFrequency")
    started: list = []
    row.editing_started.connect(started.append)
    row.editor().textEdited.emit("20")
    assert started == ["SamplingFrequency"]


def test_seeding_a_control_does_not_mark_the_file_dirty(
    qtbot, tmp_path: Path,
) -> None:
    root, target, report = _dataset(
        tmp_path, {"TaskName": "rest", "RecordingType": "continuous"},
    )
    pane = _pane(qtbot, root, target, report)
    assert not pane.is_dirty()


# ---------------------------------------------------------------------------
# Wrong data must not break the form
# ---------------------------------------------------------------------------


def test_a_value_of_the_wrong_type_still_renders(
    qtbot, tmp_path: Path,
) -> None:
    """A file may hold anything. A form that refuses to draw is worse than a
    form showing something the validator will flag."""
    root, target, report = _dataset(
        tmp_path,
        {"TaskName": "rest", "SamplingFrequency": {"unexpected": "object"}},
    )
    pane = _pane(qtbot, root, target, report)
    assert _row(pane, "SamplingFrequency").editor() is not None


def test_a_file_with_no_verdict_falls_back_without_crashing(
    qtbot, tmp_path: Path,
) -> None:
    root, target, _ = _dataset(tmp_path, {"TaskName": "rest"})
    pane = _pane(qtbot, root, target, None)
    assert _row(pane, "TaskName").editor() is not None


def test_the_editor_and_the_templates_use_one_builder(
    qtbot, tmp_path: Path,
) -> None:
    """The guarantee the whole change is for: two forms, one description of
    how a field is filled."""
    from bidsmgr.gui.widgets.template_form import build_field_widget
    from bidsmgr.metadata.template_plan import as_template_field

    root, target, report = _dataset(
        tmp_path, {"TaskName": "rest", "RecordingType": "continuous"},
    )
    pane = _pane(qtbot, root, target, report)
    from_editor = _row(pane, "RecordingType").editor()

    spec = next(
        f for f in sc.sidecar_fields("eeg", "eeg") if f.name == "RecordingType"
    )
    from_template = build_field_widget(as_template_field(spec))
    qtbot.addWidget(from_template)

    assert type(from_editor) is type(from_template)
    assert from_editor.property("template_kind") == \
        from_template.property("template_kind")
    assert from_editor.count() == from_template.count()

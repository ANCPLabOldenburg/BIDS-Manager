"""Fixing a field in many files offers the same control as fixing it in one.

The Editor's sidecar form builds its controls from the schema: a field with a
vocabulary gets a dropdown of that vocabulary, a number gets a numeric box, a
list gets a row editor, and the label carries the requirement level and the
description.

"Fix in all files" did none of that. It was a bare text box with the
placeholder "text, or JSON for a number, list or object", so repairing one
field across twelve files was strictly worse than repairing it in one: the
vocabulary was not offered, the type was not enforced, and the user was asked
to hand-write JSON for a list.

Worse than inconvenient, it was a correctness problem. ``json.loads`` on the
typed text made ``0.03`` a float by luck and ``3D`` a string by luck, and
anything that failed to parse became text: a number typed into a numeric field
could reach the sidecar quoted, which is the JSON_SCHEMA_VALIDATION_ERROR the
metadata work already had to chase once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QComboBox, QLineEdit

from bidsmgr.editor import bulk_edit as be
from bidsmgr.gui.bulk_field_dialog import BulkFieldDialog

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for sub in ("sub-01", "sub-02"):
        func = root / sub / "func"
        func.mkdir(parents=True)
        (func / f"{sub}_task-rest_bold.nii.gz").write_bytes(b"\0" * 32)
        (func / f"{sub}_task-rest_bold.json").write_text(
            json.dumps({"RepetitionTime": 2.0})
        )
        eeg = root / sub / "eeg"
        eeg.mkdir(parents=True)
        (eeg / f"{sub}_task-rest_eeg.edf").write_bytes(b"\0" * 32)
        (eeg / f"{sub}_task-rest_eeg.json").write_text(
            json.dumps({"TaskName": "rest"})
        )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0", "DatasetType": "raw"})
    )
    return root


def _dialog(qtbot, root: Path, field: str, datatype: str = "func"):
    stem, ext = ("bold", ".nii.gz") if datatype == "func" else ("eeg", ".edf")
    paths = [
        root / sub / datatype / f"{sub}_task-rest_{stem}{ext}"
        for sub in ("sub-01", "sub-02")
    ]
    dlg = BulkFieldDialog(
        root, field, candidates=be.candidates(root, field, paths=paths),
    )
    qtbot.addWidget(dlg)
    return dlg


def _set(dlg, text: str):
    widget = dlg._value_edit
    if isinstance(widget, QComboBox):
        widget.setCurrentText(text)
    else:
        widget.setText(text)
    return dlg.value()


# ---------------------------------------------------------------------------
# The control matches the field
# ---------------------------------------------------------------------------


def test_a_vocabulary_field_gets_its_vocabulary(qtbot, dataset: Path) -> None:
    dlg = _dialog(qtbot, dataset, "MRAcquisitionType")
    assert isinstance(dlg._value_edit, QComboBox)
    items = [dlg._value_edit.itemText(i) for i in range(dlg._value_edit.count())]
    assert "2D" in items and "3D" in items


def test_the_vocabulary_stays_typable(qtbot, dataset: Path) -> None:
    """A schema vocabulary is offered, never imposed: the tool records what
    the user states and the validator judges it."""
    dlg = _dialog(qtbot, dataset, "MRAcquisitionType")
    assert dlg._value_edit.isEditable()
    assert _set(dlg, "something else") == "something else"


@pytest.mark.parametrize("field,datatype,expected", [
    ("EEGReference",    "eeg",  "average"),
    ("EEGGround",       "eeg",  "AFz"),
    ("Manufacturer",    "eeg",  "Brain Products"),
    ("CapManufacturer", "eeg",  "EasyCap"),
    ("Manufacturer",    "func", "Brain Products"),
])
def test_a_curated_field_gets_its_list(
    qtbot, dataset: Path, field: str, datatype: str, expected: str,
) -> None:
    """The SECOND source of options, and the one that was still missing after
    the schema vocabularies were wired in.

    BIDS leaves these fields as free text, so they have no ``enum``; BIDS
    Manager curates a list of what people actually write. Passing no
    suggestions left the same field a dropdown in the sidecar form and a bare
    box here.
    """
    dlg = _dialog(qtbot, dataset, field, datatype=datatype)
    assert isinstance(dlg._value_edit, QComboBox), f"{field} should offer a list"
    items = [dlg._value_edit.itemText(i) for i in range(dlg._value_edit.count())]
    assert expected in items, items[:8]


@pytest.mark.parametrize("field,datatype", [
    ("EEGReference", "eeg"),
    ("EEGGround", "eeg"),
    ("Manufacturer", "eeg"),
    ("CapManufacturer", "eeg"),
    ("MRAcquisitionType", "func"),
    ("InstitutionName", "func"),
    ("EchoTime", "func"),
])
def test_it_offers_exactly_what_the_sidecar_form_offers(
    qtbot, dataset: Path, field: str, datatype: str,
) -> None:
    """The complaint, stated as a test: the same field, filled the same way.

    Builds the sidecar form's row for the field and this dialog's control for
    the same field, and compares the kind of control and the options. A future
    change that adds a vocabulary to one and not the other fails here.
    """
    from bidsmgr import schema as schema_mod
    from bidsmgr.gui.widgets.sidecar_row import SidecarRow
    from bidsmgr.metadata.template_plan import as_template_field

    suffix = "bold" if datatype == "func" else datatype
    spec = next(
        (s for s in schema_mod.sidecar_fields(datatype, suffix)
         if s.name == field),
        None,
    )
    assert spec is not None, f"{field} not declared for {datatype}"

    row = SidecarRow(
        spec.level, field, "", "missing",
        editable=True, schema_field=as_template_field(spec),
    )
    qtbot.addWidget(row)
    mine = _dialog(qtbot, dataset, field, datatype=datatype)._value_edit
    theirs = row._build_schema_editor(None, "missing", "")

    assert type(mine) is type(theirs), (
        f"{field}: dialog builds {type(mine).__name__}, "
        f"sidecar form builds {type(theirs).__name__}"
    )
    if isinstance(mine, QComboBox):
        assert (
            [mine.itemText(i) for i in range(mine.count())]
            == [theirs.itemText(i) for i in range(theirs.count())]
        ), f"{field}: the two offer different options"


def test_the_schema_is_found_for_these_files(qtbot, dataset: Path) -> None:
    for field in ("EchoTime", "InstitutionName", "MRAcquisitionType"):
        dlg = _dialog(qtbot, dataset, field)
        assert dlg._spec is not None, field
        assert dlg._spec.name == field


def test_an_eeg_field_resolves_against_eeg(qtbot, dataset: Path) -> None:
    """A field's declaration depends on the datatype, so the candidates decide
    it. EchoTime is a number under func and is not declared for eeg at all."""
    dlg = _dialog(qtbot, dataset, "PowerLineFrequency", datatype="eeg")
    assert dlg._spec is not None
    assert dlg._spec.name == "PowerLineFrequency"


# ---------------------------------------------------------------------------
# The value has the shape the schema declares
# ---------------------------------------------------------------------------


def test_a_number_is_written_as_a_number(qtbot, dataset: Path) -> None:
    """Quoted, it is a JSON_SCHEMA_VALIDATION_ERROR on a field the user
    answered correctly."""
    dlg = _dialog(qtbot, dataset, "EchoTime")
    got = _set(dlg, "0.03")
    assert got == 0.03
    assert isinstance(got, float)


def test_a_string_field_is_not_coerced(qtbot, dataset: Path) -> None:
    dlg = _dialog(qtbot, dataset, "InstitutionName")
    got = _set(dlg, "ANCP Lab")
    assert got == "ANCP Lab"
    assert isinstance(got, str)


def test_a_numeric_looking_string_stays_a_string(qtbot, dataset: Path) -> None:
    """Where the old guesswork actually broke, rather than where it happened
    to be right.

    ``json.loads`` on the typed text made "0.03" a float and "3D" a string by
    luck, so those cases looked fine. A site number typed into a STRING field
    is the case luck does not cover: JSON parses "42" to an integer, and the
    sidecar ends up with a number where the standard declares text.
    """
    dlg = _dialog(qtbot, dataset, "InstitutionName")
    got = _set(dlg, "42")
    assert got == "42", "a string field must keep the string"
    assert isinstance(got, str), f"got {type(got).__name__}"


def test_an_anyof_field_takes_either_shape(qtbot, dataset: Path) -> None:
    """PowerLineFrequency is a number OR the literal "n/a". Both have to
    survive, in their own types."""
    dlg = _dialog(qtbot, dataset, "PowerLineFrequency", datatype="eeg")
    assert _set(dlg, "50") == 50
    assert _set(dlg, "n/a") == "n/a"


def test_an_empty_value_stays_empty(qtbot, dataset: Path) -> None:
    dlg = _dialog(qtbot, dataset, "EchoTime")
    assert _set(dlg, "") == ""


# ---------------------------------------------------------------------------
# A field the schema does not declare
# ---------------------------------------------------------------------------


def test_an_unknown_field_falls_back_to_text(qtbot, dataset: Path) -> None:
    """Findings BIDS Manager raises itself are not always schema fields. The
    plain box is the FALLBACK, not the default."""
    dlg = _dialog(qtbot, dataset, "NotARealField")
    assert dlg._spec is None
    assert isinstance(dlg._value_edit, QLineEdit)


def test_the_fallback_still_parses_json(qtbot, dataset: Path) -> None:
    dlg = _dialog(qtbot, dataset, "NotARealField")
    assert _set(dlg, '{"a": 1}') == {"a": 1}
    assert _set(dlg, "42") == 42
    assert _set(dlg, "plain text") == "plain text"


def test_the_fallback_says_why_it_is_plain(qtbot, dataset: Path) -> None:
    """A box with no vocabulary should not read as a box whose field has
    none."""
    dlg = _dialog(qtbot, dataset, "NotARealField")
    assert "does not declare this field" in dlg._value_edit.toolTip()


# ---------------------------------------------------------------------------
# The preview keeps up
# ---------------------------------------------------------------------------


def test_the_preview_updates_as_you_type(qtbot, dataset: Path) -> None:
    """``connect_field_widget`` settles on editingFinished, which never fires
    while the only editable control still has focus. The preview would have
    stayed empty for the whole life of the dialog."""
    dlg = _dialog(qtbot, dataset, "InstitutionName")
    dlg._value_edit.setText("ANCP Lab")
    shown = [
        dlg._table.item(r, 2).text() for r in range(dlg._table.rowCount())
        if dlg._table.item(r, 2) is not None
    ]
    assert shown, "expected preview rows"
    assert all("ANCP Lab" in text for text in shown), shown


def test_the_preview_updates_from_a_combo(qtbot, dataset: Path) -> None:
    dlg = _dialog(qtbot, dataset, "MRAcquisitionType")
    dlg._value_edit.setCurrentText("3D")
    shown = [
        dlg._table.item(r, 2).text() for r in range(dlg._table.rowCount())
        if dlg._table.item(r, 2) is not None
    ]
    assert all("3D" in text for text in shown), shown


# ---------------------------------------------------------------------------
# And it actually writes
# ---------------------------------------------------------------------------


def test_applying_writes_the_typed_shape(qtbot, dataset: Path) -> None:
    dlg = _dialog(qtbot, dataset, "EchoTime")
    _set(dlg, "0.03")
    result = be.apply_value(
        dataset, dlg.selected(), "EchoTime", dlg.value(), label="test",
    )
    assert result.written
    written = json.loads(
        (dataset / "sub-01" / "func" / "sub-01_task-rest_bold.json").read_text()
    )
    assert written["EchoTime"] == 0.03
    assert not isinstance(written["EchoTime"], str)

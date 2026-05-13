"""Tests for the editable sidecar form (M6 Step 6a).

JSON sidecar files render with inline editors:

* ``string`` / ``todo`` / ``missing`` / ``null`` / ``array`` /
  ``object`` → :class:`QLineEdit`.
* ``number`` → :class:`QLineEdit` with a permissive float validator.
* ``bool``   → :class:`QComboBox` (``true`` / ``false``).

On commit (Enter / focus-out / combo activation) the row emits
``value_committed(key, parsed_value, value_kind)``. The pane catches
the signal, updates its cached :class:`OrderedDict`, and writes the
file back — preserving the original key order, appending new fields,
and skipping no-ops.

Per-file revalidation lives in Step 6b; this file covers the
edit-and-save round trip only.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QComboBox, QLineEdit

from bidsmgr.editor.types import (
    FieldLevel,
    FileVerdict,
    Severity,
    SidecarField,
    ValidationReport,
)
from bidsmgr.gui.widgets.sidecar_form_pane import (
    SidecarFormPane,
    _load_json_ordered,
)
from bidsmgr.gui.widgets.sidecar_row import (
    SidecarRow,
    _parse_commit_text,
)


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# _parse_commit_text — pure helper
# ---------------------------------------------------------------------------


def test_parse_commit_number_ints_and_floats() -> None:
    assert _parse_commit_text("3", "number") == 3
    assert _parse_commit_text("3.0", "number") == 3.0
    assert _parse_commit_text("1.5e2", "number") == 150.0
    # Empty number-field text means "clear" → None so we can write null.
    assert _parse_commit_text("", "number") is None


def test_parse_commit_json_literal_first_string_fallback() -> None:
    # Scalar JSON literals take effect, even for ``string``-kind fields.
    assert _parse_commit_text("true", "string") is True
    assert _parse_commit_text("42", "string") == 42
    # Container literals are accepted only for fields that were
    # already containers (kind ``array`` / ``object``).
    assert _parse_commit_text('["a", "b"]', "array") == ["a", "b"]
    assert _parse_commit_text('{"k": 1}', "object") == {"k": 1}
    # Plain identifier falls back to string.
    assert _parse_commit_text("Siemens", "string") == "Siemens"


def test_parse_commit_rejects_container_literals_for_scalar_fields() -> None:
    """A user typing ``["a","b"]`` into a string-kind field must NOT
    silently turn the field into a list. Same rule for objects."""
    assert _parse_commit_text('["a", "b"]', "string") == '["a", "b"]'
    assert _parse_commit_text('{"k": 1}', "string") == '{"k": 1}'
    # ``number`` kind falls back to raw text on parse failure, which
    # naturally covers container literals too.
    assert _parse_commit_text("[1, 2]", "number") == "[1, 2]"


# ---------------------------------------------------------------------------
# _load_json_ordered
# ---------------------------------------------------------------------------


def test_load_json_ordered_preserves_order(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(
        '{"b": 1, "a": 2, "c": 3}'
    )
    data = _load_json_ordered(p)
    assert isinstance(data, OrderedDict)
    assert list(data.keys()) == ["b", "a", "c"]


def test_load_json_ordered_handles_unreadable(tmp_path: Path) -> None:
    assert _load_json_ordered(tmp_path / "does-not-exist.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not-json")
    assert _load_json_ordered(bad) is None


# ---------------------------------------------------------------------------
# SidecarRow — editable rendering
# ---------------------------------------------------------------------------


def test_sidecar_row_readonly_uses_label(qapp) -> None:
    """Default mode renders a QLabel (no editor)."""
    row = SidecarRow("opt", "Foo", "bar", "string")
    assert row.editor() is None


def test_sidecar_row_editable_string_uses_qlineedit(qapp) -> None:
    row = SidecarRow("opt", "Foo", "bar", "string", editable=True, raw_value="bar")
    assert isinstance(row.editor(), QLineEdit)
    assert row.editor().text() == '"bar"'  # raw_value serialised as JSON


def test_sidecar_row_editable_number_uses_validated_qlineedit(qapp) -> None:
    row = SidecarRow("opt", "RT", "2.0", "number", editable=True, raw_value=2.0)
    edit = row.editor()
    assert isinstance(edit, QLineEdit)
    assert edit.validator() is not None
    assert edit.text() == "2.0"


def test_sidecar_row_editable_bool_uses_combo(qapp) -> None:
    row = SidecarRow("opt", "On", "true", "bool", editable=True, raw_value=True)
    combo = row.editor()
    assert isinstance(combo, QComboBox)
    assert combo.count() == 2
    assert combo.currentData() is True


def test_sidecar_row_missing_field_is_blank_qlineedit(qapp) -> None:
    row = SidecarRow("req", "Mag", "(missing)", "missing", editable=True)
    edit = row.editor()
    assert isinstance(edit, QLineEdit)
    assert edit.text() == ""


# ---------------------------------------------------------------------------
# SidecarRow — value_committed signal
# ---------------------------------------------------------------------------


def test_line_edit_emits_committed_with_parsed_value(qapp, qtbot) -> None:
    row = SidecarRow(
        "rec", "RepetitionTime", "2.0", "number",
        editable=True, raw_value=2.0,
    )
    edit = row.editor()
    edit.setText("3.5")
    with qtbot.waitSignal(row.value_committed, timeout=500) as sig:
        edit.editingFinished.emit()
    key, parsed, kind = sig.args
    assert key == "RepetitionTime"
    assert parsed == 3.5
    assert kind == "number"


def test_combo_emits_committed_bool(qapp, qtbot) -> None:
    row = SidecarRow(
        "opt", "OnOff", "true", "bool",
        editable=True, raw_value=True,
    )
    combo = row.editor()
    combo.setCurrentIndex(1)
    with qtbot.waitSignal(row.value_committed, timeout=500) as sig:
        combo.activated.emit(1)
    key, parsed, kind = sig.args
    assert key == "OnOff"
    assert parsed is False
    assert kind == "bool"


def test_string_edit_with_json_literal_parses_to_bool(qapp, qtbot) -> None:
    """Typing ``true`` in a string field saves as a real bool (the
    permissive-JSON rule)."""
    row = SidecarRow("opt", "Foo", "bar", "string", editable=True, raw_value="bar")
    edit = row.editor()
    edit.setText("true")
    with qtbot.waitSignal(row.value_committed, timeout=500) as sig:
        edit.editingFinished.emit()
    _, parsed, _ = sig.args
    assert parsed is True


# ---------------------------------------------------------------------------
# SidecarFormPane — round-trip save
# ---------------------------------------------------------------------------


@pytest.fixture
def bids_root(tmp_path: Path) -> Path:
    root = tmp_path / "Studyname"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-01_T1w.nii.gz").write_bytes(b"")
    (anat / "sub-01_ses-01_T1w.json").write_text(
        json.dumps({
            "Manufacturer": "Siemens",
            "RepetitionTime": 2.0,
            "EchoTime": 0.03,
        }, indent=2)
    )
    (root / "dataset_description.json").write_text(
        '{"Name": "Test", "BIDSVersion": "1.10.0"}'
    )
    return root


def test_editing_only_mutates_cache_until_save(
    qapp, qtbot, bids_root: Path,
) -> None:
    """Manual save: a commit updates the in-memory cache only. The
    disk file is unchanged until the user clicks Save."""
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    rt_row = next(r for r in pane._rows if r.key == "RepetitionTime")
    edit = rt_row.editor()
    edit.setText("3.0")
    edit.editingFinished.emit()
    qapp.processEvents()

    # Disk still has the original value.
    disk = json.loads(json_path.read_text(), object_pairs_hook=OrderedDict)
    assert disk["RepetitionTime"] == 2.0
    # Cache reflects the new value; pane is dirty; Save button on.
    assert pane._json_cache["RepetitionTime"] == 3.0
    assert pane.is_dirty()
    assert pane._save_btn.isEnabled()


def test_save_button_writes_cache_to_disk(
    qapp, qtbot, bids_root: Path,
) -> None:
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    rt_row = next(r for r in pane._rows if r.key == "RepetitionTime")
    rt_row.editor().setText("3.0")
    rt_row.editor().editingFinished.emit()

    with qtbot.waitSignal(pane.file_saved, timeout=500):
        pane.save()

    disk = json.loads(json_path.read_text(), object_pairs_hook=OrderedDict)
    assert disk["RepetitionTime"] == 3.0
    assert list(disk.keys()) == ["Manufacturer", "RepetitionTime", "EchoTime"]
    assert not pane.is_dirty()
    assert not pane._save_btn.isEnabled()


def test_revert_drops_in_memory_changes(
    qapp, qtbot, bids_root: Path,
) -> None:
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    rt_row = next(r for r in pane._rows if r.key == "RepetitionTime")
    rt_row.editor().setText("99.0")
    rt_row.editor().editingFinished.emit()
    qapp.processEvents()
    assert pane.is_dirty()

    pane.revert()
    qapp.processEvents()

    assert not pane.is_dirty()
    # Disk untouched.
    disk = json.loads(json_path.read_text(), object_pairs_hook=OrderedDict)
    assert disk["RepetitionTime"] == 2.0
    # And the in-memory cache restored to the disk value.
    assert pane._json_cache["RepetitionTime"] == 2.0


def test_dirty_count_chip_reflects_changes(
    qapp, qtbot, bids_root: Path,
) -> None:
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    # Two edits → "2 unsaved changes".
    rt_row = next(r for r in pane._rows if r.key == "RepetitionTime")
    rt_row.editor().setText("3.0")
    rt_row.editor().editingFinished.emit()
    et_row = next(r for r in pane._rows if r.key == "EchoTime")
    et_row.editor().setText("0.05")
    et_row.editor().editingFinished.emit()
    qapp.processEvents()

    assert "2 unsaved" in pane._dirty_chip.text()
    assert pane._dirty_chip.isVisibleTo(pane)
    # Save clears it.
    pane.save()
    assert not pane._dirty_chip.isVisibleTo(pane)


def test_switching_files_discards_unsaved_changes(
    qapp, bids_root: Path, tmp_path: Path,
) -> None:
    pane = SidecarFormPane()
    a = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    pane.set_file(a, bids_root, None)
    rt_row = next(r for r in pane._rows if r.key == "RepetitionTime")
    rt_row.editor().setText("99.0")
    rt_row.editor().editingFinished.emit()
    assert pane.is_dirty()

    # Switch to dataset_description.json — the previous dirty cache
    # is discarded; disk for the first file stays at the original.
    b = bids_root / "dataset_description.json"
    pane.set_file(b, bids_root, None)
    assert not pane.is_dirty()

    disk_a = json.loads(a.read_text())
    assert disk_a["RepetitionTime"] == 2.0


def test_editing_appends_a_new_field_when_missing(
    qapp, qtbot, bids_root: Path,
) -> None:
    """A schema-required-but-missing field added by the validator gets
    appended to the JSON on save (we don't keep an empty placeholder
    on disk just because the schema mentioned it)."""
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )

    report = ValidationReport(
        bids_root=bids_root,
        files=[
            FileVerdict(
                path=json_path.relative_to(bids_root),
                severity=Severity.ERR,
                datatype="anat",
                suffix="T1w",
                sidecar_fields=[
                    SidecarField(
                        level=FieldLevel.REQUIRED,
                        name="MagneticFieldStrength",
                        value=None,
                        present=False,
                        value_kind="missing",
                    ),
                ],
            ),
        ],
    )
    pane.set_file(json_path, bids_root, report)

    mfs_row = next(r for r in pane._rows if r.key == "MagneticFieldStrength")
    edit = mfs_row.editor()
    edit.setText("3")
    edit.editingFinished.emit()
    qapp.processEvents()

    with qtbot.waitSignal(pane.file_saved, timeout=500):
        pane.save()

    disk = json.loads(
        json_path.read_text(), object_pairs_hook=OrderedDict,
    )
    # Existing three keys stay in original order; the new one appends.
    assert list(disk.keys()) == [
        "Manufacturer", "RepetitionTime", "EchoTime",
        "MagneticFieldStrength",
    ]
    assert disk["MagneticFieldStrength"] == 3


def test_noop_edit_does_not_make_pane_dirty(
    qapp, bids_root: Path,
) -> None:
    """Re-committing the same value keeps the pane clean and leaves the
    Save button disabled."""
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    rt_row = next(r for r in pane._rows if r.key == "RepetitionTime")
    # Same value re-committed — no dirty.
    rt_row.editor().editingFinished.emit()
    qapp.processEvents()
    assert not pane.is_dirty()
    assert not pane._save_btn.isEnabled()


def test_save_with_no_dirty_state_is_a_noop(
    qapp, bids_root: Path,
) -> None:
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)
    mtime_before = json_path.stat().st_mtime_ns
    ok = pane.save()
    assert ok is True
    # Disk untouched.
    assert json_path.stat().st_mtime_ns == mtime_before


def test_save_failed_signal_when_disk_unwritable(
    qapp, qtbot, bids_root: Path, monkeypatch,
) -> None:
    """When the file write throws ``OSError`` :meth:`save` returns
    ``False`` and emits ``save_failed``."""
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    def _explode(self_, path):
        raise OSError("disk full")

    monkeypatch.setattr(type(pane), "_write_json_cache", _explode)

    rt_row = next(r for r in pane._rows if r.key == "RepetitionTime")
    rt_row.editor().setText("99.0")
    rt_row.editor().editingFinished.emit()
    qapp.processEvents()

    with qtbot.waitSignal(pane.save_failed, timeout=500) as sig:
        ok = pane.save()
    assert ok is False
    _, err_text = sig.args
    assert "disk full" in err_text
    # Pane stays dirty — disk write failed.
    assert pane.is_dirty()


def test_non_json_file_does_not_get_editors(qapp, bids_root: Path) -> None:
    """``.nii.gz`` files don't get a sidecar form, and certainly no
    editor."""
    pane = SidecarFormPane()
    nii_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
    )
    pane.set_file(nii_path, bids_root, None)
    assert pane._rows == []


def test_binary_file_does_not_crash_set_file(
    qapp, bids_root: Path,
) -> None:
    """Regression: clicking a binary file (``.nii.gz`` etc.) used to
    raise ``UnicodeDecodeError`` from ``_load_json_ordered`` and crash
    the GUI. The pane must skip the JSON read for non-JSON paths and
    the helper must absorb decode errors defensively.
    """
    pane = SidecarFormPane()
    nii_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
    )
    # Plant non-UTF-8 bytes so the read would explode if we naively
    # opened the file with ``encoding="utf-8"``.
    nii_path.write_bytes(b"\x1f\x8b\x08\x00\xff\xff\xff\xff")

    # The call must not raise.
    pane.set_file(nii_path, bids_root, None)
    assert pane.current_file() == nii_path
    assert pane._json_cache is None
    assert pane._rows == []

    # Belt + suspenders: pointing the helper at the same binary file
    # also returns None instead of raising.
    assert _load_json_ordered(nii_path) is None

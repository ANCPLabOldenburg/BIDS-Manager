"""Tests for the Editor's alternative JSON tree view (M6 Step 6a.2).

The sidecar pane offers two views for JSON sidecars:

* **BIDS view** (default) — the schema-aware ``"key": value`` form
  with level-coloured bars and per-row inline editors.
* **Tree view** — a 2-column ``QTreeWidget`` (Key / Value) modelled
  on the original BIDS-Manager Inspector. Add Field / Delete Field
  buttons live on the same toolbar.

Both views share a single in-memory cache; edits in either flow into
the same dirty-state pipeline; ``Save`` and ``Revert`` work the same
in either view. The user's view choice is persisted via
:class:`AppSettings`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest

from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.widgets.json_tree_view import (
    LEVEL_ROLE,
    JsonTreeView,
    text_to_value,
    value_to_text,
)
from bidsmgr.gui.widgets.sidecar_form_pane import SidecarFormPane

from bidsmgr.editor.types import (
    FieldLevel,
    FileVerdict,
    Severity,
    SidecarField,
    ValidationReport,
)


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_value_to_text_handles_scalars() -> None:
    assert value_to_text(None) == "null"
    assert value_to_text(True) == "true"
    assert value_to_text(False) == "false"
    assert value_to_text(3) == "3"
    assert value_to_text(1.5) == "1.5"
    assert value_to_text("hello") == "hello"


def test_value_to_text_returns_empty_for_containers() -> None:
    """Containers are shown as expandable parent rows (empty Value
    cell) — :func:`value_to_text` returns ``""`` for them and the
    tree code handles the children separately."""
    assert value_to_text([1, 2, 3]) == ""
    assert value_to_text({"k": 1}) == ""


def test_text_to_value_json_first_then_string() -> None:
    assert text_to_value("true") is True
    assert text_to_value("null") is None
    assert text_to_value("42") == 42
    assert text_to_value("Siemens") == "Siemens"
    assert text_to_value("") == ""


def test_text_to_value_keeps_container_literals_as_strings() -> None:
    """Container literals typed into a value cell stay as raw text —
    new sub-trees can only be created via the Add field button."""
    assert text_to_value("[1, 2]") == "[1, 2]"
    assert text_to_value('{"k": 1}') == '{"k": 1}'


# ---------------------------------------------------------------------------
# JsonTreeView basics
# ---------------------------------------------------------------------------


def test_tree_renders_top_level_dict(qapp) -> None:
    tv = JsonTreeView()
    tv.set_data(OrderedDict([("a", 1), ("b", "text"), ("c", [1, 2])]))
    assert tv.topLevelItemCount() == 3
    # Scalar rows: Value column carries the rendered string.
    assert tv.topLevelItem(0).text(0) == "a"
    assert tv.topLevelItem(0).text(1) == "1"
    # Container row: Value column is empty, children rendered recursively.
    c_item = tv.topLevelItem(2)
    assert c_item.text(0) == "c"
    assert c_item.text(1) == ""
    assert c_item.childCount() == 2
    assert c_item.child(0).text(0) == "[0]"
    assert c_item.child(0).text(1) == "1"
    assert c_item.child(1).text(0) == "[1]"
    assert c_item.child(1).text(1) == "2"


def test_tree_round_trips_via_to_dict(qapp) -> None:
    tv = JsonTreeView()
    src = OrderedDict([("RepetitionTime", 2.0), ("Manufacturer", "Siemens")])
    tv.set_data(src)
    out = tv.to_dict()
    assert out == src
    assert list(out.keys()) == ["RepetitionTime", "Manufacturer"]


def test_tree_edits_emit_model_changed(qapp, qtbot) -> None:
    tv = JsonTreeView()
    tv.set_data(OrderedDict([("k", 1)]))
    item = tv.topLevelItem(0)
    with qtbot.waitSignal(tv.model_changed, timeout=500):
        item.setText(1, "42")
    assert tv.to_dict()["k"] == 42


def test_tree_add_field_always_inserts_at_root(qapp) -> None:
    """``add_field`` ignores the selection and always inserts at the
    top level."""
    tv = JsonTreeView()
    tv.set_data(OrderedDict([
        ("outer", OrderedDict([("a", 1)])),
    ]))
    outer = tv.topLevelItem(0)
    # Even with a container selected, add_field lands at root.
    tv.setCurrentItem(outer)
    new_item = tv.add_field()
    assert new_item.parent() is None
    assert tv.topLevelItemCount() == 2


def test_tree_add_subfield_adds_child_of_selected_container(qapp) -> None:
    """``add_subfield`` puts the new row INSIDE the selected container."""
    tv = JsonTreeView()
    tv.set_data(OrderedDict([
        ("outer", OrderedDict([("a", 1)])),
    ]))
    outer = tv.topLevelItem(0)
    tv.setCurrentItem(outer)
    new_item = tv.add_subfield()
    assert new_item is not None
    assert new_item.parent() is outer
    assert outer.childCount() == 2


def test_tree_add_subfield_promotes_leaf_to_container(qapp) -> None:
    """Clicking Add subfield on a leaf converts it into a container
    with the new child. Matches the original BIDS-Manager Inspector
    behaviour; the leaf's prior value is dropped.
    """
    tv = JsonTreeView()
    tv.set_data(OrderedDict([("a", 1)]))
    leaf = tv.topLevelItem(0)
    tv.setCurrentItem(leaf)
    new_item = tv.add_subfield()
    assert new_item is not None
    assert new_item.parent() is leaf
    assert leaf.childCount() == 1
    # Prior leaf value cell is cleared (now a container row).
    assert leaf.text(1) == ""
    # to_dict round-trips: ``a`` is now a dict, not a scalar.
    out = tv.to_dict()
    assert isinstance(out["a"], dict)


def test_tree_add_subfield_returns_none_with_no_selection(qapp) -> None:
    tv = JsonTreeView()
    tv.set_data(OrderedDict([("a", 1)]))
    assert tv.add_subfield() is None


def test_tree_add_subfield_inside_list_picks_next_index(qapp) -> None:
    """When the target container is list-shaped, the new key uses the
    next ``[N]``."""
    tv = JsonTreeView()
    tv.set_data(OrderedDict([("items", [10, 20, 30])]))
    items = tv.topLevelItem(0)
    tv.setCurrentItem(items)
    new_item = tv.add_subfield()
    assert new_item is not None
    assert items.childCount() == 4
    assert new_item.text(0) == "[3]"


# ---------------------------------------------------------------------------
# Display order: sort by level
# ---------------------------------------------------------------------------


def test_tree_top_level_rows_sort_by_level(qapp) -> None:
    """Required → recommended → optional → deprecated. Within a level
    the input order is preserved."""
    tv = JsonTreeView()
    # Disk order: rec, opt, req, dep, no-level.
    data = OrderedDict([
        ("b_rec", 1),
        ("c_opt", 2),
        ("a_req", 3),
        ("d_dep", 4),
        ("e_unknown", 5),
    ])
    levels = {
        "a_req": "req",
        "b_rec": "rec",
        "c_opt": "opt",
        "d_dep": "dep",
    }
    tv.set_data(data, levels=levels)
    displayed = [tv.topLevelItem(i).text(0) for i in range(tv.topLevelItemCount())]
    # Sorted by level; unknown last.
    assert displayed == ["a_req", "b_rec", "c_opt", "d_dep", "e_unknown"]


def test_pane_save_preserves_disk_key_order_after_tree_edit(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    """Tree display sorts by level, but the on-disk save order keeps
    the file's original key order. New keys append at the end."""
    AppSettings.remember_editor_sidecar_view("tree")
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    # Edit existing key (RepetitionTime) via the tree.
    rt_item = next(
        pane._tree_view.topLevelItem(i)
        for i in range(pane._tree_view.topLevelItemCount())
        if pane._tree_view.topLevelItem(i).text(0) == "RepetitionTime"
    )
    rt_item.setText(1, "3.0")
    qapp.processEvents()

    with qtbot.waitSignal(pane.file_saved, timeout=500):
        pane.save()

    disk = json.loads(json_path.read_text(), object_pairs_hook=OrderedDict)
    # Disk keeps original order despite tree's level-sorted display.
    assert list(disk.keys()) == ["Manufacturer", "RepetitionTime"]
    assert disk["RepetitionTime"] == 3.0


def test_tree_delete_field_removes_nested_item(qapp, qtbot) -> None:
    tv = JsonTreeView()
    tv.set_data(OrderedDict([
        ("outer", OrderedDict([("a", 1), ("b", 2)])),
    ]))
    outer = tv.topLevelItem(0)
    leaf_b = outer.child(1)
    tv.setCurrentItem(leaf_b)
    with qtbot.waitSignal(tv.model_changed, timeout=500):
        removed = tv.delete_field()
    assert removed is True
    assert outer.childCount() == 1
    assert outer.child(0).text(0) == "a"


def test_tree_delete_field_removes_selection(qapp, qtbot) -> None:
    tv = JsonTreeView()
    tv.set_data(OrderedDict([("a", 1), ("b", 2)]))
    tv.setCurrentItem(tv.topLevelItem(0))
    with qtbot.waitSignal(tv.model_changed, timeout=500):
        removed = tv.delete_field()
    assert removed is True
    assert tv.topLevelItemCount() == 1
    assert tv.topLevelItem(0).text(0) == "b"


# ---------------------------------------------------------------------------
# SidecarFormPane: view-mode toggle + persistence
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
        }, indent=2)
    )
    return root


def test_default_view_is_bids(qapp, isolated_settings) -> None:
    pane = SidecarFormPane()
    assert pane.view_mode() == "bids"
    assert pane._view_stack.currentIndex() == 0
    assert not pane._add_field_btn.isVisibleTo(pane)
    assert not pane._del_field_btn.isVisibleTo(pane)


def test_persisted_tree_view_restores_on_construction(
    qapp, isolated_settings,
) -> None:
    AppSettings.remember_editor_sidecar_view("tree")
    pane = SidecarFormPane()
    assert pane.view_mode() == "tree"
    assert pane._view_stack.currentIndex() == 1
    # Add / Delete are shown in tree mode (the toolbar itself is
    # hidden until a JSON file is bound — testing visibility-to-pane
    # rather than effective visibility).
    assert pane._tree_view_btn.isChecked()


def test_view_pill_swap_persists_choice(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)
    assert pane.view_mode() == "bids"

    # Click the Tree pill (simulate the button-group signal).
    pane._tree_view_btn.setChecked(True)
    pane._view_group.idClicked.emit(1)

    assert pane.view_mode() == "tree"
    assert pane._view_stack.currentIndex() == 1
    assert AppSettings.load().editor_sidecar_view == "tree"
    # Tree view is now populated from the cache.
    assert pane._tree_view.topLevelItemCount() == 2


def test_tree_edit_marks_pane_dirty(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    AppSettings.remember_editor_sidecar_view("tree")
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    assert not pane.is_dirty()
    # Edit the RepetitionTime value cell.
    item = next(
        pane._tree_view.topLevelItem(i)
        for i in range(pane._tree_view.topLevelItemCount())
        if pane._tree_view.topLevelItem(i).text(0) == "RepetitionTime"
    )
    item.setText(1, "3.0")
    qapp.processEvents()

    assert pane.is_dirty()
    # Cache reflects the edit; disk untouched (manual save).
    assert pane._json_cache["RepetitionTime"] == 3.0
    disk = json.loads(json_path.read_text())
    assert disk["RepetitionTime"] == 2.0


def test_tree_add_field_then_save_round_trips_to_disk(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    AppSettings.remember_editor_sidecar_view("tree")
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    new_item = pane._tree_view.add_field()
    new_item.setText(0, "EchoTime")
    qapp.processEvents()
    new_item.setText(1, "0.03")
    qapp.processEvents()

    assert pane.is_dirty()
    assert pane._json_cache.get("EchoTime") == 0.03

    with qtbot.waitSignal(pane.file_saved, timeout=500):
        pane.save()

    disk = json.loads(
        json_path.read_text(), object_pairs_hook=OrderedDict,
    )
    assert disk["EchoTime"] == 0.03
    assert list(disk.keys()) == ["Manufacturer", "RepetitionTime", "EchoTime"]


def test_tree_delete_field_marks_pane_dirty(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    AppSettings.remember_editor_sidecar_view("tree")
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    # Select Manufacturer + delete via the toolbar handler.
    item = next(
        pane._tree_view.topLevelItem(i)
        for i in range(pane._tree_view.topLevelItemCount())
        if pane._tree_view.topLevelItem(i).text(0) == "Manufacturer"
    )
    pane._tree_view.setCurrentItem(item)
    pane._on_delete_field_clicked()
    qapp.processEvents()

    assert pane.is_dirty()
    assert "Manufacturer" not in pane._json_cache


def test_view_swap_after_bids_edit_carries_into_tree(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    """Editing in BIDS view then toggling to Tree view must show the
    new value (single cache shared between views)."""
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    rt_row = next(r for r in pane._rows if r.key == "RepetitionTime")
    rt_row.editor().setText("9.99")
    rt_row.editor().editingFinished.emit()
    qapp.processEvents()

    pane._tree_view_btn.setChecked(True)
    pane._view_group.idClicked.emit(1)

    rt_item = next(
        pane._tree_view.topLevelItem(i)
        for i in range(pane._tree_view.topLevelItemCount())
        if pane._tree_view.topLevelItem(i).text(0) == "RepetitionTime"
    )
    assert rt_item.text(1) == "9.99"


def test_tree_value_cell_cannot_promote_leaf_to_container(qapp, qtbot) -> None:
    """Typing a JSON array / object into a leaf's Value cell must
    keep the row as a leaf — containers come from Add field only."""
    tv = JsonTreeView()
    tv.set_data(OrderedDict([("k", "old")]))
    item = tv.topLevelItem(0)
    with qtbot.waitSignal(tv.model_changed, timeout=500):
        item.setText(1, "[1, 2, 3]")
    # The cell now reads the JSON literal text but to_dict keeps it
    # as a string, not promoting the row to a container.
    out = tv.to_dict()
    assert out["k"] == "[1, 2, 3]"
    assert isinstance(out["k"], str)


def test_tree_renders_nested_object_recursively(qapp) -> None:
    """Nested dicts/lists are real children, not stringified JSON."""
    tv = JsonTreeView()
    tv.set_data(OrderedDict([
        ("outer", OrderedDict([
            ("a", 1),
            ("b", [10, 20]),
        ])),
    ]))
    assert tv.topLevelItemCount() == 1
    outer = tv.topLevelItem(0)
    assert outer.text(0) == "outer"
    assert outer.text(1) == ""
    assert outer.childCount() == 2

    a = outer.child(0)
    assert a.text(0) == "a"
    assert a.text(1) == "1"

    b = outer.child(1)
    assert b.text(0) == "b"
    assert b.childCount() == 2
    assert b.child(0).text(0) == "[0]"
    assert b.child(0).text(1) == "10"


def test_tree_round_trips_nested_dict_and_list(qapp) -> None:
    tv = JsonTreeView()
    src = OrderedDict([
        ("Manufacturer", "Siemens"),
        ("SliceTiming", [0.0, 0.5, 1.0]),
        ("Nested", OrderedDict([("k1", 1), ("k2", "v")])),
    ])
    tv.set_data(src)
    out = tv.to_dict()
    # Lists are reconstructed as Python lists when every child key
    # matches ``[N]`` (the list-marker shape).
    assert out["Manufacturer"] == "Siemens"
    assert out["SliceTiming"] == [0.0, 0.5, 1.0]
    assert out["Nested"] == {"k1": 1, "k2": "v"}


def test_tree_levels_annotate_top_level_rows(qapp) -> None:
    """The ``levels`` map on set_data is stored at LEVEL_ROLE on each
    matching top-level item so the delegate can colour the bar."""
    tv = JsonTreeView()
    data = OrderedDict([
        ("Manufacturer", "Siemens"),
        ("RepetitionTime", 2.0),
        ("SomeOptional", "x"),
    ])
    levels = {
        "Manufacturer": "req",
        "RepetitionTime": "rec",
        "SomeOptional": "opt",
    }
    tv.set_data(data, levels=levels)
    for i in range(tv.topLevelItemCount()):
        item = tv.topLevelItem(i)
        assert item.data(0, LEVEL_ROLE) == levels[item.text(0)]


def test_sidecar_pane_passes_levels_to_tree_view(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    """When a verdict is bound, the pane forwards its FieldLevel info
    to the tree so the colour bar matches the BIDS form."""
    AppSettings.remember_editor_sidecar_view("tree")
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    # Hand-crafted report — Manufacturer is REQUIRED, RepetitionTime
    # is RECOMMENDED. The verdict-driven sidecar_fields override the
    # disk-fallback OPTIONAL classification.
    report = ValidationReport(
        bids_root=bids_root,
        files=[
            FileVerdict(
                path=json_path.relative_to(bids_root),
                severity=Severity.WARN,
                datatype="anat",
                suffix="T1w",
                sidecar_fields=[
                    SidecarField(
                        level=FieldLevel.REQUIRED,
                        name="Manufacturer",
                        value="Siemens",
                        present=True,
                        value_kind="string",
                    ),
                    SidecarField(
                        level=FieldLevel.RECOMMENDED,
                        name="RepetitionTime",
                        value=2.0,
                        present=True,
                        value_kind="number",
                    ),
                ],
            ),
        ],
    )
    pane.set_file(json_path, bids_root, report)

    levels = {
        pane._tree_view.topLevelItem(i).text(0):
            pane._tree_view.topLevelItem(i).data(0, LEVEL_ROLE)
        for i in range(pane._tree_view.topLevelItemCount())
    }
    assert levels["Manufacturer"] == "req"
    assert levels["RepetitionTime"] == "rec"


def test_view_swap_after_tree_edit_carries_into_bids(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    """Editing in Tree view then toggling back to BIDS must show the
    new value in the form rows."""
    AppSettings.remember_editor_sidecar_view("tree")
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    # Add a new field via the tree.
    new_item = pane._tree_view.add_field()
    new_item.setText(0, "EchoTime")
    qapp.processEvents()
    new_item.setText(1, "0.04")
    qapp.processEvents()

    # Swap back to BIDS view.
    pane._bids_view_btn.setChecked(True)
    pane._view_group.idClicked.emit(0)

    # The new EchoTime row appears in the BIDS form.
    keys = {r.key for r in pane._rows}
    assert "EchoTime" in keys

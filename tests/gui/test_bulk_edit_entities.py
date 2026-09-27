"""The bulk dialog offers every entity the schema allows, not four of them.

The per-row Properties panel has always built its entity list from
``schema.allowed_entities(datatype, suffix)``. The bulk dialog read a
hand-written tuple whose only entities were subject, session, task and run,
so the one control that exists to set a value on many rows at once could not
set ``acq`` at all. That is the entity that tells two otherwise identical
acquisitions apart, which is exactly the situation a multi-row selection is
usually in.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bidsmgr import schema as schema_mod
from bidsmgr.gui.bulk_edit_dialog import BulkEditDialog
from bidsmgr.gui.models import InventoryTableModel

pytestmark = pytest.mark.gui


def _df() -> pd.DataFrame:
    """Two fieldmap rows and one anatomical, as a scan would leave them."""
    return pd.DataFrame([
        {
            "subject": "OL_4925", "participant_id": "sub-001", "session": "",
            "include": 1, "sequence": "fmap", "series_uid": "uid.6|uid.7",
            "datatype": "fmap", "bids_guess_suffix": "magnitude1",
            "entities": '{"subject": "001", "acquisition": "fm2", "run": "1"}',
            "task": "", "run": "1", "bids_name": "", "dataset": "ds",
        },
        {
            "subject": "OL_4925", "participant_id": "sub-001", "session": "",
            "include": 1, "sequence": "fmap", "series_uid": "uid.12|uid.13",
            "datatype": "fmap", "bids_guess_suffix": "magnitude1",
            "entities": '{"subject": "001", "acquisition": "fm2", "run": "2"}',
            "task": "", "run": "2", "bids_name": "", "dataset": "ds",
        },
        {
            "subject": "OL_4925", "participant_id": "sub-001", "session": "",
            "include": 1, "sequence": "T1w", "series_uid": "uid.5",
            "datatype": "anat", "bids_guess_suffix": "T1w",
            "entities": '{"subject": "001"}',
            "task": "", "run": "", "bids_name": "", "dataset": "ds",
        },
    ])


@pytest.fixture()
def model(qtbot):
    return InventoryTableModel(_df())


# -- the list the model offers ------------------------------------------


def test_acquisition_is_offered(model):
    assert "acquisition" in model.bulk_editable_entities([0, 1])


def test_the_list_comes_from_the_schema(model):
    """Asserted against the schema, not against a copy of it, so this keeps
    holding when the BIDS version moves."""
    offered = model.bulk_editable_entities([0, 1])
    allowed = schema_mod.allowed_entities("fmap", "magnitude1")
    expected = [e for e in schema_mod.entity_order()
                if e in allowed and e not in ("subject", "session", "task", "run")]
    assert offered == expected


def test_entities_with_their_own_column_are_not_offered_twice(model):
    offered = model.bulk_editable_entities([0, 1, 2])
    for entity in ("subject", "session", "task", "run"):
        assert entity not in offered


def test_the_list_is_the_union_across_the_selection(model):
    """``echo`` is allowed on anat/T1w and not on fmap/magnitude1, and a
    selection holding both still offers it.

    This was an INTERSECTION and the intersection was wrong. A bulk edit is
    applied row by row, so an entity one row may not carry is skipped on
    that row rather than withheld from the whole selection. Intersecting
    meant selecting a whole study offered NOTHING, because no entity is
    legal on anat and eeg and meg and pet at once, and with an empty list
    there was no way to reach the remove tick either.
    """
    anat_only = model.bulk_editable_entities([2])
    mixed = model.bulk_editable_entities([0, 1, 2])
    assert "echo" in anat_only
    assert "echo" in mixed, "offered, and skipped on the rows that refuse it"
    assert "acquisition" in mixed


def test_setting_skips_the_rows_that_may_not_carry_it(model):
    """The other half of the union: the OFFER is wide, the WRITE is not."""
    assert model.entity_settable_on(2, "echo"), "anat/T1w may carry echo"
    assert not model.entity_settable_on(0, "echo"), "fmap/magnitude1 may not"
    written = model.bulk_set([0, 1, 2], "entity:echo", "1")
    assert written == 1, "only the row the schema allows it on"
    assert model.bulk_value(2, "entity:echo") == "1"
    assert model.bulk_value(0, "entity:echo") == ""


def test_a_derivative_row_does_not_empty_the_offer(model):
    """A datatype the schema has no rule for returns an EMPTY allowed set
    rather than raising, so it used to collapse the intersection and take
    every entity out of the menu, acq included, with nothing on screen to
    explain it."""
    model._df.at[1, "datatype"] = "derivatives/dcm2niix/sub-001/dwi"
    model._df.at[1, "bids_guess_suffix"] = "dwi"
    offered = model.bulk_editable_entities([0, 1, 2])
    assert "acquisition" in offered
    assert offered, "one unclassifiable row must not empty the menu"
    assert not model.entity_settable_on(1, "acquisition"), (
        "and it is still left alone when the edit is applied"
    )


def test_the_list_is_in_bids_filename_order(model):
    offered = model.bulk_editable_entities([0, 1])
    order = schema_mod.entity_order()
    assert offered == sorted(offered, key=order.index)


def test_an_unclassified_row_does_not_empty_the_list(model):
    """The schema cannot answer for a file it cannot identify. One such row
    in a selection of twenty must not disable the control."""
    model._df.at[1, "datatype"] = ""
    model._df.at[1, "bids_guess_suffix"] = ""
    assert "acquisition" in model.bulk_editable_entities([0, 1])


# -- applying it --------------------------------------------------------


def test_setting_an_entity_writes_it_to_every_selected_row(model):
    key = InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition"
    changed = model.bulk_set([0, 1], key, "gre")
    assert changed == 2
    assert model.entities(0)["acquisition"] == "gre"
    assert model.entities(1)["acquisition"] == "gre"


def test_setting_an_entity_rebuilds_the_basename(model):
    key = InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition"
    model.bulk_set([0, 1], key, "gre")
    for row in (0, 1):
        assert "acq-gre" in model._df.at[row, "bids_name"]


def test_rows_that_already_say_it_are_not_counted_as_changed(model):
    key = InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition"
    assert model.bulk_set([0, 1], key, "fm2") == 0, "both already say fm2"


def test_an_unselected_row_is_untouched(model):
    key = InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition"
    model.bulk_set([0], key, "gre")
    assert model.entities(1)["acquisition"] == "fm2"
    assert "acquisition" not in model.entities(2)


def test_an_unknown_column_key_still_changes_nothing(model):
    assert model.bulk_set([0, 1], "not_a_column", "x") == 0


# -- the dialog ---------------------------------------------------------


def test_the_dialog_lists_acq(qtbot, model):
    dlg = BulkEditDialog(model, [0, 1])
    qtbot.addWidget(dlg)
    labels = [dlg._col_combo.itemText(i) for i in range(dlg._col_combo.count())]
    assert any(label.startswith("acq ") for label in labels), labels


def test_the_dialog_applies_the_entity(qtbot, model):
    dlg = BulkEditDialog(model, [0, 1])
    qtbot.addWidget(dlg)
    key = InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition"
    index = next(
        i for i in range(dlg._col_combo.count())
        if dlg._col_combo.itemData(i) == key
    )
    dlg._col_combo.setCurrentIndex(index)
    # The values already in use are offered; the user may type another.
    dlg._value_combo.setCurrentText("gre")
    dlg._on_apply()
    assert dlg.changed_count() == 2
    assert model.entities(0)["acquisition"] == "gre"


def test_the_dialog_suggests_the_values_already_in_use(qtbot, model):
    model.bulk_set([0], InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition", "gre")
    dlg = BulkEditDialog(model, [0, 1])
    qtbot.addWidget(dlg)
    key = InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition"
    index = next(
        i for i in range(dlg._col_combo.count())
        if dlg._col_combo.itemData(i) == key
    )
    dlg._col_combo.setCurrentIndex(index)
    offered = [dlg._value_combo.itemText(i)
               for i in range(dlg._value_combo.count())]
    assert offered == ["fm2", "gre"]


def test_the_existing_column_targets_are_still_there(qtbot, model):
    dlg = BulkEditDialog(model, [0, 1])
    qtbot.addWidget(dlg)
    keys = [dlg._col_combo.itemData(i) for i in range(dlg._col_combo.count())]
    for key in InventoryTableModel.BULK_EDITABLE_KEYS:
        assert key in keys


# -- removing an entity --------------------------------------------------


def test_an_entity_can_be_removed_from_every_selected_row(model):
    key = InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition"
    assert model.bulk_set([0, 1], key, "") == 2
    assert "acquisition" not in model.entities(0)
    assert "acquisition" not in model.entities(1)


def test_removing_an_entity_rebuilds_the_basename(model):
    key = InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition"
    model.bulk_set([0, 1], key, "")
    for row in (0, 1):
        assert "acq-" not in model._df.at[row, "bids_name"]
        assert "run-" in model._df.at[row, "bids_name"], "the rest stays"


def test_the_remove_tick_is_only_offered_for_entities(qtbot, model):
    dlg = BulkEditDialog(model, [0, 1])
    qtbot.addWidget(dlg)

    def select(key):
        idx = next(i for i in range(dlg._col_combo.count())
                   if dlg._col_combo.itemData(i) == key)
        dlg._col_combo.setCurrentIndex(idx)

    select(InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition")
    assert dlg._remove_check.isVisible() or not dlg.isVisible()
    assert not dlg._remove_check.isHidden()

    select("task")
    assert dlg._remove_check.isHidden(), "a column is not an entity"


def test_the_dialog_removes_when_the_tick_is_on(qtbot, model):
    dlg = BulkEditDialog(model, [0, 1])
    qtbot.addWidget(dlg)
    key = InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition"
    idx = next(i for i in range(dlg._col_combo.count())
               if dlg._col_combo.itemData(i) == key)
    dlg._col_combo.setCurrentIndex(idx)
    dlg._remove_check.setChecked(True)
    dlg._on_apply()
    assert dlg.changed_count() == 2
    assert "acquisition" not in model.entities(0)


def test_the_tick_wins_over_a_value_left_in_the_box(qtbot, model):
    dlg = BulkEditDialog(model, [0, 1])
    qtbot.addWidget(dlg)
    key = InventoryTableModel.ENTITY_KEY_PREFIX + "acquisition"
    idx = next(i for i in range(dlg._col_combo.count())
               if dlg._col_combo.itemData(i) == key)
    dlg._col_combo.setCurrentIndex(idx)
    dlg._value_combo.setCurrentText("gre")
    dlg._remove_check.setChecked(True)
    dlg._on_apply()
    assert "acquisition" not in model.entities(0)


def test_an_empty_value_on_a_column_still_does_nothing(qtbot, model):
    """Unchanged for columns: an empty box is someone mid-typing."""
    dlg = BulkEditDialog(model, [0, 1])
    qtbot.addWidget(dlg)
    idx = next(i for i in range(dlg._col_combo.count())
               if dlg._col_combo.itemData(i) == "task")
    dlg._col_combo.setCurrentIndex(idx)
    dlg._value_edit.setText("")
    dlg._on_apply()
    assert dlg.changed_count() == 0

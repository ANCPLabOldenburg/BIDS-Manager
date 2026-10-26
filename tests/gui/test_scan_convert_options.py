"""The new scan and conversion options in the GUI.

Settings > Scan: where subject labels come from, and the entities a sequence
name decides. Settings > Convert: dummy volumes, the physiology layout and
cropping it to the run. Each one survives a save and a load (and a nonsense
value written by hand heals), each one is shown and saved by the dialog, and
a functional run gets its own dummy-volume count in Properties and the table,
with one Undo, while an anatomical one gets neither.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from PyQt6.QtCore import QSettings

from bidsmgr.gui.app_settings import KEYS, AppSettings

pytestmark = pytest.mark.gui


def _row(datatype="func", suffix="bold", **extra) -> dict:
    name = "sub-001_task-rest_bold" if suffix == "bold" else "sub-001_T1w"
    base = {
        "participant_id": "sub-001", "session": "", "include": 1, "modality": "mri",
        "datatype": datatype, "bids_name": name, "bids_path": name,
        "bids_guess_suffix": suffix, "bids_guess_confidence": "0.97",
        "bids_guess_skip": False, "issues": "",
        "entities": json.dumps({"subject": "001", "task": "rest"} if suffix == "bold"
                               else {"subject": "001"}),
        "task": "rest" if suffix == "bold" else "", "run": "", "source_file": "",
        "series_uid": "1.2.3", "probe_n_volumes": "300", "drop_volumes": "",
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Settings survive
# ---------------------------------------------------------------------------


def test_the_new_settings_round_trip() -> None:
    s = AppSettings()
    s.scan_subject_from = "family_name"
    s.scan_name_entities = ["acq", "run"]
    s.convert_drop_volumes = 4
    s.convert_physio_layout = "one_as_recorded"
    s.convert_physio_align = True
    s.guided_mode = True
    s.save()
    back = AppSettings.load()
    assert back.scan_subject_from == "family_name"
    assert back.scan_name_entities == ["acq", "run"]
    assert back.convert_drop_volumes == 4
    assert back.convert_physio_layout == "one_as_recorded"
    assert back.convert_physio_align is True
    assert back.guided_mode is True


def test_nonsense_written_by_hand_heals() -> None:
    q = QSettings()
    q.setValue(KEYS["scan_subject_from"], "nickname")
    q.setValue(KEYS["scan_name_entities"], json.dumps(["acq", "sub", "zzz"]))
    q.setValue(KEYS["convert_drop_volumes"], "500")
    q.setValue(KEYS["convert_physio_layout"], "per_channel")
    q.sync()
    s = AppSettings.load()
    assert s.scan_subject_from == "number"
    assert s.scan_name_entities == ["acq"]
    assert s.convert_drop_volumes == 100
    assert s.convert_physio_layout == "by_rate"


def test_the_defaults_change_nothing() -> None:
    s = AppSettings()
    assert (s.scan_subject_from, s.scan_name_entities, s.convert_drop_volumes,
            s.convert_physio_layout, s.convert_physio_align, s.guided_mode) == (
        "number", [], 0, "by_rate", False, False)


# ---------------------------------------------------------------------------
# The dialog shows and saves them
# ---------------------------------------------------------------------------


def test_the_dialog_shows_and_saves_them(qtbot) -> None:
    from bidsmgr.gui.settings_dialog import SettingsDialog

    s = AppSettings()
    s.scan_subject_from = "patient_id"
    s.scan_name_entities = ["task"]
    s.convert_drop_volumes = 2
    s.save()
    s = AppSettings.load()
    dlg = SettingsDialog(s)
    qtbot.addWidget(dlg)
    assert dlg._scan_subject_from.currentData() == "patient_id"
    assert [k for k, c in dlg._name_entities.items() if c.isChecked()] == ["task"]
    assert dlg._convert_drop_volumes.value() == 2
    assert dlg._convert_physio_layout.currentData() == "by_rate"
    assert not dlg._convert_physio_align.isChecked()
    assert dlg._convert_drop_volumes.specialValueText() == "None"

    dlg._scan_subject_from.setCurrentIndex(dlg._scan_subject_from.findData("given_name"))
    dlg._name_entities["task"].setChecked(False)
    dlg._name_entities["acq"].setChecked(True)
    dlg._name_entities["run"].setChecked(True)
    dlg._convert_drop_volumes.setValue(5)
    dlg._convert_physio_layout.setCurrentIndex(dlg._convert_physio_layout.findData("one_as_recorded"))
    dlg._convert_physio_align.setChecked(True)
    dlg._on_save()
    back = AppSettings.load()
    assert back.scan_subject_from == "given_name"
    assert back.scan_name_entities == ["acq", "run"]
    assert back.convert_drop_volumes == 5
    assert back.convert_physio_layout == "one_as_recorded"
    assert back.convert_physio_align is True


def test_the_entity_choices_are_the_ones_a_name_can_decide(qtbot) -> None:
    from bidsmgr.gui.settings_dialog import SettingsDialog
    from bidsmgr.inventory.name_entities import offered

    dlg = SettingsDialog(AppSettings())
    qtbot.addWidget(dlg)
    assert list(dlg._name_entities) == offered()
    assert "sub" not in dlg._name_entities and "ses" not in dlg._name_entities


def test_the_entity_tooltips_end_on_a_whole_sentence(qtbot) -> None:
    """Every entity control on Settings > Scan explains its entity from the
    schema. The text used to be cut at 300 characters, mid-word, with its
    Markdown showing ("...MUST be indexed with the [`run-<index>`](SPEC_ROOT"
    then "w"); now it ends a sentence, in lines Qt can show."""
    from bidsmgr.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppSettings())
    qtbot.addWidget(dlg)
    controls = {**{f"name {k}": w for k, w in dlg._name_entities.items()},
                **{f"leave out {k}": w for k, w in dlg._drop_entities.items()},
                **{f"width {k}": w for k, w in dlg._index_widths.items()}}
    assert controls
    for name, widget in controls.items():
        tip = widget.toolTip()
        assert tip.rstrip()[-1] in '.!?)"', (name, tip[-60:])
        assert "`" not in tip and "](" not in tip and "**" not in tip, name
        assert all(len(line) <= 72 for line in tip.splitlines()), name


# ---------------------------------------------------------------------------
# A run's own count: Properties and the table
# ---------------------------------------------------------------------------


def _panel(qtbot, rows):
    from bidsmgr.gui.models import InventoryTableModel
    from bidsmgr.gui.properties_panel import PropertiesPanel

    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    model = InventoryTableModel(pd.DataFrame(rows), parent=panel)
    panel.bind_model(model)
    return panel, model


def test_a_functional_run_gets_its_own_count(qtbot) -> None:
    s = AppSettings()
    s.convert_drop_volumes = 3
    s.save()
    panel, model = _panel(qtbot, [_row(), _row("anat", "T1w")])
    panel.set_selected_row(0)
    spin = panel.dummy_volumes_spin
    assert spin.value() == -1 and spin.text() == "Default (3)"
    spin.setValue(5)
    assert model.drop_volumes(0) == 5
    assert model.dataframe().at[0, "drop_volumes"] == "5"
    spin.setValue(0)
    assert model.drop_volumes(0) == 0, "0 keeps every volume whatever the default"
    spin.setValue(-1)
    assert model.drop_volumes(0) is None


def test_an_anatomical_image_has_no_count(qtbot) -> None:
    panel, model = _panel(qtbot, [_row(), _row("anat", "T1w")])
    panel.dummy_volumes_spin = None
    panel.set_selected_row(1)
    assert panel.dummy_volumes_spin is None
    assert model.column_applies(0, "drop_volumes")
    assert not model.column_applies(1, "drop_volumes")


def test_the_table_cell_takes_a_count_or_nothing(qtbot) -> None:
    from PyQt6.QtCore import Qt

    from bidsmgr.gui.models import InventoryTableModel
    from bidsmgr.gui.models.inventory import COLUMNS

    model = InventoryTableModel(pd.DataFrame([_row(), _row("anat", "T1w")]))
    col = [c.df_column for c in COLUMNS].index("drop_volumes")
    index = model.index(0, col)
    # Editable on a functional run, not on an anatomical image.
    assert model.flags(index) & Qt.ItemFlag.ItemIsEditable
    assert not model.flags(model.index(1, col)) & Qt.ItemFlag.ItemIsEditable
    assert model.setData(index, "4")
    assert model.drop_volumes(0) == 4
    for bad in ("abc", "101", "-2"):
        assert not model.setData(index, bad)
    assert model.drop_volumes(0) == 4
    assert model.setData(index, "")
    assert model.drop_volumes(0) is None
    spec = COLUMNS[col]
    assert spec.default_visible is False, "an option most studies never use stays hidden"


def test_one_change_is_one_undo(qtbot, tmp_path) -> None:
    from bidsmgr.gui.converter_panel import ConverterPanel
    from bidsmgr.project import Project, ScanImported, workspace
    from bidsmgr.cli.create import open_or_create_workspace

    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir = workspace.allocate_version_dir(root, "raw")
    vproj = Project.create(vdir, name=root.name)
    df = pd.DataFrame([_row(), _row("anat", "T1w", series_uid="1.2.4")])
    df.to_csv(workspace.version_inventory(vdir), sep="\t", index=False)
    workspace.write_version_meta(vdir, source_label="raw", raw_root=str(tmp_path / "raw"),
                                 status="curating")
    vproj.append(ScanImported(inventory_tsv=str(workspace.version_inventory(vdir)),
                              row_ids=tuple(df["series_uid"]), raw_root=None))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)
    model = panel._model
    model.set_drop_volumes(0, 6)
    qtbot.wait(20)
    assert model.drop_volumes(0) == 6
    panel._on_undo()
    assert panel._model.drop_volumes(0) is None
    panel._on_redo()
    assert panel._model.drop_volumes(0) == 6


# ---------------------------------------------------------------------------
# A physiological log follows its run: Properties says which, and how many
# ---------------------------------------------------------------------------


def _log_row(name="sub-001_task-rest_physio", **extra) -> dict:
    return _row(suffix="physio", bids_name=name, bids_path=name, series_uid="1.2.4",
                sequence="task-rest_bold_PhysioLog", acq_time="102137.78",
                study_instance_uid="1.9", task="rest",
                entities=json.dumps({"subject": "001", "task": "rest"}), **extra)


_ENTITY = {"sub": "subject", "ses": "session", "task": "task", "acq": "acquisition",
           "run": "run"}


def _run_row(name="sub-001_task-rest_bold", **extra) -> dict:
    entities = {_ENTITY[k]: v for k, v in (t.split("-", 1) for t in name.split("_")[:-1])}
    return _row(bids_name=name, bids_path=name, sequence="task-rest_bold",
                acq_time="102012.13", study_instance_uid="1.9",
                entities=json.dumps(entities), task=entities.get("task", ""),
                session=f"ses-{entities['session']}" if "session" in entities else "", **extra)


def test_a_log_names_the_run_it_follows_and_its_count(qtbot) -> None:
    s = AppSettings()
    s.convert_drop_volumes = 0
    s.save()
    # The run renamed in the table: the log still follows it (by acquisition).
    panel, model = _panel(qtbot, [_run_row("sub-001_task-rest_acq-mb4_bold", drop_volumes="4"),
                                  _log_row(), _row("anat", "T1w")])
    panel.set_selected_row(1)
    assert getattr(panel, "dummy_volumes_spin", None) is None, "the log has no count of its own"
    assert panel.physio_run_label.text() == "sub-001_task-rest_acq-mb4_bold"
    assert panel.physio_volumes_box._badge == "4 removed"
    assert panel.physio_volumes_box._subtitle == "its run"
    # The number is the run's: changed there, the log follows.
    model.set_drop_volumes(0, 2)
    panel.set_selected_row(0)
    assert panel.dummy_volumes_physio.text() == "sub-001_task-rest_physio"
    panel.set_selected_row(1)
    assert panel.physio_volumes_box._badge == "2 removed"
    # The run's default, when it has no number of its own.
    s.convert_drop_volumes = 3
    s.save()
    model.set_drop_volumes(0, None)
    panel.set_selected_row(1)
    assert panel.physio_volumes_box._badge == "3 removed"


def test_a_log_without_its_run_keeps_every_volume(qtbot) -> None:
    panel, _model = _panel(qtbot, [_log_row(), _row("anat", "T1w")])
    panel.set_selected_row(0)
    assert panel.physio_run_label is None
    assert panel.physio_volumes_box._subtitle == "no run"
    assert panel.physio_volumes_box._badge == "none removed"


def test_a_long_run_name_does_not_widen_properties(qtbot) -> None:
    """GUI_LOOK_AND_FEEL 4.23: the name elides, it never sets a minimum width."""
    long = "sub-001_ses-baselinevisit_task-restingstatewitheyesopen_acq-multiband8_run-01_bold"
    short_panel, _m = _panel(qtbot, [_run_row(), _log_row()])
    long_panel, _m2 = _panel(qtbot, [_run_row(long), _log_row()])
    for panel in (short_panel, long_panel):
        panel.set_selected_row(1)
        panel.physio_volumes_box.set_expanded(True)
    assert long_panel.physio_run_label.text() == long
    assert long_panel.minimumSizeHint().width() <= short_panel.minimumSizeHint().width()

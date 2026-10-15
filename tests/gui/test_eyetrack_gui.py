"""Eye tracking in the Converter: the run it belongs to, the screen and the
timing, per recording (Properties) and for every recording (the dataset
dialog)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.gui


def _row(**over) -> dict:
    base = {"participant_id": "sub-001", "session": "", "include": 1, "issues": "",
            "bids_guess_confidence": "0.9", "bids_guess_skip": False, "run": "",
            "dataset": "ds", "source_file": "", "source_folder": "a", "series_uid": "",
            "study_date": "20260917", "acq_time": ""}
    base.update(over)
    return base


def _panel(qtbot, tmp_path: Path):
    from bidsmgr.cli.create import open_or_create_workspace
    from bidsmgr.gui.converter_panel import ConverterPanel
    from bidsmgr.project import Project, ScanImported, workspace

    rows = [
        _row(modality="mri", datatype="func", bids_name="sub-001_task-dmaging_run-3_bold",
             bids_guess_suffix="bold", sequence="bold", task="dmaging", run="3",
             entities=json.dumps({"subject": "001", "task": "dmaging", "run": "3"}),
             series_uid="1.2.3", acq_time="103318.2"),
        _row(modality="eyetrack", datatype="beh",
             bids_name="sub-001_task-ol49253_recording-eye2_physio", bids_guess_suffix="physio",
             sequence="ol49253", task="ol49253", source_file="a/ol49253.EDF",
             entities=json.dumps({"subject": "001", "task": "ol49253",
                                  "recording": "eye2"})),
    ]
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir = workspace.allocate_version_dir(root, "raw")
    vproj = Project.create(vdir, name=root.name)
    df = pd.DataFrame(rows)
    df.to_csv(workspace.version_inventory(vdir), sep="\t", index=False)
    workspace.write_version_meta(vdir, source_label="raw", raw_root=str(tmp_path / "raw"),
                                 status="curating")
    vproj.append(ScanImported(inventory_tsv=str(workspace.version_inventory(vdir)),
                              row_ids=("1.2.3", "a/ol49253.EDF"), raw_root=None))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)
    return panel


def test_properties_pair_the_recording_with_its_run(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    props, model = panel._properties, panel._model
    props.set_selected_row(1)
    pick = props.eyetrack_run_pick
    assert pick.isEnabled() and pick.count() == 2
    assert pick.itemText(1).startswith("sub-001 · func · task-dmaging · run-3")
    props.pair_with_run(1, int(pick.itemData(1)))
    ents = model.entities(1)
    assert ents["task"] == "dmaging" and ents["run"] == "3" and ents["recording"] == "eye2"
    assert model.datatype_suffix(1) == ("func", "physio")
    assert model.dataframe().iloc[1]["bids_name"] == "sub-001_task-dmaging_run-3_recording-eye2_physio"
    # One undo takes the whole pairing back.
    panel._on_undo()
    model = panel._model                 # the table is rebuilt from the log
    assert model.datatype_suffix(1)[0] == "beh"
    assert model.entities(1)["task"] == "ol49253"


def test_properties_state_the_screen_and_the_timing(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    props, model = panel._properties, panel._model
    props.set_selected_row(1)
    fields = props.eyetrack_fields
    fields["screen_size"].setText("53.1 x 29.9")
    fields["screen_size"].editingFinished.emit()
    fields["screen_distance"].setText("60")
    fields["screen_distance"].editingFinished.emit()
    fields["sync_message"].setText("TRIGGER SIGNAL")
    fields["sync_message"].editingFinished.emit()
    assert model.eyetrack_override(1, "screen_size") == [0.531, 0.299]
    assert model.eyetrack_override(1, "screen_distance") == 0.6
    assert model.eyetrack_override(1, "sync_message") == "TRIGGER SIGNAL"
    # A typo is refused, not stored.
    fields["screen_distance"].setText("sixty")
    fields["screen_distance"].editingFinished.emit()
    assert model.eyetrack_override(1, "screen_distance") == 0.6
    # A dataset default shows as the placeholder and is what a row inherits.
    model.set_eyetrack_default("eye_tracker_distance", 0.65)
    props.set_selected_row(1)
    assert props.eyetrack_fields["eye_tracker_distance"].placeholderText().startswith("65")
    assert model.eyetrack_effective(1, "eye_tracker_distance") == 0.65
    assert model.eyetrack_effective(1, "screen_distance") == 0.6


def test_no_eye_tracking_section_on_other_rows(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    props = panel._properties
    props.eyetrack_fields = None
    props.set_selected_row(0)
    assert props.eyetrack_fields is None


def test_the_dataset_dialog_states_the_defaults(qtbot, tmp_path) -> None:
    from bidsmgr.gui.recording_meta_dialog import RecordingMetaDialog
    from bidsmgr.recording_meta import load_spec

    scaffold = tmp_path / "inv.tsv.recording_meta.json"
    dlg = RecordingMetaDialog(scaffold, {"beh"}, present_pairs=[("beh", "physio")],
                              eyetrack=True)
    qtbot.addWidget(dlg)
    dlg._eye_fields["screen_size"].setText("53.1 x 29.9")
    dlg._eye_fields["sync_message"].setText("TRIGGER SIGNAL")
    spec = dlg.build_spec()
    assert spec.eyetrack_defaults.screen_size == [0.531, 0.299]
    assert spec.eyetrack_defaults.sync_message == "TRIGGER SIGNAL"
    dlg._on_save()
    assert load_spec(scaffold).eyetrack_defaults.screen_size == [0.531, 0.299]
    again = RecordingMetaDialog(scaffold, {"beh"}, present_pairs=[("beh", "physio")],
                                eyetrack=True)
    qtbot.addWidget(again)
    assert again._eye_fields["screen_size"].text() == "53.1 x 29.9"

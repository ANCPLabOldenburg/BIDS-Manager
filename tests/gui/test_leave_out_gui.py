"""Leaving entities out of proposed names, from the GUI: the Settings group,
the setting reaching the scan, and the table already open offered the change
as one Undo."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.gui.app_settings import KEYS, AppSettings

pytestmark = pytest.mark.gui


def _rows() -> list[dict]:
    common = {"participant_id": "sub-001", "session": "", "include": 1, "issues": "",
              "modality": "mri", "bids_guess_confidence": "0.85", "bids_guess_skip": False,
              "dataset": "ds", "source_file": "", "source_folder": "s", "task": "",
              "run": "", "acq_time": ""}

    def row(uid, suffix, acq, datatype="anat"):
        ents = {"subject": "001", "acquisition": acq}
        return {**common, "series_uid": uid, "sequence": f"{suffix}_{acq}",
                "datatype": datatype, "bids_guess_suffix": suffix,
                "entities": json.dumps(ents), "bids_name": f"sub-001_acq-{acq}_{suffix}"}

    return [row("1.1", "T1w", "mprage"), row("1.2", "T1w", "highres"),
            row("1.3", "FLAIR", "space")]


def _panel(qtbot, tmp_path: Path):
    from bidsmgr.cli.create import open_or_create_workspace
    from bidsmgr.gui.converter_panel import ConverterPanel
    from bidsmgr.project import Project, ScanImported, workspace

    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir = workspace.allocate_version_dir(root, "raw")
    vproj = Project.create(vdir, name=root.name)
    tsv = workspace.version_inventory(vdir)
    pd.DataFrame(_rows()).to_csv(tsv, sep="\t", index=False)
    workspace.write_version_meta(vdir, source_label="raw", raw_root=str(tmp_path / "raw"),
                                 status="curating")
    vproj.append(ScanImported(inventory_tsv=str(tsv), row_ids=("1.1", "1.2", "1.3"),
                              raw_root=None))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)
    return panel


def _names(panel) -> list[str]:
    return panel._model.dataframe()["bids_name"].tolist()


def test_the_open_table_is_offered_the_change_as_one_undo(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    assert panel.offer_leave_out(["acq"], ask=False) == 1
    assert _names(panel) == ["sub-001_acq-mprage_T1w", "sub-001_acq-highres_T1w",
                             "sub-001_FLAIR"], "acq stays where it tells two T1w apart"
    panel._on_undo()
    assert _names(panel)[2] == "sub-001_acq-space_FLAIR", "one Undo"


def test_choosing_it_in_settings_asks_about_the_open_table(qtbot, tmp_path,
                                                          monkeypatch) -> None:
    from PyQt6.QtWidgets import QMessageBox

    panel = _panel(qtbot, tmp_path)
    asked = []

    def answer(parent, title, text, *rest):
        asked.append(text)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)
    s = AppSettings.load()
    s.scan_drop_entities = ["acq"]
    s.save()
    panel.reload_app_settings()
    assert len(asked) == 1 and "1 recording(s) would be renamed" in asked[0]
    assert "kept it on 2 recordings where it tells two apart" in asked[0]
    assert _names(panel)[2] == "sub-001_FLAIR"

    # Unchanged settings, or an entity taken back off the list: no question.
    panel.reload_app_settings()
    s.scan_drop_entities = []
    s.save()
    panel.reload_app_settings()
    assert len(asked) == 1


def test_the_scan_is_given_the_setting(qtbot, tmp_path, monkeypatch) -> None:
    panel = _panel(qtbot, tmp_path)
    s = AppSettings.load()
    s.scan_drop_entities = ["acq", "run"]
    s.save()
    panel._app_settings = AppSettings.load()
    seen = {}
    monkeypatch.setattr(panel, "start_scan", lambda *a, **kw: seen.update(kw))
    raw = tmp_path / "raw"
    raw.mkdir()
    panel._raw_root = raw
    panel._on_scan_clicked()
    assert seen.get("drop_entities") == ["acq", "run"]


def test_the_setting_survives_and_rejects_what_cannot_be_left_out() -> None:
    s = AppSettings()
    s.scan_drop_entities = ["acq", "run"]
    s.save()
    assert AppSettings.load().scan_drop_entities == ["acq", "run"]
    AppSettings._settings().setValue(KEYS["scan_drop_entities"], '["acq", "task", "sub"]')
    assert AppSettings.load().scan_drop_entities == ["acq"]
    assert AppSettings().scan_drop_entities == [], "nothing left out by default"


def test_the_settings_page_lists_them_and_saves_the_ticks(qtbot) -> None:
    from bidsmgr.gui.settings_dialog import SettingsDialog
    from bidsmgr.inventory.leave_out import offered

    s = AppSettings()
    s.scan_drop_entities = ["rec"]
    dlg = SettingsDialog(s)
    qtbot.addWidget(dlg)
    assert list(dlg._drop_entities) == list(offered())
    assert dlg._drop_entities["acq"].text() == "Acquisition", "the readable name is clicked"
    assert dlg._drop_entities["rec"].isChecked()
    dlg._drop_entities["acq"].setChecked(True)
    dlg._drop_entities["rec"].setChecked(False)
    dlg._on_save()
    assert AppSettings.load().scan_drop_entities == ["acq"]
    dlg._on_restore_defaults()
    assert not any(c.isChecked() for c in dlg._drop_entities.values())

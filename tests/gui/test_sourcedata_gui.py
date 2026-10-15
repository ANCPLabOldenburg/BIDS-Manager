"""Keeping the source data, from the GUI: the setting, and Run saying how
much it adds before anything is converted."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.gui.app_settings import AppSettings
from tests.fixtures.dicoms import write_mr_series

pytestmark = pytest.mark.gui


def test_the_setting_round_trips_and_is_off_by_default() -> None:
    s = AppSettings.load()
    assert s.convert_sourcedata == "none"
    s.convert_sourcedata = "archive"
    s.convert_sourcedata_modalities = "mri,eeg"
    s.save()
    back = AppSettings.load()
    assert back.convert_sourcedata == "archive"
    assert back.convert_sourcedata_modalities == "mri,eeg"


def test_the_settings_page_offers_it_with_the_warning(qtbot) -> None:
    from PyQt6.QtWidgets import QLabel

    from bidsmgr.converter.sourcedata import IDENTIFIERS_NOTE
    from bidsmgr.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    assert dlg._convert_sourcedata.currentData() == "none"
    assert not any(b.isEnabled() for b in dlg._sourcedata_boxes.values()), \
        "the modalities wait for the choice"
    assert any(lbl.text() == IDENTIFIERS_NOTE for lbl in dlg.findChildren(QLabel))
    dlg._convert_sourcedata.setCurrentIndex(dlg._convert_sourcedata.findData("copy"))
    assert all(b.isEnabled() for b in dlg._sourcedata_boxes.values())
    dlg._sourcedata_boxes["eeg"].setChecked(False)
    dlg._on_save()
    s = AppSettings.load()
    assert s.convert_sourcedata == "copy"
    assert "eeg" not in s.convert_sourcedata_modalities.split(",")
    assert "mri" in s.convert_sourcedata_modalities.split(",")


def _converter(qtbot, tmp_path: Path, uid: str):
    from bidsmgr.cli.create import open_or_create_workspace
    from bidsmgr.gui.converter_panel import ConverterPanel
    from bidsmgr.project import Project, ScanImported, workspace

    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir = workspace.allocate_version_dir(root, "raw")
    vproj = Project.create(vdir, name=root.name)
    df = pd.DataFrame([{
        "participant_id": "sub-001", "session": "", "include": 1, "modality": "mri",
        "datatype": "anat", "bids_name": "sub-001_T1w", "issues": "",
        "bids_guess_suffix": "T1w", "bids_guess_confidence": "0.9", "bids_guess_skip": False,
        "sequence": "t1", "entities": json.dumps({"subject": "001"}), "task": "", "run": "",
        "series_uid": uid, "dataset": "ds", "source_file": "", "source_folder": "t1"}])
    df.to_csv(workspace.version_inventory(vdir), sep="\t", index=False)
    workspace.write_version_meta(vdir, source_label="raw", raw_root=str(tmp_path / "raw"),
                                 status="curating")
    vproj.append(ScanImported(inventory_tsv=str(workspace.version_inventory(vdir)),
                              row_ids=(uid,), raw_root=None))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)
    return panel


def test_run_says_what_keeping_the_source_data_adds(qtbot, tmp_path, monkeypatch) -> None:
    from bidsmgr.inventory.sources import ScanSources

    files = write_mr_series(tmp_path / "raw" / "t1", "1.2.3.4", n_slices=3)
    panel = _converter(qtbot, tmp_path, "1.2.3.4")
    raw = tmp_path / "raw"
    panel._sources = ScanSources(raw, raw, {"1.2.3.4": [str(f) for f in files]})
    panel._bids_parent = tmp_path
    s = AppSettings.load()
    s.convert_sourcedata = "copy"
    s.save()
    asked, started = [], []
    monkeypatch.setattr(panel, "confirm_sourcedata", lambda text: asked.append(text) or True)
    monkeypatch.setattr(panel, "start_convert", lambda parent, **kw: started.append(parent))
    panel._on_run_clicked()
    assert not started, "nothing starts before the size is known"
    qtbot.waitUntil(lambda: bool(started), timeout=10000)
    assert "3 source file(s)" in asked[0] and "MRI (DICOM)" in asked[0]
    assert "not anonymised" in asked[0]


def test_run_is_refused_when_the_copies_will_not_fit(qtbot, tmp_path, monkeypatch) -> None:
    from bidsmgr.converter import sourcedata as sd
    from bidsmgr.gui import converter_panel as cp
    from bidsmgr.inventory.sources import ScanSources

    files = write_mr_series(tmp_path / "raw" / "t1", "1.2.3.4", n_slices=2)
    panel = _converter(qtbot, tmp_path, "1.2.3.4")
    raw = tmp_path / "raw"
    panel._sources = ScanSources(raw, raw, {"1.2.3.4": [str(f) for f in files]})
    panel._bids_parent = tmp_path
    s = AppSettings.load()
    s.convert_sourcedata = "archive"
    s.save()
    warned, started = [], []
    monkeypatch.setattr(sd, "space_problem", lambda root, needed: "No room.")
    monkeypatch.setattr(cp.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2]))
    monkeypatch.setattr(panel, "start_convert", lambda parent, **kw: started.append(parent))
    panel._on_run_clicked()
    qtbot.waitUntil(lambda: bool(warned), timeout=10000)
    assert not started and "No room." in warned[0]


def test_run_without_it_starts_at_once(qtbot, tmp_path, monkeypatch) -> None:
    panel = _converter(qtbot, tmp_path, "1.2.3.4")
    panel._bids_parent = tmp_path
    started = []
    monkeypatch.setattr(panel, "start_convert", lambda parent, **kw: started.append(parent))
    panel._on_run_clicked()
    assert started == [tmp_path]

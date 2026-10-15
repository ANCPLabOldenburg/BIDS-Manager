"""The Converter knows where each scan's source data is.

Selecting a scan table shows ITS source folder in the Raw input bar (not the
folder another scan read), says so when the data has moved, and lets the
user say where it went; the preview, the PSD and the conversion then read
from there.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pydicom")

from bidsmgr.cli.create import open_or_create_workspace  # noqa: E402
from bidsmgr.gui import converter_panel as cp  # noqa: E402
from bidsmgr.gui.converter_panel import ConverterPanel  # noqa: E402
from bidsmgr.inventory import sources  # noqa: E402
from bidsmgr.inventory.probe_convert import series_files  # noqa: E402
from bidsmgr.project import Project, ScanImported, workspace  # noqa: E402
from tests.fixtures.dicoms import write_mr_series  # noqa: E402

pytestmark = pytest.mark.gui


def _row(uid: str, folder: str, task: str) -> dict:
    return {
        "participant_id": "sub-001", "session": "", "include": 1, "modality": "mri",
        "datatype": "func", "bids_name": f"sub-001_task-{task}_bold",
        "bids_guess_suffix": "bold", "bids_guess_confidence": "0.9",
        "bids_guess_skip": False, "issues": "",
        "entities": json.dumps({"subject": "001", "task": task}, sort_keys=True),
        "task": task, "run": "", "series_uid": uid, "dataset": "ds",
        "source_file": "", "source_folder": folder,
    }


def _scan(bids_root: Path, raw: Path, uid: str, task: str, *, record: str | None = None):
    """A scan version of ``raw`` (one series in ``raw/s1``) as the scan
    writes it: inventory, files_by_uid and its record."""
    write_mr_series(raw / "s1", uid, description=task)
    vdir = workspace.allocate_version_dir(bids_root, raw.name)
    proj = Project.create(vdir, name=bids_root.name)
    inv = workspace.version_inventory(vdir)
    pd.DataFrame([_row(uid, "s1", task)]).to_csv(inv, sep="\t", index=False)
    with gzip.open(sources.files_by_uid_path(inv), "wb") as fh:
        fh.write(json.dumps({uid: series_files(raw, uid)}).encode("utf-8"))
    workspace.write_version_meta(vdir, source_label=raw.name,
                                 raw_root=record or str(raw), status="curating")
    proj.append(ScanImported(inventory_tsv=str(inv), row_ids=(uid,), raw_root=str(raw)))
    return vdir


def _open(qtbot, root: Path) -> ConverterPanel:
    proj = open_or_create_workspace(root)
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)
    return panel


def _select(panel: ConverterPanel, vdir: Path) -> None:
    combo = panel._scans_combo
    i = next(i for i in range(combo.count()) if Path(combo.itemData(i)) == vdir)
    combo.setCurrentIndex(i)
    panel._on_scans_combo_activated(i)


def test_each_scan_table_shows_its_own_source(qtbot, isolated_settings, tmp_path):
    root = tmp_path / "ds"
    open_or_create_workspace(root)
    first = _scan(root, tmp_path / "Shari_data", "1.2.3.1", "rest")
    second = _scan(root, tmp_path / "Shayam_data", "1.2.3.2", "nback")
    panel = _open(qtbot, root)
    assert panel._raw_pathbar.value() == str(tmp_path / "Shayam_data")
    _select(panel, first)
    assert panel._raw_pathbar.value() == str(tmp_path / "Shari_data")
    assert panel._properties._sources.root == tmp_path / "Shari_data"
    assert panel._raw_pathbar.chip_text() == ""
    _select(panel, second)
    assert panel._raw_pathbar.value() == str(tmp_path / "Shayam_data")


def test_a_record_naming_another_scan_s_folder_is_corrected(qtbot, isolated_settings,
                                                             tmp_path):
    root = tmp_path / "ds"
    open_or_create_workspace(root)
    vdir = _scan(root, tmp_path / "Shari_data", "1.2.3.1", "rest",
                 record=str(tmp_path / "Shayam_data"))
    (tmp_path / "Shayam_data").mkdir()
    logs: list[str] = []
    proj = open_or_create_workspace(root)
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.log_message.connect(logs.append)
    panel.set_project(proj, root)
    assert panel._raw_pathbar.value() == str(tmp_path / "Shari_data")
    assert workspace.read_version_meta(vdir)["raw_root"] == str(tmp_path / "Shari_data")
    assert any("record is corrected" in m for m in logs)


def test_moved_data_is_said_and_can_be_located(qtbot, isolated_settings, tmp_path,
                                               monkeypatch):
    root = tmp_path / "ds"
    open_or_create_workspace(root)
    raw, moved = tmp_path / "raw", tmp_path / "drive" / "raw"
    vdir = _scan(root, raw, "1.2.3.1", "rest")
    moved.parent.mkdir()
    shutil.move(str(raw), str(moved))

    panel = _open(qtbot, root)
    bar = panel._raw_pathbar
    assert bar.chip_text() == "Moved" and bar.action_button.text() == "Locate..."
    assert str(raw) in bar.value()
    panel._table.selectRow(0)
    panel._properties.flush()          # a selection renders once it settles
    assert not panel._properties.preview_button.isEnabled()
    assert "Locate" in panel._properties.preview_button.toolTip()

    # Convert refuses rather than failing per row.
    warned: list[str] = []
    monkeypatch.setattr(cp.QMessageBox, "warning",
                        lambda _p, title, text, *a, **k: warned.append(text))
    panel._bids_parent = tmp_path
    panel._on_run_clicked()
    assert warned and "no longer at" in warned[0] and panel._convert_worker is None

    # A folder without this scan's data is refused.
    elsewhere = tmp_path / "elsewhere"
    write_mr_series(elsewhere / "s1", "9.9.9.9", description="rest")
    monkeypatch.setattr(cp.QFileDialog, "getExistingDirectory",
                        lambda *a, **k: str(elsewhere))
    bar.action_button.click()
    assert len(warned) == 2 and "None of the" in warned[1]
    assert workspace.read_version_meta(vdir)["raw_root"] == str(raw)

    # Where it went: recorded, and everything reads from there.
    monkeypatch.setattr(cp.QFileDialog, "getExistingDirectory", lambda *a, **k: str(moved))
    bar.action_button.click()
    assert workspace.read_version_meta(vdir)["raw_root"] == str(moved)
    assert bar.chip_text() == "" and bar.value() == str(moved)
    assert panel._properties.preview_button.isEnabled()
    files = panel._sources.series_files("1.2.3.1")
    assert files and all(moved in f.parents for f in files)


def test_picking_the_moved_folder_to_scan_offers_to_relink(qtbot, isolated_settings,
                                                           tmp_path, monkeypatch):
    root = tmp_path / "ds"
    open_or_create_workspace(root)
    raw, moved = tmp_path / "raw", tmp_path / "raw_moved"
    vdir = _scan(root, raw, "1.2.3.1", "rest")
    shutil.move(str(raw), str(moved))
    panel = _open(qtbot, root)
    monkeypatch.setattr(cp.QFileDialog, "getExistingDirectory", lambda *a, **k: str(moved))
    monkeypatch.setattr(cp.QMessageBox, "question",
                        lambda *a, **k: cp.QMessageBox.StandardButton.Yes)
    panel._on_pick_raw_dir()
    assert workspace.read_version_meta(vdir)["raw_root"] == str(moved)
    assert panel._raw_pathbar.chip_text() == ""


def test_data_coming_back_is_noticed(qtbot, isolated_settings, tmp_path):
    """The drive is plugged back in while the app was in the background."""
    from PyQt6.QtCore import Qt

    root = tmp_path / "ds"
    open_or_create_workspace(root)
    raw, away = tmp_path / "raw", tmp_path / "away"
    _scan(root, raw, "1.2.3.1", "rest")
    shutil.move(str(raw), str(away))
    panel = _open(qtbot, root)
    assert panel._raw_pathbar.chip_text() == "Moved"
    assert "is not there" in panel._raw_pane._empty.text()
    shutil.move(str(away), str(raw))
    panel._on_application_state(Qt.ApplicationState.ApplicationActive)
    qtbot.waitUntil(lambda: panel._raw_pathbar.chip_text() == "", timeout=10_000)
    assert panel._source_check.state == "ok"
    assert panel._raw_pane._tree.isVisibleTo(panel._raw_pane), "the tree is back"

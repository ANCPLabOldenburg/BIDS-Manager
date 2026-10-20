"""Tools, Scans and sessions tables: a preview of every table, then one
undoable write."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from bidsmgr.gui.tables_dialog import TablesDialog
from bidsmgr.metadata.tables import read_table
from bidsmgr.project.operations import peek_undo, undo_last

pytestmark = pytest.mark.gui


def _write(path: Path, content=b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        path.write_text(json.dumps(content))
    elif isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    _write(root / "dataset_description.json", {"Name": "d", "BIDSVersion": "1.10.0"})
    for ses, stamp in (("pre", "2025-01-01T09:00:00"), ("post", "2025-02-01T09:00:00")):
        base = root / "sub-01" / f"ses-{ses}"
        _write(base / "anat" / f"sub-01_ses-{ses}_T1w.nii.gz")
        _write(base / "anat" / f"sub-01_ses-{ses}_T1w.json", {"AcquisitionDateTime": stamp})
        _write(base / "beh" / f"sub-01_ses-{ses}_task-quiz_beh.tsv", "trial\n1\n")
    _write(root / "sub-02" / "dwi" / "sub-02_dwi.nii.gz")
    _write(root / "sub-02" / "dwi" / "sub-02_dwi.bval", "0 1000\n")
    return root


def _open(qtbot, root: Path) -> TablesDialog:
    dlg = TablesDialog(root)
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: not dlg.planning(), timeout=20_000)
    return dlg


def _tables(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*_s*s.tsv")
                  if ".bidsmgr" not in p.parts)


def test_every_table_is_shown_with_its_rows(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    heads = {dlg._tree.topLevelItem(i).text(0): dlg._tree.topLevelItem(i)
             for i in range(dlg._tree.topLevelItemCount())}
    assert sorted(heads) == ["sub-01/ses-post/sub-01_ses-post_scans.tsv",
                             "sub-01/ses-pre/sub-01_ses-pre_scans.tsv",
                             "sub-01/sub-01_sessions.tsv", "sub-02/sub-02_scans.tsv"]
    pre = heads["sub-01/ses-pre/sub-01_ses-pre_scans.tsv"]
    rows = {pre.child(i).text(0): pre.child(i).text(1) for i in range(pre.childCount())}
    assert rows == {"anat/sub-01_ses-pre_T1w.nii.gz": "added, acquired 2025-01-01T09:00:00",
                    "beh/sub-01_ses-pre_task-quiz_beh.tsv": "added, time not known (n/a)"}
    dwi = heads["sub-02/sub-02_scans.tsv"]
    assert [dwi.child(i).text(0) for i in range(dwi.childCount())] == [
        "dwi/sub-02_dwi.nii.gz"], "no row for the .bval"
    assert dlg._ok.text() == "Write 4 tables"


def test_writing_is_one_undoable_step(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    dlg._on_write()
    assert len(_tables(ds)) == 4 and len(dlg.written()) == 4
    assert peek_undo(ds).label == "Write 4 scans and sessions tables"
    undo_last(ds)
    assert _tables(ds) == []


def test_an_unticked_table_is_not_written(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    for i in range(dlg._tree.topLevelItemCount()):
        item = dlg._tree.topLevelItem(i)
        if item.text(0).endswith("_sessions.tsv"):
            item.setCheckState(0, Qt.CheckState.Unchecked)
    assert dlg._ok.text() == "Write 3 tables"
    dlg._on_write()
    assert "sub-01/sub-01_sessions.tsv" not in _tables(ds)


def test_choosing_scans_only_plans_again(qtbot, ds) -> None:
    dlg = _open(qtbot, ds)
    dlg._sessions.setChecked(False)
    qtbot.waitUntil(lambda: not dlg.planning(), timeout=20_000)
    names = [dlg._tree.topLevelItem(i).text(0) for i in range(dlg._tree.topLevelItemCount())]
    assert not any(n.endswith("_sessions.tsv") for n in names) and len(names) == 3


def test_tables_already_right_say_so(qtbot, ds) -> None:
    _open(qtbot, ds)._on_write()
    dlg = _open(qtbot, ds)
    assert dlg._tree.topLevelItemCount() == 0
    assert "Every table is as it should be" in dlg._status.text()
    assert not dlg._ok.isEnabled()


def test_a_table_there_keeps_its_rows_and_columns(qtbot, ds) -> None:
    table = ds / "sub-02" / "sub-02_scans.tsv"
    table.write_text("filename\tacq_time\tnote\ndwi/sub-02_dwi.bval\tn/a\tkeep me\n")
    dlg = _open(qtbot, ds)
    item = next(dlg._tree.topLevelItem(i) for i in range(dlg._tree.topLevelItemCount())
                if dlg._tree.topLevelItem(i).text(0) == "sub-02/sub-02_scans.tsv")
    changes = {item.child(i).text(0): item.child(i).text(1) for i in range(item.childCount())}
    assert changes["dwi/sub-02_dwi.bval"] == "removed: part of a recording listed on its own row"
    dlg._on_write()
    assert read_table(table)[0] == ["filename", "acq_time", "note"]


def test_the_plan_is_made_off_the_window(qtbot, ds, monkeypatch) -> None:
    import bidsmgr.gui.tables_dialog as mod

    seen = {}
    real = mod.plan_for

    def spy(*args):
        seen["thread"] = threading.current_thread()
        return real(*args)

    monkeypatch.setattr(mod, "plan_for", spy)
    _open(qtbot, ds)
    assert seen["thread"] is not threading.main_thread()


def test_the_editor_offers_it(qtbot) -> None:
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    labels = [a.text() for a in panel._tools_menu.actions()]
    assert labels.index("Scans and sessions tables...") == labels.index("Check coherence...") + 1

"""The "Write TODO into missing fields" setting, from the switch to the files.

Three things were wrong and are pinned here. The chain passed the scope even
with the box unticked, and the engine read a scope as "on", so placeholders
were written with marking off. A narrower scope did nothing to a dataset an
earlier, wider run had marked. And the Editor's form showed every key as
optional until a validation run, so a TODO in a recommended field read as a
TODO in an optional one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.gui.app_settings import AppSettings

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "Study"
    for datatype, stem, ext in (
        ("anat", "sub-01_T1w", ".nii.gz"),
        ("eeg", "sub-01_task-rest_eeg", ".edf"),
    ):
        folder = root / "sub-01" / datatype
        folder.mkdir(parents=True)
        (folder / f"{stem}{ext}").write_bytes(b"\0" * 16)
        (folder / f"{stem}.json").write_text(
            json.dumps({"TaskName": "rest"} if "task" in stem else {}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Study", "BIDSVersion": "1.10.0"}))
    return root


def _todos(root: Path) -> set[str]:
    out = set()
    for path in root.rglob("*.json"):
        if ".bidsmgr" in path.parts:
            continue
        for key, value in json.loads(path.read_text()).items():
            if value in ("TODO", ["TODO"]):
                out.add(f"{path.relative_to(root)}:{key}")
    return out


def test_the_checkbox_and_scope_amount_to_one_value() -> None:
    s = AppSettings()
    s.metadata_fill_scope = "optional"
    s.post_metadata_fill_todos = True
    assert s.placeholder_scope() == "optional"
    # Off is "none", whatever the scope beside it says. Reading the scope
    # alone is how the chain marked fields with the box unticked.
    s.post_metadata_fill_todos = False
    assert s.placeholder_scope() == "none"


def test_the_chain_writes_no_todo_with_the_box_unticked(
    qtbot, dataset: Path, monkeypatch,
) -> None:
    from bidsmgr.gui import converter_panel as cp

    seen: dict = {}

    class _Recorder:
        def __init__(self, *args, **kwargs) -> None:
            seen.update(kwargs)
            self.progress = self.finished_with_result = self.failed = self

        def connect(self, *_args) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(cp, "MetadataWorker", _Recorder)
    panel = cp.ConverterPanel()
    qtbot.addWidget(panel)
    panel._app_settings.post_metadata_fill_todos = False
    panel._app_settings.metadata_fill_scope = "optional"
    panel._start_metadata_worker(dataset.parent)
    assert seen["fill_scope"] == "none"

    # And what "none" does to a dataset an earlier run marked.
    from bidsmgr.cli.metadata import run_metadata_cli

    run_metadata_cli(dataset.parent, fill_scope="optional")
    assert _todos(dataset)
    run_metadata_cli(dataset.parent, fill_scope=seen["fill_scope"])
    assert _todos(dataset) == set()


def test_the_settings_say_the_setting_is_about_todo(qtbot) -> None:
    from bidsmgr.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppSettings())
    qtbot.addWidget(dlg)
    box = dlg._post_metadata_fill_todos
    assert "TODO" in box.text()
    # What turning it off and narrowing it do, said where the user decides.
    assert "takes them all back" in box.toolTip()
    labels = [dlg._metadata_fill_scope.itemText(i)
              for i in range(dlg._metadata_fill_scope.count())]
    assert labels[0] == "Required fields only"


def test_fix_ups_takes_back_what_a_wider_run_wrote(
    qtbot, dataset: Path, monkeypatch,
) -> None:
    from PyQt6.QtWidgets import QMessageBox

    from bidsmgr.gui.fixups_dialog import FixupsDialog
    from bidsmgr.metadata.engine import run_metadata

    run_metadata(dataset, fill_scope="optional", write_report=False)
    wide = _todos(dataset)

    asked: list[str] = []
    told: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: (asked.append(a[2]), QMessageBox.StandardButton.Ok)[1])
    monkeypatch.setattr(
        QMessageBox, "information", lambda *a, **k: told.append(a[2]))

    dlg = FixupsDialog(dataset)
    qtbot.addWidget(dlg)
    dlg._todo_scope.setCurrentIndex(dlg._todo_scope.findData("required"))
    dlg._on_stamp_todos()

    assert "required field" in asked[0] and "recommended" not in asked[0]
    assert "taken back" in told[0]
    assert _todos(dataset) < wide
    # Persisted, so the post-convert chain agrees with what was just done.
    assert AppSettings.load().metadata_fill_scope == "required"


def test_a_dataset_change_rereads_what_the_levels_depend_on(
    qtbot, dataset: Path, monkeypatch,
) -> None:
    """Authors is recommended only while there is no CITATION.cff: the Editor
    must not keep the level it read before one was written."""
    from bidsmgr import schema
    from bidsmgr.gui.editor_panel import EditorPanel

    forgotten: list[int] = []
    monkeypatch.setattr(schema, "forget_dataset", lambda: forgotten.append(1))
    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._set_root(dataset, persist=False)
    assert forgotten
    count = len(forgotten)
    panel.tree_pane().refresh()
    assert len(forgotten) > count


def test_the_form_levels_fields_before_any_validation(qtbot, dataset: Path) -> None:
    """A TODO written into a recommended field reads as recommended, and a
    T1w with no task is not offered the task fields."""
    from bidsmgr.gui.widgets.sidecar_form_pane import SidecarFormPane
    from bidsmgr.metadata.engine import run_metadata

    run_metadata(dataset, fill_scope="recommended", write_report=False)
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_file(dataset / "sub-01" / "anat" / "sub-01_T1w.json", dataset, None)
    rows = {row._key: row for row in pane._rows}
    todo = [row for row in rows.values() if row._value_kind == "todo"]
    assert todo, "the recommended scope marks a T1w"
    assert {row._level for row in todo} <= {"req", "rec"}
    assert "TaskName" not in rows

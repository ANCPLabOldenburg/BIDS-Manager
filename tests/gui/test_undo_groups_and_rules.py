"""One user action is one Undo, and saved scan rules act on the open table.

From the 0.x feedback of the Neuroimaging unit: a misplaced click on the
study's checkbox selected everything and there was no way back (Undo reverted
one row per click), and a series added to "always exclude" changed nothing
until the data was scanned again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtCore import Qt

from bidsmgr.classifier import user_rules as ur
from bidsmgr.cli.create import open_or_create_workspace
from bidsmgr.gui import converter_panel as cp
from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.project import Project, ScanImported, UserToggleInclude, workspace

pytestmark = pytest.mark.gui


def _row(uid: str, sequence: str, task: str, *, dataset: str = "ds", include: int = 1) -> dict:
    return {
        "participant_id": "sub-001", "session": "", "include": include, "modality": "mri",
        "datatype": "func", "bids_name": f"sub-001_task-{task}_bold",
        "bids_guess_suffix": "bold", "bids_guess_confidence": "0.9",
        "bids_guess_skip": False, "issues": "", "sequence": sequence,
        "entities": json.dumps({"subject": "001", "task": task}, sort_keys=True),
        "task": task, "run": "", "series_uid": uid, "dataset": dataset,
        "source_file": "", "source_folder": "s1",
    }


def _panel(qtbot, tmp_path: Path, rows: list[dict]) -> ConverterPanel:
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir = workspace.allocate_version_dir(root, "raw")
    vproj = Project.create(vdir, name=root.name)
    df = pd.DataFrame(rows)
    df.to_csv(workspace.version_inventory(vdir), sep="\t", index=False)
    workspace.write_version_meta(vdir, source_label="raw", raw_root=str(tmp_path / "raw"),
                                 status="curating")
    vproj.append(ScanImported(inventory_tsv=str(workspace.version_inventory(vdir)),
                              row_ids=tuple(df["series_uid"]), raw_root=None))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)
    return panel


def _includes(panel: ConverterPanel) -> list[bool]:
    return [panel._model._read_include(r) for r in range(panel._model.rowCount())]


# ---------------------------------------------------------------------------
# The event log
# ---------------------------------------------------------------------------


def test_undo_takes_back_a_whole_group(tmp_path) -> None:
    proj = Project.create(tmp_path / "p", name="p")
    proj.append(UserToggleInclude(row_id="a", include=False))
    for rid in ("b", "c", "d"):
        proj.append(UserToggleInclude(row_id=rid, include=False, group="g1"))
    popped = proj.undo_last_user_edit()
    assert [e.row_id for e in popped] == ["b", "c", "d"], "oldest first, the whole group"
    assert [e.row_id for e in proj.undo_last_user_edit()] == ["a"]
    assert proj.undo_last_user_edit() == []


def test_a_log_written_before_groups_still_reads(tmp_path) -> None:
    proj = Project.create(tmp_path / "p", name="p")
    record = {"v": 1, "ts": "2026-01-01T00:00:00.000Z", "type": "user_toggle_include",
              "row_id": "a", "include": False}
    with proj.log.path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    events = list(proj.log)
    assert events[-1].group is None


# ---------------------------------------------------------------------------
# One click, one Undo
# ---------------------------------------------------------------------------


def test_a_folder_checkbox_is_one_undo(qtbot, isolated_settings, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, [_row(f"1.{i}", f"bold {i}", f"t{i}") for i in range(5)])
    tree = panel._filter_pane._tree
    root = tree.topLevelItem(0)
    subject = root.child(0)
    subject.setCheckState(0, Qt.CheckState.Unchecked)
    qtbot.wait(20)                      # the group closes when the event loop turns
    assert _includes(panel) == [False] * 5
    panel._on_undo()
    assert _includes(panel) == [True] * 5, "one Undo brought every row back"
    panel._on_redo()
    assert _includes(panel) == [False] * 5, "and one Redo took them out again"


def test_a_lone_dataset_has_no_checkbox(qtbot, isolated_settings, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, [_row("1.1", "bold", "rest")])
    root = panel._filter_pane._tree.topLevelItem(0)
    assert not root.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert root.child(0).flags() & Qt.ItemFlag.ItemIsUserCheckable, "subjects keep theirs"


def test_two_datasets_keep_their_checkboxes(qtbot, isolated_settings, tmp_path) -> None:
    """Several datasets in one table happen outside a project (a project's
    table is always its own dataset)."""
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.load_inventory(pd.DataFrame([_row("1.1", "bold", "rest"),
                                       _row("1.2", "bold", "rest", dataset="other")]),
                         output_tsv=tmp_path / "inv.tsv")
    tree = panel._filter_pane._tree
    assert tree.topLevelItemCount() == 2
    assert all(tree.topLevelItem(i).flags() & Qt.ItemFlag.ItemIsUserCheckable
               for i in range(tree.topLevelItemCount()))


def test_ctrl_z_undoes_in_the_converter(qtbot, isolated_settings, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, [_row("1.1", "bold", "rest")])
    panel.show()
    panel.activateWindow()
    qtbot.waitActive(panel)              # shortcuts fire in the active window
    panel._model.setData(panel._model.index(0, 0), False)
    assert _includes(panel) == [False]
    panel._table.setFocus()
    qtbot.wait(10)                       # let the focus settle before the key
    qtbot.keyClick(panel._table, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert _includes(panel) == [True]


# ---------------------------------------------------------------------------
# Scan rules on the open table
# ---------------------------------------------------------------------------


def test_the_plan_never_re_includes_a_row_unticked_by_hand() -> None:
    df = pd.DataFrame([
        _row("1", "localizer", "a"),
        _row("2", "bold", "b", include=0),                       # unticked by hand
        {**_row("3", "old_scout", "c", include=0),
         "issues": "user-excluded: matched the scan exclusion rule 'old_scout'"},
    ])
    rules = [ur.ExclusionRule(pattern="localizer", target="sequence", match_mode="substring")]
    plan = ur.plan_exclusions(df, rules)
    assert [i for i, _ in plan.exclude] == [0]
    assert plan.restore == (2,), "the rule that excluded it is gone"


def test_saved_rules_act_on_the_open_table(qtbot, isolated_settings, tmp_path,
                                           monkeypatch) -> None:
    panel = _panel(qtbot, tmp_path, [_row("1.1", "AAHead_Scout", "a"),
                                     _row("1.2", "bold", "rest"),
                                     _row("1.3", "AAHead_Scout_MPR", "b")])
    asked: list[str] = []

    def answer(_parent, _title, text, *a, **k):
        asked.append(text)
        return cp.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(cp.QMessageBox, "question", answer)
    settings = AppSettings.load()
    settings.scan_exclusions = [{"pattern": "Scout", "target": "sequence",
                                 "match_mode": "substring"}]
    settings.save()
    panel.reload_app_settings()          # what the window does after Settings > Save
    assert asked and "2 series would be excluded" in asked[0]
    assert _includes(panel) == [False, True, False]
    df = panel._model.dataframe()
    assert ur.USER_EXCLUDED_ISSUE_TOKEN in df.at[0, "issues"]
    # One Undo, both rows and their notes.
    panel._on_undo()
    assert _includes(panel) == [True, True, True]
    assert ur.USER_EXCLUDED_ISSUE_TOKEN not in str(panel._model.dataframe().at[0, "issues"])


def test_declining_leaves_the_table_for_the_next_scan(qtbot, isolated_settings, tmp_path,
                                                      monkeypatch) -> None:
    panel = _panel(qtbot, tmp_path, [_row("1.1", "AAHead_Scout", "a")])
    monkeypatch.setattr(cp.QMessageBox, "question",
                        lambda *a, **k: cp.QMessageBox.StandardButton.No)
    settings = AppSettings.load()
    settings.scan_exclusions = [{"pattern": "Scout", "target": "sequence",
                                 "match_mode": "substring"}]
    settings.save()
    panel.reload_app_settings()
    assert _includes(panel) == [True]


def test_removing_a_rule_includes_its_rows_again(qtbot, isolated_settings, tmp_path,
                                                 monkeypatch) -> None:
    panel = _panel(qtbot, tmp_path, [_row("1.1", "AAHead_Scout", "a")])
    monkeypatch.setattr(cp.QMessageBox, "question",
                        lambda *a, **k: cp.QMessageBox.StandardButton.Yes)
    settings = AppSettings.load()
    settings.scan_exclusions = [{"pattern": "Scout", "target": "sequence",
                                 "match_mode": "substring"}]
    settings.save()
    panel.reload_app_settings()
    assert _includes(panel) == [False]
    settings.scan_exclusions = []
    settings.save()
    panel.reload_app_settings()
    assert _includes(panel) == [True]


# ---------------------------------------------------------------------------
# The runs a fieldmap corrects, before converting
# ---------------------------------------------------------------------------


def _fmap_rows() -> list[dict]:
    def row(uid, dt, suffix, time):
        r = _row(uid, "seq " + suffix, "rest")
        r.update({"datatype": dt, "bids_guess_suffix": suffix, "acq_time": time,
                  "bids_name": f"sub-001_acq-{uid}_{suffix}", "fieldmap_targets": ""})
        return r
    return [row("f1", "fmap", "phasediff", "10:00:00"), row("b1", "func", "bold", "10:05:00"),
            row("f2", "fmap", "phasediff", "11:00:00"), row("b2", "func", "bold", "11:05:00")]


def test_a_fieldmap_shows_and_takes_its_runs(qtbot, isolated_settings, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, _fmap_rows())
    props = panel._properties
    props.set_selected_row(0)
    checks = props.fieldmap_checks
    assert {panel._model.row_id(r): c.isChecked() for r, c in checks.items()} == \
        {"b1": True, "b2": False}, "by time: the run after it, not the next fieldmap's"
    checks[3].setChecked(True)                    # the user adds the later run
    qtbot.wait(20)
    assert json.loads(panel._model.dataframe().at[0, "fieldmap_targets"]) == ["b1", "b2"]
    props.set_selected_row(0)
    assert hasattr(props, "fieldmap_reset")
    props.fieldmap_reset.click()
    qtbot.wait(20)
    assert panel._model.dataframe().at[0, "fieldmap_targets"] == ""
    # And the choice was one undoable edit.
    props.set_selected_row(0)
    props.fieldmap_checks[3].setChecked(True)
    qtbot.wait(20)
    panel._on_undo()
    assert panel._model.dataframe().at[0, "fieldmap_targets"] == ""


def test_a_choice_covers_the_whole_fieldmap_acquisition(qtbot, isolated_settings,
                                                      tmp_path) -> None:
    """A fieldmap written as two rows (a B0 map and its magnitude): ticking a
    run on one row is the choice for both, one Undo, and the other row says
    so and can reset it."""
    rows = _fmap_rows()
    rows[0]["bids_guess_suffix"] = "fieldmap"
    magnitude = dict(rows[0], series_uid="f1m", bids_guess_suffix="magnitude")
    panel = _panel(qtbot, tmp_path, [rows[0], magnitude, *rows[1:]])
    model = panel._model
    assert model.fieldmap_rows(0) == [0, 1] and model.fieldmap_rows(3) == [3]
    props = panel._properties
    props.set_selected_row(0)
    props.fieldmap_checks[4].setChecked(True)     # the later run, on the B0 map row
    qtbot.wait(20)
    df = model.dataframe()
    assert df.at[0, "fieldmap_targets"] == df.at[1, "fieldmap_targets"] != ""
    props.set_selected_row(1)
    assert hasattr(props, "fieldmap_reset"), "the magnitude row shows the same choice"
    assert props.fieldmap_checks[4].isChecked()
    panel._on_undo()
    df = panel._model.dataframe()
    assert df.at[0, "fieldmap_targets"] == df.at[1, "fieldmap_targets"] == "", "one Undo"


# ---------------------------------------------------------------------------
# New subject numbers
# ---------------------------------------------------------------------------


def _people_rows() -> list[dict]:
    rows = []
    for uid, pid, patient in (("1.1", "sub-001", "P-A"), ("1.2", "sub-001", "P-A"),
                              ("1.3", "sub-002", "P-B")):
        r = _row(uid, f"bold {uid}", f"t{uid[-1]}")
        r.update({"participant_id": pid, "PatientID": patient,
                  "issues": "existing-subject: sub-001 already exists in the dataset"})
        rows.append(r)
    return rows


def test_new_subject_numbers_in_one_step(qtbot, isolated_settings, tmp_path,
                                         monkeypatch) -> None:
    from bidsmgr.gui import renumber_dialog as rd

    panel = _panel(qtbot, tmp_path, _people_rows())
    (tmp_path / "ds" / "sub-001").mkdir()             # already converted people
    (tmp_path / "ds" / "sub-002").mkdir()
    monkeypatch.setattr(rd.RenumberDialog, "exec", lambda self: rd.RenumberDialog.DialogCode.Accepted)
    changed = panel.open_renumber([0, 1, 2])
    assert changed == 3
    ids = list(panel._model.dataframe()["participant_id"])
    assert ids[0] == ids[1] and ids[0] != ids[2], "a person keeps one number"
    assert set(ids) == {"sub-003", "sub-004"}
    panel._on_undo()
    assert list(panel._model.dataframe()["participant_id"]) == ["sub-001", "sub-001", "sub-002"]


def test_the_warning_offers_the_fix(qtbot, isolated_settings, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, _people_rows())
    asked: list = []
    panel._properties.renumber_requested.disconnect()
    panel._properties.renumber_requested.connect(asked.append)
    panel._properties.set_selected_row(0)
    panel._properties.renumber_button.click()
    assert asked == [[0, 1]], "the person's every series"


def test_the_dialog_checks_what_is_typed(qtbot) -> None:
    from bidsmgr.gui.renumber_dialog import RenumberDialog
    from bidsmgr.inventory.renumber import Person

    dlg = RenumberDialog([Person("P-A", "001", "003", (0,)), Person("P-B", "002", "004", (1,))],
                         taken={"001", "002"})
    qtbot.addWidget(dlg)
    assert dlg.apply_button.isEnabled() and not dlg.problem.text()
    dlg.table.topLevelItem(1).setText(4, "003")
    assert not dlg.apply_button.isEnabled() and "same number" in dlg.problem.text()
    dlg.table.topLevelItem(1).setText(4, "a-b")
    assert not dlg.apply_button.isEnabled() and "letters and digits" in dlg.problem.text()
    dlg.table.topLevelItem(1).setText(4, "002")
    assert dlg.apply_button.isEnabled() and "merged" in dlg.problem.text(), "allowed, said"


def test_rename_a_subject_and_a_session_from_the_tree(qtbot, isolated_settings,
                                                      tmp_path) -> None:
    rows = _people_rows()
    for r in rows:
        r["session"] = "ses-pre"
        ents = json.loads(r["entities"])
        r["entities"] = json.dumps({**ents, "session": "pre"}, sort_keys=True)
    panel = _panel(qtbot, tmp_path, rows)
    sub_node = panel._filter_pane._tree.topLevelItem(0).child(0)
    assert panel._filter_pane._rows_under(sub_node)
    assert panel.rename_from_tree("subject", [0, 1], "sub-001", new="077") == 2
    assert list(panel._model.dataframe()["participant_id"])[:2] == ["sub-077", "sub-077"]
    assert panel.rename_from_tree("session", [0, 1, 2], "ses-pre", new="baseline") == 3
    assert set(panel._model.dataframe()["session"]) == {"ses-baseline"}
    panel._on_undo()
    assert set(panel._model.dataframe()["session"]) == {"ses-pre"}, "one Undo"


def test_the_table_menu_offers_new_numbers(qtbot, isolated_settings, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path, _people_rows())
    panel.show()
    qtbot.waitExposed(panel)
    panel._table.clearSelection()
    panel._table.selectRow(2)
    pos = panel._table.visualRect(panel._model.index(2, 1)).center()
    panel._on_table_menu(pos)
    texts = [a.text() for a in panel._table_menu.actions()]
    assert "Give a new subject number..." in texts
    panel._table_menu.close()

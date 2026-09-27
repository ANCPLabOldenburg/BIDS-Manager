"""The three Editor tools added for links, padding and coherence.

Each is a thin dialog over a Qt-free engine that has its own unit tests, so
what is checked here is the wiring: that the dialog offers what the engine
found, that ticking and applying reaches the engine, and that the three
actions exist on the Tools menu.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QMessageBox

from bidsmgr.gui.coherence_dialog import CoherenceDialog
from bidsmgr.gui.linkage_dialog import LinkageDialog
from bidsmgr.gui.pad_values_dialog import PadValuesDialog

pytestmark = pytest.mark.gui


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """Two fieldmaps and three runs, timed so the rule has something to say."""
    root = tmp_path / "ds"
    (root / "sub-001/fmap").mkdir(parents=True)
    (root / "sub-001/func").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.10.0"})
    )
    (root / "participants.tsv").write_text("participant_id\nsub-001\n")

    def write(rel: str, meta: dict) -> None:
        path = root / rel
        path.write_bytes(b"x")
        path.with_name(path.name.split(".")[0] + ".json").write_text(
            json.dumps(meta)
        )

    for run, time in ((1, "09:33:59"), (2, "09:57:39"), (3, "10:17:1")):
        write(f"sub-001/func/sub-001_task-x_run-{run}_bold.nii.gz",
              {"AcquisitionTime": time})
    for fm, time in ((1, "09:31:31"), (2, "09:54:53")):
        write(f"sub-001/fmap/sub-001_run-{fm}_phasediff.nii.gz",
              {"AcquisitionTime": time})
    (root / "sub-001/sub-001_scans.tsv").write_text(
        "filename\n"
        + "".join(
            f"func/sub-001_task-x_run-{r}_bold.nii.gz\n" for r in (1, 2, 3)
        )
        + "".join(
            f"fmap/sub-001_run-{f}_phasediff.nii.gz\n" for f in (1, 2)
        )
    )
    return root


def _tree_rows(tree) -> list[tuple[str, str]]:
    """(now, becomes) for every leaf row of a two-column preview."""
    out = []
    for i in range(tree.topLevelItemCount()):
        head = tree.topLevelItem(i)
        for j in range(head.childCount()):
            child = head.child(j)
            out.append((child.text(0), child.text(1)))
    return out


def _fmap(root: Path, n: int) -> Path:
    return root / f"sub-001/fmap/sub-001_run-{n}_phasediff.json"


class TestReferencesDialog:
    """Sources on the left, targets on the right, the verbs in between.

    Three earlier versions drew one vertical tree and asked the reader to
    hold the relationship in their head. What is checked here is that both
    lists hold what the STANDARD allows, that ticking reaches the engine,
    and that a selection of several files is edited as one.
    """

    def _left_rows(self, dlg):
        return [
            (dlg._left.topLevelItem(i).text(0),
             dlg._left.topLevelItem(i).text(1),
             dlg._left.topLevelItem(i).text(2))
            for i in range(dlg._left.topLevelItemCount())
        ]

    def _right_rows(self, dlg):
        return [
            (dlg._right.topLevelItem(i).text(0),
             dlg._right.topLevelItem(i).text(1))
            for i in range(dlg._right.topLevelItemCount())
        ]

    def _tick(self, dlg, needle: str, on: bool = True) -> None:
        for i in range(dlg._right.topLevelItemCount()):
            item = dlg._right.topLevelItem(i)
            if needle in item.text(0):
                item.setCheckState(
                    0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
                )
                return
        raise AssertionError(f"{needle} is not a candidate: {self._right_rows(dlg)}")

    def test_it_opens_on_the_whole_dataset(self, qtbot, dataset):
        """No file has to be selected first."""
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        assert dlg._left.topLevelItemCount() == 2       # the two fieldmaps

    def test_the_left_list_holds_only_files_that_may_carry_the_field(
        self, qtbot, dataset,
    ):
        """IntendedFor belongs on a fieldmap, so no bold run is offered."""
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        names = [r[0] for r in self._left_rows(dlg)]
        assert all("fmap/" in n for n in names)
        assert not any("_bold" in n for n in names)

    def test_the_right_list_holds_only_legal_targets(self, qtbot, dataset):
        """And it is the runs, in the same subject, not the other fieldmap."""
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        names = [r[0] for r in self._right_rows(dlg)]
        assert names and all("func/" in n for n in names)
        assert not any("phasediff" in n for n in names)

    def test_a_file_with_nothing_set_says_so(self, qtbot, dataset):
        """The fixture's fieldmaps carry no IntendedFor at all."""
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        assert any(r[2] == "the times imply one" for r in self._left_rows(dlg))

    def test_a_missing_target_is_flagged(self, qtbot, dataset):
        from bidsmgr.editor import linkage

        linkage.apply_links(dataset, [(
            _fmap(dataset, 1), "IntendedFor", ["bids::sub-001/func/gone.nii.gz"],
        )])
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        assert any(
            r[2] == "points at a missing file" for r in self._left_rows(dlg)
        )

    def test_the_right_list_answers_the_reverse_question(self, qtbot, dataset):
        """What already points at each candidate: is this run corrected?"""
        from bidsmgr.editor import linkage

        linkage.apply_links(dataset, [(
            _fmap(dataset, 1), "IntendedFor",
            ["bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"],
        )])
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        by = {r[0]: r[1] for r in self._right_rows(dlg)}
        assert any("phasediff" in v for v in by.values())
        assert "nothing" in by.values()

    def test_nothing_is_saveable_until_something_is_edited(self, qtbot, dataset):
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        assert not dlg._save_btn.isEnabled()

    def test_ticking_a_target_then_saving_writes_a_bids_uri(
        self, qtbot, dataset,
    ):
        dlg = LinkageDialog(dataset, _fmap(dataset, 1))
        qtbot.addWidget(dlg)
        self._tick(dlg, "run-2_bold")
        assert dlg._save_btn.isEnabled()
        dlg._on_save()
        assert json.loads(_fmap(dataset, 1).read_text())["IntendedFor"] == [
            "bids::sub-001/func/sub-001_task-x_run-2_bold.nii.gz"]

    def test_what_the_times_imply_fills_it_in(self, qtbot, dataset):
        dlg = LinkageDialog(dataset, _fmap(dataset, 1))
        qtbot.addWidget(dlg)
        dlg._on_propose()
        dlg._on_save()
        assert dlg.changed_count() == 1
        assert json.loads(_fmap(dataset, 1).read_text())["IntendedFor"] == [
            "bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"]

    def test_pointing_at_nothing_removes_the_field(self, qtbot, dataset):
        from bidsmgr.editor import linkage

        linkage.apply_links(dataset, [(
            _fmap(dataset, 1), "IntendedFor",
            ["bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"],
        )])
        dlg = LinkageDialog(dataset, _fmap(dataset, 1))
        qtbot.addWidget(dlg)
        dlg._on_clear()
        dlg._on_save()
        assert "IntendedFor" not in json.loads(_fmap(dataset, 1).read_text())

    def test_several_files_take_the_same_target_at_once(self, qtbot, dataset):
        """Select both fieldmaps, tick one run, both get it."""
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        for i in range(dlg._left.topLevelItemCount()):
            dlg._left.topLevelItem(i).setSelected(True)
        dlg._on_source_changed()
        self._tick(dlg, "run-3_bold")
        dlg._on_save()
        for n in (1, 2):
            assert json.loads(_fmap(dataset, n).read_text())["IntendedFor"] == [
                "bids::sub-001/func/sub-001_task-x_run-3_bold.nii.gz"]

    def test_disagreeing_files_show_a_partial_tick(self, qtbot, dataset):
        """Nothing is silently flattened when the selection disagrees."""
        from bidsmgr.editor import linkage

        linkage.apply_links(dataset, [(
            _fmap(dataset, 1), "IntendedFor",
            ["bids::sub-001/func/sub-001_task-x_run-1_bold.nii.gz"],
        )])
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        for i in range(dlg._left.topLevelItemCount()):
            dlg._left.topLevelItem(i).setSelected(True)
        dlg._on_source_changed()
        states = [
            dlg._right.topLevelItem(i).checkState(0)
            for i in range(dlg._right.topLevelItemCount())
        ]
        assert Qt.CheckState.PartiallyChecked in states

    def test_ticking_and_unticking_does_not_take_the_process_down(
        self, qtbot, dataset,
    ):
        """It used to, and not as a Python exception.

        Ticking a box emits ``itemChanged``; the handler refilled both lists,
        and clearing a tree from inside that handler destroys the very item
        whose signal is still running. Qt goes on using it and the process
        dies with a segmentation fault, which no ``except`` clause catches.
        Five ticks were enough.
        """
        dlg = LinkageDialog(dataset, _fmap(dataset, 1))
        qtbot.addWidget(dlg)
        rows = [dlg._right.topLevelItem(i)
                for i in range(dlg._right.topLevelItemCount())]
        assert len(rows) >= 3
        for item, state in (
            (rows[0], Qt.CheckState.Checked),
            (rows[1], Qt.CheckState.Checked),
            (rows[0], Qt.CheckState.Unchecked),
            (rows[2], Qt.CheckState.Checked),
            (rows[1], Qt.CheckState.Unchecked),
        ):
            item.setCheckState(0, state)
        assert dlg._pending
        ticked = {
            Path(r.data(0, Qt.ItemDataRole.UserRole)).name for r in rows
            if r.checkState(0) == Qt.CheckState.Checked
        }
        assert ticked == {"sub-001_task-x_run-3_bold.nii.gz"}

    def test_a_tick_does_not_rebuild_the_lists_under_the_cursor(
        self, qtbot, dataset,
    ):
        """The lists are what the user is pointing at. Rebuilding them on
        every tick lost the scroll position and the row identities."""
        dlg = LinkageDialog(dataset, _fmap(dataset, 1))
        qtbot.addWidget(dlg)
        before = [dlg._right.topLevelItem(i)
                  for i in range(dlg._right.topLevelItemCount())]
        before[0].setCheckState(0, Qt.CheckState.Checked)
        after = [dlg._right.topLevelItem(i)
                 for i in range(dlg._right.topLevelItemCount())]
        assert all(a is b for a, b in zip(before, after)), (
            "the rows were replaced, so the list was rebuilt"
        )

    def test_an_edited_row_says_so_before_it_is_saved(self, qtbot, dataset):
        dlg = LinkageDialog(dataset, _fmap(dataset, 1))
        qtbot.addWidget(dlg)
        self._tick(dlg, "run-2_bold")
        edited = [
            dlg._left.topLevelItem(i).text(2)
            for i in range(dlg._left.topLevelItemCount())
            if dlg._left.topLevelItem(i).text(2) == "edited, not saved"
        ]
        assert edited

    def test_everything_saves_as_one_operation(self, qtbot, dataset):
        """A linkage pass over a session is one entry in the history."""
        dlg = LinkageDialog(dataset)
        qtbot.addWidget(dlg)
        for i in range(dlg._left.topLevelItemCount()):
            dlg._left.topLevelItem(i).setSelected(True)
        dlg._on_source_changed()
        dlg._on_propose()
        assert len(dlg._pending) == 2
        dlg._on_save()
        assert dlg.changed_count() == 2
        assert not dlg._pending

class TestIndexWidthsDialog:
    def test_every_index_the_schema_defines_gets_a_row(self, qtbot, dataset):
        """All of them, not only those in use. Listing only what the dataset
        already has made a dataset with one ``run`` read as though ``run``
        were the only index entity there is."""
        from bidsmgr.editor.values import index_entities

        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert list(dlg._widths) == index_entities()
        assert "acq" not in dlg._widths, "a label has no width"

    def test_an_unused_entity_is_shown_and_disabled(self, qtbot, dataset):
        """Shown and disabled, not hidden: the difference between "you have
        none of these" and "these do not exist"."""
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert dlg._widths["run"].isEnabled()
        assert not dlg._widths["echo"].isEnabled()

    def test_the_editor_and_settings_offer_the_same_list(self, qtbot, dataset):
        """One list, read from the schema, so the two cannot disagree."""
        from bidsmgr.gui.app_settings import AppSettings
        from bidsmgr.gui.settings_dialog import SettingsDialog

        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        settings = SettingsDialog(AppSettings())
        qtbot.addWidget(settings)
        assert list(dlg._widths) == list(settings._index_widths)

    def test_a_consistent_dataset_says_so(self, qtbot, dataset):
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert "already uses one width" in dlg._warning.text()

    def test_the_width_starts_at_the_widest_in_use(self, qtbot, dataset):
        """So pressing Apply unchanged settles a dataset that disagrees with
        itself and leaves a consistent one alone."""
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert dlg._widths["run"].value() == 1
        assert not dlg._ok.isEnabled(), "nothing to change"

    def test_an_inconsistent_dataset_is_named(self, qtbot, dataset):
        f = dataset / "sub-001/func/sub-001_task-x_run-3_bold.nii.gz"
        f.rename(f.with_name("sub-001_task-x_run-30_bold.nii.gz"))
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert "disagree with themselves" in dlg._warning.text()
        assert dlg._widths["run"].value() == 2

    def test_a_clash_that_padding_cannot_settle_says_why(self, qtbot, dataset):
        """run-1 beside run-01 is the mess this is for, and padding would
        fuse two different runs."""
        (dataset / "sub-001/func/sub-001_task-x_run-01_bold.nii.gz").write_bytes(b"x")
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        assert "Cannot pad" in dlg._status.text()
        assert not dlg._ok.isEnabled()

    def test_the_preview_lists_what_would_change(self, qtbot, dataset):
        dlg = PadValuesDialog(dataset)
        qtbot.addWidget(dlg)
        dlg._widths["run"].setValue(2)
        head = dlg._tree.topLevelItem(0)
        assert head.text(0) == "run"
        rows = [(head.child(i).text(0), head.child(i).text(1))
                for i in range(head.childCount())]
        assert rows == [("run-1", "run-01"), ("run-2", "run-02"),
                        ("run-3", "run-03")]


class TestReplaceValueDialog:
    def test_it_offers_the_scopes(self, qtbot, dataset):
        from bidsmgr.gui.replace_value_dialog import ReplaceValueDialog

        dlg = ReplaceValueDialog(dataset)
        qtbot.addWidget(dlg)
        labels = [dlg._scope.itemText(i) for i in range(dlg._scope.count())]
        assert labels == ["The whole dataset", "sub-001"]

    def test_values_are_listed_with_their_file_counts(self, qtbot, dataset):
        from bidsmgr.gui.replace_value_dialog import ReplaceValueDialog

        dlg = ReplaceValueDialog(dataset, entity="task")
        qtbot.addWidget(dlg)
        assert dlg._old.itemText(0).startswith("x  (")
        assert "file(s)" in dlg._old.itemText(0)

    def test_replacing_renames_every_matching_file(self, qtbot, dataset):
        from bidsmgr.gui.replace_value_dialog import ReplaceValueDialog

        dlg = ReplaceValueDialog(dataset, entity="task")
        qtbot.addWidget(dlg)
        dlg._new.setText("rest")
        dlg._on_apply()
        assert dlg.applied_count() > 0
        assert (dataset / "sub-001/func/sub-001_task-rest_run-1_bold.nii.gz").exists()

    def test_the_same_value_is_refused(self, qtbot, dataset):
        from bidsmgr.gui.replace_value_dialog import ReplaceValueDialog

        dlg = ReplaceValueDialog(dataset, entity="task")
        qtbot.addWidget(dlg)
        dlg._new.setText("x")
        assert "already has" in dlg._status.text()
        assert not dlg._ok.isEnabled()


class TestCoherenceDialog:
    def test_a_clean_dataset_says_so(self, qtbot, dataset):
        """The fixture's fieldmaps carry no IntendedFor, which the times DO
        imply, so it is settled first and then checked."""
        from bidsmgr.editor import coherence

        coherence.apply(dataset, coherence.check(dataset))
        dlg = CoherenceDialog(dataset)
        qtbot.addWidget(dlg)
        assert dlg._tree.topLevelItemCount() == 0
        assert "Everything agrees" in dlg._status.text()
        assert not dlg._ok.isEnabled()

    def test_selecting_a_finding_shows_the_exact_change(self, qtbot, dataset):
        """The whole point of the rebuild: not "set TaskName" but what it
        says now and what it would say."""
        side = dataset / "sub-001/func/sub-001_task-x_run-1_bold.json"
        meta = json.loads(side.read_text())
        meta["TaskName"] = "something else"
        side.write_text(json.dumps(meta))

        dlg = CoherenceDialog(dataset)
        qtbot.addWidget(dlg)
        for item in dlg._leaves():
            finding = item.data(0, Qt.ItemDataRole.UserRole)
            if finding and finding.kind.name == "TASK_NAME_MISMATCH":
                dlg._show_detail(finding)
                rows = [
                    (dlg._detail.topLevelItem(i).text(0),
                     dlg._detail.topLevelItem(i).text(1))
                    for i in range(dlg._detail.topLevelItemCount())
                ]
                assert rows == [('TaskName: "something else"',
                                 'TaskName: "x"')]
                return
        raise AssertionError("the TaskName mismatch was not found")

    def test_findings_are_grouped_by_kind(self, qtbot, dataset):
        (dataset / "sub-001/func/sub-001_task-x_run-3_bold.nii.gz").unlink()
        (dataset / "sub-001/func/sub-001_task-x_run-3_bold.json").unlink()
        (dataset / "participants.tsv").write_text(
            "participant_id\nsub-001\nsub-999\n"
        )
        dlg = CoherenceDialog(dataset)
        qtbot.addWidget(dlg)
        heads = [dlg._tree.topLevelItem(i).text(0)
                 for i in range(dlg._tree.topLevelItemCount())]
        assert any("participants row" in h for h in heads)

    def test_repairing_settles_it(self, qtbot, dataset, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox

        monkeypatch.setattr(
            QMessageBox, "question",
            lambda *a, **k: QMessageBox.StandardButton.Ok,
        )
        (dataset / "participants.tsv").write_text(
            "participant_id\nsub-001\nsub-999\n"
        )
        dlg = CoherenceDialog(dataset)
        qtbot.addWidget(dlg)
        assert dlg._checked()
        dlg._on_apply()
        assert dlg.applied_count() >= 1
        assert "sub-999" not in (dataset / "participants.tsv").read_text()
        assert "Everything agrees" in dlg._status.text()

    def test_nothing_is_written_by_opening_it(self, qtbot, dataset):
        before = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
        dlg = CoherenceDialog(dataset)
        qtbot.addWidget(dlg)
        after = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
        assert before == after


class TestTheToolsMenu:
    def test_the_three_actions_exist(self, qtbot):
        from bidsmgr.gui.editor_panel import EditorPanel

        panel = EditorPanel()
        qtbot.addWidget(panel)
        labels = [a.text() for a in panel._tools_menu.actions()]
        for wanted in ("References (IntendedFor, Sources...)...",
                       "Find and replace a value...",
                       "Index widths...", "Check coherence..."):
            assert wanted in labels, labels


class TestIndexWidthsBothWays:
    """It has to work on a dataset that is ALREADY padded.

    The reported bug: with values written ``run-001``, ticking two digits
    found nothing to change. ``pad`` was ``zfill`` alone, so it could only
    ever add zeros, and half of what the tool claims to do was missing.
    """

    @pytest.fixture()
    def padded(self, tmp_path: Path) -> Path:
        root = tmp_path / "ds"
        (root / "sub-001" / "func").mkdir(parents=True)
        (root / "dataset_description.json").write_text(
            json.dumps({"Name": "t", "BIDSVersion": "1.10.0"})
        )
        for value in ("001", "002", "003"):
            (root / "sub-001" / "func"
             / f"sub-001_task-x_run-{value}_bold.nii.gz").write_bytes(b"x")
        return root

    def test_it_offers_to_trim(self, qtbot, padded):
        dlg = PadValuesDialog(padded)
        qtbot.addWidget(dlg)
        dlg._widths["run"].setValue(2)
        rows = _tree_rows(dlg._tree)
        assert ("run-001", "run-01") in rows
        assert ("run-003", "run-03") in rows

    def test_it_says_the_width_that_would_fit(self, qtbot, padded):
        """Otherwise a dataset written at three digits looks settled, and
        trimming is something you would have to guess at."""
        dlg = PadValuesDialog(padded)
        qtbot.addWidget(dlg)
        notes = [
            w.text() for w in dlg._rows.findChildren(QLabel)
            if "would fit" in w.text()
        ]
        assert notes and "1 would fit" in notes[0]

    def test_applying_it_renames_the_files(self, qtbot, padded, monkeypatch):
        monkeypatch.setattr(
            "bidsmgr.gui.pad_values_dialog.QMessageBox.question",
            lambda *a, **k: QMessageBox.StandardButton.Ok,
        )
        dlg = PadValuesDialog(padded)
        qtbot.addWidget(dlg)
        dlg._widths["run"].setValue(2)
        dlg._on_apply()
        names = sorted(
            p.name for p in (padded / "sub-001" / "func").iterdir()
        )
        assert names == [
            "sub-001_task-x_run-01_bold.nii.gz",
            "sub-001_task-x_run-02_bold.nii.gz",
            "sub-001_task-x_run-03_bold.nii.gz",
        ]

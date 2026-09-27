"""One click on a folder row opens or closes it, arrow or no arrow.

Qt gives the expander triangle its own hit area, so a folder row had two
halves that did different things: the name selected, the triangle opened.
Which one you got depended on hitting a target a few pixels wide, and the
symptom was that selecting a file and then going to open a nearby folder
often did nothing visible.

Driven through ``itemClicked`` rather than a synthetic mouse event on
purpose: that is the signal Qt emits for a click on the row and NOT for a
click on the triangle, so exercising it is exercising the half that was
broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from bidsmgr.gui.output_fs_pane import OutputFsPane
from bidsmgr.gui.widgets.bids_tree_pane import BidsTreePane

pytestmark = pytest.mark.gui


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_T1w.nii.gz").write_bytes(b"")
    (anat / "sub-01_T1w.json").write_text(json.dumps({"EchoTime": 0.03}))
    # A folder-recording: a directory on disk that the tree draws as one
    # recording, so clicking it must stay a plain selection.
    (root / "sub-01" / "meg" / "sub-01_task-x_meg.ds").mkdir(parents=True)
    (root / "sub-01" / "meg" / "sub-01_task-x_meg.ds" / "x.res4").write_bytes(b"")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


def _click(tree, item) -> None:
    """What Qt emits for a click on the ROW, not on the triangle."""
    tree.itemClicked.emit(item, 0)


def _click_again(tree, item) -> None:
    """A SEPARATE gesture, not the second half of a double click.

    Two clicks on one row inside the system double-click interval are one
    gesture and fold the row once. Waiting past it is what makes the second
    click mean "and now close it".
    """
    QTest.qWait(QApplication.doubleClickInterval() + 40)
    tree.itemClicked.emit(item, 0)


class TestTheEditorTree:
    def _pane(self, qtbot, dataset: Path) -> BidsTreePane:
        pane = BidsTreePane()
        qtbot.addWidget(pane)
        pane.set_root(dataset)
        return pane

    def _subject(self, pane: BidsTreePane):
        root = pane._tree.topLevelItem(0)
        return next(
            root.child(i) for i in range(root.childCount())
            if root.child(i).text(0) == "sub-01"
        )

    def test_clicking_a_folder_opens_it(self, qtbot, dataset):
        pane = self._pane(qtbot, dataset)
        subject = self._subject(pane)
        assert not subject.isExpanded()
        _click(pane._tree, subject)
        assert subject.isExpanded()

    def test_clicking_it_again_closes_it(self, qtbot, dataset):
        pane = self._pane(qtbot, dataset)
        subject = self._subject(pane)
        _click(pane._tree, subject)
        _click_again(pane._tree, subject)
        assert not subject.isExpanded()

    def test_a_double_click_is_one_toggle(self, qtbot, dataset):
        """Not two, which would open the folder and close it again, and not
        none. Qt is supposed to suppress the second ``clicked`` itself, but
        whether it does depends on the platform plugin synthesising a double
        click at all, so this does not rely on it."""
        pane = self._pane(qtbot, dataset)
        subject = self._subject(pane)
        _click(pane._tree, subject)
        _click(pane._tree, subject)   # same gesture: inside the interval
        assert subject.isExpanded()

    def test_it_still_draws_the_contents(self, qtbot, dataset):
        """The click goes through the same lazy build the triangle does."""
        pane = self._pane(qtbot, dataset)
        subject = self._subject(pane)
        _click(pane._tree, subject)
        names = {subject.child(i).text(0) for i in range(subject.childCount())}
        assert names == {"anat", "meg"}

    def test_clicking_a_file_is_a_plain_selection(self, qtbot, dataset):
        pane = self._pane(qtbot, dataset)
        item = pane.reveal(dataset / "sub-01" / "anat" / "sub-01_T1w.json")
        assert item is not None
        _click(pane._tree, item)   # must not raise, must not fold anything
        assert not item.isExpanded()

    def test_a_folder_recording_is_a_plain_selection(self, qtbot, dataset):
        """A CTF ``.ds`` is a directory on disk and one recording to a
        reader, so the tree draws it as a leaf and a click selects it."""
        pane = self._pane(qtbot, dataset)
        item = pane.reveal(
            dataset / "sub-01" / "meg" / "sub-01_task-x_meg.ds"
        )
        assert item is not None
        assert item.childCount() == 0
        _click(pane._tree, item)
        assert not item.isExpanded()

    def test_a_modified_click_leaves_the_fold_alone(
        self, qtbot, dataset, monkeypatch,
    ):
        """Ctrl and Shift build a multi-selection. Folding a folder in the
        middle of building one would fight the user."""
        from bidsmgr.gui.widgets import tree_click

        pane = self._pane(qtbot, dataset)
        subject = self._subject(pane)
        monkeypatch.setattr(
            tree_click.QApplication, "keyboardModifiers",
            staticmethod(lambda: Qt.KeyboardModifier.ControlModifier),
        )
        _click(pane._tree, subject)
        assert not subject.isExpanded()

    def test_qts_own_double_click_expand_is_off(self, qtbot, dataset):
        """Left on, it would add a second toggle to the one the click makes
        and cancel it out."""
        pane = self._pane(qtbot, dataset)
        assert not pane._tree.expandsOnDoubleClick()


class TestTheOutputTree:
    def test_clicking_a_folder_opens_it(self, qtbot, tmp_path: Path):
        (tmp_path / "study" / "sub-001").mkdir(parents=True)
        (tmp_path / "study" / "sub-001" / "x.tsv").write_text("a\n")
        pane = OutputFsPane()
        qtbot.addWidget(pane)
        pane.show()
        qtbot.waitExposed(pane)
        pane.set_root(tmp_path)
        qtbot.waitUntil(lambda: not pane._scan_in_progress, timeout=10_000)

        root = pane._tree.topLevelItem(0)
        study = next(
            root.child(i) for i in range(root.childCount())
            if root.child(i).text(0) == "study"
        )
        assert not study.isExpanded()
        _click(pane._tree, study)
        assert study.isExpanded()
        names = {study.child(i).text(0) for i in range(study.childCount())}
        assert names == {"sub-001"}

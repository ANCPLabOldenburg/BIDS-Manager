"""The Filter panel's three modes, the BIDS preview, and the splitter grips.

Recordings: one checkbox per kind of recording across all subjects, a click
leaves every one of them out (one Undo). BIDS preview: computed off the GUI
thread, filled a folder at a time, a file's predicted content coloured by
where each value comes from, following the settings, the table and the theme.
Grips: every splitter's handle shows three dots and takes the accent wash
under the mouse.
"""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import pytest
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QSplitter, QTreeWidgetItem

pytestmark = pytest.mark.gui


def _row(sub: str, name: str, datatype: str, suffix: str, *, include: int = 1, uid: str = "",
         **entities) -> dict:
    ents = {"subject": sub.split("-", 1)[1], **entities}
    return {"participant_id": sub, "session": "", "include": include, "modality": "mri",
            "datatype": datatype, "bids_guess_suffix": suffix, "bids_name": name,
            "bids_guess_confidence": "0.9", "bids_guess_skip": False, "issues": "",
            "sequence": name, "series_uid": uid or f"{name}.{suffix}", "dataset": "ds",
            "source_file": "", "source_folder": "s", "acq_time": "100000",
            "task": entities.get("task", ""), "run": "", "entities": json.dumps(ents)}


def _study() -> pd.DataFrame:
    rows = []
    for i in range(1, 5):
        sub = f"sub-{i:02d}"
        rows.append(_row(sub, f"{sub}_T1w", "anat", "T1w"))
        rows.append(_row(sub, f"{sub}_task-rest_bold", "func", "bold", task="rest"))
        rows.append(_row(sub, f"{sub}_phasediff", "fmap", "phasediff"))
    return pd.DataFrame(rows)


def _settle(qtbot, ms: int = 120) -> None:
    end = time.time() + ms / 1000
    while time.time() < end:
        qtbot.wait(10)


@pytest.fixture
def panel(qtbot):
    from bidsmgr.gui.converter_panel import ConverterPanel

    conv = ConverterPanel()
    qtbot.addWidget(conv)
    conv.resize(1500, 900)
    conv.show()
    qtbot.waitExposed(conv)
    conv.load_inventory(_study())
    return conv


def _include(panel, row: int) -> bool:
    return panel._model._read_include(row)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def test_the_modes_sit_in_the_title_bar_and_name_it(qtbot, panel) -> None:
    from PyQt6.QtCore import QSettings

    from bidsmgr.gui.filter_pane import FilterPane

    pane, frame = panel._filter_pane, panel._filter_frame
    assert frame._bar.isAncestorOf(pane.mode_switch)
    assert [b.toolTip().split(":")[0] for b in pane.mode_switch.buttons.values()] == [
        "Structure", "Recordings", "BIDS preview"]
    assert all(not b.text() and not b.icon().isNull() for b in pane.mode_switch.buttons.values())
    assert frame._title_lbl.text() == "STRUCTURE"
    pane.mode_switch.buttons["recordings"].click()
    assert pane.mode() == "recordings" and frame._title_lbl.text() == "RECORDINGS"
    assert QSettings().value("filter/mode") == "recordings"
    pane.mode_switch.buttons["preview"].click()
    assert frame._title_lbl.text() == "BIDS PREVIEW"
    assert not pane.hide_left_out_btn.isVisible(), "nothing is left out of a preview"
    again = FilterPane()
    qtbot.addWidget(again)
    assert again.mode() == "preview"


def test_a_title_follows_into_a_floating_window(qtbot, panel) -> None:
    frame = panel._filter_frame
    frame.detach()
    qtbot.addWidget(frame._detached)
    panel._filter_pane.set_mode("recordings")
    assert frame._detached.windowTitle() == "BIDS-Manager - Recordings"
    frame.reattach()
    assert frame._title == "Recordings"


# ---------------------------------------------------------------------------
# Recordings across subjects
# ---------------------------------------------------------------------------


def _kind_node(pane, title: str) -> QTreeWidgetItem:
    tree = pane._kinds
    for i in range(tree.topLevelItemCount()):
        parent = tree.topLevelItem(i)
        for j in range(parent.childCount()):
            node = parent.child(j)
            if node.text(0).split("   ")[0] == title:
                return node
    raise AssertionError(f"no kind {title}")


def test_a_kind_is_left_out_of_every_subject_at_once(qtbot, panel) -> None:
    pane = panel._filter_pane
    pane.set_mode("recordings")
    node = _kind_node(pane, "task-rest_bold")
    assert node.text(0) == "task-rest_bold   4"
    assert node.childCount() == 4 and node.child(0).text(0) == "sub-01"
    assert "4 recordings in 4 subjects" in node.toolTip(0)
    node.setCheckState(0, Qt.CheckState.Unchecked)
    bold_rows = [i for i in range(12) if i % 3 == 1]
    assert not any(_include(panel, r) for r in bold_rows)
    assert all(_include(panel, r) for r in range(12) if r not in bold_rows)
    node = _kind_node(pane, "task-rest_bold")
    assert node.checkState(0) == Qt.CheckState.Unchecked
    node.setCheckState(0, Qt.CheckState.Checked)
    assert all(_include(panel, r) for r in bold_rows)


def test_hiding_what_is_left_out_works_across_subjects(qtbot, panel) -> None:
    pane = panel._filter_pane
    pane.set_mode("recordings")
    _kind_node(pane, "T1w").setCheckState(0, Qt.CheckState.Unchecked)
    pane.hide_left_out_btn.setChecked(True)
    assert _kind_node(pane, "T1w").isHidden()
    assert not _kind_node(pane, "phasediff").isHidden()
    pane.hide_left_out_btn.setChecked(False)
    assert not _kind_node(pane, "T1w").isHidden()


def test_the_tables_row_is_shown_among_its_kind(qtbot, panel) -> None:
    pane = panel._filter_pane
    pane.set_mode("recordings")
    panel._jump_to_row(5)     # sub-02's fieldmap
    current = pane._kinds.currentItem()
    assert current is not None and current.text(0) == "sub-02"
    assert current.parent().text(0).startswith("phasediff")


# ---------------------------------------------------------------------------
# The BIDS preview
# ---------------------------------------------------------------------------


def _planned(qtbot, pane):
    qtbot.waitUntil(lambda: pane.preview.plan() is not None and not pane.preview.is_planning(),
                    timeout=5000)
    return pane.preview.plan()


def test_the_preview_is_planned_off_the_gui_thread(qtbot, panel, monkeypatch) -> None:
    from bidsmgr.converter import output_plan
    from bidsmgr.gui import output_preview

    threads = []
    real = output_plan.plan_output

    def planned(*a, **k):
        threads.append(threading.current_thread() is threading.main_thread())
        return real(*a, **k)

    monkeypatch.setattr(output_preview, "plan_output", planned)
    pane = panel._filter_pane
    pane.set_mode("preview")
    _planned(qtbot, pane)
    assert threads and not any(threads)


def test_the_preview_tree(qtbot, panel) -> None:
    pane = panel._filter_pane
    pane.set_mode("preview")
    plan = _planned(qtbot, pane)
    tree = pane.preview.tree
    top = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert top[:3] == [".bidsignore", "CHANGES", "dataset_description.json"]
    assert top[-4:] == ["sub-01", "sub-02", "sub-03", "sub-04"]
    sub2 = tree.topLevelItem(top.index("sub-02"))
    assert sub2.childCount() == 1 and not sub2.child(0).text(0), "filled when opened"
    sub2.setExpanded(True)
    assert {sub2.child(i).text(0) for i in range(sub2.childCount())} == {
        "anat", "fmap", "func", "sub-02_scans.tsv"}
    readme = tree.topLevelItem(top.index("README"))
    assert readme.font(0).italic(), "written only when there is none"
    assert "unless the dataset already has one" in readme.toolTip(0)
    assert len(plan.files) > 30


def test_a_file_shows_what_it_will_contain_and_the_table_follows(qtbot, panel) -> None:
    from bidsmgr.gui.theme_manager import CUR

    pane = panel._filter_pane
    pane.set_mode("preview")
    _planned(qtbot, pane)
    assert pane.preview.select_path("sub-03/func/sub-03_task-rest_bold.json")
    qtbot.waitUntil(lambda: "TaskName" in pane.preview.content.toPlainText(), timeout=5000)
    text = pane.preview.content.toPlainText()
    assert '"TaskName": "rest"' in text and '"Modality": "MR"' in text
    html = pane.preview.content.toHtml().lower()
    assert QColor(CUR()["accent"]).name().lower() in html, "BIDS Manager's values in the accent"
    assert pane.preview.file_origin.text().startswith("Written by dcm2niix")
    assert panel._table.currentIndex().row() == 7, "sub-03's BOLD is selected in the table"


def test_the_preview_follows_the_settings(qtbot, panel) -> None:
    pane = panel._filter_pane
    pane.set_mode("preview")
    _planned(qtbot, pane)
    pane.preview.select_path("sub-01/fmap/sub-01_phasediff.json")
    qtbot.waitUntil(lambda: "IntendedFor" in pane.preview.content.toPlainText(), timeout=5000)
    assert "bids::sub-01/func/sub-01_task-rest_bold.nii.gz" in pane.preview.content.toPlainText()
    panel._app_settings.convert_intended_for = "relative"
    panel._app_settings.convert_b0_field = False
    pane.preview.refresh()
    qtbot.waitUntil(lambda: '"func/sub-01_task-rest_bold.nii.gz"' in
                    pane.preview.content.toPlainText(), timeout=5000)
    assert "B0FieldIdentifier" not in pane.preview.content.toPlainText()


def test_a_left_out_recording_leaves_the_preview(qtbot, panel) -> None:
    pane = panel._filter_pane
    pane.set_mode("preview")
    plan = _planned(qtbot, pane)
    assert "sub-04/anat/sub-04_T1w.nii.gz" in plan.paths()
    model = panel._model
    include = next(i for i, c in enumerate(model.COLUMNS) if c.key == "include")
    model.setData(model.index(9, include), False)
    qtbot.waitUntil(lambda: "sub-04/anat/sub-04_T1w.nii.gz" not in pane.preview.plan().paths(),
                    timeout=5000)


def test_the_preview_takes_the_new_theme(qtbot, qapp) -> None:
    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import CUR, ThemeManager

    manager = ThemeManager(qapp)
    manager.apply("graphite")
    win = MainWindow(manager)
    qtbot.addWidget(win)
    win.show()
    win._apply_active_view("converter", persist=False)
    conv = win.converter
    conv.load_inventory(_study())
    pane = conv._filter_pane
    pane.set_mode("preview")
    _planned(qtbot, pane)
    for theme in ("porcelain", "nord"):
        manager.apply(theme)
        _settle(qtbot, 80)
        item = pane.preview.tree.topLevelItem(pane.preview.tree.topLevelItemCount() - 1)
        assert item.foreground(0).color().name() == QColor(CUR()["tree_folder"]).name(), theme


# ---------------------------------------------------------------------------
# Grips, and nothing opening a window
# ---------------------------------------------------------------------------


def test_every_splitter_has_a_grip(qtbot, qapp) -> None:
    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager
    from bidsmgr.gui.widgets.splitter import GRIP_PX, GripHandle, Splitter

    win = MainWindow(ThemeManager(qapp))
    qtbot.addWidget(win)
    win.show()
    splitters = win.findChildren(QSplitter)
    assert splitters and all(isinstance(s, Splitter) for s in splitters), \
        [type(s).__name__ for s in splitters if not isinstance(s, Splitter)]
    for s in splitters:
        assert s.handleWidth() >= GRIP_PX
        assert all(isinstance(s.handle(i), GripHandle) for i in range(1, s.count()))


def _pixels_like(image, colour: QColor, tolerance: int = 30) -> int:
    n = 0
    for y in range(image.height()):
        for x in range(image.width()):
            c = image.pixelColor(x, y)
            if abs(c.red() - colour.red()) + abs(c.green() - colour.green()) + \
                    abs(c.blue() - colour.blue()) <= tolerance:
                n += 1
    return n


def test_a_grip_shows_its_dots_and_takes_the_accent_under_the_mouse(qtbot, panel) -> None:
    from bidsmgr.gui.theme_manager import CUR
    from bidsmgr.gui.widgets.primitives import qcolor
    from bidsmgr.gui.widgets.splitter import GripHandle

    handle = next(h for h in panel.findChildren(GripHandle)
                  if h.isVisible() and h.orientation() == Qt.Orientation.Horizontal
                  and h.height() > 100)
    rest = handle.grab().toImage()
    assert _pixels_like(rest, qcolor(CUR()["muted"])) >= 3, "three dots at rest"
    assert _pixels_like(rest, qcolor(CUR()["accent"])) == 0
    QApplication.sendEvent(handle, QEvent(QEvent.Type.Enter))
    hover = handle.grab().toImage()
    assert handle.is_active()
    assert _pixels_like(hover, qcolor(CUR()["accent"])) >= 3, "the dots take the accent"
    QApplication.sendEvent(handle, QEvent(QEvent.Type.Leave))
    assert not handle.is_active()


class _WindowSpy(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.shown: list[str] = []

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        if event.type() == QEvent.Type.Show and obj.isWidgetType() and obj.isWindow():
            self.shown.append(f"{type(obj).__name__}#{obj.objectName()}")
        return False


def test_nothing_opens_a_window(qtbot, qapp, panel) -> None:
    spy = _WindowSpy()
    qapp.installEventFilter(spy)
    try:
        pane = panel._filter_pane
        for mode in ("recordings", "preview", "structure", "preview"):
            pane.mode_switch.buttons[mode].click()
            _settle(qtbot, 80)
        _planned(qtbot, pane)
        for path in ("dataset_description.json", "sub-02/fmap/sub-02_phasediff.json",
                     "sub-02/sub-02_scans.tsv", "participants.tsv"):
            pane.preview.select_path(path)
            _settle(qtbot, 120)
        panel.load_inventory(_study().iloc[:6].reset_index(drop=True))
        _settle(qtbot, 300)
    finally:
        qapp.removeEventFilter(spy)
    assert spy.shown == [], spy.shown


# ---------------------------------------------------------------------------
# One click folds a row, in every view, as in every other tree
# ---------------------------------------------------------------------------


def _click_row(qtbot, tree, item) -> None:
    """A real click on the row's name, past the arrow and any checkbox."""
    from PyQt6.QtTest import QTest

    rect = tree.visualItemRect(item)
    tree.scrollToItem(item)
    rect = tree.visualItemRect(item)
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    # A second gesture on the same row must wait out the double-click time.
    qtbot.wait(QApplication.doubleClickInterval() + 40)


def test_one_click_opens_a_preview_folder(qtbot, panel) -> None:
    pane = panel._filter_pane
    pane.set_mode("preview")
    _planned(qtbot, pane)
    tree = pane.preview.tree
    assert not tree.expandsOnDoubleClick()
    sub = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
               if tree.topLevelItem(i).text(0) == "sub-02")
    _click_row(qtbot, tree, sub)
    assert sub.isExpanded(), "a click on the row opens the folder"
    assert {sub.child(i).text(0) for i in range(sub.childCount())} >= {"anat", "func"}
    _click_row(qtbot, tree, sub)
    assert not sub.isExpanded()


def test_one_click_folds_a_filter_row_and_a_tick_does_not(qtbot, panel) -> None:
    from PyQt6.QtTest import QTest

    pane = panel._filter_pane
    for mode, tree in (("structure", pane._tree), ("recordings", pane._kinds)):
        pane.set_mode(mode)
        qtbot.wait(20)
        from PyQt6.QtWidgets import QTreeWidgetItemIterator

        tree.expandAll()
        it, item = QTreeWidgetItemIterator(tree), None
        while it.value() is not None:
            node = it.value()
            if node.childCount() and node.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item = node
                break
            it += 1
        assert item is not None, f"{mode}: no foldable row with a checkbox"
        tree.collapseAll()
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()
        _click_row(qtbot, tree, item)
        assert item.isExpanded(), mode
        _click_row(qtbot, tree, item)
        assert not item.isExpanded(), mode

        # The checkbox ticks and folds nothing.
        before = item.checkState(0)
        rect = tree.visualItemRect(item)
        # The middle of the indicator, styled or not (it spans 7-22 px in).
        box = rect.topLeft() + type(rect.topLeft())(13, rect.height() // 2)
        QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=box)
        qtbot.wait(QApplication.doubleClickInterval() + 40)
        assert item.checkState(0) != before, f"{mode}: the tick did not land"
        assert not item.isExpanded(), f"{mode}: a tick folded the row"


# ---------------------------------------------------------------------------
# Right-click in Structure and Recordings: the table's menu for those rows
# ---------------------------------------------------------------------------


def _menu_at(qtbot, panel, tree, item) -> list[str]:
    tree.setCurrentItem(item)
    pos = tree.visualItemRect(item).center()
    panel._filter_pane._on_tree_menu(pos, tree)
    menu = panel._filter_menu
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    menu.close()
    return texts


def _first(tree, test):
    from PyQt6.QtWidgets import QTreeWidgetItemIterator

    tree.expandAll()
    it = QTreeWidgetItemIterator(tree)
    while it.value() is not None:
        if test(it.value()):
            return it.value()
        it += 1
    raise AssertionError("no such row")


def test_a_recording_in_the_tree_offers_the_tables_menu(qtbot, panel) -> None:
    from bidsmgr.gui.filter_pane import _ROW_IDS_ROLE

    pane = panel._filter_pane
    pane.set_mode("structure")
    leaf = _first(pane._tree, lambda i: bool(i.data(0, _ROW_IDS_ROLE)))
    row = int(leaf.data(0, _ROW_IDS_ROLE)[0])
    from_tree = _menu_at(qtbot, panel, pane._tree, leaf)

    panel._table.selectRow(row)
    panel._on_table_menu(panel._table.visualRect(panel._model.index(row, 1)).center())
    from_table = [a.text() for a in panel._table_menu.actions() if not a.isSeparator()]
    panel._table_menu.close()
    assert from_tree == from_table
    assert "Give a new subject number..." in from_tree


def test_a_subject_offers_its_rename_and_its_rows(qtbot, panel) -> None:
    from bidsmgr.gui.filter_pane import _KIND_ROLE

    pane = panel._filter_pane
    pane.set_mode("structure")
    subject = _first(pane._tree, lambda i: i.data(0, _KIND_ROLE) == "subject")
    texts = _menu_at(qtbot, panel, pane._tree, subject)
    assert texts[0] == f"Rename subject {subject.data(0, Qt.ItemDataRole.UserRole + 12)}..."
    assert "Give a new subject number..." in texts

    # And the rename acts on every row under it.
    seen: list = []
    panel.rename_from_tree = lambda *a, **k: seen.append(a)
    pane._on_tree_menu(pane._tree.visualItemRect(subject).center(), pane._tree)
    panel._filter_menu.actions()[0].trigger()
    panel._filter_menu.close()
    assert seen and seen[0][0] == "subject" and len(seen[0][1]) == 3


def test_a_kind_of_recording_offers_new_numbers_for_everyone(qtbot, panel) -> None:
    from bidsmgr.gui.filter_pane import _KIND_ROLE

    pane = panel._filter_pane
    pane.set_mode("recordings")
    kind = _first(pane._kinds, lambda i: i.data(0, _KIND_ROLE) == "kind")
    texts = _menu_at(qtbot, panel, pane._kinds, kind)
    assert "Give new subject numbers..." in texts, "four people under one kind"
    assert not any(t.startswith("Rename") for t in texts)


# ---------------------------------------------------------------------------
# The preview follows the metadata answers and the settings that write files
# ---------------------------------------------------------------------------


def test_a_template_reaches_the_preview_without_a_table_edit(qtbot, panel) -> None:
    from bidsmgr.recording_meta import RecordingMetaSpec

    pane = panel._filter_pane
    pane.set_mode("preview")
    _planned(qtbot, pane)
    spec = panel._model.global_spec() or RecordingMetaSpec()
    spec.sequence_templates["func/bold"] = {"TaskDescription": "Fixate the cross"}
    panel._model.set_global_spec(spec)
    panel._model.recordingSpecChanged.emit()
    qtbot.waitUntil(lambda: not pane.preview.is_planning(), timeout=5000)
    assert pane.preview.select_path("sub-02/func/sub-02_task-rest_bold.json")
    qtbot.waitUntil(lambda: "Fixate the cross" in pane.preview.content.toPlainText(),
                    timeout=5000)
    from bidsmgr.gui.output_preview import _colour
    from bidsmgr.gui.theme_manager import CUR

    assert _colour("stated", CUR()).lower() in pane.preview.content.toHtml().lower(), \
        "the user's own answer in its own colour"


def test_the_companion_setting_adds_the_stubs(qtbot, panel) -> None:
    pane = panel._filter_pane
    pane.set_mode("preview")
    plan = _planned(qtbot, pane)
    assert plan.find("sub-01/func/sub-01_task-rest_events.tsv") is None
    panel._app_settings.post_fixup_companions = True
    pane.preview.refresh()
    qtbot.waitUntil(lambda: pane.preview.plan() is not None and pane.preview.plan().find(
        "sub-01/func/sub-01_task-rest_events.tsv") is not None, timeout=5000)
    assert pane.preview.select_path("sub-01/func/sub-01_task-rest_events.tsv")
    qtbot.waitUntil(lambda: "TODO" in pane.preview.content.toPlainText(), timeout=5000)


def test_the_legend_names_every_source(qtbot, panel) -> None:
    from bidsmgr.converter.output_plan import SOURCES
    from bidsmgr.gui.output_preview import _LEGEND

    assert {key for key, _t, _token in _LEGEND} == set(SOURCES)


def test_a_purple_accent_moves_the_answers_to_teal() -> None:
    from bidsmgr.gui.output_preview import _colour

    pal = {"accent": "#a371f7", "purple": "#a371f7", "teal": "#39c5cf", "text": "#ffffff"}
    assert _colour("stated", pal) == "#39c5cf"
    pal["accent"] = "#2f81f7"
    assert _colour("stated", pal) == "#a371f7"

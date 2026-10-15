"""Inspection > Overview: what each subject will have, beside the table.

What is checked, beyond "it shows": it counts on a worker thread and only
while shown; it follows edits and never shows a count of a table that has
changed since; a count, a header or Show selects the right rows (and
Properties follows); its colours and fonts follow the theme and the font
size; it travels with the Inspection when that floats; nothing opens a
window.
"""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import pytest
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

pytestmark = pytest.mark.gui


def _row(sub: str, name: str, datatype: str, suffix: str, *, include: int = 1,
         uid: str = "", **entities) -> dict:
    ents = {"subject": sub.split("-", 1)[1], **entities}
    return {"participant_id": sub, "session": "", "include": include, "modality": "mri",
            "datatype": datatype, "bids_guess_suffix": suffix,
            "bids_name": f"{sub}_{name}" if name else f"{sub}_{suffix}",
            "bids_guess_confidence": "0.9", "bids_guess_skip": False, "issues": "",
            "sequence": name or suffix, "series_uid": uid or f"{sub}.{name}.{suffix}",
            "dataset": "ds", "source_file": "", "source_folder": "s", "acq_time": "10:00:00",
            "task": entities.get("task", ""), "run": entities.get("run", ""),
            "entities": json.dumps(ents)}


def _study() -> pd.DataFrame:
    """Eight subjects with a T1w, a resting BOLD and a fieldmap; sub-04 has
    no fieldmap, sub-03 a second T1w, sub-02's task is spelled 'resting'."""
    rows = []
    for i in range(1, 9):
        sub = f"sub-{i:02d}"
        rows.append(_row(sub, "", "anat", "T1w"))
        task = "resting" if i == 2 else "rest"
        rows.append(_row(sub, f"task-{task}_bold", "func", "bold", task=task))
        if i != 4:
            rows.append(_row(sub, "", "fmap", "phasediff"))
        if i == 3:
            rows.append(_row(sub, "run-2_T1w", "anat", "T1w", run="2"))
    return pd.DataFrame(rows)


def _settle(qtbot, ms: int = 120) -> None:
    end = time.time() + ms / 1000
    while time.time() < end:
        qtbot.wait(10)


def _counted(qtbot, pane, timeout: int = 5000):
    qtbot.waitUntil(lambda: pane.overview() is not None and not pane.is_counting()
                    and not pane._dirty, timeout=timeout)
    return pane.overview()


@pytest.fixture
def panel(qtbot):
    from bidsmgr.gui.converter_panel import ConverterPanel

    conv = ConverterPanel()
    qtbot.addWidget(conv)
    conv.resize(1400, 900)
    conv.show()
    qtbot.waitExposed(conv)
    return conv


def _overview_of(qtbot, conv, df=None):
    conv.set_inspection_view("overview")
    conv.load_inventory(df if df is not None else _study())
    return _counted(qtbot, conv._overview)


# ---------------------------------------------------------------------------
# Where it sits
# ---------------------------------------------------------------------------


def test_the_switch_sits_after_the_inspection_title(panel) -> None:
    bar = panel._inspection_unit._bar
    switch = panel._view_switch
    assert bar.isAncestorOf(switch), "lifted into the title bar"
    layout = bar.layout()
    title = layout.indexOf(panel._inspection_unit._title_lbl)
    holder = layout.indexOf(switch.parentWidget())
    detach = layout.indexOf(panel._inspection_unit._detach_btn)
    assert title < holder < detach
    assert [b.text() for b in switch.buttons.values()] == ["Table", "Overview"]
    assert switch.height() <= bar.height(), "fits the 30 px bar"


def test_the_view_is_switched_and_remembered(qtbot, panel) -> None:
    from PyQt6.QtCore import QSettings

    from bidsmgr.gui.converter_panel import ConverterPanel

    assert panel.inspection_view() == "table"
    panel._view_switch.buttons["overview"].click()
    assert panel.inspection_view() == "overview"
    assert panel._inspection_views.currentWidget() is panel._overview
    assert QSettings().value("inspector/view") == "overview"
    again = ConverterPanel()
    qtbot.addWidget(again)
    assert again.inspection_view() == "overview"
    assert again._view_switch.current() == "overview"


def test_an_empty_converter_says_what_to_do(panel) -> None:
    panel.set_inspection_view("overview")
    assert panel._overview._stack.currentIndex() == 0
    assert "Scan a raw-data folder" in panel._overview._empty.text()


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_it_counts_on_a_worker_thread_and_only_while_shown(qtbot, panel, monkeypatch) -> None:
    from bidsmgr.gui import overview_pane
    from bidsmgr.inventory.overview import build_overview

    threads = []

    def counted(*args, **kwargs):
        threads.append(threading.current_thread() is threading.main_thread())
        return build_overview(*args, **kwargs)

    monkeypatch.setattr(overview_pane, "build_overview", counted)
    panel.load_inventory(_study())
    _settle(qtbot, 400)
    assert threads == [], "nothing counted while the table is shown"
    panel.set_inspection_view("overview")
    _counted(qtbot, panel._overview)
    assert threads == [False], "counted once, off the GUI thread"


def test_what_it_shows(qtbot, panel) -> None:
    ov = _overview_of(qtbot, panel)
    pane = panel._overview
    tiles = {k: t.number.text() for k, t in pane._tiles.items()}
    assert tiles == {"subjects": "8", "sessions": "0", "recordings": "24", "datatypes": "3"}
    assert pane._facts.text() == "24 MRI"
    texts = [r.findChild(QObject, "overview-finding").text() for r in pane.finding_rows()]
    assert texts == [f.text for f in ov.findings]
    assert any("spelled two ways" in t for t in texts)
    assert any(t.startswith("sub-04 has no fmap/phasediff") for t in texts)
    assert pane._findings_title.text() == f"Worth a look ({len(ov.findings)})"
    grid = pane.grid
    assert grid.grid_model.rowCount() == 8 and grid.grid_model.columnCount() == len(ov.kinds)
    assert grid.grid_model.parent() is grid, "the model belongs to its view (guard 8c)"


def test_it_follows_edits_and_drops_stale_counts(qtbot, panel) -> None:
    ov = _overview_of(qtbot, panel)
    assert ov.recordings == 24
    model = panel._model
    include = next(i for i, c in enumerate(model.COLUMNS) if c.key == "include")
    for row in (0, 1, 2):
        model.setData(model.index(row, include), False)
    qtbot.waitUntil(lambda: panel._overview.overview().recordings == 21, timeout=5000)
    assert panel._overview._ticket_state[1] == model.revision


def test_every_new_table_is_counted(qtbot, panel) -> None:
    """Scanning again and again. The count's number travelled through a Qt
    signal as a 32-bit int and was ``binding * 10**9 + revision``: from the
    third table it wrapped negative, never matched, and the Overview stayed
    empty (recounting every 11 ms) until the app was restarted."""
    panel.set_inspection_view("overview")
    pane = panel._overview
    started = []
    real = pane._jobs.start if pane._jobs is not None else None
    for n in range(1, 7):
        df = _study().iloc[: 3 * n + 3].reset_index(drop=True)
        panel.load_inventory(df)
        qtbot.waitUntil(lambda: pane.overview() is not None
                        and pane.overview().recordings == len(df), timeout=5000)
        assert pane._stack.currentIndex() == 2, f"table {n} is shown"
        started.append(pane._ticket)
    assert all(0 < t < 2 ** 31 for t in started), "fits the signal's int"
    _settle(qtbot, 300)
    assert not pane.is_counting() and not pane._settle.isActive(), "and then it rests"
    del real


def test_a_new_table_replaces_the_old_one(qtbot, panel) -> None:
    _overview_of(qtbot, panel)
    smaller = _study().iloc[:6].reset_index(drop=True)
    panel.load_inventory(smaller)
    qtbot.waitUntil(lambda: panel._overview.overview() is not None
                    and panel._overview.overview().recordings == 6, timeout=5000)


def test_the_column_level_is_a_choice(qtbot, panel) -> None:
    from PyQt6.QtCore import QSettings

    _overview_of(qtbot, panel)
    box = panel._overview.level_box
    assert box.itemText(0) == "Columns: automatic (every label)"
    box.setCurrentIndex(box.findData("suffix"))
    qtbot.waitUntil(lambda: panel._overview.overview().level == "suffix", timeout=5000)
    assert [k.key for k in panel._overview.overview().kinds] == [
        "anat/T1w", "func/bold", "fmap/phasediff"]
    assert QSettings().value("inspector/overview_level") == "suffix"


def test_many_findings_fold_under_show_all(qtbot, panel) -> None:
    from bidsmgr.gui.overview_pane import FINDINGS_SHOWN

    rows = []
    for i in range(1, 11):
        for t in range(1, 9):
            if t != i:
                rows.append(_row(f"sub-{i:02d}", f"task-t{t}_bold", "func", "bold", task=f"t{t}"))
    ov = _overview_of(qtbot, panel, pd.DataFrame(rows))
    pane = panel._overview
    assert len(ov.findings) == 8
    assert len(pane.finding_rows()) == FINDINGS_SHOWN
    assert pane._more.isVisible() and pane._more.text() == "Show all 8"
    pane._more.click()
    assert len(pane.finding_rows()) == 8 and pane._more.text() == "Show fewer"


def test_the_findings_fold_to_give_the_grid_room(qtbot, panel) -> None:
    from PyQt6.QtCore import QSettings

    _overview_of(qtbot, panel)
    pane = panel._overview
    _settle(qtbot, 60)
    grid_before = pane.grid.height()
    assert pane._findings_box.isVisible() and not pane.findings_folded()
    pane._findings_title.click()
    _settle(qtbot, 50)
    assert pane.findings_folded() and not pane._findings_box.isVisible()
    assert pane._findings_title.text() == f"Worth a look ({len(pane.overview().findings)})"
    assert pane.grid.height() > grid_before, "the grid takes the room"
    assert QSettings().value("inspector/overview_findings_folded") in (True, "true")
    pane._findings_title.click()
    assert pane._findings_box.isVisible()


# ---------------------------------------------------------------------------
# Choosing rows
# ---------------------------------------------------------------------------


def _cell_index(grid, visit: str, kind: str):
    ov = grid.grid_model.overview
    r = next(i for i, v in enumerate(ov.visits) if v.label == visit)
    c = next(i for i, k in enumerate(ov.kinds) if k.key == kind)
    return grid.grid_model.index(r, c)


def _selected(panel) -> list[int]:
    return sorted({i.row() for i in panel._table.selectionModel().selectedRows()})


def test_a_count_selects_its_recordings_and_properties_follows(qtbot, panel) -> None:
    ov = _overview_of(qtbot, panel)
    grid = panel._overview.grid
    index = _cell_index(grid, "sub-03", "anat/T1w")
    QTest.mouseClick(grid.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, grid.visualRect(index).center())
    cell = ov.cell(index.row(), index.column())
    assert _selected(panel) == sorted(cell.rows) and len(cell.rows) == 2
    panel._properties.flush()
    assert panel._properties._row == min(cell.rows)
    assert panel.inspection_view() == "overview", "a click stays on the Overview"


def test_an_empty_count_selects_what_that_subject_has(qtbot, panel) -> None:
    ov = _overview_of(qtbot, panel)
    grid = panel._overview.grid
    index = _cell_index(grid, "sub-04", "fmap/phasediff")
    grid.clicked.emit(index)
    assert _selected(panel) == sorted(ov.visit_rows[index.row()])


def test_double_click_or_enter_opens_the_table(qtbot, panel) -> None:
    ov = _overview_of(qtbot, panel)
    grid = panel._overview.grid
    index = _cell_index(grid, "sub-03", "anat/T1w")
    # A person's double-click is a click and then the double-click event;
    # QTest's mouseDClick sends only the second.
    at = grid.visualRect(index).center()
    QTest.mouseClick(grid.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    QTest.mouseDClick(grid.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
    assert panel.inspection_view() == "table"
    assert _selected(panel) == sorted(ov.cell(index.row(), index.column()).rows)
    panel.set_inspection_view("overview")
    grid.setFocus()
    grid.setCurrentIndex(_cell_index(grid, "sub-01", "anat/T1w"))
    QTest.keyClick(grid, Qt.Key.Key_Return)
    assert panel.inspection_view() == "table"
    assert _selected(panel) == [0]


def test_arrow_keys_walk_the_grid_and_select(qtbot, panel) -> None:
    ov = _overview_of(qtbot, panel)
    grid = panel._overview.grid
    grid.setFocus()
    grid.setCurrentIndex(_cell_index(grid, "sub-02", "anat/T1w"))
    QTest.keyClick(grid, Qt.Key.Key_Down)
    index = grid.currentIndex()
    assert ov.visits[index.row()].label == "sub-03"
    assert _selected(panel) == sorted(ov.cell(index.row(), index.column()).rows)
    assert panel.inspection_view() == "overview"


def test_show_opens_the_table_on_a_findings_rows(qtbot, panel) -> None:
    from PyQt6.QtWidgets import QPushButton

    ov = _overview_of(qtbot, panel)
    pane = panel._overview
    row = next(r for r in pane.finding_rows() if r.property("finding").startswith("spelling:"))
    finding = next(f for f in ov.findings if f.key == row.property("finding"))
    show = next(b for b in row.findChildren(QPushButton) if b.text() == "Show")
    show.click()
    assert panel.inspection_view() == "table"
    assert _selected(panel) == sorted(finding.rows)
    assert panel._model.dataframe().at[finding.rows[0], "bids_name"] == "sub-02_task-resting_bold"


def test_headers_select_a_kind_or_a_subject(qtbot, panel) -> None:
    ov = _overview_of(qtbot, panel)
    grid = panel._overview.grid
    grid.horizontalHeader().sectionClicked.emit(0)
    assert _selected(panel) == sorted(ov.kind_rows[0])
    grid.verticalHeader().sectionClicked.emit(1)
    assert _selected(panel) == sorted(ov.visit_rows[1])


def test_the_tables_row_is_marked_in_the_grid(qtbot, panel) -> None:
    _overview_of(qtbot, panel)
    grid = panel._overview.grid
    panel.set_inspection_view("table")
    panel._jump_to_row(5)     # sub-02's fieldmap (T1w 3, BOLD 4, fieldmap 5)
    current = grid.currentIndex()
    ov = grid.grid_model.overview
    assert ov.visits[current.row()].label == "sub-02"
    assert ov.kinds[current.column()].key == "fmap/phasediff"


# ---------------------------------------------------------------------------
# Looking right
# ---------------------------------------------------------------------------


def _count_near(image, colour: QColor, tolerance: int = 40) -> int:
    n = 0
    for y in range(0, image.height(), 1):
        for x in range(0, image.width(), 1):
            c = image.pixelColor(x, y)
            if (abs(c.red() - colour.red()) + abs(c.green() - colour.green())
                    + abs(c.blue() - colour.blue())) <= tolerance:
                n += 1
    return n


def test_the_painted_grid_follows_the_theme(qtbot, qapp) -> None:
    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import CUR, ThemeManager
    from bidsmgr.gui.widgets.primitives import qcolor

    manager = ThemeManager(qapp)
    manager.apply("graphite")
    win = MainWindow(manager)
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    win._apply_active_view("converter", persist=False)
    conv = win.converter
    _overview_of(qtbot, conv)
    grid = conv._overview.grid
    rect = grid.visualRect(_cell_index(grid, "sub-03", "anat/T1w"))
    for theme in ("graphite", "porcelain", "hc-dark"):
        manager.apply(theme)
        _settle(qtbot, 60)
        image = grid.viewport().grab(rect).toImage()
        warning = qcolor(CUR()["warning"])
        assert _count_near(image, warning) > 3, f"{theme}: the pill's number in its warning colour"
    assert conv._overview._findings_title.isVisible()


def test_painted_text_follows_the_font_size(qtbot, qapp) -> None:
    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager

    manager = ThemeManager(qapp)
    manager.apply("graphite")
    win = MainWindow(manager)
    qtbot.addWidget(win)
    win.show()
    win._apply_active_view("converter", persist=False)
    conv = win.converter
    _overview_of(qtbot, conv)
    header = conv._overview.grid.horizontalHeader()
    before = (header.height(), header.sectionSize(1), conv._overview.grid.verticalHeader().width())
    manager.set_font_scale(1.3)
    try:
        after = (header.height(), header.sectionSize(1), conv._overview.grid.verticalHeader().width())
        assert all(a > b for a, b in zip(after, before)), (before, after)
    finally:
        manager.set_font_scale(1.0)


def test_no_floor_under_the_panel_and_one_scroll(qtbot, panel) -> None:
    _overview_of(qtbot, panel)
    pane = panel._overview
    body = pane._scroll.widget()
    assert body.minimumSizeHint().width() < 260, "the Inspection still shrinks"
    _settle(qtbot, 50)
    assert pane._scroll.verticalScrollBar().maximum() == 0, "the page fits; the grid scrolls"
    assert pane.grid.height() > pane.grid.minimumHeight()


# ---------------------------------------------------------------------------
# Floating, windows, start-up
# ---------------------------------------------------------------------------


class _WindowSpy(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.shown: list[str] = []

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        if event.type() == QEvent.Type.Show and obj.isWidgetType() and obj.isWindow():
            self.shown.append(f"{type(obj).__name__}#{obj.objectName()}")
        return False


def test_the_switch_floats_with_the_inspection(qtbot, panel) -> None:
    unit = panel._inspection_unit
    switch = panel._view_switch
    unit.detach()
    qtbot.addWidget(unit._detached)
    assert unit._detached.isAncestorOf(switch) and not unit._bar.isAncestorOf(switch)
    assert panel._view_switch_home.contentsMargins().top() > 0, "room around it when floating"
    _settle(qtbot, 60)
    pane = panel._inspection_pane
    top_left = switch.mapTo(pane, switch.rect().topLeft())
    assert top_left.y() < 20 and top_left.x() < 20, "at the top left, not stretched at the bottom"
    assert switch.width() < pane.width() / 2
    unit.reattach()
    assert unit._bar.isAncestorOf(switch)
    assert panel._view_switch_home.contentsMargins().top() == 0


def test_nothing_opens_a_window(qtbot, qapp, panel) -> None:
    spy = _WindowSpy()
    qapp.installEventFilter(spy)
    try:
        panel._view_switch.buttons["overview"].click()
        panel.load_inventory(_study())
        _counted(qtbot, panel._overview)
        grid = panel._overview.grid
        for visit, kind in (("sub-01", "anat/T1w"), ("sub-04", "fmap/phasediff")):
            index = _cell_index(grid, visit, kind)
            QTest.mouseClick(grid.viewport(), Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier, grid.visualRect(index).center())
            _settle(qtbot, 60)
        panel._overview._more.click() if panel._overview._more.isVisible() else None
        panel._overview.level_box.setCurrentIndex(3)
        _counted(qtbot, panel._overview)
        panel.load_inventory(_study().iloc[:9].reset_index(drop=True))
        _counted(qtbot, panel._overview)
        _settle(qtbot)
    finally:
        qapp.removeEventFilter(spy)
    assert spy.shown == [], spy.shown
    assert sorted(type(w).__name__ for w in QApplication.topLevelWidgets()
                  if w.isVisible()) == ["ConverterPanel"]


# ---------------------------------------------------------------------------
# Filter: Hide rows left out
# ---------------------------------------------------------------------------


def test_filter_hides_the_rows_left_out(qtbot, panel) -> None:
    from PyQt6.QtCore import QSettings

    df = _study()
    df.loc[df["participant_id"] == "sub-05", "include"] = 0
    df.loc[0, "include"] = 0                      # sub-01's T1w
    panel.load_inventory(df)
    pane = panel._filter_pane
    btn = pane.hide_left_out_btn
    assert panel._filter_frame._bar.isAncestorOf(btn), "in the Filter's header"
    assert not btn.isChecked()

    def shown_subjects():
        root = pane._tree.topLevelItem(0)
        return [root.child(i).text(0) for i in range(root.childCount())
                if not root.child(i).isHidden()]

    assert "sub-05" in shown_subjects()
    btn.setChecked(True)
    assert "sub-05" not in shown_subjects(), "a subject left out entirely goes"
    assert pane._leaf_for_row(0).isHidden() and not pane._leaf_for_row(1).isHidden()
    assert QSettings().value("filter/hide_left_out") in (True, "true")
    model = panel._model
    include = next(i for i, c in enumerate(model.COLUMNS) if c.key == "include")
    model.setData(model.index(1, include), False)
    assert pane._leaf_for_row(1).isHidden(), "unticked in the table, gone here too"
    btn.setChecked(False)
    assert "sub-05" in shown_subjects()

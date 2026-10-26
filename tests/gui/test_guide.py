"""The guide in a full main window: it asks, the user chooses, it walks.

Pinned here:

- the switch sits in the header just left of Settings, is off at first and
  is remembered; a window opened with it on ASKS on Home at once;
- a chosen task starts at its first step, whatever was already true (the
  defect: "Check and edit the table" jumped to Run because a row had been
  clicked before), and moves on by itself only when the user does the thing;
- every step has Previous and Next (Finish on the last), and All tasks goes
  back to the question, which then offers to continue;
- Not now leaves a small Guide button that brings the question back;
- the pieces are children of the window, never windows; the overlay never
  takes a click; the card never covers its control and stays in the window;
- the question measures its answers even when asked while on screen (it
  opened empty once), uses two columns for many, and scrolls when low;
- hidden under a dialog; a theme and a font-size change re-measure it.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest
from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog

pytestmark = pytest.mark.gui


def _rows(n: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "participant_id": "sub-001", "session": "", "include": 1,
            "modality": "mri", "datatype": "func", "bids_guess_suffix": "bold",
            "bids_name": f"sub-001_task-t{i}_run-1_bold",
            "bids_guess_confidence": "0.9", "bids_guess_skip": False, "issues": "",
            "sequence": f"bold_{i}", "series_uid": f"1.{i}", "dataset": "ds",
            "source_file": "", "source_folder": "s", "acq_time": f"10:{i:02d}:00",
            "task": f"t{i}", "run": "1",
            "entities": json.dumps({"subject": "001", "task": f"t{i}", "run": "1"}),
        })
    return pd.DataFrame(rows)


class _WindowSpy(QObject):
    """Records every window shown."""

    def __init__(self) -> None:
        super().__init__()
        self.shown: list[str] = []

    def eventFilter(self, obj, event):  # noqa: N802 - Qt signature
        if event.type() == QEvent.Type.Show and obj.isWidgetType() and obj.isWindow():
            self.shown.append(f"{type(obj).__name__}#{obj.objectName()}")
        return False


def _settle(qtbot, ms: int = 150) -> None:
    end = time.time() + ms / 1000
    while time.time() < end:
        qtbot.wait(10)


def _make_window(qtbot, qapp, monkeypatch, tmp_path, *, guide=False):
    from bidsmgr.gui.app_settings import AppSettings
    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager

    picks = {"dir": str(tmp_path)}
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: picks["dir"]))
    if guide:
        AppSettings.remember_guided_mode(True)
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    win.picks = picks
    win.theme_manager = theme
    _settle(qtbot)
    return win


@pytest.fixture
def window(qtbot, qapp, monkeypatch, tmp_path):
    return _make_window(qtbot, qapp, monkeypatch, tmp_path)


@pytest.fixture
def guided(qtbot, qapp, monkeypatch, tmp_path):
    return _make_window(qtbot, qapp, monkeypatch, tmp_path, guide=True)


def _choose(qtbot, win, key: str) -> None:
    rows = {r.key: r for r in win.guide.question.findChildren(QObject, "guide-choice")}
    QTest.mouseClick(rows[key], Qt.MouseButton.LeftButton)
    _settle(qtbot, 80)


def _card(win):
    win.guide.refresh()
    card = win.guide.card
    return card.step.key if card.isVisible() and card.step is not None else None


def _with_a_table(qtbot, win, tmp_path) -> None:
    win.welcome.create_project(tmp_path, "Study_Q")
    win._apply_active_view("converter", persist=False)
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    win.picks["dir"] = str(raw)
    win.converter._raw_pathbar.change_button.click()
    win.converter.load_inventory(_rows())
    _settle(qtbot)


def _rect_of(win, widget):
    return win.guide._rect_in_host(widget)


# ---------------------------------------------------------------------------
# The switch, and asking when the app opens
# ---------------------------------------------------------------------------


def test_the_switch_sits_left_of_settings_and_is_remembered(qtbot, window) -> None:
    from bidsmgr.gui.app_settings import AppSettings

    header = window._header
    switch = header.guide_switch
    assert switch.isVisible() and not switch.isChecked() and not window.guide.enabled
    gap = header._settings_btn.geometry().left() - switch.geometry().right()
    assert 0 < gap < 20, "just left of Settings"
    switch.click()
    _settle(qtbot)
    assert window.guide.enabled and AppSettings.load().guided_mode
    switch.click()
    assert not window.guide.enabled and AppSettings.load().guided_mode is False
    assert not window.guide.question.isVisible() and not window.guide.card.isVisible()


def test_opening_the_app_with_the_guide_on_asks_on_home(qtbot, guided) -> None:
    question = guided.guide.question
    assert guided.guide.enabled and question.isVisible()
    assert question.title == "What would you like to do?"
    assert question.choices() == ["create", "open", "sample"]
    focused = QApplication.focusWidget()
    assert getattr(focused, "key", None) == "create", "the usual next step has the focus"


# ---------------------------------------------------------------------------
# A task, step by step
# ---------------------------------------------------------------------------


def test_a_task_walks_with_previous_and_next(qtbot, guided) -> None:
    welcome, card = guided.welcome, guided.guide.card
    _choose(qtbot, guided, "create")
    assert _card(guided) == "create.browse"
    assert card.step.heading == "Create a new dataset · Step 1 of 3"
    assert not card._prev_btn.isEnabled() and card._next_btn.text() == "Next  ›"
    assert guided.guide.overlay.target_rect == _rect_of(guided, welcome._browse_btn)
    welcome._browse_btn.click()                       # doing it moves on
    assert _card(guided) == "create.name"
    QTest.mouseClick(card._prev_btn, Qt.MouseButton.LeftButton)
    assert _card(guided) == "create.browse"
    QTest.mouseClick(card._next_btn, Qt.MouseButton.LeftButton)
    assert _card(guided) == "create.name"
    QTest.keyClicks(welcome._name_edit, "Study_A")
    assert _card(guided) == "create.create"
    assert card._next_btn.text() == "Finish"
    QTest.mouseClick(card._next_btn, Qt.MouseButton.LeftButton)
    _settle(qtbot, 60)
    assert guided.guide.question.isVisible()
    assert guided.guide.question.title == "Done. What would you like to do next?"


def test_a_chosen_task_starts_where_it_was_chosen(qtbot, guided, tmp_path) -> None:
    """The defect: a row clicked BEFORE choosing "Check and edit the table"
    sent the guide straight to Run."""
    guided.guide.question.dismissed.emit()
    _with_a_table(qtbot, guided, tmp_path)
    table = guided.converter._table
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                     pos=table.visualRect(table.model().index(1, 2)).center())
    guided.guide.ask()
    _choose(qtbot, guided, "review")
    assert _card(guided) == "review.table"
    _settle(qtbot, 300)                               # several polls: no jump
    assert _card(guided) == "review.table"
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                     pos=table.visualRect(table.model().index(3, 2)).center())
    assert _card(guided) == "review.properties"
    card = guided.guide.card
    QTest.mouseClick(card._next_btn, Qt.MouseButton.LeftButton)
    assert _card(guided) == "review.structure"
    QTest.mouseClick(card._next_btn, Qt.MouseButton.LeftButton)
    assert _card(guided) == "review.overview"
    overview = guided.converter._view_switch.buttons["overview"]
    assert guided.guide.overlay.target_rect == _rect_of(guided, overview)


def test_more_tasks_in_the_converter(qtbot, guided, tmp_path) -> None:
    guided.guide.question.dismissed.emit()
    _with_a_table(qtbot, guided, tmp_path)
    guided.guide.ask()
    offered = guided.guide.question.choices()
    for key in ("scan", "review", "row_meta", "bulk", "columns", "convert", "settings"):
        assert key in offered, key
    _choose(qtbot, guided, "bulk")
    assert _card(guided) == "bulk.select"
    table = guided.converter._table
    table.selectRow(0)
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ShiftModifier,
                     table.visualRect(table.model().index(3, 2)).center())
    assert _card(guided) == "bulk.edit"
    assert guided.guide.overlay.target_rect == _rect_of(guided, guided.converter._bulk_btn)
    guided.guide.ask()
    _choose(qtbot, guided, "columns")
    assert guided.guide.overlay.target_rect == _rect_of(guided, guided.converter._columns_btn)
    guided.guide.ask()
    _choose(qtbot, guided, "settings")
    assert guided.guide.overlay.target_rect == _rect_of(guided, guided._header._settings_btn)


def test_the_editor_tasks(qtbot, guided, tmp_path) -> None:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text("{}")
    (root / "dataset_description.json").write_text(json.dumps(
        {"Name": "ds", "BIDSVersion": "1.10.0"}))
    guided.welcome.open_project(root)
    guided._apply_active_view("editor", persist=False)
    _settle(qtbot)
    question = guided.guide.question
    assert question.isVisible()
    assert question.choices()[:2] == ["validate", "edit"]
    for key in ("rename", "sessions", "delete", "anonymise", "quality", "repair", "history"):
        assert key in question.choices(), key
    _choose(qtbot, guided, "edit")
    editor = guided.editor
    assert guided.guide.overlay.target_rect == _rect_of(guided, editor._tree_frame)
    editor._on_file_selected(root / "sub-01" / "anat" / "sub-01_T1w.json")
    _settle(qtbot, 60)
    assert _card(guided) == "edit.change"
    assert guided.guide.overlay.target_rect == _rect_of(guided, editor._center_frame)
    guided.guide.ask()
    _choose(qtbot, guided, "quality")
    assert guided.guide.overlay.target_rect == _rect_of(guided, editor._tools_btn)


# ---------------------------------------------------------------------------
# Back to the question, Not now, Continue
# ---------------------------------------------------------------------------


def test_all_tasks_goes_back_and_offers_to_continue(qtbot, guided) -> None:
    _choose(qtbot, guided, "create")
    guided.welcome._browse_btn.click()
    assert _card(guided) == "create.name"
    QTest.mouseClick(guided.guide.card._tasks_btn, Qt.MouseButton.LeftButton)
    _settle(qtbot, 60)
    question = guided.guide.question
    assert question.isVisible() and not guided.guide.card.isVisible()
    assert question.choices()[0] == "__continue__"
    assert getattr(QApplication.focusWidget(), "key", None) == "__continue__"
    _choose(qtbot, guided, "__continue__")
    assert _card(guided) == "create.name", "where it was left"


def test_not_now_leaves_a_guide_button(qtbot, guided) -> None:
    question, rest = guided.guide.question, guided.guide.rest
    QTest.keyClick(question, Qt.Key.Key_Escape)
    _settle(qtbot, 80)
    guided.guide.refresh()
    assert not question.isVisible() and rest.isVisible()
    assert not guided.guide.card.isVisible()
    assert guided.guide._anchor().contains(rest.geometry())
    QTest.mouseClick(rest, Qt.MouseButton.LeftButton)
    _settle(qtbot, 60)
    assert question.isVisible() and not rest.isVisible()
    # A click anywhere else closes it too, and still does its work.
    QTest.mouseClick(guided.welcome._name_edit, Qt.MouseButton.LeftButton)
    _settle(qtbot, 80)
    assert not question.isVisible() and guided.welcome._name_edit.hasFocus()


def test_coming_to_a_view_asks(qtbot, guided, tmp_path) -> None:
    guided.guide.question.dismissed.emit()
    _with_a_table(qtbot, guided, tmp_path)
    guided._apply_active_view("welcome", persist=False)
    _settle(qtbot, 60)
    assert guided.guide.question.isVisible()
    assert guided.guide.question.choices()[0] == "continue"
    guided._apply_active_view("converter", persist=False)
    _settle(qtbot, 60)
    assert guided.guide.question.isVisible()
    assert "review" in guided.guide.question.choices()


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------


def test_the_pieces_are_children_and_never_windows(qtbot, qapp, monkeypatch, tmp_path) -> None:
    spy = _WindowSpy()
    QApplication.instance().installEventFilter(spy)
    try:
        win = _make_window(qtbot, qapp, monkeypatch, tmp_path, guide=True)
        _choose(qtbot, win, "create")
        win.welcome._browse_btn.click()
        win.guide.ask()
        win.guide.question.dismissed.emit()
        _settle(qtbot)
    finally:
        QApplication.instance().removeEventFilter(spy)
    host = win.centralWidget()
    g = win.guide
    for piece in (g.overlay, g.card, g.question, g.rest):
        assert not piece.isWindow() and piece.parentWidget() is host
    assert g.overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert not [w for w in spy.shown if "uide" in w or "hoice" in w], spy.shown


def test_the_card_never_covers_its_control_and_stays_in_the_window(qtbot, guided) -> None:
    _choose(qtbot, guided, "create")
    welcome = guided.welcome
    for act in (lambda: None, welcome._browse_btn.click,
                lambda: QTest.keyClicks(welcome._name_edit, "S")):
        act()
        guided.guide.refresh()
        card = guided.guide.card.geometry()
        target = guided.guide.overlay.target_rect.toAlignedRect()
        assert not card.intersects(target)
        assert guided.centralWidget().rect().contains(card)
    # The ring never takes a click: the ringed field still gets it.
    name = welcome._name_edit
    centre = name.mapTo(guided, QPoint(name.width() // 2, name.height() // 2))
    assert guided.childAt(centre) is not guided.guide.overlay


def test_a_question_asked_while_on_screen_keeps_its_answers(qtbot, guided, tmp_path) -> None:
    """It opened empty once: rows added to a question on screen were shown by
    Qt one turn later, after they had been measured as nothing."""
    question = guided.guide.question
    assert question.isVisible()
    _with_a_table(qtbot, guided, tmp_path)           # opening the project asks again
    guided.guide.ask()                               # while on screen
    assert len(question.choices()) >= 7
    assert question._scroll.height() > 200
    assert question._columns == 2


def test_a_low_window_scrolls_the_answers(qtbot, guided, tmp_path) -> None:
    guided.guide.question.dismissed.emit()
    _with_a_table(qtbot, guided, tmp_path)
    guided.guide.ask()
    question = guided.guide.question
    tall = question._scroll.height()
    low = QRect(0, 56, 1400, 360)                    # lower than the window allows
    question.place(low)
    _settle(qtbot)
    assert question._scroll.height() < tall
    assert low.contains(question.geometry())
    assert question._scroll.verticalScrollBar().maximum() > 0


def test_hidden_while_a_dialog_is_open(qtbot, guided) -> None:
    _choose(qtbot, guided, "create")
    assert guided.guide.card.isVisible()
    dialog = QDialog(guided)
    qtbot.addWidget(dialog)
    dialog.setModal(True)
    dialog.show()
    _settle(qtbot)
    guided.guide.refresh()
    assert not guided.guide.card.isVisible() and not guided.guide.overlay.isVisible()
    dialog.close()
    _settle(qtbot)
    guided.guide.refresh()
    assert guided.guide.card.isVisible()


def test_a_theme_and_font_size_change_re_measure_it(qtbot, guided) -> None:
    _choose(qtbot, guided, "create")
    card = guided.guide.card
    before = card.geometry()
    switch_before = guided._header.guide_switch.sizeHint()
    guided.theme_manager.set_font_scale(1.3)
    _settle(qtbot)
    guided.guide.refresh()
    assert card.width() > before.width() and card.height() > before.height()
    assert guided._header.guide_switch.sizeHint().width() > switch_before.width()
    for theme in ("light", "hc-dark"):
        guided.theme_manager.apply(theme)
        _settle(qtbot, 60)
        guided.guide.refresh()
        assert card.isVisible()
    guided.theme_manager.set_font_scale(1.0)


def test_a_busy_step_says_what_is_happening(qtbot, guided, tmp_path, monkeypatch) -> None:
    guided.guide.question.dismissed.emit()
    _with_a_table(qtbot, guided, tmp_path)
    guided.guide.ask()
    _choose(qtbot, guided, "scan")
    guided.converter._raw_pathbar.change_button.click()
    assert _card(guided) == "scan.scan"
    monkeypatch.setattr(guided.converter, "_scan_worker", object())
    assert _card(guided) == "scan.scan.busy"
    assert guided.guide.card.step.title == "Scanning" and guided.guide.overlay.step.busy
    monkeypatch.setattr(guided.converter, "_scan_worker", None)
    guided.converter.load_inventory(_rows())        # the scan's table arrives
    assert _card(guided) == "scan.table"

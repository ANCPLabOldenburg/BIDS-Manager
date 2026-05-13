"""pytest-qt smoke tests for the extracted GUI widgets and delegates.

The aim is narrow on purpose:

* Every widget can be instantiated under a ``QApplication`` without
  raising, with both dark and light palettes applied.
* Object names match what ``theme.qss`` selectors expect (so a future
  rename in the QSS or widget code shows up as a test failure, not a
  silent visual regression).
* Delegates' ``paint`` runs cleanly against a 1-row in-memory model
  for every variant (no rendering correctness check — just that the
  paint method doesn't crash).

Marked ``gui`` so it can be filtered out or run headless via
``QT_QPA_PLATFORM=offscreen`` per ``pyproject.toml``.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QModelIndex, QRect, Qt
from PyQt6.QtGui import QPainter, QPixmap, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QApplication, QStyleOptionViewItem

from bidsmgr.gui.delegates import (
    BADGE_ROLE,
    BidsTreeDelegate,
    CellTextDelegate,
    CheckboxDelegate,
    PAYLOAD_ROLE,
    ROW_STATE_ROLE,
    StatusDelegate,
    paint_row_state,
)
from bidsmgr.gui.theme_manager import DARK, LIGHT, ThemeManager
from bidsmgr.gui.widgets import (
    Chip,
    PaneHeader,
    PathBar,
    SidecarRow,
    StatusBadge,
    ValMessage,
    VSep,
)


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def theme(qapp: QApplication) -> ThemeManager:
    """A ThemeManager bound to the test QApplication, set to dark."""
    tm = ThemeManager(qapp)
    tm.apply("dark")
    return tm


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def test_chip_object_name_matches_qss_property(qtbot, theme) -> None:
    for kind in ("default", "success", "warn", "err", "purple", "teal"):
        chip = Chip("hello", kind=kind)
        qtbot.addWidget(chip)
        assert chip.property("chipKind") == kind


def test_vsep_has_fixed_1px_width(qtbot, theme) -> None:
    sep = VSep()
    qtbot.addWidget(sep)
    assert sep.width() == 1
    assert sep.objectName() == "vsep"


def test_pane_header_uppercases_and_uses_pane_h5(qtbot, theme) -> None:
    h = PaneHeader("inspection")
    qtbot.addWidget(h)
    assert h.text() == "INSPECTION"
    assert h.objectName() == "pane-h5"
    assert h.height() == 28


def test_pathbar_exposes_change_button(qtbot, theme) -> None:
    bar = PathBar("Raw input", "/tmp/raw", ok=True,
                  trailing_chips=[("purple", "BIDS 1.10.0")])
    qtbot.addWidget(bar)
    assert bar.objectName() == "pathbar"
    # The change button is exposed for clicked.connect wiring.
    assert bar.change_button.text() == "change…"
    # Light palette repaint via QSS doesn't crash.
    theme.apply("light")
    theme.apply("dark")


# ---------------------------------------------------------------------------
# Status badge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["ok", "warn", "err", "phys", "skip", "info"])
def test_status_badge_paints_every_kind(qtbot, theme, kind: str) -> None:
    badge = StatusBadge(kind=kind)
    qtbot.addWidget(badge)
    badge.resize(18, 18)
    # Force a paint into a QPixmap; should not raise for any kind.
    pix = QPixmap(badge.size())
    badge.render(pix)
    assert not pix.isNull()


def test_status_badge_set_kind_triggers_repaint(qtbot, theme) -> None:
    badge = StatusBadge(kind="ok")
    qtbot.addWidget(badge)
    badge.set_kind("err")
    assert badge.kind == "err"
    # set_kind to the same value is a no-op; still safe.
    badge.set_kind("err")


# ---------------------------------------------------------------------------
# SidecarRow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["req", "rec", "opt", "dep"])
def test_sidecar_row_handles_every_level(qtbot, theme, level: str) -> None:
    row = SidecarRow(level, "RepetitionTime", "2.3", "num")
    qtbot.addWidget(row)
    assert row.objectName() == "sc-row"


@pytest.mark.parametrize("value_kind", ["str", "num", "todo"])
def test_sidecar_row_handles_every_value_kind(qtbot, theme, value_kind: str) -> None:
    row = SidecarRow("rec", "Authors", "TODO", value_kind)
    qtbot.addWidget(row)


def test_sidecar_row_repaint_for_palette_swap(qtbot, theme) -> None:
    row = SidecarRow("req", "Modality", "MR", "str")
    qtbot.addWidget(row)
    row.repaint_for_palette(LIGHT)
    row.repaint_for_palette(DARK)


# ---------------------------------------------------------------------------
# ValMessage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sev,obj", [
    ("ok",   "val-msg-ok"),
    ("warn", "val-msg-warn"),
    ("err",  "val-msg-err"),
])
def test_val_message_object_name_per_severity(qtbot, theme, sev: str, obj: str) -> None:
    msg = ValMessage(sev, "RULE-1", "<code>foo</code> missing", None)
    qtbot.addWidget(msg)
    assert msg.objectName() == obj


def test_val_message_fix_button_emits_signal(qtbot, theme) -> None:
    msg = ValMessage("warn", "RULE-2", "fix me", fix_label="Auto-fill")
    qtbot.addWidget(msg)
    # Find the QPushButton and click it; signal should fire.
    from PyQt6.QtWidgets import QPushButton
    btns = msg.findChildren(QPushButton)
    assert len(btns) == 1 and btns[0].text() == "Auto-fill"
    with qtbot.waitSignal(msg.fix_requested, timeout=500):
        btns[0].click()


# ---------------------------------------------------------------------------
# Delegates
# ---------------------------------------------------------------------------


def _option(rect_w: int = 200, rect_h: int = 26) -> QStyleOptionViewItem:
    opt = QStyleOptionViewItem()
    opt.rect = QRect(0, 0, rect_w, rect_h)
    return opt


def _paint_into_pixmap(width: int = 200, height: int = 26):
    """Build a pixmap + painter the delegate can draw into."""
    pix = QPixmap(width, height)
    pix.fill(Qt.GlobalColor.transparent)
    return pix


def _index_with_data(payload, row_state: str, display: str = "") -> tuple[QStandardItemModel, QModelIndex]:
    """Build a 1-cell model + its only index.

    Returns BOTH so the caller keeps the model alive — a returned
    ``QModelIndex`` becomes invalid (and crashes the next paint) if the
    underlying ``QAbstractItemModel`` is garbage-collected.
    """
    model = QStandardItemModel(1, 1)
    item = QStandardItem(display)
    item.setData(payload, PAYLOAD_ROLE)
    item.setData(row_state, ROW_STATE_ROLE)
    model.setItem(0, 0, item)
    return model, model.index(0, 0)


@pytest.mark.parametrize("kind", ["ok", "warn", "err", "phys", "skip", "info"])
def test_status_delegate_paints_every_kind(qtbot, theme, kind: str) -> None:
    pix = _paint_into_pixmap()
    painter = QPainter(pix)
    try:
        delegate = StatusDelegate()
        model, idx = _index_with_data(kind, "")
        delegate.paint(painter, _option(), idx)
    finally:
        painter.end()


@pytest.mark.parametrize("checked", [True, False])
def test_checkbox_delegate_paints_both_states(qtbot, theme, checked: bool) -> None:
    pix = _paint_into_pixmap()
    painter = QPainter(pix)
    try:
        delegate = CheckboxDelegate()
        model, idx = _index_with_data(checked, "")
        delegate.paint(painter, _option(), idx)
    finally:
        painter.end()


@pytest.mark.parametrize("role,text", [
    ("plain",    "sub-001"),
    ("mono",     "OL_0001"),
    ("basename", "sub-001_ses-pre_T1w"),
    ("conf",     "0.94"),
    ("conf",     "—"),
])
def test_cell_text_delegate_paints_every_role(qtbot, theme, role: str, text: str) -> None:
    pix = _paint_into_pixmap()
    painter = QPainter(pix)
    try:
        delegate = CellTextDelegate(role)
        model, idx = _index_with_data(None, "", display=text)
        delegate.paint(painter, _option(), idx)
    finally:
        painter.end()


@pytest.mark.parametrize("row_state", ["", "selected", "warn", "err", "skip"])
def test_cell_text_delegate_handles_every_row_state(qtbot, theme, row_state: str) -> None:
    pix = _paint_into_pixmap()
    painter = QPainter(pix)
    try:
        delegate = CellTextDelegate("basename")
        model, idx = _index_with_data(None, row_state, display="sub-001_ses-pre_T1w")
        delegate.paint(painter, _option(), idx)
    finally:
        painter.end()


def test_paint_row_state_noop_for_empty_state(qtbot, theme) -> None:
    pix = _paint_into_pixmap()
    painter = QPainter(pix)
    try:
        paint_row_state(painter, _option(), None)
        paint_row_state(painter, _option(), "")
    finally:
        painter.end()


@pytest.mark.parametrize("badge", ["ok", "warn", "err", None])
def test_bids_tree_delegate_paints_with_or_without_badge(
    qtbot, theme, badge,
) -> None:
    model = QStandardItemModel(1, 1)
    item = QStandardItem("sub-001/")
    if badge is not None:
        item.setData(badge, BADGE_ROLE)
    model.setItem(0, 0, item)

    pix = _paint_into_pixmap()
    painter = QPainter(pix)
    try:
        delegate = BidsTreeDelegate()
        delegate.paint(painter, _option(), model.index(0, 0))
    finally:
        painter.end()


# ---------------------------------------------------------------------------
# Theme integration
# ---------------------------------------------------------------------------


def test_theme_listener_fires_on_toggle(qtbot, theme) -> None:
    seen: list[dict] = []
    theme.add_listener(lambda pal: seen.append(pal))
    theme.toggle()
    theme.toggle()
    assert len(seen) == 2
    # First toggle goes to light, second back to dark.
    assert seen[0]["bg"] == LIGHT["bg"]
    assert seen[1]["bg"] == DARK["bg"]

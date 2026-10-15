"""``FlowBar``: the wrapping bar every viewer toolbar and control row uses.

A bar is a WIDGET that places its children (a ``QLayout`` subclass in PyQt
segfaulted, see CLAUDE.md), so it has to notice by itself when children
arrive and leave.
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QCheckBox, QPushButton

from bidsmgr.gui.widgets.flow_layout import FlowBar

pytestmark = pytest.mark.gui


def _bar(qtbot, width: int = 400) -> FlowBar:
    bar = FlowBar(h_spacing=8, v_spacing=4)
    qtbot.addWidget(bar)
    bar.resize(width, 60)
    bar.show()
    qtbot.waitExposed(bar)
    return bar


def test_children_added_to_a_bar_on_screen_never_overlap(qtbot) -> None:
    """Boxes rebuilt for a new recording were all painted at the origin, one
    over the other (the MEG mag box over EEG)."""
    bar = _bar(qtbot)
    boxes = [QCheckBox(text) for text in ("MEG mag", "MEG grad", "EEG")]
    for box in boxes:
        bar.addWidget(box)
    rects = [b.geometry() for b in boxes]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not a.intersects(b), (a, b)


def test_a_child_taken_away_leaves_the_bar(qtbot) -> None:
    bar = _bar(qtbot)
    first, second = QPushButton("first"), QPushButton("second")
    bar.addWidget(first)
    bar.addWidget(second)
    first.setParent(None)
    first.deleteLater()
    assert bar.count() == 1
    assert second.geometry().left() == bar._margins[0], "the gap was not closed"
    bar.removeWidget(second)
    assert bar.count() == 0
    second.deleteLater()


def test_rows_wrap_in_a_narrow_bar(qtbot) -> None:
    bar = _bar(qtbot, width=120)
    buttons = [QPushButton(f"button {k}") for k in range(4)]
    for b in buttons:
        bar.addWidget(b)
    assert len({b.geometry().top() for b in buttons}) > 1
    assert bar.minimumSizeHint().height() >= max(b.geometry().bottom() for b in buttons)


def test_a_row_centres_its_children(qtbot) -> None:
    from PyQt6.QtWidgets import QLabel

    bar = _bar(qtbot)
    label, button = QLabel("Values"), QPushButton("Raw values")
    button.setFixedHeight(label.sizeHint().height() + 12)
    bar.addWidget(label)
    bar.addWidget(button)
    assert label.geometry().center().y() == pytest.approx(button.geometry().center().y(), abs=1), \
        "a short label sits level with the taller control beside it"


def test_a_fixed_size_child_gets_its_fixed_size(qtbot) -> None:
    bar = _bar(qtbot)
    first, second = QPushButton(), QPushButton()
    for b in (first, second):
        b.setFixedSize(28, 28)
        bar.addWidget(b)
    assert first.sizeHint().height() != 28 or first.sizeHint().width() != 28, \
        "the case this covers: a hint that is not the fixed size"
    assert first.geometry().size().height() == 28
    assert second.geometry().left() >= first.geometry().right() + 1, "no overlap"
    assert bar.heightForWidth(bar.width()) >= 28, "the row is as tall as the buttons"

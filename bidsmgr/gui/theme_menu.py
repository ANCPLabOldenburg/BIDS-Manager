"""Choosing a theme: the header's menu and the swatches beside each name.

A swatch is the theme in miniature (its background, a surface, its text
and its accent), so the themes read apart before one is picked. The menu
lists them by group (Neutral, Classic, Colour, High contrast), dark before
light, the current one ringed and in bold; Settings > Display shows the
same swatches in the same order.
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QAction, QActionGroup, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMenu, QWidget, QWidgetAction

from .theme_manager import PALETTES, THEMES


def _colour(value: str) -> QColor:
    from ..viz.theme import parse_colour

    return QColor(*parse_colour(value))


def theme_swatch(theme_id: str, side: int, *, ring: str = "") -> QIcon:
    """``theme_id`` in miniature: its background with a surface panel, a
    line of its text colour and a dot of its accent, outlined; in a ring of
    ``ring`` (a colour) when it is the theme on now."""
    pal = PALETTES[theme_id]
    app = QApplication.instance()
    ratio = app.devicePixelRatio() if app is not None else 1.0
    pm = QPixmap(max(1, int(side * ratio)), max(1, int(side * ratio)))
    pm.setDevicePixelRatio(ratio)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    box = QRectF(0.5, 0.5, side - 1, side - 1)
    radius = side * 0.22
    if ring:
        width = max(1.5, side * 0.1)
        p.setPen(QPen(_colour(ring), width))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(box.adjusted(width / 2, width / 2, -width / 2, -width / 2),
                          radius, radius)
        inset = width + 1.5
        box = box.adjusted(inset, inset, -inset, -inset)
        side_in = box.width()
        p.translate(box.left() - 0.5, box.top() - 0.5)
        box = QRectF(0.5, 0.5, side_in, side_in)
        side = side_in + 1
        radius = side * 0.22
    p.setPen(QPen(_colour(pal["border"]), 1))
    p.setBrush(_colour(pal["bg"]))
    p.drawRoundedRect(box, radius, radius)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_colour(pal["surface3"]))
    p.drawRoundedRect(QRectF(side * 0.18, side * 0.18, side * 0.64, side * 0.30),
                      side * 0.08, side * 0.08)
    p.setBrush(_colour(pal["text"]))
    p.drawRoundedRect(QRectF(side * 0.18, side * 0.58, side * 0.36, side * 0.10),
                      side * 0.05, side * 0.05)
    p.setBrush(_colour(pal["accent"]))
    p.drawEllipse(QPointF(side * 0.72, side * 0.66), side * 0.12, side * 0.12)
    p.end()
    return QIcon(pm)


def build_theme_menu(parent: QWidget, current: str,
                     on_pick: Callable[[str], None]) -> QMenu:
    """A rounded menu of every theme, ``current`` marked; picking one
    calls ``on_pick(theme_id)``."""
    from .viz import fonts
    from .viz.menus import popup_menu

    menu = popup_menu(parent)
    group = QActionGroup(menu)
    group.setExclusive(True)
    side = fonts.px(16)
    heading = None
    for t in THEMES:
        if t.group != heading:
            heading = t.group
            title = QLabel(heading.upper())
            title.setObjectName("menu-section")
            holder = QWidgetAction(menu)
            holder.setDefaultWidget(title)
            holder.setEnabled(False)
            menu.addAction(holder)
        on = t.id == current
        # The theme on now: its swatch ringed in the accent and its name in
        # bold. A tick would share the icon's column, which Qt draws apart
        # on each platform.
        act = QAction(theme_swatch(t.id, side, ring=PALETTES[current]["accent"] if on else ""),
                      t.label, menu)
        act.setCheckable(True)
        act.setChecked(on)
        if on:
            font = act.font()
            font.setWeight(QFont.Weight.DemiBold)
            act.setFont(font)
        act.setToolTip(t.description)
        act.setData(t.id)
        act.triggered.connect(lambda _c=False, tid=t.id: on_pick(tid))
        group.addAction(act)
        menu.addAction(act)
    menu.setToolTipsVisible(True)
    return menu


__all__ = ["build_theme_menu", "theme_swatch"]

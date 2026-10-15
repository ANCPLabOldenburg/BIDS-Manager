"""The header of a panel inside a viewer (the time course, the QC plots).

Two zones, the same in every panel:

* on the LEFT, the panel's own controls, grouped by purpose (what is
  plotted, how, what else is drawn on the same axis), in a row that wraps
  when the panel is narrow;
* in the UPPER-RIGHT CORNER, where a window keeps its own, the controls of
  the panel AS A PANEL: expand the plots, put the panel beside the views,
  maximise it, give it a window of its own, and a "More" menu for what is
  used rarely. Icons with a tooltip each, so they take no room from the
  controls and are always in the same place.

A header asks no width of its own beyond its widest control and one icon:
both zones wrap.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QWidget

from ...widgets.flow_layout import FlowBar
from ..menus import popup_menu

#: An icon button in the corner, square, at scale 1.0.
CORNER_PX = 28


def corner_button(icon: str, tip: str, *, checkable: bool = False,
                  button: Optional[QPushButton] = None) -> QPushButton:
    """An icon-only button for a panel's corner (or ``button`` restyled as
    one, an action's button keeping its action)."""
    from ... import icons
    from .. import fonts

    btn = button if button is not None else QPushButton()
    btn.setObjectName("viz-corner-btn")
    btn.setText("")
    if icon:
        btn.setIcon(icons.icon(icon))
        btn.setProperty("viz_icon", icon)
    side = fonts.px(CORNER_PX)
    btn.setFixedSize(side, side)
    btn.setIconSize(QSize(int(side * 0.6), int(side * 0.6)))
    btn.setCheckable(checkable or btn.isCheckable())
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    if tip:
        btn.setToolTip(tip)
    return btn


class PanelHeader(QWidget):
    """``controls`` (a wrapping bar, left) and ``corner`` (icons, top right)."""

    def __init__(self, parent=None, *, margins=(8, 4, 6, 4), h_spacing: int = 10,
                 v_spacing: int = 4) -> None:
        super().__init__(parent)
        self.setObjectName("viz-panel-header")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(self)
        row.setContentsMargins(*margins)
        row.setSpacing(10)
        self.controls = FlowBar(h_spacing=h_spacing, v_spacing=v_spacing)
        row.addWidget(self.controls, 1)
        # The corner wraps too: a row of icons that could not wrap was a
        # floor under the whole viewer's width (five icons, 160 px), and
        # the controls column stopped giving way when the viewer narrowed.
        self.corner = FlowBar(h_spacing=4, v_spacing=4)
        policy = self.corner.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Maximum)
        self.corner.setSizePolicy(policy)
        self._corner = self.corner
        row.addWidget(self.corner, 0, Qt.AlignmentFlag.AlignTop)
        # Its icons follow the theme on their own: an icon drawn for the dark
        # theme all but vanishes on the light one.
        from ..bridge import ThemeHub, connect_while_alive

        connect_while_alive(ThemeHub.instance().changed, self,
                            lambda h, _theme: h.refresh_icons())

    def add_corner(self, button: QPushButton) -> QPushButton:
        self._corner.addWidget(button)
        return button

    def insert_corner(self, buttons, *, before: Optional[QWidget] = None) -> None:
        """``buttons`` restyled as corner icons (an action's button keeps its
        action and its icon), placed before ``before`` (else last)."""
        index = self._corner.indexOf(before) if before is not None else -1
        index = self._corner.count() if index < 0 else index
        for k, btn in enumerate(buttons):
            corner_button("", btn.toolTip(), button=btn)
            self._corner.insertWidget(index + k, btn)

    def corner_buttons(self) -> list[QPushButton]:
        """The corner's buttons, left to right."""
        return [w for w in self.corner.widgets() if isinstance(w, QPushButton)]

    def more_menu(self, tip: str = "More"):
        """The corner's last button: a rounded menu of what is used rarely."""
        btn = corner_button("more", tip)
        menu = popup_menu(btn)
        menu.setToolTipsVisible(True)
        btn.setMenu(menu)
        self.add_corner(btn)
        return btn, menu

    def refresh_icons(self) -> None:
        """Every icon in the header (a button with a ``viz_icon`` property,
        corner or controls) in the theme's colours again."""
        from ... import icons

        for btn in self.findChildren(QPushButton):
            name = btn.property("viz_icon")
            if name:
                btn.setIcon(icons.icon(name))


__all__ = ["CORNER_PX", "PanelHeader", "corner_button"]

"""A collapsible / detachable wrapper around a pane (or a sub-splitter).

Each side region in the Converter / Editor is wrapped in a ``PanelFrame``
to get two affordances:

* a **collapse caret** that folds the body toward an ``edge``:
    - ``"top"`` / ``"bottom"``  fold by height; caret + title + detach share
      one thin horizontal bar on that edge.
    - ``"left"`` / ``"right"``  fold by width; the title + detach stay in a
      horizontal bar across the top, while the collapse caret sits centred in
      a thin VERTICAL strip on the panel's outer edge.
  When collapsed inside a splitter the freed space is handed to a designated
  ``grow_target`` (the inspection table / the editor viewer) so the work
  surface expands automatically, no manual drag needed.
* a **detach button** (always in the horizontal bar) that pops the body out
  into a floating window; closing it docks it back. ``inner`` may be a whole
  ``QSplitter``, so two panes (inspection + properties) detach as one unit.

The frame hides the pane's own ``PaneHeader`` so there is a single header.
A pane that has buttons belonging NEXT to its title can name a child widget
:data:`HEADER_EXTRAS` and the frame lifts it into that one header; it is
handed back when the pane detaches, so the buttons travel with the pane.
State is not persisted across restarts.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import icons
from .primitives import PaneHeader

_QWIDGETSIZE_MAX = (1 << 24) - 1
_STRIP_PX = 24  # collapsed vertical-strip thickness
_BAR_PX = 26    # horizontal title-bar height

# Object name a pane gives a child widget to have it rendered beside the
# frame's title instead of inside the pane. One name rather than a per-pane
# argument, so a pane stays buildable on its own and the frame stays generic.
HEADER_EXTRAS = "pane-header-extras"


class PanelFrame(QFrame):
    """Collapsible + detachable container for one pane (or sub-splitter)."""

    state_changed = pyqtSignal()

    def __init__(
        self,
        inner: QWidget,
        title: str = "",
        *,
        edge: str = "top",            # "top" | "bottom" | "left" | "right"
        collapsible: bool = True,
        detachable: bool = True,
        hide_inner_header: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("panel-frame")
        self._inner = inner
        self._title = title
        self._edge = edge
        self._vertical_fold = edge in ("left", "right")
        self._collapsible = collapsible
        self._detachable = detachable
        self._collapsed = False
        self._detached: Optional[QDialog] = None
        self._splitter: Optional[QSplitter] = None
        self._grow_target: Optional[QWidget] = None
        self._saved_extent: Optional[int] = None

        # Buttons the pane wants beside the title rather than under it. Kept
        # with the layout they came from, so detaching can hand them back.
        self._extras: Optional[QWidget] = None
        self._extras_home = None

        if hide_inner_header:
            hdr = inner.findChild(PaneHeader)
            if hdr is not None:
                hdr.setVisible(False)
            extras = inner.findChild(QWidget, HEADER_EXTRAS)
            if extras is not None:
                self._extras = extras
                self._extras_home = extras.parentWidget().layout()

        # Placeholder shown in place of the body while detached.
        self._placeholder = QLabel("Detached - close the window to dock it back.")
        self._placeholder.setObjectName("pane-hint")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setVisible(False)

        self._build_controls()
        self._assemble()
        self._refresh_icons()

    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        self._caret = QToolButton()
        self._caret.setObjectName("panel-frame-caret")
        self._caret.setAutoRaise(True)
        self._caret.setCursor(Qt.CursorShape.PointingHandCursor)
        self._caret.setToolTip("Collapse / expand")
        self._caret.clicked.connect(self.toggle_collapsed)

        self._detach_btn = QToolButton()
        self._detach_btn.setObjectName("panel-frame-detach")
        self._detach_btn.setAutoRaise(True)
        self._detach_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._detach_btn.setToolTip("Detach into a floating window")
        self._detach_btn.clicked.connect(self.toggle_detached)

        self._title_lbl = QLabel(self._title.upper())
        self._title_lbl.setObjectName("pane-h5")

        # Horizontal title bar. For top/bottom panels it also carries the
        # caret; for left/right panels the caret lives in the side strip and
        # the bar holds just the title + detach.
        self._bar = QFrame()
        self._bar.setObjectName("panel-frame-header")
        self._bar.setFixedHeight(_BAR_PX)
        bl = QHBoxLayout(self._bar)
        bl.setContentsMargins(6, 0, 4, 0)
        bl.setSpacing(4)
        if not self._vertical_fold:
            bl.addWidget(self._caret)
        bl.addWidget(self._title_lbl, 1)
        if self._extras is not None:
            bl.addWidget(self._extras)
        bl.addWidget(self._detach_btn)

        # Side strip (left/right only): a thin vertical bar with the caret
        # centred vertically.
        self._strip: Optional[QFrame] = None
        if self._vertical_fold:
            self._strip = QFrame()
            self._strip.setObjectName("panel-frame-strip")
            self._strip.setFixedWidth(_STRIP_PX)
            self._strip.setToolTip(self._title)
            sl = QVBoxLayout(self._strip)
            sl.setContentsMargins(0, 4, 0, 4)
            sl.setSpacing(4)
            sl.addStretch(1)
            sl.addWidget(self._caret, 0, Qt.AlignmentFlag.AlignHCenter)
            sl.addStretch(1)

        self._caret.setVisible(self._collapsible)
        self._detach_btn.setVisible(self._detachable)

    def _assemble(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if self._vertical_fold:
            # [ title bar              ]
            # [ strip | content ]   (strip on the panel's outer edge)
            outer.addWidget(self._bar)
            self._body_row = QWidget()
            brow = QHBoxLayout(self._body_row)
            brow.setContentsMargins(0, 0, 0, 0)
            brow.setSpacing(0)
            if self._edge == "left":
                brow.addWidget(self._strip)
                brow.addWidget(self._inner, 1)
                brow.addWidget(self._placeholder, 1)
                self._content_index = 1
            else:  # right
                brow.addWidget(self._inner, 1)
                brow.addWidget(self._placeholder, 1)
                brow.addWidget(self._strip)
                self._content_index = 0
            outer.addWidget(self._body_row, 1)
        else:
            self._body_row = None
            if self._edge == "bottom":
                outer.addWidget(self._inner, 1)
                outer.addWidget(self._placeholder, 1)
                outer.addWidget(self._bar)
                self._content_index = 0
            else:  # top
                outer.addWidget(self._bar)
                outer.addWidget(self._inner, 1)
                outer.addWidget(self._placeholder, 1)
                self._content_index = 1

    # ------------------------------------------------------------------
    # Splitter wiring
    # ------------------------------------------------------------------

    def attach_splitter(self, splitter: QSplitter, grow_target: Optional[QWidget] = None) -> None:
        self._splitter = splitter
        self._grow_target = grow_target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inner(self) -> QWidget:
        return self._inner

    def is_collapsed(self) -> bool:
        return self._collapsed

    def is_detached(self) -> bool:
        return self._detached is not None

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        if not self._collapsible or collapsed == self._collapsed or self.is_detached():
            return
        self._collapsed = collapsed
        self._inner.setVisible(not collapsed)
        if self._vertical_fold:
            # Fold to just the side strip: hide the top title bar too.
            self._bar.setVisible(not collapsed)
        self._apply_fold(collapsed)
        self._refresh_icons()
        self.state_changed.emit()

    def _grow_index(self, sizes_len: int, idx: int) -> Optional[int]:
        if self._grow_target is not None and self._splitter is not None:
            gi = self._splitter.indexOf(self._grow_target)
            if gi != -1 and gi != idx:
                return gi
        others = [i for i in range(sizes_len) if i != idx]
        return max(others, key=lambda i: self._splitter.sizes()[i]) if others else None

    def _apply_fold(self, collapsed: bool) -> None:
        sp = self._splitter
        if self._vertical_fold:
            if collapsed:
                self.setFixedWidth(_STRIP_PX)
            else:
                self.setMinimumWidth(0)
                self.setMaximumWidth(_QWIDGETSIZE_MAX)
        else:
            self.setMaximumHeight(_BAR_PX + 4 if collapsed else _QWIDGETSIZE_MAX)

        if sp is None or sp.indexOf(self) == -1:
            return
        sizes = sp.sizes()
        idx = sp.indexOf(self)
        grow = self._grow_index(len(sizes), idx)
        if grow is None:
            return
        strip = _STRIP_PX if self._vertical_fold else _BAR_PX + 4
        if collapsed:
            self._saved_extent = sizes[idx]
            delta = sizes[idx] - strip
            sizes[idx] = strip
            sizes[grow] += delta
        else:
            restore = self._saved_extent or 280
            delta = restore - sizes[idx]
            sizes[idx] = restore
            sizes[grow] = max(strip, sizes[grow] - delta)
        sp.setSizes(sizes)

    def toggle_detached(self) -> None:
        if self.is_detached():
            self.reattach()
        else:
            self.detach()

    def detach(self) -> None:
        if self.is_detached():
            return
        if self._collapsed:
            self.set_collapsed(False)

        win = QDialog(self.window())
        win.setWindowTitle(f"BIDS-Manager - {self._title}" if self._title else "BIDS-Manager")
        win.setObjectName("panel-frame-float")
        # A bare QDialog gets only a close button on Linux / Windows. Promote
        # it to a normal top-level window so it carries minimize + maximize
        # buttons and can be maximised / tiled like any other window.
        win.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        lay = QVBoxLayout(win)
        lay.setContentsMargins(0, 0, 0, 0)
        # Give the pane its buttons back before it leaves, or the floating
        # window arrives without the controls it owns while they sit uselessly
        # on a title bar whose body is gone.
        self._return_extras()
        self._inner.setParent(win)
        lay.addWidget(self._inner)
        self._inner.setVisible(True)
        win.resize(max(self._inner.width(), 520), max(self._inner.height(), 380))
        win.finished.connect(lambda _=0: self.reattach())
        self._detached = win

        self._placeholder.setVisible(True)
        self._caret.setVisible(False)
        self._detach_btn.setToolTip("Re-dock the floating window")
        self._refresh_icons()
        win.show()
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Header extras: buttons a pane owns but that belong beside the title
    # ------------------------------------------------------------------

    def _return_extras(self) -> None:
        """Put the pane's buttons back where the pane built them."""
        if self._extras is None or self._extras_home is None:
            return
        self._extras_home.addWidget(self._extras)
        self._extras.setVisible(True)

    def _take_extras(self) -> None:
        """Lift them back into the title bar."""
        if self._extras is None:
            return
        layout = self._bar.layout()
        layout.insertWidget(layout.count() - 1, self._extras)
        self._extras.setVisible(True)

    def reattach(self) -> None:
        if not self.is_detached():
            return
        win = self._detached
        self._detached = None
        # Move the body back next to its control, at its original position.
        parent_layout = self._body_row.layout() if self._vertical_fold else self.layout()
        self._inner.setParent(self._body_row if self._vertical_fold else self)
        parent_layout.insertWidget(self._content_index, self._inner, 1)
        self._inner.setVisible(True)
        self._take_extras()
        self._placeholder.setVisible(False)
        self._caret.setVisible(self._collapsible)
        self._detach_btn.setVisible(self._detachable)
        self._detach_btn.setToolTip("Detach into a floating window")
        self._refresh_icons()
        try:
            win.finished.disconnect()
        except TypeError:
            pass
        win.close()
        win.deleteLater()
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def repaint_for_palette(self, pal: dict) -> None:
        self._refresh_icons()
        inner_repaint = getattr(self._inner, "repaint_for_palette", None)
        if callable(inner_repaint):
            inner_repaint(pal)

    def _refresh_icons(self) -> None:
        if self._vertical_fold:
            if self._edge == "left":
                glyph = "panel_expand" if self._collapsed else "chevron_left"
            else:  # right
                glyph = "chevron_left" if self._collapsed else "panel_expand"
        else:
            glyph = "panel_expand" if self._collapsed else "panel_collapse"
        self._caret.setIcon(icons.icon(glyph))
        self._detach_btn.setIcon(
            icons.icon("reattach" if self.is_detached() else "detach")
        )


__all__ = ["PanelFrame"]

"""Small reusable widgets used by both Converter and Editor views.

Lift-and-shift from ``inspector_proto/proto.py`` (lines 181-227 of the
prototype). Same QSS object names, same paint behaviour. No logic
change here — these are pure presentation primitives the higher-level
views compose.

Object-name conventions (consumed by ``theme.qss``):

* ``Chip``          → ``chipKind`` property selects color (``default``,
  ``success``, ``warn``, ``err``, ``purple``, ``teal``).
* ``VSep``          → ``vsep`` — 1px vertical divider used in toolbars.
* ``PaneHeader``    → ``pane-h5`` — 28px uppercase pane title.
* ``PathBar``       → ``pathbar`` (root), ``path-label`` (left text),
  ``path-field`` (readonly QLineEdit), ``tb-btn-ghost`` (change button).

All four respect palette swaps through the QSS alone; no per-widget
``repaint_for_palette`` is needed.
"""

from __future__ import annotations

from typing import Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
)


class Chip(QLabel):
    """A small pill label. ``kind`` selects color via QSS property selector.

    Known kinds (from ``theme.qss``): ``default``, ``success``, ``warn``,
    ``err``, ``purple``, ``teal``. Unknown kinds fall back to ``default``.

    Emits :pyattr:`clicked` on a left-mouse release **only if**
    :meth:`set_clickable` is called with ``True``. The default chip is
    a passive label (no signal, no cursor change) so existing callers
    don't accidentally pick up click semantics.
    """

    clicked = pyqtSignal()

    def __init__(self, text: str, kind: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setProperty("chipKind", kind or "default")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self._clickable = False

    def set_clickable(self, clickable: bool) -> None:
        """Toggle whether mouse presses on the chip emit ``clicked``.

        Also flips the cursor to a hand pointer so users can tell the
        chip is an actionable element.
        """
        self._clickable = clickable
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if clickable else
            Qt.CursorShape.ArrowCursor
        )

    def mouseReleaseEvent(self, event):  # noqa: N802 — Qt naming
        if self._clickable and event.button() == Qt.MouseButton.LeftButton \
                and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class VSep(QFrame):
    """1px-wide vertical separator. Used between toolbar groups."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("vsep")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedWidth(1)


class PaneHeader(QLabel):
    """28px uppercase header used at the top of every splitter pane."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text.upper(), parent)
        self.setObjectName("pane-h5")
        self.setFixedHeight(28)


class PathBar(QFrame):
    """One-row "label · value · trailing chips · change… button" strip.

    Used for the Converter view's raw-input and BIDS-output paths, and
    for the Editor view's BIDS-root path. The value field is read-only
    by design — path edits happen via the ``change…`` button (file
    dialog wiring lives in the view that owns the bar).

    ``ok=True`` paints a ✔ prefix, ``ok=False`` a ○ prefix (these are
    inline characters today; the architecture doc allows replacing with
    bundled SVGs later).
    """

    def __init__(
        self,
        label: str,
        value: str,
        ok: bool = False,
        trailing_chips: Sequence[tuple[str, str]] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("pathbar")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 9, 14, 9)
        lay.setSpacing(10)

        lbl = QLabel(label)
        lbl.setObjectName("path-label")
        lbl.setMinimumWidth(80)
        lay.addWidget(lbl)

        ico = "✔  " if ok else "○  "
        field = QLineEdit(f"{ico}{value}")
        field.setObjectName("path-field")
        field.setReadOnly(True)
        lay.addWidget(field, 1)

        for chip_kind, chip_text in trailing_chips or ():
            lay.addWidget(Chip(chip_text, chip_kind))

        btn = QPushButton("change…")
        btn.setObjectName("tb-btn-ghost")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(btn)

        # Expose the change-button + value field so views can
        # ``.clicked.connect(...)`` / ``set_value(...)`` without
        # fishing through the layout.
        self.change_button = btn
        self._field = field

    def set_value(self, value: str, *, ok: bool = False) -> None:
        """Update the displayed path. Re-renders the ✔/○ prefix as well."""
        ico = "✔  " if ok else "○  "
        self._field.setText(f"{ico}{value}")

    def value(self) -> str:
        """Return the current value without the ✔/○ prefix."""
        raw = self._field.text()
        return raw[3:] if raw[:1] in ("✔", "○") else raw


__all__ = ["Chip", "PaneHeader", "PathBar", "VSep"]

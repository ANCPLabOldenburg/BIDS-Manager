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
* ``ElidedLabel``   → keeps the object name it is given; used wherever
  runtime text (a path, a log line) must never widen the window.

All four respect palette swaps through the QSS alone; no per-widget
``repaint_for_palette`` is needed.
"""

from __future__ import annotations

from typing import Sequence

from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPalette, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStyleOption,
)


def _chip_qcolor(token: str) -> QColor:
    """Parse a palette token (hex or ``rgba(r,g,b,a)``) into a QColor.

    ``QColor`` cannot parse the CSS ``rgba(...)`` strings the palette uses
    for translucent tints, so handle those explicitly.
    """
    s = (token or "").strip()
    if s.startswith(("rgba(", "rgb(")):
        nums = s[s.index("(") + 1: s.rindex(")")].split(",")
        try:
            r, g, b = (int(float(nums[i])) for i in range(3))
            a = int(float(nums[3]) * 255) if len(nums) > 3 else 255
            return QColor(r, g, b, a)
        except (ValueError, IndexError):
            return QColor("#000000")
    return QColor(s or "#000000")


class Chip(QLabel):
    """A small pill label that **paints its own rounded background**.

    ``kind`` selects the colour set: ``default``, ``success``, ``warn``,
    ``err``, ``accent``, ``purple``, ``teal``.

    The pill is drawn in :meth:`paintEvent` with a corner radius of
    ``height / 2``, so it is ALWAYS a perfect pill at any font size /
    DPI / platform - QSS ``border-radius`` proved unreliable across
    macOS/Linux/Windows + font scales. Colours are read live from the
    active palette (:func:`bidsmgr.gui.theme_manager.CUR`) at paint time,
    so a theme swap (which triggers ``update()``) recolours automatically.

    Emits :pyattr:`clicked` on a left-mouse release **only if**
    :meth:`set_clickable` is called with ``True``.
    """

    clicked = pyqtSignal()

    # kind -> (background token, foreground token, border token).
    _KIND_TOKENS = {
        "default": ("surface3", "dim", "border"),
        "success": ("success_bg", "success", "success_border"),
        "warn":    ("warning_bg", "warning", "warning_border"),
        "err":     ("error_bg", "error", "error_border"),
        "accent":  ("accent_bg", "accent", "accent_border"),
        "purple":  ("purple_bg", "purple", "purple_border"),
        "teal":    ("teal_bg", "teal", "teal_border"),
    }

    def __init__(self, text: str, kind: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self._kind = kind or "default"
        # Keep the property so any QSS targeting chipKind still matches.
        self.setProperty("chipKind", self._kind)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        # Transparent widget background so ONLY the painted pill shows - the
        # parent (toolbar / panel) shows through the rounded corners. Without
        # this Qt fills the widget rect first, leaving a square behind the pill.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Horizontal breathing room + a hair of vertical so the pill reads.
        self.setContentsMargins(9, 1, 9, 1)
        self._clickable = False

    def setText(self, text: str) -> None:  # noqa: N802 — Qt naming
        super().setText(text)
        self.updateGeometry()
        self.update()

    def set_clickable(self, clickable: bool) -> None:
        """Toggle whether mouse presses on the chip emit ``clicked``."""
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

    def paintEvent(self, event):  # noqa: N802 — Qt naming
        from ..theme_manager import CUR

        pal = CUR()
        bg_t, fg_t, br_t = self._KIND_TOKENS.get(
            self._kind, self._KIND_TOKENS["default"]
        )
        bg = _chip_qcolor(pal.get(bg_t, "#1c2128"))
        fg = _chip_qcolor(pal.get(fg_t, "#8b949e"))
        border = _chip_qcolor(pal.get(br_t, "#21262d"))

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2.0
        p.setBrush(QBrush(bg))
        p.setPen(QPen(border, 1))
        p.drawRoundedRect(rect, radius, radius)
        p.setPen(QPen(fg))
        p.setFont(self.font())
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


class VSep(QFrame):
    """1px-wide vertical separator. Used between toolbar groups."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("vsep")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedWidth(1)


class PaneHeader(QLabel):
    """28px uppercase header used at the top of every splitter pane.

    It elides rather than setting a floor. A pane header carries a name and
    often a path, and a plain QLabel reports all of it as its MINIMUM width, so
    the header alone stopped the Properties pane narrowing past 191 px however
    small its contents could go.
    """

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text.upper(), parent)
        self.setObjectName("pane-h5")
        self.setFixedHeight(28)
        self._full = text.upper()

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        self._full = text
        super().setText(text)
        self.setToolTip(text)
        self._elide()

    def minimumSizeHint(self):  # noqa: N802 - Qt override
        hint = super().minimumSizeHint()
        hint.setWidth(0)
        return hint

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        room = max(0, self.width() - 12)
        if not self._full or not room:
            return
        shown = self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideRight, room,
        )
        if shown != QLabel.text(self):
            QLabel.setText(self, shown)


class ElidedLabel(QLabel):
    """A single-line label that can never widen the window.

    A plain ``QLabel`` reports its full text width as *both* its size
    hint and its **minimum** size hint, so a long runtime string (a scan
    path, a converter log line, a traceback in the status bar) raises the
    layout minimum of every ancestor and Qt grows the window to honour
    it. That is why status text used to be hard-truncated at an arbitrary
    character count — a workaround that still let the window jump.

    This label reports a zero minimum width instead and elides its text
    to whatever width the layout happens to grant it, re-eliding on every
    repaint (so a window resize is enough to reveal more of it). The full
    string stays available through :meth:`text` and is mirrored into the
    tooltip, so nothing is lost.
    """

    def __init__(
        self,
        text: str = "",
        parent=None,
        *,
        mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight,
    ) -> None:
        super().__init__(text, parent)
        self._mode = mode
        # Preferred (not Ignored): the layout still offers the label the
        # room its text wants when there is spare width, but ShrinkFlag
        # lets it fall back to ``minimumSizeHint`` (width 0) when there
        # is not, instead of pushing the shortfall onto the window.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setToolTip(text)

    def setText(self, text: str) -> None:  # noqa: N802 — Qt naming
        super().setText(text)
        # Hovering recovers whatever the elision dropped.
        self.setToolTip(text)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt naming
        """Claim no horizontal room at all; keep the natural height."""
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, _evt) -> None:  # noqa: N802 — Qt naming
        painter = QPainter(self)
        opt = QStyleOption()
        # initFrom carries the QSS-resolved palette and font across, so
        # the elided text keeps the colour theme.qss assigned to it.
        opt.initFrom(self)
        elided = painter.fontMetrics().elidedText(
            self.text(), self._mode, self.width()
        )
        self.style().drawItemText(
            painter,
            self.rect(),
            int(self.alignment()),
            opt.palette,
            self.isEnabled(),
            elided,
            QPalette.ColorRole.WindowText,
        )


class ElidedPushButton(QPushButton):
    """A button whose label shortens instead of widening its window.

    A ``QPushButton`` asks for the width of its whole label and, given less,
    clips it mid-glyph with no ellipsis: the user sees a word cut in half and
    cannot tell there is more. That is what a long BIDS basename or a full
    file path did to both issue dialogs.

    This one reports a zero minimum width and elides at paint time, leaving
    :meth:`text` holding the full string. Callers (and tests) that read
    ``text()`` still get what was set; only the pixels shorten.
    """

    def __init__(
        self,
        text: str = "",
        parent=None,
        *,
        mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle,
    ) -> None:
        super().__init__(text, parent)
        self._mode = mode
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred,
        )
        self.setToolTip(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt override
        super().setText(text)
        self.setToolTip(text)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, _evt) -> None:  # noqa: N802 - Qt override
        from PyQt6.QtWidgets import QStyleOptionButton, QStylePainter

        painter = QStylePainter(self)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        # Elide against the room the style leaves for the label, not the
        # whole widget: a framed button spends some of its width on the
        # border and padding.
        room = self.style().subElementRect(
            self.style().SubElement.SE_PushButtonContents, option, self,
        ).width()
        option.text = option.fontMetrics.elidedText(
            self.text(), self._mode, max(room, 0),
        )
        painter.drawControl(
            self.style().ControlElement.CE_PushButton, option,
        )


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


__all__ = [
    "Chip",
    "ElidedLabel",
    "ElidedPushButton",
    "PaneHeader",
    "PathBar",
    "VSep",
]

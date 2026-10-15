"""What a quality measure or a QC plot means, a click away.

The info icon beside a measure, a group of measures or a QC plot opens this:
the measure's full explanation (``bidsmgr.qc.explain``) in a rounded popup
the app's way (``menus.popup_menu``), sized to the font setting, closed by
a click elsewhere or Escape. The hover text is the explanation's first
sentence or two (:func:`hover_text`); the popup is the rest.
"""

from __future__ import annotations

import html
from typing import Optional

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget, QWidgetAction

from ....qc import explain
from .. import fonts
from ..bridge import ThemeHub

#: The sections of an explanation, in the order they are read.
SECTIONS = (("measures", "What it measures"), ("computed", "How it is computed"),
            ("reading", "How to read it"), ("causes", "What makes it worse"),
            ("caveats", "Watch out"), ("mriqc", "MRIQC"))


def hover_text(key: str, *, better: str = "", extra: str = "") -> str:
    """The hover text for ``key``: the explanation's short text, which way
    is better, and where the full one is."""
    exp = explain.lookup(key)
    parts = []
    if exp is not None:
        parts.append(exp.short)
    if better and (exp is None or "better" not in exp.short.lower()):
        parts.append(f"{better.capitalize()} is better.")
    if extra:
        parts.append(extra)
    if exp is not None:
        parts.append("The info icon explains it in full.")
    return " ".join(parts)


def explanation_html(key: str, *, here: str = "", title: str = "", fallback: str = "") -> str:
    """The popup's rich text for ``key`` (``fallback`` when there is no
    explanation), with what was measured HERE (``here``) at the end."""
    theme = ThemeHub.instance().theme
    exp = explain.lookup(key)
    esc = html.escape
    head = esc(title or (exp.title if exp is not None else "")) or "About this"
    out = [f"<div style='font-weight:600; font-size:{fonts.px(14)}px'>{head}</div>"]

    def section(name: str, body: str) -> None:
        out.append(f"<div style='margin-top:{fonts.px(8)}px; color:{theme.dim}; "
                   f"font-weight:600'>{esc(name)}</div>")
        out.append(f"<div style='margin-top:2px'>{esc(body)}</div>")

    if exp is None:
        # A fallback may hold several paragraphs, separated by a blank line.
        for paragraph in (fallback or "").split("\n\n"):
            if paragraph.strip():
                out.append(f"<div style='margin-top:{fonts.px(6)}px'>{esc(paragraph.strip())}</div>")
    else:
        out.append(f"<div style='margin-top:{fonts.px(6)}px'>{esc(exp.short)}</div>")
        for field, name in SECTIONS:
            body = getattr(exp, field, "")
            if body:
                section(name, body)
        if exp.references:
            section("References", "; ".join(exp.references))
    if here:
        section("In this image", here)
    return "".join(out)


def show(anchor: QWidget, key: str, *, at: Optional[QPoint] = None, here: str = "",
         title: str = "", fallback: str = "") -> None:
    """Open the explanation of ``key`` under ``anchor`` (or at ``at``, global)."""
    from ..menus import popup_menu

    menu = popup_menu(anchor)
    body = QFrame()
    body.setObjectName("viz-explain")
    lay = QVBoxLayout(body)
    pad = fonts.px(12)
    lay.setContentsMargins(pad, fonts.px(10), pad, fonts.px(10))
    label = QLabel(explanation_html(key, here=here, title=title, fallback=fallback))
    label.setObjectName("viz-explain-text")
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    width = fonts.px(440)
    label.setFixedWidth(width)
    lay.addWidget(label)
    # Taller than most of the screen: it scrolls inside the popup.
    screen = anchor.screen() or None
    room = int(screen.availableGeometry().height() * 0.7) if screen is not None else 700
    label.adjustSize()
    if label.sizeHint().height() + 2 * pad > room:
        scroll = QScrollArea()
        scroll.setObjectName("viz-explain-scroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(body)
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedSize(width + 2 * pad + fonts.px(14), room)
        holder = scroll
    else:
        holder = body
    action = QWidgetAction(menu)
    action.setDefaultWidget(holder)
    menu.addAction(action)
    where = at if at is not None else anchor.mapToGlobal(QPoint(0, anchor.height()))
    menu.popup(where)
    global _LAST
    _LAST = (menu, label)


#: The last popup shown and its label (tests).
_LAST: Optional[tuple] = None


def last_shown() -> Optional[tuple]:
    """``(menu, label)`` of the last explanation opened (tests)."""
    return _LAST


__all__ = ["SECTIONS", "explanation_html", "hover_text", "last_shown", "show"]

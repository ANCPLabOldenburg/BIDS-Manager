"""A card around any widget: the rounded, outlined surface the main panels
sit on (``QFrame[card="true"]`` in the stylesheet), for dialogs and windows
that should look like the rest of the app."""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QWidget


def card(inner: QWidget, *, padding: int = 6) -> QFrame:
    frame = QFrame()
    frame.setProperty("card", True)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(padding, padding, padding, padding)
    lay.setSpacing(0)
    lay.addWidget(inner)
    return frame


__all__ = ["card"]

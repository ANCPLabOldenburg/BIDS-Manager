"""The command line of the view on screen, to copy into a shell or a script.

What FSLeyes calls "show command line": the files and the commands that
reproduce exactly this view, written by the Qt-free
:mod:`bidsmgr.viz.reproduce`. Three forms, one per tab: open it in a viewer,
render it to a PNG with no window, or the same from Python.
"""

from __future__ import annotations

from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout,
)

from ....viz import reproduce


class CommandLineDialog(QDialog):
    """Three tabs of read-only text and a Copy button."""

    def __init__(self, store, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Command line of this view")
        self.resize(720, 360)
        lay = QVBoxLayout(self)
        text = ("The same files, looks and crosshair, from a terminal or a script. "
                "Render writes the 2-D planes to a PNG with no window.")
        skipped = reproduce.plan(store)["skipped"]
        if skipped:
            text += (f" Not reproduced: {', '.join(skipped)} (computed in the viewer, "
                     "there is no file to open).")
        note = QLabel(text)
        note.setObjectName("sidecar-footer-summary")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.tabs = QTabWidget()
        from .. import fonts

        mono = fonts.font(12, mono=True)   # at the app's font size
        self.texts: dict[str, str] = {
            "Open": reproduce.shell(store),
            "Render": reproduce.shell(store, render="figure.png"),
            "Python": reproduce.python(store),
        }
        for title, text in self.texts.items():
            box = QPlainTextEdit(text)
            box.setReadOnly(True)
            box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            box.setFont(mono)
            self.tabs.addTab(box, title)
        lay.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.copy_button = QPushButton("Copy")
        self.copy_button.setObjectName("tb-btn-primary")
        self.copy_button.clicked.connect(self.copy)
        row.addWidget(self.copy_button)
        close = QPushButton("Close")
        close.setObjectName("tb-btn")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)

    def current_text(self) -> str:
        return list(self.texts.values())[self.tabs.currentIndex()]

    def copy(self) -> None:
        QGuiApplication.clipboard().setText(self.current_text())
        self.copy_button.setText("Copied")


__all__ = ["CommandLineDialog"]

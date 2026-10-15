"""Editor centre pane for documents: README, CHANGES, LICENSE, Markdown,
HTML and plain text (``viz.data.formats.document_kind``).

Markdown and HTML open RENDERED (``markdown_view.MarkdownView``: a
repository page's typography in the app's theme), with the text a click
away; README (which
BIDS allows as plain text or Markdown) opens as text with the rendering a
click away. Plain text is editable, saved as one undoable operation of the
dataset when it sits in one (``project.operations``), and never in a folder
open for viewing only. A file too large to hold, or not valid UTF-8, opens
read only rather than risk writing back something it is not.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .markdown_view import MarkdownView
from .primitives import Chip, PaneHeader

log = logging.getLogger(__name__)

#: Larger than this, only the start is shown, read only.
MAX_BYTES = 4 * 1024 * 1024


def _size_text(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


class TextPane(QWidget):
    """A document: rendered (Markdown, HTML) and as text, editable text."""

    file_saved = pyqtSignal(Path)
    #: A link in the rendered document to a file beside it.
    file_requested = pyqtSignal(Path)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._path: Optional[Path] = None
        self._root: Optional[Path] = None
        self._kind = ""
        self._original = ""
        self._writable = False
        #: A folder open for viewing: shown, never written.
        self._read_only = False

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._header = PaneHeader("Document")
        v.addWidget(self._header)

        bar = QFrame()
        bar.setObjectName("sidecar-toolbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 6, 14, 6)
        bl.setSpacing(8)
        self._rendered_btn = QPushButton("Rendered")
        self._text_btn = QPushButton("Text")
        group = QButtonGroup(self)
        for b, tip in ((self._rendered_btn, "The document as it reads"),
                       (self._text_btn, "The document's text, as stored")):
            b.setObjectName("tb-btn-toggle")
            b.setCheckable(True)
            b.setToolTip(tip)
            group.addButton(b)
            bl.addWidget(b)
        group.setExclusive(True)
        self._rendered_btn.toggled.connect(self._on_view)
        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        bl.addWidget(self._status, 1)
        # A saved report (validation_report.html) is styled for a browser.
        self._browser_btn = QPushButton("Open in browser")
        self._browser_btn.setObjectName("tb-btn")
        self._browser_btn.setToolTip("Open the page in your web browser, which shows it "
                                     "as it was made to look")
        self._browser_btn.clicked.connect(
            lambda: self._path is not None and QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self._path))))
        bl.addWidget(self._browser_btn)
        self._dirty = Chip("changed", "warn")
        self._dirty.setVisible(False)
        bl.addWidget(self._dirty)
        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setObjectName("tb-btn")
        self._revert_btn.clicked.connect(self.revert)
        bl.addWidget(self._revert_btn)
        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("tb-btn")
        self._save_btn.clicked.connect(self.save)
        bl.addWidget(self._save_btn)
        v.addWidget(bar)

        self._stack = QStackedWidget()
        self._browser = MarkdownView()
        self._browser.file_requested.connect(self.file_requested)
        self._editor = QPlainTextEdit()
        self._editor.setObjectName("text-document")
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._editor.textChanged.connect(self._sync)
        self._stack.addWidget(self._browser)
        self._stack.addWidget(self._editor)
        v.addWidget(self._stack, 1)
        self._sync()

    # -- binding -----------------------------------------------------------

    def set_read_only(self, on: bool) -> None:
        self._read_only = bool(on)
        self._sync()

    def set_file(self, path: Optional[Path], root: Optional[Path]) -> None:
        from ...viz.data.formats import document_kind

        self._path = Path(path) if path is not None else None
        self._root = Path(root) if root is not None else None
        self._kind = document_kind(self._path) if self._path is not None else ""
        self._header.setText(self._path.name if self._path is not None else "Document")
        self.reload()

    def current_file(self) -> Optional[Path]:
        return self._path

    def reload(self) -> None:
        """Read the file again (Revert)."""
        text, note, self._writable = "", "", False
        if self._path is not None and self._path.is_file():
            size = self._path.stat().st_size
            raw = self._path.read_bytes()[:MAX_BYTES]
            try:
                text = raw.decode("utf-8")
                self._writable = size <= MAX_BYTES
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
                note = "not UTF-8: shown read only"
            if size > MAX_BYTES:
                note = f"the first {_size_text(MAX_BYTES)} of {_size_text(size)}, read only"
            lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
            self._status.setText(", ".join(x for x in (
                f"{lines:,} lines, {_size_text(size)}", note) if x))
        else:
            self._status.setText("")
        self._original = text
        self._editor.blockSignals(True)
        self._editor.setPlainText(text)
        self._editor.blockSignals(False)
        renders = self._kind in ("markdown", "html") or (
            self._path is not None and self._path.name.lower() == "readme")
        self._rendered_btn.setVisible(renders)
        self._text_btn.setVisible(renders)
        self._browser_btn.setVisible(self._kind == "html")
        rendered = self._kind in ("markdown", "html")
        (self._rendered_btn if rendered else self._text_btn).setChecked(True)
        self._on_view(rendered)
        self._sync()

    # -- views ---------------------------------------------------------------

    def _on_view(self, rendered: bool) -> None:
        if rendered and not self._rendered_btn.isHidden():
            text = self._editor.toPlainText()
            base = self._path.parent if self._path is not None else None
            if self._kind == "html":
                self._browser.set_html(text, base)
            else:
                self._browser.set_markdown(text, base)
            self._stack.setCurrentWidget(self._browser)
        else:
            self._stack.setCurrentWidget(self._editor)

    def is_rendered(self) -> bool:
        return self._stack.currentWidget() is self._browser

    def _editable(self) -> bool:
        return (self._path is not None and self._writable and not self._read_only
                and self._kind != "html")

    def _sync(self) -> None:
        editable = self._editable()
        self._editor.setReadOnly(not editable)
        dirty = editable and self._editor.toPlainText() != self._original
        self._dirty.setVisible(dirty)
        self._save_btn.setVisible(editable)
        self._revert_btn.setVisible(editable)
        self._save_btn.setEnabled(dirty)
        self._revert_btn.setEnabled(dirty)

    def is_dirty(self) -> bool:
        return self._editable() and self._editor.toPlainText() != self._original

    # -- saving --------------------------------------------------------------

    def revert(self) -> None:
        self.reload()

    def save(self) -> bool:
        """Write the text, as one undoable operation of the dataset."""
        if not self.is_dirty():
            return False
        text = self._editor.toPlainText()
        try:
            if self._root is not None:
                from ...project.operations import begin_operation

                with begin_operation(self._root, f"Edit {self._path.name}") as op:
                    op.write_text(self._path, text)
            else:
                self._path.write_text(text, encoding="utf-8")
        except OSError as exc:
            log.warning("could not save %s: %s", self._path, exc)
            self._status.setText(f"save failed: {exc}")
            return False
        self._original = text
        self._sync()
        self.file_saved.emit(self._path)
        return True

    def repaint_for_palette(self, pal: dict) -> None:
        """QSS-styled children are re-polished; the rendered document
        re-renders itself on the theme hub."""
        del pal
        for widget in (self, self._browser, self._editor):
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)


__all__ = ["MAX_BYTES", "TextPane"]

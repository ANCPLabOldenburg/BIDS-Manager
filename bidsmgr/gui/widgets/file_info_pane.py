"""Editor centre pane for a file BIDS Manager has no view of (an ``.eeg``
data block, a ``.pdf``, an archive): what it is, how large, when it was
changed, and the system's own app to open it."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from .primitives import PaneHeader


def _size_text(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


#: What a file is, by extension, where BIDS gives it a role.
KNOWN = {
    ".eeg": "BrainVision data (open the .vhdr beside it to view the recording)",
    ".vmrk": "BrainVision markers",
    ".pdf": "PDF document",
    ".zip": "Archive",
    ".gz": "Compressed file",
    ".mat": "MATLAB data",
    ".gii": "GIFTI surface or data",
    ".h5": "HDF5 data",
    ".npy": "NumPy array",
}


class FileInfoPane(QWidget):
    """A file shown by its facts, with the system's app to open it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._path: Optional[Path] = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._header = PaneHeader("File")
        v.addWidget(self._header)
        body = QWidget()
        body.setObjectName("cff-body")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)
        hint = QLabel("BIDS Manager has no view of this kind of file.")
        hint.setObjectName("dlg-hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._kind = QLabel("")
        self._size = QLabel("")
        self._changed = QLabel("")
        self._where = QLabel("")
        self._where.setWordWrap(True)
        for label, w in (("Kind", self._kind), ("Size", self._size),
                         ("Changed", self._changed), ("Folder", self._where)):
            form.addRow(label, w)
        lay.addLayout(form)
        row = QHBoxLayout()
        self.open_button = QPushButton("Open with the system's app")
        self.open_button.setObjectName("tb-btn")
        self.open_button.setToolTip("Open the file in the application your computer uses "
                                    "for it")
        self.open_button.clicked.connect(self._open)
        row.addWidget(self.open_button)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)
        v.addWidget(body, 1)

    def set_read_only(self, on: bool) -> None:
        """Always read only; here for the Editor's mode switch."""
        del on

    def set_file(self, path: Optional[Path], root: Optional[Path]) -> None:
        del root
        self._path = Path(path) if path is not None else None
        p = self._path
        if p is None or not p.exists():
            for w in (self._kind, self._size, self._changed, self._where):
                w.setText("")
            self.open_button.setEnabled(False)
            return
        self._header.setText(p.name)
        from ...viz.bids import full_ext

        ext = full_ext(p)
        self._kind.setText(KNOWN.get(ext, KNOWN.get(p.suffix.lower(), "")) or (
            f"{ext} file" if ext else "File"))
        st = p.stat()
        self._size.setText(_size_text(st.st_size))
        self._changed.setText(_dt.datetime.fromtimestamp(st.st_mtime).strftime(
            "%Y-%m-%d %H:%M"))
        self._where.setText(str(p.parent))
        self.open_button.setEnabled(True)

    def current_file(self) -> Optional[Path]:
        return self._path

    def _open(self) -> None:
        if self._path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path)))

    def repaint_for_palette(self, pal: dict) -> None:
        """Re-polish the QSS-styled children after a theme switch."""
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()


__all__ = ["FileInfoPane"]

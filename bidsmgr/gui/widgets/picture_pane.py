"""Editor centre pane for pictures a dataset carries: electrode and
anatomical photos (``*_photo.jpg``), figures in ``derivatives`` or
``code`` (``viz.data.formats.PICTURE_EXTS``). Fitted to the pane, or at its
own size; read only."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .primitives import PaneHeader


class PicturePane(QWidget):
    """A picture, fitted or at its own size."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane")
        self._path: Optional[Path] = None
        self._pixmap = QPixmap()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._header = PaneHeader("Picture")
        v.addWidget(self._header)
        bar = QFrame()
        bar.setObjectName("sidecar-toolbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 6, 14, 6)
        bl.setSpacing(8)
        self._fit_btn = QPushButton("Fit")
        self._actual_btn = QPushButton("Actual size")
        group = QButtonGroup(self)
        for b, tip in ((self._fit_btn, "The whole picture in the pane"),
                       (self._actual_btn, "One pixel of the picture per pixel of the screen")):
            b.setObjectName("tb-btn-toggle")
            b.setCheckable(True)
            b.setToolTip(tip)
            group.addButton(b)
            bl.addWidget(b)
        self._fit_btn.setChecked(True)
        self._fit_btn.toggled.connect(lambda _on: self._show())
        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        bl.addWidget(self._status, 1)
        v.addWidget(bar)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("picture-scroll")
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._label = QLabel("")
        self._label.setObjectName("picture-label")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setWidget(self._label)
        v.addWidget(self._scroll, 1)

    def set_read_only(self, on: bool) -> None:
        """Always read only; here for the Editor's mode switch."""
        del on

    def set_file(self, path: Optional[Path], root: Optional[Path]) -> None:
        del root
        self._path = Path(path) if path is not None else None
        self._pixmap = QPixmap(str(self._path)) if self._path is not None else QPixmap()
        if self._path is not None:
            self._header.setText(self._path.name)
            if self._pixmap.isNull():
                self._status.setText("This picture could not be read.")
            else:
                self._status.setText(f"{self._pixmap.width()} x {self._pixmap.height()} px")
        else:
            self._status.setText("")
        self._show()

    def current_file(self) -> Optional[Path]:
        return self._path

    def is_fitted(self) -> bool:
        return self._fit_btn.isChecked()

    def _show(self) -> None:
        if self._pixmap.isNull():
            self._label.setPixmap(QPixmap())
            self._label.setText("" if self._path is None else "Nothing to show.")
            return
        if self._fit_btn.isChecked():
            self._scroll.setWidgetResizable(True)
            room = self._scroll.viewport().size()
            shown = self._pixmap.scaled(room, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
        else:
            self._scroll.setWidgetResizable(False)
            shown = self._pixmap
        self._label.setPixmap(shown)
        self._label.adjustSize()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        super().resizeEvent(event)
        if self._fit_btn.isChecked():
            self._show()

    def repaint_for_palette(self, pal: dict) -> None:
        """Re-polish the QSS-styled children after a theme switch."""
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()


__all__ = ["PicturePane"]

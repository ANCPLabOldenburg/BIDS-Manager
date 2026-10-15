"""One inventory row, converted on the spot and shown before the real
conversion.

From the Converter's Properties panel: the row goes through the conversion
itself (``cli.convert.preview_row``) into a temporary folder, and what it
makes opens in the viewer for its kind. A series in the volume viewer, a
fieldmap as its magnitude images and phase difference, a spectroscopy
series in the spectrum viewer, a physiology recording and an EEG or MEG
recording in the signal viewer (the last two read as they are in the
source folder). Whether the scout really is a scout, whether the "T1" is
the T1, whether a fieldmap splits the way the plan says, is a look rather
than a guess. The temporary folder is removed when the window closes; the
source data is never touched.

The window opens on the CLICK, pending, with a spinner while the row is
converted on a thread (``set_files`` fills it, ``set_failed`` says why
not). Opened when the conversion finished instead, it came up seconds after
the click, and the window system left it behind the main window.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget,
)

from ..viz.data.formats import kind_of
from .viz import Viewer

#: The viewer each kind of file opens in.
_VIEWER_KIND = {"volume": "volume", "spectrum": "spectrum", "signal": "signal",
                "physio": "signal"}

_CONVERTED = ("Converted now for a look, the way the conversion will. "
              "Nothing is written to the dataset.")
_AS_IS = ("The recording as it is in the source folder; the conversion copies it. "
          "Nothing is written to the dataset.")


class PreviewDialog(QDialog):
    def __init__(self, title: str, work_dir: Optional[Path],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from .widgets import BusySpinner

        self.setWindowTitle(f"Preview: {title}")
        self.resize(1000, 680)
        self._work_dir = Path(work_dir) if work_dir else None
        self.files: list[Path] = []
        self._viewers: dict[str, Viewer] = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        bar = QWidget()
        bar.setObjectName("toolbar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(12, 6, 12, 6)
        self.note = QLabel(_CONVERTED)
        self.note.setObjectName("sidecar-footer-summary")
        self.note.setWordWrap(True)
        h.addWidget(self.note, 1)
        self.choice = QComboBox()
        self.choice.setObjectName("ent-input")
        self.choice.setVisible(False)
        self.choice.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.choice.setMinimumContentsLength(24)
        self.choice.setToolTip("The conversion made several files (magnitude images and "
                               "a phase difference, echoes, ...): pick one")
        self.choice.currentIndexChanged.connect(self._show)
        h.addWidget(self.choice)
        v.addWidget(bar)

        # Pending: a spinner and what is happening, centred where the data
        # will be.
        self._stack = QStackedWidget()
        pending = QWidget()
        pending.setObjectName("pane")
        pl = QVBoxLayout(pending)
        pl.addStretch(1)
        self.spinner = BusySpinner()
        self.spinner.set_busy(True, message="Converting for a look...")
        pl.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignHCenter)
        self.status = QLabel("")
        self.status.setObjectName("pane-hint")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setVisible(False)
        pl.addWidget(self.status)
        pl.addStretch(1)
        self._pending = pending
        self._stack.addWidget(pending)
        v.addWidget(self._stack, 1)

    @property
    def viewer(self) -> Optional[Viewer]:
        """The viewer showing the chosen file."""
        widget = self._stack.currentWidget()
        return widget if isinstance(widget, Viewer) else None

    def is_pending(self) -> bool:
        return self._stack.currentWidget() is self._pending

    def set_files(self, files: list[Path], work_dir: Optional[Path]) -> None:
        """What the conversion made: show the first, offer the others."""
        self._work_dir = Path(work_dir) if work_dir else self._work_dir
        self.files = [Path(p) for p in files]
        converted = self._work_dir is not None and all(
            self._work_dir in p.parents for p in self.files)
        self.note.setText(_CONVERTED if converted else _AS_IS)
        self.choice.blockSignals(True)
        self.choice.clear()
        for path in self.files:
            self.choice.addItem(path.name, str(path))
            self.choice.setItemData(self.choice.count() - 1, str(path),
                                    Qt.ItemDataRole.ToolTipRole)
        self.choice.blockSignals(False)
        self.choice.setVisible(len(self.files) > 1)
        self.spinner.set_busy(False)
        self._show(0)

    def set_failed(self, message: str) -> None:
        """The row could not be converted: say why, in the window."""
        self.spinner.set_busy(False)
        self.spinner.setVisible(False)
        self.status.setText(f"This could not be converted for a look: {message}")
        self.status.setVisible(True)

    def _viewer_for(self, kind: str) -> Viewer:
        viewer = self._viewers.get(kind)
        if viewer is None:
            viewer = Viewer(kind=kind, parent=self)
            if kind == "signal":
                # The click was "show me": no second click on Load signal.
                viewer.presenter.load_on_open = True
            self._viewers[kind] = viewer
            self._stack.addWidget(viewer)
        return viewer

    def _show(self, index: int) -> None:
        if not 0 <= index < len(self.files):
            return
        path = self.files[index]
        kind = _VIEWER_KIND.get(kind_of(path), "volume")
        viewer = self._viewer_for(kind)
        self._stack.setCurrentWidget(viewer)
        viewer.set_file(path, None)

    def done(self, result: int) -> None:  # noqa: D401 - Qt override
        for viewer in self._viewers.values():
            viewer.stop_loading()
        if self._work_dir is not None:
            # Only the temporary folder: a recording shown as it is stays
            # where it is.
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = None
        super().done(result)


__all__ = ["PreviewDialog"]

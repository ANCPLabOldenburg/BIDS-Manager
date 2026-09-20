"""Look at the image before defacing and after it, side by side.

Every defacing tool ends its documentation by telling you to inspect the
result, and then leaves you to find your own viewer. That advice is not
decoration: the two ways defacing goes wrong are opposite, and both are plain
to see and impossible to reason about. Too little removed and the participant
is still identifiable, which is the failure the user was trying to avoid. Too
much removed and the cerebellum or the front of the brain is gone, which
quietly ruins the analysis and survives every validator.

The side-by-side viewer itself is :class:`~bidsmgr.gui.widgets.compare_panes
.ComparePanes`, shared with the general "compare any two images" dialog. What
is here is the part that is about DEFACING: finding the undefaced copy,
explaining when there is not one, and offering to restore from it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..deface import compare, status
from .widgets.compare_panes import ComparePanes

log = logging.getLogger(__name__)


class DefaceCompareDialog(QDialog):
    """Before and after, for one image, with a linked crosshair."""

    def __init__(self, root: Path, rel: str, parent=None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._rel = str(rel).replace("\\", "/")
        self._original = compare.original_for(self._root, self._rel)
        self._panes = None

        self.setWindowTitle(f"Before and after: {Path(self._rel).name}")
        self.setSizeGripEnabled(True)
        size_to_screen(self, 1180, 720)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        self._header = QLabel()
        self._header.setObjectName("dialog-title")
        self._header.setWordWrap(True)
        outer.addWidget(self._header)

        self._subhead = QLabel()
        self._subhead.setObjectName("dialog-subtitle")
        self._subhead.setWordWrap(True)
        outer.addWidget(self._subhead)

        if self._original is None:
            self._build_nothing_to_compare(outer)
            return

        self._build_panes(outer)

    # -- the ordinary case ------------------------------------------------

    def _build_panes(self, outer: QVBoxLayout) -> None:
        eng = status.defaced_by_us(
            status.read_sidecar(status.sidecar_for(self._root / self._rel))
        )
        self._header.setText(self._rel)
        self._subhead.setText(
            f"Left: the image before defacing, from {self._original.description}. "
            f"Right: what is in the dataset now"
            + (f", defaced with {eng.label}." if eng else ".")
        )

        self._panes = ComparePanes()
        outer.addWidget(self._panes, 1)
        self._panes.both_loaded.connect(self._on_both_loaded)

        self._restore = QPushButton("Put the face back")
        self._restore.setObjectName("tb-btn")
        self._restore.setEnabled(False)
        self._restore.setToolTip(
            "Restore this image from the undefaced copy on the left. The copy "
            "is kept, so this can be defaced again afterwards."
        )
        self._restore.clicked.connect(self._on_restore)
        self._panes.footer.addWidget(self._restore)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        self._panes.footer.addWidget(close)

        self._panes.show_images(
            self._original.path, self._root / self._rel, root=self._root,
            left_title=f"Before: {self._original.path.name}",
            right_title=f"After: {Path(self._rel).name}",
        )

    def _on_both_loaded(self) -> None:
        self._restore.setEnabled(True)

    # -- restoring --------------------------------------------------------

    def _on_restore(self) -> None:
        """Put this one image back, from the copy being shown on the left."""
        from .deface_revert_dialog import DefaceRevertDialog

        dlg = DefaceRevertDialog(self._root, [self._root / self._rel], self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # The image on the right is now the one on the left. Re-read it rather
        # than leaving the pane showing bytes that are no longer on disk.
        self._panes.right.set_file(None, None)
        self._panes.right.set_file(self._root / self._rel, self._root)
        self._restore.setEnabled(False)
        self._subhead.setText(
            f"{self._rel} has been restored. Both sides now show the same "
            "image."
        )

    # -- the case with nothing to compare ---------------------------------

    def _build_nothing_to_compare(self, outer: QVBoxLayout) -> None:
        self._header.setText("No undefaced copy of this image")
        self._subhead.setText(compare.explain_missing(self._root, self._rel))
        outer.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        outer.addLayout(row)

    # -- closing ----------------------------------------------------------

    def done(self, result: int) -> None:  # noqa: D102 - Qt signature
        if self._panes is not None:
            self._panes.stop()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        if self._panes is not None:
            self._panes.stop()
        super().closeEvent(event)

    # -- what the tests reach for -----------------------------------------

    @property
    def _before(self):
        return self._panes.left

    @property
    def _after(self):
        return self._panes.right

    @property
    def _link(self):
        return self._panes.link

    @property
    def _link_note(self):
        return self._panes.note

    @property
    def _same_grid(self) -> bool:
        return self._panes._same_grid


def size_to_screen(widget, want_w: int, want_h: int) -> None:
    """Open at the requested size, or at what the screen actually has.

    A fixed 1180x720 is bigger than the work area on a 13-inch laptop once the
    dock and the menu bar are taken out, and a window that opens larger than
    the screen cannot be resized back by dragging an edge that is off the
    display.
    """
    screen = widget.screen() or QApplication.primaryScreen()
    if screen is None:
        widget.resize(want_w, want_h)
        return
    available = screen.availableGeometry()
    widget.resize(
        min(want_w, int(available.width() * 0.92)),
        min(want_h, int(available.height() * 0.92)),
    )


__all__ = ["DefaceCompareDialog", "size_to_screen"]

"""Two NIfTI viewers side by side, driven as one.

Built for the defacing before-and-after view and then wanted for everything
else, which is the usual sign that it belonged in its own widget. Comparing
two images is not a defacing question: a preprocessed volume against its raw
input, two echoes, this run against last week's, a derivative against the scan
it came from. The only thing defacing contributed was knowing which two files
to open.

What "driven as one" covers, because a partial sync is worse than none: the
crosshair, the slice, the plane, the volume index, RAS and radiological, the
orientation labels, and on the 3-D side the camera, every effect parameter,
and the cut plane INCLUDING which side of it is kept. Two renders cut from
opposite sides look like a difference in the data.

Two things it deliberately does not do.

**Images that do not match are still linked, through the scanner.** A voxel
index is meaningless between two images unless they share a grid, so the
crosshair travels as MILLIMETRES: out through one image's affine, in through
the other's. A cropped image, a different resolution, a different storage
order and a different modality all follow to the same anatomy. The note still
says the shapes differ, because it changes what a reader should expect of
the two pictures, but it is no longer a reason to stop syncing.

The one thing this cannot do is put the crosshair somewhere the other image
does not reach. Two images that only partly overlap clamp at the edge, which
is more useful than refusing to move.

**It shows ONE toolbar while synced.** Two identical toolbars driving one
shared state is not a choice the user has, it is the same control drawn twice.
Unsyncing gives the right-hand image its own.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .nifti_viewer_pane import NiftiViewerPane
from .primitives import ElidedLabel

log = logging.getLogger(__name__)


class ComparePanes(QWidget):
    """Two viewers, one set of controls, and a switch to separate them."""

    #: Both images have finished loading and their shapes are known.
    both_loaded = pyqtSignal()

    def __init__(
        self,
        *,
        left_title: str = "Left",
        right_title: str = "Right",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        # Whether the two share a voxel grid. NOT whether they can be
        # linked: they always can, through world coordinates. This only
        # decides what the note says.
        self._same_grid = False
        self._syncing = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self._left_caption, self.left = self._column(split, left_title)
        self._right_caption, self.right = self._column(split, right_title)
        split.setSizes([560, 560])
        outer.addWidget(split, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.link = QCheckBox("Sync the two views")
        self.link.setChecked(True)
        self.link.setEnabled(False)
        self.link.setToolTip(
            "On: one set of controls drives both images, and the crosshair, "
            "slice, plane, volume, 3-D camera, effects and cut plane stay "
            "together, which is the only way a comparison means anything.\n\n"
            "Off: each image gets its own controls, for when you want to look "
            "at one of them on its own."
        )
        self.link.toggled.connect(self._on_link_toggled)
        row.addWidget(self.link)
        self.note = QLabel("Loading…")
        self.note.setObjectName("muted-note")
        self.note.setWordWrap(True)
        row.addWidget(self.note, 1)
        self.footer = row
        outer.addLayout(row)

        for pane in (self.left, self.right):
            pane.loaded.connect(self._on_one_loaded)
        self.left.crosshair_moved.connect(
            lambda voxel: self._mirror_crosshair(self.right, voxel)
        )
        self.right.crosshair_moved.connect(
            lambda voxel: self._mirror_crosshair(self.left, voxel)
        )
        self.left.view_changed.connect(
            lambda state: self._mirror_view(self.right, state)
        )
        self.right.view_changed.connect(
            lambda state: self._mirror_view(self.left, state)
        )
        self.right.set_toolbar_visible(False)

    # -- building ---------------------------------------------------------

    def _column(self, split: QSplitter, title: str):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        caption = ElidedLabel(title)
        caption.setObjectName("section-caption")
        lay.addWidget(caption)
        pane = NiftiViewerPane()
        # Explicitly shrinkable. A viewer pane's natural minimum is its widest
        # control row, which is fine for the single pane in the Editor and not
        # fine for two: the window could not be dragged narrower than the sum,
        # and it grew itself to that width when the images loaded.
        pane.setMinimumWidth(260)
        lay.addWidget(pane, 1)
        split.addWidget(box)
        return caption, pane

    # -- loading ----------------------------------------------------------

    def show_images(
        self,
        left: Path,
        right: Path,
        *,
        root: Optional[Path] = None,
        left_title: str = "",
        right_title: str = "",
    ) -> None:
        """Bind both viewers. Reading happens on their own threads."""
        if left_title:
            self._left_caption.setText(left_title)
        if right_title:
            self._right_caption.setText(right_title)
        self._same_grid = False
        self.link.setEnabled(False)
        self.note.setText("Loading…")

        # Warm nibabel on THIS thread before starting either load. Both panes
        # read on their own QThread and import nibabel lazily, so this is the
        # only place in the application where two worker threads race the same
        # COLD import. Python's import lock does not survive that: one thread
        # blocks in `importlib._bootstrap.acquire` while the other is still
        # executing the module, and the process aborts.
        import nibabel  # noqa: F401

        self.left.set_file(Path(left), root)
        self.right.set_file(Path(right), root)

    def _on_one_loaded(self, _path: Path) -> None:
        left, right = self._shape(self.left), self._shape(self.right)
        if left is None or right is None:
            return

        self.link.setEnabled(True)
        self._same_grid = left[:3] == right[:3]
        self.note.setText("" if self._same_grid else (
            f"Different sizes ({left[:3]} and {right[:3]}). The crosshair is "
            "matched by position in the scanner rather than by voxel, so it "
            "still points at the same place in both."
        ))
        # Start both at the same place, so the first thing on screen is a
        # comparison rather than two unrelated slices.
        self._mirror_crosshair(self.right, self.left.crosshair_voxel())
        self.both_loaded.emit()

    @staticmethod
    def _shape(pane: NiftiViewerPane):
        data = getattr(pane, "_data", None)
        return None if data is None else tuple(data.shape)

    # -- linking ----------------------------------------------------------

    def _mirror_crosshair(self, target: NiftiViewerPane, voxel) -> None:
        """Move the other crosshair to the same PLACE, not the same index."""
        if self._syncing or not self.link.isChecked():
            return
        source = self.left if target is self.right else self.right
        world = source.crosshair_world()
        self._syncing = True
        try:
            if world is not None:
                target.set_crosshair_world(world)
            else:
                # No affine to go through: fall back to the index, which is
                # right whenever the two share a grid and is all there is
                # when one of them has no usable header.
                target.set_crosshair_voxel(voxel)
        finally:
            self._syncing = False

    def _mirror_view(self, target: NiftiViewerPane, state: dict) -> None:
        if self._syncing or not self.link.isChecked():
            return
        self._syncing = True
        try:
            target.apply_view_state(state, with_crosshair=True)
        finally:
            self._syncing = False

    def _on_link_toggled(self, on: bool) -> None:
        """One toolbar when linked, one per image when not."""
        self.right.set_toolbar_visible(not on)
        if on:
            # Re-linking adopts the left view, so the two are comparable again
            # rather than staying however they drifted apart.
            self._mirror_view(self.right, self.left.view_state())

    # -- closing ----------------------------------------------------------

    def stop(self) -> None:
        """Abandon any in-flight read, before anything is destroyed.

        Each pane's loader is a QThread parented to it, so a pane destroyed
        mid-read destroys a RUNNING QThread, and Qt answers that by aborting
        the process rather than raising.
        """
        for pane in (self.left, self.right):
            try:
                pane.stop_loading()
            except RuntimeError:
                pass


__all__ = ["ComparePanes"]

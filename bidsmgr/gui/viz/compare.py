"""Two viewers side by side, driven as one.

Built for the defacing before-and-after view and wanted for everything else:
a preprocessed volume against its raw input, two echoes, this run against
last week's, a derivative against its scan.

"Driven as one" covers the crosshair, the slice, the plane, the frame, the
display options, and on the 3-D side the camera, every effect parameter and
the cut plane INCLUDING which side is kept (two renders cut from opposite
sides look like a difference in the data). Contrast is linked too when the
two images are versions of the same data (defacing), and optional otherwise:
two different contrasts of one head want their own windows.

The crosshair travels as MILLIMETRES, out through one image's affine and in
through the other's, so a cropped image, another resolution, another storage
order and another modality all follow to the same anatomy; the note still
says when the grids differ.

The two halves are SYMMETRIC. While synced there is one toolbar, above both
images, and one controls column, beside both (with a switch saying which
image's own settings it shows); each image has the same header, so neither
looks like the main one. Unsyncing gives each image its own toolbar and its
own column, inside its half.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ..widgets.primitives import ElidedLabel
from .link import ViewerLink
from .viewer import Viewer
from ..widgets.splitter import Splitter

log = logging.getLogger(__name__)

#: How much of the window the shared controls column takes when it first opens.
COLUMN_FRACTION = 0.22


class _PaneHead(QFrame):
    """The same header on both halves: what the image is, and a way to
    change it when the host allows."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("pathbar")
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 6, 8, 6)
        row.setSpacing(8)
        self.caption = ElidedLabel(title)
        self.caption.setObjectName("section-caption")
        row.addWidget(self.caption, 1)
        self.change = QPushButton("Change…")
        self.change.setObjectName("tb-btn-ghost")
        self.change.setCursor(Qt.CursorShape.PointingHandCursor)
        self.change.setVisible(False)
        row.addWidget(self.change)


class _ControlsHost:
    """Where the shared controls column goes (see
    ``VolumePresenter.host_controls``)."""

    def __init__(self, column, tab_row, shown: Callable[[bool], None]) -> None:
        self.column = column
        self.tab_row = tab_row
        self.shown = shown


class ComparePanes(QWidget):
    """Two viewers, one set of controls, and a switch to separate them."""

    #: Both images are read and their grids are known.
    both_loaded = pyqtSignal()

    def __init__(self, *, left_title: str = "Left", right_title: str = "Right",
                 link_contrast: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        #: Whether the two share a voxel grid. Not whether they can be linked
        #: (they always can, through the world); only what the note says.
        self._same_grid = False
        self._loaded: set[str] = set()
        #: Whose own settings the shared column shows.
        self._controls_side = "left"
        self._synced = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        # The shared toolbar's place: above BOTH images.
        self._bar_slot = QVBoxLayout()
        self._bar_slot.setContentsMargins(0, 0, 0, 0)
        self._bar_slot.setSpacing(0)
        outer.addLayout(self._bar_slot)

        self._body = Splitter(Qt.Orientation.Horizontal)
        self._body.setHandleWidth(2)
        self._body.setChildrenCollapsible(False)
        images = QWidget()
        self._tab_row = QHBoxLayout(images)
        self._tab_row.setContentsMargins(0, 0, 0, 0)
        self._tab_row.setSpacing(0)
        self._split = Splitter(Qt.Orientation.Horizontal)
        self._split.setChildrenCollapsible(False)
        self._left_head, self.left = self._half(left_title)
        self._right_head, self.right = self._half(right_title)
        self._split.setSizes([560, 560])
        self._tab_row.addWidget(self._split, 1)
        self._body.addWidget(images)
        self._body.addWidget(self._build_column())
        self._body.setStretchFactor(0, 1)
        self._body.setStretchFactor(1, 0)
        self._body.splitterMoved.connect(
            lambda *_a: self.left.presenter.remember_sizes("compare.side", self._body))
        outer.addWidget(self._body, 1)
        self._host = _ControlsHost(self._column_lay, self._tab_row, self._show_column)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.link = QCheckBox("Sync the two views")
        self.link.setChecked(True)
        self.link.setEnabled(False)
        self.link.setToolTip(
            "On: one toolbar and one column of advanced controls drive both images, and the "
            "crosshair, slice, plane, volume, 3-D camera, effects and cut plane "
            "stay together.\n\nOff: each image gets its own toolbar and controls."
        )
        self.link.toggled.connect(self._on_link_toggled)
        row.addWidget(self.link)
        self.contrast = QCheckBox("Same contrast")
        self.contrast.setChecked(link_contrast)
        self.contrast.setToolTip(
            "Give both images the same window, gamma and colour map. Right for "
            "two versions of the same image (before and after defacing); "
            "leave off for two different contrasts."
        )
        self.contrast.toggled.connect(self._on_contrast_toggled)
        row.addWidget(self.contrast)
        self.note = QLabel("Loading...")
        self.note.setObjectName("dlg-hint")
        self.note.setWordWrap(True)
        row.addWidget(self.note, 1)
        self.footer = row
        outer.addLayout(row)

        self._link = ViewerLink(self.left, self.right, window=link_contrast)
        self._link.enabled = False
        self.left.loaded.connect(lambda _p: self._on_one_loaded("left"))
        self.right.loaded.connect(lambda _p: self._on_one_loaded("right"))
        self._apply_sync(True)

    def _half(self, title: str):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = _PaneHead(title)
        lay.addWidget(head)
        viewer = Viewer(header=False)
        # Explicitly shrinkable: the window must not be held wider than the
        # sum of two toolbars, nor grow itself when the images load.
        viewer.setMinimumWidth(260)
        lay.addWidget(viewer, 1)
        self._split.addWidget(box)
        return head, viewer

    def _build_column(self) -> QWidget:
        """The shared controls column: a switch for whose settings it shows,
        then that viewer's own column (placed by ``host_controls``)."""
        self._column = QWidget()
        lay = QVBoxLayout(self._column)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("sidecar-toolbar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(6)
        label = QLabel("Settings of")
        label.setObjectName("dlg-hint")
        row.addWidget(label)
        self._whose = QButtonGroup(self)
        self._whose.setExclusive(True)
        self.whose_buttons: dict[str, QPushButton] = {}
        for side, text in (("left", "Left image"), ("right", "Right image")):
            btn = QPushButton(text)
            # The header's own segmented look (Converter | Editor): the
            # chosen side is filled with the accent, in both themes.
            btn.setObjectName("view-pill")
            btn.setCheckable(True)
            btn.setChecked(side == "left")
            btn.setToolTip(
                "Which image's own settings the column shows: its window and "
                "colour map, its overlays, its quality maps. The layout, "
                "crosshair, 3-D view and cut planes are shared while synced.")
            btn.clicked.connect(lambda _c=False, s=side: self.show_settings_of(s))
            self._whose.addButton(btn)
            self.whose_buttons[side] = btn
            row.addWidget(btn)
        row.addStretch(1)
        lay.addWidget(bar)
        self._column_lay = lay
        self._column.setVisible(False)
        return self._column

    # -- captions -----------------------------------------------------------

    def _head(self, side: str) -> _PaneHead:
        return self._left_head if side == "left" else self._right_head

    def set_caption(self, side: str, text: str) -> None:
        self._head(side).caption.setText(text)

    def caption(self, side: str) -> str:
        return self._head(side).caption.text()

    def add_chooser(self, side: str, slot, tooltip: str = "") -> QPushButton:
        """Offer a Change button in ``side``'s header, calling ``slot``."""
        btn = self._head(side).change
        btn.clicked.connect(slot)
        btn.setToolTip(tooltip or f"Choose the {side} image")
        btn.setVisible(True)
        return btn

    # -- loading -----------------------------------------------------------

    def show_images(self, left: Path, right: Path, *, root: Optional[Path] = None,
                    left_title: str = "", right_title: str = "") -> None:
        """Open both. Reading happens on each viewer's own worker."""
        if left_title:
            self.set_caption("left", left_title)
        if right_title:
            self.set_caption("right", right_title)
        self._same_grid = False
        self._loaded.clear()
        self._link.enabled = False
        self.link.setEnabled(False)
        self.note.setText("Loading...")
        # Warm nibabel on THIS thread first. Both viewers read on their own
        # QThread and import nibabel lazily; two workers racing one COLD
        # import can deadlock the import lock and abort the process.
        import nibabel  # noqa: F401

        self.left.set_file(Path(left), root)
        self.right.set_file(Path(right), root)

    def _on_one_loaded(self, side: str) -> None:
        self._loaded.add(side)
        if self._loaded != {"left", "right"}:
            return
        a = self.left.presenter.source
        b = self.right.presenter.source
        if a is None or b is None:
            return
        self.link.setEnabled(True)
        self._same_grid = tuple(a.spatial) == tuple(b.spatial)
        self.note.setText("" if self._same_grid else (
            f"Different sizes ({tuple(a.spatial)} and {tuple(b.spatial)}). The "
            "crosshair is matched by position in the scanner rather than by "
            "voxel, so it still points at the same place in both."
        ))
        self._link.enabled = self.link.isChecked()
        if self._link.enabled:
            # Start both at the same place and the same view, so the first
            # thing on screen is a comparison.
            self._link.sync_now(self.left)
        self.both_loaded.emit()

    # -- the shared or separate controls --------------------------------------

    def _apply_sync(self, on: bool) -> None:
        """One toolbar and one column for both, or one each, inside each half."""
        if on == self._synced:
            return
        self._synced = on
        if on:
            self.left.lend_toolbar(self._bar_slot)
            self.right.set_toolbar_visible(False)
            self._place_column(self._controls_side, open_=self._any_column_open())
        else:
            open_ = self._any_column_open()
            self.left.lend_toolbar(None)
            self.right.set_toolbar_visible(True)
            for viewer in (self.left, self.right):
                viewer.presenter.share_controls(None)
                viewer.presenter.host_controls(None)
            self._column.setVisible(False)
            if open_:
                for viewer in (self.left, self.right):
                    viewer.presenter.set_inspector(True, remember=False)

    def _any_column_open(self) -> bool:
        if self._synced and not self._column.isHidden():
            return True
        return any(not v.presenter.side.isHidden() for v in (self.left, self.right))

    def _place_column(self, side: str, *, open_: bool) -> None:
        owner = self.left if side == "left" else self.right
        other = self.right if owner is self.left else self.left
        for viewer in (other, owner):
            viewer.presenter.share_controls(None)
        other.presenter.host_controls(None, tab=False)
        other.presenter.set_inspector(False, remember=False)
        owner.presenter.host_controls(self._host)
        other.presenter.share_controls(owner.presenter)
        self._controls_side = side
        self.whose_buttons[side].setChecked(True)
        if open_:
            owner.presenter.set_inspector(True, remember=False)

    def show_settings_of(self, side: str) -> None:
        """Show ``side``'s own settings in the shared column."""
        if not self._synced:
            return
        if side != self._controls_side:
            self._place_column(side, open_=not self._column.isHidden())

    def _show_column(self, on: bool) -> None:
        was = not self._column.isHidden()
        self._column.setVisible(bool(on))
        if on and not was:
            self.left.presenter.restore_sizes(
                "compare.side", self._body, [1.0 - COLUMN_FRACTION, COLUMN_FRACTION])

    def column_open(self) -> bool:
        """Whether the shared controls column is open."""
        return not self._column.isHidden()

    # -- linking -----------------------------------------------------------

    def _on_link_toggled(self, on: bool) -> None:
        self._apply_sync(bool(on))
        self._link.enabled = bool(on) and self._loaded == {"left", "right"}
        if self._link.enabled:
            self._link.sync_now(self.left)

    def _on_contrast_toggled(self, on: bool) -> None:
        self._link.window = bool(on)
        if self._link.enabled and on:
            self._link.sync_now(self.left)

    @property
    def viewer_link(self) -> ViewerLink:
        return self._link

    # -- closing -----------------------------------------------------------

    def stop(self) -> None:
        """Abandon in-flight reads before anything is destroyed: destroying a
        RUNNING QThread aborts the process."""
        for viewer in (self.left, self.right):
            try:
                viewer.stop_loading()
            except RuntimeError:
                pass

    def repaint_for_palette(self, pal: dict) -> None:
        viewers = (self.left, self.right)
        for viewer in viewers:
            viewer.repaint_for_palette(pal)
        # The shared toolbar and column live HERE while synced, outside both
        # viewers, so the viewers' own pass does not reach them.
        from PyQt6.QtWidgets import QComboBox

        from ..converter_panel import _repolish_combo

        def in_a_viewer(w) -> bool:
            while w is not None and w is not self:
                if w is viewers[0] or w is viewers[1]:
                    return True
                w = w.parentWidget()
            return False

        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            if in_a_viewer(w):
                continue
            style.unpolish(w)
            style.polish(w)
            if isinstance(w, QComboBox):
                _repolish_combo(w)
            w.update()


__all__ = ["ComparePanes"]

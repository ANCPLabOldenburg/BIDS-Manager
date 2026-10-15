"""Compare any two NIfTI images, driven as one.

The side-by-side viewer was built for defacing, where the two images are
chosen for you: the copy before, the file now. It turned out to be the thing
people wanted for everything else, and the only part that was about defacing
was knowing which two files to open.

So this is the same viewer with a pair of file pickers. Raw against
preprocessed, two echoes, a derivative against the scan it came from, this
week's run against last week's, one subject against another. Nothing here
knows or cares which.

It does not require the images to match. Different shapes, resolutions and
orientations all open and all stay linked: the crosshair travels as
millimetres in the scanner rather than as a voxel index, so it points at the
same anatomy in both. The note says when the shapes differ, because that
changes what a reader should expect of the two pictures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .deface_compare import size_to_screen
from .viz.compare import ComparePanes
from .widgets.nifti_picker import ask_for_image, is_nifti

log = logging.getLogger(__name__)


class CompareDialog(QDialog):
    """Two NIfTI images side by side, with everything synchronised."""

    def __init__(
        self,
        left: Optional[Path] = None,
        right: Optional[Path] = None,
        *,
        root: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root) if root else None
        self._left: Optional[Path] = Path(left) if left else None
        self._right: Optional[Path] = Path(right) if right else None

        self.setWindowTitle("Compare images")
        self.setSizeGripEnabled(True)
        size_to_screen(self, 1180, 720)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        header = QLabel("Compare two images")
        header.setObjectName("dialog-title")
        outer.addWidget(header)

        subtitle = QLabel(
            "One toolbar and one column of advanced controls drive both: crosshair, slice, "
            "plane, volume, 4-D graph, 3-D camera, effects and the cut plane. "
            "Images of different sizes work too: the crosshair is matched by "
            "position in the scanner, not by voxel."
        )
        subtitle.setObjectName("dialog-subtitle")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        self._panes = ComparePanes()
        outer.addWidget(self._panes, 1)
        # Each half chooses its own image, in its own header: the two halves
        # look the same, so neither reads as the main one.
        self._left_btn = self._panes.add_chooser(
            "left", self._pick_left, "Choose the image on the left")
        self._right_btn = self._panes.add_chooser(
            "right", self._pick_right, "Choose the image on the right")

        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        self._panes.footer.addWidget(close)

        self._refresh_labels()
        if self._left and self._right:
            self._show()

    # -- picking ----------------------------------------------------------

    def _ask(self, side: str) -> Optional[Path]:
        """The dataset's own images, in a tree, with a filter box.

        Not the OS file dialog. Finding ``sub-014/ses-post/anat`` by
        navigating folders is slower than the comparison it stands in the way
        of, and the dialog shows every file when only a handful are images.
        The picker still offers Browse… for an image from outside.
        """
        return ask_for_image(
            self, self._root,
            title=f"Choose the {side} image",
            start=self._left if side == "left" else self._right,
        )

    def _pick_left(self) -> None:
        picked = self._ask("left")
        if picked:
            self._left = picked
            self._after_pick()

    def _pick_right(self) -> None:
        picked = self._ask("right")
        if picked:
            self._right = picked
            self._after_pick()

    def _after_pick(self) -> None:
        self._refresh_labels()
        if self._left and self._right:
            self._show()

    def _refresh_labels(self) -> None:
        for side, path in (("left", self._left), ("right", self._right)):
            self._panes.set_caption(
                side, self._describe(path) if path else "nothing chosen yet")

    def _describe(self, path: Path) -> str:
        """The dataset-relative path when there is one, else the name."""
        if self._root:
            try:
                return Path(path).resolve().relative_to(
                    Path(self._root).resolve()
                ).as_posix()
            except ValueError:
                pass
        return Path(path).name

    # -- showing ----------------------------------------------------------

    def _show(self) -> None:
        self._panes.show_images(self._left, self._right, root=self._root)

    @property
    def panes(self) -> ComparePanes:
        """The two linked viewers."""
        return self._panes

    @property
    def chosen(self) -> tuple[Optional[Path], Optional[Path]]:
        """The left and right image, as picked so far."""
        return self._left, self._right

    def captions(self) -> tuple[str, str]:
        """What the two halves' headers say."""
        return self._panes.caption("left"), self._panes.caption("right")

    # -- closing ----------------------------------------------------------

    def done(self, result: int) -> None:  # noqa: D102 - Qt signature
        self._panes.stop()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt signature
        self._panes.stop()
        super().closeEvent(event)


def open_compare(
    parent, targets: Sequence[Path], *, root: Optional[Path] = None,
) -> CompareDialog:
    """Open the dialog on whatever the caller had selected.

    Two images picked in the tree open straight away. One opens on the left
    with the right still to choose, which is the common case: you are looking
    at something and want to put another image beside it.
    """
    images = [Path(t) for t in targets if is_nifti(t) and Path(t).is_file()]
    dlg = CompareDialog(
        images[0] if images else None,
        images[1] if len(images) > 1 else None,
        root=root, parent=parent,
    )
    return dlg


__all__ = ["CompareDialog", "is_nifti", "open_compare"]

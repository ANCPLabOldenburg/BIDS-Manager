"""Choose an overlay from the dataset, or from anywhere.

The OS file dialog dropped the user into the folder of the open image and
showed every file there; an overlay is usually elsewhere in the dataset (a
segmentation under ``derivatives/``, the fieldmap of the same session, the
spectroscopy of the same subject). This is the dataset's own images in the
shape the dataset has, the open image's subject first, each with a line on
what it would be drawn as, and Browse... for a file from outside.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QCheckBox, QDialog

from ...widgets.nifti_picker import NiftiPickerDialog

#: What Browse... offers: NIfTI and FreeSurfer volumes.
OVERLAY_FILTER = ("Images (*.nii *.nii.gz *.mgz *.mgh);;NIfTI (*.nii *.nii.gz);;"
                  "All files (*)")


def subject_of(path: Path) -> Optional[str]:
    """The ``sub-<label>`` a path belongs to (folder or file name)."""
    for part in reversed(Path(path).parts):
        m = re.match(r"(sub-[A-Za-z0-9]+)", part)
        if m:
            return m.group(1)
    return None


class OverlayPickerDialog(NiftiPickerDialog):
    """The dataset's images to draw over ``base``."""

    def __init__(self, root: Optional[Path], base: Path, *, base_affine=None,
                 base_shape=None, parent=None) -> None:
        self._base = Path(base)
        self._subject = subject_of(self._base)
        self._base_affine = base_affine
        self._base_shape = base_shape
        super().__init__(root, title="Add an overlay", start=None, parent=parent,
                         browse_filter=OVERLAY_FILTER, accept_text="Add")
        self.resize(700, 580)
        self.only_subject = QCheckBox(f"Only {self._subject}" if self._subject else "")
        self.only_subject.setToolTip(
            "Show the images of the open image's subject (its derivatives "
            "included); off, every image of the dataset.")
        others = any(subject_of(p) not in (None, self._subject) for p in self._images)
        self.only_subject.setVisible(bool(self._subject and others))
        self.only_subject.setChecked(bool(self._subject and others))
        self.only_subject.toggled.connect(lambda _v: self._apply_filter(self._filter.text()))
        self.layout().insertWidget(1, self.only_subject)
        self._apply_filter(self._filter.text())
        if self._subject:
            self._tree.expandAll()

    def _include(self, path: Path) -> bool:
        return Path(path) != self._base

    def _keep(self, path: Path) -> bool:
        box = getattr(self, "only_subject", None)
        if box is None or not box.isChecked() or not self._subject:
            return True
        return subject_of(path) == self._subject

    def _describe(self, path: Path) -> str:
        from ....viz.overlays import candidate_hint

        rel = path.name
        if self._root:
            try:
                rel = path.relative_to(self._root).as_posix()
            except ValueError:
                rel = str(path)
        hint = candidate_hint(path, self._base_affine, self._base_shape)
        return f"{rel}\n{hint}" if hint else rel


def ask_for_overlay(parent, root: Optional[Path], base: Path, *, base_affine=None,
                    base_shape=None) -> Optional[Path]:
    """The overlay to add, or None when cancelled."""
    dlg = OverlayPickerDialog(root, base, base_affine=base_affine, base_shape=base_shape,
                              parent=parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return dlg.chosen()


__all__ = ["OVERLAY_FILTER", "OverlayPickerDialog", "ask_for_overlay", "subject_of"]

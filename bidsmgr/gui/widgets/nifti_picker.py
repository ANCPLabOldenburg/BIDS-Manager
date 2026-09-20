"""Pick an image from the dataset, without leaving the application.

The comparison view used to ask for the second image through the OS file
dialog, which is the wrong tool for this job twice over. It drops the user
into a folder hierarchy they then have to navigate by memory, and it shows
every file in the dataset when only a handful are images. Finding
``sub-014/ses-post/anat`` that way, in a dataset with sixty subjects, is
slower than the comparison it is standing in the way of.

So: the dataset's own images, in the shape the dataset has, with a filter box.
Typing ``014 t1`` narrows sixty subjects to one file. Nothing is read from
disk beyond the directory walk until something is selected, and then only the
348-byte header, so the dimensions can be shown next to the choice.

``Browse…`` is still there, because an image from outside the dataset is a
real case (last month's export, a colleague's file) and the OS dialog is the
right tool for THAT one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

NIFTI_SUFFIXES = (".nii", ".nii.gz")
NIFTI_FILTER = "NIfTI images (*.nii *.nii.gz);;All files (*)"

# Never walked into. The tool's own state and other tools' repositories are
# not part of the dataset a user is looking for an image in.
SKIP_DIRS = frozenset({
    ".bidsmgr", ".tmp_bidsmgr", ".git", ".datalad", ".svn", "__pycache__",
})

# Past this the walk is not the slow part any more, the tree is. A dataset
# with more images than this is one where the filter box is the only sane way
# to find anything anyway.
MAX_IMAGES = 20_000


def is_nifti(path) -> bool:
    return Path(path).name.lower().endswith(NIFTI_SUFFIXES)


def find_images(root: Path, limit: int = MAX_IMAGES) -> list[Path]:
    """Every NIfTI under ``root``, sorted, skipping tool directories."""
    root = Path(root)
    out: list[Path] = []
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in SKIP_DIRS and not entry.name.startswith("."):
                    stack.append(entry)
            elif is_nifti(entry):
                out.append(entry)
                if len(out) >= limit:
                    return sorted(out)
    return sorted(out)


def matches(haystack: str, query: str) -> bool:
    """Every whitespace-separated term must appear, in any order.

    ``014 t1`` finds ``sub-014/anat/sub-014_T1w.nii.gz`` without the user
    having to remember which comes first or type any punctuation.
    """
    terms = query.lower().split()
    if not terms:
        return True
    low = haystack.lower()
    return all(term in low for term in terms)


class NiftiPickerDialog(QDialog):
    """Choose one image from the dataset, or browse outside it."""

    def __init__(
        self,
        root: Optional[Path],
        *,
        title: str = "Choose an image",
        start: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root) if root else None
        self._start = Path(start) if start else None
        self._chosen: Optional[Path] = None
        self._images: list[Path] = []

        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(660, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(8)

        self._filter = QLineEdit()
        self._filter.setObjectName("ent-input")
        self._filter.setPlaceholderText(
            "Filter: type any parts of the path, in any order (014 t1)"
        )
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        outer.addWidget(self._filter)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setUniformRowHeights(True)
        self._tree.currentItemChanged.connect(self._on_selection)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        outer.addWidget(self._tree, 1)

        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        row = QHBoxLayout()
        browse = QPushButton("Browse…")
        browse.setObjectName("tb-btn")
        browse.setToolTip(
            "Choose an image from outside this dataset, through the system "
            "file dialog."
        )
        browse.clicked.connect(self._on_browse)
        row.addWidget(browse)
        row.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Open
        )
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Open)
        self._ok.setObjectName("tb-btn-primary")
        self._ok.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        outer.addLayout(row)

        self._populate()

    # -- the tree ---------------------------------------------------------

    def _populate(self) -> None:
        self._tree.clear()
        if self._root is None:
            self._status.setText(
                "No dataset is open, so there is nothing to list. Use "
                "Browse… to choose a file."
            )
            return

        self._images = find_images(self._root)
        if not self._images:
            self._status.setText(
                f"No NIfTI images under {self._root.name}. Use Browse… to "
                "choose one from somewhere else."
            )
            return

        # Grouped the way the dataset is shaped, and keyed on the path built
        # so far rather than the folder NAME: two subjects both have an
        # `anat`, and keying on the name merges them into one branch.
        groups: dict[str, QTreeWidgetItem] = {}
        for image in self._images:
            rel = image.relative_to(self._root)
            parent: Optional[QTreeWidgetItem] = None
            key = ""
            for part in rel.parts[:-1]:
                key = f"{key}/{part}" if key else part
                node = groups.get(key)
                if node is None:
                    node = (
                        QTreeWidgetItem(parent, [part]) if parent is not None
                        else QTreeWidgetItem(self._tree, [part])
                    )
                    node.setFirstColumnSpanned(True)
                    groups[key] = node
                parent = node
            leaf = (
                QTreeWidgetItem(parent, [rel.name]) if parent is not None
                else QTreeWidgetItem(self._tree, [rel.name])
            )
            leaf.setData(0, Qt.ItemDataRole.UserRole, str(image))
            leaf.setToolTip(0, rel.as_posix())

        self._tree.expandToDepth(0)
        self._status.setText(f"{len(self._images)} image(s) in this dataset.")
        if self._start is not None:
            self._select(self._start)

    def _select(self, path: Path) -> None:
        """Put the cursor on ``path`` if it is in the tree."""
        wanted = str(Path(path))
        for item in self._leaves():
            if item.data(0, Qt.ItemDataRole.UserRole) == wanted:
                self._tree.setCurrentItem(item)
                self._tree.scrollToItem(item)
                return

    def _leaves(self):
        stack = [self._tree.topLevelItem(i)
                 for i in range(self._tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item is None:
                continue
            if item.data(0, Qt.ItemDataRole.UserRole):
                yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    # -- filtering --------------------------------------------------------

    def _apply_filter(self, text: str) -> None:
        """Hide what does not match, and every group left with nothing."""
        query = text.strip()
        shown = 0
        for leaf in self._leaves():
            path = leaf.data(0, Qt.ItemDataRole.UserRole) or ""
            rel = (
                Path(path).relative_to(self._root).as_posix()
                if self._root else Path(path).name
            )
            visible = matches(rel, query)
            leaf.setHidden(not visible)
            shown += int(visible)

        for item in self._groups_deepest_first():
            has_visible = any(
                not item.child(i).isHidden() for i in range(item.childCount())
            )
            item.setHidden(not has_visible)
            if has_visible and query:
                item.setExpanded(True)

        total = len(self._images)
        self._status.setText(
            f"{shown} of {total} image(s) shown." if query
            else f"{total} image(s) in this dataset."
        )

    def _groups_deepest_first(self) -> list[QTreeWidgetItem]:
        groups: list[QTreeWidgetItem] = []
        stack = [self._tree.topLevelItem(i)
                 for i in range(self._tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item is None or item.data(0, Qt.ItemDataRole.UserRole):
                continue
            groups.append(item)
            stack.extend(item.child(i) for i in range(item.childCount()))
        groups.sort(key=lambda i: _depth(i), reverse=True)
        return groups

    # -- choosing ---------------------------------------------------------

    def _on_selection(self, current, _previous) -> None:
        path = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        self._chosen = Path(path) if path else None
        self._ok.setEnabled(self._chosen is not None)
        if self._chosen is not None:
            self._status.setText(self._describe(self._chosen))

    def _describe(self, path: Path) -> str:
        """The choice, with its dimensions. Reads 348 bytes, not the image."""
        rel = path.name
        if self._root:
            try:
                rel = path.relative_to(self._root).as_posix()
            except ValueError:
                rel = str(path)
        try:
            from ...deface.probe import read_dimensions

            dims = read_dimensions(path)
            return f"{rel}   {dims.shape}"
        except Exception:  # noqa: BLE001 - a picker must not fail on a header
            return rel

    def _on_double_click(self, item, _column: int) -> None:
        if item is not None and item.data(0, Qt.ItemDataRole.UserRole):
            self.accept()

    def _on_browse(self) -> None:
        start = str(self._root) if self._root else ""
        chosen, _ = QFileDialog.getOpenFileName(
            self, self.windowTitle(), start, NIFTI_FILTER,
        )
        if chosen:
            self._chosen = Path(chosen)
            self.accept()

    def chosen(self) -> Optional[Path]:
        return self._chosen


def _depth(item: QTreeWidgetItem) -> int:
    depth = 0
    while item.parent() is not None:
        item = item.parent()
        depth += 1
    return depth


def ask_for_image(
    parent,
    root: Optional[Path],
    *,
    title: str = "Choose an image",
    start: Optional[Path] = None,
) -> Optional[Path]:
    """Ask for one image. ``None`` if the user cancelled."""
    dialog = NiftiPickerDialog(root, title=title, start=start, parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.chosen()


__all__ = [
    "NIFTI_FILTER",
    "NiftiPickerDialog",
    "ask_for_image",
    "find_images",
    "is_nifti",
    "matches",
]

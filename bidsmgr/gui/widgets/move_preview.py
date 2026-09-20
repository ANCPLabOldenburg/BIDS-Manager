"""The "what this would do" tree, shared by every dialog that moves files.

It shows the dataset the way the dataset is shaped. That sounds obvious and the
first version did not do it: it grouped the moves by the full folder path, so
``sub-001/ses-01/anat`` and ``sub-001/ses-01/eeg`` came out as two unrelated
top-level rows that happened to start with the same text. There was no subject
to collapse, no session to untick, and on a dataset with more than one of
either the list read as a flat pile of paths with no shape at all.

Here a subject contains its sessions, a session contains its datatypes, and a
datatype contains its files, because that is what is on disk. Ticking is
inherited the way a person expects: untick a session and everything under it
goes with it, untick one run and its session shows as partly selected.

Three dialogs use this (rename, edit entities, sessions), which is the point.
The selection rules and the tri-state arithmetic are subtle enough that a
second copy would drift, and a preview that disagrees with what is applied is
worse than no preview.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHeaderView,
    QTreeWidget,
    QTreeWidgetItem,
)

# Where a row keeps the plan key it stands for, so nothing has to match a file
# back to the plan by its display text.
KEY_ROLE = Qt.ItemDataRole.UserRole + 1

# Above this many files the tree opens folded to the datatype level. Expanding
# everything is right for the dataset in front of most people and unreadable
# for a hundred-subject one, where the useful view is the shape, not the rows.
_EXPAND_ALL_UNDER = 250


class MovePreviewTree(QTreeWidget):
    """A checkable, nested view of a set of file moves."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("check-tree")
        self.setColumnCount(2)
        self.setHeaderLabels(["In the dataset", "Becomes"])
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(False)
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

    # -- filling it ------------------------------------------------------

    def show_moves(
        self,
        root: Path,
        moves: Sequence[tuple[str, Path, Path]],
        *,
        extras: Iterable[tuple[str, str]] = (),
        conflicts: Iterable[str] = (),
        checked: Optional[set[str]] = None,
    ) -> None:
        """Draw ``moves`` as ``(key, source, destination)``, nested by folder.

        ``extras`` are the things that follow and cannot be chosen separately
        (a folder that empties, a table that merges, a reference that travels).
        ``conflicts`` are the reasons the plan cannot go ahead.

        ``checked`` is which keys start ticked; ``None`` ticks everything.
        Opening on one file the user right-clicked should offer THAT file, not
        every file in the dataset that happens to share its subject, so the
        caller decides and this draws what it is told.
        """
        self.blockSignals(True)
        self.clear()

        folders: dict[str, QTreeWidgetItem] = {}
        count = 0
        for key, src, dst in moves:
            rel = _rel(root, src)
            parts = rel.split("/")
            parent = self._folder(folders, parts[:-1])
            leaf = QTreeWidgetItem(parent, [parts[-1], _becomes(root, src, dst)])
            leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            leaf.setCheckState(0, _state(checked is None or key in checked))
            leaf.setData(0, KEY_ROLE, key)
            leaf.setToolTip(0, rel)
            leaf.setToolTip(1, _rel(root, dst))
            count += 1

        self._add_section("Follows automatically", extras)
        self._add_section(
            "Refused, these names would collide",
            ((text, "") for text in conflicts),
        )

        self.blockSignals(False)
        self._expand(count)

    def show_removals(
        self,
        root: Path,
        removals: Sequence[tuple[str, Path]],
        *,
        extras: Iterable[tuple[str, str]] = (),
        conflicts: Iterable[str] = (),
        verb: str = "deleted",
    ) -> None:
        """Draw ``removals`` as ``(key, path)``, nested the same way.

        Same tree, same ticking, same sections, because "what is about to
        happen to my dataset" is one question whether the answer is a move or
        a removal, and two widgets answering it would drift.
        """
        self.setHeaderLabels(["In the dataset", "What happens"])
        self.blockSignals(True)
        self.clear()

        folders: dict[str, QTreeWidgetItem] = {}
        count = 0
        for key, path in removals:
            rel = _rel(root, path)
            parts = rel.split("/")
            parent = self._folder(folders, parts[:-1])
            leaf = QTreeWidgetItem(parent, [parts[-1], verb])
            leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            leaf.setCheckState(0, Qt.CheckState.Checked)
            leaf.setData(0, KEY_ROLE, key)
            leaf.setToolTip(0, rel)
            count += 1

        self._add_section("Follows automatically", extras)
        self._add_section("Refused", ((text, "") for text in conflicts))

        self.blockSignals(False)
        self._expand(count)

    def _folder(
        self, folders: dict[str, QTreeWidgetItem], parts: Sequence[str],
    ) -> Optional[QTreeWidgetItem]:
        """The item for this folder, creating the chain above it as needed.

        Keyed on the POSIX path built so far rather than on the folder's own
        name: two subjects both have an ``anat``, and keying on the name put
        the second subject's files under the first one's folder.
        """
        parent: Optional[QTreeWidgetItem] = None
        walked = ""
        for name in parts:
            walked = f"{walked}/{name}" if walked else name
            item = folders.get(walked)
            if item is None:
                item = (
                    QTreeWidgetItem(parent, [name, ""]) if parent is not None
                    else QTreeWidgetItem(self, [name, ""])
                )
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsAutoTristate
                )
                # Unchecked to start. Qt's auto-tristate raises a parent to
                # partially-checked or checked as its children are added, so
                # starting them CHECKED left a folder claiming everything
                # under it was selected when only one file was.
                item.setCheckState(0, Qt.CheckState.Unchecked)
                item.setToolTip(0, walked)
                folders[walked] = item
            parent = item
        return parent

    def _add_section(
        self, title: str, rows: Iterable[tuple[str, str]],
    ) -> None:
        rows = list(rows)
        if not rows:
            return
        head = QTreeWidgetItem(self, [title, ""])
        head.setFlags(Qt.ItemFlag.ItemIsEnabled)
        head.setExpanded(True)
        for left, right in rows:
            child = QTreeWidgetItem(head, [left, right])
            child.setFlags(Qt.ItemFlag.ItemIsEnabled)

    def _expand(self, count: int) -> None:
        if count <= _EXPAND_ALL_UNDER:
            self.expandAll()
            return
        # Subjects and their sessions, so the shape is visible and the rows
        # are not.
        self.expandToDepth(1)

    # -- reading it back --------------------------------------------------

    def selected_keys(self) -> set[str]:
        """Every ticked leaf, by the key it was given."""
        out: set[str] = set()
        for item in self._leaves():
            key = item.data(0, KEY_ROLE)
            if key and item.checkState(0) == Qt.CheckState.Checked:
                out.add(key)
        return out

    def total_keys(self) -> int:
        return sum(1 for item in self._leaves() if item.data(0, KEY_ROLE))

    def set_all(self, state: Qt.CheckState) -> None:
        """Tick or untick everything, without a signal per row.

        Only the top-level checkable rows are set: Qt's auto-tristate
        propagates DOWN from a parent, so setting each descendant as well would
        fire the same work once per row on a dataset where that is thousands of
        them.
        """
        self.blockSignals(True)
        for i in range(self.topLevelItemCount()):
            top = self.topLevelItem(i)
            if top.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                _set_recursive(top, state)
        self.blockSignals(False)

    def _leaves(self) -> Iterable[QTreeWidgetItem]:
        stack = [
            self.topLevelItem(i) for i in range(self.topLevelItemCount())
        ]
        while stack:
            item = stack.pop()
            if item is None:
                continue
            if item.childCount():
                stack.extend(item.child(j) for j in range(item.childCount()))
            else:
                yield item


def _set_recursive(item: QTreeWidgetItem, state: Qt.CheckState) -> None:
    """Set this row and everything under it.

    Done by hand rather than left to auto-tristate because auto-tristate only
    propagates when the change comes from the USER. Setting a parent's state in
    code leaves the children where they were, which is how Select none used to
    leave every file ticked underneath an unticked folder.
    """
    item.setCheckState(0, state)
    for i in range(item.childCount()):
        _set_recursive(item.child(i), state)


def _state(on: bool) -> Qt.CheckState:
    return Qt.CheckState.Checked if on else Qt.CheckState.Unchecked


def _rel(root: Path, path: Path) -> str:
    """``path`` within ``root``, spelled with forward slashes.

    ``as_posix()``, never ``str()``: on Windows the native spelling hands the
    label back with backslashes, so the tree would read in a spelling that
    appears nowhere in the dataset and nowhere in any BIDS reference.
    See CROSS_PLATFORM_RULES section 1.
    """
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, OSError):
        return Path(path).as_posix()


def _becomes(root: Path, src: Path, dst: Path) -> str:
    """What to show in the second column.

    Just the new NAME when the file stays where it is, which is the common case
    and the thing the eye is comparing. The whole new path when the file also
    changes folder, because "becomes sub-001_ses-01_T1w.nii.gz" would hide the
    fact that it has moved into a session, which is the larger half of what is
    about to happen.
    """
    if src.parent == dst.parent:
        return dst.name
    return _rel(root, dst)


def delete_extras(plan) -> list[tuple[str, str]]:
    """What a deletion does BESIDES removing the files, as display rows.

    Kept visible and not checkable for the same reason the move version is:
    a scans row for a recording that is not there, or an ``IntendedFor``
    naming a file nothing has, is exactly the damage this feature exists to
    avoid, and letting somebody untick the repair would reintroduce it.
    """
    out: list[tuple[str, str]] = []
    for drop in getattr(plan, "scans_drops", ()):
        what = (
            "emptied, so the table goes too" if drop.empties
            else f"{len(drop.rows)} row(s) removed"
        )
        out.append((drop.rel, what))
    for drop in getattr(plan, "ref_drops", ()):
        field = getattr(drop, "field", "IntendedFor")
        what = (
            f"{field} emptied, so the key is removed" if drop.empties
            else f"{len(drop.entries)} {field} entry(ies) removed"
        )
        out.append((drop.rel, what))
    for label in getattr(plan, "participants", ()):
        out.append(("participants.tsv", f"row for {label} removed"))
    for sidecar in getattr(plan, "orphaned_sidecars", ()):
        out.append((sidecar.name, "describes nothing now, so it is removed"))
    for folder in getattr(plan, "emptied", ()):
        out.append((folder.name, "folder left empty, so it is removed"))
    return out


def plan_extras(plan) -> list[tuple[str, str]]:
    """Everything a plan does BESIDES moving the files, as display rows.

    A folder that empties, the tables a merge combines, the references that
    travel. Shown so the preview is the whole plan, and not checkable, so
    nobody can produce a half-applied cross-reference by unticking the row
    that keeps ``IntendedFor`` pointing at something real.
    """
    out: list[tuple[str, str]] = []
    for src, dst in getattr(plan, "fused_dirs", ()):
        out.append((src.name, f"merged into the existing {dst.name}"))
    for src, dst in getattr(plan, "dir_moves", ()):
        out.append((src.name, dst.name))
    for src, dst in getattr(plan, "table_merges", ()):
        out.append((src.name, f"appended to {dst.name}"))
    for path, column in getattr(plan, "row_folds", ()):
        out.append((path.name, f"two {column} rows folded into one"))
    for src, dst in getattr(plan, "tool_state_moves", ()):
        out.append((
            f"{src.parent.name}/{src.name}",
            f"kept as {dst.parent.name}/{dst.name}",
        ))
    for edit in getattr(plan, "content_edits", ()):
        out.append((edit.rel, f"{edit.hits} x {edit.what} updated"))
    return out


__all__ = [
    "KEY_ROLE",
    "MovePreviewTree",
    "delete_extras",
    "plan_extras",
]

"""Which files point at which, made by picking a file on each side.

A reference has two ends, so the dialog has two lists: the files that carry
the field on the LEFT, the files it may point at on the RIGHT, and the verbs
between them. That is the shape every tool which maps one set onto another
uses, from a mail-merge field mapper to a mixer's routing matrix, and it is
the shape this always should have been. The previous versions drew one
vertical tree and asked the reader to hold the relationship in their head.

What each side is showing is decided by the standard, not by us:

* the **field** list holds only fields the selected files may carry
  (``IntendedFor`` is not offered on an anatomical);
* the **right** list holds only files the field may point AT, in the same
  subject and session, because a fieldmap does not correct another
  participant's run;
* the rule is printed above both lists in a sentence, so the reason a file is
  absent from a list is readable rather than guessable.

Three things the left list says that a JSON editor cannot:

**Status.** ``ok``, ``points at a missing file``, ``not set``, and
``differs from the times``, which is the interesting one: the link is
written, resolves, and disagrees with what the acquisition times imply.

**The reverse direction.** The right list's second column says what already
points at each candidate, so "is this run corrected by anything at all" is
answerable, which is half of what people actually come here to ask.

**Several at once.** Select four fieldmaps and tick a run: all four take it.
The tick is tri-state when the selected files disagree, so nothing is
silently flattened.

Nothing is written until Save, and Save writes every pending change as ONE
operation, so a linkage pass over a session is one entry in the history and
one undo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..editor import linkage
from ..editor import values as ev
from .dialog_chrome import build_footer_with, build_header, card, hint

log = logging.getLogger(__name__)

#: A row with unsaved edits. Not a linkage state: it is a state of the
#: DIALOG, and it has to read differently from the four the engine reports.
_EDITED = "edited, not saved"

_COLOURS = {
    _EDITED: "#58a6ff",
    linkage.BROKEN: "#f85149",
    linkage.DIFFERS: "#d29922",
    linkage.IMPLIED: "#d29922",
    linkage.UNSET: "#8b949e",
}

# What the left list is filtered to.
_ALL = "all"
_PROBLEMS = "problems"
_SET = "set"

_PATH_ROLE = Qt.ItemDataRole.UserRole


class LinkageDialog(QDialog):
    """Sources on the left, targets on the right, the verbs in between."""

    def __init__(
        self,
        root: Path,
        target: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._rows: list[linkage.SourceRow] = []
        self._incoming: dict[Path, list[tuple[Path, str]]] = {}
        # sidecar -> the targets it should end up pointing at. Only the
        # files actually edited appear, so Save writes nothing it was not
        # asked to.
        self._pending: dict[Path, list[Path]] = {}
        self._changed = 0
        self._loading = False

        self.setWindowTitle("References between files")
        self.setModal(True)
        self.resize(1280, 820)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Which files point at which",
            "A few BIDS fields hold a pointer at another file rather than a "
            "value: a fieldmap's <b>IntendedFor</b> names the runs it can "
            "correct, a derivative's <b>Sources</b> names what it was made "
            "from. Pick a file on the left, tick what it should point at on "
            "the right. Only what the standard allows appears in either list.",
        ))

        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(18, 14, 18, 12)
        bl.setSpacing(10)

        bl.addWidget(self._build_controls())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_left())
        split.addWidget(self._build_middle())
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 0)
        split.setStretchFactor(2, 5)
        split.setCollapsible(1, False)
        bl.addWidget(split, 1)
        outer.addWidget(body, 1)

        self._status = QLabel("")
        self._status.setObjectName("dlg-hint")
        self._status.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self._on_close)
        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("tb-btn-primary")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        buttons.addButton(
            self._save_btn, QDialogButtonBox.ButtonRole.AcceptRole
        )
        outer.addWidget(build_footer_with(self._status, buttons))

        if target is not None:
            self._preselect = linkage.data_file_for(
                linkage.sidecar_for(Path(target))
            )
            self._aim_field_at(Path(target))
        else:
            self._preselect = None
        self._reload()

    # -- construction ------------------------------------------------------

    def _build_controls(self) -> QWidget:
        box, cl = card()
        row = QHBoxLayout()
        row.setSpacing(8)

        row.addWidget(QLabel("Field:"))
        self._field = QComboBox()
        self._field.setObjectName("ent-input")
        for name in linkage.LINK_FIELDS:
            self._field.addItem(name, name)
        self._field.currentIndexChanged.connect(lambda _i: self._reload())
        row.addWidget(self._field, 1)

        row.addWidget(QLabel("In:"))
        self._scope = QComboBox()
        self._scope.setObjectName("ent-input")
        for scope in ev.scopes(self._root):
            self._scope.addItem(scope.label, scope.prefix)
        self._scope.currentIndexChanged.connect(lambda _i: self._reload())
        row.addWidget(self._scope, 1)

        row.addWidget(QLabel("Show:"))
        self._show = QComboBox()
        self._show.setObjectName("ent-input")
        self._show.addItem("Everything that can carry it", _ALL)
        self._show.addItem("Only what is set", _SET)
        self._show.addItem("Only problems", _PROBLEMS)
        self._show.currentIndexChanged.connect(lambda _i: self._reload())
        row.addWidget(self._show, 1)
        cl.addLayout(row)

        self._rule = hint("")
        cl.addWidget(self._rule)
        return box

    def _build_left(self) -> QWidget:
        box, ll = card("Files that carry the field")
        self._left = QTreeWidget()
        self._left.setObjectName("check-tree")
        self._left.setColumnCount(3)
        self._left.setHeaderLabels(["File", "Points at", "Status"])
        self._left.setRootIsDecorated(False)
        self._left.setUniformRowHeights(True)
        self._left.setSelectionMode(
            QTreeWidget.SelectionMode.ExtendedSelection
        )
        self._left.itemSelectionChanged.connect(self._on_source_changed)
        ll.addWidget(self._left, 1)

        # The bulk path is the reason this tool exists: "every fieldmap in
        # this dataset has nothing set, give them all what the times imply".
        # That needs selecting them all, and Ctrl+A is not discoverable.
        picks = QHBoxLayout()
        picks.setSpacing(8)
        for text, tip, fn in (
            ("Select all", "Act on every file listed here.",
             self._left.selectAll),
            ("Select none", "Start the selection again.",
             self._left.clearSelection),
        ):
            btn = QPushButton(text)
            btn.setObjectName("tb-btn")
            btn.setToolTip(tip)
            btn.clicked.connect(fn)
            picks.addWidget(btn)
        picks.addStretch(1)
        ll.addLayout(picks)

        self._left_note = hint("")
        ll.addWidget(self._left_note)
        return box

    def _build_middle(self) -> QWidget:
        holder = QWidget()
        holder.setFixedWidth(196)
        ml = QVBoxLayout(holder)
        ml.setContentsMargins(6, 28, 6, 6)
        ml.setSpacing(6)

        self._link_btn = QPushButton("Link  →")
        self._link_btn.setObjectName("tb-btn-primary")
        self._link_btn.setToolTip(
            "Point the files selected on the left at the files selected on "
            "the right. Ticking a box on the right does the same thing."
        )
        self._link_btn.clicked.connect(lambda: self._set_selected_targets(True))
        ml.addWidget(self._link_btn)

        self._unlink_btn = QPushButton("←  Unlink")
        self._unlink_btn.setObjectName("tb-btn")
        self._unlink_btn.clicked.connect(
            lambda: self._set_selected_targets(False)
        )
        ml.addWidget(self._unlink_btn)

        ml.addSpacing(12)

        self._propose_btn = QPushButton("Use what the\ntimes imply")
        self._propose_btn.setObjectName("tb-btn")
        self._propose_btn.setToolTip(
            "Apply the same rule the conversion applies: a fieldmap covers "
            "the runs acquired after it and before the next one."
        )
        self._propose_btn.clicked.connect(self._on_propose)
        ml.addWidget(self._propose_btn)

        self._clear_btn = QPushButton("Point at nothing")
        self._clear_btn.setObjectName("tb-btn")
        self._clear_btn.setToolTip(
            "Remove the field rather than writing an empty list. An empty "
            "list claims the file points at nothing, which is a different "
            "and wronger statement than not saying."
        )
        self._clear_btn.clicked.connect(self._on_clear)
        ml.addWidget(self._clear_btn)

        ml.addStretch(1)
        self._middle_note = hint("")
        ml.addWidget(self._middle_note)
        return holder

    def _build_right(self) -> QWidget:
        box, rl = card("Files it may point at")
        self._right = QTreeWidget()
        self._right.setObjectName("check-tree")
        self._right.setColumnCount(2)
        self._right.setHeaderLabels(["File", "Already pointed at by"])
        self._right.setRootIsDecorated(False)
        self._right.setUniformRowHeights(True)
        self._right.setSelectionMode(
            QTreeWidget.SelectionMode.ExtendedSelection
        )
        self._right.itemChanged.connect(self._on_target_toggled)
        rl.addWidget(self._right, 1)
        self._right_note = hint("")
        rl.addWidget(self._right_note)
        return box

    # -- helpers -----------------------------------------------------------

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self._root.resolve()).as_posix()
        except ValueError:
            return path.name

    def _field_name(self) -> str:
        return self._field.currentData() or linkage.LINK_FIELDS[0]

    def _aim_field_at(self, path: Path) -> None:
        """Start on a field the passed-in file can actually carry."""
        offered = linkage.fields_for(self._root, Path(path))
        if not offered:
            return
        at = self._field.findData(offered[0])
        if at >= 0:
            self._field.blockSignals(True)
            self._field.setCurrentIndex(at)
            self._field.blockSignals(False)

    def _targets_now(self, row: linkage.SourceRow) -> list[Path]:
        """What ``row`` points at, counting unsaved edits."""
        if row.sidecar in self._pending:
            return list(self._pending[row.sidecar])
        return list(row.live)

    def _selected_rows(self) -> list[linkage.SourceRow]:
        chosen = {
            item.data(0, _PATH_ROLE) for item in self._left.selectedItems()
        }
        return [r for r in self._rows if str(r.path) in chosen]

    # -- loading -----------------------------------------------------------

    def _reload(self) -> None:
        field = self._field_name()
        self._rule.setText(
            f"<b>{field}</b>. {linkage.FIELD_RULE.get(field, '')}"
        )

        # A field the standard allows almost anywhere would otherwise list
        # every recording in the dataset, which is true and unusable, so it
        # defaults to what is already set.
        unrestricted = linkage.restricted_to(field) is None
        mode = self._show.currentData()
        only_set = mode == _SET or (mode == _ALL and unrestricted)

        self._rows = linkage.sources(
            self._root, field,
            prefix=self._scope.currentData() or "",
            only_set=only_set,
        )
        if mode == _PROBLEMS:
            self._rows = [r for r in self._rows if r.status != linkage.OK]
        self._incoming = linkage.incoming(self._root)

        self._fill_left()
        note = (
            f"{len(self._rows)} file(s) can carry {field} here."
            if self._rows else
            f"No file here carries {field}. "
            + linkage.FIELD_RULE.get(field, "")
        )
        if unrestricted and mode == _ALL:
            self._left_note.setText(
                note + " The standard allows this field almost anywhere, so "
                "only files that already have it are listed."
            )
        else:
            self._left_note.setText(note)

    def _fill_left(self) -> None:
        self._loading = True
        self._left.clear()
        for row in self._rows:
            targets = self._targets_now(row)
            status = (
                _EDITED if row.sidecar in self._pending else row.status
            )
            item = QTreeWidgetItem(self._left, [
                self._rel(row.path),
                self._describe(targets, row),
                status,
            ])
            item.setData(0, _PATH_ROLE, str(row.path))
            item.setToolTip(0, self._rel(row.path))
            colour = _COLOURS.get(status)
            if colour:
                item.setForeground(2, QBrush(QColor(colour)))
        for column in range(3):
            self._left.resizeColumnToContents(column)
        self._loading = False

        if self._preselect is not None:
            self._select(self._preselect)
            self._preselect = None
        elif self._left.topLevelItemCount():
            self._left.setCurrentItem(self._left.topLevelItem(0))
        else:
            self._on_source_changed()

    def _describe(self, targets: list[Path], row: linkage.SourceRow) -> str:
        """The "Points at" column: names when few, a count when many.

        Shortened to the last two entity tokens (``run-01_bold.nii.gz``)
        rather than the suffix alone: every target of one fieldmap is a
        ``_bold``, so the suffix on its own printed the same word twice and
        said nothing about WHICH runs.
        """
        broken = sum(1 for r in row.resolved if r is None)
        if not targets and not broken:
            return "nothing"
        names = ["_".join(Path(t).name.split("_")[-2:]) for t in targets[:2]]
        text = ", ".join(names)
        if len(targets) > 2:
            text += f" and {len(targets) - 2} more"
        if broken:
            text += f" ({broken} missing)" if text else f"{broken} missing"
        return text or f"{len(targets)} file(s)"

    def _select(self, path: Path) -> None:
        wanted = str(path)
        for i in range(self._left.topLevelItemCount()):
            item = self._left.topLevelItem(i)
            if item.data(0, _PATH_ROLE) == wanted:
                self._left.setCurrentItem(item)
                self._left.scrollToItem(item)
                return
        if self._left.topLevelItemCount():
            self._left.setCurrentItem(self._left.topLevelItem(0))

    # -- the right list ----------------------------------------------------

    def _on_source_changed(self) -> None:
        rows = self._selected_rows()
        self._loading = True
        self._right.clear()

        for widget in (self._link_btn, self._unlink_btn, self._clear_btn):
            widget.setEnabled(bool(rows))
        self._propose_btn.setEnabled(
            any(r.proposed is not None for r in rows)
        )

        if not rows:
            self._right_note.setText("Select a file on the left.")
            self._middle_note.setText("")
            self._loading = False
            self._refresh_status()
            return

        field = self._field_name()
        # The INTERSECTION across the selected files, because a target has to
        # be legal for every file that would take it. Selecting two subjects'
        # fieldmaps therefore offers nothing rather than offering a link that
        # would cross subjects.
        shared: Optional[set[Path]] = None
        for row in rows:
            here = set(linkage.candidates(self._root, row.path, field))
            shared = here if shared is None else (shared & here)
        options = sorted(shared or set())

        ticked = [set(self._targets_now(r)) for r in rows]
        for candidate in options:
            holders = self._incoming.get(candidate, [])
            others = ", ".join(
                sorted({Path(src).name for src, _f in holders})
            ) or "nothing"
            item = QTreeWidgetItem(self._right, [self._rel(candidate), others])
            item.setData(0, _PATH_ROLE, str(candidate))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, self._tick_state(
                sum(1 for t in ticked if candidate in t), len(rows),
            ))
            self._right.addTopLevelItem(item)

        # Targets that are written and are not there. They cannot be ticked,
        # because ticking means "point at this" and there is nothing to point
        # at; unticking them is done by Save, which drops what is gone.
        for row in rows:
            for written, resolved in zip(row.targets, row.resolved):
                if resolved is not None:
                    continue
                item = QTreeWidgetItem(self._right, [written, "missing"])
                item.setForeground(
                    1, QBrush(QColor(_COLOURS[linkage.BROKEN]))
                )
                item.setForeground(
                    0, QBrush(QColor(_COLOURS[linkage.BROKEN]))
                )
                self._right.addTopLevelItem(item)

        self._right.resizeColumnToContents(0)
        self._loading = False

        if not options:
            self._right_note.setText(
                "Nothing in this dataset is a legal target for that field "
                "here. " + linkage.FIELD_RULE.get(field, "")
            )
        else:
            self._right_note.setText(
                f"{len(options)} candidate(s). The second column is the "
                "reverse question: what already points at each of them."
            )
        self._middle_note.setText(
            rows[0].reason if len(rows) == 1 and rows[0].reason else ""
        )
        self._refresh_status()

    # -- editing -----------------------------------------------------------

    def _on_target_toggled(self, item: QTreeWidgetItem, column: int) -> None:
        if self._loading or column != 0:
            return
        path = item.data(0, _PATH_ROLE)
        if not path:
            return
        self._apply_target(
            Path(path), item.checkState(0) == Qt.CheckState.Checked
        )

    def _set_selected_targets(self, on: bool) -> None:
        chosen = [
            item for item in self._right.selectedItems()
            if item.data(0, _PATH_ROLE)
        ]
        if not chosen:
            self._status.setText(
                "Select one or more files on the right first, or tick them."
            )
            return
        for item in chosen:
            self._apply_target(Path(item.data(0, _PATH_ROLE)), on)

    def _apply_target(self, target: Path, on: bool) -> None:
        """Add or remove ``target`` on every file selected on the left."""
        for row in self._selected_rows():
            current = self._targets_now(row)
            if on and target not in current:
                current.append(target)
            elif not on and target in current:
                current.remove(target)
            self._pending[row.sidecar] = sorted(set(current))
        self._after_edit(sync_ticks=False)

    def _on_propose(self) -> None:
        for row in self._selected_rows():
            if row.proposed is None:
                continue
            self._pending[row.sidecar] = sorted(set(row.proposed))
        self._after_edit()

    def _on_clear(self) -> None:
        for row in self._selected_rows():
            self._pending[row.sidecar] = []
        self._after_edit()

    def _after_edit(self, *, sync_ticks: bool = True) -> None:
        """Show an edit WITHOUT rebuilding either list.

        This used to refill the left list and then rebuild the right one, and
        it segfaulted. Ticking a box emits ``itemChanged``; clearing the tree
        from inside that handler destroys the very item whose signal is still
        running, and Qt goes on using it. Not a Python exception: the process
        dies.

        Rebuilding was also the wrong thing to do even where it survived. The
        lists are what the user is pointing at, and having them rebuild,
        re-sort and lose their scroll position under the cursor on every tick
        is what made this feel unusable.

        So an edit updates the cells that changed and nothing else.
        ``sync_ticks`` is skipped when the edit CAME from a tick, because the
        box is already in the state the user just put it in.
        """
        if sync_ticks:
            self._sync_ticks()
        self._refresh_left_cells()
        self._refresh_status()

    def _sync_ticks(self) -> None:
        """Put the right list's boxes back in step with the pending edits."""
        rows = self._selected_rows()
        if not rows:
            return
        ticked = [set(self._targets_now(r)) for r in rows]
        self._loading = True
        try:
            for i in range(self._right.topLevelItemCount()):
                item = self._right.topLevelItem(i)
                key = item.data(0, _PATH_ROLE)
                if not key:
                    continue
                on = sum(1 for t in ticked if Path(key) in t)
                item.setCheckState(0, self._tick_state(on, len(rows)))
        finally:
            self._loading = False

    @staticmethod
    def _tick_state(on: int, total: int) -> Qt.CheckState:
        if on == 0:
            return Qt.CheckState.Unchecked
        if on == total:
            return Qt.CheckState.Checked
        # Tri-state rather than a guess: the selected files disagree and
        # flattening that silently would lose an edit.
        return Qt.CheckState.PartiallyChecked

    def _refresh_left_cells(self) -> None:
        """Re-word the left list's two right-hand columns, in place."""
        by_path = {str(r.path): r for r in self._rows}
        for i in range(self._left.topLevelItemCount()):
            item = self._left.topLevelItem(i)
            row = by_path.get(str(item.data(0, _PATH_ROLE) or ""))
            if row is None:
                continue
            status = (
                _EDITED if row.sidecar in self._pending else row.status
            )
            item.setText(1, self._describe(self._targets_now(row), row))
            item.setText(2, status)
            colour = _COLOURS.get(status)
            item.setForeground(
                2, QBrush(QColor(colour)) if colour else QBrush()
            )

    def _refresh_status(self) -> None:
        pending = len(self._pending)
        self._save_btn.setEnabled(bool(pending))
        if not pending:
            self._status.setText(
                "Nothing to save. Tick a file on the right to point at it."
            )
            return
        self._status.setText(
            f"{pending} file(s) edited and not yet saved. Saving writes them "
            "as one step, so one undo puts them all back."
        )

    # -- saving ------------------------------------------------------------

    def _on_save(self) -> None:
        if not self._pending:
            return
        field = self._field_name()
        edits = [
            (sidecar, field, [linkage.to_uri(self._root, t) for t in targets])
            for sidecar, targets in sorted(self._pending.items())
        ]
        try:
            self._changed += linkage.apply_links(
                self._root, edits,
                label=f"Set {field} on {len(edits)} file(s)",
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            QMessageBox.warning(
                self, "Could not save",
                f"Nothing was changed.\n\n{exc}",
            )
            return
        self._pending.clear()
        # Keep the user where they were. Saving re-reads the dataset, and
        # dropping the selection meant a pass over twenty fieldmaps was
        # twenty saves each followed by finding your place again.
        keep = [str(r.path) for r in self._selected_rows()]
        self._reload()
        self._reselect(keep)
        self._status.setText(f"Saved. {self._changed} file(s) changed so far.")

    def _reselect(self, keys: list[str]) -> None:
        if not keys:
            return
        wanted = set(keys)
        self._left.clearSelection()
        for i in range(self._left.topLevelItemCount()):
            item = self._left.topLevelItem(i)
            if str(item.data(0, _PATH_ROLE) or "") in wanted:
                item.setSelected(True)
                if self._left.currentItem() is None:
                    self._left.setCurrentItem(item)

    def _on_close(self) -> None:
        if self._pending:
            answer = QMessageBox.question(
                self, "Unsaved references",
                f"{len(self._pending)} file(s) have edits that are not "
                "written. Close and lose them?",
                QMessageBox.StandardButton.Cancel
                | QMessageBox.StandardButton.Discard,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Discard:
                return
        self.reject()

    def changed_count(self) -> int:
        return self._changed


__all__ = ["LinkageDialog"]

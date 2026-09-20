"""The repairs a dataset can be given, listed with what each would do.

Three actions live here, and they share a shape: work out what is missing,
show it, change nothing until the user says so, and record the change so it
can be taken back.

The companion-file list is a checklist rather than a button, because
generating files is the one action here that adds content to a dataset. A
reader cannot tell a generated stub from an authored file by looking at the
tree, so the user has to see the list before it is written, and every stub
carries the TODO convention that keeps the validator reporting it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..fixups import associations as assoc
from ..fixups.citation import citation_path, write_citation
from .dialog_chrome import WrapLabel, build_footer, build_header, card


class FixupsDialog(QDialog):
    """Dataset-wide repairs: companion files, citation file, TODO stamps."""

    def __init__(self, root: Path, parent: Optional[QWidget] = None,
                 *, report=None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        # Used only to say how many errors remain; never re-run here.
        self._report = report
        self._missing: list[assoc.MissingAssociation] = []
        self.setWindowTitle("Fix up this dataset")
        self.setModal(True)
        self.resize(820, 640)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(build_header(
            "Fix up this dataset",
            "Everything here is a repair BIDS Manager can make on its own, "
            "and every one of them changes files on disk as a <b>single step "
            "in the Editor's history</b>. Nothing is applied until you press "
            "its button.",
        ))
        body = QWidget()
        body.setObjectName("issue-dialog-body")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QVBoxLayout(body)
        v.setContentsMargins(18, 14, 18, 14)
        v.setSpacing(12)

        # --- Ready to share ---------------------------------------------
        ready_card, ready_body, _ = _repair_card(
            "Ready to share",
            "Whether this dataset is fit to hand to somebody else, which is "
            "not the same question as whether it is legal BIDS. These are "
            "BIDS Manager's opinions about sharing, not rules from the "
            "standard.",
        )
        self._ready_summary = _wrapped("", hint=True)
        ready_body.addWidget(self._ready_summary)
        self._ready = QTableWidget(0, 2)
        self._ready.setHorizontalHeaderLabels(["Check", "State"])
        self._ready.verticalHeader().setVisible(False)
        self._ready.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._ready.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._ready.horizontalHeader().setStretchLastSection(True)
        # Height is set from the row count in _refresh_readiness: a fixed
        # cap put a scrollbar on a six-row checklist and hid the last
        # check, which is the one a user most needs to see.
        self._ready.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        ready_body.addWidget(self._ready)
        v.addWidget(ready_card)

        # --- Companion files ------------------------------------------
        assoc_card, assoc_body, assoc_actions = _repair_card(
            "Missing companion files",
            "The tables and sidecars the standard associates with a "
            "recording. What can be read from the recording is read from it; "
            "the rest is written as a stub that stays visible to validation "
            "until you fill it in.",
        )
        self._assoc_hint = _wrapped("", hint=True)
        assoc_body.addWidget(self._assoc_hint)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["File", "What would be written"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMinimumHeight(120)
        assoc_body.addWidget(self._table, 1)

        for label, slot in (
            ("Select all", lambda: self._set_all(True)),
            ("Select none", lambda: self._set_all(False)),
        ):
            b = QPushButton(label)
            b.setObjectName("tb-btn-ghost")
            b.clicked.connect(slot)
            assoc_actions.insertWidget(assoc_actions.count() - 1, b)
        self._generate_btn = QPushButton("Generate selected")
        self._generate_btn.setObjectName("tb-btn")
        self._generate_btn.clicked.connect(self._on_generate)
        assoc_actions.addWidget(self._generate_btn)
        v.addWidget(assoc_card, 1)

        # --- Citation file --------------------------------------------
        cite_card, cite_body, cite_actions = _repair_card(
            "Citation file",
            "BIDS treats CITATION.cff as the single source of AUTHORSHIP, so "
            "writing one moves <b>Authors</b> out of "
            "dataset_description.json: stating it in both is an error. "
            "Everything else, the licence included, is copied into the "
            "citation file and <b>left where it is</b>.",
        )
        self._citation_hint = _wrapped("", hint=True)
        cite_body.addWidget(self._citation_hint)
        # Said before it happens, naming the field. A repair that silently
        # removes something a user typed is the worst kind.
        self._citation_moves = _wrapped("", hint=True)
        cite_body.addWidget(self._citation_moves)
        self._citation_btn = QPushButton("Write CITATION.cff")
        self._citation_btn.setObjectName("tb-btn")
        self._citation_btn.clicked.connect(self._on_citation)
        cite_actions.addWidget(self._citation_btn)
        v.addWidget(cite_card)

        # --- Inheritance ------------------------------------------------
        inherit_card, inherit_body, inherit_actions = _repair_card(
            "Repeated metadata",
            "A field with the same value in every file of a group belongs "
            "once, higher up, where BIDS inheritance applies it to all of "
            "them. Only fields that are identical everywhere are offered, so "
            "moving one cannot change what any file states.",
        )
        self._consolidate_hint = _wrapped("", hint=True)
        inherit_body.addWidget(self._consolidate_hint)
        self._consolidate_btn = QPushButton("Move shared fields up")
        self._consolidate_btn.setObjectName("tb-btn")
        self._consolidate_btn.clicked.connect(self._on_consolidate)
        inherit_actions.addWidget(self._consolidate_btn)
        v.addWidget(inherit_card)

        # --- TODO placeholders -----------------------------------------
        todo_card, todo_body, todo_actions = _repair_card(
            "Unanswered metadata",
            "Writes the literal TODO into every missing required and "
            "recommended sidecar field, so what is unanswered is visible in "
            "the file and reported by validation instead of being an absence "
            "nobody notices. Existing values are never overwritten.",
        )
        todo_body.addWidget(_wrapped(
            "This runs the metadata engine, which also refreshes "
            "participants.tsv and the scans tables.",
            hint=True,
        ))

        # How much to mark. Seeded from the same setting the post-convert
        # chain uses, so a dataset gets the same result whichever moment the
        # user chooses, and changing it here changes it for both.
        scope_row = QHBoxLayout()
        scope_row.setSpacing(8)
        scope_row.addWidget(QLabel("Mark which fields:"))
        self._todo_scope = QComboBox()
        self._todo_scope.setObjectName("ent-input")
        for value, label in (
            ("required", "Required only"),
            ("recommended", "Required and recommended"),
            ("optional", "Everything declared, including optional"),
        ):
            self._todo_scope.addItem(label, userData=value)
        self._todo_scope.setToolTip(
            "The scopes nest. A field whose type admits no honest marker (a "
            "number, a boolean, a controlled vocabulary) is left absent and "
            "reported rather than given a value nobody stated, so a wider "
            "scope never introduces a validation error."
        )
        self._todo_scope.currentIndexChanged.connect(self._on_scope_changed)
        scope_row.addWidget(self._todo_scope, 1)
        todo_body.addLayout(scope_row)

        self._todo_unmarkable = _wrapped("", hint=True)
        todo_body.addWidget(self._todo_unmarkable)

        self._todo_btn = QPushButton("Mark missing metadata")
        self._todo_btn.setObjectName("tb-btn")
        self._todo_btn.clicked.connect(self._on_stamp_todos)
        todo_actions.addWidget(self._todo_btn)
        v.addWidget(todo_card)

        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("issue-dialog-scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(build_footer(buttons))

        self.refresh()

    def _on_scope_changed(self, _index: int) -> None:
        """Persist the choice, so Fix ups and the post-convert chain agree."""
        from .app_settings import AppSettings

        settings = AppSettings.load()
        settings.metadata_fill_scope = (
            self._todo_scope.currentData() or "recommended"
        )
        settings.save()

    # -- state ---------------------------------------------------------

    def refresh(self) -> None:
        """Re-scan the dataset and redraw every section."""
        from .app_settings import AppSettings

        scope = AppSettings.load().metadata_fill_scope
        index = self._todo_scope.findData(scope)
        if index >= 0 and index != self._todo_scope.currentIndex():
            self._todo_scope.blockSignals(True)
            self._todo_scope.setCurrentIndex(index)
            self._todo_scope.blockSignals(False)
        self._refresh_readiness()
        self._missing = assoc.find_missing(self._root)
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._missing))
        for r, item in enumerate(self._missing):
            box = QCheckBox(item.rel)
            box.setChecked(True)
            self._table.setCellWidget(r, 0, box)
            what = QLabel(item.label)
            what.setToolTip(f"Belongs to {item.source.name}")
            self._table.setCellWidget(r, 1, what)
        self._table.resizeColumnToContents(0)
        # An empty table is a large blank rectangle that reads as a loading
        # failure. When there is nothing to fix, the sentence is the answer.
        self._table.setVisible(bool(self._missing))
        if self._missing:
            self._assoc_hint.setText(
                f"{len(self._missing)} file(s) the standard associates with a "
                "recording are not there. What can be read from the recording "
                "is read from it; the rest is written as a stub that stays "
                "invalid, and visible to validation, until you fill it in."
            )
        else:
            self._assoc_hint.setText(
                "Every recording has the companion files the standard "
                "associates with it."
            )
        self._generate_btn.setEnabled(bool(self._missing))

        from ..fixups.citation import fields_moved_by_citation

        moving = fields_moved_by_citation(self._root)
        self._citation_moves.setText(
            "Writing it will move <b>" + "</b>, <b>".join(moving)
            + "</b> out of dataset_description.json. Nothing else is removed."
            if moving else ""
        )
        self._citation_moves.setVisible(bool(moving))

        from ..editor.inheritance import consolidation_candidates

        self._shared = consolidation_candidates(self._root)
        if self._shared:
            fields = ", ".join(sorted({c.field for c in self._shared})[:6])
            self._consolidate_hint.setText(
                f"{len(self._shared)} field(s) are stated identically by "
                f"every sibling: {fields}. BIDS inheritance lets one copy "
                "higher up cover them all, which is fewer places for the "
                "same fact to disagree. A field that differs between files "
                "is never offered."
            )
        else:
            self._consolidate_hint.setText(
                "No field is repeated identically across every sibling."
            )
        self._consolidate_btn.setEnabled(bool(self._shared))

        exists = citation_path(self._root).exists()
        self._citation_hint.setText(
            "CITATION.cff is already here. Writing it again would overwrite "
            "it, so the button is disabled."
            if exists else
            "Generated from dataset_description.json. Only <b>Authors</b> "
            "moves: stating it in both files is an error. License, "
            "HowToAcknowledge and ReferencesAndLinks are copied and kept, so "
            "nothing you typed disappears from the description. The "
            "validator will note that it would rather those lived in one "
            "place only, which is a warning, not an error."
        )
        self._citation_btn.setEnabled(not exists)

    def _refresh_readiness(self) -> None:
        """The things that block sharing but are not validator errors."""
        from ..editor.readiness import check_dataset, summarise
        from .theme_manager import CUR

        checks = check_dataset(self._root, self._report)
        # Just the tally: the card's own description already says these are
        # our opinions about sharing rather than rules from the standard, and
        # saying it twice in adjacent lines reads as a stutter.
        self._ready_summary.setText(summarise(checks))
        pal = CUR()
        self._ready.setRowCount(0)
        self._ready.setRowCount(len(checks))
        for r, check in enumerate(checks):
            name = QTableWidgetItem(check.name)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name.setToolTip(check.why + (f"\n\n{check.fix}" if check.fix else ""))
            self._ready.setItem(r, 0, name)
            state = QTableWidgetItem(check.detail)
            state.setFlags(Qt.ItemFlag.ItemIsEnabled)
            state.setForeground(_qcolor(
                pal["success"] if check.passed else pal["warning"]
            ))
            state.setToolTip(check.fix or check.why)
            self._ready.setItem(r, 1, state)
        self._ready.resizeRowsToContents()
        self._ready.setFixedHeight(_table_height(self._ready))
        self._ready.resizeColumnToContents(0)

    def _boxes(self) -> list[QCheckBox]:
        out = []
        for r in range(self._table.rowCount()):
            w = self._table.cellWidget(r, 0)
            if isinstance(w, QCheckBox):
                out.append(w)
        return out

    def _set_all(self, state: bool) -> None:
        for b in self._boxes():
            b.setChecked(state)

    # -- actions -------------------------------------------------------

    def _on_generate(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        chosen = [
            item for b, item in zip(self._boxes(), self._missing)
            if b.isChecked()
        ]
        if not chosen:
            return
        created, failed = assoc.generate(self._root, chosen)
        if failed:
            QMessageBox.warning(
                self, "Some files were not written",
                "\n".join(f"{p}: {why}" for p, why in failed[:6]),
            )
        else:
            QMessageBox.information(
                self, "Companion files written",
                f"{len(created)} file(s) created.\n\nStubs carry TODO "
                "rows and stay invalid until you fill them in: an events "
                "table will now report an error, because TODO is not a valid "
                "onset. That is deliberate. A valid but empty table would "
                "look exactly like a recording that had no events.",
            )
        self.refresh()

    def _on_consolidate(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        from ..editor.inheritance import consolidate

        if not self._shared:
            return
        detail = "\n".join(
            f"  {c.field} = {c.value!r}  ({c.count} files -> {c.target_rel})"
            for c in self._shared[:12]
        )
        more = (f"\n  and {len(self._shared) - 12} more"
                if len(self._shared) > 12 else "")
        answer = QMessageBox.question(
            self, "Move shared fields up",
            "Each field below moves into a shared sidecar and is deleted "
            f"from the files that repeat it:\n\n{detail}{more}\n\n"
            "The effective metadata of every file is unchanged.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return
        written, failed = consolidate(self._root, self._shared)
        if failed:
            QMessageBox.warning(
                self, "Some files were not written",
                "\n".join(f"{p}: {why}" for p, why in failed[:6]),
            )
        else:
            QMessageBox.information(
                self, "Fields moved",
                f"{len(written)} file(s) changed. The effective metadata is "
                "the same; it is stated in fewer places.",
            )
        self.refresh()

    def _on_stamp_todos(self) -> None:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        from ..metadata.engine import run_metadata

        answer = QMessageBox.question(
            self, "Stamp TODO placeholders",
            "Every missing required and recommended field gets the literal "
            "TODO written into it, and participants.tsv and the scans tables "
            "are refreshed.\n\nValidation will then report each TODO, which "
            "is the point: an unanswered field becomes visible rather than "
            "absent.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            report = run_metadata(
                self._root, fill_todos=True,
                fill_scope=self._todo_scope.currentData() or "recommended",
                write_report=False,
            )
        except Exception as exc:  # noqa: BLE001 - report rather than crash
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Could not stamp placeholders", str(exc))
            return
        QApplication.restoreOverrideCursor()
        filled = sum(
            len(f.fields) for f in getattr(report, "todo_fills", []) or []
        )
        files = len(getattr(report, "todo_fills", []) or [])
        unmarkable = len(getattr(report, "unmarkable", []) or [])
        tail = (
            f"\n\n{unmarkable} field(s) could take no placeholder and were "
            "left absent: a number, a boolean or a controlled vocabulary has "
            "no honest marker, and inventing one would write a value nobody "
            "stated. They stay in the validation report."
            if unmarkable else ""
        )
        QMessageBox.information(
            self, "Missing metadata marked",
            f"{filled} field(s) across {files} file(s) now carry a "
            f"placeholder. Validation will report each one until it is "
            f"answered.{tail}",
        )
        self.refresh()

    def _on_citation(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        path = write_citation(self._root)
        if path is None:
            return
        QMessageBox.information(
            self, "CITATION.cff written",
            "Written from dataset_description.json, and the fields the "
            "citation file now owns were removed from the description so "
            "the two cannot disagree.",
        )
        self.refresh()


def _qcolor(spec: str):
    from PyQt6.QtGui import QColor

    return QColor(spec)


def _wrapped(text: str, *, hint: bool = False) -> QLabel:
    """A word-wrapped label that takes the height its text needs, no more.

    See :class:`~bidsmgr.gui.dialog_chrome.WrapLabel` for why a plain wrapped
    QLabel gets this wrong in both directions.
    """
    label = WrapLabel(text)
    if hint:
        label.setObjectName("dlg-hint")
    return label


def _table_height(table) -> int:
    """Exactly tall enough for every row, so nothing is scrolled out of view.

    A checklist that hides its last item is worse than no checklist: the user
    reads five green rows and stops.
    """
    height = table.horizontalHeader().height() + 2 * table.frameWidth()
    for row in range(table.rowCount()):
        height += table.rowHeight(row)
    return height


def _repair_card(title: str, description: str):
    """One repair, as a card: what it is, what it does, and its button.

    Five repairs stacked as bare headings and right-aligned buttons read as
    one long form where every control looks equally urgent. A card per repair
    groups the explanation with the thing it explains and gives every action
    the same place to be, which is the bottom-right of its own block.

    Returns ``(card, body_layout, action_row)``.
    """
    frame, layout = card()
    heading = QLabel(title)
    heading.setObjectName("dlg-section-title")
    layout.addWidget(heading)
    layout.addWidget(_wrapped(description, hint=True))

    body = QVBoxLayout()
    body.setSpacing(8)
    layout.addLayout(body, 1)

    actions = QHBoxLayout()
    actions.setSpacing(8)
    actions.addStretch(1)
    layout.addLayout(actions)
    return frame, body, actions


__all__ = ["FixupsDialog"]

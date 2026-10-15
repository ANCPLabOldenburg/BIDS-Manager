"""Give the chosen people new subject numbers, in one step.

The plan comes from ``inventory/renumber.plan_renumber``: each distinct
person gets the next number free in the dataset, in the rest of the table
and in the project's other scans. The table shows who, how many series, the
number now and the number they get; a new number can be typed over, and is
checked as it is typed. Apply writes it as ONE undoable change.
"""

from __future__ import annotations

import re
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from ..inventory.renumber import Person
from .theme_manager import CUR
from .widgets.card import card
from .window_flags import give_window_buttons

_LABEL = re.compile(r"^[0-9A-Za-z]+$")


class RenumberDialog(QDialog):
    def __init__(self, people: list[Person], taken: set[str],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        give_window_buttons(self)
        self.setWindowTitle("Give new subject numbers")
        self.resize(760, 420)
        self._people = list(people)
        self._taken = set(taken)
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)
        intro = QLabel(
            "Each person gets the next free number. Numbers already used in the dataset, "
            "by the rest of this table or by the project's other scans are skipped. Type "
            "over a new number to choose your own.")
        intro.setObjectName("dlg-hint")
        intro.setWordWrap(True)
        v.addWidget(intro)

        self.table = QTreeWidget()
        self.table.setObjectName("renumber-table")
        self.table.setHeaderLabels(["Person", "Acquired", "Series", "Now", "New"])
        self.table.setRootIsDecorated(False)
        self.table.setUniformRowHeights(True)
        header = self.table.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        for person in self._people:
            item = QTreeWidgetItem([person.shown or "(no identifier)", person.acquired or "-",
                                    str(len(person.rows)), f"sub-{person.current}", person.new])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(0, "The PatientID the scanner recorded (names are not shown)")
            item.setToolTip(4, "Double-click to type another number")
            self.table.addTopLevelItem(item)
        self.table.itemChanged.connect(lambda *_a: self._check())
        self.table.itemDoubleClicked.connect(self._edit_new)
        v.addWidget(card(self.table), 1)

        self.problem = QLabel("")
        self.problem.setObjectName("dlg-hint")
        self.problem.setWordWrap(True)
        v.addWidget(self.problem)

        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("tb-btn")
        cancel.clicked.connect(self.reject)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("tb-btn-primary")
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.apply_button)
        v.addLayout(row)
        self._check()

    def _edit_new(self, item: QTreeWidgetItem, column: int) -> None:
        if column == 4:
            self.table.editItem(item, 4)

    def mapping(self) -> list[tuple[Person, str]]:
        """Each person and the label they get (as shown, edits included)."""
        out = []
        for i, person in enumerate(self._people):
            label = self.table.topLevelItem(i).text(4).strip()
            out.append((person, label[4:] if label.startswith("sub-") else label))
        return out

    def _check(self) -> None:
        labels = [label for _p, label in self.mapping()]
        problem = ""
        for label in labels:
            if not _LABEL.match(label):
                problem = f"'{label}' cannot be a subject label: letters and digits only."
                break
            if label in self._taken:
                problem = f"sub-{label} is already used; its data would be merged."
                break
        if not problem and len(set(labels)) != len(labels):
            problem = "Two people would get the same number."
        pal = CUR()
        self.problem.setText(problem)
        self.problem.setStyleSheet(f"color:{pal['warning']};" if problem else "")
        hard = problem and ("cannot" in problem or "Two people" in problem)
        self.apply_button.setEnabled(not hard)


def apply_renumbering(model, mapping: list[tuple[Person, str]]) -> int:
    """Write the new labels as one undoable change; returns rows changed."""
    changed = 0
    with model.edit_group():
        for person, label in mapping:
            if label == person.current:
                continue
            for row in person.rows:
                if model.set_entity(row, "subject", label):
                    changed += 1
    return changed


__all__ = ["RenumberDialog", "apply_renumbering"]

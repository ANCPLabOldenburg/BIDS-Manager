"""What is in this dataset, at a glance.

The question a user has on opening a dataset they did not make, or coming back
to one after a month: what is in it, is it evenly filled, and where is the
work. The file tree cannot answer it, because the answer is a shape across
hundreds of files.

Everything here is a count read from :mod:`bidsmgr.editor.dashboard`. No
scores, no indexes: "41 of 60 declared fields are answered" is a figure a user
can check, and "completeness 68%" is one they can only believe.

The bars are drawn rather than charted. A bar is a rectangle whose width is a
proportion, and a plotting library buys nothing for that while costing a
dependency, a theme that does not match, and a widget that does not scale with
the font.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..editor import dashboard as dash
from .dialog_chrome import build_footer, build_header, card, hint
from .theme_manager import CUR, scaled_px


class _Bar(QWidget):
    """One proportion, as a rectangle. Nothing more is needed."""

    def __init__(self, fraction: float, token: str = "accent", parent=None):
        super().__init__(parent)
        self._fraction = max(0.0, min(float(fraction), 1.0))
        self._token = token
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(scaled_px(8))

    def paintEvent(self, event):  # noqa: N802 - Qt signature
        del event
        pal = CUR()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        radius = self.height() / 2.0
        track = QColor(pal["border"])
        painter.setBrush(track)
        painter.drawRoundedRect(self.rect(), radius, radius)

        if self._fraction <= 0:
            return
        filled = QColor(pal.get(self._token, pal["accent"]))
        painter.setBrush(filled)
        width = max(int(self.width() * self._fraction), self.height())
        painter.drawRoundedRect(
            QRect(0, 0, width, self.height()), radius, radius,
        )


def _stat(label: str, value: str, tone: str = "text") -> QWidget:
    """One headline number with its name under it."""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(1)
    number = QLabel(value)
    number.setObjectName(f"stat-value-{tone}")
    caption = QLabel(label)
    caption.setObjectName("stat-caption")
    layout.addWidget(number)
    layout.addWidget(caption)
    return box


def _table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.horizontalHeader().setSectionResizeMode(
        0, QHeaderView.ResizeMode.Stretch
    )
    for column in range(1, len(headers)):
        table.horizontalHeader().setSectionResizeMode(
            column, QHeaderView.ResizeMode.ResizeToContents
        )
    return table


def _fit(table: QTableWidget) -> None:
    """Exactly tall enough for its rows. A summary that scrolls is not one."""
    table.resizeRowsToContents()
    height = table.horizontalHeader().height() + 2 * table.frameWidth()
    for row in range(table.rowCount()):
        height += table.rowHeight(row)
    table.setFixedHeight(height)


def _cell(text: str, tone: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
    if tone:
        item.setForeground(QColor(CUR()[tone]))
    return item


class DashboardDialog(QDialog):
    """The dataset, counted."""

    file_selected = pyqtSignal(Path)

    def __init__(
        self, root: Path, report=None, parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._board = dash.build(self._root, report)
        self.setWindowTitle("Dashboard")
        self.setModal(False)
        self.resize(820, 640)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        board = self._board
        outer.addWidget(build_header(
            board.name or self._root.name,
            f"{self._root}"
            + (f"  ·  BIDS {board.bids_version}" if board.bids_version else ""),
        ))

        from .dialog_chrome import scrollable_body

        scroll, body = scrollable_body()
        self._add_headline(body)
        self._add_modalities(body)
        self._add_subjects(body)
        self._add_findings(body)
        self._add_completeness(body)
        body.addStretch(1)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(build_footer(buttons))

    # ------------------------------------------------------------------

    def _add_headline(self, body: QVBoxLayout) -> None:
        board = self._board
        frame, layout = card()
        row = QHBoxLayout()
        row.setSpacing(28)
        row.addWidget(_stat("subjects", str(len(board.subjects))))
        row.addWidget(_stat("sessions", str(board.sessions)))
        row.addWidget(_stat("modalities", str(len(board.modalities))))
        row.addWidget(_stat("files", f"{board.total_files:,}"))
        row.addWidget(_stat("size", dash.human_bytes(board.total_bytes)))
        if board.validated:
            row.addWidget(_stat("errors", str(board.errors), "err"))
            row.addWidget(_stat("warnings", str(board.warnings), "warn"))
        row.addStretch(1)
        layout.addLayout(row)

        # A participants table that is absent and one that is empty are
        # different facts, and a zero would read as the second.
        if board.participants is None:
            layout.addWidget(hint(
                "There is no <b>participants.tsv</b>. A dataset with more "
                "than one subject needs one for an analyst to have anything "
                "to group by."
            ))
        elif board.participants != len(board.subjects):
            layout.addWidget(hint(
                f"<b>participants.tsv</b> lists {board.participants} "
                f"row(s) but the tree holds {len(board.subjects)} subject(s). "
                "Those should agree."
            ))
        if not board.validated:
            layout.addWidget(hint(
                "Not validated yet, so this shows what is here but not what "
                "is wrong with it. Run <b>Validate dataset</b> and reopen."
            ))
        body.addWidget(frame)

    def _add_modalities(self, body: QVBoxLayout) -> None:
        board = self._board
        if not board.modalities:
            return
        frame, layout = card("What it holds")
        layout.addWidget(hint(
            "How much of what the standard declares for each datatype is "
            "actually answered. A placeholder counts as unanswered, because "
            "that is what it is."
        ))
        table = _table([
            "Datatype", "Subjects", "Recordings", "Answered", "TODO",
            "Errors", "Warnings",
        ])
        for modality in board.modalities:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, _cell(modality.datatype))
            table.setItem(row, 1, _cell(str(modality.subjects)))
            table.setItem(row, 2, _cell(str(modality.recordings)))
            coverage = modality.coverage
            table.setItem(row, 3, _cell(
                "not declared" if coverage is None else
                f"{modality.answered} of {modality.declared}"
            ))
            table.setItem(row, 4, _cell(
                str(modality.placeholders) if modality.placeholders else "-",
                "warning" if modality.placeholders else "",
            ))
            table.setItem(row, 5, _cell(
                str(modality.errors) if modality.errors else "-",
                "error" if modality.errors else "",
            ))
            table.setItem(row, 6, _cell(
                str(modality.warnings) if modality.warnings else "-",
                "warning" if modality.warnings else "",
            ))
        _fit(table)
        layout.addWidget(table)
        body.addWidget(frame)

    def _add_subjects(self, body: QVBoxLayout) -> None:
        board = self._board
        if not board.subjects:
            return
        frame, layout = card("Subjects")
        uneven = board.uneven_subjects
        outliers = board.outlier_subjects
        if uneven:
            shown = ", ".join(uneven[:6])
            more = "" if len(uneven) <= 6 else f" and {len(uneven) - 6} more"
            verb = "does not" if len(uneven) == 1 and not more else "do not"
            layout.addWidget(hint(
                f"<b>{shown}{more}</b> {verb} carry what most of the others "
                "do. A subject missing a modality is invisible in a file "
                "tree and obvious in a table."
            ))
        for label, extra in outliers[:3]:
            layout.addWidget(hint(
                f"<b>{label}</b> is the only one with "
                f"<b>{', '.join(extra)}</b>. Usually a conversion that went "
                "further than the rest, or a file in the wrong subject."
            ))
        if not uneven and not outliers and len(board.subjects) > 1:
            layout.addWidget(hint(
                "Every subject carries the same modalities."
            ))

        table = _table([
            "Subject", "Sessions", "Files", "Modalities", "Errors", "Warnings",
        ])
        for subject in board.subjects:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, _cell(subject.label))
            table.setItem(row, 1, _cell(str(subject.sessions) or "-"))
            table.setItem(row, 2, _cell(str(subject.files)))
            table.setItem(row, 3, _cell(", ".join(subject.datatypes) or "-"))
            table.setItem(row, 4, _cell(
                str(subject.errors) if subject.errors else "-",
                "error" if subject.errors else "",
            ))
            table.setItem(row, 5, _cell(
                str(subject.warnings) if subject.warnings else "-",
                "warning" if subject.warnings else "",
            ))
        _fit(table)
        layout.addWidget(table)
        body.addWidget(frame)

    def _add_findings(self, body: QVBoxLayout) -> None:
        board = self._board
        if not board.top_rules:
            return
        frame, layout = card("Where the findings are")
        layout.addWidget(hint(
            "The same rule repeated is one mistake made many times, not many "
            "mistakes. Fixing the top row usually fixes most of the count."
        ))
        worst = max(count for _rule, count in board.top_rules)
        for rule, count in board.top_rules:
            line = QHBoxLayout()
            line.setSpacing(10)
            name = QLabel(rule)
            name.setObjectName("dlg-hint")
            name.setMinimumWidth(scaled_px(210))
            line.addWidget(name)
            line.addWidget(_Bar(count / worst, "warning"), 1)
            value = QLabel(str(count))
            value.setObjectName("dlg-hint")
            value.setMinimumWidth(scaled_px(44))
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            line.addWidget(value)
            layout.addLayout(line)
        body.addWidget(frame)

    def _add_completeness(self, body: QVBoxLayout) -> None:
        board = self._board
        frame, layout = card("Ready to share")
        layout.addWidget(hint(
            "The files that describe the dataset rather than any one "
            "recording. These are what a reader opens first."
        ))
        line = QHBoxLayout()
        line.setSpacing(6)
        for name in board.present:
            chip = QLabel(name)
            chip.setObjectName("dash-have")
            line.addWidget(chip)
        line.addStretch(1)
        layout.addLayout(line)
        if board.absent:
            layout.addWidget(hint(
                "Missing: <b>" + "</b>, <b>".join(board.absent) + "</b>. "
                "Fix ups can write several of them."
            ))
        body.addWidget(frame)


__all__ = ["DashboardDialog"]

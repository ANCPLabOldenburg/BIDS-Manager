"""The Logs window: every operation's log file, newest first.

Each scan, conversion, metadata pass and validation writes a file
(``util/oplog``). This lists them with what ran, when and how it ended, and
shows the chosen one beside the list in the Log panel's colours. Opening the
file in the system's editor or its folder is one click away.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ..util import oplog
from .widgets.card import card
from .widgets.run_log import RunLog
from .window_flags import give_window_buttons
from .widgets.splitter import Splitter

#: The tail of a file read to learn how the run ended.
_TAIL_BYTES = 4096
#: Lines shown of a long log (its end, where the outcome is).
_SHOW_LINES = 4000

_VERBS = {"scan": "Scan", "convert": "Conversion", "metadata": "Metadata",
          "validate": "Validation", "quality": "Quality check"}


_FOOTER = re.compile(
    r"bidsmgr\.oplog: \w+ (?P<state>finished|stopped by .*?) after (?P<secs>[\d.]+) s: "
    r"(?P<warn>\d+) warning\(s\), (?P<err>\d+) error\(s\)")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def describe(path: Path) -> tuple[str, str, str]:
    """``(when, what, outcome)`` for a log file, from its name and its
    footer."""
    raw_stamp, _, verb = path.stem.partition("_")
    try:
        utc = datetime.strptime(raw_stamp[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        when = utc.astimezone().strftime("%d %b %Y, %H:%M")
    except ValueError:
        when = raw_stamp
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - _TAIL_BYTES))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return when, _VERBS.get(verb, verb.capitalize()), "Unreadable"
    found = list(_FOOTER.finditer(tail))
    if not found:
        outcome = "Running, or ended without a word"
    else:
        m = found[-1]
        state = "Finished" if m["state"] == "finished" else "Stopped"
        parts = [state]
        if int(m["warn"]):
            parts.append(_plural(int(m["warn"]), "warning"))
        if int(m["err"]):
            parts.append(_plural(int(m["err"]), "error"))
        outcome = ", ".join(parts)
    return when, _VERBS.get(verb, verb.capitalize() or "Operation"), outcome


class LogHistoryDialog(QDialog):
    def __init__(self, root: Optional[Path], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        give_window_buttons(self)
        self.setWindowTitle("Logs")
        self.resize(1040, 640)
        self._root = root
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)
        where = oplog.log_dir(root)
        intro = QLabel(
            "Every scan, conversion, metadata pass and validation writes a log here, "
            f"in {where}. Times are your local time.")
        intro.setObjectName("dlg-hint")
        intro.setWordWrap(True)
        v.addWidget(intro)

        split = Splitter(Qt.Orientation.Horizontal)
        split.setObjectName("card-split")
        self.list = QTreeWidget()
        self.list.setObjectName("log-history")
        self.list.setHeaderLabels(["When", "What", "How it ended"])
        self.list.setRootIsDecorated(False)
        self.list.setUniformRowHeights(True)
        self.list.setAlternatingRowColors(False)
        self.list.currentItemChanged.connect(self._show)
        split.addWidget(card(self.list))

        self._stack = QStackedWidget()
        self.view = RunLog()
        self._empty = QLabel("No logs yet. Every scan, conversion, metadata pass and "
                             "validation will leave one here.")
        self._empty.setObjectName("pane-hint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._stack.addWidget(self.view)
        self._stack.addWidget(self._empty)
        split.addWidget(card(self._stack, padding=10))
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([400, 600])
        v.addWidget(split, 1)

        row = QHBoxLayout()
        self.open_button = QPushButton("Open in the system's app")
        self.open_button.setObjectName("tb-btn")
        self.open_button.setToolTip("Open the chosen log in your text editor")
        self.open_button.clicked.connect(lambda: self._open(self._current()))
        self.folder_button = QPushButton("Show the folder")
        self.folder_button.setObjectName("tb-btn")
        self.folder_button.setToolTip(str(where))
        self.folder_button.clicked.connect(lambda: self._open(where))
        close = QPushButton("Close")
        close.setObjectName("tb-btn-primary")
        close.clicked.connect(self.accept)
        row.addWidget(self.open_button)
        row.addWidget(self.folder_button)
        row.addStretch(1)
        row.addWidget(close)
        v.addLayout(row)
        self.reload()

    def reload(self) -> None:
        self.list.clear()
        files = oplog.history(self._root)
        for path in files:
            when, what, outcome = describe(path)
            item = QTreeWidgetItem([when, what, outcome])
            item.setData(0, Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(0, str(path))
            self.list.addTopLevelItem(item)
        for col in range(2):
            self.list.resizeColumnToContents(col)
        has = bool(files)
        self.open_button.setEnabled(has)
        self._stack.setCurrentWidget(self.view if has else self._empty)
        if has:
            self.list.setCurrentItem(self.list.topLevelItem(0))

    def _current(self) -> Optional[Path]:
        item = self.list.currentItem()
        return Path(item.data(0, Qt.ItemDataRole.UserRole)) if item is not None else None

    def _show(self, current, _previous=None) -> None:
        self.view.clear_log()
        path = self._current() if current is not None else None
        if path is None:
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            lines = [f"error: could not read {path}: {exc}"]
        if len(lines) > _SHOW_LINES:
            lines = [f"(the first {len(lines) - _SHOW_LINES} lines are in the file)"] + \
                lines[-_SHOW_LINES:]
        self.view.append_lines(lines)
        self.view.verticalScrollBar().setValue(0)

    @staticmethod
    def _open(path: Optional[Path]) -> None:
        if path is None:
            return
        if not path.suffix:
            path.mkdir(parents=True, exist_ok=True)      # the folder, before any log
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


__all__ = ["LogHistoryDialog", "describe"]

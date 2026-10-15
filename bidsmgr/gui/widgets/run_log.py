"""The Converter's Log: runs with headings, warnings and errors in colour.

It used to be a plain rolling buffer: a scan, a conversion and the metadata
pass ran together with nothing to say where one ended, and nothing was ever
cleared (the Neuroimaging unit asked for an empty log per participant).
Each operation now starts a RUN with a heading, the panel can start empty for
every run (a setting, on by default), and the lines are kept as data, so a
theme change re-draws them in the new colours instead of leaving the old
ones baked into the text.

Every operation also writes a file (``util/oplog``); the panel shows what
happened now, the file is what happened, kept.
"""

from __future__ import annotations

import html
import re
from collections import deque
from datetime import datetime
from typing import Iterable, Optional

from PyQt6.QtWidgets import QPlainTextEdit

from ..theme_manager import CUR

#: Lines kept (and shown); the oldest go first.
MAX_LINES = 2000

_ERROR = re.compile(r"\b(error|errors|failed|failure|traceback|exception)\b", re.IGNORECASE)
_WARNING = re.compile(r"\b(warning|warnings|warn|skipped|not found|missing)\b", re.IGNORECASE)
_NONE = re.compile(r"\b0 (warning|error)", re.IGNORECASE)


def line_kind(text: str) -> str:
    """``error``, ``warning`` or ``info``, read from the words in the line
    ("0 warning(s), 0 error(s)" is good news, not a warning)."""
    plain = _NONE.sub("", text)
    if _ERROR.search(plain):
        return "error"
    if _WARNING.search(plain):
        return "warning"
    return "info"


class RunLog(QPlainTextEdit):
    """Read-only text in the code typeface (``#dock-log`` in the stylesheet)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("dock-log")
        self.setReadOnly(True)
        self.setMaximumBlockCount(MAX_LINES)
        # (kind, text): kind is "run", "error", "warning" or "info".
        self._lines: deque[tuple[str, str]] = deque(maxlen=MAX_LINES)
        self.runs = 0

    # -- writing --------------------------------------------------------------

    def begin_run(self, title: str, detail: str = "", *, clear: bool = False) -> None:
        """A heading for an operation; ``clear`` empties the panel first."""
        if clear:
            self.clear_log()
        self.runs += 1
        stamp = datetime.now().strftime("%H:%M")
        text = f"{title}  ·  {stamp}" + (f"  ·  {detail}" if detail else "")
        self._add("run", text)

    def append_lines(self, lines: Iterable[str]) -> None:
        for raw in lines:
            for text in str(raw).splitlines() or [""]:
                self._add(line_kind(text), text)

    def clear_log(self) -> None:
        self._lines.clear()
        self.clear()

    def lines(self) -> list[tuple[str, str]]:
        return list(self._lines)

    def plain_text(self) -> str:
        return "\n".join(text for _kind, text in self._lines)

    # -- drawing ----------------------------------------------------------------

    def repaint_for_palette(self, _pal: Optional[dict] = None) -> None:
        """Draw every kept line again in the theme's colours."""
        kept = list(self._lines)
        self.clear()
        self.setUpdatesEnabled(False)
        try:
            for kind, text in kept:
                self._draw(kind, text)
        finally:
            self.setUpdatesEnabled(True)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def _add(self, kind: str, text: str) -> None:
        self._lines.append((kind, text))
        self._draw(kind, text)

    def _draw(self, kind: str, text: str) -> None:
        if kind == "run" and self.document().characterCount() > 1:
            self.appendHtml("&nbsp;")          # air before a new run
        self.appendHtml(self._html(kind, text))

    @staticmethod
    def _html(kind: str, text: str) -> str:
        pal = CUR()
        body = html.escape(text)
        if kind == "run":
            return (f'<span style="color:{pal["accent"]}; font-weight:600; '
                    f'white-space:pre-wrap;">{body}</span>')
        colour = {"error": pal["error"], "warning": pal["warning"]}.get(kind, pal["text"])
        return f'<span style="color:{colour}; white-space:pre-wrap;">{body}</span>'


__all__ = ["MAX_LINES", "RunLog", "line_kind"]

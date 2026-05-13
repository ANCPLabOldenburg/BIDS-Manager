"""Append-only JSONL event log.

Reference: architecture.md §9. The log is the **only** source of truth
for project history. Every other artifact in the bundle (state cache,
provenance side-table) is rebuildable from this file.

On-disk format: one JSON object per line, UTF-8, LF-terminated. A
malformed line (truncated write, manual edit, corruption) raises on
read; we do not silently skip — bad data deserves a loud failure.

Concurrency: a single project bundle is owned by one writer at a time.
We rely on POSIX file-append atomicity for individual line writes (a
single ``write(2)`` of < PIPE_BUF bytes is atomic; a typical event line
is well under that). Multi-writer support is not in scope.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from .types import Event, dump_event, parse_event

log = logging.getLogger(__name__)


class EventLog:
    """Wrapper around a single ``events.jsonl`` file.

    Construct by passing the absolute path; the file is created on first
    :meth:`append` if it does not exist. Reading an absent log yields
    zero events (so a freshly initialised bundle behaves the same as one
    with no recorded events yet).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, event: Event) -> None:
        """Serialise ``event`` and append one line to the log.

        Each call opens, writes, and closes the file. This is slower
        than holding a handle, but it makes the writer crash-safe — if
        the process dies mid-session, partially flushed lines cannot
        leak into the log because every write is its own flushed open.
        """
        record = dump_event(event)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def __iter__(self) -> Iterator[Event]:
        """Yield parsed events in chronological (file) order.

        Raises ``ValidationError`` (Pydantic) on a malformed record so
        the caller knows the log is corrupt instead of silently
        producing wrong state from a truncated event.
        """
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.rstrip("\n")
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise EventLogCorrupt(
                        f"{self.path}:{lineno}: invalid JSON ({exc})"
                    ) from exc
                try:
                    yield parse_event(record)
                except ValidationError as exc:
                    raise EventLogCorrupt(
                        f"{self.path}:{lineno}: event fails schema "
                        f"({type(exc).__name__})"
                    ) from exc

    def __len__(self) -> int:
        """Count events without parsing them. Linear in file size."""
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def truncate_last(self) -> bool:
        """Drop the last event from the log. Returns ``True`` if anything was dropped.

        This is the primitive the GUI's "undo" button rides on. Implemented
        by rewriting the file minus its final line — safe for the modest
        log sizes the project targets (< 50k events per
        architecture.md §11). If undo becomes a bottleneck we can switch
        to a tombstone-event model.
        """
        if not self.path.exists():
            return False
        lines = self.path.read_text(encoding="utf-8").splitlines(keepends=True)
        # Strip trailing blank lines and find the last real event.
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return False
        lines.pop()
        # Rewrite via a temp file so a crash mid-rewrite cannot corrupt
        # the log: a torn write would leave the original in place.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text("".join(lines), encoding="utf-8")
        tmp.replace(self.path)
        return True


class EventLogCorrupt(RuntimeError):
    """Raised when a JSONL line cannot be parsed back into an :data:`Event`."""


__all__ = ["EventLog", "EventLogCorrupt"]

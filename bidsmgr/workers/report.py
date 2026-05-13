"""``QThread`` bridge that returns an in-memory :class:`ValidationReport`.

Companion to :class:`bidsmgr.workers.ValidateWorker` (which wraps the
CLI verb and writes a JSON file to disk). The Editor view wants the
report object itself — sidecar form, tree badges, and validation
panel all bind against :class:`bidsmgr.editor.types.ValidationReport`
in memory, not a disk artifact.

Signal contract:

* ``progress(str)``                  — INFO-level log lines from
  :mod:`bidsmgr.editor.validator` while the run is in flight.
* ``finished_with_report(report, root)`` — emitted once with the
  finished :class:`ValidationReport` and the absolute BIDS root.
* ``failed(str)``                    — formatted traceback on any
  unexpected exception.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class _LogToSignal(logging.Handler):
    def __init__(self, sink) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink(record.getMessage())
        except Exception:  # pragma: no cover
            pass


class ReportWorker(QThread):
    """Run :func:`bidsmgr.editor.validator.validate` on a worker thread."""

    progress = pyqtSignal(str)
    finished_with_report = pyqtSignal(object, object)  # (ValidationReport, Path)
    failed = pyqtSignal(str)

    def __init__(
        self,
        bids_root: Path,
        *,
        strict: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._bids_root = Path(bids_root)
        self._strict = strict

    def run(self) -> None:
        from ..editor import validate

        handler = _LogToSignal(self.progress.emit)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger("bidsmgr.editor.validator")
        logger.addHandler(handler)
        prev = logger.level
        logger.setLevel(logging.INFO)

        try:
            self.progress.emit(f"Validating {self._bids_root}")
            report = validate(self._bids_root, strict=self._strict)
            self.progress.emit(
                f"Validation done: {report.counts.get('ok', 0)} ok, "
                f"{report.counts.get('warn', 0)} warn, "
                f"{report.counts.get('err', 0)} err"
            )
            self.finished_with_report.emit(report, self._bids_root)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev)


__all__ = ["ReportWorker"]

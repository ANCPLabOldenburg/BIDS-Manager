"""``QThread`` bridge for ``bidsmgr-validate``.

Signal contract mirrors the other workers in :mod:`bidsmgr.workers`.
``finished_with_result(rc, target)`` rc is 0 if every BIDS root passed
validation, 1 if any errored.
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


class ValidateWorker(QThread):
    """Run :func:`bidsmgr.cli.validate.run_validate_cli` on a worker thread."""

    progress = pyqtSignal(str)
    finished_with_result = pyqtSignal(int, object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        target: Path,
        *,
        strict: bool = False,
        strict_warn: bool = False,
        html_report: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._target = Path(target)
        self._strict = strict
        self._strict_warn = strict_warn
        self._html_report = html_report

    def run(self) -> None:
        from ..cli.validate import run_validate_cli

        handler = _LogToSignal(self.progress.emit)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger("bidsmgr.cli.validate")
        logger.addHandler(handler)
        prev = logger.level
        logger.setLevel(logging.INFO)

        try:
            self.progress.emit(f"Validating {self._target}")
            rc = run_validate_cli(
                self._target,
                strict=self._strict,
                strict_warn=self._strict_warn,
                html_report=self._html_report,
            )
            verdict = "clean" if rc == 0 else f"errored (rc={rc})"
            self.progress.emit(f"Validation {verdict}")
            self.finished_with_result.emit(rc, self._target)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev)


__all__ = ["ValidateWorker"]

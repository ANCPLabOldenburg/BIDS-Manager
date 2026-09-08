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
        schema: str | None = None,
        max_rows: int = 1000,
        flag_todos: bool = True,
        html_report: bool = False,
        datasets=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        # The dataset names this run may touch. Without it the walk finds
        # every BIDS root under the target, and projects are kept side by
        # side, so one project's metadata reached all of them.
        self._datasets = list(datasets) if datasets is not None else None
        self._target = Path(target)
        self._strict = strict
        self._strict_warn = strict_warn
        self._schema = schema or None
        self._max_rows = max_rows
        self._flag_todos = flag_todos
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
                schema=self._schema,
                max_rows=self._max_rows,
                flag_todos=self._flag_todos,
                html_report=self._html_report,
                datasets=self._datasets,
            )
            # rc=1 means the DATASET did not pass, which is an ordinary result
            # and the whole point of running a validator. Only rc=2 means the
            # validator could not run at all. Calling both "errored" told the
            # user their tool had broken when it had in fact worked and found
            # something.
            verdict = {
                0: "passed: no errors",
                1: "finished: the dataset has errors, see the report",
                2: "could not run: no BIDS dataset found at that path",
            }.get(rc, f"finished with an unexpected status (rc={rc})")
            self.progress.emit(f"Validation {verdict}")
            self.finished_with_result.emit(rc, self._target)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev)


__all__ = ["ValidateWorker"]

"""``QThread`` bridge for ``bidsmgr-metadata``.

Mirrors :class:`bidsmgr.workers.ScanWorker` /
:class:`bidsmgr.workers.ConvertWorker`:

* ``progress(str)`` — log records forwarded from
  ``bidsmgr.cli.metadata`` while the run is in flight.
* ``finished_with_result(returncode, target)`` — non-zero rc means at
  least one BIDS root errored during metadata generation.
* ``failed(traceback)`` — uncaught exception in the worker.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Optional

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


class MetadataWorker(QThread):
    """Run :func:`bidsmgr.cli.metadata.run_metadata_cli` on a worker thread."""

    progress = pyqtSignal(str)
    finished_with_result = pyqtSignal(int, object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        target: Path,
        *,
        inventory_tsv: Optional[Path] = None,
        name: Optional[str] = None,
        fill_todos: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._target = Path(target)
        self._inventory_tsv = Path(inventory_tsv) if inventory_tsv else None
        self._name = name
        self._fill_todos = fill_todos

    def run(self) -> None:
        from ..cli.metadata import run_metadata_cli

        handler = _LogToSignal(self.progress.emit)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger("bidsmgr.cli.metadata")
        logger.addHandler(handler)
        prev = logger.level
        logger.setLevel(logging.INFO)

        try:
            self.progress.emit(f"Running metadata on {self._target}")
            rc = run_metadata_cli(
                self._target,
                inventory_tsv=self._inventory_tsv,
                name=self._name,
                fill_todos=self._fill_todos,
            )
            verdict = "completed" if rc == 0 else f"completed with rc={rc}"
            self.progress.emit(f"Metadata {verdict}")
            self.finished_with_result.emit(rc, self._target)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev)


__all__ = ["MetadataWorker"]

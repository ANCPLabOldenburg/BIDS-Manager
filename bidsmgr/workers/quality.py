"""``QThread`` bridge for the dataset quality check (``bidsmgr.qc.run``).

The images are checked in separate processes (joblib) driven from this
thread, so the GUI never computes; each result is written into
``derivatives/bidsmgr-qc/`` as it arrives. ``request_stop`` stops after the
images already running.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


class QualityWorker(QThread):
    """Check ``paths`` of the dataset at ``root``."""

    #: (done, total, name of the image just finished)
    progress = pyqtSignal(int, int, str)
    #: The results, as JSON rows (``bidsmgr.qc.report.to_json``).
    finished_with_result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, root: Path, paths: list[Path], *, jobs: int = 1, config=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._paths = [Path(p) for p in paths]
        self._jobs = max(1, int(jobs))
        #: ``bidsmgr.qc.config.QcConfig`` (None: the defaults).
        self._config = config
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        from ..qc import run as R

        try:
            rows = R.run(self._root, self._paths, jobs=self._jobs, config=self._config,
                         progress=lambda d, t, n: self.progress.emit(int(d), int(t), str(n)),
                         cancel=lambda: self._stop)
        except Exception as exc:  # noqa: BLE001 - reported to the dialog
            log.debug("quality check failed", exc_info=True)
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
            return
        self.finished_with_result.emit(rows)


__all__ = ["QualityWorker"]

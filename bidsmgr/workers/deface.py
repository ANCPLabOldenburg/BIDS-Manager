"""Run defacing, and undoing it, off the GUI thread.

Defacing a single image takes a couple of seconds. Forty of them takes two
minutes, and doing that on the GUI thread gives a window that stops repainting,
stops answering the mouse and gets a spinning cursor from the OS, which is
indistinguishable from a crash. The user's only evidence that anything is
happening would be that the application appears to have died.

So the same shape every other long job here uses: a ``QThread`` that owns the
work, a ``progress`` signal per file, and a cooperative cancel flag checked
between files rather than an attempt to kill the process mid-write.

Stopping rolls the whole operation back, which is the engine's contract, not
this class's: a half-defaced dataset looks finished and is the hardest state to
recover from.
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path
from typing import Iterable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ..util.cancel import OperationCancelled

log = logging.getLogger(__name__)


class _Base(QThread):
    """Shared plumbing: progress, cancel, and never raising into Qt."""

    #: ``(done, total, relative_path)``. ``relative_path`` is "" at the end.
    progress = pyqtSignal(int, int, str)
    #: The outcome object the engine returned.
    finished_with_result = pyqtSignal(object)
    #: An exception that escaped the engine, as a formatted traceback.
    failed = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._stop = threading.Event()

    def cancel(self) -> None:
        """Ask the run to stop at the next file boundary."""
        self._stop.set()

    def _cancel_check(self) -> None:
        if self._stop.is_set():
            raise OperationCancelled("defacing stopped by the user")

    def _emit_progress(self, done: int, total: int, rel: str) -> None:
        self.progress.emit(int(done), int(total), str(rel))

    def _run_guarded(self, fn) -> None:
        try:
            self.finished_with_result.emit(fn())
        except Exception:  # noqa: BLE001 - a raise here would abort Qt
            log.exception("deface worker failed")
            self.failed.emit(traceback.format_exc())


class DefaceWorker(_Base):
    """Remove faces from a dataset, off the GUI thread."""

    def __init__(
        self,
        root: Path,
        *,
        selection=None,
        engine_id: str,
        only: Optional[Iterable[str]] = None,
        keep_original_in_sourcedata: bool = False,
        in_place=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._selection = selection
        self._engine_id = engine_id
        self._only = list(only) if only is not None else None
        self._keep = keep_original_in_sourcedata
        self._in_place = in_place

    def run(self) -> None:  # noqa: D401 - QThread entry point
        # Local import: the engine pulls in the deface package, and workers are
        # constructed at GUI start-up where that cost is not wanted.
        from ..deface.apply import deface_dataset

        self._run_guarded(lambda: deface_dataset(
            self._root,
            selection=self._selection,
            engine_id=self._engine_id,
            only=self._only,
            keep_original_in_sourcedata=self._keep,
            in_place=self._in_place,
            progress=self._emit_progress,
            cancel_check=self._cancel_check,
        ))


class DefaceRevertWorker(_Base):
    """Put faces back from the kept originals, off the GUI thread."""

    def __init__(
        self,
        root: Path,
        *,
        items=None,
        only: Optional[Iterable[str]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._root = Path(root)
        self._items = list(items) if items is not None else None
        self._only = list(only) if only is not None else None

    def run(self) -> None:  # noqa: D401 - QThread entry point
        from ..deface.revert import revert_dataset

        self._run_guarded(lambda: revert_dataset(
            self._root,
            items=self._items,
            only=self._only,
            progress=self._emit_progress,
            cancel_check=self._cancel_check,
        ))


__all__ = ["DefaceRevertWorker", "DefaceWorker"]

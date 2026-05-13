"""``QThread`` bridges for partial revalidation.

Siblings of :class:`bidsmgr.workers.ReportWorker` (dataset-wide). These
workers run :func:`bidsmgr.editor.validate_file` and
:func:`bidsmgr.editor.validate_folder` off the GUI thread so the
toolbar's "Validate file" / "Validate folder" buttons don't block
on large folders.

Both emit a list of fresh :class:`FileVerdict` records back to the
GUI; the editor panel merges them into its in-memory
:class:`ValidationReport` and refreshes the tree badge + Validation
pane sections for the affected files only.
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


class FileReportWorker(QThread):
    """Validate a single file. Emits a 1-element list for symmetry
    with :class:`FolderReportWorker` (downstream merge logic is the
    same)."""

    progress = pyqtSignal(str)
    # (verdicts, bids_root, target_path). ``verdicts`` is always
    # length 1 for this worker but kept as a list so the host can
    # share the same merge code with :class:`FolderReportWorker`.
    finished_with_verdicts = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        bids_root: Path,
        file_path: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._bids_root = Path(bids_root)
        self._file_path = Path(file_path)

    def run(self) -> None:
        from ..editor.validator import validate_file

        handler = _LogToSignal(self.progress.emit)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger("bidsmgr.editor.validator")
        logger.addHandler(handler)
        prev = logger.level
        logger.setLevel(logging.INFO)

        try:
            self.progress.emit(
                f"Validating {self._file_path.relative_to(self._bids_root)}"
            )
            verdict = validate_file(self._bids_root, self._file_path)
            self.progress.emit(
                f"Validation done — severity {verdict.severity.value}"
            )
            self.finished_with_verdicts.emit(
                [verdict], self._bids_root, self._file_path,
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev)


class FolderReportWorker(QThread):
    """Validate every file under a folder."""

    progress = pyqtSignal(str)
    finished_with_verdicts = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        bids_root: Path,
        folder_path: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._bids_root = Path(bids_root)
        self._folder_path = Path(folder_path)

    def run(self) -> None:
        from ..editor.validator import validate_folder

        handler = _LogToSignal(self.progress.emit)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger("bidsmgr.editor.validator")
        logger.addHandler(handler)
        prev = logger.level
        logger.setLevel(logging.INFO)

        try:
            self.progress.emit(
                f"Validating folder "
                f"{self._folder_path.relative_to(self._bids_root)}"
            )
            verdicts = validate_folder(self._bids_root, self._folder_path)
            self.progress.emit(
                f"Folder validation done — {len(verdicts)} files checked"
            )
            self.finished_with_verdicts.emit(
                verdicts, self._bids_root, self._folder_path,
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev)


__all__ = ["FileReportWorker", "FolderReportWorker"]

"""``QThread`` bridge for ``bidsmgr-convert``.

Runs :func:`bidsmgr.cli.convert.run_convert` on a background thread.
Same signal vocabulary as :class:`bidsmgr.workers.ScanWorker` so views
can reuse their plumbing:

* ``progress(str)`` — log messages forwarded from
  ``bidsmgr.cli.convert``'s logger for the run duration.
* ``finished_with_result(returncode, bids_parent)`` — emitted on
  success or partial failure; non-zero returncode means some subjects
  failed (the per-subject error logs live in
  ``<bids_root>/.bidsmgr/errors/``).
* ``failed(str)`` — emitted on unhandled exceptions; payload is the
  traceback text.

The worker writes the live inventory DataFrame back to its source TSV
before invoking ``run_convert`` so user edits made in the GUI (entity
changes, include toggles) are picked up. The TSV's
``files_by_uid.json.gz`` sidecar (written by the scan) stays valid as
long as series UIDs don't change.
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
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


class ConvertWorker(QThread):
    """Run :func:`bidsmgr.cli.convert.run_convert` on a worker thread.

    Parameters
    ----------
    df
        The current inventory DataFrame (typically
        ``InventoryTableModel.dataframe()``). Written back to ``tsv``
        before conversion so GUI-only edits are persisted.
    tsv
        Path the scan wrote the inventory to. The sidecar
        ``<tsv>.files_by_uid.json.gz`` must already exist alongside it
        (the scan writes both).
    bids_parent
        Destination parent dir; ``run_convert`` creates one
        ``<bids_parent>/<dataset>/`` per dataset slug.
    n_jobs, overwrite, dry_run, recording_meta
        Pass-through to the CLI verb's keyword args.
    """

    progress = pyqtSignal(str)
    finished_with_result = pyqtSignal(int, object)
    failed = pyqtSignal(str)
    # Emitted when the run stopped because the user clicked Stop. Subjects
    # committed before the stop remain on disk; the in-flight subject (if
    # any) is wiped from staging and not committed.
    stopped = pyqtSignal()

    def __init__(
        self,
        df: pd.DataFrame,
        tsv: Path,
        bids_parent: Path,
        *,
        n_jobs: int = 1,
        overwrite: bool = False,
        on_existing: Optional[str] = None,
        dry_run: bool = False,
        recording_meta: Optional[Path] = None,
        raw_root: Optional[Path] = None,
        skip_residuals: bool = True,
        preserve_curation: bool = True,
        force_edf: bool = False,
        deface: bool = False,
        deface_engine: str = "allineate",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._df = df
        self._tsv = Path(tsv)
        self._bids_parent = Path(bids_parent)
        self._n_jobs = n_jobs
        self._overwrite = overwrite
        self._on_existing = on_existing
        self._dry_run = dry_run
        self._recording_meta = Path(recording_meta) if recording_meta is not None else None
        self._raw_root = Path(raw_root) if raw_root is not None else None
        self._skip_residuals = skip_residuals
        self._preserve_curation = preserve_curation
        self._force_edf = force_edf
        self._deface = deface
        self._deface_engine = deface_engine
        # Cooperative stop flag, polled by ``run_convert`` between subjects
        # and between per-subject tasks.
        self._stop = threading.Event()

    def request_stop(self) -> None:
        """Ask the in-flight conversion to stop at the next subject / task
        boundary. Thread-safe. Already-committed subjects stay; the
        in-flight subject is not committed."""
        self._stop.set()

    def run(self) -> None:
        from ..cli.convert import run_convert  # see scan worker comment

        handler = _LogToSignal(self.progress.emit)
        handler.setLevel(logging.INFO)
        convert_logger = logging.getLogger("bidsmgr.cli.convert")
        convert_logger.addHandler(handler)
        prev_level = convert_logger.level
        convert_logger.setLevel(logging.INFO)

        try:
            self.progress.emit(f"Writing edited inventory back to {self._tsv}")
            # ``run_convert`` runs an in-memory rebuild itself, but
            # persisting first means a later CLI run sees the same
            # state the GUI converted from.
            self._df.to_csv(self._tsv, sep="\t", index=False)

            self.progress.emit(
                f"Converting → {self._bids_parent} (n_jobs={self._n_jobs})"
            )
            rc = run_convert(
                self._tsv,
                self._bids_parent,
                n_jobs=self._n_jobs,
                overwrite=self._overwrite,
                on_existing=self._on_existing,
                dry_run=self._dry_run,
                recording_meta=self._recording_meta,
                raw_root=self._raw_root,
                skip_residuals=self._skip_residuals,
                preserve_curation=self._preserve_curation,
                force_edf=self._force_edf,
                deface=self._deface,
                deface_engine=self._deface_engine,
                cancel_check=self._stop.is_set,
            )
            if rc == 130 or self._stop.is_set():
                # 130 = run_convert's "stopped before all subjects" code.
                self.progress.emit("Conversion stopped by user")
                self.stopped.emit()
                return
            verdict = "completed" if rc == 0 else f"completed with {rc} subject failure(s)"
            self.progress.emit(f"Conversion {verdict}")
            self.finished_with_result.emit(rc, self._bids_parent)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            convert_logger.removeHandler(handler)
            convert_logger.setLevel(prev_level)


__all__ = ["ConvertWorker"]

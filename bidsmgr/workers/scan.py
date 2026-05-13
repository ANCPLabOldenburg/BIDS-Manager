"""``QThread`` bridge for ``bidsmgr-scan``.

The CLI verb :func:`bidsmgr.cli.scan.run_scan` is a synchronous function
that walks DICOM and EEG/MEG trees, classifies, and writes a unified
inventory TSV. It can take tens of seconds on a real dataset; running
it on the GUI thread would freeze the window. :class:`ScanWorker` is a
``QThread`` that runs it in the background and emits signals.

Architecture note (architecture.md §12): workers import core modules
(``bidsmgr.cli.scan``) but never widgets. The view subscribes to the
worker's signals and updates its model on the main thread.

Signals
-------
* ``progress(str)``   — free-form status messages forwarded via a
  ``logging.Handler`` attached to ``bidsmgr.cli.scan``'s logger for the
  duration of the run. Lets the bottom dock's Log tab stream the same
  text the CLI prints to stderr.
* ``finished(object, object)`` — emitted on success with
  ``(DataFrame, output_tsv_Path)``. Declared as ``object`` because Qt's
  signal type system doesn't natively know pandas types.
* ``failed(str)``     — emitted on any uncaught exception; payload is
  the ``traceback.format_exc()`` text so the GUI can show a dialog.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal


class _LogToSignal(logging.Handler):
    """Forward ``logging`` records as ``progress`` signals.

    Attached for the duration of one ``run_scan`` call so info / warning
    messages stream into the GUI's Log tab. Removed in ``finally`` so we
    don't leak handlers across runs.
    """

    def __init__(self, sink) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            self._sink(record.getMessage())
        except Exception:  # pragma: no cover — never let log forwarding crash
            pass


class ScanWorker(QThread):
    """Run :func:`bidsmgr.cli.scan.run_scan` on a background thread.

    Parameters mirror the CLI verb so the GUI passes through whatever
    the user picked in the toolbar / settings:

    * ``dicom_root``  — raw input folder. The scanner auto-detects MRI
      DICOMs and EEG/MEG raw recordings in the same tree.
    * ``output_tsv``  — destination unified TSV.
    * ``dataset``     — optional dataset slug stamped into every row.
    * ``line_freq``   — optional EEG/MEG power-line freq (Hz).
    * ``montage``     — optional mne built-in montage name.
    * ``n_jobs``      — parallel workers (defaults to 1 for predictable
      behaviour in tests; views pass cpu-count-derived defaults).
    * ``skip_bids_guess`` — skip the dcm2niix BidsGuess classifier.
    * ``probe_convert``   — extra per-series dcm2niix probe pass.
    """

    progress = pyqtSignal(str)
    finished_with_result = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        dicom_root: Path,
        output_tsv: Path,
        *,
        dataset: Optional[str] = None,
        line_freq: Optional[float] = None,
        montage: Optional[str] = None,
        n_jobs: int = 1,
        skip_bids_guess: bool = False,
        probe_convert: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._dicom_root = Path(dicom_root)
        self._output_tsv = Path(output_tsv)
        self._dataset = dataset
        self._line_freq = line_freq
        self._montage = montage
        self._n_jobs = n_jobs
        self._skip_bids_guess = skip_bids_guess
        self._probe_convert = probe_convert

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the scan. Called by ``QThread.start()`` on the worker
        thread; do not invoke directly.
        """
        # Import inside ``run`` so a stuck import doesn't block worker
        # construction on the main thread. Also keeps the worker module
        # cheap to import in tests that don't actually start it.
        from ..cli.scan import run_scan

        handler = _LogToSignal(self.progress.emit)
        handler.setLevel(logging.INFO)
        scan_logger = logging.getLogger("bidsmgr.cli.scan")
        scan_logger.addHandler(handler)
        # The CLI verb's logger doesn't propagate by default; ensure the
        # handler sees INFO records regardless of the root logger's level.
        prev_level = scan_logger.level
        scan_logger.setLevel(logging.INFO)

        try:
            self.progress.emit(f"Scanning {self._dicom_root}")
            df = run_scan(
                self._dicom_root,
                self._output_tsv,
                n_jobs=self._n_jobs,
                skip_bids_guess=self._skip_bids_guess,
                probe_convert=self._probe_convert,
                dataset=self._dataset,
                line_freq=self._line_freq,
                montage=self._montage,
            )
            self.progress.emit(
                f"Scan complete: {len(df)} row(s) → {self._output_tsv}"
            )
            self.finished_with_result.emit(df, self._output_tsv)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            scan_logger.removeHandler(handler)
            scan_logger.setLevel(prev_level)


__all__ = ["ScanWorker"]

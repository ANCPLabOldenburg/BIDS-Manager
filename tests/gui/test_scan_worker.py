"""Tests for ``bidsmgr.workers.scan.ScanWorker``.

The worker wraps :func:`bidsmgr.cli.scan.run_scan`, which we already
trust at the CLI level (336+ unit tests, 49 real-data). The worker's
own contract is narrow:

* Run synchronously inside a QThread so the GUI doesn't freeze.
* Emit ``finished_with_result(df, output_tsv)`` on success.
* Emit ``failed(traceback_text)`` on any exception.
* Forward log records from ``bidsmgr.cli.scan`` as ``progress`` signals
  while the run is in flight, and detach the handler when it ends.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.workers import ScanWorker


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_empty_dir_completes_with_empty_dataframe(qtbot, tmp_path: Path) -> None:
    """A folder with no DICOMs or EEG/MEG should yield an empty inventory."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out_tsv = tmp_path / "inv.tsv"

    worker = ScanWorker(raw, out_tsv, n_jobs=1)
    with qtbot.waitSignal(worker.finished_with_result, timeout=60_000) as blocker:
        worker.start()

    df, output_tsv = blocker.args
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert Path(output_tsv) == out_tsv
    # The TSV must exist after a successful run — convert relies on it.
    assert out_tsv.exists()
    worker.wait()


def test_progress_signal_emitted_for_status_messages(qtbot, tmp_path: Path) -> None:
    """The worker prepends a 'Scanning ...' progress message before run_scan."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out_tsv = tmp_path / "inv.tsv"

    received: list[str] = []
    worker = ScanWorker(raw, out_tsv, n_jobs=1)
    worker.progress.connect(received.append)

    with qtbot.waitSignal(worker.finished_with_result, timeout=60_000):
        worker.start()

    assert any("Scanning" in m for m in received)
    assert any("Scan complete" in m for m in received)
    worker.wait()


def test_logging_handler_removed_after_run(qtbot, tmp_path: Path) -> None:
    """The progress handler must not stay attached across runs."""
    raw = tmp_path / "raw"
    raw.mkdir()
    out_tsv = tmp_path / "inv.tsv"

    scan_logger = logging.getLogger("bidsmgr.cli.scan")
    handlers_before = list(scan_logger.handlers)

    worker = ScanWorker(raw, out_tsv, n_jobs=1)
    with qtbot.waitSignal(worker.finished_with_result, timeout=60_000):
        worker.start()
    worker.wait()

    handlers_after = list(scan_logger.handlers)
    assert handlers_after == handlers_before, (
        "ScanWorker leaked a logging handler onto bidsmgr.cli.scan"
    )


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_nonexistent_input_raises_failed_signal(qtbot, tmp_path: Path) -> None:
    """A missing input folder bubbles up via the ``failed`` signal."""
    raw = tmp_path / "does_not_exist"
    out_tsv = tmp_path / "inv.tsv"

    worker = ScanWorker(raw, out_tsv, n_jobs=1)
    with qtbot.waitSignal(worker.failed, timeout=30_000) as blocker:
        worker.start()

    tb = blocker.args[0]
    assert isinstance(tb, str)
    # The traceback should at least mention the offending path or
    # something equivalent. Don't over-constrain — the underlying call
    # may evolve. Just confirm we got a non-empty traceback.
    assert tb.strip(), "failed signal carried an empty traceback"
    worker.wait()

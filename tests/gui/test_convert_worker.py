"""Tests for ``bidsmgr.workers.convert.ConvertWorker``.

The worker wraps :func:`bidsmgr.cli.convert.run_convert`, which is
already exercised by the CLI integration tests + real-data tests.
We're checking the worker's own contract:

* writes the DataFrame back to the TSV before invoking convert,
* emits ``finished_with_result(rc, bids_parent)`` on success,
* emits ``failed(tb)`` on uncaught exceptions,
* detaches its log handler at the end of every run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.workers import ConvertWorker


pytestmark = pytest.mark.gui


def _empty_inventory() -> pd.DataFrame:
    """An inventory DataFrame with the required columns but zero rows.

    ``run_convert`` accepts an empty inventory (no rows → no work)
    provided the ``dataset`` column exists.
    """
    return pd.DataFrame(columns=[
        "BIDS_name", "session", "include", "modality", "modality_bids",
        "sequence", "series_uid", "rep", "acq_time", "image_type",
        "n_files", "GivenName", "FamilyName", "PatientID", "PatientSex",
        "PatientAge", "StudyDescription",
        "proposed_datatype", "proposed_basename", "Proposed BIDS name",
        "bids_guess_classifier", "bids_guess_datatype", "bids_guess_suffix",
        "bids_guess_entities", "bids_guess_confidence", "bids_guess_skip",
        "proposed_issues", "repetition_type",
        "entities", "dataset",
        "probe_n_files", "probe_n_nifti", "probe_n_volumes", "probe_extensions",
        "study_instance_uid", "study_date", "study_time",
        "task", "run", "format", "source_file", "n_channels", "sfreq",
        "duration_sec", "n_times", "recording_time", "has_positions",
        "line_freq", "montage", "subject", "source_folder",
    ])


def test_empty_inventory_returns_zero(qtbot, tmp_path: Path) -> None:
    tsv = tmp_path / "inv.tsv"
    df = _empty_inventory()
    df.to_csv(tsv, sep="\t", index=False)

    bids_parent = tmp_path / "out"

    worker = ConvertWorker(df, tsv, bids_parent, n_jobs=1)
    with qtbot.waitSignal(worker.finished_with_result, timeout=30_000) as blocker:
        worker.start()
    rc, returned_parent = blocker.args
    assert rc == 0
    assert Path(returned_parent) == bids_parent
    worker.wait()


def test_missing_dataset_column_raises_failed(qtbot, tmp_path: Path) -> None:
    """The CLI explicitly raises if the inventory lacks ``dataset``."""
    tsv = tmp_path / "inv.tsv"
    df = pd.DataFrame([{"BIDS_name": "sub-001", "include": 1}])
    df.to_csv(tsv, sep="\t", index=False)

    worker = ConvertWorker(df, tsv, tmp_path / "out", n_jobs=1)
    with qtbot.waitSignal(worker.failed, timeout=30_000) as blocker:
        worker.start()
    tb = blocker.args[0]
    assert "dataset" in tb
    worker.wait()


def test_progress_signal_emitted(qtbot, tmp_path: Path) -> None:
    tsv = tmp_path / "inv.tsv"
    df = _empty_inventory()
    df.to_csv(tsv, sep="\t", index=False)
    bids_parent = tmp_path / "out"

    received: list[str] = []
    worker = ConvertWorker(df, tsv, bids_parent, n_jobs=1)
    worker.progress.connect(received.append)
    with qtbot.waitSignal(worker.finished_with_result, timeout=30_000):
        worker.start()
    assert any("Writing edited inventory" in m for m in received)
    assert any("Converting" in m for m in received)
    worker.wait()


def test_log_handler_removed_after_run(qtbot, tmp_path: Path) -> None:
    convert_logger = logging.getLogger("bidsmgr.cli.convert")
    handlers_before = list(convert_logger.handlers)

    tsv = tmp_path / "inv.tsv"
    df = _empty_inventory()
    df.to_csv(tsv, sep="\t", index=False)
    worker = ConvertWorker(df, tsv, tmp_path / "out", n_jobs=1)
    with qtbot.waitSignal(worker.finished_with_result, timeout=30_000):
        worker.start()
    worker.wait()

    assert list(convert_logger.handlers) == handlers_before, (
        "ConvertWorker leaked a logging handler"
    )


def test_dataframe_written_back_to_tsv(qtbot, tmp_path: Path) -> None:
    """Edits in the model must end up on disk before the verb reads it."""
    tsv = tmp_path / "inv.tsv"
    df = _empty_inventory()
    # Place a tombstone row that's distinct from any prior write.
    df = pd.concat([df, pd.DataFrame([{
        c: "" for c in df.columns
    }])], ignore_index=True)
    df.at[0, "BIDS_name"] = "sub-write-back-marker"
    df.at[0, "include"] = "0"
    df.at[0, "dataset"] = "tombstone"
    df.to_csv(tsv, sep="\t", index=False)

    # Mutate the in-memory df to a different marker before launching.
    df.at[0, "BIDS_name"] = "sub-edited"
    worker = ConvertWorker(df, tsv, tmp_path / "out", n_jobs=1)
    with qtbot.waitSignal(worker.finished_with_result, timeout=30_000):
        worker.start()
    worker.wait()

    persisted = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    assert persisted.at[0, "BIDS_name"] == "sub-edited"

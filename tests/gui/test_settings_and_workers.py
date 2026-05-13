"""Tests for AppSettings, SettingsDialog, MetadataWorker, ValidateWorker.

Plus targeted asserts on the new inspector + ConverterPanel behaviours:

* MEG/EEG source-file resolver picks up ``raw_root`` / ``tsv.parent``.
* Inspector exposes ``sequence`` + ``dataset`` columns, hides ``backend``
  + other extras by default.
* Column show/hide round-trips through QSettings.
* Dataset slug input + persistence.

QSettings is sandboxed per-test via an isolated org/app name in
``QCoreApplication``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from PyQt6.QtCore import QCoreApplication, QSettings

from bidsmgr.gui.app_settings import KEYS, AppSettings
from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.gui.models import COLUMNS, MANDATORY_COLUMN_KEYS
from bidsmgr.gui.settings_dialog import SettingsDialog
from bidsmgr.workers import MetadataWorker, ValidateWorker


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# QSettings sandbox — every test gets its own org so values don't leak.
# ---------------------------------------------------------------------------


# ``isolated_settings`` fixture is provided by ``tests/gui/conftest.py``
# so every GUI test file can sandbox its QSettings access without
# duplicating the setup logic.


# ---------------------------------------------------------------------------
# AppSettings
# ---------------------------------------------------------------------------


def test_app_settings_load_defaults(isolated_settings) -> None:
    s = AppSettings.load()
    assert s.theme == "dark"
    assert s.scan_n_jobs == 1
    assert s.scan_probe_convert is False
    assert s.post_run_metadata is True


def test_app_settings_save_and_reload(isolated_settings) -> None:
    s = AppSettings.load()
    s.theme = "light"
    s.scan_n_jobs = 8
    s.scan_probe_convert = True
    s.scan_montage = "standard_1005"
    s.post_run_validate = False
    s.save()
    reloaded = AppSettings.load()
    assert reloaded.theme == "light"
    assert reloaded.scan_n_jobs == 8
    assert reloaded.scan_probe_convert is True
    assert reloaded.scan_montage == "standard_1005"
    assert reloaded.post_run_validate is False


def test_remember_helpers_persist_individually(isolated_settings, tmp_path) -> None:
    AppSettings.remember_raw_root(tmp_path / "a")
    AppSettings.remember_bids_parent(tmp_path / "b")
    AppSettings.remember_dataset_slug("study42")
    AppSettings.remember_theme("light")

    s = AppSettings.load()
    assert s.raw_root == str(tmp_path / "a")
    assert s.bids_parent == str(tmp_path / "b")
    assert s.dataset_slug == "study42"
    assert s.theme == "light"


# ---------------------------------------------------------------------------
# SettingsDialog
# ---------------------------------------------------------------------------


def test_settings_dialog_save_writes_through(isolated_settings, qtbot) -> None:
    s = AppSettings.load()
    dlg = SettingsDialog(s)
    qtbot.addWidget(dlg)
    # Mutate a few controls then trigger _on_save.
    dlg._theme_combo.setCurrentText("light")
    dlg._scan_jobs.setValue(4)
    dlg._scan_probe.setChecked(True)
    dlg._post_run_metadata.setChecked(False)
    dlg._on_save()

    reloaded = AppSettings.load()
    assert reloaded.theme == "light"
    assert reloaded.scan_n_jobs == 4
    assert reloaded.scan_probe_convert is True
    assert reloaded.post_run_metadata is False


# ---------------------------------------------------------------------------
# Inspector columns
# ---------------------------------------------------------------------------


def test_inspector_default_columns_include_sequence_and_dataset() -> None:
    keys = {c.key for c in COLUMNS}
    assert "sequence" in keys
    assert "dataset" in keys


def test_backend_column_is_hidden_by_default() -> None:
    backend = next(c for c in COLUMNS if c.key == "backend")
    assert backend.default_visible is False


def test_mandatory_columns_set_is_complete() -> None:
    assert MANDATORY_COLUMN_KEYS == {"include", "status", "id"}


def test_column_visibility_persists_to_qsettings(isolated_settings, qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    # Toggle ``backend`` on (hidden by default).
    panel.set_column_visible("backend", True)
    # Reload — a new panel should see the persisted state.
    panel2 = ConverterPanel()
    qtbot.addWidget(panel2)
    assert panel2._column_visible["backend"] is True


def test_mandatory_columns_cannot_be_hidden(isolated_settings, qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.set_column_visible("id", False)  # should be a no-op
    assert panel._column_visible["id"] is True


# ---------------------------------------------------------------------------
# Dataset slug input
# ---------------------------------------------------------------------------


def test_dataset_column_is_editable_in_inspector() -> None:
    """The ``dataset`` column partitions the convert queue per-row, so it
    must be user-editable from the table without needing a global toolbar
    input."""
    dataset = next(c for c in COLUMNS if c.key == "dataset")
    assert dataset.editable is True


def test_tsv_filename_input_seeded_from_settings(isolated_settings, qtbot) -> None:
    AppSettings.remember_tsv_filename("custom.tsv")
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    assert panel._tsv_filename_edit.text() == "custom.tsv"


def test_tsv_filename_persists_on_edit(isolated_settings, qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel._tsv_filename_edit.setText("study.tsv")
    panel._on_tsv_filename_edited()
    assert AppSettings.load().scan_tsv_filename == "study.tsv"


def test_tsv_filename_appends_extension_if_missing(isolated_settings, qtbot) -> None:
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel._tsv_filename_edit.setText("study")
    panel._on_tsv_filename_edited()
    assert panel._tsv_filename_edit.text() == "study.tsv"
    assert AppSettings.load().scan_tsv_filename == "study.tsv"


def test_scan_click_writes_tsv_under_bids_parent(isolated_settings, qtbot, tmp_path) -> None:
    """The scan flow now places the TSV inside the BIDS output folder
    instead of next to the raw data. Verified by exercising
    ``_on_scan_clicked`` after pre-seeding raw_root + bids_parent +
    the filename input.
    """
    panel = ConverterPanel()
    qtbot.addWidget(panel)
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "bids_out"
    out.mkdir()
    panel._raw_root = raw
    panel._bids_parent = out
    panel._tsv_filename_edit.setText("inv.tsv")

    with qtbot.waitSignal(panel.scan_finished, timeout=60_000):
        panel._on_scan_clicked()
    # Worker writes TSV to <bids_parent>/<filename>.
    assert (out / "inv.tsv").exists()
    # The old behaviour wrote into the raw folder — make sure that's gone.
    assert not (raw / ".bidsmgr_scan.tsv").exists()


def test_default_dataset_slug_helper() -> None:
    assert ConverterPanel._default_dataset_slug(Path("/x/My Study v2")) == "my-study-v2"
    assert ConverterPanel._default_dataset_slug(Path("/tmp/abc__def")) == "abc__def"


# ---------------------------------------------------------------------------
# MEG/EEG resolver fix (CLI-level test)
# ---------------------------------------------------------------------------


def test_run_convert_finds_relative_source_via_raw_root(tmp_path: Path) -> None:
    """The convert CLI must resolve ``source_file`` against ``raw_root``."""
    from bidsmgr.cli.convert import run_convert

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    src_rel = "sub_x/rec.fif"
    src_abs = raw_root / src_rel
    src_abs.parent.mkdir(parents=True)
    src_abs.write_bytes(b"")  # not a real .fif, will fail at backend stage

    # Build a 1-row inventory carrying a relative source path.
    tsv = tmp_path / "inv.tsv"
    df = pd.DataFrame([{
        "BIDS_name": "sub-001",
        "session": "",
        "include": "1",
        "modality": "meg",
        "modality_bids": "meg",
        "proposed_datatype": "meg",
        "proposed_basename": "sub-001_task-rest_meg",
        "Proposed BIDS name": "sub-001_task-rest_meg",
        "bids_guess_classifier": "mne",
        "bids_guess_datatype": "meg",
        "bids_guess_suffix": "meg",
        "bids_guess_confidence": "0.99",
        "bids_guess_skip": "false",
        "proposed_issues": "",
        "entities": json.dumps({"subject": "001", "task": "rest"}, sort_keys=True),
        "task": "rest",
        "run": "",
        "format": "FIF",
        "source_file": src_rel,
        "dataset": "study",
    }])
    df.to_csv(tsv, sep="\t", index=False)

    # Without raw_root + with CWD elsewhere, the resolver used to miss.
    # Move CWD to somewhere unrelated so the test cleanly exercises the fix.
    cwd_before = Path.cwd()
    os.chdir(tmp_path / "raw")  # ensure cwd path resolution would also miss
    os.chdir(tmp_path)
    try:
        # The conversion will fail at the mne-bids backend stage (the
        # ``rec.fif`` is empty), but the resolver step must succeed —
        # i.e. the failure message no longer says ``source not found``.
        rc = run_convert(tsv, tmp_path / "out", raw_root=raw_root, n_jobs=1)
        # rc is 0 or 1 depending on whether the backend handled the
        # empty .fif gracefully. We don't care about the exact rc; we
        # care that the error log (if any) doesn't say "source not found".
        errors_dir = tmp_path / "out" / "study" / ".bidsmgr" / "errors"
        if errors_dir.exists():
            for log_file in errors_dir.glob("*.json"):
                payload = json.loads(log_file.read_text())
                for r in payload.get("results", []):
                    assert "source not found" not in (r.get("error") or "")
    finally:
        os.chdir(cwd_before)


# ---------------------------------------------------------------------------
# MetadataWorker / ValidateWorker contract
# ---------------------------------------------------------------------------


def test_metadata_worker_emits_finished_on_empty_target(qtbot, tmp_path: Path) -> None:
    """No BIDS roots → rc 0 + no errors."""
    target = tmp_path / "empty"
    target.mkdir()
    worker = MetadataWorker(target)
    with qtbot.waitSignal(worker.finished_with_result, timeout=30_000) as blocker:
        worker.start()
    rc, returned = blocker.args
    assert rc == 0
    assert Path(returned) == target
    worker.wait()


def test_validate_worker_emits_failed_for_missing_target(qtbot, tmp_path: Path) -> None:
    """Non-existent target → run_validate_cli returns rc 2; worker emits
    finished_with_result (the CLI returns the rc, doesn't raise)."""
    worker = ValidateWorker(tmp_path / "does_not_exist")
    with qtbot.waitSignal(worker.finished_with_result, timeout=30_000) as blocker:
        worker.start()
    rc, _ = blocker.args
    assert rc != 0
    worker.wait()

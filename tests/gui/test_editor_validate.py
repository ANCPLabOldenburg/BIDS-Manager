"""Tests for the Editor's dataset-validate flow (M6 Step 3).

Covers:

* :class:`BidsTreePane.set_badges` stamps the right ``BADGE_ROLE``
  values and rolls folder badges up to the worst descendant severity.
* :class:`BidsTreePane.clear_badges` wipes them.
* :class:`ReportWorker` returns an in-memory
  :class:`bidsmgr.editor.types.ValidationReport`.
* :class:`EditorPanel.start_dataset_validation` runs the worker,
  stamps the tree, updates the status chips, and re-enables the
  Validate button on completion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bidsmgr.editor.types import (
    FileVerdict,
    Severity,
    ValidationReport,
)
from bidsmgr.gui.delegates.bids_tree import BADGE_ROLE
from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.widgets.bids_tree_pane import BidsTreePane
from bidsmgr.workers import ReportWorker


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bids_root(tmp_path: Path) -> Path:
    """Minimal BIDS-shaped tree with a sidecar that's intentionally
    missing required fields, so :func:`editor.validate` produces real
    issues without needing a 30-row dataset."""
    root = tmp_path / "Studyname"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-01_T1w.nii.gz").write_bytes(b"")
    # Empty JSON triggers "required field missing" findings from the
    # schema-driven validator (no Magnetic*Field, no etc).
    (anat / "sub-01_ses-01_T1w.json").write_text("{}")
    (root / "dataset_description.json").write_text(
        '{"Name": "Test", "BIDSVersion": "1.10.0"}'
    )
    (root / "participants.tsv").write_text("participant_id\nsub-01\n")
    return root


# ---------------------------------------------------------------------------
# BidsTreePane.set_badges / clear_badges
# ---------------------------------------------------------------------------


def _find_item(tree, name: str):
    matches = []

    def visit(item) -> None:
        if item.text(0) == name:
            matches.append(item)
        for i in range(item.childCount()):
            visit(item.child(i))

    for i in range(tree.topLevelItemCount()):
        visit(tree.topLevelItem(i))
    assert matches, f"item {name!r} not found"
    return matches[0]


def test_set_badges_stamps_leaves_and_rolls_folders_up(
    qapp, bids_root: Path,
) -> None:
    pane = BidsTreePane()
    pane.set_root(bids_root)
    qapp.processEvents()

    t1w_nii = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
    t1w_json = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"

    pane.set_badges({
        t1w_nii: "ok",
        t1w_json: "err",
    })

    # Leaf badges set verbatim.
    nii_item = _find_item(pane._tree, "sub-01_ses-01_T1w.nii.gz")
    json_item = _find_item(pane._tree, "sub-01_ses-01_T1w.json")
    assert nii_item.data(0, BADGE_ROLE) == "ok"
    assert json_item.data(0, BADGE_ROLE) == "err"

    # Folder rollups pick the worst severity of any descendant.
    anat_item = _find_item(pane._tree, "anat")
    ses_item = _find_item(pane._tree, "ses-01")
    sub_item = _find_item(pane._tree, "sub-01")
    root_item = _find_item(pane._tree, bids_root.name)
    for item in (anat_item, ses_item, sub_item, root_item):
        assert item.data(0, BADGE_ROLE) == "err"


def test_set_badges_warn_does_not_clobber_err(qapp, bids_root: Path) -> None:
    pane = BidsTreePane()
    pane.set_root(bids_root)
    qapp.processEvents()

    pane.set_badges({
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz": "warn",
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json": "err",
        bids_root / "participants.tsv": "ok",
    })
    # The anat dir rolls up to err (worst), not warn.
    anat = _find_item(pane._tree, "anat")
    assert anat.data(0, BADGE_ROLE) == "err"
    # The root rolls up to err too (because of the anat subtree).
    root_item = _find_item(pane._tree, bids_root.name)
    assert root_item.data(0, BADGE_ROLE) == "err"


def test_clear_badges_removes_everything(qapp, bids_root: Path) -> None:
    pane = BidsTreePane()
    pane.set_root(bids_root)
    qapp.processEvents()

    pane.set_badges({
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json": "warn",
    })
    pane.clear_badges()

    def visit(item) -> None:
        assert item.data(0, BADGE_ROLE) is None
        for i in range(item.childCount()):
            visit(item.child(i))

    for i in range(pane._tree.topLevelItemCount()):
        visit(pane._tree.topLevelItem(i))


# ---------------------------------------------------------------------------
# ReportWorker
# ---------------------------------------------------------------------------


def test_report_worker_returns_validation_report(qapp, qtbot, bids_root: Path) -> None:
    worker = ReportWorker(bids_root, strict=False)

    with qtbot.waitSignal(worker.finished_with_report, timeout=5000) as sig:
        worker.start()

    report, root = sig.args
    assert isinstance(report, ValidationReport)
    assert root == bids_root
    # Counts shape is what the GUI consumes.
    assert set(report.counts) >= {"ok", "warn", "err"}
    # At least one FileVerdict was emitted (we provided at least the
    # T1w sidecar + dataset_description.json).
    assert any(isinstance(fv, FileVerdict) for fv in report.files)


# ---------------------------------------------------------------------------
# EditorPanel.start_dataset_validation
# ---------------------------------------------------------------------------


def test_validate_dataset_button_disabled_without_root(qapp) -> None:
    panel = EditorPanel()
    assert not panel._validate_dataset_btn.isEnabled()


def test_validate_dataset_button_enables_once_root_loaded(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    assert panel._validate_dataset_btn.isEnabled()


def test_start_dataset_validation_stamps_tree_and_chips(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)

    # Watch for the panel's log-message stream so we can verify the
    # progress signal is forwarded (this is what the status bar shows).
    progress: list[str] = []
    panel.log_message.connect(progress.append)

    with qtbot.waitSignal(
        panel.log_message, timeout=5000,
        check_params_cb=lambda msg: "Validation done" in msg,
    ):
        panel.start_dataset_validation()

    # ``log_message`` fires inside ``_on_report_ready`` — that's BEFORE
    # the worker's ``finished`` slot runs and re-enables the button.
    # Wait for the worker to be fully cleaned up before asserting.
    qtbot.waitUntil(lambda: panel._report_worker is None, timeout=5000)

    # Report stashed for later steps to consume.
    assert isinstance(panel.current_report(), ValidationReport)
    # Chips no longer explicitly hidden after the run (effective
    # visibility requires the panel to be shown; we only assert that
    # the panel un-hid them).
    for chip in (panel._chip_ok, panel._chip_warn, panel._chip_err):
        assert not chip.isHidden()
    # Button re-enabled.
    assert panel._validate_dataset_btn.isEnabled()
    # Tree carries at least one badge (the empty T1w.json must trip
    # at least a warning or error in the schema audit).
    tree = panel.tree_pane()._tree
    badges_seen: list[str] = []

    def visit(item) -> None:
        b = item.data(0, BADGE_ROLE)
        if b:
            badges_seen.append(b)
        for i in range(item.childCount()):
            visit(item.child(i))

    for i in range(tree.topLevelItemCount()):
        visit(tree.topLevelItem(i))
    assert badges_seen, "expected the report to stamp at least one badge"


def test_change_root_clears_old_badges_and_report(
    qapp, qtbot, isolated_settings, bids_root: Path, tmp_path: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)

    # Run a validation so we have a report + badges, then swap roots.
    with qtbot.waitSignal(
        panel.log_message, timeout=5000,
        check_params_cb=lambda msg: "Validation done" in msg,
    ):
        panel.start_dataset_validation()
    assert panel.current_report() is not None

    new_root = tmp_path / "OtherDataset"
    new_root.mkdir()
    panel._set_root(new_root, persist=False)

    assert panel.current_report() is None
    # Chips hidden again.
    for chip in (panel._chip_ok, panel._chip_warn, panel._chip_err):
        assert chip.isHidden()


def test_strict_toggle_defaults_off_and_persists(
    qapp, isolated_settings,
) -> None:
    """The Strict BIDS toggle starts OFF, and clicking it persists
    the new state to :class:`AppSettings`."""
    from bidsmgr.gui.app_settings import AppSettings
    from bidsmgr.gui.editor_panel import EditorPanel as _EP

    panel = _EP()
    assert panel._strict_btn.isCheckable()
    assert not panel._strict_btn.isChecked()
    assert not AppSettings.load().editor_strict_validate

    panel._strict_btn.setChecked(True)
    assert AppSettings.load().editor_strict_validate

    panel._strict_btn.setChecked(False)
    assert not AppSettings.load().editor_strict_validate


def test_persisted_strict_setting_restores_toggle(
    qapp, isolated_settings,
) -> None:
    from bidsmgr.gui.app_settings import AppSettings
    from bidsmgr.gui.editor_panel import EditorPanel as _EP

    AppSettings.remember_editor_strict_validate(True)
    panel = _EP()
    assert panel._strict_btn.isChecked()


def test_strict_toggle_drives_report_worker_strict_flag(
    qapp, qtbot, isolated_settings, bids_root: Path, monkeypatch,
) -> None:
    """Clicking Validate dataset with the Strict toggle ON constructs
    a :class:`ReportWorker` with ``strict=True``."""
    from bidsmgr.gui.editor_panel import EditorPanel as _EP
    from bidsmgr.workers import report as report_mod

    captured: dict = {}
    real_init = report_mod.ReportWorker.__init__

    def _spy_init(self, bids_root, *, strict=False, parent=None):
        captured["strict"] = strict
        return real_init(self, bids_root, strict=strict, parent=parent)

    monkeypatch.setattr(report_mod.ReportWorker, "__init__", _spy_init)

    panel = _EP()
    panel._set_root(bids_root, persist=False)
    panel._strict_btn.setChecked(True)

    panel.start_dataset_validation()
    # Wait for the worker to finish so teardown is clean (running
    # QThreads will SIGABRT pytest if their parent goes away mid-flight).
    qtbot.waitUntil(lambda: panel._report_worker is None, timeout=10000)
    assert captured.get("strict") is True


def test_strict_toggle_off_passes_strict_false(
    qapp, qtbot, isolated_settings, bids_root: Path, monkeypatch,
) -> None:
    """Default toggle state (off) still flows through as ``strict=False``."""
    from bidsmgr.gui.editor_panel import EditorPanel as _EP
    from bidsmgr.workers import report as report_mod

    captured: dict = {}
    real_init = report_mod.ReportWorker.__init__

    def _spy_init(self, bids_root, *, strict=False, parent=None):
        captured["strict"] = strict
        return real_init(self, bids_root, strict=strict, parent=parent)

    monkeypatch.setattr(report_mod.ReportWorker, "__init__", _spy_init)

    panel = _EP()
    panel._set_root(bids_root, persist=False)
    assert not panel._strict_btn.isChecked()
    panel.start_dataset_validation()
    qtbot.waitUntil(lambda: panel._report_worker is None, timeout=10000)
    assert captured.get("strict") is False


def test_strict_button_has_tooltip(qapp, isolated_settings) -> None:
    """Hovering the Strict toggle must surface explanatory text — a
    user-facing description of what the second pass adds."""
    from bidsmgr.gui.editor_panel import EditorPanel as _EP

    panel = _EP()
    tip = panel._strict_btn.toolTip()
    assert tip
    assert "bidsschematools" in tip
    # Should mention the official Python BIDS validator and a hint
    # about when to use it.
    assert "official" in tip.lower()


def test_validate_report_severity_values_round_trip(qapp) -> None:
    """The Severity enum's `.value` field is the string the tree pane
    expects (``"ok"``/``"warn"``/``"err"``). Guard against an accidental
    rename."""
    assert Severity.OK.value == "ok"
    assert Severity.WARN.value == "warn"
    assert Severity.ERR.value == "err"

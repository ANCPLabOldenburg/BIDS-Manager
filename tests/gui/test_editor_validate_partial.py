"""Tests for the Editor's Validate-file / Validate-folder toolbar buttons.

Covers:

* :func:`validate_file` / :func:`validate_folder` pure helpers — layer
  1 only, no dataset-wide pass.
* :class:`FileReportWorker` / :class:`FolderReportWorker` emit
  ``finished_with_verdicts`` with the right shape.
* :class:`EditorPanel` enables the buttons based on tree selection,
  merges the partial results into the in-memory report, and refreshes
  tree badges + Validation pane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor.types import FileVerdict, Severity, ValidationReport
from bidsmgr.editor.validator import validate_file, validate_folder
from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.widgets.bids_tree_pane import PATH_ROLE
from bidsmgr.workers import FileReportWorker, FolderReportWorker


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bids_root(tmp_path: Path) -> Path:
    """Two-subject dataset with one TODO-laden top-level JSON."""
    root = tmp_path / "DS"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-01_T1w.nii.gz").write_bytes(b"")
    (anat / "sub-01_ses-01_T1w.json").write_text(
        json.dumps({"Manufacturer": "Siemens"})
    )
    (anat / "sub-01_ses-01_T2w.nii.gz").write_bytes(b"")
    (anat / "sub-01_ses-01_T2w.json").write_text(
        json.dumps({"Manufacturer": "TODO"})
    )
    (root / "dataset_description.json").write_text(
        '{"Name": "Test", "BIDSVersion": "1.10.0"}'
    )
    (root / "participants.tsv").write_text("participant_id\nsub-01\n")
    return root


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_validate_file_returns_a_single_verdict(bids_root: Path) -> None:
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    verdict = validate_file(bids_root, json_path)
    assert isinstance(verdict, FileVerdict)
    # Relative to bids_root, so it merges cleanly with dataset reports.
    assert not verdict.path.is_absolute()
    assert verdict.datatype == "anat"
    assert verdict.suffix == "T1w"


def test_validate_file_detects_todo_in_sidecar(bids_root: Path) -> None:
    """The T2w sidecar has a TODO Manufacturer — should warn."""
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T2w.json"
    )
    verdict = validate_file(bids_root, json_path)
    assert verdict.severity in (Severity.WARN, Severity.ERR)
    assert any(i.rule_id == "bidsmgr.todo_placeholder" for i in verdict.issues)


def test_validate_folder_walks_recursively(bids_root: Path) -> None:
    folder = bids_root / "sub-01" / "ses-01" / "anat"
    verdicts = validate_folder(bids_root, folder)
    # 2 .nii.gz + 2 .json = 4 files in this folder.
    assert len(verdicts) == 4
    paths = sorted(str(v.path) for v in verdicts)
    assert any("T1w.json" in p for p in paths)
    assert any("T2w.json" in p for p in paths)


def test_validate_folder_skips_dot_dirs(bids_root: Path) -> None:
    # Drop a .bidsmgr scratch tree — must be ignored.
    junk = bids_root / "sub-01" / ".bidsmgr"
    junk.mkdir()
    (junk / "events.jsonl").write_text("noise")
    verdicts = validate_folder(bids_root, bids_root / "sub-01")
    for v in verdicts:
        assert ".bidsmgr" not in str(v.path)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def test_file_report_worker_emits_one_verdict(
    qapp, qtbot, bids_root: Path,
) -> None:
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    worker = FileReportWorker(bids_root, json_path)
    with qtbot.waitSignal(worker.finished_with_verdicts, timeout=5000) as sig:
        worker.start()
    verdicts, root, target = sig.args
    assert len(verdicts) == 1
    assert isinstance(verdicts[0], FileVerdict)
    assert target == json_path
    assert root == bids_root


def test_folder_report_worker_emits_all_verdicts(
    qapp, qtbot, bids_root: Path,
) -> None:
    folder = bids_root / "sub-01" / "ses-01" / "anat"
    worker = FolderReportWorker(bids_root, folder)
    with qtbot.waitSignal(worker.finished_with_verdicts, timeout=5000) as sig:
        worker.start()
    verdicts, _root, target = sig.args
    assert len(verdicts) == 4
    assert target == folder


# ---------------------------------------------------------------------------
# EditorPanel — toolbar enablement + end-to-end merge
# ---------------------------------------------------------------------------


def _find_tree_item(tree, path_str: str):
    def visit(item):
        if item.data(0, PATH_ROLE) == path_str:
            return item
        for i in range(item.childCount()):
            r = visit(item.child(i))
            if r is not None:
                return r
        return None
    for i in range(tree.topLevelItemCount()):
        r = visit(tree.topLevelItem(i))
        if r is not None:
            return r
    return None


def test_validate_file_button_enables_on_file_selection(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    # Buttons start disabled — nothing selected.
    assert not panel._validate_file_btn.isEnabled()
    assert not panel._validate_folder_btn.isEnabled()

    # Select a file in the tree.
    target = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    item = _find_tree_item(panel._tree_pane._tree, str(target))
    assert item is not None
    panel._tree_pane._tree.setCurrentItem(item)
    qapp.processEvents()

    # Only Validate file enables; folder stays off.
    assert panel._validate_file_btn.isEnabled()
    assert not panel._validate_folder_btn.isEnabled()


def test_validate_folder_button_enables_on_folder_selection(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)

    folder = bids_root / "sub-01" / "ses-01" / "anat"
    item = _find_tree_item(panel._tree_pane._tree, str(folder))
    assert item is not None
    panel._tree_pane._tree.setCurrentItem(item)
    qapp.processEvents()

    assert not panel._validate_file_btn.isEnabled()
    assert panel._validate_folder_btn.isEnabled()


def test_start_file_validation_merges_into_existing_report(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    # Pre-seed an in-memory report with a stale verdict for T2w.json.
    t2w_rel = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T2w.json"
    ).relative_to(bids_root)
    panel._report = ValidationReport(
        bids_root=bids_root,
        files=[FileVerdict(path=t2w_rel, severity=Severity.OK)],
    )
    # Select T2w.json (the one with the TODO).
    item = _find_tree_item(
        panel._tree_pane._tree,
        str(bids_root / t2w_rel),
    )
    panel._tree_pane._tree.setCurrentItem(item)
    qapp.processEvents()

    with qtbot.waitSignal(
        panel.log_message, timeout=5000,
        check_params_cb=lambda m: "Validation done" in m,
    ):
        panel.start_file_validation()
    qtbot.waitUntil(lambda: panel._partial_worker is None, timeout=5000)

    # The stale OK verdict was replaced by the fresh warn one.
    new_verdict = next(
        f for f in panel._report.files if f.path == t2w_rel
    )
    assert new_verdict.severity is Severity.WARN
    assert any(
        i.rule_id == "bidsmgr.todo_placeholder" for i in new_verdict.issues
    )
    # And the report's rollup severity / counts reflect the change.
    assert panel._report.severity is Severity.WARN
    assert panel._report.counts.get("warn", 0) >= 1


def test_start_folder_validation_appends_new_verdicts(
    qapp, qtbot, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    panel._report = ValidationReport(bids_root=bids_root)  # empty

    folder = bids_root / "sub-01" / "ses-01" / "anat"
    item = _find_tree_item(panel._tree_pane._tree, str(folder))
    panel._tree_pane._tree.setCurrentItem(item)
    qapp.processEvents()

    with qtbot.waitSignal(
        panel.log_message, timeout=10000,
        check_params_cb=lambda m: "Validation done" in m,
    ):
        panel.start_folder_validation()
    qtbot.waitUntil(lambda: panel._partial_worker is None, timeout=10000)

    # 4 verdicts merged into the previously-empty report.
    assert len(panel._report.files) == 4


def test_start_file_validation_noop_without_root(
    qapp, isolated_settings,
) -> None:
    panel = EditorPanel()
    # No root — must not raise.
    panel.start_file_validation()
    panel.start_folder_validation()


def test_start_file_validation_noop_without_selection(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)
    # No selection → button is disabled and the method is a noop.
    panel.start_file_validation()
    panel.start_folder_validation()

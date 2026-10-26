"""The defacing dialog, offscreen.

What is tested is what a user could get wrong by trusting the screen: that the
skipped images are visible, that unticking one keeps it, and that the option
which leaves a face inside the dataset is off unless asked for.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402

from bidsmgr.deface import run, status  # noqa: E402
from bidsmgr.deface.engines import TEMPLATE  # noqa: E402
from bidsmgr.gui.deface_dialog import DefaceDialog  # noqa: E402


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "func").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"})
    )
    for name in ("sub-01_T1w.nii.gz", "sub-01_T2w.nii.gz"):
        shutil.copyfile(TEMPLATE, root / "sub-01" / "anat" / name)
    # Something that must be refused, so the skipped list has content.
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    nib.save(
        nib.Nifti1Image(np.zeros((8, 8, 8, 20), dtype="float32"), np.eye(4)),
        str(root / "sub-01" / "func" / "sub-01_task-x_bold.nii.gz"),
    )
    return root


def test_the_dialog_lists_candidates_and_the_reason_for_each_skip(qtbot, tmp_path):
    root = _dataset(tmp_path)
    dlg = DefaceDialog(root)
    qtbot.addWidget(dlg)

    tree = dlg._preview
    tops = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    checkable = [t for t in tops if t.data(0, Qt.ItemDataRole.UserRole)]
    assert {t.text(0) for t in checkable} == {
        "sub-01/anat/sub-01_T1w.nii.gz",
        "sub-01/anat/sub-01_T2w.nii.gz",
    }

    heads = [t for t in tops if not t.data(0, Qt.ItemDataRole.UserRole)]
    assert heads, "the skipped images are not shown at all"
    head = heads[0]
    assert "(1)" in head.text(0), "the skip count is not on the header"

    # Grouped by reason: header -> reason -> the files under it.
    reasons = [head.child(i) for i in range(head.childCount())]
    assert reasons, "no reason groups"
    assert all(r.text(1) for r in reasons), "a reason group has no count"
    files = [r.child(j) for r in reasons for j in range(r.childCount())]
    assert any("bold" in f.text(0) for f in files), (
        "the 4-D image is not listed among the skips"
    )


def test_everything_starts_ticked_and_the_button_is_live(qtbot, tmp_path):
    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    if run.available():
        assert dlg._ok.isEnabled()
    assert len(dlg._checked()) == 2


def test_unticking_removes_it_from_what_would_run(qtbot, tmp_path):
    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    tree = dlg._preview
    first = next(
        tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
        if tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
    )
    first.setCheckState(0, Qt.CheckState.Unchecked)

    assert len(dlg._checked()) == 1
    assert first.text(0) not in dlg._checked()


def test_select_none_disables_the_button(qtbot, tmp_path):
    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    dlg._set_all(Qt.CheckState.Unchecked)
    assert not dlg._ok.isEnabled()
    assert "Nothing selected" in dlg._status.text()


def test_the_sourcedata_option_is_off_by_default(qtbot, tmp_path):
    """It leaves a face-bearing copy inside the dataset, so it is a choice."""
    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    assert not dlg._keep.isChecked()
    assert "still contain the face" in dlg._keep.toolTip()


def test_every_engine_is_offered_with_its_description(qtbot, tmp_path):
    from bidsmgr.deface import engines

    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    offered = [dlg._engine.itemData(i) for i in range(dlg._engine.count())]
    # DEFACE engines only. A defacing dropdown that offers a skull stripper is
    # offering to delete the skull when the user asked to blank a face.
    assert offered == engines.engine_ids(engines.KIND_DEFACE)
    assert dlg._engine.currentData() == engines.DEFAULT_ENGINE_ID
    assert dlg._engine_note.text()

    dlg._engine.setCurrentIndex(offered.index("allineate-robust"))
    assert "DIMENSIONS" in dlg._engine_note.text(), (
        "the engine that crops the neck does not warn that it does"
    )


def test_a_previously_defaced_image_says_so_rather_than_being_hidden(
    qtbot, tmp_path
):
    from bidsmgr.deface.engines import ALLINEATE

    root = _dataset(tmp_path)
    img = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    status.sidecar_for(img).write_text(json.dumps(status.record({}, ALLINEATE)))

    dlg = DefaceDialog(root)
    qtbot.addWidget(dlg)
    tree = dlg._preview
    row = next(
        tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
        if tree.topLevelItem(i).text(0).endswith("T1w.nii.gz")
    )
    assert "again" in row.text(1)
    assert row.checkState(0) == Qt.CheckState.Checked


def test_targets_narrow_the_dialog(qtbot, tmp_path):
    root = _dataset(tmp_path)
    dlg = DefaceDialog(root, [root / "sub-01" / "anat" / "sub-01_T2w.nii.gz"])
    qtbot.addWidget(dlg)
    assert dlg._checked() == ["sub-01/anat/sub-01_T2w.nii.gz"]


def test_the_scope_card_counts_both_halves(qtbot, tmp_path):
    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    text = dlg._scope.text()
    assert "2 image(s) can be defaced" in text
    assert "1 skipped" in text


# ---------------------------------------------------------------------------
# It runs off the GUI thread, with progress, and can be stopped.
#
# Defacing one image takes seconds and a whole dataset takes minutes. Run
# inline it froze the window, which the OS marks unresponsive and a user reads
# as a crash.


def test_the_work_happens_on_a_qthread_not_the_gui_thread(
    qtbot, tmp_path, monkeypatch,
):
    from PyQt6.QtCore import QThread
    from PyQt6.QtWidgets import QMessageBox

    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    if not run.available():
        pytest.skip("needs niimath")
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None),
    )

    dlg._on_apply()
    assert isinstance(dlg._worker, QThread)
    assert dlg._worker.thread() is not dlg._worker, (
        "the worker must not be running on the thread that owns it"
    )
    assert not dlg._progress.isHidden() or dlg._worker.isFinished()
    qtbot.waitUntil(lambda: dlg._worker is None, timeout=120_000)
    assert dlg._outcome is not None and dlg._outcome.ok


def test_the_controls_lock_while_it_runs_and_unlock_after(qtbot, tmp_path):
    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)

    dlg._set_running(True, 2)
    assert not dlg._preview.isEnabled()
    assert not dlg._engine.isEnabled()
    assert not dlg._keep.isEnabled()
    assert not dlg._ok.isEnabled()
    assert not dlg._stop.isHidden() and not dlg._progress.isHidden()

    dlg._set_running(False)
    assert dlg._preview.isEnabled() and dlg._engine.isEnabled()
    assert dlg._stop.isHidden() and dlg._progress.isHidden()


def test_progress_names_the_file_and_moves_the_bar(qtbot, tmp_path):
    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    dlg._set_running(True, 2)

    dlg._on_progress(0, 2, "sub-01/anat/sub-01_T1w.nii.gz")
    assert dlg._progress.value() == 0
    assert "sub-01_T1w.nii.gz" in dlg._status.text()
    assert "1 of 2" in dlg._status.text()

    dlg._on_progress(2, 2, "")
    assert dlg._progress.value() == 2


def test_a_stop_is_reported_as_stopped_not_as_a_failure(qtbot, tmp_path):
    """The dataset is rolled back either way; the wording must differ."""
    from bidsmgr.deface.apply import DefaceOutcome

    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    dlg._set_running(True, 2)

    dlg._on_done(DefaceOutcome(cancelled=True))
    assert "Stopped" in dlg._status.text()
    assert dlg.result() != DefaceDialog.DialogCode.Accepted
    assert dlg._preview.isEnabled(), "the dialog did not become usable again"


def test_closing_mid_run_stops_the_thread_first(qtbot, tmp_path, monkeypatch):
    """A running QThread destroyed with its parent aborts the process."""
    from PyQt6.QtWidgets import QMessageBox

    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    if not run.available():
        pytest.skip("needs niimath")
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
    )
    dlg._on_apply()
    assert dlg._worker is not None

    dlg.close()
    assert dlg._worker is None


# ---------------------------------------------------------------------------
# The same dialog, in skull-strip mode.


def _strip(tmp_path):
    from bidsmgr.deface import engines as e

    return DefaceDialog(_dataset(tmp_path), kind=e.KIND_STRIP)


def test_strip_mode_offers_only_strip_engines(qtbot, tmp_path):
    from bidsmgr.deface import engines as e

    dlg = _strip(tmp_path)
    qtbot.addWidget(dlg)
    offered = [dlg._engine.itemData(i) for i in range(dlg._engine.count())]
    assert offered == e.engine_ids(e.KIND_STRIP)
    assert dlg._engine.currentData() == e.DEFAULT_STRIP_ENGINE_ID


def test_strip_mode_says_the_output_is_a_derivative(qtbot, tmp_path):
    from bidsmgr.deface.derivatives import PIPELINE

    dlg = _strip(tmp_path)
    qtbot.addWidget(dlg)
    if not run.available(dlg._current_engine().id):
        pytest.skip("no strip engine available here")
    assert PIPELINE in dlg._status.text()
    assert "untouched" in dlg._status.text()


def test_strip_mode_offers_overwriting_and_does_not_default_to_it(
    qtbot, tmp_path,
):
    """A skull-stripped scan is not raw data, so in place is a choice."""
    dlg = _strip(tmp_path)
    qtbot.addWidget(dlg)
    assert not dlg._keep.isChecked()
    assert "Overwrite the original" in dlg._keep.text()
    assert "not raw data" in dlg._keep.toolTip()


def test_ticking_overwrite_changes_what_the_status_promises(qtbot, tmp_path):
    dlg = _strip(tmp_path)
    qtbot.addWidget(dlg)
    if not run.available(dlg._current_engine().id):
        pytest.skip("no strip engine available here")
    dlg._keep.setChecked(True)
    assert "derivatives" not in dlg._status.text()
    assert ".bidsmgr" in dlg._status.text()


def test_an_engine_needing_an_extra_greys_out_only_itself(qtbot, tmp_path,
                                                          monkeypatch):
    """A missing extra must not disable the engine that needs nothing."""
    from bidsmgr.gui import deface_dialog as mod

    dlg = _strip(tmp_path)
    qtbot.addWidget(dlg)
    monkeypatch.setattr(
        mod, "unavailable_reason",
        lambda engine_id=None: (
            "mindgrab needs the brainchop extra" if engine_id == "mindgrab"
            else None
        ),
    )
    ids = [dlg._engine.itemData(i) for i in range(dlg._engine.count())]

    # Start on the one that needs nothing, so each switch actually fires.
    dlg._engine.setCurrentIndex(ids.index("strip-atlas"))
    assert dlg._ok.isEnabled(), "the engine that needs nothing was greyed out"

    dlg._engine.setCurrentIndex(ids.index("mindgrab"))
    assert not dlg._ok.isEnabled()
    assert "brainchop extra" in dlg._status.text()


def test_the_two_modes_have_different_titles_and_buttons(qtbot, tmp_path):
    from bidsmgr.deface import engines as e

    root = _dataset(tmp_path)
    face = DefaceDialog(root)
    strip = DefaceDialog(root, kind=e.KIND_STRIP)
    qtbot.addWidget(face)
    qtbot.addWidget(strip)

    assert face.windowTitle() != strip.windowTitle()
    assert face._ok.text() == "Remove faces"
    assert strip._ok.text() == "Remove the skull"


# ---------------------------------------------------------------------------
# Derivatives (1.5.1): a preprocessed image carries the face of its source
# ---------------------------------------------------------------------------


def _with_derivatives(tmp_path):
    root = _dataset(tmp_path)
    pipe = root / "derivatives" / "fmriprep"
    (pipe / "sub-01" / "anat").mkdir(parents=True)
    (pipe / "dataset_description.json").write_text(json.dumps(
        {"Name": "fmriprep", "BIDSVersion": "1.11.1", "DatasetType": "derivative"}))
    shutil.copyfile(TEMPLATE, pipe / "sub-01" / "anat" / "sub-01_desc-preproc_T1w.nii.gz")
    # A mask is not an image of the head: never a candidate.
    shutil.copyfile(TEMPLATE, pipe / "sub-01" / "anat" / "sub-01_desc-brain_mask.nii.gz")
    return root


def _heads(dlg):
    tree = dlg._preview
    return {tree.topLevelItem(i).text(0): tree.topLevelItem(i)
            for i in range(tree.topLevelItemCount())}


def test_derivatives_are_offered_and_off_by_default(qtbot, tmp_path):
    dlg = DefaceDialog(_with_derivatives(tmp_path))
    qtbot.addWidget(dlg)
    assert dlg._derivatives.isEnabled() and not dlg._derivatives.isChecked()
    assert dlg._derivatives_hint.text() == "derivatives/fmriprep/"
    assert sorted(dlg._checked()) == ["sub-01/anat/sub-01_T1w.nii.gz",
                                      "sub-01/anat/sub-01_T2w.nii.gz"]
    # Raw images stay ungrouped while derivatives are out of it.
    assert "Raw data (2)" not in _heads(dlg)


def test_including_derivatives_groups_them_by_pipeline(qtbot, tmp_path):
    dlg = DefaceDialog(_with_derivatives(tmp_path))
    qtbot.addWidget(dlg)
    dlg._derivatives.setChecked(True)
    heads = _heads(dlg)
    assert "Raw data (2)" in heads and "derivatives/fmriprep/ (1)" in heads
    deriv = "derivatives/fmriprep/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz"
    assert deriv in dlg._checked()
    assert not any("mask" in c for c in dlg._checked())
    # Unticking a pipeline unticks its images, and only them.
    heads["derivatives/fmriprep/ (1)"].setCheckState(0, Qt.CheckState.Unchecked)
    assert deriv not in dlg._checked() and len(dlg._checked()) == 2
    # The mask is listed among the skips, with why.
    skipped = [h for t, h in heads.items() if t.startswith("Cannot be done")]
    reasons = [skipped[0].child(i).text(0) for i in range(skipped[0].childCount())]
    assert any("not an image of the head" in r for r in reasons)


def test_a_derivative_picked_in_the_tree_ticks_derivatives(qtbot, tmp_path):
    root = _with_derivatives(tmp_path)
    image = root / "derivatives/fmriprep/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz"
    dlg = DefaceDialog(root, [image])
    qtbot.addWidget(dlg)
    assert dlg._derivatives.isChecked()
    assert dlg._checked() == [
        "derivatives/fmriprep/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz"]


def test_without_derivatives_the_option_is_greyed_and_says_why(qtbot, tmp_path):
    dlg = DefaceDialog(_dataset(tmp_path))
    qtbot.addWidget(dlg)
    assert not dlg._derivatives.isEnabled()
    assert dlg._derivatives_hint.text() == "No derivatives in this dataset"


def test_strip_mode_names_where_a_derivative_goes(qtbot, tmp_path):
    from bidsmgr.deface import engines as e

    dlg = DefaceDialog(_with_derivatives(tmp_path), kind=e.KIND_STRIP)
    qtbot.addWidget(dlg)
    dlg._derivatives.setChecked(True)
    notes = {item.data(0, Qt.ItemDataRole.UserRole): item.text(1)
             for item in dlg._candidate_items()}
    assert notes["derivatives/fmriprep/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz"] == \
        "brain kept, written as sub-01_desc-preprocbrain_T1w.nii.gz"
    assert notes["sub-01/anat/sub-01_T1w.nii.gz"] == \
        "brain kept, written as sub-01_desc-brain_T1w.nii.gz"
    # Overwriting changes what every row promises.
    dlg._keep.setChecked(True)
    assert {item.text(1) for item in dlg._candidate_items()} == {
        "brain kept, original overwritten"}


def test_a_derivative_is_not_mirrored_to_sourcedata(qtbot, tmp_path):
    dlg = DefaceDialog(_with_derivatives(tmp_path))
    qtbot.addWidget(dlg)
    dlg._derivatives.setChecked(True)
    dlg._keep.setChecked(True)
    if dlg._ok.isEnabled():
        assert "Derivatives are not copied to sourcedata/" in dlg._status.text()

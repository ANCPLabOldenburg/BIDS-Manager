"""A dataset's name, and what changing it is taken to mean.

Two different intentions wear the same edit. A folder named in a hurry gets its
real name later, which is a rename. Or the folder name is the one everybody
uses and what is being typed is the dataset's title for publication.

BIDS Manager used to guess: whatever went into dataset_description.json became
the name shown everywhere, so typing a title silently renamed the project in the
interface while the folder on disk stayed put and the two drifted apart with
nothing to say so.
"""

from __future__ import annotations

import json

import pytest

from bidsmgr.cli.create import main as create_main
from bidsmgr.gui.recording_meta_dialog import RecordingMetaDialog
from bidsmgr.gui.rename_dataset_dialog import (
    KEEP_FOLDER,
    RENAME_ALL,
    RenameDatasetDialog,
    can_rename_to,
)
from bidsmgr.gui.welcome_panel import _dataset_display_name, _dataset_title
from bidsmgr.gui.widgets.template_form import read_field_widget, write_field_widget
from bidsmgr.recording_meta import scaffold_sidecar_path

pytestmark = pytest.mark.gui


def _dataset(tmp_path, folder: str, name: str):
    root = tmp_path / folder
    create_main([str(root), "--name", name])
    tsv = root / "inv.tsv"
    tsv.write_text("include\n1\n")
    return root, tsv


def _dialog(qtbot, root, tsv):
    dlg = RecordingMetaDialog(
        scaffold_sidecar_path(tsv), {"eeg"}, None,
        present_pairs=[("eeg", "eeg")], bids_root=root,
    )
    qtbot.addWidget(dlg)
    return dlg


# ---------------------------------------------------------------------------
# The name a dataset was created with
# ---------------------------------------------------------------------------


def test_the_template_opens_showing_the_name_it_was_created_with(qtbot, tmp_path):
    """Otherwise the form shows an empty box for a name the user already chose,
    and saving writes the blank over it."""
    root, tsv = _dataset(tmp_path, "study2", "study2")
    dlg = _dialog(qtbot, root, tsv)

    widget = dlg._template.widgets_for("dataset_description")["Name"]
    field = dlg._template._fields["dataset_description"]["Name"]
    assert read_field_widget(widget, field) == "study2"


# ---------------------------------------------------------------------------
# What a change means
# ---------------------------------------------------------------------------


def test_a_new_name_asks_what_it_means(qtbot, tmp_path, monkeypatch):
    root, tsv = _dataset(tmp_path, "study2", "study2")
    dlg = _dialog(qtbot, root, tsv)
    asked = {}

    def _fake(folder, new, *a, **k):
        asked["folder"], asked["new"] = folder, new
        return KEEP_FOLDER

    monkeypatch.setattr(
        "bidsmgr.gui.rename_dataset_dialog.ask_what_the_name_means", _fake,
    )
    write_field_widget(
        dlg._template.widgets_for("dataset_description")["Name"], "Real Title",
    )
    spec = dlg.build_spec()
    dlg._read_template(spec)

    assert dlg._settle_the_name(spec) is True
    assert asked == {"folder": "study2", "new": "Real Title"}
    assert dlg.rename_project_to is None, "keeping the folder must not rename it"


def test_choosing_rename_reports_the_new_folder_name(qtbot, tmp_path, monkeypatch):
    root, tsv = _dataset(tmp_path, "study2", "study2")
    dlg = _dialog(qtbot, root, tsv)
    monkeypatch.setattr(
        "bidsmgr.gui.rename_dataset_dialog.ask_what_the_name_means",
        lambda *a, **k: RENAME_ALL,
    )
    write_field_widget(
        dlg._template.widgets_for("dataset_description")["Name"], "Ageing",
    )
    spec = dlg.build_spec()
    dlg._read_template(spec)

    assert dlg._settle_the_name(spec) is True
    assert dlg.rename_project_to == "Ageing"


def test_backing_out_of_the_question_cancels_the_save(qtbot, tmp_path, monkeypatch):
    root, tsv = _dataset(tmp_path, "study2", "study2")
    dlg = _dialog(qtbot, root, tsv)
    monkeypatch.setattr(
        "bidsmgr.gui.rename_dataset_dialog.ask_what_the_name_means",
        lambda *a, **k: None,
    )
    write_field_widget(
        dlg._template.widgets_for("dataset_description")["Name"], "Something",
    )
    spec = dlg.build_spec()
    dlg._read_template(spec)
    assert dlg._settle_the_name(spec) is False


def test_a_name_that_has_not_changed_is_not_questioned(qtbot, tmp_path, monkeypatch):
    """Saving the form again must not interrogate a settled name every time."""
    root, tsv = _dataset(tmp_path, "study2", "Already A Title")
    dlg = _dialog(qtbot, root, tsv)

    def _boom(*a, **k):
        raise AssertionError("should not ask about an unchanged name")

    monkeypatch.setattr(
        "bidsmgr.gui.rename_dataset_dialog.ask_what_the_name_means", _boom,
    )
    spec = dlg.build_spec()
    dlg._read_template(spec)
    assert dlg._settle_the_name(spec) is True


def test_a_name_a_folder_cannot_take_offers_only_the_title(qtbot, tmp_path):
    root, _tsv = _dataset(tmp_path, "study2", "study2")
    assert can_rename_to(root, "Ageing") is True
    assert can_rename_to(root, "a/b") is False, "a separator is not a folder name"

    (tmp_path / "taken").mkdir()
    assert can_rename_to(root, "taken") is False, "a sibling already has it"

    dialog = RenameDatasetDialog("study2", "taken", can_rename=False)
    qtbot.addWidget(dialog)
    assert not dialog._rename.isEnabled()
    assert dialog.choice() == KEEP_FOLDER


# ---------------------------------------------------------------------------
# How the two names are shown
# ---------------------------------------------------------------------------


def test_the_project_is_its_folder_and_the_title_sits_beside_it(tmp_path):
    root, _ = _dataset(tmp_path, "study2", "study2")
    description = root / "dataset_description.json"
    data = json.loads(description.read_text())
    data["Name"] = "Verbal Working Memory in Ageing"
    description.write_text(json.dumps(data))

    assert _dataset_display_name(root) == "study2"
    assert _dataset_title(root) == "Verbal Working Memory in Ageing"


def test_nothing_is_shown_twice_when_they_agree(tmp_path):
    root, _ = _dataset(tmp_path, "Ageing", "Ageing")
    assert _dataset_display_name(root) == "Ageing"
    assert _dataset_title(root) == ""


def test_a_dataset_with_no_description_still_has_a_name(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    assert _dataset_display_name(root) == "bare"
    assert _dataset_title(root) == ""

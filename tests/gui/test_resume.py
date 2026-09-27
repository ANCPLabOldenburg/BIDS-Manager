"""Resume curation + versioned scans (Phase E-1 + E-2).

Each scan of a source folder is its own version under
``.bidsmgr/project/scans/<version>/`` with an isolated curation bundle. Reopening
a project resumes the latest version (replaying its cell / include / entity
edits); a second scan creates a new version without overwriting the first; the
toolbar picker switches between versions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.cli.create import open_or_create_workspace
from bidsmgr.gui.converter_panel import ConverterPanel
from bidsmgr.project import (
    Project,
    ScanImported,
    UserSetCell,
    UserSetEntity,
    UserToggleInclude,
    workspace,
)

pytestmark = pytest.mark.gui


def _inventory_df(subject: str = "001", task: str = "rest", uid: str = "UID1") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "participant_id": f"sub-{subject}", "session": "", "include": 1,
            "modality": "mri", "datatype": "func",
            "bids_name": f"sub-{subject}_task-{task}_bold",
            "bids_guess_suffix": "bold", "bids_guess_confidence": "0.9",
            "bids_guess_skip": False, "issues": "",
            "entities": json.dumps({"subject": subject, "task": task}, sort_keys=True),
            "task": task, "run": "", "series_uid": uid, "dataset": "ds",
            "source_file": "",
        },
    ])


def _add_version(bids_root: Path, source_label: str, df: pd.DataFrame, raw_root: str):
    """Create a scan version (its own bundle + inventory) and return its Project."""
    vdir = workspace.allocate_version_dir(bids_root, source_label)
    proj = Project.create(vdir, name=bids_root.name)
    df.to_csv(workspace.version_inventory(vdir), sep="\t", index=False)
    workspace.write_version_meta(
        vdir, source_label=source_label, raw_root=raw_root, status="curating",
    )
    rids = tuple(df["series_uid"].tolist())
    proj.append(ScanImported(
        inventory_tsv=str(workspace.version_inventory(vdir)),
        row_ids=rids, raw_root=raw_root,
    ))
    return vdir, proj


def test_resume_loads_latest_version_and_paths(qtbot, isolated_settings, tmp_path):
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir, _ = _add_version(root, "raw_eeg", _inventory_df(), str(tmp_path / "raw"))

    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    assert panel._model is not None
    assert panel._active_version_dir == vdir
    assert panel._raw_root == tmp_path / "raw"


def test_resume_replays_cell_include_and_entity_edits(qtbot, isolated_settings, tmp_path):
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir, vproj = _add_version(root, "raw", _inventory_df(), str(tmp_path / "raw"))
    vproj.append(UserSetCell(row_id="UID1", column="task", value="memory", previous="rest"))
    vproj.append(UserToggleInclude(row_id="UID1", include=False))

    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    df = panel._model.dataframe()
    assert df.iloc[0]["task"] == "memory"
    assert panel._model.row_state(0) == "skip"


def test_resume_replays_subject_rename(qtbot, isolated_settings, tmp_path):
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir, vproj = _add_version(root, "raw", _inventory_df(), str(tmp_path / "raw"))
    vproj.append(UserSetEntity(row_id="UID1", entity="subject", value="042", previous="001"))

    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    assert panel._model.dataframe().iloc[0]["participant_id"] == "sub-042"


def test_two_versions_coexist_and_picker_switches(qtbot, isolated_settings, tmp_path):
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    v1, _ = _add_version(root, "source_a", _inventory_df(subject="001", uid="A1"), str(tmp_path / "a"))
    v2, _ = _add_version(root, "source_b", _inventory_df(subject="002", uid="B1"), str(tmp_path / "b"))

    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    # Both versions exist; resume lands on the latest (v2 / source_b).
    assert len(workspace.list_versions(root)) == 2
    assert panel._active_version_dir == v2
    assert panel._model.dataframe().iloc[0]["participant_id"] == "sub-002"
    # isHidden() reflects the explicit setVisible flag without needing the whole
    # window shown (offscreen). The picker lists both versions.
    assert not panel._scans_combo.isHidden() and panel._scans_combo.count() == 2

    # Switch back to the first version via the picker.
    idx_v1 = next(i for i in range(panel._scans_combo.count())
                  if Path(panel._scans_combo.itemData(i)) == v1)
    panel._scans_combo.setCurrentIndex(idx_v1)
    panel._on_scans_combo_activated(idx_v1)
    assert panel._active_version_dir == v1
    assert panel._model.dataframe().iloc[0]["participant_id"] == "sub-001"


def test_undo_reverts_last_edit(qtbot, isolated_settings, tmp_path):
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    _add_version(root, "raw", _inventory_df(), str(tmp_path / "raw"))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    # An edit on the active version, then reload so the model shows it.
    panel._project.append(UserSetCell(row_id="UID1", column="task", value="memory", previous="rest"))
    panel._reload_active_version()
    assert panel._model.dataframe().iloc[0]["task"] == "memory"

    panel._on_undo()  # reverts the edit
    assert panel._model.dataframe().iloc[0]["task"] == "rest"


def test_redo_reapplies_undone_edit(qtbot, isolated_settings, tmp_path):
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    _add_version(root, "raw", _inventory_df(), str(tmp_path / "raw"))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    panel._project.append(UserSetCell(row_id="UID1", column="task", value="memory", previous="rest"))
    panel._reload_active_version()
    assert panel._model.dataframe().iloc[0]["task"] == "memory"

    panel._on_undo()
    assert panel._model.dataframe().iloc[0]["task"] == "rest"
    assert panel._redo_btn.isEnabled()

    panel._on_redo()
    assert panel._model.dataframe().iloc[0]["task"] == "memory"  # redone
    assert not panel._redo_btn.isEnabled()  # stack drained


def test_new_edit_clears_redo_stack(qtbot, isolated_settings, tmp_path):
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    _add_version(root, "raw", _inventory_df(), str(tmp_path / "raw"))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    panel._project.append(UserSetCell(row_id="UID1", column="task", value="x", previous="rest"))
    panel._reload_active_version()
    panel._on_undo()
    assert panel._redo_stack  # something is redoable

    panel._on_user_edited()  # a fresh edit diverges the timeline
    assert not panel._redo_stack


def test_undo_is_noop_when_last_event_is_structural(qtbot, isolated_settings, tmp_path):
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    _add_version(root, "raw", _inventory_df(), str(tmp_path / "raw"))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    # Last event is ScanImported (structural) -> undo must not pop it.
    panel._on_undo()
    assert panel._model.dataframe().iloc[0]["task"] == "rest"  # unchanged


def test_relocate_source_persists_to_version_meta(
    qtbot, isolated_settings, tmp_path, monkeypatch,
):
    from bidsmgr.gui import converter_panel as cp

    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir, _ = _add_version(root, "raw", _inventory_df(), str(tmp_path / "old_raw"))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    new_raw = tmp_path / "new_raw"
    new_raw.mkdir()
    monkeypatch.setattr(
        cp.QFileDialog, "getExistingDirectory", lambda *a, **k: str(new_raw),
    )
    panel._on_pick_raw_dir()

    assert panel._raw_root == new_raw
    assert workspace.read_version_meta(vdir)["raw_root"] == str(new_raw)


def test_fresh_scan_flags_rows_whose_subject_already_exists(
    qtbot, isolated_settings, tmp_path,
):
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    (root / "sub-001").mkdir()  # already converted on disk
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    df = pd.DataFrame([
        {"participant_id": "sub-001", "issues": ""},  # collides with on-disk
        {"participant_id": "sub-002", "issues": ""},  # new
    ])
    panel._flag_existing_subject_rows(df)
    # No participants.tsv identity -> generic heads-up, tagged with the token.
    assert "already in the dataset" in df.iloc[0]["issues"]
    assert "existing-subject" in df.iloc[0]["issues"]
    assert df.iloc[1]["issues"] == ""  # new subject not flagged


def test_flag_existing_subject_rows_is_idempotent_after_rename(
    qtbot, isolated_settings, tmp_path,
):
    # Renaming a clashing subject must clear the stale collision note (it bakes
    # in the old id); re-flagging the same frame is the engine behind Re-validate.
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    (root / "sub-001").mkdir()
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    df = pd.DataFrame([{"participant_id": "sub-001", "issues": ""}])
    panel._flag_existing_subject_rows(df)
    assert "existing-subject" in df.iloc[0]["issues"]

    # User renames to a free id, then re-flags: the warning is gone (no double).
    df.at[0, "participant_id"] = "sub-002"
    panel._flag_existing_subject_rows(df)
    assert df.iloc[0]["issues"] == ""


def test_revalidate_clears_collision_after_bulk_rename(
    qtbot, isolated_settings, tmp_path,
):
    # Full flow the user hit: convert sub-001, scan a second source that also
    # lands on sub-001 (warn), rename to sub-002, click Re-validate -> warning
    # clears (no lingering "sub-001 ... DIFFERENT subject" note).
    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    (root / "sub-001").mkdir()  # subject already on disk
    _add_version(root, "raw", _inventory_df(subject="001", uid="X1"), str(tmp_path / "raw"))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    # Flag collisions as the scan-complete path would.
    panel._flag_existing_subject_rows(panel._model.dataframe())
    panel._model.revalidate_all()
    assert panel._model.row_state(0) == "warn"  # collides with on-disk sub-001

    # Rename via the bulk-edit code path (id -> subject entity + participant_id).
    panel._model.bulk_set([0], "id", "002")
    # Live edit alone leaves the stale note (manual revalidation is by design).
    assert panel._model.row_state(0) == "warn"

    # Re-validate recomputes the collision against the new id + disk -> cleared.
    panel._on_revalidate_clicked()
    assert panel._model.row_state(0) == ""
    issues = panel._model.dataframe().at[0, "issues"]
    assert "existing-subject" not in issues and "sub-001" not in issues


def test_set_project_without_scan_is_noop(qtbot, isolated_settings, tmp_path):
    root = tmp_path / "fresh"
    proj = open_or_create_workspace(root)
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)
    assert panel._model is None
    assert panel._scans_combo.isHidden()  # no versions -> picker hidden


def test_binding_new_empty_project_clears_previous(qtbot, isolated_settings, tmp_path):
    # Project A with a scan -> the converter shows its rows.
    root_a = tmp_path / "A"
    proj_a = open_or_create_workspace(root_a)
    _add_version(root_a, "raw", _inventory_df(), str(tmp_path / "rawA"))
    panel = ConverterPanel(project=proj_a)
    qtbot.addWidget(panel)
    panel.set_project(proj_a, root_a)
    assert panel._model is not None

    # Switching to a fresh, empty project B must reset the converter so A's rows
    # / raw input / active version never leak across datasets.
    root_b = tmp_path / "B"
    proj_b = open_or_create_workspace(root_b)
    panel.set_project(proj_b, root_b)
    assert panel._model is None
    assert panel._raw_root is None
    assert panel._active_version_dir is None
    assert panel._scans_combo.isHidden()
    assert panel._bids_root == root_b


def test_open_existing_syncs_dataset_column_to_folder(
    qtbot, isolated_settings, tmp_path,
):
    # The dataset folder was renamed on disk, so the stored inventory's dataset
    # column is stale. Opening the project must re-sync it to the folder name so
    # conversion writes into the (renamed) locked root.
    root = tmp_path / "renamed_ds"
    proj = open_or_create_workspace(root)
    df = _inventory_df()
    df["dataset"] = "old_folder_name"
    _add_version(root, "raw", df, str(tmp_path / "raw"))

    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)

    assert (panel._model.dataframe()["dataset"] == "renamed_ds").all()

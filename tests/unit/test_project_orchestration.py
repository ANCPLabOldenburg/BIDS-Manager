"""Tests for the shared, Qt-free project orchestration (``--project`` engine).

Covers the version-import + edit-replay + active-version-load functions that
both the GUI and the CLI ``--project`` flags drive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bidsmgr.cli.create import open_or_create_workspace
from bidsmgr.project import Project, UserSetEntity, UserToggleInclude, workspace
from bidsmgr.project.orchestration import (
    apply_project_state,
    find_version,
    import_scan_as_version,
    latest_version,
    row_id,
    version_dataframe,
)


def _inv_df(subject="001", task="rest", uid="UID1") -> pd.DataFrame:
    return pd.DataFrame([{
        "participant_id": f"sub-{subject}", "session": "", "include": 1,
        "modality": "mri", "datatype": "func",
        "bids_name": f"sub-{subject}_task-{task}_bold",
        "bids_guess_suffix": "bold", "bids_guess_skip": False,
        "issues": "",
        "entities": json.dumps({"subject": subject, "task": task}, sort_keys=True),
        "task": task, "run": "", "series_uid": uid, "dataset": "ds",
        "source_file": "",
    }])


def test_row_id_prefers_series_uid_then_source_file(tmp_path):
    df = pd.DataFrame([
        {"series_uid": "U1", "source_file": ""},
        {"series_uid": "", "source_file": "/x/rec.edf"},
        {"series_uid": "", "source_file": ""},
    ])
    assert row_id(df, 0) == "U1"
    assert row_id(df, 1) == "/x/rec.edf"
    assert row_id(df, 2) == "row-2"


def test_apply_project_state_replays_entity_cell_include(tmp_path):
    df = _inv_df(uid="U1")
    proj = Project.create(tmp_path / "v", name="ds")
    proj.append(UserSetEntity(row_id="U1", entity="task", value="memory", previous="rest"))
    proj.append(UserToggleInclude(row_id="U1", include=False))

    apply_project_state(df, proj.state())
    assert df.iloc[0]["task"] == "memory"
    assert "task-memory" in df.iloc[0]["bids_name"]
    assert str(df.iloc[0]["include"]) in ("0", "False", "false")


def test_import_scan_as_version_and_latest(tmp_path):
    root = tmp_path / "ds"
    open_or_create_workspace(root)
    staged = root / ".bidsmgr" / "project" / ".scan_staging" / "inventory.tsv"
    staged.parent.mkdir(parents=True, exist_ok=True)
    df = _inv_df(uid="U1")
    df.to_csv(staged, sep="\t", index=False)

    version = import_scan_as_version(
        root, staged, raw_root=tmp_path / "raw",
        row_ids=("U1",), n_rows=len(df),
    )
    # The staged TSV moved into the version dir.
    assert version.inventory.exists()
    assert not staged.exists()
    assert latest_version(root).version_id == version.version_id
    # ScanImported event recorded in the version bundle.
    proj = Project.open(version.dir)
    from bidsmgr.project import ScanImported
    assert any(isinstance(e, ScanImported) for e in proj.log)


def test_version_dataframe_replays_edits(tmp_path):
    root = tmp_path / "ds"
    open_or_create_workspace(root)
    staged = root / ".bidsmgr" / "project" / ".scan_staging" / "inventory.tsv"
    staged.parent.mkdir(parents=True, exist_ok=True)
    _inv_df(subject="001", uid="U1").to_csv(staged, sep="\t", index=False)
    version = import_scan_as_version(root, staged, raw_root=tmp_path / "raw", row_ids=("U1",))

    # Simulate a GUI rename of the subject, recorded in the version bundle.
    proj = Project.open(version.dir)
    proj.append(UserSetEntity(row_id="U1", entity="subject", value="042", previous="001"))

    df = version_dataframe(latest_version(root), apply_edits=True)
    assert df.iloc[0]["participant_id"] == "sub-042"
    # Without replay the on-disk inventory is unchanged.
    raw = version_dataframe(latest_version(root), apply_edits=False)
    assert raw.iloc[0]["participant_id"] == "sub-001"


def test_find_version_by_id_and_index(tmp_path):
    root = tmp_path / "ds"
    open_or_create_workspace(root)
    for label in ("alpha", "beta"):
        staged = root / ".bidsmgr" / "project" / ".scan_staging" / "inventory.tsv"
        staged.parent.mkdir(parents=True, exist_ok=True)
        _inv_df().to_csv(staged, sep="\t", index=False)
        import_scan_as_version(root, staged, raw_root=tmp_path / label, source_label=label)

    versions = workspace.list_versions(root)
    assert len(versions) == 2
    # By full id, by numeric index, by zero-padded index.
    assert find_version(root, versions[0].version_id).index == 1
    assert find_version(root, "2").index == 2
    assert find_version(root, "0001").index == 1
    assert find_version(root, "nope") is None

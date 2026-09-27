"""Tests for the additive ``--project`` CLI flags + ``bidsmgr-project``.

The heavy scan/convert paths need real data (covered by the real-data flow);
here we cover the argparse wiring, the metadata ``--project`` resolution, and
the ``bidsmgr-project`` listing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.cli import convert as convert_cli
from bidsmgr.cli import metadata as metadata_cli
from bidsmgr.cli import project as project_cli
from bidsmgr.cli import scan as scan_cli
from bidsmgr.cli.create import open_or_create_workspace
from bidsmgr.project.orchestration import import_scan_as_version


def _seed_version(root: Path) -> None:
    open_or_create_workspace(root)
    staged = root / ".bidsmgr" / "project" / ".scan_staging" / "inventory.tsv"
    staged.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "participant_id": "sub-001", "participant_id": "sub-001",
        "series_uid": "U1", "dataset": root.name, "include": 1,
    }]).to_csv(staged, sep="\t", index=False)
    import_scan_as_version(root, staged, raw_root=root / "raw", row_ids=("U1",))


def test_scan_requires_output_or_project(capsys):
    with pytest.raises(SystemExit):
        scan_cli.main(["/tmp/raw"])  # no output_tsv, no --project


def test_convert_requires_positionals_or_project(capsys):
    with pytest.raises(SystemExit):
        convert_cli.main(["only_one_arg"])  # missing bids_parent + no --project


def test_metadata_requires_target_or_project(capsys):
    with pytest.raises(SystemExit):
        metadata_cli.main([])  # no target, no --project


def test_bidsmgr_project_lists_versions(tmp_path, capsys):
    root = tmp_path / "ds"
    _seed_version(root)
    rc = project_cli.main([str(root)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 scan version(s)" in out
    assert "0001__" in out and "*" in out  # active marker


def test_bidsmgr_project_rejects_non_project(tmp_path, capsys):
    plain = tmp_path / "plain"
    plain.mkdir()
    rc = project_cli.main([str(plain)])
    assert rc == 1
    assert "not a BIDS-Manager project" in capsys.readouterr().out


def test_convert_project_unknown_version_errors(tmp_path, capsys):
    root = tmp_path / "ds"
    _seed_version(root)
    with pytest.raises(SystemExit):
        convert_cli.main(["--project", str(root), "--version", "9999"])


def test_metadata_project_writes_participants(tmp_path):
    root = tmp_path / "ds"
    _seed_version(root)
    # A converted subject on disk so the metadata engine writes participants.tsv.
    (root / "sub-001" / "anat").mkdir(parents=True)
    (root / "sub-001" / "anat" / "sub-001_T1w.json").write_text("{}", encoding="utf-8")

    rc = metadata_cli.main(["--project", str(root)])
    assert rc == 0
    assert (root / "participants.tsv").exists()

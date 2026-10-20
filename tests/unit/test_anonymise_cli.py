"""``bidsmgr-anonymise``, and ``--anonymise`` on the metadata and convert verbs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bidsmgr.cli import anonymise as cli
from bidsmgr.project.operations import peek_undo


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_T1w.json").write_text(json.dumps({
        "AcquisitionDateTime": "2025-05-26T11:14:30", "PatientName": "Doe^Jane"}))
    (anat / "sub-01_T1w.nii.gz").write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03")
    (root / "participants.tsv").write_text("participant_id\tgiven_name\nsub-01\tJane\n")
    (root / "dataset_description.json").write_text('{"Name": "d", "BIDSVersion": "1.10.0"}')
    return root


def _files(root: Path) -> dict:
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()
            and ".bidsmgr" not in p.parts}


def test_a_dry_run_lists_and_writes_nothing(ds: Path, capsys) -> None:
    before = _files(ds)
    assert cli.main([str(ds), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "PatientName" in out and "(removed)" in out and "given_name" in out
    assert _files(ds) == before


def test_json_says_the_same_for_a_script(ds: Path, capsys) -> None:
    assert cli.main([str(ds), "--dry-run", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    paths = {f["path"] for f in data["files"]}
    assert "sub-01/anat/sub-01_T1w.json" in paths and data["preset"] == "recommended"


def test_applied_it_is_one_step_of_the_history(ds: Path) -> None:
    assert cli.main([str(ds), "--preset", "strict"]) == 0
    side = json.loads((ds / "sub-01/anat/sub-01_T1w.json").read_text())
    assert "PatientName" not in side and side["AcquisitionDateTime"].endswith("T00:00:00")
    assert peek_undo(ds).label == "Anonymise metadata (Strict)"


def test_not_a_dataset_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path)])


def test_the_metadata_step_anonymises_last(ds: Path) -> None:
    """participants.tsv is written by the metadata step: anonymised before
    it, the name would come back."""
    from bidsmgr.cli.metadata import run_metadata_cli

    assert run_metadata_cli(ds, anonymise="recommended", write_report=False) == 0
    text = (ds / "participants.tsv").read_text()
    assert "given_name" not in text and "Jane" not in text


def test_convert_anonymises_what_it_converted(ds: Path) -> None:
    from bidsmgr.cli.convert import _anonymise_after

    args = SimpleNamespace(dry_run=False, anonymise="recommended")
    assert _anonymise_after(0, ds.parent, [ds.name], args) == 0
    assert "PatientName" not in (ds / "sub-01/anat/sub-01_T1w.json").read_text()
    args.anonymise = "off"
    assert _anonymise_after(3, ds.parent, [ds.name], args) == 3, "a failed conversion is not"


def test_the_dry_run_shows_what_the_real_run_does(ds: Path, capsys) -> None:
    """The offsets the dry run chose are remembered, so the dates it listed
    are the dates the real run writes."""
    import re

    cli.main([str(ds), "--dry-run"])
    shown = re.search(r"AcquisitionDateTime: \S+ -> \"(\S+)\"", capsys.readouterr().out).group(1)
    cli.main([str(ds)])
    side = json.loads((ds / "sub-01/anat/sub-01_T1w.json").read_text())
    assert side["AcquisitionDateTime"] == shown

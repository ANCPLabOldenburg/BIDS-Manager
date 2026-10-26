"""``bidsmgr-sidecars``: the Editor's Sidecars from the recordings, from a shell.

Same engine as the dialog (its own tests pin the rules); these pin what the
command promises: a dry run writes nothing, a run writes one undoable step,
files not named the BIDS way are done and can be told what they are, the BIDS
version is honoured, and bad arguments are refused before anything is read.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bidsmgr.cli.sidecars import main


def _nifti(path: Path, shape=(4, 4, 3), tr=2.0, descrip=b"TE=30;Time=1;phase=1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.eye(4))
    img.header["pixdim"][4] = tr
    img.header.set_xyzt_units("mm", "sec")
    img.header["descrip"] = descrip
    nib.save(img, str(path))
    return path


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A folder of recordings with no BIDS structure, and one BIDS run."""
    root = tmp_path / "study"
    _nifti(root / "mri/subject1/t1_mprage.nii", tr=2.3)
    _nifti(root / "mri/subject1/rest.nii.gz", (4, 4, 3, 6), tr=0.8)
    _nifti(root / "sub-01/func/sub-01_task-rest_bold.nii.gz", (4, 4, 3, 6), tr=0.8)
    (root / "mri/subject1/notes.pdf").write_bytes(b"%PDF-1.4")
    return root


def _sidecars(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.json")
                  if ".bidsmgr" not in p.parts)


def test_a_dry_run_lists_and_writes_nothing(folder, capsys) -> None:
    assert main([str(folder), "--dry-run", "--no-log-file"]) == 0
    out = capsys.readouterr().out
    assert "3 new sidecars" in out and "BIDS 1." in out
    assert "mri/subject1/t1_mprage.json  [anat/T1w, guessed]" in out
    assert "sub-01/func/sub-01_task-rest_bold.json  [func/bold, BIDS name]" in out
    assert "Not a BIDS name or place" in out
    assert _sidecars(folder) == []


def test_a_run_writes_one_undoable_step(folder, capsys) -> None:
    from bidsmgr.project.operations import peek_undo, undo_last

    assert main([str(folder), "--no-log-file"]) == 0
    assert _sidecars(folder) == ["mri/subject1/rest.json", "mri/subject1/t1_mprage.json",
                                 "sub-01/func/sub-01_task-rest_bold.json"]
    t1 = json.loads((folder / "mri/subject1/t1_mprage.json").read_text())
    assert t1 == {"RepetitionTime": 2.3, "EchoTime": 0.03}
    assert "Wrote 3 sidecars" in capsys.readouterr().out
    assert peek_undo(folder).label == "Write 3 sidecars from the recordings"
    undo_last(folder)
    assert _sidecars(folder) == []


def test_paths_narrow_it_and_kinds_correct_a_guess(folder, capsys) -> None:
    rc = main([str(folder), "mri/subject1/rest.nii.gz", "--kind",
               "mri/subject1/rest.nii.gz=dwi/dwi", "--json", "--dry-run", "--no-log-file"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    (one,) = plan["files"]
    assert one["recording"] == "mri/subject1/rest.nii.gz"
    assert one["kind"] == {"datatype": "dwi", "suffix": "dwi", "how": "chosen",
                           "why": "as you chose"}
    assert {f["field"] for f in one["added"]} == {"RepetitionTime", "EchoTime"}


def test_complete_adds_to_a_sidecar_and_keeps_its_values(folder, capsys) -> None:
    side = folder / "mri/subject1/t1_mprage.json"
    side.write_text(json.dumps({"EchoTime": 0.0021, "Manufacturer": "Siemens"}))
    main([str(folder), "--dry-run", "--no-log-file"])
    assert "1 recording already has its sidecar" in capsys.readouterr().out
    assert main([str(folder), "mri/subject1/t1_mprage.nii", "--complete", "--fields",
                 "--no-log-file"]) == 0
    out = capsys.readouterr().out
    assert "RepetitionTime = 2.3  [File header]" in out
    assert json.loads(side.read_text()) == {
        "EchoTime": 0.0021, "Manufacturer": "Siemens", "RepetitionTime": 2.3}


def test_placeholders_follow_the_scope(folder, capsys) -> None:
    def origins(scope: str) -> dict:
        main([str(folder), "sub-01", "--fill-scope", scope, "--json", "--dry-run",
              "--no-log-file"])
        (one,) = json.loads(capsys.readouterr().out)["files"]
        return {f["field"]: (f["origin"], f["value"]) for f in one["added"]}

    none = origins("none")
    assert none["TaskName"] == ("filename", "rest")
    assert not [k for k, (o, _v) in none.items() if o == "placeholder"]
    recommended = origins("recommended")
    marked = {k: v for k, (o, v) in recommended.items() if o == "placeholder"}
    assert marked and all(v in ("TODO", ["TODO"]) for v in marked.values())
    assert set(recommended) > set(none)


def test_the_schema_version_is_honoured(folder, capsys) -> None:
    from bidsmgr import schema as schema_mod

    try:
        assert main([str(folder), "--schema", "1.9.0", "--json", "--dry-run",
                     "--no-log-file"]) == 0
        assert json.loads(capsys.readouterr().out)["bids_version"] == "1.9.0"
    finally:
        schema_mod.set_active_version(None)


@pytest.mark.parametrize("argv, message", [
    (["--kind", "rest.nii.gz"], "expected PATH=DATATYPE/SUFFIX"),
    (["missing.nii"], "does not exist"),
])
def test_bad_arguments_are_refused(folder, argv, message, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main([str(folder), *argv, "--no-log-file"])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_a_path_outside_the_folder_is_refused(folder, tmp_path, capsys) -> None:
    outside = _nifti(tmp_path / "elsewhere/x.nii.gz")
    with pytest.raises(SystemExit):
        main([str(folder), str(outside), "--no-log-file"])
    assert "is not inside" in capsys.readouterr().err

"""``*_scans.tsv`` and ``*_sessions.tsv``: every recording the standard
defines, the user's rows kept, times that are times, and a sessions table.

Before 1.5.1 ``beh`` and ``motion`` recordings never got a row (their data are
``.tsv``, which the engine took for companions), ``.bval``/``.bvec`` got rows
of their own, a PET row could carry a time without a date, rows a user added
for valid files were dropped, and nothing wrote ``*_sessions.tsv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.metadata.engine import run_metadata
from bidsmgr.metadata.tables import (
    is_scans_entry,
    iso_datetime,
    plan_tables,
    read_table,
    session_recordings,
    write_tables,
)


def _write(path: Path, content=b"\0") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        path.write_text(json.dumps(content), encoding="utf-8")
    elif isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _dataset(root: Path) -> Path:
    _write(root / "dataset_description.json", {"Name": "d", "BIDSVersion": "1.10.0"})
    return root


def _filenames(table: Path) -> list[str]:
    return [r["filename"] for r in read_table(table)[1]]


# ---------------------------------------------------------------------------
# What gets a row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,is_dir", [
    ("beh/sub-01_task-x_beh.tsv", False),
    ("motion/sub-01_task-x_tracksys-imu_motion.tsv", False),
    ("nirs/sub-01_task-x_nirs.snirf", False),
    ("emg/sub-01_task-x_emg.edf", False),
    ("micr/sub-01_sample-A_SEM.ome.zarr", True),
    ("ieeg/sub-01_task-x_ieeg.mefd", True),
    ("meg/sub-01_task-x_meg", True),
    ("meg/sub-01_task-x_meg.ds", True),
    ("eeg/sub-01_task-x_eeg.vhdr", False),
    ("eeg/sub-01_task-x_eeg.set", False),
    ("dwi/sub-01_dwi.nii.gz", False),
    ("pet/sub-01_pet.nii.gz", False),
    ("mrs/sub-01_svs.nii.gz", False),
    ("perf/sub-01_asl.nii.gz", False),
])
def test_every_datatype_the_standard_defines_is_listed(path, is_dir) -> None:
    assert is_scans_entry(path, is_dir=is_dir)


@pytest.mark.parametrize("path", [
    "dwi/sub-01_dwi.bval", "dwi/sub-01_dwi.bvec", "fmap/sub-01_dir-AP_epi.bvec",
    "anat/sub-01_T1w.json", "eeg/sub-01_task-x_eeg.eeg", "eeg/sub-01_task-x_eeg.vmrk",
    "eeg/sub-01_task-x_eeg.fdt", "eeg/sub-01_task-x_channels.tsv",
    "eeg/sub-01_task-x_events.tsv", "func/sub-01_task-x_physio.tsv.gz",
    "func/sub-01_task-x_stim.tsv.gz", "eeg/sub-01_photo.jpg",
    "meg/sub-01_acq-crosstalk_meg.fif", "meg/sub-01_acq-calibration_meg.dat",
    "meg/sub-01_headshape.pos", "perf/sub-01_aslcontext.tsv",
    "pet/sub-01_recording-manual_blood.tsv", "func/sub-01_bold.nii.gz",
])
def test_side_files_companions_and_invalid_names_are_not(path) -> None:
    assert not is_scans_entry(path)


def test_a_behaviour_only_events_file_is_the_recording(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    beh = root / "sub-01" / "beh"
    _write(beh / "sub-01_task-quiz_events.tsv", "onset\tduration\n1\t0\n")
    _write(beh / "sub-01_task-game_beh.tsv", "trial\n1\n")
    _write(beh / "sub-01_task-game_events.tsv", "onset\tduration\n1\t0\n")
    found = [p.name for p in session_recordings(root, root / "sub-01")]
    assert found == ["sub-01_task-game_beh.tsv", "sub-01_task-quiz_events.tsv"]


def test_bidsignore_and_hidden_files_are_left_out(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    anat = root / "sub-01" / "anat"
    _write(anat / "sub-01_T1w.nii.gz")
    _write(anat / "sub-01_T2w.nii.gz")
    _write(anat / ".DS_Store")
    (root / ".bidsignore").write_text("**/*_T2w.nii.gz\n")
    assert [p.name for p in session_recordings(root, root / "sub-01")] == ["sub-01_T1w.nii.gz"]


def test_a_directory_recording_is_one_row(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    ds = root / "sub-01" / "meg" / "sub-01_task-x_meg.ds"
    _write(ds / "inner.meg4")
    _write(ds / "sub-01_task-x_meg.res4")
    found = session_recordings(root, root / "sub-01")
    assert [p.name for p in found] == ["sub-01_task-x_meg.ds"]


# ---------------------------------------------------------------------------
# The scans table
# ---------------------------------------------------------------------------


def test_beh_and_motion_get_rows_and_bval_does_not(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    sub = root / "sub-01"
    _write(sub / "beh" / "sub-01_task-x_beh.tsv", "trial\n1\n")
    _write(sub / "motion" / "sub-01_task-x_tracksys-imu_motion.tsv", "1\t2\n")
    _write(sub / "dwi" / "sub-01_dwi.nii.gz")
    _write(sub / "dwi" / "sub-01_dwi.bval", "0 1000\n")
    _write(sub / "dwi" / "sub-01_dwi.bvec", "0 1\n0 0\n0 0\n")
    write_tables(plan_tables(root))
    assert _filenames(sub / "sub-01_scans.tsv") == [
        "beh/sub-01_task-x_beh.tsv", "dwi/sub-01_dwi.nii.gz",
        "motion/sub-01_task-x_tracksys-imu_motion.tsv"]


def test_a_table_written_before_keeps_its_rows_columns_and_values(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    sub = root / "sub-01"
    _write(sub / "anat" / "sub-01_T1w.nii.gz")
    _write(sub / "func" / "sub-01_task-x_bold.nii.gz")
    _write(sub / "func" / "sub-01_task-x_physio.tsv.gz")
    _write(sub / "dwi" / "sub-01_dwi.nii.gz")
    _write(sub / "dwi" / "sub-01_dwi.bval", "0\n")
    table = sub / "sub-01_scans.tsv"
    # A byte-order mark, as MNE-BIDS writes; a column of the user's; a row
    # the user added for a physio file; a .bval row an older version wrote;
    # a row whose file is gone; a blank cell.
    table.write_text(
        "﻿filename\tacq_time\toperator\n"
        "func/sub-01_task-x_bold.nii.gz\t2024-01-02T10:00:00\tAnna\n"
        "func/sub-01_task-x_physio.tsv.gz\t2024-01-02T10:00:00\tAnna\n"
        "dwi/sub-01_dwi.bval\t2024-01-02T10:30:00\t\n"
        "anat/sub-01_gone_T1w.nii.gz\t2024-01-02T09:00:00\tAnna\n"
        "dwi/sub-01_dwi.nii.gz\t\tBen\n",
        encoding="utf-8")
    [edit] = plan_tables(root, sessions=False)
    assert edit.added == ["anat/sub-01_T1w.nii.gz"]
    assert dict(edit.dropped) == {
        "dwi/sub-01_dwi.bval": "part of a recording listed on its own row",
        "anat/sub-01_gone_T1w.nii.gz": "the file is not there"}
    assert edit.filled == 1
    write_tables([edit])
    header, rows = read_table(table)
    assert header == ["filename", "acq_time", "operator"]
    by_name = {r["filename"]: r for r in rows}
    assert by_name["func/sub-01_task-x_bold.nii.gz"]["operator"] == "Anna"
    assert "func/sub-01_task-x_physio.tsv.gz" in by_name, "the user's row stays"
    assert by_name["dwi/sub-01_dwi.nii.gz"]["acq_time"] == "n/a"
    assert by_name["anat/sub-01_T1w.nii.gz"]["operator"] == "n/a"
    assert not table.read_bytes().startswith(b"\xef\xbb\xbf")


def test_acq_time_is_a_date_and_time_or_nothing(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    pet = root / "sub-01" / "pet"
    _write(pet / "sub-01_trc-FDG_pet.nii.gz")
    _write(pet / "sub-01_trc-FDG_pet.json", {"AcquisitionTime": "09:28:23.000000"})
    anat = root / "sub-01" / "anat"
    _write(anat / "sub-01_T1w.nii.gz")
    _write(anat / "sub-01_T1w.json", {"AcquisitionDateTime": "2025-05-26T11:14:30.502500"})
    write_tables(plan_tables(root))
    rows = {r["filename"]: r["acq_time"] for r in read_table(root / "sub-01" / "sub-01_scans.tsv")[1]}
    assert rows == {"anat/sub-01_T1w.nii.gz": "2025-05-26T11:14:30.502500",
                    "pet/sub-01_trc-FDG_pet.nii.gz": "n/a"}, "a time alone is not written"

    # What the conversion knew (the scan's study date and time) fills it.
    (root / "sub-01" / "sub-01_scans.tsv").unlink()
    times = {"sub-01_trc-FDG_pet": iso_datetime("20240301", "092823")}
    write_tables(plan_tables(root, times=times))
    rows = {r["filename"]: r["acq_time"] for r in read_table(root / "sub-01" / "sub-01_scans.tsv")[1]}
    assert rows["pet/sub-01_trc-FDG_pet.nii.gz"] == "2024-03-01T09:28:23"


def test_the_parts_of_one_recording_share_one_time(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    func = root / "sub-01" / "func"
    for echo, second in ((1, "05"), (2, "07"), (3, "09")):
        name = f"sub-01_task-x_echo-{echo}_bold"
        _write(func / f"{name}.nii.gz")
        _write(func / f"{name}.json", {"AcquisitionDateTime": f"2025-01-01T10:00:{second}"})
    write_tables(plan_tables(root))
    times = {r["acq_time"] for r in read_table(root / "sub-01" / "sub-01_scans.tsv")[1]}
    assert times == {"2025-01-01T10:00:05"}


def test_eye_tracking_alone_is_listed_and_both_eyes_share_a_time(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    beh = root / "sub-01" / "beh"
    for eye in ("eye1", "eye2"):
        _write(beh / f"sub-01_task-rest_recording-{eye}_physio.tsv.gz")
        _write(beh / f"sub-01_task-rest_recording-{eye}_physioevents.tsv.gz")
    write_tables(plan_tables(root, times={
        "sub-01_task-rest_recording-eye1_physio": "2022-10-31T06:13:24"}))
    rows = read_table(root / "sub-01" / "sub-01_scans.tsv")[1]
    assert rows == [
        {"filename": "beh/sub-01_task-rest_recording-eye1_physio.tsv.gz",
         "acq_time": "2022-10-31T06:13:24"},
        {"filename": "beh/sub-01_task-rest_recording-eye2_physio.tsv.gz",
         "acq_time": "2022-10-31T06:13:24"}], "one recording, one time; no physioevents rows"


def test_eye_tracking_beside_a_behaviour_file_is_its_companion(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    beh = root / "sub-01" / "beh"
    _write(beh / "sub-01_task-rest_beh.tsv", "trial\n1\n")
    _write(beh / "sub-01_task-rest_recording-eye1_physio.tsv.gz")
    assert [p.name for p in session_recordings(root, root / "sub-01")] == [
        "sub-01_task-rest_beh.tsv"]


def test_iso_datetime_reads_dicom_dates_and_times() -> None:
    assert iso_datetime("20250526", "111430.5025") == "2025-05-26T11:14:30.502500"
    assert iso_datetime("2025-05-26", "11:14:30") == "2025-05-26T11:14:30"
    assert iso_datetime("", "111430") == ""
    assert iso_datetime("20250526", "") == ""


# ---------------------------------------------------------------------------
# The sessions table
# ---------------------------------------------------------------------------


def _two_sessions(root: Path) -> Path:
    for ses, stamp in (("pre", "2025-01-01T09:00:00"), ("post", "2025-02-01T09:00:00")):
        anat = root / "sub-01" / f"ses-{ses}" / "anat"
        _write(anat / f"sub-01_ses-{ses}_T1w.nii.gz")
        _write(anat / f"sub-01_ses-{ses}_T1w.json", {"AcquisitionDateTime": stamp})
        func = root / "sub-01" / f"ses-{ses}" / "func"
        _write(func / f"sub-01_ses-{ses}_task-x_bold.nii.gz")
        _write(func / f"sub-01_ses-{ses}_task-x_bold.json",
               {"AcquisitionDateTime": stamp.replace("09:", "08:")})
    return root


def test_a_subject_with_sessions_gets_a_sessions_table(tmp_path: Path) -> None:
    root = _two_sessions(_dataset(tmp_path / "ds"))
    write_tables(plan_tables(root))
    header, rows = read_table(root / "sub-01" / "sub-01_sessions.tsv")
    assert header == ["session_id", "acq_time"]
    assert rows == [{"session_id": "ses-post", "acq_time": "2025-02-01T08:00:00"},
                    {"session_id": "ses-pre", "acq_time": "2025-01-01T08:00:00"}], \
        "each session's earliest scan"


def test_a_sessions_table_keeps_the_users_columns_and_drops_a_gone_session(
        tmp_path: Path) -> None:
    root = _two_sessions(_dataset(tmp_path / "ds"))
    table = root / "sub-01" / "sub-01_sessions.tsv"
    table.write_text("session_id\tpathology\nses-pre\thealthy\nses-old\thealthy\n")
    [edit] = plan_tables(root, scans=False)
    assert edit.added == ["ses-post"]
    assert edit.dropped == [("ses-old", "the session folder is not there")]
    write_tables([edit])
    header, rows = read_table(table)
    assert header == ["session_id", "pathology"], "the user's table keeps its shape"
    assert rows == [{"session_id": "ses-pre", "pathology": "healthy"},
                    {"session_id": "ses-post", "pathology": "n/a"}]


def test_no_sessions_no_sessions_table(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds")
    _write(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    write_tables(plan_tables(root))
    assert not (root / "sub-01" / "sub-01_sessions.tsv").exists()


# ---------------------------------------------------------------------------
# Through the metadata step, and as one step of the Editor's history
# ---------------------------------------------------------------------------


def test_a_second_run_changes_nothing(tmp_path: Path) -> None:
    root = _two_sessions(_dataset(tmp_path / "ds"))
    run_metadata(root, write_report=False)
    tables = sorted(root.rglob("*_s*s.tsv"))
    before = {p: p.read_bytes() for p in tables}
    assert plan_tables(root) == []
    run_metadata(root, write_report=False)
    assert {p: p.read_bytes() for p in tables} == before


def test_the_tables_can_be_turned_off(tmp_path: Path) -> None:
    root = _two_sessions(_dataset(tmp_path / "ds"))
    run_metadata(root, write_report=False, tables=False)
    assert not list(root.rglob("*_scans.tsv")) and not list(root.rglob("*_sessions.tsv"))


def test_writing_them_is_one_undoable_step(tmp_path: Path) -> None:
    from bidsmgr.project.operations import begin_operation, undo_last

    root = _two_sessions(_dataset(tmp_path / "ds"))
    edits = plan_tables(root)
    with begin_operation(root, "Write scans and sessions tables") as op:
        write_tables(edits, op)
    def tables():
        return [p for p in root.rglob("*_s*s.tsv") if ".bidsmgr" not in p.parts]

    assert len(tables()) == 3
    undo_last(root)
    assert tables() == []


def test_the_validator_finds_nothing_wrong_with_them(tmp_path: Path) -> None:
    from bidsmgr.editor.types import Severity
    from bidsmgr.editor.validator import validate

    root = _two_sessions(_dataset(tmp_path / "ds"))
    _write(root / "sub-01" / "ses-pre" / "beh" / "sub-01_ses-pre_task-x_beh.tsv", "trial\n1\n")
    _write(root / "sub-01" / "ses-pre" / "beh" / "sub-01_ses-pre_task-x_beh.json",
           {"TaskName": "x"})
    write_tables(plan_tables(root))
    report = validate(root)
    tables = [f for f in report.files
              if f.path.name.endswith(("_scans.tsv", "_sessions.tsv"))]
    assert len(tables) == 3, "the validator read both scans tables and the sessions table"
    problems = [f"{f.path.name}: {i.rule_id}" for f in tables for i in f.issues
                if i.severity is Severity.ERR]
    assert problems == []

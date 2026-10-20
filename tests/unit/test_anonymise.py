"""Anonymising a BIDS dataset's metadata.

Every carrier the engine cleans, checked on content it really writes: a
sidecar, the tables, a gzip and NIfTI header, an EDF header, a BrainVision
marker file, a FIF recording, a photograph. What must never change (the
recorded data) is compared byte for byte; what must change consistently (one
participant's dates) is compared across files; and running it twice changes
nothing more.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.anonymise import RECOMMENDED, STRICT, PRESETS, apply_anonymise, plan_anonymise
from bidsmgr.anonymise.carriers import clean_photo, parse_date
from bidsmgr.anonymise.state import STATE_FILE
from bidsmgr.metadata.tables import read_table

RAW_DATE = "2025-05-26T11:14:30.502500"          # a Monday


def _nifti_gz(path: Path, descrip: bytes = b"TE=30;Time=101530.500;phase=1") -> bytes:
    """A small .nii.gz whose gzip header states a name and a time, and whose
    header text states the time of day, as dcm2niix writes them."""
    nib = pytest.importorskip("nibabel")
    img = nib.Nifti1Image(np.arange(4 * 4 * 3, dtype=np.int16).reshape(4, 4, 3), np.eye(4))
    img.header["descrip"] = descrip
    raw = img.to_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh, gzip.GzipFile(filename="original_scan.nii", mode="wb",
                                               fileobj=fh, mtime=1_700_000_000) as gz:
        gz.write(raw)
    return raw


def _edf(path: Path) -> bytes:
    header = bytearray(b" " * 256)
    header[0:8] = b"0       "
    header[8:88] = b"MCH-0234567 F 02-MAY-1951 Haagse_Harry".ljust(80)
    header[88:168] = b"Startdate 02-MAR-2002 EMG561 BK/JOP Sony.".ljust(80)
    header[168:176] = b"02.03.02"
    header[176:184] = b"14.21.33"
    body = bytes(range(256)) * 4
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + body)
    return body


def _jpeg_with_exif() -> bytes:
    exif = b"Exif\x00\x00" + b"2024:03:01 10:00:00 Canon"
    app1 = b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    scan = b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00" + b"\x12\x34\x56" + b"\xff\xd9"
    return b"\xff\xd8" + app0 + app1 + scan


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    ses = root / "sub-01" / "ses-1"
    _nifti_gz(ses / "anat" / "sub-01_ses-1_T1w.nii.gz")
    side = {"AcquisitionDateTime": RAW_DATE, "PatientName": "Doe^Jane",
            "PatientBirthDate": "19510502", "InstitutionName": "Uni Clinic",
            "InstitutionAddress": "Main Street 1, Town", "DeviceSerialNumber": "66001",
            "StationName": "MRC35104", "SeriesInstanceUID": "1.3.12.2.1107.5.2.43.66001.1",
            "ProtocolName": "t1_mprage Jane", "AcquisitionTime": "11:14:30.502500",
            "EchoTime": 0.003}
    (ses / "anat" / "sub-01_ses-1_T1w.json").write_text(json.dumps(side, indent=4))
    func = ses / "func"
    func.mkdir(parents=True)
    _nifti_gz(func / "sub-01_ses-1_task-rest_bold.nii.gz")
    (func / "sub-01_ses-1_task-rest_bold.json").write_text(json.dumps({
        "AcquisitionDateTime": "2025-05-26T11:20:00",
        "SeriesInstanceUID": "1.3.12.2.1107.5.2.43.66001.1", "TaskName": "rest"}))
    (ses / "sub-01_ses-1_scans.tsv").write_text(
        "filename\tacq_time\nanat/sub-01_ses-1_T1w.nii.gz\t2025-05-26T11:14:30.502500\n"
        "func/sub-01_ses-1_task-rest_bold.nii.gz\t2025-05-26T11:20:00\n")
    (root / "sub-01" / "sub-01_sessions.tsv").write_text(
        "session_id\tacq_time\nses-1\t2025-05-26T11:14:30\n")
    (root / "participants.tsv").write_text(
        "participant_id\tgiven_name\tage\tsex\nsub-01\tJane\t93\tF\n")
    (root / "participants.json").write_text(json.dumps(
        {"given_name": {"Description": "first name"}, "age": {"Description": "age"}}))
    (root / "dataset_description.json").write_text(json.dumps(
        {"Name": "d", "BIDSVersion": "1.10.0", "Authors": ["A. Author"]}))
    return root


def _apply(root: Path, preset: str = RECOMMENDED, **kw):
    plan = plan_anonymise(root, PRESETS[preset])
    apply_anonymise(plan, **kw)
    return plan


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


# -- sidecars -----------------------------------------------------------------


def test_a_sidecar_loses_the_person_and_keeps_the_science(ds: Path) -> None:
    _apply(ds)
    side = _json(ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.json")
    assert "PatientName" not in side and "PatientBirthDate" not in side
    assert side["InstitutionAddress"] == "n/a" and side["InstitutionName"] == "Uni Clinic"
    assert side["DeviceSerialNumber"].startswith("device-")
    assert side["StationName"].startswith("station-")
    assert side["SeriesInstanceUID"].startswith("2.25.")
    assert side["EchoTime"] == 0.003, "the science is untouched"
    assert side["AcquisitionTime"] == "11:14:30.502500", "Recommended keeps the time of day"
    assert isinstance(side["DeidentificationMethod"], list), "an array, as the schema types it"
    assert "BIDS Manager" in side["DeidentificationMethod"][0]


def test_dates_move_back_together_and_keep_their_weekday(ds: Path) -> None:
    _apply(ds)
    anat = _json(ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.json")["AcquisitionDateTime"]
    bold = _json(ds / "sub-01/ses-1/func/sub-01_ses-1_task-rest_bold.json")[
        "AcquisitionDateTime"]
    shifted = parse_date(anat)
    assert 1901 <= shifted.year <= 1925
    assert shifted.weekday() == dt.date(2025, 5, 26).weekday()
    assert anat.endswith("T11:14:30.502500") and bold.endswith("T11:20:00")
    scans = {r["filename"]: r["acq_time"]
             for r in read_table(ds / "sub-01/ses-1/sub-01_ses-1_scans.tsv")[1]}
    assert scans["anat/sub-01_ses-1_T1w.nii.gz"] == anat, "the table moves with the sidecar"
    session = read_table(ds / "sub-01/sub-01_sessions.tsv")[1][0]["acq_time"]
    assert session.startswith(anat[:10])


def test_one_uid_gets_one_pseudonym_everywhere(ds: Path) -> None:
    _apply(ds)
    a = _json(ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.json")["SeriesInstanceUID"]
    b = _json(ds / "sub-01/ses-1/func/sub-01_ses-1_task-rest_bold.json")["SeriesInstanceUID"]
    assert a == b


def test_strict_sets_times_to_midnight_and_removes_the_site(ds: Path) -> None:
    _apply(ds, STRICT)
    side = _json(ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.json")
    assert side["AcquisitionDateTime"].endswith("T00:00:00")
    assert "AcquisitionTime" not in side and side["InstitutionName"] == "n/a"
    rows = read_table(ds / "participants.tsv")[1]
    assert rows[0]["age"] == "85", "five-year bins, capped at 89"


# -- tables -------------------------------------------------------------------


def test_participants_lose_their_names_and_ages_are_capped(ds: Path) -> None:
    _apply(ds)
    header, rows = read_table(ds / "participants.tsv")
    assert header == ["participant_id", "age", "sex"]
    assert rows[0]["age"] == "89"
    assert "given_name" not in _json(ds / "participants.json")


# -- binary headers -----------------------------------------------------------


def test_a_gzip_header_states_no_time_or_name_and_the_image_is_identical(ds: Path) -> None:
    path = ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz"
    before = gzip.decompress(path.read_bytes())
    _apply(ds)
    data = path.read_bytes()
    assert data[3] & 0x08 == 0 and struct.unpack("<I", data[4:8])[0] == 0
    assert gzip.decompress(data) == before, "byte for byte"


def test_strict_takes_the_time_out_of_the_nifti_header(ds: Path) -> None:
    path = ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz"
    before = gzip.decompress(path.read_bytes())
    _apply(ds, STRICT)
    after = gzip.decompress(path.read_bytes())
    assert after[148:228].rstrip(b"\x00") == b"TE=30;phase=1"
    assert after[:148] == before[:148] and after[228:] == before[228:], "data identical"


def test_an_edf_header_is_anonymised_and_its_data_kept(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    path = root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.edf"
    body = _edf(path)
    (root / "dataset_description.json").write_text('{"Name": "d", "BIDSVersion": "1.10.0"}')
    _apply(root)
    data = path.read_bytes()
    assert data[8:88].rstrip() == b"X X X X"
    assert data[88:168].rstrip() == b"Startdate X X X X"
    assert data[168:176] == b"01.01.85" and data[176:184] == b"14.21.33"
    assert data[256:] == body


def test_a_brainvision_marker_date_moves_with_the_participant(ds: Path) -> None:
    eeg = ds / "sub-01" / "ses-1" / "eeg"
    eeg.mkdir()
    (eeg / "sub-01_ses-1_task-rest_eeg.vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0,20250526111430502500\n")
    _apply(ds)
    text = (eeg / "sub-01_ses-1_task-rest_eeg.vmrk").read_text()
    anat = _json(ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.json")["AcquisitionDateTime"]
    assert f"Mk1=New Segment,,1,1,0,{anat[:10].replace('-', '')}111430502500" in text


def test_a_fif_recording_through_mne(tmp_path: Path) -> None:
    mne = pytest.importorskip("mne")
    root = tmp_path / "ds"
    meg = root / "sub-01" / "meg"
    meg.mkdir(parents=True)
    info = mne.create_info(["MEG 0113", "MEG 0112"], 200.0, ["grad", "grad"])
    raw = mne.io.RawArray(np.random.default_rng(0).normal(size=(2, 400)) * 1e-12, info)
    raw.set_meas_date(dt.datetime(2022, 2, 9, 10, 8, 29, tzinfo=dt.timezone.utc))
    raw.info["subject_info"] = {"first_name": "Jane", "last_name": "Doe",
                                "birthday": dt.date(1951, 5, 2), "his_id": "MCH0234"}
    raw.info["experimenter"] = "Dr Who"
    path = meg / "sub-01_task-rest_meg.fif"
    raw.save(path, verbose="error")
    data_before = mne.io.read_raw_fif(path, preload=True, verbose="error").get_data()
    (root / "dataset_description.json").write_text('{"Name": "d", "BIDSVersion": "1.10.0"}')
    plan = _apply(root)
    assert {c.what for f in plan.files for c in f.changes} >= {
        "measurement date", "subject_info", "experimenter"}
    after = mne.io.read_raw_fif(path, preload=True, verbose="error")
    assert after.info["meas_date"].year <= 1925
    assert after.info["meas_date"].weekday() == 2, "a Wednesday stays a Wednesday"
    # MNE's own anonymisation writes placeholders rather than blanks.
    assert after.info.get("experimenter") in (None, "", "mne_anonymize")
    subject = after.info.get("subject_info") or {}
    assert subject.get("first_name") in (None, "mne_anonymize")
    assert subject.get("his_id") in (None, "0", 0, "mne_anonymize")
    assert subject.get("birthday") is None or subject["birthday"].year <= 1925
    assert plan_anonymise(root, PRESETS[RECOMMENDED]).files == [], "and stays so"
    np.testing.assert_allclose(after.get_data(), data_before)


def test_a_photo_loses_its_metadata_and_nothing_else() -> None:
    jpeg = _jpeg_with_exif()
    cleaned = clean_photo(jpeg)
    assert b"Exif" not in cleaned and b"Canon" not in cleaned
    assert cleaned.endswith(b"\x12\x34\x56\xff\xd9") and b"JFIF" in cleaned
    assert clean_photo(cleaned) is None, "nothing left to remove"


# -- the whole --------------------------------------------------------------


def test_running_it_again_changes_nothing(ds: Path) -> None:
    _apply(ds)
    before = {p: p.read_bytes() for p in ds.rglob("*") if p.is_file() and ".bidsmgr" not in p.parts}
    again = plan_anonymise(ds, PRESETS[RECOMMENDED])
    assert again.files == [], [(f.path.name, f.changes) for f in again.files]
    apply_anonymise(again)
    assert {p: p.read_bytes() for p in before} == before


def test_a_session_converted_later_moves_with_its_participant(ds: Path) -> None:
    _apply(ds)
    first = parse_date(_json(ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.json")[
        "AcquisitionDateTime"])
    ses2 = ds / "sub-01" / "ses-2" / "anat"
    ses2.mkdir(parents=True)
    (ses2 / "sub-01_ses-2_T1w.json").write_text(json.dumps(
        {"AcquisitionDateTime": "2025-06-02T09:00:00"}))
    _apply(ds)
    second = parse_date(_json(ses2 / "sub-01_ses-2_T1w.json")["AcquisitionDateTime"])
    assert (second - first).days == 7, "the week between them survives"
    assert json.loads((ds / STATE_FILE).read_text())["offsets"]["sub-01"] > 0


def test_it_is_one_undoable_step(ds: Path) -> None:
    from bidsmgr.project.operations import begin_operation, undo_last

    before = {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()}
    plan = plan_anonymise(ds, PRESETS[RECOMMENDED])
    with begin_operation(ds, "Anonymise metadata") as op:
        apply_anonymise(plan, op=op)
    undo_last(ds)
    after = {p: p.read_bytes() for p in ds.rglob("*") if p.is_file()
             and ".bidsmgr" not in p.parts}
    assert after == {p: b for p, b in before.items() if ".bidsmgr" not in p.parts}
    assert not (ds / STATE_FILE).exists(), "the offsets go back with the files"


def test_what_it_cannot_change_is_said_and_free_text_is_flagged(ds: Path) -> None:
    (ds / "sourcedata" / "dicom").mkdir(parents=True)
    eeg = ds / "sub-01" / "ses-1" / "eeg"
    eeg.mkdir()
    (eeg / "sub-01_ses-1_task-rest_eeg.set").write_bytes(b"MATLAB")
    (ds / "sub-Smith").mkdir()
    (ds / "sub-Smith" / "anat").mkdir()
    (ds / "sub-Smith" / "anat" / "sub-Smith_T1w.json").write_text("{}")
    plan = plan_anonymise(ds, PRESETS[RECOMMENDED])
    cannot = dict(plan.cannot)
    assert "sourcedata/" in cannot
    assert "sub-01/ses-1/eeg/sub-01_ses-1_task-rest_eeg.set" in cannot
    reviews = {(p.name, r.what) for p, r in plan.reviews}
    assert ("sub-01_ses-1_T1w.json", "ProtocolName") in reviews
    assert ("sub-Smith", "participant label") in reviews


def test_the_dataset_description_and_readme_are_left_alone(ds: Path) -> None:
    before = (ds / "dataset_description.json").read_bytes()
    _apply(ds)
    assert (ds / "dataset_description.json").read_bytes() == before


def test_a_file_that_cannot_be_rewritten_is_named_and_the_rest_written(
        ds: Path, monkeypatch) -> None:
    import bidsmgr.anonymise.carriers as carriers

    def boom(src, dst):
        raise OSError("the disk said no")

    monkeypatch.setattr(carriers, "write_clean_nifti", boom)
    path = ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz"
    data = path.read_bytes()
    plan = plan_anonymise(ds, PRESETS[STRICT])
    apply_anonymise(plan)
    failed = dict(plan.failed)
    assert "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz" in failed
    assert "the disk said no" in failed["sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz"]
    assert path.read_bytes() == data, "left as it was"
    assert "PatientName" not in _json(ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.json")

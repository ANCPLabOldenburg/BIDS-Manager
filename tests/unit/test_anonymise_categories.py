"""Anonymisation by category and by scope, undone and redone as one step.

Reported 2026-10-20: the tool was "good, but overwhelming": it should say
what it changes by purpose, let each purpose be switched off, and let the
user choose the part of the dataset (subjects, sessions, the files at the
top). Every change now names its category; a category switched off makes
none of its changes and every other's still; a scope changes only its own
files and table rows.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.anonymise import (
    CATEGORIES,
    CATEGORY,
    CUSTOM,
    PRESETS,
    RECOMMENDED,
    STRICT,
    Scope,
    apply_anonymise,
    dataset_units,
    plan_anonymise,
    policy_for,
)
from bidsmgr.anonymise.policy import RECORD
from bidsmgr.metadata.tables import read_table

from .test_anonymise import _edf, _jpeg_with_exif, _nifti_gz

KEYS = [c.key for c in CATEGORIES]


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    """Two subjects, the first with two sessions, every carrier."""
    root = tmp_path / "ds"
    for sub, sessions in (("sub-01", ("ses-1", "ses-2")), ("sub-02", ("ses-1",))):
        for i, ses in enumerate(sessions):
            home = root / sub / ses
            _nifti_gz(home / "anat" / f"{sub}_{ses}_T1w.nii.gz")
            when = f"2025-0{5 + i}-26T11:14:30"
            (home / "anat" / f"{sub}_{ses}_T1w.json").write_text(json.dumps({
                "AcquisitionDateTime": when, "PatientName": f"Doe^{sub}",
                "InstitutionName": "Uni Clinic", "InstitutionAddress": "Main Street 1",
                "DeviceSerialNumber": "66001", "StationName": "MRC35104",
                "SeriesInstanceUID": f"1.3.12.2.1107.5.2.43.66001.{i}",
                "OriginalFile": "/Users/someone/dicom/1.dcm", "EchoTime": 0.003}))
            (home / f"{sub}_{ses}_scans.tsv").write_text(
                f"filename\tacq_time\nanat/{sub}_{ses}_T1w.nii.gz\t{when}\n")
        (root / sub / f"{sub}_sessions.tsv").write_text(
            "session_id\tacq_time\n" + "".join(
                f"{ses}\t2025-0{5 + i}-26T11:14:30\n" for i, ses in enumerate(sessions)))
    _edf(root / "sub-01" / "ses-1" / "eeg" / "sub-01_ses-1_task-x_eeg.edf")
    (root / "sub-01" / "ses-1" / "photo").mkdir()
    (root / "sub-01" / "ses-1" / "photo" / "sub-01_ses-1_photo.jpg").write_bytes(
        _jpeg_with_exif())
    (root / "participants.tsv").write_text(
        "participant_id\tgiven_name\tage\nsub-01\tJane\t93\nsub-02\tJoe\t41\n")
    (root / "participants.json").write_text(json.dumps(
        {"given_name": {"Description": "first name"}, "age": {"Description": "age"}}))
    (root / "dataset_description.json").write_text(json.dumps(
        {"Name": "d", "BIDSVersion": "1.10.0", "Authors": ["A"]}))
    return root


def _changes(plan) -> set[tuple[str, str, str]]:
    return {(plan.rel(f.path), c.what, c.category) for f in plan.files for c in f.changes}


def test_every_change_names_its_category(ds: Path) -> None:
    plan = plan_anonymise(ds, PRESETS[STRICT])
    found = {c.category for f in plan.files for c in f.changes}
    assert found <= set(KEYS) | {RECORD}
    assert found >= {"identity", "ages", "dates", "site", "devices", "uids", "traces",
                     "photos"}, "the fixture exercises every category"


@pytest.mark.parametrize("key", KEYS)
def test_a_category_switched_off_makes_none_of_its_changes_and_takes_no_other(
        ds: Path, key: str) -> None:
    full = _changes(plan_anonymise(ds, PRESETS[RECOMMENDED]))
    off = PRESETS[RECOMMENDED].with_category(key, False)
    assert off.name == CUSTOM and not off.enabled(key)
    left = _changes(plan_anonymise(ds, off))
    assert not any(cat == key for _f, _w, cat in left)
    # Switching "identity" off also keeps the EDF's patient and recording
    # fields; the rest of the EDF is still cleaned.
    assert left == {c for c in full if c[2] != key}


def test_turning_a_category_back_on_finds_the_preset_again() -> None:
    policy = PRESETS[STRICT].with_category("dates", False)
    assert policy.name == CUSTOM
    assert policy.with_category("dates", True, "midnight").name == STRICT
    assert PRESETS[RECOMMENDED].with_category("site", True, "all").name == CUSTOM
    assert policy_for("recommended", keep=("uids", "devices")).categories_on() == [
        k for k in KEYS if k not in ("uids", "devices")]
    with pytest.raises(ValueError):
        PRESETS[RECOMMENDED].with_category("everything", False)


def test_an_option_is_kept_while_its_category_is_off() -> None:
    strict = PRESETS[STRICT]
    assert strict.option("site") == "all" and strict.option("ages") == "bins"
    off = strict.with_category("site", False)
    assert off.institution == "keep" and off.option("site") == "address"
    assert off.with_category("site", True, "all").institution == "all"


def test_a_subject_scope_changes_only_its_files_and_rows(ds: Path) -> None:
    plan = plan_anonymise(ds, PRESETS[RECOMMENDED], scope=Scope(subjects=frozenset({"sub-02"})))
    files = {plan.rel(f.path) for f in plan.files}
    assert not any(f.startswith("sub-01/") for f in files)
    assert any(f.startswith("sub-02/") for f in files)
    apply_anonymise(plan)
    header, rows = read_table(ds / "participants.tsv")
    assert "given_name" in header, "sub-01 still has its value: the column stays"
    by = {r["participant_id"]: r for r in rows}
    assert by["sub-02"]["given_name"] == "n/a" and by["sub-01"]["given_name"] == "Jane"
    assert by["sub-01"]["age"] == "93", "sub-01 is not in the scope"
    side = json.loads((ds / "participants.json").read_text())
    assert "given_name" in side, "the column is still described"
    one = json.loads((ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.json").read_text())
    assert one["PatientName"] == "Doe^sub-01"


def test_a_session_scope_changes_only_its_session(ds: Path) -> None:
    scope = Scope(top=False, subjects=frozenset({"sub-01"}),
                  sessions=frozenset({"sub-01/ses-2"}))
    plan = plan_anonymise(ds, PRESETS[RECOMMENDED], scope=scope)
    files = {plan.rel(f.path) for f in plan.files}
    assert all(f.startswith(("sub-01/ses-2/", "sub-01/sub-01_sessions.tsv")) for f in files)
    apply_anonymise(plan)
    sessions = {r["session_id"]: r["acq_time"]
                for r in read_table(ds / "sub-01/sub-01_sessions.tsv")[1]}
    assert sessions["ses-1"] == "2025-05-26T11:14:30", "ses-1 is not in the scope"
    assert sessions["ses-2"] < "1926"
    assert read_table(ds / "participants.tsv")[1][0]["given_name"] == "Jane", "top left alone"


def test_sessions_done_one_at_a_time_keep_their_interval(ds: Path) -> None:
    """The weeks a participant moves are chosen from ALL their dates and
    remembered, so the rest done later moves with what was done first."""
    def when(ses: str) -> dt.datetime:
        side = json.loads((ds / f"sub-01/{ses}/anat/sub-01_{ses}_T1w.json").read_text())
        return dt.datetime.fromisoformat(side["AcquisitionDateTime"])

    gap = when("ses-2") - when("ses-1")
    for ses in ("ses-2", "ses-1"):
        scope = Scope(subjects=frozenset({"sub-01"}), sessions=frozenset({f"sub-01/{ses}"}))
        apply_anonymise(plan_anonymise(ds, PRESETS[RECOMMENDED], scope=scope))
    assert when("ses-1").year <= 1925 and when("ses-2").year <= 1925
    assert when("ses-2") - when("ses-1") == gap


def test_the_scope_says_itself_and_the_units_count_files(ds: Path) -> None:
    assert Scope().describe() == "the whole dataset"
    assert Scope(subjects=frozenset({"sub-01"})).describe() == "1 subject and the dataset files"
    assert Scope(top=False, subjects=frozenset({"sub-01"}),
                 sessions=frozenset({"sub-01/ses-1"})).describe() == "1 subject (1 session)"
    top, units = dataset_units(ds)
    assert top == 3
    assert [u.subject for u in units] == ["sub-01", "sub-02"]
    assert [s for s, _n in units[0].sessions] == ["ses-1", "ses-2"]
    assert units[0].files == sum(n for _s, n in units[0].sessions) + 1, "+ sessions.tsv"


def test_a_fif_needs_names_and_ids_and_keeps_its_date_when_dates_are_off(
        tmp_path: Path) -> None:
    mne = pytest.importorskip("mne")
    root = tmp_path / "ds"
    meg = root / "sub-01" / "meg"
    meg.mkdir(parents=True)
    info = mne.create_info(["MEG 0113"], 200.0, ["grad"])
    raw = mne.io.RawArray(np.zeros((1, 200)), info)
    raw.set_meas_date(dt.datetime(2022, 2, 9, 10, 8, 29, tzinfo=dt.timezone.utc))
    raw.info["experimenter"] = "Dr Who"
    path = meg / "sub-01_task-rest_meg.fif"
    raw.save(path, verbose="error")
    (root / "dataset_description.json").write_text('{"Name": "d", "BIDSVersion": "1.10.0"}')

    no_identity = plan_anonymise(root, PRESETS[RECOMMENDED].with_category("identity", False))
    assert not no_identity.files
    assert any("Names and IDs" in why for _p, why in no_identity.cannot)

    keep_dates = PRESETS[RECOMMENDED].with_category("dates", False)
    plan = plan_anonymise(root, keep_dates)
    assert {c.what for f in plan.files for c in f.changes} == {"experimenter"}
    apply_anonymise(plan)
    after = mne.io.read_raw_fif(path, verbose="error")
    assert after.info["meas_date"].year == 2022, "dates off: the date stays"
    assert after.info["experimenter"] in (None, "", "mne_anonymize")


def test_an_already_shifted_fif_is_not_moved_to_2000(tmp_path: Path) -> None:
    """MNE moves a date to 2000-01-01 when told no shift (None); a date
    already shifted is moved by 0 days instead."""
    mne = pytest.importorskip("mne")
    root = tmp_path / "ds"
    meg = root / "sub-01" / "meg"
    meg.mkdir(parents=True)
    raw = mne.io.RawArray(np.zeros((1, 200)), mne.create_info(["MEG 0113"], 200.0, ["grad"]))
    raw.set_meas_date(dt.datetime(1915, 2, 3, 10, 8, 29, tzinfo=dt.timezone.utc))
    raw.info["experimenter"] = "Dr Who"
    path = meg / "sub-01_task-rest_meg.fif"
    raw.save(path, verbose="error")
    (root / "dataset_description.json").write_text('{"Name": "d", "BIDSVersion": "1.10.0"}')
    apply_anonymise(plan_anonymise(root, PRESETS[RECOMMENDED]))
    assert mne.io.read_raw_fif(path, verbose="error").info["meas_date"].year == 1915


def test_undone_and_redone_as_one_step(ds: Path) -> None:
    from bidsmgr.project.operations import begin_operation, redo_last, undo_last

    def files() -> dict[Path, bytes]:
        return {p: p.read_bytes() for p in ds.rglob("*")
                if p.is_file() and ".bidsmgr" not in p.parts}

    before = files()
    policy = PRESETS[STRICT].with_category("uids", False)
    with begin_operation(ds, "Anonymise metadata (Custom, 1 subject)") as op:
        apply_anonymise(plan_anonymise(ds, policy, scope=Scope(subjects=frozenset({"sub-01"}))),
                        op=op)
    anonymised = files()
    assert anonymised != before
    undo_last(ds)
    assert files() == before
    redo_last(ds)
    assert files() == anonymised, "Redo makes exactly the same run again"


def test_the_record_says_what_was_done(ds: Path) -> None:
    plan = plan_anonymise(ds, PRESETS[RECOMMENDED])
    records = [c for f in plan.files for c in f.changes if c.category == RECORD]
    assert records and all(c.what == "DeidentificationMethod" for c in records)
    assert CATEGORY["identity"].title == "Names and IDs"


def test_what_was_right_clicked_becomes_the_scope(ds: Path) -> None:
    from bidsmgr.anonymise.scope import scope_of

    one = scope_of(ds, [ds / "sub-01" / "ses-2" / "anat"])
    assert one == Scope(subjects=frozenset({"sub-01"}), sessions=frozenset({"sub-01/ses-2"}))
    both = scope_of(ds, [ds / "sub-01", ds / "sub-02" / "ses-1"])
    assert both.subjects == frozenset({"sub-01", "sub-02"})
    assert both.has_session("sub-01", "ses-2"), "a subject picked whole keeps every session"
    assert both.has_session("sub-02", "ses-1")
    assert scope_of(ds, [ds / "participants.tsv"]).whole

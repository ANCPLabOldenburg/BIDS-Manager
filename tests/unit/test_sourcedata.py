"""Keeping the source data in sourcedata/ (converter/sourcedata), and a
conversion that does it."""

from __future__ import annotations

import hashlib
import os
import tarfile
from pathlib import Path

import pytest

from bidsmgr.converter import sourcedata as sd
from bidsmgr.converter.types import ConvertTask
from tests.fixtures.dicoms import write_mr_series


def _task(files, *, uid="1.2.3", datatype="anat", suffix="T1w", modality="mri",
          session=None, basename="sub-01_T1w", companions=()) -> ConvertTask:
    return ConvertTask(row_id=uid or str(files[0]), series_uid=uid,
                       source_files=tuple(Path(f) for f in files), dataset="ds",
                       bids_root=Path("/tmp/ds"), subject="01", session=session,
                       datatype=datatype, suffix=suffix, modality=modality,
                       basename=basename, companion_files=tuple(companions))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_options() -> None:
    assert not sd.SourcedataOptions().enabled
    opts = sd.SourcedataOptions.parse("archive", "mri, EEG")
    assert opts.enabled and opts.modalities == {"mri", "eeg"}
    assert not sd.SourcedataOptions.parse("copy", "").enabled
    with pytest.raises(ValueError):
        sd.SourcedataOptions.parse("zip")
    with pytest.raises(ValueError):
        sd.SourcedataOptions.parse("copy", "mri,ct")


def test_the_modality_folder_of_each_kind_of_task(tmp_path) -> None:
    f = [tmp_path / "x"]
    assert sd.kind_of(_task(f)) == "mri"
    assert sd.kind_of(_task(f, suffix="physio", datatype="func")) == "physio"
    assert sd.kind_of(_task(f, uid="", datatype="ieeg", modality="ieeg")) == "eeg"
    assert sd.kind_of(_task(f, uid="", datatype="meg", modality="")) == "meg"
    assert sd.kind_of(_task(f, datatype="pet", modality="pet")) == "pet"


def test_a_recording_brings_the_files_that_belong_to_it(tmp_path) -> None:
    vhdr = tmp_path / "rest.vhdr"
    vhdr.write_text("DataFile=rest_data.eeg\nMarkerFile=rest.vmrk\n", encoding="latin-1")
    for name in ("rest_data.eeg", "rest.vmrk", "other.eeg"):
        (tmp_path / name).write_bytes(b"x")
    assert {p.name for p in sd.recording_files(vhdr)} == {"rest.vhdr", "rest_data.eeg",
                                                          "rest.vmrk"}
    s = tmp_path / "task.set"
    s.write_bytes(b"x")
    s.with_suffix(".fdt").write_bytes(b"x")
    assert {p.name for p in sd.recording_files(s)} == {"task.set", "task.fdt"}
    fif = tmp_path / "meg_raw.fif"
    for name in ("meg_raw.fif", "meg_raw-1.fif", "meg_raw-2.fif", "meg_rawx.fif"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in sd.recording_files(fif)] == ["meg_raw.fif", "meg_raw-1.fif",
                                                         "meg_raw-2.fif"]
    ds = tmp_path / "run.ds"
    ds.mkdir()
    assert sd.recording_files(ds) == [ds]


def test_a_dicom_series_is_named_by_its_number_and_description(tmp_path) -> None:
    files = write_mr_series(tmp_path / "raw", "1.2.3.4", description="t1_mprage", n_slices=2)
    items = sd.plan([_task(files, session="pre")], sd.SourcedataOptions("copy"))
    assert [i.destination for i in items] == ["ses-pre/mri/003_t1_mprage"]
    assert [n for _p, n in items[0].files] == [f.name for f in files]
    items = sd.plan([_task(files)], sd.SourcedataOptions("archive"))
    assert items[0].destination == "mri/003_t1_mprage.tar.gz" and items[0].archive


def test_only_the_chosen_modalities_are_kept(tmp_path) -> None:
    files = write_mr_series(tmp_path / "raw", "1.2.3.4", n_slices=1)
    edf = tmp_path / "rest.edf"
    edf.write_bytes(b"0       ")
    tasks = [_task(files), _task([edf], uid="", datatype="eeg", modality="eeg")]
    kinds = [i.kind for i in sd.plan(tasks, sd.SourcedataOptions.parse("copy", "eeg"))]
    assert kinds == ["eeg"]
    assert sd.plan(tasks, sd.SourcedataOptions()) == [], "off by default"


def test_two_series_with_the_same_name_do_not_meet(tmp_path) -> None:
    a = write_mr_series(tmp_path / "a", "1.1", description="bold", n_slices=1)
    b = write_mr_series(tmp_path / "b", "1.2", description="bold", n_slices=1)
    items = sd.plan([_task(a, uid="1.1"), _task(b, uid="1.2")], sd.SourcedataOptions("copy"))
    assert [i.destination for i in items] == ["mri/003_bold", "mri/003_bold_2"]


def test_copied_with_checksums_times_and_a_manifest(tmp_path) -> None:
    files = write_mr_series(tmp_path / "raw", "1.2.3.4", n_slices=3)
    os.utime(files[0], (1_600_000_000, 1_600_000_000))
    items = sd.plan([_task(files)], sd.SourcedataOptions("copy"))
    rows, failed = sd.stage(items, tmp_path / "stage")
    assert not failed and len(rows) == 3
    report = sd.commit(tmp_path / "stage", tmp_path / "sourcedata" / "sub-01", rows,
                       on_existing="skip", backup_root=tmp_path / "bk", stamp="s")
    assert report.added == 3
    target = tmp_path / "sourcedata" / "sub-01" / "mri" / items[0].destination.split("/")[-1]
    copied = target / files[0].name
    assert _sha(copied) == _sha(files[0])
    assert int(copied.stat().st_mtime) == 1_600_000_000, "modification time kept"
    manifest = sd.read_manifest(tmp_path / "sourcedata" / "sub-01" / sd.MANIFEST)
    assert len(manifest) == 3
    row = manifest[f"mri/{target.name}/{files[0].name}"]
    assert row.sha256 == _sha(files[0]) and row.source == str(files[0])


def test_a_second_run_leaves_identical_files_and_follows_on_existing(tmp_path) -> None:
    files = write_mr_series(tmp_path / "raw", "1.2.3.4", n_slices=2)
    items = sd.plan([_task(files)], sd.SourcedataOptions("copy"))
    target = tmp_path / "sourcedata" / "sub-01"

    def run(on_existing: str) -> sd.Report:
        rows, _f = sd.stage(items, tmp_path / f"stage_{on_existing}")
        return sd.commit(tmp_path / f"stage_{on_existing}", target, rows,
                         on_existing=on_existing, backup_root=tmp_path / "bk", stamp="s")

    run("skip")
    assert run("skip").unchanged == 2
    files[0].write_bytes(files[0].read_bytes() + b"changed")
    assert run("skip").kept_existing == 1
    assert run("replace").replaced == 1
    assert list((tmp_path / "bk").rglob(files[0].name)), "the old copy is backed up"
    manifest = sd.read_manifest(target / sd.MANIFEST)
    assert any(r.sha256 == _sha(files[0]) for r in manifest.values())


def test_an_archive_per_series(tmp_path) -> None:
    files = write_mr_series(tmp_path / "raw", "1.2.3.4", description="dwi", n_slices=2)
    items = sd.plan([_task(files)], sd.SourcedataOptions("archive"))
    rows, failed = sd.stage(items, tmp_path / "stage")
    sd.commit(tmp_path / "stage", tmp_path / "out", rows, on_existing="skip",
              backup_root=tmp_path / "bk", stamp="s")
    archive = tmp_path / "out" / "mri" / "003_dwi.tar.gz"
    with tarfile.open(archive) as tar:
        names = sorted(tar.getnames())
        member = tar.extractfile(f"003_dwi/{files[0].name}").read()
    assert names == sorted(f"003_dwi/{f.name}" for f in files)
    assert hashlib.sha256(member).hexdigest() == _sha(files[0])
    manifest = sd.read_manifest(tmp_path / "out" / sd.MANIFEST)
    assert f"mri/003_dwi.tar.gz!003_dwi/{files[0].name}" in manifest
    # Again: the archive is recognised by its members' checksums.
    rows, _f = sd.stage(items, tmp_path / "stage2")
    report = sd.commit(tmp_path / "stage2", tmp_path / "out", rows, on_existing="skip",
                       backup_root=tmp_path / "bk", stamp="s")
    assert report.unchanged == 2 and report.added == 0


def test_a_recording_archive_holds_its_files_at_the_top(tmp_path) -> None:
    edf = tmp_path / "raw" / "rest.edf"
    edf.parent.mkdir()
    edf.write_bytes(b"0       data")
    items = sd.plan([_task([edf], uid="", datatype="eeg", modality="eeg")],
                    sd.SourcedataOptions("archive"))
    rows, _f = sd.stage(items, tmp_path / "stage")
    assert [r.destination for r in rows] == ["eeg/rest.edf.tar.gz!rest.edf"]
    with tarfile.open(tmp_path / "stage" / "eeg" / "rest.edf.tar.gz") as tar:
        assert tar.getnames() == ["rest.edf"]


def test_a_folder_shaped_recording_is_kept_whole(tmp_path) -> None:
    ds = tmp_path / "raw" / "run.ds"
    (ds / "inner").mkdir(parents=True)
    (ds / "run.meg4").write_bytes(b"abc")
    (ds / "inner" / "x.hc").write_bytes(b"def")
    items = sd.plan([_task([ds], uid="", datatype="meg", modality="meg")],
                    sd.SourcedataOptions("copy", frozenset({"meg"})))
    rows, _f = sd.stage(items, tmp_path / "stage")
    assert sorted(r.destination for r in rows) == ["meg/run.ds/inner/x.hc", "meg/run.ds/run.meg4"]


def test_a_file_that_cannot_be_read_is_reported_not_fatal(tmp_path) -> None:
    files = write_mr_series(tmp_path / "raw", "1.2.3.4", n_slices=2)
    items = sd.plan([_task(files)], sd.SourcedataOptions("copy"))
    files[1].unlink()                       # gone (or held by another program)
    rows, failed = sd.stage(items, tmp_path / "stage")
    assert len(rows) == 1 and len(failed) == 1 and files[1].name in failed[0]


def test_pet_blood_data_goes_with_the_run(tmp_path) -> None:
    files = write_mr_series(tmp_path / "raw", "1.2.3.4", n_slices=1)
    blood = tmp_path / "raw" / "blood.bld"
    blood.write_text("t\tv\n", encoding="utf-8")
    items = sd.plan([_task(files, datatype="pet", modality="pet", suffix="pet",
                           companions=[("blood", str(blood)), ("events", str(blood))])],
                    sd.SourcedataOptions("copy"))
    assert [n for _p, n in items[0].files][-1] == "blood.bld"
    assert len(items[0].files) == 2


def test_space_is_checked(tmp_path) -> None:
    assert sd.space_problem(tmp_path, 10) == ""
    problem = sd.space_problem(tmp_path / "not" / "yet", 1 << 62)
    assert "needs" in problem and "free" in problem


# ---------------------------------------------------------------------------
# Through a conversion
# ---------------------------------------------------------------------------


def _scan(raw: Path, tmp_path: Path) -> Path:
    from bidsmgr.cli.scan import run_scan

    inv = tmp_path / "inventory.tsv"
    run_scan(raw, inv, n_jobs=1, dataset="ds")
    return inv


def test_a_conversion_keeps_the_source_data(tmp_path) -> None:
    pytest.importorskip("dcm2niix")
    from bidsmgr.cli.convert import run_convert

    raw = tmp_path / "raw"
    files = write_mr_series(raw / "t1", "1.2.3.4", description="t1_mprage", n_slices=4)
    inv = _scan(raw, tmp_path)
    out = tmp_path / "bids"
    assert run_convert(inv, out, sourcedata="copy") == 0
    subjects = sorted((out / "ds" / "sourcedata").iterdir())
    assert len(subjects) == 1 and subjects[0].name.startswith("sub-")
    kept = sorted((subjects[0] / "mri" / "003_t1_mprage").iterdir())
    assert [p.name for p in kept] == sorted(f.name for f in files)
    assert (subjects[0] / sd.MANIFEST).is_file()
    assert not (out / "ds" / ".tmp_bidsmgr").exists(), "no staging left behind"
    # The validator does not look in sourcedata/.
    from bidsmgr.editor.validator import validate

    report = validate(out / "ds")
    assert not [f for f in report.files if "sourcedata" in Path(f.path).parts]
    assert not [k for k in report.folder_issues if "sourcedata" in k]
    # Converting again adds nothing.
    assert run_convert(inv, out, sourcedata="copy") == 0
    assert len(sd.read_manifest(subjects[0] / sd.MANIFEST)) == len(files)


def test_a_conversion_is_refused_when_the_copies_will_not_fit(tmp_path, monkeypatch) -> None:
    from bidsmgr.cli.convert import run_convert

    raw = tmp_path / "raw"
    write_mr_series(raw / "t1", "1.2.3.4", n_slices=2)
    inv = _scan(raw, tmp_path)
    monkeypatch.setattr(sd, "space_problem", lambda root, needed: "No room.")
    assert run_convert(inv, tmp_path / "bids", sourcedata="archive") == 1
    assert not list((tmp_path / "bids").rglob("*.nii.gz")), "nothing was converted"

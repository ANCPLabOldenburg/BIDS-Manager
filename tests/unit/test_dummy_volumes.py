"""Removing the first volumes of a functional run at conversion.

The image loses its first volumes with every stored value and header field
otherwise kept; the sidecar records it (``NumberOfVolumesDiscardedByUser``)
and its ``VolumeTiming`` follows; the events move with the new first volume;
the run's physiology starts there too; a cell in the table wins over the
default, 0 included; only functional time series are touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from bidsmgr.cli.convert import (
    TaskOptions,
    _build_tasks_for_subject,
    _drop_volumes_for,
)
from bidsmgr.converter.types import ConvertResult, ConvertTask
from bidsmgr.fixups.dummy_volumes import FIELD, applies, drop_leading_volumes


def _bold(folder: Path, volumes: int = 10, tr: float = 0.8) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    data = np.arange(4 * 4 * 3 * volumes, dtype=np.uint16).reshape(4, 4, 3, volumes)
    img = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.5, 1.0]))
    img.header.set_zooms((2.0, 2.0, 2.5, tr))
    img.header.set_xyzt_units("mm", "sec")
    img.header["descrip"] = b"TE=30;Time=102012.133;phase=1"
    path = folder / "sub-01_task-rest_bold.nii.gz"
    nib.save(img, str(path))
    (folder / "sub-01_task-rest_bold.json").write_text(json.dumps(
        {"RepetitionTime": tr, "TaskName": "rest"}))
    return path


def _result(image: Path, n: int, *, datatype="func", suffix="bold") -> ConvertResult:
    task = ConvertTask(
        row_id="r", series_uid="1.2", source_files=(image,), dataset="d",
        bids_root=image.parent, subject="01", datatype=datatype, suffix=suffix,
        basename=image.name.split(".")[0], drop_volumes=n)
    return ConvertResult(task=task, success=True,
                         staged_files=(image, image.with_name(image.name.split(".")[0] + ".json")))


def test_the_first_volumes_go_and_everything_else_stays(tmp_path) -> None:
    image = _bold(tmp_path / "func")
    before = nib.load(str(image))
    raw = np.asanyarray(before.dataobj)
    assert drop_leading_volumes([_result(image, 3)]) == 1
    after = nib.load(str(image))
    assert after.shape == (4, 4, 3, 7)
    assert after.get_data_dtype() == np.uint16
    assert np.array_equal(np.asanyarray(after.dataobj), raw[..., 3:])
    assert np.allclose(after.affine, before.affine)
    assert after.header["descrip"] == before.header["descrip"]
    assert after.header["pixdim"][4] == pytest.approx(0.8)


def test_the_sidecar_records_it(tmp_path) -> None:
    image = _bold(tmp_path / "func")
    side = image.with_name("sub-01_task-rest_bold.json")
    data = json.loads(side.read_text())
    data[FIELD] = 2                       # already discarded by someone
    data["VolumeTiming"] = [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2]
    side.write_text(json.dumps(data))
    drop_leading_volumes([_result(image, 3)])
    data = json.loads(side.read_text())
    assert data[FIELD] == 5
    assert data["VolumeTiming"][:2] == [0.0, 0.8] and len(data["VolumeTiming"]) == 7


def test_the_events_move_with_the_new_first_volume(tmp_path) -> None:
    image = _bold(tmp_path / "func")
    events = image.with_name("sub-01_task-rest_events.tsv")
    events.write_text("onset\tduration\ttrial_type\n1.6\t2\tgo\n10.5\t1\tstop\n")
    drop_leading_volumes([_result(image, 3)])
    rows = [line.split("\t") for line in events.read_text().splitlines()]
    assert rows[0] == ["onset", "duration", "trial_type"]
    # 3 volumes x 0.8 s: an event before the new first volume keeps its
    # negative onset, which the standard allows.
    assert [r[0] for r in rows[1:]] == ["-0.8", "8.1"]
    assert [r[2] for r in rows[1:]] == ["go", "stop"]


def test_a_run_too_short_is_kept_whole(tmp_path, caplog) -> None:
    image = _bold(tmp_path / "func", volumes=3)
    assert drop_leading_volumes([_result(image, 3)]) == 0
    assert nib.load(str(image)).shape[3] == 3
    assert "kept whole" in caplog.text


def test_only_functional_time_series(tmp_path) -> None:
    assert applies("func", "bold") and applies("func", "cbv")
    assert not applies("func", "sbref") and not applies("dwi", "dwi")
    assert not applies("pet", "pet") and not applies("anat", "T1w")
    image = _bold(tmp_path / "dwi")
    assert drop_leading_volumes([_result(image, 3, datatype="dwi", suffix="dwi")]) == 0
    assert nib.load(str(image)).shape[3] == 10


def test_a_cell_wins_over_the_default() -> None:
    row = pd.Series({"drop_volumes": ""})
    assert _drop_volumes_for(row, "func", "bold", 4) == 4
    assert _drop_volumes_for(pd.Series({"drop_volumes": "2"}), "func", "bold", 4) == 2
    assert _drop_volumes_for(pd.Series({"drop_volumes": "0"}), "func", "bold", 4) == 0
    assert _drop_volumes_for(pd.Series({"drop_volumes": "5"}), "anat", "T1w", 4) == 0
    assert _drop_volumes_for(pd.Series({}), "func", "bold", 0) == 0


def test_physiology_follows_its_run(tmp_path) -> None:
    rows = pd.DataFrame([
        {"participant_id": "sub-01", "series_uid": "1.1", "datatype": "func",
         "bids_name": "sub-01_task-rest_bold", "bids_guess_suffix": "bold",
         "drop_volumes": "3", "dataset": "d", "include": "1"},
        {"participant_id": "sub-01", "series_uid": "1.2", "datatype": "func",
         "bids_name": "sub-01_task-rest_physio", "bids_guess_suffix": "physio",
         "drop_volumes": "", "dataset": "d", "include": "1"},
        {"participant_id": "sub-01", "series_uid": "1.3", "datatype": "func",
         "bids_name": "sub-01_task-other_physio", "bids_guess_suffix": "physio",
         "drop_volumes": "", "dataset": "d", "include": "1"},
    ])
    files = {"1.1": [str(tmp_path / "a.dcm")], "1.2": [str(tmp_path / "b.dcm")],
             "1.3": [str(tmp_path / "c.dcm")]}
    tasks = _build_tasks_for_subject(
        rows, tmp_path, files,
        options=TaskOptions(physio_layout="one_interpolated", physio_align=True))
    by_name = {t.basename: t for t in tasks}
    assert by_name["sub-01_task-rest_bold"].drop_volumes == 3
    assert by_name["sub-01_task-rest_physio"].drop_volumes == 3
    assert by_name["sub-01_task-other_physio"].drop_volumes == 0
    assert by_name["sub-01_task-rest_physio"].physio_layout == "one_interpolated"
    assert by_name["sub-01_task-rest_physio"].physio_align is True


def test_the_command_line_hands_it_over(monkeypatch, tmp_path) -> None:
    from bidsmgr.cli import convert

    seen = {}
    monkeypatch.setattr(convert, "run_convert",
                        lambda *a, **k: seen.update(k) or 0)
    (tmp_path / "inv.tsv").write_text("participant_id\n")
    convert.main([str(tmp_path / "inv.tsv"), str(tmp_path / "out"), "--drop-volumes", "4",
                  "--physio-layout", "one-as-recorded", "--physio-align", "--no-log-file"])
    assert seen["drop_volumes"] == 4
    assert seen["physio_layout"] == "one_as_recorded" and seen["physio_align"] is True


def test_an_unknown_physio_layout_is_refused(tmp_path) -> None:
    from bidsmgr.cli.convert import run_convert

    (tmp_path / "inv.tsv").write_text("participant_id\tdataset\n")
    with pytest.raises(ValueError, match="unknown physio layout"):
        run_convert(tmp_path / "inv.tsv", tmp_path / "out", physio_layout="per_channel",
                    write_log=False)


def test_the_provenance_records_what_was_asked() -> None:
    import inspect

    from bidsmgr.cli import convert

    source = inspect.getsource(convert._write_provenance)
    for key in ("drop_volumes", "physio_layout", "physio_align"):
        assert f'"{key}"' in source


# ---------------------------------------------------------------------------
# The run starts later: its times move with it
# ---------------------------------------------------------------------------


def test_the_sidecar_times_move_to_the_first_volume_kept(tmp_path) -> None:
    """BIDS: the run's time is "when the first data point in each run was
    acquired", and after removing 3 volumes of 0.85 s that is 2.55 s later.
    Measured on OL_3844: 10:20:12.1325 had stayed, the image now starting
    at 10:20:14.6825."""
    image = _bold(tmp_path / "func", tr=0.85)
    side = image.with_name("sub-01_task-rest_bold.json")
    data = json.loads(side.read_text())
    data.update(AcquisitionTime="10:20:12.132500",
                AcquisitionDateTime="2025-05-26T23:59:59.000000")
    side.write_text(json.dumps(data))
    drop_leading_volumes([_result(image, 3)])
    data = json.loads(side.read_text())
    assert data["AcquisitionTime"] == "10:20:14.682500"
    # Past midnight, the date follows.
    assert data["AcquisitionDateTime"] == "2025-05-27T00:00:01.550000"


def test_volume_timing_sets_the_offset_for_times_and_events(tmp_path) -> None:
    """Volumes that are not evenly spaced: the image starts at the first
    volume kept, VolumeTiming[n] - VolumeTiming[0] later, whatever the
    header says a volume takes."""
    image = _bold(tmp_path / "func", tr=1.0)
    side = image.with_name("sub-01_task-rest_bold.json")
    side.write_text(json.dumps({"VolumeTiming": [0, 2, 3.5, 7, 9, 12, 14, 16, 19, 21],
                                "AcquisitionTime": "09:00:00.000000", "TaskName": "rest"}))
    events = image.with_name("sub-01_task-rest_events.tsv")
    events.write_text("onset\tduration\n10\t1\n")
    drop_leading_volumes([_result(image, 2)])
    data = json.loads(side.read_text())
    assert data["VolumeTiming"][:2] == [0.0, 3.5]
    assert data["AcquisitionTime"] == "09:00:03.500000"
    assert events.read_text().splitlines()[1].split("\t")[0] == "6.5"


def test_later_times_as_dcm2niix_writes_them() -> None:
    from bidsmgr.fixups.dummy_volumes import later_datetime, later_time

    assert later_time("23:59:59.5", 1.0) == "00:00:00.500000"
    assert later_time("10:17:1", 1.0) == "10:17:02.000000"      # dcm2niix's unpadded seconds
    assert later_time("2025-05-26T10:00:00", 1.0) == ""          # not a time of day
    assert later_time(None, 1.0) == "" and later_time("soon", 1.0) == ""
    assert later_datetime("2025-05-26T10:00:00", 2.55) == "2025-05-26T10:00:02.550000"
    assert later_datetime("26 May", 1.0) == ""


def test_the_scans_table_times_the_first_volume_kept(tmp_path) -> None:
    from bidsmgr.metadata.tables import acq_time

    func = tmp_path / "func"
    func.mkdir()
    image = func / "sub-01_task-rest_bold.nii.gz"
    image.write_bytes(b"")
    side = func / "sub-01_task-rest_bold.json"
    times = {"sub-01_task-rest_bold": "2023-09-14T10:20:12.132500"}
    # Nothing removed: the scan's time, as before.
    side.write_text(json.dumps({"AcquisitionTime": "10:20:12.132500"}))
    assert acq_time(image, times) == "2023-09-14T10:20:12.132500"
    # Removed: the sidecar's moved time on the scan's date.
    side.write_text(json.dumps({"AcquisitionTime": "10:20:14.682500", FIELD: 3}))
    assert acq_time(image, times) == "2023-09-14T10:20:14.682500"
    # A run that crossed midnight is on the next day.
    side.write_text(json.dumps({"AcquisitionTime": "00:00:01.000000", FIELD: 3}))
    assert acq_time(image, {"sub-01_task-rest_bold": "2023-09-14T23:59:58.450000"}) \
        == "2023-09-15T00:00:01.000000"
    # A sidecar that dates itself wins, as before.
    side.write_text(json.dumps({"AcquisitionDateTime": "2023-09-14T10:20:14.682500", FIELD: 3}))
    assert acq_time(image, times) == "2023-09-14T10:20:14.682500"


def test_the_preview_counts_the_removed_volumes() -> None:
    """The BIDS preview is the conversion's own steps: the run's sidecar
    records the volumes removed, its time in the scans table moves."""
    from bidsmgr.classifier.acquisition.features import SeriesFeatures
    from bidsmgr.converter.output_plan import PlanOptions, approximate, plan_output

    rows = [{"participant_id": "sub-01", "session": "", "datatype": "func",
             "bids_name": "sub-01_task-rest_bold", "bids_guess_suffix": "bold",
             "include": "1", "modality": "mri", "series_uid": "u1", "source_file": "",
             "sequence": "task-rest_bold", "acq_time": "102012.1325", "study_date": "20230914",
             "dataset": "ds", "drop_volumes": ""},
            {"participant_id": "sub-01", "session": "", "datatype": "anat",
             "bids_name": "sub-01_T1w", "bids_guess_suffix": "T1w", "include": "1",
             "modality": "mri", "series_uid": "u2", "source_file": "", "sequence": "t1",
             "acq_time": "101000", "study_date": "20230914", "dataset": "ds",
             "drop_volumes": ""}]
    df = pd.DataFrame(rows)
    feats = {"u1": SeriesFeatures(repetition_time=0.85), "u2": SeriesFeatures()}
    options = PlanOptions(drop_volumes=3)
    plan = plan_output(df, features=feats, options=options)
    scans = dict(approximate(df, plan.find("sub-01/sub-01_scans.tsv"), options,
                             features=feats).rows)
    assert scans["func/sub-01_task-rest_bold.nii.gz"] == "2023-09-14T10:20:14.682500"
    assert scans["anat/sub-01_T1w.nii.gz"] == "2023-09-14T10:10:00"
    side = approximate(df, plan.find("sub-01/func/sub-01_task-rest_bold.json"), options,
                       features=feats)
    fields = {key: (value, source) for key, value, source in side.fields}
    assert fields["NumberOfVolumesDiscardedByUser"] == (3, "settings")
    # A cell wins over the default, 0 included.
    df.loc[0, "drop_volumes"] = "0"
    side = approximate(df, plan.find("sub-01/func/sub-01_task-rest_bold.json"), options,
                       features=feats)
    assert "NumberOfVolumesDiscardedByUser" not in {key for key, _v, _s in side.fields}


# ---------------------------------------------------------------------------
# A physiological log follows the run it was recorded with
# ---------------------------------------------------------------------------


def _scan_row(uid, name, sequence, acq_time, *, datatype="func", drop="", study="1.9"):
    return {"participant_id": "sub-01", "session": "", "series_uid": uid, "datatype": datatype,
            "bids_name": name, "bids_guess_suffix": name.rsplit("_", 1)[-1],
            "sequence": sequence, "acq_time": acq_time, "study_instance_uid": study,
            "drop_volumes": drop, "dataset": "d", "include": "1"}


def test_a_log_follows_its_run_by_the_acquisition_whatever_the_names() -> None:
    """Measured on OL_3844: the log's series is its run's description plus
    _PhysioLog, written when the run ends."""
    from bidsmgr.fixups.dummy_volumes import physio_runs

    df = pd.DataFrame([
        _scan_row("1", "sub-01_task-rest_acq-mb4_bold", "task-rest_bold", "102012.13"),
        _scan_row("2", "sub-01_task-rest_physio", "task-rest_bold_PhysioLog", "102137.78"),
        _scan_row("3", "sub-01_task-two_bold", "task-two_bold", "102330.98"),
        _scan_row("4", "sub-01_task-two_physio", "task-two_bold_PhysioLog", "102417.42"),
    ])
    assert physio_runs(df) == {1: 0, 3: 2}


def test_a_protocol_repeated_under_one_name_goes_by_time() -> None:
    from bidsmgr.fixups.dummy_volumes import physio_runs

    df = pd.DataFrame([
        _scan_row("1", "sub-01_task-x_run-1_bold", "bold", "100000"),
        _scan_row("2", "sub-01_task-x_run-1_physio", "bold_PhysioLog", "100130"),
        _scan_row("3", "sub-01_task-x_run-2_bold", "bold", "100200"),
        _scan_row("4", "sub-01_task-x_run-2_physio", "bold_PhysioLog", "100330"),
    ])
    assert physio_runs(df) == {1: 0, 3: 2}


def test_a_run_stored_as_magnitude_and_phase_is_its_magnitude() -> None:
    from bidsmgr.fixups.dummy_volumes import physio_runs

    df = pd.DataFrame([
        _scan_row("1", "sub-01_task-rest_part-phase_bold", "task-rest_bold", "100000"),
        _scan_row("2", "sub-01_task-rest_part-mag_bold", "task-rest_bold", "100000"),
        _scan_row("3", "sub-01_task-rest_physio", "task-rest_bold_PhysioLog", "100130"),
    ])
    assert physio_runs(df) == {2: 1}


def test_without_a_series_name_a_log_follows_the_run_of_its_name() -> None:
    from bidsmgr.fixups.dummy_volumes import physio_runs

    df = pd.DataFrame([
        _scan_row("1", "sub-01_task-rest_bold", "", ""),
        _scan_row("2", "sub-01_task-rest_physio", "", ""),
        _scan_row("3", "sub-01_task-other_physio", "", ""),
        # Another study's run of that description is not this log's run.
        _scan_row("4", "sub-01_task-late_bold", "late_bold", "", study="2.9"),
        _scan_row("5", "sub-01_task-late2_physio", "late_bold_PhysioLog", "", study="1.9"),
    ])
    assert physio_runs(df) == {1: 0}


def test_a_renamed_run_still_carries_its_log(tmp_path) -> None:
    """The case that left the log uncut: the run renamed in the table (an
    acq- added), the log not."""
    df = pd.DataFrame([
        _scan_row("1.1", "sub-01_task-rest_acq-mb4_bold", "task-rest_bold", "102012", drop="3"),
        _scan_row("1.2", "sub-01_task-rest_physio", "task-rest_bold_PhysioLog", "102137"),
    ])
    files = {"1.1": [str(tmp_path / "a.dcm")], "1.2": [str(tmp_path / "b.dcm")]}
    tasks = {t.basename: t for t in _build_tasks_for_subject(df, tmp_path, files)}
    assert tasks["sub-01_task-rest_acq-mb4_bold"].drop_volumes == 3
    assert tasks["sub-01_task-rest_physio"].drop_volumes == 3


def test_a_log_with_no_run_is_said_when_volumes_are_removed(tmp_path, caplog) -> None:
    df = pd.DataFrame([
        _scan_row("1.1", "sub-01_task-rest_bold", "task-rest_bold", "102012", drop="3"),
        _scan_row("1.2", "sub-01_task-other_physio", "task-other_bold_PhysioLog", "102137"),
    ])
    files = {"1.1": [str(tmp_path / "a.dcm")], "1.2": [str(tmp_path / "b.dcm")]}
    tasks = {t.basename: t for t in _build_tasks_for_subject(df, tmp_path, files)}
    assert tasks["sub-01_task-other_physio"].drop_volumes == 0
    assert "sub-01_task-other_physio: no functional run" in caplog.text
    caplog.clear()
    df.loc[0, "drop_volumes"] = "0"
    _build_tasks_for_subject(df, tmp_path, files)
    assert "no functional run" not in caplog.text, "nothing removed, nothing to say"

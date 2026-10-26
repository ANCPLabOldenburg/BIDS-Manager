"""Siemens CMRR physiology: read channel by channel, written as BIDS tables.

Pinned on synthetic logs shaped like the multimodal sample's real ones
(``tests/fixtures/cmrr_physio.py``). What the previous (vendored) reader got
wrong is what these pin first: four ECG leads are four columns, not one
column four times too long; the trigger is a 0/1 column at each volume start,
not a constant 1; a clock correction does not empty half a channel.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.converter.physio_cmrr import (
    TICK,
    Channel,
    CmrrPhysio,
    build_tables,
    parse_log,
    read_cmrr,
    write_tables,
)
from tests.fixtures.cmrr_physio import START, TR_TICKS, physio_dicom, write_logs


@pytest.fixture
def logs(tmp_path: Path) -> list[Path]:
    return write_logs(tmp_path / "logs", seconds=12.0, volumes=10)


def _table(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in gzip.open(path, "rt").read().splitlines()]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_every_ecg_lead_is_its_own_channel(logs) -> None:
    physio = read_cmrr(logs)
    ecg = [ch for ch in physio.channels if ch.kind == "ECG"]
    assert [ch.label for ch in ecg] == ["ecg1", "ecg2", "ecg3", "ecg4"]
    for ch in ecg:
        assert ch.rate == 400
        # 12 s at 400 Hz, not four leads' worth of samples in one.
        assert abs(len(ch.values) / ch.rate - 12.0) < 0.01


def test_a_clock_correction_does_not_empty_a_channel(logs) -> None:
    physio = read_cmrr(logs)
    puls = next(ch for ch in physio.channels if ch.label == "cardiac")
    assert puls.rate == 200
    assert np.isnan(puls.values).sum() == 0
    # Placed by sequence: the correction (one tick back) moves no later sample.
    assert abs(puls.start - (START - 1) * TICK) < 1e-9


def test_a_dropped_sample_is_left_missing(logs) -> None:
    physio = read_cmrr(logs)
    ecg1 = next(ch for ch in physio.channels if ch.label == "ecg1")
    assert np.isnan(ecg1.values).sum() == 1
    assert np.isnan(ecg1.values[100])


def test_volumes_come_from_the_acquisition_log(logs) -> None:
    physio = read_cmrr(logs)
    assert len(physio.volumes) == 10
    first = physio.volumes[0]
    # The volume's earliest slice, not the first row listed.
    assert first.start == pytest.approx((START + 1000) * TICK)
    assert physio.volumes[1].start - first.start == pytest.approx(TR_TICKS * TICK)


def test_a_physio_dicom_is_read_like_its_logs(tmp_path, logs) -> None:
    dcm = physio_dicom(tmp_path / "physio.dcm")
    from_dicom, from_logs = read_cmrr([dcm]), read_cmrr(logs)
    assert [c.label for c in from_dicom.channels] == [c.label for c in from_logs.channels]
    for a, b in zip(from_dicom.channels, from_logs.channels):
        assert np.array_equal(a.values, b.values, equal_nan=True)
    assert [v.start for v in from_dicom.volumes] == [v.start for v in from_logs.volumes]


def test_a_dicom_without_a_log_says_so(tmp_path) -> None:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(tmp_path / "x.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    pydicom.dcmwrite(str(tmp_path / "x.dcm"), ds, enforce_file_format=True)
    with pytest.raises(ValueError, match="no CMRR physiological log"):
        read_cmrr([tmp_path / "x.dcm"])


def test_parse_log_reads_header_and_rows() -> None:
    header, samples, rows = parse_log(
        "LogDataType = PULS\nSampleTime  = 2\n\n  100   PULS  1500  PULS_TRIGGER\n"
        "  102  PULS  1510\nnot a row\n")
    assert header["SampleTime"] == "2"
    assert samples == {"PULS": [(100, 1500.0), (102, 1510.0)]}
    assert rows == []


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_one_table_per_rate_with_the_trigger_last(tmp_path, logs) -> None:
    written = write_tables(read_cmrr(logs), tmp_path / "out" / "sub-01_task-rest")
    names = sorted(p.name for p in written)
    assert names == [
        "sub-01_task-rest_recording-cardiac_physio.json",
        "sub-01_task-rest_recording-cardiac_physio.tsv.gz",
        "sub-01_task-rest_recording-ecg_physio.json",
        "sub-01_task-rest_recording-ecg_physio.tsv.gz",
        "sub-01_task-rest_recording-respiratory_physio.json",
        "sub-01_task-rest_recording-respiratory_physio.tsv.gz",
    ]
    ecg = json.loads((tmp_path / "out/sub-01_task-rest_recording-ecg_physio.json").read_text())
    assert ecg["Columns"] == ["ecg1", "ecg2", "ecg3", "ecg4", "trigger"]
    assert ecg["SamplingFrequency"] == 400
    assert "ECG channel 3" in ecg["ecg3"]["Description"]
    # StartTime: the first sample relative to the first volume.
    assert ecg["StartTime"] == pytest.approx(-(1000 + 1) * TICK, abs=1e-6)
    rows = _table(tmp_path / "out/sub-01_task-rest_recording-ecg_physio.tsv.gz")
    assert len(rows[0]) == 5
    triggers = [r[-1] for r in rows]
    assert set(triggers) == {"0", "1"} and triggers.count("1") == 10
    assert sum(1 for r in rows if r[0] == "n/a") == 1


def test_the_trigger_marks_each_volume_start_on_the_tables_grid(tmp_path, logs) -> None:
    physio = read_cmrr(logs)
    tables, t0 = build_tables(physio)
    for table in tables:
        trig = table.data[:, table.columns.index("trigger")]
        on = np.flatnonzero(trig == 1)
        times = table.start + on / table.rate
        starts = np.array([v.start for v in physio.volumes])
        # The first sample at or after each volume start, within one sample.
        assert np.all(times >= starts - 1e-9)
        assert np.all(times - starts < 1 / table.rate + 1e-9)


def test_one_table_for_the_run_interpolates_and_says_so(tmp_path, logs) -> None:
    written = write_tables(read_cmrr(logs), tmp_path / "out" / "sub-01_task-rest",
                           layout="one_interpolated")
    assert sorted(p.name for p in written) == ["sub-01_task-rest_physio.json",
                                               "sub-01_task-rest_physio.tsv.gz"]
    side = json.loads((tmp_path / "out/sub-01_task-rest_physio.json").read_text())
    assert side["Columns"] == ["cardiac", "respiratory", "ecg1", "ecg2", "ecg3", "ecg4",
                               "trigger"]
    assert side["SamplingFrequency"] == 400
    assert "resampled to 400 Hz" in side["respiratory"]["Description"]
    assert "resampled" not in side["ecg1"]["Description"]
    rows = _table(tmp_path / "out/sub-01_task-rest_physio.tsv.gz")
    assert {len(r) for r in rows} == {7}
    assert [r[-1] for r in rows].count("1") == 10


def test_one_table_as_recorded_keeps_every_sample_where_it_was(tmp_path, logs) -> None:
    """Nothing computed: each sample of a slower channel in the row of its own
    time, n/a between, its values exactly the recorded ones."""
    physio = read_cmrr(logs)
    (table,), _ = build_tables(physio, layout="one_as_recorded")
    assert table.rate == 400 and table.label == ""
    assert table.columns == ["cardiac", "respiratory", "ecg1", "ecg2", "ecg3", "ecg4",
                             "trigger"]
    by_label = {ch.label: ch for ch in physio.channels}
    for label, every in (("respiratory", 8), ("cardiac", 2), ("ecg2", 1)):
        col = table.data[:, table.columns.index(label)]
        rows = np.flatnonzero(~np.isnan(col))
        ch = by_label[label]
        recorded = ch.values[~np.isnan(ch.values)]
        # Every recorded sample is there once, unchanged, and nothing else.
        assert np.array_equal(np.sort(col[rows]), np.sort(recorded))
        assert set(np.diff(rows)) <= {every, 2 * every}       # 2x across a missing one
        # In the row of its own time.
        times = table.start + rows / table.rate
        own = ch.start + np.flatnonzero(~np.isnan(ch.values)) / ch.rate
        assert np.allclose(times, own, atol=1e-9)
    desc = table.descriptions
    assert "row of its own time (one row in 8)" in desc["respiratory"]
    assert "n/a" in desc["respiratory"] and "interpolat" not in desc["respiratory"]
    assert "Recorded at" not in desc["ecg1"]
    assert table.data[:, -1].sum() == 10


def test_a_sample_between_rows_goes_to_the_nearest_and_says_how_far() -> None:
    """Rates that do not share their grid (here a 50 Hz channel one tick after
    a 200 Hz one): the nearest row, and the column says by how much at most."""
    fast = Channel("PULS", "cardiac", "PULS", 200.0, 0.0, np.arange(200.0))
    slow = Channel("RESP", "respiratory", "RESP", 50.0, TICK, np.arange(50.0) + 1000)
    (table,), _ = build_tables(CmrrPhysio(channels=[fast, slow]), layout="one_as_recorded")
    col = table.data[:, table.columns.index("respiratory")]
    assert np.count_nonzero(~np.isnan(col)) == 50       # none lost, none merged
    assert "at most 2.5 ms" in table.descriptions["respiratory"]
    assert "nearest" not in table.descriptions["cardiac"]


def test_aligned_tables_start_at_the_first_volume(tmp_path, logs) -> None:
    physio = read_cmrr(logs)
    tables, t0 = build_tables(physio, align=True)
    end = physio.volumes[-1].finish
    for table in tables:
        assert 0 <= table.start_time(t0) < 1 / table.rate + 1e-9
        last = table.start + (len(table.data) - 1) / table.rate
        assert last <= end + 1e-9 and end - last < 1 / table.rate + 1e-9
        trig = table.data[:, table.columns.index("trigger")]
        assert trig[0] == 1 and trig.sum() == 10


def test_removed_volumes_move_the_start_and_the_trigger(tmp_path, logs) -> None:
    physio = read_cmrr(logs)
    tables, t0 = build_tables(physio, drop_volumes=3)
    assert t0 == pytest.approx(physio.volumes[3].start)
    for table in tables:
        assert table.data[:, -1].sum() == 7
        assert "3 volumes removed" in table.descriptions["trigger"]
    aligned, _ = build_tables(physio, drop_volumes=3, align=True)
    assert all(t.start_time(t0) >= 0 for t in aligned)


def test_more_volumes_removed_than_the_run_has_is_refused(logs) -> None:
    with pytest.raises(ValueError, match="only 10"):
        build_tables(read_cmrr(logs), drop_volumes=10)


def test_an_unknown_layout_is_refused(logs) -> None:
    with pytest.raises(ValueError, match="unknown physio layout"):
        build_tables(read_cmrr(logs), layout="one-per-channel")


def test_without_an_acquisition_log_there_is_no_trigger(tmp_path, logs) -> None:
    physio = read_cmrr([p for p in logs if "Info" not in p.name])
    tables, t0 = build_tables(physio)
    assert all("trigger" not in t.columns for t in tables)
    assert t0 == min(ch.start for ch in physio.channels)


def test_the_tables_are_valid_bids(tmp_path, logs) -> None:
    from bidsmgr.editor.validator import validate

    root = tmp_path / "ds"
    (root / "sub-01/func").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "p", "BIDSVersion": "1.10.0"}))
    for layout in ("by_rate", "one_interpolated", "one_as_recorded"):
        for p in (root / "sub-01/func").iterdir():
            p.unlink()
        write_tables(read_cmrr(logs), root / "sub-01/func/sub-01_task-rest", layout=layout)
        report = validate(root, strict=True)
        errors = [(f.path, i.rule_id, i.message) for f in report.files for i in f.issues
                  if str(i.severity).lower().endswith(("err", "error"))
                  and "physio" in str(f.path)]
        assert errors == [], (layout, errors)


# ---------------------------------------------------------------------------
# A log with no signal: the volume timings alone
# ---------------------------------------------------------------------------


@pytest.fixture
def timing_only(tmp_path: Path) -> CmrrPhysio:
    """What the scanner logs when physiology logging is on and no sensor is
    attached (SIna OL_4925 runs 2 and 3, the MRI guide sample): the
    acquisition log alone."""
    return read_cmrr([p for p in write_logs(tmp_path / "logs", volumes=10)
                      if "Info" in p.name])


def _marks(table) -> np.ndarray:
    return np.flatnonzero(table.data[:, table.columns.index("trigger")] == 1)


def test_a_log_with_only_volume_timings_gives_the_trigger_alone(timing_only) -> None:
    assert timing_only.channels == [] and len(timing_only.volumes) == 10
    tables, t0 = build_tables(timing_only)
    assert len(tables) == 1
    table = tables[0]
    assert table.label == "" and table.columns == ["trigger"]
    assert table.rate == 400.0
    assert table.start_time(t0) == 0.0
    # One row per scanner tick: every volume start is exactly a mark.
    assert list(_marks(table)) == [v * TR_TICKS for v in range(10)]
    end = timing_only.volumes[-1].finish
    assert table.start + (len(table.data) - 1) / table.rate == pytest.approx(end)
    assert set(np.unique(table.data)) == {0.0, 1.0}
    assert "no physiological signal" in table.descriptions["trigger"]


def test_the_layout_does_not_change_a_trigger_alone(timing_only) -> None:
    first, _ = build_tables(timing_only)
    for layout in ("one_interpolated", "one_as_recorded"):
        tables, _ = build_tables(timing_only, layout=layout)
        assert len(tables) == 1 and np.array_equal(tables[0].data, first[0].data)


def test_removed_volumes_move_a_trigger_alone(timing_only) -> None:
    tables, t0 = build_tables(timing_only, drop_volumes=3)
    table = tables[0]
    assert t0 == pytest.approx(timing_only.volumes[3].start)
    # Not cropped: the rows before the first volume kept stay, unmarked,
    # before StartTime 0.
    assert table.start_time(t0) == pytest.approx(-3 * TR_TICKS * TICK)
    assert list(_marks(table)) == [v * TR_TICKS for v in range(3, 10)]
    assert "3 volumes removed" in table.descriptions["trigger"]
    aligned, t0 = build_tables(timing_only, drop_volumes=3, align=True)
    assert aligned[0].start_time(t0) == 0.0
    assert list(_marks(aligned[0])) == [v * TR_TICKS for v in range(7)]


def test_a_trigger_alone_is_written_as_the_runs_only_table(tmp_path, timing_only) -> None:
    from bidsmgr.editor.validator import validate

    root = tmp_path / "ds"
    (root / "sub-01/func").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "p", "BIDSVersion": "1.10.0"}))
    written = write_tables(timing_only, root / "sub-01/func/sub-01_task-rest")
    assert sorted(p.name for p in written) == ["sub-01_task-rest_physio.json",
                                               "sub-01_task-rest_physio.tsv.gz"]
    sidecar = json.loads((root / "sub-01/func/sub-01_task-rest_physio.json").read_text())
    assert sidecar["Columns"] == ["trigger"]
    assert sidecar["SamplingFrequency"] == 400.0 and sidecar["StartTime"] == 0.0
    rows = _table(root / "sub-01/func/sub-01_task-rest_physio.tsv.gz")
    assert rows[0] == ["1"] and {r[0] for r in rows} == {"0", "1"}
    report = validate(root, strict=True)
    errors = [(f.path, i.rule_id) for f in report.files for i in f.issues
              if str(i.severity).lower().endswith(("err", "error")) and "physio" in str(f.path)]
    assert errors == []


def test_a_log_with_neither_signal_nor_timings_is_refused() -> None:
    with pytest.raises(ValueError, match="neither a signal nor the volume timings"):
        build_tables(CmrrPhysio())


# ---------------------------------------------------------------------------
# Channel names
# ---------------------------------------------------------------------------


def test_two_external_inputs_read_as_a_pair(tmp_path) -> None:
    """The scanner names the first input after the kind alone (EXT, then
    EXT2), as on SIna OL_4594."""
    lines = ["UUID        = 08b19a26-bf5b-4d0b-90aa-85b2893c9c75",
             "ScanDate    = 20250821_142638", "LogVersion  = EJA_1",
             "LogDataType = EXT", "SampleTime  = 8", "",
             "ACQ_TIME_TICS  CHANNEL  VALUE  SIGNAL", ""]
    for k in range(200):
        tick = START + 8 * k
        lines += [f"     {tick}     EXT       {k % 2} ", f"     {tick}     EXT2      0 "]
    log = tmp_path / "Physio_EXT.log"
    log.write_text("\n".join(lines) + "\n")
    physio = read_cmrr([log])
    assert [ch.label for ch in physio.channels] == ["external_trigger1", "external_trigger2"]
    tables, _ = build_tables(physio)
    assert len(tables) == 1 and tables[0].columns == ["external_trigger1", "external_trigger2"]


def test_a_volume_on_a_row_is_marked_in_that_row(logs) -> None:
    """Volume times are seconds of the day (~37,000 s here, as on a real
    scanner), so a row index carries float noise of a few 1e-9 rows. The ECG
    table shares the scanner's 2.5 ms ticks: every volume starts exactly on
    one of its rows and must be marked there, not 2.5 ms later (it was, on
    a real run, whenever volumes were removed or the tables cropped)."""
    physio = read_cmrr(logs)
    for drop, align in ((0, False), (3, False), (3, True), (0, True)):
        tables, t0 = build_tables(physio, drop_volumes=drop, align=align)
        ecg = next(t for t in tables if t.label == "ecg")
        marks = _marks(ecg)
        assert marks[0] / ecg.rate + ecg.start_time(t0) == pytest.approx(0.0, abs=1e-9)
        assert set(np.diff(marks)) == {TR_TICKS}
        if align:
            assert ecg.start_time(t0) == 0.0 and marks[0] == 0


def test_a_volume_the_scanner_did_not_finish_is_not_the_runs(tmp_path) -> None:
    """A run stopped by hand (SIna OL_4594: the last of 1,130 volumes has 40
    of its 48 slices): dcm2niix writes that volume apart and the image has
    1,129, so the trigger marks 1,129, never one beyond the image."""
    from tests.fixtures.cmrr_physio import info_log

    log = tmp_path / "Physio_Info.log"
    log.write_text(info_log(10, slices=4, last_slices=2))
    physio = read_cmrr([log])
    assert len(physio.volumes) == 9 and physio.unfinished == 1
    tables, _t0 = build_tables(physio, drop_volumes=2)
    assert len(_marks(tables[0])) == 7
    assert "did not finish" in tables[0].descriptions["trigger"]
    # A volume in the middle missing a line is the image's: it stays.
    text = info_log(10, slices=4).splitlines()
    middle = next(i for i, line in enumerate(text) if line.split()[:2] == ["4", "2"])
    log.write_text("\n".join(text[:middle] + text[middle + 1:]) + "\n")
    assert len(read_cmrr([log]).volumes) == 10

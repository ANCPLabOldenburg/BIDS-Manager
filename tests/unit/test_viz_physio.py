"""A physio recording as a signal: ``bidsmgr.viz.data.physio``.

A ``*_physio.tsv.gz`` is a table of numbers with no header row, and reading
one as a table answers nothing about the shape of the trace. The three facts
a viewer needs are in the sidecar, which is also what distinguishes a
continuous recording from a table of onsets, so that is what decides whether
a viewer is offered at all. Qt-free.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.viz.bids import sidecar_for
from bidsmgr.viz.data.physio import (
    build_combined_raw,
    build_raw,
    guess_channel_type,
    read_columns,
    read_timing,
    related_recordings,
)
from bidsmgr.viz.data.signal import open_physio
from tests.fixtures.signals import write_physio

pytest.importorskip("mne")


@pytest.fixture()
def recording(tmp_path: Path) -> Path:
    """Two channels at 100 Hz, starting before the scanner did."""
    rows = [[math.sin(i / 10.0), 5.0 if i % 25 == 0 else 0.0]
            for i in range(1000)]
    return write_physio(
        tmp_path, "sub-001_task-x_recording-both_physio.tsv.gz", rows,
        {"Columns": ["cardiac", "trigger"], "SamplingFrequency": 100.0,
         "StartTime": -3.5},
    )


@pytest.fixture()
def one_channel(tmp_path: Path) -> Path:
    """One channel, which is what a ``recording-`` split physio file is."""
    rows = [[math.sin(i / 8.0)] for i in range(600)]
    return write_physio(
        tmp_path, "sub-001_task-x_run-01_recording-cardiac_physio.tsv.gz", rows,
        {"Columns": ["cardiac"], "SamplingFrequency": 100.0, "StartTime": 0.0},
    )


@pytest.fixture()
def one_runs_relatives(tmp_path: Path, one_channel: Path) -> Path:
    """The same run's respiratory belt and trigger, beside the cardiac.

    BIDS splits a run's physio by the ``recording`` entity, so this is the
    ordinary shape, not an edge case.
    """
    write_physio(
        tmp_path, "sub-001_task-x_run-01_recording-respiratory_physio.tsv.gz",
        [[math.cos(i / 40.0)] for i in range(300)],
        {"Columns": ["respiratory"], "SamplingFrequency": 50.0, "StartTime": -1.0},
    )
    write_physio(
        tmp_path, "sub-001_task-x_run-01_recording-trigger_physio.tsv.gz",
        [[5.0 if i % 20 == 0 else 0.0] for i in range(600)],
        {"Columns": ["trigger"], "SamplingFrequency": 100.0, "StartTime": 0.0},
    )
    # A different run, which must NOT be pulled in.
    write_physio(
        tmp_path, "sub-001_task-x_run-02_recording-cardiac_physio.tsv.gz",
        [[0.0] for _ in range(10)],
        {"Columns": ["cardiac"], "SamplingFrequency": 100.0, "StartTime": 0.0},
    )
    return one_channel


class TestWhatCountsAsARecording:
    def test_a_recording_is(self, recording):
        assert read_timing(recording) == {
            "columns": ["cardiac", "trigger"],
            "sampling_frequency": 100.0,
            "start_time": -3.5,
            "units": "",
            "physio_type": "",
        }

    def test_a_table_of_onsets_is_not(self, tmp_path):
        """An events table has no sampling frequency, and that is exactly
        how the standard distinguishes the two."""
        folder = tmp_path / "sub-001" / "func"
        folder.mkdir(parents=True)
        events = folder / "sub-001_task-x_events.tsv"
        events.write_text("onset\tduration\ttrial_type\n0.0\t1.0\tgo\n")
        (folder / "sub-001_task-x_events.json").write_text(
            json.dumps({"trial_type": {"Description": "what happened"}}))
        assert read_timing(events) is None

    def test_a_file_with_no_sidecar_is_not(self, tmp_path):
        folder = tmp_path / "sub-001" / "func"
        folder.mkdir(parents=True)
        lonely = folder / "sub-001_task-x_physio.tsv.gz"
        with gzip.open(lonely, "wt") as handle:
            handle.write("1\n2\n")
        assert read_timing(lonely) is None

    @pytest.mark.parametrize("meta", [
        {"Columns": ["a"], "SamplingFrequency": 0},
        {"Columns": ["a"], "SamplingFrequency": -5},
        {"Columns": ["a"], "SamplingFrequency": "fast"},
        {"Columns": [], "SamplingFrequency": 100},
        {"SamplingFrequency": 100},
    ])
    def test_a_sidecar_that_does_not_say_is_not(self, tmp_path, meta):
        path = write_physio(tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0]], meta)
        assert read_timing(path) is None

    def test_a_missing_start_time_is_zero_not_a_refusal(self, tmp_path):
        path = write_physio(tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0]],
                            {"Columns": ["a"], "SamplingFrequency": 50})
        assert read_timing(path)["start_time"] == 0.0

    def test_the_sidecar_is_found_for_both_extensions(self):
        assert sidecar_for(Path("a/b_physio.tsv.gz")).name == "b_physio.json"
        assert sidecar_for(Path("a/b_physio.tsv")).name == "b_physio.json"


class TestMotion:
    """BIDS motion: no header, no ``Columns``; the names are in the run's
    ``_channels.tsv``. It is a signal like any other, so it opens like one."""

    def _motion(self, root: Path, *, with_channels: bool = True) -> Path:
        folder = root / "sub-001" / "motion"
        folder.mkdir(parents=True)
        path = folder / "sub-001_task-walk_tracksys-imu_motion.tsv"
        path.write_text("".join(f"{i * 0.1:g}\t{-i * 0.2:g}\n" for i in range(300)))
        (folder / "sub-001_task-walk_tracksys-imu_motion.json").write_text(
            json.dumps({"SamplingFrequency": 60.0, "TaskName": "walk"}))
        if with_channels:
            (folder / "sub-001_task-walk_tracksys-imu_channels.tsv").write_text(
                "name\tcomponent\ttype\ttracked_point\tunits\n"
                "head_acc_x\tx\tACC\thead\tm/s^2\n"
                "head_acc_y\ty\tACC\thead\tm/s^2\n")
        return path

    def test_the_columns_come_from_the_channels_table(self, tmp_path):
        timing = read_timing(self._motion(tmp_path))
        assert timing["columns"] == ["head_acc_x", "head_acc_y"]
        assert timing["sampling_frequency"] == 60.0

    def test_it_opens_as_a_signal(self, tmp_path):
        src = open_physio(self._motion(tmp_path))
        assert src.ch_names == ["head_acc_x", "head_acc_y"]
        assert src.n_times == 300 and src.sfreq == 60.0

    def test_without_a_channels_table_it_is_not_a_recording(self, tmp_path):
        assert read_timing(self._motion(tmp_path, with_channels=False)) is None


class TestReadingTheWholeFile:
    def test_it_reads_past_the_table_preview(self, tmp_path):
        """The table stops at five thousand rows. A view of the first five
        thousand samples of a long recording would be a picture of its first
        few seconds, drawn as though it were the whole thing."""
        rows = [[float(i)] for i in range(20000)]
        path = write_physio(tmp_path, "sub-001_task-x_physio.tsv.gz", rows,
                            {"Columns": ["x"], "SamplingFrequency": 100.0})
        columns, total, step = read_columns(path)
        assert (total, step, columns[0].size) == (20000, 1, 20000)

    def test_the_first_sample_is_not_eaten_by_a_header(self, recording):
        """The column names are in the sidecar, not the file: read with a
        header row, the first SAMPLE would vanish."""
        columns, total, _step = read_columns(recording)
        assert total == 1000
        assert columns[1][0] == 5.0, "the trigger fired on sample zero"

    def test_an_enormous_file_strides_and_says_by_how_much(self, tmp_path):
        rows = [[float(i)] for i in range(20000)]
        path = write_physio(tmp_path, "sub-001_task-x_physio.tsv.gz", rows,
                            {"Columns": ["x"], "SamplingFrequency": 100.0})
        columns, total, step = read_columns(path, limit=1000)
        assert total == 20000 and step > 1 and columns[0].size <= 1000
        # Point i of the strided column IS sample i * step.
        assert columns[0][3] == pytest.approx(3 * step)

    def test_an_unreadable_file_is_empty_not_an_exception(self, tmp_path):
        assert read_columns(tmp_path / "nope.tsv.gz") == ([], 0, 1)


class TestChannelTyping:
    """What makes the shared viewer USEFUL here rather than merely possible:
    it colours, groups, averages and reads events by MNE channel type."""

    @pytest.mark.parametrize("name,want", [
        ("cardiac", "ecg"), ("ECG", "ecg"), ("pulse", "ecg"),
        ("respiratory", "resp"), ("resp", "resp"), ("breath_belt", "resp"),
        ("trigger", "stim"), ("external_trigger", "stim"), ("TTL", "stim"),
        ("x_coordinate", "eyegaze"), ("pupil_size", "pupil"),
        ("temperature", "temperature"), ("gsr", "gsr"),
    ])
    def test_a_column_name_implies_its_kind(self, name, want):
        assert guess_channel_type(name) == want

    def test_an_unrecognised_column_is_misc_not_a_guess(self):
        assert guess_channel_type("weird_column") == "misc"

    def test_a_substring_match_does_not_fire_on_the_wrong_word(self):
        """``discard`` contains ``card``, and is not a cardiac trace."""
        assert guess_channel_type("discard") == "misc"


class TestBuildingTheRaw:
    def test_channels_rate_and_types_survive(self, recording):
        timing = read_timing(recording)
        columns, _total, step = read_columns(recording)
        raw, _gaps = build_raw(columns, timing, step)
        assert raw.ch_names == ["cardiac", "trigger"]
        assert raw.get_channel_types() == ["ecg", "stim"]
        assert raw.info["sfreq"] == 100.0
        assert raw.n_times == 1000

    def test_a_strided_read_slows_the_rate_to_match(self, recording):
        """A strided recording really IS a slower one, and every frequency
        the viewer computes depends on getting that right."""
        columns, _total, _step = read_columns(recording)
        raw, _gaps = build_raw(columns, read_timing(recording), step=4)
        assert raw.info["sfreq"] == 25.0

    def test_duplicate_column_names_are_disambiguated(self, tmp_path):
        path = write_physio(tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0, 2.0]],
                            {"Columns": ["ecg", "ecg"], "SamplingFrequency": 50})
        columns, _t, step = read_columns(path)
        raw, _gaps = build_raw(columns, read_timing(path), step)
        assert raw.ch_names == ["ecg", "ecg-1"]

    def test_a_gap_is_zero_in_the_array_and_flagged_beside_it(self, tmp_path):
        """MNE cannot carry NaN through a filter or an FFT, so its array is
        finite; a dropped sample is not a zero either, so the truth goes in
        a mask beside it and the canvas puts the NaN back at draw time."""
        path = tmp_path / "sub-001" / "func" / "sub-001_physio.tsv.gz"
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt") as handle:
            handle.write("1\nn/a\n3\n")
        (path.parent / "sub-001_physio.json").write_text(
            json.dumps({"Columns": ["ecg"], "SamplingFrequency": 10}))
        columns, _t, step = read_columns(path)
        assert math.isnan(float(columns[0][1]))
        raw, gaps = build_raw(columns, read_timing(path), step)
        assert np.isfinite(raw.get_data()).all(), "MNE gets a finite array"
        assert gaps.tolist() == [[False, True, False]], "and the truth beside it"

    def test_no_columns_is_refused_not_guessed(self):
        with pytest.raises(ValueError):
            build_raw([], {"columns": [], "sampling_frequency": 100.0})


class TestOneRunsRecordingsTogether:
    """Reading a run's physio one file at a time is reading a three-channel
    recording one channel at a time."""

    def test_the_relatives_of_a_run_are_found(self, one_runs_relatives):
        found = [p.name for p in related_recordings(one_runs_relatives)]
        assert found[0] == one_runs_relatives.name, "itself comes first"
        assert len(found) == 3
        assert all("run-01" in n for n in found), found

    def test_a_lone_recording_has_only_itself(self, recording):
        assert related_recordings(recording) == [recording]

    def test_they_land_on_the_fastest_grid(self, one_runs_relatives):
        raw, gaps, _origin = build_combined_raw(related_recordings(one_runs_relatives))
        assert raw.info["sfreq"] == 100.0, "the 50 Hz belt must not slow it"
        assert raw.ch_names == ["cardiac", "respiratory", "trigger"]
        assert gaps.shape == (3, raw.n_times)

    def test_each_keeps_its_own_start_time(self, one_runs_relatives):
        """The belt started a second before the others. Lining them up on
        sample zero would be inventing a synchronisation."""
        raw, gaps, origin = build_combined_raw(related_recordings(one_runs_relatives))
        assert origin == -1.0, "sample zero of the shared clock is the earliest start"
        assert gaps[0, :100].all(), "cardiac has no data before its start"
        assert not gaps[1, :100].any(), "the belt does"

    def test_a_recording_ends_in_a_gap_not_its_last_value(self, tmp_path):
        """Beyond its own last sample a recording has no samples. Repeating
        the last value to the end of the shared clock drew a flat line across
        the other recordings' remaining minutes."""
        short = write_physio(
            tmp_path, "sub-001_task-x_recording-cardiac_physio.tsv.gz",
            [[7.0] for _ in range(100)],
            {"Columns": ["cardiac"], "SamplingFrequency": 100.0, "StartTime": 0.0})
        write_physio(
            tmp_path, "sub-001_task-x_recording-respiratory_physio.tsv.gz",
            [[1.0] for _ in range(1000)],
            {"Columns": ["respiratory"], "SamplingFrequency": 100.0, "StartTime": 0.0})
        raw, gaps, _origin = build_combined_raw(related_recordings(short))
        assert raw.n_times == 1000
        assert not gaps[0, :100].any()
        assert gaps[0, 100:].all(), "past its end the cardiac is absent"

    def test_the_source_starts_at_the_earliest_start(self, one_runs_relatives):
        src = open_physio(one_runs_relatives, together=True)
        assert src.start_time == -1.0
        assert len(src.paths) == 3
        assert src.kind == "physio"

    def test_a_lone_recording_together_is_just_itself(self, recording):
        src = open_physio(recording, together=True)
        assert src.ch_names == ["cardiac", "trigger"]
        assert src.start_time == -3.5

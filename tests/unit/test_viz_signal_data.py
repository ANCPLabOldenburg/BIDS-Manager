"""Signals as one kind of source: formats, reading, events, filters, spectra.

``bidsmgr.viz.data.formats``, ``.signal`` and ``.events`` and
``bidsmgr.viz.compute.filters``, ``.spectral`` and ``.decimate``. Qt-free:
the same functions serve MEG, EEG and physio in every host.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bidsmgr.viz.bids import full_ext
from bidsmgr.viz.compute import filters, spectral
from bidsmgr.viz.compute.decimate import peak_decimate
from bidsmgr.viz.data.events import (
    Event,
    events_sibling,
    read_events_tsv,
    run_base,
    stim_events,
)
from bidsmgr.viz.data.formats import is_recording_path, kind_of
from bidsmgr.viz.data.signal import (
    SignalSource,
    open_meeg,
    open_physio,
    read_recording,
    resampled,
    summarize,
)
from tests.fixtures.signals import write_fif, write_physio

mne = pytest.importorskip("mne")


def _source(data: np.ndarray, sfreq: float, types=None, *, gaps=None,
            start_time: float = 0.0) -> SignalSource:
    data = np.atleast_2d(np.asarray(data, dtype=float))
    names = [f"c{i}" for i in range(data.shape[0])]
    info = mne.create_info(names, sfreq, types or ["misc"] * len(names), verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    return SignalSource(path=Path("sub-01_task-x_physio.tsv.gz"), raw=raw,
                        kind="physio", gaps=gaps, start_time=start_time)


# ---------------------------------------------------------------------------
# Which files are recordings
# ---------------------------------------------------------------------------


class TestFormats:
    def test_double_extensions_are_kept_whole(self):
        assert full_ext(Path("a.fif.gz")) == ".fif.gz"
        assert full_ext(Path("a.edf")) == ".edf"
        assert full_ext(Path("SUB.EDF")) == ".edf"
        assert full_ext(Path("rec.ds")) == ".ds"

    @pytest.mark.parametrize("name", ["x.fif", "x.fif.gz", "x.edf", "x.vhdr", "x.cnt"])
    def test_recordings_are(self, name):
        assert is_recording_path(Path(name))

    @pytest.mark.parametrize("name", ["x.eeg", "x.vmrk", "x.json", "x.tsv", "x.nii.gz"])
    def test_everything_else_is_not(self, name):
        """A BrainVision recording is opened through its ``.vhdr``."""
        assert not is_recording_path(Path(name))

    def test_a_ctf_folder_is_a_recording(self, tmp_path):
        folder = tmp_path / "sub-01_task-x_meg.ds"
        folder.mkdir()
        assert is_recording_path(folder)

    def test_routing_by_kind(self, tmp_path):
        mrs = tmp_path / "sub-01" / "mrs" / "sub-01_svs.nii.gz"
        bold = tmp_path / "sub-01" / "func" / "sub-01_task-x_bold.nii.gz"
        for p in (mrs, bold):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
        assert kind_of(mrs) == "spectrum"
        assert kind_of(bold) == "volume"
        assert kind_of(Path("x.fif")) == "signal"
        assert kind_of(Path("x.json")) == ""

    def test_a_table_is_physio_only_when_its_sidecar_says_so(self, tmp_path):
        rec = write_physio(tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0]],
                           {"Columns": ["a"], "SamplingFrequency": 10})
        events = rec.with_name("sub-001_task-x_events.tsv")
        events.write_text("onset\tduration\n0\t1\n")
        assert kind_of(rec) == "physio"
        assert kind_of(events) == ""
        assert kind_of(events, sidecar_checked=True) == "physio"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class TestReading:
    def test_read_and_summarize(self, tmp_path):
        path = write_fif(tmp_path)
        meta = summarize(read_recording(path, preload=False), path)
        assert meta["n_channels"] == 5
        assert meta["sfreq"] == 128.0
        assert meta["ch_type_counts"] == {"eeg": 3, "eog": 1, "stim": 1}
        assert meta["available_ch_types"] == ["eeg", "eog", "stim"]
        assert meta["is_ctf"] is False
        assert meta["duration"] > 0
        assert meta["name"] == path.name

    def test_preload_materialises(self, tmp_path):
        raw = read_recording(write_fif(tmp_path), preload=True)
        assert raw.preload is True and raw.n_times > 0

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_an_unreadable_file_raises_rather_than_returning_nothing(self, tmp_path):
        bad = tmp_path / "x.edf"
        bad.write_bytes(b"not an edf")
        with pytest.raises(Exception):
            read_recording(bad, preload=False)

    def test_the_source_knows_its_channels(self, tmp_path):
        src = open_meeg(write_fif(tmp_path))
        assert src.ch_names == ["Fz", "Cz", "Pz", "EOG", "STI"]
        assert src.ch_types == ["eeg", "eeg", "eeg", "eog", "stim"]
        assert src.sfreq == 128.0
        assert src.kind == "meeg"
        assert src.picks_for("eeg") == [0, 1, 2]
        assert src.picks_for("all", ["Cz", "STI"]) == [1, 4]

    def test_a_read_is_clamped_to_the_recording(self, tmp_path):
        src = open_meeg(write_fif(tmp_path))
        assert src.read([0, 1], -50, 10).shape == (2, 10)
        assert src.read([0], src.n_times - 5, src.n_times + 99).shape == (1, 5)
        assert src.read([0], 10, 10).shape == (1, 0)

    def test_a_read_is_a_copy_the_caller_may_write_to(self, tmp_path):
        """The gap blanking writes NaN into it in place; were it a view, the
        recording itself would lose those samples."""
        src = open_meeg(write_fif(tmp_path))
        out = src.read([0, 1, 2], 0, 100)
        assert not np.shares_memory(out, src.raw.get_data(picks=[0], start=0, stop=1))
        out[:] = np.nan
        assert np.isfinite(src.read([0, 1, 2], 0, 100)).all()

    def test_a_table_without_timing_is_refused_in_a_sentence(self, tmp_path):
        table = tmp_path / "sub-01_task-x_events.tsv"
        table.write_text("onset\n0\n")
        with pytest.raises(ValueError, match="SamplingFrequency"):
            open_physio(table)

    def test_physio_is_drawn_in_run_time(self, tmp_path):
        """``StartTime`` used to be read and then dropped, so the physio axis
        started at zero and an events overlay was off by that much."""
        path = write_physio(tmp_path, "sub-001_task-x_physio.tsv.gz",
                            [[float(i % 7)] for i in range(500)],
                            {"Columns": ["cardiac"], "SamplingFrequency": 100.0,
                             "StartTime": -2.5})
        assert open_physio(path).start_time == -2.5


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_the_sibling_is_found_and_parsed(self, tmp_path):
        path = write_fif(tmp_path)
        assert events_sibling(path) is None
        (tmp_path / "sub-01_task-rest_events.tsv").write_text(
            "onset\tduration\ttrial_type\n0.1\t0\tgo\n0.9\t0.5\tstop\n")
        sib = events_sibling(path)
        assert sib is not None
        assert read_events_tsv(sib) == [Event(0.1, 0.0, "go"), Event(0.9, 0.5, "stop")]

    def test_the_label_falls_back_to_value(self, tmp_path):
        p = tmp_path / "e_events.tsv"
        p.write_text("onset\tvalue\n0.0\t1\n1.0\t2\n")
        assert [e.label for e in read_events_tsv(p)] == ["1", "2"]

    def test_a_bad_row_is_skipped_and_na_duration_is_zero(self, tmp_path):
        p = tmp_path / "e_events.tsv"
        p.write_text("onset\tduration\ttrial_type\nn/a\t1\tx\n2.0\tn/a\ty\n")
        assert read_events_tsv(p) == [Event(2.0, 0.0, "y")]

    def test_a_physio_file_finds_its_runs_events(self, tmp_path):
        """The ``recording-`` entity is not part of the run. Keeping it is
        the defect that left a physio file unable to find its events."""
        name = "sub-01_task-a_run-1_recording-cardiac_physio.tsv.gz"
        assert run_base(Path(name)) == "sub-01_task-a_run-1"
        assert run_base(Path("sub-01_task-a_run-1_bold.nii.gz")) == "sub-01_task-a_run-1"
        path = write_physio(tmp_path, name, [[1.0]] * 10,
                            {"Columns": ["cardiac"], "SamplingFrequency": 10})
        (path.parent / "sub-01_task-a_run-1_events.tsv").write_text("onset\n1.0\n")
        assert events_sibling(path) is not None

    def test_stim_events_are_in_recording_time_not_acquisition_time(self, tmp_path):
        """A real MEG file's first sample sits ``first_samp`` samples into the
        acquisition (96 s on the lab's own recording). ``find_events`` counts
        from THERE, so every event used to be drawn that far late."""
        path = write_fif(tmp_path, stim_at=(128, 256), first_samp=1000)
        raw = read_recording(path, preload=True)
        assert raw.first_samp == 1000
        found = stim_events(raw, ["eeg", "eeg", "eeg", "eog", "stim"])
        assert [e.onset for e in found] == pytest.approx([1.0, 2.0])
        assert [e.label for e in found] == ["5", "5"]

    def test_the_curated_table_wins_and_stim_follows_start_time(self):
        src = _source(np.zeros((1, 100)), 10.0, start_time=-3.0)
        src.events_stim = [Event(1.0, 0.0, "5")]
        assert src.events() == [Event(-2.0, 0.0, "5")], "stim moved into run time"
        src.events_tsv = [Event(4.0, 0.0, "go")]
        assert src.events() == [Event(4.0, 0.0, "go")]
        assert src.events("stim") == [Event(-2.0, 0.0, "5")]
        assert src.event_sources() == ["events.tsv", "stim"]


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestResampling:
    def test_the_gap_mask_follows_the_samples(self):
        """The mask used to keep its old shape after a resample, and when the
        new length happened to match it blanked the wrong samples."""
        gaps = np.zeros((1, 1000), dtype=bool)
        gaps[0, 500:600] = True
        src = _source(np.sin(np.arange(1000) / 10.0), 100.0, gaps=gaps)
        out = resampled(src, 50.0)
        assert out.n_times == 500
        assert out.gaps.shape == (1, 500)
        assert out.gaps[0, 250:300].all()
        assert not out.gaps[0, :240].any() and not out.gaps[0, 310:].any()

    def test_what_the_raw_cannot_carry_is_kept(self):
        src = _source(np.zeros((1, 400)), 100.0, start_time=-1.5)
        src.events_tsv = [Event(1.0)]
        out = resampled(src, 25.0)
        assert out.sfreq == 25.0
        assert out.start_time == -1.5
        assert out.events_tsv == [Event(1.0)]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestFilters:
    @pytest.mark.parametrize("spec,word", [
        (filters.FilterSpec(lp=600.0), "low-pass"),
        (filters.FilterSpec(hp=500.0), "high-pass"),
        (filters.FilterSpec(notch=60.0), "notch"),
    ])
    def test_a_cut_off_above_nyquist_is_refused_in_a_sentence(self, spec, word):
        """The old view swallowed MNE's exception and said "Filter applied"."""
        sfreq = 1000.0 if word != "notch" else 50.0
        with pytest.raises(ValueError, match=word):
            filters.validate(spec, sfreq)

    def test_a_high_pass_above_the_low_pass_is_refused(self):
        with pytest.raises(ValueError, match="below the low-pass"):
            filters.validate(filters.FilterSpec(hp=40.0, lp=10.0), 1000.0)

    def test_zeros_mean_off(self):
        spec = filters.validate(filters.FilterSpec(hp=0.0, lp=0.0, notch=0.0), 100.0)
        assert not spec.active
        assert spec.describe() == "no filter"

    def test_the_padding_fits_the_budget_and_says_when_it_cannot(self):
        spec = filters.FilterSpec(hp=0.01)
        pad, note = filters.pad_samples(spec, 100.0, 1)
        assert pad == 33_000 and note is None, "one physio channel affords 330 s"
        pad, note = filters.pad_samples(spec, 1000.0, 300)
        assert pad == filters.SAMPLE_BUDGET // 300
        assert note and "fewer channels" in note

    def test_a_high_pass_the_stretch_cannot_resolve_is_left_out_and_said(self):
        data = np.random.default_rng(0).standard_normal((1, 1000))
        out, messages = filters.apply(data, 100.0, filters.FilterSpec(hp=0.01))
        assert np.array_equal(out, data)
        assert messages and "was not applied" in messages[0]

    def test_a_low_pass_removes_what_it_should(self):
        t = np.arange(2000) / 200.0
        data = np.atleast_2d(np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 60 * t))
        out, messages = filters.apply(data, 200.0, filters.FilterSpec(lp=20.0))
        assert messages == []
        core = slice(400, 1600)
        assert np.std(out[0, core] - np.sin(2 * np.pi * 2 * t[core])) < 0.05

    def test_a_segment_puts_the_gaps_back_after_filtering(self):
        gaps = np.zeros((1, 1000), dtype=bool)
        gaps[0, 300:320] = True
        src = _source(np.sin(np.arange(1000) / 5.0), 100.0, gaps=gaps)
        seg = filters.segment(src, [0], 2.0, 4.0, filters.FilterSpec(lp=10.0))
        assert seg.times[0] == pytest.approx(2.0)
        assert np.isnan(seg.data[0, 100:120]).all()
        assert np.isfinite(seg.data[0, :100]).all()

    def test_an_empty_segment_is_none(self):
        src = _source(np.zeros((1, 100)), 10.0)
        assert filters.segment(src, [0], 50.0, 60.0) is None
        assert filters.segment(src, [], 0.0, 5.0) is None


# ---------------------------------------------------------------------------
# The spectrum
# ---------------------------------------------------------------------------


class TestSpectrum:
    def test_a_sine_peaks_where_it_should(self):
        t = np.arange(4000) / 200.0
        src = _source(np.sin(2 * np.pi * 12.0 * t), 200.0)
        out = spectral.psd(src)
        peak = out["freqs"][int(np.argmax(out["data"][0]))]
        assert peak == pytest.approx(12.0, abs=0.3)
        assert out["filtered"] is False

    def test_a_stim_only_recording_has_a_spectrum(self):
        """``Raw.compute_psd`` drops stim channels whatever the picks, and a
        trigger-only physio file came back as "picks yielded no channels"."""
        data = np.zeros(2000)
        data[::100] = 5.0
        src = _source(data, 100.0, ["stim"])
        out = spectral.psd(src)
        assert out["data"].shape[0] == 1
        assert out["ch_types"] == ["stim"]

    def test_the_result_says_when_it_was_filtered(self):
        t = np.arange(4000) / 200.0
        src = _source(np.sin(2 * np.pi * 12.0 * t), 200.0)
        out = spectral.psd(src, spec=filters.FilterSpec(lp=30.0))
        assert out["filtered"] is True
        assert out["filter"] == "low-pass 30 Hz"

    def test_a_bad_filter_is_refused_not_ignored(self):
        src = _source(np.zeros(1000), 100.0)
        with pytest.raises(ValueError):
            spectral.psd(src, spec=filters.FilterSpec(lp=80.0))

    def test_the_default_view_stops_at_150_hz(self):
        src = _source(np.random.default_rng(1).standard_normal(20000), 2000.0)
        assert spectral.psd(src)["freqs"].max() <= 150.0

    def test_decibels_never_take_the_log_of_zero(self):
        assert np.isfinite(spectral.to_db(np.array([0.0, 1.0]))).all()


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------


class TestDecimation:
    """A pane is a thousand pixels wide and a recording a million samples.
    Taking every n-th sample would be worse than slow: a one-sample trigger
    falls between the samples kept and vanishes."""

    def test_a_short_trace_is_left_alone(self):
        t = np.arange(500) / 100.0
        y = np.sin(t)
        out_t, out_y = peak_decimate(t, y, 1000)
        assert out_t is t and out_y is y

    def test_a_long_trace_comes_down_to_about_two_per_pixel(self):
        t = np.arange(1_000_000) / 1000.0
        _out_t, out_y = peak_decimate(t, np.sin(t), 1000)
        assert 2000 <= out_y.size < 3000

    def test_a_one_sample_spike_survives(self):
        t = np.arange(200_000) / 1000.0
        y = np.sin(t)
        y[137_421] = 9.0
        _out_t, out_y = peak_decimate(t, y, 1000)
        assert np.nanmax(out_y) == pytest.approx(9.0)
        stride = y.size // out_y.size
        assert np.nanmax(y[::stride]) < 2.0, "the naive alternative loses it"

    def test_the_envelope_is_preserved_both_ways(self):
        t = np.arange(200_000) / 1000.0
        y = np.sin(t) * 3.0
        _out_t, out_y = peak_decimate(t, y, 1000)
        assert np.nanmin(out_y) == pytest.approx(np.nanmin(y), abs=0.01)
        assert np.nanmax(out_y) == pytest.approx(np.nanmax(y), abs=0.01)

    def test_a_gap_stays_a_gap(self):
        t = np.arange(100_000) / 1000.0
        y = np.sin(t)
        y[40_000:60_000] = np.nan
        _out_t, out_y = peak_decimate(t, y, 500)
        assert np.isnan(out_y).any()

    def test_one_dropped_sample_does_not_open_a_hole(self):
        t = np.arange(100_000) / 1000.0
        y = np.sin(t)
        y[::500] = np.nan
        _out_t, out_y = peak_decimate(t, y, 500)
        assert np.isfinite(out_y).sum() > out_y.size * 0.5

    def test_the_time_axis_still_reaches_both_ends(self):
        t = np.arange(123_457) / 1000.0
        out_t, _out_y = peak_decimate(t, np.sin(t), 800)
        assert out_t[0] == pytest.approx(t[0])
        assert out_t[-1] == pytest.approx(t[-1])
        assert np.all(np.diff(out_t) >= 0)



# ---------------------------------------------------------------------------
# What is drawn is right: triggers, gaps, one filter for the recording
# ---------------------------------------------------------------------------


class TestFilteringCorrectly:
    def test_a_trigger_channel_is_never_filtered(self):
        sfreq = 500.0
        ttl = np.zeros(5000)
        ttl[1000:1050] = 5.0
        eeg = np.random.default_rng(0).normal(0, 1e-5, 5000)
        src = _source(np.vstack([eeg, ttl]), sfreq, ["eeg", "stim"])
        seg = filters.segment(src, [0, 1], 0.0, 10.0, filters.FilterSpec(hp=1.0, lp=40.0))
        assert np.array_equal(seg.data[1][:5000], ttl[: seg.data.shape[1]]), \
            "a filtered TTL rings; its edges are what it is for"
        assert not np.allclose(seg.data[0], eeg[: seg.data.shape[1]])

    def test_a_gap_does_not_ring_into_its_neighbours(self):
        """The gap holds zeros; a high-pass over a step to zero rang twelve
        times the signal into the samples beside it."""
        sfreq = 200.0
        t = np.arange(4000) / sfreq
        ecg = 2050.0 + 50.0 * np.sin(2 * np.pi * 1.2 * t)
        gaps = np.zeros((1, 4000), dtype=bool)
        gaps[0, 2000:2400] = True
        data = ecg.copy()
        data[2000:2400] = 0.0
        src = _source(data[None, :], sfreq, ["ecg"], gaps=gaps)
        seg = filters.segment(src, [0], 0.0, 20.0, filters.FilterSpec(hp=0.5))
        trace = seg.data[0]
        assert np.isnan(trace[2000:2400]).all(), "the gap is still shown as one"
        near = np.r_[trace[1800:2000], trace[2400:2600]]
        far = trace[400:1200]
        assert np.nanmax(np.abs(near)) < 2.0 * np.nanmax(np.abs(far))

    def test_a_copy_filtered_once_is_what_a_window_shows(self):
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1e-5, (3, 20000))
        src = _source(data, 1000.0, ["eeg", "eeg", "eeg"])
        spec = filters.FilterSpec(hp=1.0, lp=30.0)
        assert filters.fits_in_memory(src, spec)
        copy = filters.filter_recording(src, spec)
        windowed = filters.segment(src, [0, 2], 8.0, 12.0, spec)
        sliced = filters.segment(src, [0, 2], 8.0, 12.0, spec, copy=copy)
        assert np.allclose(windowed.data, sliced.data, atol=1e-7)

    def test_nothing_to_filter_needs_no_copy(self):
        src = _source(np.zeros((1, 100)), 100.0, ["stim"])
        assert not filters.fits_in_memory(src, filters.FilterSpec())


class TestTriggersAnnotationsAndBads:
    def test_an_analog_trigger_counts_its_pulses(self):
        """find_events counted every change of value: 169 for 30 pulses."""
        from bidsmgr.viz.data.events import edge_events

        rng = np.random.default_rng(2)
        v = rng.normal(0, 0.02, 30000)
        for k in range(30):
            v[500 + k * 900: 520 + k * 900] += 5.0
        events = edge_events(v, None, 1000.0, "trigger")
        assert len(events) == 30
        assert events[0].onset == pytest.approx(0.5, abs=0.002)

    def test_annotations_are_events_and_bad_spans_are_marked(self):
        from bidsmgr.viz.data.events import annotation_events

        src = _source(np.zeros((1, 1000)), 100.0, ["eeg"])
        src.raw.set_annotations(mne.Annotations([1.0, 4.0], [0.5, 2.0], ["stim/S1", "BAD_motion"]))
        events = annotation_events(src.raw)
        assert [(e.onset, e.duration, e.label, e.kind) for e in events] == [
            (1.0, 0.5, "stim/S1", ""), (4.0, 2.0, "BAD_motion", "bad")]

    def test_channels_tsv_bads_are_read(self, tmp_path):
        from bidsmgr.viz.data.signal import open_meeg

        meg = tmp_path / "sub-01" / "meg"
        path = write_fif(meg, "sub-01_task-x_meg.fif", sfreq=200.0, seconds=5.0)
        raw = mne.io.read_raw_fif(str(path), verbose=False)
        names = raw.ch_names
        rows = "\n".join(f"{n}\t{'bad' if n == names[1] else 'good'}" for n in names)
        (meg / "sub-01_task-x_channels.tsv").write_text("name\tstatus\n" + rows + "\n")
        src = open_meeg(path)
        assert src.bads == {names[1]} and src.bads_from == "sub-01_task-x_channels.tsv"

    def test_one_scale_per_type_for_the_whole_recording(self):
        rng = np.random.default_rng(4)
        quiet = rng.normal(0, 1e-6, 10000)
        loud = rng.normal(0, 1e-4, 10000)
        src = _source(np.vstack([quiet, loud, quiet]), 1000.0, ["eeg", "ecg", "eeg"])
        scales = src.type_scales()
        assert scales["eeg"] == pytest.approx(3.92e-6, rel=0.1)   # 2.5-97.5 % of N(0, 1)
        assert scales["ecg"] > 50 * scales["eeg"]
        assert src.unit_for("eeg") == ("a.u.", 1.0), "a physio table carries no unit"
        src.kind = "meeg"
        assert src.unit_for("eeg") == ("µV", 1e6) and src.unit_for("mag") == ("fT", 1e15)


class TestWritingBads:
    def test_status_is_written_and_undoable(self, tmp_path):
        from bidsmgr.editor.channels import set_bad_channels
        from bidsmgr.project.operations import read_log

        tsv = tmp_path / "sub-01" / "eeg" / "sub-01_task-x_channels.tsv"
        tsv.parent.mkdir(parents=True)
        tsv.write_text("name\ttype\tunits\nFz\tEEG\tuV\nCz\tEEG\tuV\nEOG\tEOG\tuV\n")
        assert set_bad_channels(tmp_path, tsv, {"Cz"}) == 3
        text = tsv.read_text().splitlines()
        assert text[0] == "name\ttype\tunits\tstatus"
        assert text[2] == "Cz\tEEG\tuV\tbad" and text[1].endswith("\tgood")
        assert read_log(tmp_path)[-1]["label"] == "Mark 1 channel bad in sub-01_task-x_channels.tsv"
        assert set_bad_channels(tmp_path, tsv, {"Cz"}) == 0, "nothing changes, nothing written"


class TestByteOrderMark:
    """mne-bids writes its TSV tables with a UTF-8 byte order mark. Read as
    plain UTF-8, the first column was named "\\ufeffonset" (every event row
    dropped) or "\\ufeffname" (no bad channel ever written)."""

    def test_events_with_a_bom_are_read(self, tmp_path):
        p = tmp_path / "sub-01_task-x_events.tsv"
        p.write_bytes("﻿onset\tduration\ttrial_type\n0.0\t4.2\tT0\n4.2\t4.1\tT2\n"
                      .encode("utf-8"))
        assert [(e.onset, e.label) for e in read_events_tsv(p)] == [(0.0, "T0"), (4.2, "T2")]

    def test_bad_channels_are_written_and_the_bom_kept(self, tmp_path):
        from bidsmgr.editor.channels import set_bad_channels
        from bidsmgr.viz.data.signal import read_bad_channels

        tsv = tmp_path / "sub-01" / "eeg" / "sub-01_task-x_channels.tsv"
        tsv.parent.mkdir(parents=True)
        tsv.write_bytes("﻿name\ttype\tstatus\nFz\tEEG\tgood\nCz\tEEG\tgood\n".encode("utf-8"))
        assert set_bad_channels(tmp_path, tsv, {"Cz"}) == 1
        raw = tsv.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbfname\t"), "the mark stays, once"
        assert raw.count(b"\xef\xbb\xbf") == 1
        assert read_bad_channels(tsv) == {"Cz"}

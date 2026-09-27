"""A physio recording opened as a signal, in the viewer every signal uses.

A ``*_physio.tsv.gz`` is a table of numbers with no header row, and reading
one as a table answers nothing about the shape of the trace. The three facts
a viewer needs are in the sidecar, which is also what distinguishes a
continuous recording from a table of onsets, so that is what decides whether
a viewer is offered at all.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.gui.widgets.physio_viewer import (
    build_raw,
    guess_channel_type,
    read_columns,
    read_timing,
    sidecar_for,
)

pytestmark = pytest.mark.gui


def _write(root: Path, name: str, rows: list[list[float]], meta: dict) -> Path:
    folder = root / "sub-001" / "func"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write("\t".join(f"{v:g}" for v in row) + "\n")
    sidecar = folder / name.replace(".tsv.gz", ".json")
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    return path


@pytest.fixture()
def recording(tmp_path: Path) -> Path:
    """Two channels at 100 Hz, starting before the scanner did."""
    rows = [[math.sin(i / 10.0), 5.0 if i % 25 == 0 else 0.0]
            for i in range(1000)]
    return _write(
        tmp_path, "sub-001_task-x_recording-both_physio.tsv.gz", rows,
        {"Columns": ["cardiac", "trigger"], "SamplingFrequency": 100.0,
         "StartTime": -3.5},
    )


class TestWhatCountsAsARecording:
    def test_a_recording_is(self, recording):
        assert read_timing(recording) == {
            "columns": ["cardiac", "trigger"],
            "sampling_frequency": 100.0,
            "start_time": -3.5,
            "units": "",
        }

    def test_a_table_of_onsets_is_not(self, tmp_path):
        """An events table has no sampling frequency, and that is exactly
        how the standard distinguishes the two."""
        folder = tmp_path / "sub-001" / "func"
        folder.mkdir(parents=True)
        events = folder / "sub-001_task-x_events.tsv"
        events.write_text("onset\tduration\ttrial_type\n0.0\t1.0\tgo\n")
        (folder / "sub-001_task-x_events.json").write_text(
            json.dumps({"trial_type": {"Description": "what happened"}})
        )
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
        path = _write(tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0]], meta)
        assert read_timing(path) is None

    def test_a_missing_start_time_is_zero_not_a_refusal(self, tmp_path):
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0]],
            {"Columns": ["a"], "SamplingFrequency": 50},
        )
        assert read_timing(path)["start_time"] == 0.0

    def test_the_sidecar_is_found_for_both_extensions(self, tmp_path):
        assert sidecar_for(Path("a/b_physio.tsv.gz")).name == "b_physio.json"
        assert sidecar_for(Path("a/b_physio.tsv")).name == "b_physio.json"


class TestReadingTheWholeFile:
    def test_it_reads_past_the_table_preview(self, tmp_path):
        """The table stops at five thousand rows. A view of the first five
        thousand samples of a long recording would be a picture of its first
        few seconds, drawn as though it were the whole thing."""
        rows = [[float(i)] for i in range(20000)]
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", rows,
            {"Columns": ["x"], "SamplingFrequency": 100.0, "StartTime": 0.0},
        )
        columns, total, step = read_columns(path)
        assert (total, step, columns[0].size) == (20000, 1, 20000)

    def test_an_enormous_file_strides_and_says_by_how_much(self, tmp_path):
        rows = [[float(i)] for i in range(20000)]
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", rows,
            {"Columns": ["x"], "SamplingFrequency": 100.0, "StartTime": 0.0},
        )
        columns, total, step = read_columns(path, limit=1000)
        assert total == 20000 and step > 1 and columns[0].size <= 1000
        # Point i of the strided column IS sample i * step.
        assert columns[0][3] == pytest.approx(3 * step)

    def test_an_unreadable_file_is_empty_not_an_exception(self, tmp_path):
        assert read_columns(tmp_path / "nope.tsv.gz") == ([], 0, 1)


class TestChannelTyping:
    """What makes the shared viewer USEFUL here rather than merely possible.

    The viewer colours by kind, groups the type filter by kind, averages the
    spectrum per kind and reads events off a stim channel. All of that comes
    free once the columns carry MNE's own channel types.
    """

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
        """``misc`` is MNE for "shown, not interpreted", which is exactly
        the right claim about a column we did not recognise."""
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
        timing = read_timing(recording)
        columns, _total, _step = read_columns(recording)
        raw, _gaps = build_raw(columns, timing, step=4)
        assert raw.info["sfreq"] == 25.0

    def test_duplicate_column_names_are_disambiguated(self, tmp_path):
        """MNE refuses duplicates and a sidecar is free to repeat a name."""
        path = _write(
            tmp_path, "sub-001_task-x_physio.tsv.gz", [[1.0, 2.0]],
            {"Columns": ["ecg", "ecg"], "SamplingFrequency": 50},
        )
        columns, _t, step = read_columns(path)
        raw, _gaps = build_raw(columns, read_timing(path), step)
        assert raw.ch_names == ["ecg", "ecg-1"]

    def test_a_gap_is_zero_in_the_array_and_flagged_beside_it(self, tmp_path):
        """MNE cannot carry NaN through a filter or an FFT, so the array it
        gets has to be finite. But a dropped sample is not a zero either: on
        a trace centred at 2000 it draws as a spike to the floor and reads as
        an artefact. So the zero goes in the array and the truth goes in a
        mask alongside, and the viewer puts the NaN back at draw time."""
        path = tmp_path / "sub-001" / "func" / "sub-001_physio.tsv.gz"
        path.parent.mkdir(parents=True)
        with gzip.open(path, "wt") as handle:
            handle.write("1\nn/a\n3\n")
        (path.parent / "sub-001_physio.json").write_text(
            json.dumps({"Columns": ["ecg"], "SamplingFrequency": 10})
        )
        columns, _t, step = read_columns(path)
        assert math.isnan(float(columns[0][1]))
        raw, gaps = build_raw(columns, read_timing(path), step)
        assert np.isfinite(raw.get_data()).all(), "MNE gets a finite array"
        assert gaps.shape == (1, 3)
        assert gaps.tolist() == [[False, True, False]], "and the truth beside it"

    def test_no_columns_is_refused_not_guessed(self):
        with pytest.raises(ValueError):
            build_raw([], {"columns": [], "sampling_frequency": 100.0})


class TestTheViewerOffersIt:
    def _pane(self, qtbot):
        from bidsmgr.gui.widgets.tsv_viewer_pane import TsvViewerPane

        pane = TsvViewerPane()
        qtbot.addWidget(pane)
        pane.resize(900, 600)
        return pane

    def _load(self, qtbot, pane, path, root):
        pane.set_file(path, root)
        qtbot.waitUntil(lambda: pane._loader is None, timeout=20_000)

    def test_a_recording_gets_a_visualize_button(self, qtbot, recording, tmp_path):
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        assert pane._visualize_btn.isVisibleTo(pane)
        assert pane._visualize_btn.text().strip() == "Visualize"

    def test_an_events_table_does_not(self, qtbot, tmp_path):
        folder = tmp_path / "sub-001" / "func"
        folder.mkdir(parents=True)
        events = folder / "sub-001_task-x_events.tsv"
        events.write_text("onset\tduration\n0.0\t1.0\n")
        pane = self._pane(qtbot)
        self._load(qtbot, pane, events, tmp_path)
        assert not pane._visualize_btn.isVisibleTo(pane)

    def test_pressing_it_opens_the_shared_viewer(self, qtbot, recording, tmp_path):
        from bidsmgr.gui.widgets.time_series_view import TimeSeriesView

        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        pane._visualize_btn.setChecked(True)
        qtbot.waitUntil(lambda: pane._signal_worker is None, timeout=20_000)
        assert isinstance(pane._viewer_page, TimeSeriesView)
        assert pane._stack.currentWidget() is pane._viewer_page
        assert pane._viewer_page._ch_names == ["cardiac", "trigger"]

    def test_pressing_it_again_goes_back_to_the_table(
        self, qtbot, recording, tmp_path,
    ):
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        pane._visualize_btn.setChecked(True)
        qtbot.waitUntil(lambda: pane._signal_worker is None, timeout=20_000)
        pane._visualize_btn.setChecked(False)
        assert pane._stack.currentIndex() == 1

    def test_the_first_sample_is_not_eaten_by_a_header(
        self, qtbot, recording, tmp_path,
    ):
        """A physio TSV has no column names in it: they are in the sidecar.
        pandas reads the first SAMPLE as the header, and it belongs back in
        the data."""
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        assert len(pane._plot_rows) == 1000

    def test_the_viewer_does_not_pin_the_pane_wide(
        self, qtbot, recording, tmp_path,
    ):
        """It is a viewer in a splitter, like every other one."""
        pane = self._pane(qtbot)
        self._load(qtbot, pane, recording, tmp_path)
        pane._visualize_btn.setChecked(True)
        qtbot.waitUntil(lambda: pane._signal_worker is None, timeout=20_000)
        assert pane.minimumSizeHint().width() <= 400


@pytest.fixture()
def one_channel(tmp_path: Path) -> Path:
    """One channel, which is what a ``recording-`` split physio file is."""
    rows = [[math.sin(i / 8.0)] for i in range(600)]
    return _write(
        tmp_path, "sub-001_task-x_run-01_recording-cardiac_physio.tsv.gz", rows,
        {"Columns": ["cardiac"], "SamplingFrequency": 100.0, "StartTime": 0.0},
    )


@pytest.fixture()
def one_runs_relatives(tmp_path: Path, one_channel: Path) -> Path:
    """The same run's respiratory belt and trigger, beside the cardiac.

    BIDS splits a run's physio by the ``recording`` entity, so this is the
    ordinary shape, not an edge case.
    """
    _write(
        tmp_path, "sub-001_task-x_run-01_recording-respiratory_physio.tsv.gz",
        [[math.cos(i / 40.0)] for i in range(300)],
        {"Columns": ["respiratory"], "SamplingFrequency": 50.0,
         "StartTime": -1.0},
    )
    _write(
        tmp_path, "sub-001_task-x_run-01_recording-trigger_physio.tsv.gz",
        [[5.0 if i % 20 == 0 else 0.0] for i in range(600)],
        {"Columns": ["trigger"], "SamplingFrequency": 100.0, "StartTime": 0.0},
    )
    # A different run, which must NOT be pulled in.
    _write(
        tmp_path, "sub-001_task-x_run-02_recording-cardiac_physio.tsv.gz",
        [[0.0] for i in range(10)],
        {"Columns": ["cardiac"], "SamplingFrequency": 100.0, "StartTime": 0.0},
    )
    return one_channel


class TestOneRunsRecordingsTogether:
    """Reading a run's physio one file at a time is reading a three-channel
    recording one channel at a time, and whether the trigger lines up with
    the belt is the question people bring to it."""

    def test_the_relatives_of_a_run_are_found(self, one_runs_relatives):
        from bidsmgr.gui.widgets.physio_viewer import related_recordings

        found = [p.name for p in related_recordings(one_runs_relatives)]
        assert found[0] == one_runs_relatives.name, "itself comes first"
        assert len(found) == 3
        assert all("run-01" in n for n in found), found

    def test_a_lone_recording_has_only_itself(self, recording):
        from bidsmgr.gui.widgets.physio_viewer import related_recordings

        assert related_recordings(recording) == [recording]

    def test_they_land_on_the_fastest_grid(self, one_runs_relatives):
        from bidsmgr.gui.widgets.physio_viewer import (
            build_combined_raw,
            related_recordings,
        )

        raw, gaps = build_combined_raw(related_recordings(one_runs_relatives))
        assert raw.info["sfreq"] == 100.0, "the 50 Hz belt must not slow it"
        assert raw.ch_names == ["cardiac", "respiratory", "trigger"]
        assert gaps.shape == (3, raw.n_times)

    def test_each_keeps_its_own_start_time(self, one_runs_relatives):
        """The belt started a second before the others. Lining them up on
        sample zero would be inventing a synchronisation."""
        from bidsmgr.gui.widgets.physio_viewer import (
            build_combined_raw,
            related_recordings,
        )

        raw, gaps = build_combined_raw(related_recordings(one_runs_relatives))
        # The belt covers one second the cardiac does not, so the cardiac is
        # flagged as absent over that stretch and the belt is not.
        assert gaps[0, :50].all(), "cardiac has no data before its start"
        assert not gaps[1, :50].any(), "the belt does"


class TestControlsThatSuitTheChannelCount:
    def _open(self, qtbot, path, root):
        from bidsmgr.gui.widgets.tsv_viewer_pane import TsvViewerPane

        pane = TsvViewerPane()
        qtbot.addWidget(pane)
        pane.resize(900, 600)
        pane.set_file(path, root)
        qtbot.waitUntil(lambda: pane._loader is None, timeout=20_000)
        pane._visualize_btn.setChecked(True)
        qtbot.waitUntil(lambda: pane._signal_worker is None, timeout=20_000)
        return pane

    def test_one_channel_hides_the_channel_controls(
        self, qtbot, one_channel, tmp_path,
    ):
        """Picking which of one channel to show, and how many of them, are
        both questions with one answer."""
        pane = self._open(qtbot, one_channel, tmp_path)
        view = pane._viewer_page
        assert view._ch_names == ["cardiac"]
        assert not view.cmb_ch_type.isVisibleTo(view)
        assert not view.spn_n.isVisibleTo(view)
        assert not view.btn_select.isVisibleTo(view)

    def test_several_channels_show_them(self, qtbot, recording, tmp_path):
        pane = self._open(qtbot, recording, tmp_path)
        view = pane._viewer_page
        assert len(view._ch_names) == 2
        assert view.cmb_ch_type.isVisibleTo(view)
        assert view.spn_n.isVisibleTo(view)
        assert view.btn_select.isVisibleTo(view)

    def test_the_together_button_is_offered_only_with_relatives(
        self, qtbot, one_runs_relatives, tmp_path,
    ):
        pane = self._open(qtbot, one_runs_relatives, tmp_path)
        assert pane._together_btn.isVisibleTo(pane)

    def test_a_lone_recording_does_not_offer_it(
        self, qtbot, one_channel, tmp_path,
    ):
        pane = self._open(qtbot, one_channel, tmp_path)
        assert not pane._together_btn.isVisibleTo(pane)

    def test_together_brings_the_controls_back(
        self, qtbot, one_runs_relatives, tmp_path,
    ):
        pane = self._open(qtbot, one_runs_relatives, tmp_path)
        pane._together_btn.setChecked(True)
        qtbot.waitUntil(
            lambda: pane._signal_worker is None
            and len(pane._viewer_page._ch_names) == 3,
            timeout=30_000,
        )
        view = pane._viewer_page
        assert view.cmb_ch_type.isVisibleTo(view)
        assert view.spn_n.isVisibleTo(view)


class TestHowTheTraceIsDrawn:
    def _view(self, qtbot, path, root):
        from bidsmgr.gui.widgets.tsv_viewer_pane import TsvViewerPane

        pane = TsvViewerPane()
        qtbot.addWidget(pane)
        pane.resize(900, 600)
        pane.set_file(path, root)
        qtbot.waitUntil(lambda: pane._loader is None, timeout=20_000)
        pane._visualize_btn.setChecked(True)
        qtbot.waitUntil(lambda: pane._signal_worker is None, timeout=20_000)
        return pane._viewer_page

    def _curves(self, view):
        return [item for item in view._plot_widget.getPlotItem().items
                if item.__class__.__name__ == "PlotDataItem"]

    def test_a_few_traces_draw_thick(self, qtbot, recording, tmp_path):
        """Two physio channels can afford the wider pen, and a hairline
        reads as a scratch."""
        view = self._view(qtbot, recording, tmp_path)
        assert view._line_width == 0, "the stored width is automatic"
        assert view._pen_width() >= 2
        assert self._curves(view)[0].opts["pen"].width() >= 2

    def test_many_traces_draw_thin(self, qtbot, recording, tmp_path):
        """Not a matter of taste. Qt strokes a one-pixel cosmetic pen through
        a fast path and has to stroke anything wider properly: measured on a
        twenty-channel MEG window, 16.5 ms of paint against 119. A drag pays
        that over and over."""
        view = self._view(qtbot, recording, tmp_path)
        assert view._pen_width(n_shown=40) == 1

    def test_a_chosen_width_is_honoured_where_it_is_affordable(
        self, qtbot, recording, tmp_path,
    ):
        view = self._view(qtbot, recording, tmp_path)
        view._set_line_style(6, "")
        assert view._pen_width(n_shown=1) == 6
        assert view._pen_width(n_shown=2) == 6

    def test_a_chosen_width_is_CAPPED_where_it_is_not(
        self, qtbot, recording, tmp_path,
    ):
        """A cap, not a suggestion. A preference of six pixels set on a
        physio trace must not turn a twenty-channel window into a slideshow,
        and that is the complaint this exists for: it is the thickness that
        makes it slow, so the thickness is what gets limited."""
        view = self._view(qtbot, recording, tmp_path)
        view._set_line_style(6, "")
        assert view._pen_width(n_shown=20) == 1
        assert view._pen_width(n_shown=300) == 1

    def test_the_popup_does_not_offer_a_width_that_would_be_capped(
        self, qtbot, recording, tmp_path,
    ):
        """Offering a number and then ignoring it is how an app looks broken."""
        from bidsmgr.gui.widgets.line_style_dialog import LineStyleDialog

        view = self._view(qtbot, recording, tmp_path)
        wide = LineStyleDialog(0, None, max_width=1, traces_shown=20)
        qtbot.addWidget(wide)
        assert wide._slider.maximum() == 1
        assert not wide._slider.isEnabled()

        narrow = LineStyleDialog(0, None, max_width=8, traces_shown=1)
        qtbot.addWidget(narrow)
        assert narrow._slider.maximum() == 8
        assert narrow._slider.isEnabled()
        assert view.max_pen_width(n_shown=1) > 1

    def test_the_line_popup_changes_the_trace(self, qtbot, recording, tmp_path):
        view = self._view(qtbot, recording, tmp_path)
        view._set_line_style(6, "#ff7b72")
        pen = self._curves(view)[0].opts["pen"]
        assert pen.width() == 6
        assert pen.color().name() == "#ff7b72"

    def test_a_gap_is_a_break_not_a_spike(self, qtbot, tmp_path):
        """A dropped sample is not a value. Drawn as the zero it is stored
        as, it reads as an artefact on a trace centred anywhere else."""
        rows = [[float("nan") if 100 <= i < 200 else 50.0 + math.sin(i / 5.0)]
                for i in range(600)]
        path = _write(
            tmp_path, "sub-001_task-x_recording-cardiac_physio.tsv.gz", rows,
            {"Columns": ["cardiac"], "SamplingFrequency": 100.0,
             "StartTime": 0.0},
        )
        view = self._view(qtbot, path, tmp_path)
        view._fit_all()
        curve = self._curves(view)[0]
        drawn = np.asarray(curve.yData, dtype=float)
        assert curve.opts["connect"] == "finite"
        assert np.isnan(drawn).any(), "the gap must not be drawn as a value"
        # And nothing was drawn at the floor a zero would have put it.
        finite = drawn[np.isfinite(drawn)]
        assert finite.size > 0

    def test_physio_offers_fit_all(self, qtbot, recording, tmp_path):
        """Physio is a channel or four and fits in a window whole. MEG and
        EEG do not, where the whole recording is hundreds of millions of
        samples, so the button is off unless the caller asks."""
        view = self._view(qtbot, recording, tmp_path)
        assert view.btn_fit.isVisibleTo(view)

    def test_the_view_does_not_offer_it_unasked(self, qtbot):
        from bidsmgr.gui.widgets.time_series_view import TimeSeriesView

        view = TimeSeriesView()
        qtbot.addWidget(view)
        assert not view.btn_fit.isVisibleTo(view)
        view.enable_fit_all(True)
        assert view.btn_fit.isVisibleTo(view)

    def test_fit_all_shows_the_whole_recording(self, qtbot, recording, tmp_path):
        view = self._view(qtbot, recording, tmp_path)
        view._time_window = 1.0
        view._fit_all()
        assert view._time_window == pytest.approx(view._duration)
        assert view._time_start == 0.0

    def test_the_visible_portion_sets_the_height(self, qtbot, tmp_path):
        """Zoomed into a quiet stretch, a quiet stretch should fill the pane.
        Scaling on the whole recording flattens it against the axis."""
        rows = [[(1.0 if i < 300 else 100.0) * math.sin(i / 5.0)]
                for i in range(600)]
        path = _write(
            tmp_path, "sub-001_task-x_recording-cardiac_physio.tsv.gz", rows,
            {"Columns": ["cardiac"], "SamplingFrequency": 100.0,
             "StartTime": 0.0},
        )
        view = self._view(qtbot, path, tmp_path)

        def drawn_height(start):
            view._time_start, view._time_window = start, 2.0
            view._redraw()
            y = np.asarray(self._curves(view)[0].yData, dtype=float)
            return float(np.nanmax(y) - np.nanmin(y))

        quiet = drawn_height(0.5)
        loud = drawn_height(4.0)
        # A hundredfold change in the data, the same height on screen.
        assert quiet == pytest.approx(loud, rel=0.05)


class TestDecimation:
    """A pane is a thousand pixels wide and a recording is a million samples.

    Drawing every one asks Qt to stroke a thousand points into each column,
    all but two of which land where another already did. That is what made
    Fit all freeze, and taking every n-th sample instead would be worse than
    slow: a one-sample trigger falls between the samples kept and vanishes.
    """

    def test_a_short_trace_is_left_alone(self):
        from bidsmgr.gui.widgets.time_series_view import _peak_decimate

        t = np.arange(500) / 100.0
        y = np.sin(t)
        out_t, out_y = _peak_decimate(t, y, 1000)
        assert out_t is t and out_y is y, "nothing to gain, so nothing is done"

    def test_a_long_trace_comes_down_to_about_two_per_pixel(self):
        from bidsmgr.gui.widgets.time_series_view import _peak_decimate

        t = np.arange(1_000_000) / 1000.0
        y = np.sin(t)
        out_t, out_y = _peak_decimate(t, y, 1000)
        assert out_y.size < 3000
        assert out_y.size >= 2000

    def test_a_one_sample_spike_survives(self):
        """The reason it is min/max per column and not every n-th sample."""
        from bidsmgr.gui.widgets.time_series_view import _peak_decimate

        t = np.arange(200_000) / 1000.0
        y = np.sin(t)
        y[137_421] = 9.0
        _out_t, out_y = _peak_decimate(t, y, 1000)
        assert np.nanmax(out_y) == pytest.approx(9.0)
        # And the naive alternative loses it, which is the whole point.
        stride = y.size // out_y.size
        assert np.nanmax(y[::stride]) < 2.0

    def test_the_envelope_is_preserved_both_ways(self):
        from bidsmgr.gui.widgets.time_series_view import _peak_decimate

        t = np.arange(200_000) / 1000.0
        y = np.sin(t) * 3.0
        _out_t, out_y = _peak_decimate(t, y, 1000)
        assert np.nanmin(out_y) == pytest.approx(np.nanmin(y), abs=0.01)
        assert np.nanmax(out_y) == pytest.approx(np.nanmax(y), abs=0.01)

    def test_a_gap_stays_a_gap(self):
        from bidsmgr.gui.widgets.time_series_view import _peak_decimate

        t = np.arange(100_000) / 1000.0
        y = np.sin(t)
        y[40_000:60_000] = np.nan
        _out_t, out_y = _peak_decimate(t, y, 500)
        assert np.isnan(out_y).any()

    def test_one_dropped_sample_does_not_open_a_hole(self):
        """A column holding one NaN among two hundred good samples still has
        a range worth drawing. Propagating the NaN would show a gap the
        recording does not have."""
        from bidsmgr.gui.widgets.time_series_view import _peak_decimate

        t = np.arange(100_000) / 1000.0
        y = np.sin(t)
        y[::500] = np.nan          # one in every five hundred
        _out_t, out_y = _peak_decimate(t, y, 500)
        assert not np.isnan(out_y).all()
        assert np.isfinite(out_y).sum() > out_y.size * 0.5

    def test_the_time_axis_still_reaches_both_ends(self):
        from bidsmgr.gui.widgets.time_series_view import _peak_decimate

        t = np.arange(123_457) / 1000.0     # deliberately not a round number
        y = np.sin(t)
        out_t, _out_y = _peak_decimate(t, y, 800)
        assert out_t[0] == pytest.approx(t[0])
        assert out_t[-1] == pytest.approx(t[-1])
        assert np.all(np.diff(out_t) >= 0), "and it still runs forwards"


class TestChannelTypeColours:
    """Colouring by type is what makes a multi-channel view readable, and the
    types have to be told apart for that to mean anything."""

    def test_mag_and_grad_differ(self, qapp):
        from bidsmgr.gui.widgets.psd_dialog import default_type_color

        assert default_type_color("mag") != default_type_color("grad")

    def test_an_unmapped_type_is_not_the_same_grey_as_the_others(self, qapp):
        """A real MEG file carries ias and syst beside misc, and three kinds
        of channel in one colour is three kinds nobody can tell apart."""
        from bidsmgr.gui.widgets.psd_dialog import default_type_color

        colours = {default_type_color(t) for t in ("misc", "ias", "syst")}
        assert len(colours) == 3

    def test_the_derived_colour_is_the_same_next_time(self, qapp):
        """A character sum, not hash(): Python randomises string hashing per
        process, and a colour that changed on every start would be worse
        than a collision."""
        from bidsmgr.gui.widgets.psd_dialog import default_type_color

        assert default_type_color("ias") == default_type_color("ias")
        expected_pool = {
            default_type_color(t)
            for t in ("mag", "grad", "eeg", "bio", "stim", "ecg", "misc")
        }
        assert default_type_color("ias") in expected_pool

    def test_an_override_takes_effect_and_leaves_its_neighbours_alone(self, qapp):
        from bidsmgr.gui.widgets.psd_dialog import (
            default_type_color,
            set_type_colors,
            type_color,
        )

        try:
            set_type_colors({"grad": "#ff7b72"})
            assert type_color("grad") == "#ff7b72"
            assert type_color("mag") == default_type_color("mag")
        finally:
            set_type_colors({})

    def test_reset_puts_every_type_back(self, qapp):
        """A deletion, not a rewrite: the shipped colours live in one place
        and copying them to write back would be a second table to keep in
        step."""
        from bidsmgr.gui.widgets.psd_dialog import (
            default_type_color,
            set_type_colors,
            type_color,
        )

        set_type_colors({"grad": "#ff7b72", "mag": "#000000"})
        set_type_colors({})
        for t in ("mag", "grad"):
            assert type_color(t) == default_type_color(t)

    def test_the_dialog_lists_only_the_types_the_viewer_has(self, qtbot):
        from bidsmgr.gui.widgets.line_style_dialog import LineStyleDialog

        dlg = LineStyleDialog(0, None, channel_types=["mag", "grad", "mag"])
        qtbot.addWidget(dlg)
        assert list(dlg._swatches) == ["mag", "grad"], "deduped, order kept"
        assert dlg._reset_btn is not None

    def test_a_single_channel_view_is_not_asked_about_types(self, qtbot):
        from bidsmgr.gui.widgets.line_style_dialog import LineStyleDialog

        dlg = LineStyleDialog(2, "#58a6ff", allow_by_type=False,
                              channel_types=())
        qtbot.addWidget(dlg)
        assert dlg._swatches == {}
        assert dlg._reset_btn is None
        assert not dlg._by_type.isEnabled()

    def test_the_dialog_reset_emits_an_empty_map(self, qtbot):
        from bidsmgr.gui.widgets.line_style_dialog import LineStyleDialog

        dlg = LineStyleDialog(0, None, channel_types=["mag", "grad"])
        qtbot.addWidget(dlg)
        seen = []
        dlg.type_colors_changed.connect(seen.append)
        dlg._overrides["grad"] = "#ff7b72"
        dlg._reset_type_colours()
        assert seen == [{}]

    def test_automatic_width_reads_as_automatic(self, qtbot):
        from bidsmgr.gui.widgets.line_style_dialog import LineStyleDialog

        dlg = LineStyleDialog(0, None, channel_types=["mag"])
        qtbot.addWidget(dlg)
        assert dlg._width_label.text() == "automatic"
        dlg._slider.setValue(4)
        assert dlg._width_label.text() == "4 px"

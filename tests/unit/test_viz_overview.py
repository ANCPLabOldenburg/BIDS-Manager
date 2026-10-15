"""The recording at a glance, and the commands that navigate it.

``bidsmgr.viz.compute.overview`` (the activity profile the overview bar
draws), the filter presets of ``viz.compute.filters`` and the time cursor.
Qt-free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bidsmgr.viz.compute import filters
from bidsmgr.viz.compute.overview import BINS, MAX_CHANNELS, activity
from bidsmgr.viz.data.signal import SignalSource
from bidsmgr.viz.scene import Scene, SignalLayer, SourceRef, TracesState
from bidsmgr.viz.store import SceneStore

mne = pytest.importorskip("mne")


def _source(data, sfreq: float, types, *, gaps=None) -> SignalSource:
    data = np.atleast_2d(np.asarray(data, dtype=float))
    names = [f"c{i}" for i in range(data.shape[0])]
    info = mne.create_info(names, sfreq, types, verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    return SignalSource(path=Path("sub-01_task-x_eeg.fif"), raw=raw, kind="meeg", gaps=gaps)


def _store(src: SignalSource) -> SceneStore:
    store = SceneStore()
    store.sources = {"sig0": src}
    scene = Scene()
    scene.sources = {"sig0": SourceRef(id="sig0", path=str(src.path), kind="signal")}
    scene.layers = [SignalLayer(id="sig", source="sig0", name=src.path.name)]
    scene.traces = TracesState(width=10.0)
    store.replace_scene(scene)
    return store


class TestActivity:
    def test_a_burst_shows_where_it_happened(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1e-6, (4, 60_000))
        data[:, 30_000:31_000] *= 20          # one loud second at 30 s of 60
        out = activity(_source(data, 1000.0, ["eeg"] * 4))
        prof = out["profile"]
        assert prof.shape == (BINS,) and out["bins"] == BINS
        peak = int(np.argmax(prof)) / BINS * 60.0
        assert 29.5 <= peak <= 31.5
        assert prof.max() == pytest.approx(1.0)
        assert np.median(prof) < 0.2, "the quiet rest stays low"

    def test_one_loud_channel_does_not_set_the_profile(self):
        """Each channel is normalised by its own spread, so a channel ten
        thousand times louder than the rest weighs the same."""
        rng = np.random.default_rng(1)
        data = rng.normal(0, 1e-6, (3, 20_000))
        data[0] *= 1e4
        data[1, 10_000:10_500] *= 30
        prof = activity(_source(data, 1000.0, ["eeg"] * 3))["profile"]
        assert 9.5 <= np.argmax(prof) / prof.size * 20.0 <= 11.0

    def test_trigger_pulses_are_not_activity(self):
        rng = np.random.default_rng(2)
        eeg = rng.normal(0, 1e-6, (2, 20_000))
        stim = np.zeros((1, 20_000))
        stim[0, ::1000] = 5.0
        prof = activity(_source(np.vstack([eeg, stim]), 1000.0, ["eeg", "eeg", "stim"]))["profile"]
        assert prof.max() - np.median(prof) < 0.6, "no comb of spikes at the pulses"

    def test_a_stretch_without_samples_is_a_gap(self):
        data = np.random.default_rng(3).normal(0, 1, (2, 10_000))
        gaps = np.zeros(data.shape, dtype=bool)
        gaps[:, 4_000:6_000] = True
        out = activity(_source(data, 100.0, ["misc", "misc"], gaps=gaps), bins=100)
        assert out["gaps"][45] and out["gaps"][55]
        assert not out["gaps"][10] and not out["gaps"][90]
        assert out["profile"][50] == 0.0

    def test_many_channels_are_sampled_not_all_read(self):
        data = np.random.default_rng(4).normal(0, 1, (MAX_CHANNELS * 3, 5_000))
        src = _source(data, 500.0, ["eeg"] * data.shape[0])
        calls = []
        read = src.read
        src.read = lambda idx, s0, s1: (calls.append(idx), read(idx, s0, s1))[1]
        activity(src)
        assert len(calls) == MAX_CHANNELS

    def test_a_cancelled_job_stops(self):
        class Token:
            cancelled = True

        with pytest.raises(RuntimeError):
            activity(_source(np.ones((2, 1000)), 100.0, ["eeg", "eeg"]), cancel=Token())


class TestPresets:
    @pytest.mark.parametrize("kind", sorted(filters.PRESETS))
    def test_every_preset_is_a_valid_band(self, kind):
        for label, hp, lp in filters.PRESETS[kind]:
            assert hp is not None or lp is not None, label
            if hp is not None and lp is not None:
                assert hp < lp, label
            assert filters.preset_of(kind, hp, lp) == label

    def test_a_band_that_is_no_preset_is_none(self):
        assert filters.preset_of("meeg", 0.3, 33.0) is None
        assert filters.preset_of("meeg", 1.0, 40.0) == "1 to 40 Hz"
        assert filters.preset_of("physio", 1.0, 40.0) is None, "presets are per kind"


class TestTimeCursor:
    def test_placed_and_removed(self):
        store = _store(_source(np.zeros((2, 6000)), 100.0, ["eeg", "eeg"]))
        assert store.run("cursor.time", t=12.5) == frozenset({"cursor.time"})
        assert store.scene.cursor.time == 12.5
        assert store.run("cursor.time", t=12.5) == frozenset(), "no change, no report"
        store.run("cursor.time", t=None)
        assert store.scene.cursor.time is None

    def test_reset_keeps_the_bad_channels_and_drops_the_cursor(self):
        """Bad channels are a judgement about the data; a reset of the view
        threw them away with the scroll position."""
        store = _store(_source(np.zeros((3, 6000)), 100.0, ["eeg"] * 3))
        store.run("channels.toggle_bad", name="c1")
        store.run("traces.scale", value=3.0)
        store.run("cursor.time", t=4.0)
        paths = store.run("traces.reset")
        tr = store.scene.traces
        assert tr.scale == 1.0
        assert tr.bads == ["c1"]
        assert store.scene.cursor.time is None and "cursor.time" in paths

"""``traces.go_to``: where a click on a QC channel map takes you. Qt-free."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bidsmgr.viz.data.signal import SignalSource
from bidsmgr.viz.scene import Scene, SignalLayer, SourceRef, TracesState
from bidsmgr.viz.store import SceneStore

mne = pytest.importorskip("mne")


def _store() -> SceneStore:
    names = [f"MEG{i:03d}" for i in range(30)] + [f"EEG{i:03d}" for i in range(10)]
    types = ["mag"] * 30 + ["eeg"] * 10
    info = mne.create_info(names, 100.0, types, verbose=False)
    raw = mne.io.RawArray(np.zeros((40, 6000)), info, verbose=False)
    src = SignalSource(path=Path("sub-01_task-x_meg.fif"), raw=raw, kind="meeg")
    store = SceneStore()
    store.sources = {"sig0": src}
    scene = Scene()
    scene.sources = {"sig0": SourceRef(id="sig0", path=str(src.path), kind="signal")}
    scene.layers = [SignalLayer(id="sig", source="sig0", name=src.path.name)]
    scene.traces = TracesState(count=10, width=10.0)
    store.replace_scene(scene)
    return store


def _shown(store) -> list[str]:
    src = store.sources["sig0"]
    tr = store.scene.traces
    pool = src.picks_for(tr.ch_type, tr.picks)
    start = min(tr.offset, max(0, len(pool) - tr.count))
    return [src.ch_names[i] for i in pool[start:start + tr.count]]


def test_it_brings_the_channel_and_the_segment_on_screen():
    store = _store()
    start = store.sources["sig0"].start_time
    store.run("traces.go_to", channel="MEG025", onset=start + 40.0, duration=2.0)
    tr = store.scene.traces
    assert "MEG025" in _shown(store)
    assert tr.t0 == pytest.approx(41.0 - tr.width / 2.0), "the segment centred"
    assert store.scene.cursor.time == pytest.approx(start + 40.0)
    assert (tr.focus.channel, tr.focus.onset, tr.focus.duration) == ("MEG025", start + 40.0, 2.0)


def test_a_channel_filtered_out_is_shown_again():
    store = _store()
    store.run("traces.type", ch_type="mag")
    store.run("traces.pick", names=["MEG000", "MEG001"])
    store.run("traces.go_to", channel="EEG004", onset=10.0, duration=2.0)
    assert store.scene.traces.picks is None
    assert store.scene.traces.ch_type == "all"
    assert "EEG004" in _shown(store)


def test_a_stretch_longer_than_the_window_starts_at_its_start():
    store = _store()
    start = store.sources["sig0"].start_time
    store.run("traces.go_to", channel="MEG001", onset=start + 20.0, duration=30.0)
    assert store.scene.traces.t0 == pytest.approx(20.0)


def test_the_outline_goes_and_a_reset_removes_it():
    store = _store()
    store.run("traces.go_to", channel="MEG001", onset=5.0, duration=2.0)
    store.run("traces.focus")
    assert store.scene.traces.focus is None
    store.run("traces.go_to", channel="MEG001", onset=5.0, duration=2.0)
    store.run("traces.reset")
    assert store.scene.traces.focus is None


def test_an_unknown_channel_is_refused():
    with pytest.raises(ValueError):
        _store().run("traces.go_to", channel="nope", onset=0.0)

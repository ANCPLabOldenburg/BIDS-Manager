"""What the viewers remember between files and windows: ``viz.memory``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bidsmgr.viz import memory
from bidsmgr.viz.scene import ClipPlane, Scene, SpectrumState, TracesState

mne = pytest.importorskip("mne")


def _src(sfreq=200.0, seconds=30.0, types=("eeg", "eeg", "eog")):
    from bidsmgr.viz.data.signal import SignalSource

    names = [f"c{i}" for i in range(len(types))]
    info = mne.create_info(names, sfreq, list(types), verbose=False)
    raw = mne.io.RawArray(np.zeros((len(types), int(sfreq * seconds))), info, verbose=False)
    return SignalSource(path=Path("sub-01_task-x_eeg.fif"), raw=raw, kind="meeg")


def test_the_volume_look_round_trips():
    scene = Scene()
    scene.display.radiological = True
    scene.render.effect = "MIP"
    scene.clips = [ClipPlane(active=True, az=30.0, el=10.0, pos=0.3)]
    look = memory.volume_look(scene)
    fresh = memory.apply_volume_look(Scene(), look)
    assert fresh.display.radiological and fresh.render.effect == "MIP"
    assert fresh.clips[0].az == 30.0 and fresh.clips[0].active


def test_a_broken_look_is_skipped_not_raised():
    fresh = memory.apply_volume_look(Scene(), {"render": {"effect": 7}, "clips": "nope"})
    assert fresh.render == Scene().render and fresh.clips == Scene().clips


def test_trace_options_carry_and_are_made_to_fit():
    src = _src(sfreq=200.0, seconds=30.0)
    remembered = {"count": 64, "width": 120.0, "butterfly": True, "scale": 2.5,
                  "ch_type": "mag", "hp": 1.0, "lp": 40.0, "quality": True}
    opening = {"t0": 0.0, "width": 10.0, "count": 3, "ch_type": "all"}
    state = memory.restore_traces(opening, remembered, src)
    assert state["butterfly"] and state["scale"] == 2.5
    assert not state.get("quality"), "QC runs on opening only when the user asked for that"
    asked = memory.restore_traces(opening, remembered, src, qc_on_open=True)
    assert asked["quality"]
    assert state["count"] == 3, "no more traces than channels"
    assert state["width"] == pytest.approx(src.duration), "no longer than the recording"
    assert state["ch_type"] == "all", "this recording has no magnetometers"
    assert (state["hp"], state["lp"]) == (1.0, 40.0)


def test_a_filter_the_recording_cannot_take_is_dropped():
    src = _src(sfreq=100.0)
    state = memory.restore_traces({"width": 10.0}, {"lp": 70.0, "hp": 1.0}, src)
    assert state["lp"] is None and state["hp"] is None


def test_spectrum_options_but_never_the_phase():
    prefs = memory.spectrum_prefs(SpectrumState(part="magnitude", lb_hz=5.0, phase0=40.0))
    assert "phase0" not in prefs
    state = memory.restore_spectrum(SpectrumState(), prefs)
    assert (state.part, state.lb_hz, state.phase0) == ("magnitude", 5.0, 0.0)
    assert TracesState().quality is False

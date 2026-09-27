"""Unit tests for the Qt-free recording read/summary helpers.

These functions are pure logic: extension routing, MNE read, the display
summary, and events.tsv resolution. No QApplication is created here.

They live in three modules now. ``recording_formats`` holds what counts as
a recording and how its extension is spelled, because four things ask;
``time_series_view`` holds the events lookup, because the view draws them;
``recording_viewer_pane`` keeps the MNE read and the summary card.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from bidsmgr.gui.widgets.recording_formats import (  # noqa: E402
    full_ext as _full_ext,
    is_recording_path,
)
from bidsmgr.gui.widgets.recording_viewer_pane import (  # noqa: E402
    _read_raw,
    _summarize_raw,
)
from bidsmgr.gui.widgets.time_series_view import (  # noqa: E402
    _events_sibling,
    _read_events_tsv,
)


def _make_fif(tmp: Path, name: str = "sub-01_task-rest_eeg.fif") -> Path:
    sfreq = 128.0
    n = int(sfreq * 3)
    data = np.random.default_rng(2).standard_normal((5, n)) * 1e-5
    info = mne.create_info(
        ["Fz", "Cz", "Pz", "EOG", "STI"],
        sfreq,
        ["eeg", "eeg", "eeg", "eog", "stim"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.RawArray(data, info, verbose=False)
        path = tmp / name
        raw.save(path, overwrite=True, verbose=False)
    return path


def test_full_ext_preserves_double_suffix() -> None:
    assert _full_ext(Path("a.fif.gz")) == ".fif.gz"
    assert _full_ext(Path("a.edf")) == ".edf"
    assert _full_ext(Path("SUB.EDF")) == ".edf"
    assert _full_ext(Path("rec.ds")) == ".ds"


def test_is_recording_path() -> None:
    assert is_recording_path(Path("x.fif"))
    assert is_recording_path(Path("x.fif.gz"))
    assert is_recording_path(Path("x.edf"))
    assert is_recording_path(Path("x.vhdr"))
    assert is_recording_path(Path("x.cnt"))
    # BrainVision binary / sidecars are opened via .vhdr, never directly.
    assert not is_recording_path(Path("x.eeg"))
    assert not is_recording_path(Path("x.vmrk"))
    assert not is_recording_path(Path("x.json"))
    assert not is_recording_path(Path("x.tsv"))
    assert not is_recording_path(Path("x.nii.gz"))


def test_read_and_summarize(tmp_path: Path) -> None:
    path = _make_fif(tmp_path)
    raw = _read_raw(path, preload=False)
    meta = _summarize_raw(raw, path)
    assert meta["n_channels"] == 5
    assert meta["sfreq"] == 128.0
    assert meta["ch_type_counts"] == {"eeg": 3, "eog": 1, "stim": 1}
    assert meta["available_ch_types"] == ["eeg", "eog", "stim"]
    assert meta["is_ctf"] is False
    assert meta["duration"] > 0
    assert meta["name"] == path.name


def test_read_preload_materialises(tmp_path: Path) -> None:
    path = _make_fif(tmp_path)
    raw = _read_raw(path, preload=True)
    assert raw.preload is True
    assert raw.n_times > 0


def test_events_sibling_and_parse(tmp_path: Path) -> None:
    path = _make_fif(tmp_path)
    # No sibling yet.
    assert _events_sibling(path) is None
    # The BIDS events file shares all entities, only the suffix differs.
    (tmp_path / "sub-01_task-rest_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0.1\t0\tgo\n0.9\t0\tstop\n"
    )
    sib = _events_sibling(path)
    assert sib is not None
    assert _read_events_tsv(sib) == [(0.1, "go"), (0.9, "stop")]


def test_read_events_tsv_falls_back_to_value(tmp_path: Path) -> None:
    p = tmp_path / "e_events.tsv"
    p.write_text("onset\tvalue\n0.0\t1\n1.0\t2\n")
    assert _read_events_tsv(p) == [(0.0, "1"), (1.0, "2")]

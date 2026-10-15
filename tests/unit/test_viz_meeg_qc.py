"""The MEG/EEG quality check: what it flags, and what it must not.

``bidsmgr.viz.compute.meeg_qc``. Synthetic recordings where the answer is
known. Qt-free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bidsmgr.viz.compute.meeg_qc import bad_label, flagged_segments, quality
from bidsmgr.viz.data.signal import SignalSource
from bidsmgr.viz.settings import MeegQcSettings

mne = pytest.importorskip("mne")

SFREQ = 500.0
SECONDS = 120.0


def _src(data: np.ndarray, types=None) -> SignalSource:
    names = [f"E{i:02d}" for i in range(data.shape[0])]
    info = mne.create_info(names, SFREQ, types or ["eeg"] * len(names), verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    return SignalSource(path=Path("sub-01_task-x_eeg.fif"), raw=raw, kind="meeg")


def _noise(n_ch: int = 16, seed: int = 0, scale: float = 10e-6) -> np.ndarray:
    """Channels of one type as real ones are: a field they all see (with
    their own gains) plus a little noise of their own, so they correlate."""
    rng = np.random.default_rng(seed)
    n = int(SECONDS * SFREQ)
    shared = rng.normal(0, 1, (3, n))
    gains = rng.uniform(0.5, 1.0, (n_ch, 3))
    return scale * (gains @ shared + 0.3 * rng.normal(0, 1, (n_ch, n)))


class TestChannels:
    def test_a_noisy_and_a_flat_channel(self):
        data = _noise()
        data[3] += np.random.default_rng(9).normal(0, 80e-6, data.shape[1])
        data[7] *= 1e-5
        r = quality(_src(data))
        verdict = {c["name"]: c["reasons"] for c in r["channels"]}
        assert verdict["E03"] == ["noisy"]
        assert verdict["E07"] == ["flat"]
        assert sorted(r["suggested_bads"]) == ["E03", "E07"]
        assert sum(bool(v) for v in verdict.values()) == 2

    def test_blinks_do_not_condemn_a_channel(self):
        """Large, frequent but intermittent artefacts (blinks on the frontal
        channels) lift a channel's median; its quiet level stays normal."""
        data = _noise()
        n = data.shape[1]
        for start in range(0, n, int(3.0 * SFREQ)):
            data[0:2, start:start + int(0.4 * SFREQ)] += 150e-6
        r = quality(_src(data))
        assert r["suggested_bads"] == []

    def test_line_noise_against_its_type(self):
        data = _noise()
        t = np.arange(data.shape[1]) / SFREQ
        data[5] += 60e-6 * np.sin(2 * np.pi * 50.0 * t)
        r = quality(_src(data), line_freq=50.0)
        assert [c["name"] for c in r["channels"] if "line noise" in c["reasons"]] == ["E05"]
        assert quality(_src(data))["line_freq"] is None, "no frequency stated, not checked"

    def test_bad_channels_are_left_out(self):
        data = _noise()
        data[3] += np.random.default_rng(9).normal(0, 80e-6, data.shape[1])
        r = quality(_src(data), exclude={"E03"})
        assert "E03" not in [c["name"] for c in r["channels"]]


class TestSegments:
    def test_a_burst_is_flagged_where_it_happened(self):
        data = _noise()
        burst = slice(int(60 * SFREQ), int(64 * SFREQ))
        data[:, burst] *= 6.0
        r = quality(_src(data))
        assert flagged_segments(r) == [(60.0, 4.0)]
        assert "EEG: 100 % of channels off" in r["segment_reasons"][30]
        assert bad_label(r["segment_kinds"][30]) == "BAD_noise"

    def test_muscle_in_its_band(self):
        data = _noise()
        t = np.arange(data.shape[1]) / SFREQ
        rng = np.random.default_rng(1)
        hf = rng.normal(0, 1, data.shape[1])
        win = slice(int(80 * SFREQ), int(82 * SFREQ))
        burst = np.zeros(data.shape[1])
        burst[win] = 1.0
        # 125 Hz bursts on every channel, small in amplitude (real EMG in
        # EEG is tens of microvolts).
        data += 15e-6 * burst * np.sin(2 * np.pi * 125.0 * t) * (1 + 0.1 * hf)
        r = quality(_src(data))
        assert r["types"]["eeg"]["muscle"] is not None
        assert r["flagged"][40] and any("muscle" in x for x in r["segment_reasons"][40])
        assert bad_label(r["segment_kinds"][40]) == "BAD_muscle"

    def test_a_jump_on_many_channels(self):
        data = _noise()
        at = int(30 * SFREQ)
        data[:4, at:at + 5] += 2e-3
        r = quality(_src(data))
        assert r["flagged"][15] and "jump" in r["segment_kinds"][15]
        assert bad_label(r["segment_kinds"][15]) == "BAD_jump"

    def test_a_clean_recording_flags_little(self):
        r = quality(_src(_noise(seed=5)))
        assert r["flagged"].sum() <= 1
        assert r["suggested_bads"] == []

    def test_neighbours_merge(self):
        r = {"segment_s": 2.0, "times": np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
             "flagged": np.array([False, True, True, False, True])}
        assert flagged_segments(r) == [(2.0, 4.0), (8.0, 2.0)]


def test_too_short_or_nothing_to_check():
    with pytest.raises(ValueError):
        quality(_src(_noise()[:, :1000]))
    with pytest.raises(ValueError):
        quality(_src(_noise(2), types=["stim", "misc"]))


def _meg_eeg():
    """Magnetometers (~1e-13 T), gradiometers (~1e-11 T/m) and EEG
    (~1e-5 V) in one recording: units a hundred million apart."""
    mag = _noise(8, seed=2, scale=2e-13)
    grad = _noise(10, seed=3, scale=4e-12)
    eeg = _noise(8, seed=4, scale=10e-6)
    return np.vstack([mag, grad, eeg]), ["mag"] * 8 + ["grad"] * 10 + ["eeg"] * 8


class TestTypesNeverMix:
    def test_units_alone_flag_nothing(self):
        data, types = _meg_eeg()
        r = quality(_src(data, types))
        assert r["suggested_bads"] == []
        assert set(r["types"]) == {"mag", "grad", "eeg"}
        assert "MEG mag: 0 noisy" in r["summary"] and "EEG: 0 noisy" in r["summary"]

    def test_a_channel_is_judged_against_its_own_type(self):
        data, types = _meg_eeg()
        # A gradiometer with noise of its own, noisy among gradiometers.
        data[10] += np.random.default_rng(5).normal(0, 30e-12, data.shape[1])
        r = quality(_src(data, types))
        assert r["suggested_bads"] == ["E10"]
        assert [c["type"] for c in r["channels"] if c["reasons"]] == ["grad"]

    def test_a_segment_is_judged_per_type(self):
        data, types = _meg_eeg()
        burst = slice(int(50 * SFREQ), int(52 * SFREQ))
        data[8:12, burst] *= 8.0  # 4 of 10 gradiometers, 4 of 26 channels overall
        r = quality(_src(data, types))
        assert r["flagged"][25]
        assert r["segment_reasons"][25] == ["MEG grad: 40 % of channels off"]
        assert not r["types"]["eeg"]["flagged"][25] and not r["types"]["mag"]["flagged"][25]
        from bidsmgr.viz.compute.meeg_qc import tracks

        ids = [t["id"] for t in tracks(r, ["off"])]
        assert ids == ["off:mag", "off:grad", "off:eeg"]


class TestSettings:
    def test_thresholds_are_the_users(self):
        data = _noise()
        data[3] += np.random.default_rng(9).normal(0, 40e-6, data.shape[1])
        assert quality(_src(data))["suggested_bads"] == ["E03"]
        relaxed = MeegQcSettings(noisy_z=8.0, min_correlation=0.0)
        assert quality(_src(data), settings=relaxed)["suggested_bads"] == []

    def test_one_measure_alone(self):
        data = _noise()
        at = int(30 * SFREQ)
        data[:6, at:at + 5] += 2e-3
        only_std = quality(_src(data), settings=MeegQcSettings(use_ptp=False))
        only_ptp = quality(_src(data), settings=MeegQcSettings(use_std=False))
        assert only_ptp["flagged"][15]
        assert only_ptp["settings"]["use_std"] is False
        assert only_std["settings"]["use_ptp"] is False
        with pytest.raises(ValueError):
            quality(_src(data), settings=MeegQcSettings(use_std=False, use_ptp=False))

    def test_segment_length_and_muscle_off(self):
        r = quality(_src(_noise()), settings=MeegQcSettings(segment_s=4.0, muscle=False))
        assert r["segment_s"] == pytest.approx(4.0) and len(r["times"]) == 30
        assert r["types"]["eeg"]["muscle"] is None and not r["muscle_checked"]


class TestCorrelation:
    def test_a_loud_channel_that_follows_its_type_is_not_noisy(self):
        """A sensor near the heart or the room's field is loud with what the
        others also see: not a broken sensor (MNE's Maxwell check agrees)."""
        data = _noise()
        data[3] *= 8.0
        r = quality(_src(data))
        e03 = [c for c in r["channels"] if c["name"] == "E03"][0]
        assert e03["loud"] and e03["follows"] > 0.9 and e03["reasons"] == []
        level_only = quality(_src(data), settings=MeegQcSettings(min_correlation=0.0))
        assert level_only["suggested_bads"] == ["E03"]

    def test_a_channel_at_a_normal_level_that_follows_nothing(self):
        data = _noise()
        data[5] = np.random.default_rng(11).normal(0, data[5].std(), data.shape[1])
        r = quality(_src(data))
        assert {c["name"]: c["reasons"] for c in r["channels"]}["E05"] == ["uncorrelated"]
        assert r["suggested_bads"] == ["E05"]

    def test_the_rooms_field_is_projected_away(self):
        """An SSP projector stored with the recording is applied."""
        data = _noise()
        src = _src(data)
        vec = np.ones(16) / 4.0
        proj = mne.Projection(data=dict(nrow=1, ncol=16, row_names=None,
                                        col_names=src.ch_names, data=vec[None]),
                              kind=1, desc="room", active=False)
        src.raw.add_proj([proj])
        assert quality(src)["projected"]
        assert not quality(src, settings=MeegQcSettings(apply_proj=False))["projected"]


class TestTracksAndTypes:
    def test_qc_on_chosen_types_only(self):
        data, types = _meg_eeg()
        r = quality(_src(data, types), settings=MeegQcSettings(types=["grad"]))
        assert set(r["types"]) == {"grad"}
        assert {c["type"] for c in r["channels"]} == {"grad"}
        # A type this recording does not have: every type, not nothing.
        r = quality(_src(data, types), settings=MeegQcSettings(types=["seeg"]))
        assert set(r["types"]) == {"mag", "grad", "eeg"}

    def test_tracks_never_put_two_types_on_one_axis(self):
        from bidsmgr.viz.compute.meeg_qc import tracks

        data, types = _meg_eeg()
        r = quality(_src(data, types))
        made = tracks(r, ["off", "std", "map"], ["mag", "eeg"])
        assert [t["id"] for t in made] == ["off:mag", "off:eeg", "std:mag", "std:eeg",
                                           "map:mag", "map:eeg"]
        off = made[0]
        assert off["unit"] == "%" and off["rule"] == pytest.approx(20.0)
        assert off["x"][0] == pytest.approx(1.0), "segment centres, from the start"
        channel_map = made[-1]
        assert channel_map["kind"] == "image"
        assert channel_map["image"].shape == (8, len(r["times"]))
        assert channel_map["rows"][0] == "E18"

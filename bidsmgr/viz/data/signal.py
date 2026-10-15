"""Signals: any multichannel time series, as one kind of source.

MEG, EEG, iEEG, NIRS, physio, a BIDS ``_stim`` or ``_motion`` table: all of
them are channels, a sampling rate and samples, which is what an
``mne.io.Raw`` is made of (decision D8). So every one is held as a Raw, and
one filter, one decimation, one spectrum and one traces canvas serve all of
them.

What a :class:`SignalSource` adds to the Raw is what the Raw cannot carry:

* the GAP MASK. MNE cannot carry NaN through a filter or an FFT, so the array
  holds zeros where the recording has no sample and the mask says where. The
  filter sees a continuous signal and the drawing shows the gaps. A real ECG
  in this lab's data is 28 percent gaps;
* ``start_time``: where sample zero sits relative to the run (physio starts
  before the scanner). Traces are drawn in RUN time, so events from the
  run's ``_events.tsv`` line up;
* the facts the controls need (types, CTF), computed once.

Qt-free. Reading, filtering and resampling run on a worker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..bids import full_ext
from .events import Event

log = logging.getLogger(__name__)

#: Extensions ``mne.io.read_raw`` refuses to auto-dispatch (ambiguous
#: between vendors). Tried in order after the generic reader. ANT ``.cnt``
#: needs the optional ``antio`` package; absent it the error surfaces.
_SPECIFIC_READERS: dict[str, tuple[str, ...]] = {
    ".cnt": ("read_raw_cnt", "read_raw_ant"),
    ".egi": ("read_raw_egi",),
    ".mff": ("read_raw_egi",),
}


def read_recording(path, *, preload: bool):
    """Read a recording with MNE (``preload`` False: header only)."""
    import mne

    p = str(path)
    try:
        return mne.io.read_raw(p, preload=preload, verbose=False)
    except Exception as primary:
        readers = _SPECIFIC_READERS.get(full_ext(path))
        if not readers:
            raise
        last = primary
        for name in readers:
            fn = getattr(mne.io, name, None)
            if fn is None:
                continue
            try:
                return fn(p, preload=preload, verbose=False)
            except Exception as exc:  # noqa: BLE001 - try the next reader
                last = exc
        raise last


def channel_types(raw) -> list[str]:
    import mne

    return [mne.channel_type(raw.info, i) for i in range(len(raw.ch_names))]


def is_ctf(raw, types: list[str], path=None) -> bool:
    """CTF MEG: axial gradiometers that MNE files as ``mag``."""
    available = set(types)
    return ("mag" in available and "grad" not in available and (
        "ref_meg" in available
        or getattr(raw, "compensation_grade", None) is not None
        or (path is not None and full_ext(path) == ".ds")
    ))


def summarize(raw, path) -> dict:
    """The metadata card: plain values, no Qt."""
    info = raw.info
    types = channel_types(raw)
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    try:
        duration = float(raw.times[-1]) if raw.n_times else 0.0
    except Exception:  # noqa: BLE001
        duration = 0.0
    meas = info.get("meas_date")
    return {
        "filename": str(path),
        "name": Path(path).name,
        "n_channels": len(raw.ch_names),
        "sfreq": float(info["sfreq"]),
        "duration": duration,
        "n_times": int(raw.n_times),
        "highpass": info.get("highpass"),
        "lowpass": info.get("lowpass"),
        "line_freq": info.get("line_freq"),
        "meas_date": str(meas) if meas else None,
        "ch_type_counts": counts,
        "available_ch_types": sorted(set(types)),
        "is_ctf": is_ctf(raw, types, path),
        "bads": list(info.get("bads") or []),
    }


@dataclass
class SignalSource:
    """One recording, preloaded, with what the Raw cannot carry."""

    path: Path
    raw: Any
    kind: str = "meeg"                    # "meeg" | "physio"
    gaps: Optional[np.ndarray] = None
    start_time: float = 0.0
    #: Every file the source was built from (a run's physio, combined).
    paths: list[Path] = field(default_factory=list)
    #: Something the reader wants said (a strided read).
    note: str = ""
    events_tsv: list[Event] = field(default_factory=list)
    events_stim: list[Event] = field(default_factory=list)
    #: The recording's own annotations (BAD spans among them).
    events_annotations: list[Event] = field(default_factory=list)
    #: Channels marked bad: the recording's own list and the BIDS
    #: ``_channels.tsv`` ``status`` column.
    bads: set = field(default_factory=set)
    #: Where the bads came from, for the card ("" when none).
    bads_from: str = ""
    #: The recording's bad segments, in RUN time: the ``_events.tsv`` BAD
    #: rows when the run has a table (mne-bids reads that back as the
    #: annotations, so it is the authority), else the file's own BAD
    #: annotations.
    bad_spans: list[Event] = field(default_factory=list)
    # -- derived, set by refresh() ---------------------------------------
    sfreq: float = 1.0
    ch_names: list[str] = field(default_factory=list)
    ch_types: list[str] = field(default_factory=list)
    n_times: int = 0
    duration: float = 0.0
    ctf: bool = False

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        raw = self.raw
        self.sfreq = float(raw.info["sfreq"])
        self.ch_names = list(raw.ch_names)
        self.ch_types = channel_types(raw)
        self.n_times = int(raw.n_times)
        self.duration = float(raw.times[-1]) if raw.n_times else 0.0
        self.ctf = is_ctf(raw, self.ch_types, self.path)

    @property
    def available_types(self) -> list[str]:
        return sorted(set(self.ch_types))

    def type_label(self, ch_type: str) -> str:
        """How a type is NAMED to the user (CTF axial gradiometers)."""
        return "mag (axial grad)" if self.ctf and ch_type == "mag" else ch_type

    def events(self, source: str = "auto") -> list[Event]:
        """Events in RUN time. ``auto`` prefers the curated table, else the
        trigger channel, and always adds the recording's annotations."""
        out: list[Event] = []
        if source in ("auto", "events.tsv") and self.events_tsv:
            out = list(self.events_tsv)
        elif source in ("auto", "stim") and self.events_stim:
            out = [Event(e.onset + self.start_time, e.duration, e.label)
                   for e in self.events_stim]
        if source in ("auto", "annotations") and self.events_annotations:
            out += [Event(e.onset + self.start_time, e.duration, e.label, e.kind)
                    for e in self.events_annotations]
        return out

    def event_sources(self) -> list[str]:
        out = []
        if self.events_tsv:
            out.append("events.tsv")
        if self.events_stim:
            out.append("stim")
        if self.events_annotations:
            out.append("annotations")
        return out

    # -- units and scale ----------------------------------------------------

    #: How each channel type is shown: (unit, factor from SI). The values a
    #: reader knows: picotesla in femtotesla, volts in microvolts.
    UNITS = {"mag": ("fT", 1e15), "grad": ("fT/cm", 1e13), "eeg": ("µV", 1e6),
             "seeg": ("µV", 1e6), "ecog": ("µV", 1e6), "dbs": ("µV", 1e6),
             "eog": ("µV", 1e6), "emg": ("µV", 1e6), "ecg": ("µV", 1e6)}

    def unit_for(self, ch_type: str) -> tuple[str, float]:
        """``(unit, factor)`` to show a channel of ``ch_type``. A physio
        table carries no unit: arbitrary units."""
        if self.kind == "physio":
            return "a.u.", 1.0
        return self.UNITS.get(ch_type, ("a.u.", 1.0))

    def type_scales(self) -> dict[str, float]:
        """One amplitude per channel TYPE for the whole recording: the median
        over (up to 16) channels of the 2.5 to 97.5 percentile range of a
        strided sample. Measured once, so a quiet page and an artefact page
        are drawn at the same gain and can be compared (the per-page scale
        it replaces changed with every page)."""
        cached = getattr(self, "_type_scales", None)
        if cached is not None:
            return cached
        out: dict[str, float] = {}
        by_type: dict[str, list[int]] = {}
        for i, t in enumerate(self.ch_types):
            by_type.setdefault(t, []).append(i)
        step = max(1, self.n_times // 100_000)
        for t, idx in by_type.items():
            if len(idx) > 16:
                idx = [idx[k] for k in np.linspace(0, len(idx) - 1, 16).astype(int)]
            data = np.asarray(self.raw.get_data(picks=idx)[:, ::step], dtype=float)
            if self.gaps is not None:
                mask = self.gaps[np.asarray(idx), ::step][:, : data.shape[1]]
                data = np.where(mask, np.nan, data)
            spans = []
            for row in data:
                row = row[np.isfinite(row)]
                if row.size > 10:
                    lo, hi = np.percentile(row, (2.5, 97.5))
                    if hi > lo:
                        spans.append(hi - lo)
            out[t] = float(np.median(spans)) if spans else 1.0
        self._type_scales = out
        return out

    #: Housekeeping channels: the system's own status (Elekta's internal
    #: active shielding, system and head-position coil channels). Not
    #: signals: under "all" eleven IAS channels flickering between levels
    #: filled half the window. Shown when their type is chosen.
    HOUSEKEEPING = frozenset({"ias", "syst", "chpi", "exci"})

    def picks_for(self, ch_type: str, picks: Optional[list[str]] = None) -> list[int]:
        """Channel indices of a type filter ("all" signals, "mag+grad", a
        type), narrowed to ``picks`` (names) when given."""
        if ch_type == "all":
            idx = [i for i, t in enumerate(self.ch_types) if t not in self.HOUSEKEEPING]
            if not idx:
                idx = list(range(len(self.ch_names)))
        elif ch_type == "mag+grad":
            idx = [i for i, t in enumerate(self.ch_types) if t in ("mag", "grad")]
        else:
            idx = [i for i, t in enumerate(self.ch_types) if t == ch_type]
        if picks is not None:
            wanted = set(picks)
            idx = [i for i in idx if self.ch_names[i] in wanted]
        return idx

    def read(self, indices: list[int], s0: int, s1: int) -> np.ndarray:
        """Samples ``[s0, s1)`` of ``indices`` (float64), as a COPY the caller
        may write to: the filter and the gap blanking work in place on it."""
        s0 = max(0, int(s0))
        s1 = min(self.n_times, int(s1))
        if s1 <= s0 or not indices:
            return np.empty((len(indices), 0))
        data, _times = self.raw[indices, s0:s1]
        return np.asarray(data, dtype=np.float64)

    def gap_mask(self, indices: list[int], s0: int, s1: int) -> Optional[np.ndarray]:
        if self.gaps is None:
            return None
        try:
            mask = self.gaps[np.asarray(indices), max(0, s0):min(self.n_times, s1)]
        except IndexError:
            return None
        return mask


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def channels_sibling(path) -> Optional[Path]:
    """The recording's ``_channels.tsv`` beside it, if any."""
    from .events import run_base

    p = Path(path)
    stem = p.name.split(".")[0]
    candidates = [p.parent / (stem.rsplit("_", 1)[0] + "_channels.tsv")]
    candidates.append(p.parent / f"{run_base(p)}_channels.tsv")
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def line_frequency(src) -> Optional[float]:
    """The mains frequency of a recording: its own (``info["line_freq"]``),
    else ``PowerLineFrequency`` from its BIDS sidecar; None when neither
    says (``n/a`` is allowed by BIDS and means unknown)."""
    value = None
    try:
        value = src.raw.info.get("line_freq")
    except Exception:  # noqa: BLE001 - no info, no frequency
        value = None
    if not value:
        import json

        p = Path(src.path)
        sidecar = p.parent / (p.name.split(".")[0] + ".json")
        try:
            value = json.loads(sidecar.read_text(encoding="utf-8-sig")).get("PowerLineFrequency")
        except (OSError, ValueError, AttributeError):
            value = None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def read_bad_channels(path) -> set:
    """Channel names whose ``status`` is ``bad`` in a ``_channels.tsv``."""
    import csv

    out = set()
    try:
        # utf-8-sig: a byte order mark (mne-bids writes one) is not part
        # of the first column's name.
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if str(row.get("status", "")).strip().lower() == "bad" and row.get("name"):
                    out.add(row["name"])
    except Exception as exc:  # noqa: BLE001 - an unreadable table is no bads
        log.debug("could not read %s: %s", path, exc)
    return out


def _with_events(src: SignalSource, root=None) -> SignalSource:
    from .events import (annotation_events, edge_events, events_sibling, read_events_tsv,
                         stim_events)

    sibling = events_sibling(src.path)
    if sibling is not None:
        src.events_tsv = read_events_tsv(sibling)
    if src.kind == "physio":
        # An analog trigger: rises through its midpoint, not every change of
        # value (find_events found 169 events in 30 noisy pulses).
        found = []
        for i, kind in enumerate(src.ch_types):
            if kind != "stim":
                continue
            values = src.raw.get_data(picks=[i])[0]
            missing = src.gaps[i] if src.gaps is not None else None
            found += edge_events(values, missing, src.sfreq, src.ch_names[i])
        src.events_stim = sorted(found, key=lambda e: e.onset)
    else:
        src.events_stim = stim_events(src.raw, src.ch_types)
    src.events_annotations = annotation_events(src.raw)
    if sibling is not None:
        src.bad_spans = [e for e in src.events_tsv if e.kind == "bad"]
    else:
        src.bad_spans = [Event(e.onset + src.start_time, e.duration, e.label, "bad")
                         for e in src.events_annotations if e.kind == "bad"]
    bads = set(src.raw.info.get("bads") or [])
    tsv = channels_sibling(src.path)
    if tsv is not None:
        listed = read_bad_channels(tsv)
        if listed:
            src.bads_from = tsv.name
        bads |= listed
    src.bads = {b for b in bads if b in src.raw.ch_names}
    return src


def open_meeg(path, root=None, *, cancel=None) -> SignalSource:
    """A MEG/EEG/iEEG recording, preloaded, with its events."""
    raw = read_recording(path, preload=True)
    if cancel is not None and cancel():
        from .volume import StreamCancelled

        raise StreamCancelled(str(path))
    src = SignalSource(path=Path(path), raw=raw, kind="meeg", paths=[Path(path)])
    return _with_events(src, root)


def _eye_events(paths, origin: float) -> list:
    """The blinks and messages of the eye trackers among ``paths``, on the
    combined clock (which starts at ``origin``, run time)."""
    from . import physio
    from .events import Event

    out = []
    for p in paths:
        timing = physio.read_timing(p)
        if timing is None or timing.get("physio_type") != "eyetrack":
            continue
        columns, _total, step = physio.read_columns(p)
        _cols, timing, stamps = physio.split_timestamps(columns, timing)
        shift = float(timing.get("start_time", 0.0)) - origin
        out += [Event(e.onset + shift, e.duration, e.label, e.kind)
                for e in physio.eyetrack_events(p, stamps, float(timing["sampling_frequency"]),
                                                step)]
    return out


def open_physio(path, root=None, *, together: bool = False) -> SignalSource:
    """A continuous BIDS table (physio, stim), or every physio file of the
    run on one clock (``together``)."""
    from . import physio

    path = Path(path)
    if together:
        siblings = physio.related_recordings(path)
        if len(siblings) > 1:
            raw, gaps, start = physio.build_combined_raw(siblings)
            src = SignalSource(path=path, raw=raw, kind="physio", gaps=gaps,
                               start_time=start, paths=siblings)
            src = _with_events(src, root)
            src.events_annotations = list(src.events_annotations) + _eye_events(
                siblings, start)
            return src
    timing = physio.read_timing(path)
    if timing is None:
        raise ValueError(f"{path.name} has no sidecar stating SamplingFrequency "
                         "and Columns, so it is a table, not a recording")
    columns, total, step = physio.read_columns(path)
    if not columns:
        raise ValueError(f"no numeric columns could be read from {path.name}")
    columns, timing, stamps = physio.split_timestamps(columns, timing)
    raw, gaps = physio.build_raw(columns, timing, step)
    note = ""
    if step > 1:
        note = (f"{total:,} samples held as {raw.n_times:,}: the recording was "
                f"read one sample in {step} to stay within memory")
    src = SignalSource(path=path, raw=raw, kind="physio", gaps=gaps,
                       start_time=float(timing.get("start_time", 0.0)),
                       paths=[path], note=note)
    src = _with_events(src, root)
    if stamps is not None:
        # An eye tracker's blinks and the experiment's messages.
        src.events_annotations = list(src.events_annotations) + physio.eyetrack_events(
            path, stamps, float(timing["sampling_frequency"]), step)
    return src


def resampled(src: SignalSource, sfreq: float) -> SignalSource:
    """A new source at ``sfreq``, the gap mask carried along.

    The mask used to keep its old shape after a resample, and when the new
    length happened to match it blanked the wrong samples.
    """
    raw = src.raw.copy().resample(float(sfreq), verbose=False)
    gaps = None
    if src.gaps is not None and src.n_times:
        # Nearest original sample of every new sample.
        idx = np.clip(np.round(np.arange(raw.n_times) * (src.sfreq / float(raw.info["sfreq"])))
                      .astype(int), 0, src.n_times - 1)
        gaps = src.gaps[:, idx]
    out = SignalSource(path=src.path, raw=raw, kind=src.kind, gaps=gaps,
                       start_time=src.start_time, paths=list(src.paths),
                       note=src.note, events_tsv=list(src.events_tsv),
                       events_stim=list(src.events_stim),
                       events_annotations=list(src.events_annotations),
                       bads=set(src.bads), bads_from=src.bads_from,
                       bad_spans=list(src.bad_spans))
    return out


__all__ = [
    "SignalSource", "channel_types", "channels_sibling", "is_ctf", "open_meeg",
    "open_physio", "read_bad_channels", "read_recording", "resampled", "summarize",
]

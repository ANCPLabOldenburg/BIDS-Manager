"""Filtering a signal for display: ONE implementation, every cut-off checked.

Zero-phase high-pass, low-pass and notch through ``mne.filter`` (decision D8:
every signal is an mne ``Raw``). Three things the old per-widget code did
not all do:

* every cut-off is checked against the Nyquist frequency and REFUSED with a
  sentence (a 600 Hz low-pass on a 1000 Hz recording, a 60 Hz notch on a
  50 Hz respiratory belt). The old view swallowed MNE's exception and the
  status bar said "Filter applied";
* a filter is never applied to a stretch too short to resolve it: a 0.01 Hz
  high-pass needs about 330 s of signal. Segments are PADDED, within a
  sample budget, and a cut-off the recording cannot resolve at all is
  refused rather than drawn as ringing;
* the gaps go back in AFTER filtering, and before it they are BRIDGED (a
  straight line across), never left as zeros: a 0.5 Hz high-pass over a
  zero-filled 2 s gap in an ECG rang twelve times its signal into the
  samples around it;
* trigger and status channels (``stim``, ``syst``, ``ias``, ``chpi``) are
  never filtered: a filtered TTL rings, and its edges are what it is for;
* a whole recording can be filtered ONCE (:func:`filter_recording`, on a
  worker) when it fits in memory, so scrolling is slicing; filtering each
  window on the GUI thread cost a second a page at 306 channels.

Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Samples a single redraw may read for filter padding, across all visible
#: channels. One physio channel can afford the 330 s either side a 0.01 Hz
#: high-pass wants; three hundred MEG channels cannot.
SAMPLE_BUDGET = 4_000_000

#: MNE sizes its FIR kernel at about this many periods of the cut-off.
_PERIODS = 3.3

#: Channel types a display filter never touches: their edges are the data.
NOT_FILTERED = frozenset({"stim", "syst", "ias", "chpi", "exci"})

#: The most a filtered copy of a whole recording may take (float32).
FILTERED_COPY_BYTES = 768 * 1024 * 1024

#: Band-passes people reach for, by kind of recording: ``(label, hp, lp)``.
#: A preset sets the two cut-offs and leaves the notch alone, because the
#: line frequency is the site's, not the analysis's.
PRESETS: dict[str, tuple[tuple[str, Optional[float], Optional[float]], ...]] = {
    "meeg": (
        ("0.1 to 40 Hz", 0.1, 40.0),
        ("1 to 40 Hz", 1.0, 40.0),
        ("0.5 to 30 Hz", 0.5, 30.0),
        ("1 to 100 Hz", 1.0, 100.0),
        ("High-pass 1 Hz", 1.0, None),
        ("Low-pass 40 Hz", None, 40.0),
    ),
    "physio": (
        ("Breathing: 0.05 to 1 Hz", 0.05, 1.0),
        ("Pulse: 0.5 to 10 Hz", 0.5, 10.0),
        ("ECG: 0.5 to 40 Hz", 0.5, 40.0),
        ("Drift: high-pass 0.01 Hz", 0.01, None),
    ),
}


def preset_of(kind: str, hp: Optional[float], lp: Optional[float]) -> Optional[str]:
    """The label of the preset ``(hp, lp)`` is, if it is one."""
    for label, p_hp, p_lp in PRESETS.get(kind, ()):
        if p_hp == hp and p_lp == lp:
            return label
    return None


@dataclass(frozen=True)
class FilterSpec:
    hp: Optional[float] = None
    lp: Optional[float] = None
    notch: Optional[float] = None

    @property
    def active(self) -> bool:
        return bool(self.hp or self.lp or self.notch)

    def describe(self) -> str:
        parts = []
        if self.hp:
            parts.append(f"high-pass {self.hp:g} Hz")
        if self.lp:
            parts.append(f"low-pass {self.lp:g} Hz")
        if self.notch:
            parts.append(f"notch {self.notch:g} Hz")
        return ", ".join(parts) or "no filter"


def validate(spec: FilterSpec, sfreq: float) -> FilterSpec:
    """``spec`` with zeros as None, or ``ValueError`` saying why it cannot
    be applied to a recording sampled at ``sfreq``."""
    nyq = float(sfreq) / 2.0
    hp = spec.hp if spec.hp and spec.hp > 0 else None
    lp = spec.lp if spec.lp and spec.lp > 0 else None
    notch = spec.notch if spec.notch and spec.notch > 0 else None
    for name, value in (("high-pass", hp), ("low-pass", lp), ("notch", notch)):
        if value is not None and value >= nyq:
            raise ValueError(
                f"A {value:g} Hz {name} is at or above the Nyquist frequency of "
                f"this recording ({nyq:g} Hz, half its {sfreq:g} Hz sampling "
                f"rate): there is nothing at {value:g} Hz to filter.")
    if hp is not None and lp is not None and hp >= lp:
        raise ValueError(
            f"The high-pass ({hp:g} Hz) must be below the low-pass ({lp:g} Hz): "
            "together they would remove everything.")
    return FilterSpec(hp, lp, notch)


def pad_samples(spec: FilterSpec, sfreq: float, n_channels: int,
                budget: int = SAMPLE_BUDGET) -> tuple[int, Optional[str]]:
    """Extra samples to read either side so the filter has room, and a
    sentence when the budget cannot afford what the filter needs."""
    if not spec.active:
        return 0, None
    if not spec.hp:
        # A low-pass or a notch settles within a second at these rates.
        return int(sfreq), None
    needed = int(round((_PERIODS / spec.hp) * sfreq))
    channels = max(1, int(n_channels))
    affordable = int(budget / channels)
    if needed <= affordable:
        return needed, None
    resolvable = _PERIODS * sfreq / max(1, affordable)
    return affordable, (
        f"A {spec.hp:g} Hz high-pass needs {needed / sfreq:.0f} s of signal either "
        f"side and {channels} channels only allow {affordable / sfreq:.0f} s. "
        f"Showing the closest this view can resolve (about {resolvable:.3g} Hz); "
        "show fewer channels for the full depth.")


def apply(data: np.ndarray, sfreq: float, spec: FilterSpec) -> tuple[np.ndarray, list[str]]:
    """Filter ``data`` (channels, samples). Returns the data and what the
    reader should be told: a high-pass the stretch is too short for is
    left out and said so; a failure is said, never swallowed."""
    import mne

    messages: list[str] = []
    hp, lp = spec.hp, spec.lp
    n = data.shape[-1]
    if hp:
        needed = int(round((_PERIODS / hp) * sfreq))
        if n < needed:
            messages.append(
                f"A {hp:g} Hz high-pass needs {needed / sfreq:.0f} s of signal and "
                f"{n / sfreq:.1f} s is available. The high-pass was not applied.")
            hp = None
    out = data
    if hp or lp:
        try:
            out = mne.filter.filter_data(out, sfreq, hp, lp, verbose=False)
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            messages.append(f"The filter could not be applied: {exc}")
    if spec.notch:
        try:
            out = mne.filter.notch_filter(out, sfreq, spec.notch, verbose=False)
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            messages.append(f"The {spec.notch:g} Hz notch could not be applied: {exc}")
    return out, messages


@dataclass
class Segment:
    times: np.ndarray        # seconds of RECORDING time
    data: np.ndarray         # (channels, samples), NaN where the recording has none
    messages: list[str]


@dataclass
class FilteredCopy:
    """A whole recording filtered once: what scrolling slices."""

    spec: FilterSpec
    #: Row of each filtered channel index.
    rows: dict
    data: np.ndarray         # (filtered channels, samples), float32
    messages: list


def bridge(data: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """``data`` with every masked sample replaced by a straight line between
    the samples either side (the first and last held), in place: what a
    filter is given instead of the zeros the gaps hold."""
    if mask is None or not mask.any():
        return data
    idx = np.arange(data.shape[-1])
    for row, gaps in zip(data, mask):
        if not gaps.any():
            continue
        good = ~gaps
        if not good.any():
            row[:] = 0.0
            continue
        row[gaps] = np.interp(idx[gaps], idx[good], row[good])
    return data


def filterable(src, indices: list[int]) -> list[bool]:
    return [src.ch_types[i] not in NOT_FILTERED for i in indices]


def fits_in_memory(src, spec: FilterSpec, limit: int = FILTERED_COPY_BYTES) -> bool:
    n = sum(1 for t in src.ch_types if t not in NOT_FILTERED)
    return spec.active and n * src.n_times * 4 <= limit


def filter_recording(src, spec: FilterSpec, *, cancel=None) -> FilteredCopy:
    """Filter every filterable channel of ``src`` once, in blocks of 32
    channels, gaps bridged first. Runs on a worker (a QThread: scipy)."""
    spec = validate(spec, src.sfreq)
    idx = [i for i, t in enumerate(src.ch_types) if t not in NOT_FILTERED]
    out = np.empty((len(idx), src.n_times), dtype=np.float32)
    messages: list[str] = []
    for start in range(0, len(idx), 32):
        if cancel is not None and getattr(cancel, "cancelled", False):
            raise RuntimeError("cancelled")
        block = idx[start:start + 32]
        data = src.read(block, 0, src.n_times)
        bridge(data, src.gap_mask(block, 0, src.n_times))
        data, notes = apply(data, src.sfreq, spec)
        for note in notes:
            if note not in messages:
                messages.append(note)
        out[start:start + len(block)] = data
    return FilteredCopy(spec, {ch: k for k, ch in enumerate(idx)}, out, messages)


def segment(src, indices: list[int], t0: float, t1: float,
            spec: FilterSpec = FilterSpec(), budget: int = SAMPLE_BUDGET,
            copy: Optional[FilteredCopy] = None) -> Optional[Segment]:
    """What a canvas draws: ``[t0, t1)`` of ``indices``, filtered with room
    to settle (or sliced from a ``copy`` filtered once), gaps restored.
    Trigger and status channels are left as recorded. ``src`` is a
    :class:`~..data.signal.SignalSource`."""
    sfreq = src.sfreq
    s0 = max(0, int(t0 * sfreq))
    s1 = min(src.n_times, int(np.ceil(t1 * sfreq)) + 1)
    if s1 <= s0 or not indices:
        return None
    messages: list[str] = []
    use_copy = copy is not None and spec.active and copy.spec == spec
    if use_copy:
        data = src.read(indices, s0, s1)
        for row, ch in enumerate(indices):
            k = copy.rows.get(ch)
            if k is not None:
                data[row] = copy.data[k, s0:s1]
        messages.extend(copy.messages)
    else:
        pad, note = pad_samples(spec, sfreq, len(indices), budget)
        if note:
            messages.append(note)
        p0 = max(0, s0 - pad)
        p1 = min(src.n_times, s1 + pad)
        data = src.read(indices, p0, p1)
        if spec.active:
            which = filterable(src, indices)
            rows = [k for k, ok in enumerate(which) if ok]
            if rows:
                part = data[rows]
                mask = src.gap_mask([indices[k] for k in rows], p0, p1)
                bridge(part, mask if mask is not None and mask.shape == part.shape else None)
                part, notes = apply(part, sfreq, spec)
                data[rows] = part
                messages.extend(notes)
        lo = s0 - p0
        data = data[:, lo:lo + (s1 - s0)]
    mask = src.gap_mask(indices, s0, s1)
    if mask is not None and mask.shape == data.shape and mask.any():
        # In place: ``read`` hands over a copy, and a second full-size array
        # per redraw is what a long physio run cannot afford.
        data[mask] = np.nan
    times = np.arange(s0, s1, dtype=float) / sfreq
    return Segment(times, data, messages)


__all__ = ["FILTERED_COPY_BYTES", "FilterSpec", "FilteredCopy", "NOT_FILTERED",
           "PRESETS", "SAMPLE_BUDGET", "Segment", "apply", "bridge", "filter_recording",
           "filterable", "fits_in_memory", "pad_samples", "preset_of", "segment",
           "validate"]

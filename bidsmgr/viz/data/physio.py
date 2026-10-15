"""A ``*_physio.tsv.gz`` (or ``_stim``, or any continuous BIDS table) as a
signal.

A physio recording is a table of numbers with no header row, and reading one
as a table tells you almost nothing. Whether the trigger fired where you
expect, whether the ECG is flat for the first minute, whether the
respiratory belt came loose halfway through, are questions about the SHAPE
of the signal. So the file is turned into an ``mne.io.RawArray`` (decision
D8: every signal is an mne ``Raw``) and drawn by the same traces canvas, the
same filters and the same spectrum as MEG and EEG.

Three facts the plot needs are not in the file, and BIDS puts all three in
the sidecar, which is why this reads it rather than guessing:

``Columns``            what each column is, since the file has no header
``SamplingFrequency``  how far apart the samples are
``StartTime``          where sample zero sits relative to the run, which is
                       usually NEGATIVE because recording starts before the
                       scanner does

``StartTime`` used to be read and then dropped, so the physio time axis
started at zero and an events overlay drawn on it was off by exactly that
offset. It is now carried on the :class:`~.signal.SignalSource` and the
canvas draws in run time.

Channel TYPES are guessed from the column names: a column called
``cardiac`` becomes an MNE ``ecg`` channel, ``respiratory`` becomes
``resp``, a trigger becomes ``stim``. The viewer then colours them by kind,
groups them in the type filter, averages the spectrum per kind, and reads
events off the stim channel, all without knowing that physio exists.

Qt-free.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

from ..bids import full_ext, sidecar_for
from .events import run_base

log = logging.getLogger(__name__)

#: Beyond this many samples in one channel the read strides rather than
#: keeping everything. A memory bound, not a drawing one: the viewer draws
#: a window at a time. Four million samples costs 16 MB as float32, which at
#: 1000 Hz is over an hour of recording.
_MAX_SAMPLES = 4_000_000

#: Column name (lower-cased, non-letters stripped) -> MNE channel type.
#: Matched as a WHOLE word first and then as a substring, so ``cardiac`` and
#: ``card1`` both land on ``ecg`` while ``discard`` does not.
#:
#: The right-hand side is deliberately MNE's vocabulary rather than ours:
#: the viewer, the colour map and the spectrum all group by it, so inventing
#: a parallel set of names here would mean translating twice.
_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("cardiac", "ecg"),
    ("ecg", "ecg"),
    ("ekg", "ecg"),
    ("pulse", "ecg"),
    ("ppg", "ecg"),
    ("plethysmograph", "ecg"),
    ("respiratory", "resp"),
    ("respiration", "resp"),
    ("resp", "resp"),
    ("breath", "resp"),
    ("belt", "resp"),
    ("trigger", "stim"),
    ("stim", "stim"),
    ("scannertrigger", "stim"),
    ("ttl", "stim"),
    ("eyegaze", "eyegaze"),
    ("xcoordinate", "eyegaze"),
    ("ycoordinate", "eyegaze"),
    ("gaze", "eyegaze"),
    ("pupil", "pupil"),
    ("emg", "emg"),
    ("eog", "eog"),
    ("temperature", "temperature"),
    ("gsr", "gsr"),
    ("skinconductance", "gsr"),
)

#: What a column becomes when nothing matches. ``misc`` rather than ``bio``
#: because MNE treats ``misc`` as "shown, not interpreted", which is exactly
#: the right claim about a column we did not recognise.
_DEFAULT_TYPE = "misc"


def read_timing(path: Path) -> Optional[dict]:
    """``{columns, sampling_frequency, start_time, units}`` or ``None``.

    ``None`` means this is not a continuous recording, which is how the
    caller decides whether to offer a viewer at all. A table of onsets
    (``_events.tsv``) has no sampling frequency and is not one. Deciding on
    the SIDECAR rather than the filename means ``_stim.tsv.gz``, and any
    future continuous suffix, are offered a viewer without this knowing
    they exist.
    """
    sidecar = sidecar_for(path)
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        rate = float(data["SamplingFrequency"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(rate) or rate <= 0:
        return None
    columns = data.get("Columns")
    if not isinstance(columns, list) or not columns:
        # BIDS motion names its columns in the run's ``_channels.tsv``
        # instead (``*_motion.tsv`` has no header and its sidecar no
        # ``Columns``), so that is where they are read from.
        columns = _channel_table_names(path)
        if not columns:
            return None
    try:
        start = float(data.get("StartTime", 0.0))
    except (TypeError, ValueError):
        start = 0.0
    units = data.get("Units")
    return {
        "columns": [str(c) for c in columns],
        "sampling_frequency": rate,
        "start_time": start if math.isfinite(start) else 0.0,
        "units": str(units) if isinstance(units, str) else "",
        "physio_type": str(data.get("PhysioType") or ""),
    }


#: Eye-tracker messages that are its own set-up chatter, not the experiment's.
_TRACKER_MESSAGES = ("!CAL", "VALIDATE", "RECCFG", "ELCL", "GAZE_COORDS", "THRESHOLDS",
                     "CAMERA_LENS", "DISPLAY_COORDS", "!MODE", "PUPIL_DATA_TYPE",
                     "DRIFTCORRECT", "BUTTON", "INPUT")


def split_timestamps(columns: list[np.ndarray], timing: dict):
    """An eye-tracking recording's ``timestamp`` column is its clock, not a
    channel: ``(columns, timing, timestamps)`` without it (``None`` when
    there is none)."""
    names = list(timing.get("columns", []))
    if timing.get("physio_type") != "eyetrack" or "timestamp" not in names:
        return columns, timing, None
    i = names.index("timestamp")
    if i >= len(columns):
        return columns, timing, None
    stamps = np.asarray(columns[i], dtype=np.float64)
    rest = [c for k, c in enumerate(columns) if k != i]
    timing = {**timing, "columns": [n for k, n in enumerate(names) if k != i]}
    return rest, timing, stamps


def eyetrack_events(path: Path, stamps: np.ndarray, sfreq: float, step: int = 1) -> list:
    """Blinks and the experiment's messages from the recording's
    ``_physioevents.tsv.gz``, in seconds from its first sample, placed on the
    SAMPLES (a pause between trials is not drawn, so an event after one sits
    where its sample is drawn, not where the clock says)."""
    import csv
    import gzip

    from .events import Event

    events_path = Path(path).with_name(Path(path).name.replace("_physio.", "_physioevents."))
    if not events_path.is_file() or stamps is None or not len(stamps):
        return []
    out = []
    try:
        with gzip.open(events_path, "rt", encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if len(row) < 4:
                    continue
                try:
                    onset = float(row[0])
                except ValueError:
                    continue
                kind, message = row[2], row[3]
                if kind == "blink":
                    label = "blink"
                elif kind in ("n/a", "") and message not in ("n/a", "") and \
                        not message.startswith(_TRACKER_MESSAGES):
                    label = message
                else:
                    continue
                try:
                    duration = max(0.0, float(row[1]))
                except ValueError:
                    duration = 0.0
                index = int(np.clip(np.searchsorted(stamps, onset), 0, len(stamps) - 1))
                out.append(Event(index * step / sfreq, duration, label))
    except (OSError, ValueError) as exc:
        log.debug("could not read %s: %s", events_path, exc)
    return out


def _channel_table_names(path: Path) -> list[str]:
    """The ``name`` column of the ``_channels.tsv`` that shares ``path``'s
    entities, or ``[]``."""
    import csv

    from ..bids import stem_of

    stem = stem_of(Path(path))
    if "_" not in stem:
        return []
    table = Path(path).with_name(stem.rsplit("_", 1)[0] + "_channels.tsv")
    try:
        with open(table, encoding="utf-8-sig", newline="") as fh:
            return [str(row["name"]) for row in csv.DictReader(fh, delimiter="\t")
                    if row.get("name")]
    except (OSError, KeyError, csv.Error):
        return []


def guess_channel_type(name: str) -> str:
    """The MNE channel type a physio column name implies.

    Whole-word match first, then substring, so ``cardiac`` and ``card_1``
    both reach ``ecg`` while ``discard`` reaches neither.
    """
    token = re.sub(r"[^a-z]", "", str(name).lower())
    if not token:
        return _DEFAULT_TYPE
    for needle, ch_type in _TYPE_HINTS:
        if token == needle:
            return ch_type
    for needle, ch_type in _TYPE_HINTS:
        if needle in token:
            return ch_type
    return _DEFAULT_TYPE


def read_columns(
    path: Path, limit: int = _MAX_SAMPLES,
) -> tuple[list[np.ndarray], int, int]:
    """``(columns, total_samples, step)``: the whole file, and what it took.

    Reads the WHOLE file rather than the preview the table shows. The table
    is bounded at five thousand rows because nobody reads more than that; a
    view of the first five thousand samples of a 1.4-million-sample trigger
    channel would be a picture of the first four seconds, drawn as though it
    were the recording. Silently.

    ``step`` is 1 unless the file is longer than ``limit``, at which point
    the read strides and each kept point stands for ``step`` real samples.
    Both numbers are returned because both are needed to say anything true
    about the result: ``total`` keeps the caption honest and ``step`` keeps
    the SAMPLING RATE honest, since a strided recording is a slower one.

    Parsed with pandas' C engine, which releases the GIL: 1.4 million rows
    read in 46 ms. Run on a worker anyway, because "fast on the files we
    happened to test" is how a freeze gets shipped.
    """
    import pandas as pd

    try:
        frame = pd.read_csv(
            path, sep="\t", header=None, engine="c",
            compression="gzip" if str(path).lower().endswith(".gz") else "infer",
            na_values=["n/a", "N/A", ""],
        )
    except Exception as exc:  # noqa: BLE001 - a bad file must not crash the pane
        log.warning("could not read %s for viewing: %s", path, exc)
        return [], 0, 1

    total = int(len(frame))
    step = max(1, math.ceil(total / limit)) if total > limit else 1
    out: list[np.ndarray] = []
    for name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=np.float32,
        )
        out.append(values[::step] if step > 1 else values)
    return out, total, step


def _unique(names: list[str]) -> list[str]:
    """MNE refuses duplicate channel names, and a sidecar is free to repeat
    one: the second ``ecg`` becomes ``ecg-1``."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}-{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def build_raw(columns: list[np.ndarray], timing: dict, step: int = 1):
    """``(RawArray, gaps)`` from physio columns, typed by column name.

    ``step`` divides the sampling rate, because a strided read really is a
    slower recording and every frequency the viewer computes depends on
    getting that right.

    **The gaps are returned, not discarded.** MNE cannot carry NaN through a
    filter or an FFT, so the array holds zeros where the recording has no
    sample and the mask says where they are. The viewer then filters a
    continuous signal and DRAWS the gaps as breaks.

    That matters more than it sounds. A real ECG in this lab's own data is
    28 percent gaps, because the scanner dropped samples: filling them with
    zero and drawing the result puts a spike to the floor of the plot at
    every one, on a trace centred near 2050. It was reported as the viewer
    showing artefacts, and it was.
    """
    import mne

    names = [str(c) for c in timing.get("columns", [])]
    n = min(len(names), len(columns))
    if n == 0:
        raise ValueError("no physio columns to show")
    names = names[:n]
    unique = _unique(names)

    rate = float(timing.get("sampling_frequency", 1.0)) or 1.0
    rate = rate / max(1, int(step))
    types = [guess_channel_type(name) for name in names]
    stacked = np.vstack([
        np.asarray(columns[i], dtype=np.float64) for i in range(n)
    ])
    gaps = ~np.isfinite(stacked)
    data = np.where(gaps, 0.0, stacked)
    info = mne.create_info(unique, sfreq=rate, ch_types=types, verbose=False)
    return mne.io.RawArray(data, info, verbose=False), gaps


def related_recordings(path: Path) -> list[Path]:
    """Every physio recording of the same run, this one first.

    BIDS splits a run's physio by the ``recording`` entity: the cardiac
    trace, the respiratory belt and the trigger are three files describing
    one acquisition. Reading them apart is reading a three-channel
    recording one channel at a time, and the question people bring to
    physio (did the trigger fire where the ECG says it should) cannot be
    answered that way at all.

    Matched on the basename with the ``recording`` entity removed, so it is
    the run that groups them and not the folder: two runs in the same
    ``func/`` do not get mixed.
    """
    path = Path(path)
    mine = run_base(path)
    found = [
        sibling for sibling in sorted(path.parent.glob("*_physio.tsv*"))
        if full_ext(sibling) in (".tsv", ".tsv.gz") and run_base(sibling) == mine
    ]
    # The one that was clicked comes first, so it keeps the colour and the
    # position a reader already has in their head.
    found.sort(key=lambda q: (q != path, q.name))
    return found


def build_combined_raw(paths: list[Path]):
    """``(RawArray, gaps, start_time)`` holding every recording in ``paths``.

    They are resampled onto the FASTEST one's grid, because that is the
    only rate at which nothing is thrown away, and a trigger sampled at
    1000 Hz beside a belt at 50 is exactly the case this exists for.
    Upsampling is nearest-sample, not interpolation: inventing values
    between two samples of a trigger would invent edges.

    Each recording keeps its own StartTime by being placed at its own
    offset on the shared clock, which is what makes the comparison mean
    anything: physio starts before the scanner does, and by a different
    amount per device.
    """
    import mne

    loaded = []
    for path in paths:
        timing = read_timing(path)
        if timing is None:
            continue
        columns, _total, step = read_columns(path)
        if not columns:
            continue
        # An eye tracker's timestamp is its clock, not a channel.
        columns, timing, _stamps = split_timestamps(columns, timing)
        rate = (float(timing["sampling_frequency"]) or 1.0) / max(1, step)
        loaded.append((path, timing, columns, rate))
    if not loaded:
        raise ValueError("none of these files could be read as a recording")

    target = max(rate for _p, _t, _c, rate in loaded)
    starts = [t["start_time"] for _p, t, _c, _r in loaded]
    origin = min(starts)
    # How long the shared clock has to be to hold all of them.
    span = max(
        (t["start_time"] - origin) + len(c[0]) / r
        for _p, t, c, r in loaded
    )
    n_out = max(1, int(round(span * target)))

    names, types, rows, gaps = [], [], [], []
    for path, timing, columns, rate in loaded:
        offset = int(round((timing["start_time"] - origin) * target))
        for index, column in enumerate(columns):
            label = (
                timing["columns"][index]
                if index < len(timing["columns"]) else f"{path.stem}-{index}"
            )
            # Nearest-sample placement onto the shared grid. Beyond its own
            # last sample a recording has NO samples: a gap, never its last
            # value repeated to the end of the clock (which drew a flat line
            # across the other recordings' remaining minutes).
            source = (np.arange(n_out - offset) * (rate / target)).astype(int)
            inside = source < len(column)
            stretch = np.full(n_out - offset, np.nan)
            stretch[inside] = column[source[inside]]
            placed = np.full(n_out, np.nan)
            placed[offset:] = stretch
            rows.append(np.nan_to_num(placed, nan=0.0))
            gaps.append(~np.isfinite(placed))
            names.append(label)
            types.append(guess_channel_type(label))

    info = mne.create_info(_unique(names), sfreq=target, ch_types=types, verbose=False)
    raw = mne.io.RawArray(np.vstack(rows), info, verbose=False)
    # Sample zero of the shared clock is the EARLIEST start: that is the
    # combined recording's StartTime.
    return raw, np.vstack(gaps), float(origin)


# ---------------------------------------------------------------------------
# Events and gaps: what the samples MEAN, for drawing them honestly
# ---------------------------------------------------------------------------

#: A channel missing at least this often, its values only at isolated
#: samples, marks EVENTS (a scanner trigger, a detected heartbeat), not a
#: waveform: physio from a scanner log writes a value where the trigger fired
#: and nothing in between. Drawn as a line it is a row of dots and dashes.
EVENT_MISSING_FRACTION = 0.8
#: Gaps this short in a waveform are dropped samples (the ds_4 ECG has 6381
#: of them, one or two samples each). They are bridged for DISPLAY, or the
#: trace is drawn as thousands of fragments; longer gaps stay visible.
BRIDGE_SECONDS = 0.05


def channel_role(values, missing=None, ch_type: str = "") -> str:
    """``waveform``, ``events`` or ``empty``.

    ``events`` is a channel present at under a fifth of its samples, or a
    stim channel with at most three levels (a TTL line: 0 and 5 V).
    """
    v = np.asarray(values, dtype=float)
    miss = np.asarray(missing, dtype=bool) if missing is not None else ~np.isfinite(v)
    present = ~miss
    n_present = int(present.sum())
    if n_present == 0:
        return "empty"
    if 1.0 - n_present / max(v.size, 1) >= EVENT_MISSING_FRACTION:
        return "events"
    if ch_type == "stim":
        kept = v[present]
        if kept.size > 200_000:
            kept = kept[:: kept.size // 200_000]
        if np.unique(kept).size <= 3:
            return "events"
    return "waveform"


def event_onsets(values, missing, seconds) -> tuple[np.ndarray, np.ndarray]:
    """``(times, values)`` of the events in an ``events`` channel.

    A sparse channel: the first sample of every run of present samples (a
    trigger sampled at 50 Hz and placed on a 400 Hz clock is eight samples,
    one event). A level channel: every rise through the midpoint.
    """
    v = np.asarray(values, dtype=float)
    miss = np.asarray(missing, dtype=bool) if missing is not None else ~np.isfinite(v)
    t = np.asarray(seconds, dtype=float)
    present = ~miss
    if not present.any():
        return np.empty(0), np.empty(0)
    if 1.0 - present.mean() >= EVENT_MISSING_FRACTION:
        starts = present & ~np.r_[False, present[:-1]]
    else:
        kept = v[present]
        lo, hi = float(kept.min()), float(kept.max())
        if not hi > lo:
            return np.empty(0), np.empty(0)
        above = present & (v > (lo + hi) / 2.0)
        starts = above & ~np.r_[False, above[:-1]]
    return t[starts], v[starts]


def bridge_gaps(values, missing, max_samples: int) -> tuple[np.ndarray, np.ndarray, int]:
    """``(values, still_missing, n_bridged)``: runs of at most
    ``max_samples`` missing samples, with a sample on both sides, filled by a
    straight line between those two; everything else missing stays NaN."""
    v = np.array(values, dtype=float, copy=True)
    miss = np.asarray(missing, dtype=bool) if missing is not None else ~np.isfinite(v)
    v[miss] = np.nan
    if max_samples < 1 or not miss.any() or miss.all():
        return v, miss.copy(), 0
    edges = np.diff(np.r_[0, miss.astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    short = ((ends - starts) <= max_samples) & (starts > 0) & (ends < v.size)
    if not short.any():
        return v, miss.copy(), 0
    mark = np.zeros(v.size + 1, dtype=np.int32)
    np.add.at(mark, starts[short], 1)
    np.add.at(mark, ends[short], -1)
    fill = np.cumsum(mark[:-1]) > 0
    idx = np.arange(v.size)
    good = ~miss
    v[fill] = np.interp(idx[fill], idx[good], v[good])
    return v, miss & ~fill, int(short.sum())


def describe_events(times) -> str:
    """"428 marks, 2.25 s apart (median)"."""
    t = np.asarray(times, dtype=float)
    if t.size == 0:
        return "no marks"
    if t.size == 1:
        return "1 mark"
    gap = float(np.median(np.diff(np.sort(t))))
    return f"{t.size:,} marks, {gap:.3g} s apart (median)"


__all__ = [
    "BRIDGE_SECONDS", "EVENT_MISSING_FRACTION", "bridge_gaps", "channel_role",
    "describe_events", "event_onsets",
    "build_combined_raw",
    "build_raw",
    "guess_channel_type",
    "read_columns",
    "read_timing",
    "related_recordings",
]

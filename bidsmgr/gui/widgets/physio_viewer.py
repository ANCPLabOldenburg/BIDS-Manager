"""A ``*_physio.tsv.gz`` seen as a signal, through the shared viewer.

A physio recording is a table of numbers with no header row, and reading one
as a table tells you almost nothing. Whether the trigger fired where you
expect, whether the ECG is flat for the first minute, whether the
respiratory belt came loose halfway through, are all questions about the
SHAPE of the signal, and none of them can be answered from a grid of
six-decimal numbers.

**It is not a second viewer.** The file is turned into an ``mne.io.RawArray``
and handed to :class:`~bidsmgr.gui.widgets.time_series_view.TimeSeriesView`,
the same widget the MEG/EEG pane uses. A physio recording is channels, a
sampling rate and samples, which is exactly what a ``RawArray`` is made of,
so the conversion costs a few lines and buys the channel picker, the
navigation, the amplitude and window controls, zero-phase filtering,
resampling, the interactive spectrum and the events overlay, all of them
already written and already tested. The alternative was a second
implementation of each, drifting from the first one feature at a time.

Three facts the plot needs are not in the file, and BIDS puts all three in
the sidecar, which is why this reads it rather than guessing:

``Columns``            what each column is, since the file has no header
``SamplingFrequency``  how far apart the samples are
``StartTime``          where sample zero sits relative to the run, which is
                       usually NEGATIVE because recording starts before the
                       scanner does

Channel TYPES are guessed from the column names, and that is what makes the
shared viewer useful here rather than merely possible: a column called
``cardiac`` becomes an MNE ``ecg`` channel, ``respiratory`` becomes ``resp``,
a trigger becomes ``stim``. The viewer then colours them by kind, groups
them in the type filter, averages the spectrum per kind, and reads events
off the stim channel, all without knowing that physio exists.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

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


def sidecar_for(path: Path) -> Path:
    """The ``.json`` beside a ``.tsv`` or ``.tsv.gz``."""
    name = path.name
    for ext in (".tsv.gz", ".tsv"):
        if name.endswith(ext):
            return path.with_name(name[: -len(ext)] + ".json")
    return path.with_suffix(".json")


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
    }


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
    # MNE refuses duplicate channel names, and a sidecar is free to repeat one.
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}-{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)

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
    stem = path.name
    for ext in (".tsv.gz", ".tsv"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    def key(name: str) -> str:
        return "_".join(
            part for part in name.split("_")
            if not part.startswith("recording-")
        )

    mine = key(stem)
    found = []
    for sibling in sorted(path.parent.glob("*_physio.tsv*")):
        if sibling.suffix not in (".gz",) and not sibling.name.endswith(".tsv"):
            continue
        name = sibling.name
        for ext in (".tsv.gz", ".tsv"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        if key(name) == mine:
            found.append(sibling)
    # The one that was clicked comes first, so it keeps the colour and the
    # position a reader already has in their head.
    found.sort(key=lambda q: (q != path, q.name))
    return found


def build_combined_raw(paths: list[Path]):
    """``(RawArray, gaps)`` holding every recording in ``paths``.

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
            # Nearest-sample placement onto the shared grid.
            source = np.arange(n_out - offset) * (rate / target)
            source = np.clip(source.astype(int), 0, len(column) - 1)
            placed = np.full(n_out, np.nan)
            placed[offset:] = column[source]
            rows.append(np.nan_to_num(placed, nan=0.0))
            gaps.append(~np.isfinite(placed))
            names.append(label)
            types.append(guess_channel_type(label))

    seen: dict[str, int] = {}
    unique = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}-{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)

    info = mne.create_info(unique, sfreq=target, ch_types=types, verbose=False)
    return mne.io.RawArray(np.vstack(rows), info, verbose=False), np.vstack(gaps)


__all__ = [
    "build_combined_raw",
    "build_raw",
    "related_recordings",
    "guess_channel_type",
    "read_columns",
    "read_timing",
    "sidecar_for",
]

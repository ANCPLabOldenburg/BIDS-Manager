"""Siemens CMRR physiological logs, read and written as BIDS tables.

The CMRR multiband sequences log the physiology of a run in plain text, one
log per kind of signal, and store the logs together in one ``_PhysioLog``
DICOM (private tag ``7fe1,1010``) or as ``.log`` files:

- ``PULS``: pulse oximetry, one channel;
- ``RESP``: respiratory belt, one channel;
- ``ECG``: up to four leads, ``ECG1`` to ``ECG4``, interleaved line by line;
- ``EXT``: external inputs, ``EXT1`` and ``EXT2``;
- ``ACQUISITION_INFO`` (``_Info.log``): when every slice of every volume was
  acquired.

Every sample carries its time in scanner ticks of 2.5 ms, so each channel is
placed on an exact grid at its own rate, a missing sample left ``n/a``.

Read here rather than with the vendored ``bidsphysio`` because that parser
put every line whose channel name contains ``ECG`` into one list: four leads
became one column four times too long, with the leads interleaved. And its
trigger was a table of its own, sampled only at the volume starts and filled
in between with a missing value that a boolean array turns into ``True``: a
constant 1, a flat line. Measured on the multimodal sample (2026-10-22): ECG
146,944 rows at 400 Hz (367 s) for a 92 s run; trigger 84,151 rows, all 1.

Written (:func:`write_tables`), the scanner trigger is a ``trigger`` column,
1 at the first sample of each volume of the run's image and 0 elsewhere, the
LAST column of every table. The standard asks ONE sampling frequency per
file, and the channels are recorded at several (ECG 400 Hz, pulse 200 Hz,
respiration 50 Hz), so there are three layouts, all BIDS compliant:

- ``by_rate`` (default): one table per sampling rate, the samples exactly as
  recorded (the four ECG leads together, pulse and respiration apart).
- ``one_interpolated``: one table for the run at the fastest rate, slower
  channels linearly interpolated between their samples (said so per column).
- ``one_as_recorded``: one table for the run at the fastest rate, every sample
  of a slower channel in the row of its own time and ``n/a`` in the rows
  between (a 50 Hz channel fills every eighth row of a 400 Hz table): nothing
  computed, nothing invented. Each sample is in its exact row when the rates
  share the scanner's tick grid, which CMRR rates do; otherwise in the nearest
  row, and its column's description says how far that is at most.

A log with no signal at all (logging on, no sensor attached) still says
when each volume was acquired: it is written as ONE table holding the trigger
column alone, on the scanner's clock (400 Hz, one row per tick), from the
first volume to the end of the last. 1.5.1 wrote a trigger of 1 in every row
for such a log, and 1.5.2 at first wrote nothing.

``align`` crops the tables to the run: from the first volume of the image to
the end of its last volume, so ``StartTime`` is 0 (to within one sample).
``drop_volumes`` is the number of volumes removed from the start of the image
(dummy scans): the run, ``StartTime`` and the trigger column then start at the
first volume kept. Measured on nine real runs (two scanner generations, 50 to
1,225 volumes): volume i of the image is volume i of the log, to the
millisecond, so removing N from both keeps them together. A run stopped by
hand ends in a volume the scanner did not finish: the log lists it, the
image does not (dcm2niix writes it apart), so it is not a volume of the run
here either and the trigger counts exactly the image's volumes.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

#: One scanner tick, in seconds.
TICK = 0.0025

LAYOUT_BY_RATE = "by_rate"
LAYOUT_ONE_INTERPOLATED = "one_interpolated"
LAYOUT_ONE_AS_RECORDED = "one_as_recorded"
LAYOUTS = (LAYOUT_BY_RATE, LAYOUT_ONE_INTERPOLATED, LAYOUT_ONE_AS_RECORDED)

#: Log type -> (column label, what it is).
_KINDS = {
    "PULS": ("cardiac", "Pulse oximetry"),
    "RESP": ("respiratory", "Respiratory belt"),
    "ECG": ("ecg", "ECG"),
    "EXT": ("external_trigger", "External input"),
}

#: Column order in a table: what a reader looks for first, the trigger last.
_ORDER = ("cardiac", "respiratory", "ecg", "external_trigger")

_ROWS_PER_CHUNK = 100_000


@dataclass
class Channel:
    """One recorded channel on its own grid."""

    name: str              # the scanner's (ECG2, PULS)
    label: str             # the column (ecg2, cardiac)
    kind: str              # the log type (ECG, PULS, RESP, EXT)
    rate: float            # samples per second
    start: float           # time of the first sample, seconds on the scanner clock
    values: np.ndarray     # float, NaN where a sample is missing

    @property
    def end(self) -> float:
        """Time of the last sample."""
        return self.start + (len(self.values) - 1) / self.rate


@dataclass
class Volume:
    """When one volume of the run was acquired (first echo)."""

    index: int
    start: float
    finish: float


@dataclass
class CmrrPhysio:
    """Everything one physio DICOM (or set of logs) holds."""

    channels: list[Channel] = field(default_factory=list)
    volumes: list[Volume] = field(default_factory=list)
    uuid: str = ""
    scan_date: str = ""
    #: Volumes the scanner started and did not finish when the run was
    #: stopped: in the log, not in the image (:func:`_volumes`).
    unfinished: int = 0


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def logs_in_dicom(path: Path) -> list[tuple[str, str]]:
    """``(file name, text)`` of every log a CMRR physio DICOM carries."""
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    if (0x7FE1, 0x1010) not in ds:
        raise ValueError(f"{Path(path).name} holds no CMRR physiological log "
                         "(private tag 7fe1,1010 is missing)")
    data = bytes(ds[0x7FE1, 0x1010].value)
    rows = int(ds.get("AcquisitionNumber", 0) or 0)
    if rows <= 0 or len(data) % rows or (len(data) // rows) % 1024:
        raise ValueError(f"{Path(path).name}: the physiological log is not laid out "
                         "as a CMRR sequence writes it")
    waves = (len(data) // rows) // 1024
    length = len(data) // waves
    out = []
    for w in range(waves):
        block = data[w * length:(w + 1) * length]
        size = int.from_bytes(block[0:4], sys.byteorder)
        name_size = int.from_bytes(block[4:8], sys.byteorder)
        name = block[8:8 + name_size].decode("utf-8", "ignore").rstrip("\x00")
        text = block[1024:1024 + size].decode("utf-8", "ignore")
        out.append((name, text))
    return out


def _header(lines: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        key, eq, value = line.partition("=")
        if eq and key.strip() and " " not in key.strip():
            out[key.strip()] = value.strip()
    return out


_INT = re.compile(r"^-?\d+$")


def parse_log(text: str) -> tuple[dict[str, str], dict[str, list[tuple[int, float]]],
                                  list[tuple[int, int, int, int, int]]]:
    """``(header, samples by channel, acquisition rows)`` of one log.

    Samples are ``(tick, value)``; acquisition rows (an ``_Info`` log) are
    ``(volume, slice, start tick, finish tick, echo)``.
    """
    lines = text.splitlines()
    header = _header(lines)
    info = header.get("LogDataType", "") == "ACQUISITION_INFO"
    samples: dict[str, list[tuple[int, float]]] = {}
    rows: list[tuple[int, int, int, int, int]] = []
    for line in lines:
        parts = line.split()
        if not parts or not _INT.match(parts[0]):
            continue
        if info:
            if len(parts) >= 5 and all(_INT.match(p) for p in parts[:5]):
                rows.append(tuple(int(p) for p in parts[:5]))   # type: ignore[arg-type]
            continue
        if len(parts) < 3:
            continue
        try:
            value = float(parts[2])
        except ValueError:
            continue
        samples.setdefault(parts[1], []).append((int(parts[0]), value))
    return header, samples, rows


def _label(kind: str, name: str, several: bool) -> str:
    base = _KINDS.get(kind, (re.sub(r"[^0-9a-z]+", "", kind.lower()) or "signal", ""))[0]
    if not several:
        return base
    digits = re.sub(r"\D", "", name)
    if not digits and name.upper() == kind.upper():
        # The first of several channels of a kind is named after the kind
        # alone: EXT, then EXT2.
        digits = "1"
    return f"{base}{digits}" if digits else f"{base}_{name.lower()}"


def read_cmrr(paths: Sequence[Path]) -> CmrrPhysio:
    """Read one CMRR physio DICOM, or a set of CMRR ``.log`` files."""
    paths = [Path(p) for p in paths]
    logs: list[tuple[str, str]] = []
    for path in paths:
        if path.suffix.lower() == ".log":
            logs.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))
        else:
            logs.extend(logs_in_dicom(path))
    physio = CmrrPhysio()
    for _name, text in logs:
        header, samples, rows = parse_log(text)
        kind = header.get("LogDataType", "")
        physio.uuid = physio.uuid or header.get("UUID", "")
        physio.scan_date = physio.scan_date or header.get("ScanDate", "")
        if kind == "ACQUISITION_INFO":
            physio.volumes, physio.unfinished = _volumes(rows, _int(header.get("NumSlices")))
            continue
        step = _sample_ticks(header, samples)
        several = len(samples) > 1
        for channel_name in sorted(samples, key=_natural):
            ticks = np.array([t for t, _v in samples[channel_name]], dtype=np.int64)
            values = np.array([v for _t, v in samples[channel_name]], dtype=float)
            if ticks.size == 0:
                continue
            index, start = _place(ticks, step)
            grid = np.full(int(index[-1]) + 1, np.nan)
            grid[index] = values
            physio.channels.append(Channel(
                name=channel_name, label=_label(kind, channel_name, several), kind=kind,
                rate=1.0 / (step * TICK), start=start * TICK, values=grid))
    return physio


def _place(ticks: np.ndarray, step: float) -> tuple[np.ndarray, float]:
    """``(grid index of each sample, tick of grid slot 0)``.

    By SEQUENCE, not by rounding each tick: the scanner corrects its clock
    once in a while (measured: every channel at once, one tick back, about
    0.1 s into a run), so the ticks after it sit half a sample off the ticks
    before. Rounding them one by one put two samples in one slot every other
    time and left half the pulse channel empty. A step of ``k`` samples is
    ``max(1, round(gap / step))``: a real gap leaves slots empty, a corrected
    clock does not. Slot 0's tick is the median offset, so nearly every
    sample keeps its logged time exactly (those before a correction move by
    at most one tick, 2.5 ms).
    """
    order = np.argsort(ticks, kind="stable")
    if not np.array_equal(order, np.arange(ticks.size)):
        ticks = ticks[order]
    steps = np.maximum(1, np.floor(np.diff(ticks) / step + 0.5)).astype(np.int64)
    index = np.concatenate([[0], np.cumsum(steps)])
    start = float(np.median(ticks - index * step))
    if not np.array_equal(order, np.arange(ticks.size)):
        # Back to the order the values were read in.
        unsorted = np.empty_like(index)
        unsorted[order] = index
        index = unsorted
    return index, start


def _natural(name: str):
    digits = re.sub(r"\D", "", name)
    return (re.sub(r"\d", "", name), int(digits) if digits else 0)


def _sample_ticks(header: dict[str, str], samples: dict) -> float:
    """Ticks between two samples: the log's ``SampleTime``, else measured."""
    try:
        step = float(header.get("SampleTime", ""))
        if step > 0:
            return step
    except ValueError:
        pass
    diffs = [np.diff(sorted({t for t, _v in values})) for values in samples.values()]
    diffs = [d for d in diffs if d.size]
    return float(np.median(np.concatenate(diffs))) if diffs else 1.0


def _volumes(rows: list[tuple[int, int, int, int, int]],
             slices: int = 0) -> tuple[list[Volume], int]:
    """Each volume's first start and last finish, from its first echo; and
    how many volumes at the end the scanner did not finish.

    A run stopped by hand stops in the middle of a volume: the log lists the
    slices acquired (40 of 48 on SIna OL_4594, 8 of 48 on OL_4925), dcm2niix
    writes that volume to a file of its own (``_bolda``, "partial volume")
    and the conversion leaves it out, so it is not a volume of the image.
    With ``slices`` (the log's ``NumSlices``), trailing volumes with fewer
    slices are left out; a volume in the middle missing a line is kept,
    since the image has it."""
    starts: dict[int, int] = {}
    finishes: dict[int, int] = {}
    counts: dict[int, int] = {}
    echoes = [r[4] for r in rows]
    first_echo = min(echoes) if echoes else 0
    for volume, _slice, start, finish, echo in rows:
        if echo != first_echo:
            continue
        starts[volume] = min(starts.get(volume, start), start)
        finishes[volume] = max(finishes.get(volume, finish), finish)
        counts[volume] = counts.get(volume, 0) + 1
    order = sorted(starts)
    unfinished = 0
    while slices > 0 and order and counts[order[-1]] < slices:
        order.pop()
        unfinished += 1
    return [Volume(v, starts[v] * TICK, finishes[v] * TICK) for v in order], unfinished


def _int(text) -> int:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


@dataclass
class Table:
    """One BIDS physio table, before it is written."""

    label: str                       # the recording label ("" for the only table)
    rate: float
    start: float                     # time of row 0, scanner clock
    columns: list[str]
    data: np.ndarray                 # rows x columns (float, NaN = n/a)
    descriptions: dict[str, str]

    def start_time(self, t0: float) -> float:
        return round(self.start - t0, 6)


def _ordered(channels: list[Channel]) -> list[Channel]:
    def key(ch: Channel):
        base = re.sub(r"\d+$", "", ch.label)
        rank = _ORDER.index(base) if base in _ORDER else len(_ORDER)
        return (rank, _natural(ch.name))
    return sorted(channels, key=key)


def _describe(ch: Channel, rate: Optional[float] = None, *, as_recorded: bool = False,
              shift: float = 0.0) -> str:
    """What a column holds; how it was put on the table's rate when that is
    not its own (``shift``: the most a sample was moved to a row, seconds)."""
    what = _KINDS.get(ch.kind, ("", ch.kind))[1] or ch.kind
    text = f"{what} (Siemens {ch.name}), in the scanner's units."
    if ch.kind == "ECG" and ch.label != "ecg":
        lead = re.sub(r"\D", "", ch.name)
        text = f"ECG channel {lead} (Siemens {ch.name}), in the scanner's units."
    if rate is None or abs(rate - ch.rate) <= 1e-9:
        return text
    if not as_recorded:
        return text + (f" Recorded at {ch.rate:g} Hz and resampled to {rate:g} Hz by "
                       "linear interpolation.")
    every = rate / ch.rate
    text += (f" Recorded at {ch.rate:g} Hz: each sample is in the row of its own time "
             + (f"(one row in {every:g}) " if abs(every - round(every)) < 1e-9 else "")
             + "and the rows between are n/a.")
    if shift > 1e-9:
        text += (f" Its times fall between the table's rows, so each sample is in the "
                 f"nearest row, at most {shift * 1000:.3g} ms from when it was recorded.")
    return text


def _on_grid(ch: Channel, start: float, rate: float, n: int) -> np.ndarray:
    """``ch`` sampled at ``start + k / rate`` for ``k < n``: placed exactly when
    the rate and phase match, else linearly interpolated, never across a gap
    and never beyond the channel's own first and last sample."""
    out = np.full(n, np.nan)
    if abs(ch.rate - rate) < 1e-9:
        offset = (ch.start - start) * rate
        shift = int(round(offset))
        if abs(offset - shift) < 1e-6:
            lo, hi = max(0, shift), min(n, shift + len(ch.values))
            if hi > lo:
                out[lo:hi] = ch.values[lo - shift:hi - shift]
            return out
    times = ch.start + np.arange(len(ch.values)) / ch.rate
    grid = start + np.arange(n) / rate
    valid = ~np.isnan(ch.values)
    if valid.sum() < 2:
        return out
    inside = (grid >= times[0] - 1e-9) & (grid <= times[-1] + 1e-9)
    out[inside] = np.interp(grid[inside], times[valid], ch.values[valid])
    # A grid point next to a missing sample is not interpolated across it.
    pos = np.clip(np.searchsorted(times, grid[inside]), 1, len(times) - 1)
    gap = np.isnan(ch.values[pos - 1]) | np.isnan(ch.values[pos])
    out[np.flatnonzero(inside)[gap]] = np.nan
    return out


def _as_recorded(ch: Channel, start: float, rate: float, n: int) -> tuple[np.ndarray, float]:
    """``ch``'s own samples in the rows of their times on a grid of ``n``
    rows at ``rate`` from ``start``, ``n/a`` elsewhere; and the most a sample
    was moved to reach its row (0 when the grids share their ticks)."""
    out = np.full(n, np.nan)
    exact = (ch.start - start) * rate + np.arange(len(ch.values)) * (rate / ch.rate)
    rows = np.rint(exact).astype(np.int64)
    inside = (rows >= 0) & (rows < n)
    out[rows[inside]] = ch.values[inside]
    moved = float(np.max(np.abs(exact[inside] - rows[inside])) / rate) if inside.any() else 0.0
    return out, (moved if moved > 1e-6 / rate else 0.0)


#: How near a row a time must be to be ON it, in rows. A row index computed
#: from scanner times (seconds of the day, ~10^4 to 10^5) carries float noise
#: of a few 1e-9 rows: a tolerance below that put a volume that starts
#: exactly on a row one row late (2.5 ms on a 400 Hz table, measured on a
#: real ECG table).
_ON_ROW = 1e-6


def _first_row(seconds: float, rate: float) -> int:
    """The first row at or after ``seconds`` from row 0."""
    return int(np.ceil(seconds * rate - _ON_ROW))


def _row_count(seconds: float, rate: float) -> int:
    """Rows from row 0 to the last at or before ``seconds``, both included."""
    return int(np.floor(seconds * rate + _ON_ROW)) + 1


def _trigger(volumes: list[Volume], start: float, rate: float, n: int) -> np.ndarray:
    col = np.zeros(n)
    for vol in volumes:
        k = _first_row(vol.start - start, rate)
        if 0 <= k < n:
            col[k] = 1
    return col


def _trigger_description(drop_volumes: int, *, alone: bool = False,
                         unfinished: int = 0) -> str:
    text = ("Scanner volume trigger from the acquisition log: 1 at the first sample of "
            "each volume of the run's image, 0 elsewhere.")
    if alone:
        text += (" The log recorded no physiological signal, only when each volume was "
                 "acquired, on the scanner's clock (one row per 2.5 ms tick).")
    if drop_volumes > 0:
        text += (f" The {drop_volumes} volumes removed from the start of the image are not "
                 "marked.")
    if unfinished > 0:
        text += (" The last volume, which the scanner did not finish when the run was "
                 "stopped and which is not in the image, is not marked."
                 if unfinished == 1 else
                 f" The last {unfinished} volumes, which the scanner did not finish and "
                 "which are not in the image, are not marked.")
    return text


def _trigger_only(volumes: list[Volume], kept: list[Volume], *, align: bool,
                  drop_volumes: int, unfinished: int = 0) -> Table:
    """The table of a log with no signal: the trigger alone, one row per
    scanner tick, from the first volume (the first kept when ``align``) to
    the end of the last. Every volume starts on a tick, so each mark is in
    the row of its exact time."""
    rate = 1.0 / TICK
    start = (kept if align else volumes)[0].start
    end = max(vol.finish for vol in kept)
    n = _row_count(end - start, rate)
    return Table(label="", rate=rate, start=start, columns=["trigger"],
                 data=_trigger(kept, start, rate, n).reshape(-1, 1),
                 descriptions={"trigger": _trigger_description(
                     drop_volumes, alone=True, unfinished=unfinished)})


def build_tables(physio: CmrrPhysio, *, layout: str = LAYOUT_BY_RATE, align: bool = False,
                 drop_volumes: int = 0) -> tuple[list[Table], float]:
    """The tables to write, and the time (scanner clock) ``StartTime`` is
    measured from: the first volume kept, or the first sample when the log
    says nothing about volumes. A log with volumes and no signal gives one
    table, the trigger alone (:func:`_trigger_only`)."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown physio layout {layout!r}; choose from {', '.join(LAYOUTS)}")
    channels = _ordered(physio.channels)
    if not channels and not physio.volumes:
        raise ValueError("the physiological log holds neither a signal nor the volume timings")
    kept = physio.volumes[drop_volumes:] if drop_volumes > 0 else list(physio.volumes)
    if physio.volumes and not kept:
        raise ValueError(f"{drop_volumes} volumes to remove, and the run has only "
                         f"{len(physio.volumes)}")
    if not channels:
        return [_trigger_only(physio.volumes, kept, align=align, drop_volumes=drop_volumes,
                              unfinished=physio.unfinished)], kept[0].start
    t0 = kept[0].start if kept else min(ch.start for ch in channels)
    window = None
    if align and kept:
        window = (kept[0].start, kept[-1].finish)

    if layout in (LAYOUT_ONE_INTERPOLATED, LAYOUT_ONE_AS_RECORDED):
        groups = [channels]
    else:
        by_rate: dict[float, list[Channel]] = {}
        for ch in channels:
            by_rate.setdefault(round(ch.rate, 6), []).append(ch)
        groups = list(by_rate.values())

    tables: list[Table] = []
    for group in groups:
        rate = max(ch.rate for ch in group)
        start = min(ch.start for ch in group)
        end = max(ch.end for ch in group)
        if window is not None:
            # From the first sample at or after the run's start to the last
            # at or before its end, on this table's grid.
            start = start + _first_row(window[0] - start, rate) / rate
            end = min(end, window[1])
        n = _row_count(end - start, rate)
        if n <= 0:
            continue
        columns, data, descriptions = [], [], {}
        for ch in group:
            columns.append(ch.label)
            if layout == LAYOUT_ONE_AS_RECORDED:
                values, shift = _as_recorded(ch, start, rate, n)
                data.append(values)
                descriptions[ch.label] = _describe(ch, rate, as_recorded=True, shift=shift)
            else:
                data.append(_on_grid(ch, start, rate, n))
                descriptions[ch.label] = _describe(ch, rate)
        if kept:
            columns.append("trigger")
            data.append(_trigger(kept, start, rate, n))
            descriptions["trigger"] = _trigger_description(drop_volumes,
                                                           unfinished=physio.unfinished)
        tables.append(Table(label=_group_label(group), rate=rate, start=start,
                            columns=columns, data=np.column_stack(data),
                            descriptions=descriptions))
    if len(tables) == 1:
        tables[0].label = ""
    else:
        from ..vendor.bidsphysio.base.bidsphysio import unique_bids_labels

        for table, label in zip(tables, unique_bids_labels([t.label for t in tables])):
            table.label = label
    return tables, t0


def _group_label(group: list[Channel]) -> str:
    bases = {re.sub(r"\d+$", "", ch.label) for ch in group}
    return bases.pop() if len(bases) == 1 else group[0].label


def table_paths(prefix: Path, table: Table) -> tuple[Path, Path]:
    """``(<prefix>[_recording-<label>]_physio.tsv.gz, .json)``. ``prefix`` is
    the recording's name without ``_physio``."""
    prefix = Path(prefix)
    stem = prefix.name + (f"_recording-{table.label}" if table.label else "") + "_physio"
    return prefix.with_name(stem + ".tsv.gz"), prefix.with_name(stem + ".json")


def write_tables(physio: CmrrPhysio, prefix: Path, *, layout: str = LAYOUT_BY_RATE,
                 align: bool = False, drop_volumes: int = 0) -> list[Path]:
    """Write the BIDS physio tables of ``physio`` next to ``prefix`` (the
    recording's name without ``_physio``). Returns every file written."""
    tables, t0 = build_tables(physio, layout=layout, align=align, drop_volumes=drop_volumes)
    written: list[Path] = []
    for table in tables:
        data_path, json_path = table_paths(prefix, table)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tsv_gz(data_path, table)
        sidecar = {
            "SamplingFrequency": table.rate,
            "StartTime": table.start_time(t0),
            "Columns": table.columns,
        }
        for column in table.columns:
            sidecar[column] = {"Description": table.descriptions[column]}
        json_path.write_text(json.dumps(sidecar, indent=4) + "\n", encoding="utf-8")
        written += [data_path, json_path]
    return written


def _write_tsv_gz(path: Path, table: Table) -> None:
    """No header row (BIDS physio), ``n/a`` for a missing sample, the trigger
    as 0/1, in chunks so a long recording is never all text at once."""
    fmt = ["%d" if c == "trigger" else "%.10g" for c in table.columns]
    with gzip.open(path, "wb") as out:
        for lo in range(0, len(table.data), _ROWS_PER_CHUNK):
            buf = io.BytesIO()
            np.savetxt(buf, table.data[lo:lo + _ROWS_PER_CHUNK], fmt=fmt, delimiter="\t")
            out.write(buf.getvalue().replace(b"nan", b"n/a"))


__all__ = ["Channel", "CmrrPhysio", "LAYOUTS", "LAYOUT_BY_RATE", "LAYOUT_ONE_AS_RECORDED",
           "LAYOUT_ONE_INTERPOLATED", "TICK",
           "Table", "Volume", "build_tables", "logs_in_dicom", "parse_log", "read_cmrr",
           "table_paths", "write_tables"]

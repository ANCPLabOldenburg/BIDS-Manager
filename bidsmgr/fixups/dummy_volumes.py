"""Remove the first volumes of a functional run, before it reaches the dataset.

The first volumes of a BOLD run are often acquired before the signal reaches
its steady state (dummy scans the scanner did not discard). A study that drops
them would otherwise do it in every analysis; here it is done once, at
conversion, and recorded where BIDS asks:

- the image (every echo) loses its first ``n`` volumes, header and stored
  values otherwise exactly as dcm2niix wrote them;
- the sidecar gains ``NumberOfVolumesDiscardedByUser`` (added to any value
  already there), and a ``VolumeTiming`` loses its first ``n`` entries and is
  measured from the new first volume;
- the image now starts later, by the OFFSET from its first volume to its
  first volume kept: ``VolumeTiming[n] - VolumeTiming[0]`` when the sidecar
  times its volumes, else ``n * RepetitionTime``, else ``n`` times the
  header's repetition time. The sidecar's own times (``AcquisitionTime``,
  ``AcquisitionDateTime``) move by it, so the run's ``acq_time`` in the
  scans table is the first volume kept, as BIDS defines it ("when the first
  data point in each run was acquired"; ``metadata.tables.acq_time``);
- the run's events (``*_events.tsv`` beside it) move by the same offset:
  BIDS measures an onset from the first volume of the image. An event that
  now starts before the image is kept, with its negative onset, which the
  standard allows.

The run's physiology starts at the first volume kept too; that is done when
its tables are written (:mod:`bidsmgr.converter.physio_cmrr`), from the same
number, which the conversion hands the log recorded with the run
(:func:`physio_runs`).

Only functional runs: removing volumes from diffusion or PET data would need
the gradient table or the frame timing rewritten with them.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

#: The suffixes volumes can be removed from: a functional time series.
DROPPABLE_SUFFIXES = frozenset({"bold", "cbv", "phase"})

FIELD = "NumberOfVolumesDiscardedByUser"


def applies(datatype: str, suffix: str) -> bool:
    """Whether volumes can be removed from a recording of this kind."""
    return datatype == "func" and suffix in DROPPABLE_SUFFIXES


def volumes_to_remove(cell, datatype: str, suffix: str, default: int) -> int:
    """How many volumes the conversion removes from the start of a row: its
    ``drop_volumes`` cell when that is a number (0 included), else
    ``default``; 0 for anything but a functional time series. Raises
    ``ValueError`` for a cell that is not a number."""
    if not applies(datatype, suffix):
        return 0
    text = "" if cell is None or (isinstance(cell, float) and cell != cell) else str(cell).strip()
    if text:
        return max(0, int(float(text)))
    return max(0, int(default or 0))


#: What a Siemens CMRR sequence adds to its run's series description for the
#: physiological log it records with the run (``task-rest_bold`` ->
#: ``task-rest_bold_PhysioLog``).
PHYSIO_SERIES_SUFFIX = "_PhysioLog"


def physio_runs(df) -> dict:
    """For every physiology row of a functional run (``func``, ``_physio``),
    the row of the run it was recorded with: ``{physio row: run row}``, both
    index labels of ``df`` (a scan table). A log linked to no run is absent.

    By the ACQUISITION first, so that names do not matter: the log's series
    is named after its run's (``<run>_PhysioLog``, in the same study) and is
    written when the run ends, so among the runs of that description the
    last to start before the log is its run (a protocol repeated under one
    name). Then by the NAME, for a log whose series says nothing: the run
    whose name differs from the log's by the suffix alone. Where a run is
    stored as magnitude and phase, the magnitude is the run.
    """
    from ..inventory._time import parse_time_seconds

    def cell(i, col: str) -> str:
        if col not in df.columns:
            return ""
        value = df.at[i, col]
        return "" if value is None or (isinstance(value, float) and value != value) \
            else str(value).strip()

    def suffix(i) -> str:
        return cell(i, "bids_name").rsplit("_", 1)[-1]

    def study(i) -> tuple:
        uid = cell(i, "study_instance_uid")
        return (uid,) if uid else (cell(i, "participant_id"), cell(i, "session"))

    def when(i) -> float:
        seconds = parse_time_seconds(cell(i, "acq_time"))
        return float("-inf") if seconds is None else seconds

    def phase(i) -> bool:
        name = cell(i, "bids_name")
        return "_part-phase_" in name or suffix(i) == "phase"

    def best(candidates):
        # The run, before the phase stored beside it.
        return max(candidates, key=lambda r: (when(r), not phase(r)))

    runs = [i for i in df.index if applies(cell(i, "datatype"), suffix(i))]
    out = {}
    for i in df.index:
        if cell(i, "datatype") != "func" or suffix(i) != "physio":
            continue
        run = None
        description = cell(i, "sequence")
        if description.lower().endswith(PHYSIO_SERIES_SUFFIX.lower()):
            of = description[: -len(PHYSIO_SERIES_SUFFIX)].lower()
            same = [r for r in runs
                    if cell(r, "sequence").lower() == of and study(r) == study(i)]
            before = [r for r in same if when(r) <= when(i)]
            if before or same:
                run = best(before or same)
        if run is None:
            key = cell(i, "bids_name").rsplit("_", 1)[0]
            named = [r for r in runs if cell(r, "bids_name").rsplit("_", 1)[0] == key]
            if named:
                run = best(named)
        if run is not None:
            out[i] = run
    return out


def drop_leading_volumes(results: Iterable) -> int:
    """Remove the first ``task.drop_volumes`` volumes of every functional run
    converted in ``results``. Returns how many runs changed. A run too short
    for it is left whole, with a warning."""
    changed = 0
    for result in results:
        task = getattr(result, "task", None)
        n = int(getattr(task, "drop_volumes", 0) or 0)
        if n <= 0 or not getattr(result, "success", False) \
                or not applies(task.datatype, task.suffix):
            continue
        images = [Path(p) for p in result.staged_files
                  if str(p).endswith((".nii.gz", ".nii"))]
        if not images:
            continue
        try:
            done = [_drop_image(p, n) for p in images]
        except ValueError as exc:
            log.warning("%s: %s; the run is kept whole", task.basename, exc)
            continue
        offset = None
        for image in images:
            sidecar = _sidecar(image)
            if sidecar.exists():
                offset = _record(sidecar, n) or offset
        if offset is None and done and done[0]:
            offset = n * done[0]
        if offset:
            for image in images:
                _shift_events(image, offset)
        log.info("%s: the first %d volume(s) removed", task.basename, n)
        changed += 1
    return changed


def _sidecar(image: Path) -> Path:
    name = image.name
    stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name[: -len(".nii")]
    return image.with_name(stem + ".json")


def _drop_image(path: Path, n: int) -> float:
    """Cut the first ``n`` volumes; returns the repetition time in the
    header (seconds, 0 when unknown)."""
    import nibabel as nib
    import numpy as np

    img = nib.load(str(path))
    shape = img.shape
    if len(shape) < 4 or shape[3] <= n:
        volumes = shape[3] if len(shape) > 3 else 1
        raise ValueError(f"{n} volume(s) to remove, and {path.name} has {volumes}")
    raw = np.asanyarray(img.dataobj.get_unscaled()) if hasattr(img.dataobj, "get_unscaled") \
        else np.asanyarray(img.dataobj)
    out = img.__class__(raw[..., n:], img.affine, header=img.header)
    tmp = path.with_name(path.name + ".part" + (".nii.gz" if path.name.endswith(".gz") else ".nii"))
    nib.save(out, str(tmp))
    tmp.replace(path)
    units = img.header.get_xyzt_units()[1]
    scale = {"msec": 1e-3, "usec": 1e-6}.get(units, 1.0)
    return float(img.header["pixdim"][4]) * scale


def _record(sidecar: Path, n: int) -> Optional[float]:
    """Write the record into the sidecar and move its times; returns the
    offset in seconds from the image's first volume to its first volume
    kept, or ``None`` when the sidecar does not say."""
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    before = data.get(FIELD, 0)
    data[FIELD] = (before if isinstance(before, int) and not isinstance(before, bool) else 0) + n
    offset = None
    timing = data.get("VolumeTiming")
    if isinstance(timing, list) and len(timing) > n and all(_is_number(t) for t in timing):
        first = timing[n]
        offset = float(first) - float(timing[0])
        data["VolumeTiming"] = [round(t - first, 6) for t in timing[n:]]
    elif _is_number(data.get("RepetitionTime")):
        offset = n * float(data["RepetitionTime"])
    if offset:
        moved = {"AcquisitionTime": later_time(data.get("AcquisitionTime"), offset),
                 "AcquisitionDateTime": later_datetime(data.get("AcquisitionDateTime"), offset)}
        data.update({key: value for key, value in moved.items() if value})
    sidecar.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return offset


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def later_time(value, seconds: float) -> str:
    """A time of day as dcm2niix writes it (``10:20:12.132500``), ``seconds``
    later, past midnight from 00:00; empty when ``value`` is not a time."""
    from ..inventory._time import parse_time_seconds

    if not isinstance(value, str) or "T" in value:
        return ""
    start = parse_time_seconds(value)
    if start is None:
        return ""
    micro = int(round(((start + seconds) % 86400) * 1e6))
    whole, frac = divmod(micro, 1_000_000)
    hours, rest = divmod(whole, 3600)
    return f"{hours:02d}:{rest // 60:02d}:{rest % 60:02d}.{frac:06d}"


def later_datetime(value, seconds: float) -> str:
    """An ISO date and time (``2025-05-26T11:36:33.285000``) ``seconds``
    later, its date too; empty when ``value`` is not one."""
    import datetime as dt

    if not isinstance(value, str):
        return ""
    try:
        stamp = dt.datetime.fromisoformat(value.strip())
    except ValueError:
        return ""
    return (stamp + dt.timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def _shift_events(image: Path, seconds: float) -> None:
    """Move the onsets of the run's events by ``-seconds``."""
    name = image.name
    stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else name[: -len(".nii")]
    base = stem.rsplit("_", 1)[0]
    events = image.with_name(base + "_events.tsv")
    if not events.exists():
        return
    text = events.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
    if not rows or "onset" not in rows[0]:
        return
    col = rows[0].index("onset")
    for row in rows[1:]:
        if col < len(row):
            try:
                value = float(row[col]) - seconds
            except ValueError:
                continue
            row[col] = _number(value)
    out = io.StringIO()
    csv.writer(out, delimiter="\t", lineterminator="\n").writerows(rows)
    events.write_text(out.getvalue(), encoding="utf-8")
    log.info("%s: onsets moved by -%g s with the removed volumes", events.name, seconds)


def _number(value: float) -> str:
    text = f"{round(value, 6):.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


__all__ = ["DROPPABLE_SUFFIXES", "FIELD", "applies", "drop_leading_volumes",
           "PHYSIO_SERIES_SUFFIX", "later_datetime", "later_time", "physio_runs",
           "volumes_to_remove"]

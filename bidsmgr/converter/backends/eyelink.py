"""EyeLink eye tracking to BIDS (BEP020, in the standard since 1.11).

One recording becomes, per eye recorded (``recording-eye1`` left,
``recording-eye2`` right; a binocular file gives both)::

    <match>_recording-eye1_physio.tsv.gz        timestamp, x_coordinate, y_coordinate, pupil_size
    <match>_recording-eye1_physio.json          PhysioType "eyetrack", RecordedEye, ...
    <match>_recording-eye1_physioevents.tsv.gz  onset, duration, trial_type, message
    <match>_recording-eye1_physioevents.json    OnsetSource "timestamp", ...

and the screen the gaze is measured on goes in the run's ``_events.json``
(``StimulusPresentation``), which the standard requires for gaze-on-screen
coordinates once the run has events (``fixups/eyetrack``, after the events
are attached).

* ``timestamp`` and ``onset`` are the tracker's own clock in milliseconds,
  so pauses between trials stay where they were; ``duration`` is seconds.
* What the file says is written: the tracker, its serial number, the
  calibration and validation (type, count, errors), the pupil measure and
  how it was fitted, the screen resolution.
* What it cannot say comes from the user (Properties > Eye tracking, or the
  dataset defaults): the screen's size and distance, and how the recording
  lines up with the run (``StartTime``: a message, such as the scanner's
  first trigger, that marks the run's start, or a number of seconds).

The file is read by ``inventory/eyelink`` (eyelinkio, else edf2asc).
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import ConvertResult, ConvertTask

log = logging.getLogger(__name__)

RECORDING_LABEL = {"left": "eye1", "right": "eye2"}


class EyelinkBackend:
    """Convert an EyeLink ``.edf`` / ``.asc`` to BIDS eye-tracking files."""

    name = "eyelink"

    def can_handle(self, task: ConvertTask) -> bool:
        return task.modality == "eyetrack"

    def convert(self, task: ConvertTask, staging_dir: Path) -> ConvertResult:
        t0 = time.monotonic()
        try:
            from ...inventory.eyelink import read_recording

            source = next((Path(f) for f in task.source_files if Path(f).exists()), None)
            if source is None:
                raise FileNotFoundError("the recording is not where the scan found it")
            rec = read_recording(source)
            files = write_bids(rec, staging_dir / task.datatype, task.basename,
                               settings=dict(task.eyetrack or {}))
            return ConvertResult(task=task, success=True, staged_files=tuple(files),
                                 duration_s=time.monotonic() - t0)
        except Exception as exc:  # noqa: BLE001 - one recording never stops a subject
            log.warning("eyelink: %s: %s", task.basename, exc)
            return ConvertResult(task=task, success=False,
                                 error=f"{type(exc).__name__}: {exc}",
                                 duration_s=time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def name_for(basename: str, eye: str, suffix: str = "physio") -> str:
    """``basename`` with ``recording-eye<N>`` for ``eye`` and ``suffix``."""
    label = RECORDING_LABEL[eye]
    head = basename.rsplit("_", 1)[0] if "_" in basename else basename
    if re.search(r"_recording-[^_]+", head):
        head = re.sub(r"_recording-[^_]+", f"_recording-{label}", head)
    else:
        head = f"{head}_recording-{label}"
    return f"{head}_{suffix}"


def match_of(basename: str) -> str:
    """The recording's entities without ``recording`` and the suffix
    (``sub-01_task-x_run-1``), the name its events file shares."""
    head = basename.rsplit("_", 1)[0] if "_" in basename else basename
    return re.sub(r"_recording-[^_]+", "", head)


def start_time(rec, settings: dict) -> float:
    """``StartTime``: the user's number, else the first message containing
    their sync text (the run's start), else 0 (the recording starts the run)."""
    if settings.get("start_time") not in (None, ""):
        return float(settings["start_time"])
    sync = str(settings.get("sync_message") or "").strip()
    if sync and len(rec.times):
        marks = rec.messages_matching(sync)
        if marks:
            return round((float(rec.times[0]) - marks[0]) / 1000.0, 6)
        log.warning("eyelink: no message contains %r in %s; StartTime is 0", sync,
                    rec.path.name)
    return 0.0


def _number_text(values: np.ndarray, decimals: int) -> np.ndarray:
    out = np.char.mod(f"%.{decimals}f", np.nan_to_num(values, nan=0.0))
    out = out.astype(object)
    out[np.isnan(values)] = "n/a"
    return out


def _write_tsv_gz(path: Path, columns: list[np.ndarray]) -> None:
    rows = ["\t".join(map(str, r)) for r in zip(*columns)]
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows))
        if rows:
            fh.write("\n")


def _timestamps(times: np.ndarray) -> np.ndarray:
    whole = np.all(np.mod(times, 1) == 0)
    return (times.astype(np.int64).astype(str) if whole
            else np.char.mod("%.1f", times)).astype(object)


def physio_sidecar(rec, eye: str, settings: dict) -> dict:
    from ...inventory.eyelink import calibration_facts

    info = rec.info
    pupil = rec.pupil or "area"
    side = {
        "SamplingFrequency": rec.sfreq,
        "StartTime": start_time(rec, settings),
        "Columns": ["timestamp", "x_coordinate", "y_coordinate", "pupil_size"],
        "PhysioType": "eyetrack",
        "RecordedEye": eye,
        "SampleCoordinateSystem": "gaze-on-screen",
        "Manufacturer": "SR Research",
    }
    model = info.get("tracker") or info.get("version")
    if model:
        side["ManufacturersModelName"] = model
    if info.get("serial"):
        side["DeviceSerialNumber"] = info["serial"]
    side.update(calibration_facts(rec, eye))
    if rec.filters[0]:
        side["RawDataFilters"] = (f"EyeLink heuristic filter level {rec.filters[0]} on the "
                                  f"recorded data (level {rec.filters[1]} over the link)")
    if settings.get("eye_tracker_distance") not in (None, ""):
        side["EyeTrackerDistance"] = float(settings["eye_tracker_distance"])
    side["timestamp"] = {
        "Description": "The EyeLink tracker's clock at each sample. Pauses between "
                       "recording blocks keep their length.",
        "Units": "ms",
        "Origin": "The EyeLink tracker's start-up",
    }
    for axis in ("x", "y"):
        side[f"{axis}_coordinate"] = {
            "LongName": f"Gaze position ({axis})",
            "Description": f"Gaze position {axis}-coordinate of the recorded eye on the "
                           "screen, from its top-left corner.",
            "Units": "pixel",
        }
    side["pupil_size"] = {
        "Description": f"Pupil {pupil} of the recorded eye as measured by the EyeLink "
                       "tracker, in arbitrary units (see the EyeLink documentation for a "
                       "conversion).",
        "Units": "arbitrary",
    }
    return side


EVENTS_SIDECAR = {
    "Columns": ["onset", "duration", "trial_type", "message"],
    "Description": "Fixations, saccades and blinks as the EyeLink tracker parsed them, "
                   "and the messages it logged.",
    "OnsetSource": "timestamp",
    "onset": {"Description": "The tracker's clock at the start of the event, as in "
                             "the timestamp column of the recording.", "Units": "ms"},
    "duration": {"Description": "How long the event lasted.", "Units": "s"},
    "trial_type": {
        "Description": "The kind of event, as the EyeLink tracker's parser identified it.",
        "Levels": {"fixation": "A fixation.", "saccade": "A saccade.",
                   "blink": "A blink (the pupil was lost)."},
    },
    "message": {"Description": "A message logged by the experiment or the tracker."},
}


def write_bids(rec, out_dir: Path, basename: str, *, settings: Optional[dict] = None
               ) -> list[Path]:
    """Write ``rec`` as BIDS eye-tracking files into ``out_dir``; returns
    them."""
    settings = settings or {}
    if not rec.eyes:
        raise ValueError(f"{rec.path.name} holds no eye samples")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    stamps = _timestamps(rec.times)
    for eye in rec.eyes:
        gaze = rec.gaze[eye]
        name = name_for(basename, eye)
        tsv = out_dir / f"{name}.tsv.gz"
        _write_tsv_gz(tsv, [stamps, _number_text(gaze[:, 0], 1), _number_text(gaze[:, 1], 1),
                            _number_text(gaze[:, 2], 1)])
        side = out_dir / f"{name}.json"
        side.write_text(json.dumps(physio_sidecar(rec, eye, settings), indent=2) + "\n",
                        encoding="utf-8")
        written += [tsv, side]

        rows = [(e.start, (e.end - e.start) / 1000.0, e.kind, "n/a")
                for e in rec.events if e.eye == eye]
        rows += [(t, 0.0, "n/a", m.replace("\t", " ").replace("\n", " ").strip() or "n/a")
                 for t, m in rec.messages]
        rows.sort(key=lambda r: r[0])
        events_name = name_for(basename, eye, "physioevents")
        events = out_dir / f"{events_name}.tsv.gz"
        whole = all(float(r[0]).is_integer() for r in rows)
        _write_tsv_gz(events, [
            np.array([str(int(r[0])) if whole else f"{r[0]:.1f}" for r in rows], dtype=object),
            np.array([f"{r[1]:.3f}" for r in rows], dtype=object),
            np.array([r[2] for r in rows], dtype=object),
            np.array([r[3] for r in rows], dtype=object),
        ])
        events_side = out_dir / f"{events_name}.json"
        events_side.write_text(json.dumps(EVENTS_SIDECAR, indent=2) + "\n", encoding="utf-8")
        written += [events, events_side]
    return written


def stimulus_presentation(display, settings: dict) -> dict:
    """The ``StimulusPresentation`` the screen gives: the resolution the file
    records, and the size, distance and origin the user stated."""
    screen: dict = {}
    if display:
        screen["ScreenResolution"] = [int(display[0]), int(display[1])]
    screen["ScreenOrigin"] = list(settings.get("screen_origin") or ["top", "left"])
    if settings.get("screen_size"):
        screen["ScreenSize"] = [float(v) for v in settings["screen_size"]]
    if settings.get("screen_distance") not in (None, ""):
        screen["ScreenDistance"] = float(settings["screen_distance"])
    return screen


__all__ = ["EyelinkBackend", "match_of", "name_for", "physio_sidecar", "start_time",
           "stimulus_presentation", "write_bids"]

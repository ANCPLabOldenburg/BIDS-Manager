"""Events: what happened when, from the run's ``_events.tsv`` or a stim channel.

Two sources, because BIDS has two: the curated ``*_events.tsv`` beside a run
(onset, duration, trial_type), and the trigger codes a recording carries on
its stim channel. Both become :class:`Event` in seconds of RUN time, so a
BOLD graph, a physio trace and an MEG recording all place an onset in the
same place.

Qt-free.
"""

from __future__ import annotations

import csv
import gzip
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..bids import stem_of

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    onset: float
    duration: float = 0.0
    label: str = ""
    #: "" for an event, "bad" for a span marked bad (an annotation whose
    #: description starts with BAD), drawn in the error colour.
    kind: str = ""


def run_base(path) -> str:
    """The run part of a BIDS name: suffix and ``recording-`` entity removed.

    ``sub-01_task-a_run-1_recording-cardiac_physio.tsv.gz`` and
    ``sub-01_task-a_run-1_bold.nii.gz`` are the same run. Keeping the
    ``recording-`` entity is the defect that left a physio file unable to
    find its own run's events.
    """
    stem = stem_of(Path(path))
    parts = stem.split("_")
    if len(parts) > 1:
        parts = parts[:-1]          # the suffix
    return "_".join(p for p in parts if not p.startswith("recording-"))


def events_sibling(path) -> Optional[Path]:
    """The run's ``_events.tsv`` (or ``.tsv.gz``) beside ``path``, if any."""
    p = Path(path)
    base = run_base(p)
    for ext in ("_events.tsv", "_events.tsv.gz"):
        cand = p.parent / f"{base}{ext}"
        if cand.exists():
            return cand
    return None


def read_events_tsv(path) -> list[Event]:
    """Every row with a numeric onset. The label is ``trial_type``, else
    ``value``, else ``event_type``; a duration of ``n/a`` is zero."""
    out: list[Event] = []
    is_gz = str(path).lower().endswith(".gz")
    try:
        # utf-8-sig: mne-bids writes its tables with a byte order mark, and
        # read as plain UTF-8 the first column is "\ufeffonset": every row
        # was dropped.
        opener = (gzip.open(path, "rt", encoding="utf-8-sig", newline="") if is_gz
                  else open(path, "r", encoding="utf-8-sig", newline=""))
        with opener as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                try:
                    onset = float(row.get("onset", ""))
                except (TypeError, ValueError):
                    continue
                try:
                    duration = float(row.get("duration", 0) or 0)
                except (TypeError, ValueError):
                    duration = 0.0
                if not np.isfinite(duration) or duration < 0:
                    duration = 0.0
                label = (row.get("trial_type") or row.get("value")
                         or row.get("event_type") or "")
                # A BAD row is a segment marked bad (the mne-bids spelling
                # of an annotation), not an event.
                kind = "bad" if str(row.get("trial_type") or "").upper().startswith("BAD") else ""
                out.append(Event(onset, duration, str(label), kind))
    except Exception as exc:  # noqa: BLE001 - an unreadable table is no events
        log.debug("could not read events %s: %s", path, exc)
        return []
    return out


def stim_events(raw, ch_types: list[str]) -> list[Event]:
    """Trigger codes off the stim channels, as events (code as the label).

    Runs ``mne.find_events``, which walks every sample of the stim channel:
    call it on a worker.
    """
    import mne

    stim = [name for name, kind in zip(raw.ch_names, ch_types) if kind == "stim"]
    if not stim:
        return []
    try:
        found = mne.find_events(raw, stim_channel=stim, shortest_event=1, verbose=False)
    except Exception:  # noqa: BLE001 - a stim channel that is not one
        return []
    sfreq = float(raw.info["sfreq"])
    first = int(getattr(raw, "first_samp", 0))
    return [Event((int(ev[0]) - first) / sfreq, 0.0, str(int(ev[2])))
            for ev in found if int(ev[2]) != 0]


def annotation_events(raw) -> list[Event]:
    """The recording's own annotations (BrainVision markers, EDF+
    annotations, FIF ``BAD_`` spans), in seconds from the recording's start.
    A description starting with ``BAD`` is a span to leave out."""
    ann = getattr(raw, "annotations", None)
    if ann is None or len(ann) == 0:
        return []
    first = float(getattr(raw, "first_time", 0.0) or 0.0)
    shift = first if ann.orig_time is not None else 0.0
    out = []
    for onset, duration, description in zip(ann.onset, ann.duration, ann.description):
        label = str(description)
        kind = "bad" if label.upper().startswith("BAD") else ""
        out.append(Event(float(onset) - shift, float(max(duration, 0.0)), label, kind))
    return out


def edge_events(values, missing, sfreq: float, label: str = "") -> list[Event]:
    """Trigger onsets of an ANALOG channel (a physio trigger, a photodiode):
    every rise through the midpoint, or every present sample of a sparse
    log, at least 1.5 samples apart. ``mne.find_events`` counts every
    change of value, so 30 pulses with a little noise came out as 169."""
    from .physio import event_onsets

    v = np.asarray(values, dtype=float)
    times = np.arange(v.size, dtype=float) / float(sfreq)
    onsets, _vals = event_onsets(v, missing, times)
    out: list[Event] = []
    last = -np.inf
    for t in onsets:
        if t - last >= 1.5 / float(sfreq):
            out.append(Event(float(t), 0.0, label))
            last = t
    return out


__all__ = ["Event", "annotation_events", "edge_events", "events_sibling",
           "read_events_tsv", "run_base", "stim_events"]

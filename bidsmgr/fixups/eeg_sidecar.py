"""Post-write enrichment of EEG / MEG / iEEG / NIRS BIDS outputs.

Runs in the per-subject Phase 2 of the converter, on the staged tree, after
mne-bids has written the recording plus its ``channels.tsv``, datatype JSON
sidecar, and ``events.tsv``. mne-bids fills only what it can read from the
recording (channel counts, sampling rate, power-line frequency, electrode
positions); this step folds in the information the recording cannot carry on its
own, taken from a resolved :class:`~bidsmgr.recording_meta.RecordingMetaSpec`:

* sidecar JSON: reference, ground, hardware/software filters, manufacturer and
  model, software versions, cap details, institution, and optional extras.
* ``channels.tsv``: upgrade generically-typed auxiliary channels (ECG, EOG, ...)
  to their real BIDS type and fill units + description.
* ``events.tsv`` + ``events.json``: rename trigger codes to human-readable labels
  via the task's event map.
* the task sidecar: TaskDescription / Instructions from the task protocol.

The step is additive and best-effort: a missing file or a per-file error is
logged and skipped, never raised, so a single bad row cannot abort the subject.
``PowerLineFrequency`` and electrode positions are intentionally NOT touched here
- the backend already writes them from the resolved per-row value during the
write (they shape the data file and the coordinate files).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from ..recording_meta import (
    RecordingMetaSpec,
    resolve_effective,
)
from ..recording_meta.resolve import EffectiveSpec

log = logging.getLogger(__name__)

_SUPPORTED_DATATYPES = frozenset({"eeg", "meg", "ieeg", "nirs"})
# Channel ``type`` values we consider safe to overwrite with a user-specified
# BIDS type (we never clobber an already-specific type the reader assigned).
_GENERIC_CH_TYPES = frozenset({"MISC", "OTHER", "N/A", ""})


def enrich_recording_sidecars(
    subject_staging_dir: Path,
    tasks: Iterable,
    spec: Optional[RecordingMetaSpec],
) -> int:
    """Enrich every staged EEG/MEG/iEEG/NIRS recording for one subject.

    Parameters
    ----------
    subject_staging_dir
        Per-subject staging tree (``<bids_root>/.tmp_bidsmgr/sub-<id>/``); the
        recording's datatype directory lives at ``<staging>/[ses-Y/]<datatype>/``.
    tasks
        The :class:`ConvertTask` objects for this subject. Non-EEG/MEG tasks are
        ignored.
    spec
        The resolved recording-metadata spec. ``None`` makes this a no-op.

    Returns
    -------
    int
        Count of files (sidecar JSON / channels.tsv / events.tsv) modified.
    """
    if spec is None:
        return 0

    n_modified = 0
    for task in tasks:
        datatype = getattr(task, "datatype", "")
        if datatype not in _SUPPORTED_DATATYPES:
            continue
        basename = getattr(task, "basename", "") or ""
        if not basename:
            continue

        sidecar = _find_sidecar(subject_staging_dir, basename)
        if sidecar is None:
            # mne-bids should always emit the datatype JSON; if it is absent the
            # conversion of this row likely failed and was already reported.
            continue

        eff = resolve_effective(
            spec,
            getattr(task, "row_id", "") or "",
            (getattr(task, "entities", {}) or {}).get("task"),
            datatype,
        )

        # The sidecar's own FIELDS are not written here any more. They are what
        # the user stated, and stating them is the metadata step's job; this
        # pass used to walk the same chain, so the same field could be written
        # twice and the winner depended on the order the two ran in.
        #
        # What stays is the work that needs this context: retyping the channels
        # the reader saw as generic, giving trigger codes their labels, and the
        # task protocol. None of that is expressible as a sidecar field.
        n_modified += _retype_channels(sidecar, basename, eff)
        n_modified += _map_events(sidecar, basename, eff)
        n_modified += _apply_task_protocol(sidecar, eff)

    return n_modified


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_sidecar(staging: Path, basename: str) -> Optional[Path]:
    """Locate the staged ``<basename>.json`` datatype sidecar, if present."""
    if not staging.is_dir():
        return None
    matches = list(staging.rglob(f"{basename}.json"))
    return matches[0] if matches else None


def _entity_prefix(basename: str) -> str:
    """``sub-01_task-rest_eeg`` -> ``sub-01_task-rest`` (strip the suffix token)."""
    return basename.rsplit("_", 1)[0] if "_" in basename else basename


def _read_json(path: Path) -> dict:
    # utf-8-sig tolerates a leading BOM (mne-bids writes its TSVs with one;
    # JSON usually has none, but reading sig-aware is harmless and safe).
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        log.warning("enrich: could not read %s: %s", path.name, exc)
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def _extras_to_keys(extras) -> dict:
    out: dict = {}
    if extras.acceptable_impedance is not None:
        imp = extras.acceptable_impedance
        val = {k: v for k, v in {"value": imp.value, "units": imp.units}.items() if v is not None}
        if val:
            out["AcceptableImpedance"] = val
    if extras.electrode_type is not None:
        out["ElectrodeType"] = extras.electrode_type
    if extras.conductive_medium is not None:
        out["ConductiveMedium"] = extras.conductive_medium
    if extras.faraday_cage is not None:
        out["FaradayCage"] = extras.faraday_cage
    if extras.sound_proof is not None:
        out["SoundProofing"] = extras.sound_proof
    if extras.lighting_conditions is not None:
        lc = extras.lighting_conditions
        val = {
            k: v for k, v in {
                "description": lc.description, "measurement": lc.measurement,
            }.items() if v is not None
        }
        if val:
            out["LightingConditions"] = val
    return out


def _retype_channels(sidecar: Path, basename: str, eff: EffectiveSpec) -> int:
    """Upgrade generic aux-channel rows in channels.tsv and fill units/description."""
    aux = eff.acquisition.aux_channels
    if not aux:
        return 0
    tsv = sidecar.with_name(f"{_entity_prefix(basename)}_channels.tsv")
    if not tsv.is_file():
        return 0

    try:
        with open(tsv, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as exc:  # pragma: no cover - defensive
        log.warning("enrich: could not read %s: %s", tsv.name, exc)
        return 0
    if "name" not in fieldnames:
        return 0

    present = {row.get("name", "") for row in rows}
    for missing in sorted(set(aux) - present):
        log.warning("enrich: aux channel %r in spec not found in %s", missing, tsv.name)

    changed = False
    for row in rows:
        name = row.get("name", "")
        spec_ch = aux.get(name)
        if spec_ch is None:
            continue
        if spec_ch.bids_type and "type" in fieldnames:
            current = str(row.get("type", "")).upper()
            if current in _GENERIC_CH_TYPES:
                row["type"] = spec_ch.bids_type
                changed = True
        if spec_ch.units and "units" in fieldnames:
            current_u = str(row.get("units", "")).strip()
            if current_u in ("", "n/a"):
                row["units"] = spec_ch.units
                changed = True
        if spec_ch.description and "description" in fieldnames:
            current_d = str(row.get("description", "")).strip()
            if current_d in ("", "n/a"):
                row["description"] = spec_ch.description
                changed = True

    if not changed:
        return 0
    with open(tsv, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    log.info("enrich: %s channel types/units updated", tsv.name)
    return 1


def _map_events(sidecar: Path, basename: str, eff: EffectiveSpec) -> int:
    """Rename trigger codes to labels in events.tsv and write an events.json."""
    event_map = eff.event_map
    if not event_map:
        return 0
    tsv = sidecar.with_name(f"{_entity_prefix(basename)}_events.tsv")
    if not tsv.is_file():
        return 0

    try:
        with open(tsv, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as exc:  # pragma: no cover - defensive
        log.warning("enrich: could not read %s: %s", tsv.name, exc)
        return 0

    # Map on the column that carries the trigger label/code. mne-bids puts the
    # annotation description in ``trial_type`` and the numeric code in ``value``.
    label_col = "trial_type" if "trial_type" in fieldnames else None
    if label_col is None:
        return 0

    used: dict[str, str] = {}
    changed = False
    for row in rows:
        code = str(row.get(label_col, ""))
        # Skip blank labels: a scaffold seeds detected codes with empty
        # labels for the user to fill; an unedited entry must not rename
        # the trial_type to "".
        if code in event_map and event_map[code]:
            row[label_col] = event_map[code]
            used[event_map[code]] = code
            changed = True
    if not changed:
        return 0

    with open(tsv, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    # Companion events.json documenting the trial_type levels we produced.
    events_json = sidecar.with_name(f"{_entity_prefix(basename)}_events.json")
    data = _read_json(events_json) if events_json.is_file() else {}
    levels = {label: f"trigger code {code}" for label, code in used.items()}
    tt = data.get("trial_type", {})
    tt.setdefault("Description", "Event category")
    tt["Levels"] = {**tt.get("Levels", {}), **levels}
    data["trial_type"] = tt
    _write_json(events_json, data)
    log.info("enrich: %s mapped %d event label(s)", tsv.name, len(used))
    return 1


def _apply_task_protocol(sidecar: Path, eff: EffectiveSpec) -> int:
    """Write TaskDescription / Instructions into the datatype sidecar."""
    proto = eff.task_protocol
    if proto is None:
        return 0
    updates: dict = {}
    if proto.task_description:
        updates["TaskDescription"] = proto.task_description
    if proto.instructions:
        updates["Instructions"] = proto.instructions
    if not updates:
        return 0
    data = _read_json(sidecar)
    data.update(updates)
    _write_json(sidecar, data)
    return 1


__all__ = ["enrich_recording_sidecars"]

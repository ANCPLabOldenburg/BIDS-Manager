"""What the conversion will answer by itself, per kind of file.

``derivable.py`` says WHICH fields a converter usually fills. That is enough to
keep the template from asking for them, and not enough to be reassuring: a form
that simply omits a required field looks like a form that forgot it, and a
per-row panel that shows the field empty looks like a dataset missing metadata.
Neither is true. dcm2niix has already read the flip angle out of the header.

This module answers the other half: what the value will BE. It reads the
sidecars a probe conversion produced and the header facts the scan read out of
each recording, and reports them per ``<datatype>/<suffix>`` so a form can show
"EchoTime 0.03, from the DICOM" instead of an empty box.

Two rules make it honest rather than decorative:

* A field is reported only when every probed file of that kind agrees. Where
  they differ the answer is :data:`~bidsmgr.recording_meta.VARIES`, which is
  exactly what that sentinel is for: it says the answer lives further down, per
  recording, not here.
* Nothing here is ever written to a sidecar. It is a preview of what the
  conversion does anyway, so writing it would at best duplicate the converter
  and at worst freeze one file's value onto a class of files.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .. import schema as schema_mod
from ..recording_meta import VARIES
from .derivable import CONVERTER_PRIVATE


def _key(datatype: str, suffix: str) -> str:
    return f"{datatype}/{suffix}"


def _is_answer(value: Any) -> bool:
    """Is this a value, or a placeholder standing in for one?

    BIDS requires some keys whatever the answer, so mne-bids writes ``"n/a"``
    when it does not know. Counting that as answered is what hid EEGReference
    and EEGGround from the form: the very fields a user has to supply.
    """
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in ("n/a", "na", ""):
        return False
    if isinstance(value, (list, dict)) and not value:
        return False
    return True


def _agree(values: list[Any], complete: bool = True) -> Any:
    """One answer for a group of files, or VARIES.

    VARIES covers both ways a kind can fail to have one answer: the files
    disagree, or some of them did not answer at all. Either way the answer lives
    per recording, which is what the sentinel says.
    """
    if not complete:
        return VARIES
    first = values[0]
    for other in values[1:]:
        if other != first:
            return VARIES
    return first


def _collapse(
    seen: dict[str, dict[str, list[Any]]],
    counted: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Per kind, one value per field the conversion answers for any of them.

    Three states, and the difference between them matters:

    * every file gave the same value: that value, and the form says so;
    * the files disagree, or only some of them answered: VARIES, meaning the
      answer is per recording rather than one thing about the kind.

    Requiring EVERY file to answer before saying the conversion does was the
    first attempt and it was too strict for real data. Seventy-five functional
    runs from several studies agree on fourteen fields and no more, so the form
    asked for the repetition time, the echo time and thirty others that
    dcm2niix reads out of the header for almost all of them. Being asked for
    what the converter fills is the noise the template exists to remove.

    Nothing is lost by relaxing it: the block is editable, so a value stated
    there covers the recordings whose header lacked one.
    """
    out: dict[str, dict[str, Any]] = {}
    for key, fields in seen.items():
        datatype, _, suffix = key.partition("/")
        n_files = counted.get(key, 0)
        answers = {
            name: _agree(values, complete=len(values) >= n_files)
            for name, values in sorted(fields.items())
            if name not in CONVERTER_PRIVATE
            and schema_mod.field_applies(name, datatype, suffix)
        }
        if answers:
            out[key] = answers
    return out


def preview_from_inventory(df) -> dict[str, dict[str, Any]]:
    """What the scan already knows the conversion will write, per kind.

    Free: the scanner opened every recording to build the inventory and asked it
    the same questions mne-bids will ask, so no conversion has to run.

    The task label is added here rather than there because it is a curation
    decision, not a fact in the file: mne-bids writes whatever the row says.
    """
    if df is None or not len(df) or "_derived_fields" not in df.columns:
        return {}

    seen: dict[str, dict[str, list[Any]]] = {}
    counted: dict[str, int] = {}
    for _, row in df.iterrows():
        if str(row.get("include", "1")).strip() in ("0", "False", "false"):
            continue
        datatype = str(row.get("proposed_datatype", "") or "").strip()
        suffix = str(row.get("bids_guess_suffix", "") or "").strip()
        if not datatype or not suffix:
            continue
        raw = row.get("_derived_fields")
        try:
            derived = json.loads(raw) if isinstance(raw, str) and raw else {}
        except (ValueError, TypeError):
            derived = {}
        if not isinstance(derived, dict) or not derived:
            continue
        # Counted only now: a row that measured nothing is not a row that
        # failed to answer.
        key = _key(datatype, suffix)
        counted[key] = counted.get(key, 0) + 1

        task = str(row.get("task", "") or "").strip()
        if task:
            derived = {**derived, "TaskName": task}

        fields = seen.setdefault(_key(datatype, suffix), {})
        for name, value in derived.items():
            fields.setdefault(name, []).append(value)
    return _collapse(seen, counted)


def preview_from_probe(df, probe_stats: Optional[dict] = None) -> dict[str, dict[str, Any]]:
    """What a probe conversion actually produced, per kind.

    The join runs through the inventory rather than the scanned rows: a row
    knows its series UID but not its datatype, which the classifier decides
    afterwards and writes into the table.
    """
    if not probe_stats or df is None or not len(df):
        return {}

    kind_of: dict[str, tuple[str, str]] = {}
    for _, row in df.iterrows():
        uid = str(row.get("series_uid", "") or "").strip()
        datatype = str(row.get("proposed_datatype", "") or "").strip()
        if not datatype:
            datatype = str(row.get("bids_guess_datatype", "") or "").strip()
        suffix = str(row.get("bids_guess_suffix", "") or "").strip()
        if uid and datatype and suffix:
            kind_of[uid] = (datatype, suffix)

    seen: dict[str, dict[str, list[Any]]] = {}
    counted: dict[str, int] = {}
    for uid, stats in probe_stats.items():
        kind = kind_of.get(str(uid))
        if not kind:
            continue
        fields = getattr(stats, "sidecar_fields", None)
        if not fields:
            continue
        key = _key(*kind)
        counted[key] = counted.get(key, 0) + 1
        bucket = seen.setdefault(key, {})
        for name, value in fields.items():
            if _is_answer(value):
                bucket.setdefault(name, []).append(value)
    return _collapse(seen, counted)


def merge_previews(*previews: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Combine previews from several sources; later ones win per field."""
    out: dict[str, dict[str, Any]] = {}
    for preview in previews:
        for key, fields in (preview or {}).items():
            out.setdefault(key, {}).update(fields)
    return out


__all__ = [
    "merge_previews",
    "preview_by_row",
    "preview_from_inventory",
    "preview_from_probe",
]


def preview_by_row(df) -> dict[str, dict[str, Any]]:
    """What the conversion answers for each individual recording.

    Keyed by the same row id the converter uses: the source path for a
    recording, the series UID for a DICOM series. The per-file form reads it to
    say what THIS file already answers, which is not the same question as what
    every file of its kind answers.
    """
    if df is None or not len(df) or "_derived_fields" not in df.columns:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        row_id = str(
            row.get("source_file", "") or row.get("series_uid", "") or ""
        ).strip()
        raw = row.get("_derived_fields")
        if not row_id or not isinstance(raw, str) or not raw:
            continue
        try:
            derived = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(derived, dict):
            continue

        datatype = str(row.get("proposed_datatype", "") or "").strip()
        suffix = str(row.get("bids_guess_suffix", "") or "").strip()
        kept = {
            name: value
            for name, value in derived.items()
            if _is_answer(value)
            and name not in CONVERTER_PRIVATE
            and (
                not datatype
                or schema_mod.field_applies(name, datatype, suffix or datatype)
            )
        }
        if kept:
            out[row_id] = kept
    return out

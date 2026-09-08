"""Schema-driven sidecar repairs that apply to every datatype.

The datatype-specific fixups next door know what a particular modality needs.
This one knows only what the SCHEMA says, and so applies everywhere:

* :func:`repair_key_names` fixes a key whose spelling differs from the
  standard's only in case. Real example, and the reason this exists: BIDS
  spells the field ``MISCChannelCount`` for EEG and ``MiscChannelCount`` for
  MEG and iEEG, and mne-bids writes the MEG spelling into EEG sidecars. The
  value is right and the key is not, so every EEG dataset carried an undeclared
  field while the declared one read as missing.
* :func:`repair_array_types` wraps a bare scalar in a list where the schema
  types the field as an array of one entry per frame or per run. Generalised
  from the PET fixup, which had it first because dcm2niix does this on
  single-frame PET.
* :func:`repair_datetimes` zero-pads an acquisition time whose seconds field
  dcm2niix wrote unpadded. ``11:22:1.995`` is not a date-time to any validator,
  and it is copied on into ``scans.tsv``, so one converter slip is reported
  twice.
* :func:`fill_agnostic_fields` writes the values that belong to the DATASET
  rather than to a modality, chiefly where the study was done, into every
  sidecar whose datatype declares them. They used to reach electrophysiology
  alone, because the only enrichment pass that ran was the EEG/MEG one.

None of this guesses. A key is renamed only when the schema declares that exact
name modulo case, a value is wrapped only where the schema says array, and a
field is written only when the user supplied it and the schema declares it for
that datatype.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .. import schema as schema_mod
from ..schema.loader import register_cache
from ..recording_meta import RecordingMetaSpec, is_varies, resolve_sidecar_fields

log = logging.getLogger(__name__)


# Values the dataset as a whole carries, mapped to the sidecar key that holds
# them. Agnostic by nature: which building a scanner sits in has nothing to do
# with whether it is a scanner, an amplifier or a dewar. The schema decides
# whether a given datatype actually declares each one.
_AGNOSTIC_FIELDS: dict[str, str] = {
    "institution_name": "InstitutionName",
    "institution_dept": "InstitutionalDepartmentName",
}


@lru_cache(maxsize=256)
def _declared_fields(datatype: str, suffix: str) -> tuple[str, ...]:
    """Every field name the schema declares for this kind of file."""
    try:
        return tuple(
            f.name for f in schema_mod.optional_sidecar_fields(datatype, suffix)
        ) + tuple(
            f.name for f in schema_mod.required_sidecar_fields(datatype, suffix)
        ) + tuple(
            f.name for f in schema_mod.recommended_sidecar_fields(datatype, suffix)
        ) + tuple(
            f.name for f in schema_mod.deprecated_sidecar_fields(datatype, suffix)
        )
    except Exception:  # noqa: BLE001 - an unknown datatype is not an error here
        return ()


@lru_cache(maxsize=256)
def _array_typed(datatype: str, suffix: str) -> frozenset[str]:
    """Fields the schema types as an array for this kind of file.

    Fields the schema leaves untyped (an ``anyOf`` that accepts both a scalar
    and an array) are excluded on purpose: it permits both shapes there, so
    reshaping the converter's choice would be meddling rather than repair.
    """
    out = set()
    for level in (
        schema_mod.required_sidecar_fields,
        schema_mod.recommended_sidecar_fields,
        schema_mod.optional_sidecar_fields,
    ):
        try:
            for f in level(datatype, suffix):
                if f.type == "array":
                    out.add(f.name)
        except Exception:  # noqa: BLE001
            continue
    return frozenset(out)


def repair_key_names(data: dict, datatype: str, suffix: str) -> int:
    """Rename keys that differ from the standard's spelling only in case.

    Returns the number of keys renamed. A key whose correct spelling is already
    present is dropped rather than merged: the correctly-spelled one wins,
    because that is the one every reader will look at.
    """
    declared = _declared_fields(datatype, suffix)
    if not declared:
        return 0
    canonical = {name.lower(): name for name in declared}

    renamed = 0
    for key in list(data):
        proper = canonical.get(key.lower())
        if proper is None or proper == key:
            continue
        if proper in data:
            data.pop(key)
        else:
            data[proper] = data.pop(key)
        renamed += 1
    return renamed


def repair_array_types(data: dict, datatype: str, suffix: str) -> int:
    """Wrap a bare scalar where the schema declares an array. Returns the count."""
    fixed = 0
    for name in _array_typed(datatype, suffix):
        if name in data and not isinstance(data[name], list):
            value = data[name]
            if value is None or value == "":
                continue
            data[name] = [value]
            fixed += 1
    return fixed


# Fields the standard types as a date-time, written by a converter rather than
# by a user. Scoped rather than applied to every string that looks like a date,
# so a free-text field that happens to contain one is never rewritten.
_DATETIME_KEYS: tuple[str, ...] = ("AcquisitionDateTime", "AcquisitionTime")

_DATETIME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}T)?"
    r"(?P<h>\d{1,2}):(?P<m>\d{1,2}):(?P<s>\d{1,2})(?P<frac>\.\d+)?$"
)


def repair_datetimes(data: dict, datatype: str, suffix: str) -> int:
    """Zero-pad an unpadded time field. Returns the count.

    dcm2niix writes the seconds without a leading zero when the DICOM had one,
    producing ``2025-05-26T11:22:1.995000``. BIDS wants a real date-time, so
    that value is rejected, and it is rejected twice: once in the sidecar and
    again in ``scans.tsv``, which is generated from it. Nothing about the
    instant is wrong, only how it was spelled.
    """
    fixed = 0
    for name in _DATETIME_KEYS:
        value = data.get(name)
        if not isinstance(value, str) or not value:
            continue
        m = _DATETIME_RE.match(value)
        if m is None:
            continue
        padded = "%s%02d:%02d:%02d%s" % (
            m.group("date") or "", int(m.group("h")), int(m.group("m")),
            int(m.group("s")), m.group("frac") or "",
        )
        if padded != value:
            data[name] = padded
            fixed += 1
    return fixed


def fill_agnostic_fields(
    data: dict, datatype: str, suffix: str, spec: Optional[RecordingMetaSpec],
) -> int:
    """Write the dataset-wide values the schema declares for this datatype.

    Never overwrites a value already present: a converter that read the
    institution out of a DICOM header knows better than a dataset-wide default.
    """
    if spec is None:
        return 0
    declared = set(_declared_fields(datatype, suffix))
    if not declared:
        return 0

    written = 0
    for attr, key in _AGNOSTIC_FIELDS.items():
        if key not in declared or key in data:
            continue
        value = getattr(spec.defaults, attr, None)
        if value in (None, "") or is_varies(value):
            continue
        data[key] = value
        written += 1
    return written


def apply_sequence_template(
    data: dict,
    datatype: str,
    suffix: str,
    task: Optional[str],
    spec: Optional[RecordingMetaSpec],
    row_id: str = "",
    row_values: Optional[dict] = None,
) -> int:
    """Write what the chain resolved for this file. Returns how many landed.

    Reads :func:`resolve_sidecar_fields`, the single chain, rather than the
    templates alone. Before this, templates were applied here while the
    acquisition blocks were applied in the EEG/MEG fixup, so the same field
    could be written by either pass and the winner depended on their order.

    What the user stated WINS over what the converter wrote.

    It used to be the other way round: an existing value stood unless the row
    itself contradicted it, on the reasoning that a file's own header beats a
    statement about a class of files. That reasoning is wrong about who is
    talking. Nothing in this chain is a guess; every layer of it is somebody
    having typed an answer into a form, and the form only offers a field at all
    when it is worth asking about. A user who opens "already answered by the
    conversion" and corrects the manufacturer has said the header is wrong,
    which is the entire reason that block is editable. Skipping their answer
    made the correction vanish on the next run with no explanation.

    The converter still supplies everything nobody stated, which is almost all
    of it.
    """
    resolved = resolve_sidecar_fields(
        spec, datatype, suffix, row_id=row_id, task=task, row_values=row_values,
    )
    # A MEG recording only carries an EEG reference if it carries EEG at all.
    # The schema declares EEGReference for MEG because simultaneous EEG is
    # common, not because every MEG run has it, and the converter has already
    # counted the channels. Writing a dataset-wide reference into a MEG sidecar
    # with no EEG channels states something untrue about the recording.
    simultaneous_eeg = bool(data.get("EEGChannelCount") or 0)

    written = 0
    for name, field in resolved.items():
        if (
            name in ("EEGReference", "EEGGround")
            and datatype == "meg"
            and not simultaneous_eeg
        ):
            continue
        value = _as_schema_types_it(name, field.value)
        if data.get(name) == value:
            continue
        data[name] = value
        written += 1
    return written


def _as_schema_types_it(name: str, value):
    """``value`` in the shape the schema declares, where that is unambiguous.

    Only numeric arrays are touched, and only when an item spells a number.
    A number written as its own decimal string is the same number, and storing
    it as text fails validation for an answer that is otherwise right.

    This exists because the answers are PERSISTED. A metadata template saved
    before the form learned to read numeric arrays holds ``["0"]`` on disk, and
    that file keeps being applied on every later run. Repairing it as it is
    applied means a stored answer does not have to be re-entered to become
    valid, and it also covers a hand-edited template.
    """
    if not isinstance(value, list) or not any(isinstance(x, str) for x in value):
        return value
    from .. import schema as schema_mod

    try:
        info = schema_mod.field_metadata(name)
    except (KeyError, ValueError):
        return value
    if info.type != "array" or info.item_type not in ("number", "integer"):
        return value
    return [_number_or_text(item) for item in value]


def _number_or_text(value):
    """``value`` as the number it spells, or unchanged if it spells none."""
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _rewrite_each_sidecar(root: Path, apply) -> int:
    """Walk every sidecar under ``root`` and let ``apply`` edit it in place.

    Walks the tree rather than a task list, so a sidecar a backend wrote for an
    output nobody enumerated, a fieldmap split or a multi-echo series, is
    treated like any other. Returns how many files changed.
    """
    from ..editor.bidsmgr_checks import infer_datatype_suffix

    changed = 0
    for sidecar in sorted(root.rglob("*.json")):
        if ".bidsmgr" in sidecar.parts:
            continue
        datatype, suffix = infer_datatype_suffix(sidecar, root)
        if not (datatype and suffix):
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue

        before = json.dumps(data, sort_keys=True)
        if not apply(sidecar, data, datatype, suffix):
            continue
        if json.dumps(data, sort_keys=True) == before:
            continue
        try:
            sidecar.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:  # noqa: BLE001
            log.warning("could not rewrite %s: %s", sidecar, exc)
            continue
        changed += 1
    return changed


def repair_converter_output(staging: Path) -> int:
    """Fix what the CONVERTER wrote. Returns files changed.

    Three repairs, all about the backend's own output rather than about
    anything a user said, which is why this belongs to conversion:

    * a key whose spelling differs from the standard's only in case, the real
      case being that mne-bids writes MEG's ``MiscChannelCount`` into EEG
      sidecars where BIDS spells it ``MISCChannelCount``;
    * a bare scalar where the schema declares an array, which dcm2niix writes
      for a single-frame acquisition;
    * an acquisition time whose seconds field is not zero-padded, which
      dcm2niix also writes and which no validator will accept as a date-time.

    What the USER stated is applied by the metadata step instead: see
    :func:`apply_stated_metadata`. Keeping the two apart means ``bidsmgr-convert``
    produces a faithful conversion and no opinions, which is what the verb says.
    """
    return _rewrite_each_sidecar(
        staging,
        lambda _path, data, datatype, suffix: (
            repair_key_names(data, datatype, suffix)
            + repair_array_types(data, datatype, suffix)
            + repair_datetimes(data, datatype, suffix)
        ),
    )


def apply_stated_metadata(
    bids_root: Path,
    spec: Optional[RecordingMetaSpec],
    inventory=None,
) -> int:
    """Write what the USER stated into every sidecar that takes it.

    The single place user-stated metadata reaches the files. It used to happen
    during conversion, in two passes that both walked the same chain, so which
    one won depended on the order they ran in and a CLI user got opinions from a
    verb that promises a conversion.

    ``inventory`` is the scan table, used to recover which row produced which
    file so a correction aimed at one recording lands on that recording and no
    other, and to read the cells a user typed in the table.
    """
    if spec is None:
        return 0

    row_by_basename, cells_by_row = _rows_from_inventory(inventory)

    def apply(sidecar: Path, data: dict, datatype: str, suffix: str) -> int:
        row_id = _row_id_for(sidecar, row_by_basename)
        return (
            apply_sequence_template(
                data, datatype, suffix, _task_of(sidecar), spec,
                row_id=row_id, row_values=cells_by_row.get(row_id),
            )
            + fill_agnostic_fields(data, datatype, suffix, spec)
            + _apply_extras(data, datatype, spec, row_id, _task_of(sidecar))
        )

    return _rewrite_each_sidecar(bids_root, apply)


def _apply_extras(
    data: dict, datatype: str, spec, row_id: str, task: Optional[str],
) -> int:
    """The supplemental acquisition conditions BIDS does not name.

    Impedance, electrode type, conductive medium, Faraday cage. BIDS permits
    extra keys and these are real facts about how a recording was made, so the
    spec models them explicitly and they are written as given. Not for MEG,
    which has none of them.
    """
    if spec is None or datatype == "meg":
        return 0
    from ..fixups.eeg_sidecar import _extras_to_keys
    from ..recording_meta import resolve_effective

    acq = resolve_effective(spec, row_id, task, datatype).acquisition
    if acq is None or acq.extras is None:
        return 0
    written = 0
    for key, value in _extras_to_keys(acq.extras).items():
        if data.get(key) != value:
            data[key] = value
            written += 1
    return written


# Inventory columns that are also sidecar fields, in the chain's vocabulary.
# The user types them in the table, so the table is where they live.
_CELL_COLUMNS: dict[str, str] = {
    "eeg_reference": "eeg_reference",
    "eeg_ground": "eeg_ground",
    "line_freq": "power_line_freq",
}


def _rows_from_inventory(inventory) -> tuple[dict[str, str], dict[str, dict]]:
    """``(basename -> row_id, row_id -> the cells that are sidecar fields)``."""
    if inventory is None or not len(inventory):
        return {}, {}

    by_basename: dict[str, str] = {}
    cells: dict[str, dict] = {}
    for _, row in inventory.iterrows():
        basename = str(row.get("proposed_basename", "") or "").strip()
        row_id = str(
            row.get("source_file", "") or row.get("series_uid", "") or ""
        ).strip()
        if not basename or not row_id:
            continue
        by_basename[basename] = row_id
        stated = {}
        for column, attr in _CELL_COLUMNS.items():
            value = str(row.get(column, "") or "").strip()
            if value and value.lower() not in ("nan", "none"):
                stated[attr] = value
        if stated:
            cells[row_id] = stated
    return by_basename, cells


_TASK_RE = re.compile(r"_task-([A-Za-z0-9]+)")


def _task_of(path: Path) -> Optional[str]:
    """The BIDS task label in a filename, if it carries one."""
    m = _TASK_RE.search(path.name)
    return m.group(1) if m else None


def _row_id_for(sidecar: Path, row_by_basename: dict[str, str]) -> str:
    """The row that produced this sidecar, by its basename.

    A backend may add its own tail to a name it was given: dcm2niix writes
    ``..._bold_e2`` for a second echo and ``..._ph`` for a phase image. Those
    are still that row's outputs, so the longest task basename the filename
    starts with wins, and an unmatched file simply has no row layer.
    """
    stem = sidecar.name[: -len(".json")]
    if stem in row_by_basename:
        return row_by_basename[stem]
    matches = [b for b in row_by_basename if stem.startswith(b)]
    return row_by_basename[max(matches, key=len)] if matches else ""


# These hold answers about one BIDS version, so they are dropped when it changes.
for _cached in (_declared_fields, _array_typed):
    register_cache(_cached)


__all__ = [
    "apply_sequence_template",
    "apply_stated_metadata",
    "fill_agnostic_fields",
    "repair_array_types",
    "repair_datetimes",
    "repair_converter_output",
    "repair_key_names",
]

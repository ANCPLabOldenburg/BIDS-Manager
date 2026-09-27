"""Make a converted physio recording say what the standard asks it to.

The vendored ``bidsphysio`` writes the three fields a physio sidecar cannot
do without: ``Columns``, ``SamplingFrequency`` and ``StartTime``. It was
written against an earlier BIDS, so everything the standard has asked for
since is simply absent, and the validator reports the same two warnings on
every physio file we produce::

    SIDECAR_KEY_RECOMMENDED: missing recommended field 'PhysioType'
    SIDECAR_KEY_RECOMMENDED: missing recommended field 'TaskName'

**Why this is a fixup and not a patch to the vendored writer.** Teaching
``bidsphysio`` about ``PhysioType`` would hard-code one version of BIDS into
third-party code, and the next version that adds a field would need the same
edit again, in a file whose whole point is that it is somebody else's. This
module instead asks the ACTIVE schema which fields the datatype and suffix
declare, and fills the ones it can derive. A BIDS version that has never
heard of ``PhysioType`` is not given one; a version that adds a field we can
derive gets it with no edit here. That is the same rule the EEG, MEG and PET
sidecar fixups follow, and the same rule guard 9 states for the tool as a
whole.

What can be DERIVED, and nothing else:

``TaskName``
    From the ``task-`` entity, by the standard's own rule. A physio
    recording belongs to a task run and carries the entity already.

``PhysioType``
    From the column names, which BIDS itself standardises, against the
    VALUES THE SCHEMA ALLOWS. The field is not free text: the schema gives
    it an enum, currently ``generic`` and ``eyetrack``. Deriving a value the
    schema has never heard of turns a missing recommended field into a
    validation ERROR, which is worse than leaving it out, so the derived
    value is checked against the enum before it is written and a schema with
    no enum for it is left alone.

Everything else a physio sidecar may carry (``Manufacturer``,
``Description``, the spoiling-gradient trio) is a fact about the equipment
that no converter can read out of a PMU dump. Those are left to the metadata
step, which marks them as unanswered rather than inventing them.

Nothing already written is overwritten: a value in the file is the user's or
the converter's, and this only fills blanks.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .. import schema as schema_mod

log = logging.getLogger(__name__)

#: Column names BIDS standardises for EYE TRACKING. A physio file carrying
#: any of them is an eye-tracking recording; anything else is the ordinary
#: kind. These are column names from the standard, not a guess at the
#: equipment.
_EYETRACK_COLUMNS: frozenset[str] = frozenset({
    "x_coordinate", "y_coordinate", "pupil_size",
})

#: The two values the field takes, named so the mapping above can be read
#: against them. Both are still checked against the schema's own enum
#: before anything is written.
_EYETRACK = "eyetrack"
_GENERIC = "generic"

#: Only ``[0-9a-zA-Z+]`` survives into a task label, so a ``TaskName``
#: derived back out of one keeps exactly those characters.
_LABEL_CHARS = re.compile(r"[^0-9a-zA-Z+]")


def enrich_physio_sidecars(staging: Path) -> int:
    """Fill the derivable fields of every physio sidecar under ``staging``.

    Returns the number of files changed. Safe to run on a tree with no
    physio in it, which is most of them.
    """
    staging = Path(staging)
    if not staging.is_dir():
        return 0

    changed = 0
    for sidecar in sorted(staging.rglob("*_physio.json")):
        if _enrich_one(staging, sidecar):
            changed += 1
    return changed


def _enrich_one(staging: Path, sidecar: Path) -> bool:
    data = _read_json(sidecar)
    if data is None:
        return False

    datatype = _datatype_of(staging, sidecar)
    if not datatype:
        return False
    declared = _declared_fields(datatype, "physio")
    if not declared:
        return False

    before = dict(data)

    if "TaskName" in declared and not data.get("TaskName"):
        task = _entity_value(sidecar.name, "task")
        if task:
            data["TaskName"] = task

    if "PhysioType" in declared and not data.get("PhysioType"):
        physio_type = _physio_type(
            data.get("Columns"), _allowed_values(datatype, "PhysioType"),
        )
        if physio_type:
            data["PhysioType"] = physio_type

    if data == before:
        return False
    _write_json(sidecar, data)
    log.info(
        "physio: filled %s in %s",
        ", ".join(sorted(set(data) - set(before))) or "nothing",
        sidecar.name,
    )
    return True


def _physio_type(columns: object, allowed: frozenset[str]) -> str:
    """The ``PhysioType`` the columns imply, if the schema allows that value.

    The columns the standard defines for eye tracking make a file an
    eye-tracking recording; everything else is ``generic``, which the field
    documents as its default.

    ``allowed`` is the schema's own enum, and a value outside it is not
    written. Guessing at a vocabulary is how a missing RECOMMENDED field
    became a validation ERROR the first time this was written: it derived
    "cardiac" and "trigger" from the column names, which read perfectly well
    and are not values this field takes.
    """
    if not isinstance(columns, list) or not allowed:
        return ""
    names = {str(name).strip().lower() for name in columns}
    kind = _EYETRACK if names & _EYETRACK_COLUMNS else _GENERIC
    return kind if kind in allowed else ""


def _allowed_values(datatype: str, field: str) -> frozenset[str]:
    """The values the ACTIVE schema lets ``field`` take, or empty for any."""
    try:
        for info in schema_mod.sidecar_fields(datatype, "physio"):
            if info.name == field:
                return frozenset(info.enum or ())
    except Exception:  # noqa: BLE001 - a conversion must not fail over this
        pass
    return frozenset()


def _declared_fields(datatype: str, suffix: str) -> set[str]:
    """Field names the ACTIVE schema declares for this kind of file."""
    try:
        return {f.name for f in schema_mod.sidecar_fields(datatype, suffix)}
    except Exception:  # noqa: BLE001 - a conversion must not fail over this
        log.debug("schema has no sidecar rules for %s/%s", datatype, suffix)
        return set()


def _datatype_of(staging: Path, sidecar: Path) -> str:
    """The datatype folder the sidecar sits in, per the schema."""
    known = set(schema_mod.list_datatypes())
    try:
        parts = sidecar.relative_to(staging).parts
    except ValueError:
        parts = sidecar.parts
    for part in reversed(parts[:-1]):
        if part in known:
            return part
    return ""


def _entity_value(name: str, entity: str) -> str:
    """The value of ``entity`` in a BIDS basename, or ``""``."""
    for token in name.split("_"):
        if token.startswith(f"{entity}-"):
            return _LABEL_CHARS.sub("", token[len(entity) + 1:])
    return ""


def _read_json(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    try:
        path.write_text(
            json.dumps(data, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("could not write %s: %s", path, exc)


__all__ = ["enrich_physio_sidecars"]

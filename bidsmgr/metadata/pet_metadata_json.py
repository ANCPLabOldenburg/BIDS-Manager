"""Read a PET metadata JSON keyed by BIDS field names.

PET needs metadata no scanner records: what was injected, how much, when,
by what route. That has to come from the operator, and every tool in this
space asks for it differently. BIDS Manager already accepts two shapes, the
GUI form and a dose spreadsheet (``metadata/pet_spreadsheet``). This is the
third, and it is deliberately **not a new format**: it is the shape
pypet2bids already accepts through ``--set-default-metadata-json``, so a lab
that has one of those files can hand us the same file.

The format
----------

A flat object keyed by BIDS sidecar field names applies to every PET run::

    {"TracerName": "FDG", "InjectedRadioactivity": 81.24,
     "InjectedRadioactivityUnits": "MBq", "ModeOfAdministration": "bolus"}

An object whose keys look like subject labels or BIDS names scopes it::

    {"sub-01": {"InjectedRadioactivity": 81.24},
     "sub-02": {"InjectedRadioactivity": 74.10}}

pypet2bids wraps its own file in a ``nifti_json`` block alongside
``blood_json`` and ``blood_tsv``; that wrapper is unwrapped here so their
file works unmodified. Their blood blocks are ignored rather than rejected:
blood metadata reaches us through the recording-metadata form, and silently
dropping a block we do not read is better than refusing a file over it.

Two rules that are ours, not theirs
-----------------------------------

**A key BIDS does not define is reported, never written.** An unrestricted
overlay is a way to put arbitrary JSON into somebody's dataset. Unknown keys
are logged with the file they came from and dropped.

**A value is only ever a suggestion until the conversion applies it**, the
same contract the scan-time tracer and montage hints use. This module
returns specs; it writes nothing.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from ..recording_meta.models import PetAcquisitionSpec

log = logging.getLogger(__name__)

# The wrapper pypet2bids puts its PET sidecar block in. Their file also
# carries ``blood_json`` and ``blood_tsv``, which are not ours to read.
_NIFTI_BLOCK = "nifti_json"
_IGNORED_BLOCKS = ("blood_json", "blood_tsv")

# A key that scopes a block rather than naming a field. ``sub-01`` and a full
# BIDS basename both start this way, and BIDS field names never do.
_SCOPE_RE = re.compile(r"^sub-[0-9a-zA-Z]+")


def _bids_to_field() -> dict[str, str]:
    """BIDS sidecar key -> ``PetAcquisitionSpec`` field, inverted from chain.

    Inverted from the maps the tool already uses to WRITE these fields, so a
    field added to the spec becomes readable here without anyone remembering
    to add it in two places.
    """
    from ..recording_meta.chain import PET_LIST_TO_BIDS, PET_SCALAR_TO_BIDS

    return {
        bids: field
        for mapping in (PET_SCALAR_TO_BIDS, PET_LIST_TO_BIDS)
        for field, bids in mapping.items()
    }


def _spec_from_block(block: dict, source: str) -> Optional[PetAcquisitionSpec]:
    """One flat ``{BIDS key: value}`` block into a spec, or ``None``."""
    mapping = _bids_to_field()
    fields: dict[str, object] = {}
    unknown: list[str] = []
    for key, value in block.items():
        if value is None or value == "":
            continue
        field = mapping.get(key)
        if field is None:
            unknown.append(key)
            continue
        fields[field] = value
    if unknown:
        log.warning(
            "PET metadata %s: ignoring %d key(s) BIDS does not define for a "
            "PET sidecar: %s",
            source, len(unknown), ", ".join(sorted(unknown)[:8]),
        )
    if not fields:
        return None
    try:
        return PetAcquisitionSpec(**fields)
    except Exception:  # noqa: BLE001 - try the shapes BIDS also allows
        pass

    # BIDS types several PET fields as ARRAYS that the spec models as
    # scalars, because a value is stated once and the conversion wraps it.
    # A file written for pet2bids, or copied out of a finished sidecar,
    # carries the array. Unwrapping a one-element list is reading the file
    # the standard describes rather than refusing it over a bracket.
    relaxed = {
        name: (value[0] if isinstance(value, list) and len(value) == 1
               and not _expects_list(name) else value)
        for name, value in fields.items()
    }
    try:
        return PetAcquisitionSpec(**relaxed)
    except Exception as exc:  # noqa: BLE001 - a bad file must not abort a conversion
        log.warning("PET metadata %s: could not read the values (%s)", source, exc)
        return None


def _expects_list(field: str) -> bool:
    """Does the spec itself model this field as a list?"""
    from ..recording_meta.chain import PET_LIST_TO_BIDS

    return field in PET_LIST_TO_BIDS


def read_pet_metadata_json(path: Path) -> dict[str, PetAcquisitionSpec]:
    """``{scope: spec}`` from a PET metadata JSON. ``""`` scopes everything.

    The empty-string key is the dataset default, matching how the rest of
    the tool expresses "applies unless a row overrides it".

    An unreadable or unrecognised file yields ``{}`` and a warning. A
    metadata import must never be able to abort a conversion: the worst it
    should cost is the metadata.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("could not read PET metadata %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        log.warning("PET metadata %s is not a JSON object; ignoring", path)
        return {}

    # Unwrap pypet2bids' container so their file works unmodified.
    if _NIFTI_BLOCK in data and isinstance(data[_NIFTI_BLOCK], dict):
        ignored = [b for b in _IGNORED_BLOCKS if b in data]
        if ignored:
            log.info(
                "PET metadata %s: reading %r, ignoring %s",
                path.name, _NIFTI_BLOCK, ", ".join(ignored),
            )
        data = data[_NIFTI_BLOCK]

    scoped = {
        key: value for key, value in data.items()
        if isinstance(value, dict) and _SCOPE_RE.match(str(key))
    }

    out: dict[str, PetAcquisitionSpec] = {}
    if scoped:
        for key, block in scoped.items():
            spec = _spec_from_block(block, f"{path.name}[{key}]")
            if spec is not None:
                out[str(key)] = spec
        # Anything NOT scoped, alongside scoped blocks, is the default.
        flat = {k: v for k, v in data.items() if k not in scoped}
        if flat:
            spec = _spec_from_block(flat, path.name)
            if spec is not None:
                out[""] = spec
    else:
        spec = _spec_from_block(data, path.name)
        if spec is not None:
            out[""] = spec

    log.info(
        "PET metadata %s: %d block(s) (%s)",
        path.name, len(out),
        "dataset default" if list(out) == [""] else ", ".join(sorted(out)) or "none",
    )
    return out


__all__ = ["read_pet_metadata_json"]

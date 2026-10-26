"""Where a scan's subject labels come from: numbers, or the DICOM identity.

By default every subject the scan tells apart is numbered (``sub-001``,
``sub-002``, ...). A lab whose participants already have IDs (a code in the
DICOM PatientID, an anonymised name typed by the operator) can have the label
taken from one of the identity fields instead:

- ``patient_id``: DICOM PatientID (``OL_0001`` -> ``sub-OL0001``);
- ``given_name`` / ``family_name``: the parts of DICOM PatientName.

The value is curated into a valid BIDS label: accents lose their marks
(``Müller`` -> ``Muller``), everything but letters and digits goes
(``Smith Jr.`` -> ``SmithJr``), so the label is valid under every BIDS
version. A subject whose field is empty, or holds an anonymisation stamp
(``ANONYMOUS``, ``DE-IDENTIFIED``), is numbered as before, with numbers that no
taken label uses. Two subjects whose values curate to the same label keep it
apart with a number (``Smith``, ``Smith2``): two people never share a label.

The subjects are still told apart the same way (``subject_identity``): this
only decides what each is called.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping, Optional

NUMBER = "number"
PATIENT_ID = "patient_id"
GIVEN_NAME = "given_name"
FAMILY_NAME = "family_name"

#: Every source, with its name for a person.
SOURCES: dict[str, str] = {
    NUMBER: "Numbers (sub-001, sub-002, ...)",
    PATIENT_ID: "Patient ID",
    GIVEN_NAME: "Given name",
    FAMILY_NAME: "Family name",
}

#: Position of each field in an identity tuple ``(PatientID, given, family)``.
_FIELD = {PATIENT_ID: 0, GIVEN_NAME: 1, FAMILY_NAME: 2}

_NOT_LABEL = re.compile(r"[^0-9A-Za-z]+")


def curate(value: object) -> str:
    """``value`` as a BIDS label: ASCII letters and digits only."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _NOT_LABEL.sub("", text)


def _usable(value: str) -> bool:
    from .subject_identity import _looks_like_placeholder_token

    return bool(value) and not _looks_like_placeholder_token(value)


def assign(clusters: Mapping[object, Iterable[tuple[str, str, str]]],
           source: str = NUMBER) -> dict[object, str]:
    """``{cluster: "sub-<label>"}`` for every cluster.

    ``clusters`` maps each subject (a cluster key, in the order to number
    them) to its members' identity tuples ``(PatientID, given, family)``.
    With ``source`` a field, the first usable value of that field among the
    members gives the label; the rest are numbered.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown subject label source {source!r}; choose from "
                         f"{', '.join(SOURCES)}")
    keys = list(clusters)
    named: dict[object, str] = {}
    if source != NUMBER:
        field = _FIELD[source]
        taken: dict[str, int] = {}
        for key in keys:
            value = next((t[field] for t in clusters[key] if _usable(t[field])), "")
            label = curate(value)
            if not label:
                continue
            seen = taken.get(label.lower(), 0) + 1
            taken[label.lower()] = seen
            named[key] = label if seen == 1 else f"{label}{seen}"
    used = {label.lower() for label in named.values()}
    out: dict[object, str] = {}
    n = 0
    for key in keys:
        if key in named:
            out[key] = f"sub-{named[key]}"
            continue
        n += 1
        while f"{n:03d}" in used:
            n += 1
        out[key] = f"sub-{n:03d}"
    return out


def describe(source: Optional[str]) -> str:
    return SOURCES.get(source or NUMBER, SOURCES[NUMBER])


__all__ = ["FAMILY_NAME", "GIVEN_NAME", "NUMBER", "PATIENT_ID", "SOURCES", "assign", "curate",
           "describe"]

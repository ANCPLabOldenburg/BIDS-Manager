"""Whether an image has been defaced, recorded where BIDS says to record it.

BIDS has two fields for this, and both are used:

``DeidentificationMethodCodeSequence``
    An array of DICOM-style code objects. Machine-readable, and the one a tool
    should look at. Our entries carry ``CodingSchemeDesignator: "BIDSManager"``
    so nobody mistakes them for a genuine DICOM code.

``DeidentificationMethod``
    An array of strings. For a person reading the sidecar.

Both are legal on ``anat``, ``pet``, ``func`` and ``dwi`` (checked against the
schema, not assumed).

The file is the source of truth. There is no separate record of what has been
defaced, because a second record is a second thing that can be wrong: a user
who defaces in BIDS Manager, copies the dataset somewhere, and opens it again
should see the state their own sidecar reports.

**Only our own entries are touched.** A dataset defaced by another tool
carries its entries here too, and removing them would be a lie about what was
done to the image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .engines import CODING_SCHEME, ENGINES, Engine

CODE_SEQUENCE = "DeidentificationMethodCodeSequence"
METHOD = "DeidentificationMethod"


def sidecar_for(image: Path) -> Path:
    """The JSON sidecar beside an image, whatever its compound extension."""
    image = Path(image)
    name = image.name
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            return image.with_name(name[: -len(ext)] + ".json")
    return image.with_suffix(".json")


def read_sidecar(path: Path) -> dict[str, Any]:
    """Parse a sidecar. A missing or unreadable one reads as empty.

    Unreadable rather than raising, because this is called to decide what to
    show in a list. A malformed sidecar is a reason to offer defacing, not a
    reason for the dialog to fail to open.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _entries(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    seq = sidecar.get(CODE_SEQUENCE)
    if not isinstance(seq, list):
        return []
    return [e for e in seq if isinstance(e, dict)]


def _is_ours(entry: dict[str, Any]) -> bool:
    return str(entry.get("CodingSchemeDesignator", "")) == CODING_SCHEME


def defaced_by_us(sidecar: dict[str, Any]) -> Optional[Engine]:
    """The engine we last defaced with, or ``None``.

    Walks back from the end: the array is a stack, each deface appends and
    each revert pops, so the current state is the last entry we wrote.
    """
    for entry in reversed(_entries(sidecar)):
        if not _is_ours(entry):
            continue
        code = str(entry.get("CodeValue", ""))
        for eng in ENGINES:
            if eng.code_value == code:
                return eng
        # Ours, but an engine or revision this build does not know. It was
        # still defaced, and saying so matters more than naming the engine.
        return None
    return None


def defaced_by_us_at_all(sidecar: dict[str, Any]) -> bool:
    """True when any entry is ours, known engine or not."""
    return any(_is_ours(e) for e in _entries(sidecar))


def defaced_by_others(sidecar: dict[str, Any]) -> list[str]:
    """What other tools claim to have done. Descriptions, for showing a user."""
    out: list[str] = []
    for entry in _entries(sidecar):
        if _is_ours(entry):
            continue
        label = (
            entry.get("CodeMeaning")
            or entry.get("CodeValue")
            or entry.get("CodingSchemeDesignator")
        )
        if label:
            out.append(str(label))
    methods = sidecar.get(METHOD)
    if isinstance(methods, list):
        out += [
            str(m) for m in methods
            if isinstance(m, str) and not m.startswith("BIDS Manager ")
        ]
    return out


def record(sidecar: dict[str, Any], eng: Engine) -> dict[str, Any]:
    """A copy of ``sidecar`` with this engine recorded. Does not write.

    Replaces our previous entry rather than stacking a second one: an image
    defaced twice was defaced once from the original, so the file should say
    what it is, not narrate how it got there. Other tools' entries keep their
    order and their place.
    """
    out = dict(sidecar)

    kept = [e for e in _entries(sidecar) if not _is_ours(e)]
    out[CODE_SEQUENCE] = kept + [eng.deid_entry()]

    methods = sidecar.get(METHOD)
    existing = [m for m in methods if isinstance(m, str)] if isinstance(methods, list) else []
    existing = [m for m in existing if not m.startswith("BIDS Manager ")]
    out[METHOD] = existing + [eng.deid_method()]
    return out


def clear(sidecar: dict[str, Any]) -> dict[str, Any]:
    """A copy with our entries removed, for undoing a deface.

    An empty array is dropped rather than left as ``[]``: an empty
    deidentification sequence asserts that deidentification was considered and
    produced nothing, which is not what a reverted file means.
    """
    out = dict(sidecar)

    kept = [e for e in _entries(sidecar) if not _is_ours(e)]
    if kept:
        out[CODE_SEQUENCE] = kept
    else:
        out.pop(CODE_SEQUENCE, None)

    methods = sidecar.get(METHOD)
    if isinstance(methods, list):
        rest = [
            m for m in methods
            if not (isinstance(m, str) and m.startswith("BIDS Manager "))
        ]
        if rest:
            out[METHOD] = rest
        else:
            out.pop(METHOD, None)
    return out


__all__ = [
    "CODE_SEQUENCE",
    "METHOD",
    "clear",
    "defaced_by_others",
    "defaced_by_us",
    "defaced_by_us_at_all",
    "read_sidecar",
    "record",
    "sidecar_for",
]

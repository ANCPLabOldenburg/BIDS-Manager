"""Demographic normalization + optional participant-spreadsheet import.

Modality-agnostic helpers used by the metadata engine to write a clean
``participants.tsv``. Sex and handedness arrive in many forms (DICOM
``PatientSex`` codes, MNE-derived ``M``/``R`` letters, free-text ``female`` /
``left-handed`` from a hand-kept spreadsheet); these map them to the
BIDS-conventional single letters. An optional participants spreadsheet
(TSV/CSV/XLSX/ODS) keyed by ``participant_id`` can supply or override the
demographic columns.
"""

from __future__ import annotations

import re

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# String -> BIDS letter maps (lower-cased lookup). Numeric codes follow the
# common 0/1/2 (sex) and 1/2/3 (hand) conventions.
_SEX_MAP = {
    "m": "M", "male": "M", "1": "M",
    "f": "F", "female": "F", "2": "F",
    "o": "O", "other": "O", "intersex": "O",
    "u": "", "unknown": "", "0": "",
}
_HAND_MAP = {
    "r": "R", "right": "R", "right-handed": "R", "1": "R",
    "l": "L", "left": "L", "left-handed": "L", "2": "L",
    "a": "A", "ambi": "A", "ambidextrous": "A", "both": "A", "3": "A",
}


def normalize_sex(value: object) -> str:
    """Map a sex value to ``M`` / ``F`` / ``O``; ``""`` when unknown/blank."""
    s = str(value or "").strip()
    if not s:
        return ""
    if s in ("M", "F", "O"):  # already canonical
        return s
    return _SEX_MAP.get(s.lower(), "")


def normalize_handedness(value: object) -> str:
    """Map a handedness value to ``R`` / ``L`` / ``A``; ``""`` when unknown."""
    s = str(value or "").strip()
    if not s:
        return ""
    if s in ("R", "L", "A"):  # already canonical
        return s
    return _HAND_MAP.get(s.lower(), "")


_DICOM_AGE_RE = re.compile(r"^\s*(\d{1,3})\s*([DWMY])\s*$", re.IGNORECASE)

# How many of each DICOM age unit make a year. Days and weeks are approximate
# by nature; BIDS wants years and does not ask for more precision than that.
_AGE_UNIT_YEARS = {"Y": 1.0, "M": 12.0, "W": 52.0, "D": 365.0}


def normalize_age(value: object) -> str:
    """Return an age in YEARS as a bare number, or ``""`` when unusable.

    DICOM writes ``PatientAge`` as three digits and a unit, so a 65-year-old is
    ``065Y`` and an infant may be ``018M``. BIDS wants ``age`` to be a number,
    and the raw DICOM string was being copied into participants.tsv unchanged,
    which the validator rightly rejects as not a number.

    A value that is already numeric passes through. Anything else that cannot
    be read as an age becomes blank rather than a guess, because a wrong age is
    worse than a missing one.
    """
    s = str(value or "").strip()
    if not s:
        return ""

    match = _DICOM_AGE_RE.match(s)
    if match:
        count, unit = int(match.group(1)), match.group(2).upper()
        years = count / _AGE_UNIT_YEARS[unit]
        # Whole years stay whole, so a 65-year-old is "65" and not "65.0".
        return str(int(years)) if years == int(years) else f"{years:.2f}".rstrip("0")

    try:
        number = float(s)
    except ValueError:
        return ""
    return str(int(number)) if number == int(number) else str(number)


def _participant_key(value: object) -> str:
    """Normalise a participant identifier to the ``sub-XXX`` form."""
    s = str(value or "").strip()
    if not s:
        return ""
    return s if s.startswith("sub-") else f"sub-{s}"


def load_participants_table(path: Path) -> dict[str, dict[str, str]]:
    """Read a participants spreadsheet, keyed by ``sub-XXX`` participant id.

    Supports ``.tsv`` / ``.csv`` natively and ``.xlsx`` / ``.ods`` via pandas'
    optional Excel engines (a clear error is logged if the engine is missing).
    The file must carry a ``participant_id`` column; every other column is
    returned verbatim so the caller can pick the demographic fields it knows.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in (".tsv", ".txt"):
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        elif suffix == ".csv":
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        else:  # .xlsx / .ods / .xls
            df = pd.read_excel(path, dtype=str).fillna("")
    except Exception as exc:
        log.warning("could not read participants file %s: %s", path, exc)
        return {}

    if "participant_id" not in df.columns:
        log.warning(
            "participants file %s has no 'participant_id' column; ignoring", path,
        )
        return {}

    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        key = _participant_key(row.get("participant_id"))
        if not key:
            continue
        out[key] = {
            str(c): str(row.get(c, "") or "")
            for c in df.columns
            if c != "participant_id"
        }
    return out


def merge_demographics(
    base: dict[str, dict[str, str]],
    overlay: Optional[dict[str, dict[str, str]]],
) -> dict[str, dict[str, str]]:
    """Overlay participant-file demographics on the inventory lookup.

    A non-empty overlay value wins (the spreadsheet is the human-authored
    source of truth); blank overlay cells leave the inventory value intact.
    """
    if not overlay:
        return base
    merged = {k: dict(v) for k, v in base.items()}
    for pid, fields in overlay.items():
        target = merged.setdefault(pid, {})
        for col, val in fields.items():
            if str(val).strip():
                target[col] = val
    return merged


__all__ = [
    "normalize_sex",
    "normalize_handedness",
    "load_participants_table",
    "merge_demographics",
]

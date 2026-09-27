"""Compare an incoming scan's subjects against an already-converted BIDS dataset.

Powers the specific incremental-conversion hint: when a scanned subject id is
already present on disk, is it the SAME person (and is this a NEW session), or a
DIFFERENT person reusing the id? The comparison uses ``participants.tsv``
(``patient_id`` / ``given_name`` / ``family_name``, written by the metadata
engine) plus the on-disk ``ses-*`` directories. When there is nothing to compare
(no participants.tsv identity), it degrades to a generic heads-up.

Qt-free and pure-data so it is unit-testable without the GUI.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

_SUBNUM_RE = re.compile(r"^sub-(\d+)$")

# Stable marker prefixed to every incremental-collision note written into
# ``issues``. It lets the GUI find and strip a stale collision note so
# the warning can be recomputed against the CURRENT subject id + on-disk state
# (e.g. after the user renames sub-001 -> sub-002 to resolve the clash). The
# wording after the token deliberately avoids the model's error-token
# substrings so the row reads as a warning, not an error.
EXISTING_SUBJECT_TOKEN = "existing-subject"


def read_existing_identities(bids_root: Path) -> dict[str, dict]:
    """Map ``participant_id`` -> identity dict from ``participants.tsv``.

    Returns ``{}`` when the file is absent or unreadable. Identity dict carries
    ``patient_id`` / ``given_name`` / ``family_name`` (blank when the column is
    missing).
    """
    p = Path(bids_root) / "participants.tsv"
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    try:
        with open(p, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                pid = (row.get("participant_id") or "").strip()
                if not pid:
                    continue
                out[pid] = {
                    "patient_id": (row.get("patient_id") or "").strip(),
                    "given_name": (row.get("given_name") or "").strip(),
                    "family_name": (row.get("family_name") or "").strip(),
                }
    except OSError:
        return {}
    return out


def existing_sessions(bids_root: Path, subject: str) -> set[str]:
    """Return the ``ses-*`` directory names already present for ``subject``."""
    sub = Path(bids_root) / subject
    if not sub.is_dir():
        return set()
    return {p.name for p in sub.iterdir() if p.is_dir() and p.name.startswith("ses-")}


def next_free_subject_id(bids_root: Path) -> str:
    """Return the next free zero-padded ``sub-<NNN>`` (max numeric + 1)."""
    mx = 0
    width = 3
    for p in Path(bids_root).glob("sub-*"):
        if not p.is_dir():
            continue
        m = _SUBNUM_RE.match(p.name)
        if m:
            mx = max(mx, int(m.group(1)))
            width = max(width, len(m.group(1)))
    return f"sub-{mx + 1:0{width}d}"


_FIELD_LABEL = {
    "patient_id": "PatientID",
    "given_name": "given name",
    "family_name": "family name",
}


def compare_identity(existing: Optional[dict], scanned: dict) -> tuple[list[str], list[str]]:
    """Return ``(matched, differing)`` field names among the three identity fields.

    Only fields that are non-empty on BOTH sides are comparable. Names are
    compared case-insensitively. A field is ``matched`` when the values are equal,
    ``differing`` otherwise. Fields blank on either side are ignored (no info).
    """
    matched: list[str] = []
    differing: list[str] = []
    if not existing:
        return matched, differing
    for field in ("patient_id", "given_name", "family_name"):
        e = (existing.get(field) or "").strip()
        s = (scanned.get(field) or "").strip()
        if field != "patient_id":
            e, s = e.lower(), s.lower()
        if e and s:
            (matched if e == s else differing).append(field)
    return matched, differing


def identity_match(existing: Optional[dict], scanned: dict) -> Optional[bool]:
    """``True`` if ANY field coincides (possible same subject), ``False`` if
    fields are comparable but none coincide, ``None`` if nothing is comparable.

    A single coincidence (e.g. given name) is treated as a possible match even
    when another field differs (technicians sometimes relabel PatientID per
    session while the name stays the same).
    """
    matched, differing = compare_identity(existing, scanned)
    if matched:
        return True
    if differing:
        return False
    return None


def _labels(fields: list[str]) -> str:
    return ", ".join(_FIELD_LABEL[f] for f in fields)


def classify(
    subject: str,
    scanned_ident: dict,
    scanned_session: str,
    existing_ident: Optional[dict],
    on_disk_sessions: set[str],
    next_free: str,
) -> str:
    """Return a specific hint for a scanned subject whose id is already on disk.

    Any single identity coincidence (PatientID / given name / family name) is
    surfaced as a possible same-subject, even when another field differs.
    """
    matched, differing = compare_identity(existing_ident, scanned_ident)
    is_new_session = bool(scanned_session) and scanned_session not in on_disk_sessions

    if matched and not differing:
        # Every comparable field agrees: confident same subject.
        if is_new_session:
            return (
                f"same subject already in dataset ({subject}; {_labels(matched)} "
                f"match); new session ({scanned_session}) - safe to add"
            )
        sess = f", {scanned_session}" if scanned_session else ""
        return (
            f"re-scan of {subject} (same subject{sess}; {_labels(matched)} match); "
            f"convert merges per the Existing-subjects policy"
        )

    if matched:
        # At least one coincidence but something else differs: flag the possible
        # match and what to check (the relabeled-PatientID case).
        sess_hint = (
            f" If it is, this looks like a new session ({scanned_session})."
            if is_new_session else ""
        )
        return (
            f"{subject} may be the same subject ({_labels(matched)} match) but "
            f"{_labels(differing)} differ - verify.{sess_hint} If it is a "
            f"different person, rename to {next_free}"
        )

    if differing:
        return (
            f"{subject} appears to be a DIFFERENT subject ({_labels(differing)} "
            f"differ); rename to {next_free} if this is a new person"
        )

    # No comparable identity info.
    return (
        f"subject id {subject} already in the dataset; rename if this is a "
        f"different subject"
    )


__all__ = [
    "EXISTING_SUBJECT_TOKEN",
    "read_existing_identities",
    "existing_sessions",
    "next_free_subject_id",
    "compare_identity",
    "identity_match",
    "classify",
]

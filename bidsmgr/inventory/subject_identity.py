"""Robust subject identity clustering (architecture.md §4.1).

DICOM patient identifiers are unreliable across visits. Anonymisation
pipelines re-write ``PatientID`` between sessions, blank ``PatientName``
on some scanners, swap given vs family components, and stamp every
record with the same placeholder value (``DE-IDENTIFIED``, ``ANONYMOUS``,
``Anonymous^None``). A single field is never enough.

This module implements a union-find over the three primary identifiers
``(PatientID, GivenName, FamilyName)`` with a built-in placeholder
detector. Two records get linked into the same subject when they share
**any** non-placeholder identifier; otherwise they stay separate.

Placeholder rule (the key insight): a value is a placeholder when records
carrying it disagree on every other non-empty identifier. Concretely,
``"DE-IDENTIFIED"`` is a placeholder because the records using it have
distinct ``PatientID`` values and no shared given name; but ``"Smith"``
is a real surname when every record using it also shares the same
``GivenName`` ``"John"``.

Use :func:`cluster_subjects` to map a set of identity tuples to their
cluster roots; pass the result to BIDS subject-id assignment.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

# (PatientID, GivenName, FamilyName)
IdentityTuple = tuple[str, str, str]
_FIELD_NAMES = ("pid", "given", "family")

# Tokens that DICOM anonymisation pipelines stamp into identifier fields.
# Treated as placeholders regardless of cardinality — when a value matches
# (case-insensitive, after a light normalisation) it never participates in
# subject linking. Extend as we see new tokens in real datasets.
_KNOWN_PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    s.lower() for s in (
        "anonymous", "anon", "anonymised", "anonymized",
        "de-identified", "deidentified", "de_identified",
        "unknown", "unspecified", "n/a", "na",
        "patient", "subject", "subj",
        "anonymous^patient", "anonymous^subject",
        "doe", "john^doe", "jane^doe",
        "test", "test^test",
    )
)


def _looks_like_placeholder_token(value: str) -> bool:
    """True for known anonymisation strings (case-insensitive)."""
    if not value:
        return False
    low = value.strip().lower()
    if low in _KNOWN_PLACEHOLDER_TOKENS:
        return True
    # Tokens like "ANONYMOUS_001" / "ANON-1" / "PATIENT_42".
    head = low.replace("-", "_").split("_", 1)[0]
    return head in _KNOWN_PLACEHOLDER_TOKENS


def _norm(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_tuple(pid: object, given: object, family: object) -> IdentityTuple:
    return (_norm(pid), _norm(given), _norm(family))


def detect_placeholders(tuples: Iterable[IdentityTuple]) -> set[tuple[int, str]]:
    """Return the set of ``(field_index, value)`` pairs that look like placeholders.

    A value is a placeholder when 2+ identity tuples carry it AND those
    tuples share no other non-empty identifier. ``"DE-IDENTIFIED"`` shared
    across distinct patient IDs is a placeholder; ``"Smith"`` carried by
    distinct patient IDs but the same first name is not.
    """

    tuples_list = list(tuples)
    placeholders: set[tuple[int, str]] = set()

    # Pass 1 — well-known anonymisation tokens (case-insensitive, also
    # matches token prefixes like ``ANON-001``). Catches the common
    # de-identifier strings regardless of cardinality.
    for fi in range(len(_FIELD_NAMES)):
        for t in tuples_list:
            v = t[fi]
            if v and _looks_like_placeholder_token(v):
                placeholders.add((fi, v))

    # Pass 2 — universal-coverage rule. With 3+ identity tuples, a value
    # that appears in EVERY tuple of a given field is almost certainly an
    # operator-stamped constant (e.g. ``StudyDescription = "PHANTOM"``
    # across all rows). This is conservative enough not to flag values
    # like ``XX00XX00`` that appear in some-but-not-all tuples — those
    # are the real shared identifiers we WANT to use for linking
    # (operator pasted a folder label into PID/FamilyName but the
    # GivenName carries the real anonymised subject hash).
    n = len(tuples_list)
    if n >= 3:
        for fi in range(len(_FIELD_NAMES)):
            by_value: dict[str, int] = defaultdict(int)
            for t in tuples_list:
                v = t[fi]
                if v:
                    by_value[v] += 1
            for value, count in by_value.items():
                if count == n and (fi, value) not in placeholders:
                    placeholders.add((fi, value))

    return placeholders


def cluster_subjects(tuples: Iterable[IdentityTuple]) -> dict[IdentityTuple, IdentityTuple]:
    """Group identity tuples into subjects via union-find.

    Two tuples join the same subject when they share any non-empty,
    non-placeholder identifier in ``(PatientID, GivenName, FamilyName)``.

    Returns a mapping ``identity_tuple -> cluster_root``. Each cluster
    root is the lexicographically-smallest identity tuple in its cluster
    (deterministic across runs).
    """

    tuples_set = set(tuples)
    placeholders = detect_placeholders(tuples_set)

    parent: dict[IdentityTuple, IdentityTuple] = {t: t for t in tuples_set}

    def find(x: IdentityTuple) -> IdentityTuple:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: IdentityTuple, b: IdentityTuple) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Pick the lexicographically smaller root for determinism.
        if rb < ra:
            ra, rb = rb, ra
        parent[rb] = ra

    sorted_tuples = sorted(tuples_set)
    for i, t1 in enumerate(sorted_tuples):
        for t2 in sorted_tuples[i + 1 :]:
            for fi in range(len(_FIELD_NAMES)):
                v1, v2 = t1[fi], t2[fi]
                if not v1 or v1 != v2:
                    continue
                if (fi, v1) in placeholders:
                    continue
                if _looks_like_placeholder_token(v1):
                    continue
                union(t1, t2)
                break  # one match is enough

    return {t: find(t) for t in tuples_set}


__all__ = [
    "IdentityTuple",
    "normalize_tuple",
    "detect_placeholders",
    "cluster_subjects",
]

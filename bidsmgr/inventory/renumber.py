"""Give the people in a scan new subject numbers.

A second batch of data scanned into a dataset that already has ``sub-001``
and ``sub-002`` often comes back as ``sub-001`` and ``sub-002`` again: the
scanner numbers people from one in every scan. The collision warning
(``dataset_identity``) says so; this plans the fix: each distinct PERSON in
the chosen rows gets the next free number.

* A person is the scanned identity (PatientID and name for DICOM), else the
  current subject label (a recording's folder), so a person's MRI and their
  second session keep one number.
* People are numbered in order of acquisition (study date, then time), then
  by their current label.
* Free means: not a ``sub-`` folder of the dataset, not used by a row left
  out of the selection, and not used by the project's other scans.
* The width is the index-width setting for ``sub``, else the widest number
  already in use, at least three digits.

Qt-free; applying it is ``InventoryTableModel.set_entity`` per row, as one
undoable action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

_NUMERIC = re.compile(r"^(\d+)$")


@dataclass(frozen=True)
class Person:
    """One person among the chosen rows and the number they get."""

    identity: str          # what makes them one person (names included)
    current: str           # their subject label now ("001", "OL4594")
    new: str               # the label they get
    rows: tuple[int, ...]
    #: How the person is SHOWN: the PatientID, never a name (screens get
    #: shared), else the subject label.
    shown: str = ""
    #: The earliest acquisition date of their series ("2026-03-01"), to
    #: tell people apart.
    acquired: str = ""


def _text(df, i, col: str) -> str:
    if col not in df.columns:
        return ""
    v = df.at[i, col]
    return "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()


def _label(participant_id: str) -> str:
    return participant_id[4:] if participant_id.startswith("sub-") else participant_id


def identity(df, i) -> str:
    """Who the row's recording is of: the scanned identity, else its label."""
    parts = [_text(df, i, c) for c in ("PatientID", "GivenName", "FamilyName")]
    if any(parts):
        return " / ".join(p for p in parts if p)
    return _label(_text(df, i, "participant_id"))


def taken_labels(df, chosen: Iterable[int], *, bids_root: Optional[Path] = None,
                 others: Iterable = ()) -> set[str]:
    """Subject labels already spoken for, outside the chosen rows."""
    chosen = set(chosen)
    taken = {_label(_text(df, i, "participant_id")) for i in df.index if i not in chosen}
    if bids_root is not None and Path(bids_root).is_dir():
        taken |= {p.name[4:] for p in Path(bids_root).glob("sub-*") if p.is_dir()}
    for other in others:
        if other is not None and "participant_id" in getattr(other, "columns", ()):
            taken |= {_label(str(v)) for v in other["participant_id"].dropna()}
    taken.discard("")
    return taken


def plan_renumber(df, rows: Sequence[int], taken: Iterable[str], *,
                  width: Optional[int] = None) -> list[Person]:
    """Each distinct person in ``rows`` with the next free number."""
    taken = set(taken)
    people: dict[str, list[int]] = {}
    for i in rows:
        people.setdefault(identity(df, i), []).append(int(i))

    def order(item) -> tuple:
        _who, idx = item
        dates = sorted((_text(df, i, "study_date"), _text(df, i, "acq_time")) for i in idx)
        return (dates[0] if dates else ("", ""), _label(_text(df, idx[0], "participant_id")))

    numbers = [int(m.group(1)) for m in (_NUMERIC.match(t) for t in taken) if m]
    widths = [len(t) for t in taken if _NUMERIC.match(t)]
    width = width or max([3, *widths])
    following = max(numbers, default=0) + 1
    out: list[Person] = []
    for who, idx in sorted(people.items(), key=order):
        while f"{following:0{width}d}" in taken:
            following += 1
        label = f"{following:0{width}d}"
        taken.add(label)
        following += 1
        current = _label(_text(df, idx[0], "participant_id"))
        dates = sorted(d for d in (_text(df, i, "study_date") for i in idx) if d)
        when = dates[0] if dates else ""
        if len(when) == 8 and when.isdigit():
            when = f"{when[:4]}-{when[4:6]}-{when[6:]}"
        out.append(Person(identity=who, current=current, new=label, rows=tuple(idx),
                          shown=_text(df, idx[0], "PatientID") or current, acquired=when))
    return out


def rows_of_person(df, row: int) -> list[int]:
    """Every row of the person ``row`` is of (their other series and
    sessions)."""
    who = identity(df, row)
    return [int(i) for i in df.index if identity(df, i) == who]


__all__ = ["Person", "identity", "plan_renumber", "rows_of_person", "taken_labels"]

"""Keeping curated metadata when a subject is converted again.

Conversion is not a one-shot act. A subject arrives, gets converted, somebody
spends an afternoon in the Editor filling in the fields no scanner records, and
then the subject is converted again: a corrected series, a missing session, a
newer dcm2niix. Until now that second pass decided at **file** level. Replacing
``sub-01_task-rest_bold.json`` threw the afternoon away, and skipping it meant
the corrections from the new pass never arrived.

Neither answer is right, because the two passes know different things. The
converter knows what the scanner recorded. The person knows everything the
scanner did not. So the merge here is by FIELD, and the rule is one sentence:

    **A value a person stated wins. A value only the converter stated loses.**

Which is decidable without guessing, because the two sides differ in kind:

* A field only the fresh conversion has is **added**. It is new knowledge and
  nobody has contradicted it.
* A field only the existing file has is **kept**. The converter never had an
  opinion about it, so it cannot be overruling one.
* A field both have keeps the **existing** value, because that is the one a
  person may have touched, with two exceptions:
  - the existing value is a placeholder (``TODO``, blank): nobody stated it, it
    is what BIDS Manager writes when nothing could answer the field, so the
    fresh value replaces it;
  - the fresh value is not a placeholder and the existing one came from a
    conversion that is being superseded for that field. That case is NOT
    detectable from the files alone, which is why it is not attempted. When a
    user wants the fresh conversion to win outright, they turn preservation off
    rather than have this module guess.

The same idea applies to ``*_scans.tsv``, the one table a re-conversion
rewrites: rows are merged on ``filename``, columns are unioned, and a curated
cell is never replaced by ``n/a``.

Nothing else is merged. Merging an arbitrary table without knowing which column
identifies a row would silently interleave two datasets, and being wrong there
is worse than being conservative.

Qt-free, no I/O beyond the two files it is handed.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# What BIDS Manager writes when nothing could answer a field. It is a note to
# self, not a value, so it never outranks something a converter derived.
PLACEHOLDERS = ("todo", "")

SCANS_KEY = "filename"
NA = "n/a"


def is_placeholder(value: Any) -> bool:
    """Is this a stand-in rather than a stated value?

    Only strings qualify. ``0`` and ``False`` are answers.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in PLACEHOLDERS
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and all(is_placeholder(v) for v in value)
    return False


def merge_sidecar(existing: dict, fresh: dict) -> tuple[dict, list[str]]:
    """Merge a freshly converted sidecar into a curated one.

    Returns the merged mapping and the fields whose curated value was kept in
    the face of a different fresh one, which is the interesting half of what
    happened and the half worth logging.
    """
    out = dict(existing)
    kept: list[str] = []
    for key, fresh_value in fresh.items():
        if key not in out:
            out[key] = fresh_value          # new knowledge, nobody objected
            continue
        if is_placeholder(out[key]):
            out[key] = fresh_value          # a TODO is not an opinion
            continue
        if out[key] != fresh_value and not is_placeholder(fresh_value):
            # Reported only when the fresh pass had a real, different answer.
            # Keeping a stated value over a blank one is not a disagreement
            # and logging it would bury the cases that are.
            kept.append(key)
    return out, kept


def merge_sidecar_files(
    existing_path: Path, fresh_path: Path,
) -> Optional[list[str]]:
    """Merge ``fresh_path`` into ``existing_path`` in place.

    On any read failure the caller's ordinary replace path should run instead,
    so this returns ``None`` rather than half-writing a file. A merge that
    cannot be done correctly must not be done at all.
    """
    existing = _read_json(existing_path)
    fresh = _read_json(fresh_path)
    if existing is None or fresh is None:
        return None
    merged, kept = merge_sidecar(existing, fresh)
    existing_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return kept


def merge_scans_table(existing_rows, fresh_rows) -> tuple[list[str], list[dict]]:
    """Merge two ``*_scans.tsv`` bodies on the ``filename`` column.

    Column order follows the existing table, with any new columns appended, so
    a curated column the user added stays where they put it.
    """
    fields: list[str] = []
    for row in list(existing_rows) + list(fresh_rows):
        for key in row:
            if key not in fields:
                fields.append(key)

    by_name: dict[str, dict] = {}
    order: list[str] = []
    for row in existing_rows:
        name = row.get(SCANS_KEY, "")
        by_name[name] = dict(row)
        order.append(name)
    for row in fresh_rows:
        name = row.get(SCANS_KEY, "")
        if name not in by_name:
            by_name[name] = dict(row)
            order.append(name)
            continue
        current = by_name[name]
        for key, value in row.items():
            if key not in current or _is_blank_cell(current[key]):
                current[key] = value
    return fields, [by_name[n] for n in order]


def merge_scans_files(existing_path: Path, fresh_path: Path) -> Optional[int]:
    """Merge one ``*_scans.tsv`` into another. Returns the row count written."""
    existing = _read_tsv(existing_path)
    fresh = _read_tsv(fresh_path)
    if existing is None or fresh is None:
        return None
    if SCANS_KEY not in (existing[0] or []) or SCANS_KEY not in (fresh[0] or []):
        # Without the column that identifies a row there is no safe merge.
        return None
    fields, rows = merge_scans_table(existing[1], fresh[1])
    with existing_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fields, delimiter="\t", lineterminator="\n",
            restval=NA, extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def merge_file(existing_path: Path, fresh_path: Path):
    """Merge whatever the pair is, if it is a kind we can merge safely.

    Returns ``None`` when it is not, which tells the caller to fall back to its
    ordinary behaviour. The two kinds are JSON sidecars and ``*_scans.tsv``.
    """
    name = existing_path.name
    if name.endswith(".json"):
        return merge_sidecar_files(existing_path, fresh_path)
    if name.endswith("_scans.tsv"):
        return merge_scans_files(existing_path, fresh_path)
    return None


def is_mergeable(path: Path) -> bool:
    """Would :func:`merge_file` attempt this pair?"""
    name = Path(path).name
    return name.endswith(".json") or name.endswith("_scans.tsv")


# ---------------------------------------------------------------------------
# Reading, defensively: a merge that cannot be done correctly is not done
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.debug("cannot merge %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _read_tsv(path: Path):
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            return list(reader.fieldnames or []), list(reader)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        log.debug("cannot merge %s: %s", path, exc)
        return None


def _is_blank_cell(value: Any) -> bool:
    return value is None or str(value).strip() in ("", NA)


__all__ = [
    "NA",
    "PLACEHOLDERS",
    "is_mergeable",
    "is_placeholder",
    "merge_file",
    "merge_scans_files",
    "merge_scans_table",
    "merge_sidecar",
    "merge_sidecar_files",
]

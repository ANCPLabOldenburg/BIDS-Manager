"""Repairs to a BIDS tabular file, applied across as many as you like.

The bulk editor started JSON-only, which left half of every dataset out: a
``participants.tsv`` with a missing column, an ``events.tsv`` whose ``onset``
holds text, a ``channels.tsv`` whose columns are in the wrong order. Those are
the findings that repeat identically across a hundred files and are exactly
what a bulk repair is for.

Three operations, chosen because each has one obviously right answer:

**Add a column.** The standard says a column must be there and it is not. The
value is the user's to give; ``n/a`` is offered because BIDS defines it as
"not available", which is true of a column nobody has filled in.

**Coerce a column's values.** The schema declares the column's type and some
cells do not match. Only the cells that fail are touched, and a cell that
cannot be converted is left alone rather than replaced with a guess: losing a
value to make a validator quiet is the worst outcome available here.

**Reorder columns.** The standard defines an order for the columns it knows
about. Reordering moves no data, so it is the one repair that cannot go wrong,
and columns the standard has never heard of keep their relative order at the
end.

Tables are read and written with the ``csv`` module rather than pandas: these
are small files, the whole point is to preserve what is there, and pandas
would helpfully retype a column on the way through.

Qt-free.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# BIDS's own "no value here". Not the same as an empty cell, which is why a
# generated column says this rather than nothing.
NA = "n/a"


@dataclass
class TsvTable:
    """A parsed tabular file: a header, and rows as lists of strings."""

    header: list[str]
    rows: list[list[str]]

    def column(self, name: str) -> list[str]:
        if name not in self.header:
            return []
        idx = self.header.index(name)
        return [r[idx] if idx < len(r) else "" for r in self.rows]

    def to_text(self) -> str:
        out = io.StringIO()
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(self.header)
        writer.writerows(self.rows)
        return out.getvalue()


def read_table(path: Path) -> Optional[TsvTable]:
    """Parse a ``.tsv``. ``None`` when unreadable or empty."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        log.debug("cannot read %s: %s", path, exc)
        return None
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    try:
        rows = [r for r in reader]
    except csv.Error as exc:
        log.debug("cannot parse %s: %s", path, exc)
        return None
    if not rows:
        return None
    header = [h.strip() for h in rows[0]]
    return TsvTable(header=header, rows=[list(r) for r in rows[1:]])


# --------------------------------------------------------------------------
# What the schema says about a column


def column_type(path: Path, column: str, root: Optional[Path] = None) -> str:
    """The type the standard declares for ``column`` in this kind of table.

    One of ``"number"``, ``"integer"``, ``"string"`` or ``""`` when the
    standard says nothing, in which case nothing here will touch it.
    """
    suffix = _suffix_of(Path(path).name)
    if not suffix:
        return ""
    try:
        from .. import schema as schema_mod

        info = schema_mod.column_metadata(suffix, column)  # type: ignore[attr-defined]
        return str(getattr(info, "type", "") or "")
    except Exception:  # noqa: BLE001 - the adapter may not expose columns
        pass
    # Fall back to the handful the standard fixes for every dataset. These are
    # the ones a bulk repair is actually asked for; anything else is left
    # alone rather than guessed at.
    known = {
        "onset": "number",
        "duration": "number",
        "sample": "integer",
        "response_time": "number",
        "age": "number",
        "acq_time": "string",
        "sampling_frequency": "number",
        "low_cutoff": "number",
        "high_cutoff": "number",
    }
    return known.get(column, "")


def _suffix_of(name: str) -> str:
    stem = name[:-4] if name.endswith(".tsv") else name
    parts = stem.split("_")
    last = parts[-1] if parts else ""
    return "" if "-" in last else last


def _as_number(text: str, kind: str) -> Optional[str]:
    """Reformat one cell as the declared type, or ``None`` if it cannot be."""
    raw = text.strip()
    if raw == "" or raw == NA:
        # BIDS's own "no value" is valid for a typed column; leave it.
        return None
    try:
        if kind == "integer":
            return str(int(float(raw)))
        return repr(float(raw))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# The repairs


@dataclass
class TsvChange:
    """What a repair would do to one file, before it does it."""

    path: Path
    rel: str
    applicable: bool
    reason: str = ""
    detail: str = ""          # human summary: "3 of 240 cells"
    n_cells: int = 0


def plan_add_column(
    root: Path, paths: list[Path], column: str, value: str = NA,
) -> list[TsvChange]:
    """Which tables lack ``column``, and how many rows would gain it."""
    out: list[TsvChange] = []
    for p in paths:
        table = read_table(p)
        rel = _rel(root, p)
        if table is None:
            out.append(TsvChange(p, rel, False, "not readable as a table"))
            continue
        if column in table.header:
            out.append(TsvChange(
                p, rel, False, f"already has a {column!r} column",
            ))
            continue
        out.append(TsvChange(
            p, rel, True, detail=f"{len(table.rows)} rows would get {value!r}",
            n_cells=len(table.rows),
        ))
    return out


def plan_coerce_column(
    root: Path, paths: list[Path], column: str,
) -> list[TsvChange]:
    """Which cells in ``column`` do not match the type the standard declares."""
    out: list[TsvChange] = []
    for p in paths:
        rel = _rel(root, p)
        table = read_table(p)
        if table is None:
            out.append(TsvChange(p, rel, False, "not readable as a table"))
            continue
        if column not in table.header:
            out.append(TsvChange(p, rel, False, f"no {column!r} column"))
            continue
        kind = column_type(p, column, root)
        if kind not in ("number", "integer"):
            out.append(TsvChange(
                p, rel, False,
                f"the standard does not declare a numeric type for {column!r}",
            ))
            continue
        values = table.column(column)
        fixable = sum(1 for v in values if _as_number(v, kind) not in (None, v))
        unfixable = sum(
            1 for v in values
            if _as_number(v, kind) is None and v.strip() not in ("", NA)
        )
        detail = f"{fixable} of {len(values)} cells"
        if unfixable:
            detail += f", {unfixable} cannot be converted and stay as they are"
        out.append(TsvChange(
            p, rel, fixable > 0,
            reason="" if fixable else "every cell already matches",
            detail=detail, n_cells=fixable,
        ))
    return out


def plan_reorder(root: Path, paths: list[Path]) -> list[TsvChange]:
    """Which tables have their known columns out of the standard's order."""
    out: list[TsvChange] = []
    for p in paths:
        rel = _rel(root, p)
        table = read_table(p)
        if table is None:
            out.append(TsvChange(p, rel, False, "not readable as a table"))
            continue
        wanted = _ordered_header(p, table.header)
        if wanted == table.header:
            out.append(TsvChange(p, rel, False, "already in order"))
            continue
        out.append(TsvChange(
            p, rel, True,
            detail=" then ".join(wanted[:4]) + (" ..." if len(wanted) > 4 else ""),
        ))
    return out


def _ordered_header(path: Path, header: list[str]) -> list[str]:
    """The standard's order for the columns it knows, then the rest as found."""
    suffix = _suffix_of(Path(path).name)
    try:
        from .. import schema as schema_mod

        declared = list(schema_mod.column_order(suffix))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        declared = _FALLBACK_ORDER.get(suffix, [])
    known = [c for c in declared if c in header]
    rest = [c for c in header if c not in known]
    return known + rest


# The orders the standard fixes and that a real dataset actually hits. Used
# when the schema adapter does not expose column order.
_FALLBACK_ORDER: dict[str, list[str]] = {
    "events": ["onset", "duration", "trial_type", "response_time",
               "value", "sample", "stim_file", "HED"],
    "participants": ["participant_id", "age", "sex", "handedness"],
    "channels": ["name", "type", "units", "description",
                 "sampling_frequency", "low_cutoff", "high_cutoff",
                 "notch", "status", "status_description"],
    "electrodes": ["name", "x", "y", "z", "size", "type",
                   "material", "impedance"],
    "scans": ["filename", "acq_time"],
}


# --------------------------------------------------------------------------
# Doing it


def apply_add_column(
    root: Path, changes: list[TsvChange], column: str, value: str = NA,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Add ``column`` to every applicable table, reversibly."""
    def _edit(table: TsvTable) -> bool:
        if column in table.header:
            return False
        table.header.append(column)
        for row in table.rows:
            while len(row) < len(table.header) - 1:
                row.append("")
            row.append(value)
        return True

    return _apply(root, changes, _edit, f"Add column {column}")


def apply_coerce_column(
    root: Path, changes: list[TsvChange], column: str,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Rewrite the cells in ``column`` that do not match its declared type."""
    def _edit_for(path: Path):
        kind = column_type(path, column)

        def _edit(table: TsvTable) -> bool:
            if column not in table.header or kind not in ("number", "integer"):
                return False
            idx = table.header.index(column)
            touched = False
            for row in table.rows:
                if idx >= len(row):
                    continue
                fixed = _as_number(row[idx], kind)
                if fixed is not None and fixed != row[idx]:
                    row[idx] = fixed
                    touched = True
            return touched

        return _edit

    return _apply(
        root, changes, _edit_for, f"Fix the type of column {column}",
        per_path=True,
    )


def apply_reorder(
    root: Path, changes: list[TsvChange],
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Put each table's known columns into the standard's order."""
    def _edit_for(path: Path):
        def _edit(table: TsvTable) -> bool:
            wanted = _ordered_header(path, table.header)
            if wanted == table.header:
                return False
            index = {c: i for i, c in enumerate(table.header)}
            table.rows[:] = [
                [row[index[c]] if index[c] < len(row) else "" for c in wanted]
                for row in table.rows
            ]
            table.header[:] = wanted
            return True

        return _edit

    return _apply(root, changes, _edit_for, "Reorder columns", per_path=True)


def _apply(root, changes, edit, label, *, per_path: bool = False):
    from ..project.operations import begin_operation

    written: list[Path] = []
    failed: list[tuple[Path, str]] = []
    targets = [c for c in changes if c.applicable]
    if not targets:
        return written, failed
    with begin_operation(Path(root), f"{label} in {len(targets)} file(s)") as op:
        for change in targets:
            table = read_table(change.path)
            if table is None:
                failed.append((change.path, "not readable as a table"))
                continue
            fn = edit(change.path) if per_path else edit
            try:
                if not fn(table):
                    continue
                op.write_text(change.path, table.to_text())
            except OSError as exc:
                failed.append((change.path, str(exc)))
                continue
            written.append(change.path)
    return written, failed


def _rel(root: Path, path: Path) -> str:
    """``path`` relative to ``root``, with forward slashes on every platform.

    BIDS spells relative paths POSIX-style, so anything compared against a
    value read out of a dataset file has to as well. See the longer note on
    the twin of this function in ``editor/rename.py``.
    """
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


__all__ = [
    "NA",
    "TsvChange",
    "TsvTable",
    "apply_add_column",
    "apply_coerce_column",
    "apply_reorder",
    "column_type",
    "plan_add_column",
    "plan_coerce_column",
    "plan_reorder",
    "read_table",
]

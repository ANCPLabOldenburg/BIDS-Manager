"""Whether a dataset's files still agree with each other.

The validator answers "is this legal BIDS". It does not answer "do these
files still describe the same dataset", and that second question is the one
that fails after somebody renames a folder in Finder, deletes a subject by
hand, or copies a session in from somewhere else. Every symptom below is
legal BIDS and wrong:

* a ``*_scans.tsv`` or ``*_sessions.tsv`` row naming a file or session that
  is not there any more, or one missing the row the table should have
* a ``participants.tsv`` row for a subject folder that was deleted, or a
  subject folder with no row
* ``IntendedFor`` and the four other pointer fields aimed at nothing, or
  into another subject; a ``B0FieldSource`` no fieldmap identifies
* a sidecar or a companion table whose recording is gone
* ``dataset_description.json`` and ``CITATION.cff`` disagreeing

None of that is invented here. Each check is a question something in the
tool could already answer (the tables come from ``metadata.tables``, the
links from ``editor.linkage``), asked as a read-only pass and reported in
one place instead of being a side effect of a conversion.

**Read-only until asked.** Nothing in this module writes. :func:`check`
returns findings, each carrying a repair the caller may choose to apply, and
:func:`apply` runs the chosen ones through one operation so the whole pass
is a single undo. Only a repair that cannot lose anything is ``safe`` and
offered ticked: adding a missing row, dropping a row that names nothing.
Deleting a file, dropping a participant's row, rewriting a link or a name
waits for a person to tick it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field as dc_field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional

from .. import schema as schema_mod
from ..project.operations import begin_operation
from . import linkage
from .rename import _skip_top_level

log = logging.getLogger(__name__)


class Kind(str, Enum):
    """What sort of disagreement this is. Groups the report, in this order."""

    SCANS_ROW_MOVED = "scans row names a file renamed outside the application"
    SCANS_ROW_ORPHAN = "scans row names a missing file"
    SCANS_ROW_EXTRA = "scans row that should not be there"
    SCANS_ROW_MISSING = "recording has no scans row"
    SCANS_TABLE_MISSING = "scans table missing"
    SCANS_TIMES_DIFFER = "the parts of one recording have different times"
    SESSION_ROW_ORPHAN = "sessions row names a missing session"
    SESSION_ROW_MISSING = "session has no sessions row"
    SESSIONS_TABLE_MISSING = "sessions table missing"
    TABLE_BLANK_CELLS = "table with empty cells"
    PARTICIPANTS_TABLE_MISSING = "there is no participants.tsv"
    PARTICIPANT_ORPHAN = "participants row names a missing subject"
    PARTICIPANT_MISSING = "subject has no participants row"
    LINK_BROKEN = "link points at a missing file"
    LINK_OTHER_SUBJECT = "IntendedFor points into another subject"
    B0_SOURCE_UNMATCHED = "B0FieldSource names no fieldmap"
    SIDECAR_ORPHAN = "sidecar has no recording"
    COMPANION_ORPHAN = "companion table has no recording"
    ENTITY_WIDTH_SPLIT = "one entity written at two widths"
    ENTITY_CASE_SPLIT = "one value written in two letter cases"
    TASK_NAME_MISMATCH = "TaskName does not match the task entity"
    INTENDED_FOR_DIFFERS = "IntendedFor differs from what the times imply"
    SESSIONS_UNEVEN = "some subjects use sessions and some do not"
    CITATION_DIVERGED = "CITATION.cff disagrees with dataset_description.json"


@dataclass
class Finding:
    """One disagreement, and what it would take to settle it."""

    kind: Kind
    path: Path                  # the file that is wrong, or would be edited
    detail: str                 # one line, naming the other side
    repair: Optional[str] = None      # what applying it would do, in words
    # The exact before and after, per thing that would change:
    # ``("run-1", "run-01", 18)``. An empty ``before`` is an addition, an
    # empty ``after`` a removal. This is what turns "pad every run" from a
    # promise into something a user can check before agreeing to it.
    changes: tuple[tuple[str, str, int], ...] = ()
    # The files the finding is about, when it is about more than the one in
    # ``path``. Shown so "3 rows name a missing file" can be opened up.
    files: tuple[Path, ...] = ()
    # What the finding is ABOUT, dataset-relative, when that is not ``path``:
    # a missing scans row is about the recording, not the table it goes in.
    about: str = ""
    # A repair that loses nothing (adds a row, drops a row naming nothing):
    # offered ticked. Everything else waits for a person.
    safe: bool = False
    # Why there is no repair, when there is none: said, not assumed.
    no_repair: str = ""
    _apply: Optional[Callable] = dc_field(default=None, repr=False, compare=False)
    # A repair that CANNOT run inside the shared operation because it opens
    # its own: a rename is a whole operation with folder moves and reference
    # rewrites, and nesting one inside another would make the outer undo
    # restore a half-renamed tree. Run after the batch, each its own undo.
    _standalone: Optional[Callable] = dc_field(
        default=None, repr=False, compare=False
    )

    @property
    def fixable(self) -> bool:
        return self._apply is not None or self._standalone is not None


# --------------------------------------------------------------------------
# The dataset, walked once


@dataclass
class _Tree:
    """Every file worth checking, and the directory recordings, from ONE walk.

    The walk the rename uses, plus three exclusions it does not need: a
    ``.bidsignore``d path (the dataset said it is not BIDS), any hidden folder
    at any depth (``.tmp_bidsmgr`` was walked), and the inside of a directory
    recording (a CTF ``.ds``, a BTi folder, an OME-Zarr), whose vendor files
    no rule names and which were reported as orphans.
    """

    root: Path
    files: list[Path]
    recordings: set[Path]

    @classmethod
    def build(cls, root: Path) -> "_Tree":
        from ..metadata.tables import is_scans_entry
        from ..vendor.bidsval.files.bidsignore import load_bidsignore

        root = Path(root)
        ignore = load_bidsignore(root)
        skip = set(_skip_top_level())
        datatypes = set(schema_mod.list_datatypes())
        files: list[Path] = []
        recordings: set[Path] = set()
        for dirpath, dirnames, filenames in os.walk(root):
            base = Path(dirpath)
            rel_base = "" if base == root else base.relative_to(root).as_posix() + "/"
            keep = []
            for name in dirnames:
                rel = rel_base + name
                if name.startswith(".") or (not rel_base and name in skip):
                    continue
                if ignore.match(rel) or ignore.match(rel + "/"):
                    continue
                if base.name in datatypes and is_scans_entry(
                        f"{base.name}/{name}", is_dir=True):
                    recordings.add(base / name)
                    continue
                keep.append(name)
            dirnames[:] = sorted(keep)
            for name in sorted(filenames):
                if name.startswith(".") or ignore.match(rel_base + name):
                    continue
                files.append(base / name)
        return cls(root, files, recordings)


_CURRENT: dict[Path, _Tree] = {}


def _tree(root: Path) -> _Tree:
    """The walk :func:`check` made, or a fresh one for a check run alone."""
    root = Path(root)
    tree = _CURRENT.get(root)
    return tree if tree is not None else _Tree.build(root)


def _rel(root: Path, path: Path) -> str:
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).name


def _parse(name: str) -> tuple[dict[str, str], str, str]:
    """``(entities, suffix, extension)`` of a filename, as the validator reads
    it."""
    from ..vendor.bidsval.context.entities import parse_filename

    return parse_filename(schema_mod.get_schema(), name)


def _count(n: int, word: str, plural: str = "") -> str:
    return f"{n} {word}" if n == 1 else f"{n} {plural or word + 's'}"


# --------------------------------------------------------------------------
# Tables: rows read and written as dictionaries


def _write_table(op, path: Path, header: list[str], rows: list[dict]) -> None:
    from ..metadata.tables import table_text

    op.write_text(path, table_text(header, rows))


def _table_drop(op, path: Path, key: str, names: set[str]) -> None:
    from ..metadata.tables import read_table

    header, rows = read_table(path)
    seen: set[str] = set()
    kept = []
    for row in rows:
        value = row.get(key, "").strip()
        if value in names or (not value and "(empty)" in names) \
                or (value in seen and f"{value} (twice)" in names):
            continue
        seen.add(value)
        kept.append(row)
    _write_table(op, path, header, kept)


def _table_add(op, path: Path, key: str, new_rows: list[dict]) -> None:
    from ..metadata.tables import read_table

    header, rows = read_table(path)
    header = header or [key, "acq_time"]
    for col in (c for r in new_rows for c in r):
        if col not in header:
            header.append(col)
    present = {r.get(key, "") for r in rows}
    rows += [r for r in new_rows if r.get(key, "") not in present]
    _write_table(op, path, header, rows)


def _table_rename_row(op, path: Path, key: str, old: str, new: str) -> None:
    from ..metadata.tables import read_table

    header, rows = read_table(path)
    for row in rows:
        if row.get(key, "") == old:
            row[key] = new
    _write_table(op, path, header, rows)


def _table_fill(op, path: Path, key: str, fills: dict[tuple[str, str], str]) -> None:
    from ..metadata.tables import read_table

    header, rows = read_table(path)
    for row in rows:
        for col in header:
            if not str(row.get(col, "")).strip():
                row[col] = fills.get((row.get(key, ""), col), "n/a")
    _write_table(op, path, header, rows)


def _table_set(op, path: Path, key: str, names: set[str], col: str, value: str) -> None:
    from ..metadata.tables import read_table

    header, rows = read_table(path)
    for row in rows:
        if row.get(key, "") in names:
            row[col] = value
    _write_table(op, path, header, rows)


def _table_create(op, path: Path, text: str) -> None:
    op.write_text(path, text)


# --------------------------------------------------------------------------
# The checks: scans and sessions tables


def _suffix_key(name: str) -> tuple[str, str, str]:
    """What a renamed file keeps: its folder, suffix and extension, without
    regard to case (a row naming ``sub-01_t1w`` is the ``sub-01_T1w`` file)."""
    folder, _, base = name.rpartition("/")
    _ents, suffix, ext = _parse(base)
    return folder.lower(), suffix.lower(), ext.lower()


def _pairs(gone: list[str], added: list[str]) -> list[tuple[str, str]]:
    """A row naming a missing file, and a recording with no row, that are
    one file renamed: the only such pair for their folder, suffix and
    extension."""
    by_key_gone: dict[tuple, list[str]] = {}
    by_key_added: dict[tuple, list[str]] = {}
    for name in gone:
        by_key_gone.setdefault(_suffix_key(name), []).append(name)
    for name in added:
        by_key_added.setdefault(_suffix_key(name), []).append(name)
    return [(olds[0], by_key_added[k][0]) for k, olds in by_key_gone.items()
            if len(olds) == 1 and len(by_key_added.get(k, [])) == 1]


def _blank_fills(edit) -> dict[tuple[str, str], str]:
    """What the plan writes into each empty cell of an existing table."""
    from ..metadata.tables import read_table

    _header, rows = read_table(edit.path)
    planned = {r.get(edit.key, ""): r for r in edit.rows}
    fills = {}
    for row in rows:
        name = row.get(edit.key, "")
        for col, value in row.items():
            if not str(value).strip() and name in planned:
                fills[(name, col)] = planned[name].get(col, "n/a") or "n/a"
    return fills


def _row_text(name: str, when: str, verb: str) -> str:
    """A row a new table will have, as the detail lists it."""
    return f"{name}, {verb} {when}" if when and when != "n/a" else f"{name}, time not known"


def _scans_findings(root: Path, edit, others_exist: bool, expect: bool) -> list[Finding]:
    """``expect``: a missing table is a finding (Settings > Convert, "Write
    scans and sessions tables", or another session already has one)."""
    out: list[Finding] = []
    table = edit.path
    table_rel = _rel(root, table)
    home = table.parent
    if edit.created:
        if expect and edit.rows:
            out.append(Finding(
                kind=Kind.SCANS_TABLE_MISSING, path=table, about=_rel(root, home),
                detail=(f"{_rel(root, home)} has {_count(len(edit.rows), 'recording')} "
                        "and no scans table, while other sessions have one" if others_exist
                        else f"{_rel(root, home)} has "
                        f"{_count(len(edit.rows), 'recording')} and no scans table. The "
                        "standard makes it optional; it is where a dataset says when each "
                        "recording was acquired"),
                repair=f"create {table.name} with {_count(len(edit.rows), 'row')}",
                changes=tuple(("", _row_text(r["filename"], r.get("acq_time", "n/a"),
                                             "acquired"), 1) for r in edit.rows),
                files=tuple(home / r["filename"] for r in edit.rows),
                safe=True,
                _apply=lambda op, t=table, text=edit.text(): _table_create(op, t, text),
            ))
        return out

    gone = [name for name, why in edit.dropped if why == "the file is not there"]
    extra = [(name, why) for name, why in edit.dropped if why != "the file is not there"]
    planned = {r["filename"]: r for r in edit.rows}
    paired = _pairs(gone, edit.added)
    for old, new in paired:
        out.append(Finding(
            kind=Kind.SCANS_ROW_MOVED, path=table, about=_rel(root, home / new),
            detail=(f"{table.name} names {old}, which is not there; {new} is, "
                    "with no row. The same recording, renamed outside the application"),
            repair="point the row at the new name, keeping its time and other columns",
            changes=((old, new, 1),), files=(home / new,), safe=True,
            _apply=lambda op, t=table, o=old, n=new: _table_rename_row(op, t, "filename", o, n),
        ))
    moved_old = {o for o, _n in paired}
    moved_new = {n for _o, n in paired}

    orphans = [g for g in gone if g not in moved_old]
    if orphans:
        out.append(Finding(
            kind=Kind.SCANS_ROW_ORPHAN, path=table, about=table_rel,
            detail=(f"{_count(len(orphans), 'row')} of {table.name} "
                    f"{'names a file' if len(orphans) == 1 else 'name files'} not there"),
            repair=f"drop {_count(len(orphans), 'row')}",
            changes=tuple((name, "", 1) for name in orphans), safe=True,
            _apply=lambda op, t=table, n=set(orphans): _table_drop(op, t, "filename", n),
        ))
    if extra:
        names = {name if why != "listed twice" else f"{name} (twice)" for name, why in extra}
        out.append(Finding(
            kind=Kind.SCANS_ROW_EXTRA, path=table, about=table_rel,
            detail="; ".join(f"{name}: {why}" for name, why in extra[:4])
                   + ("; ..." if len(extra) > 4 else ""),
            repair=f"drop {_count(len(extra), 'row')}, so each recording has one",
            changes=tuple((name, "", 1) for name, _why in extra), safe=True,
            _apply=lambda op, t=table, n=names: _table_drop(op, t, "filename", n),
        ))
    for name in edit.added:
        if name in moved_new:
            continue
        row = planned[name]
        when = row.get("acq_time", "n/a")
        out.append(Finding(
            kind=Kind.SCANS_ROW_MISSING, path=table, about=_rel(root, home / name),
            detail=f"{name} is a recording and {table.name} has no row for it",
            repair="add the row" + (f", acquired {when}" if when != "n/a" else ""),
            changes=(("", name, 1),), files=(home / name,), safe=True,
            _apply=lambda op, t=table, r=dict(row): _table_add(op, t, "filename", [r]),
        ))
    if edit.filled:
        fills = _blank_fills(edit)
        out.append(Finding(
            kind=Kind.TABLE_BLANK_CELLS, path=table, about=table_rel,
            detail=(f"{table.name} has {_count(edit.filled, 'empty cell')}; the standard "
                    "writes n/a for a value that is not known"),
            repair="fill them: the acquisition time where the sidecar has it, else n/a",
            changes=tuple((f"{n} ({c})", v, 1) for (n, c), v in list(fills.items())[:50]),
            safe=True,
            _apply=lambda op, t=table, f=fills: _table_fill(op, t, "filename", f),
        ))

    return out


def _times_findings(root: Path, table: Path) -> list[Finding]:
    """The parts of one recording (echo, part, split, chunk) MUST share one
    ``acq_time``. Every scans table is read, changed by the plan or not."""
    from ..metadata.tables import _recording_key, is_datetime, read_table

    out: list[Finding] = []
    home = table.parent
    _header, rows = read_table(table)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_recording_key(row.get("filename", "")), []).append(row)
    for members in groups.values():
        times = sorted({r.get("acq_time", "") for r in members
                        if is_datetime(r.get("acq_time", ""))})
        if len(members) < 2 or len(times) < 2:
            continue
        names = {r["filename"] for r in members}
        out.append(Finding(
            kind=Kind.SCANS_TIMES_DIFFER, path=table,
            about=_rel(root, home / sorted(names)[0]),
            detail=(f"{_count(len(members), 'file')} of one recording have "
                    f"{len(times)} acquisition times; the standard says they MUST be "
                    "identical"),
            repair=f"use the earliest, {times[0]}, for all of them",
            changes=tuple((f"{r['filename']}: {r.get('acq_time', '')}", times[0], 1)
                          for r in members if r.get("acq_time") != times[0]),
            files=tuple(home / n for n in sorted(names)),
            _apply=lambda op, t=table, n=names, v=times[0]: _table_set(
                op, t, "filename", n, "acq_time", v),
        ))
    return out


def _sessions_findings(root: Path, edit, others_exist: bool, expect: bool) -> list[Finding]:
    out: list[Finding] = []
    table = edit.path
    subject = table.parent
    if edit.created:
        if expect and edit.rows:
            out.append(Finding(
                kind=Kind.SESSIONS_TABLE_MISSING, path=table, about=subject.name,
                detail=(f"{subject.name} has {_count(len(edit.rows), 'session')} and no "
                        "sessions table, while other subjects have one" if others_exist
                        else f"{subject.name} has {_count(len(edit.rows), 'session')} and "
                        "no sessions table. The standard makes it optional; it lists the "
                        "sessions, with when each began"),
                repair=f"create {table.name} with {_count(len(edit.rows), 'row')}",
                changes=tuple(("", _row_text(r["session_id"], r.get("acq_time", "n/a"),
                                             "began"), 1) for r in edit.rows),
                safe=True,
                _apply=lambda op, t=table, text=edit.text(): _table_create(op, t, text),
            ))
        return out
    gone = [name for name, why in edit.dropped]
    if gone:
        names = {n if why != "listed twice" else f"{n} (twice)" for n, why in edit.dropped}
        out.append(Finding(
            kind=Kind.SESSION_ROW_ORPHAN, path=table, about=_rel(root, table),
            detail="; ".join(f"{n}: {why}" for n, why in edit.dropped[:4]),
            repair=f"drop {_count(len(gone), 'row')}",
            changes=tuple((n, "", 1) for n in gone), safe=True,
            _apply=lambda op, t=table, n=names: _table_drop(op, t, "session_id", n),
        ))
    planned = {r["session_id"]: r for r in edit.rows}
    for sid in edit.added:
        row = planned[sid]
        out.append(Finding(
            kind=Kind.SESSION_ROW_MISSING, path=table, about=f"{subject.name}/{sid}",
            detail=f"{sid} is a session of {subject.name} and {table.name} has no row for it",
            repair="add the row" + (f", acquired {row['acq_time']}"
                                    if row.get("acq_time", "n/a") != "n/a" else ""),
            changes=(("", sid, 1),), safe=True,
            _apply=lambda op, t=table, r=dict(row): _table_add(op, t, "session_id", [r]),
        ))
    if edit.filled:
        fills = _blank_fills(edit)
        out.append(Finding(
            kind=Kind.TABLE_BLANK_CELLS, path=table, about=_rel(root, table),
            detail=f"{table.name} has {_count(edit.filled, 'empty cell')}",
            repair="fill them: the session's earliest scan for acq_time, else n/a",
            changes=tuple((f"{n} ({c})", v, 1) for (n, c), v in fills.items()), safe=True,
            _apply=lambda op, t=table, f=fills: _table_fill(op, t, "session_id", f),
        ))
    return out


def check_scans_tables(root: Path, *, expect: bool = True) -> list[Finding]:
    """Scans and sessions tables against what is on disk, by the one rule the
    metadata step writes them with (``metadata.tables``): what Tools, Scans
    and sessions tables would change is what this reports.

    ``expect`` is Settings > Convert, "Write scans and sessions tables" (on
    by default): every table that is not there is a finding. Off, a dataset
    with none is coherent (the standard makes them optional), but one where
    some sessions have a table and some do not is still not. Tables that are
    there are checked either way.
    """
    from ..metadata.tables import plan_tables

    root = Path(root)
    tree = _tree(root)
    has_scans = any(p.name.endswith("_scans.tsv") for p in tree.files)
    has_sessions = any(p.name.endswith("_sessions.tsv") for p in tree.files)
    if not (expect or has_scans or has_sessions):
        return []
    out: list[Finding] = []
    for edit in plan_tables(root):
        if edit.kind == "scans":
            out.extend(_scans_findings(root, edit, has_scans, expect or has_scans))
        else:
            out.extend(_sessions_findings(root, edit, has_sessions, expect or has_sessions))
    for table in sorted(p for p in tree.files if p.name.endswith("_scans.tsv")):
        out.extend(_times_findings(root, table))
    return out


# --------------------------------------------------------------------------
# participants.tsv


def _phenotype_ids(root: Path) -> set[str]:
    """Participants the phenotype tables describe. A participant with
    phenotype data and no imaging is legitimate, so their row is not an
    orphan."""
    from ..metadata.tables import read_table

    out: set[str] = set()
    for table in sorted((root / "phenotype").glob("*.tsv")):
        _header, rows = read_table(table)
        out.update(r.get("participant_id", "").strip() for r in rows)
    out.discard("")
    return out


def check_participants(root: Path) -> list[Finding]:
    """Rows for subjects that are gone, subjects with no row, no table."""
    from ..metadata.tables import read_table

    root = Path(root)
    table = root / "participants.tsv"
    on_disk = {p.name for p in root.glob("sub-*") if p.is_dir()}
    if not table.is_file():
        if not on_disk:
            return []
        rows = [{"participant_id": s} for s in sorted(on_disk)]
        return [Finding(
            kind=Kind.PARTICIPANTS_TABLE_MISSING, path=table, about="participants.tsv",
            detail=(f"{_count(len(on_disk), 'subject folder')} and no participants.tsv, "
                    "which the standard RECOMMENDS to list them"),
            repair=f"create participants.tsv listing the {_count(len(on_disk), 'subject')}",
            changes=tuple(("", s, 1) for s in sorted(on_disk)), safe=True,
            _apply=lambda op, r=rows: _write_table(op, table, ["participant_id"], r),
        )]

    header, rows = read_table(table)
    if "participant_id" not in header:
        return []
    listed = {r.get("participant_id", "").strip() for r in rows}
    out: list[Finding] = []
    orphans = sorted(listed - on_disk - _phenotype_ids(root) - {""})
    if orphans:
        out.append(Finding(
            kind=Kind.PARTICIPANT_ORPHAN, path=table, about="participants.tsv",
            detail=(f"{_count(len(orphans), 'row')} name a subject folder that is not "
                    f"there and no phenotype table describes: {', '.join(orphans[:3])}"
                    f"{', ...' if len(orphans) > 3 else ''}. Dropping a row loses what "
                    "it says about that person (age, sex, group)"),
            repair=f"drop {_count(len(orphans), 'row')}",
            changes=tuple((o, "", 1) for o in orphans),
            _apply=lambda op, n=set(orphans): _table_drop(op, table, "participant_id", n),
        ))
    for missing in sorted(on_disk - listed):
        out.append(Finding(
            kind=Kind.PARTICIPANT_MISSING, path=table, about=missing,
            detail=f"{missing} has a folder and no row in participants.tsv",
            repair="add the row (n/a in every other column)",
            changes=(("", missing, 1),), files=(root / missing,), safe=True,
            _apply=lambda op, m=missing: _table_add(
                op, table, "participant_id", [{"participant_id": m}]),
        ))
    return out


# --------------------------------------------------------------------------
# Links


def _subject_of(root: Path, path: Path) -> str:
    rel = _rel(root, path)
    first = rel.split("/", 1)[0]
    return first if first.startswith("sub-") else ""


def check_links(root: Path) -> list[Finding]:
    """Pointer fields aimed at nothing or into another subject, and
    ``B0FieldSource`` values no fieldmap identifies."""
    root = Path(root)
    tree = _tree(root)
    out: list[Finding] = []
    identifiers: dict[str, set[str]] = {}
    sources: list[tuple[Path, str]] = []
    for path in tree.files:
        if not path.name.endswith(".json"):
            continue
        for link in linkage.read_links(root, path):
            for target in link.broken:
                out.append(Finding(
                    kind=Kind.LINK_BROKEN, path=link.source, about=_rel(root, link.source),
                    detail=f"{link.field} points at {target}, which is not there",
                    repair="remove that entry",
                    changes=((f"{link.field}: {target}", "", 1),),
                    _apply=lambda op, s=link.source, f=link.field, t=target: _drop_link(
                        op, s, f, t),
                ))
            if link.field != "IntendedFor":
                continue
            own = _subject_of(root, link.source)
            for target, found in zip(link.targets, link.resolved):
                other = _subject_of(root, found) if found is not None else ""
                if own and other and other != own:
                    out.append(Finding(
                        kind=Kind.LINK_OTHER_SUBJECT, path=link.source,
                        about=_rel(root, link.source),
                        detail=(f"a fieldmap of {own} names {_rel(root, found)}, a run of "
                                f"{other}; a fieldmap corrects the runs of its own subject"),
                        repair="remove that entry",
                        changes=((f"IntendedFor: {target}", "", 1),),
                        _apply=lambda op, s=link.source, t=target: _drop_link(
                            op, s, "IntendedFor", t),
                    ))
        data = linkage._load(path)
        subject = _subject_of(root, path)
        for key, bucket in (("B0FieldIdentifier", identifiers), ("B0FieldSource", None)):
            value = data.get(key)
            values = value if isinstance(value, list) else [value] if value else []
            for v in values:
                if not isinstance(v, str) or v in ("", "TODO", "n/a"):
                    continue
                if bucket is not None:
                    bucket.setdefault(subject, set()).add(v)
                else:
                    sources.append((path, v))
    for path, value in sources:
        if value in identifiers.get(_subject_of(root, path), set()):
            continue
        out.append(Finding(
            kind=Kind.B0_SOURCE_UNMATCHED, path=path, about=_rel(root, path),
            detail=(f'B0FieldSource is "{value}", and no fieldmap of this subject has '
                    "that B0FieldIdentifier, so no correction is applied to it"),
            no_repair=("Which fieldmap was meant is not something the files say. Set "
                       "it in the sidecar, or with Tools, References."),
        ))
    return out


def _drop_link(op, source: Path, field: str, target: str) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    raw = data.get(field)
    values = raw if isinstance(raw, list) else [raw]
    kept = [v for v in values if str(v) != target]
    if kept:
        data[field] = kept if isinstance(raw, list) else kept[0]
    else:
        # An empty list claims "points at nothing", which is a different and
        # wronger statement than not saying.
        data.pop(field, None)
    op.write_json(source, data)


# --------------------------------------------------------------------------
# Sidecars and companion tables with no recording

#: Companion tables that belong to one recording (or, inherited, to several).
_COMPANION_SUFFIXES = frozenset({"events", "channels", "physio", "physioevents", "stim"})


def _rule_extensions(relpath: str) -> Optional[list[str]]:
    from ..vendor.bidsval.rules.filenames import match_file_rule

    schema = schema_mod.get_schema()
    rule = match_file_rule(schema, relpath)
    if rule is None:
        return None
    node = schema
    for part in rule.split("."):
        node = node[part]
    return [str(e) for e in node.get("extensions", [])]


def _has_partner(entities: dict, suffix: Optional[str], names: Iterable[str],
                 ignore: tuple[str, ...] = (), recorded: frozenset = frozenset()) -> bool:
    """A data file in the folder with ``suffix`` (any recording, when
    ``None``) whose entities include all of ``entities``: by inheritance, a
    file with fewer entities describes every recording that has more.
    ``recorded`` are the folder's recordings, by the scans tables' rule: eye
    tracking alone in ``beh/`` is one, and its ``physioevents`` its partner."""
    want = {k: v for k, v in entities.items() if k not in ignore}
    for name in names:
        if name.endswith(".json"):
            continue
        ents, sfx, _ext = _parse(name)
        if suffix is not None and sfx != suffix:
            continue
        if suffix is None and sfx in _COMPANION_SUFFIXES and name not in recorded:
            continue
        if all(ents.get(k) == v for k, v in want.items()):
            return True
    return False


def check_orphan_sidecars(root: Path) -> list[Finding]:
    """A sidecar or a companion table whose recording is gone.

    Read with the standard's inheritance: ``task-x_bold.json`` describes every
    run of task x in its folder, and a ``part-mag`` and ``part-phase`` pair
    can share one sidecar without the part. A JSON file the standard defines
    on its own (``coordsystem.json``) has no recording to lose. Only
    datatype folders: a sidecar higher up describes everything below it.
    """
    from ..metadata.tables import list_recordings

    root = Path(root)
    tree = _tree(root)
    datatypes = set(schema_mod.list_datatypes())
    by_folder: dict[Path, list[str]] = {}
    for path in tree.files:
        by_folder.setdefault(path.parent, []).append(path.name)
    for rec in tree.recordings:
        by_folder.setdefault(rec.parent, []).append(rec.name)
    recorded: dict[Path, frozenset] = {}

    def recordings_in(folder: Path) -> frozenset:
        if folder not in recorded:
            dirs = {r.name for r in tree.recordings if r.parent == folder}
            recorded[folder] = frozenset(list_recordings(
                folder.name, ((n, n in dirs) for n in by_folder[folder])))
        return recorded[folder]

    out: list[Finding] = []
    for path in tree.files:
        folder = path.parent
        if folder.name not in datatypes:
            continue
        name = path.name
        entities, suffix, ext = _parse(name)
        tail = f"{folder.name}/{name}"
        if ext == ".json":
            extensions = _rule_extensions(tail)
            if not extensions or all(e == ".json" for e in extensions):
                continue                  # not BIDS, or a JSON file of its own
            if _has_partner(entities, suffix, by_folder[folder]):
                continue
            out.append(Finding(
                kind=Kind.SIDECAR_ORPHAN, path=path, about=_rel(root, path),
                detail=f"{name} describes a {suffix} recording that is not in {folder.name}/",
                repair="delete the sidecar",
                changes=((name, "", 1),),
                _apply=lambda op, p=path: op.delete(p),
            ))
        elif suffix in _COMPANION_SUFFIXES and ext in (".tsv", ".tsv.gz"):
            here = recordings_in(folder)
            if name in here:
                continue                  # behaviour or eye tracking alone: the data
            if _has_partner(entities, None, by_folder[folder], ignore=("recording",),
                            recorded=here):
                continue
            out.append(Finding(
                kind=Kind.COMPANION_ORPHAN, path=path, about=_rel(root, path),
                detail=f"{name} belongs to a recording that is not in {folder.name}/",
                repair="delete it",
                changes=((name, "", 1),),
                _apply=lambda op, p=path: op.delete(p),
            ))
    return out


# --------------------------------------------------------------------------
# CITATION.cff


def _real(value) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_real(v) for v in value)
    if isinstance(value, dict):
        return any(_real(v) for v in value.values())
    return bool(str(value or "").strip()) and str(value).strip() not in ("TODO", "n/a")


def check_citation(root: Path) -> list[Finding]:
    """``CITATION.cff`` and ``dataset_description.json`` out of step.

    Only what BOTH files actually state is compared, and only real values: a
    description with no authors used to produce a citation of TODO authors
    and offered to write it over the real ones.
    """
    from . import cff

    root = Path(root)
    citation = root / "CITATION.cff"
    description = root / "dataset_description.json"
    if not (citation.is_file() and description.is_file()):
        return []
    try:
        have = cff.load(citation) or {}
        desc = json.loads(description.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    expected = cff.from_dataset_description(desc)
    stated = {"title": desc.get("Name"), "authors": desc.get("Authors"),
              "license": desc.get("License")}
    diverged = [
        key for key in ("title", "authors", "license")
        if _real(stated[key]) and key in have and _real(have.get(key))
        and have.get(key) != expected.get(key)
    ]
    if not diverged:
        return []
    names = {"title": "Name", "authors": "Authors", "license": "License"}
    update = {k: expected[k] for k in diverged}
    return [Finding(
        kind=Kind.CITATION_DIVERGED, path=citation, about="CITATION.cff",
        detail=(f"{', '.join(names[k] for k in diverged)} "
                f"{'differs' if len(diverged) == 1 else 'differ'} between the two files"),
        repair=f"write the description's {', '.join(names[k] for k in diverged)} "
               "into CITATION.cff",
        changes=tuple((f"{k}: {json.dumps(have.get(k), default=str)[:60]}",
                       json.dumps(expected.get(k), default=str)[:60], 1) for k in diverged),
        _apply=lambda op, c=citation, e={**have, **update}: op.write_text(c, cff.dumps(e)),
    )]


# --------------------------------------------------------------------------
# Names


def check_entity_widths(root: Path) -> list[Finding]:
    """An index entity written padded and unpadded in one dataset.

    ``run-01`` beside ``run-2`` is legal BIDS, which is why no validator
    mentions it, and it is still what makes a dataset sort wrongly and a
    glob miss half its files. Numbering without padding (``run-1`` to
    ``run-10``) is one consistent choice and not reported.
    """
    from . import values as ev

    out: list[Finding] = []
    for split in ev.inconsistent_widths(root):
        width = split.suggested
        detail = (
            f"{split.entity} is written at {len(split.widths)} widths "
            f"({', '.join(split.examples)}) across {_count(split.files, 'file')}"
        )
        try:
            plan = ev.plan_padding(root, split.entity, width)
        except Exception as exc:  # noqa: BLE001 - say WHY it cannot be padded
            out.append(Finding(
                kind=Kind.ENTITY_WIDTH_SPLIT, path=Path(root), about=f"{split.entity}-",
                detail=detail, no_repair=str(exc),
            ))
            continue
        if not plan:
            continue
        out.append(Finding(
            kind=Kind.ENTITY_WIDTH_SPLIT, path=Path(root), about=f"{split.entity}-",
            detail=detail,
            repair=f"pad every {split.entity} value to {width} digits (a rename)",
            changes=tuple(
                (f"{split.entity}-{r.old}", f"{split.entity}-{r.new}", r.files)
                for r in plan
            ),
            _standalone=lambda e=split.entity, w=width: _pad_entity(root, e, w),
        ))
    return out


def _pad_entity(root: Path, entity: str, width: int) -> None:
    """Repad every value of ``entity``, one rename per value.

    Each ``apply_rename`` opens its own operation, which is why this is a
    STANDALONE repair rather than one that runs inside the batch: nesting
    an operation inside another would make the outer undo restore a
    half-repadded tree.
    """
    from . import values as ev
    from .rename import apply_rename, plan_rename

    for repad in ev.plan_padding(root, entity, width):
        apply_rename(root, plan_rename(root, entity, repad.old, repad.new))


def check_entity_case(root: Path) -> list[Finding]:
    """Two spellings of one value that differ only in case.

    ``task-Rest`` and ``task-rest`` are two different tasks to BIDS and to
    every pipeline that reads it, and are almost never meant to be. One walk
    for every entity (it was one per entity, 33 of them).
    """
    root = Path(root)
    seen: dict[tuple[str, str], dict[str, list[Path]]] = {}
    for path in _tree(root).files:
        entities, _suffix, _ext = _parse(path.name)
        for key, value in entities.items():
            if value:
                seen.setdefault((key, value.lower()), {}).setdefault(value, []).append(path)
    out: list[Finding] = []
    for (entity, _lowered), spellings in sorted(seen.items()):
        if len(spellings) < 2:
            continue
        files = tuple(p for group in spellings.values() for p in group)
        out.append(Finding(
            kind=Kind.ENTITY_CASE_SPLIT, path=root, about=f"{entity}-",
            detail=(f"{entity} is spelled "
                    f"{' and '.join(sorted(f'{entity}-{s}' for s in spellings))}"
                    ", which BIDS reads as different values"),
            changes=tuple((f"{entity}-{s}", "", len(g)) for s, g in sorted(spellings.items())),
            files=files,
            no_repair=("Which spelling is right is yours to say. Rename the other with "
                       "Tools, Rename entity."),
        ))
    return out


def check_task_names(root: Path) -> list[Finding]:
    """``TaskName`` in a sidecar that does not derive to the task entity.

    The standard defines the relationship: the label is the name with
    everything but letters and digits removed. Compared without regard to
    case: ``"Rest"`` on ``task-rest`` is how most datasets write it.
    """
    from .rename import _derived_label, entity_value

    root = Path(root)
    out: list[Finding] = []
    for path in _tree(root).files:
        if not path.name.endswith(".json"):
            continue
        label = entity_value(path.name, "task")
        if not label:
            continue
        name = _load_json(path).get("TaskName")
        if not isinstance(name, str) or not name:
            continue
        if _derived_label(name).lower() == label.lower():
            continue
        out.append(Finding(
            kind=Kind.TASK_NAME_MISMATCH, path=path, about=_rel(root, path),
            detail=(f'TaskName is "{name}", which does not derive to the '
                    f"task-{label} in the filename"),
            repair=f'set TaskName to "{label}"',
            changes=((f'TaskName: "{name}"', f'TaskName: "{label}"', 1),),
            _apply=lambda op, p=path, v=label: _set_field(op, p, "TaskName", v),
        ))
    return out


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _set_field(op, path: Path, field: str, value) -> None:
    data = _load_json(path)
    if value in (None, [], ""):
        data.pop(field, None)
    else:
        data[field] = value
    op.write_json(path, data)


_PHASE_PART = re.compile(r"_part-phase_")


def check_intended_for(root: Path) -> list[Finding]:
    """``IntendedFor`` that differs from what the acquisition times imply.

    Advice, never ticked: a protocol may want one fieldmap applied to
    everything, and a user may have set it deliberately. Left out entirely
    where the advice would be worth nothing: when the times could not be read
    (the suggestion falls back to every fieldmap for every run), on the phase
    part of a fieldmap (it corrects nothing on its own), and on a fieldmap
    linked with ``B0FieldIdentifier``, which is the link pipelines follow.
    Entries are compared by the file they name, so ``bids::`` and the older
    relative spelling of one file are the same entry.
    """
    from ..fixups.intended_for import FALLBACK_NOTE, suggest_intended_for

    root = Path(root)
    out: list[Finding] = []
    for subject in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        sessions = [p for p in subject.glob("ses-*") if p.is_dir()]
        for scope in (sessions or [subject]):
            label = subject.name[len("sub-"):]
            ses = scope.name[len("ses-"):] if scope is not subject else None
            pairs, why = suggest_intended_for(scope, label, ses)
            if FALLBACK_NOTE in (why or ""):
                continue
            for members, uris in pairs:
                for sidecar in members:
                    data = _load_json(sidecar)
                    if _PHASE_PART.search(sidecar.name) or data.get("B0FieldIdentifier"):
                        continue
                    raw = data.get("IntendedFor") or []
                    have = [str(v) for v in (raw if isinstance(raw, list) else [raw])]
                    have_files = {linkage.resolve(root, v): v for v in have}
                    want_files = {linkage.resolve(root, u): u for u in uris}
                    if set(have_files) == set(want_files):
                        continue
                    extra = [v for f, v in have_files.items() if f not in want_files]
                    missing = [u for f, u in want_files.items() if f not in have_files]
                    out.append(Finding(
                        kind=Kind.INTENDED_FOR_DIFFERS, path=sidecar,
                        about=_rel(root, sidecar),
                        detail=(f"{_count(len(extra), 'entry', 'entries')} the times do "
                                f"not imply, {len(missing)} the times imply and it does "
                                f"not have ({why})"),
                        repair="use what the times imply",
                        changes=tuple((u.rsplit("/", 1)[-1], "", 1) for u in extra)
                        + tuple(("", u.rsplit("/", 1)[-1], 1) for u in missing),
                        _apply=lambda op, p=sidecar, u=list(uris): _set_field(
                            op, p, "IntendedFor", u),
                    ))
    return out


def check_sessions_even(root: Path) -> list[Finding]:
    """Some subjects in session folders and some not.

    Legal, and almost always a conversion that treated two subjects
    differently rather than a study that genuinely measured one of them
    once. ``sub-emptyroom`` is the MEG empty-room recording and is not a
    participant, so it is not counted.
    """
    with_ses, without = [], []
    for subject in sorted(p for p in Path(root).glob("sub-*") if p.is_dir()):
        if subject.name == "sub-emptyroom":
            continue
        if any(p.is_dir() for p in subject.glob("ses-*")):
            with_ses.append(subject.name)
        else:
            without.append(subject.name)
    if not (with_ses and without):
        return []
    return [Finding(
        kind=Kind.SESSIONS_UNEVEN, path=Path(root), about=", ".join(without[:3]),
        detail=(f"{_count(len(with_ses), 'subject')} use session folders and "
                f"{len(without)} do not ({', '.join(without[:3])}"
                f"{', ...' if len(without) > 3 else ''})"),
        files=tuple(Path(root) / s for s in without),
        no_repair=("Putting a subject into a session needs a label only you know. "
                   "Tools, Sessions does it."),
    )]


CHECKS: tuple[Callable[[Path], list[Finding]], ...] = (
    check_scans_tables,
    check_participants,
    check_links,
    check_orphan_sidecars,
    check_entity_widths,
    check_entity_case,
    check_task_names,
    check_intended_for,
    check_sessions_even,
    check_citation,
)


def check(root: Path, *, tables: bool = True) -> list[Finding]:
    """Run every check. Read-only. Ordered by :class:`Kind`.

    ``tables``: a scans or sessions table that is not there is a finding
    (Settings > Convert, "Write scans and sessions tables").

    A check that raises is reported and does not stop the others: a coherence
    pass that gives up on the first malformed table is no use on exactly the
    dataset it was written for. The dataset is walked once for all of them.
    """
    root = Path(root)
    _CURRENT[root] = _Tree.build(root)
    out: list[Finding] = []
    try:
        for fn in CHECKS:
            try:
                out.extend(fn(root, expect=tables) if fn is check_scans_tables
                           else fn(root))
            except Exception as exc:  # noqa: BLE001 - one bad check is not all of them
                log.warning("coherence check %s failed: %s", fn.__name__, exc)
    finally:
        _CURRENT.pop(root, None)
    order = {kind: i for i, kind in enumerate(Kind)}
    return sorted(out, key=lambda f: order[f.kind])


def apply(root: Path, findings: Iterable[Finding], *,
          label: Optional[str] = None) -> int:
    """Apply the chosen repairs. Returns how many ran.

    One operation for all of them, so the pass is one entry in the history;
    renames (standalone) after it, each its own.
    """
    chosen = [f for f in findings if f.fixable]
    if not chosen:
        return 0

    batch = [f for f in chosen if f._apply is not None]
    standalone = [f for f in chosen if f._apply is None and f._standalone]

    done = 0
    if batch:
        with begin_operation(
            Path(root), label or f"Repair {_count(len(batch), 'inconsistency', 'inconsistencies')}"
        ) as op:
            for finding in batch:
                finding._apply(op)
                done += 1
    for finding in standalone:
        finding._standalone()
        done += 1
    return done


__all__ = ["CHECKS", "Finding", "Kind", "apply", "check"]

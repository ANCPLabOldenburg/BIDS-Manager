"""The ``*_scans.tsv`` and ``*_sessions.tsv`` tables, from what is on disk.

ONE rule decides which files a scans table lists, and it is the standard's:
a file gets a row when it fits one of the schema's DATATYPE file rules
(``rules.files.raw.anat.*``, ``beh.*``, ``motion.*``, ...), through the
validator's own matcher (``vendor.bidsval.rules.filenames.match_file_rule``).
So every datatype is covered, including the ones BIDS Manager does not
convert (fNIRS, motion, microscopy, EMG), because a user curating a dataset
converted elsewhere needs its tables too. Before, the engine kept its own
lists of suffixes and extensions, and ``beh`` and ``motion`` recordings,
whose data are ``.tsv``, never got a row.

Not listed, because they are not a recording of their own: sidecars
(``.json``), the side files of a multi-file recording (``.bval``/``.bvec``
beside the image, BrainVision's ``.eeg``/``.vmrk``, EEGLAB's ``.fdt``),
companion tables and timeseries (events, channels, physio, stim, photos), and
a few datatype files that describe a recording rather than being one (MEG
calibration, crosstalk, headshape and markers; ASL context and labelling
image; PET blood). In ``beh/`` an ``_events.tsv`` with no ``_beh`` file of its
own IS the recording: a behavioural experiment with no other data.

Existing tables are kept: their rows, their extra columns, their values.
Rows are added for recordings not listed, blank cells become ``n/a``, and a
row goes only when its file is gone or when it lists the side part of a
recording that has its own row (one row per recording).

``acq_time`` is a full date and time or ``n/a``, never a time alone: the
sidecar's ``AcquisitionDateTime``, else what the conversion knew (``times``),
else ``n/a``. The parts of one recording (echo, part, split, chunk) share
one value, as the standard requires. A session's ``acq_time`` is its earliest
scan.

Qt-free. Writes through an editor operation when given one, so the Editor's
tool is one undoable step.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional

NA = "n/a"

#: Datatype rules whose files describe a recording rather than being one.
SIDE_RULES = frozenset({
    "meg.calibration", "meg.crosstalk", "meg.headshape", "meg.markers",
    "perf.aslcontext", "perf.asllabeling", "pet.blood",
})

#: Extensions that are a part of a recording listed under another extension:
#: the sidecar, the gradient tables beside a diffusion image, BrainVision's
#: binary and markers beside its header, EEGLAB's data beside its ``.set``.
PART_EXTENSIONS = frozenset({".json", ".bval", ".bvec", ".eeg", ".vmrk", ".fdt"})

#: Entities that split ONE recording into files that share its time. The
#: schema defines ``recording`` as exactly that: one acquisition written to
#: several files (two eyes of one eye-tracking session, physio channels at
#: different rates).
SAME_RECORDING = ("echo", "part", "split", "chunk", "recording")

_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$")


# ---------------------------------------------------------------------------
# Which files get a row
# ---------------------------------------------------------------------------


def _schema():
    from .. import schema as schema_mod

    return schema_mod.get_schema()


@lru_cache(maxsize=1)
def _datatypes() -> frozenset:
    from .. import schema as schema_mod

    return frozenset(schema_mod.list_datatypes())


def _register() -> None:
    from ..schema.loader import register_cache

    register_cache(_datatypes)
    register_cache(_rule)


@lru_cache(maxsize=65536)
def _rule(relpath: str, is_dir: bool) -> Optional[str]:
    from ..vendor.bidsval.rules.filenames import match_file_rule

    return match_file_rule(_schema(), relpath, is_dir=is_dir)


def _rule_name(relpath: str, is_dir: bool) -> Optional[str]:
    """``"anat.nonparametric"`` for a raw file rule, else ``None``."""
    rule = _rule(relpath, is_dir)
    if not rule or not rule.startswith("rules.files.raw."):
        return None
    return rule[len("rules.files.raw."):]


def _extension(name: str) -> str:
    from ..vendor.bidsval.schema import introspect

    try:
        return introspect.split_extension(_schema(), name)[1]
    except Exception:  # noqa: BLE001 - an odd name has no extension worth reading
        return Path(name).suffix


def _tail(relpath: str) -> str:
    """The datatype folder and the name: all the rule needs."""
    parts = relpath.replace("\\", "/").strip("/").split("/")
    return "/".join(parts[-2:])


def is_scans_entry(relpath: str, *, is_dir: bool = False) -> bool:
    """Does the file at ``relpath`` get a row of its own in ``*_scans.tsv``?

    ``relpath`` from the dataset or session root, POSIX or not; only the
    datatype folder and the name are read, so a planned path works as well as
    one on disk. ``is_dir`` for a directory recording (CTF ``.ds``, MEF
    ``.mefd``, OME-Zarr, a BTi folder).

    A ``beh`` events file is listed by :func:`session_recordings`, which can
    see whether a ``_beh`` file shares its name; alone, this says no.
    """
    tail = _tail(relpath)
    name = tail.rsplit("/", 1)[-1]
    if not name or name.startswith("."):
        return False
    rule = _rule_name(tail, is_dir)
    if rule is None:
        return False
    group = rule.split(".", 1)[0]
    if group not in _datatypes() or rule in SIDE_RULES:
        return False
    return is_dir or _extension(name) not in PART_EXTENSIONS


#: In ``beh/``, what is the recording when no ``_beh`` file is: the events
#: of a behavioural experiment, or eye tracking (physio) on its own.
_BEH_ALONE = frozenset({"events", "physio"})


def _entities_of(name: str) -> tuple[dict, str]:
    from ..vendor.bidsval.context.entities import parse_filename

    entities, suffix, _ext = parse_filename(_schema(), name)
    return entities, suffix


def list_recordings(datatype: str, entries: Iterable[tuple[str, bool]]) -> list[str]:
    """The names in one datatype folder that get a row: ``entries`` are
    ``(name, is_dir)``, on disk or planned. Pure, so the BIDS preview and
    the metadata step decide by the same rule.

    :func:`is_scans_entry` for each, and, in ``beh/``, an ``_events.tsv`` or
    ``_physio.tsv.gz`` that no ``_beh`` file shares its name with: a
    behavioural experiment, or eye tracking, with nothing else recorded.
    """
    entries = [(n, d) for n, d in entries if n and not n.startswith(".")]
    out = []
    beh = [_entities_of(n)[0] for n, _d in entries
           if datatype == "beh" and _entities_of(n)[1] == "beh"]
    for name, is_dir in entries:
        tail = f"{datatype}/{name}"
        if is_scans_entry(tail, is_dir=is_dir):
            out.append(name)
            continue
        if datatype != "beh" or is_dir or _rule_name(tail, False) is None:
            continue
        entities, suffix = _entities_of(name)
        if suffix not in _BEH_ALONE or _extension(name) not in (".tsv", ".tsv.gz"):
            continue
        want = {k: v for k, v in entities.items() if k != "recording"}
        if not any(all(b.get(k) == v for k, v in want.items()) for b in beh):
            out.append(name)
    return out


def _ignored(root: Path):
    from ..vendor.bidsval.files.bidsignore import load_bidsignore

    return load_bidsignore(root)


def session_recordings(dataset_root: Path, session_root: Path) -> list[Path]:
    """Every recording in one session (or subject, without sessions), in path
    order, as the scans table lists them.

    Walks the session's datatype folders only, takes a directory recording as
    one entry without descending into it, and honours ``.bidsignore``.
    """
    dataset_root = Path(dataset_root)
    session_root = Path(session_root)
    ignore = _ignored(dataset_root)
    out: list[Path] = []
    if not session_root.is_dir():
        return out
    for folder in sorted(session_root.iterdir(), key=lambda p: p.name):
        if not folder.is_dir() or folder.name not in _datatypes():
            continue
        entries = [e for e in sorted(folder.iterdir(), key=lambda p: p.name)
                   if not ignore.match(e.relative_to(dataset_root).as_posix())]
        listed = set(list_recordings(folder.name, ((e.name, e.is_dir()) for e in entries)))
        out.extend(e for e in entries if e.name in listed)
    return out


# ---------------------------------------------------------------------------
# Times
# ---------------------------------------------------------------------------


def _sidecar(recording: Path) -> Path:
    name = recording.name
    base = name.split(".", 1)[0]
    return recording.with_name(base + ".json")


def is_datetime(value: str) -> bool:
    """A value ``acq_time`` may hold (a full date and time), ``n/a`` aside."""
    return bool(_DATETIME.match(str(value or "").strip()))


def acq_time(recording: Path, times: Optional[Mapping[str, str]] = None) -> str:
    """When ``recording`` started, as ``acq_time`` writes it, or ``n/a``.

    The sidecar's ``AcquisitionDateTime``; else ``times[stem]``, what the
    conversion knew (the scan's study date and time); else ``n/a``. A
    sidecar's ``AcquisitionTime`` alone is a time without a date, which the
    standard does not accept here, so it is not used.
    """
    sidecar = _sidecar(recording)
    if sidecar.is_file():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            meta = {}
        value = meta.get("AcquisitionDateTime") if isinstance(meta, dict) else None
        if isinstance(value, str) and is_datetime(value):
            return value
    if times:
        value = times.get(recording.name.split(".", 1)[0], "")
        if is_datetime(value):
            return value
    return NA


def _recording_key(name: str) -> str:
    """The name with the parts of one recording (echo, part, ...) taken out."""
    base = name.split(".", 1)[0]
    tokens = [t for t in base.split("_")
              if t.split("-", 1)[0] not in SAME_RECORDING]
    return "_".join(tokens)


# ---------------------------------------------------------------------------
# Reading and writing a table
# ---------------------------------------------------------------------------


def read_table(path: Path) -> tuple[list[str], list[dict]]:
    """``(header, rows)`` of a TSV, a leading byte-order mark ignored (MNE-BIDS
    writes one), every value a string. Missing file: empty."""
    path = Path(path)
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        rows = [{k: ("" if v is None else str(v)) for k, v in row.items() if k is not None}
                for row in reader]
    return header, rows


def table_text(header: list[str], rows: list[dict]) -> str:
    """A TSV body: tab separated, ``\\n`` line ends, blanks as ``n/a``."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n",
                        quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    for row in rows:
        writer.writerow([_cell(row.get(col, "")) for col in header])
    return buffer.getvalue()


def _cell(value) -> str:
    text = "" if value is None else str(value)
    return NA if not text.strip() else text


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass
class TableEdit:
    """One table as it will be written, and why."""

    path: Path
    kind: str                       # "scans" | "sessions"
    header: list[str]
    rows: list[dict]
    created: bool = False
    added: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)   # (key, reason)
    filled: int = 0                 # blank cells given n/a, or a time
    times: int = 0                  # of which, acquisition times found

    @property
    def key(self) -> str:
        return "filename" if self.kind == "scans" else "session_id"

    def text(self) -> str:
        return table_text(self.header, self.rows)

    def summary(self) -> str:
        parts = []
        if self.created:
            parts.append(f"new, {_count(len(self.rows), 'row')}")
        elif self.added:
            parts.append(f"{_count(len(self.added), 'row')} added")
        if self.dropped:
            parts.append(f"{_count(len(self.dropped), 'row')} removed")
        if self.filled and not self.created:
            parts.append(f"{_count(self.filled, 'blank cell')} filled")
        return ", ".join(parts) or "unchanged"


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _session_roots(dataset_root: Path) -> list[tuple[Path, Path]]:
    """``(subject folder, session root)`` for every session, or the subject
    itself where it has none."""
    out = []
    for subject in sorted(Path(dataset_root).glob("sub-*")):
        if not subject.is_dir():
            continue
        sessions = [s for s in sorted(subject.glob("ses-*")) if s.is_dir()]
        out.extend((subject, s) for s in (sessions or [subject]))
    return out


def scans_path(subject: Path, session_root: Path) -> Path:
    ses = f"_{session_root.name}" if session_root != subject else ""
    return session_root / f"{subject.name}{ses}_scans.tsv"


def _plan_scans(dataset_root: Path, subject: Path, session_root: Path,
                times: Optional[Mapping[str, str]]) -> Optional[TableEdit]:
    path = scans_path(subject, session_root)
    recordings = session_recordings(dataset_root, session_root)
    header, existing = read_table(path)
    if not recordings and not existing:
        return None
    if path.is_file() and "filename" not in header:
        return None             # not a table this can read safely; left alone
    listed = {r.relative_to(session_root).as_posix(): r for r in recordings}

    # A table that has no acq_time keeps that shape: the column is optional,
    # and the table is the user's. A new table gets one.
    header = header or ["filename", "acq_time"]
    edit = TableEdit(path, "scans", header, [], created=not path.is_file())

    # Existing rows: kept unless the file is gone, or the row lists a part of
    # a recording that has its own row.
    primary_stems = {Path(rel).as_posix().split(".", 1)[0] for rel in listed}
    listings: dict[Path, frozenset] = {}
    seen = set()
    for row in existing:
        name = row.get("filename", "").strip()
        if not name or name in seen:
            edit.dropped.append((name or "(empty)", "listed twice" if name else "no filename"))
            continue
        if not _exists_exactly(session_root, name, listings):
            edit.dropped.append((name, "the file is not there"))
            continue
        if name not in listed and name.split(".", 1)[0] in primary_stems:
            edit.dropped.append((name, "part of a recording listed on its own row"))
            continue
        seen.add(name)
        edit.rows.append(dict(row))

    for name, recording in listed.items():
        if name in seen:
            continue
        row = {"filename": name}
        if "acq_time" in header:
            row["acq_time"] = acq_time(recording, times)
        edit.rows.append(row)
        edit.added.append(name)

    # Blank cells: a time where one can be found, else n/a.
    for row in edit.rows:
        for col in header:
            if str(row.get(col, "")).strip():
                continue
            if col == "acq_time" and row.get("filename", "") in listed:
                row[col] = acq_time(listed[row["filename"]], times)
            else:
                row[col] = NA
            if row["filename"] not in edit.added:
                edit.filled += 1

    # The parts of one recording share one time: the earliest known.
    groups: dict[str, list[dict]] = {}
    for row in edit.rows:
        groups.setdefault(_recording_key(row["filename"]), []).append(row)
    for rows in groups.values():
        added = [r for r in rows if r["filename"] in edit.added]
        known = sorted(r.get("acq_time", "") for r in rows
                       if is_datetime(r.get("acq_time", "")))
        if len(rows) > 1 and added and known:
            for r in added:
                r["acq_time"] = known[0]

    edit.times = sum(1 for r in edit.rows if r["filename"] in edit.added
                     and is_datetime(r.get("acq_time", "")))
    if edit.created:
        edit.rows.sort(key=lambda r: r["filename"])
    return edit


def _exists_exactly(base: Path, rel: str, listings: dict) -> bool:
    """Is ``base/rel`` there, spelled exactly so?

    ``Path.exists`` answers yes on macOS and Windows for ``anat/sub-01_t1w``
    when the file is ``sub-01_T1w``: those file systems ignore case, BIDS
    does not, and a row naming the wrong spelling went unreported.
    """
    folder = base
    for part in Path(rel).parts:
        names = listings.get(folder)
        if names is None:
            try:
                names = frozenset(p.name for p in folder.iterdir())
            except OSError:
                names = frozenset()
            listings[folder] = names
        if part not in names:
            return False
        folder = folder / part
    return True


def _plan_sessions(subject: Path, scans: list[TableEdit]) -> Optional[TableEdit]:
    sessions = [s for s in sorted(subject.glob("ses-*")) if s.is_dir()]
    path = subject / f"{subject.name}_sessions.tsv"
    header, existing = read_table(path)
    if not sessions and not existing:
        return None
    if path.is_file() and "session_id" not in header:
        return None
    header = header or ["session_id", "acq_time"]
    edit = TableEdit(path, "sessions", header, [], created=not path.is_file())

    # Each session's earliest scan, from the scans tables as they will be.
    earliest: dict[str, str] = {}
    for table in scans:
        if table.path.parent.parent != subject:
            continue
        found = sorted(r.get("acq_time", "") for r in table.rows
                       if is_datetime(r.get("acq_time", "")))
        if found:
            earliest[table.path.parent.name] = found[0]

    present = {s.name for s in sessions}
    seen = set()
    for row in existing:
        sid = row.get("session_id", "").strip()
        if not sid or sid in seen:
            edit.dropped.append((sid or "(empty)", "listed twice" if sid else "no session_id"))
            continue
        if sid not in present:
            edit.dropped.append((sid, "the session folder is not there"))
            continue
        seen.add(sid)
        edit.rows.append(dict(row))
    for sid in sorted(present - seen):
        row = {"session_id": sid}
        if "acq_time" in header:
            row["acq_time"] = earliest.get(sid, NA)
        edit.rows.append(row)
        edit.added.append(sid)
    for row in edit.rows:
        for col in header:
            if str(row.get(col, "")).strip():
                continue
            row[col] = earliest.get(row["session_id"], NA) if col == "acq_time" else NA
            if row["session_id"] not in edit.added:
                edit.filled += 1
    edit.times = sum(1 for r in edit.rows if r["session_id"] in edit.added
                     and is_datetime(r.get("acq_time", "")))
    return edit


def plan_tables(dataset_root: Path, *, scans: bool = True, sessions: bool = True,
                times: Optional[Mapping[str, str]] = None,
                subjects: Optional[Iterable[str]] = None) -> list[TableEdit]:
    """Every table that would change, as it would be written.

    ``times`` maps a recording's name without its extension to an
    ``acq_time`` the conversion knew; ``subjects`` limits the plan to those
    ``sub-`` folders. Tables already as they should be are left out.
    """
    dataset_root = Path(dataset_root)
    wanted = None if subjects is None else set(subjects)
    planned: list[TableEdit] = []
    by_subject: dict[Path, list[TableEdit]] = {}
    for subject, root in _session_roots(dataset_root):
        if wanted is not None and subject.name not in wanted:
            continue
        edit = _plan_scans(dataset_root, subject, root, times)
        if edit is not None:
            by_subject.setdefault(subject, []).append(edit)
            if scans:
                planned.append(edit)
        else:
            by_subject.setdefault(subject, [])
    if sessions:
        for subject, tables in by_subject.items():
            edit = _plan_sessions(subject, tables)
            if edit is not None:
                planned.append(edit)
    return [e for e in planned if _changes(e)]


def _changes(edit: TableEdit) -> bool:
    if not edit.path.is_file():
        return bool(edit.rows)
    try:
        current = edit.path.read_text(encoding="utf-8-sig")
    except OSError:
        return True
    return current != edit.text()


def write_tables(edits: Iterable[TableEdit], op=None) -> list[Path]:
    """Write ``edits``; through ``op`` (an editor operation) when given, so
    the whole is one undoable step. Returns the tables written."""
    written = []
    for edit in edits:
        if op is not None:
            op.write_text(edit.path, edit.text())
        else:
            edit.path.parent.mkdir(parents=True, exist_ok=True)
            edit.path.write_text(edit.text(), encoding="utf-8", newline="")
        written.append(edit.path)
    return written


def times_from_inventory(df) -> dict[str, str]:
    """``{name without extension: acq_time}`` for the rows of a scan
    inventory whose study date and time are known (``study_date`` +
    ``acq_time`` from the DICOM header)."""
    out: dict[str, str] = {}
    if df is None or not len(df) or "bids_name" not in df.columns:
        return out
    for _, row in df.iterrows():
        name = str(row.get("bids_name", "") or "").strip()
        stamp = iso_datetime(str(row.get("study_date", "") or ""),
                             str(row.get("acq_time", "") or ""))
        if name and stamp:
            out[name] = stamp
    return out


def iso_datetime(date: str, time: str) -> str:
    """``20250526`` and ``111430.5025`` (or ``11:14:30.5025``) as
    ``2025-05-26T11:14:30.502500``; empty when either is missing."""
    digits = re.sub(r"\D", "", date or "")
    whole, _dot, frac = (time or "").replace(":", "").strip().partition(".")
    whole = re.sub(r"\D", "", whole)
    if len(digits) < 8 or len(whole) < 6:
        return ""
    stamp = (f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
             f"T{whole[:2]}:{whole[2:4]}:{whole[4:6]}")
    return stamp + (f".{(frac + '000000')[:6]}" if frac.isdigit() else "")


_register()

__all__ = [
    "NA", "PART_EXTENSIONS", "SAME_RECORDING", "SIDE_RULES", "TableEdit", "acq_time",
    "is_datetime", "is_scans_entry", "iso_datetime", "list_recordings", "plan_tables",
    "read_table",
    "scans_path", "session_recordings", "table_text", "times_from_inventory",
    "write_tables",
]

"""Delete recordings, datatypes and sessions, and everything that names them.

Deleting a file is the easy half. The half that makes a dataset wrong is what
is left pointing at it: a ``*_scans.tsv`` row for a recording that is not
there, an ``IntendedFor`` naming a file nothing has, a ``participants.tsv``
row for a subject with no data, and the empty ``anat/`` that every tool
listing datatypes will read as "this subject has anatomy".

So this is the same shape as :mod:`bidsmgr.editor.restructure`: work out
everything the deletion touches, show it, then do all of it as ONE operation.
It is applied through :func:`bidsmgr.project.operations.begin_operation`,
which copies each file aside before removing it, so the whole thing is a
single undoable step in the Editor's history rather than something you have to
be sure about beforehand.

**What it refuses.** The dataset root, anything inside ``.bidsmgr/`` (that is
the operation log, and deleting it would destroy the record that makes the
deletion reversible), and ``dataset_description.json``, without which the
directory stops being a BIDS dataset at all. Everything at or under a subject
is fair game.

Qt-free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterable, Optional

from .rename import (
    RenameError,
    _load_json,
    _rel,
    PATH_FIELDS,
    build_ref_map,
    scans_home,
    scans_table_in,
    walk_dataset,
)
from .restructure import _under_root, _within, companions

log = logging.getLogger(__name__)

# Files whose loss stops the directory being a dataset, rather than just
# making it a smaller one.
_DATASET_CRITICAL = ("dataset_description.json",)

# Ours, not the standard's. The log inside it is what makes this reversible.
_TOOL_DIRS = (".bidsmgr", ".git", ".datalad")

# The one tool directory that belongs to whatever folder it sits INSIDE rather
# than to the dataset. Conversions before 1.3 wrote each subject's provenance
# to ``sub-XXX/.bidsmgr/``, and because tool state is excluded from a delete,
# deleting such a subject removed every recording and then left the folder
# standing with a hidden file in it. New conversions write to the dataset
# root instead, but datasets built by an older version are still out there.
#
# ``.git`` and ``.datalad`` are deliberately NOT in here: nested ones mean a
# DataLad subdataset, which is somebody else's repository, not our bookkeeping.
_OWN_STATE_DIR = ".bidsmgr"


@dataclass
class ScansDrop:
    """Rows to remove from one ``*_scans.tsv``."""

    table: Path
    rel: str
    rows: list[str] = dc_field(default_factory=list)
    # Every row it has would go, so the table describes nothing and goes too.
    empties: bool = False


@dataclass
class RefDrop:
    """``IntendedFor`` entries in one sidecar that name a deleted file."""

    path: Path
    rel: str
    # Which field: IntendedFor, AssociatedEmptyRoom, Sources, ...
    field: str = "IntendedFor"
    entries: list[str] = dc_field(default_factory=list)
    # Nothing would be left, so the key is removed rather than left as an
    # empty list. An empty list is a claim that the fieldmap is intended for
    # nothing, which is a different statement from not saying.
    empties: bool = False


@dataclass
class DeletePlan:
    """Everything a deletion would do, before any of it is done."""

    files: list[Path] = dc_field(default_factory=list)
    scans_drops: list[ScansDrop] = dc_field(default_factory=list)
    ref_drops: list[RefDrop] = dc_field(default_factory=list)
    # ``sub-`` labels whose participants.tsv row would go, because the whole
    # subject is being removed.
    participants: list[str] = dc_field(default_factory=list)
    # ``(sessions table, ses- label)`` rows that go with a session folder the
    # deletion empties, in a subject that stays.
    session_rows: list[tuple[Path, str]] = dc_field(default_factory=list)
    # Inherited sidecars left describing nothing once the files go.
    orphaned_sidecars: list[Path] = dc_field(default_factory=list)
    emptied: list[Path] = dc_field(default_factory=list)
    conflicts: list[str] = dc_field(default_factory=list)
    title: str = ""
    # The picked folders themselves: every folder inside them goes when it
    # ends up empty, the ones that never held a file included.
    folders: list[Path] = dc_field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.files and not self.emptied

    @property
    def only_folders(self) -> bool:
        """Nothing but empty folders to remove."""
        return not self.files and bool(self.emptied)

    @property
    def n_files(self) -> int:
        return len(self.files)

    def file_key(self, root: Path, path: Path) -> str:
        """The stable identifier for one removal: its path within the root."""
        return _rel(root, path)

    def file_keys(self, root: Path) -> list[str]:
        return [self.file_key(root, p) for p in self.files]

    def selected(
        self, root: Path, only: Optional[set[str]],
    ) -> list[Path]:
        if only is None:
            return list(self.files)
        return [p for p in self.files if self.file_key(root, p) in only]

    def bytes_freed(self) -> int:
        total = 0
        for path in self.files:
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def summary(self) -> str:
        if self.only_folders:
            return f"{len(self.emptied)} empty folder(s)"
        parts = [f"{len(self.files)} file(s)"]
        rows = sum(len(d.rows) for d in self.scans_drops)
        if rows:
            parts.append(f"{rows} scans row(s)")
        refs = sum(len(d.entries) for d in self.ref_drops)
        if refs:
            parts.append(f"{refs} IntendedFor entry(ies)")
        if self.participants:
            parts.append(f"{len(self.participants)} participants row(s)")
        if self.session_rows:
            parts.append(f"{len(self.session_rows)} sessions row(s)")
        if self.orphaned_sidecars:
            parts.append(f"{len(self.orphaned_sidecars)} orphaned sidecar(s)")
        if self.emptied:
            parts.append(f"{len(self.emptied)} folder(s)")
        return ", ".join(parts)

    def verb(self) -> str:
        return self.title or f"Delete {len(self.files)} file(s)"


# --------------------------------------------------------------------------
# Planning


def _refuse(root: Path, target: Path) -> Optional[str]:
    """Why this target may not be deleted, or ``None`` if it may."""
    if target == root:
        return "the dataset root itself cannot be deleted from here"
    if not _within(root, target):
        return f"{target} is outside the dataset"
    parts = target.relative_to(root).parts
    if parts and parts[0] in _TOOL_DIRS:
        return (
            f"{parts[0]}/ holds the operation log that makes deletions "
            "reversible, so it is not deletable from here"
        )
    if len(parts) == 1 and parts[0] in _DATASET_CRITICAL:
        return (
            f"{parts[0]} is what makes this a BIDS dataset; removing it "
            "would leave an ordinary folder"
        )
    return None


def _files_under(root: Path, targets: Iterable[Path]) -> list[Path]:
    """Every file the selection covers.

    A folder contributes everything beneath it, INCLUDING its
    ``*_scans.tsv``: the folder is going, so the table describing it goes with
    it. That is the one place this differs from
    :func:`bidsmgr.editor.restructure.expand`, which drops those tables
    because a rename relocates their rows rather than removing them.

    A file contributes itself and its companions, because deleting a
    recording and leaving its sidecar produces an orphan that no validator can
    attribute to anything.
    """
    targets = list(targets)
    folders = [t for t in targets if t.is_dir()]
    everything = walk_dataset(root) if folders else []

    picked: list[Path] = []
    for target in targets:
        if target.is_dir():
            picked.extend(p for p in everything if _within(target, p))
            # walk_dataset skips the tool directories and top-level files, so
            # a table living directly in the folder is picked up explicitly.
            picked.extend(
                p for p in sorted(target.rglob("*_scans.tsv")) if p.is_file()
            )
            # A .bidsmgr/ INSIDE the folder is that folder's own provenance,
            # written there by conversions before 1.3. Leaving it behind is
            # what made "delete this subject" remove every recording and then
            # leave the subject folder standing.
            picked.extend(
                p for p in sorted(target.rglob("*"))
                if p.is_file()
                and _OWN_STATE_DIR in p.relative_to(target).parts
            )
        elif target.is_file():
            picked.extend(companions(root, target))
            if target.name.endswith("_scans.tsv"):
                picked.append(target)
    return sorted(dict.fromkeys(picked))


def _keep_tables_still_needed(files: list[Path]) -> list[Path]:
    """Take any ``*_scans.tsv`` back out of the deletion if it still has work.

    A table inside a folder that is going goes with it, which is right when
    the whole session or subject goes. It is wrong the moment somebody unticks
    part of the selection: the table is a leaf in the preview like any other
    file, so leaving it ticked while keeping a recording would delete the
    thing that describes what survived.

    That cannot be left to the user to notice. A table is dropped from the
    deletion whenever anything it describes is staying, and it is then EDITED
    instead, which is what the scans handling does for every other partial
    case anyway.
    """
    going = set(files)
    kept: list[Path] = []
    for path in files:
        if not path.name.endswith("_scans.tsv"):
            kept.append(path)
            continue
        survivors = [
            p for p in path.parent.rglob("*")
            if p.is_file() and p not in going
            and not p.name.endswith("_scans.tsv")
            and not any(part in _TOOL_DIRS for part in p.parts)
        ]
        if not survivors:
            kept.append(path)
    return kept


def plan_delete(root: Path, targets: Iterable[Path]) -> DeletePlan:
    """Work out everything a deletion would touch.

    Nothing is removed: the result is a plan, for the same reason the rename
    and restructure dialogs show one first, and more so here, because this is
    the operation where a surprise is expensive.
    """
    # Spelled against the caller's root and never resolved, so the plan and
    # the applier agree about what a path IS. See
    # ``restructure._under_root`` and CROSS_PLATFORM_RULES section 1.5.
    root = Path(root)
    targets = [_under_root(root, Path(t)) for t in targets]

    plan = DeletePlan()
    keep: list[Path] = []
    for target in targets:
        why = _refuse(root, target)
        if why:
            plan.conflicts.append(why)
        else:
            keep.append(target)
    if not keep:
        return plan

    plan.files = _keep_tables_still_needed(_files_under(root, keep))
    plan.folders = [t for t in keep if t.is_dir()]
    if not plan.files:
        # Nothing but empty folders (or folders holding only the operating
        # system's droppings): they can be deleted too.
        plan.emptied = _emptied(root, [], set(), plan.folders)
        if plan.emptied:
            plan.title = (f"Delete the empty folder {_rel(root, plan.emptied[-1])}"
                          if len(plan.emptied) == 1
                          else f"Delete {len(plan.emptied)} empty folders")
        return plan

    going = set(plan.files)
    plan.title = _title(root, keep, plan.files)
    plan.scans_drops = _scans_drops(root, plan.files, going)
    plan.ref_drops = _ref_drops(root, plan.files, going)
    # A table that ends up describing nothing is deleted too, but by
    # ``_drop_scans_rows`` rather than by the file loop, so it is not in
    # ``going``. Everything downstream has to count it as leaving anyway: it
    # is the last thing in a session folder, and a check that treated it as a
    # survivor concluded the session was still in use. The symptom was an
    # empty ``sub-001/ses-01/`` left on disk after its only datatype went, and
    # a participants row for a subject with no data.
    gone = _also_going(going, plan.scans_drops)
    plan.orphaned_sidecars = _orphaned_sidecars(root, gone)
    gone |= set(plan.orphaned_sidecars)
    plan.participants = _orphaned_subjects(root, gone)
    plan.emptied = _emptied(
        root, plan.files + plan.orphaned_sidecars, gone, plan.folders,
    )
    plan.session_rows = _session_rows(plan.emptied, gone)
    return plan


def _also_going(going: set[Path], drops: list[ScansDrop]) -> set[Path]:
    """``going`` plus the scans tables that empty, so they are not survivors."""
    return going | {drop.table for drop in drops if drop.empties}


def _title(root: Path, targets: list[Path], files: list[Path]) -> str:
    """What this is, in words, for the history entry and the dialog."""
    if len(targets) == 1:
        one = targets[0]
        rel = _rel(root, one)
        if one.is_dir():
            return f"Delete {rel} and its {_count(len(files), 'file')}"
        if len(files) > 1:
            return f"Delete {rel} and its {_count(len(files) - 1, 'companion')}"
        return f"Delete {rel}"
    return f"Delete {_count(len(targets), 'item')}, {_count(len(files), 'file')}"


def _count(n: int, word: str) -> str:
    """``1 file``, ``3 files``: the title shows in the Editor's history."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _scans_drops(
    root: Path, files: list[Path], going: set[Path],
) -> list[ScansDrop]:
    """Which ``*_scans.tsv`` rows name a file that is about to go.

    A table that is itself being deleted is skipped: there is no point
    editing a file on its way out, and doing so would make the operation
    record a write it then undoes.
    """
    from .tsv_edit import read_table

    by_table: dict[Path, ScansDrop] = {}
    for path in files:
        if path.name.endswith("_scans.tsv"):
            continue
        home = scans_home(path, root)
        if home is None:
            continue
        table = scans_table_in(home)
        if table in going or not table.exists():
            continue
        drop = by_table.get(table)
        if drop is None:
            drop = ScansDrop(table=table, rel=_rel(root, table))
            by_table[table] = drop
        drop.rows.append(_rel(home, path))

    out = []
    for table, drop in by_table.items():
        current = read_table(table)
        if current is None or "filename" not in current.header:
            continue
        present = {
            _cell(row, current.header, "filename") for row in current.rows
        }
        # Only rows the table ACTUALLY has. Every file going contributes a
        # candidate, but a scans table lists recordings, not their sidecars,
        # so deleting a bold and its .json and its _events.tsv is three files
        # and ONE row. Counting the candidates instead told the user three
        # rows would go and then removed one, which is the kind of small lie
        # that makes a preview not worth reading.
        drop.rows = [rel for rel in drop.rows if rel in present]
        if not drop.rows:
            continue
        remaining = [
            row for row in current.rows
            if _cell(row, current.header, "filename") not in set(drop.rows)
        ]
        drop.empties = not remaining
        out.append(drop)
    return out


def _cell(row: list[str], header: list[str], column: str) -> str:
    try:
        index = header.index(column)
    except ValueError:
        return ""
    return row[index] if index < len(row) else ""


def _ref_drops(
    root: Path, files: list[Path], going: set[Path],
) -> list[RefDrop]:
    """Which ``IntendedFor`` entries name a file that is about to go.

    The set of spellings comes from :func:`build_ref_map`, so a reference is
    recognised in all three forms BIDS allows it to take rather than in
    whichever one this module happened to think of.
    """
    keys = set(build_ref_map(root, [(p, p) for p in files]))
    out: list[RefDrop] = []
    for path in walk_dataset(root):
        if path.suffix != ".json" or path in going:
            continue
        data = _load_json(path)
        if not data:
            continue
        # Every field that points at another file. A deleted empty-room
        # recording has to be taken out of the MEG sidecar naming it just as
        # much as out of a fieldmap's IntendedFor. See rename.PATH_FIELDS.
        for field in PATH_FIELDS:
            if field not in data:
                continue
            value = data[field]
            items = [
                str(v) for v in
                (value if isinstance(value, list) else [value])
            ]
            doomed = [v for v in items if _body(v) in keys]
            if not doomed:
                continue
            out.append(RefDrop(
                path=path,
                rel=_rel(root, path),
                field=field,
                entries=doomed,
                empties=len(doomed) == len(items),
            ))
    return out


def _body(text: str) -> str:
    """The path inside a BIDS URI, or the text unchanged.

    Shares rename's scheme handling so bids:derivatives: is understood
    too, not only the bare bids:: form.
    """
    from .rename import _split_uri

    return _split_uri(text)[1]


def _orphaned_sidecars(root: Path, going: set[Path]) -> list[Path]:
    """Inherited sidecars that would describe nothing, so they go too.

    BIDS lets a sidecar sit ABOVE the recordings it applies to: a
    ``magnitude1.json`` at the dataset root supplies fields to every
    ``*_magnitude1.nii.gz`` beneath it. Those are deliberately NOT treated as
    companions, because they describe more than one recording and must not
    travel with any single one. But once the last recording they feed is gone,
    they feed nothing, and a sidecar applying to no data file is exactly what
    the validator reports as ``SIDECAR_WITHOUT_DATAFILE``.

    Two questions have to be answered from the schema rather than guessed, and
    getting either wrong deletes something it should not:

    * **Is this JSON a sidecar at all?** ``meg/coordsystem.json`` and
      ``dataset_description.json`` are JSON files that ARE the data, not
      descriptions of other files. ``list_extensions`` settles it: when
      ``.json`` is the only extension the suffix allows, the file is the
      thing itself and is left alone.
    * **Is this suffix a recording suffix?** ``datatypes_with_suffix``
      returns nothing for ``description``, ``participants`` and ``scans``, so
      those are not recording sidecars and are never considered here.

    "Feeds" means the inheritance rule: a surviving non-JSON file somewhere
    below this sidecar's own directory, with the same suffix and a SUPERSET of
    its entities. A superset, because the sidecar is more general than what it
    applies to; an equal set is the ordinary companion case and is true too.
    """
    from ..schema import datatypes_with_suffix, list_extensions

    from .restructure import describe_file

    candidates: list[tuple[Path, str, dict[str, str]]] = []
    survivors: list[tuple[Path, str, dict[str, str], Path]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(p in _TOOL_DIRS for p in path.parts):
            continue
        facts = describe_file(root, path)
        if facts is None or not facts.suffix:
            continue
        if path.suffix == ".json":
            if path in going:
                continue
            datatypes = datatypes_with_suffix(facts.suffix)
            if not datatypes:
                continue        # not a recording suffix at all
            if all(
                list_extensions(dt, facts.suffix) == [".json"]
                for dt in datatypes
            ):
                continue        # the JSON is the data, not a description of it
            candidates.append((path, facts.suffix, facts.entities))
        elif path not in going:
            survivors.append((path, facts.suffix, facts.entities, path.parent))

    out: list[Path] = []
    for sidecar, suffix, entities in candidates:
        scope = sidecar.parent
        fed = any(
            other_suffix == suffix
            and _within(scope, other)
            and entities.items() <= other_entities.items()
            for other, other_suffix, other_entities, _parent in survivors
        )
        if not fed:
            out.append(sidecar)
    return out


def _orphaned_subjects(root: Path, going: set[Path]) -> list[str]:
    """Subjects that would have no data left, so their row should go too.

    A ``participants.tsv`` row for a subject with no recordings is a dataset
    claiming a participant it does not have, which is the same class of
    wrongness as an empty ``anat/`` claiming a modality.
    """
    out: list[str] = []
    for subject in sorted(root.glob("sub-*")):
        if not subject.is_dir():
            continue
        survivors = [
            p for p in subject.rglob("*")
            if p.is_file() and p not in going
            and not any(part in _TOOL_DIRS for part in p.parts)
        ]
        if not survivors:
            out.append(subject.name)
    return out


def _session_rows(emptied: list[Path], going: set[Path]) -> list[tuple[Path, str]]:
    """The ``*_sessions.tsv`` rows of the session folders a deletion empties.

    A row for a session that is gone is the sessions table claiming one the
    subject no longer has. Only where the table stays: a whole subject takes
    its table with it.
    """
    out: list[tuple[Path, str]] = []
    for folder in emptied:
        subject = folder.parent
        if not (folder.name.startswith("ses-") and subject.name.startswith("sub-")):
            continue
        table = subject / f"{subject.name}_sessions.tsv"
        if table.is_file() and table not in going and subject not in emptied:
            out.append((table, folder.name))
    return out


def _emptied(root: Path, files: list[Path], going: set[Path],
             folders: Iterable[Path] = ()) -> list[Path]:
    """Folders that would have nothing left in them, deepest first.

    The folders the files leave, every folder inside a picked folder (one
    that never held a file would otherwise stop its parent from going), and
    the operating system's droppings (``.DS_Store``) do not count as
    something left: they go with the folder."""
    import os

    from ..project.operations import is_os_junk

    candidates: list[Path] = []
    for path in files:
        folder = path.parent
        while folder != root and _within(root, folder):
            candidates.append(folder)
            folder = folder.parent
    for picked in folders:
        if picked.is_dir() and picked != root and _within(root, picked):
            candidates.append(picked)
            for dirpath, _dirs, _files in os.walk(picked):
                candidates.append(Path(dirpath))

    out: list[Path] = []
    for folder in dict.fromkeys(candidates):
        survivors = [
            p for p in folder.rglob("*")
            if p.is_file() and p not in going and not is_os_junk(p.name)
        ]
        if not survivors:
            out.append(folder)
    return sorted(out, key=lambda p: len(p.parts), reverse=True)


# --------------------------------------------------------------------------
# Applying


def apply_delete(
    root: Path, plan: DeletePlan, *, only: Optional[set[str]] = None,
) -> tuple[int, list[str]]:
    """Perform ``plan``. Returns ``(items_touched, errors)``.

    ``only`` is the set of relative paths to delete, as
    :meth:`DeletePlan.file_key` reports them. ``None`` deletes everything the
    plan found.

    A PARTIAL deletion is a different operation from a whole one, and the
    difference is handled rather than assumed away: the tables, references,
    participants rows and folders are all recomputed from what is ACTUALLY
    going, so unticking a recording leaves its scans row, its ``IntendedFor``
    entry and its folder exactly where they were.
    """
    from ..project.operations import begin_operation

    root = Path(root)
    if plan.conflicts and only is None:
        raise RenameError(
            "this deletion was refused:\n  " + "\n  ".join(plan.conflicts)
        )

    files = plan.selected(root, only)
    if not files and not plan.only_folders:
        return 0, []

    # Everything downstream is derived from the files actually going, not from
    # the plan, so a partial selection cannot remove a row or a reference
    # belonging to a file that stays.
    whole = only is None or len(files) == len(plan.files)
    if whole:
        scans_drops = plan.scans_drops
        ref_drops = plan.ref_drops
        participants = plan.participants
        orphans = plan.orphaned_sidecars
        emptied = plan.emptied
        session_rows = plan.session_rows
    else:
        # Recomputed from what is ACTUALLY going, including which tables are
        # still needed: unticking a recording must leave the table that
        # describes it, even when the table was ticked.
        files = _keep_tables_still_needed(files)
        going = set(files)
        scans_drops = _scans_drops(root, files, going)
        ref_drops = _ref_drops(root, files, going)
        gone = _also_going(going, scans_drops)
        orphans = _orphaned_sidecars(root, gone)
        gone |= set(orphans)
        participants = _orphaned_subjects(root, gone)
        emptied = _emptied(root, files + orphans, gone, plan.folders)
        session_rows = _session_rows(emptied, gone)

    errors: list[str] = []
    touched = 0
    label = plan.verb() if whole else (
        f"{plan.verb()} ({len(files)} selected)"
    )

    with begin_operation(root, label) as op:
        # Tables and references FIRST, while the files they name still exist.
        # The other order works but reads as a tool editing around holes it
        # has already made.
        for drop in scans_drops:
            try:
                touched += _drop_scans_rows(op, drop)
            except OSError as exc:
                errors.append(f"{drop.rel}: {exc}")

        for drop in ref_drops:
            try:
                _drop_intended_for(op, drop)
                touched += 1
            except OSError as exc:
                errors.append(f"{drop.rel}: {exc}")

        if participants:
            try:
                touched += _drop_participants(op, root, participants)
            except OSError as exc:
                errors.append(f"participants.tsv: {exc}")

        for table in dict.fromkeys(t for t, _s in session_rows):
            try:
                touched += _drop_session_rows(
                    op, table, {s for t, s in session_rows if t == table})
            except OSError as exc:
                errors.append(f"{_rel(root, table)}: {exc}")

        # The orphans first, then the files. Either order works; this one
        # reads as "clean up what these leave behind, then remove them".
        for path in list(orphans) + list(files):
            try:
                op.delete(path)
                touched += 1
            except OSError as exc:
                errors.append(f"{_rel(root, path)}: {exc}")

        # Deepest first, so a datatype goes before the session holding it.
        for folder in sorted(
            emptied, key=lambda p: len(p.parts), reverse=True,
        ):
            if _remove_if_empty(op, folder, root, errors):
                touched += 1

    return touched, errors


def _drop_scans_rows(op, drop: ScansDrop) -> int:
    """Remove the named rows, or the whole table when nothing would remain."""
    from .tsv_edit import read_table

    table = read_table(drop.table)
    if table is None or "filename" not in table.header:
        return 0
    named = set(drop.rows)
    kept = [
        row for row in table.rows
        if _cell(row, table.header, "filename") not in named
    ]
    if not kept:
        # A scans table with only a header describes nothing. Leaving it is
        # not neutral: it is a file asserting that this subject or session has
        # recordings, and listing none.
        op.delete(drop.table)
        return 1
    table.rows = kept
    op.write_text(drop.table, table.to_text())
    return 1


def _drop_intended_for(op, drop: RefDrop) -> None:
    data = _load_json(drop.path)
    if data is None or drop.field not in data:
        return
    value = data[drop.field]
    was_list = isinstance(value, list)
    items = [str(v) for v in (value if was_list else [value])]
    kept = [v for v in items if v not in set(drop.entries)]
    if kept:
        data[drop.field] = kept if was_list else kept[0]
    else:
        # Removed rather than left as an empty list. An empty list states that
        # this fieldmap is intended for nothing, which is a claim; saying
        # nothing is the honest result of the files having gone.
        data.pop(drop.field, None)
    op.write_json(drop.path, data)


def _drop_participants(op, root: Path, labels: list[str]) -> int:
    from .tsv_edit import read_table

    table_path = root / "participants.tsv"
    if not table_path.exists():
        return 0
    table = read_table(table_path)
    if table is None or "participant_id" not in table.header:
        return 0
    going = set(labels)
    kept = [
        row for row in table.rows
        if _cell(row, table.header, "participant_id") not in going
    ]
    if len(kept) == len(table.rows):
        return 0
    table.rows = kept
    op.write_text(table_path, table.to_text())
    return 1


def _drop_session_rows(op, table_path: Path, sessions: set[str]) -> int:
    """Remove the rows of ``sessions``, or the table when none would remain
    (a sessions table listing nothing describes nothing, like a scans one)."""
    from .tsv_edit import read_table

    table = read_table(table_path)
    if table is None or "session_id" not in table.header:
        return 0
    kept = [row for row in table.rows
            if _cell(row, table.header, "session_id") not in sessions]
    if len(kept) == len(table.rows):
        return 0
    if not kept:
        op.delete(table_path)
        return 1
    table.rows = kept
    op.write_text(table_path, table.to_text())
    return 1


def _remove_if_empty(op, folder: Path, root: Path, errors: list[str]) -> bool:
    """Remove ``folder`` when nothing is left in it but the operating
    system's droppings, which go with it. Recorded, so undo puts the folder
    back and redo removes it again (an empty folder holds nothing to restore,
    but a folder the step removed must reappear on undo even if it was
    empty). True when it went."""
    if folder == root or not _within(root, folder):
        return False
    try:
        return op.remove_if_empty(folder)
    except OSError as exc:
        errors.append(f"{_rel(root, folder)}: {exc}")
        return False


__all__ = [
    "DeletePlan",
    "RefDrop",
    "ScansDrop",
    "apply_delete",
    "plan_delete",
]

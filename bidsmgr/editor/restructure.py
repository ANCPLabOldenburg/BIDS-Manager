"""Change which entities a BIDS filename carries.

Three things a user asks for turn out to be one operation:

* add an entity a file is allowed to have and does not (``acq-highres``);
* remove one it has and does not need (``run-01`` on a subject with one run);
* put a datatype into a session, or take the last session back out.

A session is not a special case of anything. ``ses`` is an ordinary entity that
happens to name a DIRECTORY as well as appearing in the filename, so "create a
session for this datatype" is "add ``ses-<label>`` to these files", and the
folder follows because the schema says that entity has a folder. Writing it any
other way would give two engines that have to agree about scans tables,
``IntendedFor`` and empty folders, and they would stop agreeing.

Everything here produces a :class:`~bidsmgr.editor.rename.RenamePlan` and is
applied by :func:`~bidsmgr.editor.rename.apply_rename`, so the preview, the
per-file selection, the scans-table relocation, the emptied-folder cleanup and
the single undoable history entry are the ones that already work rather than a
second set that looks like them.

**What the schema decides, and what it therefore refuses.** Which entities a
file may carry, which of them are required, what a value may look like and what
order they appear in are all read from the active schema (guard 8: schema facts
have one implementation). So this will not offer ``echo`` on an EEG recording,
will not let go of ``task`` on a ``_bold``, and never has to be told that
``run`` is digits while ``acq`` is a label.

Qt-free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from .rename import (
    PATH_FIELDS,
    ContentEdit,
    RenameError,
    RenamePlan,
    _load_json,
    _rel,
    _stem_and_ext,
    build_ref_map,
    entity_value,
    walk_dataset,
)

log = logging.getLogger(__name__)

# Tables that describe a subject or a session rather than belonging to one
# recording. They are never renamed directly: their rows are relocated by
# ``apply_scans_rows``, which also writes the table at the level the rows now
# belong to and deletes the one it emptied. Renaming the file here as well
# would race that and leave two tables, one of them wrong.
_MANAGED_TABLES = ("_scans.tsv",)


# --------------------------------------------------------------------------
# Reading a filename the way the schema reads it


@dataclass(frozen=True)
class FileFacts:
    """What the schema knows about one file in the tree."""

    path: Path
    rel: str
    datatype: str
    suffix: str
    extension: str
    # Short key to value (``{"sub": "001", "task": "rest"}``), in the order
    # they appear in the name.
    entities: dict[str, str]

    @property
    def is_typed(self) -> bool:
        """Does this sit in a datatype folder with a suffix we can reason about?

        A file that does not is not refused for being unusual; it is refused
        because nothing can say which entities it is allowed to have, and
        guessing is how a tool writes an invalid name confidently.
        """
        return bool(self.datatype and self.suffix)


@lru_cache(maxsize=8)
def _tables(version: str) -> tuple[dict[str, str], frozenset[str]]:
    """``({"acq": "acquisition"}, {datatype folder names})`` for one schema.

    Cached because it is read once per file per plan and a plan covers a whole
    subject, and keyed on the schema VERSION rather than cached outright so
    that switching schemas in Settings is picked up instead of serving the
    previous standard's answer for the rest of the session.
    """
    from ..schema import entity_key_info, entity_keys, list_datatypes

    del version
    short_to_long: dict[str, str] = {}
    for key in entity_keys():
        try:
            short_to_long[key] = entity_key_info(key).key
        except KeyError:
            continue
    return short_to_long, frozenset(list_datatypes())


def _active() -> str:
    from ..schema import schema_version

    try:
        return str(schema_version())
    except Exception:            # a schema that cannot say is still usable
        return ""


def _short_to_long() -> dict[str, str]:
    """``{"acq": "acquisition"}``, from the active schema."""
    return _tables(_active())[0]


def _long_to_short() -> dict[str, str]:
    return {long: short for short, long in _short_to_long().items()}


def _datatypes() -> frozenset[str]:
    """The datatype folder names the ACTIVE schema defines.

    Read rather than assumed, so a folder somebody made called ``notes`` is not
    treated as a datatype and one added in a newer BIDS release is.
    """
    return _tables(_active())[1]


def describe_file(root: Path, path: Path) -> Optional[FileFacts]:
    """Parse one file into the facts the schema needs to reason about it.

    ``None`` when the path is not a file inside the dataset.
    """
    root, path = Path(root), Path(path)
    if not path.is_file():
        return None
    stem, ext = _stem_and_ext(path.name)
    parts = stem.split("_")
    suffix = parts[-1] if parts and "-" not in parts[-1] else ""
    entities: dict[str, str] = {}
    for short in _short_to_long():
        value = entity_value(path.name, short)
        if value:
            entities[short] = value
    return FileFacts(
        path=path,
        rel=_rel(root, path),
        datatype=path.parent.name if path.parent.name in _datatypes() else "",
        suffix=suffix,
        extension=ext,
        entities=entities,
    )


# --------------------------------------------------------------------------
# What may be added, and what may be taken away


@dataclass(frozen=True)
class EntitySlot:
    """One entity offered for this selection, with the schema's terms."""

    key: str            # short form, as it appears in a filename
    long: str           # schema's own name
    display: str
    kind: str           # "label" or "index"
    pattern: str
    description: str
    required: bool      # required by at least one selected file's suffix
    present: int        # how many of the selection already carry it
    total: int
    values: tuple[str, ...] = ()   # values already in use, for a dropdown

    @property
    def is_folder(self) -> bool:
        from .rename import folder_entities

        return self.key in folder_entities()


def _slot(short: str, long: str, *, required: bool, present: int,
          total: int, values: Iterable[str] = ()) -> Optional[EntitySlot]:
    from ..schema import entity_key_info

    try:
        info = entity_key_info(short)
    except KeyError:
        return None
    return EntitySlot(
        key=short,
        long=long,
        display=info.display_name,
        kind=info.format.name,
        pattern=info.format.pattern,
        description=info.description,
        required=required,
        present=present,
        total=total,
        values=tuple(sorted(set(values))),
    )


def _typed(root: Path, files: Iterable[Path]) -> list[FileFacts]:
    out = []
    for path in files:
        facts = describe_file(root, path)
        if facts is not None and facts.is_typed:
            out.append(facts)
    return out


def addable_entities(root: Path, files: Iterable[Path]) -> list[EntitySlot]:
    """Entities at least one selected file is ALLOWED to carry, in filename order.

    The UNION across the selection, the same rule as removal and as the
    Converter's bulk edit. It used to be the intersection, on the grounds that
    offering ``echo`` for a set that is half ``_bold`` and half ``_eeg`` would
    produce a name the standard rejects for one half. That stopped being true
    when :func:`plan_entity_edit` learned to SKIP a file that may not carry the
    entity rather than refuse the whole operation: the rejected name is never
    produced, and the preview lists exactly the files that change. The
    intersection meant a selection spanning datatypes was offered almost
    nothing, while the same selection could have the same entity REMOVED file
    by file, which is two rules for one tool.

    ``sub`` is excluded. Every BIDS file has a subject and no file may gain or
    lose one, and moving a file to a different subject is a rename, which the
    Rename dialog does properly (it carries the participants row with it).
    """
    from ..schema import allowed_entities, entity_order, required_entities

    facts = _typed(root, files)
    if not facts:
        return []

    long_to_short = _long_to_short()
    allowed: set[str] = set()
    required: set[str] = set()
    for item in facts:
        here = set(allowed_entities(item.datatype, item.suffix))
        if not here:
            # No rule for this datatype and suffix pair, so it has no opinion
            # to contribute, and plan_entity_edit moves it without asserting
            # anything about it.
            continue
        allowed |= here
        required |= set(required_entities(item.datatype, item.suffix))

    # Values already used ANYWHERE in the dataset, not just in the selection:
    # adding a session almost always means adding the one the other subjects
    # already have, and typing it again by hand is how it gets typed
    # differently. Gathered in ONE walk for every entity at once; asking
    # ``list_values`` per entity walked the tree a dozen times and was most of
    # what made opening the dialog slow on a real dataset.
    in_use = values_in_use(root)

    out: list[EntitySlot] = []
    for long in entity_order():
        if long not in allowed:
            continue
        short = long_to_short.get(long)
        if not short or short == "sub":
            continue
        present = [f.entities[short] for f in facts if short in f.entities]
        slot = _slot(
            short, long,
            required=long in required,
            present=len(present),
            total=len(facts),
            values=in_use.get(short, ()),
        )
        if slot is not None:
            out.append(slot)
    return out


def removable_entities(root: Path, files: Iterable[Path]) -> list[EntitySlot]:
    """Entities at least one selected file carries and can do without.

    Offered PER FILE, not per selection. Select a whole study and ask for
    ``acq`` to go and it goes from the files that have one and are allowed
    to lose it, leaving the rest alone. Requiring every file in the
    selection to qualify meant the answer to "take the acquisition label
    off this dataset" was almost always "no", because one ``_bold``
    somewhere requires ``task`` and one file somewhere carries no ``acq``.

    So an entity appears here when at least one file both carries it and is
    allowed to do without it. ``present`` counts the files that will
    actually change, which is what the preview then lists, and ``required``
    says whether some OTHER file in the selection requires it, so the
    dialog can say that those will be left alone rather than leaving it to
    be discovered in the preview.
    """
    from ..schema import entity_order, required_entities

    facts = _typed(root, files)
    if not facts:
        return []

    long_to_short = _long_to_short()
    out: list[EntitySlot] = []
    for long in entity_order():
        short = long_to_short.get(long)
        if not short or short == "sub":
            continue
        carried: list[str] = []
        blocked = 0
        for item in facts:
            if short not in item.entities:
                continue
            if long in set(required_entities(item.datatype, item.suffix)):
                blocked += 1
                continue
            carried.append(item.entities[short])
        if not carried:
            continue
        slot = _slot(
            short, long, required=bool(blocked), present=len(carried),
            total=len(facts), values=carried,
        )
        if slot is not None:
            out.append(slot)
    return out


def values_in_use(root: Path) -> dict[str, tuple[str, ...]]:
    """Every value every entity takes in this dataset, from a single walk."""
    seen: dict[str, set[str]] = {}
    shorts = tuple(_short_to_long())
    for path in walk_dataset(Path(root)):
        for short in shorts:
            value = entity_value(path.name, short)
            if value:
                seen.setdefault(short, set()).add(value)
    return {k: tuple(sorted(v)) for k, v in seen.items()}


# --------------------------------------------------------------------------
# What else has to move when one file does


def companions(root: Path, path: Path) -> list[Path]:
    """Every file that is the SAME recording under a different suffix.

    ``sub-01_task-rest_bold.nii.gz`` does not travel alone: its ``.json``
    sidecar, its ``_events.tsv``, a DWI's ``.bval`` and ``.bvec`` all carry the
    same entities and mean nothing apart from it. Renaming the image and
    leaving them is how a converted dataset acquires an orphaned sidecar that
    no validator can attribute to anything.

    Matched on the entity set being IDENTICAL, which is the test that keeps a
    file that applies to several recordings out of it. ``_electrodes.tsv`` and
    an inherited ``task-rest_bold.json`` at the top of a session carry FEWER
    entities on purpose, because they describe more than one file, and moving
    them with any single recording would be wrong.
    """
    root, path = Path(root), Path(path)
    facts = describe_file(root, path)
    if facts is None:
        return [path]
    out = []
    for sibling in sorted(path.parent.glob("*")):
        if not sibling.is_file():
            continue
        other = describe_file(root, sibling)
        if other is not None and other.entities == facts.entities:
            out.append(sibling)
    return out or [path]


def expand(root: Path, targets: Iterable[Path]) -> list[Path]:
    """Turn what the user picked into the files that actually have to move.

    A directory contributes everything under it, a file contributes itself and
    its companions, and scans tables are dropped because their rows are
    relocated rather than their names rewritten.
    """
    # Normalised ONCE, here, and never again below. ``walk_dataset`` yields
    # paths built from the root it was given, so re-spelling the targets
    # against that same root puts both sides of every comparison in one
    # spelling. Mixing them is what CROSS_PLATFORM_RULES section 1.4 warns
    # about: ``/private/var`` against ``/var`` on macOS, drive-letter case and
    # UNC forms on Windows, and a containment test that quietly answers no.
    root = Path(root)
    targets = [_under_root(root, Path(t)) for t in targets]
    folders = [t for t in targets if t.is_dir()]
    # One walk however many folders were picked, and none at all when only
    # files were.
    everything = walk_dataset(root) if folders else []

    picked: list[Path] = []
    for target in targets:
        if target.is_dir():
            picked.extend(p for p in everything if _within(target, p))
        elif target.is_file():
            picked.extend(companions(root, target))
    out: list[Path] = []
    for path in dict.fromkeys(picked):
        if path.name.endswith(_MANAGED_TABLES):
            continue
        out.append(path)
    return sorted(out)


def _under_root(root: Path, path: Path) -> Path:
    """``path`` re-spelled as a descendant of ``root`` exactly as given.

    NOT ``path.resolve()``, and the difference is the whole point.

    Resolving puts every path in the OS's canonical spelling, which makes the
    comparisons here valid and makes the PLAN disagree with the applier:
    ``apply_rename`` is handed the root the caller has, and on macOS that is
    ``/var/folders/...`` while a resolved plan carries ``/private/var/...``.
    Nothing raises. The applier simply fails every ``in`` test it makes
    against the plan, so the scans rows are rewritten in place instead of
    being relocated, and the emptied folders fail their inside-the-root guard
    and are left behind. Both were real, and both were invisible to the tests,
    because pytest's ``tmp_path`` is ALREADY canonical on macOS, so the two
    spellings coincided in every test and diverged on the first real dataset.

    Re-spelling against the caller's root gives the same guarantee the resolve
    was for, since both sides now descend from one root object, and costs no
    syscall per file. See CROSS_PLATFORM_RULES section 1.4.
    """
    rel = _rel(root, path)
    if rel.startswith("/") or ":" in rel.split("/", 1)[0]:
        return path                   # not under the root at all
    return root.joinpath(*rel.split("/"))


def _within(folder: Path, path: Path) -> bool:
    """Is ``path`` inside ``folder``? Both sides must already be resolved.

    Not resolved here on purpose: this is called once per file in the dataset
    and ``resolve`` is a syscall. The callers resolve their roots once, which
    is what makes the comparison valid without paying for it per file.
    """
    try:
        path.relative_to(folder)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------
# Planning


def _renamed(facts: FileFacts, short: str, value: Optional[str]) -> str:
    """This file's basename with ``short`` set to ``value``, or removed.

    Built with :func:`bidsmgr.schema.build_basename` rather than by splicing
    the string, so the new entity lands in the position the standard puts it
    in. Splicing is what produces ``sub-01_run-02_task-rest_bold``, which is
    every bit as invalid as leaving the entity out.
    """
    from ..schema import build_basename

    short_to_long = _short_to_long()
    entities = {
        short_to_long[key]: val
        for key, val in facts.entities.items()
        if key in short_to_long
    }
    long = short_to_long.get(short)
    if long is None:
        raise RenameError(f"{short!r} is not an entity in this schema")
    if value is None:
        entities.pop(long, None)
    else:
        entities[long] = value
    return build_basename(entities, facts.datatype, facts.suffix,
                          facts.extension)


def _destination(root: Path, facts: FileFacts, short: str,
                 value: Optional[str], name: str) -> Path:
    """Where the renamed file lands, folder included.

    Only a folder entity moves the file to a different directory, and which
    entities those are comes from the schema. Setting one means the file goes
    into that folder; clearing one means it comes back out of it.
    """
    from .rename import folder_entities

    parent = facts.path.parent
    if short not in folder_entities():
        return parent / name

    token = f"{short}-"
    parts = list(parent.relative_to(root).parts)
    kept = [p for p in parts if not p.startswith(token)]
    if value is not None:
        # Directly under the level that owns it. For ``ses`` that is the
        # subject, which is where BIDS puts a session and the only place a
        # datatype folder may be found beneath one.
        insert_at = 1 if kept and kept[0].startswith("sub-") else 0
        kept.insert(insert_at, f"{short}-{value}")
    return root.joinpath(*kept) / name


def plan_entity_edit(
    root: Path,
    entity: str,
    targets: Iterable[Path],
    *,
    value: Optional[str] = None,
) -> RenamePlan:
    """Plan adding, changing or removing one entity across ``targets``.

    ``value`` is the label to set; ``None`` removes the entity. Nothing is
    touched on disk: the result is a plan, for the same reason the rename
    dialog shows one first.

    Raises :class:`RenameError` for a request the standard cannot honour: a
    value the entity's format rejects, a required entity being removed, or an
    entity the selected files are not allowed to carry.
    """
    from ..schema import allowed_entities, required_entities
    from .rename import _label_pattern

    # Left exactly as the caller spelled it. ``apply_rename`` is handed this
    # same root, and a plan whose paths are spelled differently from the root
    # the applier holds is the defect ``_under_root`` documents.
    root = Path(root)
    entity = entity.strip()
    if entity == "sub":
        raise RenameError(
            "every BIDS file has a subject and none may gain or lose one. "
            "Use Rename to move a recording to a different subject, so its "
            "participants.tsv row travels with it."
        )
    if value is not None:
        value = value.strip()
        if not value:
            raise RenameError("a value is required to add an entity")
        if not _label_pattern(entity).match(value):
            from .rename import _bad_label_message

            raise RenameError(_bad_label_message(entity, value))

    files = expand(root, targets)
    facts = _typed(root, files)
    if not facts:
        raise RenameError(
            "none of the selected files sit in a datatype folder with a "
            "recognised suffix, so the schema cannot say which entities they "
            "are allowed to carry."
        )

    long = _short_to_long().get(entity, entity)
    # A file the edit cannot apply to is LEFT ALONE, not made to refuse the
    # whole operation. Selecting a study and asking for the acquisition label
    # to go should take it off the files that have one and can lose it; with
    # an all-or-nothing rule the answer was almost always no, because one
    # _bold somewhere requires task and one file somewhere carries no acq.
    # The preview then lists exactly what moves, so what was skipped is
    # visible before anything happens rather than asserted in a message.
    skipped: list[str] = []
    for item in list(facts):
        allowed = set(allowed_entities(item.datatype, item.suffix))
        if not allowed:
            # The schema has no rule for this datatype and suffix together, so
            # there is nothing to check against and nothing is asserted. It
            # still MOVES: it reached this list by carrying exactly the same
            # entities as a recording that is moving, which is what makes it
            # that recording's companion, and a companion left behind while
            # its recording is renamed is an orphan no validator can attribute
            # to anything.
            continue
        if value is not None:
            if long not in allowed:
                skipped.append(item.rel)
                facts.remove(item)
        elif entity in item.entities:
            if long in set(required_entities(item.datatype, item.suffix)):
                skipped.append(item.rel)
                facts.remove(item)

    if not facts:
        what = "take" if value is not None else "do without"
        raise RenameError(
            f"none of the {len(skipped)} selected recordings can {what} "
            f"{entity}-, so there is nothing to do. "
            + (f"{entity}- is required for {skipped[0]} and the rest."
               if value is None else
               f"{skipped[0]} and the rest may not carry it.")
        )

    # Count what will actually CHANGE, not what survived the filter. With a
    # whole study selected, "Remove acq- from 84 files" when 24 carry one is
    # a title that contradicts the preview underneath it.
    if value is None:
        touched = sum(1 for item in facts if entity in item.entities)
    else:
        touched = sum(1 for item in facts
                      if item.entities.get(entity) != value)

    plan = RenamePlan(
        entity=entity,
        old="",
        new=(value or ""),
        ref_by_path=True,
        standard_scans_home=True,
        title=(
            f"Remove {entity}- from {touched} file(s)" if value is None
            else f"Set {entity}-{value} on {touched} file(s)"
        ),
    )

    taken: dict[Path, Path] = {}
    for item in facts:
        if value is None and entity not in item.entities:
            continue                         # nothing to take away
        if value is not None and item.entities.get(entity) == value:
            continue                         # already says that
        try:
            name = _renamed(item, entity, value)
        except (KeyError, ValueError) as exc:
            plan.conflicts.append(f"{item.rel}: {exc}")
            continue
        target = _destination(root, item, entity, value, name)
        if target == item.path:
            continue
        if target in taken:
            plan.conflicts.append(
                f"{_rel(root, item.path)} and {_rel(root, taken[target])} "
                f"would both become {_rel(root, target)}"
            )
            continue
        if target.exists():
            plan.conflicts.append(f"{_rel(root, target)} already exists")
            continue
        taken[target] = item.path
        plan.file_moves.append((item.path, target))

    plan.emptied = _vacated(root, plan.file_moves)
    plan.content_edits = _plan_ref_edits(root, plan.file_moves)
    return plan


def _vacated(root: Path, moves: list[tuple[Path, Path]]) -> list[Path]:
    """Directories this plan could leave behind, deepest first.

    ``apply_rename`` already removes the folder each moved file came out of.
    What it cannot see is the level ABOVE that: taking the last datatype out of
    ``sub-01/ses-01`` empties the session folder too, and a session folder with
    nothing in it is not untidy, it is a session the dataset claims to have.
    """
    out: list[Path] = []
    for src, _dst in moves:
        folder = src.parent
        while folder != root and _within(root, folder):
            out.append(folder)
            folder = folder.parent
    return sorted(
        dict.fromkeys(out), key=lambda p: len(p.parts), reverse=True,
    )


def _plan_ref_edits(
    root: Path, moves: list[tuple[Path, Path]],
) -> list[ContentEdit]:
    """Which files MENTION something that is about to move, and how often.

    Counted against the same map the applier will use, so the preview cannot
    promise an edit that will not happen or stay silent about one that will.
    """
    if not moves:
        return []
    ref_map = build_ref_map(root, moves)
    out: list[ContentEdit] = []

    for path in walk_dataset(root):
        if path.suffix == ".json":
            data = _load_json(path)
            if not data:
                continue
            # Every field that points at another file, not just IntendedFor.
            # A recording moving into a session has to be followed by the MEG
            # sidecar naming it as an empty room just as much as by a
            # fieldmap. See rename.PATH_FIELDS.
            for field in PATH_FIELDS:
                if field not in data:
                    continue
                value = data[field]
                items = value if isinstance(value, list) else [value]
                hits = sum(
                    1 for v in items
                    if _lookup(str(v), ref_map) is not None
                )
                if hits:
                    out.append(ContentEdit(path, _rel(root, path),
                                           field, hits))
        elif path.name.endswith("_scans.tsv"):
            hits = _count_rows(path, "filename", ref_map)
            if hits:
                out.append(ContentEdit(path, _rel(root, path),
                                       "scans.tsv filename", hits))
    return out


def _lookup(text: str, ref_map: dict[str, str]) -> Optional[str]:
    from .rename import _split_uri

    return ref_map.get(_split_uri(text)[1])


def _count_rows(path: Path, column: str, ref_map: dict[str, str]) -> int:
    from .tsv_edit import read_table

    table = read_table(path)
    if table is None or column not in table.header:
        return 0
    return sum(1 for v in table.column(column)
               if _lookup(v, ref_map) is not None)


# --------------------------------------------------------------------------
# Sessions, which are the above with a folder


def sessions_in(root: Path) -> list[str]:
    """Every session label the dataset uses, sorted."""
    from .rename import list_values

    return list_values(Path(root), "ses")


def session_scope(root: Path, path: Path) -> list[Path]:
    """The files a session operation on ``path`` should cover.

    Clicking a subject means the whole subject; clicking a datatype folder
    means that datatype; clicking a file means its recording. The distinction
    matters because a session is a property of a GROUP of recordings, and the
    natural unit is whatever the user pointed at.
    """
    return expand(Path(root), [Path(path)])


def plan_add_session(
    root: Path, label: str, targets: Iterable[Path],
) -> RenamePlan:
    """Move the selection into ``ses-<label>``, creating the folder."""
    plan = plan_entity_edit(root, "ses", targets, value=label)
    plan.title = f"Create session ses-{label}"
    return plan


def plan_remove_session(root: Path, targets: Iterable[Path]) -> RenamePlan:
    """Take the selection back out of its session folder.

    The folder goes once it is empty, its ``*_scans.tsv`` rows are folded up
    into the subject's table, and any ``IntendedFor`` that reached into the
    session follows. Removing a session from SOME of a subject's recordings
    leaves the session in place holding the rest, which is what a partial
    selection has to mean.
    """
    plan = plan_entity_edit(root, "ses", targets, value=None)
    plan.title = "Remove the session"
    return plan


__all__ = [
    "EntitySlot",
    "FileFacts",
    "addable_entities",
    "companions",
    "describe_file",
    "expand",
    "plan_add_session",
    "plan_entity_edit",
    "plan_remove_session",
    "removable_entities",
    "session_scope",
    "sessions_in",
]

"""Whether a dataset's files still agree with each other.

The validator answers "is this legal BIDS". It does not answer "do these
files still describe the same dataset", and that second question is the one
that fails after somebody renames a folder in Finder, deletes a subject by
hand, or copies a session in from somewhere else. Every symptom below is
legal BIDS and wrong:

* a ``*_scans.tsv`` row naming a recording that is not there any more, or a
  recording with no row in the table that is supposed to list it
* a ``participants.tsv`` row for a subject folder that was deleted, or a
  subject folder with no row
* ``IntendedFor`` and the four other pointer fields aimed at nothing
* a sidecar whose recording is gone
* ``dataset_description.json`` and ``CITATION.cff`` disagreeing about who
  the authors are

None of that is invented here. Each check is a question something in the
tool could already answer, asked as a read-only pass and reported in one
place instead of being a side effect of a conversion.

**Read-only until asked.** Nothing in this module writes. :func:`check`
returns findings, each carrying a repair the caller may choose to apply, and
:func:`apply` runs the chosen ones through one operation so the whole pass
is a single undo. A tool that silently rewrote ``participants.tsv`` when a
dataset was opened would be one nobody could trust, which is the same reason
``run_dataset_fixups`` defaults everything off.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field as dc_field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional

from .. import schema as schema_mod
from ..project.operations import begin_operation
from . import linkage
from .rename import _skip, scans_home, scans_table_in, walk_dataset

log = logging.getLogger(__name__)


class Kind(str, Enum):
    """What sort of disagreement this is. Groups the report."""

    ENTITY_WIDTH_SPLIT = "one entity written at two widths"
    ENTITY_CASE_SPLIT = "one value written in two letter cases"
    TASK_NAME_MISMATCH = "TaskName does not match the task entity"
    INTENDED_FOR_DIFFERS = "IntendedFor differs from what the times imply"
    SESSIONS_UNEVEN = "some subjects use sessions and some do not"
    SCANS_ROW_ORPHAN = "scans row names a missing file"
    SCANS_ROW_MISSING = "recording has no scans row"
    PARTICIPANT_ORPHAN = "participants row names a missing subject"
    PARTICIPANT_MISSING = "subject has no participants row"
    LINK_BROKEN = "link points at a missing file"
    SIDECAR_ORPHAN = "sidecar has no recording"
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
# The checks


def _recordings(root: Path) -> list[Path]:
    """Every file a ``*_scans.tsv`` is supposed to list.

    "Each neural recording file", per the standard, which is what
    ``metadata.engine`` decided too. A folder-shaped recording (CTF ``.ds``,
    EGI ``.mff``) is ONE of these and is not descended into.
    """
    from ..metadata.engine import _is_recording_file, _scans_rows_for  # noqa: F401

    out: list[Path] = []
    known = set(schema_mod.list_datatypes())
    for subject in sorted(root.glob("sub-*")):
        if not subject.is_dir():
            continue
        for datatype_dir in sorted(subject.rglob("*")):
            if not datatype_dir.is_dir() or datatype_dir.name not in known:
                continue
            for entry in sorted(datatype_dir.iterdir()):
                if _is_recording_file(entry) and _is_bids_name(entry):
                    out.append(entry)
    return out


def _is_bids_name(path: Path) -> bool:
    """Second gate: does the standard actually accept this filename?

    Belt and braces on purpose. A repair WRITES into the dataset, and the
    first version of this trusted one predicate and offered to add a
    ``.DS_Store`` row to ``*_scans.tsv``. A check that costs a schema
    lookup is worth having in front of every write, so that a change to
    the predicate can never again turn junk into data.
    """
    datatype = path.parent.name
    if datatype not in set(schema_mod.list_datatypes()):
        return False
    try:
        # The folder-shaped recordings are missing from the validator's
        # default extension list, because it was written for files. A CTF
        # ``.ds`` and an EGI ``.mff`` are one recording each and have to be
        # accepted, or the scans-row repair would never offer them.
        verdicts = schema_mod.validate_basename(
            path.name, datatype,
            extensions=schema_mod.validate_basename.__defaults__[0]
            + (".ds", ".mff"),
        )
    except Exception:  # noqa: BLE001 - an unparseable name is not BIDS
        return False
    # A list of verdicts, one per rule. Anything at ERROR means the name is
    # not one the standard accepts here.
    return not any(
        getattr(v, "severity", None) is schema_mod.Severity.ERROR
        for v in verdicts
    )


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh, delimiter="\t"))
    except OSError:
        return [], []
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _write_tsv(op, path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = ["\t".join(header)]
    lines += ["\t".join(r) for r in rows]
    op.write_text(path, "\n".join(lines) + "\n")


def check_scans_tables(root: Path) -> list[Finding]:
    """Rows naming files that are gone, and files with no row."""
    out: list[Finding] = []
    listed: set[Path] = set()

    # ``walk_dataset`` rather than ``rglob``: an operation keeps a copy of
    # every file it changes under ``.bidsmgr/editor/originals/``, and those
    # copies are scans tables too. Walking into them reported the BACKUP's
    # rows as orphans, because they are relative to a folder that holds no
    # data, and a coherence pass that flags its own undo history is worse
    # than useless.
    for table in sorted(p for p in walk_dataset(root)
                        if p.name.endswith("_scans.tsv")):
        header, rows = _read_tsv(table)
        if "filename" not in header:
            continue
        col = header.index("filename")
        home = table.parent
        keep: list[list[str]] = []
        gone: list[str] = []
        for row in rows:
            if col >= len(row):
                continue
            named = (home / row[col]).resolve()
            if named.exists():
                listed.add(named)
                keep.append(row)
            else:
                gone.append(row[col])
        if gone:
            out.append(Finding(
                kind=Kind.SCANS_ROW_ORPHAN,
                path=table,
                detail=(
                    f"{len(gone)} row(s) name a file that is not there: "
                    f"{', '.join(gone[:3])}{', ...' if len(gone) > 3 else ''}"
                ),
                repair=f"drop {len(gone)} row(s)",
                _apply=lambda op, t=table, h=header, k=keep: _write_tsv(op, t, h, k),
            ))

    for recording in _recordings(root):
        if recording.resolve() in listed:
            continue
        home = scans_home(recording, root)
        if home is None:
            continue
        table = scans_table_in(home)
        if not table.is_file():
            # No table at all is not an inconsistency. ``*_scans.tsv`` is
            # RECOMMENDED, not required, and a dataset that does not use
            # them is coherent: reporting one finding per recording would
            # bury the real disagreements under a list of every file.
            continue
        out.append(Finding(
            kind=Kind.SCANS_ROW_MISSING,
            path=table,
            detail=f"{recording.name} is not listed in {table.name}",
            repair="add the row",
            _apply=lambda op, t=table, r=recording, h=home: _add_scans_row(op, t, r, h),
        ))
    return out


def _add_scans_row(op, table: Path, recording: Path, home: Path) -> None:
    header, rows = _read_tsv(table)
    if not header:
        header = ["filename"]
    if "filename" not in header:
        header = ["filename"] + header
    col = header.index("filename")
    rel = recording.resolve().relative_to(home.resolve()).as_posix()
    row = ["n/a"] * len(header)
    row[col] = rel
    rows.append(row)
    rows.sort(key=lambda r: r[col] if col < len(r) else "")
    _write_tsv(op, table, header, rows)


def check_participants(root: Path) -> list[Finding]:
    """Rows for subjects that are gone, and subjects with no row."""
    table = root / "participants.tsv"
    on_disk = {
        p.name for p in root.glob("sub-*")
        if p.is_dir() and not _skip(p, root)
    }
    if not table.is_file():
        if on_disk:
            return [Finding(
                kind=Kind.PARTICIPANT_MISSING,
                path=table,
                detail=f"{len(on_disk)} subject folder(s) and no participants.tsv",
                repair=None,
            )]
        return []

    header, rows = _read_tsv(table)
    if "participant_id" not in header:
        return []
    col = header.index("participant_id")
    listed = {r[col] for r in rows if col < len(r)}

    out: list[Finding] = []
    orphans = sorted(listed - on_disk)
    if orphans:
        keep = [r for r in rows if col >= len(r) or r[col] in on_disk]
        out.append(Finding(
            kind=Kind.PARTICIPANT_ORPHAN,
            path=table,
            detail=(
                f"{len(orphans)} row(s) name a subject folder that is not "
                f"there: {', '.join(orphans[:3])}"
                f"{', ...' if len(orphans) > 3 else ''}"
            ),
            repair=f"drop {len(orphans)} row(s)",
            _apply=lambda op, h=header, k=keep: _write_tsv(op, table, h, k),
        ))

    for missing in sorted(on_disk - listed):
        row = ["n/a"] * len(header)
        row[col] = missing
        out.append(Finding(
            kind=Kind.PARTICIPANT_MISSING,
            path=table,
            detail=f"{missing} has a folder but no row",
            repair="add the row",
            _apply=lambda op, r=row: _append_row(op, table, r),
        ))
    return out


def _append_row(op, table: Path, row: list[str]) -> None:
    header, rows = _read_tsv(table)
    rows.append(row)
    rows.sort(key=lambda r: r[0] if r else "")
    _write_tsv(op, table, header, rows)


def check_links(root: Path) -> list[Finding]:
    """Pointer fields aimed at files that are not there."""
    out: list[Finding] = []
    for broken in linkage.broken_links(root):
        out.append(Finding(
            kind=Kind.LINK_BROKEN,
            path=broken.source,
            detail=f"{broken.field} points at {broken.target}, which is missing",
            repair="remove that entry",
            _apply=lambda op, b=broken: _drop_link(op, b),
        ))
    return out


def _drop_link(op, broken: linkage.BrokenLink) -> None:
    data = json.loads(broken.source.read_text(encoding="utf-8"))
    raw = data.get(broken.field)
    values = raw if isinstance(raw, list) else [raw]
    kept = [v for v in values if str(v) != broken.target]
    if kept:
        data[broken.field] = kept if isinstance(raw, list) else kept[0]
    else:
        # An empty list claims "points at nothing", which is a different and
        # wronger statement than not saying.
        data.pop(broken.field, None)
    op.write_json(broken.source, data)


def check_orphan_sidecars(root: Path) -> list[Finding]:
    """A ``.json`` whose recording is gone.

    Only a sidecar that names a specific file counts. One that sits higher up
    the tree describes every recording below it by inheritance, so it has no
    single partner to be missing.
    """
    out: list[Finding] = []
    known = set(schema_mod.list_datatypes())
    for path in walk_dataset(root):
        if not path.name.endswith(".json"):
            continue
        if path.parent.name not in known:
            continue          # inherited, not a partner
        stem = path.name[: -len(".json")]
        siblings = [
            p for p in path.parent.iterdir()
            if p.name != path.name and p.name.startswith(stem + ".")
        ]
        if siblings:
            continue
        out.append(Finding(
            kind=Kind.SIDECAR_ORPHAN,
            path=path,
            detail=f"{path.name} describes a file that is not there",
            repair="delete the sidecar",
            _apply=lambda op, p=path: op.delete(p),
        ))
    return out


def check_citation(root: Path) -> list[Finding]:
    """``CITATION.cff`` and ``dataset_description.json`` out of step."""
    from . import cff

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
    diverged = [
        key for key in ("title", "authors", "license")
        if key in expected and have.get(key) != expected.get(key)
    ]
    if not diverged:
        return []
    return [Finding(
        kind=Kind.CITATION_DIVERGED,
        path=citation,
        detail=(
            f"{', '.join(diverged)} differ(s) from dataset_description.json"
        ),
        repair="rewrite CITATION.cff from the description",
        _apply=lambda op, c=citation, e={**have, **expected}: op.write_text(
            c, cff.dumps(e)
        ),
    )]


def check_entity_widths(root: Path) -> list[Finding]:
    """An index entity written at two widths in one dataset.

    ``run-1`` beside ``run-01`` is legal BIDS, which is why no validator
    mentions it, and it is still what makes a dataset sort wrongly and a
    glob miss half its files.

    The repair pads everything up to the widest in use, which never loses a
    digit. It is offered here as well as in its own tool because this is
    where somebody looking for what is wrong with a dataset will look.
    """
    from . import values as ev

    out: list[Finding] = []
    for split in ev.inconsistent_widths(root):
        width = split.suggested
        detail = (
            f"{split.entity} is written at {len(split.widths)} widths "
            f"({', '.join(split.examples)}) across {split.files} file(s)"
        )
        try:
            plan = ev.plan_padding(root, split.entity, width)
        except Exception as exc:  # noqa: BLE001 - say WHY it cannot be padded
            # The interesting case, not a failure: the dataset holds both
            # spellings of ONE number, so padding would fuse two runs. That
            # is a decision for a person, and a finding that says "no
            # repair" without saying why is one nobody can act on.
            out.append(Finding(
                kind=Kind.ENTITY_WIDTH_SPLIT,
                path=Path(root),
                detail=f"{detail}. {exc}",
                repair=None,
            ))
            continue
        if not plan:
            continue
        out.append(Finding(
            kind=Kind.ENTITY_WIDTH_SPLIT,
            path=Path(root),
            detail=detail,
            repair=f"pad every {split.entity} to {width} digits",
            changes=tuple(
                (f"{split.entity}-{r.old}", f"{split.entity}-{r.new}",
                 r.files) for r in plan
            ),
            _standalone=lambda e=split.entity, w=width: _pad_entity(root, e, w),
        ))
    return out


def _pad_entity(root: Path, entity: str, width: int) -> None:
    """Repad every value of ``entity``, one rename per value.

    Each ``apply_rename`` opens its own operation, which is why this is a
    STANDALONE repair rather than one that runs inside the batch: nesting
    an operation inside another would make the outer undo restore a
    half-repadded tree. The cost is that padding three values is three undo
    steps rather than one, which is the honest shape of it.
    """
    from . import values as ev
    from .rename import apply_rename, plan_rename

    for repad in ev.plan_padding(root, entity, width):
        apply_rename(root, plan_rename(root, entity, repad.old, repad.new))


def check_entity_case(root: Path) -> list[Finding]:
    """Two spellings of one value that differ only in case.

    ``task-Rest`` and ``task-rest`` are two different tasks to BIDS and to
    every pipeline that reads it, and are almost never meant to be.
    """
    from .rename import entity_value, walk_dataset

    out: list[Finding] = []
    for entity in schema_mod.entity_keys():
        seen: dict[str, set[str]] = {}
        for path in walk_dataset(root):
            value = entity_value(path.name, entity)
            if value:
                seen.setdefault(value.lower(), set()).add(value)
        for lowered, spellings in seen.items():
            if len(spellings) < 2:
                continue
            out.append(Finding(
                kind=Kind.ENTITY_CASE_SPLIT,
                path=Path(root),
                detail=(
                    f"{entity} is spelled "
                    f"{' and '.join(sorted(f'{entity}-{s}' for s in spellings))}"
                    ", which BIDS reads as different values"
                ),
                repair=None,   # which spelling is right is not ours to decide
            ))
    return out


def check_task_names(root: Path) -> list[Finding]:
    """``TaskName`` in a sidecar that does not derive to the task entity.

    The standard defines the relationship: the label is the name with
    everything but letters and digits removed. A sidecar saying
    ``"TaskName": "rest"`` on a ``task-nback`` file is one of the two
    disagreeing, and no validator checks it.
    """
    from .rename import _derived_label, entity_value, walk_dataset

    out: list[Finding] = []
    for path in walk_dataset(root):
        if not path.name.endswith(".json"):
            continue
        label = entity_value(path.name, "task")
        if not label:
            continue
        data = _load_json(path)
        name = data.get("TaskName")
        if not isinstance(name, str) or not name:
            continue
        if _derived_label(name) == label:
            continue
        out.append(Finding(
            kind=Kind.TASK_NAME_MISMATCH,
            path=path,
            detail=(
                f'TaskName is "{name}", which does not derive to the '
                f"task-{label} in the filename"
            ),
            repair=f'set TaskName so it derives to "{label}"',
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
    data[field] = value
    op.write_json(path, data)


def check_intended_for(root: Path) -> list[Finding]:
    """``IntendedFor`` that differs from what the acquisition times imply.

    Not an error: a protocol may genuinely want one fieldmap applied to
    everything, and a user may have set it deliberately. It is reported
    because the usual cause is that the times were unreadable when the
    conversion ran, and the difference is invisible otherwise.
    """
    from ..fixups.intended_for import suggest_intended_for

    out: list[Finding] = []
    for subject in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        sessions = [p for p in subject.glob("ses-*") if p.is_dir()]
        for scope in (sessions or [subject]):
            label = subject.name[len("sub-"):]
            ses = scope.name[len("ses-"):] if scope is not subject else None
            pairs, why = suggest_intended_for(scope, label, ses)
            for members, uris in pairs:
                for sidecar in members:
                    have = [
                        str(v) for v in (_load_json(sidecar).get("IntendedFor") or [])
                    ]
                    if sorted(have) == sorted(uris):
                        continue
                    # Only the DIFFERENCE. Listing every current entry as a
                    # removal and every suggested one as an addition made a
                    # one-file difference read as four changes.
                    extra = [u for u in have if u not in uris]
                    missing = [u for u in uris if u not in have]
                    out.append(Finding(
                        kind=Kind.INTENDED_FOR_DIFFERS,
                        path=sidecar,
                        detail=(
                            f"{len(extra)} entry(ies) the times do not imply, "
                            f"{len(missing)} the times imply and it does not "
                            f"have. {why}"
                        ),
                        repair="use what the times imply",
                        changes=tuple(
                            (u.rsplit("/", 1)[-1], "", 1) for u in extra
                        ) + tuple(
                            ("", u.rsplit("/", 1)[-1], 1) for u in missing
                        ),
                        _apply=lambda op, p=sidecar, u=list(uris): _set_field(
                            op, p, "IntendedFor", u
                        ),
                    ))
    return out


def check_sessions_even(root: Path) -> list[Finding]:
    """Some subjects in session folders and some not.

    Legal, and almost always a conversion that treated two subjects
    differently rather than a study that genuinely measured one of them
    once. Reported without a repair: moving a subject into or out of a
    session is the Sessions tool's job and needs a label chosen by a person.
    """
    with_ses, without = [], []
    for subject in sorted(p for p in root.glob("sub-*") if p.is_dir()):
        if any(p.is_dir() for p in subject.glob("ses-*")):
            with_ses.append(subject.name)
        else:
            without.append(subject.name)
    if not (with_ses and without):
        return []
    return [Finding(
        kind=Kind.SESSIONS_UNEVEN,
        path=Path(root),
        detail=(
            f"{len(with_ses)} subject(s) use session folders and "
            f"{len(without)} do not ({', '.join(without[:3])}"
            f"{', ...' if len(without) > 3 else ''})"
        ),
        repair=None,
    )]


CHECKS: tuple[Callable[[Path], list[Finding]], ...] = (
    check_entity_widths,
    check_entity_case,
    check_task_names,
    check_intended_for,
    check_sessions_even,
    check_scans_tables,
    check_participants,
    check_links,
    check_orphan_sidecars,
    check_citation,
)


def check(root: Path) -> list[Finding]:
    """Run every check. Read-only.

    A check that raises is reported and does not stop the others: a coherence
    pass that gives up on the first malformed table is no use on exactly the
    dataset it was written for.
    """
    root = Path(root)
    out: list[Finding] = []
    for fn in CHECKS:
        try:
            out.extend(fn(root))
        except Exception as exc:  # noqa: BLE001 - one bad check is not all of them
            log.warning("coherence check %s failed: %s", fn.__name__, exc)
    return out


def apply(root: Path, findings: Iterable[Finding], *,
          label: Optional[str] = None) -> int:
    """Apply the chosen repairs. Returns how many ran.

    One operation for all of them, so the pass is one entry in the history.
    """
    chosen = [f for f in findings if f.fixable]
    if not chosen:
        return 0

    batch = [f for f in chosen if f._apply is not None]
    standalone = [f for f in chosen if f._apply is None and f._standalone]

    done = 0
    if batch:
        with begin_operation(
            Path(root), label or f"Repair {len(batch)} inconsistency(ies)"
        ) as op:
            for finding in batch:
                finding._apply(op)
                done += 1
    # After the batch, and outside it. Each of these opens its own
    # operation, so it is its own undo step; running them inside the batch
    # would nest operations and break the outer one's rollback.
    for finding in standalone:
        finding._standalone()
        done += 1
    return done


__all__ = ["CHECKS", "Finding", "Kind", "apply", "check"]

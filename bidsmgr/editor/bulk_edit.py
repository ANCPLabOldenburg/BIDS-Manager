"""Write one answer into every file it belongs in, and no others.

Two requests share this engine. Stating a field once for a whole study, and
fixing a finding in every file it fired on, are the same operation seen from
two directions: work out which files the field applies to, show the user what
each one says now and what it would say, and write only the ones they tick.

Three rules the engine keeps:

**Applicability comes from the schema, not from a filename pattern.** A file is
a candidate for ``EchoTime`` because the standard declares ``EchoTime`` for its
datatype and suffix. Globbing ``*_bold.json`` would sweep in a derivative that
should not have it and miss a suffix that should.

**Nothing is written that is already right.** A file whose value already equals
the new one is offered but marked unchanged, so a count of "12 files" means
twelve files that will actually differ.

**Nothing is written without the caller seeing it first.** :func:`candidates`
answers what would happen; :func:`apply_value` does it. They are separate calls
so a dialog can sit between them.

Qt-free.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

# How wide to cast the net, from an anchor file.
SCOPE_SAME_KIND = "same_kind"      # same datatype + suffix
SCOPE_SAME_TASK = "same_task"      # same datatype + suffix + task entity
SCOPE_SAME_SUBJECT = "same_subject"
SCOPE_DATASET = "dataset"          # every file the field applies to
SCOPES = (SCOPE_SAME_KIND, SCOPE_SAME_TASK, SCOPE_SAME_SUBJECT, SCOPE_DATASET)

SCOPE_LABELS = {
    SCOPE_SAME_KIND: "Same datatype and suffix",
    SCOPE_SAME_TASK: "Same task",
    SCOPE_SAME_SUBJECT: "Same subject",
    SCOPE_DATASET: "Anywhere in the dataset",
}


@dataclass
class FileCandidate:
    """One file a bulk edit could touch, and what it would do to it."""

    path: Path                  # absolute
    rel: str                    # relative to the dataset root, for display
    datatype: Optional[str]
    suffix: Optional[str]
    current: Any = None
    present: bool = False
    applicable: bool = True
    reason: str = ""            # why not, when ``applicable`` is False

    def would_change(self, new_value: Any) -> bool:
        return self.applicable and self.current != new_value

    def current_text(self) -> str:
        if not self.present:
            return "(not set)"
        try:
            return json.dumps(self.current, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(self.current)


@dataclass
class BulkResult:
    """What :func:`apply_value` did."""

    written: list[Path]
    skipped: list[Path]
    failed: list[tuple[Path, str]]

    @property
    def ok(self) -> bool:
        return not self.failed


# --------------------------------------------------------------------------
# Finding the files


def _entities(name: str) -> dict[str, str]:
    """Parse ``sub-01_ses-2_task-rest_bold.json`` into its entity pairs."""
    stem = name
    for ext in (".json", ".tsv", ".nii.gz", ".nii", ".tsv.gz"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    out: dict[str, str] = {}
    for part in stem.split("_"):
        if "-" in part:
            key, _, val = part.partition("-")
            out[key] = val
    return out


def _suffix_of(name: str) -> Optional[str]:
    """The BIDS suffix: the last underscore-separated part with no ``-``."""
    stem = name
    for ext in (".json", ".tsv", ".nii.gz", ".nii", ".tsv.gz"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    parts = stem.split("_")
    if not parts:
        return None
    last = parts[-1]
    return None if "-" in last else last


def _datatype_of(path: Path, root: Path) -> Optional[str]:
    """The datatype folder a file sits in, if it sits in one."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    # <sub>/<datatype>/<file> or <sub>/<ses>/<datatype>/<file>
    if len(parts) >= 2:
        return parts[-2]
    return None


def _known_datatypes() -> frozenset[str]:
    """The datatype folder names the standard defines, from the schema."""
    try:
        from .. import schema as schema_mod

        return frozenset(str(d) for d in schema_mod.list_datatypes())
    except Exception:  # noqa: BLE001 - never let this break a dialog
        return frozenset()


def _declares(datatype: Optional[str], suffix: Optional[str], field: str) -> bool:
    """Does the standard declare ``field`` for this kind of file?

    Answered through the schema adapter, so it follows the BIDS version the
    user selected rather than a list kept here.

    A file that is not inside a datatype folder is NOT applicable. Those are
    the dataset-level files (``dataset_description.json``,
    ``participants.json``) and they have their own schemas; letting a bulk
    edit reach them would write an MRI acquisition field into the dataset
    description.
    """
    if not datatype or datatype not in _known_datatypes():
        return False
    if not suffix:
        return True
    try:
        from ..metadata.template_plan import sidecar_section

        section = sidecar_section(datatype, suffix, include_derived=True)
    except Exception:  # noqa: BLE001 - schema lookup is best-effort here
        return True
    names = {f.name for f in section.fields} | set(
        getattr(section, "supplied", ()) or ()
    )
    declared = getattr(section, "declared", ()) or ()
    names |= {getattr(f, "name", "") for f in declared}
    return field in names


def _read_json(path: Path) -> Optional[OrderedDict]:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh, object_pairs_hook=OrderedDict)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, OrderedDict) else None


def candidates(
    root: Path,
    field: str,
    *,
    anchor: Optional[Path] = None,
    scope: str = SCOPE_SAME_KIND,
    paths: Optional[Iterable[Path]] = None,
) -> list[FileCandidate]:
    """Every JSON sidecar the edit could touch, with what it says now.

    ``paths`` short-circuits the search: a caller that already knows the files
    (a grouped finding does) passes them and gets them back annotated.
    Otherwise the dataset is walked and filtered by ``scope`` relative to
    ``anchor``.
    """
    root = Path(root)
    if paths is not None:
        found = [Path(p) for p in paths]
    else:
        found = sorted(
            p for p in root.rglob("*.json")
            if ".bidsmgr" not in p.parts and not p.name.startswith(".")
        )
        if anchor is not None:
            a_ent = _entities(anchor.name)
            a_suffix = _suffix_of(anchor.name)
            a_datatype = _datatype_of(anchor, root)

            def _keep(p: Path) -> bool:
                if scope == SCOPE_DATASET:
                    return True
                if _suffix_of(p.name) != a_suffix:
                    return False
                if _datatype_of(p, root) != a_datatype:
                    return False
                ent = _entities(p.name)
                if scope == SCOPE_SAME_TASK:
                    return ent.get("task") == a_ent.get("task")
                if scope == SCOPE_SAME_SUBJECT:
                    return ent.get("sub") == a_ent.get("sub")
                return True

            found = [p for p in found if _keep(p)]

    out: list[FileCandidate] = []
    for p in found:
        # A finding on a data file points at the file, not its sidecar. The
        # thing to edit is the sidecar, so follow it there.
        target = p if p.suffix == ".json" else _sidecar_for(p)
        if target is None:
            continue
        datatype = _datatype_of(target, root)
        suffix = _suffix_of(target.name)
        data = _read_json(target)
        applicable = _declares(datatype, suffix, field)
        if applicable:
            reason = ""
        elif not datatype or datatype not in _known_datatypes():
            reason = "not inside a datatype folder, so it is dataset-level metadata"
        else:
            reason = (
                f"the standard does not declare {field} for {datatype}/{suffix}"
            )
        if data is None and target.exists():
            applicable, reason = False, "not readable as a JSON object"
        try:
            # POSIX separators on every platform: this is matched against
            # BIDS-declared paths and shown to a user who reads them with
            # slashes. See editor/rename.py::_rel.
            rel = target.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = Path(target).as_posix()
        out.append(FileCandidate(
            path=target,
            rel=rel,
            datatype=datatype,
            suffix=suffix,
            current=(data or {}).get(field),
            present=bool(data is not None and field in data),
            applicable=applicable,
            reason=reason,
        ))
    # Stable, and de-duplicated: two data files can share one sidecar.
    seen: set[str] = set()
    unique: list[FileCandidate] = []
    for c in sorted(out, key=lambda c: c.rel):
        if c.rel in seen:
            continue
        seen.add(c.rel)
        unique.append(c)
    return unique


def _sidecar_for(path: Path) -> Optional[Path]:
    """The ``.json`` that belongs to a data file."""
    name = path.name
    for ext in (".nii.gz", ".tsv.gz"):
        if name.endswith(ext):
            return path.with_name(name[: -len(ext)] + ".json")
    return path.with_suffix(".json")


# --------------------------------------------------------------------------
# Doing it


def apply_value(
    root: Path,
    selection: list[FileCandidate],
    field: str,
    value: Any,
    *,
    label: Optional[str] = None,
) -> BulkResult:
    """Write ``field = value`` into every selected file, reversibly.

    One operation for the whole batch, so undo puts every file back at once
    rather than needing as many undos as there were files.
    """
    from ..project.operations import begin_operation

    written: list[Path] = []
    skipped: list[Path] = []
    failed: list[tuple[Path, str]] = []
    targets = [c for c in selection if c.applicable]
    if not targets:
        return BulkResult(written, skipped, failed)

    text = label or f"Set {field} in {len(targets)} file(s)"
    with begin_operation(Path(root), text) as op:
        for cand in targets:
            data = _read_json(cand.path)
            if data is None:
                if cand.path.exists():
                    failed.append((cand.path, "not readable as JSON"))
                    continue
                data = OrderedDict()
            if data.get(field) == value and field in data:
                skipped.append(cand.path)
                continue
            data[field] = value
            try:
                op.write_json(cand.path, data)
            except OSError as exc:
                failed.append((cand.path, str(exc)))
                continue
            written.append(cand.path)
    return BulkResult(written, skipped, failed)


def remove_field(
    root: Path,
    selection: list[FileCandidate],
    field: str,
    *,
    label: Optional[str] = None,
) -> BulkResult:
    """Delete ``field`` from every selected file, reversibly."""
    from ..project.operations import begin_operation

    written: list[Path] = []
    skipped: list[Path] = []
    failed: list[tuple[Path, str]] = []
    text = label or f"Remove {field} from {len(selection)} file(s)"
    with begin_operation(Path(root), text) as op:
        for cand in selection:
            data = _read_json(cand.path)
            if data is None or field not in data:
                skipped.append(cand.path)
                continue
            del data[field]
            try:
                op.write_json(cand.path, data)
            except OSError as exc:
                failed.append((cand.path, str(exc)))
                continue
            written.append(cand.path)
    return BulkResult(written, skipped, failed)


__all__ = [
    "BulkResult",
    "apply_coercions",
    "declared_type",
    "plan_coercions",
    "FileCandidate",
    "SCOPES",
    "SCOPE_DATASET",
    "SCOPE_LABELS",
    "SCOPE_SAME_KIND",
    "SCOPE_SAME_SUBJECT",
    "SCOPE_SAME_TASK",
    "apply_value",
    "candidates",
    "remove_field",
]


# --------------------------------------------------------------------------
# Repairing a value's SHAPE rather than replacing it
#
# The distinction matters and is the rule the whole fix-all feature rests on:
# a repair may change the TYPE of a value the file already states, because the
# standard says what that type must be and the value carries the fact. It may
# never invent the fact itself.


def declared_type(datatype: Optional[str], suffix: Optional[str],
                  field: str) -> tuple[str, str]:
    """``(type, item_type)`` the standard declares, or ``("", "")``."""
    if not datatype or not suffix:
        return ("", "")
    try:
        from .. import schema as schema_mod

        info = schema_mod.field_metadata(field)
    except Exception:  # noqa: BLE001 - unknown field, nothing to say
        return ("", "")
    return (
        str(getattr(info, "type", "") or ""),
        str(getattr(info, "item_type", "") or ""),
    )


def _coerce(value: Any, kind: str, item_kind: str) -> tuple[Any, bool]:
    """Reshape ``value`` to ``kind``. Returns ``(new, changed)``.

    Only conversions that cannot lose information are performed. ``"3"`` to
    ``3`` is safe; ``"about three"`` to anything is not, and is left alone so
    the finding stays visible rather than being papered over with a guess.
    """
    if kind == "array":
        if not isinstance(value, list):
            items = [value]
        else:
            items = list(value)
        if item_kind in ("number", "integer"):
            out: list[Any] = []
            changed = not isinstance(value, list)
            for item in items:
                new, did = _coerce(item, item_kind, "")
                changed = changed or did
                out.append(new)
            return out, changed
        return items, not isinstance(value, list)

    if kind in ("number", "integer") and isinstance(value, str):
        text = value.strip()
        try:
            number = int(text) if kind == "integer" else float(text)
        except (TypeError, ValueError):
            return value, False
        return number, True

    if kind == "string" and not isinstance(value, str) and value is not None:
        if isinstance(value, (dict, list)):
            return value, False          # not a lossless direction
        return str(value), True

    if kind == "boolean" and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "false"):
            return low == "true", True

    return value, False


def plan_coercions(
    root: Path, field: str, *, paths: Optional[Iterable[Path]] = None,
    scope: str = SCOPE_DATASET, anchor: Optional[Path] = None,
) -> list[FileCandidate]:
    """Candidates whose ``field`` is the wrong shape for what BIDS declares.

    ``FileCandidate.current`` is what the file says now and ``reason`` carries
    the repair in words, so the dialog can show both without re-deriving them.
    """
    out: list[FileCandidate] = []
    for cand in candidates(root, field, paths=paths, scope=scope, anchor=anchor):
        if not cand.applicable or not cand.present:
            continue
        kind, item_kind = declared_type(cand.datatype, cand.suffix, field)
        if not kind:
            continue
        new, changed = _coerce(cand.current, kind, item_kind)
        if not changed:
            continue
        cand.reason = (
            f"the standard declares {kind}"
            + (f" of {item_kind}" if item_kind else "")
        )
        out.append(cand)
    return out


def apply_coercions(
    root: Path, selection: list[FileCandidate], field: str,
) -> BulkResult:
    """Rewrite ``field`` in each selected file as the type BIDS declares."""
    from ..project.operations import begin_operation

    written: list[Path] = []
    skipped: list[Path] = []
    failed: list[tuple[Path, str]] = []
    if not selection:
        return BulkResult(written, skipped, failed)
    label = f"Fix the type of {field} in {len(selection)} file(s)"
    with begin_operation(Path(root), label) as op:
        for cand in selection:
            data = _read_json(cand.path)
            if data is None or field not in data:
                skipped.append(cand.path)
                continue
            kind, item_kind = declared_type(cand.datatype, cand.suffix, field)
            new, changed = _coerce(data[field], kind, item_kind)
            if not changed:
                skipped.append(cand.path)
                continue
            data[field] = new
            try:
                op.write_json(cand.path, data)
            except OSError as exc:
                failed.append((cand.path, str(exc)))
                continue
            written.append(cand.path)
    return BulkResult(written, skipped, failed)

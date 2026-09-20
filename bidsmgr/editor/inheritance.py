"""Where a sidecar value actually comes from, and moving it where it belongs.

BIDS lets a field common to many recordings live once, higher up the tree: a
``task-rest_bold.json`` at the dataset root applies to every matching file
below it. Two things follow, and the Editor helped with neither.

**A user editing a sidecar may be looking at a value that is not in it.** The
form shows the effective metadata, so ``RepetitionTime`` can appear on
``sub-01/func/..._bold.json`` while living in a file three levels up. Editing
it there writes a second copy and the two silently disagree from then on.
:func:`explain` answers "where is this value from", which is the cheaper and
more valuable half.

**A field repeated identically in every sibling belongs higher up.**
:func:`consolidation_candidates` finds those, and :func:`consolidate` moves one
up and deletes the copies, which is what the standard's inheritance is for.

The inheritance rule itself is BIDS's, so the applicable files are worked out
the way BIDS defines them: a sidecar applies to a data file when it sits at or
above the file's directory and its entities are a subset of the file's. That
is re-derived here rather than asked of bidsval only because bidsval answers it
per data file during validation and this needs the inverse question, which
sidecars could hold a value for a set of files.

Qt-free.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class Source:
    """One sidecar that contributes a value to a data file."""

    path: Path
    rel: str
    value: Any
    level: int          # 0 = beside the data file, larger = further up
    winner: bool        # does this one decide the effective value?


@dataclass
class ConsolidationCandidate:
    """A field every sibling states identically, and where it could move to."""

    field: str
    value: Any
    target: Path              # the sidecar it would move into
    target_rel: str
    sources: list[Path] = dc_field(default_factory=list)
    creates_target: bool = False

    @property
    def count(self) -> int:
        return len(self.sources)


# --------------------------------------------------------------------------
# Entities


def _stem(name: str) -> str:
    for ext in (".nii.gz", ".tsv.gz", ".fif.gz"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def entities_of(name: str) -> "OrderedDict[str, str]":
    out: "OrderedDict[str, str]" = OrderedDict()
    for part in _stem(name).split("_"):
        if "-" in part:
            key, _, val = part.partition("-")
            out[key] = val
    return out


def suffix_of(name: str) -> str:
    parts = _stem(name).split("_")
    last = parts[-1] if parts else ""
    return "" if "-" in last else last


def applies_to(sidecar: Path, target: Path) -> bool:
    """Does ``sidecar`` apply to ``target`` under BIDS inheritance?

    Three conditions, all of them the standard's: it is at or above the
    target's directory, it shares the suffix, and its entities are a subset of
    the target's with the same values.
    """
    try:
        if sidecar.parent != target.parent and \
                sidecar.parent not in target.parents:
            return False
    except (OSError, ValueError):
        return False
    if suffix_of(sidecar.name) != suffix_of(target.name):
        return False
    side = entities_of(sidecar.name)
    tgt = entities_of(target.name)
    return all(tgt.get(k) == v for k, v in side.items())


def _load(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------
# Explaining


def explain(root: Path, target: Path, field: str) -> list[Source]:
    """Every sidecar that states ``field`` for ``target``, nearest first.

    The nearest is the one that wins, which is the whole point: a user editing
    a value they can see needs to know whether they are about to create a
    second copy of it.
    """
    root, target = Path(root), Path(target)
    if target.suffix == ".json":
        target = _data_file_for(target) or target
    found: list[Source] = []
    for candidate in sorted(root.rglob("*.json")):
        if ".bidsmgr" in candidate.parts:
            continue
        if not applies_to(candidate, target):
            continue
        data = _load(candidate)
        if data is None or field not in data:
            continue
        depth = len(target.parent.parts) - len(candidate.parent.parts)
        try:
            # POSIX separators on every platform. See editor/rename.py::_rel.
            rel = candidate.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = Path(candidate).as_posix()
        found.append(Source(
            path=candidate, rel=rel, value=data[field],
            level=max(depth, 0), winner=False,
        ))
    found.sort(key=lambda s: s.level)
    if found:
        found[0].winner = True
    return found


def _data_file_for(sidecar: Path) -> Optional[Path]:
    """The data file a sidecar belongs to, for resolving inheritance."""
    stem = sidecar.stem
    for ext in (".nii.gz", ".nii", ".edf", ".bdf", ".set", ".vhdr", ".fif",
                ".tsv", ".snirf", ".ds"):
        candidate = sidecar.with_name(stem + ext)
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------
# Consolidating


def consolidation_candidates(
    root: Path, *, minimum: int = 2,
) -> list[ConsolidationCandidate]:
    """Fields every sibling sidecar of a kind states with the same value.

    Grouped by (directory-independent) suffix and entity set, because that is
    what a shared sidecar can address. A field stated by only some of the
    siblings is NOT offered: moving it up would silently apply it to the ones
    that did not state it.
    """
    root = Path(root)
    groups: dict[tuple, list[Path]] = {}
    for p in sorted(root.rglob("*.json")):
        if ".bidsmgr" in p.parts or p.parent == root:
            continue
        suffix = suffix_of(p.name)
        if not suffix:
            continue
        ent = entities_of(p.name)
        # Files that a root-level sidecar could address together: same suffix
        # and same task, which is the grouping BIDS inheritance is used for.
        key = (suffix, ent.get("task", ""))
        groups.setdefault(key, []).append(p)

    out: list[ConsolidationCandidate] = []
    for (suffix, task), paths in groups.items():
        if len(paths) < minimum:
            continue
        loaded = {p: _load(p) for p in paths}
        loaded = {p: d for p, d in loaded.items() if d is not None}
        if len(loaded) < minimum:
            continue
        shared = set.intersection(*(set(d) for d in loaded.values()))
        for field in sorted(shared):
            values = [d[field] for d in loaded.values()]
            first = values[0]
            if any(v != first for v in values):
                continue
            name = (f"task-{task}_{suffix}.json" if task
                    else f"{suffix}.json")
            target = root / name
            out.append(ConsolidationCandidate(
                field=field, value=first, target=target,
                target_rel=name, sources=list(loaded),
                creates_target=not target.exists(),
            ))
    out.sort(key=lambda c: (-c.count, c.field))
    return out


def consolidate(
    root: Path, items: list[ConsolidationCandidate],
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Move each field up and delete the copies, reversibly.

    One operation for the batch, so an undo puts every sidecar back at once.
    """
    from ..project.operations import begin_operation

    written: list[Path] = []
    failed: list[tuple[Path, str]] = []
    if not items:
        return written, failed
    label = f"Move {len(items)} field(s) to a shared sidecar"
    with begin_operation(Path(root), label) as op:
        for item in items:
            shared = _load(item.target) or {}
            shared[item.field] = item.value
            try:
                op.write_json(item.target, shared)
            except OSError as exc:
                failed.append((item.target, str(exc)))
                continue
            written.append(item.target)
            for source in item.sources:
                data = _load(source)
                if data is None or item.field not in data:
                    continue
                del data[item.field]
                try:
                    op.write_json(source, data)
                except OSError as exc:
                    failed.append((source, str(exc)))
                    continue
                written.append(source)
    return written, failed


__all__ = [
    "ConsolidationCandidate",
    "Source",
    "applies_to",
    "consolidate",
    "consolidation_candidates",
    "entities_of",
    "explain",
    "suffix_of",
]

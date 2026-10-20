"""Entities a user does not want in names, left out when the scan names them.

Some labs never want ``acq-`` in a filename, or want ``run-`` only where there
really are two runs. Removing it afterwards in the Editor works, but the table
a scan proposes is where the names are first seen, so the preference applies
there (``bidsmgr-scan --drop-entity``, Settings > Scan).

Leaving an entity out is never allowed to break a name:

* **Required stays.** Where the standard requires the entity for that kind of
  file (``echo`` on a ``MEGRE``, ``inv`` on an ``MP2RAGE``), it is kept.
* **What tells two recordings apart stays.** Two T1w images that differ only by
  ``acq`` would otherwise claim one name, and one would overwrite the other at
  conversion. Numbering them as runs instead would be wrong too: a run is a
  repeat of the SAME acquisition. So on those rows the entity is kept, and the
  report says how many.

Qt-free. Only the ``entities`` JSON is changed; the names are rebuilt from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from .. import schema
from ..schema.loader import register_cache


@lru_cache(maxsize=1)
def offered() -> tuple[str, ...]:
    """The entities a user may choose to leave out, by short key, in filename
    order.

    Every entity the standard lets some kind of file go without, except the
    ones that name a folder (``sub``, ``ses``: leaving those out would merge
    subjects or sessions) and ``task``, which says what the recording is.
    Read from the schema, so a BIDS version adding an entity offers it.
    """
    optional: set[str] = set()
    for datatype in schema.list_datatypes():
        for suffix in schema.list_suffixes(datatype):
            optional.update(schema.optional_entities(datatype, suffix))
    keep = set(schema.directory_entity_keys()) | {"task"}
    out = []
    for long_name in schema.entity_order():
        if long_name not in optional:
            continue
        try:
            key = schema.entity_info(long_name).name
        except KeyError:
            continue
        if key not in keep:
            out.append(key)
    return tuple(out)


register_cache(offered)


def _long(key: str) -> str:
    return schema.entity_key_info(key).key


@dataclass
class LeaveOutPlan:
    """What leaving entities out would do to an inventory.

    ``changes`` maps a row index to its new entities; ``kept_apart`` and
    ``kept_required`` count, per short key, the rows that keep the entity
    because it tells recordings apart or because the standard requires it.
    """

    entities: tuple[str, ...] = ()
    changes: dict = field(default_factory=dict)
    removed: dict = field(default_factory=dict)
    kept_apart: dict = field(default_factory=dict)
    kept_required: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.changes)

    def summary(self) -> str:
        """One sentence per entity, for the scan's output and the GUI."""
        lines = []
        for key in self.entities:
            gone = self.removed.get(key, 0)
            apart = self.kept_apart.get(key, 0)
            required = self.kept_required.get(key, 0)
            if not (gone or apart or required):
                continue
            kept = []
            if apart:
                kept.append(f"{_count(apart, 'recording')} where it tells two apart")
            if required:
                verb = "requires" if required == 1 else "require"
                kept.append(f"{_count(required, 'recording')} that {verb} it")
            if gone:
                text = f"Left {key} out of {_count(gone, 'name')}"
                if kept:
                    text += "; kept it on " + " and ".join(kept)
            else:
                text = f"Kept {key} on " + " and ".join(kept)
            lines.append(text + ".")
        return " ".join(lines)


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _entities(raw) -> dict | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _text(row, column: str) -> str:
    value = row.get(column, "")
    return "" if value is None or value != value else str(value).strip()


def _suffix(row) -> str:
    suffix = _text(row, "bids_guess_suffix")
    if suffix:
        return suffix
    name = _text(row, "bids_name")
    return name.rsplit("_", 1)[-1] if "_" in name else ""


def _included(row) -> bool:
    return _text(row, "include") not in ("0", "False", "false", "")


def _name(entities: dict, datatype: str, suffix: str) -> str | None:
    try:
        return schema.build_basename(entities, datatype, suffix)
    except (KeyError, ValueError, TypeError):
        return None


def plan_leave_out(df, keys: Iterable[str]) -> LeaveOutPlan:
    """What leaving ``keys`` (short entity keys) out of ``df``'s names does.

    One entity at a time, in filename order, so a row that keeps ``acq``
    because it tells two T1w images apart can still lose ``rec``.
    """
    wanted = [k for k in offered() if k in set(keys)]
    plan = LeaveOutPlan(entities=tuple(wanted))
    if not wanted or df is None or not len(df) or "entities" not in df.columns:
        return plan

    # The working state: every row's entities and destination, updated as
    # each entity is taken out, so the next entity is judged on the result.
    rows: dict = {}
    for idx, row in df.iterrows():
        ents = _entities(row.get("entities", ""))
        datatype, suffix = _text(row, "datatype"), _suffix(row)
        if ents is None or not datatype or not suffix:
            continue
        name = _name(ents, datatype, suffix)
        if name is None:
            continue
        rows[idx] = {
            "entities": ents, "datatype": datatype, "suffix": suffix,
            "dataset": _text(row, "dataset"), "included": _included(row),
            "name": name,
        }

    for key in wanted:
        long_name = _long(key)
        candidates: dict = {}
        for idx, r in rows.items():
            if not r["entities"].get(long_name):
                continue
            if long_name in schema.required_entities(r["datatype"], r["suffix"]):
                plan.kept_required[key] = plan.kept_required.get(key, 0) + 1
                continue
            ents = {k: v for k, v in r["entities"].items() if k != long_name}
            name = _name(ents, r["datatype"], r["suffix"])
            if name is not None:
                candidates[idx] = (ents, name)

        # Group by where each included row would land afterwards. A group
        # whose rows already shared a name before is an existing clash, not
        # one this makes; any other group keeps the entity on its candidates.
        groups: dict = {}
        for idx, r in rows.items():
            if not r["included"]:
                continue
            name = candidates[idx][1] if idx in candidates else r["name"]
            groups.setdefault((r["dataset"], r["datatype"], name), []).append(idx)
        held = set()
        for members in groups.values():
            if len(members) < 2:
                continue
            if len({rows[i]["name"] for i in members}) == 1:
                continue
            held.update(i for i in members if i in candidates)

        for idx, (ents, name) in candidates.items():
            if idx in held:
                plan.kept_apart[key] = plan.kept_apart.get(key, 0) + 1
                continue
            rows[idx]["entities"] = ents
            rows[idx]["name"] = name
            plan.changes[idx] = ents
            plan.removed[key] = plan.removed.get(key, 0) + 1
    return plan


def apply_leave_out(df, plan: LeaveOutPlan) -> int:
    """Write ``plan`` into ``df`` in place and rebuild the names. Returns the
    rows changed."""
    if not plan:
        return 0
    from .rebuild import rebuild_from_entities

    for idx, ents in plan.changes.items():
        df.at[idx, "entities"] = json.dumps(ents, sort_keys=True)
    rebuild_from_entities(df, in_place=True)
    return len(plan.changes)


__all__ = ["LeaveOutPlan", "apply_leave_out", "offered", "plan_leave_out"]

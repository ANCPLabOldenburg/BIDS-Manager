"""Reconcile the ``entities`` JSON column with derived display cells.

The unified inventory TSV's ``entities`` column is the **source of truth**
for the BIDS basename: a JSON-encoded dict like
``{"subject": "001", "session": "pre", "task": "rest", "run": "1"}``.
Display cells (``proposed_basename``, ``Proposed BIDS name``, ``session``,
``task``, ``run``) are derived from it.

Two reconciliation directions:

* :func:`rebuild_from_entities` — the user (or the GUI) edited the
  ``entities`` JSON. Recompute the basename and mirror cells. **This is
  the default direction**: the convert verb runs it in memory before
  reading rows so stale displays never poison the conversion.

* :func:`rebuild_from_columns` — the user edited individual cells like
  ``task`` or ``run`` in a spreadsheet. Reverse-derive the JSON so the
  next ``rebuild_from_entities`` (or convert) sees the changes.

Both functions are pure data — they take a DataFrame, return a new
DataFrame plus a list of human-readable changes for the CLI to print.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .. import schema as schema_mod

log = logging.getLogger(__name__)


# Mirror cells: BIDS entity name → TSV column name. Updated by
# ``rebuild_from_entities`` so a non-technical user can read the table
# without parsing JSON. Reverse-mapped by ``rebuild_from_columns``.
_MIRROR: dict[str, str] = {
    "session": "session",
    "task": "task",
    "run": "run",
}


@dataclass
class RebuildReport:
    """Summary of changes applied during a rebuild pass."""

    rows_updated: int = 0
    basename_changes: int = 0
    mirror_changes: int = 0
    json_repaired: int = 0
    warnings: list[str] = field(default_factory=list)
    diffs: list[dict] = field(default_factory=list)  # one entry per row that changed

    def add_diff(
        self,
        row_idx: int,
        field_name: str,
        before: object,
        after: object,
    ) -> None:
        self.diffs.append({
            "row": int(row_idx),
            "field": field_name,
            "before": str(before),
            "after": str(after),
        })


# ---------------------------------------------------------------------------
# entities → display cells
# ---------------------------------------------------------------------------


def rebuild_from_entities(
    df: pd.DataFrame,
    *,
    in_place: bool = False,
) -> tuple[pd.DataFrame, RebuildReport]:
    """Recompute display cells (``proposed_basename``, mirror columns)
    from each row's ``entities`` JSON.

    Rows with missing or malformed ``entities`` are skipped with a
    warning recorded in the report.

    The orchestrator (``cli/convert.py``) runs this in memory before
    reading rows so a TSV that's been hand-edited (entities changed,
    but rebuild not yet run) still converts to the correct BIDS names.
    """
    out = df if in_place else df.copy()
    report = RebuildReport()

    if "entities" not in out.columns:
        report.warnings.append(
            "no 'entities' column in inventory; rebuild is a no-op"
        )
        return out, report

    for idx in out.index:
        raw = str(out.at[idx, "entities"]).strip()
        if not raw:
            continue
        try:
            entities = json.loads(raw)
        except json.JSONDecodeError as exc:
            report.warnings.append(
                f"row {idx}: cannot parse entities JSON ({exc.msg}): {raw[:80]}"
            )
            continue
        if not isinstance(entities, dict):
            report.warnings.append(
                f"row {idx}: entities is not a JSON object: {raw[:80]}"
            )
            continue

        datatype = str(out.at[idx, "proposed_datatype"]).strip()
        suffix = (
            str(out.at[idx, "bids_guess_suffix"]).strip()
            or _suffix_from_basename(str(out.at[idx, "proposed_basename"]).strip())
        )

        row_changed = False
        if datatype and suffix:
            try:
                new_basename = schema_mod.build_basename(entities, datatype, suffix)
            except (ValueError, KeyError, TypeError) as exc:
                report.warnings.append(
                    f"row {idx}: build_basename failed ({exc}); "
                    f"basename left as-is"
                )
            else:
                old_basename = str(out.at[idx, "proposed_basename"]).strip()
                if new_basename != old_basename:
                    out.at[idx, "proposed_basename"] = new_basename
                    out.at[idx, "Proposed BIDS name"] = new_basename
                    report.add_diff(
                        idx, "proposed_basename", old_basename, new_basename,
                    )
                    report.basename_changes += 1
                    row_changed = True

        # Mirror cells.
        for ent_key, col_name in _MIRROR.items():
            if col_name not in out.columns:
                continue
            new_value = entities.get(ent_key, "")
            if ent_key == "session" and new_value:
                # Mirror column carries the BIDS form ``ses-X``.
                new_value = (
                    new_value if str(new_value).startswith("ses-")
                    else f"ses-{new_value}"
                )
            new_str = "" if new_value is None else str(new_value)
            old_str = str(out.at[idx, col_name]).strip()
            if old_str != new_str:
                out.at[idx, col_name] = new_str
                report.add_diff(idx, col_name, old_str, new_str)
                report.mirror_changes += 1
                row_changed = True

        if row_changed:
            report.rows_updated += 1

    return out, report


# ---------------------------------------------------------------------------
# display cells → entities
# ---------------------------------------------------------------------------


def rebuild_from_columns(
    df: pd.DataFrame,
    *,
    in_place: bool = False,
) -> tuple[pd.DataFrame, RebuildReport]:
    """Reverse direction: read ``session`` / ``task`` / ``run`` cells
    and update each row's ``entities`` JSON to match.

    Useful after the user edits individual cells in a spreadsheet —
    sync the JSON before calling :func:`rebuild_from_entities` (or
    just run convert, which auto-rebuilds in memory).
    """
    out = df if in_place else df.copy()
    report = RebuildReport()

    if "entities" not in out.columns:
        report.warnings.append(
            "no 'entities' column in inventory; rebuild is a no-op"
        )
        return out, report

    for idx in out.index:
        raw = str(out.at[idx, "entities"]).strip()
        entities: dict = {}
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    entities = parsed
            except json.JSONDecodeError:
                report.json_repaired += 1
                report.warnings.append(
                    f"row {idx}: malformed entities JSON; rebuilding from columns"
                )

        # Subject identity is locked at scan time — read from BIDS_name.
        bids_name = str(out.at[idx, "BIDS_name"]).strip()
        if bids_name.startswith("sub-"):
            entities["subject"] = bids_name[len("sub-"):]
        elif bids_name:
            entities["subject"] = bids_name

        # Mirror cells back into entities.
        for ent_key, col_name in _MIRROR.items():
            if col_name not in out.columns:
                continue
            cell = str(out.at[idx, col_name]).strip()
            if not cell:
                # Empty cell removes the entity (so the basename rebuild
                # leaves it out).
                entities.pop(ent_key, None)
                continue
            if ent_key == "session" and cell.startswith("ses-"):
                cell = cell[len("ses-"):]
            entities[ent_key] = cell

        new_json = json.dumps(entities, sort_keys=True)
        if new_json != raw:
            out.at[idx, "entities"] = new_json
            report.add_diff(idx, "entities", raw, new_json)
            report.rows_updated += 1

    # Once the JSON is in sync, also refresh derived cells.
    out, secondary = rebuild_from_entities(out, in_place=True)
    report.basename_changes += secondary.basename_changes
    report.mirror_changes += secondary.mirror_changes
    report.warnings.extend(secondary.warnings)

    return out, report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suffix_from_basename(basename: str) -> str:
    """Last underscore-delimited token of a BIDS basename = its suffix."""
    if not basename:
        return ""
    return basename.rsplit("_", 1)[-1]


__all__ = [
    "RebuildReport",
    "rebuild_from_columns",
    "rebuild_from_entities",
]

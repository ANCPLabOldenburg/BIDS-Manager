"""``QAbstractTableModel`` over the unified-TSV inventory DataFrame.

The keystone of the GUI: every view that shows inventory rows binds to
this model. The model holds a working ``pandas.DataFrame`` (loaded from
a ``bidsmgr-scan`` TSV), overlays any user edits recorded in an
optional :class:`bidsmgr.project.Project`, and exposes 12 display
columns mapped onto the 51-column unified schema.

Responsibilities (kept narrow so the model stays testable without Qt):

* Provide ``data`` / ``headerData`` / ``flags`` / ``setData`` for the
  view.
* Publish per-cell **row state** (selected/warn/err/skip) and **status
  kind** (ok/warn/err/phys/skip) via the delegate roles defined in
  :mod:`bidsmgr.gui.delegates`.
* On user edit, mutate the working DataFrame, run the existing
  :func:`bidsmgr.inventory.rebuild.rebuild_from_columns` reconciliation
  for the affected row, append a ``UserSetCell`` / ``UserToggleInclude``
  event to the project (if one is attached), and emit ``dataChanged``.

Not handled here:

* File I/O (the controller loads the TSV; this model takes a ready DataFrame).
* Threading (workers live in :mod:`bidsmgr.workers`).
* Properties-panel editing (a future controller will edit ``entities``
  JSON directly and call :meth:`refresh_row`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt

from ... import schema as schema_mod
from ...inventory.rebuild import rebuild_from_columns, rebuild_from_entities
from ...project import (
    Project,
    UserSetCell,
    UserSetEntity,
    UserToggleInclude,
)
from ...project.types import ProjectState
from ..delegates import HIGHLIGHT_ROLE, PAYLOAD_ROLE, ROW_STATE_ROLE

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """One display column of the inventory table.

    ``key`` is the stable identifier used in events + tests; ``header``
    is the human-readable label. ``role`` selects the delegate paint
    style (matches the strings :class:`CellTextDelegate` understands).
    ``df_column`` is the underlying DataFrame column name; ``None``
    means the value is derived (status badge, backend).
    """

    key: str
    header: str
    role: str          # 'checkbox' | 'status' | 'plain' | 'mono' | 'conf' | 'basename'
    editable: bool
    width: int
    stretch: bool = False
    df_column: Optional[str] = None
    default_visible: bool = True  # toggleable via the column-visibility menu


# Curated set of columns the inspector can show. ``default_visible``
# controls the initial state; the column-visibility menu persists user
# choices via QSettings. **The model never deletes columns from the
# underlying TSV** — visibility is purely view-state.
COLUMNS: tuple[ColumnSpec, ...] = (
    # Mandatory (cannot be hidden by the user — they're the row identity).
    ColumnSpec("include",   "",                       "checkbox", True,  28),
    ColumnSpec("status",    "",                       "status",   False, 28),
    ColumnSpec("id",        "id",                     "mono",     False, 50, df_column="BIDS_name"),
    # Default-visible curated set.
    ColumnSpec("dataset",   "dataset",                "plain",    True,  100, df_column="dataset"),
    ColumnSpec("ses",       "ses",                    "mono",     True,  50, df_column="session"),
    ColumnSpec("mod",       "mod",                    "plain",    False, 38, df_column="modality"),
    ColumnSpec("datatype",  "data",                   "plain",    True,  50, df_column="proposed_datatype"),
    ColumnSpec("suffix",    "suffix",                 "plain",    True,  80, df_column="bids_guess_suffix"),
    ColumnSpec("task",      "task",                   "plain",    True,  60, df_column="task"),
    ColumnSpec("run",       "run",                    "mono",     True,  50, df_column="run"),
    ColumnSpec("conf",      "conf",                   "conf",     False, 50, df_column="bids_guess_confidence"),
    ColumnSpec("sequence",  "sequence / source",      "mono",     False, 200, df_column="sequence"),
    ColumnSpec("basename",  "predicted basename",     "basename", False, 320, stretch=True, df_column="proposed_basename"),
    # Default-hidden — show via the column-visibility menu.
    ColumnSpec("backend",      "backend",     "mono",  False, 90,  default_visible=False),
    ColumnSpec("source_file",  "source file", "mono",  False, 220, df_column="source_file", default_visible=False),
    ColumnSpec("n_files",      "n_files",     "mono",  False, 60,  df_column="n_files",     default_visible=False),
    ColumnSpec("acq_time",     "acq_time",    "mono",  False, 80,  df_column="acq_time",    default_visible=False),
    ColumnSpec("image_type",   "image_type",  "mono",  False, 120, df_column="image_type",  default_visible=False),
    ColumnSpec("study_date",   "study_date",  "mono",  False, 90,  df_column="study_date",  default_visible=False),
    ColumnSpec("PatientID",    "PatientID",   "mono",  False, 100, df_column="PatientID",   default_visible=False),
    ColumnSpec("GivenName",    "GivenName",   "plain", False, 100, df_column="GivenName",   default_visible=False),
    ColumnSpec("FamilyName",   "FamilyName",  "plain", False, 100, df_column="FamilyName",  default_visible=False),
    ColumnSpec("PatientSex",   "sex",         "mono",  False, 40,  df_column="PatientSex",  default_visible=False),
    ColumnSpec("PatientAge",   "age",         "mono",  False, 50,  df_column="PatientAge",  default_visible=False),
    ColumnSpec("n_channels",   "n_chan",      "mono",  False, 60,  df_column="n_channels",  default_visible=False),
    ColumnSpec("sfreq",        "sfreq",       "mono",  False, 70,  df_column="sfreq",       default_visible=False),
    ColumnSpec("duration_sec", "duration",    "mono",  False, 70,  df_column="duration_sec", default_visible=False),
    ColumnSpec("line_freq",    "line_freq",   "mono",  True,  60,  df_column="line_freq",   default_visible=False),
    ColumnSpec("montage",      "montage",     "plain", True,  100, df_column="montage",     default_visible=False),
    ColumnSpec("probe_n_nifti","probe_nifti", "mono",  False, 80,  df_column="probe_n_nifti", default_visible=False),
    ColumnSpec("probe_n_vols", "probe_vols",  "mono",  False, 70,  df_column="probe_n_volumes", default_visible=False),
    ColumnSpec("repetition_type","repetition","plain", False, 110, df_column="repetition_type", default_visible=False),
    ColumnSpec("proposed_issues","issues",    "plain", False, 240, df_column="proposed_issues", default_visible=False),
)

# Columns the user cannot hide (lose them and rows become unidentifiable).
MANDATORY_COLUMN_KEYS: frozenset[str] = frozenset({"include", "status", "id"})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class InventoryTableModel(QAbstractTableModel):
    """Inventory rows surfaced to the Converter view's table.

    Parameters
    ----------
    df
        The unified-TSV DataFrame as produced by :func:`bidsmgr.cli.scan.run_scan`.
        Index is reset to ``0..N-1`` on construction; callers should not
        rely on the original index after passing it in.
    project
        Optional :class:`bidsmgr.project.Project`. When attached, the
        model appends ``UserSetCell`` / ``UserToggleInclude`` events on
        every edit and applies the project's ``ProjectState`` overrides
        on construction (so reopening a project re-applies user edits).
    """

    COLUMNS: tuple[ColumnSpec, ...] = COLUMNS

    def __init__(
        self,
        df: pd.DataFrame,
        project: Optional[Project] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._df: pd.DataFrame = df.reset_index(drop=True).copy()
        self._project: Optional[Project] = project

        if project is not None:
            self._apply_project_overlay(project.state())

        # Cache per-row state so delegates don't re-derive on every paint.
        # Invalidated for one row by :meth:`refresh_row`.
        self._row_states: list[str] = [
            self._derive_row_state(i) for i in range(len(self._df))
        ]
        # "Highlight aborts" toggle: when True, rows the scanner
        # flagged as ``suspected_abort`` (operator likely re-recorded
        # immediately after this attempt) publish ``True`` on
        # ``HIGHLIGHT_ROLE`` so the delegates paint a purple tint.
        # These rows are already auto-unchecked by the scan, but
        # highlighting helps the user spot them at a glance.
        self._highlight_aborts: bool = False

    # ------------------------------------------------------------------
    # Qt model API
    # ------------------------------------------------------------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section].header
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        f = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        spec = self.COLUMNS[index.column()]
        if spec.editable:
            f |= Qt.ItemFlag.ItemIsEditable
        return f

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if not (0 <= row < len(self._df) and 0 <= col < len(self.COLUMNS)):
            return None
        spec = self.COLUMNS[col]

        # Row state is published on every cell — delegates read it before
        # drawing so the tint is applied beneath text/badges/checkboxes.
        if role == ROW_STATE_ROLE:
            return self._row_states[row]

        # Highlight overlay (purple tint) — published on every cell
        # so delegates can paint it on top of the row-state tint.
        if role == HIGHLIGHT_ROLE:
            return self._highlight_aborts and self.is_row_aborted(row)

        # Checkbox and status cells communicate via PAYLOAD_ROLE instead
        # of DisplayRole; they have no text body.
        if spec.role == "checkbox":
            if role == PAYLOAD_ROLE:
                return self._read_include(row)
            return None
        if spec.role == "status":
            if role == PAYLOAD_ROLE:
                return self._status_kind(row)
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_value(row, spec)
        if role == Qt.ItemDataRole.EditRole:
            return self._raw_value(row, spec)
        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if not index.isValid():
            return False
        row, col = index.row(), index.column()
        spec = self.COLUMNS[col]
        if not spec.editable:
            return False

        if spec.role == "checkbox":
            new_inc = bool(value)
            self._set_include(row, new_inc)
            self.refresh_row(row)
            return True

        df_col = spec.df_column
        if df_col is None:
            return False
        new_str = "" if value is None else str(value)
        old_str = "" if pd.isna(self._df.at[row, df_col]) else str(self._df.at[row, df_col])
        if new_str == old_str:
            return False

        # Order: record the event first (durable), then mutate.
        if self._project is not None:
            self._project.append(
                UserSetCell(
                    row_id=self.row_id(row),
                    column=df_col,
                    value=new_str,
                    previous=old_str,
                )
            )
        self._df.at[row, df_col] = new_str

        # Mirror cells (session / task / run) feed the entities JSON.
        # Reconcile in both directions so the basename column stays
        # consistent with the user-visible cell.
        if spec.key in ("ses", "task", "run", "datatype", "suffix"):
            self._rebuild_one_row(row)

        self.refresh_row(row)
        return True

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def row_id(self, row: int) -> str:
        """Return a stable per-row identifier used in project events.

        Prefers ``series_uid`` (MRI) → ``source_file`` (EEG/MEG) →
        ``f"row-{row}"``. The fallback is purely positional so it
        survives reopen only when the row order does; in practice
        every real row carries one of the first two identifiers.
        """
        for col in ("series_uid", "source_file"):
            if col in self._df.columns:
                v = self._df.at[row, col]
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return f"row-{row}"

    def entities(self, row: int) -> dict[str, str]:
        """Return the row's parsed entities dict, or empty if malformed.

        Used by the Properties panel to populate its schema-driven form.
        Returns a fresh dict each call so the caller can mutate freely.
        """
        if not (0 <= row < len(self._df)):
            return {}
        if "entities" not in self._df.columns:
            return {}
        raw = self._df.at[row, "entities"]
        if pd.isna(raw):
            return {}
        text = str(raw).strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("row %d: malformed entities JSON; returning {}", row)
            return {}
        return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def set_entity(
        self,
        row: int,
        entity: str,
        value: Optional[str],
    ) -> bool:
        """Set (or delete with ``value=None`` / ``""``) one BIDS entity for ``row``.

        This is the path the Properties panel uses for entities that
        have **no mirror column** (``acquisition``, ``direction``,
        ``echo``, ``ceagent``, ``reconstruction``, ``part``, ``chunk``).
        For mirror entities (``session``, ``task``, ``run``) it also
        works but the table's mirror cell stays in sync because
        ``rebuild_from_entities`` runs after the write.

        Order: record event, update entities JSON, rebuild, refresh.
        Returns ``True`` if anything changed.
        """
        if not (0 <= row < len(self._df)):
            return False
        if "entities" not in self._df.columns:
            return False

        current = self.entities(row)
        new_value = "" if value is None else str(value).strip()
        previous = current.get(entity, "")

        if new_value == previous:
            return False

        if self._project is not None:
            self._project.append(
                UserSetEntity(
                    row_id=self.row_id(row),
                    entity=entity,
                    value=new_value or None,
                    previous=previous or None,
                )
            )

        if new_value:
            current[entity] = new_value
        else:
            current.pop(entity, None)

        self._df.at[row, "entities"] = json.dumps(current, sort_keys=True)

        # ``BIDS_name`` is the de-facto mirror for the ``subject``
        # entity: ``rebuild_from_columns`` reads it back as ground truth.
        # Keep it in sync so subsequent mirror-cell edits don't undo
        # this change.
        if entity == "subject" and "BIDS_name" in self._df.columns:
            self._df.at[row, "BIDS_name"] = (
                f"sub-{new_value}" if new_value else ""
            )

        # entities-direction rebuild: push the JSON into mirror cells +
        # rebuild basename.
        self._rebuild_one_row(row, direction="entities")
        self.refresh_row(row)
        return True

    def row_issues(self, row: int) -> list[str]:
        """Return the parsed list of scanner-detected issues for ``row``.

        ``proposed_issues`` in the unified TSV is a `` | ``-joined
        string assembled by the scan step (suspected aborts, B0
        reroutes, missing required entities, etc.). Splitting it here
        keeps the inspector + Properties panel + IssuesDialog reading
        from the same source.
        """
        if not (0 <= row < len(self._df)):
            return []
        if "proposed_issues" not in self._df.columns:
            return []
        raw = self._df.at[row, "proposed_issues"]
        if pd.isna(raw):
            return []
        text = str(raw).strip()
        if not text:
            return []
        return [p.strip() for p in text.split(" | ") if p.strip()]

    def row_state(self, row: int) -> str:
        """Convenience accessor for the cached row state (``""``/``warn``/
        ``err``/``skip``). Used by views that want to drive coloring
        without going through ``data(index, ROW_STATE_ROLE)``.
        """
        if 0 <= row < len(self._row_states):
            return self._row_states[row]
        return ""

    # ------------------------------------------------------------------
    # Aborted-sequence highlighting
    # ------------------------------------------------------------------

    def is_row_aborted(self, row: int) -> bool:
        """``True`` if the scanner flagged this row as ``suspected_abort``.

        Aborts are operator-restart cases: same SeriesDescription as a
        later companion, within the scanner's redo window, neither side
        trivial. The scan already sets ``include=0`` for these so they
        won't convert by default; this helper just powers the
        "Highlight aborts" toolbar overlay.
        """
        if not (0 <= row < len(self._df)):
            return False
        if "repetition_type" not in self._df.columns:
            return False
        v = self._df.at[row, "repetition_type"]
        if pd.isna(v):
            return False
        return str(v).strip() == "suspected_abort"

    def set_highlight_aborts(self, enabled: bool) -> None:
        """Toggle the purple-tint overlay for ``suspected_abort`` rows.

        Emits ``dataChanged`` over the whole table so every cell
        repaints with the new highlight state. Cheap: only sets a flag
        on the model — the per-row classification is read on-demand.
        """
        enabled = bool(enabled)
        if enabled == self._highlight_aborts:
            return
        self._highlight_aborts = enabled
        if self.rowCount() == 0:
            return
        top = self.index(0, 0)
        bot = self.index(self.rowCount() - 1, self.columnCount() - 1)
        self.dataChanged.emit(top, bot, [HIGHLIGHT_ROLE])

    def highlight_aborts(self) -> bool:
        """Current state of the "Highlight aborts" toggle."""
        return self._highlight_aborts

    # ------------------------------------------------------------------
    # Bulk edit
    # ------------------------------------------------------------------

    # Keys the bulk-edit dialog can target. The order is intentional —
    # the dropdown shows these in this sequence so the most common
    # targets (subject, dataset, task) come first.
    BULK_EDITABLE_KEYS: tuple[str, ...] = (
        "id",        # → subject entity + BIDS_name
        "dataset",
        "ses",
        "task",
        "run",
        "datatype",
        "suffix",
        "line_freq",
        "montage",
    )

    def bulk_set(
        self,
        rows: list[int],
        column_key: str,
        value: str,
    ) -> int:
        """Apply ``value`` to ``column_key`` across every row in ``rows``.

        Dispatches to the right per-row API so entities, mirror cells,
        and the basename column stay consistent. Returns the count of
        rows actually changed (no-ops are excluded).

        * ``id``                → :meth:`set_entity` on ``subject``
          (this also updates ``BIDS_name``).
        * ``datatype`` / ``suffix`` → :meth:`set_datatype_suffix`
          (the unchanged half is preserved per-row).
        * Everything else      → :meth:`setData` on the column index
          (mirror-cell rebuilds happen inside ``setData``).
        """
        if column_key not in self.BULK_EDITABLE_KEYS:
            return 0

        changed = 0
        col_idx = next(
            (i for i, c in enumerate(self.COLUMNS) if c.key == column_key),
            None,
        )

        for row in rows:
            if not (0 <= row < self.rowCount()):
                continue

            if column_key == "id":
                if self.set_entity(row, "subject", value):
                    changed += 1
                continue

            if column_key == "datatype":
                _dt, sf = self.datatype_suffix(row)
                if self.set_datatype_suffix(row, value, sf):
                    changed += 1
                continue

            if column_key == "suffix":
                dt, _sf = self.datatype_suffix(row)
                if self.set_datatype_suffix(row, dt, value):
                    changed += 1
                continue

            if col_idx is not None:
                if self.setData(self.index(row, col_idx), value):
                    changed += 1

        return changed

    def datatype_suffix(self, row: int) -> tuple[str, str]:
        """Return ``(datatype, suffix)`` for ``row``, both possibly empty.

        Reads ``proposed_datatype`` + ``bids_guess_suffix`` (with empty
        fallback). Used by the Properties panel to drive its combos.
        """
        if not (0 <= row < len(self._df)):
            return ("", "")
        dt = ""
        sf = ""
        if "proposed_datatype" in self._df.columns:
            dt = "" if pd.isna(self._df.at[row, "proposed_datatype"]) else str(self._df.at[row, "proposed_datatype"])
        if "bids_guess_suffix" in self._df.columns:
            sf = "" if pd.isna(self._df.at[row, "bids_guess_suffix"]) else str(self._df.at[row, "bids_guess_suffix"])
        return (dt, sf)

    def set_datatype_suffix(self, row: int, datatype: str, suffix: str) -> bool:
        """Set both columns simultaneously and rebuild the row.

        Treated as one logical edit (one project event each, but
        sequential). Returns ``True`` if anything changed.
        """
        if not (0 <= row < len(self._df)):
            return False
        changed = False
        for col, val, key in (
            ("proposed_datatype", datatype, "datatype"),
            ("bids_guess_suffix", suffix, "suffix"),
        ):
            if col not in self._df.columns:
                continue
            old = "" if pd.isna(self._df.at[row, col]) else str(self._df.at[row, col])
            new = str(val)
            if new == old:
                continue
            if self._project is not None:
                self._project.append(
                    UserSetCell(
                        row_id=self.row_id(row),
                        column=col,
                        value=new,
                        previous=old,
                    )
                )
            self._df.at[row, col] = new
            changed = True
        if changed:
            self._rebuild_one_row(row)
            self.refresh_row(row)
        return changed

    def refresh_row(self, row: int) -> None:
        """Re-derive cached state for ``row`` and notify the view.

        Call after any mutation that could affect row state (include
        toggle, entity edit). Emits ``dataChanged`` for every column of
        the row so per-row tints, badges, and mirror-cell text refresh
        together.
        """
        if not (0 <= row < len(self._df)):
            return
        self._row_states[row] = self._derive_row_state(row)
        left = self.index(row, 0)
        right = self.index(row, self.columnCount() - 1)
        self.dataChanged.emit(left, right)

    def dataframe(self) -> pd.DataFrame:
        """Return the live working DataFrame.

        The caller MUST treat it as read-only; the model is the
        authority on edits. Use this for save-to-disk by the controller.
        """
        return self._df

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_project_overlay(self, state: ProjectState) -> None:
        """Apply a ``ProjectState`` overlay to the working DataFrame.

        Reverses the on-save semantics: for each row whose ``row_id``
        appears in ``state``, the saved overrides are written back into
        the DataFrame so the view shows the same thing as before the
        save. Unknown row_ids are ignored (the inventory may have been
        rescanned, dropping rows the project remembers).
        """
        if not state.cell_overrides and not state.include_overrides:
            return

        # Build row_id → df_index map once.
        index_for_rid: dict[str, int] = {
            self.row_id(i): i for i in range(len(self._df))
        }

        for rid, cells in state.cell_overrides.items():
            r = index_for_rid.get(rid)
            if r is None:
                continue
            for col, val in cells.items():
                if col in self._df.columns:
                    self._df.at[r, col] = val
            # Force a rebuild for the row so derived cells stay consistent.
            self._rebuild_one_row(r)

        if "include" in self._df.columns:
            for rid, inc in state.include_overrides.items():
                r = index_for_rid.get(rid)
                if r is None:
                    continue
                self._df.at[r, "include"] = 1 if inc else 0

    def _rebuild_one_row(self, row: int, *, direction: str = "columns") -> None:
        """Reconcile ``entities`` ↔ display cells for a single row.

        ``direction="columns"`` runs :func:`rebuild_from_columns`: read
        mirror cells → update entities JSON → rebuild basename. Use
        after a mirror-cell edit.

        ``direction="entities"`` runs :func:`rebuild_from_entities`:
        read entities JSON → update mirror cells + basename. Use after
        a Properties-panel entity edit.

        Both call paths mirror the validated CLI logic exactly so we
        don't duplicate name-building rules here.
        """
        sub = self._df.iloc[[row]].copy()
        if direction == "entities":
            new_sub, _report = rebuild_from_entities(sub)
        else:
            new_sub, _report = rebuild_from_columns(sub)
        for col in new_sub.columns:
            self._df.at[row, col] = new_sub.iloc[0][col]

    def _read_include(self, row: int) -> bool:
        """Return the include flag for ``row``.

        Defaults to ``True`` when the column is missing or empty (a
        fresh scan TSV may have blank includes; the convert verb treats
        blank as included).
        """
        if "include" not in self._df.columns:
            return True
        v = self._df.at[row, "include"]
        if pd.isna(v):
            return True
        if isinstance(v, str):
            return v.strip() not in ("0", "false", "False", "")
        return bool(v)

    def _set_include(self, row: int, included: bool) -> None:
        """Persist a new include flag (event first, then DataFrame)."""
        if self._project is not None:
            self._project.append(
                UserToggleInclude(row_id=self.row_id(row), include=included)
            )
        if "include" not in self._df.columns:
            self._df["include"] = 1
        self._df.at[row, "include"] = 1 if included else 0

    # -------- derived values --------

    def _display_value(self, row: int, spec: ColumnSpec) -> str:
        if spec.key == "backend":
            return self._backend(row)
        if spec.df_column is None:
            return ""
        if spec.df_column not in self._df.columns:
            # Column referenced by the spec isn't present in this
            # DataFrame (e.g. an old TSV missing ``dataset``, or a
            # synthetic test row). Show blank instead of crashing.
            return ""
        raw = self._df.at[row, spec.df_column]
        if pd.isna(raw):
            return "—"
        text = str(raw)

        if spec.key == "id":
            return text[len("sub-"):] if text.startswith("sub-") else text
        if spec.key == "ses":
            stripped = text[len("ses-"):] if text.startswith("ses-") else text
            return stripped or "—"
        if spec.key == "conf":
            return self._format_confidence(text)
        if not text.strip():
            return "—"
        return text

    def _raw_value(self, row: int, spec: ColumnSpec) -> str:
        """The EditRole value — same as display but without dashes / formatting.

        The view sends this back through :meth:`setData` after an edit;
        we don't want the user to see ``—`` in the editor and accidentally
        commit it as text. ``—`` collapses to empty string.
        """
        if spec.df_column is None:
            return ""
        if spec.df_column not in self._df.columns:
            return ""
        raw = self._df.at[row, spec.df_column]
        if pd.isna(raw):
            return ""
        text = str(raw)
        if spec.key == "ses" and text.startswith("ses-"):
            text = text[len("ses-"):]
        return "" if text == "—" else text

    def _backend(self, row: int) -> str:
        """Derive the converter backend that will handle this row."""
        suffix = ""
        if "bids_guess_suffix" in self._df.columns:
            suffix = str(self._df.at[row, "bids_guess_suffix"] or "")
        if suffix == "physio":
            return "bidsphysio"
        if "source_file" in self._df.columns:
            src = self._df.at[row, "source_file"]
            if isinstance(src, str) and src.strip():
                return "mne-bids"
        return "dcm2niix"

    @staticmethod
    def _format_confidence(text: str) -> str:
        """Format a numeric confidence as ``.97`` (the proto's style)."""
        try:
            v = float(text)
        except ValueError:
            return text
        # ".97" — no leading zero, two decimal places.
        s = f"{v:.2f}"
        return s[1:] if s.startswith("0.") else s

    # -------- per-row state --------

    def _derive_row_state(self, row: int) -> str:
        """One of ``""``, ``"warn"``, ``"err"``, ``"skip"``.

        Drives the row tint applied by every delegate. Selection is
        applied by the view at paint time, not stored here.
        """
        if not self._read_include(row):
            return "skip"

        if "bids_guess_skip" in self._df.columns:
            v = self._df.at[row, "bids_guess_skip"]
            if isinstance(v, str):
                if v.strip() in ("1", "true", "True"):
                    return "skip"
            elif bool(v):
                return "skip"

        issues = ""
        if "proposed_issues" in self._df.columns:
            issues = str(self._df.at[row, "proposed_issues"] or "")
        if issues:
            lowered = issues.lower()
            if any(tok in lowered for tok in (
                "suspected_abort",
                "required",
                "build_basename",
                "missing",
            )):
                return "err"
            return "warn"

        basename = ""
        if "proposed_basename" in self._df.columns:
            basename = str(self._df.at[row, "proposed_basename"] or "")
        datatype = ""
        if "proposed_datatype" in self._df.columns:
            datatype = str(self._df.at[row, "proposed_datatype"] or "")
        if not basename or not datatype:
            return "err"

        return ""

    def _status_kind(self, row: int) -> str:
        """Kind passed to ``StatusDelegate`` (the badge column).

        ``phys`` overrides ``ok`` so the physio rows in the prototype
        keep their distinct icon.
        """
        state = self._row_states[row]
        if state == "skip":
            return "skip"
        if state == "err":
            return "err"
        if state == "warn":
            return "warn"
        if self._backend(row) == "bidsphysio":
            return "phys"
        return "ok"


__all__ = ["COLUMNS", "ColumnSpec", "InventoryTableModel"]

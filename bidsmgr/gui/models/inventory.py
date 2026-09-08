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
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal

from ... import schema as schema_mod
from ...inventory.rebuild import rebuild_from_columns, rebuild_from_entities
from ...project import (
    Project,
    UserSetCell,
    UserSetEntity,
    UserToggleInclude,
)
from ...project.types import ProjectState
from ...recording_meta import (
    AcquisitionSpec,
    describe_origin,
    is_varies,
    resolve_attribute,
    resolve_sidecar_fields,
    PetAcquisitionSpec,
    RecordingMetaSpec,
)
from ..delegates import HIGHLIGHT_ROLE, INHERITED_ROLE, PAYLOAD_ROLE, ROW_STATE_ROLE

log = logging.getLogger(__name__)

# Per-row columns whose value inherits from the dataset-wide recording-metadata
# default when the cell is blank. Maps the TSV/df column -> the attribute on
# ``RecordingMetaSpec.defaults`` that supplies the default. Editing a cell to a
# value equal to the default clears it back to "inherited".
_INHERITANCE_FIELDS: dict[str, str] = {
    "line_freq": "power_line_freq",
    "montage": "montage",
    "eeg_reference": "eeg_reference",
    "eeg_ground": "eeg_ground",
}

# Which rows a metadata column means anything for.
#
# Without this every one of the columns below renders on every row, so with
# dataset defaults set an anatomical scan showed a line frequency, a montage,
# a reference and a ground. One row in nine was right.
#
# The answer comes from the schema wherever the column corresponds to a real
# BIDS field, so it stays correct as the standard changes, and it is worth
# noting where that disagrees with intuition: EEGReference applies to MEG as
# well as EEG, because a MEG recording may carry simultaneous EEG, and iEEG
# has its OWN pair of fields rather than reusing the EEG ones.
_COLUMN_BIDS_FIELDS: dict[str, tuple[str, ...]] = {
    "line_freq": ("PowerLineFrequency",),
    "eeg_reference": ("EEGReference", "iEEGReference"),
    "eeg_ground": ("EEGGround", "iEEGGround"),
}

# Columns with no BIDS field behind them. A montage is a BIDS Manager notion
# (which standard electrode layout to apply on conversion), so the standard has
# nothing to say and the scope is ours to state.
_COLUMN_DATATYPES: dict[str, frozenset[str]] = {
    "montage": frozenset({"eeg", "ieeg"}),
}

# Per-row recording-acquisition fields that live in the recording-metadata
# scaffold's ``overrides[row_id]`` block rather than a TSV column (the convert
# step's ``resolve_effective`` already layers them over the dataset defaults and
# the enrichment fixup writes them into the EEG/MEG sidecar). These are the
# device / institution values that can differ between subjects but are too
# low-frequency to warrant a spreadsheet column. Maps the panel key -> the
# attribute on ``AcquisitionSpec``. A blank override inherits the dataset
# default; setting a value equal to the default clears the override.
# Device fields are modality-specific (the EEG amplifier and the MEG system are
# different devices); institution is agnostic and lives at dataset level in the
# Dataset-metadata dialog, NOT here. MEG fields are only the ones mne-bids
# cannot derive. All are string-valued.
def _as_text(value) -> str:
    """Render a spec value for a form field, without a float's trailing .0."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_ACQ_OVERRIDE_FIELDS: dict[str, str] = {
    "manufacturer": "manufacturer",
    "amplifier_model": "amplifier_model",
    "software_versions": "software_versions",
    "cap_manufacturer": "cap_manufacturer",
    # MEG-specific (manual only)
    "dewar_position": "dewar_position",
    "associated_empty_room": "associated_empty_room",
    "subject_artefact_description": "subject_artefact_description",
}

# The PET equivalent. Kept as its own map because the two blocks live in
# separate spec fields (pet_defaults / pet_overrides) and share no field names,
# so one combined map would only invite a lookup into the wrong block.
_PET_OVERRIDE_FIELDS: dict[str, str] = {
    "tracer_name": "tracer_name",
    "tracer_radionuclide": "tracer_radionuclide",
    "injected_radioactivity": "injected_radioactivity",
    "injected_radioactivity_units": "injected_radioactivity_units",
    "injected_mass": "injected_mass",
    "injected_mass_units": "injected_mass_units",
    "specific_radioactivity": "specific_radioactivity",
    "specific_radioactivity_units": "specific_radioactivity_units",
    "mode_of_administration": "mode_of_administration",
    "injection_start": "injection_start",
    "time_zero": "time_zero",
    "scan_start": "scan_start",
    "acquisition_mode": "acquisition_mode",
    "units": "units",
    "body_part": "body_part",
    "recon_method_name": "recon_method_name",
    "recon_filter_type": "recon_filter_type",
}

# PET spec fields typed as numbers. A blank stays None; anything unparseable is
# rejected rather than coerced, so a typo cannot become a silent zero.
_PET_NUMERIC_FIELDS: frozenset[str] = frozenset({
    "injected_radioactivity", "injected_mass", "specific_radioactivity",
    "injection_start", "scan_start",
})


# ---------------------------------------------------------------------------
# Live entity-validation (re-runs on every user edit)
# ---------------------------------------------------------------------------
#
# ``proposed_issues`` carries two kinds of note. *Static* notes describe the
# source and cannot change once scanned (suspected_abort, B0 reroute, fmap
# multi-output, non-image series, user-excluded, mixed-study / collision
# hints). *Managed* notes are the schema entity-validation issues derived from
# (entities, datatype, suffix); these DO change when the user edits a row, so
# the model recomputes them on every edit and splices them back in, leaving the
# static notes untouched. That keeps the valid / warning / error chips live.

# Prefixes (the part before ``": "``) of a managed entity-validation issue.
# Mirror ``schema.validate_entity_set`` rule_ids plus the scan's
# ``build_basename: <exc>`` note. Kept narrow so a static note never matches.
_MANAGED_ISSUE_PREFIXES: tuple[str, ...] = (
    "datatype.",
    "suffix.",
    "entity.",
    "basename.",
    "build_basename",
)
# Managed notes that carry no ``": "`` separator.
_MANAGED_ISSUE_EXACT: frozenset[str] = frozenset({"BIDS_name missing"})

# Sentinel values the scan inserts for a missing required entity (see
# ``cli/scan._placeholder_for_entity``). Live validation treats them as still
# missing so a freshly-scanned row stays flagged until the user supplies a real
# value, and reverts to valid the moment they do.
_ENTITY_PLACEHOLDERS: dict[str, str] = {"task": "TASK", "subject": "TBD"}
_DEFAULT_PLACEHOLDER = "TBD"


def _is_managed_issue(token: str) -> bool:
    """True when ``token`` is a recomputable entity-validation issue."""
    t = token.strip()
    if t in _MANAGED_ISSUE_EXACT:
        return True
    head = t.split(":", 1)[0].strip()
    return head.startswith(_MANAGED_ISSUE_PREFIXES)


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
    # Read-only: the dataset name is owned by the project (the locked output
    # folder). Editing it by hand would point conversion at the wrong folder,
    # so it is informative only and excluded from bulk edits.
    ColumnSpec("dataset",   "dataset",                "plain",    False, 100, df_column="dataset"),
    ColumnSpec("ses",       "ses",                    "mono",     True,  50, df_column="session"),
    ColumnSpec("mod",       "mod",                    "plain",    False, 38, df_column="modality"),
    ColumnSpec("datatype",  "data",                   "plain",    True,  50, df_column="proposed_datatype"),
    ColumnSpec("suffix",    "suffix",                 "plain",    True,  80, df_column="bids_guess_suffix"),
    ColumnSpec("task",      "task",                   "plain",    True,  60, df_column="task"),
    ColumnSpec("run",       "run",                    "mono",     True,  50, df_column="run"),
    ColumnSpec("conf",      "conf",                   "conf",     False, 50, df_column="bids_guess_confidence"),
    # WHERE a row came from, WHAT it was read from, and WHAT the scanner called
    # it. These were three different answers hiding behind one header: the old
    # "sequence / source" column carried only the sequence name, source_folder
    # was correct in the TSV and absent from the table, and format was filled
    # for EEG/MEG alone, so no modality ever showed both an origin and a format.
    ColumnSpec("source_folder", "origin",             "plain",    False, 160, df_column="source_folder"),
    ColumnSpec("format",    "format",                 "mono",     False, 80,  df_column="format"),
    ColumnSpec("sequence",  "sequence",               "mono",     False, 200, df_column="sequence"),
    ColumnSpec("basename",  "predicted basename",     "basename", False, 320, stretch=True, df_column="proposed_basename"),
    # Default-hidden — show via the column-visibility menu.
    ColumnSpec("backend",      "backend",     "mono",  False, 90,  default_visible=False),
    ColumnSpec("source_file",  "source file", "mono",  False, 220, df_column="source_file", default_visible=False),
    ColumnSpec("n_files",      "n_files",     "mono",  False, 60,  df_column="n_files",     default_visible=False),
    ColumnSpec("acq_time",     "acq_time",    "mono",  False, 80,  df_column="acq_time",    default_visible=False),
    ColumnSpec("image_type",   "image_type",  "mono",  False, 120, df_column="image_type",  default_visible=False),
    ColumnSpec("study_date",   "study_date",  "mono",  False, 90,  df_column="study_date",  default_visible=False),
    ColumnSpec("StudyDescription", "study name", "plain", False, 160, df_column="StudyDescription", default_visible=False),
    ColumnSpec("PatientID",    "PatientID",   "mono",  False, 100, df_column="PatientID",   default_visible=False),
    ColumnSpec("GivenName",    "GivenName",   "plain", False, 100, df_column="GivenName",   default_visible=False),
    ColumnSpec("FamilyName",   "FamilyName",  "plain", False, 100, df_column="FamilyName",  default_visible=False),
    ColumnSpec("PatientSex",   "sex",         "mono",  True,  40,  df_column="PatientSex",  default_visible=False),
    ColumnSpec("PatientAge",   "age",         "mono",  True,  50,  df_column="PatientAge",  default_visible=False),
    ColumnSpec("Handedness",   "hand",        "mono",  True,  50,  df_column="Handedness",  default_visible=False),
    ColumnSpec("n_channels",   "n_chan",      "mono",  False, 60,  df_column="n_channels",  default_visible=False),
    ColumnSpec("sfreq",        "sfreq",       "mono",  False, 70,  df_column="sfreq",       default_visible=False),
    ColumnSpec("duration_sec", "duration",    "mono",  False, 70,  df_column="duration_sec", default_visible=False),
    ColumnSpec("line_freq",    "line_freq",   "mono",  True,  60,  df_column="line_freq",   default_visible=False),
    ColumnSpec("montage",      "montage",     "plain", True,  100, df_column="montage",     default_visible=False),
    ColumnSpec("eeg_reference","reference",   "plain", True,  90,  df_column="eeg_reference", default_visible=False),
    ColumnSpec("eeg_ground",   "ground",      "plain", True,  90,  df_column="eeg_ground",  default_visible=False),
    ColumnSpec("companion_files","companion", "plain", True,  140, df_column="companion_files", default_visible=False),
    ColumnSpec("probe_n_nifti","probe_nifti", "mono",  False, 80,  df_column="probe_n_nifti", default_visible=False),
    ColumnSpec("probe_n_vols", "probe_vols",  "mono",  False, 70,  df_column="probe_n_volumes", default_visible=False),
    ColumnSpec("repetition_type","repetition","plain", False, 110, df_column="repetition_type", default_visible=False),
    ColumnSpec("proposed_issues","issues",    "plain", False, 240, df_column="proposed_issues", default_visible=False),
)

# Columns the user cannot hide (lose them and rows become unidentifiable).
MANDATORY_COLUMN_KEYS: frozenset[str] = frozenset({"include", "status", "id"})


# Plain-language description per column, surfaced in the "Manage columns"
# dialog so a non-technical user understands what each one means. Keyed by
# ``ColumnSpec.key``.
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "include":   "Whether this row is converted. Untick to skip the series.",
    "status":    "At-a-glance state badge: ok, warning, error, skipped, non-image, or physio.",
    "id":        "Subject label (sub-XXX) the row converts under.",
    "dataset":   "BIDS dataset slug. Rows with different datasets become sibling BIDS roots.",
    "ses":       "Session label (ses-XXX). Blank when the study has no sessions.",
    "mod":       "Detected modality (mri, eeg, meg, physio, ...).",
    "datatype":  "BIDS datatype folder the row lands in (anat, func, dwi, fmap, eeg, ...).",
    "suffix":    "BIDS suffix (T1w, bold, dwi, physio, ...).",
    "task":      "Task entity (task-XXX) for func / eeg / meg rows.",
    "run":       "Run index (run-N) when a series was repeated.",
    "conf":      "Classifier confidence (0-1) for the predicted datatype + suffix.",
    "source_folder": "Where the row came from: the folder inside the raw tree that holds its files.",
    "format":    "What the row was read from: DICOM, ECAT, FIF, EDF, BrainVision, EEGLAB, CTF.",
    "sequence":  "Scanner sequence name used for classification. Empty for formats that do not name one.",
    "basename":  "Full predicted BIDS filename (without extension).",
    "backend":   "Converter backend that will handle the row: dcm2niix, mne-bids, ecat, or bidsphysio.",
    "source_file": "Source recording path (EEG / MEG). Blank for DICOM rows.",
    "n_files":   "Number of source files in the series.",
    "acq_time":  "Acquisition time from the DICOM / recording header.",
    "image_type": "DICOM ImageType (ORIGINAL / DERIVED, MAGNITUDE / PHASE, ...).",
    "study_date": "Study date, used to cluster longitudinal sessions.",
    "StudyDescription": "DICOM StudyDescription (the study / protocol name). Read-only, MRI only. Not a BIDS entity; shown for awareness. Multiple distinct values in one scan raise a warning.",
    "PatientID": "DICOM PatientID, a key part of subject identity.",
    "GivenName": "Patient given name from the header.",
    "FamilyName": "Patient family name from the header.",
    "PatientSex": "Participant sex (M/F/O). Seeded from the header; editable; written to participants.tsv.",
    "PatientAge": "Participant age in years. Seeded from the header; editable; written to participants.tsv.",
    "Handedness": "Participant handedness (R/L/A). Seeded from the recording header; editable.",
    "n_channels": "EEG / MEG channel count.",
    "sfreq":     "Sampling frequency (Hz) of the recording.",
    "duration_sec": "Recording duration in seconds.",
    "line_freq": "Power-line frequency (Hz) written to the sidecar. Dropdown (50 / 60); blank uses the dataset default.",
    "montage":   "EEG / MEG montage applied on conversion. Dropdown of MNE built-in montages; blank leaves positions as-is.",
    "eeg_reference": "Per-row EEG/iEEG reference electrode override (e.g. Cz). Blank uses the dataset default.",
    "eeg_ground":    "Per-row EEG/iEEG ground electrode override. Blank uses the dataset default.",
    "companion_files": "Already-curated companion files (events / beh / stim) linked to this row and copied into BIDS on convert. Edit via the Properties panel.",
    "probe_n_nifti": "NIfTI files dcm2niix actually produced in a --probe-convert run.",
    "probe_n_vols":  "Volumes dcm2niix actually produced in a --probe-convert run.",
    "repetition_type": "Repeat classification, e.g. suspected_abort (operator restart).",
    "proposed_issues": "Scanner-detected notes: non-image series, B0 reroute, missing entity, ...",
}


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

    # Emitted when a per-row acquisition override changes (set/cleared via
    # :meth:`set_acq_override`). The controller (ConverterPanel) listens and
    # persists the recording-metadata scaffold to disk so the convert verb,
    # which reloads it, sees the edit.
    recordingSpecChanged = pyqtSignal()
    # Emitted whenever the user makes a curation edit (cell / entity / include).
    # The converter uses it to invalidate its redo stack: a fresh edit diverges
    # the timeline, so previously-undone edits can no longer be redone.
    userEdited = pyqtSignal()

    def __init__(
        self,
        df: pd.DataFrame,
        project: Optional[Project] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._df: pd.DataFrame = df.reset_index(drop=True).copy()
        self._project: Optional[Project] = project
        # Dataset-wide recording-metadata defaults. Per-row cells in the
        # inheritance fields show this value (muted) when blank; set via
        # :meth:`set_global_spec` when the scan scaffold loads / the dataset
        # metadata dialog saves.
        self._global_spec: Optional[RecordingMetaSpec] = None

        # Whether a metadata column means anything for a given kind of row,
        # keyed by (column, datatype, suffix). Keyed on the TYPE rather than the
        # row, so a row that changes datatype simply looks up a different entry
        # and the cache never goes stale; it holds at most one entry per column
        # per datatype seen.
        self._applies_cache: dict[tuple[str, str, str], bool] = {}

        # Populate the mirror cells (session / task / run) from each row's
        # ``entities`` JSON on load. A fresh ``bidsmgr-scan`` TSV carries the
        # entities in JSON but leaves the mirror columns blank. Without this
        # pass, editing one mirror cell (e.g. ``task``) runs
        # ``rebuild_from_columns``, which reads the *other* still-blank mirror
        # cells (e.g. ``run``) as "user cleared it" and drops those entities —
        # so changing the task silently deleted ``run``. Mirroring entities →
        # cells up front means a later single-cell edit only ever changes the
        # field the user actually touched; every other entity is preserved.
        # This matches what ``cli/convert.py`` does in memory before reading
        # rows. Idempotent: ``proposed_basename`` is rebuilt from the same
        # entities, so a well-formed scan TSV is unchanged.
        rebuild_from_entities(self._df, in_place=True)

        self._warm_applies_cache()

        if project is not None:
            self._apply_project_overlay(project.state())

        # Normalise ``proposed_issues`` to the live entity-validation state so a
        # freshly-loaded TSV and a user-edited one read identically (load ==
        # edit). Static scan notes are preserved; only the schema-validation
        # segment is recomputed. Pure DataFrame mutation, no signals (we build
        # the row-state cache from the result immediately below).
        for i in range(len(self._df)):
            self._revalidate_row(i)

        # Which rows currently want a name another row also wants. Derived
        # live rather than read from a cell: a clash is a property of the whole
        # table as it stands right now, so anything stored at scan time is
        # stale the moment the user edits an entity, and a red name that stays
        # red after you have fixed it is worse than no warning at all.
        self._colliding: set[int] = set()
        self._recompute_collisions()

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
    # Recording-metadata inheritance (dataset default <- per-row override)
    # ------------------------------------------------------------------

    def set_global_spec(self, spec: Optional[RecordingMetaSpec]) -> None:
        """Set the dataset-wide recording-metadata defaults and refresh display.

        Inherited cells (blank in an inheritance field) now show the new
        default; the underlying DataFrame is untouched. Emits ``dataChanged``
        so every row re-renders with the new inherited values.
        """
        self._global_spec = spec
        if len(self._df) and len(self.COLUMNS):
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._df) - 1, len(self.COLUMNS) - 1),
            )

    def inherited_from(self, row: int, df_col: str) -> str:
        """Which layer supplies this cell's value, as a phrase for a tooltip.

        "Inherited" alone does not tell a user enough to act: the value could
        come from the dataset defaults, from this modality's block, or from a
        sequence template, and which one it is decides where to go and change
        it.
        """
        attr = _INHERITANCE_FIELDS.get(df_col)
        if attr is None or self._global_spec is None:
            return ""
        datatype, suffix = self.effective_datatype_suffix(row)
        field = resolve_attribute(
            self._global_spec, attr, datatype, suffix,
            task=self._raw_cell(row, "task") or None,
        )
        if field is None:
            return ""
        return describe_origin(field.origin, datatype, suffix)

    def _global_default(self, df_col: str, datatype: str = "", row: int = -1) -> str:
        """The inherited value for a cell, resolved through the one chain.

        ``datatype`` selects the per-modality block, and the chain then layers
        any sequence template over it, so the table shows the same answer the
        converter will write rather than only the part of it the dataset block
        knows about.
        """
        attr = _INHERITANCE_FIELDS.get(df_col)
        if attr is None or self._global_spec is None:
            return ""
        suffix = ""
        task = None
        if row >= 0:
            datatype, suffix = self.effective_datatype_suffix(row)
            task = self._raw_cell(row, "task") or None
        field = resolve_attribute(
            self._global_spec, attr, datatype, suffix, task=task,
        )
        val = field.value if field is not None else None
        if val is None:
            return ""
        if isinstance(val, float):
            return str(int(val)) if val == int(val) else str(val)
        return str(val)

    def _raw_cell(self, row: int, df_col: str) -> str:
        if df_col not in self._df.columns or not (0 <= row < len(self._df)):
            return ""
        v = self._df.at[row, df_col]
        return "" if pd.isna(v) else str(v)

    def _warm_applies_cache(self) -> None:
        """Decide the modality-scoped columns up front, once per row TYPE.

        Answering costs a full walk of the schema's rules, and the answer is
        the same for every row of a given datatype and suffix. Left to the
        paint path, the first scroll that reveals these columns pays for one
        walk per distinct type, which reads as a stall part-way across the
        table. Doing it at bind time costs the same work where a wait is
        already expected, and is bounded by the number of distinct types in
        the scan rather than the number of rows.
        """
        if not len(self._df):
            return
        scoped = set(_COLUMN_BIDS_FIELDS) | set(_COLUMN_DATATYPES)
        seen: set[tuple[str, str]] = set()
        for row in range(len(self._df)):
            pair = self.effective_datatype_suffix(row)
            if pair in seen:
                continue
            seen.add(pair)
            for df_col in scoped:
                self.column_applies(row, df_col)

    def column_applies(self, row: int, df_col: str) -> bool:
        """True when ``df_col`` means anything for this row's datatype.

        A metadata column that does not apply is not merely empty, it is
        inapplicable: an anatomical scan has no power line frequency, and
        offering to inherit one is offering nonsense. Columns not covered by
        either map apply everywhere, which keeps the identity and provenance
        columns untouched.
        """
        datatypes = _COLUMN_DATATYPES.get(df_col)
        fields = _COLUMN_BIDS_FIELDS.get(df_col)
        if datatypes is None and fields is None:
            return True
        datatype, suffix = self.effective_datatype_suffix(row)
        # Asked once per cell per repaint, and the answer depends only on the
        # column and the row's type, not on the row. Without this a table of
        # any size re-derives the same handful of answers thousands of times.
        cache_key = (df_col, datatype, suffix)
        cached = self._applies_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._compute_applies(df_col, datatype, suffix, datatypes, fields)
        self._applies_cache[cache_key] = result
        return result

    def effective_datatype_suffix(self, row: int) -> tuple[str, str]:
        """``(datatype, suffix)`` for the row, falling back to the guess.

        A blank ``proposed_datatype`` does not mean "undecided". Scanner
        derivatives carry one: a Siemens scout, a PhoenixZIPReport or a TENSOR
        map is excluded from conversion and left with no proposed datatype,
        while the classifier's guess still records what it is (``anat``,
        ``discard``, ``derivatives``). Reading only the proposed value made
        those rows look like blank slates, which is how a bulk edit wrote a
        power line frequency onto a localiser.
        """
        datatype, suffix = self.datatype_suffix(row)
        if not datatype and "bids_guess_datatype" in self._df.columns:
            if 0 <= row < len(self._df):
                raw = self._df.at[row, "bids_guess_datatype"]
                datatype = "" if pd.isna(raw) else str(raw)
        return datatype, suffix

    def _compute_applies(self, df_col, datatype, suffix, datatypes, fields) -> bool:
        if not datatype:
            # Nothing anywhere says what this row is. A metadata field cannot
            # be meaningfully set on it either, and it becomes editable the
            # moment the row is classified.
            return False
        if datatypes is not None:
            return datatype in datatypes
        if datatype not in set(schema_mod.list_datatypes()):
            # ``discard`` and ``derivatives`` are BIDS Manager routing labels,
            # not datatypes. The standard declares nothing for them, and the
            # schema lookup below would find no suffixes and wave the row
            # through.
            return False

        # Ask about the exact pair only when the two agree. Mid-edit they often
        # do not: a user who switches a row's datatype before its suffix leaves
        # an eeg row still carrying "bold", and asking about eeg/bold returns
        # False for everything, which would silently refuse the very edit that
        # fixes the row. Falling back to "does any suffix of this datatype take
        # the field" keeps the answer stable while the row is in flux.
        try:
            suffixes = [suffix] if suffix in schema_mod.list_suffixes(datatype) \
                else list(schema_mod.list_suffixes(datatype))
        except Exception:
            suffixes = [suffix] if suffix else []
        if not suffixes:
            return False
        return any(
            schema_mod.field_applies(name, datatype, sfx)
            for sfx in suffixes
            for name in fields
        )

    def needs_per_row_answer(self, row: int, df_col: str) -> bool:
        """True when the dataset says this field differs per recording.

        The cell then wants a prompt rather than a value: the dataset-wide
        answer is VARIES, which is a statement about where the answer lives,
        not an answer. Distinct from blank, which means nobody has said
        anything, and from not-applicable, which means the datatype has no
        such field.
        """
        if df_col not in _INHERITANCE_FIELDS or not self.column_applies(row, df_col):
            return False
        if self._raw_cell(row, df_col).strip():
            return False  # answered for this row already
        return is_varies(self._global_default(df_col, self.effective_datatype_suffix(row)[0], row))

    def is_inherited(self, row: int, df_col: str) -> bool:
        """True when an inheritance-field cell is blank and a default exists."""
        if df_col not in _INHERITANCE_FIELDS:
            return False
        if not self.column_applies(row, df_col):
            return False
        if is_varies(self._global_default(df_col, self.effective_datatype_suffix(row)[0], row)):
            # Not inherited: there is nothing to inherit yet.
            return False
        return not self._raw_cell(row, df_col).strip() and bool(
            self._global_default(df_col, self.effective_datatype_suffix(row)[0], row)
        )

    def effective_value(self, row: int, df_col: str) -> str:
        """The per-row override (cell) when set, else the inherited default.

        A VARIES default resolves to nothing: it says the answer differs per
        recording, so until this row supplies one there is no value here.
        """
        cell = self._raw_cell(row, df_col).strip()
        if cell:
            return cell
        default = self._global_default(df_col, self.effective_datatype_suffix(row)[0], row)
        return "" if is_varies(default) else default

    # ------------------------------------------------------------------
    # Per-row acquisition overrides (recording-metadata scaffold-backed)
    # ------------------------------------------------------------------
    #
    # These device / institution fields are NOT TSV columns; they live in the
    # scaffold's ``overrides[row_id]`` and inherit from ``defaults`` the same
    # way the TSV-backed inheritance fields do. The convert step's
    # ``resolve_effective`` already merges them and the enrichment fixup writes
    # them into the EEG/MEG sidecar.

    def global_spec(self) -> Optional[RecordingMetaSpec]:
        """The live recording-metadata spec (defaults + per-row overrides)."""
        return self._global_spec

    def acq_default(self, field: str, row: int = -1) -> str:
        """What this field inherits, resolved through the one chain.

        ``row`` supplies the datatype, suffix and task, without which the chain
        cannot pick the per-modality block or the sequence template. Reading
        only ``spec.defaults``, as this used to, meant the panel showed the
        shared answer while the converter wrote the modality's or the
        template's, which is the disconnect this chain exists to close.
        """
        attr = _ACQ_OVERRIDE_FIELDS.get(field)
        if attr is None or self._global_spec is None:
            return ""
        datatype = suffix = ""
        task = None
        if row >= 0:
            datatype, suffix = self.effective_datatype_suffix(row)
            task = self._raw_cell(row, "task") or None
        # No row id is passed, so the chain stops above the row layers: this
        # is what the field INHERITS, and acq_effective adds the override.
        resolved = resolve_attribute(
            self._global_spec, attr, datatype, suffix, task=task,
        )
        val = resolved.value if resolved is not None else None
        return "" if val is None else str(val)

    def acq_inherited_from(self, row: int, field: str) -> str:
        """Which layer supplies this panel field, as a readable phrase."""
        attr = _ACQ_OVERRIDE_FIELDS.get(field)
        if attr is None or self._global_spec is None:
            return ""
        datatype, suffix = self.effective_datatype_suffix(row)
        resolved = resolve_attribute(
            self._global_spec, attr, datatype, suffix,
            task=self._raw_cell(row, "task") or None,
        )
        if resolved is None:
            return ""
        return describe_origin(resolved.origin, datatype, suffix)

    def _row_override(self, row: int) -> Optional[AcquisitionSpec]:
        if self._global_spec is None:
            return None
        return self._global_spec.overrides.get(self.row_id(row))

    def acq_override(self, row: int, field: str) -> str:
        """The raw per-row override for an acquisition field (``""`` if unset)."""
        attr = _ACQ_OVERRIDE_FIELDS.get(field)
        if attr is None:
            return ""
        over = self._row_override(row)
        if over is None:
            return ""
        val = getattr(over, attr, None)
        return "" if val is None else str(val)

    def acq_effective(self, row: int, field: str) -> str:
        """Per-row override when set, else what the chain resolves for it."""
        ov = self.acq_override(row, field)
        return ov if ov else self.acq_default(field, row)

    def acq_is_inherited(self, row: int, field: str) -> bool:
        """True when no per-row override is set and something is inherited."""
        return not self.acq_override(row, field) and bool(self.acq_default(field, row))

    def set_acq_override(self, row: int, field: str, value: str) -> bool:
        """Set or clear a per-row acquisition override in the scaffold spec.

        Writing a value equal to the dataset default (or an empty value) clears
        the override so the row inherits again. Mutates the live spec and emits
        :attr:`recordingSpecChanged` (the controller persists the scaffold) plus
        ``dataChanged`` for the row. Returns ``True`` if anything changed.
        """
        attr = _ACQ_OVERRIDE_FIELDS.get(field)
        if attr is None or not (0 <= row < len(self._df)):
            return False
        if self._global_spec is None:
            self._global_spec = RecordingMetaSpec()

        new_val = (value or "").strip()
        # Equal to the dataset default -> inherit (store no override).
        if new_val == self.acq_default(field):
            new_val = ""

        if new_val == self.acq_override(row, field):
            return False

        rid = self.row_id(row)
        overrides = dict(self._global_spec.overrides)
        current = overrides.get(rid) or AcquisitionSpec()
        updated = current.model_copy(update={attr: (new_val or None)})

        if updated == AcquisitionSpec():
            overrides.pop(rid, None)
        else:
            overrides[rid] = updated
        self._global_spec = self._global_spec.model_copy(update={"overrides": overrides})

        self.recordingSpecChanged.emit()
        self.refresh_row(row)
        return True

    # ------------------------------------------------------------------
    # Per-row sidecar fields, in the schema's own vocabulary
    # ------------------------------------------------------------------
    #
    # The block above models a dozen acquisition attributes. This one carries
    # any field the schema declares for the row's file, so the properties panel
    # can offer exactly what the dataset dialog offers, one scope down. Both
    # read the same chain, so a value set anywhere shows everywhere it applies.

    # A few sidecar fields are also inventory columns, because the user needs
    # them in the table to sort and bulk-edit by. Those keep the cell as their
    # one home: a form that wrote a second copy elsewhere would show one value
    # while the table showed another, and the converter would have to pick.
    _CELL_BACKED_FIELDS: dict[str, tuple[str, str]] = {
        "PowerLineFrequency": ("line_freq", "power_line_freq"),
        "EEGReference": ("eeg_reference", "eeg_reference"),
        "iEEGReference": ("eeg_reference", "eeg_reference"),
        "EEGGround": ("eeg_ground", "eeg_ground"),
        "iEEGGround": ("eeg_ground", "eeg_ground"),
    }

    def _row_context(self, row: int) -> tuple[str, str, Optional[str]]:
        datatype, suffix = self.effective_datatype_suffix(row)
        return datatype, suffix, (self._raw_cell(row, "task") or None)

    def _row_cell_values(self, row: int) -> dict:
        """This row's own cells, in the chain's vocabulary.

        The same three the conversion passes, so the panel's picture of what
        will be written is the one that gets written.
        """
        out: dict = {}
        for column, attr in {
            "eeg_reference": "eeg_reference",
            "eeg_ground": "eeg_ground",
            "line_freq": "power_line_freq",
        }.items():
            value = self._raw_cell(row, column)
            if value:
                out[attr] = value
        return out

    def resolved_sidecar(self, row: int, *, with_row: bool = True) -> dict:
        """Every stated field for this row's sidecar, and which layer said it.

        ``with_row=False`` stops above this recording, which is what the field
        would say if the row had never been touched. That is the comparison that
        decides whether an answer is worth storing.
        """
        if self._global_spec is None or not (0 <= row < len(self._df)):
            return {}
        datatype, suffix, task = self._row_context(row)
        if not datatype:
            return {}
        return resolve_sidecar_fields(
            self._global_spec, datatype, suffix or datatype,
            row_id=self.row_id(row) if with_row else "", task=task,
            row_values=self._row_cell_values(row) if with_row else None,
        )

    def row_answered(self, row: int) -> dict:
        """What the conversion will fill in for THIS recording, by BIDS name.

        Two sources, both about this one file. The scan asked the recording the
        same questions the converter will ask and stashed the answers; and the
        row's own entities settle a field or two by themselves, a task label
        being the obvious one, since the converter writes whatever the row says.
        """
        if not (0 <= row < len(self._df)):
            return {}

        answered: dict = {}
        # Written by the scan into the scaffold, because the inventory drops it:
        # it is derived rather than curated, and a column of JSON in a table
        # people read helps nobody.
        if self._global_spec is not None:
            stored = self._global_spec.row_preview.get(self.row_id(row))
            if isinstance(stored, dict):
                answered.update(stored)
        raw = self._raw_cell(row, "_derived_fields")
        if raw:
            try:
                measured = json.loads(raw)
            except (ValueError, TypeError):
                measured = {}
            if isinstance(measured, dict):
                answered.update(measured)

        task = (self.entities(row) or {}).get("task") or self._raw_cell(row, "task")
        if task:
            answered["TaskName"] = task
        return answered

    def row_template(self, row: int) -> dict:
        """What this one recording states, by BIDS field name."""
        if self._global_spec is None or not (0 <= row < len(self._df)):
            return {}
        return dict(self._global_spec.row_templates.get(self.row_id(row), {}))

    def sidecar_origin(self, row: int, name: str) -> str:
        """Where this field's current answer comes from, as a readable phrase."""
        resolved = self.resolved_sidecar(row).get(name)
        if resolved is None:
            return ""
        datatype, suffix, task = self._row_context(row)
        return describe_origin(resolved.origin, datatype, suffix, task or "")

    def set_row_template_field(self, row: int, name: str, value) -> bool:
        """State a sidecar field for this recording alone, or stop stating it.

        An answer equal to what the row already inherits is not stored: keeping
        it would freeze today's inherited value into this row, so a later change
        to the dataset default would silently skip this one recording.
        """
        if not (0 <= row < len(self._df)):
            return False

        backing = self._CELL_BACKED_FIELDS.get(name)
        if backing is not None:
            column = backing[0]
            if column in self._df.columns:
                return self.bulk_set(
                    [row], column, "" if value in (None, "", [], {}) else str(value),
                )

        if self._global_spec is None:
            self._global_spec = RecordingMetaSpec()

        rid = self.row_id(row)
        current = dict(self._global_spec.row_templates.get(rid, {}))
        blank = value in (None, "", [], {})

        if not blank:
            inherited = self.resolved_sidecar(row, with_row=False).get(name)
            if inherited is not None and inherited.value == value:
                blank = True

        if blank and name not in current:
            return False
        if not blank and current.get(name) == value:
            return False

        if blank:
            current.pop(name, None)
        else:
            current[name] = value

        templates = dict(self._global_spec.row_templates)
        if current:
            templates[rid] = current
        else:
            templates.pop(rid, None)
        self._global_spec = self._global_spec.model_copy(
            update={"row_templates": templates}
        )
        self.recordingSpecChanged.emit()
        self.refresh_row(row)
        return True

    # ------------------------------------------------------------------
    # PET per-row overrides (same inheritance contract as acquisition)
    # ------------------------------------------------------------------

    def pet_default(self, field: str) -> str:
        """The dataset default for a PET field, as a string."""
        attr = _PET_OVERRIDE_FIELDS.get(field)
        if attr is None or self._global_spec is None:
            return ""
        return _as_text(getattr(self._global_spec.pet_defaults, attr, None))

    def pet_override(self, row: int, field: str) -> str:
        """The raw per-row PET override (``""`` if unset)."""
        attr = _PET_OVERRIDE_FIELDS.get(field)
        if attr is None or self._global_spec is None:
            return ""
        over = self._global_spec.pet_overrides.get(self.row_id(row))
        if over is None:
            return ""
        return _as_text(getattr(over, attr, None))

    def pet_effective(self, row: int, field: str) -> str:
        """Per-row override when set, else the inherited dataset default."""
        ov = self.pet_override(row, field)
        return ov if ov else self.pet_default(field)

    def pet_is_inherited(self, row: int, field: str) -> bool:
        return not self.pet_override(row, field) and bool(self.pet_default(field))

    def set_pet_override(self, row: int, field: str, value: str) -> bool:
        """Set or clear a per-row PET override in the scaffold spec.

        Mirrors :meth:`set_acq_override`: writing the dataset default (or a
        blank) clears the override so the row inherits again.
        """
        attr = _PET_OVERRIDE_FIELDS.get(field)
        if attr is None or not (0 <= row < len(self._df)):
            return False
        if self._global_spec is None:
            self._global_spec = RecordingMetaSpec()

        new_val = (value or "").strip()
        if new_val == self.pet_default(field):
            new_val = ""
        if new_val == self.pet_override(row, field):
            return False

        typed: object = new_val or None
        if typed is not None and field in _PET_NUMERIC_FIELDS:
            try:
                typed = float(new_val)
            except ValueError:
                # Not a number: refuse rather than store a coerced value that
                # would look deliberate in the sidecar.
                return False

        rid = self.row_id(row)
        overrides = dict(self._global_spec.pet_overrides)
        current = overrides.get(rid) or PetAcquisitionSpec()
        updated = current.model_copy(update={attr: typed})

        if updated == PetAcquisitionSpec():
            overrides.pop(rid, None)
        else:
            overrides[rid] = updated
        self._global_spec = self._global_spec.model_copy(
            update={"pet_overrides": overrides})

        self.recordingSpecChanged.emit()
        self.refresh_row(row)
        return True

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
            # A column that means nothing for this row's datatype refuses the
            # edit rather than accepting a value that would never be written.
            if spec.df_column and not self.column_applies(index.row(), spec.df_column):
                return f
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

        # Hovering any cell of a flagged row surfaces the scanner's
        # ``proposed_issues`` (e.g. the "non-image series" reason) so the
        # user sees *why* a row is highlighted without unhiding the issues
        # column. Newline-split the `` | ``-joined notes for readability.
        if role == Qt.ItemDataRole.ToolTipRole:
            # An inherited metadata cell says where its value came from, so a
            # user knows which level to go and change.
            spec_col = self.COLUMNS[col].df_column
            if spec_col in _INHERITANCE_FIELDS and self.column_applies(row, spec_col):
                if self.needs_per_row_answer(row, spec_col):
                    return (
                        "The dataset says this differs per recording.\n"
                        "Type the value for this one."
                    )
                if self.is_inherited(row, spec_col):
                    where = self.inherited_from(row, spec_col)
                    if where:
                        return f"Inherited from {where}.\nType here to override it."
            if row in self._colliding:
                # Derived, not stored, so it disappears the moment the clash is
                # resolved rather than lingering on a row that is now fine.
                from ...inventory.name_collisions import DUPLICATE_ISSUE

                return DUPLICATE_ISSUE
            if "proposed_issues" in self._df.columns:
                issues = str(self._df.at[row, "proposed_issues"] or "").strip()
                if issues:
                    return issues.replace(" | ", "\n")
            return None

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

        # Inheritance fields: a blank cell shows the dataset default (and the
        # delegate paints it muted via INHERITED_ROLE).
        if spec.df_column and not self.column_applies(row, spec.df_column):
            # Blank, not "—": the em dash means "inherited and empty", which
            # would imply the field could be filled in here. It cannot.
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
                return ""
            if role == INHERITED_ROLE:
                return False

        if spec.df_column in _INHERITANCE_FIELDS:
            if role == INHERITED_ROLE:
                return self.is_inherited(row, spec.df_column)
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
                eff = self.effective_value(row, spec.df_column)
                if eff:
                    return eff
                if role == Qt.ItemDataRole.DisplayRole:
                    # Three empties that mean different things. "varies" asks
                    # for an answer, the em dash reports an empty inherited
                    # default, and a truly blank cell says nothing was stated.
                    return "varies?" if self.needs_per_row_answer(
                        row, spec.df_column,
                    ) else "—"
                return ""

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
        if not self.column_applies(row, df_col):
            return False
        new_str = "" if value is None else str(value)
        # Inheritance fields: writing a value equal to the dataset default
        # clears the cell back to "inherited" instead of storing a redundant
        # per-row override (the convert/display layers re-resolve the default).
        if df_col in _INHERITANCE_FIELDS and new_str == self._global_default(
            df_col, self.effective_datatype_suffix(row)[0]
        ):
            new_str = ""
        old_str = "" if pd.isna(self._df.at[row, df_col]) else str(self._df.at[row, df_col])
        if new_str == old_str:
            return False

        # Order: record the event first (durable), then mutate.
        self._record_edit(
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
        ``f"row-{row}"``. Delegates to the shared, Qt-free
        :func:`bidsmgr.project.orchestration.row_id` so GUI-recorded edits
        replay identically in the CLI (``--project``).
        """
        from ...project.orchestration import row_id as _row_id
        return _row_id(self._df, row)

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

    def _record_edit(self, event) -> None:
        """Append a user-edit event to the project (if any) and signal it."""
        if self._project is not None:
            self._project.append(event)
        self.userEdited.emit()

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

        self._record_edit(
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
        # "dataset" is intentionally excluded: it is owned by the project /
        # locked output folder and must never be changed by hand (see ColumnSpec).
        "ses",
        "task",
        "run",
        "datatype",
        "suffix",
        "line_freq",
        "montage",
        "eeg_reference",
        "eeg_ground",
        "PatientSex",
        "PatientAge",
        "Handedness",
        "companion_files",
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
            self._record_edit(
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

    def _recompute_collisions(self) -> bool:
        """Refresh which rows clash. True if the answer changed.

        Cheap enough to run on every edit: it is one pass building a dict of
        names.
        """
        from ...inventory.name_collisions import find_collisions

        try:
            found = find_collisions(self._df)
        except Exception:  # noqa: BLE001 - a paint hint must never raise
            found = {}
        colliding = {idx for rows in found.values() for idx in rows}
        changed = colliding != self._colliding
        self._colliding = colliding
        return changed

    def colliding_rows(self) -> set:
        """Rows whose BIDS name another included row also wants."""
        return set(self._colliding)

    def refresh_row(self, row: int) -> None:
        """Re-derive cached state for ``row`` and notify the view.

        Call after any mutation that could affect row state (include
        toggle, entity edit). Re-runs live entity-validation first so the
        warn / err / valid state reflects the edit, then emits
        ``dataChanged`` for every column of the row so per-row tints,
        badges, and mirror-cell text refresh together.
        """
        if not (0 <= row < len(self._df)):
            return
        self._revalidate_row(row)

        # Renaming ONE row can clear the clash on ANOTHER: two rows wanted the
        # same name and now they do not. Refreshing only the edited row left
        # its partner painted red with nothing wrong with it, which reads as
        # the warning being stuck.
        if self._recompute_collisions():
            for i in range(len(self._df)):
                self._row_states[i] = self._derive_row_state(i)
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
            )
            return

        self._row_states[row] = self._derive_row_state(row)
        left = self.index(row, 0)
        right = self.index(row, self.columnCount() - 1)
        self.dataChanged.emit(left, right)

    def revalidate_all(self) -> None:
        """Recompute every row's validation issues + state, then repaint.

        Forces a full live re-validation: for each row the schema entity checks
        run against its current entities / datatype / suffix, the managed
        segment of ``proposed_issues`` is refreshed (static scan notes kept),
        and the cached row state is rebuilt. Emits one ``dataChanged`` over the
        whole table so delegates repaint and the controller's chip / preview /
        stats listeners recompute. Backs the Converter's "Re-validate" button so
        the valid / warning / error tallies are guaranteed current after any
        batch of edits, no matter which edit path produced them.
        """
        if self.rowCount() == 0:
            return
        for i in range(len(self._df)):
            self._revalidate_row(i)
        self._recompute_collisions()
        for i in range(len(self._df)):
            self._row_states[i] = self._derive_row_state(i)
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1),
        )

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

        Replays the saved curation edits (entity / cell / include overrides)
        onto the table so reopening a project shows the same thing as before.
        Delegates to the shared, Qt-free
        :func:`bidsmgr.project.orchestration.apply_project_state` so the CLI
        (``--project``) replays edits identically.
        """
        from ...project.orchestration import apply_project_state
        apply_project_state(self._df, state)

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
        self._record_edit(
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

    # Datatypes MneBidsBackend claims, mirroring its own ``can_handle``.
    _MNE_BIDS_DATATYPES = frozenset({"eeg", "meg", "ieeg", "nirs"})

    def _backend(self, row: int) -> str:
        """Which converter backend will handle this row.

        Mirrors ``converter.registry`` in the same order the dispatcher uses:
        physio, then ECAT, then mne-bids, then dcm2niix as the broad fallback.

        This used to read "any row with a source file goes to mne-bids", which
        is not what the registry does. An ECAT PET scan has a source file and
        is converted by EcatDirect through nibabel, so the table named the
        wrong tool for every ECAT row. The registry decides ECAT by reading the
        file's signature, which is far too expensive to do per cell per
        repaint; the ``format`` column records the same fact at scan time.
        """
        suffix = ""
        if "bids_guess_suffix" in self._df.columns:
            suffix = str(self._df.at[row, "bids_guess_suffix"] or "")
        if suffix == "physio":
            return "bidsphysio"

        datatype, _ = self.effective_datatype_suffix(row)
        if datatype == "pet" and self._raw_cell(row, "format").strip().upper() == "ECAT":
            return "ecat"
        if datatype in self._MNE_BIDS_DATATYPES:
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

    # -------- live entity-validation --------

    def _live_entity_issues(self, row: int) -> list[str]:
        """Schema entity-validation issues for the row's *current* entities.

        Returns the same ``"<rule_id>: <message>"`` strings the scan emits, so
        they slot straight back into ``proposed_issues``. Skipped rows and rows
        without a datatype/suffix produce none. Placeholder sentinels are
        treated as missing (see :data:`_ENTITY_PLACEHOLDERS`).
        """
        if not (0 <= row < len(self._df)):
            return []
        if not self._read_include(row):
            return []
        # A row the scanner marked skip is not converted; don't validate it.
        if "bids_guess_skip" in self._df.columns:
            v = self._df.at[row, "bids_guess_skip"]
            if isinstance(v, str):
                if v.strip() in ("1", "true", "True"):
                    return []
            elif not pd.isna(v) and bool(v):
                return []

        datatype, suffix = self.datatype_suffix(row)
        if not datatype or not suffix:
            return []
        # Derivatives use a bespoke path that the schema entity validator
        # rejects wholesale; leave their notes to the scanner.
        if datatype == "derivatives":
            return []

        ents = self.entities(row)
        cleaned = {
            k: v
            for k, v in ents.items()
            if v and v != _ENTITY_PLACEHOLDERS.get(k, _DEFAULT_PLACEHOLDER)
        }
        try:
            verdicts = schema_mod.validate_entity_set(cleaned, datatype, suffix)
        except Exception:  # pragma: no cover - schema lookups are defensive
            log.debug("live entity validation failed for row %d", row, exc_info=True)
            return []
        return [
            f"{v.rule_id}: {v.message}"
            for v in verdicts
            if v.severity is schema_mod.Severity.ERROR
        ]

    def _revalidate_row(self, row: int) -> bool:
        """Recompute the managed (entity-validation) segment of ``proposed_issues``.

        Static scan notes are preserved verbatim; only the schema-validation
        issues are replaced with a fresh recompute of the row's current
        entities. Mutates the DataFrame in place (no signal) and returns
        ``True`` when the cell changed. Callers emit ``dataChanged`` via
        :meth:`refresh_row`.
        """
        if not (0 <= row < len(self._df)):
            return False
        if "proposed_issues" not in self._df.columns:
            return False
        old_val = "" if pd.isna(self._df.at[row, "proposed_issues"]) else str(
            self._df.at[row, "proposed_issues"]
        )
        existing = [t.strip() for t in old_val.split(" | ") if t.strip()]
        static = [t for t in existing if not _is_managed_issue(t)]
        fresh = self._live_entity_issues(row)
        # Validation issues first (matches the scan's ordering), then static.
        new_val = " | ".join(fresh + static)
        if new_val == old_val:
            return False
        self._df.at[row, "proposed_issues"] = new_val
        return True

    # -------- per-row state --------

    def _derive_row_state(self, row: int) -> str:
        """One of ``""``, ``"warn"``, ``"err"``, ``"skip"``.

        Drives the row tint applied by every delegate. Selection is
        applied by the view at paint time, not stored here.
        """
        # Non-image DERIVED objects (no pixel data; e.g. a Siemens TENSOR
        # map) are flagged by the scanner via the ``non-image series``
        # token in ``proposed_issues`` (see ``cli/scan.NONIMAGE_ISSUE_TOKEN``).
        # They are excluded from conversion (include=0) but must still read
        # as a deliberate, highlighted "not an image" state rather than a
        # de-emphasised skip — so this check wins over the include check.
        if "proposed_issues" in self._df.columns:
            issues_l = str(self._df.at[row, "proposed_issues"] or "").lower()
            if "non-image series" in issues_l:
                return "noimg"

        # Two included recordings wanting one BIDS name means one overwrites
        # the other, and conversion refuses until it is resolved. An error
        # rather than a warning, and checked before the include and skip states
        # because the danger comes precisely from the row being included.
        if row in self._colliding:
            return "err"

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
        if state == "noimg":
            return "noimg"
        if state == "skip":
            return "skip"
        if state == "err":
            return "err"
        if state == "warn":
            return "warn"
        if self._backend(row) == "bidsphysio":
            return "phys"
        return "ok"


__all__ = [
    "COLUMNS",
    "COLUMN_DESCRIPTIONS",
    "ColumnSpec",
    "InventoryTableModel",
    "MANDATORY_COLUMN_KEYS",
]

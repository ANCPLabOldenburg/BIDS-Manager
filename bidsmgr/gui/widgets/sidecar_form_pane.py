"""Center pane of the Editor — schema-aware sidecar form (read-only).

Visual reference: ``inspector_proto/proto.py`` ``EditorView._center_pane``
(lines 879-940), simplified for the bidsmgr port:

* No multi-file tab strip — the pane shows the **single file** the
  user clicked in the tree (peer files are reached by clicking them in
  the tree, not auto-opened here).
* JSON content is always loaded **directly from disk** the moment a
  file is bound; validation only adds colour-coding (REQUIRED red,
  RECOMMENDED amber, OPTIONAL grey, DEPRECATED strikethrough) and
  surfaces schema fields that are missing. Without a
  :class:`ValidationReport` we still show the JSON's actual keys with
  OPTIONAL styling.

Three stacked sections:

1. **Schema legend** — small colour swatches for the four
   :class:`bidsmgr.editor.types.FieldLevel` values plus a context
   label like ``schema · anat/T1w`` (or the filename when no verdict
   is available).
2. **Form** — one :class:`SidecarRow` per field, sorted
   ``required → recommended → optional → deprecated``.
3. **Status footer** — the relative path + a summary count.

Theme handling: every palette-coloured widget here is **QSS-driven**
(same pattern as the converter's ``QPlainTextEdit#dock-log``). The
legend swatches carry per-level object names (``legend-swatch-req``
etc.); the footer + its labels use ``sidecar-footer*`` object names;
all colours live in ``theme.qss``. A dark↔light swap is therefore a
plain global ``setStyleSheet`` re-apply by the theme manager — no
per-widget repaint dance, no widget reconstruction.
:meth:`repaint_for_palette` is kept as a no-op for API compatibility
with the existing cascade.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import copy

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...editor.types import FieldLevel, FileVerdict, SidecarField
from .json_tree_view import JsonTreeView
from .primitives import PaneHeader
from .sidecar_row import SidecarRow

log = logging.getLogger(__name__)

# Sentinel used to distinguish "not present in JSON" from "present with
# value None" when checking whether a commit is a no-op.
_UNSET: Any = object()

# Which fields both views render. The BIDS form and the Tree view used to
# answer two different questions without saying so: the form offered every
# field the schema declares for the datatype and the tree rendered the file,
# so the same sidecar showed 130 rows in one and 90 in the other. Both now
# render the same set and this names it.
SCOPE_ALL = "all"          # what the file states + what the standard declares
SCOPE_PRESENT = "present"  # only what the file states
SCOPE_ABSENT = "absent"    # only what is declared and missing
FIELD_SCOPES = (SCOPE_ALL, SCOPE_PRESENT, SCOPE_ABSENT)


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------


_LEVEL_CODE: dict[FieldLevel, str] = {
    FieldLevel.REQUIRED:    "req",
    FieldLevel.RECOMMENDED: "rec",
    FieldLevel.OPTIONAL:    "opt",
    FieldLevel.DEPRECATED:  "dep",
}

_LEVEL_SORT: dict[FieldLevel, int] = {
    FieldLevel.REQUIRED:    0,
    FieldLevel.RECOMMENDED: 1,
    FieldLevel.OPTIONAL:    2,
    FieldLevel.DEPRECATED:  3,
}


def _python_value_kind(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, (int, float)):
        return "number"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        return "array"
    if isinstance(val, dict):
        return "object"
    return "string"


def _schema_specs(verdict, current_file) -> dict:
    """What the standard declares about each field of this kind of file.

    Keyed by field name, so a row can be given a control that matches the
    field's actual type, vocabulary and unit. Returns an empty mapping rather
    than raising when the file is not a kind the schema knows: a user is
    still entitled to keep a key the standard has never heard of, and it gets
    the plain text box.
    """
    from ...metadata.template_plan import as_template_field
    from ...schema import dataset_description_fields, sidecar_fields

    try:
        if current_file is not None and \
                current_file.name == "dataset_description.json":
            specs = dataset_description_fields()
        elif verdict is not None and verdict.datatype and verdict.suffix:
            specs = sidecar_fields(verdict.datatype, verdict.suffix)
        else:
            return {}
    except Exception as exc:  # noqa: BLE001 - a form must still render
        log.debug("no schema specs available: %s", exc)
        return {}
    # Converted to the shape the shared control builder takes, which is
    # also the shape the metadata templates use.
    return {spec.name: as_template_field(spec) for spec in specs}


def _format_value(field: SidecarField) -> tuple[str, str]:
    """Map a :class:`SidecarField` to ``(display_text, value_kind)``.

    The returned ``value_kind`` is the **validator's full vocabulary**
    (``"string"``, ``"number"``, ``"bool"``, ``"array"``, ``"object"``,
    ``"null"``, ``"todo"``, ``"missing"``), not a display-only short
    code — :class:`SidecarRow` accepts both vocabularies for the
    read-only label and needs the full one to pick an editor widget
    in :data:`editable` mode.
    """
    vk = field.value_kind
    if vk == "todo":
        return (
            str(field.value) if field.value is not None else "TODO",
            "todo",
        )
    if vk == "missing":
        return ("(missing)", "missing")
    if vk == "number":
        if field.value is None:
            return ("", "number")
        return (json.dumps(field.value), "number")
    if vk == "bool":
        return (json.dumps(bool(field.value)), "bool")
    if vk == "null":
        return ("null", "null")
    if vk in ("array", "object"):
        try:
            return (json.dumps(field.value, ensure_ascii=False), vk)
        except (TypeError, ValueError):
            return (repr(field.value), vk)
    if field.value is None:
        return ("", "string")
    return (str(field.value), "string")


def find_peer_files(path: Path) -> list[Path]:
    """Return sibling files that share a BIDS stem with ``path``.

    Kept as a utility even though :class:`SidecarFormPane` no longer
    uses it — future features (e.g. a "jump to peer" context menu)
    can call it directly.
    """
    def _stem(p: Path) -> str:
        name = p.name
        if name.endswith(".nii.gz"):
            return name[: -len(".nii.gz")]
        return p.stem

    parent = path.parent
    target_stem = _stem(path)
    try:
        peers = [
            child for child in parent.iterdir()
            if child.is_file() and _stem(child) == target_stem
        ]
    except (PermissionError, FileNotFoundError):
        return [path]
    if path not in peers:
        peers.append(path)

    _PRIORITY = {".json": 0, ".tsv": 1, ".nii.gz": 2}

    def _ext_rank(p: Path) -> tuple[int, str]:
        name = p.name.lower()
        for ext, rank in _PRIORITY.items():
            if name.endswith(ext):
                return (rank, name)
        return (10, name)

    peers.sort(key=_ext_rank)
    return peers


def _load_json_ordered(path: Path) -> Optional[OrderedDict[str, Any]]:
    """Read ``path`` into an :class:`OrderedDict` (key order preserved).

    Returns ``None`` if the file isn't readable / isn't valid JSON /
    isn't a top-level object / isn't UTF-8 (e.g. a binary ``.nii.gz``
    accidentally passed in). Used by the editable form to round-trip
    saves without losing the original field order.
    """
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.debug("could not read JSON %s: %s", path, exc)
        return None
    if not isinstance(data, OrderedDict):
        return None
    return data


def _load_json_fields_from_disk(path: Path) -> list[SidecarField]:
    """Read ``path`` and return one OPTIONAL SidecarField per top-level key.

    Used when we have no :class:`FileVerdict` to drive level info.
    Every field is OPTIONAL because without validation we don't claim
    required-vs-optional.
    """
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.debug("could not read JSON %s: %s", path, exc)
        return []
    if not isinstance(data, dict):
        return []
    fields: list[SidecarField] = []
    for key, val in data.items():
        if isinstance(val, str) and val.strip().upper() == "TODO":
            kind = "todo"
        else:
            kind = _python_value_kind(val)
        fields.append(
            SidecarField(
                level=FieldLevel.OPTIONAL,
                name=key,
                value=val,
                present=True,
                value_kind=kind,
            )
        )
    return fields


def _find_verdict(report, root: Path, path: Path) -> Optional[FileVerdict]:
    """Return the FileVerdict matching ``path`` (absolute)."""
    try:
        target_abs = str(path.resolve())
    except OSError:
        target_abs = str(path)
    try:
        root_resolved = root.resolve()
    except OSError:
        root_resolved = root
    for fv in report.files:
        fp = fv.path
        candidate = fp if fp.is_absolute() else root_resolved / fp
        try:
            candidate_abs = str(candidate.resolve())
        except OSError:
            candidate_abs = str(candidate)
        if candidate_abs == target_abs:
            return fv
    return None


# --------------------------------------------------------------------------
# The pane
# --------------------------------------------------------------------------


class SidecarFormPane(QWidget):
    """Schema-aware sidecar form (Editor center pane).

    JSON sidecars are **editable**. Edits update an in-memory cache
    only; the user clicks **Save** in the toolbar to flush changes to
    disk, or **Revert** to drop them. Switching files silently
    discards any unsaved changes (the dirty-count chip in the toolbar
    makes the unsaved state visible while the file is bound).

    The pane preserves key order via an :class:`OrderedDict` cached at
    bind time; existing fields keep position, new fields append.
    """

    # Emitted after a successful save to disk. Per-file revalidation
    # (Step 6b) and the project-event audit log (M7) hook here.
    file_saved = pyqtSignal(Path)

    # Emitted when saving to disk fails (permission error etc.).
    # Args: (file_path, error_message).
    save_failed = pyqtSignal(Path, str)

    # Emitted whenever the in-memory dirty state changes (after a
    # commit, save, revert, or set_file). Allows the toolbar / window
    # title / status bar to react.
    dirty_changed = pyqtSignal(int)  # number of fields changed since load
    # Emitted when undo/redo availability changes (Editor toolbar syncs).
    history_changed = pyqtSignal()
    # A row asked to state its field across more than one file. Carries
    # the field name; the host owns the dataset root and the picker.
    apply_to_others_requested = pyqtSignal(str)
    # A row asked where its value is actually stated.
    explain_requested = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")

        # Undo/redo of in-memory JSON edits (snapshot = deep-copied cache).
        from .edit_history import SnapshotHistory
        self._history = SnapshotHistory()
        self._pre_edit: Optional[OrderedDict[str, Any]] = None

        # Bound state — kept so the form can be rebuilt on file change.
        self._current_file: Optional[Path] = None
        self._current_root: Optional[Path] = None
        self._current_report = None  # ValidationReport | None
        self._rows: list[SidecarRow] = []
        # Working copy of the bound file's JSON. Edits mutate this
        # dict; ``save()`` flushes it to disk; ``revert()`` copies the
        # disk snapshot back over the top.
        self._json_cache: Optional[OrderedDict[str, Any]] = None
        # Pristine snapshot of what's on disk for the bound file. Used
        # to compute the dirty diff and to power revert.
        self._original_json: Optional[OrderedDict[str, Any]] = None

        # Which view is currently shown — restored from AppSettings so
        # the user's pick survives launches.
        from ..app_settings import AppSettings
        _s = AppSettings.load()
        self._view_mode: str = _s.editor_sidecar_view
        # Which fields both views show. See ``SCOPE_*`` below.
        self._field_scope: str = _s.editor_field_scope

        # Save as you go. A field commits on focus-out or Enter, and the
        # commits are coalesced so a burst of typing is one write rather than
        # one per keystroke: a half-typed value should never reach the file,
        # and revalidation should not run on every character. The write is
        # reversible (see ``_write_json_cache``), which is what makes saving
        # without being asked acceptable in the first place.
        self._autosave: bool = _s.editor_autosave
        # True between the first keystroke in a field and its commit.
        # Only affects what the toolbar SAYS; the cache is not touched.
        self._typing: bool = False
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(400)
        self._autosave_timer.timeout.connect(self._on_autosave)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        v.addWidget(PaneHeader("Sidecar"))

        # --- Edit toolbar ----------------------------------------------
        # View-mode pills (BIDS / Tree) on the left, then Add / Delete
        # field buttons (Tree mode only), then a stretch, the unsaved
        # chip, Revert + Save. The toolbar is hidden when the bound
        # file isn't editable JSON.
        self._edit_toolbar = QFrame()
        self._edit_toolbar.setObjectName("sidecar-toolbar")
        et = QHBoxLayout(self._edit_toolbar)
        et.setContentsMargins(14, 6, 14, 6)
        et.setSpacing(8)

        # View-mode pills — re-uses the ``#view-pill`` QSS rules the
        # top-header view switcher already ships, so dark/light is
        # automatic.
        self._bids_view_btn = QPushButton("BIDS view")
        self._bids_view_btn.setObjectName("view-pill")
        self._bids_view_btn.setCheckable(True)
        self._bids_view_btn.setChecked(True)
        self._tree_view_btn = QPushButton("Tree view")
        self._tree_view_btn.setObjectName("view-pill")
        self._tree_view_btn.setCheckable(True)
        self._view_group = QButtonGroup(self._edit_toolbar)
        self._view_group.setExclusive(True)
        self._view_group.addButton(self._bids_view_btn, 0)
        self._view_group.addButton(self._tree_view_btn, 1)
        self._view_group.idClicked.connect(self._on_view_pill_clicked)
        et.addWidget(self._bids_view_btn)
        et.addWidget(self._tree_view_btn)

        # Which fields either view shows. The two views used to disagree
        # silently: the BIDS form offers every field the schema declares for
        # the datatype (130 rows on a typical BOLD sidecar) and the Tree view
        # renders the file (90). Both now render the SAME set and this says
        # which set it is.
        self._scope_combo = QComboBox()
        self._scope_combo.setObjectName("ent-input")
        self._scope_combo.addItem("In file + declared", SCOPE_ALL)
        self._scope_combo.addItem("Only what is in the file", SCOPE_PRESENT)
        self._scope_combo.addItem("Only what is missing", SCOPE_ABSENT)
        self._scope_combo.setToolTip(
            "Which fields this form OFFERS. It changes what you see, never "
            "what is in the file.\n\n"
            "In file + declared: everything the file states, plus every field "
            "the standard declares for this kind of file, so you can fill one "
            "in without knowing its name. A declared field you do not type "
            "into is never written.\n"
            "Only what is in the file: exactly the keys the file holds.\n"
            "Only what is missing: the declared fields the file does not "
            "carry, which is the list of work.\n\n"
            "If fields seem to appear after validating, this is why: a "
            "validated file knows its datatype, so the form can offer what "
            "the standard declares for it. Nothing was added to the file."
        )
        idx = self._scope_combo.findData(self._field_scope)
        if idx >= 0:
            self._scope_combo.setCurrentIndex(idx)
        self._scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        et.addWidget(self._scope_combo)

        # Add / Delete field — tree-only. Both are visible in tree mode
        # and hidden in BIDS mode (the BIDS form's notion of "add field"
        # is the validator-surfaced missing-field row).
        # ``+ Add field`` always inserts at root.
        self._add_field_btn = QPushButton("+ Add field")
        self._add_field_btn.setObjectName("tb-btn")
        self._add_field_btn.setVisible(False)
        self._add_field_btn.clicked.connect(self._on_add_field_clicked)
        et.addWidget(self._add_field_btn)
        # ``+ Add subfield`` inserts inside the currently-selected
        # container. Enabled state is synced from the tree's selection.
        self._add_subfield_btn = QPushButton("+ Add subfield")
        self._add_subfield_btn.setObjectName("tb-btn")
        self._add_subfield_btn.setVisible(False)
        self._add_subfield_btn.setEnabled(False)
        self._add_subfield_btn.setToolTip(
            "Add a new field inside the selected field. "
            "Promotes a leaf into a container if needed."
        )
        self._add_subfield_btn.clicked.connect(self._on_add_subfield_clicked)
        et.addWidget(self._add_subfield_btn)
        self._del_field_btn = QPushButton("− Delete field")
        self._del_field_btn.setObjectName("tb-btn")
        self._del_field_btn.setVisible(False)
        self._del_field_btn.clicked.connect(self._on_delete_field_clicked)
        et.addWidget(self._del_field_btn)

        self._dirty_chip = QLabel("")
        self._dirty_chip.setObjectName("sidecar-dirty-chip")
        self._dirty_chip.setVisible(False)
        et.addWidget(self._dirty_chip)
        et.addStretch(1)
        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setObjectName("tb-btn")
        self._revert_btn.setEnabled(False)
        self._revert_btn.clicked.connect(self.revert)
        et.addWidget(self._revert_btn)
        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("tb-btn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save)
        et.addWidget(self._save_btn)
        self._edit_toolbar.setVisible(False)
        v.addWidget(self._edit_toolbar)

        # --- Schema legend ---------------------------------------------
        # Built once in __init__ — the swatches are QSS-driven so the
        # palette swap reaches them automatically via the global QSS
        # re-apply. No need to rebuild on theme change.
        self._legend = QFrame()
        self._legend.setObjectName("schema-legend")
        self._legend_layout = QHBoxLayout(self._legend)
        self._legend_layout.setContentsMargins(14, 6, 14, 6)
        self._legend_layout.setSpacing(14)
        self._build_legend_once()
        v.addWidget(self._legend)

        # --- Stacked content: BIDS form vs Tree view -------------------
        # Index 0 is the schema-aware BIDS form (current behavior).
        # Index 1 is the 2-column JSON tree editor.
        self._view_stack = QStackedWidget()

        # BIDS form body (scrollable).
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(14, 8, 14, 12)
        self._body_layout.setSpacing(0)
        self._body_layout.addStretch(1)

        self._empty_hint = QLabel(
            "Select a file in the BIDS tree to view its sidecar."
        )
        self._empty_hint.setObjectName("pane-hint")
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_hint.setWordWrap(True)
        self._body_layout.insertWidget(0, self._empty_hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(self._body)
        self._view_stack.addWidget(scroll)              # index 0

        # Tree editor view.
        self._tree_view = JsonTreeView()
        self._tree_view.model_changed.connect(self._on_tree_changed)
        # Sync the Add-subfield button enable-state with tree selection
        # (and with edits that turn a container into a leaf, etc.).
        self._tree_view.itemSelectionChanged.connect(
            self._sync_subfield_btn_enabled,
        )
        self._tree_view.model_changed.connect(
            self._sync_subfield_btn_enabled,
        )
        self._view_stack.addWidget(self._tree_view)     # index 1

        v.addWidget(self._view_stack, 1)

        # --- Status footer ---------------------------------------------
        # Styling lives in theme.qss under ``QFrame#sidecar-footer`` and
        # ``QLabel#sidecar-footer-*`` (same QSS-only pattern as the
        # converter's ``QPlainTextEdit#dock-log``).
        self._footer = QFrame()
        self._footer.setObjectName("sidecar-footer")
        fl = QHBoxLayout(self._footer)
        fl.setContentsMargins(14, 6, 14, 6)
        fl.setSpacing(10)
        self._footer_path = QLabel("")
        self._footer_path.setObjectName("sidecar-footer-path")
        self._footer_summary = QLabel("")
        self._footer_summary.setObjectName("sidecar-footer-summary")
        fl.addWidget(self._footer_path, 1)
        fl.addWidget(self._footer_summary)
        v.addWidget(self._footer)

        # Apply the persisted view-mode now that every widget is built.
        # ``_apply_view_mode`` toggles the pills + swaps the stack + sets
        # the tree-only-button visibility in one place.
        self._apply_view_mode(self._view_mode, persist=False)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def current_file(self) -> Optional[Path]:
        return self._current_file

    def set_file(
        self,
        path: Optional[Path],
        root: Optional[Path],
        report,
    ) -> None:
        """Bind the pane to a file.

        ``path``   — absolute path of the file to display, or ``None``.
        ``root``   — current BIDS root (used for relative-path display
                     and to resolve the validator's report paths).
        ``report`` — :class:`ValidationReport` or ``None``. The report
                     is optional: file content is loaded directly from
                     disk; the report only contributes level
                     colour-coding and surfaces missing required fields.
        """
        # A debounced write must land on the file it belongs to, not on the
        # next one the user clicks.
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
            if self._current_file is not None and self.is_dirty():
                self.save()

        self._typing = False
        self._current_file = path
        self._current_root = root
        self._current_report = report

        if path is None:
            self._json_cache = None
            self._original_json = None
            self._rebuild_form_with_fields([], None)
            self._tree_view.set_data(None)
            self._update_footer(None, None, None)
            self._refresh_dirty_ui()
            self._history.clear()
            self._pre_edit = None
            self.history_changed.emit()
            return

        # Seed the JSON cache from disk so subsequent saves preserve
        # the original key order. We re-read on every ``set_file`` (not
        # just first time) so an external edit between selections is
        # picked up. Only attempt this for ``.json`` files — opening a
        # binary ``.nii.gz`` would either crash on decode or waste a
        # large read for no benefit.
        if path.name.lower().endswith(".json"):
            disk = _load_json_ordered(path)
            self._json_cache = (
                copy.deepcopy(disk) if disk is not None else None
            )
            self._original_json = disk
        else:
            self._json_cache = None
            self._original_json = None

        verdict = (
            _find_verdict(report, root, path)
            if (report is not None and root is not None)
            else None
        )

        # One render path for both views, so the field scope cannot apply to
        # one of them and not the other. This used to render each view
        # separately here, which is how the two drifted apart.
        if self._json_cache is not None:
            self._sync_active_view()
        else:
            fields = self._fields_for(path, verdict)
            self._rebuild_form_with_fields(fields, verdict)
            self._tree_view.set_data(None)
        self._update_footer(path, root, verdict)
        self._refresh_dirty_ui()
        self._history.clear()
        self._pre_edit = self._snapshot()
        self.history_changed.emit()

    def repaint_for_palette(self, pal: dict) -> None:
        """Force Qt to re-evaluate the global QSS for every widget here.

        Background: every palette-coloured widget in this pane is
        QSS-driven (same pattern as ``QPlainTextEdit#dock-log``), so in
        theory ``QApplication.setStyleSheet`` from the theme manager
        should restyle everything for free. In practice, Qt's QSS
        engine caches the computed style per widget and sometimes does
        **not** invalidate that cache when only the *values* of QSS
        token-substitutions change (only when *rules* change). The
        symptom: every other pane refreshes, this one stays stale
        until app restart.

        The canonical Qt workaround is to call
        ``style.unpolish(w); style.polish(w)`` on each widget — that
        drops the cache and forces a fresh QSS lookup against the
        (now updated) global stylesheet. We iterate self + all
        descendants so the legend swatches, SidecarRow bars, footer
        chrome, etc., all refresh.
        """
        del pal
        style = self.style()
        for w in [self, *self.findChildren(QWidget)]:
            style.unpolish(w)
            style.polish(w)
            w.update()

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _fields_for(
        self,
        path: Path,
        verdict: Optional[FileVerdict],
    ) -> list[SidecarField]:
        """Pick the field list to render based on what's available.

        Priority:

        1. If a :class:`FileVerdict` is bound and carries
           ``sidecar_fields``, use them — they include schema-required
           "missing" entries the validator pre-computed.
        2. Otherwise, if the file is JSON, parse it and emit one
           OPTIONAL field per present key.
        3. For non-JSON files we return an empty list — the empty-hint
           explains why.
        """
        if verdict is not None and verdict.sidecar_fields:
            return list(verdict.sidecar_fields)
        if path.name.lower().endswith(".json"):
            return _load_json_fields_from_disk(path)
        return []

    def _build_legend_once(self) -> None:
        """Build the legend strip. QSS handles all colors going forward."""
        levels: list[tuple[str, str]] = [
            ("req", "required"),
            ("rec", "recommended"),
            ("opt", "optional"),
            ("dep", "deprecated"),
        ]
        for code, label in levels:
            chip = QFrame()
            ch = QHBoxLayout(chip)
            ch.setContentsMargins(0, 0, 0, 0)
            ch.setSpacing(6)
            sw = QFrame()
            sw.setObjectName(f"legend-swatch-{code}")
            sw.setFixedSize(8, 12)
            ch.addWidget(sw)
            t = QLabel(label)
            t.setObjectName("legend-text")
            ch.addWidget(t)
            self._legend_layout.addWidget(chip)
        self._legend_layout.addStretch(1)
        self._legend_context = QLabel("")
        self._legend_context.setObjectName("legend-text")
        self._legend_layout.addWidget(self._legend_context)

    def _rebuild_form_with_fields(
        self,
        fields: list[SidecarField],
        verdict: Optional[FileVerdict],
    ) -> None:
        # Drop every existing SidecarRow. ``setParent(None)`` is the
        # critical bit — without it the old rows stay visually present
        # until ``deleteLater`` processes on the next event-loop tick.
        for row in self._rows:
            self._body_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

        if not fields:
            if self._current_file is None:
                self._empty_hint.setText(
                    "Select a file in the BIDS tree to view its sidecar."
                )
            else:
                lower = self._current_file.name.lower()
                if lower.endswith(".json"):
                    self._empty_hint.setText(
                        "This sidecar is empty or could not be parsed."
                    )
                else:
                    self._empty_hint.setText(
                        "No sidecar form for this file type.\n\n"
                        "Pick the matching “.json” sibling in the tree "
                        "to view its metadata."
                    )
            self._empty_hint.setVisible(True)
            self._legend_context.setText(
                self._current_file.name if self._current_file else ""
            )
            return

        self._empty_hint.setVisible(False)
        if verdict is not None and verdict.datatype and verdict.suffix:
            self._legend_context.setText(
                f"schema · {verdict.datatype}/{verdict.suffix}"
            )
        elif self._current_file is not None:
            self._legend_context.setText(self._current_file.name)
        else:
            self._legend_context.setText("")

        fields_sorted = sorted(
            fields, key=lambda f: _LEVEL_SORT.get(f.level, 99)
        )
        # Editable when the bound file is a JSON sidecar and we have a
        # readable on-disk cache to save back into.
        editable = (
            self._current_file is not None
            and self._current_file.name.lower().endswith(".json")
            and self._json_cache is not None
        )
        insert_idx = self._body_layout.count() - 1  # before the stretch
        # What the standard says about each field, so the editor for it can be
        # the one the standard implies rather than a text box for everything.
        specs = _schema_specs(verdict, self._current_file)
        for field in fields_sorted:
            level_code = _LEVEL_CODE.get(field.level, "opt")
            text, kind = _format_value(field)
            row = SidecarRow(
                level_code,
                field.name,
                text,
                kind,
                editable=editable,
                raw_value=field.value,
                schema_field=specs.get(field.name),
            )
            if field.description:
                row.setToolTip(field.description)
            if editable:
                row.value_committed.connect(self._on_field_committed)
                row.apply_to_others_requested.connect(
                    self.apply_to_others_requested
                )
                row.editing_started.connect(self._on_editing_started)
                row.explain_requested.connect(self.explain_requested)
            self._body_layout.insertWidget(insert_idx, row)
            insert_idx += 1
            self._rows.append(row)

    def _on_field_committed(
        self,
        key: str,
        parsed_value: Any,
        value_kind: str,
    ) -> None:
        """Mutate the in-memory cache. Disk write happens on :meth:`save`."""
        del value_kind  # parsing already handled by SidecarRow
        self._typing = False
        if self._current_file is None or self._json_cache is None:
            return
        from ...editor.field_values import REMOVE

        if parsed_value is REMOVE:
            # The field was cleared and its type has no empty form: a number
            # and a boolean have none, and ``null`` is a validation error.
            # Deleting the key is the only reading of "empty" that stays
            # valid, and an absent field is reported honestly as missing.
            if self._json_cache.pop(key, REMOVE) is not REMOVE:
                self._after_edit()
            return
        # No-ops don't touch the cache — keeps the dirty count honest.
        current = self._json_cache.get(key, _UNSET)
        if current == parsed_value:
            return
        self._json_cache[key] = parsed_value
        self._after_edit()

    # ----------------------------------------------------------------------
    # Undo / redo (snapshot-based, in-memory; disk write still via Save)
    # ----------------------------------------------------------------------

    def _on_autosave(self) -> None:
        """The debounce elapsed: write, quietly."""
        if not self._autosave or self._current_file is None:
            return
        if not self.is_dirty():
            return
        self.save()

    def set_autosave(self, enabled: bool) -> None:
        """Turn save-as-you-go on or off for this pane."""
        self._autosave = bool(enabled)
        if not enabled:
            self._autosave_timer.stop()

    def flush_pending_save(self) -> bool:
        """Write now if a debounced save is still waiting.

        Called before the pane moves to another file, so switching selection
        cannot drop an edit the timer had not reached yet.
        """
        if not self._autosave_timer.isActive():
            return True
        self._autosave_timer.stop()
        return self.save()

    def _snapshot(self) -> Optional[OrderedDict]:
        return copy.deepcopy(self._json_cache) if self._json_cache is not None else None

    def _on_editing_started(self, key: str) -> None:
        """A field is being typed into: say so now, not on focus-out.

        The value is not committed yet, so the cache is untouched and there is
        nothing to save. What changes is only what the user is told: the
        toolbar says there are unsaved changes from the first keystroke
        instead of staying silent until they click somewhere else.
        """
        del key
        if self._typing:
            return
        self._typing = True
        self._refresh_dirty_ui()

    def _after_edit(self) -> None:
        """Record the pre-edit cache for undo, then refresh the dirty UI.

        Called at every in-memory mutation site (field commit + tree edits)
        in place of a bare ``_refresh_dirty_ui`` so each edit is undoable.
        """
        if self._pre_edit is not None:
            self._history.record(self._pre_edit)
        self._pre_edit = self._snapshot()
        self._refresh_dirty_ui()
        self.history_changed.emit()
        if self._autosave and self._current_file is not None:
            self._autosave_timer.start()

    def _restore(self, snap: Optional[OrderedDict]) -> None:
        self._json_cache = copy.deepcopy(snap) if snap is not None else None
        self._sync_active_view()
        self._pre_edit = self._snapshot()
        self._refresh_dirty_ui()
        self.history_changed.emit()

    def can_undo(self) -> bool:
        return self._json_cache is not None and self._history.can_undo

    def can_redo(self) -> bool:
        return self._json_cache is not None and self._history.can_redo

    def undo(self) -> None:
        if self._json_cache is None:
            return
        was_saved = not self.is_dirty()
        snap = self._history.undo(self._snapshot())
        if snap is not None:
            self._restore(snap)
            self._persist_if_it_was_saved(was_saved)

    def redo(self) -> None:
        if self._json_cache is None:
            return
        was_saved = not self.is_dirty()
        snap = self._history.redo(self._snapshot())
        if snap is not None:
            self._restore(snap)
            self._persist_if_it_was_saved(was_saved)

    def _persist_if_it_was_saved(self, was_saved: bool) -> None:
        """Write the file back when the step being undone had been saved.

        Without this, undoing after a save changed only the in-memory cache:
        the file kept the value that had been written, the pane went dirty
        again, and the user saw a button that did nothing. Half an undo is
        worse than none, because it leaves the screen and the disk disagreeing.

        The rule is the one a user already holds: what you see is what is on
        disk, unless you have unsaved changes. So an undo from a CLEAN state
        writes, and an undo from a dirty one does not, because in that case
        nothing had reached the disk to revert.
        """
        if not was_saved or self._current_file is None:
            return
        try:
            self._write_json_cache(self._current_file)
        except OSError as exc:
            log.warning("undo could not write %s: %s", self._current_file, exc)
            self.save_failed.emit(self._current_file, str(exc))
            return
        self._original_json = copy.deepcopy(self._json_cache)
        self._refresh_dirty_ui()
        self.file_saved.emit(self._current_file)

    # ----------------------------------------------------------------------
    # View-mode toggle + tree handlers
    # ----------------------------------------------------------------------

    def focus_field(self, name: str, severity: Optional[str] = None) -> bool:
        """Scroll to, focus, and highlight the field named ``name``.

        Works in both views — the BIDS form jumps to and highlights the
        matching :class:`SidecarRow`; the Tree view selects the matching
        top-level row and begins editing its Value cell. ``severity``
        (``"err"`` / ``"warn"``) tints the row in that colour; ``None`` uses
        the neutral focus tint. Returns ``True`` if a matching field was found.
        """
        if not name or self._json_cache is None:
            return False
        if self._view_mode == "tree":
            return self._focus_field_in_tree(name, severity)
        return self._focus_field_in_form(name, severity)

    def highlight_fields(self, severities: dict[str, str]) -> int:
        """Highlight every field in ``severities`` (``{name: "err"|"warn"}``).

        Clears any prior highlights first, tints each matching row in its
        severity colour, and scrolls the first one into view. Returns the
        number of rows highlighted. Only meaningful in the BIDS form view.
        """
        self.clear_field_highlights()
        first = None
        n = 0
        for row in self._rows:
            sev = severities.get(row.key)
            if sev:
                row.set_highlight(sev)
                n += 1
                if first is None:
                    first = row
        if first is not None:
            self._scroll_widget_visible(first)
        # Mirror onto the Tree view so the highlight shows in either mode.
        self._tree_view.set_highlights(severities)
        return n

    def clear_field_highlights(self) -> None:
        """Remove all row highlights (BIDS form + Tree view)."""
        for row in self._rows:
            row.set_highlight(None)
        self._tree_view.clear_highlights()

    def _focus_field_in_form(self, name: str, severity: Optional[str] = None) -> bool:
        self.clear_field_highlights()
        for row in self._rows:
            if row.key == name:
                row.set_highlight(severity or "focus")
                # Scroll the row into view inside the QScrollArea parent.
                editor = row.editor()
                if editor is not None:
                    self._scroll_widget_visible(editor)
                    editor.setFocus(Qt.FocusReason.OtherFocusReason)
                    # Highlight what is there, so the field reads as ready to
                    # be typed over. A combo has no ``selectAll`` of its own;
                    # its text lives in the line edit underneath, and a field
                    # with a curated vocabulary is a combo, so asking the
                    # outer widget silently did nothing for exactly the
                    # fields most likely to be jumped to.
                    target = (
                        editor.lineEdit()
                        if hasattr(editor, "lineEdit") and editor.lineEdit()
                        else editor
                    )
                    if hasattr(target, "selectAll"):
                        target.selectAll()
                else:
                    # Read-only mode: still scroll so the user sees it.
                    self._scroll_widget_visible(row)
                return True
        return False

    def _focus_field_in_tree(self, name: str, severity: Optional[str] = None) -> bool:
        for i in range(self._tree_view.topLevelItemCount()):
            item = self._tree_view.topLevelItem(i)
            if item.text(0) == name:
                # Tint this row (clears the others) so the jump is visible in
                # the Tree view too.
                self._tree_view.set_highlights({name: severity or "focus"})
                self._tree_view.setCurrentItem(item)
                self._tree_view.scrollToItem(item)
                # Start editing the Value cell so the user can type
                # immediately.
                self._tree_view.editItem(item, 1)
                return True
        return False

    def _scroll_widget_visible(self, widget) -> None:
        """Find the enclosing :class:`QScrollArea` and ensure ``widget``
        is visible. Walks up the parent chain (the QStackedWidget
        layout puts the scroll area a couple of levels above the row).
        """
        from PyQt6.QtWidgets import QScrollArea
        parent = widget.parent()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                parent.ensureWidgetVisible(widget)
                return
            parent = parent.parent()

    def view_mode(self) -> str:
        return self._view_mode

    def _on_view_pill_clicked(self, idx: int) -> None:
        new_mode = "tree" if idx == 1 else "bids"
        self._apply_view_mode(new_mode, persist=True)

    def _apply_view_mode(self, mode: str, *, persist: bool) -> None:
        """Swap the visible view (and the tree-only buttons), persist
        the choice, and re-render the now-visible view from the cache.
        """
        if mode not in ("bids", "tree"):
            mode = "bids"
        self._view_mode = mode
        # Sync the pills silently — ``setChecked`` doesn't emit
        # ``idClicked`` so we won't re-enter this handler.
        target = self._tree_view_btn if mode == "tree" else self._bids_view_btn
        target.setChecked(True)
        # Swap the stack.
        self._view_stack.setCurrentIndex(1 if mode == "tree" else 0)
        # Toggle the tree-only toolbar buttons.
        self._add_field_btn.setVisible(mode == "tree")
        self._add_subfield_btn.setVisible(mode == "tree")
        self._del_field_btn.setVisible(mode == "tree")
        if mode == "tree":
            self._sync_subfield_btn_enabled()
        # Re-render the now-visible view from the working cache so any
        # edits made while the other view was active flow through.
        self._sync_active_view()
        if persist:
            from ..app_settings import AppSettings
            AppSettings.remember_editor_sidecar_view(mode)

    def _sync_active_view(self) -> None:
        """Re-render whichever view is currently visible from the cache."""
        if self._json_cache is None:
            return
        # BIDS form derives from current cache + verdict; we also use
        # the same field list to compute the tree's level-bar map.
        verdict = (
            _find_verdict(
                self._current_report,
                self._current_root,
                self._current_file,
            )
            if (
                self._current_report is not None
                and self._current_root is not None
                and self._current_file is not None
            )
            else None
        )
        fields = (
            self._fields_for(self._current_file, verdict)
            if self._current_file is not None else []
        )
        fields = self._patch_fields_from_cache(fields)
        fields = self._apply_field_scope(fields)
        if self._view_mode == "tree":
            # The tree renders the file, so the fields the scope keeps that
            # the file does not carry are passed as placeholders: shown,
            # dimmed, and not written unless the user types into them. That
            # is what makes the two views hold the same rows.
            shown = {f.name for f in fields}
            data = OrderedDict(
                (k, v) for k, v in self._json_cache.items() if k in shown
            )
            # The same schema facts the BIDS view puts on each row, so the
            # two views explain a field identically rather than one of them
            # showing a bare key and a value.
            specs = _schema_specs(verdict, self._current_file)
            self._tree_view.set_data(
                data,
                levels=self._levels_from_fields(fields),
                placeholders=[
                    f.name for f in fields if f.name not in self._json_cache
                ],
                describe={
                    name: spec.description
                    for name, spec in specs.items() if spec.description
                },
                units={
                    name: spec.unit
                    for name, spec in specs.items() if spec.unit
                },
            )
        else:
            self._rebuild_form_with_fields(fields, verdict)

    def _apply_field_scope(
        self, fields: list[SidecarField],
    ) -> list[SidecarField]:
        """Keep the fields the current scope asks for.

        Presence is decided against the live cache, not against the
        ``present`` flag the verdict was built with, so a field the user has
        just typed counts as present without a revalidation.
        """
        if self._field_scope == SCOPE_ALL or self._json_cache is None:
            return fields
        in_file = set(self._json_cache.keys())
        if self._field_scope == SCOPE_PRESENT:
            return [f for f in fields if f.name in in_file]
        return [f for f in fields if f.name not in in_file]

    def field_scope(self) -> str:
        """Which fields both views are showing (one of :data:`FIELD_SCOPES`)."""
        return self._field_scope

    def set_field_scope(self, scope: str) -> None:
        """Set the field scope from code, keeping the combo in step."""
        if scope not in FIELD_SCOPES:
            return
        idx = self._scope_combo.findData(scope)
        if idx >= 0:
            # Drives ``_on_scope_changed``, which stores and re-renders.
            self._scope_combo.setCurrentIndex(idx)

    def _on_scope_changed(self, idx: int) -> None:
        """Field scope changed: remember it and re-render the active view."""
        scope = self._scope_combo.itemData(idx)
        if scope not in FIELD_SCOPES or scope == self._field_scope:
            return
        self._field_scope = scope
        from ..app_settings import AppSettings
        AppSettings.remember_editor_field_scope(scope)
        self._sync_active_view()

    def _patch_fields_from_cache(
        self, fields: list[SidecarField],
    ) -> list[SidecarField]:
        """Update / extend ``fields`` with current cache contents.

        For each known field, replace its value/value_kind with the
        cache's. For each cache key not in ``fields``, append a new
        OPTIONAL SidecarField. This is how tree-side edits propagate
        into the BIDS form when the user toggles back.
        """
        if self._json_cache is None:
            return fields
        by_name = {f.name: i for i, f in enumerate(fields)}
        patched: list[SidecarField] = []
        for f in fields:
            if f.name in self._json_cache:
                val = self._json_cache[f.name]
                f = SidecarField(
                    level=f.level,
                    name=f.name,
                    value=val,
                    present=True,
                    value_kind=_python_value_kind(val),
                    description=f.description,
                )
            patched.append(f)
        for key, val in self._json_cache.items():
            if key not in by_name:
                patched.append(
                    SidecarField(
                        level=FieldLevel.OPTIONAL,
                        name=key,
                        value=val,
                        present=True,
                        value_kind=_python_value_kind(val),
                    )
                )
        return patched

    def _levels_from_fields(
        self, fields: list[SidecarField],
    ) -> dict[str, str]:
        """Map ``field.name → level code`` for every present field.

        The tree view consumes this to paint a level bar on the left of
        each top-level row, matching the BIDS form's colour coding.
        Missing/required fields are skipped — the tree only shows keys
        actually present in the JSON, so a bar for an absent key would
        have nowhere to land.
        """
        out: dict[str, str] = {}
        for f in fields:
            if not f.present:
                continue
            code = _LEVEL_CODE.get(f.level)
            if code is not None:
                out[f.name] = code
        return out

    def _on_add_field_clicked(self) -> None:
        """Add a new field at the top level (regardless of selection)."""
        if self._view_mode != "tree" or self._json_cache is None:
            return
        self._tree_view.add_field()

    def _on_add_subfield_clicked(self) -> None:
        """Add a new field inside the selected container.

        No-op when the selection isn't a container — the button is
        disabled in that case so this is just defensive.
        """
        if self._view_mode != "tree" or self._json_cache is None:
            return
        self._tree_view.add_subfield()

    def _on_delete_field_clicked(self) -> None:
        if self._view_mode != "tree" or self._json_cache is None:
            return
        self._tree_view.delete_field()  # acts on the current selection

    def _sync_subfield_btn_enabled(self) -> None:
        """Enable the Add-subfield button whenever the tree has a
        selection. Works on leaves too — clicking promotes them into
        containers, matching the original BIDS-Manager Inspector.
        """
        sel = self._tree_view.currentItem()
        self._add_subfield_btn.setEnabled(sel is not None)

    def _on_tree_changed(self) -> None:
        """Pull the latest tree state into the cache + refresh UI.

        Preserves the cache's existing key order across edits — the
        tree's display sort (by validation level) must not bleed into
        the on-disk save order. Existing keys keep position, deleted
        keys are removed, genuinely new keys append at the end.
        """
        if self._json_cache is None:
            return
        new_dict = self._tree_view.to_dict()
        old_keys = list(self._json_cache.keys())
        # Update / remove pass.
        for k in old_keys:
            if k in new_dict:
                self._json_cache[k] = new_dict[k]
            else:
                del self._json_cache[k]
        # Append truly-new keys at the end.
        for k, v in new_dict.items():
            if k not in self._json_cache:
                self._json_cache[k] = v
        self._after_edit()

    # ----------------------------------------------------------------------
    # Save / Revert / dirty introspection
    # ----------------------------------------------------------------------

    def is_dirty(self) -> bool:
        """``True`` when the in-memory cache differs from disk."""
        return self._dirty_count() > 0

    def save(self) -> bool:
        """Flush the in-memory cache to disk.

        Returns ``True`` on success (or no-op when there's nothing to
        save), ``False`` on I/O error (``save_failed`` is emitted).
        """
        if self._current_file is None or self._json_cache is None:
            return True
        if not self.is_dirty():
            return True
        try:
            self._write_json_cache(self._current_file)
        except OSError as exc:
            log.warning("save failed for %s: %s", self._current_file, exc)
            self.save_failed.emit(self._current_file, str(exc))
            return False
        # Snapshot the new on-disk state so the next dirty check
        # compares against what we just wrote.
        self._original_json = copy.deepcopy(self._json_cache)
        self._refresh_dirty_ui()
        self.file_saved.emit(self._current_file)
        return True

    def revert(self) -> None:
        """Discard in-memory edits, restore the disk snapshot, rebuild."""
        if self._original_json is None:
            return
        self._json_cache = copy.deepcopy(self._original_json)
        # Re-render the form from the restored cache so editors reset.
        verdict = (
            _find_verdict(
                self._current_report,
                self._current_root,
                self._current_file,
            )
            if (
                self._current_report is not None
                and self._current_root is not None
                and self._current_file is not None
            )
            else None
        )
        fields = (
            self._fields_for(self._current_file, verdict)
            if self._current_file is not None else []
        )
        # Replace the FileVerdict-sourced values with whatever the
        # original disk snapshot says — so reverting a "missing" field
        # the user filled in (and saved → then edited again → reverted)
        # behaves correctly. ``_fields_for`` already returns the verdict
        # rows when present; we patch values from the disk snapshot.
        if self._original_json is not None:
            patched: list[SidecarField] = []
            for f in fields:
                if f.name in self._original_json:
                    val = self._original_json[f.name]
                    f = SidecarField(
                        level=f.level,
                        name=f.name,
                        value=val,
                        present=True,
                        value_kind=_python_value_kind(val),
                        description=f.description,
                    )
                patched.append(f)
            fields = patched
        # Same single render path as ``set_file``: the cache is already back
        # to the disk snapshot, so this re-renders whichever view is showing
        # with the current field scope applied.
        self._sync_active_view()
        self._refresh_dirty_ui()
        # Revert discards unsaved edits, so the undo history of those edits is
        # no longer meaningful: reset it to the just-restored state.
        self._history.clear()
        self._pre_edit = self._snapshot()
        self.history_changed.emit()

    def _dirty_count(self) -> int:
        """Number of keys whose value differs between cache and disk."""
        if self._json_cache is None or self._original_json is None:
            return 0
        keys = set(self._json_cache.keys()) | set(self._original_json.keys())
        n = 0
        for k in keys:
            a = self._json_cache.get(k, _UNSET)
            b = self._original_json.get(k, _UNSET)
            if a != b:
                n += 1
        return n

    def _refresh_dirty_ui(self) -> None:
        """Sync the toolbar visibility + chip text + button enablement
        with the current dirty count."""
        editable = (
            self._current_file is not None
            and self._current_file.name.lower().endswith(".json")
            and self._json_cache is not None
        )
        self._edit_toolbar.setVisible(editable)
        n = self._dirty_count() if editable else 0
        if n > 0:
            self._dirty_chip.setText(
                f"{n} unsaved change" + ("s" if n != 1 else "")
            )
            self._dirty_chip.setVisible(True)
        elif editable and self._typing:
            # Typed into but not committed yet. There is nothing to count and
            # nothing to save, so say what is true: an edit is in progress.
            self._dirty_chip.setText("editing")
            self._dirty_chip.setVisible(True)
        else:
            self._dirty_chip.setVisible(False)
        self._save_btn.setEnabled(editable and n > 0)
        self._revert_btn.setEnabled(editable and n > 0)
        self.dirty_changed.emit(n)

    def _write_json_cache(self, path: Path) -> None:
        """Write ``self._json_cache`` to ``path``, reversibly where possible.

        Inside a dataset the write goes through the operation log, so it is
        atomic, the previous bytes are kept, and it can be undone after the
        pane has moved on. That matters more now that saving can happen by
        itself: a user who did not press anything should still be able to get
        back what they had.

        Outside a dataset, or when the root is unwritable, fall back to a
        plain atomic write rather than refusing to save at all.
        """
        assert self._json_cache is not None
        text = json.dumps(self._json_cache, indent=2, ensure_ascii=False) + "\n"
        root = self._current_root
        if root is not None:
            try:
                from ...project.operations import begin_operation

                with begin_operation(root, f"Edit {path.name}") as op:
                    op.write_text(path, text)
                return
            except Exception as exc:  # noqa: BLE001 - fall back, never block
                log.debug("operation log unavailable for %s: %s", path, exc)
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise

    def _update_footer(
        self,
        path: Optional[Path],
        root: Optional[Path],
        verdict: Optional[FileVerdict],
    ) -> None:
        if path is None:
            self._footer_path.setText("")
            self._footer_summary.setText("")
            return
        if root is not None:
            try:
                rel = path.resolve().relative_to(root.resolve())
                self._footer_path.setText(str(rel))
            except ValueError:
                self._footer_path.setText(str(path))
        else:
            self._footer_path.setText(str(path))
        if verdict is None:
            n_fields = len(self._rows)
            if n_fields:
                self._footer_summary.setText(
                    f"{n_fields} fields · not yet validated"
                )
            else:
                self._footer_summary.setText("not yet validated")
            return
        fields = verdict.sidecar_fields
        if not fields:
            self._footer_summary.setText(f"{len(verdict.issues)} issues")
            return
        missing = sum(1 for f in fields if not f.present)
        self._footer_summary.setText(
            f"{len(fields)} fields · {missing} missing · "
            f"{len(verdict.issues)} issues"
        )


__all__ = ["SidecarFormPane", "find_peer_files"]

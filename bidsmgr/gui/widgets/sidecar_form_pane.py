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
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import copy

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pane-dark")

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
        self._view_mode: str = AppSettings.load().editor_sidecar_view

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

        fields = self._fields_for(path, verdict)
        self._rebuild_form_with_fields(fields, verdict)
        # Tree view stays in sync regardless of which view is visible —
        # set_data is cheap and avoids any drift if the user toggles
        # the pill mid-session. We pass the level map so the tree
        # delegate can paint matching colour bars on top-level rows.
        self._tree_view.set_data(
            self._json_cache, levels=self._levels_from_fields(fields),
        )
        self._update_footer(path, root, verdict)
        self._refresh_dirty_ui()

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
            )
            if field.description:
                row.setToolTip(field.description)
            if editable:
                row.value_committed.connect(self._on_field_committed)
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
        if self._current_file is None or self._json_cache is None:
            return
        # No-ops don't touch the cache — keeps the dirty count honest.
        current = self._json_cache.get(key, _UNSET)
        if current == parsed_value:
            return
        self._json_cache[key] = parsed_value
        self._refresh_dirty_ui()

    # ----------------------------------------------------------------------
    # View-mode toggle + tree handlers
    # ----------------------------------------------------------------------

    def focus_field(self, name: str) -> bool:
        """Scroll to and focus the editor for the field named ``name``.

        Works in both views — the BIDS form jumps to and focuses the
        matching :class:`SidecarRow`; the Tree view selects the
        matching top-level row and begins editing its Value cell.
        Returns ``True`` if a matching field was found.
        """
        if not name or self._json_cache is None:
            return False
        if self._view_mode == "tree":
            return self._focus_field_in_tree(name)
        return self._focus_field_in_form(name)

    def _focus_field_in_form(self, name: str) -> bool:
        for row in self._rows:
            if row.key == name:
                # Scroll the row into view inside the QScrollArea
                # parent. ``ensureWidgetVisible`` walks up to find the
                # nearest scroll area for us.
                editor = row.editor()
                if editor is not None:
                    self._scroll_widget_visible(editor)
                    editor.setFocus(Qt.FocusReason.OtherFocusReason)
                    # ``QLineEdit`` highlights its content on focus
                    # when ``setFocus`` is called this way — gives the
                    # user a clear visual cue + lets them type to
                    # overwrite.
                    if hasattr(editor, "selectAll"):
                        editor.selectAll()
                else:
                    # Read-only mode: still scroll so the user sees it.
                    self._scroll_widget_visible(row)
                return True
        return False

    def _focus_field_in_tree(self, name: str) -> bool:
        for i in range(self._tree_view.topLevelItemCount()):
            item = self._tree_view.topLevelItem(i)
            if item.text(0) == name:
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
        if self._view_mode == "tree":
            self._tree_view.set_data(
                self._json_cache,
                levels=self._levels_from_fields(fields),
            )
        else:
            self._rebuild_form_with_fields(fields, verdict)

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
        self._refresh_dirty_ui()

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
        self._rebuild_form_with_fields(fields, verdict)
        # Keep the tree view in sync too. Pass the level map so the
        # delegate paints colour bars matching the BIDS form.
        self._tree_view.set_data(
            self._json_cache, levels=self._levels_from_fields(fields),
        )
        self._refresh_dirty_ui()

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
        else:
            self._dirty_chip.setVisible(False)
        self._save_btn.setEnabled(editable and n > 0)
        self._revert_btn.setEnabled(editable and n > 0)
        self.dirty_changed.emit(n)

    def _write_json_cache(self, path: Path) -> None:
        """Serialise ``self._json_cache`` to ``path`` (overwrite)."""
        assert self._json_cache is not None
        path.write_text(
            json.dumps(self._json_cache, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

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

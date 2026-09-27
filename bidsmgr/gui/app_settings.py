r"""Typed wrapper over ``QSettings`` for cross-platform persistence.

``QSettings`` stores values per-platform in the right native location:

* macOS  → ``~/Library/Preferences/com.bidsmgr.bidsmgr.plist``
* Linux  → ``~/.config/bidsmgr/bidsmgr.conf``
* Windows → ``HKEY_CURRENT_USER\Software\bidsmgr\bidsmgr`` (registry)

This module wraps it with type-safe getters / setters and one schema
the rest of the GUI can rely on. Keep the keys in :data:`KEYS` —
adding a new one outside that namespace is a code smell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings


# Canonical setting keys. Grouped by section as a flat string namespace
# so QSettings shows them under ``[section]`` headers in INI / plist.
KEYS = {
    "theme":              "ui/theme",                # "dark" | "light"
    "raw_root":           "paths/raw_root",          # last raw input dir
    "bids_parent":        "paths/bids_parent",       # last BIDS output dir
    "scan_tsv_filename":  "scan/tsv_filename",       # filename of the scan TSV
    "highlight_aborts":   "inspector/highlight_aborts",   # toolbar toggle
    "active_view":        "ui/active_view",          # "converter" | "editor"
    "editor_bids_root":   "editor/bids_root",        # last BIDS root opened in the Editor view
    "editor_sidecar_view": "editor/sidecar_view",    # "bids" | "tree"
    "editor_field_scope": "editor/field_scope",      # "all"|"present"|"absent"
    "editor_show_hidden": "editor/show_hidden",      # dotfiles in the tree
    "editor_autosave": "editor/autosave",            # save as you go
    "editor_strict_validate": "editor/strict_validate",  # "deep checks": bidsval read_headers on/off
    # Validation engine (bidsval) knobs, controllable from Settings.
    "template_colour_levels": "ui/template_colour_levels",  # colour the level marks
    "validate_schema_version": "validate/schema_version",  # "" = bidsval bundled default
    "validate_max_rows": "validate/max_rows",              # TSV rows scanned per table
    "validate_show": "validate/show",                      # which severities the Editor lists
    "validate_flag_todos": "validate/flag_todos",          # flag literal TODO placeholders
    # Which of the four viewer layouts to open a scan in. "" means the user
    # has never chosen, so the GPU-dependent default applies.
    "nifti_view_mode": "editor/nifti_view_mode",                # single|multi|3d|combo
    "nifti_orientation": "editor/nifti_orientation",           # 0 sag | 1 cor | 2 ax
    # How every signal viewer draws a trace. One preference, not one per
    # viewer: somebody who wants thicker lines wants them everywhere.
    "trace_line_width": "editor/trace_line_width",
    "trace_line_color": "editor/trace_line_color",
    "trace_type_colors": "editor/trace_type_colors",   # JSON, channel type -> hex
    "nifti_crosshair_color": "editor/nifti_crosshair_color",   # hex string e.g. "#4FC3F7"
    "nifti_crosshair_thickness": "editor/nifti_crosshair_thickness",  # px, 1..5
    # Scan defaults
    "scan_n_jobs":        "scan/n_jobs",
    "scan_probe_convert": "scan/probe_convert",
    "scan_converter_preview": "scan/converter_preview",
    "scan_skip_bids_guess": "scan/skip_bids_guess",
    "scan_index_widths":  "scan/index_widths",       # JSON: entity -> width
    # Convert defaults
    "convert_n_jobs":     "convert/n_jobs",
    "convert_overwrite":  "convert/overwrite",      # legacy; migrated to on_existing
    "convert_on_existing": "convert/on_existing",    # skip|update|replace|error
    "convert_skip_residuals": "convert/skip_residuals",
    "convert_preserve_curation": "convert/preserve_curation",
    "convert_force_edf":  "convert/force_edf",       # re-encode EEG/iEEG to EDF
    "convert_deface":     "convert/deface",          # remove faces before commit
    "convert_deface_engine": "convert/deface_engine",
    # Scan rules (user-extensible classifier hints + series exclusions).
    # Stored as JSON-encoded lists - see ``bidsmgr.classifier.user_rules``.
    "user_hints":         "classifier/user_hints",
    "scan_exclusions":    "classifier/scan_exclusions",
    # Post-convert chain
    "post_run_metadata":  "post_convert/run_metadata",
    "post_run_validate":  "post_convert/run_validate",
    "post_metadata_fill_todos": "post_convert/metadata_fill_todos",
    "metadata_fill_scope": "post_convert/metadata_fill_scope",
    "post_validate_strict": "post_convert/validate_strict",
    "post_validate_html": "post_convert/validate_html",
    "post_fixup_companions": "post_convert/fixup_companions",
    "post_fixup_citation": "post_convert/fixup_citation",
    # Self-update
    "skipped_update_version": "update/skipped_version",
    # UI font scale (1.0 = default size baseline; values <1 shrink,
    # >1 enlarge every font-size + icon size proportionally).
    "font_scale": "ui/font_scale",
    # Which artwork the top-header brand mark renders.
    # "default" → ``assets/logo.png`` (monochrome, palette-inverted on dark).
    # "app_icon" → ``assets/macos/AppIcon128.png`` (full-color BIDS-Manager logo).
    "header_logo": "ui/header_logo",
    # Recently opened/created project dataset roots (JSON list, most-recent
    # first), shown on the Welcome tab.
    "recent_projects": "project/recent",
}


@dataclass
class AppSettings:
    """Strongly-typed snapshot of the persistent settings.

    Construct via :meth:`load` to read the current QSettings state.
    Call :meth:`save` to write back. The dataclass shape is the single
    source of truth for what's persistable.
    """

    # UI
    theme: str = "dark"
    # Which top-level view is shown on launch. Persisted across runs so
    # users land on the pane they were last using.
    active_view: str = "converter"
    # Last BIDS root opened in the Editor view (post-convert browser).
    editor_bids_root: Optional[str] = None
    # Which sidecar pane layout is active for JSON files.
    editor_sidecar_view: str = "bids"  # "bids" | "tree"
    # Which fields BOTH sidecar views show. "all" keeps the schema-declared
    # fields the file does not carry; "present" makes the two views identical.
    editor_field_scope: str = "all"   # "all" | "present" | "absent"
    # Dotfiles and dot-folders in the BIDS tree. Off by default: a dataset
    # has .bidsmgr/, .git/ and .bidsignore in it and none of them are the
    # data. On, they are shown dimmed rather than mixed in.
    editor_show_hidden: bool = False
    # Write a sidecar edit as soon as the field commits, debounced. OFF by
    # default: saving without being asked is a surprise, and the thing that
    # was actually wanted was for the toolbar to SAY there are unsaved
    # changes from the first keystroke, which it now does regardless of this.
    editor_autosave: bool = False
    # "Deep checks" toggle for the Editor's "Validate dataset". When True the
    # validator (bidsval) reads NIfTI headers and file contents (slower, more
    # thorough); when False it runs the fast structural pass used for live
    # revalidation. (Historically this enabled a second bidsschematools pass;
    # validation is now a single engine and the toggle maps to read_headers.)
    editor_strict_validate: bool = False
    # Validation engine (bidsval) knobs. ``validate_schema_version`` selects the
    # BIDS schema version to validate against ("" = bidsval's bundled default);
    # ``validate_max_rows`` bounds how many rows of each TSV are scanned;
    # ``validate_show`` filters which severities the Editor's Validation pane
    # lists ("error_warning" | "error" | "warning").
    validate_schema_version: str = ""
    validate_max_rows: int = 1000
    validate_show: str = "error_warning"
    # Flag literal "TODO" placeholder values as warnings (a BIDS Manager
    # convention: the metadata engine writes TODO into missing recommended
    # fields). On by default; off gives exact bidsval parity.
    validate_flag_todos: bool = True
    # NIfTI viewer crosshair style. Persisted so the user's chosen
    # colour + thickness survives across sessions.
    # Empty on purpose: "no choice made yet" is a different thing from any
    # particular layout, and it is what lets the first run pick the best
    # default this machine can show rather than a stored one.
    # How a trace is drawn in the signal viewers. ZERO means "decide from
    # what is on screen": two pixels for a physio channel, one for a wall of
    # MEG, because Qt strokes a wider pen 7x slower and a 300-channel view
    # cannot pay it. A width the user picks in the Line popup is stored as
    # that number and honoured everywhere.
    trace_line_width: int = 0
    trace_line_color: str = ""
    # Per-channel-type trace colours, type -> hex. EMPTY is the shipped
    # scheme, which is palette TOKENS rather than literals (mag takes the
    # accent colour, grad the success colour) so it follows the theme and
    # stays legible in both. A type appears here only when somebody chose
    # a colour for it, which is what lets Reset defaults be a deletion
    # rather than a second hardcoded table to keep in step.
    trace_type_colors: dict = field(default_factory=dict)
    nifti_view_mode: str = ""
    # Which plane a single-pane view opens on. Axial by convention when the
    # user has never chosen.
    nifti_orientation: int = 2
    nifti_crosshair_color: str = "#4FC3F7"
    nifti_crosshair_thickness: int = 1
    # Colour the requirement-level marks in the metadata template. Off, the
    # marks (* required, . recommended) remain, so the information does not
    # depend on being able to see the colour.
    template_colour_levels: bool = True

    # Recently-used paths (paths come back as str; callers wrap in Path).
    raw_root: Optional[str] = None
    bids_parent: Optional[str] = None
    # Scan-TSV filename only (the TSV lives under ``<bids_parent>/``).
    # The user can override this from the toolbar field.
    scan_tsv_filename: str = "inventory.tsv"
    # Toggle: when True, the inspector paints a purple tint on rows
    # the scanner flagged as ``suspected_abort``.
    highlight_aborts: bool = False

    # Scan defaults
    scan_n_jobs: int = 1
    # Default on: probe-convert runs dcm2niix per series at scan time to
    # enrich the BIDS guess with sidecar-derived hints.
    scan_probe_convert: bool = True
    # How wide to write each index entity in the names a scan proposes, as
    # ``{"run": 2}``. Empty means whatever the source says, which is what
    # the standard allows and what every earlier version did. Stored as JSON
    # because QSettings has no dict type and a flat key per entity would need
    # editing here every time BIDS adds one.
    scan_index_widths: dict = field(default_factory=dict)
    # Record what the conversion answers by itself, so the metadata form
    # shows it instead of an empty box for a field nobody has to fill in.
    scan_converter_preview: bool = True
    scan_skip_bids_guess: bool = False

    # Convert defaults
    convert_n_jobs: int = 1
    convert_overwrite: bool = False  # legacy; migrated into convert_on_existing
    # Policy when an incoming subject already exists during convert:
    # skip (default, keep existing) | update (replace changed) | replace
    # (back up + replace colliding) | error (abort on any collision).
    convert_on_existing: str = "skip"
    # Drop dcm2niix residual/secondary outputs (e.g. ``..._bolda`` next to
    # ``..._bold``). Default on: they are derived duplicates, not real images.
    convert_skip_residuals: bool = True
    # Re-converting a subject somebody already curated in the Editor:
    # merge the sidecars field by field rather than overwrite them, so an
    # afternoon of annotation survives the second pass. Only bites when a
    # file would otherwise be replaced.
    convert_preserve_curation: bool = True
    # Re-encode EEG / iEEG recordings to EDF on convert (mne-bids format="EDF").
    convert_force_edf: bool = False
    # Off by default. Defacing is destructive and must be chosen, not
    # discovered after the fact.
    convert_deface: bool = False
    convert_deface_engine: str = "allineate"

    # Post-convert chain. All steps on by default: run metadata + validation
    # with TODO placeholders, strict validation, and an HTML report.
    post_run_metadata: bool = True
    post_run_validate: bool = True
    post_metadata_fill_todos: bool = True
    # How much of what the standard declares the placeholder fill marks.
    # required | recommended | optional, nested. Used by the post-convert
    # chain AND by the Editor's Fix ups, so the two cannot disagree about
    # what the user asked for.
    metadata_fill_scope: str = "recommended"
    post_validate_strict: bool = True
    post_validate_html: bool = True
    # Dataset repairs, run after metadata and before validation. Both default
    # OFF: one adds files to the dataset and the other moves fields between
    # files, and a tool that does either without being asked is a tool whose
    # output cannot be trusted.
    post_fixup_companions: bool = False
    post_fixup_citation: bool = False

    # PyPI version string the user picked "Skip this version" on, so the
    # startup update check doesn't nag them about the same release on
    # every launch. Cleared implicitly when a newer version appears.
    skipped_update_version: str = ""

    # Global UI font-size multiplier. 1.0 = baseline sizing in
    # ``theme.qss``. The Settings dialog exposes four presets
    # (0.85 / 1.00 / 1.15 / 1.30) but any positive float is persisted.
    font_scale: float = 1.0

    # Header brand artwork.
    # "default"  → minimalist mark in ``assets/logo.png``, inverted on dark.
    # "app_icon" → full-color BIDS-Manager app icon (``assets/macos/AppIcon128.png``).
    header_logo: str = "default"

    # Scan rules (JSON-serialisable list[dict]); converted to/from the
    # engine's frozen dataclasses at the boundary via ``to_user_hints`` /
    # ``to_exclusions`` (keeps the classifier import out of this module's
    # hot path and the engine free of any GUI dependency).
    # hint:      {"patterns": [...], "datatype", "suffix", "task",
    #             "entities": {k: v}, "match_mode", "force"}
    # exclusion: {"pattern", "target": "sequence"|"path", "match_mode"}
    user_hints: list = field(default_factory=list)
    scan_exclusions: list = field(default_factory=list)

    # Recently opened/created project dataset roots (most-recent first).
    recent_projects: list = field(default_factory=list)

    # ------------------------------------------------------------------
    @staticmethod
    def _settings() -> QSettings:
        """Return a ``QSettings`` bound to the current QApplication's
        org/app names. The ``bidsmgr`` CLI entry point sets those once
        at startup; tests override them via :func:`QCoreApplication`
        so each test gets an isolated INI file.
        """
        return QSettings()

    @classmethod
    def load(cls) -> "AppSettings":
        s = cls._settings()
        out = cls()

        def _as_bool(v, default: bool) -> bool:
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in ("1", "true", "yes")

        def _as_int(v, default: int) -> int:
            try:
                return int(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        def _as_float(v, default: float) -> float:
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        def _as_str(v, default: str) -> str:
            return str(v) if v not in (None, "") else default

        def _as_json_list(v, default: list) -> list:
            """Parse a JSON list stored in QSettings; corrupt/non-list -> default."""
            if not v:
                return list(default)
            try:
                out = json.loads(v) if isinstance(v, str) else v
            except (ValueError, TypeError):
                return list(default)
            return out if isinstance(out, list) else list(default)

        out.theme = _as_str(s.value(KEYS["theme"]), out.theme)
        if out.theme not in ("dark", "light"):
            out.theme = "dark"
        out.active_view = _as_str(s.value(KEYS["active_view"]), out.active_view)
        if out.active_view not in ("converter", "editor"):
            out.active_view = "converter"
        out.editor_bids_root = s.value(KEYS["editor_bids_root"]) or None
        out.editor_sidecar_view = _as_str(
            s.value(KEYS["editor_sidecar_view"]), out.editor_sidecar_view,
        )
        if out.editor_sidecar_view not in ("bids", "tree"):
            out.editor_sidecar_view = "bids"
        out.editor_field_scope = _as_str(
            s.value(KEYS["editor_field_scope"]), out.editor_field_scope,
        )
        if out.editor_field_scope not in ("all", "present", "absent"):
            out.editor_field_scope = "all"
        out.editor_show_hidden = _as_bool(
            s.value(KEYS["editor_show_hidden"]), out.editor_show_hidden,
        )
        out.editor_autosave = _as_bool(
            s.value(KEYS["editor_autosave"]), out.editor_autosave,
        )
        out.editor_strict_validate = _as_bool(
            s.value(KEYS["editor_strict_validate"]),
            out.editor_strict_validate,
        )
        out.validate_schema_version = _as_str(
            s.value(KEYS["validate_schema_version"]),
            out.validate_schema_version,
        )
        out.validate_max_rows = _as_int(
            s.value(KEYS["validate_max_rows"]), out.validate_max_rows,
        )
        if out.validate_max_rows < 1:
            out.validate_max_rows = 1000
        out.validate_show = _as_str(
            s.value(KEYS["validate_show"]), out.validate_show,
        )
        if out.validate_show not in ("error_warning", "error", "warning"):
            out.validate_show = "error_warning"
        out.validate_flag_todos = _as_bool(
            s.value(KEYS["validate_flag_todos"]), out.validate_flag_todos,
        )
        try:
            out.trace_line_width = int(
                s.value(KEYS["trace_line_width"], out.trace_line_width)
            )
        except (TypeError, ValueError):
            pass
        # 0 is legal and means automatic; anything outside 1..8 is not.
        if out.trace_line_width and not (1 <= out.trace_line_width <= 8):
            out.trace_line_width = 0
        out.trace_line_color = _as_str(
            s.value(KEYS["trace_line_color"]), out.trace_line_color,
        )
        try:
            raw = s.value(KEYS["trace_type_colors"], "")
            parsed = json.loads(raw) if isinstance(raw, str) and raw else {}
            if isinstance(parsed, dict):
                out.trace_type_colors = {
                    str(k): str(v) for k, v in parsed.items() if v
                }
        except (TypeError, ValueError):
            pass
        out.nifti_view_mode = _as_str(
            s.value(KEYS["nifti_view_mode"]), out.nifti_view_mode,
        )
        if out.nifti_view_mode not in ("", "single", "multi", "3d", "combo"):
            out.nifti_view_mode = ""
        try:
            out.nifti_orientation = int(
                s.value(KEYS["nifti_orientation"], out.nifti_orientation)
            )
        except (TypeError, ValueError):
            pass
        if out.nifti_orientation not in (0, 1, 2):
            out.nifti_orientation = 2
        out.nifti_crosshair_color = _as_str(
            s.value(KEYS["nifti_crosshair_color"]),
            out.nifti_crosshair_color,
        )
        out.nifti_crosshair_thickness = _as_int(
            s.value(KEYS["nifti_crosshair_thickness"]),
            out.nifti_crosshair_thickness,
        )
        out.nifti_crosshair_thickness = max(
            1, min(out.nifti_crosshair_thickness, 5),
        )
        out.template_colour_levels = _as_bool(
            s.value(KEYS["template_colour_levels"]),
            out.template_colour_levels,
        )
        out.raw_root = s.value(KEYS["raw_root"]) or None
        out.bids_parent = s.value(KEYS["bids_parent"]) or None
        out.scan_tsv_filename = _as_str(
            s.value(KEYS["scan_tsv_filename"]), out.scan_tsv_filename,
        )
        out.highlight_aborts = _as_bool(
            s.value(KEYS["highlight_aborts"]), out.highlight_aborts,
        )

        out.scan_n_jobs = _as_int(s.value(KEYS["scan_n_jobs"]), out.scan_n_jobs)
        out.scan_probe_convert = _as_bool(s.value(KEYS["scan_probe_convert"]),
                                          out.scan_probe_convert)
        try:
            raw = s.value(KEYS["scan_index_widths"], "")
            parsed = json.loads(raw) if raw else {}
            out.scan_index_widths = {
                str(k): int(v) for k, v in parsed.items()
                if str(v).isdigit() and 1 <= int(v) <= 6
            } if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            out.scan_index_widths = {}
        out.scan_converter_preview = _as_bool(
            s.value(KEYS["scan_converter_preview"]), out.scan_converter_preview,
        )
        out.scan_skip_bids_guess = _as_bool(s.value(KEYS["scan_skip_bids_guess"]),
                                            out.scan_skip_bids_guess)

        out.convert_n_jobs = _as_int(s.value(KEYS["convert_n_jobs"]), out.convert_n_jobs)
        out.convert_overwrite = _as_bool(s.value(KEYS["convert_overwrite"]),
                                          out.convert_overwrite)
        # Migrate the legacy overwrite checkbox: if no explicit policy is stored
        # but overwrite was on, default the policy to "replace".
        out.convert_on_existing = _as_str(
            s.value(KEYS["convert_on_existing"]),
            "replace" if out.convert_overwrite else "skip",
        )
        if out.convert_on_existing not in ("skip", "update", "replace", "error"):
            out.convert_on_existing = "skip"
        out.convert_skip_residuals = _as_bool(
            s.value(KEYS["convert_skip_residuals"]), out.convert_skip_residuals,
        )
        out.convert_preserve_curation = _as_bool(
            s.value(KEYS["convert_preserve_curation"]),
            out.convert_preserve_curation,
        )
        out.convert_force_edf = _as_bool(
            s.value(KEYS["convert_force_edf"]), out.convert_force_edf,
        )
        out.convert_deface = _as_bool(
            s.value(KEYS["convert_deface"]), out.convert_deface,
        )
        out.convert_deface_engine = str(
            s.value(KEYS["convert_deface_engine"]) or out.convert_deface_engine
        )
        # Self-heal a value that is not an engine. Settings written by an
        # older build can be anything, and an id the converter cannot resolve
        # used to raise inside the subject commit, which lost the whole
        # conversion rather than just the defacing.
        from ..deface.engines import engine_ids as _deface_engine_ids

        if out.convert_deface_engine not in _deface_engine_ids():
            out.convert_deface_engine = cls.convert_deface_engine

        out.post_run_metadata = _as_bool(s.value(KEYS["post_run_metadata"]),
                                         out.post_run_metadata)
        out.post_run_validate = _as_bool(s.value(KEYS["post_run_validate"]),
                                         out.post_run_validate)
        out.post_metadata_fill_todos = _as_bool(s.value(KEYS["post_metadata_fill_todos"]),
                                                out.post_metadata_fill_todos)
        out.metadata_fill_scope = _as_str(
            s.value(KEYS["metadata_fill_scope"]), out.metadata_fill_scope,
        )
        if out.metadata_fill_scope not in (
            "required", "recommended", "optional",
        ):
            out.metadata_fill_scope = "recommended"
        out.post_validate_strict = _as_bool(s.value(KEYS["post_validate_strict"]),
                                            out.post_validate_strict)
        out.post_validate_html = _as_bool(s.value(KEYS["post_validate_html"]),
                                          out.post_validate_html)
        out.post_fixup_companions = _as_bool(
            s.value(KEYS["post_fixup_companions"]), out.post_fixup_companions,
        )
        out.post_fixup_citation = _as_bool(
            s.value(KEYS["post_fixup_citation"]), out.post_fixup_citation,
        )
        out.skipped_update_version = _as_str(
            s.value(KEYS["skipped_update_version"]),
            out.skipped_update_version,
        )
        out.font_scale = _as_float(s.value(KEYS["font_scale"]), out.font_scale)
        # Clamp to a sensible range so a corrupted setting can't render
        # the GUI unusable.
        if out.font_scale <= 0:
            out.font_scale = 1.0
        out.font_scale = max(0.5, min(out.font_scale, 2.0))
        out.header_logo = _as_str(
            s.value(KEYS["header_logo"]), out.header_logo,
        )
        if out.header_logo not in ("default", "app_icon"):
            out.header_logo = "default"
        out.user_hints = _as_json_list(s.value(KEYS["user_hints"]), [])
        out.scan_exclusions = _as_json_list(s.value(KEYS["scan_exclusions"]), [])
        out.recent_projects = _as_json_list(s.value(KEYS["recent_projects"]), [])
        return out

    def save(self) -> None:
        s = self._settings()
        # Strings.
        s.setValue(KEYS["theme"], self.theme)
        s.setValue(KEYS["scan_tsv_filename"], self.scan_tsv_filename)
        s.setValue(KEYS["editor_field_scope"], self.editor_field_scope)
        if self.raw_root is not None:
            s.setValue(KEYS["raw_root"], self.raw_root)
        if self.bids_parent is not None:
            s.setValue(KEYS["bids_parent"], self.bids_parent)
        # Ints / floats.
        s.setValue(KEYS["scan_n_jobs"], int(self.scan_n_jobs))
        s.setValue(KEYS["convert_n_jobs"], int(self.convert_n_jobs))
        # Bools — store as strings so the load path doesn't depend on
        # platform-specific QVariant→Python bool quirks.
        for key, val in (
            ("scan_probe_convert",       self.scan_probe_convert),
            ("scan_converter_preview",   self.scan_converter_preview),
            ("scan_skip_bids_guess",     self.scan_skip_bids_guess),
            ("convert_overwrite",        self.convert_overwrite),
            ("convert_skip_residuals",   self.convert_skip_residuals),
            ("convert_preserve_curation", self.convert_preserve_curation),
            ("convert_force_edf",        self.convert_force_edf),
            ("convert_deface",           self.convert_deface),
            ("post_run_metadata",        self.post_run_metadata),
            ("post_run_validate",        self.post_run_validate),
            ("post_metadata_fill_todos", self.post_metadata_fill_todos),
            ("metadata_fill_scope",      self.metadata_fill_scope),
            ("post_validate_strict",     self.post_validate_strict),
            ("post_validate_html",       self.post_validate_html),
            ("post_fixup_companions",    self.post_fixup_companions),
            ("post_fixup_citation",      self.post_fixup_citation),
            ("editor_strict_validate",   self.editor_strict_validate),
            ("validate_flag_todos",      self.validate_flag_todos),
            ("editor_show_hidden",       self.editor_show_hidden),
            ("editor_autosave",          self.editor_autosave),
        ):
            s.setValue(KEYS[key], "1" if val else "0")
        # Strings. Keep them OUT of the loop above: it writes "1" for anything
        # truthy, so a string setting put there is saved as "1" and read back
        # as "1". That is not a hypothetical. `convert_deface_engine` was in
        # that list, every conversion loaded the engine id "1", and the lookup
        # raised inside the subject commit, so nothing converted at all.
        s.setValue(KEYS["convert_on_existing"], self.convert_on_existing)
        s.setValue(KEYS["convert_deface_engine"], self.convert_deface_engine)
        s.setValue(KEYS["trace_line_width"], int(self.trace_line_width))
        s.setValue(KEYS["trace_line_color"], self.trace_line_color)
        s.setValue(KEYS["trace_type_colors"], json.dumps(self.trace_type_colors))
        s.setValue(KEYS["nifti_view_mode"], self.nifti_view_mode)
        s.setValue(KEYS["scan_index_widths"], json.dumps(self.scan_index_widths))
        s.setValue(KEYS["nifti_orientation"], int(self.nifti_orientation))
        s.setValue(KEYS["validate_schema_version"], self.validate_schema_version)
        s.setValue(KEYS["validate_max_rows"], int(self.validate_max_rows))
        s.setValue(KEYS["validate_show"], self.validate_show)
        s.setValue(KEYS["skipped_update_version"], self.skipped_update_version)
        s.setValue(KEYS["font_scale"], float(self.font_scale))
        s.setValue(KEYS["header_logo"], self.header_logo)
        # Scan rules as JSON blobs.
        s.setValue(KEYS["user_hints"], json.dumps(self.user_hints))
        s.setValue(KEYS["scan_exclusions"], json.dumps(self.scan_exclusions))
        s.setValue(KEYS["recent_projects"], json.dumps(self.recent_projects))
        s.sync()

    # ------------------------------------------------------------------
    # Convenience helpers used by panels when they only need to persist
    # one value (e.g. last raw_root after the file dialog).
    # ------------------------------------------------------------------

    @classmethod
    def remember_raw_root(cls, path: Path) -> None:
        cls._settings().setValue(KEYS["raw_root"], str(path))

    @classmethod
    def remember_bids_parent(cls, path: Path) -> None:
        cls._settings().setValue(KEYS["bids_parent"], str(path))

    @classmethod
    def remember_tsv_filename(cls, filename: str) -> None:
        cls._settings().setValue(KEYS["scan_tsv_filename"], filename)

    @classmethod
    def remember_highlight_aborts(cls, enabled: bool) -> None:
        cls._settings().setValue(
            KEYS["highlight_aborts"], "1" if enabled else "0",
        )

    @classmethod
    def remember_recent_project(cls, path: Path, *, cap: int = 10) -> None:
        """Push a project dataset root to the front of the recent list.

        De-duplicates (most-recent-first) and caps the list length so the
        Welcome tab stays tidy.
        """
        s = cls._settings()
        existing = cls.load().recent_projects
        p = str(path)
        out = [p] + [x for x in existing if x != p]
        s.setValue(KEYS["recent_projects"], json.dumps(out[:cap]))

    @classmethod
    def forget_recent_project(cls, path: Path) -> None:
        """Drop a project from the recent list (does not touch the dataset)."""
        s = cls._settings()
        p = str(path)
        out = [x for x in cls.load().recent_projects if x != p]
        s.setValue(KEYS["recent_projects"], json.dumps(out))

    @classmethod
    def remember_theme(cls, theme: str) -> None:
        cls._settings().setValue(KEYS["theme"], theme)

    @classmethod
    def remember_active_view(cls, view: str) -> None:
        cls._settings().setValue(KEYS["active_view"], view)

    @classmethod
    def remember_editor_bids_root(cls, path: Path) -> None:
        cls._settings().setValue(KEYS["editor_bids_root"], str(path))

    @classmethod
    def remember_editor_sidecar_view(cls, view: str) -> None:
        cls._settings().setValue(KEYS["editor_sidecar_view"], view)

    @classmethod
    def remember_editor_field_scope(cls, scope: str) -> None:
        cls._settings().setValue(KEYS["editor_field_scope"], scope)

    @classmethod
    def remember_editor_show_hidden(cls, show: bool) -> None:
        cls._settings().setValue(KEYS["editor_show_hidden"], bool(show))

    @classmethod
    def remember_editor_strict_validate(cls, enabled: bool) -> None:
        cls._settings().setValue(
            KEYS["editor_strict_validate"], "1" if enabled else "0",
        )

    # ------------------------------------------------------------------
    # Scan-rules boundary: list[dict] (JSON) <-> engine frozen dataclasses.
    # The conversion lives here so the engine never imports settings and the
    # GUI / CLI share one (de)serialiser (``classifier.user_rules``).
    # ------------------------------------------------------------------

    def to_user_hints(self) -> list:
        """Return the persisted hints as ``list[UserHint]`` for the scanner."""
        from ..classifier import user_rules
        hints, _ = user_rules.from_json({"user_hints": self.user_hints})
        return hints

    def to_exclusions(self) -> list:
        """Return the persisted exclusions as ``list[ExclusionRule]``."""
        from ..classifier import user_rules
        _, excl = user_rules.from_json({"scan_exclusions": self.scan_exclusions})
        return excl

    @classmethod
    def remember_trace_style(cls, width: int, colour: str) -> None:
        """Store how a trace is drawn. One preference for every viewer."""
        cls._settings().setValue(KEYS["trace_line_width"], int(width))
        cls._settings().setValue(KEYS["trace_line_color"], str(colour))

    @classmethod
    def remember_type_colors(cls, mapping: dict) -> None:
        """Store per-channel-type trace colours. Empty restores the defaults."""
        clean = {str(k): str(v) for k, v in dict(mapping or {}).items() if v}
        cls._settings().setValue(
            KEYS["trace_type_colors"], json.dumps(clean),
        )

    @classmethod
    def remember_nifti_view_mode(cls, mode: str) -> None:
        """Store the layout the user is in, so the next scan opens in it.

        Written as the user switches rather than at shut-down: the Editor is
        not always closed cleanly, and a preference that only survives a
        graceful exit is one that mostly does not survive.
        """
        if mode not in ("single", "multi", "3d", "combo"):
            return
        cls._settings().setValue(KEYS["nifti_view_mode"], str(mode))

    @classmethod
    def remember_nifti_orientation(cls, axis: int) -> None:
        """Store the plane, so a single-pane view opens on the same one."""
        if int(axis) not in (0, 1, 2):
            return
        cls._settings().setValue(KEYS["nifti_orientation"], int(axis))

    @classmethod
    def remember_nifti_crosshair(cls, color: str, thickness: int) -> None:
        s = cls._settings()
        s.setValue(KEYS["nifti_crosshair_color"], str(color))
        s.setValue(
            KEYS["nifti_crosshair_thickness"],
            int(max(1, min(thickness, 5))),
        )


__all__ = ["AppSettings", "KEYS"]

"""Typed wrapper over ``QSettings`` for cross-platform persistence.

``QSettings`` stores values per-platform in the right native location:

* macOS  → ``~/Library/Preferences/com.bidsmgr.bidsmgr.plist``
* Linux  → ``~/.config/bidsmgr/bidsmgr.conf``
* Windows → ``HKEY_CURRENT_USER\Software\bidsmgr\bidsmgr`` (registry)

This module wraps it with type-safe getters / setters and one schema
the rest of the GUI can rely on. Keep the keys in :data:`KEYS` —
adding a new one outside that namespace is a code smell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings


# Canonical setting keys. Grouped by section as a flat string namespace
# so QSettings shows them under ``[section]`` headers in INI / plist.
KEYS = {
    "theme":              "ui/theme",                # "dark" | "light"
    "raw_root":           "paths/raw_root",          # last raw input dir
    "bids_parent":        "paths/bids_parent",       # last BIDS output dir
    "dataset_slug":       "scan/dataset_slug",       # default dataset name
    "scan_tsv_filename":  "scan/tsv_filename",       # filename of the scan TSV
    "highlight_aborts":   "inspector/highlight_aborts",   # toolbar toggle
    "active_view":        "ui/active_view",          # "converter" | "editor"
    "editor_bids_root":   "editor/bids_root",        # last BIDS root opened in the Editor view
    "editor_sidecar_view": "editor/sidecar_view",    # "bids" | "tree"
    "editor_strict_validate": "editor/strict_validate",  # layer 2 (bidsschematools) on/off
    "nifti_crosshair_color": "editor/nifti_crosshair_color",   # hex string e.g. "#4FC3F7"
    "nifti_crosshair_thickness": "editor/nifti_crosshair_thickness",  # px, 1..5
    # Scan defaults
    "scan_n_jobs":        "scan/n_jobs",
    "scan_probe_convert": "scan/probe_convert",
    "scan_line_freq":     "scan/line_freq",
    "scan_montage":       "scan/montage",
    "scan_skip_bids_guess": "scan/skip_bids_guess",
    # Convert defaults
    "convert_n_jobs":     "convert/n_jobs",
    "convert_overwrite":  "convert/overwrite",
    # Post-convert chain
    "post_run_metadata":  "post_convert/run_metadata",
    "post_run_validate":  "post_convert/run_validate",
    "post_metadata_fill_todos": "post_convert/metadata_fill_todos",
    "post_validate_strict": "post_convert/validate_strict",
    "post_validate_html": "post_convert/validate_html",
    # Self-update
    "skipped_update_version": "update/skipped_version",
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
    # When True, "Validate dataset" runs ``bidsschematools.validator``
    # (the official Python BIDS validator) in addition to bidsmgr's
    # schema-driven layer 1 checks.
    editor_strict_validate: bool = False
    # NIfTI viewer crosshair style. Persisted so the user's chosen
    # colour + thickness survives across sessions.
    nifti_crosshair_color: str = "#4FC3F7"
    nifti_crosshair_thickness: int = 1

    # Recently-used paths (paths come back as str; callers wrap in Path).
    raw_root: Optional[str] = None
    bids_parent: Optional[str] = None
    dataset_slug: str = ""
    # Scan-TSV filename only (the TSV lives under ``<bids_parent>/``).
    # The user can override this from the toolbar field.
    scan_tsv_filename: str = "inventory.tsv"
    # Toggle: when True, the inspector paints a purple tint on rows
    # the scanner flagged as ``suspected_abort``.
    highlight_aborts: bool = False

    # Scan defaults
    scan_n_jobs: int = 1
    scan_probe_convert: bool = False
    scan_line_freq: float = 50.0
    scan_montage: str = ""
    scan_skip_bids_guess: bool = False

    # Convert defaults
    convert_n_jobs: int = 1
    convert_overwrite: bool = False

    # Post-convert chain
    post_run_metadata: bool = True
    post_run_validate: bool = True
    post_metadata_fill_todos: bool = True
    post_validate_strict: bool = False
    post_validate_html: bool = False

    # PyPI version string the user picked "Skip this version" on, so the
    # startup update check doesn't nag them about the same release on
    # every launch. Cleared implicitly when a newer version appears.
    skipped_update_version: str = ""

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
        out.editor_strict_validate = _as_bool(
            s.value(KEYS["editor_strict_validate"]),
            out.editor_strict_validate,
        )
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
        out.raw_root = s.value(KEYS["raw_root"]) or None
        out.bids_parent = s.value(KEYS["bids_parent"]) or None
        out.dataset_slug = _as_str(s.value(KEYS["dataset_slug"]), out.dataset_slug)
        out.scan_tsv_filename = _as_str(
            s.value(KEYS["scan_tsv_filename"]), out.scan_tsv_filename,
        )
        out.highlight_aborts = _as_bool(
            s.value(KEYS["highlight_aborts"]), out.highlight_aborts,
        )

        out.scan_n_jobs = _as_int(s.value(KEYS["scan_n_jobs"]), out.scan_n_jobs)
        out.scan_probe_convert = _as_bool(s.value(KEYS["scan_probe_convert"]),
                                          out.scan_probe_convert)
        out.scan_line_freq = _as_float(s.value(KEYS["scan_line_freq"]), out.scan_line_freq)
        out.scan_montage = _as_str(s.value(KEYS["scan_montage"]), out.scan_montage)
        out.scan_skip_bids_guess = _as_bool(s.value(KEYS["scan_skip_bids_guess"]),
                                            out.scan_skip_bids_guess)

        out.convert_n_jobs = _as_int(s.value(KEYS["convert_n_jobs"]), out.convert_n_jobs)
        out.convert_overwrite = _as_bool(s.value(KEYS["convert_overwrite"]),
                                          out.convert_overwrite)

        out.post_run_metadata = _as_bool(s.value(KEYS["post_run_metadata"]),
                                         out.post_run_metadata)
        out.post_run_validate = _as_bool(s.value(KEYS["post_run_validate"]),
                                         out.post_run_validate)
        out.post_metadata_fill_todos = _as_bool(s.value(KEYS["post_metadata_fill_todos"]),
                                                out.post_metadata_fill_todos)
        out.post_validate_strict = _as_bool(s.value(KEYS["post_validate_strict"]),
                                            out.post_validate_strict)
        out.post_validate_html = _as_bool(s.value(KEYS["post_validate_html"]),
                                          out.post_validate_html)
        out.skipped_update_version = _as_str(
            s.value(KEYS["skipped_update_version"]),
            out.skipped_update_version,
        )
        return out

    def save(self) -> None:
        s = self._settings()
        # Strings.
        s.setValue(KEYS["theme"], self.theme)
        s.setValue(KEYS["dataset_slug"], self.dataset_slug)
        s.setValue(KEYS["scan_tsv_filename"], self.scan_tsv_filename)
        s.setValue(KEYS["scan_montage"], self.scan_montage)
        if self.raw_root is not None:
            s.setValue(KEYS["raw_root"], self.raw_root)
        if self.bids_parent is not None:
            s.setValue(KEYS["bids_parent"], self.bids_parent)
        # Ints / floats.
        s.setValue(KEYS["scan_n_jobs"], int(self.scan_n_jobs))
        s.setValue(KEYS["scan_line_freq"], float(self.scan_line_freq))
        s.setValue(KEYS["convert_n_jobs"], int(self.convert_n_jobs))
        # Bools — store as strings so the load path doesn't depend on
        # platform-specific QVariant→Python bool quirks.
        for key, val in (
            ("scan_probe_convert",       self.scan_probe_convert),
            ("scan_skip_bids_guess",     self.scan_skip_bids_guess),
            ("convert_overwrite",        self.convert_overwrite),
            ("post_run_metadata",        self.post_run_metadata),
            ("post_run_validate",        self.post_run_validate),
            ("post_metadata_fill_todos", self.post_metadata_fill_todos),
            ("post_validate_strict",     self.post_validate_strict),
            ("post_validate_html",       self.post_validate_html),
            ("editor_strict_validate",   self.editor_strict_validate),
        ):
            s.setValue(KEYS[key], "1" if val else "0")
        s.setValue(KEYS["skipped_update_version"], self.skipped_update_version)
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
    def remember_dataset_slug(cls, slug: str) -> None:
        cls._settings().setValue(KEYS["dataset_slug"], slug)

    @classmethod
    def remember_tsv_filename(cls, filename: str) -> None:
        cls._settings().setValue(KEYS["scan_tsv_filename"], filename)

    @classmethod
    def remember_highlight_aborts(cls, enabled: bool) -> None:
        cls._settings().setValue(
            KEYS["highlight_aborts"], "1" if enabled else "0",
        )

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
    def remember_editor_strict_validate(cls, enabled: bool) -> None:
        cls._settings().setValue(
            KEYS["editor_strict_validate"], "1" if enabled else "0",
        )

    @classmethod
    def remember_nifti_crosshair(cls, color: str, thickness: int) -> None:
        s = cls._settings()
        s.setValue(KEYS["nifti_crosshair_color"], str(color))
        s.setValue(
            KEYS["nifti_crosshair_thickness"],
            int(max(1, min(thickness, 5))),
        )


__all__ = ["AppSettings", "KEYS"]

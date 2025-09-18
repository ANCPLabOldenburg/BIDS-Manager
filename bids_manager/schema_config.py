"""Public wrapper around the renaming configuration helpers."""

from __future__ import annotations

try:
    from ._renaming_loader import load_module
except ImportError:  # pragma: no cover - fallback for direct execution
    # See ``schema_renamer`` for the rationale.  When the package is not
    # initialised (e.g. running helper scripts from a source checkout) the
    # relative import fails, so fall back to the plain module import.
    from _renaming_loader import load_module  # type: ignore

_config = load_module("config")

# Copy the exported symbols so callers keep interacting with this thin wrapper
# instead of the implementation module directly.  Using ``getattr`` avoids mypy
# complaints while keeping the code compact and explicit.
DEFAULT_SCHEMA_DIR = getattr(_config, "DEFAULT_SCHEMA_DIR")
ENABLE_SCHEMA_RENAMER = getattr(_config, "ENABLE_SCHEMA_RENAMER")
ENABLE_FIELDMap_NORMALIZATION = getattr(_config, "ENABLE_FIELDMap_NORMALIZATION")
ENABLE_DWI_DERIVATIVES_MOVE = getattr(_config, "ENABLE_DWI_DERIVATIVES_MOVE")
DERIVATIVES_PIPELINE_NAME = getattr(_config, "DERIVATIVES_PIPELINE_NAME")

__all__ = (
    "DEFAULT_SCHEMA_DIR",
    "ENABLE_SCHEMA_RENAMER",
    "ENABLE_FIELDMap_NORMALIZATION",
    "ENABLE_DWI_DERIVATIVES_MOVE",
    "DERIVATIVES_PIPELINE_NAME",
)


"""Configuration helpers for the schema-driven renaming utilities.

The implementation intentionally lives in the plain ``bids_manager/renaming``
folder (which lacks an ``__init__``) so the renaming logic is grouped together
without forming an additional Python package.  The public API is re-exported
via thin wrappers in :mod:`bids_manager.schema_config`.
"""

from __future__ import annotations

from pathlib import Path

# ``Path(__file__)`` points to ``bids_manager/renaming/config.py``.  Hopping two
# levels up lands on the installed ``bids_manager`` package directory regardless
# of whether the project is executed from a source checkout or from an installed
# wheel.  From there we can reliably reach the bundled schema files.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA_DIR = _PACKAGE_ROOT / "miscellaneous" / "schema"

# Feature toggles exposed to the GUI.  They remain defined here so the GUI and
# CLI wrappers can keep importing from :mod:`bids_manager.schema_config` without
# noticing that the underlying implementation moved into the ``renaming``
# folder.
ENABLE_SCHEMA_RENAMER = True
ENABLE_FIELDMap_NORMALIZATION = True
ENABLE_DWI_DERIVATIVES_MOVE = True
DERIVATIVES_PIPELINE_NAME = "dcm2niix"  # or "BIDS-Manager" if you prefer

__all__ = [
    "DEFAULT_SCHEMA_DIR",
    "ENABLE_SCHEMA_RENAMER",
    "ENABLE_FIELDMap_NORMALIZATION",
    "ENABLE_DWI_DERIVATIVES_MOVE",
    "DERIVATIVES_PIPELINE_NAME",
]


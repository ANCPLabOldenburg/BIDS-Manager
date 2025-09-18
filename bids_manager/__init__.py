"""BIDS Manager package."""

from importlib import metadata

# The schema-driven renaming helpers live in the plain ``renaming`` folder
# (without ``__init__``) and are exposed through the wrappers below.  Import
# them here so callers can continue to access the functionality via the
# canonical ``bids_manager`` module.
from .schema_config import (
    DEFAULT_SCHEMA_DIR,
    DERIVATIVES_PIPELINE_NAME,
    ENABLE_DWI_DERIVATIVES_MOVE,
    ENABLE_FIELDMap_NORMALIZATION,
    ENABLE_SCHEMA_RENAMER,
)
from .schema_renamer import (
    SchemaInfo,
    SeriesInfo,
    apply_post_conversion_rename,
    build_preview_names,
    load_bids_schema,
    propose_bids_basename,
)

__all__ = [
    "__version__",
    "DEFAULT_SCHEMA_DIR",
    "DERIVATIVES_PIPELINE_NAME",
    "ENABLE_DWI_DERIVATIVES_MOVE",
    "ENABLE_FIELDMap_NORMALIZATION",
    "ENABLE_SCHEMA_RENAMER",
    "SchemaInfo",
    "SeriesInfo",
    "apply_post_conversion_rename",
    "build_preview_names",
    "load_bids_schema",
    "propose_bids_basename",
]

try:  # pragma: no cover - version resolution
    __version__ = metadata.version("bids-manager")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


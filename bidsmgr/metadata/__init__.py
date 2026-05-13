"""Schema-driven post-conversion metadata engine.

Produces dataset-level files BIDS expects but that dcm2niix doesn't
write: ``dataset_description.json`` (merged with what the converter
already wrote), ``participants.tsv`` + ``participants.json``, ``README``,
``CHANGES``, per-subject ``*_scans.tsv``, and a schema-driven sidecar
audit (required + recommended fields per ``(datatype, suffix)``).

Reference: architecture.md §7 tail. Port of v0.2.5
``BIDS-Manager/bids_manager/bids_metadata_engine.py``, refactored to
function-style (architectural rule 6) and rebuilt on
:mod:`bidsmgr.schema` instead of v0.2.5's hardcoded required-field
table.
"""

from .engine import run_metadata
from .types import DatasetMetadata, MetadataReport, SidecarFill

__all__ = ["DatasetMetadata", "MetadataReport", "SidecarFill", "run_metadata"]

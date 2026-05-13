"""Pure-data types for the metadata engine.

Reference: architecture.md §7 tail. ``DatasetMetadata`` is the
user-supplied input; ``MetadataReport`` is what the engine returns
(files written, sidecar fills applied, warnings to surface to the user).
Both are Pydantic v2 models with no I/O methods (architectural rule 5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DatasetMetadata(BaseModel):
    """Caller-supplied fields for ``dataset_description.json``.

    Anything left blank falls back to a sane default. Only ``name`` is
    truly required by the BIDS spec; everything else has a workable
    default and is omitted from the written JSON when empty.
    """

    model_config = ConfigDict(frozen=False)

    name: str = "Untitled BIDS Dataset"
    bids_version: str = "1.10.0"
    dataset_type: str = "raw"  # 'raw' | 'derivative'
    license: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    acknowledgements: Optional[str] = None
    how_to_acknowledge: Optional[str] = None
    funding: list[str] = Field(default_factory=list)
    ethics_approvals: list[str] = Field(default_factory=list)
    references_and_links: list[str] = Field(default_factory=list)
    dataset_doi: Optional[str] = None
    source_datasets: list[dict[str, str]] = Field(default_factory=list)


class SidecarFill(BaseModel):
    """One filename-derived fill applied to a sidecar JSON.

    Today this is just ``TaskName`` derived from the ``_task-<label>``
    token in the filename. Distinct from :class:`TodoFill`, which
    inserts placeholder ``"TODO"`` strings for fields the engine
    cannot derive.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sidecar: Path
    fields: dict[str, object]


class TodoFill(BaseModel):
    """Placeholder ``"TODO"`` strings inserted into one sidecar.

    Created only when ``fill_todos=True`` is passed to ``run_metadata``.
    Each entry records which fields were missing and got the literal
    string ``"TODO"`` written. Existing values are never overwritten.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sidecar: Path
    fields: list[str]


class MetadataReport(BaseModel):
    """What ``run_metadata`` did and what it could not do.

    Written as JSON to ``<bids_root>/.bidsmgr/metadata_report.json`` on
    every run. The CLI prints a summary to stdout; the future GUI binds
    this directly into a tree view. Warnings are advisory — the engine
    never raises on validation failures, it records them here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bidsmgr_version: str = ""
    bids_root: Optional[Path] = None
    generated_at: Optional[str] = None
    files_written: list[Path] = Field(default_factory=list)
    sidecar_fills: list[SidecarFill] = Field(default_factory=list)
    todo_fills: list[TodoFill] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    missing_recommended: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


__all__ = ["DatasetMetadata", "MetadataReport", "SidecarFill", "TodoFill"]

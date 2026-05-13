"""Pure-data types emitted by inventory scanners.

Reference: architecture.md §2.1.

``InventoryRow`` is the modality-agnostic record produced by every
per-modality scanner. The classifier consumes it; the planner consumes
classifications produced from it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

Modality = Literal["mri", "eeg", "meg", "ieeg", "pet", "physio", "nirs", "unknown"]


class InventoryRow(BaseModel):
    """One row per discovered recording, regardless of modality.

    ``raw_metadata`` is whatever the scanner extracted (DICOM headers, mne
    raw.info, EDF header, …). It is the evidence pool for downstream
    classification and identity inference (architecture.md §4).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    row_id: UUID = Field(default_factory=uuid4)
    modality: Modality = "unknown"
    source: Path
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    n_files: int = 0
    discovered_at: datetime = Field(default_factory=datetime.utcnow)

    # Identity hints filled by inventory scanners. The planner / GUI may
    # override these; the originals stay in ``raw_metadata``.
    subject_hint: Optional[str] = None
    session_hint: Optional[str] = None
    task_hint: Optional[str] = None
    series_uid: Optional[str] = None  # MRI-specific but cheap to keep here.
    series_description: Optional[str] = None
    acq_time: Optional[str] = None  # for cross-row run ordering
    fine_modality: Optional[str] = None  # legacy regex label (T1w, bold, …)
    image_type: Optional[str] = None  # third DICOM ImageType element (M / P / ND / …)


__all__ = ["InventoryRow", "Modality"]

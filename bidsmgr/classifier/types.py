"""Pure-data types produced by the classifier chain.

Reference: architecture.md §2.2, §4.2.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Classification(BaseModel):
    """One candidate (datatype, suffix) classification with confidence + reason.

    A single ``InventoryRow`` may produce multiple ``Classification`` records,
    one per classifier that fires. The planner picks the highest-confidence
    schema-valid one (or the user picks manually).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    row_id: UUID
    classifier: str  # e.g. "dcm2niix_bidsguess", "mne_channel_types", "sequence_dict"
    datatype: str
    suffix: str
    candidate_entities: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0  # 0.0–1.0
    rationale: str = ""

    # Some classifiers (notably ``dcm2niix_bidsguess``) produce a "discard"
    # verdict for derived/localizer scans the user almost never wants.
    skip: bool = False


__all__ = ["Classification"]

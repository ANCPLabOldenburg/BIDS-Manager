"""The inventory columns that hold curation decisions, appended last so every
earlier column keeps its index (the unified TSV is 66 columns).

* ``fieldmap_targets``: for a fieldmap row, the runs the user chose for it
  to correct (a JSON list of row ids; empty = the linking rule decides,
  ``fixups.intended_for``).
* ``classification_confidence``, ``classification_alternatives``,
  ``classification_evidence``: what the acquisition-parameter classifier
  concluded for the row, how sure it is, the next candidates and the
  strongest evidence (``classifier.acquisition``).
"""

from __future__ import annotations

CURATION_COLUMNS: tuple[str, ...] = (
    "fieldmap_targets",
    "classification_confidence",
    "classification_alternatives",
    "classification_evidence",
)

__all__ = ["CURATION_COLUMNS"]

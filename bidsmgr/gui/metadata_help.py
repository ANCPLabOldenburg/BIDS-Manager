"""Schema-sourced tooltips for the recording-metadata fields.

Every editable metadata field maps to a real BIDS sidecar key (or a documented
BIDS convention). The tooltip text is pulled live from the ``bidsschematools``
schema so it always matches the installed BIDS version - which also guarantees
the field we expose is a genuine schema key. A small set of fallbacks covers the
handful of fields that are conventions (montage) or participants.tsv columns
rather than sidecar-metadata keys.
"""

from __future__ import annotations

import functools
import re

from .. import schema as schema_mod
from ..recording_meta.chain import _ACQ_TO_BIDS

# UI field key -> BIDS ``objects.metadata`` key.
#
# Read from the chain rather than listed here. This module used to keep its own
# copy, so "which BIDS field does ``dewar_position`` mean" had two answers that
# had to be kept equal by hand: a tooltip could describe one field while the
# converter wrote another. The chain owns that translation because it is what
# the chain does.
_FIELD_BIDS_KEY: dict[str, str] = {
    attr: names[0] for attr, names in _ACQ_TO_BIDS if names
}
# One alias the table adds: a read-only column showing what the scan detected,
# which documents the same field.
_FIELD_BIDS_KEY["manufacturer_suggestion"] = "Manufacturer"
_FIELD_BIDS_KEY["line_freq"] = "PowerLineFrequency"

# Fields that are BIDS conventions or participants.tsv columns (not sidecar
# metadata keys), so they are not in ``objects.metadata``.
_FALLBACKS: dict[str, str] = {
    "montage": (
        "MNE montage applied on conversion; sets the electrode positions written "
        "to electrodes.tsv. A BIDS convention, not a sidecar key."
    ),
    "PatientSex": "Participant sex (M / F / O). Written to participants.tsv (sex column).",
    "PatientAge": "Participant age in years at acquisition. Written to participants.tsv (age column).",
    "Handedness": "Participant handedness (R / L / A). Written to participants.tsv (handedness column).",
    "event": (
        "Map a recorded trigger code to a human-readable trial_type label in "
        "events.tsv. Blank labels are left untouched."
    ),
}


def _clean(text: str) -> str:
    """Strip BIDS markdown (links, backticks) and collapse whitespace."""
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    t = t.replace("`", "")
    return re.sub(r"\s+", " ", t).strip()


@functools.lru_cache(maxsize=256)
def bids_tooltip(bids_key: str) -> str:
    """Cleaned ``<display name>: <description>`` from the schema, or ``""``."""
    try:
        fi = schema_mod.field_metadata(bids_key)
    except Exception:
        return ""
    desc = _clean(fi.description)
    return f"{fi.display_name}: {desc}" if desc else fi.display_name


def tooltip_for(ui_field_key: str) -> str:
    """Tooltip for a metadata UI field (schema description, else a fallback)."""
    bids_key = _FIELD_BIDS_KEY.get(ui_field_key)
    if bids_key:
        t = bids_tooltip(bids_key)
        if t:
            return t
    return _FALLBACKS.get(ui_field_key, "")


__all__ = ["bids_tooltip", "tooltip_for"]

"""Anonymise a BIDS dataset's metadata (Qt-free).

``plan_anonymise`` lists every change a policy makes, ``apply_anonymise``
makes them (through an editor operation: one undoable step). Categories by
purpose and the presets that set them in ``policy``; which part of the
dataset in ``scope``; what each kind of file carries in ``carriers``; the
per-dataset memory (date offsets, the pseudonym secret) in ``state``.
"""

from .plan import AnonymisePlan, FilePlan, anonymise, apply_anonymise, plan_anonymise
from .policy import (
    CATEGORIES,
    CATEGORY,
    CUSTOM,
    OFF,
    PRESET_LABELS,
    PRESETS,
    PURPOSES,
    RECOMMENDED,
    STRICT,
    AnonymisePolicy,
    policy_for,
)
from .scope import Scope, dataset_units

__all__ = [
    "AnonymisePlan", "AnonymisePolicy", "CATEGORIES", "CATEGORY", "CUSTOM", "FilePlan", "OFF",
    "PRESETS", "PRESET_LABELS", "PURPOSES", "RECOMMENDED", "STRICT", "Scope", "anonymise",
    "apply_anonymise", "dataset_units", "plan_anonymise", "policy_for",
]

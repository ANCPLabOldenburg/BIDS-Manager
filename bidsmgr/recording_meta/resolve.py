"""Resolve the effective enrichment for one recording.

Merges the dataset-wide ``defaults`` with the per-recording ``overrides`` and
picks the task-scoped event map and protocol. The result is a flat object the
enrichment step consumes. Per-row inventory cells (montage, line_freq, reference,
ground) take precedence over the resolved spec and are layered on by the caller
(the convert verb), so this module stays spec-only and pure.

Precedence (low to high): ``defaults`` < ``overrides[row_id]`` < inventory cells.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from .models import (
    AcquisitionSpec,
    EventMap,
    PetAcquisitionSpec,
    RecordingMetaSpec,
    TaskProtocol,
)

_GLOBAL_EVENT_KEY = "*"


class EffectiveSpec(BaseModel):
    """The resolved enrichment for a single recording."""

    model_config = ConfigDict(extra="forbid")

    acquisition: AcquisitionSpec
    task_protocol: Optional[TaskProtocol] = None
    event_map: EventMap = {}


def merge_acquisition(
    base: AcquisitionSpec, over: Optional[AcquisitionSpec]
) -> AcquisitionSpec:
    """Return ``base`` with every set field of ``over`` layered on top.

    Scalars: a non-None override value wins. ``aux_channels``: the two maps are
    merged key-by-key (override entries replace same-named base entries).
    ``filters``: a non-empty override list replaces the base list outright (the
    set of filters is treated as a unit, not merged element-wise).
    """
    if over is None:
        return base.model_copy(deep=True)

    merged = base.model_dump()
    over_dump = over.model_dump(exclude_none=True)

    for key, value in over_dump.items():
        if key == "aux_channels":
            combined = dict(merged.get("aux_channels") or {})
            combined.update(value or {})
            merged["aux_channels"] = combined
        elif key == "filters":
            # exclude_none drops a None list; an explicit empty list would not
            # reach here, so a present value means "replace".
            if value:
                merged["filters"] = value
        else:
            merged[key] = value

    return AcquisitionSpec.model_validate(merged)


def merge_pet(
    base: PetAcquisitionSpec, over: Optional[PetAcquisitionSpec]
) -> PetAcquisitionSpec:
    """Return ``base`` with every set field of ``over`` layered on top.

    Same contract as :func:`merge_acquisition`: a non-None override wins. The
    three ``recon_method_parameter_*`` lists are replaced as a unit rather than
    merged element-wise, because their entries are positional (label N goes
    with value N and unit N) and interleaving two lists would silently
    mis-pair them.
    """
    if over is None:
        return base.model_copy(deep=True)

    merged = base.model_dump()
    for key, value in over.model_dump(exclude_none=True).items():
        if key.startswith("recon_method_parameter_"):
            if value:
                merged[key] = value
        else:
            merged[key] = value

    return PetAcquisitionSpec.model_validate(merged)


def resolve_pet(spec: RecordingMetaSpec, row_id: str) -> PetAcquisitionSpec:
    """The effective PET block for one recording."""
    return merge_pet(spec.pet_defaults, spec.pet_overrides.get(row_id))


def resolve_effective(
    spec: RecordingMetaSpec,
    row_id: str,
    task_label: Optional[str] = None,
    datatype: Optional[str] = None,
) -> EffectiveSpec:
    """Resolve the effective enrichment for the recording identified by ``row_id``.

    ``task_label`` selects the task protocol and event map; the event map falls
    back to the ``"*"`` global map when no task-specific map exists.

    ``datatype`` selects the per-modality block. EEG and MEG are different
    instruments and a study may run both, so what is stated for one must not
    leak into the other. The layers, weakest first: the shared defaults, then
    this datatype's own, then the per-recording override.
    """
    base = spec.defaults
    if datatype:
        per_modality = spec.modality_defaults.get(datatype)
        if per_modality is not None:
            base = merge_acquisition(base, per_modality)
    acquisition = merge_acquisition(base, spec.overrides.get(row_id))

    task_protocol = spec.task_protocols.get(task_label) if task_label else None

    event_map: EventMap = {}
    if task_label and task_label in spec.event_maps:
        event_map = dict(spec.event_maps[task_label])
    elif _GLOBAL_EVENT_KEY in spec.event_maps:
        event_map = dict(spec.event_maps[_GLOBAL_EVENT_KEY])

    return EffectiveSpec(
        acquisition=acquisition,
        task_protocol=task_protocol,
        event_map=event_map,
    )


__all__ = [
    "EffectiveSpec",
    "merge_acquisition",
    "merge_pet",
    "resolve_effective",
    "resolve_pet",
]

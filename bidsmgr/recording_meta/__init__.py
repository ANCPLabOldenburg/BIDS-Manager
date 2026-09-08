"""Recording-level enrichment metadata: pure-data models + resolution + I/O.

This subpackage holds the information a raw recording cannot carry on its own,
as I/O-free Pydantic models, plus the logic to resolve a dataset-default-plus-
per-row-override spec into a single effective view for one recording.

Two modality blocks live here, because two modalities have facts their files
cannot express. EEG/MEG/iEEG needs reference, ground, filters, device, cap,
event-code meaning and task protocol. PET needs the radiochemistry: how much
tracer, in what form, administered when. MRI needs neither, since dcm2niix
reads everything BIDS wants straight out of the DICOM header. Alongside both
sit the modality-agnostic parts (institution, events, phenotype, participants),
which is what lets one scaffold file serve a PET/MR study.

It imports only from :mod:`bidsmgr.schema` and standard scientific deps; it is
Qt-free and depends on no other ``bidsmgr`` subpackage.

The enrichment is *applied* to written BIDS files by
:mod:`bidsmgr.fixups.eeg_sidecar` and :mod:`bidsmgr.fixups.pet_sidecar`; this
subpackage only describes and resolves.
"""

from __future__ import annotations

from .chain import (
    LAYERS,
    describe_origin,
    resolve_attribute,
    ResolvedField,
    resolve_sidecar_fields,
    DATASET_DESCRIPTION_TO_BIDS,
    dataset_description_as_bids,
    dataset_description_from_bids,
    PET_LIST_TO_BIDS,
    PET_SCALAR_TO_BIDS,
)
from .templates import (
    parse_template_key,
    resolve_sequence_template,
    template_key,
    validate_sequence_templates,
)
from .models import (
    VARIES,
    is_varies,
    DatasetDescriptionSpec,
    COMMON_CAP_MANUFACTURERS,
    COMMON_MANUFACTURERS,
    CURATED_SUGGESTIONS,
    SCAN_SUGGESTION_COLUMNS,
    COMMON_RADIONUCLIDES,
    COMMON_TRACERS,
    MODES_OF_ADMINISTRATION,
    PET_ACQUISITION_MODES,
    RADIOACTIVITY_UNITS,
    MASS_UNITS,
    SPECIFIC_RADIOACTIVITY_UNITS,
    MOLAR_ACTIVITY_UNITS,
    PET_IMAGE_UNITS,

    AcceptableImpedance,
    AcquisitionSpec,
    AuxChannelSpec,
    EventMap,
    ExtrasSpec,
    FilterSpec,
    LightingConditions,
    PetAcquisitionSpec,
    RecordingMetaSpec,
    TaskProtocol,
)
from .resolve import (
    EffectiveSpec,
    merge_acquisition,
    merge_pet,
    resolve_effective,
    resolve_pet,
)
from .schema_types import bids_channel_types, mne_channel_types
from .serialize import (
    DEFAULT_POWER_LINE_FREQ,
    RECORDING_META_SIDECAR,
    default_spec,
    dump_spec,
    load_spec,
    scaffold_sidecar_path,
)

__all__ = [
    "LAYERS",
    "describe_origin",
    "resolve_attribute",
    "ResolvedField",
    "resolve_sidecar_fields",
    "DATASET_DESCRIPTION_TO_BIDS",
    "dataset_description_as_bids",
    "dataset_description_from_bids",
    "PET_LIST_TO_BIDS",
    "PET_SCALAR_TO_BIDS",
    "VARIES",
    "is_varies",
    "parse_template_key",
    "resolve_sequence_template",
    "template_key",
    "validate_sequence_templates",
    "DatasetDescriptionSpec",
    "COMMON_MANUFACTURERS",
    "CURATED_SUGGESTIONS",
    "SCAN_SUGGESTION_COLUMNS",
    "COMMON_CAP_MANUFACTURERS",
    "COMMON_RADIONUCLIDES",
    "COMMON_TRACERS",
    "MODES_OF_ADMINISTRATION",
    "PET_ACQUISITION_MODES",
    "RADIOACTIVITY_UNITS",
    "MASS_UNITS",
    "SPECIFIC_RADIOACTIVITY_UNITS",
    "MOLAR_ACTIVITY_UNITS",
    "PET_IMAGE_UNITS",
    "AcceptableImpedance",
    "AcquisitionSpec",
    "AuxChannelSpec",
    "EventMap",
    "ExtrasSpec",
    "FilterSpec",
    "LightingConditions",
    "PetAcquisitionSpec",
    "RecordingMetaSpec",
    "TaskProtocol",
    "EffectiveSpec",
    "merge_acquisition",
    "merge_pet",
    "resolve_effective",
    "resolve_pet",
    "bids_channel_types",
    "mne_channel_types",
    "DEFAULT_POWER_LINE_FREQ",
    "RECORDING_META_SIDECAR",
    "default_spec",
    "dump_spec",
    "load_spec",
    "scaffold_sidecar_path",
]

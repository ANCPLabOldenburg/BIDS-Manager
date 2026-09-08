"""One inheritance chain, resolved to BIDS field names.

The same fact about the same file could be stated in two places and two
vocabularies: as a spec attribute (``modality_defaults["eeg"].manufacturer``)
or as a BIDS field in a sequence template
(``sequence_templates["eeg/eeg"]["Manufacturer"]``). Two write paths applied
them independently, so which one won was decided by the order the fixups
happened to run in, and the per-row surfaces in the GUI knew about only one of
them: you could set a template, see nothing change in the table, and then find
values in the sidecars afterwards.

This module is the single answer. It walks every layer, weakest first, and
returns one value per BIDS field together with WHERE it came from, so a form can
say "from the EEG defaults" rather than a bare "inherited".

The layers, least specific first:

===============  ==========================================================
``dataset``      the agnostic block: the site, and anything shared by all
``modality``     that datatype's own block: the instrument
``template``     the per-sequence template for this datatype/suffix
``template@task`` the same, narrowed to one task
``row``          this recording's override, from the properties panel
``cell``         this recording's inventory cell, typed into the table
===============  ==========================================================

``row`` and ``cell`` are both "this one recording" and differ only in which
surface the user typed into. The table cell wins, preserving the behaviour the
enrichment always had, and because it is the value the user can see in front of
them while the panel override is a level down.

What the converter read out of the file is NOT a layer here: it is already in
the sidecar, and the writer's rule is that it stands unless a ``row`` override
contradicts it. A statement about a class of files should not overrule the
file's own header; correcting one specific file is exactly what the row
override is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .models import RecordingMetaSpec, is_varies
from .templates import template_key

# Weakest first. The order IS the precedence.
LAYERS: tuple[str, ...] = (
    "dataset", "modality", "template", "template@task", "row", "cell",
)


class _AttrView:
    """Read a plain mapping as if it were an acquisition block."""

    def __init__(self, values: dict) -> None:
        self._values = values

    def __getattr__(self, name: str) -> Any:
        return self._values.get(name)

    @property
    def filters(self):
        return self._values.get("filters") or []


@dataclass(frozen=True)
class ResolvedField:
    """One field's answer, and which layer supplied it."""

    name: str
    value: Any
    origin: str

    @property
    def is_row_override(self) -> bool:
        """True when this recording states it directly, which beats the file."""
        return self.origin in ("row", "cell")


# Spec attribute -> the BIDS field names it can land under, most preferred
# first. More than one candidate exists where datatypes disagree on the name:
# the same value is ``EEGReference`` on a scalp recording and
# ``iEEGReference`` on an intracranial one.
_ACQ_TO_BIDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("manufacturer", ("Manufacturer",)),
    ("amplifier_model", ("ManufacturersModelName",)),
    ("institution_name", ("InstitutionName",)),
    ("institution_dept", ("InstitutionalDepartmentName",)),
    ("eeg_reference", ("EEGReference", "iEEGReference")),
    ("eeg_ground", ("EEGGround", "iEEGGround")),
    ("cap_manufacturer", ("CapManufacturer",)),
    ("cap_model", ("CapManufacturersModelName",)),
    ("dewar_position", ("DewarPosition",)),
    ("associated_empty_room", ("AssociatedEmptyRoom",)),
    ("subject_artefact_description", ("SubjectArtefactDescription",)),
    ("power_line_freq", ("PowerLineFrequency",)),
    # Listed even though the loop's result is overwritten below, because this
    # table is also read as "which BIDS field does this attribute mean", and
    # leaving it out made that question unanswerable for this one attribute.
    ("software_versions", ("SoftwareVersions",)),
)


# The same translation for PET. It lived in the PET fixup, which meant the fixup
# could write a field the chain knew nothing about, so a dose imported from a
# lab spreadsheet reached the sidecar without ever appearing in the form that
# claims to show what this recording will say. One map, one home.
#
# Split scalars from lists because they are read back differently, not because
# they mean anything different.
PET_SCALAR_TO_BIDS: dict[str, str] = {
    "tracer_name": "TracerName",
    "tracer_radionuclide": "TracerRadionuclide",
    "tracer_molecular_weight": "TracerMolecularWeight",
    "tracer_molecular_weight_units": "TracerMolecularWeightUnits",
    "tracer_radlex": "TracerRadLex",
    "tracer_snomed": "TracerSNOMED",
    "injected_radioactivity": "InjectedRadioactivity",
    "injected_radioactivity_units": "InjectedRadioactivityUnits",
    "injected_mass": "InjectedMass",
    "injected_mass_units": "InjectedMassUnits",
    "specific_radioactivity": "SpecificRadioactivity",
    "specific_radioactivity_units": "SpecificRadioactivityUnits",
    "molar_activity": "MolarActivity",
    "molar_activity_units": "MolarActivityUnits",
    "injected_volume": "InjectedVolume",
    "purity": "Purity",
    "mode_of_administration": "ModeOfAdministration",
    "injection_start": "InjectionStart",
    "injection_end": "InjectionEnd",
    "infusion_radioactivity": "InfusionRadioactivity",
    "infusion_start": "InfusionStart",
    "infusion_speed": "InfusionSpeed",
    "infusion_speed_units": "InfusionSpeedUnits",
    "time_zero": "TimeZero",
    "scan_start": "ScanStart",
    "acquisition_mode": "AcquisitionMode",
    "image_decay_corrected": "ImageDecayCorrected",
    "image_decay_correction_time": "ImageDecayCorrectionTime",
    "attenuation_correction": "AttenuationCorrection",
    "units": "Units",
    "body_part": "BodyPart",
    "recon_method_name": "ReconMethodName",
    "recon_filter_type": "ReconFilterType",
    "recon_filter_size": "ReconFilterSize",
    "manufacturer": "Manufacturer",
    "manufacturers_model_name": "ManufacturersModelName",
}

PET_LIST_TO_BIDS: dict[str, str] = {
    "recon_method_parameter_labels": "ReconMethodParameterLabels",
    "recon_method_parameter_units": "ReconMethodParameterUnits",
    "recon_method_parameter_values": "ReconMethodParameterValues",
}


def _coerce(name: str, value: Any, datatype: str, suffix: str) -> Any:
    """Put a value into the shape the standard declares, if it is not already.

    Everything arrives as text somewhere: the inventory is a TSV, and a combo
    box hands back whatever was typed. BIDS declares PowerLineFrequency a
    number, and the string "60" fails validation for a field answered right.
    """
    from .. import schema as schema_mod

    try:
        return schema_mod.coerce(name, value, datatype, suffix)
    except Exception:  # noqa: BLE001 - a shape we cannot check is left alone
        return value


def _acquisition_as_bids(acq, datatype: str, field_applies) -> dict[str, Any]:
    """Translate one acquisition block into the BIDS names this datatype uses.

    Accepts an :class:`AcquisitionSpec` or a plain mapping of the same
    attribute names, which is how the inventory's own cells join the chain.
    """
    if acq is None:
        return {}
    if isinstance(acq, dict):
        acq = _AttrView(acq)

    out: dict[str, Any] = {}

    def put(candidates: tuple[str, ...], value: Any) -> None:
        if value in (None, "", [], {}) or is_varies(value):
            # VARIES points at where the answer lives; it is never the answer.
            return
        for name in candidates:
            if field_applies(name, datatype, datatype):
                out[name] = _coerce(name, value, datatype, datatype)
                return

    for attr, candidates in _ACQ_TO_BIDS:
        put(candidates, getattr(acq, attr, None))

    # The spec has two names for one BIDS field, so the more specific one wins
    # and the older one is the fallback. This runs after the loop above, which
    # already put software_versions, and says the same thing when it is set.
    put(("SoftwareVersions",), acq.software_versions or acq.software)
    put(("HardwareFilters",), {f.name: f.info for f in acq.filters if f.kind == "Hardware"})
    put(("SoftwareFilters",), {f.name: f.info for f in acq.filters if f.kind == "Software"})
    return out


def _pet_as_bids(pet, datatype: str, suffix: str, field_applies) -> dict[str, Any]:
    """One PET block as BIDS fields, minus anything this file does not take."""
    if pet is None or datatype != "pet":
        return {}
    out: dict[str, Any] = {}
    for attr, name in {**PET_SCALAR_TO_BIDS, **PET_LIST_TO_BIDS}.items():
        value = getattr(pet, attr, None)
        if value in (None, "", [], {}) or is_varies(value):
            continue
        if field_applies(name, datatype, suffix):
            out[name] = _coerce(name, value, datatype, suffix)
    return out


def _template_as_bids(values: Any, datatype: str, suffix: str, field_applies) -> dict[str, Any]:
    """A template's fields, minus anything this datatype does not accept."""
    if not isinstance(values, dict):
        return {}
    return {
        name: _coerce(name, value, datatype, suffix)
        for name, value in values.items()
        if value not in (None, "", [], {})
        and not is_varies(value)
        and field_applies(name, datatype, suffix)
    }


def resolve_sidecar_fields(
    spec: Optional[RecordingMetaSpec],
    datatype: str,
    suffix: str,
    row_id: str = "",
    task: Optional[str] = None,
    row_values: Optional[dict] = None,
    field_applies=None,
) -> dict[str, ResolvedField]:
    """Every field the user has stated for this file, and which layer said it.

    ``field_applies`` is injected so this stays a pure-data module; the default
    resolves it lazily from the schema layer.
    """
    # An empty spec is not the same as nothing to resolve: the row's own cells
    # are a layer too, and a user who typed a line frequency into the table
    # without ever opening the dataset dialog got nothing at all.
    spec = spec if spec is not None else RecordingMetaSpec()
    if field_applies is None:
        from .. import schema as schema_mod

        field_applies = schema_mod.field_applies

    contributions: list[tuple[str, dict[str, Any]]] = [
        ("dataset", _acquisition_as_bids(spec.defaults, datatype, field_applies)),
        (
            "modality",
            {
                **_acquisition_as_bids(
                    spec.modality_defaults.get(datatype), datatype, field_applies,
                ),
                # PET keeps its own block because an injected dose and an EEG cap
                # have nothing to say to each other. It is still this datatype's
                # instrument block, so it is the same layer.
                **_pet_as_bids(spec.pet_defaults, datatype, suffix, field_applies),
            },
        ),
        (
            "template",
            _template_as_bids(
                spec.sequence_templates.get(template_key(datatype, suffix)),
                datatype, suffix, field_applies,
            ),
        ),
    ]
    if task:
        contributions.append((
            "template@task",
            _template_as_bids(
                spec.sequence_templates.get(template_key(datatype, suffix, task)),
                datatype, suffix, field_applies,
            ),
        ))
    if row_id:
        # One recording states things two ways: through the acquisition block
        # the older surfaces write, and through the BIDS-named row template the
        # properties panel writes. Both are this recording speaking, so they are
        # one layer; the template goes second because it is the surface a user
        # types into now, and because it can say things the block has no
        # attribute for.
        contributions.append((
            "row",
            {
                **_acquisition_as_bids(
                    spec.overrides.get(row_id), datatype, field_applies,
                ),
                **_pet_as_bids(
                    spec.pet_overrides.get(row_id), datatype, suffix, field_applies,
                ),
                **_template_as_bids(
                    spec.row_templates.get(row_id), datatype, suffix, field_applies,
                ),
            },
        ))
    if row_values:
        contributions.append((
            "cell",
            _acquisition_as_bids(
                {k: v for k, v in row_values.items() if v not in (None, "")},
                datatype, field_applies,
            ),
        ))

    resolved: dict[str, ResolvedField] = {}
    for origin, values in contributions:
        for name, value in values.items():
            resolved[name] = ResolvedField(name=name, value=value, origin=origin)
    return resolved


# How each layer reads in a form or a tooltip.
LAYER_LABELS: dict[str, str] = {
    "dataset": "the dataset defaults",
    "modality": "the {datatype} defaults",
    "template": "the {datatype}/{suffix} template",
    "template@task": "the {datatype}/{suffix} template for task {task}",
    "row": "this recording",
    "cell": "this row",
}


def describe_origin(
    origin: str, datatype: str = "", suffix: str = "", task: str = "",
) -> str:
    """A phrase a user can read, e.g. "the eeg defaults"."""
    template = LAYER_LABELS.get(origin, origin)
    return template.format(datatype=datatype or "?", suffix=suffix or "?", task=task or "?")


def resolve_attribute(
    spec: Optional[RecordingMetaSpec],
    attr: str,
    datatype: str,
    suffix: str = "",
    row_id: str = "",
    task: Optional[str] = None,
    row_values: Optional[dict] = None,
    field_applies=None,
) -> Optional[ResolvedField]:
    """The same chain, asked about one of OUR attribute names.

    The GUI works in the spec's own vocabulary, and one of its fields has no
    BIDS name at all: a montage is which electrode layout to apply on
    conversion, which the standard has no opinion about. So this walks the same
    layers reading the attribute directly, and consults a template only where
    the attribute does have a BIDS name to look up.
    """
    if spec is None:
        return None
    if field_applies is None:
        from .. import schema as schema_mod

        field_applies = schema_mod.field_applies

    bids_names = dict(_ACQ_TO_BIDS).get(attr, ())
    suffix = suffix or datatype

    def from_block(block) -> Any:
        return getattr(block, attr, None) if block is not None else None

    def from_values(values: Any) -> Any:
        """This attribute's answer inside a BIDS-named mapping, if it has one."""
        if not isinstance(values, dict):
            return None
        for name in bids_names:
            if name in values and field_applies(name, datatype, suffix):
                return values[name]
        return None

    def from_template(key: str) -> Any:
        return from_values(spec.sequence_templates.get(key))

    candidates: list[tuple[str, Any]] = [
        ("dataset", from_block(spec.defaults)),
        ("modality", from_block(spec.modality_defaults.get(datatype))),
        ("template", from_template(template_key(datatype, suffix))),
    ]
    if task:
        candidates.append(("template@task", from_template(template_key(datatype, suffix, task))))
    if row_id:
        # The row template is the surface a user types into now, so it answers
        # first; the acquisition block still answers for anything it has no
        # BIDS-named home for.
        from_row = from_values(spec.row_templates.get(row_id))
        candidates.append((
            "row",
            from_row if from_row is not None else from_block(spec.overrides.get(row_id)),
        ))
    if row_values:
        candidates.append(("cell", row_values.get(attr)))

    winner: Optional[ResolvedField] = None
    for origin, value in candidates:
        if value in (None, "", [], {}):
            continue
        winner = ResolvedField(name=attr, value=value, origin=origin)
    return winner


__all__ = [
    "DATASET_DESCRIPTION_TO_BIDS",
    "LAYERS",
    "LAYER_LABELS",
    "ResolvedField",
    "dataset_description_as_bids",
    "dataset_description_from_bids",
    "describe_origin",
    "resolve_attribute",
    "resolve_sidecar_fields",
]


# ``DatasetDescriptionSpec`` attribute -> its BIDS field. The third of these
# tables, and the last: the dialog kept its own copy inline, spelled out twice
# (once to show the stored answers, once to put them back), so adding a field
# meant editing two lists in a GUI module and hoping they agreed.
DATASET_DESCRIPTION_TO_BIDS: dict[str, str] = {
    "name": "Name",
    "authors": "Authors",
    "license": "License",
    "acknowledgements": "Acknowledgements",
    "how_to_acknowledge": "HowToAcknowledge",
    "funding": "Funding",
    "ethics_approvals": "EthicsApprovals",
    "references_and_links": "ReferencesAndLinks",
    "dataset_doi": "DatasetDOI",
}

# Which of those hold a list, so reading one back knows what shape to keep.
_DATASET_DESCRIPTION_LISTS = frozenset(
    {"authors", "funding", "ethics_approvals", "references_and_links"}
)


def dataset_description_as_bids(dd) -> dict[str, Any]:
    """One ``DatasetDescriptionSpec`` as the fields a form asks for."""
    out: dict[str, Any] = {}
    for attr, name in DATASET_DESCRIPTION_TO_BIDS.items():
        value = getattr(dd, attr, None)
        if value not in (None, "", [], {}):
            out[name] = value
    for name, value in (getattr(dd, "extra", None) or {}).items():
        if value not in (None, "", [], {}):
            out.setdefault(name, value)
    return out


def dataset_description_from_bids(dd, answers: dict) -> None:
    """Write a form's answers back onto a ``DatasetDescriptionSpec``, in place.

    An answer to a field the model does not name is kept rather than dropped:
    the schema declares more fields than the CLI has flags for, and a form built
    from the schema offers all of them.
    """
    named = set(DATASET_DESCRIPTION_TO_BIDS.values())
    for attr, name in DATASET_DESCRIPTION_TO_BIDS.items():
        value = answers.get(name)
        if attr in _DATASET_DESCRIPTION_LISTS:
            setattr(dd, attr, list(value or []))
        else:
            setattr(dd, attr, value or None)
    dd.extra = {
        name: value
        for name, value in answers.items()
        if name not in named and value not in (None, "", [], {})
    }

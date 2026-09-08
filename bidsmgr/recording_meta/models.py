"""Pure-data models for recording-level enrichment metadata.

These describe the information a recording file cannot carry on its own:
the reference and ground electrodes, hardware/software filters, amplifier and
cap details, institution, the meaning of trigger codes, and per-task protocol
notes. A scan seeds what it can detect; the user fills the rest. The models are
deliberately I/O-free (Pydantic v2 only): reading, writing, resolving, and
applying live in sibling modules.

Every leaf field is optional. Enrichment is *additive*: a field left unset means
"leave whatever the converter already wrote". The structure separates a single
dataset-wide ``defaults`` block from sparse per-recording ``overrides`` keyed by
the inventory row id, so a heterogeneous cohort (mixed caps, amplifiers, or
references) is expressible without repeating shared values.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator

from .schema_types import bids_channel_types, mne_channel_types

# A trigger-code -> human-label mapping (e.g. {"S 20": "eyes_open"}).
EventMap = dict[str, str]

# Electrophysiology amplifier / system manufacturers, offered as an editable
# dropdown in the dataset dialog and the per-row recording section (the user can
# always type any other value). Grouped by modality but presented as one list.
# Pure reference data - no Qt - so both GUI surfaces import the one list instead
# of duplicating it.
COMMON_MANUFACTURERS: tuple[str, ...] = (
    # EEG amplifiers / systems
    "Brain Products", "BioSemi", "EGI / Philips Neuro", "ANT Neuro",
    "Compumedics Neuroscan", "g.tec", "Cognionics / CGX", "mBrainTrain",
    "OpenBCI", "Wearable Sensing", "Neuroelectrics", "Bittium", "Nihon Kohden",
    "Natus", "Nicolet", "Grass", "Cadwell", "Deymed", "Emotiv", "InteraXon",
    "Advanced Brain Monitoring",
    # MEG systems (BIDS Manufacturer convention)
    "MEGIN / Elekta / Neuromag", "CTF", "4D Neuroimaging / BTi",
    "KIT / Yokogawa", "ITAB", "KRISS",
    # OPM-MEG
    "FieldLine", "QuSpin", "Cerca Magnetics",
    # iEEG / ECoG / sEEG acquisition + electrodes
    "Blackrock Neurotech", "Ripple Neuro", "Tucker-Davis Technologies",
    "Plexon", "Medtronic", "Ad-Tech", "PMT Corporation", "DIXI Medical",
    # fNIRS
    "NIRx", "Artinis", "Hitachi", "Shimadzu", "Gowerlabs", "Kernel",
    "Cortivision",
)

# Common EEG/iEEG cap (electrode-holder) manufacturers. Editable dropdown too.
COMMON_CAP_MANUFACTURERS: tuple[str, ...] = (
    "EasyCap", "Brain Products", "BioSemi", "EGI", "ANT Neuro",
    "Compumedics Neuroscan", "Electro-Cap International", "g.tec",
    "Cognionics", "Greentek",
)


# PET controlled vocabularies, offered as dropdowns in the dataset dialog. The
# user can always type another value: these are the common cases, not a closed
# set. Positron-emitting radionuclides in routine clinical and research use.
COMMON_RADIONUCLIDES: tuple[str, ...] = (
    "F18", "C11", "O15", "N13", "Ga68", "Cu64", "Zr89", "Rb82", "I124",
    "Br76", "Sc44", "Y86", "Ge68",
)

# Tracer short labels as they conventionally appear in BIDS PET datasets.
COMMON_TRACERS: tuple[str, ...] = (
    "FDG", "PIB", "AV45", "FBB", "FMM", "FTP", "MK6240", "RAC", "CFN",
    "FDOPA", "FET", "FLT", "PSMA", "DOTATATE", "UCBJ", "SV2A", "FEOBV",
)

# BIDS ModeOfAdministration is a closed set in practice.
MODES_OF_ADMINISTRATION: tuple[str, ...] = ("bolus", "infusion", "bolus-infusion")

# BIDS AcquisitionMode.
PET_ACQUISITION_MODES: tuple[str, ...] = ("list mode", "sinogram")

# Unit vocabularies, one per quantity, so the dialog can offer the right list
# beside each field instead of one undifferentiated pile.
RADIOACTIVITY_UNITS: tuple[str, ...] = ("MBq", "kBq", "Bq", "mCi", "uCi", "nCi")
MASS_UNITS: tuple[str, ...] = ("ug", "mg", "g", "umol", "nmol", "mol")
SPECIFIC_RADIOACTIVITY_UNITS: tuple[str, ...] = (
    "Bq/g", "MBq/ug", "GBq/umol", "MBq/nmol", "Bq/umol",
)
MOLAR_ACTIVITY_UNITS: tuple[str, ...] = ("GBq/umol", "MBq/nmol", "Bq/mol")
PET_IMAGE_UNITS: tuple[str, ...] = ("Bq/mL", "kBq/mL", "MBq/mL", "SUV", "counts")


class _Model(BaseModel):
    """Shared config: forbid unknown keys so typos surface at load time."""

    model_config = ConfigDict(extra="forbid")


class AcceptableImpedance(_Model):
    """Impedance acceptance threshold recorded during acquisition."""

    value: Optional[float] = None
    units: Optional[str] = None


class LightingConditions(_Model):
    """Ambient lighting of the recording environment."""

    description: Optional[str] = None
    measurement: Optional[str] = None


class ExtrasSpec(_Model):
    """Supplemental, non-required acquisition conditions.

    These are not BIDS-required sidecar keys; they are written as additional
    fields (BIDS permits extra keys) when present.
    """

    acceptable_impedance: Optional[AcceptableImpedance] = None
    electrode_type: Optional[str] = None
    conductive_medium: Optional[str] = None
    faraday_cage: Optional[bool] = None
    sound_proof: Optional[bool] = None
    lighting_conditions: Optional[LightingConditions] = None


class FilterSpec(_Model):
    """One hardware or software filter applied during acquisition.

    ``info`` is copied verbatim into the sidecar under the filter's ``name``,
    grouped by ``kind`` into ``HardwareFilters`` / ``SoftwareFilters``.
    """

    name: str
    kind: str  # "Hardware" | "Software"
    info: dict[str, Any] = {}

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, v: str) -> str:
        if v not in ("Hardware", "Software"):
            raise ValueError("filter kind must be 'Hardware' or 'Software'")
        return v


class AuxChannelSpec(_Model):
    """How one auxiliary (non-data) channel should be typed and described.

    Used to upgrade a channel the reader saw as generic ``misc`` to its real
    BIDS type and to fill its units / description in ``channels.tsv``.
    """

    mne_type: Optional[str] = None
    bids_type: Optional[str] = None
    description: Optional[str] = None
    units: Optional[str] = None
    location: Optional[Union[str, dict[str, str]]] = None

    @field_validator("bids_type")
    @classmethod
    def _check_bids_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in bids_channel_types():
            raise ValueError(
                f"{v!r} is not a BIDS channel type "
                f"(allowed: {', '.join(sorted(bids_channel_types()))})"
            )
        return v

    @field_validator("mne_type")
    @classmethod
    def _check_mne_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in mne_channel_types():
            raise ValueError(
                f"{v!r} is not an MNE channel type "
                f"(allowed: {', '.join(sorted(mne_channel_types()))})"
            )
        return v


class TaskProtocol(_Model):
    """Free-form per-task protocol notes that map into the task sidecar."""

    task_description: Optional[str] = None
    instructions: Optional[str] = None


class AcquisitionSpec(_Model):
    """The recording-level technical block (defaults and per-row overrides).

    Carries both the values applied during the write (``power_line_freq``,
    ``montage``) and the values folded into the sidecar afterwards (reference,
    ground, device, institution, filters, extras, aux channels).
    """

    power_line_freq: Optional[float] = None
    montage: Optional[str] = None
    eeg_reference: Optional[str] = None
    eeg_ground: Optional[str] = None
    manufacturer: Optional[str] = None
    amplifier_model: Optional[str] = None  # -> ManufacturersModelName
    software: Optional[str] = None
    software_versions: Optional[str] = None
    cap_manufacturer: Optional[str] = None
    cap_model: Optional[str] = None
    institution_name: Optional[str] = None
    institution_dept: Optional[str] = None
    # MEG-specific sidecar fields (written only into the meg sidecar). Only the
    # ones mne-bids CANNOT derive from the recording are exposed: dewar position
    # (a physical setup fact), the associated empty-room link (a curation
    # decision), and the subject-artefact note (free text). Channel-derived
    # facts (ContinuousHeadLocalization, DigitizedLandmarks, DigitizedHeadPoints,
    # HeadCoilFrequency, ...) are left to mne-bids - never duplicated here.
    dewar_position: Optional[str] = None            # -> DewarPosition
    associated_empty_room: Optional[str] = None     # -> AssociatedEmptyRoom
    subject_artefact_description: Optional[str] = None  # -> SubjectArtefactDescription
    aux_channels: dict[str, AuxChannelSpec] = {}
    filters: list[FilterSpec] = []
    extras: Optional[ExtrasSpec] = None


class PetAcquisitionSpec(_Model):
    """The PET block: what a PET scan cannot record about itself.

    PET is the modality where this subsystem earns its keep. The BIDS PET
    sidecar requires around forty fields and the DICOM header carries only
    about half of them: a scanner records how it reconstructed an image, but
    not how much tracer went into the person, in what form, or when relative to
    the scan. Those are facts from the radiochemistry lab and the injection
    record, so they can only ever come from the user.

    Grouped the way the acquisition itself divides, which is also how the
    dataset dialog lays the fields out. Every field is optional and additive:
    unset means "leave whatever the converter wrote".
    """

    # --- tracer -------------------------------------------------------
    tracer_name: Optional[str] = None                  # -> TracerName
    tracer_radionuclide: Optional[str] = None          # -> TracerRadionuclide
    tracer_molecular_weight: Optional[float] = None    # -> TracerMolecularWeight
    tracer_molecular_weight_units: Optional[str] = None
    tracer_radlex: Optional[str] = None                # -> TracerRadLex
    tracer_snomed: Optional[str] = None                # -> TracerSNOMED

    # --- radiochemistry and dose --------------------------------------
    injected_radioactivity: Optional[float] = None
    injected_radioactivity_units: Optional[str] = None
    injected_mass: Optional[float] = None
    injected_mass_units: Optional[str] = None
    specific_radioactivity: Optional[float] = None
    specific_radioactivity_units: Optional[str] = None
    molar_activity: Optional[float] = None
    molar_activity_units: Optional[str] = None
    injected_volume: Optional[float] = None            # -> InjectedVolume (mL)
    purity: Optional[float] = None                     # -> Purity (percent)

    # --- administration -----------------------------------------------
    mode_of_administration: Optional[str] = None       # -> ModeOfAdministration
    injection_start: Optional[float] = None            # -> InjectionStart (s)
    injection_end: Optional[float] = None              # -> InjectionEnd (s)
    infusion_radioactivity: Optional[float] = None
    infusion_start: Optional[float] = None
    infusion_speed: Optional[float] = None
    infusion_speed_units: Optional[str] = None

    # --- timing -------------------------------------------------------
    time_zero: Optional[str] = None                    # -> TimeZero (hh:mm:ss)
    scan_start: Optional[float] = None                 # -> ScanStart (s)

    # --- acquisition --------------------------------------------------
    acquisition_mode: Optional[str] = None             # -> AcquisitionMode
    image_decay_corrected: Optional[bool] = None       # -> ImageDecayCorrected
    image_decay_correction_time: Optional[float] = None
    attenuation_correction: Optional[str] = None       # -> AttenuationCorrection
    units: Optional[str] = None                        # -> Units
    body_part: Optional[str] = None                    # -> BodyPart

    # --- reconstruction -----------------------------------------------
    recon_method_name: Optional[str] = None            # -> ReconMethodName
    recon_method_parameter_labels: list[str] = []
    recon_method_parameter_units: list[str] = []
    recon_method_parameter_values: list[float] = []
    recon_filter_type: Optional[str] = None            # -> ReconFilterType
    recon_filter_size: Optional[float] = None          # -> ReconFilterSize

    # --- device / site (PET-side mirrors of the agnostic block) -------
    manufacturer: Optional[str] = None                 # -> Manufacturer
    manufacturers_model_name: Optional[str] = None     # -> ManufacturersModelName


# The third state a metadata field can be in.
#
# A field used to be either answered or blank, which forces a bad choice when
# the answer differs between recordings. Put one value in the dataset default
# and every recording claims it, including the ones it is wrong for, and a
# wrong value is indistinguishable from a right one once written. Leave it
# blank and the answer is lost for the recordings you DID know.
#
# ``VARIES`` says "this differs per recording; the answer lives further down".
# It is never itself written to a sidecar: it is a statement about where to
# look, not a value. With the applicability check from the schema layer, a
# field then has four honest states: a value, VARIES, not stated, and not
# applicable to this datatype.
VARIES = "VARIES"


def is_varies(value: Any) -> bool:
    """True when ``value`` is the VARIES sentinel rather than an answer."""
    return isinstance(value, str) and value.strip().upper() == VARIES


class DatasetDescriptionSpec(_Model):
    """What goes in ``dataset_description.json``, held where it can persist.

    Every field here was already writable through ``bidsmgr-metadata``
    (``--author``, ``--license``, ``--funding`` and friends) and already
    modelled by ``metadata.DatasetMetadata``. What did not exist was anywhere
    to KEEP the answers: they lived only in the flags of one command. So the
    automatic metadata run that follows a conversion never had them, the GUI
    had nothing to offer, and re-running metadata lost whatever the last run
    was told. That is why every dataset we produce reports NO_AUTHORS.

    Holding them in the scaffold, beside the modality blocks, means they are
    stated once for the dataset and survive every later step.

    Blank fields are omitted from the written JSON rather than emitted empty,
    so an unanswered field stays absent instead of becoming an empty string
    that looks answered.
    """

    name: Optional[str] = None
    authors: list[str] = []
    license: Optional[str] = None
    acknowledgements: Optional[str] = None
    how_to_acknowledge: Optional[str] = None
    funding: list[str] = []
    ethics_approvals: list[str] = []
    references_and_links: list[str] = []
    dataset_doi: Optional[str] = None
    # Everything else the standard declares for this file.
    #
    # The nine above are named because the CLI has a flag for each. The schema
    # declares more than nine and will declare more again, and a form built from
    # the schema offers all of them, so a model that could hold only its named
    # nine ACCEPTED three answers and threw them away on save: HEDVersion,
    # DatasetLinks and Keywords went in and never came out. Anything the form
    # asks for that has no attribute here lands in this dict instead of on the
    # floor.
    extra: dict[str, Any] = {}


class RecordingMetaSpec(_Model):
    """Root enrichment object for one dataset.

    ``defaults`` applies to every recording; ``overrides`` carries the sparse
    per-recording deltas keyed by the inventory ``row_id`` (the recording's
    source path for EEG/MEG). ``event_maps`` and ``task_protocols`` are keyed by
    BIDS task label, with ``"*"`` as a fallback event map for all tasks.
    """

    schema_version: int = 1
    # The electrophysiology defaults shared by every such datatype. Kept
    # because every scaffold ever written has one, and because the genuinely
    # shared facts (the site, the mains frequency of the building) do belong
    # to all of them.
    defaults: AcquisitionSpec = AcquisitionSpec()
    # Per-datatype electrophysiology defaults, keyed "eeg" / "meg" / "ieeg" /
    # "nirs", layered ON TOP of ``defaults``.
    #
    # EEG and MEG are different modalities recorded on different instruments.
    # A study can run a Brain Products amplifier and an Elekta dewar, and one
    # shared block forced a single answer for both: state the amplifier and the
    # MEG sidecars claim it too. Anything stated here belongs to that datatype
    # alone; anything left blank falls back to ``defaults``, so a study with
    # one modality never has to say where it goes.
    modality_defaults: dict[str, AcquisitionSpec] = {}
    task_protocols: dict[str, TaskProtocol] = {}
    event_maps: dict[str, EventMap] = {}
    overrides: dict[str, AcquisitionSpec] = {}
    # PET's equivalent pair. Kept as separate fields rather than folded into
    # ``defaults`` because the two blocks share almost no fields: an EEG cap
    # manufacturer and an injected dose have nothing to say to each other. One
    # scaffold file still holds both, so a PET/MR study needs only one place
    # for the site and event information they DO share.
    # Older scaffolds have no PET section at all; the defaults here make those
    # load unchanged.
    pet_defaults: PetAcquisitionSpec = PetAcquisitionSpec()
    pet_overrides: dict[str, PetAcquisitionSpec] = {}
    # The dataset's own description. Agnostic: it says who made the dataset and
    # under what terms, which has nothing to do with what recorded it. Older
    # scaffolds have no such block and load unchanged.
    dataset_description: DatasetDescriptionSpec = DatasetDescriptionSpec()
    # Per-sequence metadata: the scope between "the whole dataset" and "this one
    # recording", which is where most of what a study knows actually lives. All
    # of a study's bold runs share a TaskDescription and a set of Instructions;
    # its T1w runs share none of that with them.
    #
    # Keyed ``"<datatype>/<suffix>"``, optionally narrowed to one task with
    # ``"<datatype>/<suffix>@<task>"``. The general key applies first and the
    # task-specific one refines it, so a study can state what every bold run
    # shares once and then say what is different about the localiser task.
    #
    # Values are plain ``{BIDS field: value}`` dicts. They are checked against
    # the schema before anything is written, so a template cannot smuggle in a
    # field its datatype does not accept: that check is what stops this from
    # becoming a second place to make the mistake the table used to make.
    sequence_templates: dict[str, dict[str, Any]] = {}
    # The same thing one scope further down: what is true of ONE recording,
    # keyed by its inventory ``row_id``.
    #
    # ``overrides`` above already holds per-recording facts, but only the twelve
    # or so an ``AcquisitionSpec`` models, so the properties panel could offer
    # only those twelve while the dataset dialog offered everything the schema
    # declares. A user who found the right field in one surface could not state
    # it for one recording in the other. This is BIDS-named like
    # ``sequence_templates`` precisely so both surfaces speak one vocabulary and
    # the same field means the same thing at every scope.
    #
    # ``overrides`` stays: every scaffold ever written has one, and the montage
    # it carries has no BIDS name to move to.
    row_templates: dict[str, dict[str, Any]] = {}
    # What the conversion will answer by itself, keyed like the templates.
    #
    # NOT user input and never written to a sidecar: the converter writes these
    # anyway. It is here so a form can show "EchoTime 0.03, read from the DICOM"
    # where it would otherwise show an empty box for a required field and look
    # like a dataset missing its metadata. Filled by the scan; VARIES where the
    # probed files of one kind disagreed.
    converter_preview: dict[str, dict[str, Any]] = {}
    # The same thing per recording, keyed by inventory row id. The inventory
    # drops it: it is derived, not curated, and it would add a column of JSON to
    # a table people read. It belongs here, where the per-file form can find it
    # after the table has been written and read back.
    row_preview: dict[str, dict[str, Any]] = {}
    # Dataset-level phenotype measure tables (TSV/CSV/XLSX/ODS paths keyed by
    # participant_id). Written to ``phenotype/<measure>.tsv`` + ``.json`` by the
    # metadata engine. Agnostic: applies to any modality.
    phenotype_files: list[str] = []
    # Optional participants spreadsheet (TSV/CSV/XLSX/ODS keyed by
    # participant_id). Its demographic columns override the inventory-derived
    # values and any extra columns are carried into participants.tsv (described
    # in participants.json, optionally via a sibling ``.json`` codebook).
    # Agnostic: applies to any modality.
    participants_file: str = ""


__all__ = [
    "CURATED_SUGGESTIONS",
    "SCAN_SUGGESTION_COLUMNS",
    "EventMap",
    "COMMON_MANUFACTURERS",
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
    "LightingConditions",
    "ExtrasSpec",
    "FilterSpec",
    "AuxChannelSpec",
    "TaskProtocol",
    "AcquisitionSpec",
    "PetAcquisitionSpec",
    "RecordingMetaSpec",
]


# What BIDS Manager offers where BIDS leaves a field free text.
#
# Product opinion, not schema: the standard does not list these, and none of
# them restricts what a user may type. They live beside the vocabularies they
# are made of, and both metadata surfaces read this one table, because when the
# dialog and the properties panel each kept their own the two offered different
# answers for the same field.
#
# Units and modes are here for a concrete reason: BIDS accepts free text, so a
# lab typing "mbq" instead of "MBq" fails validation for nothing.
CURATED_SUGGESTIONS: dict[str, tuple[str, ...]] = {
    # --- who and what recorded it ------------------------------------
    "Manufacturer": tuple(COMMON_MANUFACTURERS),
    "ManufacturersModelName": (
        # EEG amplifiers
        "BrainAmp", "BrainAmp DC", "BrainAmp MR plus", "actiCHamp",
        "actiCHamp Plus", "LiveAmp", "ActiveTwo", "NetAmps 300", "NetAmps 400",
        "eego mylab", "eego sports", "SynAmps RT", "Grael", "Quick-20",
        "Cyton", "SAGA",
        # MEG systems
        "Neuromag-122", "Vectorview", "TRIUX", "CTF-275", "Magnes 3600 WH",
        "PQ1160R-N2", "OPM",
        # MR scanners
        "Prisma", "Prisma_fit", "Skyra", "Trio", "TrioTim", "Vida", "Cima.X",
        "Achieva", "Ingenia", "Ingenia Elition", "Elition X",
        "DISCOVERY MR750", "SIGNA Premier", "SIGNA Architect",
    ),
    "CapManufacturer": tuple(COMMON_CAP_MANUFACTURERS),
    "CapManufacturersModelName": (
        "actiCAP 64Ch", "actiCAP snap", "EasyCap M1", "EasyCap M10",
        "EasyCap M34", "BioSemi 64", "BioSemi 128", "HydroCel GSN 128",
        "HydroCel GSN 256", "waveguard original", "waveguard connect",
    ),
    "DewarPosition": ("upright", "supine"),

    # --- how the electrodes sat --------------------------------------
    #
    # BIDS leaves these free text and they are the fields nobody can read from
    # the data, so an editable list of what people actually write is the
    # difference between an answer and a blank.
    "EEGReference": (
        "average", "Cz", "FCz", "CPz", "Pz", "nose", "left mastoid",
        "right mastoid", "linked mastoids", "linked earlobes", "REST", "n/a",
    ),
    "iEEGReference": ("average", "bipolar", "a white matter electrode", "n/a"),
    "EEGGround": (
        "AFz", "Fpz", "FCz", "Fz", "left mastoid", "right mastoid", "n/a",
    ),
    "iEEGGround": ("a scalp electrode", "n/a"),
    "EEGPlacementScheme": (
        "10-20", "10-10", "10-5",
        "based on the digitized electrode positions", "custom",
    ),
    "ElectrodeMaterial": ("Ag/AgCl", "silver", "gold", "tin", "platinum", "carbon"),
    "ElectrodeManufacturer": tuple(COMMON_CAP_MANUFACTURERS),

    # --- the recording environment -----------------------------------
    "PowerLineFrequency": ("50", "60"),
    "HardwareFilters": ("n/a",),
    "SoftwareFilters": ("n/a",),
    "RecordingType": ("continuous", "epoched", "discontinuous"),

    # --- MRI acquisition, where the vendors share a vocabulary -------
    "PulseSequenceType": (
        "MPRAGE", "MP2RAGE", "FLASH", "GRE", "SPGR", "FSPGR", "TSE", "FSE",
        "SPACE", "CUBE", "VISTA", "bSSFP", "TrueFISP", "FIESTA", "EPI",
        "SE-EPI", "GE-EPI", "TFL", "IR-TSE", "FLAIR",
    ),
    "ParallelAcquisitionTechnique": (
        "GRAPPA", "SENSE", "mSENSE", "ASSET", "ARC", "CAIPIRINHA", "SMS",
        "HyperSense", "Compressed SENSE",
    ),
    "PartialFourierDirection": ("phase", "frequency", "slice", "combination"),
    "WaterSuppressionTechnique": ("CHESS", "VAPOR", "MEGA", "WET", "none"),

    # --- PET ---------------------------------------------------------
    "TracerName": tuple(COMMON_TRACERS),
    "TracerRadionuclide": tuple(COMMON_RADIONUCLIDES),
    "TracerMolecularWeightUnits": ("g/mol",),
    "InjectedRadioactivityUnits": tuple(RADIOACTIVITY_UNITS),
    "InjectedMassUnits": tuple(MASS_UNITS),
    "InjectedMassPerWeightUnits": ("ug/kg", "mg/kg", "nmol/kg", "umol/kg"),
    "SpecificRadioactivityUnits": tuple(SPECIFIC_RADIOACTIVITY_UNITS),
    "MolarActivityUnits": tuple(MOLAR_ACTIVITY_UNITS),
    "InfusionSpeedUnits": ("mL/min", "mL/s", "uL/min"),
    "PharmaceuticalDoseUnits": ("mg", "ug", "mg/kg", "ug/kg", "mL"),
    "ModeOfAdministration": tuple(MODES_OF_ADMINISTRATION),
    "AcquisitionMode": tuple(PET_ACQUISITION_MODES),
    "ReconFilterType": (
        "Gaussian", "Hann", "Hamming", "Shepp-Logan", "Ramp", "Butterworth",
        "none",
    ),
    "Units": tuple(PET_IMAGE_UNITS),

    # --- the dataset itself ------------------------------------------
    "License": (
        "CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0", "PDDL-1.0", "ODbL-1.0", "ODC-BY-1.0", "None",
    ),
    "HEDVersion": ("8.3.0", "8.2.0", "8.1.0", "8.0.0"),
}

# Inventory column holding what the scan read out of a recording's own header,
# per field. The better hint of the two, and the reason it is offered rather
# than applied: a vendor string can always parse wrongly.
SCAN_SUGGESTION_COLUMNS: dict[str, str] = {
    "Manufacturer": "manufacturer_suggestion",
    "TracerName": "tracer_suggestion",
    "TracerRadionuclide": "radionuclide_suggestion",
    "InjectedRadioactivity": "injected_dose_suggestion",
    "ReconMethodName": "recon_method_suggestion",
}

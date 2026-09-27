"""Fill the BIDS PET sidecar after conversion.

Runs in convert phase 2, beside ``eeg_sidecar.enrich_recording_sidecars``, and
does three separable jobs on each staged ``*_pet.json``:

1. **Rename** what the converter wrote under a DICOM name to its BIDS name.
   dcm2niix writes ``ReconstructionMethod``; BIDS calls it ``ReconMethodName``.
2. **Fill** the fields no scanner records, from the user's metadata spec. This
   is most of the PET sidecar: a scanner knows how it reconstructed an image
   but not how much tracer went into the person, in what form, or when.
3. **Prune** keys that do not belong in a BIDS sidecar. dcm2niix is run with
   ``-ba n`` so BIDS Manager keeps ``SeriesInstanceUID`` for provenance, and
   that also keeps the patient identifiers alongside it.

Order matters: rename before fill, so a user value always wins over the
converter's, and prune last so a pruned key cannot be resurrected.

Everything is additive. A field the user left unset means "leave whatever the
converter wrote", which is what keeps a good dcm2niix value from being
clobbered by a blank.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from .identifiers import IDENTIFYING_KEYS
from ..recording_meta import (
    PET_LIST_TO_BIDS,
    PET_SCALAR_TO_BIDS,
    PetAcquisitionSpec,
    RecordingMetaSpec,
    resolve_pet,
)

log = logging.getLogger(__name__)

_SUPPORTED_DATATYPES = frozenset({"pet"})

# DICOM-derived key -> BIDS key. dcm2niix emits the left-hand names; the BIDS
# PET spec asks for the right-hand ones. A rename never overwrites a key that
# is already present under its BIDS name.
_RENAMES: dict[str, str] = {
    "ReconstructionMethod": "ReconMethodName",
}

# The identifying keys live in ``fixups/identifiers`` now, because the MRS
# header fixup prunes the same facts under partly different spellings
# (``PatientDoB`` rather than ``PatientBirthDate``) and a key one pass
# removes while another leaves is worse than a key neither removes: it makes
# the dataset look cleaned.
#
# BIDS Manager passes ``-ba n`` to keep SeriesInstanceUID for provenance,
# which also keeps these. The study and series UIDs are deliberately NOT
# pruned: they are pseudonymous, they are what lets a converted image be
# traced back to its source series, and BIDS permits extra keys.

# Non-identifying keys that are simply not part of BIDS and add noise. Kept
# separate from the identifying set so the two can be reasoned about apart.
NON_BIDS_KEYS: frozenset[str] = frozenset({
    # dcm2niix's own classification hint. Purely internal to the scan, and
    # meaningless in a shipped dataset.
    "BidsGuess",
    "Modality",
    "ProtocolName",
    "SeriesDescription",
    "StudyDescription",
    "SeriesTime",
    "SeriesNumber",
    "DecayCorrection",
    "RadionuclideHalfLife",
    "RadionuclidePositronFraction",
    "ImageType",
    "ConvolutionKernel",
})

# spec field -> BIDS sidecar key, for the values that map straight across.


@lru_cache(maxsize=1)
def _array_typed_keys() -> frozenset[str]:
    """Sidecar keys the BIDS schema types as arrays.

    dcm2niix writes a bare scalar for a single-frame acquisition where the
    schema wants a one-element array, and a validator rightly rejects that.
    Wrapping such a value is a type fix rather than a guess.

    Read from the schema instead of hardcoded, so the set follows the BIDS
    version in use. Fields the schema types with an ``anyOf`` (``type`` absent)
    are deliberately excluded: it accepts both shapes there, so touching the
    converter's choice would be meddling, not fixing.
    """
    from .. import schema as schema_mod

    keys = set()
    for key in _PET_SIDECAR_KEYS:
        try:
            if schema_mod.field_metadata(key).type == "array":
                keys.add(key)
        except KeyError:
            continue
    return frozenset(keys)


# Every key the fixup might write or touch, used to scope the schema lookup
# above. Values come from the spec maps plus the converter-written fields.
_PET_SIDECAR_KEYS: tuple[str, ...] = (
    "ScatterFraction", "DecayCorrectionFactor", "ReconFilterSize",
    "FrameDuration", "FrameTimesStart", "DoseCalibrationFactor",
    "ReconMethodParameterValues", "ReconMethodParameterLabels",
    "ReconMethodParameterUnits",
)


def enrich_pet_sidecars(
    subject_staging_dir: Path,
    tasks: Iterable,
    spec: Optional[RecordingMetaSpec],
    *,
    prune_identifiers: bool = True,
) -> int:
    """Enrich every staged PET sidecar for one subject.

    Parameters
    ----------
    subject_staging_dir
        Per-subject staging tree (``<bids_root>/.tmp_bidsmgr/sub-<id>/``).
    tasks
        The :class:`ConvertTask` objects for this subject. Non-PET tasks are
        ignored.
    spec
        The dataset's metadata spec. ``None`` still runs the rename, type-fix
        and prune passes, which need no user input.
    prune_identifiers
        Remove patient identifiers. On by default; the study and series UIDs
        are kept either way.

    Returns
    -------
    int
        Count of sidecar files modified.
    """
    n_modified = 0
    for task in tasks:
        if getattr(task, "datatype", "") not in _SUPPORTED_DATATYPES:
            continue
        basename = getattr(task, "basename", "") or ""
        if not basename:
            continue

        sidecar = _find_sidecar(subject_staging_dir, basename)
        if sidecar is None:
            continue

        # Read the DICOM header for the BIDS fields dcm2niix leaves out, before
        # anything the user stated is applied: this is reading, not stating, so
        # a template answer must still win over it. The rename, type-repair and
        # prune passes below then treat what it added like everything else.
        changed = _enrich_from_dicom(sidecar, task)

        pet = None
        if spec is not None:
            pet = resolve_pet(spec, str(getattr(task, "row_id", "")))

        if _apply_sidecar(sidecar, pet, prune_identifiers=prune_identifiers):
            changed = True
        if changed:
            n_modified += 1

    return n_modified


# ----------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------


def _enrich_from_dicom(sidecar: Path, task) -> bool:
    """Fill this sidecar from its own DICOM header. True if anything was added.

    ECAT tasks are skipped: that format is enriched as it is converted, by the
    backend that reads it, and its source files are not DICOM.

    Best-effort throughout. pet2bids crashes outright on at least one published
    phantom, so a failure here is logged and the conversion keeps every field it
    already had.
    """
    from ..inventory.pet_dicom_meta import enrich_pet_sidecar
    from ..inventory.pet_ecat import is_ecat_file

    for source in getattr(task, "source_files", ()) or ():
        candidate = Path(source)
        if not candidate.is_file() or is_ecat_file(candidate):
            continue
        added = enrich_pet_sidecar(sidecar, candidate)
        if added:
            log.info(
                "pet: %s gained %s from the DICOM header",
                sidecar.name, ", ".join(added),
            )
        return bool(added)
    return False


def _find_sidecar(staging: Path, basename: str) -> Optional[Path]:
    """Locate ``<basename>.json`` anywhere under the subject staging tree."""
    matches = sorted(staging.rglob(f"{basename}.json"))
    return matches[0] if matches else None


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read PET sidecar %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _apply_sidecar(
    sidecar: Path,
    pet: Optional[PetAcquisitionSpec],
    *,
    prune_identifiers: bool,
) -> bool:
    """Rename, derive, fill, type-fix and prune one sidecar. True if changed.

    **The order is the design.** Derive first, overlay second, prune last.
    Every step reads what the ones before it produced, and the prune is last
    so it is the final word: nothing a later step adds can reintroduce an
    identifier, and nothing the prune removes is still needed.

    Getting that order wrong is how ``TimeZero`` came to be missing from
    seven of twenty-three conversions. ``SeriesTime`` is in the prune list
    (it is not a BIDS field), and it is also the only thing dcm2niix gives
    us to derive ``TimeZero`` FROM, so pruning before deriving deleted the
    answer and then reported the question unanswered.
    """
    data = _read_json(sidecar)
    if not data:
        return False
    before = json.dumps(data, sort_keys=True)

    # 1. rename DICOM-derived keys, never clobbering an existing BIDS key
    for old, new in _RENAMES.items():
        if old in data:
            value = data.pop(old)
            data.setdefault(new, value)

    # 2. derive what the converter left for us, from what it DID write.
    _derive_time_zero(data)

    # 3. fill from the spec. A user value always wins: it is a deliberate
    #    statement, where the converter's is an inference.
    if pet is not None:
        for field, key in PET_SCALAR_TO_BIDS.items():
            value = getattr(pet, field, None)
            if value is not None and value != "":
                data[key] = value
        for field, key in PET_LIST_TO_BIDS.items():
            value = getattr(pet, field, None)
            if value:
                data[key] = list(value)

    # 4. derive the radio inputs the user did not supply, from the two they
    #    did. AFTER the overlay, so a stated value is never overwritten.
    _derive_radio_inputs(data)

    # 5. normalise the vocabularies BIDS defines as closed sets.
    _normalise_enums(data)

    # 6. type fixes: BIDS types several PET fields as one-entry-per-frame
    #    arrays, and dcm2niix writes a bare scalar for a single frame.
    for key in _array_typed_keys():
        if key in data and not isinstance(data[key], list):
            data[key] = [data[key]]

    # 7. prune the PET-specific noise. LAST, so nothing above can undo it.
    #
    # The IDENTIFIERS are not pruned here any more: ``fixups.identifiers``
    # does that for every sidecar in the dataset, which is where it belongs,
    # since an MRI sidecar carries exactly the same keys and used to keep
    # them. One owner per concern.
    drop = set(NON_BIDS_KEYS)
    if prune_identifiers:
        drop |= IDENTIFYING_KEYS
    for key in drop:
        data.pop(key, None)

    if json.dumps(data, sort_keys=True) == before:
        return False
    _write_json(sidecar, data)
    return True


def _derive_time_zero(data: dict) -> None:
    """Fill ``TimeZero`` from the clock time the converter wrote.

    BIDS REQUIRES ``TimeZero`` and dcm2niix deliberately stopped deriving it
    in v1.0.20260416, emitting ``SeriesTime`` instead so that each downstream
    tool can pick its own time-zero convention. Ours is the scan start, which
    is what ``ScanStart: 0`` (which dcm2niix also writes) already asserts.

    Read through :func:`parse_time_seconds` rather than trusting the string:
    dcm2niix writes ``"15:51:04"`` but a sidecar from another tool, or from
    an older dcm2niix, may carry the raw DICOM **TM** form ``"155104"``.
    ``parse_time_seconds`` reads both; a date parser reads neither correctly,
    which is the defect this whole function exists to route around.

    **``SeriesTime`` is preferred over ``AcquisitionTime``, and the order is
    not arbitrary.** ``AcquisitionTime`` is the more specific-sounding field
    and is the intuitive wrong answer. dcm2niix computes ``FrameTimesStart``
    as ``AcquisitionTime - SeriesTime``, so the frame times are measured
    FROM ``SeriesTime``; verified on 32 real phantom series where
    ``FrameTimesStart[0]`` equals that difference exactly, to the
    millisecond. Taking ``AcquisitionTime`` as time zero while the frame
    times are relative to ``SeriesTime`` puts every frame out by the gap
    between them, which on this data runs from 16 seconds to 32 minutes.

    An existing value is kept ONLY if it parses and is not midnight.
    ``"00:00:00"`` is the specific wrong answer a date parser returns when
    handed ``HHMMSS`` it happens not to choke on, and it is indistinguishable
    from a real answer by shape, so it is re-derived rather than trusted.
    """
    from ..inventory._time import parse_time_seconds

    def as_clock(raw) -> Optional[str]:
        seconds = parse_time_seconds(str(raw)) if raw not in (None, "") else None
        if seconds is None:
            return None
        total = int(round(seconds))
        return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"

    current = data.get("TimeZero")

    # ``SeriesTime`` is authoritative when it is there, and that means
    # CORRECTING a value the converter chain already wrote, not merely
    # filling a gap. Nothing the user stated can be lost this way: the
    # spec overlay runs after this, so a value from the form still wins.
    #
    # It has to correct rather than defer because pypet2bids writes
    # ``TimeZero`` from ``AcquisitionTime`` whenever the sidecar has one,
    # and dcm2niix v1.0.20260724 restored that field. On this lab's phantom
    # data that put six conversions out by between 16 seconds and 32
    # minutes: a well-formed, plausible clock time that no validator would
    # ever question, measured against the wrong reference.
    from_series = as_clock(data.get("SeriesTime"))
    if from_series is not None:
        if current not in (None, "") and current != from_series:
            log.info(
                "TimeZero %r replaced with SeriesTime %r: FrameTimesStart is "
                "measured from SeriesTime, so anything else puts every frame "
                "out by the difference",
                current, from_series,
            )
        data["TimeZero"] = from_series
        # Our convention: time zero IS the scan start, so the offset is zero.
        data.setdefault("ScanStart", 0)
        return

    if current not in (None, "", "00:00:00"):
        # No SeriesTime to check it against, and it is not the
        # sentinel-shaped wrong answer. Leave it.
        return

    clock = as_clock(data.get("AcquisitionTime"))
    if clock is None:
        return
    if current == "00:00:00" and clock == "00:00:00":
        # The header really does say midnight. Nothing to correct.
        return
    data["TimeZero"] = clock
    data.setdefault("ScanStart", 0)


def _derive_radio_inputs(data: dict) -> None:
    """Complete the dose / mass / specific-activity trio from any two of it.

    ``SpecificRadioactivity`` and its units are BIDS-REQUIRED, and
    ``InjectedRadioactivity`` comes straight off the DICOM on most scanners,
    so a user who enters the injected MASS has already supplied everything
    needed. Leaving the third for them to compute is asking for a number we
    can derive.

    **The arithmetic is ours, deliberately.** pypet2bids has a function for
    this (``check_meta_radio_inputs``) and its dose-over-mass branch reads
    ``(InjectedRadioactivity * 1e6) / (InjectedMass * 1e6)`` labelled
    ``Bq/g``. Micrograms to grams is ``x1e-6``, not ``x1e6``, so it is wrong
    by a factor of 1e12, and it contradicts the other two branches of the
    same function, which are right. Round-tripping its own output back
    through it yields an injected mass of five million kilograms.

    Only ever fills a blank. A stated value is a statement.
    """
    from ..editor.pet_checks import specific_activity_bq_per_g

    if data.get("SpecificRadioactivity") not in (None, ""):
        return
    derived = specific_activity_bq_per_g(
        data.get("InjectedRadioactivity"),
        data.get("InjectedRadioactivityUnits"),
        data.get("InjectedMass"),
        data.get("InjectedMassUnits"),
    )
    if derived is None:
        return
    data["SpecificRadioactivity"] = derived
    data["SpecificRadioactivityUnits"] = "Bq/g"


def _normalise_enums(data: dict) -> None:
    """Spell ``ModeOfAdministration`` the way the standard's examples do.

    **BIDS does NOT define a closed vocabulary here.** The schema types the
    field as a plain string; ``field_metadata("ModeOfAdministration").enum``
    is empty, and the three spellings everyone uses (``bolus``, ``infusion``,
    ``bolus-infusion``) appear only as examples in its description. Checked,
    rather than assumed, because lower-casing a free-text field would be
    vandalism: "Bolus then infusion over 60 min" is a legitimate value and
    must survive untouched.

    So this is a PRODUCT OPINION, not a schema fact, and it reads the
    curated vocabulary the rest of the tool already offers in its forms
    (``recording_meta.MODES_OF_ADMINISTRATION``) rather than a list of its
    own. A value that case-insensitively matches one of those three is
    written in the documented spelling; anything else is left exactly as
    the user wrote it.
    """
    raw = data.get("ModeOfAdministration")
    if not isinstance(raw, str) or not raw.strip():
        return
    text = raw.strip()
    from ..recording_meta.models import MODES_OF_ADMINISTRATION

    for option in MODES_OF_ADMINISTRATION:
        if text.lower() == option.lower():
            data["ModeOfAdministration"] = option
            return
    data["ModeOfAdministration"] = text



__all__ = [
    "IDENTIFYING_KEYS",
    "NON_BIDS_KEYS",
    "enrich_pet_sidecars",
]

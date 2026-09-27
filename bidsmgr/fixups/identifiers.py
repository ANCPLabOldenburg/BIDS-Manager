"""What counts as an identifier, in one place.

Three things prune identifiers now: the PET sidecar fixup, the MRS header
fixup, and anything added later. They must agree, because a key that one of
them removes and another leaves is worse than a key neither removes: it
makes the dataset look cleaned.

The spellings differ by carrier. DICOM and dcm2niix say ``PatientBirthDate``;
the NIfTI-MRS header standard says ``PatientDoB``. Both are here, and both
name the same fact about a person.
"""

from __future__ import annotations

# Keys that identify the participant, or the person who operated the
# scanner, or the exact machine. None is a BIDS field.
#
# Study and series UIDs are deliberately NOT here. They are pseudonymous,
# they are what lets a converted image be traced back to its source series,
# and BIDS permits extra keys.
IDENTIFYING_KEYS: frozenset[str] = frozenset({
    # the participant
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientDoB",            # the NIfTI-MRS spelling of the same thing
    "PatientSex",
    "PatientAge",
    "PatientWeight",
    "PatientSize",
    # the visit
    "AccessionNumber",
    # the people
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "RequestingPhysician",
    # The department's own address, which dcm2niix carries and BIDS does not
    # define. NOT ``InstitutionAddress``: the schema declares that one as a
    # RECOMMENDED field for every datatype, it describes the site rather than
    # the participant, and pruning it would be this tool overriding the
    # standard on a privacy question the standard has already answered.
    # Checked against the schema rather than assumed.
    "InstitutionalDepartmentAddress",
})

# Keys that name the source files on the converting machine. Not identifying
# in themselves, but they carry a local directory layout into a dataset that
# is meant to be shareable, and they are not BIDS fields.
PROVENANCE_PATH_KEYS: frozenset[str] = frozenset({
    "OriginalFile",
})


def strip_identifiers(data: dict, *, paths: bool = True) -> list[str]:
    """Remove identifying keys from ``data`` in place; return what went.

    ``paths`` also removes the source-file listing, which is on by default
    because the default is a dataset somebody intends to share.
    """
    drop = set(IDENTIFYING_KEYS)
    if paths:
        drop |= PROVENANCE_PATH_KEYS
    removed = [key for key in list(data) if key in drop]
    for key in removed:
        del data[key]
    return removed


__all__ = [
    "IDENTIFYING_KEYS",
    "PROVENANCE_PATH_KEYS",
    "prune_identifiers",
    "strip_identifiers",
]


def prune_identifiers(staging_dir) -> int:
    """Remove identifying keys from every staged JSON sidecar. Count changed.

    **This has to be dataset-wide, not per modality.** BIDS Manager runs
    dcm2niix with ``-ba n`` so that ``SeriesInstanceUID`` survives for
    provenance, and that flag keeps the patient identifiers alongside it.
    Until now only the PET fixup pruned them, so a PET dataset was clean and
    an MRI one was not: measured on this lab's own data, 58 of 71 MRI
    sidecars carried the participant's name, ID, date of birth, age, sex,
    size and weight, plus the institution's street address.

    The BIDS specification is explicit that a sidecar must not carry
    identifying information, so there is no "off" here and no per-modality
    opinion about it. The study and series UIDs stay: they are pseudonymous
    and they are what lets an image be traced back to its source series.
    """
    from pathlib import Path
    import json
    import logging

    log = logging.getLogger(__name__)
    changed = 0
    for path in sorted(Path(staging_dir).rglob("*.json")):
        if ".bidsmgr" in path.parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        removed = strip_identifiers(data, paths=False)
        if not removed:
            continue
        try:
            path.write_text(
                json.dumps(data, indent=4) + "\n", encoding="utf-8",
            )
        except OSError as exc:
            log.warning("could not rewrite %s: %s", path.name, exc)
            continue
        changed += 1
        log.info(
            "%s: removed %d identifying field(s): %s",
            path.name, len(removed), ", ".join(sorted(removed)),
        )
    return changed

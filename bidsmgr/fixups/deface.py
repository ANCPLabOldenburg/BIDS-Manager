"""Remove faces while the subject is still in staging, before it is committed.

This is the same defacing the Editor does, run at the one moment where it is
strictly better: **the face-bearing image has never been in the BIDS tree.**

Conversion writes each subject into ``<bids_root>/.tmp_bidsmgr/sub-XXX/`` and
moves it into place atomically when everything has succeeded. Defacing inside
that staging directory means the identifiable image exists only there, for a
few seconds, and what lands in the dataset was never identifiable. Nothing is
backed up because there is nothing to restore to: the original is a file the
conversion made a moment ago from source data the user still has.

That is the difference from :mod:`bidsmgr.deface.apply`, which edits a dataset
that already exists and therefore has to be reversible. Both call the same
engine; only what surrounds it differs.

A failure here does not fail the conversion. A subject whose face could not be
removed is still a correctly converted subject, and refusing to write it would
be worse for the user than writing it and saying so. The count and the failures
are returned for the caller to log.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from ..deface.engines import DEFAULT_ENGINE_ID, engine
from ..deface.run import DefaceFailed, available, deface_to_temp
from ..deface.select import FACE_BEARING_DATATYPES
from ..deface.status import read_sidecar, record, sidecar_for
from ..deface.probe import is_single_volume

log = logging.getLogger(__name__)

NIFTI_SUFFIXES = (".nii.gz", ".nii")


def _staged_candidates(staging: Path) -> list[Path]:
    """Face-bearing 3-D images under a staged subject.

    The staging tree is ``<staging>/[ses-X/]<datatype>/<file>``, so the
    datatype is the parent folder, the same as in a committed dataset. The
    location checks :mod:`bidsmgr.deface.select` makes do not apply: there is
    no ``derivatives/`` here, and everything present was just written by us.
    """
    out: list[Path] = []
    for path in sorted(Path(staging).rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if not any(name.endswith(s) for s in NIFTI_SUFFIXES):
            continue
        if path.parent.name not in FACE_BEARING_DATATYPES:
            continue
        try:
            single = is_single_volume(path)
        except Exception as exc:  # noqa: BLE001 - see the module docstring
            log.warning("deface: could not read %s: %s", path.name, exc)
            continue
        if not single:
            log.info("deface: %s is not a single volume; left alone", path.name)
            continue
        out.append(path)
    return out


def deface_staged(
    staging: Path,
    tasks: Optional[Iterable] = None,
    *,
    engine_id: str = DEFAULT_ENGINE_ID,
) -> int:
    """Deface the staged anatomical and PET images. Returns how many.

    ``tasks`` is accepted and unused, so this matches the signature every other
    phase-2 fixup has and can be called from the same place without a special
    case.
    """
    del tasks

    staging = Path(staging)
    if not staging.is_dir():
        return 0
    if not available():
        log.warning(
            "deface: niimath is not available, so no faces were removed."
        )
        return 0

    try:
        eng = engine(engine_id)
    except KeyError as exc:
        # An id this build does not know, almost always a stale setting. It
        # used to raise here, INSIDE the subject commit, so a bad string in
        # QSettings lost the entire conversion instead of just the defacing.
        log.warning("deface: %s; no faces were removed", exc)
        return 0

    candidates = _staged_candidates(staging)
    if not candidates:
        return 0

    done = 0
    for image in candidates:
        try:
            result = deface_to_temp(
                image, engine_id=eng.id, directory=image.parent,
            )
        except DefaceFailed as exc:
            # Not fatal. A subject whose face could not be removed is still a
            # correctly converted subject, and refusing to write it would lose
            # the conversion as well as the defacing.
            log.warning("deface: %s was not defaced: %s", image.name, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            # Anything else the engine or the filesystem can throw. The
            # module's contract is that defacing cannot fail a conversion, and
            # a contract that only holds for the ONE exception type we
            # remembered is not a contract.
            log.warning(
                "deface: %s was not defaced (%s: %s)",
                image.name, type(exc).__name__, exc,
            )
            continue

        try:
            # In staging, so a plain replace is right: there is nothing to
            # back up and nothing that could want the original back.
            result.output.replace(image)
        except OSError as exc:
            log.warning("deface: could not replace %s: %s", image.name, exc)
            result.output.unlink(missing_ok=True)
            continue

        sidecar = sidecar_for(image)
        try:
            import json

            sidecar.write_text(
                json.dumps(record(read_sidecar(sidecar), eng), indent=4) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            # The image IS defaced; only the record of it failed. Say so
            # loudly, because a defaced image that does not say it is defaced
            # will be defaced again by the next person who looks.
            log.warning(
                "deface: %s was defaced but its sidecar could not be "
                "updated: %s", image.name, exc,
            )
        done += 1

    if done:
        log.info("deface: removed faces from %d image(s) with %s",
                 done, eng.label)
    return done


__all__ = ["deface_staged"]

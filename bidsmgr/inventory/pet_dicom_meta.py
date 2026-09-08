"""PET DICOM fields dcm2niix does not write, read through pet2bids.

dcm2niix reads a PET DICOM well but not exhaustively. It leaves behind fields
BIDS asks for that the header does state: ``TimeZero``, which BIDS REQUIRES and
which every frame time and every blood sample is measured against; when the
injection started; and the reconstruction method broken into its name, its
iterations and its subsets.

pet2bids has a pass that goes back to the DICOM header for exactly those, and it
is the reference implementation from the group that wrote this part of the
standard. So this reads through them rather than parsing the header again.

What is ours is everything about how the result is used:

* **Ours wins.** Only fields absent from what we already wrote are added. This
  is not mere caution. Run against our sidecar, their pass rewrites
  ``TracerRadionuclide`` from ``F18`` to ``18Fluorine``, and the standard's own
  example for that field is ``"C11"``. Filling gaps takes their reading without
  taking their spelling.
* **The schema decides what is a field.** Their pass leaves dcm2niix's
  ``BidsGuess``, ``ProtocolName`` and ``SeriesDescription`` behind. Those are
  converter output, not BIDS metadata.
* **A placeholder is not an answer.** They write ``ReconFilterType`` as the
  literal ``"n/a"`` when the header does not say. Rejected, so the form still
  asks.
* **Their crash is our warning.** Their DICOM path raises
  ``ParserError: month must be in 1..12`` on the Philips Gemini phantom, a
  published OpenNeuroPET file, and writes nothing at all. Here enrichment is a
  bonus, so it is logged and the conversion keeps every field it already had.
  The reconstruction fields are read through a separate call for the same
  reason: a failure in their date handling should not cost us the recon.

Qt-free. Used by the scan probe and by the conversion fixup, so the two can
never disagree about what the conversion answers.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Guards their module-level state. Their ECAT reader turned out to share one
# mutable template between every instance in a process, which corrupted frame
# timing across a batch; until this pass is proven free of the same habit, it is
# called one at a time. Conversion runs several series at once.
_PET2BIDS_LOCK = threading.Lock()

# DICOM (0054,1103), the reconstruction method as the scanner spells it.
_RECON_METHOD_TAG = (0x0054, 0x1103)

# DICOM (0018,1210), the reconstruction filter, which BIDS splits into a type
# and a size in millimetres.
_CONVOLUTION_KERNEL_TAG = (0x0018, 0x1210)


def _schema_fields() -> set[str]:
    """The names BIDS declares for a PET sidecar, or an empty set."""
    try:
        from .. import schema as schema_mod

        return {f.name for f in schema_mod.sidecar_fields("pet", "pet")}
    except Exception:  # noqa: BLE001
        return set()


def _read_header(dicom_path: Path):
    """One representative DICOM of the series, headers only."""
    import pydicom

    return pydicom.dcmread(str(dicom_path), stop_before_pixels=True)


def _recon_fields(header) -> dict:
    """The reconstruction method, split the way BIDS wants it.

    A scanner writes one string, ``"OSEM 4i16s"`` or ``"3D-OSEM-PSF 3i21s"``.
    BIDS wants the method named in full plus its parameters as labels, units and
    values. pet2bids knows the vendor spellings, which is a body of knowledge
    worth borrowing rather than rebuilding.

    Called separately from the main pass so that a failure in their date
    handling, which is where they crash on real files, does not cost us this.
    """
    try:
        raw = header[_RECON_METHOD_TAG].value
    except (KeyError, TypeError, AttributeError):
        return {}
    text = str(raw or "").strip()
    if not text:
        return {}

    try:
        from pypet2bids import helper_functions

        parsed = helper_functions.get_recon_method(text)
    except Exception as exc:  # noqa: BLE001
        log.info("could not parse reconstruction method %r: %s", text, exc)
        return {}
    return dict(parsed or {})


# The three reconstruction-parameter fields describe one another: a label says
# what a value means, a unit says what it is measured in. BIDS presents them as
# parallel lists.
_RECON_TRIO = (
    "ReconMethodParameterLabels",
    "ReconMethodParameterUnits",
    "ReconMethodParameterValues",
)


def _coherent_recon(fields: dict) -> dict:
    """Drop reconstruction parameters that do not describe anything.

    Their parser knows ``"Filtered Back Projection"`` and returns no parameters
    for it, correctly, because it has none. Given the same method spelled
    ``"2D Filtered Backprojection"``, which is how a real GE Advance writes it,
    it falls through to a default and returns
    ``ReconMethodParameterLabels: ["none", "none"]`` with no values at all.

    That is worse than saying nothing. It is two labels for a method with no
    parameters, naming them after a unit, and a reader has no way to tell it
    from a real answer. So the trio is kept only when the values exist and the
    labels line up with them.
    """
    values = fields.get("ReconMethodParameterValues")
    labels = fields.get("ReconMethodParameterLabels")

    coherent = (
        isinstance(values, (list, tuple)) and len(values) > 0
        and isinstance(labels, (list, tuple)) and len(labels) == len(values)
        and not all(str(v).strip().lower() in ("none", "n/a", "") for v in labels)
    )
    if coherent:
        return fields
    return {k: v for k, v in fields.items() if k not in _RECON_TRIO}


def _what_is_missing(sidecar: dict) -> dict:
    """Every PET field the schema declares that this sidecar has not answered.

    Their updater only looks for fields their own ``check_json`` reports as
    missing, and that runs off a frozen list of what they consider mandatory
    and recommended. Fields outside it are never attempted, which is why
    ``ReconFilterType`` stayed absent from our sidecars even though their code
    can read it.

    So the question "what is still missing" is answered from the schema, which
    is where facts about BIDS come from, and their reader is pointed at all of
    it. The shape is theirs: ``{name: {"key": bool, "value": bool}}``.
    """
    declared = _schema_fields()
    if not declared:
        return {}

    from .pet_ecat import _is_answer

    return {
        name: {"key": False, "value": False}
        for name in declared
        if not _is_answer(sidecar.get(name))
    }


def _filter_fields(header) -> dict:
    """``ReconFilterType`` and ``ReconFilterSize`` from the convolution kernel.

    BIDS declares both and the header states both, so nobody should have to
    type them. Vendors pack the two together and disagree about how:

    * ``"All-pass"``
    * ``["hanning", "  4.000000 mm", " order 0"]``
    * ``["Rad:", "rectangle", "4.000000 mm", "Ax:", "rectangle", "8.500000 mm"]``

    Parsed with the splitter the inventory already uses for its filter hint, so
    the table and the sidecar cannot disagree about what the kernel says.

    pet2bids gets this by deleting the size from the string with a regular
    expression and keeping the remainder, which turns ``"hanning 4.00000mm"``
    into ``"hanning 00000 mm"``. Taking the type and the number separately
    gives ``"hanning"`` and ``4.0``.
    """
    try:
        raw = header[_CONVOLUTION_KERNEL_TAG].value
    except (KeyError, TypeError, AttributeError):
        return {}

    # pydicom hands back a MultiValue for a multi-valued tag. It is a sequence
    # but not a list, so an isinstance check against list misses it and str()
    # then yields the repr, "['hanning', ' 4.000000 mm']", which parses to
    # "['hanning'". Anything iterable that is not a string is joined the way
    # DICOM stores it.
    if isinstance(raw, str):
        text = raw.strip()
    elif raw is None:
        text = ""
    else:
        try:
            text = "\\".join(str(part) for part in raw).strip()
        except TypeError:
            text = str(raw).strip()
    if not text:
        return {}

    from .pet import parse_recon_filter

    ftype, size = parse_recon_filter(text)
    out: dict = {}
    if ftype:
        out["ReconFilterType"] = ftype
    if size is not None:
        out["ReconFilterSize"] = size
    return out


def _their_pass(sidecar_path: Path, header) -> dict:
    """Run their enrichment against a COPY and return what it produced.

    A copy because it writes in place: our real sidecar is never exposed to a
    half-finished write, and what survives is our decision rather than theirs.
    """
    from ..util.pet2bids import quiet_pet2bids

    with tempfile.TemporaryDirectory() as scratch:
        scratch_json = Path(scratch) / "sidecar.json"
        try:
            shutil.copyfile(sidecar_path, scratch_json)
        except OSError as exc:  # noqa: BLE001
            log.warning("could not stage %s for enrichment: %s", sidecar_path, exc)
            return {}

        try:
            with quiet_pet2bids():
                from pypet2bids.update_json_pet_file import (
                    check_json,
                    update_json_with_dicom_value,
                )

                # Union of their view and the schema's. Theirs carries
                # spelling quirks their updater keys on; the schema carries
                # everything BIDS actually declares.
                missing = dict(check_json(str(scratch_json), silent=True) or {})
                try:
                    current = json.loads(scratch_json.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    current = {}
                for name, state in _what_is_missing(current).items():
                    missing.setdefault(name, state)

                update_json_with_dicom_value(
                    str(scratch_json), missing, header, silent=True,
                )
        except Exception as exc:  # noqa: BLE001 - their known crash lands here
            log.warning(
                "pet2bids could not enrich %s (%s: %s); the sidecar keeps what "
                "the conversion already wrote",
                sidecar_path.name, type(exc).__name__, exc,
            )
            return {}

        try:
            return json.loads(scratch_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}


def dicom_sidecar_fields(
    dicom_path: Path,
    already_have: Optional[dict] = None,
    *,
    sidecar_path: Optional[Path] = None,
) -> dict:
    """Fields to ADD to a PET sidecar, read from the DICOM header.

    ``already_have`` is what the conversion already wrote; nothing in it is
    touched, and nothing already answered is returned. ``sidecar_path`` is used
    as the starting point for their pass when given, because their code decides
    what to look for from what is missing.

    Returns ``{}`` for anything unreadable. Enrichment is a bonus and must never
    cost a user their conversion.
    """
    from .pet_ecat import _is_answer  # same rule, one implementation

    have = dict(already_have or {})
    path = Path(dicom_path)
    if not path.is_file():
        return {}

    try:
        header = _read_header(path)
    except Exception as exc:  # noqa: BLE001
        log.info("could not read DICOM header %s: %s", path.name, exc)
        return {}

    with _PET2BIDS_LOCK:
        produced: dict = {}
        if sidecar_path is not None and Path(sidecar_path).is_file():
            produced.update(_their_pass(Path(sidecar_path), header))
        # Read separately so their date handling cannot cost us the recon.
        produced.update(_recon_fields(header))
        produced.update(_filter_fields(header))
    produced = _coherent_recon(produced)

    declared = _schema_fields()
    return {
        name: value
        for name, value in produced.items()
        if name not in have
        and _is_answer(value)
        and (not declared or name in declared)
    }


def enrich_pet_sidecar(sidecar_path: Path, dicom_path: Path) -> list[str]:
    """Fill a written PET sidecar in place. Returns the names added.

    The single place both the scan probe and the conversion reach for, so what
    the template shows as already answered is exactly what conversion writes.
    """
    sidecar = Path(sidecar_path)
    try:
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(existing, dict):
        return []

    added = dicom_sidecar_fields(dicom_path, existing, sidecar_path=sidecar)
    if not added:
        return []

    existing.update(added)
    try:
        sidecar.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:  # noqa: BLE001
        log.warning("could not write enriched sidecar %s: %s", sidecar.name, exc)
        return []
    return sorted(added)


__all__ = ["dicom_sidecar_fields", "enrich_pet_sidecar"]

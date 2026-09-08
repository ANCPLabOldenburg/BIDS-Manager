"""PET-specific inventory columns and the DICOM-to-suggestion derivations.

The BIDS PET sidecar needs roughly forty required fields. dcm2niix already
supplies most of the ones that live in the DICOM header verbatim. This module
covers the rest of what the *scan* can contribute: the tags that need renaming,
unit conversion or vendor-string parsing before they mean anything in BIDS.

Everything here produces a **suggestion**, never a value that is silently
written. The reconstruction string in particular is free text whose grammar
varies by manufacturer (``PSF+TOF 3i21s``, ``3D Kinahan - Rogers``,
``OSEM:i3s21``), so a wrong parse must be visible and correctable rather than
baked into the sidecar. This mirrors the montage and manufacturer suggestions
the EEG/MEG scanner already emits.

Qt-free; imports only from :mod:`bidsmgr.schema` and the standard library.

Reconstruction-string grammars were cross-checked against the OpenNeuroPET
phantom set (12 scanner models) and against the conventions used by the
openneuropet/pet2bids project, whose accumulated vendor knowledge informed the
parameter-label table below.
"""

from __future__ import annotations

import re
from typing import Optional

# PET-specific inventory columns. Appended after EEG_MEG_COLUMNS, so the
# pre-PET column order is untouched and an MRI-only or EEG-only inventory is
# unaffected (the orchestrator backfills them with empty strings).
PET_COLUMNS: tuple[str, ...] = (
    "tracer_suggestion",          # Radiopharmaceutical, cleaned to a BIDS-ish label
    "radionuclide_suggestion",    # RadionuclideCodeSequence CodeMeaning, e.g. "F18"
    "recon_method_suggestion",    # ReconstructionMethod with parameters stripped
    "recon_filter_suggestion",    # ConvolutionKernel, first component
    "injected_dose_suggestion",   # RadionuclideTotalDose in MBq
)

# Radionuclide code meanings vary in spelling across vendors ("18F", "F-18",
# "F18", "Fluorine-18"). BIDS wants the compact form.
_RADIONUCLIDE_RE = re.compile(r"^\s*(\d+)\s*[-]?\s*([A-Za-z]{1,2})\s*$")
_RADIONUCLIDE_ALT_RE = re.compile(r"^\s*([A-Za-z]{1,2})\s*[-]?\s*(\d+)\s*$")

_ELEMENT_NAMES = {
    "fluorine": "F", "carbon": "C", "oxygen": "O", "nitrogen": "N",
    "gallium": "Ga", "zirconium": "Zr", "copper": "Cu", "iodine": "I",
    "rubidium": "Rb", "bromine": "Br", "technetium": "Tc",
    "germanium": "Ge", "yttrium": "Y", "lutetium": "Lu", "indium": "In",
    "thallium": "Tl", "nitrogen13": "N", "scandium": "Sc", "titanium": "Ti",
}

# DICOM's RadionuclideCodeSequence CodeMeaning uses a caret notation for the
# mass number: "^18^Fluorine", "^68^Germanium", "^11^Carbon". Seen in 8 of the
# 12 phantom scanner models, so it is the common case rather than the exotic one.
_CARET_NUCLIDE_RE = re.compile(r"\^?(\d+)\^?\s*([A-Za-z]+)")

# Common tracer names as they appear in DICOM, mapped to the short label BIDS
# datasets conventionally use. Anything unmatched is passed through cleaned up
# rather than dropped, because the user can always correct a suggestion.
_TRACER_ALIASES = {
    "fluorodeoxyglucose": "FDG",
    "fdg": "FDG",
    "fdg -- fluorodeoxyglucose": "FDG",
    "2-fluoro-2-deoxy-d-glucose": "FDG",
    "flortaucipir": "FTP",
    "florbetapir": "AV45",
    "florbetaben": "FBB",
    "flutemetamol": "FMM",
    "pib": "PIB",
    "raclopride": "RAC",
    "carfentanil": "CFN",
    "dopa": "FDOPA",
    "fdopa": "FDOPA",
}

# Iterative reconstruction parameter shorthands, by manufacturer convention.
# ``3i21s`` means 3 iterations, 21 subsets; ``i3s21`` the same the other way
# round. Both appear in the phantom set.
_ITER_SUBSET_RE = re.compile(
    r"(?:(?P<i1>\d+)\s*i\D{0,3}(?P<s1>\d+)\s*s)"
    r"|(?:i\s*(?P<i2>\d+)\D{0,3}s\s*(?P<s2>\d+))",
    re.IGNORECASE,
)


def normalise_radionuclide(raw: str) -> str:
    """Reduce any vendor spelling of a radionuclide to the BIDS short form.

    ``"18F"``, ``"F-18"``, ``"Fluorine-18"`` and DICOM's caret notation
    ``"^18^Fluorine"`` all become ``"F18"``. An unrecognised string is returned
    unchanged rather than dropped, because this is a suggestion the user can
    correct, and a visible odd value beats a silently empty field.
    """
    text = (raw or "").strip()
    if not text:
        return ""

    # Caret notation first: it embeds both parts unambiguously.
    m = _CARET_NUCLIDE_RE.search(text)
    if m and "^" in text:
        mass, element = m.group(1), m.group(2).lower()
        sym = _ELEMENT_NAMES.get(element)
        if sym:
            return f"{sym}{mass}"
        if len(element) <= 2:
            return f"{element.capitalize()}{mass}"

    lowered = text.lower().replace(" ", "")
    for name, sym in _ELEMENT_NAMES.items():
        if lowered.startswith(name):
            digits = re.sub(r"\D", "", lowered)
            return f"{sym}{digits}" if digits else sym

    m = _RADIONUCLIDE_RE.match(text)          # "18F" / "18-F"
    if m:
        return f"{m.group(2).capitalize()}{m.group(1)}"
    m = _RADIONUCLIDE_ALT_RE.match(text)      # "F18" / "F-18"
    if m:
        return f"{m.group(1).capitalize()}{m.group(2)}"
    return text


def normalise_tracer(raw: str) -> str:
    """Map a DICOM ``Radiopharmaceutical`` string to a short tracer label."""
    text = (raw or "").strip()
    if not text:
        return ""
    key = text.lower().strip()
    if key in _TRACER_ALIASES:
        return _TRACER_ALIASES[key]
    # "FDG -- fluorodeoxyglucose" style: take the part before the separator.
    head = re.split(r"\s*--\s*|\s*,\s*|\s*\(", text)[0].strip()
    return _TRACER_ALIASES.get(head.lower(), head)


def parse_recon_method(raw: str) -> tuple[str, list[str], list[int]]:
    """Split a vendor reconstruction string into name plus parameters.

    ``"PSF+TOF 3i21s"`` becomes ``("PSF+TOF", ["iterations", "subsets"], [3, 21])``.
    A string with no recognisable iteration/subset shorthand returns the whole
    string as the name and empty parameter lists, which is the correct answer
    for analytic reconstructions such as ``"3D Kinahan - Rogers"``.
    """
    text = (raw or "").strip()
    if not text:
        return "", [], []

    m = _ITER_SUBSET_RE.search(text)
    if not m:
        return text, [], []

    iterations = m.group("i1") or m.group("i2")
    subsets = m.group("s1") or m.group("s2")
    name = (text[: m.start()] + text[m.end():]).strip(" ,-:")
    return name or text, ["iterations", "subsets"], [int(iterations), int(subsets)]


def parse_recon_filter(raw: str) -> tuple[str, Optional[float]]:
    """Split a ``ConvolutionKernel`` into a filter type and its size in mm.

    Vendors pack several components into one string, backslash separated:
    ``"Rad:\\rectangle\\4.000000 mm\\Ax:\\rectangle\\8.500000 mm"``. BIDS wants
    a single type and size, so take the first component that names a shape and
    the first millimetre figure alongside it.
    """
    text = (raw or "").strip()
    if not text:
        return "", None

    parts = [p.strip() for p in re.split(r"[\\|,]", text) if p.strip()]
    size: Optional[float] = None
    ftype = ""
    for part in parts:
        mm = re.search(r"([\d.]+)\s*mm", part, re.IGNORECASE)
        if mm and size is None:
            try:
                size = float(mm.group(1))
            except ValueError:
                pass
            continue
        if not ftype and not part.endswith(":") and not re.fullmatch(r"[\d.]+", part):
            ftype = part

    # Some vendors run the two together with no separator and no unit, as in
    # Siemens' "XYZGAUSSIAN3.00". Split only when a size was not found any
    # other way, so a string that merely ends in a digit is left alone.
    if size is None and ftype:
        packed = re.fullmatch(r"([A-Za-z][A-Za-z\-\s]*?)\s*([\d.]+)", ftype)
        if packed:
            try:
                size = float(packed.group(2))
                ftype = packed.group(1).strip()
            except ValueError:
                pass
    return ftype or (parts[0] if parts else ""), size


def becquerel_to_megabecquerel(raw: str) -> Optional[float]:
    """DICOM ``RadionuclideTotalDose`` is in Bq; BIDS conventionally uses MBq."""
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return round(value / 1e6, 4)


def derive_suggestions(pet_tags: dict) -> dict[str, str]:
    """Turn one series' raw PET tag block into the suggestion columns.

    Returns only the keys it can fill, so callers can update blindly.
    """
    if not pet_tags:
        return {}

    out: dict[str, str] = {}

    tracer = normalise_tracer(pet_tags.get("tracer", ""))
    if tracer:
        out["tracer_suggestion"] = tracer

    nuclide = normalise_radionuclide(pet_tags.get("radionuclide", ""))
    if nuclide:
        out["radionuclide_suggestion"] = nuclide

    name, _labels, values = parse_recon_method(pet_tags.get("recon_method", ""))
    if name:
        out["recon_method_suggestion"] = (
            f"{name} ({values[0]}i {values[1]}s)" if values else name
        )

    ftype, fsize = parse_recon_filter(pet_tags.get("recon_filter", ""))
    if ftype:
        out["recon_filter_suggestion"] = (
            f"{ftype} {fsize} mm" if fsize is not None else ftype
        )

    dose = becquerel_to_megabecquerel(pet_tags.get("injected_dose", ""))
    if dose is not None:
        out["injected_dose_suggestion"] = f"{dose} MBq"

    return out


__all__ = [
    "PET_COLUMNS",
    "becquerel_to_megabecquerel",
    "derive_suggestions",
    "normalise_radionuclide",
    "normalise_tracer",
    "parse_recon_filter",
    "parse_recon_method",
]

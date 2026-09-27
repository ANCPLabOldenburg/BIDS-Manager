"""Read PET metadata from a spreadsheet.

A PET study's radiochemistry usually already exists as a table: one row per
scan, columns for tracer, dose, mass and injection time, exported from the lab's
own records. Retyping that into a dialog is both tedious and a fresh chance to
make a transcription error, so read the table instead.

The reader is deliberately forgiving about column naming, because the source is
somebody else's spreadsheet rather than a format we control. It is deliberately
strict about values: a cell that will not parse as the number BIDS expects is
skipped and logged, never coerced, because a silently coerced dose looks exactly
like a deliberate one in the resulting sidecar.

Qt-free.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from ..recording_meta import (
    PET_LIST_TO_BIDS,
    PET_SCALAR_TO_BIDS,
    PetAcquisitionSpec,
)

log = logging.getLogger(__name__)

# Column header -> spec field. Matched case-insensitively after stripping
# spaces, underscores and hyphens, so "Injected Radioactivity",
# "injected_radioactivity" and "InjectedRadioactivity" all land together.
_COLUMN_ALIASES: dict[str, str] = {
    "tracername": "tracer_name",
    "tracer": "tracer_name",
    "tracerradionuclide": "tracer_radionuclide",
    "radionuclide": "tracer_radionuclide",
    "isotope": "tracer_radionuclide",
    "injectedradioactivity": "injected_radioactivity",
    "dose": "injected_radioactivity",
    "injecteddose": "injected_radioactivity",
    "injectedradioactivityunits": "injected_radioactivity_units",
    "doseunits": "injected_radioactivity_units",
    "injectedmass": "injected_mass",
    "injectedmassunits": "injected_mass_units",
    "specificradioactivity": "specific_radioactivity",
    "specificradioactivityunits": "specific_radioactivity_units",
    "molaractivity": "molar_activity",
    "molaractivityunits": "molar_activity_units",
    "injectedvolume": "injected_volume",
    "modeofadministration": "mode_of_administration",
    "administration": "mode_of_administration",
    "injectionstart": "injection_start",
    "injectionend": "injection_end",
    "timezero": "time_zero",
    "scanstart": "scan_start",
    "acquisitionmode": "acquisition_mode",
    "imagedecaycorrected": "image_decay_corrected",
    "imagedecaycorrectiontime": "image_decay_correction_time",
    "attenuationcorrection": "attenuation_correction",
    "units": "units",
    "bodypart": "body_part",
    "reconmethodname": "recon_method_name",
    "reconfiltertype": "recon_filter_type",
    "reconfiltersize": "recon_filter_size",
    "manufacturer": "manufacturer",
    "manufacturersmodelname": "manufacturers_model_name",
}

# Columns that identify which scan a row describes.
_KEY_ALIASES: tuple[str, ...] = (
    "participantid", "participant", "subject", "subjectid", "sub",
    "bidsname", "rowid", "filename", "sourcefile",
)

_NUMERIC_FIELDS: frozenset[str] = frozenset({
    "injected_radioactivity", "injected_mass", "specific_radioactivity",
    "molar_activity", "injected_volume", "injection_start", "injection_end",
    "scan_start", "image_decay_correction_time", "recon_filter_size",
})

_BOOL_FIELDS: frozenset[str] = frozenset({"image_decay_corrected"})

_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


def _normalise(header: str) -> str:
    return re.sub(r"[\s_\-.]", "", str(header)).lower()


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()
    try:
        if suffix in (".tsv", ".txt"):
            return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if suffix == ".csv":
            return pd.read_csv(path, dtype=str, keep_default_na=False)
        return pd.read_excel(path, dtype=str).fillna("")
    except Exception as exc:  # noqa: BLE001 - a bad file must not kill the run
        log.warning("could not read PET metadata spreadsheet %s: %s", path, exc)
        return None


def _coerce(field: str, raw: str) -> Optional[object]:
    """Parse one cell, or return ``None`` when it cannot be trusted."""
    text = str(raw).strip()
    if not text or text.lower() in ("n/a", "na", "none", "-"):
        return None
    if field in _NUMERIC_FIELDS:
        try:
            return float(text)
        except ValueError:
            log.warning(
                "PET spreadsheet: %r is not a number for %s; skipping the cell",
                text, field,
            )
            return None
    if field in _BOOL_FIELDS:
        low = text.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        log.warning(
            "PET spreadsheet: %r is not a yes/no value for %s; skipping", text, field,
        )
        return None
    if field in _LIST_FIELDS:
        # BIDS types the reconstruction-parameter trio as arrays, and a
        # spreadsheet cell is one string. Split on the separators a person
        # actually types; a single value becomes a one-element list, which
        # is what the schema wants and what a bare scalar is not.
        parts = [p.strip() for p in re.split(r"[;,|]", text) if p.strip()]
        return [_maybe_number(p) for p in parts]
    return text


def _maybe_number(text: str):
    """A number when the text is one, the text otherwise."""
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() else value


#: Fields BIDS types as arrays. A spreadsheet cell is one string, so these
#: are split rather than stored whole.
_LIST_FIELDS: frozenset[str] = frozenset(PET_LIST_TO_BIDS)

#: Column names for the vertical shape: one row per field, value beside it.
_FIELD_ALIASES: tuple[str, ...] = ("field", "key", "name", "parameter")
_VALUE_ALIASES: tuple[str, ...] = ("value", "answer")


def _read_vertical(df, columns: dict, path: Path):
    """One row per FIELD, keyed by BIDS name. ``None`` if that is not it.

    Returns a single spec applying to every PET run, which is what a sheet
    with no scan identifier can honestly mean.
    """
    field_col = next((columns[a] for a in _FIELD_ALIASES if a in columns), None)
    value_col = next((columns[a] for a in _VALUE_ALIASES if a in columns), None)
    if field_col is None or value_col is None:
        return None

    by_bids = {
        bids.lower(): field
        for field, bids in {**PET_SCALAR_TO_BIDS, **PET_LIST_TO_BIDS}.items()
    }
    fields: dict[str, object] = {}
    unknown: list[str] = []
    for _, row in df.iterrows():
        name = str(row.get(field_col, "")).strip()
        if not name:
            continue
        target = by_bids.get(name.lower()) or (
            name if name in PetAcquisitionSpec.model_fields else None
        )
        if target is None:
            unknown.append(name)
            continue
        value = _coerce(target, row.get(value_col, ""))
        if value is not None:
            fields[target] = value
    if unknown:
        log.warning(
            "PET spreadsheet %s: ignoring %d row(s) naming no PET field: %s",
            path.name, len(unknown), ", ".join(sorted(unknown)[:8]),
        )
    if not fields:
        return None
    log.info(
        "PET spreadsheet %s: read %d field(s) as a dataset-wide block",
        path.name, len(fields),
    )
    return PetAcquisitionSpec(**fields)


def read_pet_spreadsheet(path: Path) -> dict[str, PetAcquisitionSpec]:
    """Read a PET metadata table into per-row spec blocks.

    Returns a mapping of key to :class:`PetAcquisitionSpec`, where the key is
    whatever the identifying column held (a participant label, a BIDS name, or
    a source filename). The caller decides how to match those onto inventory
    rows, because only it knows which identifier the study used.

    An unreadable file, a missing identifier column, or a row with no usable
    values yields nothing for that row rather than an exception: a metadata
    import must never be able to abort a conversion.
    """
    path = Path(path)
    df = _read_table(path)
    if df is None or df.empty:
        return {}

    columns = {_normalise(c): c for c in df.columns}

    key_column = next(
        (columns[a] for a in _KEY_ALIASES if a in columns),
        None,
    )
    if key_column is None:
        # A sheet with no identifying column may still be the OTHER shape a
        # lab writes: one row per FIELD rather than one row per scan, with
        # the value beside it and often a column explaining what it is.
        # That is the natural way to write a study with a single dose, and
        # it is what BIDS Manager's own PET sample ships, where it silently
        # did nothing until this existed.
        vertical = _read_vertical(df, columns, path)
        if vertical is not None:
            return {"": vertical}
        log.warning(
            "PET spreadsheet %s has no identifying column (looked for %s) "
            "and no field/value pair either; ignoring the file",
            path, ", ".join(_KEY_ALIASES[:4]),
        )
        return {}

    mapped = {
        columns[norm]: field
        for norm, field in _COLUMN_ALIASES.items()
        if norm in columns
    }
    if not mapped:
        log.warning(
            "PET spreadsheet %s has no recognised metadata columns; ignoring", path,
        )
        return {}

    out: dict[str, PetAcquisitionSpec] = {}
    for _, row in df.iterrows():
        key = str(row.get(key_column, "")).strip()
        if not key:
            continue
        fields: dict[str, object] = {}
        for column, field in mapped.items():
            value = _coerce(field, row.get(column, ""))
            if value is not None:
                fields[field] = value
        if fields:
            out[key] = PetAcquisitionSpec(**fields)

    log.info(
        "PET spreadsheet %s: read %d row(s) across %d recognised column(s)",
        path.name, len(out), len(mapped),
    )
    return out


__all__ = ["read_pet_spreadsheet"]

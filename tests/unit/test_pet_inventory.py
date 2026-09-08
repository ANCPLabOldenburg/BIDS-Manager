"""PET scanning: tag extraction, vendor-string derivations, row flagging.

The derivation cases are taken verbatim from the OpenNeuroPET phantom set, so
they cover twelve real scanner models rather than invented strings. Everything
here produces a *suggestion*: a wrong parse must stay visible and correctable,
never silently written into a sidecar.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bidsmgr.cli.scan import (
    CT_COMPANION_ISSUE_TOKEN,
    _classify_pet_rows,
    _fill_pet_suggestions,
    _flag_ct_companion_rows,
)
from bidsmgr.inventory.pet import (
    PET_COLUMNS,
    becquerel_to_megabecquerel,
    derive_suggestions,
    normalise_radionuclide,
    normalise_tracer,
    parse_recon_filter,
    parse_recon_method,
)


# ---------------------------------------------------------------------------
# Radionuclide
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # DICOM caret notation, the common case across the phantom set
        ("^18^Fluorine", "F18"),
        ("^11^Carbon", "C11"),
        ("^15^Oxygen", "O15"),
        ("^68^Germanium", "Ge68"),
        # the other spellings vendors use
        ("18F", "F18"),
        ("F-18", "F18"),
        ("F18", "F18"),
        ("Fluorine-18", "F18"),
        ("Ge68", "Ge68"),
        ("", ""),
    ],
)
def test_normalise_radionuclide(raw, expected) -> None:
    assert normalise_radionuclide(raw) == expected


def test_unknown_radionuclide_passes_through() -> None:
    """A suggestion the user can see and fix beats a silently empty field."""
    assert normalise_radionuclide("Unobtainium-1") == "Unobtainium-1"


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("FDG -- fluorodeoxyglucose", "FDG"),
        ("Fluorodeoxyglucose", "FDG"),
        ("FDG", "FDG"),
        ("Flortaucipir", "FTP"),
        ("", ""),
    ],
)
def test_normalise_tracer(raw, expected) -> None:
    assert normalise_tracer(raw) == expected


def test_unmapped_tracer_is_kept() -> None:
    assert normalise_tracer("F-18-Fallypride") == "F-18-Fallypride"


# ---------------------------------------------------------------------------
# Reconstruction strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, name, values",
    [
        ("PSF+TOF 3i21s", "PSF+TOF", [3, 21]),
        ("OP-OSEM 4i21s", "OP-OSEM", [4, 21]),
        ("OSEM:i3s21", "OSEM", [3, 21]),
        # analytic reconstructions carry no iteration/subset shorthand
        ("3D Kinahan - Rogers", "3D Kinahan - Rogers", []),
        ("VPFXS", "VPFXS", []),
        ("LOR-RAMLA", "LOR-RAMLA", []),
        ("", "", []),
    ],
)
def test_parse_recon_method(raw, name, values) -> None:
    got_name, got_labels, got_values = parse_recon_method(raw)
    assert got_name == name
    assert got_values == values
    assert got_labels == (["iterations", "subsets"] if values else [])


@pytest.mark.parametrize(
    "raw, ftype, size",
    [
        (r"Rad:\rectangle\4.000000 mm\Ax:\rectangle\8.500000 mm", "rectangle", 4.0),
        ("All-pass", "All-pass", None),
        ("", "", None),
    ],
)
def test_parse_recon_filter(raw, ftype, size) -> None:
    assert parse_recon_filter(raw) == (ftype, size)


def test_becquerel_conversion() -> None:
    assert becquerel_to_megabecquerel("44400000.") == 44.4
    assert becquerel_to_megabecquerel("0") is None
    assert becquerel_to_megabecquerel("not a number") is None
    assert becquerel_to_megabecquerel(None) is None


def test_derive_suggestions_end_to_end() -> None:
    """The Siemens Biograph phantom's real tag block."""
    got = derive_suggestions({
        "tracer": "Fluorodeoxyglucose",
        "radionuclide": "^18^Fluorine",
        "injected_dose": "44400000.",
        "recon_method": "PSF+TOF 3i21s",
        "recon_filter": "All-pass",
    })
    assert got["tracer_suggestion"] == "FDG"
    assert got["radionuclide_suggestion"] == "F18"
    assert got["injected_dose_suggestion"] == "44.4 MBq"
    assert got["recon_method_suggestion"] == "PSF+TOF (3i 21s)"
    assert got["recon_filter_suggestion"] == "All-pass"


def test_derive_suggestions_on_empty_tags() -> None:
    assert derive_suggestions({}) == {}


# ---------------------------------------------------------------------------
# Row flagging
# ---------------------------------------------------------------------------


def _frame(**cols) -> pd.DataFrame:
    base = {
        "include": [1],
        "proposed_issues": [""],
        "bids_guess_skip": [False],
        "bids_guess_datatype": [""],
        "bids_guess_suffix": [""],
        "bids_guess_classifier": [""],
        "bids_guess_confidence": [0.0],
        "modality": ["unknown"],
    }
    base.update({k: [v] for k, v in cols.items()})
    return pd.DataFrame(base).astype(object)


def test_pet_row_classified_from_modality_tag() -> None:
    df = _frame(_dicom_modality="PT")
    _classify_pet_rows(df)
    assert df.at[0, "bids_guess_datatype"] == "pet"
    assert df.at[0, "bids_guess_suffix"] == "pet"
    assert df.at[0, "bids_guess_classifier"] == "dicom_modality"
    assert df.at[0, "modality"] == "pet"


def test_existing_probe_guess_wins_over_the_modality_tag() -> None:
    """dcm2niix ran and said something; the cheap fallback must not override."""
    df = _frame(_dicom_modality="PT", bids_guess_datatype="pet",
                bids_guess_classifier="dcm2niix", bids_guess_confidence=0.85)
    _classify_pet_rows(df)
    assert df.at[0, "bids_guess_classifier"] == "dcm2niix"
    assert df.at[0, "bids_guess_confidence"] == 0.85


def test_mr_row_is_untouched() -> None:
    """REGRESSION: the MR half of a PET/MR study converts as normal."""
    df = _frame(_dicom_modality="MR", bids_guess_datatype="anat",
                bids_guess_suffix="T1w")
    _classify_pet_rows(df)
    _flag_ct_companion_rows(df)
    assert df.at[0, "bids_guess_datatype"] == "anat"
    assert df.at[0, "include"] == 1


def test_ct_companion_is_excluded_with_a_reason() -> None:
    """BIDS 1.11 has no ct datatype, so the CT half has nowhere to go."""
    df = _frame(_dicom_modality="CT")
    _flag_ct_companion_rows(df)
    assert df.at[0, "include"] == 0
    assert df.at[0, "bids_guess_skip"] is True
    assert CT_COMPANION_ISSUE_TOKEN in df.at[0, "proposed_issues"]


def test_ct_flag_preserves_an_existing_issue() -> None:
    df = _frame(_dicom_modality="CT", proposed_issues="something earlier")
    _flag_ct_companion_rows(df)
    assert "something earlier" in df.at[0, "proposed_issues"]


def test_suggestions_are_created_even_when_columns_are_absent() -> None:
    """The unified backfill runs later, so the fill has to make its own columns."""
    df = _frame(_dicom_modality="PT", _pet_tags={"tracer": "FDG", "radionuclide": "18F"})
    _fill_pet_suggestions(df)
    for col in PET_COLUMNS:
        assert col in df.columns
    assert df.at[0, "tracer_suggestion"] == "FDG"
    assert df.at[0, "radionuclide_suggestion"] == "F18"


def test_rows_without_pet_tags_are_left_blank() -> None:
    df = _frame(_dicom_modality="MR", _pet_tags={})
    _fill_pet_suggestions(df)
    assert df.at[0, "tracer_suggestion"] == ""

"""PET DICOM fields dcm2niix leaves out, read through pet2bids and filtered here.

The filtering is the part under test. Their reader is good and we use it; what
it returns still needs deciding on, because it carries dcm2niix leftovers, its
own placeholders, and in one case parameter labels for a method that has no
parameters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bidsmgr.inventory.pet import parse_recon_filter
from bidsmgr.inventory.pet_dicom_meta import (
    _coherent_recon,
    _filter_fields,
    _what_is_missing,
    dicom_sidecar_fields,
    enrich_pet_sidecar,
)

PHANTOMS = Path(
    "/Users/karelo/Development/datasets/BIDS_Manager/raw_data/PET_DICOMS"
    "/PN000001/OpenNeuroPET-Phantoms/sourcedata"
)
real_pet = pytest.mark.skipif(
    os.environ.get("BIDS_MANAGER_REAL_PET_DATA") != "1" or not PHANTOMS.is_dir(),
    reason="needs BIDS_MANAGER_REAL_PET_DATA=1 and the OpenNeuroPET phantom set",
)


class _Header:
    """The little of a pydicom dataset this module touches."""

    def __init__(self, tags: dict):
        self._tags = tags

    def __getitem__(self, key):
        if key not in self._tags:
            raise KeyError(key)
        return type("Element", (), {"value": self._tags[key]})()


CONVOLUTION_KERNEL = (0x0018, 0x1210)


# --------------------------------------------------------------------------
# reconstruction parameters that describe nothing
# --------------------------------------------------------------------------


def test_labels_without_values_are_dropped() -> None:
    """The defect this rule exists for.

    Their parser handles "Filtered Back Projection" correctly and returns no
    parameters, because it has none. Given "2D Filtered Backprojection", which
    is how a real GE Advance writes it, it falls through to a default and
    returns two labels named after a unit, with no values at all.
    """
    got = _coherent_recon({
        "ReconMethodName": "2DFilteredBackprojection",
        "ReconMethodParameterLabels": ["none", "none"],
    })
    assert got == {"ReconMethodName": "2DFilteredBackprojection"}


def test_real_parameters_are_kept() -> None:
    fields = {
        "ReconMethodName": "Ordered Subset Expectation Maximization",
        "ReconMethodParameterLabels": ["subsets", "iterations"],
        "ReconMethodParameterUnits": ["none", "none"],
        "ReconMethodParameterValues": [21, 4],
    }
    assert _coherent_recon(dict(fields)) == fields


def test_labels_that_do_not_line_up_with_values_are_dropped() -> None:
    got = _coherent_recon({
        "ReconMethodParameterLabels": ["subsets", "iterations"],
        "ReconMethodParameterValues": [21],
    })
    assert "ReconMethodParameterLabels" not in got


# --------------------------------------------------------------------------
# the reconstruction filter
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "ftype", "size"),
    [
        # Siemens runs the two together with no unit.
        ("XYZGAUSSIAN3.00", "XYZGAUSSIAN", 3.0),
        # A type with no size at all.
        ("All-pass", "All-pass", None),
    ],
)
def test_a_packed_kernel_splits_into_type_and_size(kernel, ftype, size) -> None:
    assert parse_recon_filter(kernel) == (ftype, size)


def test_a_multi_valued_kernel_is_joined_before_parsing() -> None:
    """pydicom returns a MultiValue, which is a sequence but not a list.

    An isinstance check against list misses it, str() then yields the repr,
    and the kernel parses to "['hanning'".
    """
    class MultiValue(list):
        """Stands in for pydicom's own, which is likewise not a plain list."""

    header = _Header({
        CONVOLUTION_KERNEL: MultiValue(["hanning", "  4.000000 mm", " order 0"]),
    })
    assert _filter_fields(header) == {
        "ReconFilterType": "hanning", "ReconFilterSize": 4.0,
    }


def test_a_kernel_naming_two_axes_takes_the_first() -> None:
    """GE names a radial and an axial filter in one string."""
    header = _Header({
        CONVOLUTION_KERNEL: [
            "Rad:", "rectangle", "4.000000 mm", "Ax:", "rectangle", "8.500000 mm",
        ],
    })
    assert _filter_fields(header) == {
        "ReconFilterType": "rectangle", "ReconFilterSize": 4.0,
    }


@pytest.mark.parametrize("value", ["", "   ", None])
def test_an_empty_kernel_yields_nothing(value) -> None:
    assert _filter_fields(_Header({CONVOLUTION_KERNEL: value})) == {}


def test_an_absent_kernel_yields_nothing() -> None:
    assert _filter_fields(_Header({})) == {}


# --------------------------------------------------------------------------
# what we ask them to look for
# --------------------------------------------------------------------------


def test_missing_is_decided_from_the_schema() -> None:
    """Their updater only looks for what their own frozen list calls missing.

    Fields outside it were never attempted, which is why ReconFilterType stayed
    absent from our sidecars even though their code can read it.
    """
    missing = _what_is_missing({"Manufacturer": "GE"})
    assert "Manufacturer" not in missing
    assert "TimeZero" in missing
    assert missing["TimeZero"] == {"key": False, "value": False}


def test_a_placeholder_counts_as_missing() -> None:
    """A field answered "n/a" has not been answered."""
    assert "TracerName" in _what_is_missing({"TracerName": "n/a"})


# --------------------------------------------------------------------------
# never at the cost of the conversion
# --------------------------------------------------------------------------


def test_an_unreadable_dicom_adds_nothing(tmp_path: Path) -> None:
    junk = tmp_path / "not-a-dicom.dcm"
    junk.write_bytes(b"nope")
    assert dicom_sidecar_fields(junk, {}) == {}


def test_a_missing_file_adds_nothing(tmp_path: Path) -> None:
    assert dicom_sidecar_fields(tmp_path / "gone.dcm", {}) == {}


def test_enriching_an_unreadable_sidecar_changes_nothing(tmp_path: Path) -> None:
    sidecar = tmp_path / "s.json"
    sidecar.write_text("{not json", encoding="utf-8")
    assert enrich_pet_sidecar(sidecar, tmp_path / "any.dcm") == []
    assert sidecar.read_text(encoding="utf-8") == "{not json"


# --------------------------------------------------------------------------
# real phantom DICOMs
# --------------------------------------------------------------------------


def _first_dicom(folder: Path) -> Path:
    return next(f for f in sorted(folder.rglob("*")) if f.is_file())


@real_pet
def test_timezero_is_read_from_the_header(tmp_path: Path) -> None:
    """The required field our DICOM path could not supply before."""
    sidecar = tmp_path / "sub-001_pet.json"
    sidecar.write_text(json.dumps({"Manufacturer": "GE"}), encoding="utf-8")

    added = enrich_pet_sidecar(sidecar, _first_dicom(PHANTOMS / "GeneralElectricAdvance-NIMH"))
    assert "TimeZero" in added
    assert json.loads(sidecar.read_text(encoding="utf-8"))["TimeZero"]


@real_pet
def test_what_we_already_wrote_is_never_overwritten(tmp_path: Path) -> None:
    """Their pass rewrites TracerRadionuclide from F18 to 18Fluorine.

    The standard's own example for that field is "C11". Filling gaps takes
    their reading without taking their spelling.
    """
    sidecar = tmp_path / "sub-001_pet.json"
    sidecar.write_text(
        json.dumps({"TracerRadionuclide": "F18", "Units": "Bq/mL"}),
        encoding="utf-8",
    )
    enrich_pet_sidecar(sidecar, _first_dicom(PHANTOMS / "GeneralElectricAdvance-NIMH"))

    written = json.loads(sidecar.read_text(encoding="utf-8"))
    assert written["TracerRadionuclide"] == "F18"
    assert written["Units"] == "Bq/mL"


@real_pet
def test_nothing_outside_the_schema_is_added(tmp_path: Path) -> None:
    """Their pass leaves ProtocolName, SeriesDescription and ImageType behind."""
    from bidsmgr import schema as schema_mod

    declared = {f.name for f in schema_mod.sidecar_fields("pet", "pet")}
    sidecar = tmp_path / "sub-001_pet.json"
    sidecar.write_text("{}", encoding="utf-8")

    for name in enrich_pet_sidecar(sidecar, _first_dicom(PHANTOMS / "GeneralElectricAdvance-NIMH")):
        assert name in declared, f"{name} is not a field BIDS declares for PET"


@real_pet
def test_a_series_that_crashes_them_still_enriches_quietly(tmp_path: Path) -> None:
    """Their DICOM path dies on the Philips Gemini phantom, a published file.

    For them that is the whole conversion. Here it is a bonus that did not
    arrive, so the sidecar keeps everything it already had.
    """
    sidecar = tmp_path / "sub-001_pet.json"
    original = {"Manufacturer": "Philips", "Units": "Bq/mL"}
    sidecar.write_text(json.dumps(original), encoding="utf-8")

    enrich_pet_sidecar(
        sidecar, _first_dicom(PHANTOMS / "PhilipsGeminiPETMR-Unimedizin" / "reqCTAC"),
    )
    written = json.loads(sidecar.read_text(encoding="utf-8"))
    for name, value in original.items():
        assert written[name] == value

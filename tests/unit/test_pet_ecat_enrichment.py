"""ECAT sidecars, enriched by pet2bids' reader and filtered by ours.

An ECAT7 header carries roughly sixty fields. BIDS Manager parsed five of them,
so an ECAT user got a far thinner sidecar than a DICOM user for no better reason
than that nobody had written the parser. pet2bids had, and it is the reference
implementation, so we read through it.

What is ours is the filtering, and most of these tests are about that: the
schema decides what counts as a field, a blank is not an answer, a placeholder
is worse than a blank, and a value derived from an impossible timestamp is worse
than a missing one.

The isolation test is the important one. See ``_isolated_template``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bidsmgr.inventory.pet_ecat import (
    _is_answer,
    _units_from_ecat,
    ecat_sidecar_fields,
)

ECAT_DIR = Path(
    "/Users/karelo/Development/datasets/BIDS_Manager/raw_data/PET_DICOMS/"
    "PN000001/OpenNeuroPET-Phantoms/sourcedata"
)
ECAT_FILES = {
    "jhu": ECAT_DIR / "SiemensHRRT-JHU" / "Hoffman.v",
    "nru": ECAT_DIR / "SiemensHRRT-NRU" / "XCal-Hrrt-2022.04.21.15.43.05_EM_3D.v",
    "fzj": (
        ECAT_DIR / "SiemensMagnetomTrioBrainPET-FZJ"
        / "XB1BN998N-BI-01_XB298-1-Global_reco_W3_nas.v"
    ),
}
real_ecat = pytest.mark.skipif(
    os.environ.get("BIDS_MANAGER_REAL_PET_DATA") != "1"
    or not all(f.is_file() for f in ECAT_FILES.values()),
    reason="needs BIDS_MANAGER_REAL_PET_DATA=1 and the OpenNeuroPET phantom set",
)


# --------------------------------------------------------------------------
# what counts as an answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["Siemens", 0, 0.0, False, [1], {"a": 1}])
def test_a_real_value_is_an_answer(value) -> None:
    """Zero and False are answers. Only emptiness and placeholders are not."""
    assert _is_answer(value)


@pytest.mark.parametrize("value", [None, "", "   ", [], {}, ()])
def test_emptiness_is_not_an_answer(value) -> None:
    """pet2bids returns its whole template, most of it blank.

    An empty string in a sidecar reads as "we know it is nothing", which stops
    the form asking and satisfies a validator that should have complained.
    """
    assert not _is_answer(value)


@pytest.mark.parametrize("value", ["unknown", "Unknown", "n/a", "NONE", "null"])
def test_a_placeholder_is_worse_than_a_blank(value) -> None:
    assert not _is_answer(value)


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------


def test_units_are_recovered_from_a_run_on_label() -> None:
    """The real defect in the real files.

    DATA_UNITS is a fixed-width character array. A reader that does not stop at
    the terminator runs into whatever the previous write left behind, so a real
    HRRT file reads back as "Bq/mlounts/s": "Bq/ml" followed by the tail of an
    earlier "counts/s".
    """
    assert _units_from_ecat({"DATA_UNITS": "Bq/mlounts/s"}) == "Bq/mL"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Bq/ml", "Bq/mL"),      # CMIXF capitalises the litre
        ("bq/cc", "Bq/mL"),      # cc and ml are the same volume
        ("kBq/ml", "kBq/mL"),
        ("MBq/cc", "MBq/mL"),
        ("nCi/ml", "nCi/mL"),
        ("ECAT counts/sec", "counts/s"),
        ("counts/sec", "counts/s"),
    ],
)
def test_known_unit_labels_are_spelled_the_way_bids_asks(label, expected) -> None:
    assert _units_from_ecat({"DATA_UNITS": label}) == expected


@pytest.mark.parametrize("label", ["", "   ", "something nobody uses"])
def test_an_unrecognised_unit_is_left_for_the_form(label) -> None:
    """Units is required, and a units string we invented is a quantitative
    error nobody would catch. Missing is recoverable; wrong is not."""
    assert _units_from_ecat({"DATA_UNITS": label}) == ""


# --------------------------------------------------------------------------
# never at the cost of the conversion
# --------------------------------------------------------------------------


def test_an_unreadable_file_enriches_with_nothing(tmp_path: Path) -> None:
    """Enrichment is a bonus. It must never cost a user their conversion."""
    junk = tmp_path / "not-an-ecat.v"
    junk.write_bytes(b"nothing like an ECAT header")
    assert ecat_sidecar_fields(junk) == {}


def test_an_absent_dependency_enriches_with_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("pypet2bids"):
            raise ImportError("pypet2bids is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert ecat_sidecar_fields(tmp_path / "anything.v") == {}


# --------------------------------------------------------------------------
# real ECAT files
# --------------------------------------------------------------------------


@real_ecat
def test_the_sidecar_grows_well_past_what_we_parsed() -> None:
    """Our own reader gets five fields. The header holds far more."""
    fields = ecat_sidecar_fields(ECAT_FILES["jhu"])
    assert len(fields) > 12

    # The ones that were simply unavailable before.
    assert fields["Manufacturer"] == "Siemens"
    assert fields["ManufacturersModelName"] == "HRRT"
    assert fields["Units"] == "Bq/mL"
    assert fields["TimeZero"]                       # REQUIRED by BIDS
    assert fields["ReconMethodName"]
    assert fields["ReconMethodParameterValues"]


@real_ecat
def test_the_radionuclide_is_spelled_the_way_bids_spells_it() -> None:
    """They report the header's "F-18"; BIDS has a vocabulary and it says F18."""
    assert ecat_sidecar_fields(ECAT_FILES["jhu"])["TracerRadionuclide"] == "F18"


@real_ecat
def test_nothing_outside_the_schema_reaches_the_sidecar() -> None:
    """Which fields exist is a fact about BIDS, and facts have one source.

    Their template also carries the output filename, the array shape and the
    name of their own converter. None of that is metadata about the scan.
    """
    from bidsmgr import schema as schema_mod

    declared = {f.name for f in schema_mod.sidecar_fields("pet", "pet")}
    for name in ecat_sidecar_fields(ECAT_FILES["jhu"]):
        assert name in declared, f"{name} is not a field BIDS declares for PET"


@real_ecat
def test_each_file_gets_its_own_sidecar() -> None:
    """The regression this module's isolation exists for, and it is severe.

    ``Ecat.__init__`` assigns the module-level template by REFERENCE, so every
    Ecat object in a process shares one dict and populate_sidecar appends into
    it. Converting a folder of ECAT files therefore gave the second file two
    frames, the third three, and so on, while TimeZero stuck at the first
    file's value. Frame timing is what PET quantification is built on.

    This was found because three phantom scans from three sites, acquired years
    apart, all reported the same second.
    """
    read_in_one_process = {
        name: ecat_sidecar_fields(path) for name, path in ECAT_FILES.items()
    }

    # Each file has one frame, and still has one after the others were read.
    for name, fields in read_in_one_process.items():
        assert len(fields.get("FrameTimesStart", [])) == 1, name
        assert len(fields.get("FrameDuration", [])) == 1, name

    # Two scans years apart do not share a start time.
    assert (
        read_in_one_process["jhu"]["TimeZero"]
        != read_in_one_process["nru"]["TimeZero"]
    )

    # Reading one alone agrees with reading it among others.
    assert ecat_sidecar_fields(ECAT_FILES["jhu"]) == read_in_one_process["jhu"]


@real_ecat
def test_a_scan_that_started_before_the_epoch_yields_no_time() -> None:
    """One phantom file carries a negative SCAN_START_TIME.

    It renders as a plausible-looking time of day in 1936. TimeZero is what
    every frame time and every blood sample is measured against, so a wrong one
    is silently wrong for the whole study.
    """
    fields = ecat_sidecar_fields(ECAT_FILES["fzj"])
    assert "TimeZero" not in fields
    assert "ScanStart" not in fields
    assert "InjectionStart" not in fields
    # The rest of the file is still perfectly good.
    assert fields["Manufacturer"] == "Siemens"
    assert fields["Units"] == "Bq/mL"

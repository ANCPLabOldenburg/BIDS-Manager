"""PET consistency checks and the metadata spreadsheet reader.

The checks answer a question a schema validator structurally cannot: not "is
each field present and well typed" but "do the fields agree with each other".
That is arithmetic and domain knowledge, so it lives outside the schema layer.

Every finding is a warning. The spec permits all of these files; the checks
only say "this looks like a data-entry mistake". Being wrong must cost a glance,
not a blocked conversion, which is why the quiet cases below matter as much as
the noisy ones.
"""

from __future__ import annotations

import textwrap

import pytest

from bidsmgr.editor.pet_checks import check_pet_sidecar, pet_issues_for
from bidsmgr.editor.types import Severity
from bidsmgr.metadata.pet_spreadsheet import read_pet_spreadsheet


def _rules(data: dict) -> set[str]:
    return {i.rule_id.rsplit(".", 1)[-1] for i in check_pet_sidecar(data)}


# ---------------------------------------------------------------------------
# Dose plausibility
# ---------------------------------------------------------------------------


def test_dose_in_the_wrong_unit_is_flagged() -> None:
    """The classic slip: becquerels typed into a field labelled MBq."""
    assert "dose_implausible" in _rules({
        "InjectedRadioactivity": 44400000,
        "InjectedRadioactivityUnits": "MBq",
    })


@pytest.mark.parametrize(
    "dose, units",
    [(44.4, "MBq"), (1.2, "mCi"), (370.0, "MBq"), (44400000, "Bq")],
)
def test_a_plausible_dose_is_quiet(dose, units) -> None:
    assert "dose_implausible" not in _rules({
        "InjectedRadioactivity": dose,
        "InjectedRadioactivityUnits": units,
    })


def test_an_unknown_unit_is_not_second_guessed() -> None:
    """Silence beats a warning built on a unit the check cannot convert."""
    assert "dose_implausible" not in _rules({
        "InjectedRadioactivity": 1e9,
        "InjectedRadioactivityUnits": "photons",
    })


# ---------------------------------------------------------------------------
# Specific radioactivity
# ---------------------------------------------------------------------------


def test_specific_activity_that_does_not_follow_is_flagged() -> None:
    assert "specific_activity_mismatch" in _rules({
        "InjectedRadioactivity": 44.4, "InjectedRadioactivityUnits": "MBq",
        "InjectedMass": 5.0, "InjectedMassUnits": "ug",
        "SpecificRadioactivity": 1.0, "SpecificRadioactivityUnits": "Bq/g",
    })


def test_consistent_specific_activity_is_quiet() -> None:
    # 44.4 MBq over 5 ug is 8.88e12 Bq/g.
    assert "specific_activity_mismatch" not in _rules({
        "InjectedRadioactivity": 44.4, "InjectedRadioactivityUnits": "MBq",
        "InjectedMass": 5.0, "InjectedMassUnits": "ug",
        "SpecificRadioactivity": 8.88e12, "SpecificRadioactivityUnits": "Bq/g",
    })


def test_unconvertible_unit_pairs_are_left_alone() -> None:
    """A study using an exotic unit pair must not be warned about wrongly."""
    assert "specific_activity_mismatch" not in _rules({
        "InjectedRadioactivity": 44.4, "InjectedRadioactivityUnits": "MBq",
        "InjectedMass": 5.0, "InjectedMassUnits": "umol",
        "SpecificRadioactivity": 1.0, "SpecificRadioactivityUnits": "GBq/umol",
    })


# ---------------------------------------------------------------------------
# Frame timing
# ---------------------------------------------------------------------------


def test_frames_out_of_order_are_flagged() -> None:
    assert "frame_times_not_monotonic" in _rules({
        "FrameTimesStart": [0, 300, 200], "FrameDuration": [300, 300, 300],
    })


def test_mismatched_frame_counts_are_flagged() -> None:
    assert "frame_count_mismatch" in _rules({
        "FrameTimesStart": [0, 300], "FrameDuration": [300],
    })


def test_a_nonzero_start_without_time_zero_is_flagged() -> None:
    """BIDS defines frame times relative to TimeZero, so the offset is
    uninterpretable without it."""
    assert "frame_start_offset" in _rules({
        "FrameTimesStart": [967.0], "FrameDuration": [300],
    })


def test_a_nonzero_start_with_time_zero_is_quiet() -> None:
    assert "frame_start_offset" not in _rules({
        "FrameTimesStart": [967.0], "FrameDuration": [300], "TimeZero": "08:08:01",
    })


def test_normal_dynamic_framing_is_quiet() -> None:
    """Uneven frame lengths are how PET works, not an error."""
    assert _rules({
        "FrameTimesStart": [0, 10, 20, 50, 110, 410],
        "FrameDuration": [10, 10, 30, 60, 300, 300],
    }) == set()


# ---------------------------------------------------------------------------
# Timing and reconstruction
# ---------------------------------------------------------------------------


def test_a_malformed_time_zero_is_flagged() -> None:
    assert "time_zero_format" in _rules({"TimeZero": "half past ten"})


def test_a_valid_time_zero_is_quiet() -> None:
    for value in ("08:08:01", "8:08", "08:08:01.5"):
        assert "time_zero_format" not in _rules({"TimeZero": value})


def test_an_offset_without_time_zero_is_flagged() -> None:
    assert "offset_without_time_zero" in _rules({"ScanStart": 120.0})


def test_misaligned_recon_lists_are_flagged() -> None:
    """The lists are positional, so entry N of each must describe one thing."""
    assert "recon_parameter_length_mismatch" in _rules({
        "ReconMethodParameterLabels": ["iterations", "subsets"],
        "ReconMethodParameterValues": [3],
    })


def test_aligned_recon_lists_are_quiet() -> None:
    assert "recon_parameter_length_mismatch" not in _rules({
        "ReconMethodParameterLabels": ["iterations", "subsets"],
        "ReconMethodParameterValues": [3, 21],
        "ReconMethodParameterUnits": ["none", "none"],
    })


# ---------------------------------------------------------------------------
# Severity and scope
# ---------------------------------------------------------------------------


def test_every_finding_is_a_warning() -> None:
    """These are judgement calls, so they must never block a conversion."""
    found = check_pet_sidecar({
        "InjectedRadioactivity": 44400000, "InjectedRadioactivityUnits": "MBq",
        "FrameTimesStart": [0, 300, 200], "FrameDuration": [300, 300, 300],
    })
    assert found
    assert all(i.severity is Severity.WARN for i in found)


def test_only_pet_sidecars_are_checked(tmp_path) -> None:
    """REGRESSION: an MRI sidecar must not be touched."""
    bold = tmp_path / "sub-001_task-rest_bold.json"
    bold.write_text('{"InjectedRadioactivity": 44400000, "InjectedRadioactivityUnits": "MBq"}')
    assert pet_issues_for(bold) == []


def test_a_pet_sidecar_on_disk_is_checked(tmp_path) -> None:
    pet = tmp_path / "sub-001_pet.json"
    pet.write_text('{"InjectedRadioactivity": 44400000, "InjectedRadioactivityUnits": "MBq"}')
    assert pet_issues_for(pet)


def test_unreadable_json_is_survived(tmp_path) -> None:
    pet = tmp_path / "sub-001_pet.json"
    pet.write_text("{ not json")
    assert pet_issues_for(pet) == []


# ---------------------------------------------------------------------------
# Spreadsheet
# ---------------------------------------------------------------------------


def _sheet(tmp_path, body: str, name: str = "doses.tsv"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def test_a_lab_spreadsheet_is_read(tmp_path) -> None:
    """Header spellings vary by lab, so matching is loose on purpose."""
    path = _sheet(tmp_path, """\
        participant_id\tTracer\tRadionuclide\tInjected Dose\tdose_units\tMode of Administration
        sub-001\tFDG\tF18\t44.4\tMBq\tbolus
        sub-002\tPIB\tC11\t370\tMBq\tinfusion
        """)
    got = read_pet_spreadsheet(path)
    assert set(got) == {"sub-001", "sub-002"}
    assert got["sub-001"].tracer_name == "FDG"
    assert got["sub-001"].injected_radioactivity == 44.4
    assert got["sub-002"].mode_of_administration == "infusion"


@pytest.mark.parametrize(
    "header", ["InjectedRadioactivity", "injected_radioactivity", "Injected Dose", "dose"],
)
def test_header_spellings_all_land_on_one_field(tmp_path, header) -> None:
    path = _sheet(tmp_path, f"participant_id\t{header}\nsub-001\t44.4\n")
    assert read_pet_spreadsheet(path)["sub-001"].injected_radioactivity == 44.4


def test_a_bad_number_is_skipped_not_coerced(tmp_path) -> None:
    """A coerced dose looks deliberate in the sidecar, which is worse."""
    path = _sheet(tmp_path, """\
        participant_id\tTracer\tInjected Dose
        sub-001\tFDG\tnot recorded
        """)
    got = read_pet_spreadsheet(path)
    assert got["sub-001"].tracer_name == "FDG"
    assert got["sub-001"].injected_radioactivity is None


@pytest.mark.parametrize("value, expected", [("yes", True), ("TRUE", True), ("no", False), ("0", False)])
def test_boolean_spellings(tmp_path, value, expected) -> None:
    path = _sheet(tmp_path, f"participant_id\timage_decay_corrected\nsub-001\t{value}\n")
    assert read_pet_spreadsheet(path)["sub-001"].image_decay_corrected is expected


def test_na_cells_are_treated_as_unset(tmp_path) -> None:
    path = _sheet(tmp_path, """\
        participant_id\tTracer\tInjected Dose
        sub-001\tn/a\t44.4
        """)
    assert read_pet_spreadsheet(path)["sub-001"].tracer_name is None


def test_a_row_with_nothing_usable_is_dropped(tmp_path) -> None:
    path = _sheet(tmp_path, """\
        participant_id\tTracer\tInjected Dose
        sub-001\tFDG\t44.4
        sub-002\t\t
        """)
    assert set(read_pet_spreadsheet(path)) == {"sub-001"}


def test_no_identifying_column_is_reported_not_guessed(tmp_path) -> None:
    path = _sheet(tmp_path, "Tracer\tInjected Dose\nFDG\t44.4\n")
    assert read_pet_spreadsheet(path) == {}


def test_no_recognised_columns_yields_nothing(tmp_path) -> None:
    path = _sheet(tmp_path, "participant_id\tfavourite_colour\nsub-001\tblue\n")
    assert read_pet_spreadsheet(path) == {}


def test_a_missing_file_is_survived(tmp_path) -> None:
    assert read_pet_spreadsheet(tmp_path / "nope.tsv") == {}


def test_csv_is_read_too(tmp_path) -> None:
    path = _sheet(tmp_path, "participant_id,Tracer,Injected Dose\nsub-001,FDG,44.4\n", "d.csv")
    assert read_pet_spreadsheet(path)["sub-001"].tracer_name == "FDG"

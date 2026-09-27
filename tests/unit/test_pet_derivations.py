"""The three things the conversion now derives for a PET sidecar, and why.

Each of these exists because a value we could compute was being left for the
user, or because a value we were given was wrong. The tests say which.
"""

from __future__ import annotations

import pytest

from bidsmgr.editor.pet_checks import check_pet_sidecar, specific_activity_bq_per_g
from bidsmgr.fixups.sidecar_schema import drop_stringified_nulls
from bidsmgr.fixups.pet_sidecar import (
    _derive_radio_inputs,
    _derive_time_zero,
    _normalise_enums,
)


class TestTimeZero:
    """BIDS REQUIRES it, and dcm2niix deliberately stopped deriving it."""

    def test_from_the_formatted_series_time(self):
        """v1.0.20260416 emits SeriesTime for the downstream tool to use."""
        data = {"SeriesTime": "15:51:04"}
        _derive_time_zero(data)
        assert data["TimeZero"] == "15:51:04"

    def test_from_a_raw_dicom_tm_string(self):
        """A sidecar from another tool may carry the unseparated DICOM form.

        This is the shape that makes a DATE parser raise or, worse, return
        midnight. Ours reads it.
        """
        data = {"SeriesTime": "155104"}
        _derive_time_zero(data)
        assert data["TimeZero"] == "15:51:04"

    def test_series_time_is_preferred_over_acquisition_time(self):
        """The intuitive answer is the wrong one, so this is locked in.

        dcm2niix computes ``FrameTimesStart`` as
        ``AcquisitionTime - SeriesTime``, so the frame times are measured
        FROM ``SeriesTime``. Verified on 32 real phantom series where
        ``FrameTimesStart[0]`` equals that difference to the millisecond.
        Taking the more specific-sounding ``AcquisitionTime`` as time zero
        puts every frame out by the gap, which on that data ran from 16
        seconds to 32 minutes.
        """
        data = {"AcquisitionTime": "11:40:24.016", "SeriesTime": "11:12:06",
                "FrameTimesStart": [1698.02]}
        _derive_time_zero(data)
        assert data["TimeZero"] == "11:12:06"

    def test_acquisition_time_is_the_fallback(self):
        """For a sidecar from a tool that writes no SeriesTime."""
        data = {"AcquisitionTime": "125524.016"}
        _derive_time_zero(data)
        assert data["TimeZero"] == "12:55:24"

    def test_scan_start_is_asserted_with_it(self):
        """Our convention is that time zero IS the scan start."""
        data = {"SeriesTime": "15:51:04"}
        _derive_time_zero(data)
        assert data["ScanStart"] == 0

    def test_a_value_measured_from_the_wrong_clock_is_corrected(self):
        """pypet2bids writes TimeZero from AcquisitionTime when it can.

        That is the wrong reference, and the result is a well-formed,
        plausible clock time no validator would question. On this lab's
        phantom data it put six conversions out by 16 seconds to 32
        minutes. SeriesTime is authoritative when present.
        """
        data = {"TimeZero": "11:40:24", "SeriesTime": "11:12:06"}
        _derive_time_zero(data)
        assert data["TimeZero"] == "11:12:06"

    def test_a_value_with_no_series_time_to_check_it_is_left_alone(self):
        data = {"TimeZero": "09:25:32"}
        _derive_time_zero(data)
        assert data["TimeZero"] == "09:25:32"
        assert "ScanStart" not in data

    def test_the_user_still_wins(self):
        """The spec overlay runs AFTER this, so a form value is never lost.

        Guards the ordering in ``_apply_sidecar`` rather than this function.
        """
        import inspect

        from bidsmgr.fixups import pet_sidecar

        body = inspect.getsource(pet_sidecar._apply_sidecar)
        assert body.index("_derive_time_zero") < body.index("PET_SCALAR_TO_BIDS")

    def test_midnight_is_re_derived_not_trusted(self):
        """``00:00:00`` is the specific wrong answer a date parser returns.

        It is well-formed, so no validator questions it, and it is
        indistinguishable by shape from a real answer. The header is asked
        again rather than believed.
        """
        data = {"TimeZero": "00:00:00", "SeriesTime": "092532"}
        _derive_time_zero(data)
        assert data["TimeZero"] == "09:25:32"

    def test_a_genuine_midnight_survives(self):
        """A scan really started at midnight. Nothing to correct."""
        data = {"TimeZero": "00:00:00", "SeriesTime": "00:00:00"}
        _derive_time_zero(data)
        assert data["TimeZero"] == "00:00:00"

    def test_nothing_to_go_on_invents_nothing(self):
        data: dict = {}
        _derive_time_zero(data)
        assert "TimeZero" not in data


class TestSpecificRadioactivity:
    """BIDS-REQUIRED, and derivable from two values we usually have."""

    DOSE = {"InjectedRadioactivity": 81.24, "InjectedRadioactivityUnits": "MBq",
            "InjectedMass": 5.0, "InjectedMassUnits": "ug"}

    def test_derived_from_dose_and_mass(self):
        data = dict(self.DOSE)
        _derive_radio_inputs(data)
        assert data["SpecificRadioactivity"] == pytest.approx(1.6248e13, rel=1e-6)
        assert data["SpecificRadioactivityUnits"] == "Bq/g"

    def test_the_value_pet2bids_would_give_is_off_by_1e12(self):
        """Their ``(MBq * 1e6) / (ug * 1e6)`` treats ug to g as x1e6.

        Recorded as a test so nobody 'simplifies' our formula into theirs.
        [18F]FDG specific activity is of order 1e11 to 1e14 Bq/g; 16 Bq/g is
        not a tracer.
        """
        theirs = (81.24 * 10**6) / (5.0 * 10**6)
        ours = specific_activity_bq_per_g(81.24, "MBq", 5.0, "ug")
        assert ours / theirs == pytest.approx(1e12, rel=1e-6)

    def test_our_checker_accepts_our_own_derivation(self):
        """A derivation and a check that disagree is worse than neither."""
        data = dict(self.DOSE)
        _derive_radio_inputs(data)
        assert check_pet_sidecar(data) == []

    def test_our_checker_flags_theirs(self):
        data = dict(self.DOSE, SpecificRadioactivity=16.248,
                    SpecificRadioactivityUnits="Bq/g")
        rules = [i.rule_id for i in check_pet_sidecar(data)]
        assert "bidsmgr.pet.specific_activity_mismatch" in rules

    def test_a_stated_value_is_never_overwritten(self):
        data = dict(self.DOSE, SpecificRadioactivity=9.9e12)
        _derive_radio_inputs(data)
        assert data["SpecificRadioactivity"] == 9.9e12

    @pytest.mark.parametrize("missing", ["InjectedRadioactivity", "InjectedMass"])
    def test_two_values_are_needed(self, missing):
        data = {k: v for k, v in self.DOSE.items() if k != missing}
        _derive_radio_inputs(data)
        assert "SpecificRadioactivity" not in data

    def test_an_unknown_unit_pair_derives_nothing(self):
        """Left alone rather than guessed at."""
        data = dict(self.DOSE, InjectedMassUnits="umol")
        _derive_radio_inputs(data)
        assert "SpecificRadioactivity" not in data

    def test_a_non_positive_mass_derives_nothing(self):
        assert specific_activity_bq_per_g(81.24, "MBq", 0.0, "ug") is None


class TestModeOfAdministration:
    """BIDS does NOT define a closed vocabulary here, so this is careful."""

    @pytest.mark.parametrize("raw,want", [
        ("Bolus", "bolus"),
        ("BOLUS-INFUSION", "bolus-infusion"),
        ("Infusion", "infusion"),
    ])
    def test_a_documented_spelling_is_normalised(self, raw, want):
        data = {"ModeOfAdministration": raw}
        _normalise_enums(data)
        assert data["ModeOfAdministration"] == want

    def test_free_text_survives_untouched(self):
        """The schema types this as a plain string; lower-casing it blindly
        would be vandalism."""
        text = "Bolus then infusion over 60 min"
        data = {"ModeOfAdministration": text}
        _normalise_enums(data)
        assert data["ModeOfAdministration"] == text


class TestStringifiedNulls:
    """Dataset-wide, not PET-only: every modality gets these from dcm2niix."""

    def test_the_literal_string_none_is_dropped(self):
        """dcm2niix v1.0.20260724 writes "None" where the DICOM says nothing.

        ``InstitutionalDepartmentName`` is a real BIDS field, so an absent
        value and the four-character string ``None`` are different claims:
        the second says the department is called None. It also reaches the
        metadata form as an ANSWER, which stops the form asking.
        """
        data = {"InstitutionalDepartmentName": "None", "Manufacturer": "Siemens",
                "MatrixCoilMode": "None"}
        assert drop_stringified_nulls(data) == 2
        assert data == {"Manufacturer": "Siemens"}

    def test_a_real_value_survives(self):
        data = {"InstitutionalDepartmentName": "Neurology"}
        assert drop_stringified_nulls(data) == 0
        assert data == {"InstitutionalDepartmentName": "Neurology"}

    def test_the_preview_does_not_offer_it_as_an_answer(self):
        from bidsmgr.metadata.converter_preview import _is_answer

        assert _is_answer("None") is False
        assert _is_answer("Neurology") is True

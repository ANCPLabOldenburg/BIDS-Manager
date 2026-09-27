"""Four producers spell an acquisition time four ways, and all four mean it.

``parse_time_seconds`` used to parse only the DICOM TM form and read the
others by taking their leading characters as hours, minutes and seconds.
That is not a failure to parse, it is a wrong answer that looks right:
``2026-09-17T09:33:59`` came back as 20:26:00 and every caller believed it.

The two defects these lock down, both found on
``raw_data/MRI/SIna_data/OL_4925``:

* dcm2niix writes ``AcquisitionTime`` as ``09:33:59.365000``, with colons.
  The old parser returned ``None`` for it, so no functional run had a time
  and the IntendedFor fixup fell back to pairing every fieldmap with every
  run instead of with the runs that followed it.
* dcm2niix leaves the seconds UNPADDED below ten: ``10:17:1.572500``.
  Requiring two digits dropped that one run, and one untimed run is enough
  to trigger the same fallback.
"""

from __future__ import annotations

import pytest

from bidsmgr.inventory._time import parse_time_seconds


class TestEveryFormAgrees:
    """09:33:59.365 is 34439.365 seconds however it was written."""

    EXPECTED = 9 * 3600 + 33 * 60 + 59 + 0.365

    @pytest.mark.parametrize("text", [
        "093359.365000",                 # DICOM TM, from the scanner
        "20260917093359.365000",         # DICOM DT, Siemens XA
        "09:33:59.365000",               # ISO time, dcm2niix
        "2026-09-17T09:33:59.365000",    # ISO date-time, dcm2niix
    ])
    def test_form(self, text):
        assert parse_time_seconds(text) == pytest.approx(self.EXPECTED)


class TestUnpaddedComponents:
    """dcm2niix drops the leading zero on a component below ten."""

    def test_unpadded_seconds(self):
        assert parse_time_seconds("10:17:1.572500") == pytest.approx(
            10 * 3600 + 17 * 60 + 1 + 0.5725
        )

    def test_unpadded_seconds_in_a_date_time(self):
        assert parse_time_seconds("2026-09-17T10:17:1.5") == pytest.approx(
            10 * 3600 + 17 * 60 + 1.5
        )

    def test_unpadded_everything(self):
        assert parse_time_seconds("9:5:3") == pytest.approx(9 * 3600 + 5 * 60 + 3)


class TestARefusalBeatsAGuess:
    """The old parser's real sin was answering when it should not."""

    @pytest.mark.parametrize("text", [
        None, "", "   ",
        "nonsense",
        "0931",              # truncated TM: DICOM allows it, we will not guess
        "2026-09-17",        # a date with no time
        "25:00:00",          # not a time of day
        "10:61:00",
        "12345",
    ])
    def test_returns_none(self, text):
        assert parse_time_seconds(text) is None

    def test_a_date_time_is_never_read_as_a_time(self):
        """The specific old defect: ``2026-09-17T09:33:59`` read as 20:26:00."""
        wrong = 20 * 3600 + 26 * 60
        assert parse_time_seconds("2026-09-17T09:33:59") != pytest.approx(wrong)


class TestOrdering:
    """What every caller actually uses it for."""

    def test_mixed_forms_sort_correctly(self):
        times = [
            ("fmap 09:31", "093131.650000"),
            ("bold 09:33", "09:33:59.365000"),
            ("fmap 09:54", "2026-09-17T09:54:53.052500"),
            ("bold 10:17", "10:17:1.572500"),
        ]
        ordered = sorted(times, key=lambda item: parse_time_seconds(item[1]))
        assert [label for label, _ in ordered] == [
            "fmap 09:31", "bold 09:33", "fmap 09:54", "bold 10:17",
        ]

    def test_midnight_and_the_last_second_of_the_day(self):
        assert parse_time_seconds("000000.000000") == 0.0
        assert parse_time_seconds("23:59:59") == pytest.approx(86399)

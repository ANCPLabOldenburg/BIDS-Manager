"""Unit tests for ``classifier.dcm2niix_bidsguess.parse_bids_guess``."""

from __future__ import annotations

import pytest

from bidsmgr.classifier.dcm2niix_bidsguess import parse_bids_guess


def test_parses_anat_t1w_strips_run():
    """``run-N`` from BidsGuess is DICOM SeriesNumber, not BIDS-semantic — drop it."""
    datatype, entities, suffix = parse_bids_guess(["anat", "_acq-tfl3p2_run-6_T1w"])
    assert datatype == "anat"
    assert suffix == "T1w"
    assert entities == {"acquisition": "tfl3p2"}
    assert "run" not in entities


def test_parses_func_bold_strips_run():
    datatype, entities, suffix = parse_bids_guess(["func", "_acq-epfid2p2_dir-AP_run-9_bold"])
    assert datatype == "func"
    assert suffix == "bold"
    assert entities == {"acquisition": "epfid2p2", "direction": "AP"}


def test_parses_fmap_phasediff():
    datatype, entities, suffix = parse_bids_guess(["fmap", "_acq-fm2_phasediff"])
    assert datatype == "fmap"
    assert suffix == "phasediff"
    assert entities == {"acquisition": "fm2"}


def test_parses_discard_localizer_strips_run():
    datatype, entities, suffix = parse_bids_guess(["discard", "_acq-fl2_run-1_localizer"])
    assert datatype == "discard"
    assert suffix == "localizer"
    assert entities == {"acquisition": "fl2"}


def test_rejects_malformed():
    with pytest.raises(ValueError):
        parse_bids_guess([])
    with pytest.raises(ValueError):
        parse_bids_guess(["anat"])


class TestTheSidecarIsJoinedToTheRightRow:
    """dcm2niix does not always write a usable ``SeriesInstanceUID``.

    Measured 2026-09-26 on a Siemens study whose spectroscopy is stored under
    the STANDARD MR Spectroscopy Storage SOP class: the sidecar's
    ``SeriesInstanceUID`` came out as ``133347.357000``, which is the series
    TIME. The image series in the same folder got a correct UID, and a second
    study whose spectroscopy uses the Siemens private CSA class got one too.
    Joining on that field alone meant both spectroscopy series matched no
    inventory row, got no BidsGuess, and the scan reported them with no
    datatype at all: the tool "found nothing".
    """

    def test_a_real_uid_is_recognised(self):
        from bidsmgr.classifier.dcm2niix_bidsguess import looks_like_uid

        assert looks_like_uid(
            "1.3.12.2.1107.5.2.61.237021.2026060513392604620209355.0.0.0"
        )

    def test_the_series_time_is_not(self):
        from bidsmgr.classifier.dcm2niix_bidsguess import looks_like_uid

        assert not looks_like_uid("133347.357000")

    @pytest.mark.parametrize("value", ["", None, "svs_se", "1.2", "1.2.a"])
    def test_nothing_else_passes_either(self, value):
        from bidsmgr.classifier.dcm2niix_bidsguess import looks_like_uid

        assert not looks_like_uid(value)


class TestSpectroscopySuffixSurvivesTheSchema:
    """``mrs`` is a BIDS 1.11 datatype and ``svs`` one of its suffixes, so a
    BidsGuess of ``['mrs', '_svs']`` must survive parsing, canonicalisation
    and validation intact. It did; the join was what failed."""

    def test_it_parses_canonicalises_and_validates(self):
        from bidsmgr.classifier.dcm2niix_bidsguess import (
            _validate_classification, canonicalise, parse_bids_guess,
        )

        datatype, entities, suffix = parse_bids_guess(["mrs", "_svs"])
        assert (datatype, suffix) == ("mrs", "svs")
        assert canonicalise(datatype, suffix) == ("mrs", "svs")
        assert _validate_classification(datatype, suffix, entities)

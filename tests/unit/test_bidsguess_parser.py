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

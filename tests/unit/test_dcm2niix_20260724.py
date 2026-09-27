"""What changed when dcm2niix went from v1.0.20260416 to v1.0.20260724.

Two behaviours in this tool had to move with it, and both are the kind that
would otherwise have been noticed months later by a user missing files.
"""

from __future__ import annotations

import pandas as pd

from bidsmgr.cli.scan import _flag_nonimage_rows, _is_spectroscopy_row


def _row(**kwargs) -> pd.DataFrame:
    base = {
        "_has_pixel_data": False,
        "bids_guess_datatype": "",
        "bids_guess_suffix": "",
        "bids_guess_skip": False,
        "sequence_kind": "",
        "bids_name": "",
        "include": 1,
        "issues": "",
    }
    base.update(kwargs)
    return pd.DataFrame([base])


class TestSpectroscopyIsNoLongerANonImage:
    """dcm2niix writes NIfTI-MRS from v1.0.20260724.

    The non-image rule's stated reason is that dcm2niix cannot turn the
    series into a NIfTI. For spectroscopy that stopped being true, and a row
    the converter can handle must not stay excluded by a rule whose premise
    has expired. Verified on real data: two `_svs` series in the lab's own
    MRI set produce 32 KB NIfTI-MRS volumes carrying `SpectrometerFrequency`
    and `ResonantNucleus`.
    """

    def test_an_mrs_row_is_recognised(self):
        df = _row(bids_guess_datatype="mrs", bids_guess_suffix="svs")
        assert _is_spectroscopy_row(df, df.index[0]) is True

    def test_an_mrs_row_is_not_excluded(self):
        df = _row(bids_guess_datatype="mrs", bids_guess_suffix="svs")
        _flag_nonimage_rows(df)
        assert df.at[0, "include"] == 1
        assert df.at[0, "issues"] == ""

    def test_a_tensor_map_is_still_excluded(self):
        """The rule still does its job: a TENSOR map really is unconvertible."""
        df = _row(bids_guess_datatype="dwi", bids_guess_suffix="TENSOR")
        _flag_nonimage_rows(df)
        assert df.at[0, "include"] == 0
        assert "non-image series" in df.at[0, "issues"]

    def test_physio_is_still_exempt(self):
        df = _row(bids_guess_suffix="physio")
        _flag_nonimage_rows(df)
        assert df.at[0, "include"] == 1

    def test_the_series_description_alone_does_not_exempt(self):
        """'svs' appears in protocol names that are not spectroscopy, and the
        DATATYPE is what decides where a file lands."""
        df = _row(bids_guess_datatype="anat", bids_name="sub-01_acq-svs_T1w")
        assert _is_spectroscopy_row(df, df.index[0]) is False


class TestDiscardDoesNotPreemptAClassifier:
    """v1.0.20260724 renamed the DWI-derivative label ``derived`` to ``discard``.

    ``derived`` was never a schema datatype, so it was rejected and the row
    fell through to ``sequence_dict``, which knows ``FA`` / ``colFA`` /
    ``trace`` are the raw ``dwi/`` suffixes BIDS 1.11 defines. ``discard`` IS
    accepted as a no-emit decision, so taking it at face value silently
    stopped converting eight real files. A recommendation with no positive
    content must not outrank a classifier that can name the series.
    """

    def test_a_discard_row_still_reaches_the_fallback(self, monkeypatch):
        from pathlib import Path

        from bidsmgr.classifier import dcm2niix_bidsguess, sequence_dict
        from bidsmgr.classifier.types import Classification
        from bidsmgr.cli import scan as scan_mod
        from bidsmgr.inventory.types import InventoryRow

        row = InventoryRow(
            source=Path("x/FA.dcm"), series_description="ep2d_diff_FA", n_files=10,
        )
        rid = row.row_id

        monkeypatch.setattr(
            dcm2niix_bidsguess, "classify",
            lambda rows, **kw: [Classification(
                row_id=rid, classifier="dcm2niix_bidsguess",
                datatype="discard", suffix="FA", candidate_entities={},
                confidence=0.0, rationale="test", skip=True,
            )],
        )
        monkeypatch.setattr(
            sequence_dict, "classify",
            lambda rows, **kw: [Classification(
                row_id=rid, classifier="sequence_dict",
                datatype="dwi", suffix="FA", candidate_entities={},
                confidence=0.45, rationale="test", skip=False,
            )],
        )
        chosen = scan_mod._run_classifier_chain([row])
        assert chosen[rid.hex].datatype == "dwi"
        assert chosen[rid.hex].suffix == "FA"
        assert chosen[rid.hex].skip is False

    def test_a_discard_row_nothing_else_claims_keeps_the_recommendation(
        self, monkeypatch,
    ):
        from pathlib import Path

        from bidsmgr.classifier import dcm2niix_bidsguess, sequence_dict
        from bidsmgr.classifier.types import Classification
        from bidsmgr.cli import scan as scan_mod
        from bidsmgr.inventory.types import InventoryRow

        row = InventoryRow(
            source=Path("x/report.dcm"),
            series_description="PhoenixZIPReport", n_files=1,
        )
        rid = row.row_id
        monkeypatch.setattr(
            dcm2niix_bidsguess, "classify",
            lambda rows, **kw: [Classification(
                row_id=rid, classifier="dcm2niix_bidsguess",
                datatype="discard", suffix="report", candidate_entities={},
                confidence=0.0, rationale="test", skip=True,
            )],
        )
        monkeypatch.setattr(sequence_dict, "classify", lambda rows, **kw: [])
        chosen = scan_mod._run_classifier_chain([row])
        assert chosen[rid.hex].skip is True

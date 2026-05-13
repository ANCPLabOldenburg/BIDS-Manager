"""Unit tests for DWI scanner-derivative + B0-reference + fmap detection.

Three concerns covered here:

1. ``sequence_dict.detect_dwi_derivative`` correctly maps Siemens-style
   sequence names (``..._FA``, ``..._ADC``, ``..._TRACEW``, ``..._ColFA``,
   ``..._ExpADC``, ``..._TENSOR``) to BIDS suffixes.
2. ``sequence_dict.looks_like_b0_reference`` recognises common B0
   reference markers and ``cli.scan._reroute_b0_references_to_fmap_epi``
   reroutes small B0-marker DWI rows to ``fmap/epi`` when there's a
   substantially-larger DWI peer in the same session.
3. The classifier chain emits the correct (datatype, suffix) for these
   sequences end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from bidsmgr.classifier.sequence_dict import (
    classify,
    detect_dwi_derivative,
    looks_like_b0_reference,
)
from bidsmgr.classifier.types import Classification
from bidsmgr.cli.scan import _reroute_b0_references_to_fmap_epi
from bidsmgr.inventory.types import InventoryRow


# ---------------------------------------------------------------------------
# detect_dwi_derivative
# ---------------------------------------------------------------------------


def test_dwi_fa_suffix():
    assert detect_dwi_derivative("ses-pre_dir-ap_dwi_FA") == ("FA", "dwi")
    assert detect_dwi_derivative("ep2d_diff_mddw_20_p2_FA") == ("FA", "dwi")


def test_dwi_adc_suffix():
    assert detect_dwi_derivative("ses-pre_dir-ap_dwi_ADC") == ("ADC", "dwi")


def test_dwi_tracew_maps_to_trace():
    assert detect_dwi_derivative("ses-pre_dir-ap_dwi_TRACEW") == ("trace", "dwi")
    assert detect_dwi_derivative("DTI_TRACE") == ("trace", "dwi")


def test_dwi_colfa_suffix():
    assert detect_dwi_derivative("DTI_RL_ColFA") == ("colFA", "dwi")


def test_dwi_expadc_suffix():
    assert detect_dwi_derivative("DTI_LR_ExpADC") == ("expADC", "dwi")


def test_dwi_tensor_routes_to_derivatives():
    assert detect_dwi_derivative("ses-pre_dir-ap_dwi_TENSOR") == ("TENSOR", "derivatives")


def test_no_false_positive_on_unrelated_sequences():
    """Generic anatomical / functional names with FA-like substrings must
    not mis-trigger the DWI detector."""
    assert detect_dwi_derivative("MPRAGE") is None
    assert detect_dwi_derivative("MPRAGE_FATsat") is None
    assert detect_dwi_derivative("T1w") is None
    assert detect_dwi_derivative("bold_AP_run01") is None
    assert detect_dwi_derivative("") is None


# ---------------------------------------------------------------------------
# looks_like_b0_reference
# ---------------------------------------------------------------------------


def test_b0_explicit_marker():
    assert looks_like_b0_reference("acq-15_acq-1b0_dir-ap_dwi")
    assert looks_like_b0_reference("DTI_b0")
    assert looks_like_b0_reference("ep2d_b0_map")


def test_b0_b0map_marker():
    assert looks_like_b0_reference("ep2d_b0map")


def test_b0_no_marker():
    assert not looks_like_b0_reference("ep2d_diff_mddw_20")
    assert not looks_like_b0_reference("DTI_RL")
    assert not looks_like_b0_reference("MPRAGE")


# ---------------------------------------------------------------------------
# classifier chain end-to-end
# ---------------------------------------------------------------------------


def _row(*, series_description: str, n_files: int = 50,
         subject: str = "001", session: str = "pre",
         image_type: str = "DIFFUSION") -> InventoryRow:
    return InventoryRow(
        modality="mri",
        source=Path("/tmp/x"),
        subject_hint=subject,
        session_hint=session,
        n_files=n_files,
        series_description=series_description,
        image_type=image_type,
        fine_modality="dwi",
    )


def test_classifier_emits_FA_suffix():
    rows = [_row(series_description="ep2d_diff_FA")]
    out = classify(rows)
    assert len(out) == 1
    assert out[0].datatype == "dwi"
    assert out[0].suffix == "FA"


def test_classifier_emits_TENSOR_routes_to_derivatives():
    rows = [_row(series_description="ep2d_diff_TENSOR")]
    out = classify(rows)
    assert len(out) == 1
    assert out[0].datatype == "derivatives"
    assert out[0].suffix == "TENSOR"


# ---------------------------------------------------------------------------
# B0 reroute (cross-row)
# ---------------------------------------------------------------------------


def _dwi_classification(row: InventoryRow) -> Classification:
    return Classification(
        row_id=row.row_id, classifier="test",
        datatype="dwi", suffix="dwi",
        candidate_entities={"direction": "AP"}, confidence=0.5,
    )


def test_b0_reroute_with_large_peer():
    """A small B0-marker DWI row sitting next to a much larger DWI peer
    in the same session is rerouted to ``fmap/epi``."""
    big_peer = _row(series_description="acq-15_dir-ap_dwi", n_files=8775)
    b0_ref = _row(series_description="acq-15_acq-1b0_dir-ap_dwi", n_files=75)
    rows = [big_peer, b0_ref]
    chosen = {r.row_id.hex: _dwi_classification(r) for r in rows}
    chosen = _reroute_b0_references_to_fmap_epi(rows, chosen)
    assert chosen[b0_ref.row_id.hex].datatype == "fmap"
    assert chosen[b0_ref.row_id.hex].suffix == "epi"
    assert chosen[big_peer.row_id.hex].datatype == "dwi"
    assert chosen[big_peer.row_id.hex].suffix == "dwi"


def test_b0_no_reroute_when_size_similar_to_peers():
    """If the b0-marker series has a similar file count to its peers, it's
    likely a real b=0-only DWI run, not a fmap reference."""
    a = _row(series_description="DTI_RL", n_files=70)
    b = _row(series_description="DTI_b0_RL", n_files=70)  # same size — keep as dwi
    rows = [a, b]
    chosen = {r.row_id.hex: _dwi_classification(r) for r in rows}
    chosen = _reroute_b0_references_to_fmap_epi(rows, chosen)
    assert chosen[b.row_id.hex].datatype == "dwi"
    assert chosen[b.row_id.hex].suffix == "dwi"


def test_b0_reroute_singleton_dwi_with_marker_NOT_rerouted():
    """Without a substantially larger peer, a b0-marker series stays as
    ``dwi/dwi``. The series might be a real b=0-only DWI run rather than
    a fmap reference — the user reviews. (Symmetric to the abort heuristic:
    we only reroute when there's clear cross-row evidence.)"""
    only = _row(series_description="ep2d_b0_map", n_files=20)
    rows = [only]
    chosen = {only.row_id.hex: _dwi_classification(only)}
    chosen = _reroute_b0_references_to_fmap_epi(rows, chosen)
    assert chosen[only.row_id.hex].datatype == "dwi"
    assert chosen[only.row_id.hex].suffix == "dwi"

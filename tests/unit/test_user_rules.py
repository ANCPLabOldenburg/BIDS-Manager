"""Tests for user-configurable scan rules: classifier hints + exclusions.

Covers the pure-data layer (``classifier/user_rules``), the chain priority
in ``cli/scan._run_classifier_chain`` (a default hint beats the regex
fallback but loses to dcm2niix BidsGuess; a ``force`` hint beats BidsGuess),
schema rejection of bad hints, and the row-level exclusion flag.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.classifier import sequence_dict, user_rules as ur
from bidsmgr.classifier.types import Classification
from bidsmgr.cli import scan
from bidsmgr.inventory.types import InventoryRow


# ---------------------------------------------------------------------------
# user_rules: matchers + JSON
# ---------------------------------------------------------------------------


def test_hint_matches_substring_and_regex() -> None:
    h_sub = ur.UserHint(patterns=("lab_t1",), datatype="anat", suffix="T1w")
    assert ur.hint_matches(h_sub, "study_LAB_T1_v2")     # case-insensitive
    assert not ur.hint_matches(h_sub, "t2_space")
    h_rx = ur.UserHint(patterns=(r"mb\d+",), datatype="func", suffix="bold", match_mode="regex")
    assert ur.hint_matches(h_rx, "rest_MB6")
    assert not ur.hint_matches(h_rx, "rest_mb")


def test_exclusion_matches_targets() -> None:
    seq_rule = ur.ExclusionRule(pattern="calib", target="sequence")
    path_rule = ur.ExclusionRule(pattern="scratch", target="path")
    assert ur.exclusion_matches(seq_rule, sequence="pre_CALIB_scan", path="")
    assert not ur.exclusion_matches(seq_rule, sequence="t1", path="scratch/x")
    assert ur.exclusion_matches(path_rule, sequence="t1", path="data/scratch/x")


def test_validate_regex() -> None:
    assert ur.validate_regex("ok.*") is None
    assert ur.validate_regex("(") is not None


def test_bad_regex_never_raises_in_matcher() -> None:
    h = ur.UserHint(patterns=("(",), datatype="anat", suffix="T1w", match_mode="regex")
    assert ur.hint_matches(h, "anything") is False  # degrades, no raise


def test_from_json_to_json_round_trip_and_tolerance() -> None:
    obj = {
        "user_hints": [
            {"patterns": ["a", "b"], "datatype": "anat", "suffix": "T1w",
             "task": "rest", "match_mode": "regex", "force": True},
            {"patterns": [], "datatype": "anat", "suffix": "T1w"},   # dropped (no patterns)
            "garbage",                                                # ignored
        ],
        "scan_exclusions": [
            {"pattern": "x", "target": "path", "match_mode": "substring"},
            {"pattern": "", "target": "sequence"},                    # dropped (no pattern)
        ],
    }
    hints, excl = ur.from_json(obj)
    assert len(hints) == 1 and len(excl) == 1
    assert hints[0].force is True and hints[0].task == "rest"
    # to_json/from_json is stable.
    again_h, again_e = ur.from_json(ur.to_json(hints, excl))
    assert again_h == hints and again_e == excl


# ---------------------------------------------------------------------------
# classify_user_hints
# ---------------------------------------------------------------------------


def _row(desc: str) -> InventoryRow:
    return InventoryRow(source=Path("x"), series_description=desc, n_files=10)


def test_classify_user_hints_confidence_and_first_match() -> None:
    rows = [_row("LABSEQ acquisition")]
    hints = [
        ur.UserHint(patterns=("labseq",), datatype="func", suffix="bold", force=False),
        ur.UserHint(patterns=("labseq",), datatype="anat", suffix="T1w", force=True),
    ]
    cls = sequence_dict.classify_user_hints(rows, hints)
    assert len(cls) == 1
    # First matching hint (document order) wins -> func/bold, non-force 0.60.
    assert (cls[0].datatype, cls[0].suffix, cls[0].classifier) == ("func", "bold", "user_hint")
    assert cls[0].confidence == pytest.approx(0.60)

    forced = sequence_dict.classify_user_hints(rows, [hints[1]])
    assert forced[0].confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Chain priority
# ---------------------------------------------------------------------------


def test_default_hint_loses_to_bidsguess(monkeypatch) -> None:
    row = _row("LABSEQ scan")
    monkeypatch.setattr(
        scan.dcm2niix_bidsguess, "classify",
        lambda rows: [Classification(row_id=row.row_id, classifier="dcm2niix_bidsguess",
                                     datatype="anat", suffix="T2w", confidence=0.85)],
    )
    hints = [ur.UserHint(patterns=("labseq",), datatype="anat", suffix="T1w", force=False)]
    chosen = scan._run_classifier_chain([row], user_hints=hints)
    c = chosen[row.row_id.hex]
    assert (c.datatype, c.suffix) == ("anat", "T2w")   # BidsGuess won


def test_force_hint_beats_bidsguess(monkeypatch) -> None:
    row = _row("LABSEQ scan")
    monkeypatch.setattr(
        scan.dcm2niix_bidsguess, "classify",
        lambda rows: [Classification(row_id=row.row_id, classifier="dcm2niix_bidsguess",
                                     datatype="anat", suffix="T2w", confidence=0.85)],
    )
    hints = [ur.UserHint(patterns=("labseq",), datatype="anat", suffix="T1w", force=True)]
    chosen = scan._run_classifier_chain([row], user_hints=hints)
    c = chosen[row.row_id.hex]
    assert (c.datatype, c.suffix, c.classifier) == ("anat", "T1w", "user_hint")


def test_default_hint_beats_regex_fallback(monkeypatch) -> None:
    # MPRAGE would classify as anat/T1w by the built-in regex; a non-force
    # hint mapping the same series to anat/T2w must win when BidsGuess is silent.
    row = _row("my_MPRAGE")
    monkeypatch.setattr(scan.dcm2niix_bidsguess, "classify", lambda rows: [])
    hints = [ur.UserHint(patterns=("mprage",), datatype="anat", suffix="T2w", force=False)]
    chosen = scan._run_classifier_chain([row], user_hints=hints)
    c = chosen[row.row_id.hex]
    assert (c.datatype, c.suffix, c.classifier) == ("anat", "T2w", "user_hint")


def test_schema_invalid_hint_is_dropped(monkeypatch) -> None:
    row = _row("my_MPRAGE")
    monkeypatch.setattr(scan.dcm2niix_bidsguess, "classify", lambda rows: [])
    # anat/bold is not a valid pair -> hint rejected; falls through to regex.
    hints = [ur.UserHint(patterns=("mprage",), datatype="anat", suffix="bold")]
    chosen = scan._run_classifier_chain([row], user_hints=hints)
    c = chosen[row.row_id.hex]
    assert c.classifier != "user_hint"
    assert (c.datatype, c.suffix) == ("anat", "T1w")   # built-in regex result


def test_derivatives_hint_rejected(monkeypatch) -> None:
    row = _row("weird_seq")
    monkeypatch.setattr(scan.dcm2niix_bidsguess, "classify", lambda rows: [])
    hints = [ur.UserHint(patterns=("weird_seq",), datatype="derivatives", suffix="TENSOR")]
    chosen = scan._user_hint_classifications([row], hints)
    assert chosen == {}   # derivatives not allowed for user hints


# ---------------------------------------------------------------------------
# Exclusions (row-level, reversible)
# ---------------------------------------------------------------------------


def _excl_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"sequence": "t1_mprage", "source_folder": "sub-01/anat", "include": 1,
         "bids_guess_skip": False, "issues": ""},
        {"sequence": "AAHead_Scout", "source_folder": "sub-01/loc", "include": 1,
         "bids_guess_skip": False, "issues": "existing note"},
        {"sequence": "rest_bold", "source_folder": "scratch/junk", "include": 1,
         "bids_guess_skip": False, "issues": ""},
    ])


def test_apply_user_exclusions_sequence_match() -> None:
    df = _excl_df()
    scan._apply_user_exclusions(df, [ur.ExclusionRule(pattern="scout", target="sequence")])
    # Row 1 (scout) excluded, others untouched.
    assert df.at[0, "include"] == 1
    assert df.at[1, "include"] == 0
    assert bool(df.at[1, "bids_guess_skip"]) is True
    assert scan.USER_EXCLUDED_ISSUE_TOKEN in df.at[1, "issues"]
    assert "existing note" in df.at[1, "issues"]   # prepended, not clobbered
    assert df.at[2, "include"] == 1
    # Reversible: no rows dropped.
    assert len(df) == 3


def test_apply_user_exclusions_path_target_and_regex() -> None:
    df = _excl_df()
    scan._apply_user_exclusions(df, [ur.ExclusionRule(pattern=r"scratch/.*", target="path", match_mode="regex")])
    assert df.at[2, "include"] == 0       # path matched
    assert df.at[0, "include"] == 1
    assert df.at[1, "include"] == 1


def test_apply_user_exclusions_empty_is_noop() -> None:
    df = _excl_df()
    scan._apply_user_exclusions(df, None)
    scan._apply_user_exclusions(df, [])
    assert list(df["include"]) == [1, 1, 1]

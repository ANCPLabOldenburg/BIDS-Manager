"""Tests for the multiple-StudyDescription scan heads-up.

``bidsmgr.cli.scan._flag_mixed_study_descriptions`` surfaces a heads-up when one
scan pooled DICOMs from more than one study. It reuses the existing severity
system: a one-line summary on the ``bidsmgr.cli.scan`` logger (CLI + GUI Log
dock) plus a non-fatal note appended to each affected row's ``issues``
so the rows read as ``warn`` (warnings chip + Issues dialog). Nothing is excluded
or rewritten.
"""

from __future__ import annotations

import logging

import pandas as pd

from bidsmgr.cli.scan import (
    MIXED_STUDY_ISSUE_TOKEN,
    _flag_mixed_study_descriptions,
)

# Tokens the GUI inventory model treats as fatal ("err"). The mixed-study note
# must avoid all of them so the row classifies as a warning, not an error.
_ERR_TOKENS = ("suspected_abort", "required", "build_basename", "missing")

LOGGER = "bidsmgr.cli.scan"


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "issues" not in df.columns:
        df["issues"] = ""
    return df


def test_flags_rows_and_logs_on_multiple_studies(caplog):
    df = _df(
        [
            {"participant_id": "sub-001", "StudyDescription": "StudyA"},
            {"participant_id": "sub-002", "StudyDescription": "StudyB"},
        ]
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _flag_mixed_study_descriptions(df)

    msgs = " ".join(_warnings(caplog))
    assert "Multiple distinct DICOM StudyDescriptions" in msgs
    assert "StudyA" in msgs and "StudyB" in msgs

    # Every affected row gets a non-fatal issues note.
    issues = df["issues"].tolist()
    assert all(MIXED_STUDY_ISSUE_TOKEN in str(v) for v in issues)
    # row 0 names its own study and points at the other.
    assert "StudyA" in issues[0] and "StudyB" in issues[0]


def test_note_is_warning_not_error(caplog):
    df = _df(
        [
            {"participant_id": "sub-001", "StudyDescription": "StudyA"},
            {"participant_id": "sub-002", "StudyDescription": "StudyB"},
        ]
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _flag_mixed_study_descriptions(df)
    # The model classifies a issues note containing any error token as
    # "err"; the mixed-study note must stay clear of all of them so the row reads
    # as "warn".
    for note in df["issues"]:
        low = str(note).lower()
        assert not any(tok in low for tok in _ERR_TOKENS)


def test_existing_issues_are_preserved(caplog):
    df = _df(
        [
            {"participant_id": "sub-001", "StudyDescription": "StudyA",
             "issues": "B0 reroute"},
            {"participant_id": "sub-002", "StudyDescription": "StudyB",
             "issues": ""},
        ]
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _flag_mixed_study_descriptions(df)
    # The prior note is kept; the mixed-study note is appended after it.
    assert df["issues"].iloc[0].startswith("B0 reroute")
    assert MIXED_STUDY_ISSUE_TOKEN in df["issues"].iloc[0]


def test_silent_on_single_study(caplog):
    df = _df(
        [
            {"participant_id": "sub-001", "StudyDescription": "OnlyStudy"},
            {"participant_id": "sub-002", "StudyDescription": "OnlyStudy"},
        ]
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _flag_mixed_study_descriptions(df)
    assert _warnings(caplog) == []
    assert (df["issues"] == "").all()


def test_silent_without_studydescription_column(caplog):
    # EEG/MEG-only inventory has no StudyDescription column.
    df = pd.DataFrame({"participant_id": ["sub-001"], "datatype": ["eeg"]})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _flag_mixed_study_descriptions(df)
    assert _warnings(caplog) == []


def test_blank_values_are_ignored(caplog):
    # Mixed EEG (blank study) + single-study MRI must not trip the heads-up.
    df = _df(
        [
            {"participant_id": "sub-001", "StudyDescription": "OneStudy"},
            {"participant_id": "sub-002", "StudyDescription": ""},
        ]
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _flag_mixed_study_descriptions(df)
    assert _warnings(caplog) == []
    assert (df["issues"] == "").all()


def test_per_subject_span_is_called_out(caplog):
    # One subject whose series span two studies is the strongest signal.
    df = _df(
        [
            {"participant_id": "sub-001", "StudyDescription": "StudyA"},
            {"participant_id": "sub-001", "StudyDescription": "StudyB"},
        ]
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _flag_mixed_study_descriptions(df)
    msgs = " ".join(_warnings(caplog))
    assert "subject sub-001 spans 2 StudyDescriptions" in msgs

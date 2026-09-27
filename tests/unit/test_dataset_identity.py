"""Tests for incremental-conversion identity comparison (Phase F item #5).

``dataset_identity`` compares a scanned subject against an existing dataset to
produce a specific collision hint: same subject (new session vs re-scan),
different subject (with a suggested free id), or unknown.
"""

from __future__ import annotations

from pathlib import Path

from bidsmgr.inventory import dataset_identity as di


def _participants(root: Path, rows: list[dict]) -> None:
    cols = ["participant_id", "patient_id", "given_name", "family_name"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in cols))
    (root / "participants.tsv").write_text("\n".join(lines) + "\n")


def test_read_existing_identities(tmp_path: Path):
    _participants(tmp_path, [
        {"participant_id": "sub-001", "patient_id": "P1", "given_name": "Ann", "family_name": "Lee"},
    ])
    ids = di.read_existing_identities(tmp_path)
    assert ids["sub-001"]["patient_id"] == "P1"
    assert ids["sub-001"]["family_name"] == "Lee"


def test_identity_match_by_patient_id_and_name():
    assert di.identity_match({"patient_id": "P1"}, {"patient_id": "P1"}) is True
    assert di.identity_match({"patient_id": "P1"}, {"patient_id": "P2"}) is False
    assert di.identity_match(
        {"given_name": "Ann", "family_name": "Lee"},
        {"given_name": "ann", "family_name": "LEE"},
    ) is True
    assert di.identity_match({}, {"patient_id": "P1"}) is None      # no existing info
    assert di.identity_match({"patient_id": ""}, {"patient_id": ""}) is None  # nothing comparable


def test_any_single_coincidence_counts_as_possible_match():
    # Technician relabelled PatientID per session, but the given name is the
    # same -> must be treated as a POSSIBLE same subject, not a different one.
    assert di.identity_match(
        {"patient_id": "P1", "given_name": "Ann"},
        {"patient_id": "P2_ses2", "given_name": "Ann"},
    ) is True
    matched, differing = di.compare_identity(
        {"patient_id": "P1", "given_name": "Ann", "family_name": "Lee"},
        {"patient_id": "P2", "given_name": "Ann", "family_name": "Lee"},
    )
    assert matched == ["given_name", "family_name"]
    assert differing == ["patient_id"]


def test_next_free_subject_id(tmp_path: Path):
    (tmp_path / "sub-001").mkdir()
    (tmp_path / "sub-003").mkdir()
    assert di.next_free_subject_id(tmp_path) == "sub-004"
    assert di.next_free_subject_id(tmp_path / "empty") == "sub-001"


def test_classify_same_subject_new_session():
    note = di.classify(
        "sub-001",
        {"patient_id": "P1"}, "ses-post",
        {"patient_id": "P1"}, {"ses-pre"}, "sub-002",
    )
    assert "same subject" in note and "new session" in note and "ses-post" in note


def test_classify_same_subject_rescan():
    note = di.classify(
        "sub-001",
        {"patient_id": "P1"}, "ses-pre",
        {"patient_id": "P1"}, {"ses-pre"}, "sub-002",
    )
    assert "re-scan" in note and "same subject" in note


def test_classify_partial_match_flags_possible_same_subject():
    # PatientID differs (relabelled per session) but given name matches: must
    # tell the user it MAY be the same subject and name the differing field.
    note = di.classify(
        "sub-001",
        {"patient_id": "P2_ses2", "given_name": "Ann"}, "ses-post",
        {"patient_id": "P1", "given_name": "Ann"}, {"ses-pre"}, "sub-002",
    )
    assert "may be the same subject" in note
    assert "given name" in note          # the coincidence is named
    assert "PatientID" in note           # the difference is named
    assert "ses-post" in note            # new-session hint included


def test_classify_different_subject_suggests_free_id():
    note = di.classify(
        "sub-001",
        {"patient_id": "P9", "given_name": "Bob"}, "",
        {"patient_id": "P1", "given_name": "Ann"}, set(), "sub-007",
    )
    assert "DIFFERENT subject" in note and "sub-007" in note


def test_classify_unknown_when_no_identity():
    note = di.classify("sub-001", {}, "", None, set(), "sub-002")
    assert "already in the dataset" in note


def test_notes_are_warnings_not_errors():
    # The model treats these issues tokens as fatal; the hints must
    # avoid all of them so the row reads as a warning.
    err_tokens = ("suspected_abort", "required", "build_basename", "missing")
    notes = [
        di.classify("sub-001", {"patient_id": "P1"}, "ses-post",
                    {"patient_id": "P1"}, {"ses-pre"}, "sub-002"),
        di.classify("sub-001", {"patient_id": "P9"}, "",
                    {"patient_id": "P1"}, set(), "sub-007"),
        di.classify("sub-001", {}, "", None, set(), "sub-002"),
        di.classify("sub-001", {"patient_id": "P2", "given_name": "Ann"}, "ses-post",
                    {"patient_id": "P1", "given_name": "Ann"}, {"ses-pre"}, "sub-002"),
    ]
    for note in notes:
        assert not any(tok in note.lower() for tok in err_tokens)

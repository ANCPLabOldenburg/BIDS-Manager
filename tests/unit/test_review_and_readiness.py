"""Reviewer decisions, readiness, and the prose behind a rule code.

Three small engines that share one property worth locking in: none of them may
change the dataset's meaning. Accepting a warning records an opinion, readiness
reads and reports, and rule help is prose. If any of them starts editing data
the guarantee the Editor makes is gone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import readiness, review, rule_help
from bidsmgr.editor.types import (
    FileVerdict,
    Issue,
    Severity,
    ValidationReport,
)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-02" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "ds", "BIDSVersion": "1.10.0",
    }))
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"EchoTime": 0.03, "InstitutionName": "TODO"})
    )
    return root


# ---------------------------------------------------------------------------
# Accepted warnings
# ---------------------------------------------------------------------------


def test_accepting_a_finding_survives_a_reload(dataset: Path) -> None:
    review.accept(
        dataset, file="sub-01/anat/sub-01_T1w.json",
        rule_id="SIDECAR_KEY_RECOMMENDED", field="InstitutionName",
        note="Scanner was decommissioned; nobody can answer this.",
    )
    loaded = review.load(dataset)
    found = review.is_accepted(
        loaded, file="sub-01/anat/sub-01_T1w.json",
        rule_id="SIDECAR_KEY_RECOMMENDED", field="InstitutionName",
    )
    assert found is not None
    assert found.note.startswith("Scanner was decommissioned")
    assert found.who        # whoever ran it, never blank
    assert found.at


def test_an_acceptance_is_about_one_file_not_a_whole_rule(
    dataset: Path,
) -> None:
    """Accepting a rule everywhere would silence files nobody looked at."""
    review.accept(
        dataset, file="sub-01/anat/sub-01_T1w.json",
        rule_id="SIDECAR_KEY_RECOMMENDED", field="InstitutionName",
    )
    loaded = review.load(dataset)
    assert review.is_accepted(
        loaded, file="sub-02/anat/sub-02_T1w.json",
        rule_id="SIDECAR_KEY_RECOMMENDED", field="InstitutionName",
    ) is None


def test_a_decision_can_be_taken_back(dataset: Path) -> None:
    review.accept(dataset, file="a.json", rule_id="R", field="F")
    assert review.unaccept(dataset, file="a.json", rule_id="R", field="F")
    assert not review.load(dataset)
    # Undoing something that was never decided is not an error.
    assert not review.unaccept(dataset, file="a.json", rule_id="R", field="F")


def test_acceptances_live_inside_the_dataset(dataset: Path) -> None:
    """The decision travels with the data, so the next reviewer sees it."""
    review.accept(dataset, file="a.json", rule_id="R")
    assert (dataset / review.ACCEPTED_FILE).exists()


def test_accepting_never_touches_the_data(dataset: Path) -> None:
    fp = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    before = fp.read_bytes()
    review.accept(
        dataset, file="sub-01/anat/sub-01_T1w.json", rule_id="R", field="F",
    )
    assert fp.read_bytes() == before


def test_a_corrupt_decision_file_is_not_fatal(dataset: Path) -> None:
    target = dataset / review.ACCEPTED_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json")
    assert review.load(dataset) == {}


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def _check(checks, name):
    return next(c for c in checks if c.name == name)


def test_a_fresh_conversion_is_not_ready_to_share(dataset: Path) -> None:
    checks = readiness.check_dataset(dataset)
    assert not _check(checks, "Authorship").passed
    assert not _check(checks, "Licence").passed
    assert not _check(checks, "README").passed
    assert not _check(checks, "Placeholders").passed   # the TODO above
    assert not _check(checks, "Participants table").passed
    assert "checks pass" in readiness.summarise(checks)


def test_filling_the_gaps_makes_it_ready(dataset: Path) -> None:
    (dataset / "dataset_description.json").write_text(json.dumps({
        "Name": "ds", "BIDSVersion": "1.10.0",
        "Authors": ["A Person"], "License": "CC0",
    }))
    (dataset / "README").write_text("x" * 400)
    (dataset / "CHANGES").write_text("1.0.0 2026-01-01\n - first release\n")
    (dataset / "participants.tsv").write_text("participant_id\nsub-01\n")
    (dataset / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"EchoTime": 0.03})
    )
    checks = readiness.check_dataset(dataset)
    assert all(c.passed for c in checks), [c.name for c in checks if not c.passed]
    assert readiness.summarise(checks) == "Ready to share."


def test_a_citation_file_counts_as_authorship(dataset: Path) -> None:
    """BIDS makes CITATION.cff and Authors mutually exclusive, so the check
    cannot demand the field."""
    assert not _check(readiness.check_dataset(dataset), "Authorship").passed
    (dataset / "CITATION.cff").write_text("cff-version: 1.2.0\n")
    assert _check(readiness.check_dataset(dataset), "Authorship").passed


def test_a_stub_readme_does_not_pass(dataset: Path) -> None:
    (dataset / "README").write_text("Converted by BIDS Manager.\n")
    check = _check(readiness.check_dataset(dataset), "README")
    assert not check.passed
    assert "stub" in check.detail


def test_one_subject_needs_no_participants_table(tmp_path: Path) -> None:
    root = tmp_path / "solo"
    (root / "sub-01" / "anat").mkdir(parents=True)
    assert _check(
        readiness.check_dataset(root), "Participants table",
    ).passed


def test_validation_is_only_reported_when_a_report_is_given(
    dataset: Path,
) -> None:
    """Running the validator to draw a checklist would be absurd."""
    names = [c.name for c in readiness.check_dataset(dataset)]
    assert "Validation" not in names

    report = ValidationReport(
        bids_root=dataset,
        files=[FileVerdict(
            path=Path("sub-01/anat/sub-01_T1w.json"), severity=Severity.ERR,
            issues=[Issue(
                severity=Severity.ERR, rule_id="X", message="broken",
            )],
        )],
    )
    check = _check(readiness.check_dataset(dataset, report), "Validation")
    assert not check.passed and "1 error" in check.detail


def test_readiness_reads_and_never_writes(dataset: Path) -> None:
    before = {p: p.stat().st_mtime_ns for p in dataset.rglob("*") if p.is_file()}
    readiness.check_dataset(dataset)
    after = {p: p.stat().st_mtime_ns for p in dataset.rglob("*") if p.is_file()}
    assert before == after


def test_every_failing_check_says_why_it_matters(dataset: Path) -> None:
    for check in readiness.check_dataset(dataset):
        assert check.why.strip(), f"{check.name} gives no reason"
        assert check.detail.strip()


# ---------------------------------------------------------------------------
# Rule help
# ---------------------------------------------------------------------------


def test_an_unknown_rule_gets_silence_not_a_guess() -> None:
    assert rule_help.explain("SOMETHING_WE_HAVE_NEVER_SEEN") is None
    assert rule_help.as_text("SOMETHING_WE_HAVE_NEVER_SEEN") == ""
    assert rule_help.explain(None) is None
    assert rule_help.explain("") is None


def test_help_covers_the_rules_users_actually_hit() -> None:
    for rule in (
        "SIDECAR_KEY_RECOMMENDED",
        "JSON_SCHEMA_VALIDATION_ERROR",
        "TSV_COLUMN_MISSING",
        "AUTHORS_AND_CITATION_FILE_MUTUALLY_EXCLUSIVE",
        "bidsmgr.todo_placeholder",
    ):
        assert rule_help.explain(rule) is not None, rule


def test_every_entry_says_what_it_means_and_what_to_do() -> None:
    for rule in rule_help.known_rules():
        meaning, action = rule_help.explain(rule)
        assert meaning.strip() and action.strip(), rule
        assert meaning.endswith("."), rule


def test_a_todo_licence_is_not_a_licence(dataset: Path) -> None:
    """The metadata step writes TODO into fields nothing could answer. A
    checklist that reads it as an answer says "ready to share" when it is
    not."""
    (dataset / "dataset_description.json").write_text(json.dumps({
        "Name": "ds", "BIDSVersion": "1.10.0",
        "License": "TODO", "Authors": ["TODO"],
    }))
    checks = readiness.check_dataset(dataset)
    licence = _check(checks, "Licence")
    assert not licence.passed
    assert "TODO" in licence.detail
    assert not _check(checks, "Authorship").passed

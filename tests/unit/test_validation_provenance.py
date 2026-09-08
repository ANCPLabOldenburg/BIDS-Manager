"""Every finding says where in the BIDS schema it comes from.

A validator that only asserts is hard to trust and hard to argue with. bidsval
records the schema rule behind each finding, so the Editor can show it and a
user can go and read what the standard actually says rather than take the
message on faith.

Findings BIDS Manager raises itself have no schema rule behind them, and show
none rather than an invented one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import validator as v
from bidsmgr.editor.types import Issue, Severity


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps(
        {"Name": "t", "BIDSVersion": "1.11.1", "DatasetType": "raw"}
    ))
    (root / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_bytes(b"x")
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"MagneticFieldStrength": 3})
    )
    return root


def test_the_issue_type_can_carry_provenance() -> None:
    assert "schema_rule" in Issue.model_fields


def test_it_defaults_to_nothing() -> None:
    """A finding with no schema rule behind it must not invent one."""
    issue = Issue(severity=Severity.WARN, rule_id="bidsmgr.todo_placeholder", message="x")
    assert issue.schema_rule is None


def test_schema_findings_carry_the_rule_that_produced_them(dataset: Path) -> None:
    report = v.validate(dataset)
    from_schema = [
        i for f in report.files for i in f.issues
        if i.rule_id.startswith(("SIDECAR_", "JSON_", "README", "NO_AUTHORS"))
    ]
    assert from_schema, "expected some schema-driven findings to check"
    for issue in from_schema:
        assert issue.schema_rule, f"{issue.rule_id} lost its provenance"
        assert issue.schema_rule.startswith("rules."), issue.schema_rule


def test_the_rule_points_where_the_finding_came_from(dataset: Path) -> None:
    """Not just any path: the one for this kind of finding."""
    report = v.validate(dataset)
    sidecar = next(
        i for f in report.files for i in f.issues
        if i.rule_id == "SIDECAR_KEY_RECOMMENDED"
    )
    assert sidecar.schema_rule.startswith("rules.sidecars.")


def test_our_own_findings_show_no_provenance(dataset: Path) -> None:
    """The TODO-placeholder check is BM's convention, not the standard's."""
    (dataset / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"MagneticFieldStrength": 3, "Manufacturer": "TODO"})
    )
    verdict = v.validate_file(dataset, dataset / "sub-01" / "anat" / "sub-01_T1w.json")
    ours = [i for i in verdict.issues if i.rule_id.startswith("bidsmgr.")]
    assert ours, "expected the TODO placeholder to be flagged"
    for issue in ours:
        assert issue.schema_rule is None


def test_it_survives_the_single_file_and_folder_passes(dataset: Path) -> None:
    """All three entry points, because the Editor uses all three."""
    target = dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"

    one = v.validate_file(dataset, target)
    assert any(i.schema_rule for i in one.issues)

    folder = v.validate_folder(dataset, dataset / "sub-01" / "anat")
    assert any(i.schema_rule for fv in folder for i in fv.issues)


def test_the_html_report_shows_it(dataset: Path) -> None:
    from bidsmgr.editor import html_report

    rendered = html_report.render_html(v.validate(dataset))
    assert 'class="provenance"' in rendered
    assert "rules.sidecars." in rendered

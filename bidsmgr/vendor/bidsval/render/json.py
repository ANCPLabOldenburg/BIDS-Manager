"""Machine-readable JSON output.

A flat, stable shape: the run's metadata and counts, then one list of every
finding (dataset-level and per-file together) with its location. Easy to parse,
diff, and feed to other tools.
"""

from __future__ import annotations

import json as _json
from typing import Any

from .. import __version__
from ..issues import Issue
from ..report import ValidationReport


def _issue_dict(issue: Issue) -> dict[str, Any]:
    out: dict[str, Any] = {
        "severity": issue.severity.value,
        "code": issue.code,
        "location": issue.location,
    }
    if issue.sub_code:
        out["subCode"] = issue.sub_code
    if issue.message:
        out["message"] = issue.message
    if issue.suggestion:
        out["suggestion"] = issue.suggestion
    if issue.rule:
        out["rule"] = issue.rule
    return out


def to_dict(report: ValidationReport) -> dict[str, Any]:
    """Return the report as a plain, JSON-serialisable dict."""
    issues = [_issue_dict(i) for i in report.dataset_issues.issues]
    for verdict in report.files:
        issues.extend(_issue_dict(i) for i in verdict.issues)
    out: dict[str, Any] = {
        "bidsvalVersion": __version__,
        "bidsVersion": report.bids_version,
        "schemaVersion": report.schema_version,
        "datasetRoot": str(report.dataset_root) if report.dataset_root else None,
        "valid": report.is_valid,
        "counts": report.counts,
        "issues": issues,
    }
    if report.derivatives:
        out["derivatives"] = {name: to_dict(deriv) for name, deriv in report.derivatives.items()}
    return out


def to_json(report: ValidationReport, *, indent: int | None = 2) -> str:
    """Return the report as a JSON string."""
    return _json.dumps(to_dict(report), indent=indent)

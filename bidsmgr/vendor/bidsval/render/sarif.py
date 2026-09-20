"""SARIF 2.1.0 output.

SARIF is the format code-scanning tools (GitHub, GitLab, IDE "Problems" panels)
read, so a SARIF report lets BIDS findings show up as inline annotations in CI
and editors. Each finding becomes a result keyed by its issue code; the codes
seen in the run are also listed as rules.
"""

from __future__ import annotations

import json as _json
from typing import Any

from .. import __version__
from ..issues import Issue, Severity
from ..report import ValidationReport

_INFORMATION_URI = "https://github.com/karellopez/bidsval"

# SARIF levels: error / warning / note / none.
_LEVEL = {Severity.ERROR: "error", Severity.WARNING: "warning", Severity.IGNORE: "note"}


def _all_issues(report: ValidationReport) -> list[Issue]:
    issues = list(report.dataset_issues.issues)
    for verdict in report.files:
        issues.extend(verdict.issues)
    return issues


def _result(issue: Issue) -> dict[str, Any]:
    text = issue.message or issue.code
    if issue.suggestion:
        text = f"{text}\n\nHow to fix: {issue.suggestion}"
    result: dict[str, Any] = {
        "ruleId": issue.code,
        "level": _LEVEL.get(issue.severity, "warning"),
        "message": {"text": text},
    }
    if issue.location:
        result["locations"] = [
            {"physicalLocation": {"artifactLocation": {"uri": issue.location}}}
        ]
    return result


def to_dict(report: ValidationReport) -> dict[str, Any]:
    """Return the report as a SARIF 2.1.0 log (a plain dict)."""
    issues = _all_issues(report)
    rules = [{"id": code} for code in sorted({i.code for i in issues})]
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "bidsval",
                        "version": __version__,
                        "informationUri": _INFORMATION_URI,
                        "rules": rules,
                    }
                },
                "results": [_result(i) for i in issues],
            }
        ],
    }


def to_sarif(report: ValidationReport, *, indent: int | None = 2) -> str:
    """Return the report as a SARIF JSON string."""
    return _json.dumps(to_dict(report), indent=indent)

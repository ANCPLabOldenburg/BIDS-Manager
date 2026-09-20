"""Plain-text summary of a validation report (the default CLI output)."""

from __future__ import annotations

from .. import __version__
from ..issues import Issue
from ..report import ValidationReport


def _line(issue: Issue) -> str:
    field = f" [{issue.sub_code}]" if issue.sub_code else ""
    message = f" - {issue.message}" if issue.message else ""
    where = issue.location or ""
    line = f"  {issue.severity.value.upper():7s} {issue.code}{field}  {where}{message}"
    if issue.suggestion:
        # The actionable hint: what the file/field/column should contain, with an
        # example, indented under the finding so it is easy to read and to fix.
        line += f"\n           how to fix: {issue.suggestion}"
    return line


def to_text(report: ValidationReport) -> str:
    """Render the report as a human-readable text summary."""
    lines = [
        f"bidsval {__version__}  schema {report.schema_version}  BIDS {report.bids_version}",
        f"{report.dataset_root}",
    ]
    findings = list(report.dataset_issues.issues)
    for verdict in report.files:
        findings.extend(verdict.issues)
    lines.extend(_line(issue) for issue in findings)
    counts = report.counts
    lines.append("")
    lines.append(f"{counts.get('error', 0)} error(s), {counts.get('warning', 0)} warning(s)")
    lines.append("VALID" if report.is_valid else "INVALID")
    for name, deriv in report.derivatives.items():
        verdict = "VALID" if deriv.is_valid else "INVALID"
        errs = deriv.counts.get("error", 0)
        warns = deriv.counts.get("warning", 0)
        lines.append(f"  derivatives/{name}: {verdict} ({errs} error(s), {warns} warning(s))")
    return "\n".join(lines)

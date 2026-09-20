"""The top-level validation result.

A :class:`ValidationReport` is what a full run returns: the schema it ran
against, the findings (dataset-wide and per file), a rolled-up severity, and
counts. It is pure data, so it serialises cleanly and binds directly to a CLI
summary, a GUI, or a machine-readable report.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .issues import DatasetIssues, Issue, Severity


class FileVerdict(BaseModel):
    """The outcome for one file in the dataset.

    ``severity`` is the rollup of this file's own findings (``None`` means the
    file is clean). Kept per-file so a tree view or editor can show a status
    next to each path without re-scanning the whole report.
    """

    path: Path  # relative to the dataset root
    severity: Severity | None = None
    issues: list[Issue] = Field(default_factory=list)

    def recompute_severity(self) -> None:
        """Set :attr:`severity` from the file's current issues."""
        self.severity = _highest([issue.severity for issue in self.issues])


class ValidationReport(BaseModel):
    """The result of validating a dataset."""

    dataset_root: Path | None = None
    # Both BIDS version axes are recorded so a report is unambiguous about what
    # it was checked against (the spec version and the schema's own version).
    bids_version: str = ""
    schema_version: str = ""
    severity: Severity | None = None  # dataset-wide rollup; None == clean
    counts: dict[str, int] = Field(default_factory=lambda: {"error": 0, "warning": 0, "ignore": 0})
    dataset_issues: DatasetIssues = Field(default_factory=DatasetIssues)
    files: list[FileVerdict] = Field(default_factory=list)
    # Nested BIDS derivative datasets, validated on their own (only when recursive).
    derivatives: dict[str, ValidationReport] = Field(default_factory=dict)

    def recompute(self) -> None:
        """Refresh :attr:`severity` and :attr:`counts` from current findings.

        Call after all findings have been added. Counts every finding once,
        whether it sits on a file or at the dataset level.
        """
        all_severities: list[Severity] = [issue.severity for issue in self.dataset_issues.issues]
        for verdict in self.files:
            all_severities.extend(issue.severity for issue in verdict.issues)
        self.severity = _highest(all_severities)
        self.counts = {
            "error": sum(s is Severity.ERROR for s in all_severities),
            "warning": sum(s is Severity.WARNING for s in all_severities),
            "ignore": sum(s is Severity.IGNORE for s in all_severities),
        }

    @property
    def is_valid(self) -> bool:
        """True when the dataset has no error-level findings."""
        return self.counts.get("error", 0) == 0

    def filtered(self, severities: set[Severity]) -> ValidationReport:
        """A copy keeping only findings whose severity is in ``severities``.

        Used to show only selected requirement levels; the original report's
        validity/exit status is unaffected (validity always depends on errors).
        """
        kept = ValidationReport(
            dataset_root=self.dataset_root,
            bids_version=self.bids_version,
            schema_version=self.schema_version,
        )
        kept.dataset_issues = DatasetIssues(
            issues=[i for i in self.dataset_issues.issues if i.severity in severities]
        )
        for verdict in self.files:
            keep = [i for i in verdict.issues if i.severity in severities]
            if keep:
                kept.files.append(FileVerdict(path=verdict.path, issues=keep))
        kept.recompute()
        return kept


ValidationReport.model_rebuild()  # resolve the self-referential ``derivatives`` field


def _highest(severities: list[Severity]) -> Severity | None:
    """Return the most serious severity, or ``None`` for an empty list."""
    if not severities:
        return None
    return max(severities, key=lambda s: s.rank)


__all__ = ["FileVerdict", "ValidationReport"]

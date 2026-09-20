"""Typed validation findings.

Every problem the validator reports is an :class:`Issue`. The field set mirrors
the reference validator's issue shape (``code``, ``severity``, ``location``,
``rule`` ...) so that output can be made interchangeable, and adds two fields
that enable features the reference validator does not offer:

* :attr:`Issue.provenance` - the schema rule that produced the finding, so a
  user can be shown *why* something is an error ("explain" mode).
* :attr:`Issue.fix` - a machine-actionable remediation hint, so a tool or GUI
  can offer a one-click fix.

These are pure-data pydantic models with no I/O, suitable for serialising to
JSON, SARIF, or binding directly to a GUI.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """How serious a finding is.

    Ordered by attention, low to high: ``IGNORE`` < ``WARNING`` < ``ERROR``.
    ``IGNORE`` is used for findings that are explicitly silenced (for example by
    a waiver or a severity override) but kept in the record for transparency.
    """

    IGNORE = "ignore"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        """Numeric attention rank, for rolling several severities up to one."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.IGNORE: 0,
    Severity.WARNING: 1,
    Severity.ERROR: 2,
}


class RuleProvenance(BaseModel):
    """Where a finding came from in the schema, for "explain" mode.

    Lets the validator answer "why is this an error?" by pointing at the exact
    rule, its selectors and checks, and the relevant field description.
    """

    rule_path: str | None = None  # e.g. "rules.checks.anat.T1wFileWithTooManyDimensions"
    selectors: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    field_definition: str | None = None  # the schema description of a metadata field


class Fix(BaseModel):
    """A machine-actionable remediation hint attached to a finding.

    ``action`` is an opaque token a consumer maps to a handler (the validator
    never applies fixes itself). ``field`` / ``value`` carry the specifics where
    relevant, e.g. ``action="add_field"``, ``field="RepetitionTime"``.
    """

    action: str
    label: str | None = None
    field: str | None = None
    value: Any | None = None


class Issue(BaseModel):
    """A single validation finding."""

    code: str
    severity: Severity = Severity.ERROR
    location: str | None = None  # dataset-relative path of the offending file
    sub_code: str | None = None  # finer category within ``code`` (e.g. a field name)
    message: str | None = None
    suggestion: str | None = None
    affects: list[str] = Field(default_factory=list)  # entity/participant labels affected
    rule: str | None = None  # schema rule path that produced the finding
    line: int | None = None  # 1-based, for tabular/text findings (the first one)
    lines: list[int] = Field(default_factory=list)  # all 1-based rows a column finding spans
    character: int | None = None  # 1-based
    provenance: RuleProvenance | None = None
    fix: Fix | None = None


class DatasetIssues(BaseModel):
    """An ordered collection of findings, with small query helpers.

    A thin wrapper rather than a bare list so reports have a stable, typed
    container that is easy to extend (filtering, grouping, severity rollup)
    without changing call sites.
    """

    issues: list[Issue] = Field(default_factory=list)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def extend(self, issues: Iterable[Issue]) -> None:
        self.issues.extend(issues)

    def by_severity(self, severity: Severity) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity is severity]

    def highest_severity(self) -> Severity | None:
        """The most serious severity present, or ``None`` if there are none."""
        if not self.issues:
            return None
        return max((issue.severity for issue in self.issues), key=lambda s: s.rank)

    def __len__(self) -> int:
        return len(self.issues)

    # Note: we deliberately do NOT override __iter__. On a pydantic ``BaseModel``
    # that would shadow the framework's field iteration (used by ``dict(model)``)
    # and surprise callers. Iterate over ``.issues`` instead.


__all__ = ["Severity", "RuleProvenance", "Fix", "Issue", "DatasetIssues"]

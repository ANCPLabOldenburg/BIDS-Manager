"""Collapse a validation report into one row per finding, not per file.

A rule that fires on every file in a dataset produces one finding per file,
and a reader scrolling 200 identical messages learns nothing the first one
did not already tell them. Worse, the one finding that is different is buried
among them.

Grouping is by what the finding *is*, not where it landed:
``(rule_id, field, severity)``. Two files missing ``units`` on the same column
are one group; the same rule on a different column is a different group,
because the fix is different.

Qt-free, so the same grouping backs the pane, a report, and any future CLI
summary. Nothing here changes a finding: a group holds the issues it collapsed
and the files they came from, so the caller can still show every one.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

from .types import Issue, Severity, ValidationReport

# Severity order for sorting, worst first.
_SEVERITY_RANK: dict[str, int] = {"error": 0, "warning": 1, "info": 2}


@dataclass
class FindingGroup:
    """One finding, and every file it fired on."""

    rule_id: str
    field: Optional[str]
    severity: Severity
    message: str
    schema_rule: Optional[str] = None
    fix_label: Optional[str] = None
    fix_action: Optional[str] = None
    # Parallel lists: ``files[i]`` produced ``issues[i]``.
    files: list[Path] = dc_field(default_factory=list)
    issues: list[Issue] = dc_field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.issues)

    @property
    def key(self) -> tuple:
        return (self.rule_id, self.field, self.severity)

    def title(self) -> str:
        """A one-line label: the rule, and the field when it names one."""
        if self.field:
            return f"{self.rule_id}  ·  {self.field}"
        return self.rule_id


def group_report(
    report: Optional[ValidationReport],
    *,
    allowed: Optional[set] = None,
    include_mirrored: bool = False,
) -> list[FindingGroup]:
    """Group every per-file finding in ``report``.

    ``allowed`` filters by severity, matching the pane's display setting.
    Mirrored findings are copies of a data file's finding placed on its
    editable sidecar; counting them would double every metadata finding, so
    they are excluded unless asked for.

    Groups come back worst-severity first, then most-frequent first, then by
    rule id so the order is stable between runs.
    """
    if report is None:
        return []
    groups: "OrderedDict[tuple, FindingGroup]" = OrderedDict()
    for verdict in report.files or []:
        for issue in verdict.issues or []:
            if getattr(issue, "mirrored", False) and not include_mirrored:
                continue
            if allowed is not None and issue.severity not in allowed:
                continue
            key = (issue.rule_id, issue.field, issue.severity)
            grp = groups.get(key)
            if grp is None:
                grp = FindingGroup(
                    rule_id=issue.rule_id,
                    field=issue.field,
                    severity=issue.severity,
                    message=issue.message,
                    schema_rule=issue.schema_rule,
                    fix_label=issue.fix_label,
                    fix_action=issue.fix_action,
                )
                groups[key] = grp
            grp.files.append(Path(verdict.path))
            grp.issues.append(issue)

    def _rank(g: FindingGroup) -> tuple:
        sev = g.severity.value if isinstance(g.severity, Severity) else str(g.severity)
        return (_SEVERITY_RANK.get(sev, 9), -g.count, g.rule_id, g.field or "")

    return sorted(groups.values(), key=_rank)


def summarise(groups: list[FindingGroup]) -> str:
    """A one-line summary for a header chip: how much a reader is being spared."""
    if not groups:
        return "no findings"
    total = sum(g.count for g in groups)
    if total == len(groups):
        return f"{total} findings"
    return f"{total} findings in {len(groups)} kinds"


__all__ = ["FindingGroup", "group_report", "summarise"]

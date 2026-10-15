"""User-supplied scan rules: classifier hints + series exclusions.

Pure data + matchers, Qt-free, so the scan engine consumes them without
importing the GUI / settings (architecture guard: the classifier imports
nothing from ``gui/``). The GUI persists these as JSON in QSettings and
converts to/from these dataclasses at the boundary; the CLI loads the same
JSON via ``bidsmgr-scan --rules-file``. :func:`from_json` / :func:`to_json`
are the single shared (de)serialiser so the GUI and CLI can never drift.

* :class:`UserHint` extends the regex-fallback classifier: when one of its
  ``patterns`` matches a series description, the series is classified to the
  hint's ``datatype`` / ``suffix`` (see ``cli/scan`` for the priority rule -
  a hint beats the built-in regex layer; ``force`` makes it beat dcm2niix
  BidsGuess too).
* :class:`ExclusionRule` marks matching series excluded from conversion:
  ``include=0`` and a reversible ``user-excluded`` note. :func:`plan_exclusions`
  decides which rows, for a scan (``cli/scan``) and for a table already open
  in the GUI alike, so a rule saved after the scan acts without a rescan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

MATCH_MODES: tuple[str, ...] = ("substring", "regex")
#: Starts the note a rule leaves in a row's ``issues``; how a row excluded
#: by a rule is told from one the user unticked.
USER_EXCLUDED_ISSUE_TOKEN = "user-excluded"
EXCLUSION_TARGETS: tuple[str, ...] = ("sequence", "path")


@dataclass(frozen=True)
class UserHint:
    """A user-defined classifier hint: pattern(s) -> datatype / suffix [/ task]."""

    patterns: tuple[str, ...]
    datatype: str
    suffix: str
    task: Optional[str] = None
    entities: tuple[tuple[str, str], ...] = ()   # extra entity overrides (acq, dir, ...)
    match_mode: str = "substring"                # "substring" | "regex"
    force: bool = False                          # override even dcm2niix BidsGuess


@dataclass(frozen=True)
class ExclusionRule:
    """A user-defined exclusion: matching series are dropped from conversion."""

    pattern: str
    target: str = "sequence"      # "sequence" (SeriesDescription) | "path"
    match_mode: str = "substring"  # "substring" | "regex"


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def validate_regex(pattern: str) -> Optional[str]:
    """Return an error string if ``pattern`` is not a compilable regex, else None."""
    try:
        re.compile(pattern)
        return None
    except re.error as exc:
        return str(exc)


def _matches(text: Optional[str], pattern: str, mode: str) -> bool:
    if not pattern:
        return False
    text = text or ""
    if mode == "regex":
        try:
            return re.search(pattern, text, re.IGNORECASE) is not None
        except re.error:
            # A bad regex never aborts a scan; it simply doesn't match.
            return False
    return pattern.lower() in text.lower()


def hint_matches(hint: UserHint, sequence: Optional[str]) -> bool:
    """True if any of ``hint``'s patterns match ``sequence``."""
    return any(_matches(sequence, p, hint.match_mode) for p in hint.patterns)


def exclusion_matches(rule: ExclusionRule, *, sequence: Optional[str], path: Optional[str]) -> bool:
    """True if ``rule`` matches the row's sequence description or path."""
    target = sequence if rule.target == "sequence" else path
    return _matches(target, rule.pattern, rule.match_mode)


def matching_rule(rules: Iterable[ExclusionRule], *, sequence: Optional[str],
                  path: Optional[str]) -> Optional[ExclusionRule]:
    """The first rule matching the row, else ``None``."""
    return next((r for r in rules if exclusion_matches(r, sequence=sequence, path=path)), None)


def exclusion_note(rule: ExclusionRule) -> str:
    return (f"{USER_EXCLUDED_ISSUE_TOKEN}: matched the scan exclusion rule "
            f"'{rule.pattern}' ({rule.target}/{rule.match_mode}); excluded from "
            "conversion. Re-tick 'include' to convert it anyway.")


def with_exclusion_note(issues: str, rule: ExclusionRule) -> str:
    """``issues`` with the rule's note first (any older rule note replaced)."""
    rest = without_exclusion_note(issues)
    note = exclusion_note(rule)
    return f"{note} | {rest}" if rest else note


def without_exclusion_note(issues: str) -> str:
    """``issues`` with every rule note taken out."""
    parts = [p.strip() for p in str(issues or "").split(" | ")]
    return " | ".join(p for p in parts if p and not p.startswith(USER_EXCLUDED_ISSUE_TOKEN))


@dataclass(frozen=True)
class ExclusionPlan:
    """What a set of rules changes in a table: rows to exclude (with the
    rule that matched) and rows a removed rule had excluded, to include
    again."""

    exclude: tuple[tuple[int, ExclusionRule], ...] = ()
    restore: tuple[int, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.exclude or self.restore)


def plan_exclusions(df, rules: Iterable[ExclusionRule]) -> ExclusionPlan:
    """Which rows of ``df`` (an inventory) the rules exclude now, and which
    a rule no longer covers. A row the user unticked by hand (no rule note)
    is never included again; a row already excluded is left alone."""
    rules = list(rules)
    cols = getattr(df, "columns", ())

    def cell(i, col: str) -> str:
        if col not in cols:
            return ""
        v = df.at[i, col]
        return "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()

    exclude: list[tuple[int, ExclusionRule]] = []
    restore: list[int] = []
    for i in df.index:
        sequence = cell(i, "sequence")
        path = cell(i, "source_folder") or cell(i, "source_file")
        rule = matching_rule(rules, sequence=sequence, path=path)
        included = cell(i, "include") not in ("0", "false", "False")
        noted = USER_EXCLUDED_ISSUE_TOKEN in cell(i, "issues")
        if rule is not None and included:
            exclude.append((int(i), rule))
        elif rule is None and noted and not included:
            restore.append(int(i))
    return ExclusionPlan(tuple(exclude), tuple(restore))


# ---------------------------------------------------------------------------
# JSON (de)serialisation - the single shared form used by the GUI + CLI
# ---------------------------------------------------------------------------


def hint_to_dict(h: UserHint) -> dict:
    return {
        "patterns": list(h.patterns),
        "datatype": h.datatype,
        "suffix": h.suffix,
        "task": h.task or "",
        "entities": {k: v for k, v in h.entities},
        "match_mode": h.match_mode,
        "force": bool(h.force),
    }


def hint_from_dict(d: dict) -> UserHint:
    raw_patterns = d.get("patterns", [])
    if isinstance(raw_patterns, str):
        raw_patterns = [raw_patterns]
    patterns = tuple(str(p).strip() for p in raw_patterns if str(p).strip())
    ents = d.get("entities", {}) or {}
    entities = tuple((str(k), str(v)) for k, v in ents.items()) if isinstance(ents, dict) else ()
    mode = str(d.get("match_mode", "substring"))
    if mode not in MATCH_MODES:
        mode = "substring"
    task = str(d.get("task", "") or "").strip() or None
    return UserHint(
        patterns=patterns,
        datatype=str(d.get("datatype", "")).strip(),
        suffix=str(d.get("suffix", "")).strip(),
        task=task,
        entities=entities,
        match_mode=mode,
        force=bool(d.get("force", False)),
    )


def exclusion_to_dict(e: ExclusionRule) -> dict:
    return {"pattern": e.pattern, "target": e.target, "match_mode": e.match_mode}


def exclusion_from_dict(d: dict) -> ExclusionRule:
    target = str(d.get("target", "sequence"))
    if target not in EXCLUSION_TARGETS:
        target = "sequence"
    mode = str(d.get("match_mode", "substring"))
    if mode not in MATCH_MODES:
        mode = "substring"
    return ExclusionRule(
        pattern=str(d.get("pattern", "")).strip(),
        target=target,
        match_mode=mode,
    )


def from_json(obj: Optional[dict]) -> tuple[list[UserHint], list[ExclusionRule]]:
    """Parse ``{"user_hints": [...], "scan_exclusions": [...]}`` into dataclasses.

    Tolerant of missing keys / malformed entries (skips blanks); never raises
    on shape, so a hand-edited rules file degrades gracefully.
    """
    obj = obj or {}
    hints = [hint_from_dict(d) for d in obj.get("user_hints", []) if isinstance(d, dict)]
    hints = [h for h in hints if h.patterns and h.datatype and h.suffix]
    excl = [exclusion_from_dict(d) for d in obj.get("scan_exclusions", []) if isinstance(d, dict)]
    excl = [e for e in excl if e.pattern]
    return hints, excl


def to_json(hints: list[UserHint], exclusions: list[ExclusionRule]) -> dict:
    return {
        "user_hints": [hint_to_dict(h) for h in hints],
        "scan_exclusions": [exclusion_to_dict(e) for e in exclusions],
    }


__all__ = [
    "ExclusionPlan",
    "USER_EXCLUDED_ISSUE_TOKEN",
    "exclusion_note",
    "matching_rule",
    "plan_exclusions",
    "with_exclusion_note",
    "without_exclusion_note",
    "MATCH_MODES",
    "EXCLUSION_TARGETS",
    "UserHint",
    "ExclusionRule",
    "validate_regex",
    "hint_matches",
    "exclusion_matches",
    "hint_to_dict",
    "hint_from_dict",
    "exclusion_to_dict",
    "exclusion_from_dict",
    "from_json",
    "to_json",
]

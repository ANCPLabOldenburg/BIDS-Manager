"""The schema rule interpreter.

Two kinds of rule are handled here, both driven entirely by the schema:

* ``checks`` (``rules.checks``) - selector-gated boolean expressions. When all
  selectors pass and a check evaluates to a determinate failure, the rule's issue
  is emitted.
* ``fields`` (``rules.sidecars``) - required / recommended sidecar fields per
  datatype and suffix. A missing required field is an error; a missing
  recommended one is a warning, with conditional levels honoured.

Robustness against a partial context: a check that evaluates to ``null`` (because
some content was not available, e.g. an associated file the engine does not yet
load) is treated as "not determinable" and skipped, never reported. Only an
explicit, non-null falsy result raises a finding. This keeps the validator from
emitting false errors while the content layers are still being filled in.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from bidsschematools.types.namespace import Namespace

from ..expr import EvaluationError, evaluate_string
from ..expr.functions import truthy
from ..issues import Issue, RuleProvenance, Severity
from ..schema import introspect
from .guidance import entity_guidance, field_guidance, value_guidance
from .tables import eval_columns
from .values import validate_value

# Rule groups whose rules this engine evaluates: content checks, sidecar-field
# requirements, dataset_description.json field requirements, and the fields of
# JSON files that are files of their own (``rules.json``: every
# ``coordsystem.json`` rule). Other groups (files, directories, modalities,
# ...) describe structure and are handled elsewhere.
#
# VENDORED CHANGE: ``json`` was missing, so no coordsystem.json was ever
# checked: an MEG one without MEGCoordinateSystem passed, which the reference
# validator reports as JSON_KEY_REQUIRED.
_EVALUATED_GROUPS = ("checks", "sidecars", "dataset_metadata", "tabular_data", "json")

# Context fields/aggregates not yet populated with real data. A rule that
# depends on one of these cannot be determined, so it is skipped rather than
# evaluated against empty data (which would otherwise produce false findings).
# coordsystems and atlas_description ARE now built (context/associations.py); only
# the microscopy/gzip headers remain. (nifti_header is not listed: the schema's
# selectors gate header checks on `nifti_header != null`.)
_UNPOPULATED_FIELDS = re.compile(r"\b(gzip|ome|tiff)\b")

_LEVEL_TO_SEVERITY = {
    "required": Severity.ERROR,
    "recommended": Severity.WARNING,
    "optional": Severity.IGNORE,
    "prohibited": Severity.IGNORE,
    # The schema marks a handful of fields deprecated (RawSources,
    # AcquisitionDuration, ScanDate, ...). Nothing is reported for them, exactly
    # as before this entry existed, but the LEVEL now survives: an editor should
    # be able to say "deprecated" rather than "optional", which is what an
    # unrecognised level used to collapse to.
    "deprecated": Severity.IGNORE,
}

# Files that cannot themselves carry a sidecar, so sidecar rules do not apply.
_SIDECAR_EXEMPT_EXTENSIONS = (".json", "", ".md", ".txt", ".rst", ".cff")

_ADDENDUM_RE = re.compile(r"(required|recommended) if `(\w+)` is `(\w+)`")


def apply_rules(schema: Namespace, context: Mapping[str, Any]) -> list[Issue]:
    """Evaluate the schema's checks and sidecar-field rules for one file."""
    issues: list[Issue] = []
    for path, rule in _flat_rules(schema):
        if _selectors_pass(rule.get("selectors", []), context):
            _apply_rule(schema, rule, context, issues, path)
    issues.extend(_validate_present_values(schema, context))
    return _dedupe(issues)


# VENDORED CHANGE (speed): the rules are flattened ONCE per schema, into plain
# dicts, with the rules that cannot be evaluated already left out. The walk
# through the schema's namespace used to run for every file (4 million lookups
# on a 1,500-file folder), with a regex per rule per file on top.
_FLAT: dict[int, list[tuple[str, dict]]] = {}


def _flat_rules(schema: Namespace) -> list[tuple[str, dict]]:
    found = _FLAT.get(id(schema))
    if found is None:
        found = []
        rules = schema["rules"]
        for group in _EVALUATED_GROUPS:
            if group in rules:
                _collect_rules(rules[group], f"rules.{group}", found)
        _FLAT[id(schema)] = found
    return found


def _collect_rules(node: Any, path: str, out: list[tuple[str, dict]]) -> None:
    if not isinstance(node, Mapping):
        return
    if "selectors" in node:
        rule = node.to_dict() if hasattr(node, "to_dict") else dict(node)
        if _is_evaluable(rule):
            out.append((path, rule))
        return
    for key, child in node.items():
        _collect_rules(child, f"{path}.{key}", out)


def _validate_present_values(schema: Namespace, context: Mapping[str, Any]) -> list[Issue]:
    """Validate the value of EVERY present JSON field against its schema metadata
    definition (not only fields a rule names). A field whose name maps to several
    definitions is valid if it matches any of them, so context-specific duplicate
    names never cause a false positive.

    Values are validated only on the ``.json`` files that carry them, matching the
    reference validator's attribution. A data file's merged-sidecar values live in
    those same JSON files, so validating them again on the data file would
    double-report (once on the .json, once on the .nii.gz)."""
    if context.get("extension") != ".json":
        return []
    if not context.get("__json_ok__", True):
        return []  # the JSON itself is malformed; integrity reports that instead
    data = context.get("json")
    if not isinstance(data, Mapping):
        return []
    by_name = introspect.metadata_by_name(schema)
    formats = _formats(schema)
    location = _location(context)
    issues: list[Issue] = []
    for field_name, value in data.items():
        definitions = by_name.get(field_name)
        if not definitions:
            continue
        problems = [validate_value(value, definition, formats) for definition in definitions]
        if all(problems):  # the value fails every definition for this name
            issues.append(
                Issue(
                    code="JSON_SCHEMA_VALIDATION_ERROR",
                    sub_code=field_name,
                    severity=Severity.ERROR,
                    location=location,
                    message=f"{field_name} {problems[0][0]}",
                    suggestion=value_guidance(field_name, definitions[0]),
                    rule="objects.metadata",
                )
            )
    return issues


_FORMATS_MEMO: dict[int, dict[str, str]] = {}


def _formats(schema: Namespace) -> dict[str, str]:
    """The schema's format patterns, once per schema (VENDORED CHANGE: see
    ``rules/values.py``)."""
    found = _FORMATS_MEMO.get(id(schema))
    if found is None:
        from .values import format_patterns

        found = _FORMATS_MEMO[id(schema)] = format_patterns(schema)
    return found


def _dedupe(issues: list[Issue]) -> list[Issue]:
    """Drop duplicate findings (the same field can be named by several rules)."""
    seen: set[tuple] = set()
    out: list[Issue] = []
    for issue in issues:
        key = (issue.code, issue.sub_code, issue.location, issue.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    return out


def _apply_rule(
    schema: Namespace,
    rule: Mapping[str, Any],
    context: Mapping[str, Any],
    issues: list[Issue],
    path: str,
) -> None:
    """A rule whose selectors passed: its checks, fields and columns."""
    if "checks" in rule:
        _eval_checks(rule, context, issues, path)
    if "fields" in rule:
        _eval_fields(schema, rule, context, issues, path)
    if "columns" in rule and path.startswith("rules.tabular_data"):
        issues.extend(eval_columns(schema, rule, context, path))


def _is_evaluable(rule: Mapping[str, Any]) -> bool:
    """False if the rule references a context field we do not yet populate."""
    text = " ".join([*rule.get("selectors", []), *rule.get("checks", [])])
    return not _UNPOPULATED_FIELDS.search(text)


def _selectors_pass(selectors: list[str], context: Mapping[str, Any]) -> bool:
    """True only if every selector evaluates truthy. A selector that cannot be
    evaluated (e.g. an unknown function in the schema) means the rule's
    applicability is undeterminable, so the rule is skipped."""
    for selector in selectors:
        try:
            if not truthy(evaluate_string(selector, context)):
                return False
        except EvaluationError:
            return False
    return True


# VENDORED CHANGE: rules where a check that evaluates to null IS the failure.
# Elsewhere null means "the context cannot say" and is skipped, which keeps a
# partial context from inventing findings. Here the null is the answer: an
# eye tracker looking at a screen with no events sidecar describing the
# screen (StimulusPresentation) is exactly the incomplete recording the rule
# exists for, and the reference validator reports it as an error. Listed by
# rule, not made the general behaviour, because the general behaviour copies
# the reference's cascades too.
_NULL_IS_FAILURE = frozenset({
    "rules.checks.eyetrack.EyetrackingStimulusPresentation",
})


def _eval_checks(
    rule: Mapping[str, Any],
    context: Mapping[str, Any],
    issues: list[Issue],
    path: str,
) -> None:
    for check in rule.get("checks", []):
        try:
            result = evaluate_string(check, context)
        except EvaluationError:
            continue  # unsupported construct: skip this check, do not report
        if result is None and path in _NULL_IS_FAILURE:
            issues.append(_issue_from_rule(rule, context, path, check))
            return
        if result is None:
            continue  # not determinable from the available context
        if not truthy(result):
            issues.append(_issue_from_rule(rule, context, path, check))
            return  # one finding per rule (a rule's checks are an AND)


def _issue_from_rule(
    rule: Mapping[str, Any],
    context: Mapping[str, Any],
    path: str,
    failed_check: str,
) -> Issue:
    issue_def = rule.get("issue") or {}
    level = issue_def.get("level", "error")
    severity = {
        "error": Severity.ERROR,
        "warning": Severity.WARNING,
    }.get(level, Severity.ERROR)
    return Issue(
        code=issue_def.get("code") or "CHECK_ERROR",
        severity=severity,
        location=_location(context),
        message=(issue_def.get("message") or "").strip() or None,
        rule=path,
        provenance=RuleProvenance(
            rule_path=path,
            selectors=list(rule.get("selectors", [])),
            checks=list(rule.get("checks", [])),
        ),
    )


def _eval_fields(
    schema: Namespace,
    rule: Mapping[str, Any],
    context: Mapping[str, Any],
    issues: list[Issue],
    path: str,
) -> None:
    is_sidecar_rule = path.startswith("rules.sidecars")
    if is_sidecar_rule and context.get("extension") in _SIDECAR_EXEMPT_EXTENSIONS:
        return
    # A malformed dataset_description.json (or other validated JSON) is reported as
    # JSON_INVALID by the integrity check; do not also flag all its fields missing.
    if not is_sidecar_rule and context.get("extension") == ".json" and not context.get(
        "__json_ok__", True
    ):
        return
    data = context.get("sidecar") if is_sidecar_rule else context.get("json")
    if not isinstance(data, Mapping):
        return
    # VENDORED CHANGE: on a derivative dataset, a sidecar rule written for raw
    # data is not enforced; only one whose selectors name derivatives is. This
    # is what the reference validator (3.0.2) does, and the comment here used
    # to claim the opposite: about 4,400 missing-field findings on the
    # bids-examples derivatives came from it.
    if is_sidecar_rule and _dataset_type(context) == "derivative" and not any(
            "DatasetType" in str(sel) or "derivative" in str(sel)
            for sel in rule.get("selectors", [])):
        return

    metadata = schema["objects"].get("metadata", {})

    for field_key, requirement in rule["fields"].items():
        meta_def = metadata.get(field_key, {})
        field_name = str(meta_def.get("name", field_key))
        if field_name in data:
            continue  # presence satisfied; value validation is a separate global pass

        severity = _field_severity(requirement, context)
        if severity is Severity.IGNORE:
            continue

        issues.append(
            _missing_field_issue(
                schema, requirement, field_name, severity, context, path, is_sidecar_rule
            )
        )


def _dataset_type(context: Mapping[str, Any]) -> str:
    dataset = context.get("dataset") or {}
    description = dataset.get("dataset_description") if isinstance(dataset, Mapping) else None
    if isinstance(description, Mapping):
        return str(description.get("DatasetType", "raw"))
    return "raw"


def level_of(requirement: Any, context: Mapping[str, Any] | None = None) -> str:
    """Resolve a schema field requirement to its BIDS requirement level.

    A requirement is either a bare level string (``"required"``) or an object
    carrying ``level`` plus an optional ``level_addendum`` that raises the level
    conditionally, e.g. *"required if `MTState` is `true`"*. When ``context``
    carries the sidecar being checked, such a condition is evaluated against it;
    without a context the unconditional level is returned, which is what a
    caller asking "what does the standard say about this field in general"
    wants.

    Returns ``"optional"`` for anything unrecognised, because an unknown
    requirement must not be reported as a problem.

    Kept separate from :func:`field_severity` on purpose. Level is what the
    standard says about a field; severity is how bad one particular finding is.
    Validation needs the second, but a form or an editor needs the first, and
    collapsing them loses information that cannot be recovered afterwards.
    """
    if isinstance(requirement, str):
        return requirement if requirement in _LEVEL_TO_SEVERITY else "optional"

    if isinstance(requirement, Mapping):
        level = requirement.get("level")
        level = level if level in _LEVEL_TO_SEVERITY else "optional"

        addendum = requirement.get("level_addendum")
        if addendum and context is not None:
            match = _ADDENDUM_RE.search(str(addendum))
            if match:
                conditional_level, key, value = match.groups()
                sidecar = context.get("sidecar") or {}
                if isinstance(sidecar, Mapping) and str(sidecar.get(key)) == value:
                    if conditional_level in _LEVEL_TO_SEVERITY:
                        level = conditional_level
        return level

    return "optional"


def field_severity(requirement: Any, context: Mapping[str, Any]) -> Severity:
    """Severity for a MISSING field, derived from its resolved level."""
    return _LEVEL_TO_SEVERITY.get(level_of(requirement, context), Severity.IGNORE)


# Historical private name, kept so existing call sites and tests keep working.
_field_severity = field_severity


def _missing_field_issue(
    schema: Namespace,
    requirement: Any,
    field_name: str,
    severity: Severity,
    context: Mapping[str, Any],
    path: str,
    is_sidecar_rule: bool,
) -> Issue:
    requirement_issue = requirement.get("issue") if isinstance(requirement, Mapping) else None
    if isinstance(requirement_issue, Mapping) and requirement_issue.get("code"):
        code = requirement_issue["code"]
    else:
        kind = "SIDECAR" if is_sidecar_rule else "JSON"
        tier = "REQUIRED" if severity is Severity.ERROR else "RECOMMENDED"
        code = f"{kind}_KEY_{tier}"
    return Issue(
        code=code,
        sub_code=field_name,
        severity=severity,
        location=_location(context),
        message=f"missing {'required' if severity is Severity.ERROR else 'recommended'} "
        f"field {field_name!r}",
        suggestion=field_guidance(schema, field_name),
        rule=path,
        fix={"action": "add_field", "field": field_name},
    )


def validate_basename(schema: Namespace, context: Mapping[str, Any], name: str) -> list[Issue]:
    """Check that each known entity's value matches the schema's pattern.

    Conservative on purpose: unknown entity names are not flagged here (they may
    be legitimate custom or derivative entities), and value patterns are only
    checked for entities the schema defines.
    """
    from ..context.entities import parse_filename

    issues: list[Issue] = []
    short_entities, _suffix, _ext = parse_filename(schema, name)
    short_to_long = introspect.short_to_long(schema)
    for short, value in short_entities.items():
        long_name = short_to_long.get(short)
        if long_name is None:
            continue
        pattern = introspect.entity_pattern(schema, long_name)
        if pattern and not re.fullmatch(pattern, value):
            issues.append(
                Issue(
                    code="ENTITY_VALUE_INVALID",
                    sub_code=short,
                    severity=Severity.ERROR,
                    location=_location(context),
                    message=f"value {value!r} for entity {short!r} does not match /{pattern}/",
                    suggestion=entity_guidance(short, pattern),
                    rule="objects.entities",
                )
            )
    return issues


def _location(context: Mapping[str, Any]) -> str:
    return str(context.get("path", "")).lstrip("/")

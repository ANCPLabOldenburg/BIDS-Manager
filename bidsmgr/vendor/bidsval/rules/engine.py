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
# requirements, and dataset_description.json field requirements. Other groups
# (files, directories, modalities, ...) describe structure and are handled
# elsewhere.
_EVALUATED_GROUPS = ("checks", "sidecars", "dataset_metadata", "tabular_data")

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
    rules = schema["rules"]
    for group in _EVALUATED_GROUPS:
        if group in rules:
            _descend(schema, rules[group], context, issues, f"rules.{group}")
    issues.extend(_validate_present_values(schema, context))
    return _dedupe(issues)


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
    location = _location(context)
    issues: list[Issue] = []
    for field_name, value in data.items():
        definitions = by_name.get(field_name)
        if not definitions:
            continue
        problems = [validate_value(value, definition) for definition in definitions]
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


def _descend(
    schema: Namespace,
    node: Any,
    context: Mapping[str, Any],
    issues: list[Issue],
    path: str,
) -> None:
    if not isinstance(node, Mapping):
        return
    if "selectors" in node:
        _eval_rule(schema, node, context, issues, path)
        return
    for key, child in node.items():
        _descend(schema, child, context, issues, f"{path}.{key}")


def _eval_rule(
    schema: Namespace,
    rule: Mapping[str, Any],
    context: Mapping[str, Any],
    issues: list[Issue],
    path: str,
) -> None:
    if not _is_evaluable(rule):
        return
    if not _selectors_pass(rule.get("selectors", []), context):
        return
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

    metadata = schema["objects"].get("metadata", {})

    for field_key, requirement in rule["fields"].items():
        meta_def = metadata.get(field_key, {})
        field_name = str(meta_def.get("name", field_key))
        if field_name in data:
            continue  # presence satisfied; value validation is a separate global pass

        severity = _field_severity(requirement, context)
        if severity is Severity.IGNORE:
            continue
        # Required and recommended sidecar/dataset fields are reported on derivative
        # datasets too: the reference validator applies these field rules regardless
        # of DatasetType, so to stay aligned with it bidsval does not suppress them.

        issues.append(
            _missing_field_issue(
                schema, requirement, field_name, severity, context, path, is_sidecar_rule
            )
        )


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

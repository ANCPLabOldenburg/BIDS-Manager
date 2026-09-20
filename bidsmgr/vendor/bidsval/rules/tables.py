"""Validate a TSV file's columns against the schema's ``rules.tabular_data``.

For a tabular file (events, channels, participants, scans ...) the schema defines
which columns may appear, which are required, whether extra columns are allowed,
and each column's type. This checks the file's actual columns against that, and
is deliberately conservative so it never reports a column problem it cannot be
sure of:

* a required defined column that is absent -> error;
* an extra column that is neither schema-defined nor documented in the sidecar ->
  warning (or error if the rule forbids extra columns);
* a value that cannot be the column's numeric type -> error.

Enum/string value checks are intentionally left out for now (an incomplete schema
enum could otherwise produce a false positive); numeric type coercion is safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bidsschematools.types.namespace import Namespace

from ..issues import Fix, Issue, Severity
from .column_types import check_value, compile_spec, is_trivial, value_signature
from .guidance import column_guidance


def eval_columns(
    schema: Namespace,
    rule: Mapping[str, Any],
    context: Mapping[str, Any],
    path: str,
) -> list[Issue]:
    columns = context.get("columns")
    if not isinstance(columns, Mapping) or not columns:
        return []  # not a populated TSV; nothing to check
    if str(context.get("extension", "")) == ".tsv.gz":
        # Gzipped TSVs (physio, stim) are headerless: their column names come from
        # the sidecar's "Columns" field, not a header row. Reading the first data
        # row as a header would mis-name every column, so skip column checks here.
        return []

    location = str(context.get("path", "")).lstrip("/")
    object_columns = schema["objects"].get("columns", {})

    # Map each rule column key to its real name + definition + requirement.
    defined: dict[str, tuple[Mapping[str, Any], Any]] = {}
    for key, requirement in rule.get("columns", {}).items():
        definition = object_columns.get(key, {})
        name = str(definition.get("name", key))
        defined[name] = (definition, requirement)

    issues: list[Issue] = []

    # Required columns must be present.
    for name, (_definition, requirement) in defined.items():
        if _level(requirement) == "required" and name not in columns:
            issues.append(
                Issue(
                    code="TSV_COLUMN_MISSING",
                    sub_code=name,
                    severity=Severity.ERROR,
                    location=location,
                    message=f"required column {name!r} is missing",
                    suggestion=column_guidance(schema, name),
                    rule=path,
                    fix=Fix(action="add_column", field=name),
                )
            )

    # Extra columns: handled per the rule's additional_columns mode (the three real
    # modes; "n/a" or anything else means no extra-column check, never a finding).
    sidecar = context.get("sidecar")
    documented = set(sidecar.keys()) if isinstance(sidecar, Mapping) else set()
    additional = str(rule.get("additional_columns") or "")
    if additional in ("allowed", "allowed_if_defined", "not_allowed"):
        for name in columns:
            if name in defined or name in documented:
                continue
            if additional == "allowed":
                issues.append(
                    Issue(
                        code="TSV_ADDITIONAL_COLUMNS_UNDEFINED",
                        sub_code=name,
                        severity=Severity.WARNING,
                        location=location,
                        message=f"column {name!r} is not defined by the schema or the sidecar",
                        suggestion="Document this column in the accompanying JSON sidecar.",
                        rule=path,
                    )
                )
            elif additional == "allowed_if_defined":
                issues.append(
                    Issue(
                        code="TSV_ADDITIONAL_COLUMNS_MUST_DEFINE",
                        sub_code=name,
                        severity=Severity.ERROR,
                        location=location,
                        message=f"extra column {name!r} must be documented in the JSON sidecar",
                        suggestion=f"Add a description of {name!r} to the JSON sidecar.",
                        rule=path,
                    )
                )
            else:  # not_allowed
                issues.append(
                    Issue(
                        code="TSV_ADDITIONAL_COLUMNS_NOT_ALLOWED",
                        sub_code=name,
                        severity=Severity.ERROR,
                        location=location,
                        message=f"column {name!r} is not allowed in this file",
                        rule=path,
                    )
                )

    issues += _pseudo_age(columns, location, path)
    issues += _index_unique(rule, object_columns, columns, location, path)
    issues += _initial_columns_order(rule, object_columns, columns, location, path)
    issues += _value_types(schema, defined, columns, context, location, path)

    return issues


def _value_types(
    schema: Namespace,
    defined: Mapping[str, tuple[Mapping[str, Any], Any]],
    columns: Mapping[str, list[Any]],
    context: Mapping[str, Any],
    location: str,
    path: str,
) -> list[Issue]:
    """Check each schema-defined column's values against its (possibly sidecar-refined)
    type signature, and flag an incompatible sidecar redefinition."""
    formats = schema["objects"].get("formats", {})
    sidecar = context.get("sidecar")
    sidecar = sidecar if isinstance(sidecar, Mapping) else {}
    issues: list[Issue] = []
    for name, (column_object, _requirement) in defined.items():
        if name not in columns:
            continue
        signature, redefine = value_signature(column_object, sidecar.get(name))
        if redefine:
            issues.append(
                Issue(
                    code="TSV_COLUMN_TYPE_REDEFINED",
                    sub_code=name,
                    severity=Severity.WARNING,
                    location=location,
                    message=f"the sidecar redefinition of column {name!r} is ignored: {redefine}",
                    suggestion="A sidecar may only refine a schema column (same base type, a "
                    "subset of any levels, within any bounds). Remove or correct the redefinition.",
                    rule=path,
                )
            )
        if is_trivial(signature):
            continue
        spec = compile_spec(signature, formats)
        bad_lines: list[int] = []
        first_value: Any = None
        for index, value in enumerate(columns[name]):
            text = str(value)
            if name == "age" and text == "89+":
                continue  # reported as TSV_PSEUDO_AGE_DEPRECATED instead
            if not check_value(text, spec):
                if not bad_lines:
                    first_value = value
                bad_lines.append(index + 2)  # 1-based, +1 for the header row
        if bad_lines:
            # One finding per column (matching the reference validator: same
            # code / first line / message). ``lines`` lists every offending row
            # so a consumer can point at all the bad cells, without changing the
            # finding count.
            issues.append(
                Issue(
                    code="TSV_VALUE_INCORRECT_TYPE",
                    sub_code=name,
                    severity=Severity.ERROR,
                    location=location,
                    line=bad_lines[0],
                    lines=bad_lines,
                    message=f"column {name!r}: value {first_value!r} is not valid for its type",
                    suggestion=_value_suggestion(signature),
                    rule=path,
                )
            )
    return issues


def _value_suggestion(signature: Any) -> str:
    if signature.levels:
        shown = ", ".join(signature.levels[:8])
        more = ", ..." if len(signature.levels) > 8 else ""
        return f"Use one of the allowed values ({shown}{more}), or 'n/a'."
    bounds = []
    if signature.minimum is not None:
        bounds.append(f">= {signature.minimum}")
    if signature.maximum is not None:
        bounds.append(f"<= {signature.maximum}")
    kind = signature.formats[0] if signature.formats else "the expected type"
    text = f"Each value must be a valid {kind}"
    if bounds:
        text += " (" + " and ".join(bounds) + ")"
    return text + ", or 'n/a'."


def _index_unique(
    rule: Mapping[str, Any],
    object_columns: Mapping[str, Any],
    columns: Mapping[str, list[Any]],
    location: str,
    path: str,
) -> list[Issue]:
    """The combination of the rule's index columns must be unique across rows."""
    keys = [str(k) for k in rule.get("index_columns", [])]
    names = [str(object_columns.get(k, {}).get("name", k)) for k in keys]
    names = [name for name in names if name in columns]  # the present index columns
    if not names:
        return []
    seen: set[tuple] = set()
    for row in zip(*(columns[name] for name in names), strict=False):
        if row in seen:
            label = " + ".join(names)
            return [
                Issue(
                    code="TSV_INDEX_VALUE_NOT_UNIQUE",
                    sub_code=label,
                    severity=Severity.ERROR,
                    location=location,
                    message=f"the index column(s) {label} must be unique, but a value repeats",
                    suggestion=f"Make every row's {label} value unique (it identifies the row).",
                    rule=path,
                )
            ]
        seen.add(row)
    return []


def _initial_columns_order(
    rule: Mapping[str, Any],
    object_columns: Mapping[str, Any],
    columns: Mapping[str, list[Any]],
    location: str,
    path: str,
) -> list[Issue]:
    """The rule's ``initial_columns`` must appear first, in the given order.

    Only present columns are checked for order (a missing required column is left to
    the column-presence check, so it is not reported twice). Mirrors the reference's
    ``evalInitialColumns``.
    """
    initial = rule.get("initial_columns")
    if not initial:
        return []
    headers = list(columns.keys())
    rule_columns = rule.get("columns", {})
    resolved: list[tuple[str, int]] = []
    for raw_key in initial:
        key = str(raw_key)
        name = str(object_columns.get(key, {}).get("name", key))
        requirement = _level(rule_columns.get(key)) if key in rule_columns else ""
        index = headers.index(name) if name in headers else -1
        if requirement == "required" or index != -1:
            resolved.append((name, index))
    issues: list[Issue] = []
    for target_index, (name, index) in enumerate(resolved):
        if index != -1 and index != target_index:
            issues.append(
                Issue(
                    code="TSV_COLUMN_ORDER_INCORRECT",
                    sub_code=name,
                    severity=Severity.ERROR,
                    location=location,
                    message=f"column {name!r} should be at position {target_index + 1} "
                    f"but is at position {index + 1}",
                    suggestion=f"Move {name!r} to position {target_index + 1}; the schema's "
                    "initial columns must come first, in order.",
                    rule=path,
                )
            )
    return issues


def _pseudo_age(columns: Mapping[str, Any], location: str, path: str) -> list[Issue]:
    """Flag the deprecated ``89+`` value in an ``age`` column."""
    age = columns.get("age")
    if not isinstance(age, list) or not any(str(v).strip() == "89+" for v in age):
        return []
    return [
        Issue(
            code="TSV_PSEUDO_AGE_DEPRECATED",
            sub_code="age",
            severity=Severity.WARNING,
            location=location,
            message="the value '89+' in the 'age' column is deprecated",
            suggestion="Use 89 for all ages 89 and over (the cap, not a '+', preserves privacy).",
            rule=path,
        )
    ]


def _level(requirement: Any) -> str:
    if isinstance(requirement, str):
        return requirement
    if isinstance(requirement, Mapping):
        return str(requirement.get("level", ""))
    return ""

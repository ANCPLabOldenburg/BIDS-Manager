"""Filename and path legality, ported from the reference validator.

The schema's ``rules.files`` describes every legal filename: which suffix goes in
which datatype folder, which entities are required or allowed, and which
extensions. This module identifies which rule(s) a file matches and then checks the
file against them, reporting the same findings the reference validator does:

* ``NOT_INCLUDED`` - the file matches no BIDS rule (a typo, or a file that belongs
  in ``sourcedata``/``derivatives``, or one that should be listed in ``.bidsignore``);
* ``ENTITY_WITH_NO_LABEL`` - an entity present with no label (``acq-``);
* ``INVALID_ENTITY_LABEL`` - an entity's label breaks the schema's value pattern;
* ``MISSING_REQUIRED_ENTITY`` / ``ENTITY_NOT_IN_RULE`` - too few / too many entities;
* ``DATATYPE_MISMATCH`` / ``EXTENSION_MISMATCH`` / ``INVALID_LOCATION`` - right name,
  wrong folder / extension / place;
* ``FILENAME_MISMATCH`` - entities duplicated or out of order;
* ``ALL_FILENAME_RULES_HAVE_ISSUES`` - several rules matched and each had a problem.

The no-false-positives discipline is preserved: a ``.bidsignore``-d file is never
``NOT_INCLUDED``, directory recordings (``.ds`` ...) are not name-checked as files,
and the rule-matching mirrors the reference so bidsval flags exactly what it flags.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping
from typing import Any

from bidsschematools.types.namespace import Namespace

from ..files import BIDSFile
from ..issues import Issue, Severity

# Per-schema caches (schema objects are cached for the process, so id() is stable).
_RULES_MEMO: dict[int, list[tuple[str, Mapping[str, Any]]]] = {}
_ENTITY_BY_SHORT_MEMO: dict[int, dict[str, Mapping[str, Any]]] = {}
_ORDERED_SHORT_MEMO: dict[int, list[str]] = {}


def filename_checks(
    schema: Namespace, context: Mapping[str, Any], bids_file: BIDSFile
) -> list[Issue]:
    """Identify the matching ``rules.files`` rule(s) and validate the filename.

    A directory recording (CTF ``.ds``, MEF ``.mefd``, OME-Zarr) goes through here
    like any other entry. It is ONE unit, so what is checked is the directory's own
    name; its vendor-named internals are never indexed and so are never name-checked.

    This used to return early for such a recording, on the reasoning that the
    reference validates them by a different path. The effect was that no directory
    recording was checked at all: not its entities, not its suffix, not whether it
    was in a datatype directory. A CTF recording could sit in a folder called
    ``awwww`` and nothing was reported. The reason for the early return was real,
    though: the schema spells the extension ``.ds/`` and the parser reported
    ``.ds``, so nothing matched and every valid recording came back
    ``NOT_INCLUDED``. That is fixed where the context is built, by restoring the
    trailing slash, which is what makes the ordinary rules apply here.
    """
    dataset_type = _dataset_type(context)
    matched = _find_rule_matches(schema, context, dataset_type)

    if not matched:
        dataset = context.get("dataset") or {}
        bidsignore = dataset.get("bidsignore") if isinstance(dataset, Mapping) else None
        if bidsignore is not None and bidsignore.match(bids_file.relpath):
            return []  # the user listed this file in .bidsignore
        return [
            Issue(
                code="NOT_INCLUDED",
                severity=Severity.ERROR,
                location=bids_file.relpath,
                message=(
                    f"{bids_file.name} does not match any BIDS naming rule for this dataset"
                ),
                suggestion=(
                    "Check the file name against the BIDS specification (a typo in the suffix or "
                    "an entity is the usual cause). Source data belongs in sourcedata/, processed "
                    "data in derivatives/, and anything intentionally outside BIDS can be listed "
                    "in a .bidsignore file at the dataset root."
                ),
            )
        ]

    matched = _narrow(schema, context, matched)
    issues: list[Issue] = []
    issues += _missing_label(schema, context, bids_file, matched)
    issues += _entity_label_check(schema, context, bids_file)
    issues += _check_rules(schema, context, bids_file, matched)
    issues += _missing_datatype_directory(context, bids_file, matched)
    issues += _reconstruction_failure(schema, context, bids_file)
    return issues


# --- rule identification --------------------------------------------------


def _file_rules(schema: Namespace) -> list[tuple[str, Mapping[str, Any]]]:
    """Flatten ``rules.files`` to ``[(path, leaf_rule)]`` once per schema."""
    cached = _RULES_MEMO.get(id(schema))
    if cached is not None:
        return cached
    out: list[tuple[str, Mapping[str, Any]]] = []
    files = schema["rules"].get("files", {})
    for group in files.keys():
        _collect(files[group], f"rules.files.{group}", out)
    _RULES_MEMO[id(schema)] = out
    return out


def _collect(node: Any, path: str, out: list[tuple[str, Mapping[str, Any]]]) -> None:
    if not _is_mapping(node):
        return
    if "path" in node or "stem" in node or "suffixes" in node:
        out.append((path, node))
        return
    for key in node.keys():
        _collect(node[key], f"{path}.{key}", out)


def _find_rule_matches(
    schema: Namespace, context: Mapping[str, Any], dataset_type: str
) -> list[tuple[str, Mapping[str, Any]]]:
    out: list[tuple[str, Mapping[str, Any]]] = []
    for path, node in _file_rules(schema):
        if path.startswith("rules.files.deriv") and dataset_type != "derivative":
            continue
        if _rule_matches(node, context):
            out.append((path, node))
    return out


def _rule_matches(node: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    if "path" in node and "/" + str(node["path"]) == context.get("path"):
        return True
    if "stem" in node and _match_stem(node, context):
        return True
    if "suffixes" in node and context.get("suffix") in list(node["suffixes"]):
        return True
    return False


def _match_stem(node: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    stem = str(context.get("__name__", "")).split(".")[0]
    if not fnmatch.fnmatchcase(stem, str(node["stem"])):
        return False
    if "datatypes" in node:
        return context.get("datatype") in list(node["datatypes"])
    return True


def _narrow(
    schema: Namespace,
    context: Mapping[str, Any],
    matched: list[tuple[str, Mapping[str, Any]]],
) -> list[tuple[str, Mapping[str, Any]]]:
    """When several rules match, prefer the one sharing the file's datatype, then
    the one whose entities/extensions fit (mirrors the reference)."""
    if len(matched) <= 1:
        return matched
    datatype = context.get("datatype")
    by_datatype = [
        (p, n) for p, n in matched if "datatypes" in n and datatype in list(n["datatypes"])
    ]
    if by_datatype:
        matched = by_datatype
    if len(matched) <= 1:
        return matched
    by_ent_ext = [(p, n) for p, n in matched if _entities_extensions_fit(schema, context, n)]
    return by_ent_ext or matched


def _entities_extensions_fit(
    schema: Namespace, context: Mapping[str, Any], rule: Mapping[str, Any]
) -> bool:
    ext_ok = "extensions" not in rule or context.get("extension") in list(rule["extensions"])
    if "entities" not in rule:
        return ext_ok
    rule_entities = {_short(schema, key) for key in rule["entities"].keys()}
    file_entities = set(context.get("entities", {}).keys())
    return ext_ok and file_entities.issubset(rule_entities)


# --- per-file checks ------------------------------------------------------


def _missing_label(
    schema: Namespace,
    context: Mapping[str, Any],
    bids_file: BIDSFile,
    matched: list[tuple[str, Mapping[str, Any]]],
) -> list[Issue]:
    if not any("suffixes" in node for _p, node in matched):
        return []
    empty = [key for key, value in context.get("entities", {}).items() if value == ""]
    if not empty:
        return []
    return [
        Issue(
            code="ENTITY_WITH_NO_LABEL",
            sub_code=", ".join(empty),
            severity=Severity.ERROR,
            location=bids_file.relpath,
            message=f"entit{'y' if len(empty) == 1 else 'ies'} with no label: {', '.join(empty)}",
            suggestion=(
                "Every entity needs a label, written as 'key-label'. For example, write "
                "'acq-highres' rather than 'acq-'."
            ),
        )
    ]


def _entity_label_check(
    schema: Namespace, context: Mapping[str, Any], bids_file: BIDSFile
) -> list[Issue]:
    formats = schema["objects"].get("formats", {})
    by_short = _entity_by_short(schema)
    issues: list[Issue] = []
    for short, label in context.get("entities", {}).items():
        if label == "":
            continue  # an empty label is reported as ENTITY_WITH_NO_LABEL instead
        definition = by_short.get(short)
        fmt = definition.get("format") if isinstance(definition, Mapping) else None
        if not fmt or str(fmt) not in formats:
            continue
        pattern = str(formats[str(fmt)].get("pattern", ""))
        if pattern and not re.fullmatch(pattern, label):
            issues.append(
                Issue(
                    code="INVALID_ENTITY_LABEL",
                    sub_code=short,
                    severity=Severity.ERROR,
                    location=bids_file.relpath,
                    message=f"label {label!r} for entity {short!r} does not match /{pattern}/",
                    suggestion=(
                        f"The value after '{short}-' must match the pattern /{pattern}/. "
                        f"For example, '{short}-01' if labels are zero-padded numbers."
                    ),
                )
            )
    return issues


def _check_rules(
    schema: Namespace,
    context: Mapping[str, Any],
    bids_file: BIDSFile,
    matched: list[tuple[str, Mapping[str, Any]]],
) -> list[Issue]:
    if len(matched) == 1:
        return _rule_issues(schema, context, bids_file, matched[0])
    # Several rules still match: if any matches cleanly, accept it (no finding);
    # otherwise report that every candidate had a problem.
    per_rule = [(path, _rule_issues(schema, context, bids_file, (path, node))) for path, node in
                matched]
    if any(not issues for _path, issues in per_rule):
        return []
    return [
        Issue(
            code="ALL_FILENAME_RULES_HAVE_ISSUES",
            severity=Severity.ERROR,
            location=bids_file.relpath,
            message="the file resembles several BIDS rules, but does not fully satisfy any of them",
            suggestion=(
                "The name is close to more than one valid pattern but matches none exactly. "
                "Compare it to the specification for the suffix you intend."
            ),
        )
    ]


def _rule_issues(
    schema: Namespace,
    context: Mapping[str, Any],
    bids_file: BIDSFile,
    matched: tuple[str, Mapping[str, Any]],
) -> list[Issue]:
    path, rule = matched
    issues: list[Issue] = []
    issues += _entity_rule_issue(schema, context, bids_file, path, rule)
    issues += _datatype_mismatch(context, bids_file, path, rule)
    issues += _extension_mismatch(context, bids_file, path, rule)
    issues += _invalid_location(context, bids_file)
    return issues


def _entity_rule_issue(
    schema: Namespace,
    context: Mapping[str, Any],
    bids_file: BIDSFile,
    path: str,
    rule: Mapping[str, Any],
) -> list[Issue]:
    if "entities" not in rule:
        return []
    file_entities = list(context.get("entities", {}).keys())
    rule_entities = [_short(schema, key) for key in rule["entities"].keys()]
    issues: list[Issue] = []

    # Required-entity checks do not apply to a file at the dataset root (a shared
    # sidecar inherited downward), matching the reference.
    if not _is_at_root(bids_file):
        required = [
            _short(schema, key)
            for key, level in rule["entities"].items()
            if str(level) == "required"
        ]
        missing = [ent for ent in required if ent not in file_entities]
        if missing:
            issues.append(
                Issue(
                    code="MISSING_REQUIRED_ENTITY",
                    sub_code=", ".join(missing),
                    severity=Severity.ERROR,
                    location=bids_file.relpath,
                    message=f"missing required entit{'y' if len(missing) == 1 else 'ies'}: "
                    f"{', '.join(missing)}",
                    suggestion=(
                        "Add the missing entity to the filename, for example "
                        f"'{missing[0]}-01'."
                    ),
                    rule=path,
                )
            )

    extra = [ent for ent in file_entities if ent not in rule_entities]
    if extra:
        issues.append(
            Issue(
                code="ENTITY_NOT_IN_RULE",
                sub_code=", ".join(extra),
                severity=Severity.ERROR,
                location=bids_file.relpath,
                message=f"entit{'y' if len(extra) == 1 else 'ies'} not allowed for this file "
                f"type: {', '.join(extra)}",
                suggestion=(
                    "Remove the entity, or check that you are using the right suffix; "
                    "this entity is not part of the rule for this file type."
                ),
                rule=path,
            )
        )
    return issues


def _datatype_mismatch(
    context: Mapping[str, Any], bids_file: BIDSFile, path: str, rule: Mapping[str, Any]
) -> list[Issue]:
    datatype = context.get("datatype")
    if datatype and "datatypes" in rule and datatype not in list(rule["datatypes"]):
        allowed = ", ".join(str(d) for d in rule["datatypes"])
        return [
            Issue(
                code="DATATYPE_MISMATCH",
                severity=Severity.ERROR,
                location=bids_file.relpath,
                message=f"the file is in the '{datatype}' folder but its suffix belongs in: "
                f"{allowed}",
                suggestion=f"Move the file into one of: {allowed}.",
                rule=path,
            )
        ]
    return []


def _extension_mismatch(
    context: Mapping[str, Any], bids_file: BIDSFile, path: str, rule: Mapping[str, Any]
) -> list[Issue]:
    if "extensions" in rule and context.get("extension") not in list(rule["extensions"]):
        allowed = ", ".join(str(e) for e in rule["extensions"])
        return [
            Issue(
                code="EXTENSION_MISMATCH",
                severity=Severity.ERROR,
                location=bids_file.relpath,
                message=f"extension {context.get('extension')!r} is not allowed for this file "
                f"type; allowed: {allowed}",
                suggestion=f"Use one of the allowed extensions: {allowed}.",
                rule=path,
            )
        ]
    return []


def _invalid_location(context: Mapping[str, Any], bids_file: BIDSFile) -> list[Issue]:
    entities = context.get("entities", {})
    path = context.get("path", "")
    issues: list[Issue] = []
    if "tpl" not in entities:
        issues += _validate_location(entities, path, bids_file, "sub", "ses")
    if "sub" not in entities:
        issues += _validate_location(entities, path, bids_file, "tpl", "cohort")
    return issues


def _validate_location(
    entities: Mapping[str, str], path: str, bids_file: BIDSFile, top: str, sub: str
) -> list[Issue]:
    issues: list[Issue] = []
    top_val = entities.get(top)
    sub_val = entities.get(sub)
    if top_val:
        expected = f"/{top}-{top_val}/"
        if sub_val:
            expected += f"{sub}-{sub_val}/"
        if not path.startswith(expected):
            issues.append(_location_issue(bids_file, f"expected to be under {expected}"))
    if not top_val and re.match(rf"^/{top}-", path):
        issues.append(_location_issue(bids_file, f"a '{top}-' folder but no '{top}' in the name"))
    if not sub_val and re.search(rf"/{sub}-", path):
        issues.append(_location_issue(bids_file, f"a '{sub}-' folder but no '{sub}' in the name"))
    return issues


# Metadata may sit higher in the tree than the data it describes, which is what
# the inheritance principle is for, so these are never "missing a datatype
# directory".
_INHERITABLE_EXTENSIONS = frozenset({".json", ".tsv"})


def _at_inheritance_level(relpath: str) -> bool:
    """Is this file at a level the inheritance principle actually allows?

    The principle lets a sidecar sit ABOVE the data it describes: at the dataset
    root, beside a subject, or beside a session. Those are the three places, and
    in every one of them the file sits DIRECTLY in that folder.

    A ``.json`` inside ``sub-01/ses-pre/awwww/`` is not inheriting anything. It
    is in a folder BIDS does not read, exactly as lost as the image beside it,
    and exempting it because of its extension told the user only half of what
    was wrong with that folder.
    """
    parts = [p for p in relpath.strip("/").split("/") if p]
    depth = 0
    if depth < len(parts) and parts[depth].startswith("sub-"):
        depth += 1
        if depth < len(parts) and parts[depth].startswith("ses-"):
            depth += 1
    # What is left should be the filename alone; anything more means the file is
    # inside a container directory, and that container is not a datatype.
    return len(parts) - depth <= 1


def _missing_datatype_directory(
    context: Mapping[str, Any],
    bids_file: BIDSFile,
    matched: list[tuple[str, Mapping[str, Any]]],
) -> list[Issue]:
    """Report a data file that is not inside a recognised datatype directory.

    Deliberately STRICTER than the reference TypeScript validator, which misses
    this case: its suffix matching ignores the datatype, and ``DATATYPE_MISMATCH``
    is skipped when the parent directory is not a known datatype, so a perfectly
    named ``sub-01_T1w.nii.gz`` in a folder called ``awwww`` passed every check.
    The legacy ``is_bids`` regex does catch it, because its patterns cover the
    whole path, so leaving this out would lose coverage the schema-driven rules
    are meant to replace.

    Ported from python-validator's check of the same name so the two agree, in
    behaviour and in the code they emit.
    """
    if context.get("datatype"):
        return []  # the file is in a recognised datatype directory
    if context.get("extension") in _INHERITABLE_EXTENSIONS and _at_inheritance_level(
        bids_file.relpath
    ):
        return []  # metadata legitimately sitting above the data it describes
    if not matched or not all("datatypes" in node for _path, node in matched):
        return []  # this file type need not live in a datatype directory

    allowed: list[str] = []
    for _path, node in matched:
        for datatype in node["datatypes"]:
            if str(datatype) not in allowed:
                allowed.append(str(datatype))
    return [
        Issue(
            code="INVALID_LOCATION",
            severity=Severity.ERROR,
            location=bids_file.relpath,
            message=(
                "the file has a valid name but is not in a datatype directory, "
                "expected one of: " + ", ".join(allowed)
            ),
            suggestion=f"Move the file into one of: {', '.join(allowed)}.",
        )
    ]


def _location_issue(bids_file: BIDSFile, detail: str) -> Issue:
    return Issue(
        code="INVALID_LOCATION",
        severity=Severity.ERROR,
        location=bids_file.relpath,
        message=f"the file has a valid name but is in the wrong place ({detail})",
        suggestion=(
            "BIDS files live under sub-<label>/[ses-<label>/]<datatype>/. Move the file so its "
            "folders match the sub- (and ses-) entities in its name."
        ),
    )


def _reconstruction_failure(
    schema: Namespace, context: Mapping[str, Any], bids_file: BIDSFile
) -> list[Issue]:
    entities = context.get("entities", {})
    if not entities:
        return []
    ordered = [short for short in _ordered_short(schema) if short in entities]
    parts = [f"{short}-{entities[short]}" for short in ordered]
    suffix = context.get("suffix", "") or ""
    # A directory recording carries the schema's trailing slash in its extension so
    # the rules match, but the thing on disk is a directory called
    # ``sub-01_task-rest_meg.ds``, with no slash in its name. Rebuilding the name
    # has to compare against what is actually there.
    extension = (context.get("extension", "") or "").rstrip("/")
    expected = "_".join([*parts, suffix + extension])
    if bids_file.name != expected:
        return [
            Issue(
                code="FILENAME_MISMATCH",
                severity=Severity.ERROR,
                location=bids_file.relpath,
                message=f"the filename is not in canonical form; expected {expected!r}",
                suggestion=(
                    "Entities must appear once each, in the BIDS order, before the suffix. "
                    f"Rename the file to {expected!r}."
                ),
            )
        ]
    return []


# --- schema helpers (memoised) --------------------------------------------


def _entity_by_short(schema: Namespace) -> dict[str, Mapping[str, Any]]:
    cached = _ENTITY_BY_SHORT_MEMO.get(id(schema))
    if cached is not None:
        return cached
    out: dict[str, Mapping[str, Any]] = {}
    for definition in schema["objects"]["entities"].values():
        name = definition.get("name")
        if name:
            out[str(name)] = definition
    _ENTITY_BY_SHORT_MEMO[id(schema)] = out
    return out


def _ordered_short(schema: Namespace) -> list[str]:
    cached = _ORDERED_SHORT_MEMO.get(id(schema))
    if cached is not None:
        return cached
    entities = schema["objects"]["entities"]
    out: list[str] = []
    for long_name in schema["rules"].get("entities", []):
        if long_name in entities:
            name = entities[long_name].get("name")
            if name:
                out.append(str(name))
    _ORDERED_SHORT_MEMO[id(schema)] = out
    return out


def _short(schema: Namespace, long_name: str) -> str:
    entities = schema["objects"]["entities"]
    if long_name in entities:
        return str(entities[long_name].get("name", long_name))
    return long_name


def _dataset_type(context: Mapping[str, Any]) -> str:
    dataset = context.get("dataset") or {}
    description = dataset.get("dataset_description") if isinstance(dataset, Mapping) else {}
    if isinstance(description, Mapping):
        return str(description.get("DatasetType", "raw"))
    return "raw"


def _is_at_root(bids_file: BIDSFile) -> bool:
    return "/" not in bids_file.relpath


def _is_mapping(node: Any) -> bool:
    return isinstance(node, Mapping) or hasattr(node, "keys")

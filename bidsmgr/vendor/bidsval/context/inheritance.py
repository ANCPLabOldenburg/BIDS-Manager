"""Apply the BIDS inheritance principle to assemble a file's sidecar.

A JSON metadata file applies to a data file when it sits in the same directory or
an ancestor, its entities are a subset of the data file's (with matching values),
and its suffix matches. More specific sidecars (deeper directory, more entities)
override less specific ones.
"""

from __future__ import annotations

from bidsschematools.types.namespace import Namespace

from ..files import BIDSFile, FileTree
from ..issues import Issue, Severity
from .entities import parse_filename
from .loaders import load_json


def inheritance_checks(schema: Namespace, tree: FileTree, data_file: BIDSFile) -> list[Issue]:
    """Inheritance findings for a data file (ported from the reference validator):

    * ``MULTIPLE_INHERITABLE_FILES`` - a directory has more than one applicable
      sidecar and none is an exact match, so which one applies is ambiguous;
    * ``SIDECAR_FIELD_OVERRIDE`` - a less specific sidecar sets a field to a value a
      more specific sidecar overrides (the more specific one wins, silently).
    """
    name = data_file.name
    if name.endswith(".json"):
        return []
    source_entities, source_suffix, _ = parse_filename(schema, name)
    if not source_suffix:
        return []

    issues: list[Issue] = []
    merged_value: dict[str, object] = {}
    merged_origin: dict[str, str] = {}
    for dir_relpath in tree.ancestor_dirs(data_file.relpath):  # closest first
        candidates: list[tuple[dict[str, str], BIDSFile]] = []
        exact: BIDSFile | None = None
        for candidate in tree.json_sidecars_in(dir_relpath):
            cand_entities, cand_suffix, _ = parse_filename(schema, candidate.name)
            if cand_suffix != source_suffix or not _is_subset(cand_entities, source_entities):
                continue
            if cand_entities == source_entities:
                exact = candidate
            candidates.append((cand_entities, candidate))

        if exact is None and len(candidates) > 1:
            paths = sorted(c.relpath for _e, c in candidates)
            issues.append(
                Issue(
                    code="MULTIPLE_INHERITABLE_FILES",
                    severity=Severity.ERROR,
                    location=paths[0],
                    message="more than one sidecar in this directory applies, and none matches "
                    "exactly, so the metadata is ambiguous: " + ", ".join(paths),
                    suggestion="Keep a single applicable sidecar per directory, or name one to "
                    "match the data file's entities exactly.",
                )
            )
            break  # ambiguous: stop merging (the reference stops here too)

        chosen = exact
        if chosen is None and candidates:
            candidates.sort(key=lambda item: (len(item[0]), item[1].relpath))
            chosen = candidates[-1][1]
        if chosen is None:
            continue

        data = load_json(chosen)
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if key in merged_value and merged_value[key] != value:
                issues.append(
                    Issue(
                        code="SIDECAR_FIELD_OVERRIDE",
                        sub_code=key,
                        severity=Severity.WARNING,
                        location=merged_origin[key],
                        message=f"field {key!r} is overridden by a more specific sidecar; "
                        f"this value is ignored",
                        suggestion="Remove the duplicate field from one sidecar, or make the "
                        "values agree. The more specific sidecar takes precedence.",
                    )
                )
            merged_value.setdefault(key, value)
            merged_origin.setdefault(key, chosen.relpath)
    return issues


def merged_sidecar(schema: Namespace, tree: FileTree, data_file: BIDSFile) -> dict[str, object]:
    """Return the inheritance-merged JSON sidecar for ``data_file``.

    Returns an empty dict for a file that is itself JSON (JSON files have no
    sidecar) or that has no suffix.
    """
    name = data_file.name
    if name.endswith(".json"):
        return {}
    source_entities, source_suffix, _ = parse_filename(schema, name)
    if not source_suffix:
        return {}

    # The inheritance principle applies AT MOST ONE sidecar per directory level
    # (the most specific subset match), then merges across levels with the
    # closest level winning. Picking one-per-level avoids pulling fields from a
    # less specific sidecar that the spec would not apply.
    merged: dict[str, object] = {}
    for dir_relpath in tree.ancestor_dirs(data_file.relpath):  # closest first
        chosen = _best_sidecar(schema, tree, dir_relpath, source_entities, source_suffix)
        if chosen is None:
            continue
        data = load_json(chosen)
        if isinstance(data, dict):
            for key, value in data.items():
                merged.setdefault(key, value)  # closest level (seen first) wins
    return merged


def applicable_sidecar_files(schema: Namespace, tree: FileTree, data_file: BIDSFile) -> list[str]:
    """The relpaths of the JSON sidecars that apply to ``data_file`` (one per level).

    Used to mark sidecars as "in use", so a sidecar that applies to no data file can
    be reported. Empty for a JSON file or a file with no suffix.
    """
    name = data_file.name
    if name.endswith(".json"):
        return []
    source_entities, source_suffix, _ = parse_filename(schema, name)
    if not source_suffix:
        return []
    out: list[str] = []
    for dir_relpath in tree.ancestor_dirs(data_file.relpath):
        chosen = _best_sidecar(schema, tree, dir_relpath, source_entities, source_suffix)
        if chosen is not None:
            out.append(chosen.relpath)
    return out


def _best_sidecar(
    schema: Namespace,
    tree: FileTree,
    dir_relpath: str,
    source_entities: dict[str, str],
    source_suffix: str,
) -> BIDSFile | None:
    """The single most-specific sidecar in one directory that applies, or None.

    Prefers a sidecar whose entities exactly match the data file's; otherwise the
    one with the most entities (deterministic by path on a tie).
    """
    candidates: list[tuple[dict[str, str], BIDSFile]] = []
    for candidate in tree.json_sidecars_in(dir_relpath):
        cand_entities, cand_suffix, _ = parse_filename(schema, candidate.name)
        if cand_suffix != source_suffix:
            continue
        if not _is_subset(cand_entities, source_entities):
            continue
        if cand_entities == source_entities:  # exact match: use it directly
            return candidate
        candidates.append((cand_entities, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (len(item[0]), item[1].relpath))
    return candidates[-1][1]


def _is_subset(candidate: dict[str, str], source: dict[str, str]) -> bool:
    """True if every entity in ``candidate`` is present in ``source`` with the same value."""
    return all(source.get(key) == value for key, value in candidate.items())

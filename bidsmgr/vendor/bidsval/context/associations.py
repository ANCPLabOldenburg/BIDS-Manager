"""Resolve a data file's associated files into the ``associations`` context.

Many schema checks look at files that travel with a data file: a ``dwi`` file's
``.bval``/``.bvec``, a task recording's ``events.tsv``, an electrophysiology
recording's ``channels.tsv``, an ASL run's ``aslcontext.tsv``, and so on. The
schema describes each of these in ``meta.associations`` (a selector saying when
it applies, a target suffix/extension to look for, and whether it inherits up the
tree).

This module finds those files (using the same proximity walk as the inheritance
principle) and exposes them under ``associations.<name>`` with the fields the
checks read: a TSV's columns plus ``n_rows``/``n_cols`` and its sidecar; a
``.bval``/``.bvec``'s ``values``/``n_rows``/``n_cols``; or just the path for
plain existence checks.

Association names that need a more complex aggregate (``coordsystems``,
``atlas_description``) are intentionally not built here; the rule engine skips
rules that reference them, so they are never guessed at.
"""

from __future__ import annotations

from typing import Any

from bidsschematools.types.namespace import Namespace

from ..expr import EvaluationError, evaluate_string
from ..expr.functions import truthy
from ..files import BIDSFile, FileTree
from .entities import parse_filename
from .inheritance import _is_subset, merged_sidecar
from .loaders import load_columns, load_json

# Built here (the rule engine relies on these being populated).
_BUILT = {
    "events", "bval", "bvec", "channels", "aslcontext", "m0scan",
    "magnitude", "magnitude1", "coordsystem", "electrodes", "physio",
    "atlas_description", "coordsystems",
}


def build_associations(
    schema: Namespace,
    tree: FileTree,
    data_file: BIDSFile,
    source_entities: dict[str, str],
    source_suffix: str,
    source_extension: str,
    source_datatype: str = "",
) -> dict[str, Any]:
    """Return the ``associations`` mapping for one data file."""
    if not source_suffix:
        return {}
    specs = schema["meta"].get("associations", {})
    selector_context = {
        "suffix": source_suffix,
        "extension": source_extension,
        "entities": source_entities,
        "datatype": source_datatype,
    }
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        if name not in _BUILT:
            continue
        if not _spec_applies(spec.get("selectors", []), selector_context):
            continue
        if name == "coordsystems":
            # An aggregate of all coordsystem files (one per space-), with the fields
            # the EMG rules read; not a single target.
            aggregate = _build_coordsystems(schema, tree, data_file, source_entities)
            if aggregate is not None:
                out[name] = aggregate
            continue
        target = spec.get("target", {})
        found = _find_target(
            schema,
            tree,
            data_file,
            source_entities,
            str(target.get("suffix", source_suffix)),
            _as_list(target.get("extension")),
            bool(spec.get("inherit", False)),
        )
        if found is None:
            continue
        out[name] = _association_object(schema, tree, found)
    return out


def _build_coordsystems(
    schema: Namespace,
    tree: FileTree,
    data_file: BIDSFile,
    source_entities: dict[str, str],
) -> dict[str, Any] | None:
    """Collect every applicable ``coordsystem`` JSON (one per ``space-``) and expose
    ``paths`` / ``spaces`` / ``ParentCoordinateSystems`` (the EMG rules read these).

    A coordsystem matches when its entities are a subset of the source's, except the
    ``space`` entity may differ (the target allows it), mirroring the reference's
    ``targetEntities=['space']`` walk.
    """
    found: list[tuple[BIDSFile, dict[str, str]]] = []
    for dir_relpath in tree.ancestor_dirs(data_file.relpath):  # inherit up the tree
        for candidate in tree.files_in(dir_relpath):
            cand_entities, cand_suffix, cand_ext = parse_filename(schema, candidate.name)
            if cand_suffix != "coordsystem" or cand_ext != ".json":
                continue
            if all(source_entities.get(k) == v or k == "space" for k, v in cand_entities.items()):
                found.append((candidate, cand_entities))
    if not found:
        return None
    parents: list[str] = []
    for candidate, _entities in found:
        data = load_json(candidate)
        parent = data.get("ParentCoordinateSystem") if isinstance(data, dict) else None
        if parent:
            parents.append(parent)
    return {
        "paths": ["/" + candidate.relpath for candidate, _ in found],
        "spaces": [ent["space"] for _f, ent in found if "space" in ent],
        "ParentCoordinateSystems": parents,
    }


def _spec_applies(selectors: list[str], context: dict[str, Any]) -> bool:
    for selector in selectors:
        try:
            if not truthy(evaluate_string(selector, context)):
                return False
        except EvaluationError:
            return False
    return True


def _find_target(
    schema: Namespace,
    tree: FileTree,
    data_file: BIDSFile,
    source_entities: dict[str, str],
    target_suffix: str,
    target_extensions: list[str],
    inherit: bool,
) -> BIDSFile | None:
    """The closest file matching the target suffix/extension with a subset of the
    source's entities. Walks up the tree when the association inherits."""
    dirs = tree.ancestor_dirs(data_file.relpath) if inherit else [data_file.parent]
    for dir_relpath in dirs:  # closest first
        best: BIDSFile | None = None
        best_specificity = -1
        for candidate in tree.files_in(dir_relpath):
            if candidate.relpath == data_file.relpath:
                continue
            cand_entities, cand_suffix, cand_ext = parse_filename(schema, candidate.name)
            if target_suffix and cand_suffix != target_suffix:
                continue
            if target_extensions and cand_ext not in target_extensions:
                continue
            if not _is_subset(cand_entities, source_entities):
                continue
            if len(cand_entities) > best_specificity:
                best, best_specificity = candidate, len(cand_entities)
        if best is not None:
            return best
    return None


def _association_object(schema: Namespace, tree: FileTree, found: BIDSFile) -> Any:
    """Build the object exposed under ``associations.<name>`` for a found file."""
    name = found.name
    path = "/" + found.relpath
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        columns = load_columns(found, max_rows=-1)
        n_rows = max((len(values) for values in columns.values()), default=0)
        obj: dict[str, Any] = dict(columns)
        obj.update(
            n_rows=n_rows,
            n_cols=len(columns),
            sidecar=merged_sidecar(schema, tree, found),
            path=path,
        )
        return obj
    if name.endswith(".bval") or name.endswith(".bvec"):
        return _numeric_matrix(found, path)
    if name.endswith(".json"):
        data = load_json(found)
        data = dict(data) if isinstance(data, dict) else {}
        data["path"] = path
        return data
    # A plain data file (e.g. m0scan, magnitude): only existence/path matters.
    return {"path": path}


def _numeric_matrix(found: BIDSFile, path: str) -> dict[str, Any]:
    """Parse a whitespace-delimited ``.bval``/``.bvec`` into values + shape."""
    try:
        text = found.read_text()
    except OSError:
        return {"values": [], "n_rows": 0, "n_cols": 0, "path": path}
    rows = [line.split() for line in text.splitlines() if line.strip()]
    values: list[float] = []
    for row in rows:
        for token in row:
            try:
                values.append(float(token))
            except ValueError:
                pass
    return {
        "values": values,
        "n_rows": len(rows),
        "n_cols": len(rows[0]) if rows else 0,
        "path": path,
    }


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)

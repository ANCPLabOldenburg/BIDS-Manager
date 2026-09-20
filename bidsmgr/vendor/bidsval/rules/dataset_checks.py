"""Dataset-level checks that look across files, ported from the reference validator.

Unlike the per-file checks, these need the whole dataset at once:

* ``CASE_COLLISION`` - two files whose paths differ only by letter case (a hazard on
  case-insensitive filesystems);
* ``SIDECAR_WITHOUT_DATAFILE`` - a JSON sidecar that applies to no data file;
* ``UNUSED_STIMULUS`` - a file in ``stimuli/`` referenced by no ``events.tsv``.

The last two rely on a "viewed" set gathered while the per-file contexts are built
(which sidecars a data file used through inheritance or association, which stimuli an
events table referenced), so a file counts as used exactly when the reference
validator would count it.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..files import BIDSFile, FileTree
from ..issues import Issue, Severity
from .citation import citation_checks

# JSON files that legitimately stand alone (no data file), never reported.
_STANDALONE_JSON = {"dataset_description.json", "genetic_info.json"}


def dataset_checks(
    tree: FileTree, files: list[BIDSFile], viewed_json: set[str], viewed_stimuli: set[str]
) -> list[Issue]:
    """Run the dataset-wide checks over the validated ``files``."""
    issues: list[Issue] = []
    issues += _case_collisions(files)
    issues += _sidecar_without_datafile(files, viewed_json)
    issues += _unused_stimulus(tree, viewed_stimuli)
    issues += citation_checks(tree)
    return issues


def _sidecar_without_datafile(files: list[BIDSFile], viewed_json: set[str]) -> list[Issue]:
    issues: list[Issue] = []
    for bids_file in files:
        name = bids_file.name
        if not name.endswith(".json") or name in _STANDALONE_JSON:
            continue
        # ``*_description.json`` (atlas / segmentation descriptors) describe an entity,
        # not a single data file, so they never pair with one.
        if name.endswith("_description.json"):
            continue
        if bids_file.relpath in viewed_json:
            continue
        issues.append(
            Issue(
                code="SIDECAR_WITHOUT_DATAFILE",
                severity=Severity.ERROR,
                location=bids_file.relpath,
                message="this JSON sidecar applies to no data file in the dataset",
                suggestion=(
                    "A sidecar must describe at least one data file it sits beside or above "
                    "(matching suffix and a subset of its entities), or be an association such as "
                    "coordsystem. Add the data file, or remove or rename the sidecar."
                ),
            )
        )
    return issues


def _case_collisions(files: list[BIDSFile]) -> list[Issue]:
    by_lower: dict[str, list[str]] = {}
    for bids_file in files:
        by_lower.setdefault(bids_file.relpath.lower(), []).append(bids_file.relpath)
    issues: list[Issue] = []
    for collisions in by_lower.values():
        if len(collisions) > 1:
            for relpath in sorted(collisions):
                others = ", ".join(sorted(set(collisions) - {relpath}))
                issues.append(
                    Issue(
                        code="CASE_COLLISION",
                        severity=Severity.ERROR,
                        location=relpath,
                        message=f"another file has the same name but a different case: {others}",
                        suggestion=(
                            "On a case-insensitive filesystem these files clash. Rename so the "
                            "paths differ by more than letter case."
                        ),
                    )
                )
    return issues


def _unused_stimulus(tree: FileTree, viewed_stimuli: set[str]) -> list[Issue]:
    stimuli = [f.relpath for f in tree.files() if f.relpath.split("/", 1)[0] == "stimuli"]
    unused = sorted(relpath for relpath in stimuli if relpath not in viewed_stimuli)
    if not unused:
        return []
    return [
        Issue(
            code="UNUSED_STIMULUS",
            severity=Severity.WARNING,
            location="stimuli",
            message=f"{len(unused)} file(s) in stimuli/ are not referenced by any events.tsv",
            suggestion=(
                "Reference each stimulus from an events.tsv 'stim_file' column, or remove the "
                "unused files. First few: " + ", ".join(unused[:5])
            ),
            affects=unused,
        )
    ]


def collect_viewed(
    schema,
    tree: FileTree,
    bids_file: BIDSFile,
    context: Mapping[str, object],
    viewed_json: set[str],
    viewed_stimuli: set[str],
) -> None:
    """Record, from one data file's context, which sidecars and stimuli it uses."""
    from ..context.inheritance import applicable_sidecar_files

    if not bids_file.name.endswith(".json"):
        for relpath in applicable_sidecar_files(schema, tree, bids_file):
            viewed_json.add(relpath)
        associations = context.get("associations") or {}
        if isinstance(associations, Mapping):
            for obj in associations.values():
                path = obj.get("path") if isinstance(obj, Mapping) else None
                if isinstance(path, str) and path.endswith(".json"):
                    viewed_json.add(path.lstrip("/"))
        # A coordinate-system sidecar is associated with a recording in the same
        # directory even when it carries extra entities (e.g. space-), which the
        # entity-subset match above does not capture. Any recording present in the
        # directory marks the coordsystem sidecars there as used.
        for sidecar in tree.json_sidecars_in(bids_file.parent):
            if sidecar.name.endswith("_coordsystem.json") or sidecar.name == "coordsystem.json":
                viewed_json.add(sidecar.relpath)
    columns = context.get("columns") or {}
    if isinstance(columns, Mapping):
        for value in columns.get("stim_file", []):
            if value and value not in ("n/a", ""):
                viewed_stimuli.add("stimuli/" + str(value).lstrip("/"))

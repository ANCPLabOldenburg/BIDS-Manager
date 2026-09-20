"""Validate a dataset, a subject, or a single file.

These are the orchestration entry points. Each resolves a schema, indexes the
files, builds a context per file, and runs the rule engine. They return typed
results (:class:`~bidsval.report.ValidationReport`,
:class:`~bidsval.report.FileVerdict`) and never raise on an invalid dataset -
problems are recorded as findings.

Granularity:

* :func:`validate` - a whole dataset.
* :func:`validate_subject` - one subject within a dataset.
* :func:`validate_file` - one file (with the rest of the dataset available for
  inheritance and existence checks).
"""

from __future__ import annotations

from pathlib import Path

from .context import ContextBuilder
from .context.inheritance import inheritance_checks
from .files import BIDSFile, FileTree
from .files.bidsignore import load_bidsignore
from .issues import Issue, Severity
from .report import FileVerdict, ValidationReport
from .rules import apply_rules
from .rules.bespoke import bespoke_checks
from .rules.dataset_checks import collect_viewed, dataset_checks
from .rules.filenames import filename_checks
from .rules.integrity import integrity_checks
from .schema import SchemaSelector, bids_version, introspect, resolve, schema_version


def _file_tree(root: str | Path, schema_ns) -> FileTree:
    """A FileTree that knows the schema's directory-recording extensions."""
    return FileTree(root, directory_recordings=tuple(introspect.directory_recordings(schema_ns)))


def validate(
    root: str | Path,
    *,
    schema: SchemaSelector = None,
    read_headers: bool = True,
    max_rows: int = 1000,
    subjects: list[str] | None = None,
    recursive: bool = False,
) -> ValidationReport:
    """Validate the BIDS dataset at ``root``.

    ``schema`` selects the schema version (default: the bundled default).
    ``read_headers`` reads NIfTI headers for header checks (default on, needs
    nibabel; set False to skip for speed). ``subjects``, if given, restricts
    validation to those ``sub-*`` directories. ``recursive`` also validates any
    BIDS-conformant datasets under ``derivatives/`` (each on its own).
    """
    schema_ns = resolve(schema)
    tree = _file_tree(root, schema_ns)
    report = ValidationReport(
        dataset_root=Path(root),
        bids_version=bids_version(schema_ns),
        schema_version=schema_version(schema_ns),
    )

    _check_dataset_description(tree, report)
    if not tree.subjects():
        report.dataset_issues.add(
            Issue(
                code="NO_SUBJECTS",
                severity=Severity.WARNING,
                message="no sub-* directories found under the dataset root",
            )
        )

    builder = ContextBuilder(schema_ns, tree, read_headers=read_headers, max_rows=max_rows)
    files = tree.files()
    if subjects is not None:
        wanted = set(subjects)
        files = [f for f in files if f.relpath.split("/", 1)[0] in wanted]

    # Skip files the dataset (or the BIDS defaults) declares outside validation:
    # stimuli/, code/, sourcedata/, logs, hidden files, and .bidsignore entries.
    # They stay indexed in the tree (so existence checks still resolve), but are
    # not validated, matching the reference validator.
    ignore = load_bidsignore(root)
    files = [f for f in files if not ignore.match(f.relpath)]

    # As each file is validated, record which sidecars and stimuli it uses, so the
    # dataset-level checks can flag the ones nothing uses.
    viewed_json: set[str] = set()
    viewed_stimuli: set[str] = set()
    for bids_file in files:
        verdict, context = _validate_one(schema_ns, builder, bids_file)
        report.files.append(verdict)
        if context is not None:
            collect_viewed(schema_ns, tree, bids_file, context, viewed_json, viewed_stimuli)

    report.dataset_issues.extend(dataset_checks(tree, files, viewed_json, viewed_stimuli))

    report.recompute()

    if recursive:
        _validate_derivatives(root, report, schema, read_headers, max_rows)

    return report


def _validate_derivatives(
    root: str | Path,
    report: ValidationReport,
    schema: SchemaSelector,
    read_headers: bool,
    max_rows: int,
) -> None:
    """Validate each BIDS-conformant dataset under ``derivatives/`` on its own."""
    deriv_root = Path(root) / "derivatives"
    if not deriv_root.is_dir():
        return
    for sub in sorted(p for p in deriv_root.iterdir() if p.is_dir()):
        if not (sub / "dataset_description.json").is_file():
            continue  # not a BIDS derivative dataset; skip
        report.derivatives[sub.name] = validate(
            sub,
            schema=schema,
            read_headers=read_headers,
            max_rows=max_rows,
            recursive=True,
        )


def validate_subject(
    root: str | Path,
    subject: str,
    *,
    schema: SchemaSelector = None,
    read_headers: bool = True,
    max_rows: int = 1000,
) -> ValidationReport:
    """Validate a single subject. ``subject`` may be given with or without the ``sub-`` prefix."""
    sub_dir = subject if subject.startswith("sub-") else f"sub-{subject}"
    return validate(
        root,
        schema=schema,
        read_headers=read_headers,
        max_rows=max_rows,
        subjects=[sub_dir],
    )


def validate_file(
    root: str | Path,
    relpath: str,
    *,
    schema: SchemaSelector = None,
    read_headers: bool = True,
    max_rows: int = 1000,
) -> FileVerdict:
    """Validate one file within a dataset.

    The rest of the dataset is indexed so inheritance and existence checks still
    work, but only the named file's findings are returned.
    """
    schema_ns = resolve(schema)
    tree = _file_tree(root, schema_ns)
    bids_file = tree.get(relpath)
    if bids_file is None:
        verdict = FileVerdict(path=Path(relpath))
        verdict.issues.append(
            Issue(
                code="FILE_NOT_FOUND",
                severity=Severity.ERROR,
                location=relpath,
                message=f"{relpath} is not under the dataset root",
            )
        )
        verdict.recompute_severity()
        return verdict
    builder = ContextBuilder(schema_ns, tree, read_headers=read_headers, max_rows=max_rows)
    verdict, _context = _validate_one(schema_ns, builder, bids_file)
    return verdict


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_one(
    schema_ns, builder: ContextBuilder, bids_file: BIDSFile
) -> tuple[FileVerdict, dict | None]:
    verdict = FileVerdict(path=Path(bids_file.relpath))
    context: dict | None = None
    try:
        context = builder.build(bids_file)
        verdict.issues.extend(bespoke_checks(bids_file, context, read_headers=builder.read_headers))
        verdict.issues.extend(integrity_checks(bids_file, context))
        verdict.issues.extend(filename_checks(schema_ns, context, bids_file))
        verdict.issues.extend(inheritance_checks(schema_ns, builder.tree, bids_file))
        verdict.issues.extend(apply_rules(schema_ns, context))
    except Exception as error:  # never let one file abort the whole run
        verdict.issues.append(
            Issue(
                code="bidsval.internal_error",
                severity=Severity.WARNING,
                location=bids_file.relpath,
                message=f"could not fully validate this file: {error}",
            )
        )
    verdict.recompute_severity()
    return verdict, context


def _check_dataset_description(tree: FileTree, report: ValidationReport) -> None:
    # Only the file-missing case is handled here; the required/recommended fields
    # of dataset_description.json come from the schema's dataset_metadata rules
    # (evaluated when the file itself is validated), so nothing is hardcoded.
    if tree.get("dataset_description.json") is None:
        report.dataset_issues.add(
            Issue(
                code="MISSING_DATASET_DESCRIPTION",
                severity=Severity.ERROR,
                location="dataset_description.json",
                message="dataset_description.json is missing at the dataset root",
            )
        )


__all__ = ["validate", "validate_subject", "validate_file"]

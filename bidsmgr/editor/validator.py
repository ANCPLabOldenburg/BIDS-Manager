"""Schema-driven BIDS validator.

Two layers, both producing the same :class:`Issue` shape:

* **Layer 1 — schema-driven** (always on, no external deps beyond
  ``bidsmgr.schema``): per-file sidecar audit, TODO-placeholder
  detection, IntendedFor URI resolution, basename / entity validation,
  and dataset-level layout sanity.
* **Layer 2 — bidsschematools structural** (``--strict`` flag): wraps
  ``bidsschematools.validator.validate_bids`` and surfaces unrecognised
  paths and missing mandatory schema rules.

Both layers emit Pydantic :class:`Issue` records that the CLI prints
and the GUI binds to its editor pane (``inspector_proto/proto.py`` is
the visual reference).

Reference: architecture.md §7 (post-conversion) and the editor
layout described in ``../inspector_proto/data.py``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import bidsmgr
from .. import schema as schema_mod
from .types import (
    FieldLevel,
    FileVerdict,
    Issue,
    Severity,
    SidecarField,
    ValidationReport,
    rollup_severity,
)

log = logging.getLogger(__name__)


_BIDS_DATATYPE_NAMES: tuple[str, ...] = (
    "anat", "func", "dwi", "fmap", "perf",
    "meg", "eeg", "ieeg", "beh", "pet", "micr", "nirs",
)

# Files allowed at the dataset root that the validator should not
# treat as orphans.
_DATASET_ROOT_FILES: frozenset[str] = frozenset({
    "dataset_description.json", "participants.tsv", "participants.json",
    "README", "CHANGES", "LICENSE", "CITATION.cff", "samples.tsv",
    "samples.json", ".bidsignore",
})

# Suffixes considered top-level data atoms when validating a sidecar's
# IntendedFor URIs (the URI must point at one of these).
_INTENDED_FOR_DATA_EXTS: tuple[str, ...] = (".nii.gz", ".nii")

# bidsschematools rule names sometimes carry ``__<rule>`` to disambiguate
# context (e.g. ``EchoTime__fmap``). The actual JSON key is the prefix.
def _canonical(name: str) -> str:
    return name.split("__", 1)[0]


# Mutual-exclusion groups for required-field audit. Mirrors
# ``metadata.engine._REQUIRED_ALTERNATIVES``.
_REQUIRED_ALTERNATIVES: dict[tuple[str, str], tuple[tuple[str, ...], ...]] = {
    ("func", "bold"):  (("RepetitionTime", "VolumeTiming"),),
    ("func", "sbref"): (("RepetitionTime", "VolumeTiming"),),
    ("func", "cbv"):   (("RepetitionTime", "VolumeTiming"),),
    ("func", "phase"): (("RepetitionTime", "VolumeTiming"),),
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate(
    bids_root: Path,
    *,
    strict: bool = False,
) -> ValidationReport:
    """Validate the BIDS dataset at ``bids_root``.

    Layer 1 (schema-driven) always runs. Layer 2 (bidsschematools
    structural) runs when ``strict=True`` — it adds path-shape
    validation against the spec and is slower on large trees.

    The function never raises on invalid datasets; problems land in
    ``report.files[*].issues``, ``report.folder_issues``, and
    ``report.dataset_issues``. ``report.severity`` rolls up to the
    highest severity seen.
    """
    bids_root = Path(bids_root).resolve()
    if not bids_root.is_dir():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    report = ValidationReport(
        bids_root=bids_root,
        bidsmgr_version=str(getattr(bidsmgr, "__version__", "0.0.0")),
        bids_version=schema_mod.bids_version() if hasattr(schema_mod, "bids_version") else "",
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    # Layer 1 builds the per-file verdicts and dataset / folder issues.
    file_verdicts: dict[Path, FileVerdict] = {}
    _layer1_dataset_root(bids_root, file_verdicts, report)
    _layer1_walk_subjects(bids_root, file_verdicts, report)

    # Layer 2 — structural validation, optional.
    if strict:
        _layer2_bidsschematools(bids_root, file_verdicts, report)

    # Promote dict to a sorted list and roll up severities.
    report.files = [file_verdicts[p] for p in sorted(file_verdicts)]
    severities: list[Severity] = []
    for f in report.files:
        severities.append(f.severity)
    for issues in report.folder_issues.values():
        severities.extend(i.severity for i in issues)
    severities.extend(i.severity for i in report.dataset_issues)
    report.severity = rollup_severity(severities)

    counts = {"ok": 0, "warn": 0, "err": 0}
    for sev in severities:
        counts[sev.value] += 1
    report.counts = counts

    return report


# ---------------------------------------------------------------------------
# Public partial-validate helpers — used by the GUI's
# "Validate file" / "Validate folder" toolbar buttons.
# ---------------------------------------------------------------------------


def validate_file(bids_root: Path, file_path: Path) -> FileVerdict:
    """Run layer-1 per-file checks on a single file and return its verdict.

    Includes:
    * basename / entity validation against the schema;
    * sidecar audit for ``.json`` files (required + recommended +
      optional + deprecated field rollup, TODO placeholder detection).

    Skips layer 2 (``bidsschematools`` structural) — it's a
    dataset-wide pass that can't sensibly run on a single path in
    isolation. For full strict validation the user clicks "Validate
    dataset" with the Strict toggle on.

    ``file_path`` must live under ``bids_root``; raises ``ValueError``
    otherwise. The returned :class:`FileVerdict.path` is relative to
    ``bids_root`` so it merges cleanly with a dataset-wide report.
    """
    bids_root = Path(bids_root).resolve()
    file_path = Path(file_path).resolve()
    rel = file_path.relative_to(bids_root)
    datatype, suffix = _infer_datatype_suffix(file_path, bids_root)
    verdict = FileVerdict(path=rel, datatype=datatype, suffix=suffix)
    _check_filename_entities(file_path, bids_root, datatype, suffix, verdict)
    if file_path.name.endswith(".json"):
        _audit_sidecar(file_path, datatype, suffix, bids_root, verdict)
    return verdict


def validate_folder(
    bids_root: Path,
    folder_path: Path,
) -> list[FileVerdict]:
    """Run :func:`validate_file` on every file under ``folder_path``.

    Walks ``folder_path`` recursively, skipping dotfiles / dot-dirs
    (``.bidsmgr``, ``.git``, …) the dataset-wide validator also
    ignores. Returns one :class:`FileVerdict` per file. Like
    :func:`validate_file`, no dataset-level or folder-level issues are
    produced — those are reserved for the full dataset pass.
    """
    bids_root = Path(bids_root).resolve()
    folder_path = Path(folder_path).resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(folder_path)
    out: list[FileVerdict] = []
    for fp in sorted(folder_path.rglob("*")):
        if not fp.is_file():
            continue
        # Skip anything under a dot-dir (covers .bidsmgr, .git, …).
        try:
            rel_parts = fp.relative_to(bids_root).parts
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue
        out.append(validate_file(bids_root, fp))
    return out


# ---------------------------------------------------------------------------
# Layer 1 — dataset-root checks
# ---------------------------------------------------------------------------


def _layer1_dataset_root(
    bids_root: Path,
    file_verdicts: dict[Path, FileVerdict],
    report: ValidationReport,
) -> None:
    """Validate dataset-level files and presence of mandatory ones."""
    # Mandatory: dataset_description.json
    dd = bids_root / "dataset_description.json"
    if not dd.exists():
        report.dataset_issues.append(Issue(
            severity=Severity.ERR,
            rule_id="bids.missing_dataset_description",
            message="dataset_description.json is missing at the BIDS root",
        ))
    else:
        rel = dd.relative_to(bids_root)
        verdict = FileVerdict(path=rel)
        try:
            data = json.loads(dd.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            verdict.severity = Severity.ERR
            verdict.issues.append(Issue(
                severity=Severity.ERR,
                rule_id="bids.invalid_json",
                message=f"could not parse dataset_description.json: {exc}",
            ))
        else:
            if not isinstance(data, dict):
                verdict.severity = Severity.ERR
                verdict.issues.append(Issue(
                    severity=Severity.ERR,
                    rule_id="bids.invalid_dataset_description",
                    message="dataset_description.json is not a JSON object",
                ))
            else:
                for required in ("Name", "BIDSVersion"):
                    if required not in data:
                        verdict.issues.append(Issue(
                            severity=Severity.ERR,
                            rule_id="bids.required_field_missing",
                            message=f"missing required field {required!r}",
                            field=required,
                        ))
                        verdict.severity = Severity.ERR
                # Flag literal "TODO" placeholders (warn).
                for k, v in data.items():
                    if v == "TODO":
                        verdict.issues.append(Issue(
                            severity=Severity.WARN,
                            rule_id="bidsmgr.todo_placeholder",
                            message=f"field {k!r} contains TODO placeholder",
                            field=k,
                            fix_label="Set a real value",
                            fix_action="set_field",
                        ))
                        if verdict.severity is Severity.OK:
                            verdict.severity = Severity.WARN
        file_verdicts[rel] = verdict

    # participants.tsv → every NIfTI subject should appear in it (warn-level).
    participants = bids_root / "participants.tsv"
    if participants.exists():
        rel = participants.relative_to(bids_root)
        verdict = FileVerdict(path=rel)
        try:
            header = participants.read_text().splitlines()[0].split("\t")
        except (OSError, IndexError):
            verdict.severity = Severity.ERR
            verdict.issues.append(Issue(
                severity=Severity.ERR,
                rule_id="bids.invalid_tsv",
                message="participants.tsv could not be read",
            ))
        else:
            if "participant_id" not in header:
                verdict.severity = Severity.ERR
                verdict.issues.append(Issue(
                    severity=Severity.ERR,
                    rule_id="bids.required_column_missing",
                    message="participants.tsv missing required column 'participant_id'",
                    field="participant_id",
                ))
        file_verdicts[rel] = verdict


# ---------------------------------------------------------------------------
# Layer 1 — walk every subject / session / datatype
# ---------------------------------------------------------------------------


def _layer1_walk_subjects(
    bids_root: Path,
    file_verdicts: dict[Path, FileVerdict],
    report: ValidationReport,
) -> None:
    subjects = sorted(p for p in bids_root.glob("sub-*") if p.is_dir())
    if not subjects:
        report.dataset_issues.append(Issue(
            severity=Severity.WARN,
            rule_id="bidsmgr.no_subjects",
            message="no sub-* directories found under the BIDS root",
        ))
        return

    for sub_dir in subjects:
        ses_dirs = sorted(s for s in sub_dir.glob("ses-*") if s.is_dir())
        if ses_dirs:
            for ses_dir in ses_dirs:
                _layer1_walk_subject_root(
                    ses_dir, sub_dir, ses_dir, bids_root,
                    file_verdicts, report,
                )
        else:
            _layer1_walk_subject_root(
                sub_dir, sub_dir, None, bids_root,
                file_verdicts, report,
            )


def _layer1_walk_subject_root(
    root: Path,
    sub_dir: Path,
    ses_dir: Optional[Path],
    bids_root: Path,
    file_verdicts: dict[Path, FileVerdict],
    report: ValidationReport,
) -> None:
    """Walk one sub-XXX[/ses-YYY] folder; one file_verdict per file."""
    folder_key = str(root.relative_to(bids_root))
    folder_issues: list[Issue] = []

    # Reject orphan directories that aren't recognised BIDS datatypes.
    # Skip dotfiles / dot-dirs (.bidsmgr, .git, .DS_Store, …) — they're
    # bookkeeping, not BIDS content.
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if child.name not in _BIDS_DATATYPE_NAMES:
            # Sessions are handled at the level above; here a sub-dir
            # that's not a datatype is unexpected.
            folder_issues.append(Issue(
                severity=Severity.WARN,
                rule_id="bids.unknown_datatype_dir",
                message=f"unrecognised datatype directory: {child.name!r}",
            ))

    # Per-datatype files.
    for dt_dir in sorted(child for child in root.iterdir()
                         if child.is_dir() and child.name in _BIDS_DATATYPE_NAMES):
        for fp in sorted(dt_dir.iterdir()):
            if not fp.is_file():
                continue
            rel = fp.relative_to(bids_root)
            verdict = file_verdicts.setdefault(rel, FileVerdict(path=rel))
            datatype, suffix = _infer_datatype_suffix(fp, bids_root)
            verdict.datatype = datatype
            verdict.suffix = suffix

            _check_filename_entities(fp, bids_root, datatype, suffix, verdict)

            if fp.name.endswith(".json"):
                _audit_sidecar(fp, datatype, suffix, bids_root, verdict)

    # Hidden-file noise filter: skip .DS_Store etc. (already excluded by
    # filtering only datatype dirs above).

    if folder_issues:
        report.folder_issues[folder_key] = folder_issues


# ---------------------------------------------------------------------------
# Layer 1 — per-file checks
# ---------------------------------------------------------------------------


def _check_filename_entities(
    fp: Path,
    bids_root: Path,
    datatype: Optional[str],
    suffix: Optional[str],
    verdict: FileVerdict,
) -> None:
    """Validate the BIDS basename (entities + suffix) against the schema."""
    if not (datatype and suffix):
        return
    try:
        # validate_basename accepts the basename portion of the file name.
        verdicts = schema_mod.validate_basename(fp.name, datatype)
    except Exception as exc:  # pragma: no cover - schema bugs should surface
        verdict.issues.append(Issue(
            severity=Severity.WARN,
            rule_id="bidsmgr.schema_error",
            message=f"schema engine raised on basename {fp.name!r}: {exc}",
        ))
        if verdict.severity is Severity.OK:
            verdict.severity = Severity.WARN
        return

    for v in verdicts or []:
        # ValidationVerdict has fields: severity, kind, message, ...
        ok = bool(getattr(v, "ok", True))
        if ok:
            continue
        verdict.issues.append(Issue(
            severity=Severity.ERR,
            rule_id=f"bids.basename.{getattr(v, 'kind', 'invalid')}",
            message=str(getattr(v, "message", v)),
        ))
        verdict.severity = Severity.ERR


def _audit_sidecar(
    fp: Path,
    datatype: Optional[str],
    suffix: Optional[str],
    bids_root: Path,
    verdict: FileVerdict,
) -> None:
    """Required+recommended audit, TODO detection, sidecar_fields population.

    Builds :class:`SidecarField` rows for every schema-defined field
    (so the GUI form has a complete list), plus any extra keys the file
    actually carries. Issues are appended to ``verdict.issues``.
    """
    try:
        data = json.loads(fp.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        verdict.severity = Severity.ERR
        verdict.issues.append(Issue(
            severity=Severity.ERR,
            rule_id="bids.invalid_json",
            message=f"could not parse JSON sidecar: {exc}",
        ))
        return
    if not isinstance(data, dict):
        verdict.severity = Severity.ERR
        verdict.issues.append(Issue(
            severity=Severity.ERR,
            rule_id="bids.invalid_json",
            message="JSON sidecar is not an object",
        ))
        return

    if not (datatype and suffix):
        # Untyped sidecar (top-level JSON, etc.) — only TODO scan.
        for k, v in data.items():
            if v == "TODO":
                verdict.issues.append(Issue(
                    severity=Severity.WARN,
                    rule_id="bidsmgr.todo_placeholder",
                    message=f"field {k!r} contains TODO placeholder",
                    field=k,
                    fix_label="Set a real value",
                    fix_action="set_field",
                ))
                if verdict.severity is Severity.OK:
                    verdict.severity = Severity.WARN
        return

    # Pull the field set per level.
    seen_names: set[str] = set()
    fields: list[SidecarField] = []

    def _add_level(level: FieldLevel, getter):
        try:
            schema_fields = getter(datatype, suffix)
        except (KeyError, ValueError, AttributeError):
            schema_fields = []
        for fi in schema_fields or []:
            name = _canonical(fi.name)
            if name in seen_names:
                continue
            seen_names.add(name)
            present = name in data
            value = data.get(name) if present else None
            value_kind = _value_kind(value) if present else "missing"
            fields.append(SidecarField(
                level=level, name=name, value=value, present=present,
                value_kind=value_kind,
                description=getattr(fi, "description", None),
            ))

    _add_level(FieldLevel.REQUIRED, schema_mod.required_sidecar_fields)
    _add_level(FieldLevel.RECOMMENDED, schema_mod.recommended_sidecar_fields)
    _add_level(FieldLevel.OPTIONAL, schema_mod.optional_sidecar_fields)
    _add_level(FieldLevel.DEPRECATED, schema_mod.deprecated_sidecar_fields)

    # Keys present in the file but not in any schema list — append as OPTIONAL.
    for k, v in data.items():
        if k in seen_names:
            continue
        seen_names.add(k)
        fields.append(SidecarField(
            level=FieldLevel.OPTIONAL, name=k, value=v, present=True,
            value_kind=_value_kind(v),
        ))

    verdict.sidecar_fields = fields

    # Required-field audit, with mutual-exclusivity respected.
    required_names: set[str] = {
        f.name for f in fields if f.level is FieldLevel.REQUIRED
    }
    for alternatives in _REQUIRED_ALTERNATIVES.get((datatype, suffix), ()):
        if any(alt in data for alt in alternatives):
            required_names -= set(alternatives)
    for name in sorted(required_names):
        if name not in data:
            verdict.issues.append(Issue(
                severity=Severity.ERR,
                rule_id="bids.required_sidecar_field_missing",
                message=f"missing required field {name!r}",
                field=name,
                fix_label=f"Add {name}",
                fix_action="add_required_field",
            ))
            verdict.severity = Severity.ERR

    # Recommended audit (advisory).
    recommended_names: list[str] = [
        f.name for f in fields if f.level is FieldLevel.RECOMMENDED
    ]
    for name in recommended_names:
        if name not in data:
            verdict.issues.append(Issue(
                severity=Severity.WARN,
                rule_id="bids.recommended_sidecar_field_missing",
                message=f"missing recommended field {name!r}",
                field=name,
                fix_label=f"Add {name}",
                fix_action="add_recommended_field",
            ))
            if verdict.severity is Severity.OK:
                verdict.severity = Severity.WARN

    # TODO placeholder detection.
    for k, v in data.items():
        if v == "TODO":
            verdict.issues.append(Issue(
                severity=Severity.WARN,
                rule_id="bidsmgr.todo_placeholder",
                message=f"field {k!r} contains TODO placeholder",
                field=k,
                fix_label="Set a real value",
                fix_action="set_field",
            ))
            if verdict.severity is Severity.OK:
                verdict.severity = Severity.WARN

    # IntendedFor URIs must resolve.
    intended = data.get("IntendedFor")
    if isinstance(intended, list):
        for entry in intended:
            if not isinstance(entry, str):
                continue
            target = _resolve_intended_for(entry, fp, bids_root)
            if target is not None and not target.exists():
                verdict.issues.append(Issue(
                    severity=Severity.ERR,
                    rule_id="bids.intended_for_unresolved",
                    message=f"IntendedFor entry does not resolve: {entry}",
                    field="IntendedFor",
                ))
                verdict.severity = Severity.ERR


# ---------------------------------------------------------------------------
# Layer 2 — bidsschematools structural
# ---------------------------------------------------------------------------


def _layer2_bidsschematools(
    bids_root: Path,
    file_verdicts: dict[Path, FileVerdict],
    report: ValidationReport,
) -> None:
    """Run ``bidsschematools.validator.validate_bids`` and translate findings.

    Layer 2 is structural: it checks whether each on-disk path matches a
    BIDS schema regex, and whether mandatory schema rules are
    satisfied. JSON content is NOT inspected here (layer 1 does that).
    """
    try:
        import bidsschematools
        from bidsschematools.validator import validate_bids
    except ImportError:
        report.dataset_issues.append(Issue(
            severity=Severity.WARN,
            rule_id="bidsmgr.bst_unavailable",
            message="bidsschematools not installed; --strict layer skipped",
        ))
        return

    # bidsschematools logs "No BIDS reference root provided." at WARNING.
    # That's expected for our usage; quiet its logger for the duration.
    bst_logger = logging.getLogger(getattr(bidsschematools, "__name__", "bidsschematools"))
    prev_level = bst_logger.level
    bst_logger.setLevel(logging.ERROR)
    try:
        try:
            out = validate_bids([str(bids_root)], suppress_errors=True)
        except Exception as exc:
            report.dataset_issues.append(Issue(
                severity=Severity.WARN,
                rule_id="bidsmgr.bst_failed",
                message=f"bidsschematools validate_bids raised: {exc}",
            ))
            return
    finally:
        bst_logger.setLevel(prev_level)

    matched: set[str] = {it["path"] for it in out.get("itemwise", []) if it.get("match")}
    all_paths: list[str] = list(out.get("path_listing", []))

    for raw in sorted(set(all_paths) - matched):
        try:
            rel = Path(raw).resolve().relative_to(bids_root)
        except ValueError:
            continue
        # Skip dotfiles bidsschematools may have included anyway.
        if any(part.startswith(".") for part in rel.parts):
            continue
        verdict = file_verdicts.setdefault(rel, FileVerdict(path=rel))
        verdict.issues.append(Issue(
            severity=Severity.WARN,
            rule_id="bids.unknown_path",
            message="path does not match any BIDS schema rule",
        ))
        if verdict.severity is Severity.OK:
            verdict.severity = Severity.WARN

    # Mandatory rules with no satisfying path → dataset-level error.
    rule_to_match = {it["regex"]: it["match"] for it in out.get("itemwise", [])
                     if it.get("match")}
    for rule in out.get("schema_tracking", []):
        if not rule.get("mandatory"):
            continue
        if rule["regex"] not in rule_to_match:
            report.dataset_issues.append(Issue(
                severity=Severity.ERR,
                rule_id="bids.missing_mandatory_path",
                message=(
                    f"no path satisfies mandatory schema rule: "
                    f"{rule['regex']}"
                ),
            ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _value_kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value == "TODO":
        return "todo"
    if isinstance(value, str):
        return "string"
    return "string"


def _infer_datatype_suffix(
    json_path: Path, bids_root: Path,
) -> tuple[Optional[str], Optional[str]]:
    try:
        rel_parts = json_path.relative_to(bids_root).parts
    except ValueError:
        return None, None

    datatype: Optional[str] = None
    for part in rel_parts:
        if part in _BIDS_DATATYPE_NAMES:
            datatype = part
            break

    stem = json_path.name
    for ext in (".nii.gz", ".nii", ".json", ".tsv", ".tsv.gz", ".bval", ".bvec"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    suffix: Optional[str] = None
    for tok in reversed(stem.split("_")):
        if tok and "-" not in tok:
            suffix = tok
            break
    return datatype, suffix


_BIDS_URI_RE = re.compile(r"^bids::(?P<rel>.+)$")


def _resolve_intended_for(entry: str, sidecar: Path, bids_root: Path) -> Optional[Path]:
    """Return the absolute Path the IntendedFor entry points at.

    Supports two forms (BIDS spec):
    * ``bids::sub-X[/ses-Y]/datatype/file`` (1.6+ URI form, recommended).
    * legacy relative path: ``ses-Y/func/...`` (relative to the subject root).
    """
    m = _BIDS_URI_RE.match(entry.strip())
    if m:
        return (bids_root / m.group("rel")).resolve()

    # Legacy relative form: relative to the subject directory.
    rel = entry.strip()
    if not rel:
        return None
    # Walk up to sub-XXX (sidecar's first ancestor named sub-*).
    sub = sidecar.parent
    while sub.parent != sub and not sub.name.startswith("sub-"):
        sub = sub.parent
    if not sub.name.startswith("sub-"):
        return None
    return (sub / rel).resolve()


__all__ = ["validate"]

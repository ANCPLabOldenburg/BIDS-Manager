"""``bidsmgr-validate`` — schema-driven BIDS validation.

Runs the editor validator against a BIDS root (or every BIDS root
under a parent directory, like ``bidsmgr-convert``'s ``<bids_parent>``
shape). Always writes a JSON report at
``<bids_root>/.bidsmgr/validation_report.json`` so the GUI can pick
it up. Exit code reflects the dataset-wide severity:

* ``0`` — only ``ok``/``warn`` verdicts (or ``warn`` with ``--strict-warn``).
* ``1`` — at least one ``err``, OR any ``warn`` when ``--strict-warn``.

The ``--strict`` flag turns on Layer 2 (``bidsschematools.validator``)
in addition to Layer 1 (schema-driven, always on).

Architectural rule: orchestration is straight-line code here; the
validator itself (``bidsmgr.editor.validator.validate``) is a single
function returning Pydantic data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

from ..editor import Severity, ValidationReport, render_html, validate

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_validate_cli(
    target: Path,
    *,
    dataset: Optional[str] = None,
    strict: bool = False,
    strict_warn: bool = False,
    write_report: bool = True,
    html_report: bool = False,
) -> int:
    """Validate every BIDS root under ``target``.

    Returns 0 on success, 1 if any root has errors (or warnings under
    ``--strict-warn``), 2 if ``target`` doesn't exist or no roots
    were found.
    """
    target = Path(target)
    if not target.is_dir():
        log.error("not a directory: %s", target)
        return 2

    bids_roots = list(_iter_bids_roots(target, dataset=dataset))
    if not bids_roots:
        log.warning("no BIDS roots found under %s", target)
        return 2

    overall_failed = 0
    for bids_root in bids_roots:
        try:
            report = validate(bids_root, strict=strict)
        except Exception:
            log.exception("validator raised on %s", bids_root)
            overall_failed += 1
            continue

        if write_report:
            _write_validation_report(bids_root, report)
        if html_report:
            _write_html_report(bids_root, report)

        _print_summary(bids_root, report)

        if _is_failing(report.severity, strict_warn=strict_warn):
            overall_failed += 1

    return 0 if overall_failed == 0 else 1


def _iter_bids_roots(
    target: Path, *, dataset: Optional[str] = None,
) -> Iterable[Path]:
    """Yield directories that look like BIDS roots under ``target``."""
    if dataset:
        candidate = target / dataset
        if _looks_like_bids_root(candidate):
            yield candidate
        else:
            log.warning(
                "no BIDS root at %s (looking for at least one sub-*/ child)",
                candidate,
            )
        return

    if _looks_like_bids_root(target):
        yield target
        return

    for child in sorted(target.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in {"__pycache__"}:
            continue
        if _looks_like_bids_root(child):
            yield child


def _looks_like_bids_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(p.is_dir() and p.name.startswith("sub-") for p in path.iterdir())


def _is_failing(severity: Severity, *, strict_warn: bool) -> bool:
    if severity is Severity.ERR:
        return True
    if strict_warn and severity is Severity.WARN:
        return True
    return False


# ---------------------------------------------------------------------------
# Report writing & CLI output
# ---------------------------------------------------------------------------


def _write_validation_report(bids_root: Path, report: ValidationReport) -> None:
    """Persist the full Pydantic ``ValidationReport`` next to the dataset."""
    out_dir = bids_root / ".bidsmgr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "validation_report.json"
    payload = report.model_dump(mode="json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_html_report(bids_root: Path, report: ValidationReport) -> None:
    """Persist a self-contained HTML validation report.

    Lives next to the JSON report at
    ``<bids_root>/.bidsmgr/validation_report.html``. Inline-CSS only;
    safe to copy or zip and share.
    """
    out_dir = bids_root / ".bidsmgr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "validation_report.html"
    out.write_text(render_html(report), encoding="utf-8")
    log.info("html report written to %s", out)


def _print_summary(bids_root: Path, report: ValidationReport) -> None:
    print(f"\n=== {bids_root} ===")
    print(
        f"  severity: {report.severity.value.upper()}  "
        f"(ok={report.counts.get('ok', 0)} "
        f"warn={report.counts.get('warn', 0)} "
        f"err={report.counts.get('err', 0)})"
    )

    if report.dataset_issues:
        print(f"  dataset-level issues ({len(report.dataset_issues)}):")
        for issue in report.dataset_issues[:20]:
            print(f"    [{issue.severity.value}] {issue.rule_id}: {issue.message}")
        if len(report.dataset_issues) > 20:
            print(f"    ... and {len(report.dataset_issues) - 20} more")

    if report.folder_issues:
        print(f"  folder-level issues:")
        for folder, issues in list(report.folder_issues.items())[:20]:
            for issue in issues[:5]:
                print(f"    [{issue.severity.value}] {folder}: {issue.message}")

    bad_files = [f for f in report.files if f.severity is not Severity.OK]
    if bad_files:
        print(f"  files with issues ({len(bad_files)}):")
        for f in bad_files[:30]:
            print(f"    [{f.severity.value}] {f.path}")
            for issue in f.issues[:5]:
                print(f"      - {issue.rule_id}: {issue.message}")
            if len(f.issues) > 5:
                print(f"      ... and {len(f.issues) - 5} more")
        if len(bad_files) > 30:
            print(f"    ... and {len(bad_files) - 30} more")


# ---------------------------------------------------------------------------
# argparse entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-validate",
        description=(
            "Schema-driven validation for a BIDS dataset. Runs Layer 1 "
            "(bidsmgr.schema-based) by default; add --strict to also "
            "run Layer 2 (bidsschematools structural validation)."
        ),
    )
    parser.add_argument(
        "target",
        help="BIDS root, or a parent containing one or more BIDS roots.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="When `target` is a parent, limit to this single dataset name.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=(
            "Also run Layer 2: bidsschematools structural validation. "
            "Adds path-shape checks against the BIDS spec and is slower "
            "on large trees."
        ),
    )
    parser.add_argument(
        "--strict-warn", action="store_true",
        help=(
            "Treat warnings as errors for the exit code. By default the "
            "command exits 1 only when there is at least one error; "
            "with this flag, any warning also fails."
        ),
    )
    parser.add_argument(
        "--no-report",
        dest="write_report", action="store_false",
        help=(
            "Skip writing <bids_root>/.bidsmgr/validation_report.json. "
            "By default the JSON report is always written so the GUI "
            "and CI tooling can pick it up after the run."
        ),
    )
    parser.set_defaults(write_report=True)
    parser.add_argument(
        "--html",
        dest="html_report", action="store_true",
        help=(
            "In addition to the JSON report, write a self-contained "
            "HTML report at <bids_root>/.bidsmgr/validation_report.html. "
            "Inline CSS, no external assets — safe to share or archive. "
            "Issues are colour-coded green/amber/red and grouped by "
            "scope (dataset / folder / file)."
        ),
    )
    parser.set_defaults(html_report=False)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG)",
    )

    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    return run_validate_cli(
        Path(args.target),
        dataset=args.dataset,
        strict=args.strict,
        strict_warn=args.strict_warn,
        write_report=args.write_report,
        html_report=args.html_report,
    )


if __name__ == "__main__":
    sys.exit(main())

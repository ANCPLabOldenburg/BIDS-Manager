"""``bidsmgr-metadata`` — generate dataset-level BIDS metadata.

Runs against the output of ``bidsmgr-convert``. Either point at a single
BIDS root (the directory that contains ``sub-*/`` and ``dataset_description.json``)
or at a parent of multiple BIDS roots (the same ``<bids_parent>`` shape
``bidsmgr-convert`` writes to). When pointed at a parent, every
sub-directory that looks like a BIDS root is processed; ``--dataset NAME``
limits to one.

Architectural notes (rule 3 — no Pipeline orchestrator): orchestration is
straight-line code here; the engine itself
(``bidsmgr.metadata.engine.run_metadata``) is a single function.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence, Iterable, Optional

from ..metadata import DatasetMetadata, MetadataReport, run_metadata

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_metadata_cli(
    target: Path,
    *,
    dataset: Optional[str] = None,
    inventory_tsv: Optional[Path] = None,
    name: Optional[str] = None,
    bids_version: Optional[str] = None,
    license: Optional[str] = None,
    authors: Optional[list[str]] = None,
    acknowledgements: Optional[str] = None,
    how_to_acknowledge: Optional[str] = None,
    funding: Optional[list[str]] = None,
    ethics_approvals: Optional[list[str]] = None,
    references_and_links: Optional[list[str]] = None,
    dataset_doi: Optional[str] = None,
    fill_todos: bool = False,
    write_report: bool = True,
    participants_file: Optional[Path] = None,
    phenotype_files: Optional[list[Path]] = None,
    datasets: Optional[Sequence[str]] = None,
) -> int:
    """Run the metadata engine on every BIDS root under ``target``.

    Returns 0 if every root processed cleanly, 1 if any errored.

    ``datasets`` names the roots this run is allowed to touch. Without it the
    walk finds every BIDS root under ``target``, which is fine for a deliberate
    command over a folder of datasets and wrong for the step that runs after a
    conversion: projects are normally kept side by side, so the parent of one is
    the home of all of them, and this run's dataset name was written into every
    one of its neighbours.

    ``participants_file`` is an optional demographics spreadsheet whose
    ``age`` / ``sex`` / ``handedness`` columns override the inventory; it is
    applied to every BIDS root. ``phenotype_files`` are measure tables written
    to ``phenotype/`` in each root.
    """
    target = Path(target)
    if not target.is_dir():
        log.error("not a directory: %s", target)
        return 2

    bids_roots = list(_iter_bids_roots(target, dataset=dataset, datasets=datasets))
    if not bids_roots:
        log.warning("no BIDS roots found under %s", target)
        return 0

    n_failed = 0
    for bids_root in bids_roots:
        meta = DatasetMetadata(
            # Omit name when unset so the engine preserves an existing
            # dataset_description Name (e.g. from `bidsmgr create`) instead of
            # overwriting it with the folder name.
            **({"name": name} if name else {}),
            # Omit when unset so DatasetMetadata's schema-sourced default
            # applies (no hardcoded BIDS version anywhere).
            **({"bids_version": bids_version} if bids_version else {}),
            license=license,
            authors=list(authors or []),
            acknowledgements=acknowledgements,
            how_to_acknowledge=how_to_acknowledge,
            funding=list(funding or []),
            ethics_approvals=list(ethics_approvals or []),
            references_and_links=list(references_and_links or []),
            dataset_doi=dataset_doi,
        )
        try:
            report = run_metadata(
                bids_root,
                inventory_tsv=inventory_tsv,
                dataset_meta=meta,
                fill_todos=fill_todos,
                write_report=write_report,
                participants_file=participants_file,
                phenotype_files=phenotype_files,
            )
        except Exception:
            log.exception("metadata engine failed on %s", bids_root)
            n_failed += 1
            continue
        _print_report(bids_root, report)

    return 0 if n_failed == 0 else 1


def _iter_bids_roots(
    target: Path,
    *,
    dataset: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
) -> Iterable[Path]:
    """Yield directories that look like BIDS roots under ``target``.

    A "BIDS root" is a directory that contains at least one ``sub-*``
    child. ``target`` itself is checked first; if it qualifies it's
    yielded and we stop. Otherwise immediate subdirectories are checked.
    """
    if datasets is not None:
        # Only the roots this run produced. A caller that says which datasets it
        # means is never given somebody else's.
        for name in datasets:
            candidate = target if target.name == name else target / name
            if _looks_like_bids_root(candidate):
                yield candidate
        return

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


def _print_report(bids_root: Path, report: MetadataReport) -> None:
    print(f"\n=== {bids_root} ===")
    print(f"  files written: {len(report.files_written)}")
    for p in report.files_written:
        try:
            rel = p.relative_to(bids_root)
        except ValueError:
            rel = p
        print(f"    - {rel}")
    if report.sidecar_fills:
        print(f"  sidecar fills: {len(report.sidecar_fills)}")
        for fill in report.sidecar_fills:
            try:
                rel = fill.sidecar.relative_to(bids_root)
            except ValueError:
                rel = fill.sidecar
            print(f"    - {rel}: {sorted(fill.fields)}")
    if report.todo_fills:
        total = sum(len(f.fields) for f in report.todo_fills)
        print(
            f"  TODO placeholders inserted: {total} fields across "
            f"{len(report.todo_fills)} files"
        )
    if report.missing_required:
        print(f"  missing REQUIRED ({len(report.missing_required)}):")
        for msg in report.missing_required[:20]:
            print(f"    - {msg}")
        if len(report.missing_required) > 20:
            print(f"    ... and {len(report.missing_required) - 20} more")
    if report.missing_recommended:
        print(
            f"  missing recommended ({len(report.missing_recommended)}, advisory)"
        )
    if report.warnings:
        print(f"  warnings ({len(report.warnings)}):")
        for msg in report.warnings:
            print(f"    - {msg}")


# ---------------------------------------------------------------------------
# argparse entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-metadata",
        description=(
            "Generate dataset-level BIDS metadata after `bidsmgr-convert`. "
            "Pass either a single BIDS root or the parent directory used "
            "by `bidsmgr-convert`; every BIDS root underneath is processed."
        ),
    )
    parser.add_argument(
        "target", nargs="?", default=None,
        help="BIDS root, or a parent containing one or more BIDS roots "
             "(omit when using --project).",
    )
    parser.add_argument(
        "--project", default=None, type=Path,
        help=(
            "Run on a project dataset folder: the target is the project root "
            "and the inventory is resolved from its active scan version "
            "(so demographics / phenotype / participants flow through). "
            "Supersedes the positional target."
        ),
    )
    parser.add_argument(
        "--version", default=None,
        help=(
            "With --project, take the inventory from a specific scan version "
            "instead of the latest (version id or index; see bidsmgr-project)."
        ),
    )
    parser.add_argument(
        "--dataset", default=None,
        help="When `target` is a parent, limit to this single dataset name.",
    )
    parser.add_argument(
        "--inventory-tsv", default=None, type=Path,
        help=(
            "Optional inventory TSV produced by `bidsmgr-scan`. Used to "
            "enrich participants.tsv with demographics; without it, "
            "demographic columns default to 'n/a'."
        ),
    )
    parser.add_argument(
        "--participants", default=None, type=Path, dest="participants_file",
        help=(
            "Optional participants spreadsheet (TSV/CSV/XLSX/ODS) keyed by a "
            "'participant_id' column. Its 'age' / 'sex' / 'handedness' columns "
            "override the inventory-derived demographics in participants.tsv."
        ),
    )
    parser.add_argument(
        "--phenotype", action="append", default=None, type=Path,
        dest="phenotype_files",
        help=(
            "Phenotype measure table (TSV/CSV/XLSX/ODS) keyed by "
            "'participant_id'. Repeat for each instrument; each is written to "
            "phenotype/<measure>.tsv + .json."
        ),
    )
    parser.add_argument(
        "--name", default=None,
        help="Dataset Name (defaults to the BIDS root directory name).",
    )
    parser.add_argument(
        "--bids-version", default=None,
        help="BIDS version for dataset_description.json (defaults to the "
             "bundled schema's version).",
    )
    parser.add_argument("--license", default=None)
    parser.add_argument(
        "--author", action="append", dest="authors", default=None,
        help="Repeat for each author.",
    )
    parser.add_argument("--acknowledgements", default=None)
    parser.add_argument("--how-to-acknowledge", default=None)
    parser.add_argument("--funding", action="append", default=None)
    parser.add_argument("--ethics-approvals", action="append", default=None)
    parser.add_argument("--references-and-links", action="append", default=None)
    parser.add_argument("--dataset-doi", default=None)
    parser.add_argument(
        "--fill-todos",
        action="store_true",
        help=(
            "For every sidecar with a missing required or recommended "
            "field (and for missing recommended fields of "
            "dataset_description.json), write the literal string "
            "\"TODO\" as the value. Existing values are never "
            "overwritten. Lets you sweep through the BIDS root and fill "
            "the placeholders by hand later."
        ),
    )
    parser.add_argument(
        "--no-report",
        dest="write_report",
        action="store_false",
        help=(
            "Skip writing <bids_root>/.bidsmgr/metadata_report.json. "
            "By default the JSON report is always written so the GUI "
            "(and CI tooling) can pick it up after the run."
        ),
    )
    parser.set_defaults(write_report=True)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG)",
    )

    parser.add_argument(
        "--schema", default=None, metavar="VERSION",
        help=(
            "Which BIDS version to work to (for example 1.11.1). Decides which "
            "fields are declared and which entities a filename may carry. "
            "Default: the newest version shipped."
        ),
    )

    args = parser.parse_args(argv)
    # One version for the whole run, adopted before any schema question is
    # asked. Without this the flag would reach the validator alone and a
    # dataset could be filled in against one version and checked against
    # another.
    from ..schema import set_active_version
    set_active_version(args.schema)

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    target = args.target
    inventory_tsv = args.inventory_tsv
    if args.project is not None:
        # Project mode: target is the project root; inventory is a scan
        # version (so demographics + the scaffold's phenotype / participants
        # files are auto-discovered relative to it).
        from ..project.orchestration import find_version, latest_version
        bids_root = Path(args.project)
        target = bids_root
        if inventory_tsv is None:
            version = (
                find_version(bids_root, args.version) if args.version
                else latest_version(bids_root)
            )
            if args.version and version is None:
                parser.error(
                    f"no scan version {args.version!r} in {bids_root} "
                    f"(see `bidsmgr-project {bids_root}`)"
                )
            if version is not None:
                inventory_tsv = version.inventory
    elif not target:
        parser.error("provide a target directory, or use --project <dataset>")

    return run_metadata_cli(
        Path(target),
        dataset=args.dataset,
        inventory_tsv=inventory_tsv,
        name=args.name,
        bids_version=args.bids_version,
        license=args.license,
        authors=args.authors,
        acknowledgements=args.acknowledgements,
        how_to_acknowledge=args.how_to_acknowledge,
        funding=args.funding,
        ethics_approvals=args.ethics_approvals,
        references_and_links=args.references_and_links,
        dataset_doi=args.dataset_doi,
        fill_todos=args.fill_todos,
        write_report=args.write_report,
        participants_file=args.participants_file,
        phenotype_files=args.phenotype_files,
    )


if __name__ == "__main__":
    sys.exit(main())

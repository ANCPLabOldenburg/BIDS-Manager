"""``bidsmgr-rebuild`` — reconcile the inventory TSV's entities + display cells.

Workflow this verb supports:

1. The user runs ``bidsmgr-scan`` → TSV is created with the
   ``entities`` JSON column populated alongside derived display cells
   (``proposed_basename``, ``session``, ``task``, ``run``).
2. The user edits the inventory in a spreadsheet:
   * Power user: edit the ``entities`` JSON cell directly →
     run ``bidsmgr-rebuild`` (default ``--from entities``) to
     regenerate the basename + mirror cells.
   * Casual user: edit ``task`` / ``run`` cells →
     run ``bidsmgr-rebuild --from columns`` to regenerate the
     ``entities`` JSON, then the basename + mirrors.
3. The user runs ``bidsmgr-convert`` — which always rebuilds in memory
   first, so the conversion uses whatever the TSV says today.

The verb prints a per-row diff of what changed; ``--dry-run`` skips the
write.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from ..inventory.rebuild import (
    RebuildReport,
    rebuild_from_columns,
    rebuild_from_entities,
)

log = logging.getLogger(__name__)


def run_rebuild_cli(
    tsv: Path,
    *,
    direction: str = "entities",
    dry_run: bool = False,
) -> int:
    """Reconcile the TSV at ``tsv``.

    Parameters
    ----------
    direction
        Either ``"entities"`` (default — entities JSON is the source of
        truth, regenerate display cells) or ``"columns"`` (display
        cells are the source, regenerate entities JSON, then mirror back).
    dry_run
        When ``True``, print the diff but don't write the TSV back.
    """
    tsv = Path(tsv)
    if not tsv.is_file():
        log.error("not a file: %s", tsv)
        return 2

    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)

    if direction == "entities":
        rebuilt, report = rebuild_from_entities(df)
    elif direction == "columns":
        rebuilt, report = rebuild_from_columns(df)
    else:
        log.error("unknown --from value: %r (use 'entities' or 'columns')",
                  direction)
        return 2

    _print_summary(tsv, report, direction=direction, dry_run=dry_run)

    if dry_run:
        return 0
    if report.rows_updated == 0 and report.basename_changes == 0:
        return 0

    rebuilt.to_csv(tsv, sep="\t", index=False)
    print(f"\nrebuilt TSV written to {tsv}")
    return 0


def _print_summary(
    tsv: Path, report: RebuildReport, *, direction: str, dry_run: bool,
) -> None:
    print(f"=== {tsv} ===")
    print(f"  direction: --from {direction}")
    print(f"  rows updated:     {report.rows_updated}")
    print(f"  basename changes: {report.basename_changes}")
    print(f"  mirror changes:   {report.mirror_changes}")
    if direction == "columns":
        print(f"  json repaired:    {report.json_repaired}")
    if dry_run:
        print("  (dry-run — no file written)")
    if report.diffs:
        print(f"\n  Per-row diff (first 30):")
        for d in report.diffs[:30]:
            before = d["before"][:80] + ("…" if len(d["before"]) > 80 else "")
            after = d["after"][:80] + ("…" if len(d["after"]) > 80 else "")
            print(f"    [{d['row']}] {d['field']}: {before!r} → {after!r}")
        if len(report.diffs) > 30:
            print(f"    … and {len(report.diffs) - 30} more")
    if report.warnings:
        print(f"\n  Warnings ({len(report.warnings)}):")
        for w in report.warnings[:10]:
            print(f"    - {w}")
        if len(report.warnings) > 10:
            print(f"    … and {len(report.warnings) - 10} more")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-rebuild",
        description=(
            "Reconcile the inventory TSV's entities JSON column with "
            "derived display cells (proposed_basename, session, task, run). "
            "The entities column is the source of truth for the BIDS "
            "basename; this verb regenerates whatever the user didn't "
            "edit. Pass --from columns when you edited the cells directly."
        ),
    )
    parser.add_argument("tsv", help="Inventory TSV produced by `bidsmgr-scan`")
    parser.add_argument(
        "--from",
        dest="direction",
        choices=["entities", "columns"],
        default="entities",
        help=(
            "Source of truth for this rebuild. 'entities' (default) "
            "regenerates display cells from the entities JSON. 'columns' "
            "regenerates the entities JSON from individual cells (use "
            "after editing task/run/session cells in a spreadsheet)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the diff but don't write the TSV back.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG)",
    )

    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    return run_rebuild_cli(
        Path(args.tsv),
        direction=args.direction,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())

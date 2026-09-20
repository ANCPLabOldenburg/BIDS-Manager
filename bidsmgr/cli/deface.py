"""``bidsmgr-deface`` — remove identifiable faces from anatomical images.

A head MRI contains a face, and a face can be rendered from one, so a dataset
shared with its faces intact is a dataset shared with its participants
identifiable. This blanks the face and leaves the brain alone.

Defacing is destructive, so the defaults are cautious:

* ``--dry-run`` prints exactly what would happen and writes nothing. Run it
  first; it is fast, because it reads headers rather than images.
* Every run is one entry in the Editor's history, so ``bidsmgr`` can undo it.
* Files that cannot be defaced are **listed with a reason**, not silently
  passed over. A user told "4 images defaced" and not told that three were
  skipped has a dataset with a face in it and no reason to suspect one.

The engines ship with BIDS Manager: niimath for defacing and the atlas
skull strip, brainchop for mindgrab.

Exit codes: ``0`` nothing went wrong, ``1`` at least one image failed (in
which case nothing was changed, because the whole run rolls back), ``2``
defacing is unavailable or the arguments were wrong.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from ..deface import engines
from ..deface.derivatives import PIPELINE
from ..deface.apply import deface_dataset
from ..deface.run import available, unavailable_reason
from ..deface.select import Selection, walk

log = logging.getLogger(__name__)


def _format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def _report(selection: Selection, eng: engines.Engine) -> None:
    """Print candidates and skips. The skips are the half people forget."""
    if selection.candidates:
        print(f"Would deface {len(selection.candidates)} image(s) "
              f"with {eng.label} ({_format_size(selection.total_bytes)}):")
        for c in selection.candidates:
            note = ""
            if c.previous_engine:
                note = f"  [already defaced with {c.previous_engine}]"
            elif c.foreign_methods:
                note = f"  [deidentified by {', '.join(c.foreign_methods)}]"
            print(f"  {c.relative}{note}")
    else:
        print("No images to deface.")

    if selection.skipped:
        print(f"\nSkipped {len(selection.skipped)}:")
        for s in selection.skipped:
            detail = f" ({s.detail})" if s.detail else ""
            print(f"  {s.relative}: {s.reason.value}{detail}")


def run_deface_cli(
    root: Path,
    *,
    engine_id: str,
    in_place: Optional[bool] = None,
    targets: Optional[Sequence[Path]] = None,
    dry_run: bool = False,
    keep_original: bool = False,
    quiet: bool = False,
) -> int:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    try:
        eng = engines.engine(engine_id)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    reason = unavailable_reason(eng.id)
    if reason and not dry_run:
        # A dry run only reads headers, so it is allowed to work without the
        # engine: somebody checking what would be defaced should not have to
        # install a binary first.
        print(f"error: {reason}", file=sys.stderr)
        return 2

    selection = walk(root, [Path(t) for t in targets] if targets else None)
    if not quiet or dry_run:
        _report(selection, eng)

    if dry_run:
        if reason:
            print(f"\nNote: {reason}", file=sys.stderr)
        return 0
    if not selection.candidates:
        return 0

    def progress(done: int, total: int, rel: str) -> None:
        if not quiet and rel:
            print(f"  [{done + 1}/{total}] {rel}", flush=True)

    print()
    outcome = deface_dataset(
        root,
        selection=selection,
        engine_id=eng.id,
        keep_original_in_sourcedata=keep_original,
        in_place=in_place,
        progress=progress,
    )

    if outcome.failed:
        where, why = outcome.failed[0]
        print(f"\nerror: defacing failed at {where}: {why}", file=sys.stderr)
        print("Nothing was changed; the whole run was rolled back.",
              file=sys.stderr)
        return 1

    verb = "Stripped" if eng.is_strip else "Defaced"
    print(f"\n{verb} {len(outcome.defaced)} image(s) in "
          f"{outcome.seconds:.1f}s with {eng.label}.")
    if outcome.produced:
        print(f"The results are under derivatives/{PIPELINE}/; the original "
              "scans are untouched.")
    if keep_original:
        print("The originals are in sourcedata/. They still contain the "
              "face: remove them before sharing the dataset.")
    print("Undo it in the Editor (Edit > Undo) if this was not what you wanted.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bidsmgr-deface",
        description="Remove identifiable faces from anatomical images.",
    )
    # Optional so --list-engines works on its own; checked in main().
    p.add_argument(
        "bids_root", type=Path, nargs="?", help="the BIDS dataset",
    )
    p.add_argument(
        "--target", type=Path, action="append", dest="targets",
        metavar="PATH",
        help="limit to this file or folder; repeatable. Default: the dataset",
    )
    p.add_argument(
        "--engine", default=engines.DEFAULT_ENGINE_ID,
        choices=engines.engine_ids(),
        help=(
            "which engine to use (default: %(default)s). The allineate "
            "engines remove the face; mindgrab and strip-atlas keep only the "
            "brain and write a derivative. See --list-engines."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print what would be defaced and what would be skipped, and stop",
    )
    p.add_argument(
        "--keep-original-in-sourcedata", action="store_true",
        dest="keep_original",
        help=(
            "also copy each original to sourcedata/ before defacing. Those "
            "copies still contain the face, so a dataset shared with them "
            "in place is not deidentified"
        ),
    )
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument(
        "--in-place", action="store_true",
        help=(
            "for a skull-strip engine: overwrite the original instead of "
            "writing to derivatives/. A stripped scan is not raw data, so "
            "this is not the default."
        ),
    )
    p.add_argument(
        "--list-engines", action="store_true",
        help="describe the available engines and exit",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.list_engines:
        for eng in engines.ENGINES:
            default = "  (default)" if eng.id == engines.DEFAULT_ENGINE_ID else ""
            default += "  [keeps only the brain]" if eng.is_strip else ""
            print(f"{eng.id}{default}\n    {eng.description}")
        if not available():
            print(f"\nNote: {unavailable_reason()}", file=sys.stderr)
        return 0

    if args.bids_root is None:
        build_parser().error("a BIDS dataset is required")

    return run_deface_cli(
        args.bids_root,
        engine_id=args.engine,
        in_place=True if args.in_place else None,
        targets=args.targets,
        dry_run=args.dry_run,
        keep_original=args.keep_original,
        quiet=args.quiet,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

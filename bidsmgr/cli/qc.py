"""``bidsmgr-qc``: a fast quality check of every anatomical and diffusion
image of a BIDS dataset (``bidsmgr.qc``), with nothing extra to install.

Writes ``derivatives/bidsmgr-qc/``: one JSON per image (the measures, what
each means, the findings) and ``group_<suffix>.tsv`` per suffix, and prints
the findings. The air around the head is measured only in images that are
not defaced.

Exit codes: ``0`` checked (findings or not), ``1`` an image could not be
checked, ``2`` bad arguments.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ..qc import run as R

    parser = argparse.ArgumentParser(
        prog="bidsmgr-qc",
        description="Fast quality check of anatomical and diffusion images.")
    parser.add_argument("bids_root", type=Path, nargs="?", help="The BIDS dataset.")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="Images to check (default: every T1w, T2w, FLAIR, PDw, T2starw "
                             "and dwi image).")
    parser.add_argument("--participant-label", nargs="+", default=None, metavar="LABEL",
                        help="Only these participants (with or without 'sub-').")
    parser.add_argument("--datatype", nargs="+", choices=R.DATATYPES, default=list(R.DATATYPES))
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Images checked at once.")
    parser.add_argument("--no-flip-check", action="store_true",
                        help="Skip the (experimental) check for flipped or swapped b-vectors.")
    parser.add_argument("--fast", action="store_true",
                        help="Approximate masks from the numpy fallback instead of mindgrab, "
                             "the tissue model and niimath (seconds faster, less accurate).")
    parser.add_argument("--config", type=Path, default=None, metavar="JSON",
                        help="Methods and thresholds, as --print-config writes them (any "
                             "field left out keeps its default).")
    parser.add_argument("--print-config", action="store_true",
                        help="Print the configuration that would be used, as JSON, and "
                             "exit.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Print only the summary.")
    args = parser.parse_args(argv)

    from ..qc import config as C

    cfg = C.QcConfig()
    if args.config is not None:
        try:
            cfg = C.QcConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"bidsmgr-qc: cannot read the configuration {args.config}: {exc}",
                  file=sys.stderr)
            return 2
    if args.fast:
        cfg = C.fast(cfg)
    if args.no_flip_check:
        cfg.dwi.check_flips = False
    if args.print_config:
        print(cfg.model_dump_json(indent=2))
        return 0
    if args.bids_root is None:
        parser.error("the BIDS dataset is required")

    root = args.bids_root.expanduser().resolve()
    if not (root / "dataset_description.json").is_file():
        print(f"bidsmgr-qc: {root} is not a BIDS dataset (no dataset_description.json)",
              file=sys.stderr)
        return 2
    if args.paths:
        paths = [p.expanduser().resolve() for p in args.paths]
        unknown = [p for p in paths if R.kind_of(p) is None]
        if unknown:
            print("bidsmgr-qc: no quality check for: " + ", ".join(p.name for p in unknown),
                  file=sys.stderr)
            return 2
    else:
        paths = R.find_images(root, datatypes=args.datatype,
                              participants=args.participant_label)
    if not paths:
        print("No anatomical or diffusion images to check.")
        return 0
    t0 = time.perf_counter()

    def progress(done: int, total: int, name: str) -> None:
        if not args.quiet:
            print(f"[{done}/{total}] {name}", flush=True)

    rows = R.run(root, paths, jobs=max(1, args.jobs), config=cfg, progress=progress)
    failed = 0
    print(f"\nChecked {len(rows)} image(s) in {time.perf_counter() - t0:.0f} s; results in "
          f"{(root / 'derivatives' / 'bidsmgr-qc').as_posix()}")
    for row in rows:
        serious = [f for f in row.get("findings", []) if f["level"] in ("warning", "error")]
        if any(f["key"] == "unreadable" for f in serious):
            failed += 1
        if serious and not args.quiet:
            print(f"\n{row['path']}")
            for f in serious:
                print(f"  {f['level']}: {f['title']}. {f['message']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

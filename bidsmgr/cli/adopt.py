"""``bidsmgr-adopt`` - manage a dataset BIDS Manager did not convert.

    bidsmgr-adopt <dataset>            create the project bundle and a baseline
    bidsmgr-adopt <dataset> --status   what has changed since it was adopted

The project model assumes BIDS Manager wrote the dataset. A dataset from
anywhere else has no scan version, no event log and no baseline, so the Editor
could change its files with nothing to return to. Adopting supplies the missing
first step.

Nothing outside ``.bidsmgr/`` is touched, so an adopted dataset validates
exactly as it did before.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def run_adopt(dataset: Path, *, name: Optional[str] = None) -> int:
    from ..project.adopt import NotABidsDataset, adopt

    try:
        result = adopt(dataset, name=name)
    except NotABidsDataset as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not write into {dataset}: {exc}", file=sys.stderr)
        return 1

    if result.already_managed:
        print(f"{result.root} was already managed; the baseline was refreshed.")
    else:
        print(f"Adopted {result.root}.")
    print(
        f"  {result.files} files recorded, {result.bytes_seen / 1e6:.1f} MB."
    )
    print(f"  Baseline: {result.manifest_path}")
    print(
        "  Nothing outside .bidsmgr/ was changed, so the dataset validates "
        "exactly as before."
    )
    return 0


def run_status(dataset: Path) -> int:
    from ..project.adopt import changed_since_adoption, read_manifest

    manifest = read_manifest(dataset)
    if manifest is None:
        print(
            f"{dataset} has no adoption baseline. Run bidsmgr-adopt on it "
            "first.",
            file=sys.stderr,
        )
        return 2
    changed = changed_since_adoption(dataset)
    print(f"Adopted {manifest.get('recorded_at', 'at an unknown time')}.")
    if not changed:
        print("Nothing has changed since.")
        return 0
    print(f"{len(changed)} file(s) differ from the baseline:")
    for rel in changed[:40]:
        print(f"  {rel}")
    if len(changed) > 40:
        print(f"  and {len(changed) - 40} more")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr-adopt",
        description=(
            "Bring an existing BIDS dataset under BIDS Manager's management, "
            "so its edits can be tracked and reverted."
        ),
    )
    parser.add_argument("dataset", type=Path, help="BIDS dataset folder.")
    parser.add_argument(
        "--name", default=None,
        help="Project name. Defaults to the folder name.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Report what has changed since adoption instead of adopting.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    if args.status:
        return run_status(args.dataset)
    return run_adopt(args.dataset, name=args.name)


if __name__ == "__main__":
    sys.exit(main())

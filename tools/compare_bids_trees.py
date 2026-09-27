"""Diff two BIDS trees produced by the same tool from the same sources.

Written for a dependency upgrade, where the question is not "is the output
valid" but "what moved, and can every difference be named". A converter
upgrade that renames files is fine; a converter upgrade that renames files
without anyone noticing is how a lab's dataset silently stops matching its
own analysis scripts.

Run::

    python tools/compare_bids_trees.py <before_root> <after_root>

Reports four things and deliberately nothing else:

* paths only in BEFORE  -- a file we stopped producing
* paths only in AFTER   -- a file we started producing
* sidecar keys added or removed, per file
* sidecar values changed, per file

Nothing here is a test. It is a measurement, re-runnable whenever either
side changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Keys every conversion writes about ITSELF. A version string changing is
# the upgrade working, not a finding, and leaving it in buries the real
# differences under one line per file.
SELF_DESCRIBING = {"ConversionSoftware", "ConversionSoftwareVersion"}


def tree_paths(root: Path) -> set[str]:
    """Every file under ``root``, relative, excluding our own bookkeeping."""
    out = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if ".bidsmgr/" in rel or rel.startswith(".bidsmgr"):
            continue
        out.add(rel)
    return out


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def compare(before: Path, after: Path) -> int:
    a, b = tree_paths(before), tree_paths(after)

    only_a = sorted(a - b)
    only_b = sorted(b - a)

    print(f"BEFORE {before}   {len(a)} files")
    print(f"AFTER  {after}   {len(b)} files")
    print()

    print(f"== paths only in BEFORE ({len(only_a)}) ==")
    for p in only_a:
        print(f"   - {p}")
    if not only_a:
        print("   (none)")
    print()

    print(f"== paths only in AFTER ({len(only_b)}) ==")
    for p in only_b:
        print(f"   + {p}")
    if not only_b:
        print("   (none)")
    print()

    # Sidecars present in both: what changed inside them.
    shared_json = sorted(p for p in (a & b) if p.endswith(".json"))
    added_keys: Counter = Counter()
    removed_keys: Counter = Counter()
    changed_keys: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for rel in shared_json:
        old = {k: v for k, v in read_json(before / rel).items()
               if k not in SELF_DESCRIBING}
        new = {k: v for k, v in read_json(after / rel).items()
               if k not in SELF_DESCRIBING}
        for k in new.keys() - old.keys():
            added_keys[k] += 1
            if len(examples[f"+{k}"]) < 2:
                examples[f"+{k}"].append(f"{rel}: {new[k]!r}")
        for k in old.keys() - new.keys():
            removed_keys[k] += 1
            if len(examples[f"-{k}"]) < 2:
                examples[f"-{k}"].append(f"{rel}: was {old[k]!r}")
        for k in old.keys() & new.keys():
            if old[k] != new[k]:
                changed_keys[k] += 1
                if len(examples[f"~{k}"]) < 2:
                    examples[f"~{k}"].append(
                        f"{rel}: {old[k]!r} -> {new[k]!r}"
                    )

    print(f"== sidecar fields, across {len(shared_json)} shared .json files ==")
    for title, counter, sign in (
        ("ADDED", added_keys, "+"),
        ("REMOVED", removed_keys, "-"),
        ("CHANGED", changed_keys, "~"),
    ):
        print(f"  -- {title} --")
        if not counter:
            print("     (none)")
        for key, n in counter.most_common():
            print(f"     {sign} {key:<34} {n} file(s)")
            for line in examples[f"{sign}{key}"]:
                print(f"         {line}")
    print()

    differences = len(only_a) + len(only_b) + sum(
        c.total() for c in (added_keys, removed_keys, changed_keys)
    )
    print(f"TOTAL DIFFERENCES: {differences}")
    return 0 if differences == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    args = ap.parse_args()
    for root in (args.before, args.after):
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 2
    return compare(args.before, args.after)


if __name__ == "__main__":
    raise SystemExit(main())

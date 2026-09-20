#!/usr/bin/env python3
"""Prove a built wheel carries the non-Python files it needs.

Setuptools finds ``.py`` automatically and nothing else, so package data is
opt-in and a correct source tree can still build a broken wheel. That is not
hypothetical here: the vendored bidsval schema files, 2.9 MB across six BIDS
versions, were absent from every wheel until 2026-09-14, and the tree looked
complete the whole time.

Checking the artefact rather than the checkout is the only thing that sees it.

    python tools/check_wheel.py dist/
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

# (description, glob, how many at least). Each entry is here because its
# absence would ship a broken install rather than a failing build.
REQUIRED = [
    ("vendored bidsval schemas", "bidsmgr/vendor/bidsval/schema/bundled/*.json", 6),
    ("bidsval licence",          "bidsmgr/vendor/bidsval/LICENSE", 1),
    ("bidsphysio licence",       "bidsmgr/vendor/bidsphysio/LICENSE", 1),
    ("the Qt stylesheet",        "bidsmgr/gui/*.qss", 1),
    ("app icons",                "bidsmgr/gui/assets/macos/*.png", 1),
    ("the Windows icon",         "bidsmgr/gui/assets/windows/*.ico", 1),
]


def main(argv: list[str]) -> int:
    where = Path(argv[1] if len(argv) > 1 else "dist")
    wheels = sorted(where.rglob("*.whl"))
    if not wheels:
        print(f"no wheel under {where}")
        return 1
    wheel = wheels[-1]
    names = zipfile.ZipFile(wheel).namelist()

    print(f"checking {wheel.name} ({len(names)} entries)")
    bad = []
    for label, pattern, least in REQUIRED:
        import fnmatch

        hits = [n for n in names if fnmatch.fnmatch(n, pattern)]
        mark = "ok " if len(hits) >= least else "MISSING"
        print(f"  {mark:8} {label:26} {len(hits):>3} / {least:<3} {pattern}")
        if len(hits) < least:
            bad.append(label)

    if bad:
        print("\nthe wheel is missing: " + ", ".join(bad))
        print("add a glob under [tool.setuptools.package-data] in pyproject.toml")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

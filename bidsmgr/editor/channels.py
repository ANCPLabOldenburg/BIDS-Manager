"""Writing channel status (good or bad) into a recording's ``_channels.tsv``.

The viewer lets a reader mark channels bad by clicking their names; this is
how that judgement reaches the dataset. One reversible operation through
:func:`bidsmgr.project.operations.begin_operation`, so it is in the Editor's
history and can be undone like any other edit.

BIDS: ``status`` is an optional column of ``_channels.tsv`` taking ``good``
or ``bad``. It is added when the table lacks it; every other column and the
row order are kept as they were.

Qt-free.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable

from ..project.operations import begin_operation


#: The UTF-8 byte order mark. mne-bids writes its tables with one; read as
#: plain UTF-8 it became part of the first column's name, every table was
#: refused for having "no name column", and no bad channel was ever saved.
BOM = "\ufeff"


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def has_bom(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(3) == BOM.encode("utf-8")


def status_table(path: Path, bads: Iterable[str]) -> tuple[str, int]:
    """The table's new text with ``status`` set from ``bads``, and how many
    rows change."""
    fields, rows = read_rows(path)
    if "name" not in fields:
        raise ValueError(f"{Path(path).name} has no name column")
    bad = set(bads)
    if "status" not in fields:
        fields.append("status")
    changed = 0
    for row in rows:
        new = "bad" if row.get("name") in bad else "good"
        if row.get("status", "n/a") != new:
            changed += 1
        row["status"] = new
    out = io.StringIO()
    # Written as it was found: a table that carried a byte order mark keeps
    # it, so the only difference in the file is the status column.
    if has_bom(path):
        out.write(BOM)
    writer = csv.DictWriter(out, fieldnames=fields, delimiter="\t", lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("n/a" if row.get(k) in (None, "") else row[k]) for k in fields})
    return out.getvalue(), changed


def set_bad_channels(root: Path, path: Path, bads: Iterable[str]) -> int:
    """Write ``bads`` into ``path``'s ``status`` column, every other channel
    ``good``, as one undoable operation of the dataset at ``root``. Returns
    the number of rows changed (0: nothing written)."""
    bad = sorted(set(bads))
    text, changed = status_table(Path(path), bad)
    if not changed:
        return 0
    label = (f"Mark {len(bad)} channel{'s' if len(bad) != 1 else ''} bad in "
             f"{Path(path).name}" if bad else f"Mark every channel good in {Path(path).name}")
    with begin_operation(Path(root), label) as op:
        op.write_text(Path(path), text)
    return changed


__all__ = ["BOM", "has_bom", "read_rows", "set_bad_channels", "status_table"]

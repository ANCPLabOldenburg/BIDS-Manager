"""Writing a recording's review into the dataset: bad channels and bad
segments, as one undoable operation.

Where BIDS keeps them, and how mne-bids reads them back:

* bad CHANNELS: the ``status`` column of the recording's ``_channels.tsv``
  (``good`` / ``bad``), see :mod:`bidsmgr.editor.channels`;
* bad SEGMENTS: rows of the run's ``_events.tsv`` whose ``trial_type`` is
  ``BAD_`` and a reason (``BAD_muscle``), with their onset and duration.
  ``mne_bids.read_raw_bids`` turns those rows into the recording's
  annotations, and every MNE step that rejects by annotation leaves them out.

The BAD rows of the table are REPLACED by the review's segments; every other
row (the task's events) and every column are kept as they were, sorted by
onset, the byte order mark kept if the table had one. A run with no table
gets one with the three columns BIDS requires.

Qt-free.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..project.operations import begin_operation
from .channels import BOM, has_bom, read_rows, status_table

#: The columns of a table this module has to create.
NEW_COLUMNS = ("onset", "duration", "trial_type")


def _is_bad(row: dict) -> bool:
    return str(row.get("trial_type") or "").strip().upper().startswith("BAD")


def _number(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _onset_key(row: dict) -> float:
    try:
        return float(row.get("onset", ""))
    except (TypeError, ValueError):
        return float("inf")


def events_table(path: Optional[Path], spans: Sequence, sfreq: Optional[float] = None
                 ) -> tuple[str, int]:
    """The table's new text with its BAD rows replaced by ``spans`` (objects
    with ``onset``, ``duration``, ``label``; seconds), and how many BAD rows
    differ from before."""
    path = Path(path) if path is not None else None
    if path is not None and path.exists():
        fields, rows = read_rows(path)
        bom = has_bom(path)
    else:
        fields, rows, bom = list(NEW_COLUMNS), [], False
    for column in NEW_COLUMNS:
        if column not in fields:
            fields.append(column)
    old = sorted((_number(_onset_key(r)), _number(float(r.get("duration") or 0)),
                  r.get("trial_type")) for r in rows if _is_bad(r))
    kept = [r for r in rows if not _is_bad(r)]
    new_rows = []
    for sp in spans:
        row = {f: "n/a" for f in fields}
        row["onset"] = _number(sp.onset)
        row["duration"] = _number(sp.duration)
        row["trial_type"] = str(sp.label)
        if "sample" in fields and sfreq:
            row["sample"] = str(int(round(float(sp.onset) * float(sfreq))))
        new_rows.append(row)
    new = sorted((r["onset"], r["duration"], r["trial_type"]) for r in new_rows)
    changed = len(set(old) ^ set(new))
    out = io.StringIO()
    if bom:
        out.write(BOM)
    writer = csv.DictWriter(out, fieldnames=fields, delimiter="\t", lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    for row in sorted(kept + new_rows, key=_onset_key):
        writer.writerow({k: ("n/a" if row.get(k) in (None, "") else row[k]) for k in fields})
    return out.getvalue(), changed


def save_review(root: Path, *, recording: str, channels_tsv: Optional[Path] = None,
                bads: Optional[Iterable[str]] = None, events_tsv: Optional[Path] = None,
                spans: Optional[Sequence] = None, sfreq: Optional[float] = None) -> dict:
    """Write what changed of a recording's review (bad channels into
    ``channels_tsv``, bad segments into ``events_tsv``) as ONE undoable
    operation of the dataset at ``root``. ``{"channels": n, "segments": n}``:
    rows changed in each (0 for a file not written)."""
    writes: list[tuple[Path, str]] = []
    result = {"channels": 0, "segments": 0}
    if channels_tsv is not None and bads is not None:
        text, n = status_table(Path(channels_tsv), sorted(set(bads)))
        if n:
            writes.append((Path(channels_tsv), text))
            result["channels"] = n
    if events_tsv is not None and spans is not None:
        text, n = events_table(Path(events_tsv), spans, sfreq)
        if n:
            writes.append((Path(events_tsv), text))
            result["segments"] = n
    if not writes:
        return result
    parts = []
    if bads is not None and result["channels"]:
        k = len(set(bads))
        parts.append(f"{k} bad channel{'s' if k != 1 else ''}")
    if spans is not None and result["segments"]:
        k = len(spans)
        parts.append(f"{k} bad segment{'s' if k != 1 else ''}")
    with begin_operation(Path(root), f"Review of {recording}: {', '.join(parts)}") as op:
        for path, text in writes:
            op.write_text(path, text)
    return result


__all__ = ["NEW_COLUMNS", "events_table", "save_review"]

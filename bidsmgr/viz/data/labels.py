"""Label images (atlases, segmentations): what each integer means.

BIDS describes a discrete segmentation (``*_dseg.nii.gz``) with a lookup
table beside it, ``*_dseg.tsv``: an ``index`` and a ``name`` per label, and
optionally a ``color`` (``#rrggbb``). That table is what turns "voxel = 17"
into "Left-Hippocampus" under the mouse, so it is read whenever it exists,
and a label image without one still gets one stable colour per value.

Qt-free.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..bids import full_ext, stem_of
from ..scene import LabelTable

log = logging.getLogger(__name__)

#: More distinct values than this and an integer image is a measurement
#: (counts, a quantised map), not a set of labels.
MAX_LABELS = 1000


def _hex(value: str) -> Optional[tuple[int, int, int]]:
    text = (value or "").strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def read_label_tsv(path: Path) -> Optional[LabelTable]:
    """A BIDS lookup table (``index``, ``name``, optional ``color``), or None."""
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
    except (OSError, csv.Error):
        return None
    labels: dict[int, str] = {}
    colors: dict[int, tuple[int, int, int]] = {}
    for row in rows:
        try:
            index = int(float(row.get("index", "")))
        except (TypeError, ValueError):
            continue
        labels[index] = str(row.get("name") or row.get("abbreviation") or index)
        colour = _hex(row.get("color", ""))
        if colour is not None:
            colors[index] = colour
    if not labels:
        return None
    return LabelTable(labels=labels, colors=colors)


def table_path_for(image: Path) -> Optional[Path]:
    """The lookup table that describes ``image``: the ``.tsv`` with its name,
    else one at the root of its derivative named by its ``desc`` (BIDS lets a
    pipeline share one table, ``desc-aseg_dseg.tsv``, across subjects)."""
    image = Path(image)
    if full_ext(image) not in (".nii", ".nii.gz", ".mgz", ".mgh"):
        return None
    beside = image.with_name(stem_of(image) + ".tsv")
    if beside.is_file():
        return beside
    stem = stem_of(image)
    suffix = stem.rsplit("_", 1)[-1] if "_" in stem else ""
    desc = next((p[5:] for p in stem.split("_") if p.startswith("desc-")), "")
    if not suffix:
        return None
    shared = f"desc-{desc}_{suffix}.tsv" if desc else f"{suffix}.tsv"
    for parent in list(image.parents)[:6]:
        candidate = parent / shared
        if candidate.is_file():
            return candidate
        if (parent / "dataset_description.json").is_file():
            break
    return None


def distinct_values(values: np.ndarray, limit: int = MAX_LABELS) -> Optional[np.ndarray]:
    """The distinct values of ``values`` when they are integers and there are
    at most ``limit`` of them, else None (not a label image)."""
    flat = np.asarray(values).reshape(-1)
    if flat.size > 4_000_000:
        flat = flat[:: flat.size // 4_000_000]
    if flat.dtype.kind == "f":
        flat = flat[np.isfinite(flat)]
        if flat.size and not np.all(flat == np.round(flat)):
            return None
    elif flat.dtype.kind not in "iub":
        return None
    found = np.unique(flat)
    if found.size > limit:
        return None
    return found


def generated_table(values) -> LabelTable:
    """One stable colour per value, named by the number (0 is background)."""
    from ..compute.intensity import stable_colour

    labels: dict[int, str] = {}
    colors: dict[int, tuple[int, int, int]] = {}
    for v in values:
        i = int(v)
        if i == 0:
            continue
        labels[i] = str(i)
        colors[i] = stable_colour(i)
    return LabelTable(labels=labels, colors=colors)


__all__ = ["MAX_LABELS", "distinct_values", "generated_table", "read_label_tsv",
           "table_path_for"]

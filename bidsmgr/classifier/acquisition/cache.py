"""The series' acquisition features, kept beside the inventory.

``<inventory>.tsv.series_features.json.gz`` maps each ``series_uid`` to its
:class:`~bidsmgr.classifier.acquisition.features.SeriesFeatures`, so the
classifier can score the open table again (a changed setting, a new
rulebook) in milliseconds, without reading a DICOM. Nothing identifying is
in it.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Optional

from .features import SeriesFeatures

SUFFIX = ".series_features.json.gz"


def features_path(tsv) -> Path:
    tsv = Path(tsv)
    return tsv.with_suffix(tsv.suffix + SUFFIX)


def write(tsv, features_by_uid: dict) -> Optional[Path]:
    """Write the cache; ``None`` when there is nothing to keep."""
    if not features_by_uid:
        return None
    target = features_path(tsv)
    payload = {uid: (f.to_dict() if isinstance(f, SeriesFeatures) else dict(f))
               for uid, f in features_by_uid.items()}
    with gzip.open(target, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return target


def read(tsv) -> dict[str, SeriesFeatures]:
    """The cache as ``{series_uid: SeriesFeatures}`` (``{}`` when absent)."""
    target = features_path(tsv)
    if not target.is_file():
        return {}
    try:
        with gzip.open(target, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {uid: SeriesFeatures.from_dict(d) for uid, d in data.items() if d}


__all__ = ["SUFFIX", "features_path", "read", "write"]

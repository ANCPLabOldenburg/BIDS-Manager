"""The check of the image on screen: from the viewer's own copy, computed
once and shared.

The viewer asks twice for a diffusion series: its report (Check quality) and
the plots under the series (QC plots). Both run on workers and may start
together; the first computes, the second waits for it and gets the same
result. A result is kept per file, number of volumes read and
configuration: changing a setting in Settings > Quality control computes
it again.

Qt-free.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

from . import anat, report
from .config import DEFAULT, QcConfig
from .config import key as config_key
from .types import QCResult

_LOCK = threading.Lock()
_KEY_LOCKS: dict[tuple, threading.Lock] = {}
_RESULTS: dict[tuple, QCResult] = {}
#: Results kept (the last few files looked at).
KEEP = 8


def kind_of(src, bids_ctx=None) -> Optional[str]:
    """"anat", "dwi" or None for the image on screen."""
    from ..viz import bids as VB

    if src is None or getattr(src, "is_rgb", False):
        return None
    name = Path(str(src.path)).name
    suffix = VB.suffix_of(name)
    bvals = getattr(src, "bvals", None)
    if suffix == "dwi" or (bvals is not None and getattr(src, "is_4d", False)):
        return "dwi"
    if suffix in anat.SUFFIXES and not getattr(src, "is_4d", False):
        return "anat"
    return None


def _key(src, config: Optional[QcConfig]) -> tuple:
    return (str(src.path), int(getattr(src, "loaded_frames", 1)),
            config_key(config or DEFAULT))


def cached(src, config: Optional[QcConfig] = None) -> Optional[QCResult]:
    with _LOCK:
        return _RESULTS.get(_key(src, config))


def forget(src=None) -> None:
    """Drop the cached results of ``src``, whatever their configuration
    (all of them when None)."""
    with _LOCK:
        if src is None:
            _RESULTS.clear()
        else:
            for k in [k for k in _RESULTS if k[0] == str(src.path)]:
                _RESULTS.pop(k, None)


def result_for(src, sidecar: Optional[dict] = None, *, config: Optional[QcConfig] = None,
               cancel: Optional[Callable[[], bool]] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> QCResult:
    """The check of ``src`` (a ``VolumeSource`` read whole) with ``config``,
    computed once."""
    key = _key(src, config)
    with _LOCK:
        if key in _RESULTS:
            return _RESULTS[key]
        lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    with lock:
        with _LOCK:
            if key in _RESULTS:
                return _RESULTS[key]
        kind = kind_of(src)
        if kind == "dwi":
            from . import dwi
            from .series import from_source

            series, problems = from_source(src, sidecar)
            res = dwi.check(series, problems=problems, cancel=cancel, progress=progress,
                            config=config)
        elif kind == "anat":
            import numpy as np

            from ..viz import bids as VB

            data = np.asarray(src.scale(np.asarray(src.raw_frame(0))), dtype=np.float32)
            header = None
            try:
                import nibabel as nib

                header = nib.load(str(src.path)).header
            except Exception:  # noqa: BLE001 - the header checks are optional
                header = None
            res = anat.check(data, src.affine, suffix=VB.suffix_of(Path(str(src.path)).name),
                             sidecar=sidecar or {}, path=str(src.path), header=header,
                             cancel=cancel, progress=progress, config=config)
        else:
            raise ValueError("no quality check for this image")
        with _LOCK:
            _RESULTS[key] = res
            while len(_RESULTS) > KEEP:
                _RESULTS.pop(next(iter(_RESULTS)))
        return res


def dataset_context(result: QCResult, root: Optional[Path]) -> dict:
    """Where the image's measures sit among the dataset's saved results:
    ``{metric: robust z}``, empty when fewer than three comparable images
    have been checked."""
    if root is None or not result.path:
        return {}
    rows = report.load_all(root)
    mine = report.to_json(result, root=root)
    rows = [r for r in rows if r.get("path") != mine["path"]] + [mine]
    return report.group_z(rows).get(mine["path"], {})


__all__ = ["KEEP", "cached", "dataset_context", "forget", "kind_of", "result_for"]

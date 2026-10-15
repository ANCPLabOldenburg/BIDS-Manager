"""A 4-D series the checks read volume by volume, from a file or from the
viewer's own copy (so a series on screen is not read from disk twice).

Pure data plus one reader; Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class Series:
    """``n`` volumes on one grid, read by ``frame(i)`` as float32 (x, y, z)."""

    n: int
    shape: tuple[int, int, int]
    affine: np.ndarray
    frame: Callable[[int], np.ndarray]
    path: str = ""
    header: object = None
    sidecar: dict = field(default_factory=dict)
    bvals: Optional[np.ndarray] = None
    #: 3 x n, as stored (image axes, FSL's convention).
    bvecs: Optional[np.ndarray] = None

    @property
    def zooms(self) -> np.ndarray:
        return np.sqrt((np.asarray(self.affine)[:3, :3] ** 2).sum(axis=0))


def read_gradients(path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[str]]:
    """``(bvals, bvecs 3 x n, problems)`` from the ``.bval`` and ``.bvec``
    beside ``path``. A file that is missing or unreadable is a problem, not
    an exception: the check reports it."""
    from ..viz import bids as VB

    stem = VB.stem_of(Path(path))
    problems: list[str] = []
    bvals = bvecs = None
    bval = Path(path).with_name(stem + ".bval")
    bvec = Path(path).with_name(stem + ".bvec")
    if not bval.is_file():
        problems.append(f"{bval.name} is missing")
    else:
        try:
            bvals = np.loadtxt(bval, dtype=float, ndmin=1).ravel()
        except ValueError as exc:
            problems.append(f"{bval.name} could not be read: {exc}")
    if not bvec.is_file():
        problems.append(f"{bvec.name} is missing")
    else:
        try:
            bvecs = np.loadtxt(bvec, dtype=float, ndmin=2)
        except ValueError as exc:
            problems.append(f"{bvec.name} could not be read: {exc}")
    return bvals, bvecs, problems


def from_file(path: Path, root: Optional[Path] = None) -> tuple[Series, list[str]]:
    """The series at ``path``, its sidecar (with inheritance) and its
    gradients, with any problem reading the gradients."""
    import nibabel as nib

    from ..viz import bids as VB

    path = Path(path)
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 3:
        data = data[..., None]
    root = root or VB.dataset_root(path)
    bvals, bvecs, problems = read_gradients(path)

    def frame(i: int) -> np.ndarray:
        return np.asarray(data[..., int(i)], dtype=np.float32)

    return Series(n=int(data.shape[3]), shape=tuple(int(s) for s in data.shape[:3]),
                  affine=np.asarray(img.affine, dtype=float), frame=frame, path=str(path),
                  header=img.header, sidecar=VB.inherited_sidecar(path, root),
                  bvals=bvals, bvecs=bvecs), problems


def from_source(src, sidecar: Optional[dict] = None) -> tuple[Series, list[str]]:
    """The viewer's ``VolumeSource`` (read whole) as a series, with the
    gradients beside its file."""
    bvals, bvecs, problems = read_gradients(Path(src.path))

    def frame(i: int) -> np.ndarray:
        return np.asarray(src.scale(np.asarray(src.raw_frame(int(i)))), dtype=np.float32)

    return Series(n=int(src.loaded_frames), shape=tuple(int(s) for s in src.spatial),
                  affine=np.asarray(src.affine, dtype=float), frame=frame,
                  path=str(src.path), sidecar=dict(sidecar or {}), bvals=bvals,
                  bvecs=bvecs), problems


__all__ = ["Series", "from_file", "from_source", "read_gradients"]

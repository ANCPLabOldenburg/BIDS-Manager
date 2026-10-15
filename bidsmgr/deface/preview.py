"""What defacing would remove, without removing anything.

Defacing is reversible here (the original is kept aside), but looking first
is still better than undoing: an engine that takes a slice of frontal lobe
with the nose, or misses the chin on a scan with an unusual field of view,
shows at once as a coloured region on the image.

The engine runs exactly as it would for real, into a temporary file; the
voxels it blanked (non-zero before, zero after) are the preview. An engine
that also CROPS (``robustfov``) returns a smaller grid, so the result is
compared in world space: anything outside what it kept was removed too.

Qt-free. Seconds of work: run it on a worker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .engines import DEFAULT_ENGINE_ID
from .run import DEFAULT_TIMEOUT_S, deface_to_temp


#: A voxel is part of the head above this fraction of the image's robust top
#: (its 98th percentile). Below it is the scanner's background noise, which
#: every engine also zeroes and which is nobody's face.
HEAD_FRACTION = 0.1


def head_of(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=bool)
    top = float(np.percentile(finite, 98))
    return image > HEAD_FRACTION * top


def removed_mask(source: Path, *, engine_id: str = DEFAULT_ENGINE_ID,
                 timeout: int = DEFAULT_TIMEOUT_S,
                 binary: Optional[Path] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mask, head, affine)`` on ``source``'s own grid: 1 where the engine
    would blank a voxel of the HEAD (background noise left out), and the
    head itself."""
    import nibabel as nib

    result = deface_to_temp(Path(source), engine_id=engine_id, timeout=timeout, binary=binary)
    try:
        before_img = nib.load(str(source))
        after_img = nib.load(str(result.output))
        before = np.asarray(before_img.dataobj, dtype=np.float32)
        after = np.asarray(after_img.dataobj, dtype=np.float32)
        before = before.reshape(before.shape[:3])
        after = after.reshape(after.shape[:3])
        if before.shape == after.shape and np.allclose(before_img.affine, after_img.affine,
                                                       atol=1e-4):
            kept = after
        else:
            kept = _on_grid(after, after_img.affine, before.shape, before_img.affine)
        head = head_of(before)
        mask = head & (kept == 0)
        return mask.astype(np.float32), head, np.asarray(before_img.affine, dtype=float)
    finally:
        Path(result.output).unlink(missing_ok=True)


def _on_grid(data: np.ndarray, affine: np.ndarray, shape, target_affine: np.ndarray) -> np.ndarray:
    """``data`` sampled at the voxel centres of the target grid (nearest;
    zero outside it, which is what a crop removed)."""
    from scipy.ndimage import map_coordinates

    ijk = np.indices(shape, dtype=float).reshape(3, -1)
    world = target_affine[:3, :3] @ ijk + target_affine[:3, 3:4]
    inv = np.linalg.inv(affine)
    src = inv[:3, :3] @ world + inv[:3, 3:4]
    out = map_coordinates(data, src, order=0, mode="constant", cval=0.0, prefilter=False)
    return out.reshape(shape)


def summary(mask: np.ndarray, head: np.ndarray) -> str:
    """A sentence: how much of the head the engine would take."""
    n_head = int(np.count_nonzero(head))
    n_gone = int(np.count_nonzero(mask))
    share = n_gone / n_head if n_head else 0.0
    return f"{share:.1%} of the head's voxels would be blanked"


__all__ = ["HEAD_FRACTION", "head_of", "removed_mask", "summary"]

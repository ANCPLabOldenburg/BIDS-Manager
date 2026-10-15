"""Where the head is, where the air is, and the air not to measure.

The air around the head is where noise and artefacts are measured (ghosts,
ringing, motion, wrap-around). Three kinds of "air" are not air and would
make every such measure meaningless:

- **zero fill**: voxels set to exactly zero by the scanner or a resampling
  (a rotated field of view, a background the scanner blanks);
- **defacing**: the face and ears set to zero to protect identity. The
  check measures air ONLY in images that are not defaced (the sidecar
  records defacing, ``deface.status``; an image whose face region is blank
  is treated the same, with or without a record);
- **the face, neck and shoulders**: below the plane through the glabella
  and the inion, where folding and motion are normal (MRIQC's "hat").

Head and air masks are made on the working grid (about 2 mm) and expanded
to the image's grid only where a measure needs it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import stats


def zero_fill(image: np.ndarray, *, min_voxels: int = 500) -> np.ndarray:
    """Voxels at exactly zero (or below) connected to the volume's border:
    padding, a blanked background, a defaced region. MRIQC's rotation mask,
    extended to keep every border-connected part rather than the two
    largest. Empty when fewer than ``min_voxels``."""
    from scipy import ndimage as ndi

    zero = np.asarray(image) <= 0
    if int(zero.sum()) < min_voxels:
        return np.zeros(zero.shape, dtype=bool)
    padded = np.pad(zero, 1, constant_values=True)
    padded = ndi.binary_opening(padded, structure=ndi.generate_binary_structure(3, 2))
    labels, _n = ndi.label(padded)
    border = np.unique(np.concatenate([labels[0].ravel(), labels[-1].ravel(),
                                       labels[:, 0].ravel(), labels[:, -1].ravel(),
                                       labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))
    border = border[border > 0]
    out = np.isin(labels, border)[1:-1, 1:-1, 1:-1]
    return out if int(out.sum()) >= min_voxels else np.zeros(zero.shape, dtype=bool)


def head(image: np.ndarray, *, exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """The head: Otsu's threshold between air and tissue (zero fill left
    out of the histogram), closed over gaps, the largest part kept, holes
    filled. On the working grid."""
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float32)
    keep = np.isfinite(img) & (img > 0)
    if exclude is not None:
        keep &= ~exclude
    values = img[keep]
    if values.size < 100:
        return np.zeros(img.shape, dtype=bool)
    # Otsu on the square root splits air from tissue, but with two tissue
    # populations (a dim scalp, a bright brain) it can split the tissues
    # instead. The air's own noise sets a floor: just above it is head.
    otsu = stats.otsu(np.sqrt(values)) ** 2
    thr = otsu
    level = air_level(img, keep=keep, below=0.5 * otsu, blank=exclude)
    if level is not None:
        thr = min(otsu, level[0] + 4.0 * level[1])
    mask = img > thr
    # Noise and ghosts just above that floor come as specks and thin
    # strands: an opening cuts them off before the largest part is kept.
    mask = ndi.binary_opening(mask, structure=stats.ball(1), iterations=2)
    mask = stats.largest_components(mask, 1)
    mask = ndi.binary_closing(mask, structure=stats.ball(2), iterations=1)
    filled = ndi.binary_fill_holes(mask)
    # A head open to the air (mouth, a nasal cavity cut by the field of
    # view) is not filled by a 3-D fill: fill slice by slice as well.
    for axis in range(3):
        filled = filled | np.stack([ndi.binary_fill_holes(s) for s in np.moveaxis(
            filled, axis, 0)], axis=axis)
    return filled


def air_level(image: np.ndarray, *, keep: Optional[np.ndarray] = None,
              below: Optional[float] = None,
              blank: Optional[np.ndarray] = None) -> Optional[tuple[float, float]]:
    """``(median, spread)`` of the air's noise, or None: too little to
    measure, no spread (zero fill), or brighter than ``below`` (the head
    reaches it). The air sampled is the border of the field of view (air,
    in a head scan) and, when the scanner blanked the background
    (``blank``, its zero fill), the band just inside the blanked part: on a
    Philips scan with 70 % of the volume at zero, the only non-zero border
    left was the neck."""
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float32)
    if keep is None:
        keep = np.isfinite(img) & (img > 0)
    edge = np.zeros(img.shape, dtype=bool)
    b = max(2, min(img.shape[:3]) // 20)
    edge[:b] = edge[-b:] = True
    edge[:, :b] = edge[:, -b:] = True
    edge[:, :, :b] = edge[:, :, -b:] = True
    if blank is not None and np.asarray(blank).any():
        blank = np.asarray(blank, dtype=bool)
        edge |= ndi.binary_dilation(blank, iterations=2) & ~blank
    border = img[edge & keep]
    if border.size <= 100:
        return None
    med = float(np.median(border))
    spread = float(np.median(np.abs(border - med))) * stats.MAD_TO_SD
    if spread <= 0 or (below is not None and med >= below):
        return None
    return med, spread


def face_blank_share(image: np.ndarray, face: np.ndarray) -> float:
    """Share of the voxels where a face should be (``face``, the template's
    face region on the image grid, inside the template's head) that are
    exactly zero. Defacing blanks them; an image that was never defaced has
    tissue there."""
    face = np.asarray(face, dtype=bool)
    n = int(face.sum())
    if n == 0:
        return 0.0
    return float(np.count_nonzero(np.asarray(image)[face] <= 0)) / n


def air(head_mask: np.ndarray, *, excluded: np.ndarray, above: Optional[np.ndarray] = None,
        margin: int = 2) -> np.ndarray:
    """The air to measure: outside the head (by ``margin`` voxels), not
    ``excluded`` (zero fill), and above the glabella-inion plane when
    ``above`` (a boolean volume) is given."""
    from scipy import ndimage as ndi

    grown = ndi.binary_dilation(head_mask, structure=stats.ball(1), iterations=max(margin, 1))
    out = ~grown & ~np.asarray(excluded, dtype=bool)
    if above is not None:
        out &= np.asarray(above, dtype=bool)
    return out


def artefacts(image: np.ndarray, air_mask: np.ndarray, head_mask: np.ndarray, *,
              z: float = 10.0) -> np.ndarray:
    """Air voxels that are not noise (QI1's artefact mask, Mortamet 2009):
    an intensity above ``z`` times the robust spread of the air's own
    intensities (MRIQC's test: the value over the spread), more than a
    tenth of the largest distance away from the head (closer than that is
    the head's own edge and its partial volume), cleaned by an opening.

    MRIQC meant this and computes something else: it zeroes the distance
    over the whole air before the test, so its mask is always empty and its
    QI1 always 0 (``interfaces/anatomical.py:326``)."""
    from scipy import ndimage as ndi

    air_mask = np.asarray(air_mask, dtype=bool)
    out = np.zeros(air_mask.shape, dtype=bool)
    values = np.asarray(image, dtype=np.float32)[air_mask]
    positive = values[values > 0]
    if positive.size < 10:
        return out
    # MRIQC's scale: the median absolute deviation about the median (scaled
    # to an SD); a voxel is then tested by its value over that spread.
    med = float(np.median(positive))
    spread = float(np.median(np.abs(positive - med))) * stats.MAD_TO_SD
    if spread <= 0:
        return out
    dist = ndi.distance_transform_edt(~np.asarray(head_mask, dtype=bool))
    far = dist > 0.1 * float(dist[air_mask].max() if air_mask.any() else 0.0)
    out[air_mask] = values / spread > z
    out &= far
    return ndi.binary_opening(out, structure=ndi.generate_binary_structure(3, 1))


__all__ = ["air", "air_level", "artefacts", "face_blank_share", "head", "zero_fill"]

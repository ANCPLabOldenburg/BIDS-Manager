"""Three tissue classes (CSF, grey matter, white matter) by expectation
maximisation, with the template's tissue maps as priors.

Each class is a Gaussian in intensity; a voxel's probability of each class
is its likelihood under that Gaussian times the class's prior there. The
prior blends the template's map at that place (registered, ``register``)
with the class's share of the brain, so a voxel the template calls white
matter still becomes grey matter when its intensity says so. One pass of
neighbourhood smoothing (an ICM step) then favours the class the voxel's
neighbours hold, which is what a Markov random field does in Atropos.

The classes are NAMED by the template, never by their intensity order, so
the same code segments a T1w (CSF dark, white matter bright) and a T2w
(the opposite) without being told which it is.

Alternates with ``bias.fit_field`` (N3-style): classes, field, classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import bias as B

CLASSES = ("csf", "gm", "wm")
#: How much the template's map weighs against the global share.
PRIOR_WEIGHT = 0.5
#: Strength of the neighbourhood term.
SMOOTHING = 1.0


@dataclass
class Segmentation:
    #: (3, x, y, z) float32 posteriors, CSF, GM, WM; 0 outside the mask.
    posteriors: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    #: The bias field (log), on the same grid, 0 outside the mask.
    field_log: np.ndarray
    iterations: int

    def labels(self, mask: np.ndarray) -> np.ndarray:
        """1 CSF, 2 GM, 3 WM, 0 outside ``mask``."""
        lab = (np.argmax(self.posteriors, axis=0) + 1).astype(np.uint8)
        lab[~np.asarray(mask, dtype=bool)] = 0
        return lab


def em(values: np.ndarray, priors: np.ndarray, means: np.ndarray, sds: np.ndarray,
        iterations: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EM on ``values`` (N) with per-voxel ``priors`` (3 x N)."""
    post = priors.copy()
    for _ in range(iterations):
        like = np.exp(-0.5 * ((values[None, :] - means[:, None]) / sds[:, None]) ** 2) / sds[:, None]
        post = like * priors
        total = post.sum(axis=0)
        total[total <= 0] = 1.0
        post /= total
        w = post.sum(axis=1)
        w[w <= 0] = 1.0
        new_means = (post @ values) / w
        var = np.array([(post[k] * (values - new_means[k]) ** 2).sum() / w[k] for k in range(3)])
        new_sds = np.sqrt(np.maximum(var, 1e-12))
        done = np.allclose(new_means, means, rtol=1e-4)
        means, sds = new_means, new_sds
        if done:
            break
    return post, means, sds


def from_classes(image: np.ndarray, fractions: np.ndarray, mask: np.ndarray, *,
                 rounds: int = 2) -> Optional[Segmentation]:
    """The bias field and class statistics of ``image`` for classes found by
    something better (``fractions``: (3, x, y, z), each class's share of a
    voxel, from a tissue model's labels): the classes stay as given, the
    field is fitted to what their means do not explain (N3-style)."""
    img = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(img) & (img > 0)
    if int(mask.sum()) < 1000:
        return None
    post = np.clip(np.asarray(fractions, dtype=np.float64)[:, mask], 0.0, 1.0)
    total = post.sum(axis=0)
    total[total <= 0] = 1.0
    post /= total
    values = img[mask].astype(np.float64)
    log_img = np.log(np.maximum(img, 1e-6))
    field = np.zeros(img.shape, dtype=np.float32)
    weights = np.maximum(post.sum(axis=1), 1e-9)
    means = (post @ values) / weights
    for _ in range(max(rounds, 1)):
        corrected = values / np.exp(field[mask].astype(np.float64))
        means = (post @ corrected) / weights
        expected = np.log(np.maximum(post.T @ means, 1e-6))
        resid = np.zeros(img.shape, dtype=np.float32)
        resid[mask] = (log_img[mask] - expected).astype(np.float32)
        field = B.fit_field(resid, mask.astype(np.float32))
    corrected = values / np.exp(field[mask].astype(np.float64))
    means = (post @ corrected) / weights
    var = np.array([(post[k] * (corrected - means[k]) ** 2).sum() / weights[k] for k in range(3)])
    posteriors = np.zeros((3,) + img.shape, dtype=np.float32)
    posteriors[:, mask] = post.astype(np.float32)
    field_out = np.where(mask, field, 0.0).astype(np.float32)
    return Segmentation(posteriors, means, np.sqrt(np.maximum(var, 1e-12)), field_out, rounds)


def segment(image: np.ndarray, mask: np.ndarray, tissue_maps: np.ndarray, *,
            rounds: int = 3, smooth: bool = True) -> Optional[Segmentation]:
    """Classes and bias field of ``image`` inside ``mask``. ``tissue_maps``
    (3, x, y, z): the template's CSF, GM and WM maps on the image grid.
    None when the mask is too small to segment."""
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(img) & (img > 0)
    n = int(mask.sum())
    if n < 1000:
        return None
    tpm = np.clip(np.asarray(tissue_maps, dtype=np.float32), 0.0, 1.0)
    tpm_in = tpm[:, mask].astype(np.float64)
    tpm_in += 1e-3
    tpm_in /= tpm_in.sum(axis=0)
    log_img = np.log(np.maximum(img, 1e-6))
    field = np.zeros(img.shape, dtype=np.float32)
    values = img[mask].astype(np.float64)
    # Start: each class's mean where the template is sure of it.
    means = np.array([np.average(values, weights=tpm_in[k] ** 4 + 1e-9) for k in range(3)])
    sds = np.full(3, max(float(values.std()) / 3.0, 1e-6))
    post = tpm_in
    it = 0
    for r in range(max(rounds, 1)):
        corrected = values / np.exp(field[mask].astype(np.float64))
        share = post.mean(axis=1)
        priors = PRIOR_WEIGHT * tpm_in + (1.0 - PRIOR_WEIGHT) * share[:, None]
        post, means, sds = em(corrected, priors, means, sds)
        it += 1
        if r == rounds - 1:
            break
        # What the classes predict, against what is there: the field.
        expected = np.log(np.maximum(post.T @ means, 1e-6))
        resid = np.zeros(img.shape, dtype=np.float32)
        resid[mask] = (log_img[mask] - expected).astype(np.float32)
        field = B.fit_field(resid, mask.astype(np.float32))
    if smooth:
        # One ICM-like step: a voxel leans to the class its neighbours hold.
        full = np.zeros((3,) + img.shape, dtype=np.float32)
        full[:, mask] = post
        neigh = np.stack([ndi.uniform_filter(full[k], size=3) for k in range(3)])
        corrected = values / np.exp(field[mask].astype(np.float64))
        like = np.exp(-0.5 * ((corrected[None, :] - means[:, None]) / sds[:, None]) ** 2) / sds[:, None]
        post = like * np.exp(SMOOTHING * neigh[:, mask]) * (
            PRIOR_WEIGHT * tpm_in + (1.0 - PRIOR_WEIGHT) * post.mean(axis=1)[:, None])
        total = post.sum(axis=0)
        total[total <= 0] = 1.0
        post /= total
    posteriors = np.zeros((3,) + img.shape, dtype=np.float32)
    posteriors[:, mask] = post.astype(np.float32)
    field_out = np.where(mask, field, 0.0).astype(np.float32)
    return Segmentation(posteriors, means, sds, field_out, it)


__all__ = ["CLASSES", "Segmentation", "em", "from_classes", "segment"]

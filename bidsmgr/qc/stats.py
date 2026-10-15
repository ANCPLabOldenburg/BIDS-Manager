"""Small statistics and morphology the checks share. numpy and scipy only.

Robust where it matters: a quality check is run on the images most likely to
be wrong, so a mean or a standard deviation alone is the wrong summary of a
region that may hold an artefact.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: MAD to SD for a normal distribution.
MAD_TO_SD = 1.4826


def mad(values: np.ndarray, centre: Optional[float] = None) -> float:
    """Median absolute deviation scaled to an SD (normal data)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    c = float(np.median(v)) if centre is None else float(centre)
    return float(np.median(np.abs(v - c)) * MAD_TO_SD)


def nanmedian_rows(values: np.ndarray) -> np.ndarray:
    """The median of each row ignoring NaN, as a column; NaN (quietly) for
    a row with nothing in it."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.asarray(values, dtype=np.float64), axis=1, keepdims=True)


def robust_z(values: np.ndarray, axis=None) -> np.ndarray:
    """(x - median) / MAD-SD, along ``axis``; 0 where the spread is 0."""
    import warnings

    v = np.asarray(values, dtype=np.float64)
    with warnings.catch_warnings():
        # A row with nothing in it (a slice outside the head) is NaN, not news.
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(v, axis=axis, keepdims=axis is not None)
        spread = np.nanmedian(np.abs(v - med), axis=axis,
                              keepdims=axis is not None) * MAD_TO_SD
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (v - med) / spread
    return np.where(np.isfinite(z), z, 0.0)


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Mean, SD (population), median, P5, P95 of ``values`` weighted by
    ``weights`` (a soft tissue map), plus MAD and kurtosis over the voxels
    the class holds for sure (weight above half its maximum), and ``n``, the
    sum of the weights. The summary MRIQC reports, in numpy."""
    x = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    nan = float("nan")
    if w.sum() <= 0:
        return {"mean": nan, "stdv": nan, "median": nan, "p05": nan, "p95": nan,
                "mad": nan, "k": nan, "n": 0.0}
    total = float(w.sum())
    mean = float(np.sum(w * x) / total)
    sd = float(np.sqrt(np.sum(w * (x - mean) ** 2) / total))
    order = np.argsort(x)
    xs, cw = x[order], np.cumsum(w[order]) / total

    def q(p: float) -> float:
        i = int(np.searchsorted(cw, p, side="left"))
        return float(xs[min(i, xs.size - 1)])

    sure = x[w > 0.5 * float(w.max())]
    from scipy import stats as st

    k = float(st.kurtosis(sure)) if sure.size > 3 else nan
    return {"mean": mean, "stdv": sd, "median": q(0.5), "p05": q(0.05), "p95": q(0.95),
            "mad": mad(sure), "k": k, "n": total}


def otsu(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold of ``values``."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    # Every threshold inside an empty gap is as good: take the gap's middle,
    # not its first bin (which hugs the lower population).
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return float(centres[int(best[len(best) // 2])])


def largest_components(mask: np.ndarray, n: int = 1) -> np.ndarray:
    """The ``n`` largest connected parts of ``mask`` (26-connected)."""
    from scipy import ndimage as ndi

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if count <= n:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = np.argsort(sizes)[::-1][:n]
    return np.isin(labels, keep)


def ball(radius: int) -> np.ndarray:
    """A solid ball of ``radius`` voxels, as a structuring element."""
    r = int(max(radius, 0))
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (g ** 2).sum(axis=0) <= r * r


def block_factor(zooms, target_mm: float) -> np.ndarray:
    """Whole voxels per block that bring ``zooms`` up to about ``target_mm``."""
    z = np.asarray(zooms, dtype=float)[:3]
    return np.maximum(1, np.round(target_mm / np.maximum(z, 1e-3))).astype(int)


def block_mean(volume: np.ndarray, factor) -> np.ndarray:
    """Means of ``factor`` blocks, the volume padded by edge values to a
    whole number of blocks (so the grid covers every voxel)."""
    v = np.asarray(volume, dtype=np.float32)
    f = [int(x) for x in factor]
    if f == [1, 1, 1]:
        return v
    pad = [(0, (-s) % k) for s, k in zip(v.shape[:3], f)]
    if any(p[1] for p in pad):
        v = np.pad(v, pad, mode="edge")
    sx, sy, sz = v.shape[:3]
    return v.reshape(sx // f[0], f[0], sy // f[1], f[1], sz // f[2], f[2]).mean(axis=(1, 3, 5))


def block_affine(affine: np.ndarray, factor) -> np.ndarray:
    """The affine of :func:`block_mean`'s grid: voxels ``factor`` times
    larger, the first centred on the first block's centre."""
    a = np.array(affine, dtype=float)
    f = np.asarray(factor, dtype=float)
    out = a.copy()
    out[:3, :3] = a[:3, :3] * f[None, :]
    out[:3, 3] = a[:3, :3] @ ((f - 1.0) / 2.0) + a[:3, 3]
    return out


def block_expand(small: np.ndarray, factor, shape) -> np.ndarray:
    """Each block's value back on the full grid of ``shape`` (nearest)."""
    f = [int(x) for x in factor]
    out = np.asarray(small)
    for axis, k in enumerate(f):
        if k > 1:
            out = np.repeat(out, k, axis=axis)
    return out[: shape[0], : shape[1], : shape[2]]


def gaussian_kde_grid(samples: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """A Gaussian kernel density of ``samples`` on ``grid`` (normalised to
    integrate to one), by binning on the grid first: exact up to the bin
    width, and linear in the sample count."""
    s = np.asarray(samples, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    if s.size == 0 or g.size < 2:
        return np.zeros_like(g)
    step = float(g[1] - g[0])
    edges = np.concatenate([g - step / 2.0, [g[-1] + step / 2.0]])
    hist, _ = np.histogram(s, bins=edges)
    radius = int(np.ceil(4.0 * bandwidth / step))
    k = np.arange(-radius, radius + 1) * step
    kernel = np.exp(-0.5 * (k / bandwidth) ** 2)
    dens = np.convolve(hist.astype(np.float64), kernel, mode="same")
    area = float(dens.sum() * step)
    return dens / area if area > 0 else dens


__all__ = [
    "MAD_TO_SD", "ball", "block_affine", "block_expand", "block_factor", "block_mean",
    "gaussian_kde_grid", "largest_components", "mad", "nanmedian_rows", "otsu", "robust_z",
    "weighted_stats",
]

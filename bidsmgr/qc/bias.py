"""The intensity non-uniformity (bias field) of an anatomical image.

N3's idea (Sled 1998) in its simplest form: inside the brain, the log of the
image is the log of the tissue it shows plus a smooth field. Given the
tissue classes, the field is a smooth fit to what the classes do not
explain; given the field, the classes are found again (``segment``). Two or
three rounds are enough for a quality check.

The smooth fit is a third-order polynomial in x, y and z (20 terms) on a
4 mm grid: a field that varies faster than that is not a bias field.
"""

from __future__ import annotations

import numpy as np

#: Degree of the polynomial field.
DEGREE = 3


def _powers(degree: int = DEGREE) -> list[tuple[int, int, int]]:
    """Every (i, j, k) of total degree <= ``degree``."""
    return [(i, j, k) for i in range(degree + 1) for j in range(degree + 1 - i)
            for k in range(degree + 1 - i - j)]


def _axis_polys(n: int, degree: int) -> list[np.ndarray]:
    """Legendre polynomials 0..``degree`` on ``n`` points of [-1, 1]."""
    from numpy.polynomial import legendre

    a = np.linspace(-1.0, 1.0, int(n))
    return [legendre.legval(a, [0] * d + [1]) for d in range(degree + 1)]


#: The second, finer stage: a normalised Gaussian smoothing of what the
#: polynomial leaves, sigma in voxels of the working grid (5 at 2 mm: about
#: the finest level of N4's B-spline at MRIQC's settings).
FINE_SIGMA = 5.0


def fit_field(log_residual: np.ndarray, weights: np.ndarray, *,
              degree: int = DEGREE, fine_sigma: float = FINE_SIGMA) -> np.ndarray:
    """The smooth (log) field that best explains ``log_residual`` where
    ``weights`` > 0, on the same grid, mean zero inside the weights: a
    polynomial, then what it leaves smoothed (normalised convolution) so the
    field can follow a coil's sensitivity closer than a cubic can. The basis
    is evaluated only at the weighted voxels and the field term by term, so
    memory stays at a few copies of the grid."""
    poly = _fit_polynomial(log_residual, weights, degree=degree)
    if not fine_sigma:
        return poly
    from scipy import ndimage as ndi

    w = np.asarray(weights, dtype=np.float64)
    if not (w > 0).any():
        return poly
    rest = (np.asarray(log_residual, dtype=np.float64) - poly) * w
    num = ndi.gaussian_filter(rest, fine_sigma)
    den = ndi.gaussian_filter(w, fine_sigma)
    fine = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    field = poly + fine
    keep = w > 0
    field -= float(np.average(field[keep], weights=w[keep]))
    return field.astype(np.float32)


def _fit_polynomial(log_residual: np.ndarray, weights: np.ndarray, *,
                    degree: int = DEGREE) -> np.ndarray:
    """The polynomial stage of :func:`fit_field`."""
    shape = log_residual.shape
    w = np.asarray(weights, dtype=np.float64)
    keep = w > 0
    powers = _powers(degree)
    if int(keep.sum()) < len(powers) * 4:
        return np.zeros(shape, dtype=np.float32)
    polys = [_axis_polys(n, degree) for n in shape]
    idx = np.nonzero(keep)
    a = np.empty((idx[0].size, len(powers)))
    for c, (i, j, k) in enumerate(powers):
        a[:, c] = polys[0][i][idx[0]] * polys[1][j][idx[1]] * polys[2][k][idx[2]]
    sw = np.sqrt(w[keep])
    b = np.asarray(log_residual, dtype=np.float64)[keep]
    coef, *_ = np.linalg.lstsq(a * sw[:, None], b * sw, rcond=None)
    field = np.zeros(shape, dtype=np.float64)
    for c, (i, j, k) in enumerate(powers):
        field += coef[c] * (polys[0][i][:, None, None] * polys[1][j][None, :, None]
                            * polys[2][k][None, None, :])
    field -= float(np.average(field[keep], weights=w[keep]))
    return field.astype(np.float32)


def nonuniformity(field_log: np.ndarray, mask: np.ndarray) -> dict:
    """The field as a multiplier normalised to median 1 inside ``mask``:
    its median (1 by construction), 5th and 95th percentiles, and the range
    between them, the number reported."""
    f = np.exp(np.asarray(field_log, dtype=np.float64)[np.asarray(mask, dtype=bool)])
    if f.size == 0:
        return {"p05": float("nan"), "p95": float("nan"), "range": float("nan")}
    f /= float(np.median(f))
    p05, p95 = (float(x) for x in np.percentile(f, (5, 95)))
    return {"p05": p05, "p95": p95, "range": p95 - p05}


__all__ = ["DEGREE", "FINE_SIGMA", "fit_field", "nonuniformity"]

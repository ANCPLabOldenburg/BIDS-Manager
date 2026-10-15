"""The quality check of a diffusion series.

In order, all numpy and scipy:

1. **The gradient table** (``.bval``, ``.bvec``): present, one entry per
   volume, unit vectors, at least one b=0, the shells and how well each
   covers the sphere, duplicated directions. A wrong table ruins every
   analysis downstream and no image measure shows it, so it comes first.
2. **Head motion**: each b=0 registered rigidly to the b=0 reference, each
   diffusion-weighted volume to the mean of ITS OWN shell (a b=1000 volume
   and a b=0 do not share a contrast), the shell means to the b=0
   reference; framewise displacement from the composed motion, and how far
   each gradient direction turns with the head.
3. **Signal drift** over the scan, from the interleaved b=0 volumes.
4. **A tensor** fitted to every brain voxel (weighted least squares on
   b <= 1500, batched): FA, MD, the principal direction, and the signal it
   PREDICTS for every volume.
5. **Per slice and volume, the observed signal against that prediction**:
   a slice far below it in one volume is a **dropout** (the commonest
   diffusion artefact, and one MRIQC does not look for), odd slices against
   even ones an **interleave** artefact, single voxels far above it
   **spikes**.
6. **Noise**: from repeated b=0 volumes, and by MP-PCA (Veraart 2016) on
   sampled patches; the SNR in the corpus callosum per shell, along and
   across its fibres; the share of the highest shell at the noise floor.
7. **Neighbouring directions**: each volume's correlation with the volume
   nearest it in q-space (a volume unlike its neighbours is suspect).
8. **Flipped or swapped b-vectors** (experimental): the tensor's principal
   directions are most continuous along themselves with the right table.

Where MRIQC's diffusion code differs from its own definitions, we follow the
definitions (its drift is multiplied rather than divided out, its spike mask
indexes volumes 0 and 1, its corpus callosum SNR is shifted by one shell):
the plan records each.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import masks as K
from . import stats
from .config import DEFAULT, QcConfig
from .series import Series
from .types import Finding, Metric, QCMap, QCResult

# The thresholds are settings (``config.QcDwi``). Measured when they were
# chosen: a volume out of place at 2 mm is about five times a single
# volume's registration noise; on the tutorial's single-shell scan a planted
# flip of x, y or z was found every time with the true table ahead by 4.4 %,
# and the stored table of the unchanged scan led the next by the same
# margin, so a flipped table must lead by 2 %.

#: A slice needs this many brain voxels to be judged (and the share of a
#: typical slice's set in ``QcDwi.edge_slice_pct``).
MIN_SLICE_VOXELS = 40


def _metric(key: str, title: str, value, *, unit: str = "", group: str = "diffusion",
            better: str = "", mriqc: str = "", help: str = "", fmt: str = "{:.3g}",
            why: str = "") -> Metric:
    v = None if value is None or not np.isfinite(value) else float(value)
    return Metric(key=key, title=title, value=v, unit=unit, group=group, help=help,
                  better=better, mriqc=mriqc, fmt=fmt, why_missing=why if v is None else "")


# ---------------------------------------------------------------------------
# The gradient table
# ---------------------------------------------------------------------------


def shells_of(bvals: np.ndarray, b0_max: float = DEFAULT.dwi.b0_max) -> np.ndarray:
    """Each volume's shell (the viewer's rule: rounded to the nearest 50),
    0 for every b=0 volume."""
    from ..viz.bids import shells_of as viz_shells

    s = viz_shells(bvals)
    s[np.asarray(bvals, dtype=float) <= b0_max] = 0
    return s


def coverage_gap_deg(vectors: np.ndarray, samples: int = 4000) -> float:
    """The largest angle (degrees) from any direction on the sphere to the
    nearest gradient direction of ``vectors`` (n x 3), either sign: the
    radius of the largest empty cap. Small is even coverage."""
    v = np.asarray(vectors, dtype=float)
    v = v[np.linalg.norm(v, axis=1) > 0.5]
    if len(v) == 0:
        return 180.0
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    rng = np.random.default_rng(5)
    p = rng.normal(size=(samples, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    cos = np.abs(p @ v.T).max(axis=1)
    return float(np.degrees(np.arccos(np.clip(cos.min(), -1.0, 1.0))))


def duplicates(vectors: np.ndarray, limit_deg: float = DEFAULT.dwi.duplicate_deg) -> int:
    """Pairs of directions closer than ``limit_deg``, either sign."""
    v = np.asarray(vectors, dtype=float)
    v = v[np.linalg.norm(v, axis=1) > 0.5]
    if len(v) < 2:
        return 0
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    cos = np.abs(v @ v.T)
    np.fill_diagonal(cos, 0.0)
    return int(np.count_nonzero(np.triu(cos > np.cos(np.radians(limit_deg)))))


def check_table(n_volumes: int, bvals, bvecs, problems: list[str], *,
                b0_max: float = DEFAULT.dwi.b0_max) -> tuple[list[Finding], dict]:
    """Findings about the gradient table, and its usable form (``bvals``,
    ``bvecs`` n x 3) or None when there is none to use."""
    found: list[Finding] = []
    usable: dict = {}
    for p in problems:
        found.append(Finding("table_missing", "No gradient table", "error",
                             f"{p[:1].upper()}{p[1:]}: the series cannot be read as "
                             "diffusion data.", None))
    if bvals is None or bvecs is None:
        return found, usable
    b = np.asarray(bvals, dtype=float).ravel()
    g = np.asarray(bvecs, dtype=float)
    if g.ndim != 2 or 3 not in g.shape:
        found.append(Finding("bvec_shape", "The .bvec is not three rows", "error",
                             f"It holds an array of shape {g.shape}; BIDS asks for three "
                             "rows (x, y, z), one column per volume.", None))
        return found, usable
    if g.shape[0] != 3:
        g = g.T
    if b.size != n_volumes or g.shape[1] != n_volumes:
        found.append(Finding(
            "table_count", "The table does not match the volumes", "error",
            f"The series has {n_volumes} volumes, the .bval {b.size} values and the .bvec "
            f"{g.shape[1]} vectors: every volume needs exactly one of each.", None))
        return found, usable
    g = g.T                                                    # n x 3
    norms = np.linalg.norm(g, axis=1)
    weighted = b > b0_max
    zero = weighted & (norms < 1e-6)
    if zero.any():
        found.append(Finding(
            "bvec_zero", "Diffusion volumes with no direction", "error",
            f"{int(zero.sum())} volumes have b > {b0_max:g} and a zero b-vector: volumes "
            f"{', '.join(str(i) for i in np.flatnonzero(zero)[:8])}.", None))
    off = weighted & ~zero & (np.abs(norms - 1.0) > 0.01)
    if off.any():
        found.append(Finding(
            "bvec_norm", "b-vectors that are not unit vectors", "warning",
            f"{int(off.sum())} b-vectors have a length off 1 by more than 0.01 (from "
            f"{norms[off].min():.3f} to {norms[off].max():.3f}). Some tools rescale the "
            "b-value by it, others do not.", None))
    if not (b <= b0_max).any():
        found.append(Finding("no_b0", "No b=0 volume", "error",
                             "Without a b=0 volume nothing can be normalised, and the "
                             "tensor, SNR and motion checks are skipped.", None))
    usable = {"bvals": b, "bvecs": np.where(norms[:, None] > 1e-6,
                                            g / np.maximum(norms[:, None], 1e-6), 0.0)}
    return found, usable


#: A volume registered further than this from the reference is taken as
#: lost, not as moved: a head does not move 3 cm or turn 15 degrees inside a
#: head coil between two volumes.


def _implausible(rot: np.ndarray, trans: np.ndarray, lost_mm: float, lost_deg: float) -> bool:
    angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)))
    return bool(np.linalg.norm(trans) > lost_mm or angle > lost_deg)


def brain_mask(b0: np.ndarray, *, exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """The brain on a b=0 image, as dipy's ``median_otsu`` finds it: a median
    filter, Otsu's threshold, the largest part, holes filled, one voxel off
    the edge. On an EPI b=0 the brain is the bright part and fat is
    suppressed; the anatomical head mask, whose threshold sits just above
    the air's noise, takes in an EPI's ghosts and background (half the
    field of view on the tutorial's multi-shell scan)."""
    from scipy import ndimage as ndi

    img = ndi.median_filter(np.asarray(b0, dtype=np.float32), size=3)
    keep = np.isfinite(img) & (img > 0)
    if exclude is not None:
        keep &= ~exclude
    values = img[keep]
    if values.size < 100:
        return np.zeros(img.shape, dtype=bool)
    mask = keep & (img > stats.otsu(values))
    mask = ndi.binary_opening(mask, structure=stats.ball(1), iterations=1)
    mask = stats.largest_components(mask, 1)
    mask = ndi.binary_closing(mask, structure=stats.ball(2), iterations=1)
    for axis in range(3):
        mask = mask | np.stack([ndi.binary_fill_holes(s) for s in np.moveaxis(
            mask, axis, 0)], axis=axis)
    return ndi.binary_erosion(mask, structure=stats.ball(1), iterations=1)


# ---------------------------------------------------------------------------
# The tensor
# ---------------------------------------------------------------------------


def design(bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
    """The log-linear tensor design matrix: [1, -b gx^2, -b gy^2, -b gz^2,
    -2b gx gy, -2b gx gz, -2b gy gz] per volume."""
    b = np.asarray(bvals, dtype=float)
    g = np.asarray(bvecs, dtype=float)
    return np.column_stack([np.ones_like(b), -b * g[:, 0] ** 2, -b * g[:, 1] ** 2,
                            -b * g[:, 2] ** 2, -2 * b * g[:, 0] * g[:, 1],
                            -2 * b * g[:, 0] * g[:, 2], -2 * b * g[:, 1] * g[:, 2]])


#: Voxels fitted at once: bounds the memory of the batched solve.
CHUNK = 40_000


def fit_tensor(signal: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray) -> dict:
    """:func:`fit_chunk` over chunks of :data:`CHUNK` voxels."""
    parts = [fit_chunk(signal[i:i + CHUNK], bvals, bvecs)
             for i in range(0, max(len(signal), 1), CHUNK)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def fit_chunk(signal: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray) -> dict:
    """Weighted least squares, batched over voxels (``signal``: voxels x
    volumes). OLS on the log signal, then one WLS step weighted by the OLS
    prediction squared (Salvador 2005), as dipy's default. Returns the
    coefficients and, per voxel, FA, MD, the principal eigenvector, and
    whether any eigenvalue was negative before clipping."""
    s = np.maximum(np.asarray(signal, dtype=np.float64), 1e-3)
    y = np.log(s)
    a = design(bvals, bvecs)
    coef, *_ = np.linalg.lstsq(a, y.T, rcond=None)
    w = np.exp(coef.T @ a.T) ** 2                               # voxels x volumes
    ata = np.einsum("vk,ki,kj->vij", w, a, a)
    aty = np.einsum("vk,ki,vk->vi", w, a, y)
    try:
        coef = np.linalg.solve(ata, aty[..., None])[..., 0]
    except np.linalg.LinAlgError:
        coef = coef.T
    d = coef[:, 1:]
    t = np.empty((len(d), 3, 3))
    t[:, 0, 0], t[:, 1, 1], t[:, 2, 2] = d[:, 0], d[:, 1], d[:, 2]
    t[:, 0, 1] = t[:, 1, 0] = d[:, 3]
    t[:, 0, 2] = t[:, 2, 0] = d[:, 4]
    t[:, 1, 2] = t[:, 2, 1] = d[:, 5]
    t = np.where(np.isfinite(t), t, 0.0)
    evals, evecs = np.linalg.eigh(t)
    negative = (evals < 0).any(axis=1)
    raw_md = evals.mean(axis=1)
    raw_fa = np.sqrt(1.5 * ((evals - raw_md[:, None]) ** 2).sum(axis=1)
                     / np.maximum((evals ** 2).sum(axis=1), 1e-30))
    ev = np.maximum(evals, 0.0)
    md = ev.mean(axis=1)
    fa = np.sqrt(1.5 * ((ev - md[:, None]) ** 2).sum(axis=1)
                 / np.maximum((ev ** 2).sum(axis=1), 1e-30))
    return {"coef": coef, "fa": np.clip(fa, 0.0, 1.0), "md": md, "e1": evecs[:, :, 2],
            "negative": negative, "fa_above_1": raw_fa > 1.0}


def predict(coef: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
    """The tensor's log signal for every volume (voxels x volumes)."""
    return coef @ design(bvals, bvecs).T


def world_directions(vectors: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Directions given in image axes (FSL's convention, as BIDS stores
    them: x reversed when the affine's determinant is positive) in world
    (RAS) terms."""
    v = np.array(vectors, dtype=float)
    lin = np.asarray(affine)[:3, :3]
    if np.linalg.det(lin) > 0:
        v[..., 0] = -v[..., 0]
    axes = lin / np.maximum(np.linalg.norm(lin, axis=0), 1e-9)
    out = v @ axes.T
    n = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(n, 1e-9)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def mppca_sigma(patch: np.ndarray) -> Optional[float]:
    """The noise SD of one patch (voxels x volumes) by Marchenko-Pastur PCA
    (Veraart 2016): the largest number of components whose eigenvalues
    still follow the noise law."""
    x = np.asarray(patch, dtype=np.float64)
    n_vox, m = x.shape
    if n_vox <= m or m < 4:
        return None
    x = x - x.mean(axis=0, keepdims=True)
    lam = np.linalg.eigvalsh(x.T @ x / n_vox)[::-1]           # descending, m values
    lam = np.maximum(lam, 0.0)
    for p in range(m - 1):
        rest = lam[p:]
        sigma2 = float(rest.mean())
        gamma = (m - p) / n_vox
        width = float(rest[0] - rest[-1])
        if width < 4.0 * np.sqrt(gamma) * sigma2:
            return float(np.sqrt(sigma2))
    return None


def mppca_noise(series: Series, volumes: np.ndarray, brain: np.ndarray, *,
                patches: int = 300, seed: int = 3) -> Optional[float]:
    """The median MP-PCA noise SD over sampled patches of ``volumes`` inside
    ``brain``: seconds, where denoising the whole series takes minutes."""
    vols = np.asarray(volumes, dtype=int)
    m = vols.size
    if m < 6:
        return None
    r = 2 if m < 100 else 3                                    # 5^3 or 7^3 voxels
    shape = brain.shape
    core = brain.copy()
    core[:r] = core[-r:] = False
    core[:, :r] = core[:, -r:] = False
    core[:, :, :r] = core[:, :, -r:] = False
    centres = np.argwhere(core)
    if len(centres) == 0:
        return None
    rng = np.random.default_rng(seed)
    pick = centres[rng.choice(len(centres), min(patches, len(centres)), replace=False)]
    frames = [series.frame(int(v)) for v in vols]
    sigmas = []
    for c in pick:
        sl = tuple(slice(int(k) - r, int(k) + r + 1) for k in c)
        if brain[sl].mean() < 0.8:
            continue
        patch = np.stack([f[sl].ravel() for f in frames], axis=1)
        s = mppca_sigma(patch)
        if s is not None and np.isfinite(s):
            sigmas.append(s)
    del frames, shape
    return float(np.median(sigmas)) if sigmas else None


# ---------------------------------------------------------------------------
# Flipped b-vectors
# ---------------------------------------------------------------------------


_PERMS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
_FLIPS = ((1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1))


def table_name(perm, flip) -> str:
    axes = "xyz"
    parts = []
    if tuple(perm) != (0, 1, 2):
        parts.append("axes as " + "".join(axes[i] for i in perm))
    for i, s in enumerate(flip):
        if s < 0:
            parts.append(f"{axes[i]} reversed")
    return ", ".join(parts) or "as stored"


def coherence(fa: np.ndarray, e1: np.ndarray, mask: np.ndarray, zooms) -> float:
    """How continuous the principal directions are along themselves: the
    mean |e1(x) . e1(x + a voxel along e1)| over voxels with FA > 0.3.
    ``e1`` (x, y, z, 3) in image axes."""
    sel = mask & (fa > 0.3)
    idx = np.argwhere(sel)
    if len(idx) < 200:
        return float("nan")
    v = e1[sel]
    step = np.rint(v / (np.asarray(zooms) / np.min(zooms))).astype(int)
    out = []
    for sign in (1, -1):
        j = idx + sign * step
        ok = np.all((j >= 0) & (j < np.asarray(mask.shape)), axis=1)
        jj = j[ok]
        # Every step inside the brain counts: one that leaves the fibre lands
        # where the direction is noise (about 0.5 on average). Skipping those
        # would reward a wrong table for stepping out of the tract.
        keep = mask[tuple(jj.T)]
        a = v[ok][keep]
        b = e1[tuple(jj[keep].T)]
        out.append(np.abs((a * b).sum(axis=1)))
    vals = np.concatenate(out)
    return float(vals.mean()) if vals.size else float("nan")


def flip_check(signal: np.ndarray, coords: np.ndarray, shape, bvals, bvecs, zooms,
               use: np.ndarray) -> dict:
    """Fit the tensor under every sign flip and axis permutation of the
    table and score each by :func:`coherence`. ``signal`` voxels x volumes
    on a coarse grid at ``coords`` (voxel indices)."""
    scores = []
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    candidates = [((0, 1, 2), f) for f in _FLIPS] + [(p, (1, 1, 1)) for p in _PERMS[1:]]
    for perm, flip in candidates:
        if True:
            g = bvecs[:, perm] * np.asarray(flip, dtype=float)
            fit = fit_tensor(signal[:, use], bvals[use], g[use])
            fa = np.zeros(shape, dtype=np.float32)
            e1 = np.zeros(tuple(shape) + (3,), dtype=np.float32)
            fa[tuple(coords.T)] = fit["fa"]
            e1[tuple(coords.T)] = fit["e1"]
            scores.append((coherence(fa, e1, mask, zooms), perm, flip))
    scores = [s for s in scores if np.isfinite(s[0])]
    if not scores:
        return {}
    scores.sort(key=lambda s: -s[0])
    stored = next((s for s in scores if s[1] == (0, 1, 2) and s[2] == (1, 1, 1)), None)
    best = scores[0]
    return {"best": table_name(best[1], best[2]), "best_score": best[0],
            "stored_score": stored[0] if stored else float("nan"),
            "stored_is_best": stored is best,
            "ranking": [(table_name(p, f), round(c, 4)) for c, p, f in scores[:5]]}


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def _brain_from_tools(b0: np.ndarray, affine: np.ndarray, method: str, cancel=None,
                      timeout: float = 600.0):
    """``(mask, method name, note)``: mindgrab's brain on the mean b=0 when
    asked for and the tools can run, else ``(None, "median-Otsu (numpy)",
    why)`` (``why`` empty when median-Otsu was asked for)."""
    import tempfile

    import nibabel as nib

    from . import tools as T

    fallback = "median-Otsu (numpy)"
    if method != "mindgrab":
        return None, fallback, ""
    why = T.unavailable_reason()
    if why:
        return None, fallback, why
    folder = Path(tempfile.mkdtemp(prefix="bidsmgr-qc-b0-"))
    try:
        path = folder / "b0.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(b0, dtype=np.float32), affine), str(path))
        got = T.run_models(path, b0.shape, affine, [T.BRAIN_MODEL], cancel=cancel,
                           timeout=timeout)
        mask = stats.largest_components(got[T.BRAIN_MODEL] > 0, 1)
        if mask.sum() < 1000:
            return None, fallback, "mindgrab found no brain on the b=0"
        return mask, "mindgrab", ""
    except T.ToolFailed as exc:
        return None, fallback, str(exc)
    finally:
        import shutil

        shutil.rmtree(folder, ignore_errors=True)


def check(series: Series, *, problems: Optional[list[str]] = None,
          cancel: Optional[Callable[[], bool]] = None,
          progress: Optional[Callable[[int, int], None]] = None,
          config: Optional[QcConfig] = None) -> QCResult:
    """The quality check of one diffusion series, with the methods and
    thresholds of ``config`` (default: :data:`config.DEFAULT`)."""
    from scipy import ndimage as ndi

    from ..viz.compute import motion as MO
    from .anat import defacing_record

    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    cfg = config or DEFAULT
    D = cfg.dwi

    def step(name: str, n: int) -> None:
        if cancel is not None and cancel():
            raise MO.Cancelled()
        timings[name] = round(time.perf_counter() - t_start, 3)
        if progress is not None:
            progress(n, 10)

    from ..viz import bids as VB

    from pathlib import Path as _P

    result = QCResult(path=series.path, kind="dwi",
                      suffix=VB.suffix_of(_P(series.path).name) if series.path else "dwi")
    facts = result.facts
    zooms = series.zooms
    facts.update(shape=list(series.shape) + [series.n],
                 zooms=[round(float(z), 4) for z in zooms])
    # What it was made with, so a saved result can be reproduced.
    facts["config"] = {"methods": cfg.methods.model_dump(mode="json"),
                       "dwi": D.model_dump(mode="json")}
    table_findings, table = check_table(series.n, series.bvals, series.bvecs, problems or [],
                                        b0_max=D.b0_max)
    result.findings += table_findings
    metrics = result.metrics
    if not table:
        facts["seconds"] = round(time.perf_counter() - t_start, 2)
        return result
    bvals, bvecs = table["bvals"], table["bvecs"]
    shells = shells_of(bvals, D.b0_max)
    b0 = np.flatnonzero(bvals <= D.b0_max)
    shell_values = sorted(int(s) for s in set(shells.tolist()) if s > 0)
    facts["shells"] = {str(s): int(np.count_nonzero(shells == s)) for s in [0] + shell_values}
    for s in shell_values:
        sel = shells == s
        gap = coverage_gap_deg(bvecs[sel])
        dup = duplicates(bvecs[sel], D.duplicate_deg)
        metrics.append(_metric(f"coverage_gap_b{s}", f"Largest gap between directions (b={s})",
                               gap, unit="degrees", group="gradients", better="lower",
                               help="The radius of the largest cap of the sphere with no "
                                    "gradient direction in it, either sign: small when the "
                                    "directions are spread evenly.", fmt="{:.1f}"))
        if dup:
            result.findings.append(Finding(
                f"duplicates_b{s}", f"Repeated directions at b={s}", "info",
                f"{dup} pairs of directions are within {D.duplicate_deg:g} degrees of each "
                "other: fine for averaging, but they add no angular information.", None))
        if np.count_nonzero(sel) < 6:
            result.findings.append(Finding(
                f"few_b{s}", f"Too few directions at b={s}", "warning",
                f"{int(np.count_nonzero(sel))} directions: a tensor needs at least six.",
                None))
    step("table", 1)
    if b0.size == 0:
        facts["seconds"] = round(time.perf_counter() - t_start, 2)
        return result

    # b=0 reference, head and brain.
    frames_b0 = [series.frame(int(i)) for i in b0]
    b0_ref = np.median(np.stack(frames_b0), axis=0) if len(frames_b0) > 1 else frames_b0[0]
    zero = K.zero_fill(b0_ref)
    head = K.head(b0_ref, exclude=zero)
    brain, brain_engine, tool_note = _brain_from_tools(
        b0_ref, series.affine, cfg.methods.dwi_brain, cancel, cfg.methods.tool_timeout_s)
    if cancel is not None and cancel():
        raise MO.Cancelled()
    if brain is None:
        brain = brain_mask(b0_ref, exclude=zero)
    facts["engine"] = {"brain": brain_engine}
    if tool_note:
        facts["tool_note"] = tool_note
    if brain.sum() < 1000:
        result.findings.append(Finding("no_brain", "No brain found", "error",
                                       "The b=0 volumes hold too little signal to find the "
                                       "brain: the image measures are skipped.", None))
        facts["seconds"] = round(time.perf_counter() - t_start, 2)
        return result
    step("brain", 2)

    # Every brain voxel's signal, and the tensor fitted to it (before the
    # motion: the motion registers each volume to what the tensor predicts).
    coords = np.argwhere(brain)
    sig = np.empty((len(coords), series.n), dtype=np.float32)
    flat = np.ravel_multi_index(coords.T, series.shape)
    shell_mean: dict[int, np.ndarray] = {}
    sums = {s: np.zeros(series.shape, dtype=np.float64) for s in shell_values}
    for i in range(series.n):
        f = series.frame(i)
        sig[:, i] = f.ravel()[flat]
        if shells[i] > 0:
            sums[int(shells[i])] += f
    for s in shell_values:
        shell_mean[s] = (sums[s] / max(int(np.count_nonzero(shells == s)), 1)).astype(np.float32)
    del sums
    use = bvals <= D.dti_max_b
    can_fit = (np.count_nonzero(use & (bvals > D.b0_max)) >= 6
               and np.linalg.matrix_rank(design(bvals[use], bvecs[use])) == 7)
    fit = fit_tensor(sig[:, use], bvals[use], bvecs[use]) if can_fit else None
    step("tensor", 3)

    # Head motion. A diffusion-weighted volume's contrast depends on its
    # direction, so it is registered to what the tensor PREDICTS for that
    # direction (as FSL eddy and SHORELine do), not to its shell's mean,
    # whose contrast is another: against the mean, every volume of the
    # tutorial's single-shell scan "moved" 0.4 mm along y. Without a tensor,
    # the shell's mean is the reference.
    factor = MO.working_grid(zooms, series.shape, 3.0)
    spacing = zooms * factor
    down = lambda v: MO._down(v, factor)  # noqa: E731 - the one grid of the estimate
    params = np.zeros((series.n, 6))
    ref0 = MO.RigidReference(down(b0_ref), spacing)
    shell_refs: dict[int, MO.RigidReference] = {}
    to_b0: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if fit is None:
        for s in shell_values:
            small = down(shell_mean[s])
            try:
                shell_refs[s] = MO.RigidReference(small, spacing)
                to_b0[s] = ref0.register(small)
            except ValueError:
                pass
    last: dict[int, tuple] = {}
    lost: list[int] = []
    for i in range(series.n):
        if cancel is not None and cancel():
            raise MO.Cancelled()
        s = int(shells[i])
        frame = down(series.frame(i))
        if s == 0:
            rot, trans = ref0.register(frame, *last.get(0, (None, None)))
            last[0] = (rot, trans)
        elif fit is not None:
            predicted = shell_mean[s].copy()
            predicted.ravel()[flat] = np.exp(predict(fit["coef"], bvals[i:i + 1],
                                                     bvecs[i:i + 1])[:, 0])
            try:
                ref = MO.RigidReference(down(predicted), spacing)
                rot, trans = ref.register(frame, *last.get(s, (None, None)))
            except ValueError:
                rot, trans = np.eye(3), np.zeros(3)
            if _implausible(rot, trans, D.lost_mm, D.lost_deg):
                # Lost (a low-SNR volume, a prediction far off at high b):
                # start again from where the head was, not from the wreck.
                try:
                    rot, trans = MO.RigidReference(down(predicted), spacing).register(frame)
                except ValueError:
                    rot, trans = np.eye(3), np.zeros(3)
                if _implausible(rot, trans, D.lost_mm, D.lost_deg):
                    lost.append(i)
                    rot, trans = np.eye(3), np.zeros(3)
            else:
                last[s] = (rot, trans)
        else:
            ref = shell_refs.get(s, ref0)
            rot_v, t_v = ref.register(frame, *last.get(s, (None, None)))
            last[s] = (rot_v, t_v)
            rot_s, t_s = to_b0.get(s, (np.eye(3), np.zeros(3)))
            rot, trans = rot_v @ rot_s, rot_v @ t_s + t_v
        params[i, :3] = trans
        params[i, 3:] = MO._rotation_vector(rot)
    # Relative to the first b=0, as a reader expects motion to start at 0.
    params -= params[b0[0]]
    # Eddy currents shift a diffusion-weighted volume along the phase-
    # encoding axis by an amount linear in its gradient's components (a
    # rigid registration reads that as head motion; MRIQC's FD does). Per
    # shell, the part of each translation the gradient explains is taken as
    # eddy current, but only when it explains a real share of them: on the
    # tutorial scans it explains 4 to 8 %, and what is left is the
    # registration's own noise on single low-SNR volumes (about 0.4 mm along
    # the phase-encoding axis, white from volume to volume, where head
    # motion would carry over).
    eddy = np.zeros((series.n, 3))
    eddy_r2 = []
    for s in shell_values:
        idx = np.flatnonzero(shells == s)
        if idx.size < 10:
            continue
        x = np.column_stack([np.ones(idx.size), bvecs[idx]])
        coef, *_ = np.linalg.lstsq(x, params[idx, :3], rcond=None)
        part = bvecs[idx] @ coef[1:]
        var = params[idx, :3].var(axis=0).sum()
        r2 = float(1.0 - (params[idx, :3] - part - (params[idx, :3] - part).mean(axis=0)
                          ).var(axis=0).sum() / var) if var > 0 else 0.0
        eddy_r2.append(r2)
        if r2 >= D.eddy_explained_pct / 100.0:
            eddy[idx] = part
    params[:, :3] -= eddy
    disp = (np.abs(params[:, :3]).sum(axis=1)
            + D.fd_radius_mm * np.abs(params[:, 3:]).sum(axis=1))  # mm from the first b0
    angles = np.degrees(np.linalg.norm(params[:, 3:], axis=1))
    # The b=0 volumes: high SNR, one contrast, the head's own track.
    b0_disp = disp[b0]
    b0_fd = (MO.framewise_displacement(params[b0], D.fd_radius_mm)[1:] if b0.size > 1
             else np.zeros(0))
    # A volume far off: well outside the scan's own spread, and by more
    # than the noise of a single volume can explain.
    z_disp = np.zeros(series.n)
    for idx in [b0] + [np.flatnonzero(shells == s) for s in shell_values]:
        if idx.size >= 4:
            z_disp[idx] = stats.robust_z(disp[idx])
    far = np.flatnonzero((z_disp > 5.0) & (disp > D.far_mm))
    fd = MO.framewise_displacement(params, D.fd_radius_mm)
    fd_mean = float(np.nanmean(fd)) if np.isfinite(fd).any() else None
    metrics += [
        _metric("motion_b0", "Head motion across the b=0 volumes",
                float(b0_disp.max()) if b0_disp.size else None, unit="mm",
                group="motion", better="lower", fmt="{:.2f}",
                why="Needs two b=0 volumes or more.",
                help="How far the head moved at most between the first b=0 and any other, "
                     "as framewise displacement measures it (translations plus rotations on "
                     "a 50 mm sphere). The b=0 volumes are the reliable record of the head: "
                     "high signal, one contrast."),
        _metric("fd_b0_max", "Largest step between b=0 volumes",
                float(b0_fd.max()) if b0_fd.size else None, unit="mm", group="motion",
                better="lower", fmt="{:.2f}", why="Needs two b=0 volumes or more.",
                help="The largest change between consecutive b=0 volumes."),
        _metric("volumes_far", "Volumes far out of place", float(far.size), group="motion",
                better="lower", fmt="{:.0f}",
                help=f"Volumes displaced by more than {D.far_mm:g} mm from the first b=0 and "
                     "far outside the spread of their shell (robust z above 5): a movement "
                     "during that volume."),
        _metric("fd_mean", "Mean framewise displacement", fd_mean, unit="mm",
                group="motion", better="lower", mriqc="fd_mean", fmt="{:.3f}",
                help="Mean change from one volume to the next over the whole series, each "
                     "diffusion-weighted volume registered to what the tensor predicts for "
                     "it. Includes the registration's noise on single low-SNR volumes, as "
                     "MRIQC's does: read it against other scans of the same protocol, not "
                     "against a BOLD threshold."),
        _metric("eddy_shift", "Eddy-current shift",
                float(np.abs(eddy).max()) if np.abs(eddy).max() > 0 else 0.0,
                unit="mm", group="motion", better="lower", fmt="{:.2f}",
                help="The largest shift of a diffusion-weighted volume explained by its "
                     "gradient direction (linear in its components, per shell), counted only "
                     f"when the gradient explains at least {D.eddy_explained_pct:.0f} % of the "
                     "shifts: eddy currents, which distortion correction removes."),
        _metric("rotation_max", "Largest rotation of the gradients", float(angles.max()),
                unit="degrees", group="motion", better="lower", fmt="{:.2f}",
                help="How far the head (and so every gradient direction relative to it) "
                     "turned at most, against the first b=0. Above a few degrees, the "
                     "b-vectors should be rotated with the motion correction."),
    ]
    facts["eddy_explained"] = [round(r, 3) for r in eddy_r2]
    facts["registration_lost"] = lost
    if lost:
        result.findings.append(Finding(
            "lost", "Volumes the motion check could not place", "info",
            f"{len(lost)} volume{'s' if len(lost) > 1 else ''} could not be registered "
            f"(volume{'s' if len(lost) > 1 else ''} {', '.join(str(v) for v in lost[:10])}"
            + (" and more" if len(lost) > 10 else "") + "): their motion is not known, "
            "often because the signal is too low (a high b-value).",
            {"volume": int(lost[0]), "slice": None}))
    result.tracks["displacement"] = {
        "title": "Displacement from the first b=0", "unit": "mm", "ys": [disp],
        "ticks": far.astype(float), "rule": D.far_mm, "colour": "accent",
        "summary": ((f"b=0 volumes up to {b0_disp.max():.2f} mm; " if b0_disp.size else "")
                    + (f"{far.size} volume{'s' if far.size != 1 else ''} far out of place"
                       if far.size else "no volume far out of place")),
        "help": "How far each volume sits from the first b=0 (translations plus rotations "
                "on a 50 mm sphere). The b=0 volumes are the head's reliable track; the "
                "diffusion-weighted ones scatter by the registration's noise. Marked: "
                "volumes far out of place."}
    result.tracks["translation"] = {"title": "Translation", "unit": "mm",
                                    "ys": [params[:, 0], params[:, 1], params[:, 2]],
                                    "legend": ["x", "y", "z"],
                                    "help": "Where each volume sits against the first b=0, "
                                            "mm, eddy-current shifts taken out when the "
                                            "gradients explain them."}
    deg = np.degrees(params[:, 3:])
    result.tracks["rotation"] = {"title": "Rotation", "unit": "degrees",
                                 "ys": [deg[:, 0], deg[:, 1], deg[:, 2]],
                                 "legend": ["pitch", "roll", "yaw"],
                                 "help": "How each volume is turned against the first b=0."}
    if np.abs(eddy).max() > 0:
        result.tracks["eddy"] = {"title": "Eddy-current shift", "unit": "mm",
                                 "ys": [eddy[:, 0], eddy[:, 1], eddy[:, 2]],
                                 "legend": ["x", "y", "z"],
                                 "help": "The shift of each diffusion-weighted volume its "
                                         "gradient direction explains: eddy currents."}
    step("motion", 4)

    # Signal drift, from the b=0 volumes in acquisition order.
    signal_b0 = np.array([float(np.median(f[brain])) for f in frames_b0])
    if b0.size >= 2 and signal_b0[0] > 0:
        rel = signal_b0 / signal_b0[0]
        slope, _icpt = np.polyfit(b0.astype(float), np.log(np.maximum(rel, 1e-6)), 1)
        drift = (np.exp(slope * (series.n - 1)) - 1.0) * 100.0
        metrics.append(_metric("drift", "Signal drift over the scan", drift, unit="%",
                               group="motion", better="", fmt="{:+.2f}",
                               help="Change of the b=0 signal from the first volume to the "
                                    "last, from a fit through the b=0 volumes. A few percent "
                                    "is the scanner warming; it biases every diffusion "
                                    "measure unless corrected."))
        if abs(drift) > D.drift_pct:
            result.findings.append(Finding(
                "drift", "Signal drift", "warning",
                f"The b=0 signal changes by {drift:+.1f} % over the scan: correct the drift "
                "before fitting.", None))
        b0_track = np.full(series.n, np.nan)
        b0_track[b0] = rel * 100.0
        result.tracks["b0_signal"] = {"title": "b=0 signal", "unit": "% of the first",
                                      "ys": [b0_track], "colour": "purple", "points": True,
                                      "summary": f"drift {drift:+.2f} %",
                                      "help": "The brain's median signal in each b=0 volume, "
                                              "against the first: a slope is drift."}
    else:
        metrics.append(_metric("drift", "Signal drift over the scan", None, unit="%",
                               group="motion", why="Needs two b=0 volumes or more."))
    del frames_b0
    step("drift", 4)

    # What the tensor predicts, against what is there.
    fa_map = md_map = None
    if fit is not None:
        dm = design(bvals, bvecs).astype(np.float32)
        resid = np.empty_like(sig)                              # voxels x volumes
        for i in range(0, len(sig), CHUNK):
            resid[i:i + CHUNK] = (np.log(np.maximum(sig[i:i + CHUNK], 1e-3))
                                  - fit["coef"][i:i + CHUNK].astype(np.float32) @ dm.T)
        fa_map = np.zeros(series.shape, dtype=np.float32)
        md_map = np.zeros(series.shape, dtype=np.float32)
        fa_map[tuple(coords.T)] = fit["fa"]
        md_map[tuple(coords.T)] = fit["md"] * 1000.0
        metrics += [
            _metric("fa_median", "Median FA in the brain", float(np.median(fit["fa"])),
                    group="diffusion", fmt="{:.3f}",
                    help="Fractional anisotropy of the tensor, median over the brain."),
            _metric("md_median", "Median MD in the brain", float(np.median(fit["md"])) * 1000,
                    unit="um^2/ms", group="diffusion", fmt="{:.3f}",
                    help="Mean diffusivity, median over the brain (about 0.7 to 0.9 in "
                         "tissue, 3 in CSF at body temperature)."),
            _metric("negative_eigen", "Voxels with a negative eigenvalue",
                    100.0 * float(fit["negative"].mean()), unit="%", group="diffusion",
                    better="lower", fmt="{:.2f}",
                    help="Share of the brain where the fitted tensor is not physical: noise, "
                         "motion or artefacts larger than the diffusion contrast."),
        ]
    else:
        resid = None
        for key, title in (("fa_median", "Median FA in the brain"),
                           ("md_median", "Median MD in the brain"),
                           ("negative_eigen", "Voxels with a negative eigenvalue")):
            metrics.append(_metric(key, title, None, why="A tensor needs six or more "
                                   f"directions at b <= {D.dti_max_b:g} and a b=0."))
    step("residuals", 5)

    # Per slice and volume: dropout, interleave, spikes.
    slice_axis = {"i": 0, "j": 1, "k": 2}.get(
        str(series.sidecar.get("SliceEncodingDirection", "k"))[:1], 2)
    n_slices = series.shape[slice_axis]
    if resid is not None:
        zslice = coords[:, slice_axis]
        counts = np.bincount(zslice, minlength=n_slices)
        sums = np.zeros((n_slices, series.n))
        np.add.at(sums, zslice, resid)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_slice = sums / counts[:, None]
        # A slice is judged when it holds a fair share of brain: the top
        # and bottom slices hold a sliver whose mean moves with every
        # tenth of a millimetre of motion, and read as dropouts.
        typical = float(np.median(counts[counts > 0])) if (counts > 0).any() else 0.0
        judged = counts >= max(MIN_SLICE_VOXELS, D.edge_slice_pct / 100.0 * typical)
        z = np.zeros_like(per_slice)
        groups = [np.flatnonzero(shells == s) for s in [0] + shell_values]
        for idx in groups:
            if idx.size < 4:
                continue
            block = per_slice[:, idx]
            z[:, idx] = stats.robust_z(block, axis=1)
        z[~judged] = 0.0
        typical = stats.nanmedian_rows(per_slice)
        # Log ratios: losing a share p is log(1 - p) below the prediction.
        loss = -float(np.log1p(-min(D.dropout_loss_pct, 99.0) / 100.0))
        drop = (z < -D.dropout_z) & (per_slice - typical < -loss)
        drop &= judged[:, None]
        dropped = np.argwhere(drop)                              # (slice, volume)
        vols_drop = np.unique(dropped[:, 1]) if len(dropped) else np.empty(0, int)
        metrics.append(_metric(
            "dropout_slices", "Slices with signal dropout", float(len(dropped)),
            group="artefacts", better="lower", fmt="{:.0f}",
            help="Slices (in any volume) whose signal falls far below what the tensor "
                 "predicts for that direction, against the same slice in the shell's other "
                 "volumes: motion during the diffusion encoding, or a vibration artefact."))
        if len(dropped):
            worst = sorted(dropped.tolist(), key=lambda sv: z[sv[0], sv[1]])[:8]
            result.findings.append(Finding(
                "dropout", "Signal dropout", "error" if len(vols_drop) > 0.05 * series.n
                else "warning",
                f"{len(dropped)} slice{'s' if len(dropped) != 1 else ''} in "
                f"{len(vols_drop)} volume{'s' if len(vols_drop) != 1 else ''} lost signal: "
                + ", ".join(f"volume {v} slice {s_}" for s_, v in worst)
                + (" and more" if len(dropped) > 8 else "") + ".",
                {"volume": int(worst[0][1]), "slice": int(worst[0][0]),
                 "axis": slice_axis}))
        # Interleave: odd slices against even ones, per volume.
        odd = judged & (np.arange(n_slices) % 2 == 1)
        even = judged & (np.arange(n_slices) % 2 == 0)
        inter = np.zeros(series.n)
        if odd.any() and even.any():
            diff = np.nanmedian(per_slice[odd], axis=0) - np.nanmedian(per_slice[even], axis=0)
            for idx in groups:
                if idx.size >= 4:
                    inter[idx] = stats.robust_z(diff[idx])
            bad_inter = np.flatnonzero((np.abs(inter) > D.interleave_z)
                                       & (np.abs(diff) > 0.05))
        else:
            bad_inter = np.empty(0, int)
        metrics.append(_metric(
            "interleave_volumes", "Volumes with an interleave artefact",
            float(bad_inter.size), group="artefacts", better="lower", fmt="{:.0f}",
            help="Volumes whose odd slices differ from their even slices far more than "
                 "in the rest of the shell: motion between the two passes of an "
                 "interleaved acquisition."))
        if bad_inter.size:
            result.findings.append(Finding(
                "interleave", "Interleave artefact", "warning",
                f"Odd and even slices disagree in volumes "
                f"{', '.join(str(v) for v in bad_inter[:10])}"
                + (" and more" if bad_inter.size > 10 else "") + ".",
                {"volume": int(bad_inter[0]), "slice": None, "axis": slice_axis}))
        # Spikes: single voxels far above the prediction.
        spike_count = np.zeros(series.n)
        for idx in groups:
            if idx.size < 4:
                continue
            # Each voxel against ITSELF across the shell: the tensor's own
            # misfit (CSF, crossing fibres at high b) is the same in every
            # volume and is not a spike.
            for i in range(0, len(resid), CHUNK):
                r = resid[i:i + CHUNK][:, idx]
                med = np.median(r, axis=1, keepdims=True)
                spread = np.median(np.abs(r - med), axis=1, keepdims=True) * stats.MAD_TO_SD
                spread = np.maximum(spread, 0.05)
                spike_count[idx] += np.count_nonzero(
                    ((r - med) > D.spike_z * spread) & ((r - med) > 0.5), axis=0)
        spike_ppm = spike_count / len(coords) * 1e6
        metrics.append(_metric(
            "spikes_ppm", "Spiking voxels", float(spike_ppm.mean()), unit="ppm",
            group="artefacts", better="lower", fmt="{:.1f}",
            help=f"Voxels more than {D.spike_z:g} robust spreads (and 65 %) above the "
                 "tensor's prediction, against the same voxel in the shell's other "
                 "volumes, per million brain voxels per volume."))
        result.tracks["slices"] = {
            "kind": "image", "title": "Slice signal against the tensor",
            # The top slice at the top, as in the head.
            "image": np.clip(np.where(judged[:, None], z, np.nan), -10, 10)[::-1]
            .astype(np.float32),
            "levels": (-8.0, 8.0), "value_name": "z", "colormap": "diverging",
            "rows": [f"slice {k}" for k in range(n_slices - 1, -1, -1)], "height": 2.5,
            "flagged": dropped.tolist(),
            "summary": ((f"{len(dropped)} dropped slice{'s' if len(dropped) != 1 else ''} in "
                         f"{len(vols_drop)} volume{'s' if len(vols_drop) != 1 else ''}")
                        if len(dropped) else "no dropout"),
            "help": "Every slice (rows, the top slice at the top) of every volume (columns): its signal "
                    "against what the tensor predicts, as a robust z within its shell. Blue "
                    "is signal lost (a dropout), red is signal added. Click a cell to go to "
                    "that volume and slice."}
        result.tracks["spikes"] = {"title": "Spiking voxels", "unit": "ppm", "ys": [spike_ppm],
                                   "colour": "warning",
                                   "help": "Voxels far above the tensor's prediction."}
        result.tracks["interleave"] = {"title": "Odd against even slices", "unit": "z",
                                       "ys": [inter], "rule": D.interleave_z, "colour": "teal",
                                       "ticks": bad_inter.astype(float),
                                       "help": "Odd slices minus even slices against the "
                                               "prediction, robust z within the shell."}
    else:
        for key, title in (("dropout_slices", "Slices with signal dropout"),
                           ("interleave_volumes", "Volumes with an interleave artefact"),
                           ("spikes_ppm", "Spiking voxels")):
            metrics.append(_metric(key, title, None, group="artefacts",
                                   why="Judged against the tensor's prediction, which "
                                       "could not be fitted."))
    step("slices", 6)

    # Neighbouring directions in q-space.
    weighted = np.flatnonzero(bvals > D.b0_max)
    if weighted.size >= 2:
        q = np.sqrt(bvals[weighted])[:, None] * bvecs[weighted]
        dist = np.minimum(np.linalg.norm(q[:, None] - q[None], axis=2),
                          np.linalg.norm(q[:, None] + q[None], axis=2))
        np.fill_diagonal(dist, np.inf)
        nearest = np.argmin(dist, axis=1)
        sub = sig[:: max(1, len(sig) // 50000)].astype(np.float64)
        sub = sub - sub.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(sub, axis=0)
        ndc_v = np.full(series.n, np.nan)
        for a_i, b_i in zip(weighted, weighted[nearest]):
            denom = norm[a_i] * norm[b_i]
            ndc_v[a_i] = float(sub[:, a_i] @ sub[:, b_i]) / denom if denom > 0 else np.nan
        metrics.append(_metric(
            "ndc", "Neighbouring direction correlation", float(np.nanmean(ndc_v)),
            group="diffusion", better="higher", mriqc="ndc", fmt="{:.3f}",
            help="Mean correlation of each diffusion volume with its nearest neighbour in "
                 "q-space (Yeh 2019): a volume unlike its neighbours is corrupted."))
        low = np.flatnonzero(stats.robust_z(np.nan_to_num(ndc_v[weighted], nan=1.0)) < -5)
        result.tracks["ndc"] = {"title": "Correlation with the nearest direction", "unit": "r",
                                "ys": [ndc_v], "colour": "blue", "points": True,
                                "ticks": weighted[low].astype(float),
                                "help": "Each diffusion volume's correlation with the volume "
                                        "nearest it in q-space."}
    step("neighbours", 7)

    # Noise and SNR.
    eroded = ndi.binary_erosion(brain, iterations=2)
    sigma_b0 = None
    if b0.size >= 3:
        stack = np.stack([series.frame(int(i))[eroded] for i in b0], axis=1)
        sigma_b0 = float(np.median(stack.std(axis=1, ddof=1)))
    biggest = max(shell_values, key=lambda s: np.count_nonzero(shells == s)) if shell_values else None
    sigma_pca = (mppca_noise(series, np.flatnonzero(shells == biggest), eroded)
                 if biggest is not None else None)
    metrics.append(_metric("sigma_b0", "Noise from the b=0 repeats", sigma_b0, group="noise",
                           fmt="{:.3g}", why="Needs three b=0 volumes or more.",
                           help="Median over the brain of the SD across the b=0 volumes "
                                "(motion between them adds to it)."))
    metrics.append(_metric("sigma_pca", "Noise by MP-PCA", sigma_pca, group="noise",
                           mriqc="sigma_pca", fmt="{:.3g}",
                           help="Marchenko-Pastur PCA (Veraart 2016) on sampled patches of "
                                "the largest shell, median over patches."))
    # MP-PCA first: the b=0 repeats carry the motion between them.
    sigma = sigma_pca if sigma_pca else sigma_b0
    facts["sigma_used"] = "MP-PCA" if sigma_pca else "b=0 repeats" if sigma_b0 else ""
    if sigma and fa_map is not None:
        world_e1 = world_directions(fit["e1"], series.affine)
        world_xyz = coords @ series.affine[:3, :3].T + series.affine[:3, 3]
        cx = float(np.median(world_xyz[:, 0]))
        cc_sel = ((fit["fa"] > 0.4) & (np.abs(world_e1[:, 0]) > 0.85)
                  & (np.abs(world_xyz[:, 0] - cx) < 10.0))
        cc = np.zeros(series.shape, dtype=bool)
        cc[tuple(coords[cc_sel].T)] = True
        cc = stats.largest_components(cc, 1)
        cc_in = cc[tuple(coords.T)]
        facts["cc_voxels"] = int(cc_in.sum())
        if cc_in.sum() >= 5:
            snr0 = float(np.mean(sig[cc_in][:, b0])) / sigma
            metrics.append(_metric("snr_cc_b0", "SNR in the corpus callosum (b=0)", snr0,
                                   group="noise", better="higher", mriqc="snr_cc_shell0",
                                   fmt="{:.1f}",
                                   help="Mean b=0 signal in the corpus callosum over the noise."))
            gw = world_directions(bvecs, series.affine)
            for s in shell_values:
                idx = np.flatnonzero(shells == s)
                along = idx[np.argmax(np.abs(gw[idx, 0]))]
                across = idx[np.argsort(np.abs(gw[idx, 0]))[:2]]
                metrics.append(_metric(
                    f"snr_cc_b{s}_along", f"SNR in the corpus callosum, b={s}, along",
                    float(np.mean(sig[cc_in][:, along])) / sigma, group="noise",
                    better="higher", fmt="{:.1f}",
                    help="Gradient along the callosal fibres (left-right): the most "
                         "attenuated signal, the worst case."))
                metrics.append(_metric(
                    f"snr_cc_b{s}_across", f"SNR in the corpus callosum, b={s}, across",
                    float(np.mean(sig[cc_in][:, across])) / sigma, group="noise",
                    better="higher", fmt="{:.1f}",
                    help="Gradients across the callosal fibres: the least attenuated."))
            result.maps["cc"] = QCMap("cc", "Corpus callosum", cc.astype(np.uint8),
                                      series.affine, "mask",
                                      "Where the callosal SNR is measured: FA above 0.4, "
                                      "left-right, at the midline.", colour="warm")
        top = max(shell_values)
        idx = np.flatnonzero(shells == top)
        # Tissue only: in CSF the signal is meant to vanish at high b.
        tissue = fit["md"] * 1000.0 < 1.5
        floor = (100.0 * float(np.mean(sig[tissue][:, idx] < 2.0 * sigma))
                 if tissue.any() else None)
        metrics.append(_metric(
            "noise_floor", f"Signal at the noise floor (b={top})", floor, unit="%",
            group="noise", better="lower", fmt="{:.1f}",
            help="Share of the highest shell's signal in tissue (MD below 1.5, so not "
                 "CSF) under twice the noise SD, where magnitude data is biased upwards "
                 "(the Rician floor)."))
    step("noise", 8)

    # Shell means: EFC and FBER (air measures, only for an image not defaced).
    record = defacing_record(series.sidecar)
    air_ok = not record
    if not air_ok:
        facts["air_reason"] = f"The image is defaced {record}: its air is not measured."
    air = K.air(head, excluded=zero, margin=2) if air_ok else None
    for s in [0] + shell_values:
        mean_img = b0_ref if s == 0 else shell_mean[s]
        name = "b=0" if s == 0 else f"b={s}"
        if air_ok and air is not None and air.sum() > 500:
            x = np.clip(mean_img[~zero], 0, None).astype(np.float64)
            n = x.size
            bmax = float(np.sqrt(np.sum(x * x)))
            efc = (float(np.sum((x / bmax) * np.log((x + 1e-16) / bmax)))
                   / (n * (1 / np.sqrt(n)) * np.log(1 / np.sqrt(n)))) if bmax > 0 else None
            air_e = float(np.median(mean_img[air] ** 2))
            fber = float(np.median(mean_img[head] ** 2)) / air_e if air_e > 1e-6 else None
        else:
            efc = fber = None
        why = facts.get("air_reason", "Too little air in the field of view.")
        metrics.append(_metric(f"efc_b{s}", f"Entropy focus criterion ({name})", efc,
                               group="artefacts", better="lower", fmt="{:.3f}", why=why,
                               help="Ghosting and blurring of the shell's mean image."))
        metrics.append(_metric(f"fber_b{s}", f"Foreground to background energy ({name})",
                               fber, group="artefacts", better="higher", fmt="{:.0f}", why=why,
                               help="Median energy of the head over that of the air."))
    step("shells", 9)

    # Flipped or swapped b-vectors.
    if D.check_flips and can_fit:
        fdown = MO.working_grid(zooms, series.shape, 4.0)
        small_brain = stats.block_mean(brain.astype(np.float32), fdown) > 0.5
        sc = np.argwhere(small_brain)
        if len(sc) > 500:
            sflat = np.ravel_multi_index(sc.T, small_brain.shape)
            ssig = np.empty((len(sc), series.n), dtype=np.float32)
            for i in range(series.n):
                ssig[:, i] = stats.block_mean(series.frame(i), fdown).ravel()[sflat]
            fc = flip_check(ssig, sc, small_brain.shape, bvals, bvecs, zooms * fdown, use)
            facts["flip_check"] = fc
            margin = 1.0 + D.flip_margin_pct / 100.0
            if fc and not fc["stored_is_best"] and fc["best_score"] > margin * fc["stored_score"]:
                result.findings.append(Finding(
                    "bvec_flip", "The b-vectors may be flipped or swapped", "warning",
                    f"The tensor's directions are most continuous with the table "
                    f"{fc['best']} (coherence {fc['best_score']:.3f}) rather than as stored "
                    f"({fc['stored_score']:.3f}). Experimental: check the colour FA before "
                    "changing the .bvec.", "fa"))
    step("flips", 10)

    # Findings from motion.
    if b0_disp.size and b0_disp.max() > D.motion_b0_mm:
        result.findings.append(Finding(
            "motion", "Head motion", "warning" if b0_disp.max() < 3 * D.motion_b0_mm else "error",
            f"The head moved up to {b0_disp.max():.1f} mm between the b=0 volumes.",
            {"volume": int(b0[int(np.argmax(b0_disp))]), "slice": None}))
    if far.size:
        result.findings.append(Finding(
            "far", "Volumes out of place", "warning" if far.size < 0.05 * series.n else "error",
            (f"{far.size} volumes sit" if far.size > 1 else "1 volume sits")
            + f" far from the rest of their shell (more than {D.far_mm:g} mm): "
            + ("volumes " if far.size > 1 else "volume ") + ", ".join(str(v) for v in far[:10])
            + (" and more" if far.size > 10 else "") + ".",
            {"volume": int(far[0]), "slice": None}))
    if angles.max() > 3.0:
        result.findings.append(Finding(
            "rotation", "The gradients turned with the head", "info",
            f"The head turned up to {angles.max():.1f} degrees: rotate the b-vectors with the "
            "motion correction.", None))

    # Maps for the viewer.
    result.maps["brain"] = QCMap("brain", "Brain mask", brain.astype(np.uint8), series.affine,
                                 "mask", "Where the diffusion measures are taken ("
                                 + facts["engine"]["brain"] + ").", colour="red")
    if fa_map is not None:
        result.maps["fa"] = QCMap("fa", "Fractional anisotropy", fa_map, series.affine,
                                  "field", "FA of the tensor fitted here.", colour="gray")
        result.maps["md"] = QCMap("md", "Mean diffusivity", md_map, series.affine, "field",
                                  "MD of the tensor, um^2/ms.", colour="viridis")
    facts["seconds"] = round(time.perf_counter() - t_start, 2)
    facts["timings"] = timings
    facts["b0_volumes"] = b0.tolist()
    facts["shell_of_volume"] = shells.tolist()
    return result


def check_file(path, *, root=None, cancel=None, progress=None,
               config: Optional[QcConfig] = None) -> QCResult:
    """Read ``path``, its gradients and sidecar, and check it."""
    from .series import from_file

    series, problems = from_file(path, root)
    return check(series, problems=problems, cancel=cancel, progress=progress, config=config)


__all__ = ["check", "check_file", "check_table", "coherence", "coverage_gap_deg",
           "design", "duplicates", "fit_tensor", "flip_check", "mppca_noise", "mppca_sigma",
           "predict", "shells_of", "world_directions"]

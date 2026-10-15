"""Affine registration of an anatomical image to the bundled template.

What it is for: where the brain is (the template's brain mask), what tissue
to expect where (the template's tissue maps, the segmentation's priors),
where the face and the neck are (so the air the artefact measures use is
the air above them), and how much of the brain the field of view cuts off.
None of those needs a nonlinear registration; MRIQC's default is affine too.

Two stages, both on a sample of template points, solved with
``scipy.optimize.least_squares``:

1. **The head's outline.** The template's smoothed head mask against the
   image's, nine parameters (translation, rotation, a scale per axis). It
   needs no contrast in common, so it works for any anatomical image, and it
   starts from the top of the head and the head's centre, which a field of
   view showing the shoulders does not move.
2. **The intensities** (T1w and T2w, which have a template of their own):
   all twelve parameters, maximising the correlation inside the template's
   brain, at 4 mm and then 2 mm.

The result maps template millimetres to image millimetres (``matrix``). A
registration whose final correlation is low is reported as uncertain, and
what depends on it says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import stats
from .templates import Template, template

#: Template points each stage registers on.
POINTS = 12_000
#: Bounds: a head is not half or twice the template's size, nor turned
#: more than about 35 degrees in a scanner.
SCALE_BOUNDS = (np.log(0.7), np.log(1.45))
ROT_BOUND = 0.6
SHEAR_BOUND = 0.15
#: Below this correlation the registration is reported as uncertain. Low,
#: because two contrasts that are only alike (a TSE T2 against the
#: template's) correlate at about 0.45 when aligned; ``anat`` also checks
#: that the template's brain lands inside the head.
GOOD_CORRELATION = 0.3


class Cancelled(Exception):
    pass


@dataclass
class Registration:
    #: 4 x 4: template world mm -> image world mm.
    matrix: np.ndarray
    #: Final correlation of the last stage (outline or intensity).
    correlation: float
    #: Which stages ran ("outline", "intensity").
    stages: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def uncertain(self) -> bool:
        return not np.isfinite(self.correlation) or self.correlation < GOOD_CORRELATION


def _rotation(w) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(w, dtype=float) / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def matrix_of(p: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """The 4 x 4 of parameters ``p`` (translation 3, rotation vector 3, log
    scale 3, shear 3), about ``centre`` (template mm)."""
    p = np.asarray(p, dtype=float)
    lin = _rotation(p[3:6])
    shear = np.eye(3)
    if p.size >= 12:
        shear[0, 1], shear[0, 2], shear[1, 2] = p[9], p[10], p[11]
    scale = np.diag(np.exp(p[6:9]))
    lin = lin @ shear @ scale
    out = np.eye(4)
    out[:3, :3] = lin
    out[:3, 3] = centre + p[:3] - lin @ centre
    return out


def _level(volume: np.ndarray, affine: np.ndarray, target_mm: float):
    zooms = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))
    f = stats.block_factor(zooms, target_mm)
    return stats.block_mean(volume, f), stats.block_affine(affine, f)


def _sampler(volume: np.ndarray, affine: np.ndarray):
    """A function: world mm points (N x 3) -> (trilinear values, inside),
    ``inside`` False for points outside the volume's field of view."""
    from scipy import ndimage as ndi

    inv = np.linalg.inv(affine)
    vol = np.asarray(volume, dtype=np.float32)
    top = np.asarray(vol.shape[:3], dtype=float) - 0.5

    def sample(points: np.ndarray):
        vox = points @ inv[:3, :3].T + inv[:3, 3]
        inside = np.all((vox > -0.5) & (vox < top), axis=1)
        return ndi.map_coordinates(vol, vox.T, order=1, mode="nearest",
                                   prefilter=False), inside
    return sample


def _prior(p: np.ndarray) -> np.ndarray:
    """A weak pull of the scales towards 1 and the shears towards 0, so a
    field of view that shows part of the head (a slab) cannot be matched
    by squashing the template into it."""
    out = [p[6:9] / 0.15]
    if p.size >= 12:
        out.append(p[9:12] / 0.05)
    return np.concatenate(out)


def _zscore(v: np.ndarray) -> np.ndarray:
    sd = float(v.std())
    return (v - float(v.mean())) / sd if sd > 1e-12 else v * 0.0


def _points(tpl_volume: np.ndarray, tpl_affine: np.ndarray, where: np.ndarray,
            n: int, seed: int = 7) -> np.ndarray:
    idx = np.argwhere(where)
    if len(idx) > n:
        rng = np.random.default_rng(seed)
        idx = idx[np.sort(rng.choice(len(idx), n, replace=False))]
    return idx @ tpl_affine[:3, :3].T + tpl_affine[:3, 3]


def _top_and_centre(mask: np.ndarray, affine: np.ndarray) -> Optional[np.ndarray]:
    """(x, y) of the head's centre over its top 100 mm, and z of its top,
    in world mm. The neck and shoulders a large field of view shows do not
    move either."""
    idx = np.argwhere(mask)
    if len(idx) < 100:
        return None
    world = idx @ affine[:3, :3].T + affine[:3, 3]
    top = float(np.percentile(world[:, 2], 99.8))
    upper = world[world[:, 2] > top - 100.0]
    return np.array([upper[:, 0].mean(), upper[:, 1].mean(), top])


def _search_translation(p: np.ndarray, centre: np.ndarray, pts: np.ndarray,
                        t_vals: np.ndarray, sample, *, reach=(12.0, 16.0, 40.0),
                        step: float = 4.0) -> np.ndarray:
    """The translation, on a grid around ``p``'s, at which the intensities
    correlate best with the template's (over points inside the field of
    view, at least half of them). The outline leaves a slab's position
    along its axis open, and the correlation of two contrasts that are only
    alike (a TSE T2 against the template's) is too flat for a local fit to
    find from far away."""
    m0 = matrix_of(p, centre)
    moved0 = pts @ m0[:3, :3].T + m0[:3, 3]
    best, best_shift = -np.inf, np.zeros(3)
    axes = [np.arange(-r, r + 1e-9, step) for r in reach]
    for dx in axes[0]:
        for dy in axes[1]:
            for dz in axes[2]:
                values, inside = sample(moved0 + (dx, dy, dz))
                if inside.mean() < 0.5:
                    continue
                v, t = values[inside], t_vals[inside]
                if v.std() <= 0 or t.std() <= 0:
                    continue
                c = float(np.corrcoef(v, t)[0, 1])
                if c > best:
                    best, best_shift = c, np.array([dx, dy, dz])
    out = np.array(p, dtype=float)
    # A translation in image mm, expressed in the parameters: added to t.
    out[:3] = out[:3] + best_shift
    return out


def _cut_at_top(head: np.ndarray, affine: np.ndarray) -> bool:
    """Whether the head reaches the top face of the field of view (in world
    terms: the face of the volume most towards +z)."""
    lin = np.asarray(affine)[:3, :3]
    axis = int(np.argmax(np.abs(lin[2])))
    last = -1 if lin[2, axis] > 0 else 0
    face = np.take(np.asarray(head, dtype=bool), [last], axis=axis)
    return bool(face.sum() > 0.02 * max(face.size, 1))


def register(image: np.ndarray, affine: np.ndarray, head: np.ndarray, *,
             contrast: str = "", cancel: Optional[Callable[[], bool]] = None,
             tpl: Optional[Template] = None) -> Registration:
    """Register ``image`` (with its ``head`` mask, same grid) to the template.
    ``contrast`` "T1w" or "T2w" adds the intensity stage."""
    import time

    from scipy import ndimage as ndi
    from scipy.optimize import least_squares

    t0 = time.perf_counter()
    tpl = tpl or template()
    affine = np.asarray(affine, dtype=float)

    def check() -> None:
        if cancel is not None and cancel():
            raise Cancelled()

    # --- stage 1: the head's outline, at 4 mm --------------------------------
    head_f = np.asarray(head, dtype=np.float32)
    sub_head, sub_aff = _level(head_f, affine, 4.0)
    sub_head = ndi.gaussian_filter(sub_head, 1.0)
    tpl_head, tpl_aff4 = _level(tpl["head"], tpl.affine, 4.0)
    tpl_head = ndi.gaussian_filter(tpl_head, 1.0)
    centre = (tpl.affine[:3, :3] @ ((np.asarray(tpl.shape) - 1) / 2.0)) + tpl.affine[:3, 3]

    start_t = np.zeros(3)
    a = _top_and_centre(head > 0, affine)
    b = _top_and_centre(tpl["head"] > 0.5, tpl.affine)
    if a is not None and b is not None:
        start_t = a - b
        if _cut_at_top(head, affine):
            # A slab: the top of the field of view is not the top of the
            # head. Its centre is about the brain's.
            mid = affine[:3, :3] @ ((np.asarray(head.shape) - 1) / 2.0) + affine[:3, 3]
            brain = np.argwhere(tpl["brain"] > 0.5)
            brain_c = (brain @ tpl.affine[:3, :3].T + tpl.affine[:3, 3]).mean(axis=0)
            start_t[2] = mid[2] - brain_c[2]
    near = ndi.binary_dilation(tpl_head > 0.1, iterations=2)
    pts = _points(tpl_head, tpl_aff4, near, POINTS)
    t_vals = _sampler(tpl_head, tpl_aff4)(pts)[0]
    s_sample = _sampler(sub_head, sub_aff)

    def outline(p9: np.ndarray) -> np.ndarray:
        check()
        m = matrix_of(p9, centre)
        moved = pts @ m[:3, :3].T + m[:3, 3]
        values, inside = s_sample(moved)
        # Outside the field of view there is nothing to compare, and the
        # cost is a MEAN over what is inside: summed, pushing the head out
        # of a slab would lower it.
        n_in = max(int(inside.sum()), 1)
        r = (values - t_vals) * inside * np.sqrt(inside.size / n_in)
        return np.concatenate([r, _prior(p9)])

    lo = np.r_[-80.0, -80.0, -80.0, [-ROT_BOUND] * 3, [SCALE_BOUNDS[0]] * 3]
    hi = np.r_[80.0, 80.0, 80.0, [ROT_BOUND] * 3, [SCALE_BOUNDS[1]] * 3]
    p0 = np.r_[np.clip(start_t, lo[:3] + 1, hi[:3] - 1), np.zeros(6)]
    fit = least_squares(outline, p0, bounds=(lo, hi), method="trf",
                        x_scale=np.r_[[10.0] * 3, [0.1] * 3, [0.05] * 3],
                        diff_step=1e-3, max_nfev=200)
    p = np.r_[fit.x, np.zeros(3)]
    stages = ["outline"]
    m1 = matrix_of(fit.x, centre)
    vals1, in1 = s_sample(pts @ m1[:3, :3].T + m1[:3, 3])
    corr = (float(np.corrcoef(vals1[in1], t_vals[in1])[0, 1])
            if in1.sum() > 100 and t_vals[in1].std() > 0 else 0.0)

    # --- stage 2: intensities, T1w and T2w -----------------------------------
    key = "T1w" if contrast.startswith("T1") else "T2w" if contrast.startswith("T2") else ""
    if key:
        img = np.asarray(image, dtype=np.float32)
        # A few very bright voxels (vessels, fat, a Philips scan's 99,807
        # against a 99th percentile of 22,790) dominate a correlation: cut
        # them at the head's 99.5th percentile.
        inside = img[np.asarray(head, dtype=bool) & (img > 0)]
        if inside.size > 100:
            img = np.minimum(img, float(np.percentile(inside, 99.5)))
        lo12 = np.r_[lo, [-SHEAR_BOUND] * 3]
        hi12 = np.r_[hi, [SHEAR_BOUND] * 3]
        for mm in (4.0, 2.0):
            check()
            sub_img, sub_aff_i = _level(img, affine, mm)
            tpl_img, tpl_aff_i = _level(tpl[key], tpl.affine, mm)
            brain, _ = _level(tpl["brain"], tpl.affine, mm)
            # The brain only: outside it the two contrasts disagree (a
            # TSE's bright scalp against the template's), and a fit that
            # reaches for the scalp inflates the brain.
            region = ndi.binary_dilation(brain > 0.5, iterations=1)
            pts_i = _points(tpl_img, tpl_aff_i, region, POINTS, seed=11)
            t_raw = _sampler(tpl_img, tpl_aff_i)(pts_i)[0]
            s_i = _sampler(sub_img, sub_aff_i)

            def intensity(p12: np.ndarray) -> np.ndarray:
                check()
                m = matrix_of(p12, centre)
                moved = pts_i @ m[:3, :3].T + m[:3, 3]
                values, inside = s_i(moved)
                out = np.zeros(values.size)
                n_in = int(inside.sum())
                if n_in > 100:
                    out[inside] = ((_zscore(values[inside]) - _zscore(t_raw[inside]))
                                   * np.sqrt(values.size / n_in))
                return np.concatenate([out, _prior(p12)])

            if mm == 4.0:
                p = _search_translation(p, centre, pts_i, t_raw, s_i)
            p = np.clip(p, lo12 + 1e-6, hi12 - 1e-6)
            fit2 = least_squares(intensity, p, bounds=(lo12, hi12), method="trf",
                                 x_scale=np.r_[[5.0] * 3, [0.05] * 3, [0.03] * 3, [0.03] * 3],
                                 diff_step=1e-3, max_nfev=120)
            p = fit2.x
            m2 = matrix_of(p, centre)
            vals2, in2 = s_i(pts_i @ m2[:3, :3].T + m2[:3, 3])
            corr = (float(np.corrcoef(vals2[in2], t_raw[in2])[0, 1])
                    if in2.sum() > 100 and t_raw[in2].std() > 0 else 0.0)
        stages.append("intensity")
    return Registration(matrix=matrix_of(p, centre), correlation=corr, stages=stages,
                        seconds=time.perf_counter() - t0)


def to_image(volume: np.ndarray, vol_affine: np.ndarray, reg: Registration,
             shape, affine: np.ndarray, *, order: int = 1) -> np.ndarray:
    """A template-space ``volume`` on the image grid (``shape``, ``affine``)."""
    from scipy import ndimage as ndi

    affine = np.asarray(affine, dtype=float)
    # image voxel -> image mm -> template mm -> template voxel
    m = np.linalg.inv(vol_affine) @ np.linalg.inv(reg.matrix) @ affine
    return ndi.affine_transform(np.asarray(volume, dtype=np.float32), m[:3, :3],
                                offset=m[:3, 3], output_shape=tuple(int(s) for s in shape[:3]),
                                order=order, mode="constant", cval=0.0, prefilter=False)


def template_coords(reg: Registration, shape, affine: np.ndarray, axis: int) -> np.ndarray:
    """One template coordinate (``axis`` 0, 1, 2 = x, y, z mm) of every voxel
    of the image grid."""
    affine = np.asarray(affine, dtype=float)
    m = np.linalg.inv(reg.matrix) @ affine            # image voxel -> template mm
    row = m[axis]
    i, j, k = (np.arange(int(n), dtype=np.float32) for n in shape[:3])
    return (row[0] * i[:, None, None] + row[1] * j[None, :, None]
            + row[2] * k[None, None, :] + row[3]).astype(np.float32)


def outside_fov(reg: Registration, shape, affine: np.ndarray,
                tpl: Optional[Template] = None) -> dict:
    """How much of the template's brain falls outside the image's field of
    view, in total and per side ("top", "bottom", "front", "back", "left",
    "right" in the template's terms)."""
    tpl = tpl or template()
    brain = tpl["brain"] > 0.5
    idx = np.argwhere(brain)
    world_t = idx @ tpl.affine[:3, :3].T + tpl.affine[:3, 3]
    world_s = world_t @ reg.matrix[:3, :3].T + reg.matrix[:3, 3]
    vox = world_s @ np.linalg.inv(affine)[:3, :3].T + np.linalg.inv(affine)[:3, 3]
    shape = np.asarray(shape[:3], dtype=float)
    out = np.any((vox < -0.5) | (vox > shape - 0.5), axis=1)
    sides = {}
    if out.any():
        centre = world_t[~out].mean(axis=0) if (~out).any() else world_t.mean(axis=0)
        w = world_t[out] - centre
        names = (("right", "left"), ("front", "back"), ("top", "bottom"))
        main = np.argmax(np.abs(w), axis=1)
        for axis in range(3):
            pick = main == axis
            for sign, name in ((1, names[axis][0]), (-1, names[axis][1])):
                count = int(np.sum(pick & (np.sign(w[:, axis]) == sign)))
                if count:
                    sides[name] = count / len(idx)
    return {"fraction": float(out.mean()) if len(idx) else 0.0, "sides": sides}


__all__ = ["Cancelled", "GOOD_CORRELATION", "Registration", "matrix_of", "outside_fov",
           "register", "template_coords", "to_image"]

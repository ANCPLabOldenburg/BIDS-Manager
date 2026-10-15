"""Quality maps of a time series: mean, standard deviation, temporal SNR.

The three images a functional run is judged by before anything else: where
the signal is (mean), where it moves (standard deviation: motion at the
edges, vessels, ghosts), and how much of it is signal (temporal SNR, mean
over SD, the number acquisition protocols are compared by).

Accumulated frame by frame in float64 sums, never as a float copy of the
series: the 1224-volume BOLD the viewer is measured on is 679 MB as stored
and would be 2.7 GB in float64. Runs on a worker.

Each map comes with what it takes to READ it (:data:`HELP`, :func:`summary`):
a colour without a scale and a sentence on what it means is a picture, not
a measurement.

Qt-free.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np

from ...qc.config import DEFAULT

Map = Literal["mean", "sd", "tsnr"]

#: What each map is called on screen.
TITLES: dict[str, str] = {"mean": "Mean", "sd": "Standard deviation",
                          "tsnr": "Temporal SNR"}


class Cancelled(Exception):
    pass


def non_steady_state(src, *, z: float = DEFAULT.bold.nonsteady_z,
                     cancel: Optional[Callable[[], bool]] = None) -> int:
    """How many volumes at the START are non-steady-state (dummy scans not
    discarded by the scanner): they are much brighter than the rest, and
    left in they dominate every statistic. Counted from the mean of a
    central block of each of the first frames against the median of the
    run: brighter by more than ``z`` robust deviations. 0 when none."""
    n = src.loaded_frames
    if n < 8:
        return 0
    probe = min(n, 64)
    sx, sy, sz = (max(1, d // 4) for d in src.spatial)
    centre = tuple(slice(d // 2 - k, d // 2 + k) for d, k in zip(src.spatial, (sx, sy, sz)))
    means = []
    for t in range(probe):
        if cancel is not None and cancel():
            raise Cancelled()
        means.append(float(np.mean(src.scale(np.asarray(src.raw_frame(t))[centre]))))
    means = np.asarray(means)
    rest = means[min(10, probe // 2):]
    med = float(np.median(rest))
    mad = float(np.median(np.abs(rest - med))) * 1.4826 or 1e-9
    k = 0
    while k < min(10, probe) and means[k] > med + z * mad:
        k += 1
    return k


def series_moments(src, *, skip: int = 0, detrend: int = 2,
                   cancel: Optional[Callable[[], bool]] = None,
                   progress: Optional[Callable[[int, int], None]] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, sd)`` per voxel over the loaded frames after the first
    ``skip``, the SD taken around a polynomial trend of order ``detrend``
    (0, 1 or 2): slow scanner drift is not instability, and left in it
    lowers every temporal SNR (AFNI and MRIQC detrend too).

    One pass in float64 sums, never a float copy of the series. The trend
    uses discrete orthogonal polynomials (1, u, u^2 - mean(u^2)), so the
    residual sum of squares is the plain sum of squares less each
    projection: ``S2 - S0^2/m - A1^2/N1 - A2^2/N2``.
    """
    n = src.loaded_frames
    skip = max(0, min(int(skip), n - 3))
    m = n - skip
    if m < 2:
        raise ValueError("a quality map needs a series of at least two volumes")
    order = max(0, min(int(detrend), 2, m - 2))
    u = np.linspace(-1.0, 1.0, m) if m > 1 else np.zeros(1)
    q1 = u
    q2 = u * u - float(np.mean(u * u))
    n1 = float((q1 * q1).sum()) or 1.0
    n2 = float((q2 * q2).sum()) or 1.0
    s0 = s2 = a1 = a2 = None
    for k in range(m):
        if cancel is not None and cancel():
            raise Cancelled()
        frame = src.scale(src.raw_frame(skip + k)).astype(np.float64, copy=False)
        if s0 is None:
            s0 = np.zeros(frame.shape, dtype=np.float64)
            s2 = np.zeros(frame.shape, dtype=np.float64)
            a1 = np.zeros(frame.shape, dtype=np.float64) if order >= 1 else None
            a2 = np.zeros(frame.shape, dtype=np.float64) if order >= 2 else None
        s0 += frame
        s2 += frame * frame
        if a1 is not None:
            a1 += q1[k] * frame
        if a2 is not None:
            a2 += q2[k] * frame
        if progress is not None and (k % 64 == 0 or k == m - 1):
            progress(k + 1, m)
    mean = s0 / m
    rss = s2 - s0 * s0 / m
    if a1 is not None:
        rss -= a1 * a1 / n1
    if a2 is not None:
        rss -= a2 * a2 / n2
    dof = max(m - 1 - order, 1)
    return mean, np.sqrt(np.maximum(rss, 0.0) / dof)


def brain_mask(mean: np.ndarray) -> np.ndarray:
    """Voxels bright enough to be in the head: above a fifth of the robust
    top of the mean image. Outside it every map is noise over noise."""
    finite = mean[np.isfinite(mean)]
    if finite.size == 0:
        return np.zeros(mean.shape, dtype=bool)
    top = float(np.percentile(finite, 98))
    return mean > 0.2 * top


def quality_map(src, which: Map, *, skip: Optional[int] = None,
                detrend: int = DEFAULT.bold.detrend,
                cancel: Optional[Callable[[], bool]] = None,
                progress: Optional[Callable[[int, int], None]] = None
                ) -> tuple[np.ndarray, dict]:
    """``(map, facts)``: one map, (X, Y, Z) float32, NaN outside the head
    (drawn as nothing, not as the lowest colour), and how it was made:
    ``volumes`` used, ``skipped`` at the start, ``detrend`` order."""
    if skip is None:
        skip = non_steady_state(src, cancel=cancel)
    mean, sd = series_moments(src, skip=skip, detrend=detrend, cancel=cancel,
                              progress=progress)
    mask = brain_mask(mean)
    if which == "mean":
        out = mean
    elif which == "sd":
        out = sd
    else:
        out = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    facts = {"volumes": src.loaded_frames - skip, "skipped": skip, "detrend": detrend,
             "map": which}
    return np.where(mask, out, np.nan).astype(np.float32), facts


def summary(values: np.ndarray, which: Map = "tsnr", *,
            tsnr_low: float = DEFAULT.bold.tsnr_low) -> dict:
    """The numbers a map is read by, over the head: ``median``, ``q1``,
    ``q3``, ``p5``, ``p95``, ``voxels`` and, for temporal SNR, the share of
    the head ``below`` ``tsnr_low`` (a reading aid, not a pass mark: see
    HELP)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"voxels": 0}
    p5, q1, med, q3, p95 = (float(x) for x in np.percentile(v, (5, 25, 50, 75, 95)))
    out = {"voxels": int(v.size), "p5": p5, "q1": q1, "median": med, "q3": q3, "p95": p95}
    if which == "tsnr":
        out["below"] = float(np.mean(v < tsnr_low))
        out["low"] = float(tsnr_low)
    return out


#: How to read each map: shown beside it. Careful wording on purpose:
#: temporal SNR depends on field strength, voxel size, acceleration, coil and
#: TR, so the honest comparison is with runs of the same protocol.
HELP: dict[str, str] = {
    "tsnr": (
        "Temporal SNR: each voxel's mean over its standard deviation through "
        "the run, after slow drift is removed. Higher is steadier. Compare "
        "runs of the SAME protocol: it depends on field strength, voxel size, "
        "acceleration, the coil and the repetition time, so no single number "
        "is a pass mark (Krueger and Glover 2001; Triantafyllou 2005). "
        "Patterns to look for: a low ring at the edge of the brain (motion), "
        "low values in the ventricles and along large vessels (pulsation), "
        "low where the mean image is also dark, as in orbitofrontal cortex "
        "and the temporal poles (susceptibility dropout), and alternating "
        "slices (slice-wise motion or spikes)."),
    "sd": (
        "Standard deviation through the run, after slow drift is removed, in "
        "the scanner's units, inside the head. Bright means the signal moves: "
        "expected in vessels and ventricles, a warning as a rim at the brain's "
        "edge (motion) or as a band or a faint copy of the head across the brain "
        "along the phase-encode direction (ghosting). Ghosts outside the head are "
        "not shown: the map is drawn only where the mean image is bright."),
    "mean": (
        "Mean through the run. Where it is dark inside the head, signal is "
        "lost (susceptibility dropout near the sinuses and ear canals); the "
        "temporal SNR there will be low for that reason, not because of "
        "noise."),
}


def describe_summary(stats: dict, facts: dict, which: Map) -> str:
    """One line for the status bar and the controls column."""
    if not stats.get("voxels"):
        return "no voxels in the head"
    text = (f"median {stats['median']:.3g} (middle half {stats['q1']:.3g} to "
            f"{stats['q3']:.3g}) over {stats['voxels']:,} voxels")
    if which == "tsnr" and "below" in stats:
        text += f"; {stats['below'] * 100:.0f} % below {stats.get('low', 20.0):g}"
    if facts:
        text += f"; {facts.get('volumes', '?')} volumes"
        if facts.get("skipped"):
            text += f" ({facts['skipped']} non-steady-state skipped)"
        if facts.get("detrend"):
            text += f", order-{facts['detrend']} drift removed"
    return text


def per_volume(src, *, skip: int = 0, fence_iqr: float = DEFAULT.bold.dvars_fence_iqr,
               cancel: Optional[Callable[[], bool]] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """What each VOLUME looks like, over the head: ``global`` (the mean
    signal), ``dvars`` (the root mean square change from the previous
    volume, in percent of the head's mean: Power 2012), and ``flagged``,
    the volumes whose DVARS is above the upper box-plot fence (75th
    percentile plus ``fence_iqr`` interquartile ranges, 1.5 being FSL's
    default). Cheap: two
    passes over frames already in memory. Volume 0 has no DVARS (NaN)."""
    n = src.loaded_frames
    if n < 3:
        raise ValueError("a series of at least three volumes is needed")
    ref = src.scale(np.asarray(src.raw_frame(min(skip + 1, n - 1))))
    mask = brain_mask(np.asarray(ref, dtype=np.float64))
    if not mask.any():
        raise ValueError("no voxel is bright enough to be in the head")
    gs = np.empty(n)
    dvars = np.full(n, np.nan)
    prev = None
    for t in range(n):
        if cancel is not None and cancel():
            raise Cancelled()
        frame = src.scale(np.asarray(src.raw_frame(t)))[mask].astype(np.float64)
        gs[t] = frame.mean()
        if prev is not None:
            dvars[t] = float(np.sqrt(np.mean((frame - prev) ** 2)))
        prev = frame
        if progress is not None and (t % 64 == 0 or t == n - 1):
            progress(t + 1, n)
    level = float(np.median(gs[skip:])) or 1.0
    dvars_pct = dvars / level * 100.0
    valid = dvars_pct[skip + 1:]
    valid = valid[np.isfinite(valid)]
    if valid.size:
        q1, q3 = np.percentile(valid, (25, 75))
        fence = float(q3 + fence_iqr * (q3 - q1))
    else:
        fence = float("inf")
    flagged = np.flatnonzero(np.isfinite(dvars_pct) & (dvars_pct > fence)
                             & (np.arange(n) > skip))
    return {"global": gs, "dvars": dvars_pct, "fence": fence, "flagged": flagged,
            "skipped": skip}


def display_for(which: Map, values: np.ndarray):
    """The look a quality map opens with: its own colour map and a window
    from ZERO (the low end is what a quality map is for: ventricles,
    dropout, edge rings) to the head's 98th percentile; the mean map, a
    picture of the anatomy, from its 2nd percentile. Outside the head the
    map is NaN, which is drawn as nothing."""
    from ..scene import VolumeDisplay

    inside = values[np.isfinite(values)]
    if inside.size:
        p2, p98 = (float(v) for v in np.percentile(inside, (2, 98)))
    else:
        p2, p98 = 0.0, 1.0
    lo = p2 if which == "mean" else 0.0
    hi = p98 if p98 > lo else lo + 1.0
    colormap = {"mean": "gray", "sd": "hot", "tsnr": "viridis"}[which]
    return VolumeDisplay(colormap=colormap, window=(lo, hi), threshold_mode="range",
                         gamma=1.0, opacity=1.0)


#: What each map's values are, for its colour bar.
QUANTITY: dict[str, str] = {"mean": "signal (a.u.)", "sd": "SD (a.u.)",
                            "tsnr": "tSNR (mean / SD)"}


def quality_overlay(src, which: Map, *, config=None, cancel=None, progress=None):
    """The map as an :class:`~bidsmgr.viz.overlays.Overlay` over ``src``,
    with the BOLD settings of ``config`` (a ``QcConfig``)."""
    from ..data.volume import array_volume
    from ..overlays import Overlay

    bold = (config or DEFAULT).bold
    skip = non_steady_state(src, z=bold.nonsteady_z, cancel=cancel) \
        if bold.skip_nonsteady else 0
    values, facts = quality_map(src, which, skip=skip, detrend=bold.detrend, cancel=cancel,
                                progress=progress)
    vol = array_volume(values, src.affine, path=src.path, name=TITLES[which])
    stats = summary(values, which, tsnr_low=bold.tsnr_low)
    display = display_for(which, values)
    vol.quantity = QUANTITY[which]
    vol.notes = {"qc": which, "stats": stats, "facts": facts, "window": display.window,
                 "summary": describe_summary(stats, facts, which), "help": HELP[which]}
    note = f"{TITLES[which].lower()}: {describe_summary(stats, facts, which)}"
    return Overlay(source=vol, display=display, kind="image",
                   note=note, name=f"{TITLES[which]} of {src.path.name}")


# ---------------------------------------------------------------------------
# Per-volume rows beyond DVARS: outlier voxels, slice spikes, the carpet
# ---------------------------------------------------------------------------

#: The QC rows under the time course, in the order they are drawn, with what
#: each is for. ``motion`` is FD and the six parameters (``compute.motion``).
QC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("fd", "Framewise displacement",
     "How far the head moved from one volume to the next, in mm (Power 2012): "
     "the sum of the changes of the three translations and of the three "
     "rotations as arcs on a 50 mm sphere. From fMRIPrep's confounds when the "
     "run has them, else estimated here."),
    ("translation", "Translation (x, y, z)",
     "Where the head is, in mm, along the image's three axes, relative to the "
     "reference volume: left-right, back-front and down-up for an axial "
     "acquisition, other directions for a tilted or sagittal one."),
    ("rotation", "Rotation (pitch, roll, yaw)",
     "How the head is turned, in degrees, about the image's three axes: for an "
     "axial acquisition pitch is about the left-right axis (nodding), roll about "
     "the back-front axis (tilting to a shoulder) and yaw about the vertical axis "
     "(shaking the head)."),
    ("dvars", "DVARS",
     "How much the whole image changes from one volume to the next, with the "
     "volumes above the box-plot fence marked."),
    ("outliers", "Outlier voxels",
     "The share of the head's voxels that are outliers in each volume "
     "(AFNI's 3dToutcount rule, on a sample of voxels)."),
    ("spikes", "Slice spikes",
     "The largest robust z of any slice against its own course, per volume: "
     "a spike, or a slice dropped by motion during the volume, stands out."),
    ("global", "Global signal", "The mean over the head, per volume."),
    ("carpet", "Carpet plot",
     "Every sampled voxel's signal over time, one row each, from the edge of "
     "the head inwards (Power 2017): motion and spikes show as vertical "
     "bands across many rows."),
)
#: The QC rows of a DIFFUSION series (``bidsmgr.qc.dwi``), offered by the
#: same Plots menu when the series is one. Translation and rotation are the
#: same rows as for BOLD; the rest are diffusion's own.
DWI_QC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("displacement", "Displacement",
     "How far each volume sits from the first b=0 (translations plus rotations on a "
     "50 mm sphere): the b=0 volumes are the head's reliable track, the diffusion-"
     "weighted ones scatter by the registration's noise. Marked: volumes far out of "
     "place."),
    ("translation", "Translation (x, y, z)",
     "Where each volume sits against the first b=0, mm, eddy-current shifts taken out "
     "when the gradients explain them."),
    ("rotation", "Rotation (pitch, roll, yaw)",
     "How each volume is turned against the first b=0, in degrees."),
    ("slices", "Slice signal",
     "Every slice of every volume against what the diffusion tensor predicts for it, "
     "as a robust z within its shell: blue is signal lost (a dropout), red signal "
     "added. Click a cell to go to that volume and slice."),
    ("b0_signal", "b=0 signal",
     "The brain's median signal in each b=0 volume against the first: a slope is "
     "signal drift."),
    ("spikes", "Spiking voxels",
     "Voxels far above the tensor's prediction, per million brain voxels."),
    ("interleave", "Odd against even slices",
     "Odd slices against even slices, robust z within the shell: motion between the "
     "two passes of an interleaved acquisition."),
    ("ndc", "Nearest-direction correlation",
     "Each diffusion volume's correlation with the volume nearest it in q-space: a "
     "volume unlike its neighbours is suspect."),
    ("eddy", "Eddy-current shift",
     "The shift of each diffusion-weighted volume its gradient direction explains."),
)
#: Shown for a diffusion series when none of its own rows were chosen.
DWI_DEFAULT_ROWS = ("displacement", "slices", "b0_signal", "spikes")
QC_ROW_IDS = tuple(dict.fromkeys([r[0] for r in QC_ROWS] + [r[0] for r in DWI_QC_ROWS]))
BOLD_ROW_IDS = tuple(r[0] for r in QC_ROWS)
DWI_ROW_IDS = tuple(r[0] for r in DWI_QC_ROWS)
# The share of outlier voxels worth a look (afni_proc.py's 5 %), the slice
# spike threshold and the voxels sampled are settings:
# ``bidsmgr.qc.config.QcBold``.
#: Rows the carpet is drawn with (voxels are averaged in groups).
CARPET_ROWS = 240


def sample_series(src, *, skip: int = 0, voxels: int = DEFAULT.bold.sample_voxels,
                  slice_axis: int = 2,
                  seed: int = 0, cancel: Optional[Callable[[], bool]] = None,
                  progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """One pass over the frames: ``series`` (frames x voxels, a fixed random
    sample of the head), ``depth`` (each sampled voxel's distance from the
    head's edge, voxels) and ``slice_means`` (frames x slices along
    ``slice_axis``, over the whole head)."""
    from scipy import ndimage

    n = src.loaded_frames
    if n < 3:
        raise ValueError("a series of at least three volumes is needed")
    ref = src.scale(np.asarray(src.raw_frame(min(skip + 1, n - 1))))
    mask = brain_mask(np.asarray(ref, dtype=np.float64))
    if not mask.any():
        raise ValueError("no voxel is bright enough to be in the head")
    # Fortran order: a frame is an (X, Y, Z) view of a (Z, Y, X) block, so
    # its F-order ravel is free where a C-order one copies the volume.
    inside = np.flatnonzero(mask.ravel(order="F"))
    rng = np.random.default_rng(seed)
    pick = (np.sort(rng.choice(inside, voxels, replace=False))
            if inside.size > voxels else inside)
    slice_of = np.unravel_index(inside, mask.shape, order="F")[slice_axis]
    n_slices = mask.shape[slice_axis]
    counts = np.bincount(slice_of, minlength=n_slices).astype(float)
    series = np.empty((n, pick.size), dtype=np.float32)
    slice_means = np.full((n, n_slices), np.nan)
    for t in range(n):
        if cancel is not None and cancel():
            raise Cancelled()
        flat = np.asarray(src.raw_frame(t)).ravel(order="F")
        series[t] = src.scale(flat[pick])
        sums = np.bincount(slice_of, weights=src.scale(flat[inside]).astype(np.float64),
                           minlength=n_slices)
        slice_means[t] = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
        if progress is not None and (t % 64 == 0 or t == n - 1):
            progress(t + 1, n)
    depth = ndimage.distance_transform_edt(mask).ravel(order="F")[pick]
    return {"series": series, "depth": depth, "slice_means": slice_means}


def detrend(values: np.ndarray, order: int = 2) -> np.ndarray:
    """Each column less its polynomial trend of ``order`` (slow drift is not
    an outlier, a spike or a band in a carpet)."""
    y = np.asarray(values, dtype=np.float64)
    m = y.shape[0]
    order = max(0, min(int(order), m - 2))
    u = np.linspace(-1.0, 1.0, m)
    basis = np.stack([u ** k for k in range(order + 1)], axis=1)
    q, _r = np.linalg.qr(basis)
    return y - q @ (q.T @ y)


def outlier_fraction(series: np.ndarray, *, skip: int = 0) -> np.ndarray:
    """The share of the sampled voxels that are outliers in each volume
    (NaN for the skipped ones). AFNI's 3dToutcount rule: after the drift is
    removed, a value is an outlier when it is further from its voxel's median
    than ``qginv(0.001 / N) * sqrt(pi / 2) * MAD``, N the number of volumes."""
    from scipy.special import ndtri

    n = series.shape[0]
    out = np.full(n, np.nan)
    m = n - skip
    if m < 3:
        return out
    r = detrend(series[skip:])
    med = np.median(r, axis=0)
    dev = np.abs(r - med)
    mad = np.median(dev, axis=0)
    alpha = float(-ndtri(0.001 / m))
    limit = alpha * np.sqrt(np.pi / 2.0) * mad
    hits = (dev > limit) & (mad > 0)
    out[skip:] = hits.mean(axis=1)
    return out


def slice_spikes(slice_means: np.ndarray, *, skip: int = 0,
                 z: float = DEFAULT.bold.spike_z) -> dict:
    """Volumes where a slice departs from its own course, beyond what the
    volume as a whole did (a global change, additive or a scaling, is not a
    spike): ``score`` (the
    largest robust z of any slice, per volume), ``flagged`` (volumes above
    ``z``) and ``slice`` (the slice that did it, per flagged volume)."""
    n = slice_means.shape[0]
    score = np.full(n, np.nan)
    sm = slice_means[skip:]
    ok = np.all(np.isfinite(sm), axis=0)
    if sm.shape[0] < 3 or not ok.any():
        return {"score": score, "flagged": np.empty(0, int), "slice": np.empty(0, int)}
    sm = sm[:, ok]
    # What the volume as a whole did (an offset and a scaling of the run's
    # typical slice profile) is taken out, so only a slice that moved on its
    # own is left.
    profile = np.median(sm, axis=0)
    design = np.stack([np.ones_like(profile), profile], axis=1)
    resid = sm - (design @ (np.linalg.pinv(design) @ sm.T)).T
    r = detrend(resid)
    med = np.median(r, axis=0)
    mad = np.median(np.abs(r - med), axis=0) * 1.4826
    mad[mad == 0] = np.inf
    zs = np.abs(r - med) / mad
    score[skip:] = zs.max(axis=1)
    flagged = np.flatnonzero(np.nan_to_num(score) > z)
    slices = np.flatnonzero(ok)[zs.argmax(axis=1)][flagged - skip] if flagged.size else \
        np.empty(0, int)
    return {"score": score, "flagged": flagged, "slice": slices}


def carpet(series: np.ndarray, depth: np.ndarray, *, skip: int = 0,
           rows: int = CARPET_ROWS) -> np.ndarray:
    """(rows x volumes) z-scores clipped to +-3: the sampled voxels from the
    head's edge (top) inwards, averaged in groups to ``rows``; NaN for the
    skipped volumes."""
    n, k = series.shape
    out = np.full((min(rows, max(k, 1)), n), np.nan, dtype=np.float32)
    if n - skip < 3 or k == 0:
        return out
    r = detrend(series[skip:])
    sd = r.std(axis=0)
    sd[sd == 0] = 1.0
    zs = np.clip(r / sd, -3.0, 3.0)
    order = np.argsort(depth, kind="stable")
    groups = np.array_split(order, out.shape[0])
    for i, g in enumerate(groups):
        out[i, skip:] = zs[:, g].mean(axis=1)
    return out


__all__ = ["BOLD_ROW_IDS", "CARPET_ROWS", "Cancelled", "DWI_DEFAULT_ROWS", "DWI_QC_ROWS",
           "DWI_ROW_IDS", "HELP", "Map", "QC_ROWS",
           "QC_ROW_IDS", "QUANTITY", "TITLES",
           "brain_mask", "carpet", "describe_summary", "detrend", "display_for",
           "non_steady_state", "outlier_fraction", "per_volume", "quality_map",
           "quality_overlay", "sample_series", "series_moments", "slice_spikes", "summary"]

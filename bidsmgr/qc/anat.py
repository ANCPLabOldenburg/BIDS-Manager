"""The quality check of an anatomical image (T1w, T2w; FLAIR, PDw, T2starw
without the tissue measures).

What it does, in order, on a working grid of about 2 mm (the measures that
depend on noise are taken back at the image's own resolution):

1. the zero fill and the head (``masks``);
2. a registration to the template (``register``), which gives the brain,
   the tissue priors, the face and the field of view;
3. whether the image is defaced: the sidecar's record, or a blank face;
4. three tissue classes and the bias field (``segment``, ``bias``);
5. the measures (:data:`METRICS`), each named for what it measures;
6. what BIDS Manager can say that a generic check cannot: the field of
   view cutting the brain, wrap-around and ghosting along the phase-encoding
   direction the sidecar names, a face still present, saturation, the header.

**The air is measured only in images that are not defaced** (nor skull
stripped, nor with a background the scanner set to zero): defacing replaces
part of the air with zeros, and every measure of the air's noise would then
describe the defacing, not the scan. Those measures are reported as not
computed, with the reason.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import masks as K
from . import register as R
from . import stats
from .config import DEFAULT, QcConfig
from .types import Finding, Metric, QCMap, QCResult

#: Suffixes the check runs on, and those that get the tissue measures.
SUFFIXES = ("T1w", "T2w", "FLAIR", "PDw", "T2starw")
#: The numpy fallback segments these (it needs a template of the contrast).
TISSUE_SUFFIXES = ("T1w", "T2w")
#: The tissue model segments these (checked by eye on T1w, T2w and FLAIR).
TOOL_TISSUE_SUFFIXES = ("T1w", "T2w", "FLAIR")
#: Native voxels sampled for the tissue statistics.
SAMPLE = 1_000_000
#: The shell outside the brain whose signal tells a skull-stripped image
#: (``QcAnat.stripped_shell_pct``), in mm. The volumes are no test: a
#: defaced head cropped to the skull measured 1.2 x its brain.
SHELL_MM = (2.0, 8.0)
#: Dietrich's correction for the SD of magnitude noise in the air.
RAYLEIGH = 0.6551364

#: key -> (title, unit, group, better, MRIQC equivalent, help, format)
METRICS: dict[str, tuple] = {
    "snr_csf": ("SNR in CSF", "", "noise", "higher", "snr_csf",
                "Median of the CSF over its standard deviation.", "{:.2f}"),
    "snr_gm": ("SNR in grey matter", "", "noise", "higher", "snr_gm",
               "Median of the grey matter over its standard deviation.", "{:.2f}"),
    "snr_wm": ("SNR in white matter", "", "noise", "higher", "snr_wm",
               "Median of the white matter over its standard deviation: the "
               "most stable of the three.", "{:.2f}"),
    "snr_total": ("SNR (mean of the tissues)", "", "noise", "higher", "snr_total",
                  "The mean of the three tissue SNRs.", "{:.2f}"),
    "snrd_wm": ("SNR against the air (white matter)", "", "noise", "higher", "snrd_wm",
                "Dietrich's SNR: 0.655 x the white matter's median over the SD of "
                "the air's noise. Needs real air: not computed on a defaced image.",
                "{:.1f}"),
    "snrd_total": ("SNR against the air (mean)", "", "noise", "higher", "snrd_total",
                   "Dietrich's SNR averaged over CSF, grey and white matter.", "{:.1f}"),
    "cnr": ("Contrast-to-noise ratio", "", "noise", "higher", "cnr",
            "How far grey and white matter are apart against the noise of the air "
            "and of both tissues. Needs real air.", "{:.2f}"),
    "cjv": ("Coefficient of joint variation", "", "noise", "lower", "cjv",
            "Spread of grey and white matter over the distance between them "
            "(Ganzetti 2016): higher with noise, bias and motion.", "{:.3f}"),
    "efc": ("Entropy focus criterion", "", "artefacts", "lower", "efc",
            "How spread the intensities are (Atkinson 1997): ghosting and motion "
            "blurring raise it. Over the whole image, so it needs real air.", "{:.3f}"),
    "fber": ("Foreground to background energy", "", "artefacts", "higher", "fber",
             "Median energy of the head over that of the air (Shehzad 2015).",
             "{:.0f}"),
    "qi1": ("Artefacts in the air (QI1)", "%", "artefacts", "lower", "",
            "Share of the air above the face holding structured signal: ghosts, "
            "ringing, motion, wrap-around (Mortamet 2009). MRIQC's QI1 is always 0 "
            "through a bug; this is the measure as defined.", "{:.3f}"),
    "qi2": ("Noise fit (QI2)", "", "artefacts", "lower", "qi_2",
            "How badly a chi-squared distribution fits the tail of the air's noise "
            "(Mortamet 2009): structured signal in the air raises it.", "{:.4f}"),
    "ghost_ratio": ("Ghost to signal ratio", "", "artefacts", "lower", "",
                    "Mean signal where a ghost of the head falls along the phase-"
                    "encoding direction, above the rest of the air, over the head's "
                    "mean.", "{:.3f}"),
    "wm2max": ("White matter to maximum", "", "tissues", "", "wm2max",
               "The white matter's median over the 99.9th percentile of the non-zero "
               "image (the ceiling MRIQC clips to): how much of the intensity range a "
               "long tail of bright voxels (vessels, fat) takes up. MRIQC reads 0.6 to "
               "0.8 as usual for a T1w.", "{:.2f}"),
    "inu_range": ("Intensity non-uniformity", "", "tissues", "lower", "",
                  "Spread of the bias field over the brain, 95th minus 5th "
                  "percentile of the field normalised to 1: 0 is uniform.", "{:.3f}"),
    "fraction_csf": ("CSF fraction", "", "tissues", "", "icvs_csf",
                     "Share of the brain's volume classified as CSF.", "{:.3f}"),
    "fraction_gm": ("Grey matter fraction", "", "tissues", "", "icvs_gm",
                    "Share of the brain's volume classified as grey matter.", "{:.3f}"),
    "fraction_wm": ("White matter fraction", "", "tissues", "", "icvs_wm",
                    "Share of the brain's volume classified as white matter.", "{:.3f}"),
    "partial_volume": ("Partial volume", "", "tissues", "lower", "",
                       "Share of the brain where no tissue class reaches 0.9: blur, "
                       "motion and thick voxels raise it. Not MRIQC's rpve, which is "
                       "not a fraction.", "{:.3f}"),
    "template_overlap_gm": ("Overlap with the template (grey matter)", "", "tissues",
                            "higher", "tpm_overlap_gm",
                            "Fuzzy Jaccard index of the grey matter with the template's "
                            "map after an affine registration.", "{:.3f}"),
    "template_overlap_wm": ("Overlap with the template (white matter)", "", "tissues",
                            "higher", "tpm_overlap_wm",
                            "Fuzzy Jaccard index of the white matter with the template's "
                            "map.", "{:.3f}"),
    "fwhm_x": ("Smoothness along x", "mm", "coverage", "", "",
               "Full width at half maximum of the image's own correlation between "
               "neighbours (Forman 1995), inside the brain.", "{:.2f}"),
    "fwhm_y": ("Smoothness along y", "mm", "coverage", "", "", "As along x.", "{:.2f}"),
    "fwhm_z": ("Smoothness along z", "mm", "coverage", "", "", "As along x.", "{:.2f}"),
    "fwhm_avg": ("Smoothness (mean)", "mm", "coverage", "", "",
                 "The mean of the three. MRIQC reports it in voxels.", "{:.2f}"),
    "fov_cut": ("Brain outside the field of view", "%", "coverage", "lower", "",
                "Share of the template's brain that falls outside the image after "
                "registration: a cut top of the head or cerebellum.", "{:.2f}"),
    "saturation": ("Saturated voxels", "%", "header", "lower", "",
                   "Share of the head at the image's maximum value.", "{:.3f}"),
}


def _why_no_tissue(suffix: str, reg, thinnest: float) -> str:
    if suffix not in TOOL_TISSUE_SUFFIXES:
        return f"Tissue measures are made for T1w, T2w and FLAIR images; this is {suffix}."
    if suffix not in TISSUE_SUFFIXES:
        return ("Without the tissue model (see the note on the methods), only T1w and T2w "
                "are segmented.")
    if reg is None:
        return f"A slab of {thinnest:.0f} mm: no template priors to segment with."
    return "The brain could not be segmented."


def _metric(key: str, value, why: str = "") -> Metric:
    title, unit, group, better, mriqc, help_text, fmt = METRICS[key]
    v = None if value is None or not np.isfinite(value) else float(value)
    return Metric(key=key, title=title, value=v, unit=unit, group=group, help=help_text,
                  better=better, mriqc=mriqc, fmt=fmt, why_missing=why if v is None else "")


def defacing_record(sidecar: dict) -> str:
    """How the sidecar says the face was removed, to follow "defaced"
    ("by BIDS Manager (allineate)", "with pydeface"); "" when it says
    nothing."""
    from ..deface import status

    try:
        ours = status.defaced_by_us(sidecar)
    except Exception:  # noqa: BLE001 - a malformed record is no record
        ours = None
    if ours is not None:
        return f"by BIDS Manager ({ours.label})"
    try:
        if status.defaced_by_us_at_all(sidecar):
            return "by BIDS Manager"
        others = status.defaced_by_others(sidecar)
    except Exception:  # noqa: BLE001
        others = []
    return ("with " + "; ".join(others)) if others else ""


def _pe_axis(sidecar: dict) -> Optional[int]:
    """The phase-encoding axis of the image (0, 1, 2), when the sidecar says."""
    pe = str(sidecar.get("PhaseEncodingDirection") or "")
    if pe[:1] in ("i", "j", "k"):
        return "ijk".index(pe[0])
    inplane = str(sidecar.get("InPlanePhaseEncodingDirectionDICOM") or "").upper()
    if inplane == "ROW":
        return 0
    if inplane == "COL":
        return 1
    return None


def _fwhm(data: np.ndarray, mask: np.ndarray, zooms) -> list[float]:
    """Forman 1995 per axis, mm: the lag-1 correlation of neighbours both
    inside ``mask``, turned into the FWHM of a Gaussian of that width."""
    out = []
    values = data[mask]
    var = float(values.var())
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis] = slice(1, None)
        b[axis] = slice(None, -1)
        both = mask[tuple(a)] & mask[tuple(b)]
        if var <= 0 or not both.any():
            out.append(float("nan"))
            continue
        diff = (data[tuple(a)] - data[tuple(b)])[both]
        rho = 1.0 - float(diff.var()) / (2.0 * var)
        if not 0.0 < rho < 1.0:
            out.append(float("nan"))
            continue
        out.append(float(zooms[axis]) * float(np.sqrt(-2.0 * np.log(2.0) / np.log(rho))))
    return out


def _qi2(values: np.ndarray) -> Optional[float]:
    """MRIQC's QI2 (Mortamet 2009): a chi fit to the air's intensities and
    the mean gap to their density in the tail beyond half its peak."""
    from scipy import stats as st

    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size < 1000:
        return None
    data = data * (100.0 / float(np.percentile(data, 99)))
    rng = np.random.default_rng(1191935)
    if data.size > 100_000:
        data = rng.choice(data, size=100_000, replace=False)
    grid = np.linspace(0.0, 110.0, 1000)
    kde = stats.gaussian_kde_grid(data, grid, 4.0)
    cut = int(np.argmax(kde[::-1] > kde.max() * 0.5))
    if cut <= 0:
        return None
    try:
        df, loc, scale = st.chi2.fit(data, 32)
        pdf = st.chi2.pdf(grid, df, loc=loc, scale=scale)
    except Exception:  # noqa: BLE001 - a fit that fails is a missing value
        return None
    return float(np.abs(kde[-cut:] - pdf[-cut:]).mean())


def _efc(values: np.ndarray) -> Optional[float]:
    x = np.clip(np.asarray(values, dtype=np.float64), 0, None)
    n = x.size
    if n < 2:
        return None
    b_max = float(np.sqrt(np.sum(x * x)))
    if b_max <= 0:
        return None
    efc_max = n * (1.0 / np.sqrt(n)) * np.log(1.0 / np.sqrt(n))
    e = float(np.sum((x / b_max) * np.log((x + 1e-16) / b_max)))
    return e / efc_max


def shell_signal_share(image: np.ndarray, brain: np.ndarray, zooms) -> float:
    """The share of the shell just outside ``brain`` (``SHELL_MM``) holding
    signal (above a tenth of the brain's median): the scalp, the marrow, the
    meninges. Near zero when the image holds the brain only. NaN without a
    brain."""
    from scipy import ndimage as ndi

    if not brain.any():
        return float("nan")
    step = float(np.min(zooms))
    inner = max(1, int(round(SHELL_MM[0] / step)))
    outer = max(inner + 1, int(round(SHELL_MM[1] / step)))
    shell = ndi.binary_dilation(brain, iterations=outer) & ~ndi.binary_dilation(
        brain, iterations=inner)
    if not shell.any():
        return float("nan")
    level = 0.1 * float(np.median(image[brain]))
    return float(np.count_nonzero(image[shell] > level)) / float(np.count_nonzero(shell))


def check(data: np.ndarray, affine: np.ndarray, *, suffix: str, sidecar: Optional[dict] = None,
          path: str = "", header=None, cancel: Optional[Callable[[], bool]] = None,
          progress: Optional[Callable[[int, int], None]] = None,
          config: Optional[QcConfig] = None) -> QCResult:
    """The quality check of one anatomical image (``data``, 3-D).

    ``config`` (default: :data:`config.DEFAULT`) chooses the methods and
    every threshold. A method that needs a tool (``qc.tools``: mindgrab,
    robust_tissue, niimath) falls back to the fast one when the tool
    cannot run on ``path``, and the result says why."""
    from scipy import ndimage as ndi

    from .templates import FACE_Y_MM, FACE_Z_MM, GLABELLA_Z_MM, template

    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    sidecar = sidecar or {}
    cfg = config or DEFAULT
    A, how = cfg.anat, cfg.methods

    def step(name: str, n: int) -> None:
        if cancel is not None and cancel():
            raise R.Cancelled()
        timings[name] = round(time.perf_counter() - t_start, 3)
        if progress is not None:
            progress(n, 8)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.where(np.isfinite(data), data, 0.0).astype(np.float32)
    affine = np.asarray(affine, dtype=float)
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    result = QCResult(path=str(path), kind="anat", suffix=suffix)
    facts = result.facts
    facts.update(shape=list(data.shape), zooms=[round(float(z), 4) for z in zooms])
    # What it was made with, so a saved result can be reproduced.
    facts["config"] = {"methods": how.model_dump(mode="json"),
                       "anat": A.model_dump(mode="json")}

    # 1. Zero fill and head, on the working grid.
    factor = stats.block_factor(zooms, A.work_mm)
    work = stats.block_mean(data, factor)
    waff = stats.block_affine(affine, factor)
    wzoom = zooms * factor
    zero_w = K.zero_fill(work)
    head_w = K.head(work, exclude=zero_w)
    step("head", 1)

    # 2. The tools (mindgrab, robust_tissue, niimath), when they can run.
    from . import tools as T

    extent = np.asarray(data.shape[:3], dtype=float) * zooms
    thinnest = float(extent.min())
    slab = thinnest < A.slab_mm
    facts.update(extent_mm=[round(float(e), 1) for e in extent], slab=bool(slab))
    engines = {"brain": "template (numpy)", "tissues": "", "registration": ""}
    # Which methods want a tool, and whether the tools can run here; a
    # method whose tool cannot falls back to the fast one, with a note.
    want_brain = how.brain == "mindgrab"
    want_tissue = how.tissues == "robust_tissue"
    want_reg = how.registration == "niimath"
    wanted = want_brain or want_tissue or want_reg
    tool_note = ""
    got: dict = {}
    can_tools = wanted and bool(path) and Path(path).is_file()
    if can_tools:
        tool_note = T.unavailable_reason() or ""
        can_tools = not tool_note
    elif wanted:
        tool_note = "the image is not a file the tools can read"
    tissue_suffix = suffix in (TOOL_TISSUE_SUFFIXES if can_tools and want_tissue
                               else TISSUE_SUFFIXES)
    models = ([T.BRAIN_MODEL] if want_brain else []) + (
        [T.TISSUE_MODEL] if want_tissue and tissue_suffix else [])
    if can_tools and models:
        try:
            got = T.run_models(Path(path), data.shape, affine, models, cancel=cancel,
                               timeout=how.tool_timeout_s)
        except T.ToolFailed as exc:
            tool_note = str(exc)
    step("models", 2)

    # 3. Registration, unless the field of view is too thin a slab for a
    # whole-head template to be placed in it.
    tpl = template()
    if thinnest < A.no_registration_mm:
        reg = None
        facts["registration"] = {"skipped": True, "uncertain": True}
        engines["registration"] = "skipped (thin slab)"
        brain_p = ndi.binary_erosion(head_w, iterations=2).astype(np.float32)
        head_p = head_w.astype(np.float32)
        tz = np.full(work.shape, np.inf, dtype=np.float32)
        ty = np.full(work.shape, -np.inf, dtype=np.float32)
    else:
        reg = None
        if can_tools and want_reg:
            key = "T2w" if suffix in ("T2w", "FLAIR", "PDw", "T2starw") else "T1w"
            try:
                if T.BRAIN_MODEL in got:
                    # Brain to brain: what is outside it (a face defacing
                    # removed, a neck the field of view cut) cannot pull the
                    # fit. Measured: Dice with mindgrab's brain 0.76 -> 0.94 on
                    # a defaced T1w, and never worse on five other images.
                    matrix = T.allineate_brain(data, affine, got[T.BRAIN_MODEL] > 0, key,
                                               cancel=cancel,
                                               timeout=min(how.tool_timeout_s, 300))
                    stage = "niimath -allineate (brain to brain)"
                else:
                    matrix = T.allineate(Path(path), T.template_file(key), cancel=cancel,
                                         timeout=min(how.tool_timeout_s, 300))
                    stage = "niimath -allineate"
                reg = R.Registration(matrix=matrix, correlation=float("nan"), stages=[stage])
                engines["registration"] = stage
            except T.ToolFailed as exc:
                tool_note = tool_note or str(exc)
        if reg is None:
            reg = R.register(work, waff, head_w,
                             contrast=suffix if suffix in TISSUE_SUFFIXES else "",
                             cancel=cancel)
            engines["registration"] = "numpy"
        facts["registration"] = {"stages": reg.stages, "uncertain": reg.uncertain}
        if np.isfinite(reg.correlation):
            facts["registration"]["correlation"] = round(reg.correlation, 3)
        brain_p = R.to_image(tpl["brain"], tpl.affine, reg, work.shape, waff)
        head_p = R.to_image(tpl["head"], tpl.affine, reg, work.shape, waff)
        tz = R.template_coords(reg, work.shape, waff, 2)
        ty = R.template_coords(reg, work.shape, waff, 1)
    step("registration", 3)

    # 3. Defaced, skull stripped, blanked background?
    record = defacing_record(sidecar)
    face = (head_p > 0.5) & (brain_p < 0.1) & (ty > FACE_Y_MM) & (tz < FACE_Z_MM)
    blank = K.face_blank_share(work, face)
    shape = data.shape
    if T.BRAIN_MODEL in got:
        # mindgrab's brain, on the image's own grid.
        brain_n = stats.largest_components(got[T.BRAIN_MODEL] > 0, 1)
        brain_w = stats.block_mean(brain_n.astype(np.float32), factor) > 0.5
        engines["brain"] = "mindgrab"
    else:
        # The template's brain inside the head, in one piece (the edge of the
        # head mask can leave an island at a sinus).
        brain_w = stats.largest_components((brain_p > 0.5) & head_w, 1)
        brain_n = stats.block_expand(brain_w, factor, shape)
    # A registration that put the brain in the air is wrong whatever its
    # correlation says; with mindgrab's brain, the template's must agree
    # with it.
    brain_in_head = (float(np.count_nonzero(head_w[brain_p > 0.5]))
                     / max(int(np.count_nonzero(brain_p > 0.5)), 1))
    uncertain = reg is not None and (reg.uncertain or brain_in_head < 0.9)
    if reg is not None and engines["brain"] == "mindgrab":
        tb = brain_p > 0.5
        dice = 2.0 * float((tb & brain_w).sum()) / max(float(tb.sum() + brain_w.sum()), 1.0)
        facts["registration"]["dice_with_brain"] = round(dice, 3)
        uncertain = dice < A.min_registration_dice
    facts["registration"]["brain_in_head"] = round(brain_in_head, 3)
    facts["registration"]["uncertain"] = uncertain
    head_ml = float(head_w.sum() * np.prod(wzoom)) / 1000.0
    brain_ml = float(brain_w.sum() * np.prod(wzoom)) / 1000.0
    shell = shell_signal_share(work, brain_w, wzoom)
    stripped = brain_ml > 0 and shell < A.stripped_shell_pct / 100.0
    defaced = bool(record) or blank > A.defaced_face_pct / 100.0
    facts.update(defaced=defaced, defacing_record=record, face_blank_share=round(blank, 3),
                 skull_stripped=bool(stripped), shell_signal_share=round(shell, 3),
                 head_ml=round(head_ml, 1),
                 brain_ml=round(brain_ml, 1))

    above = tz > GLABELLA_Z_MM
    air_w = K.air(head_w, excluded=zero_w, above=above)
    candidate_air = air_w.sum()
    # The whole air above the face, blanked part included: a scanner that
    # sets the background to zero leaves a thin band of noise around the
    # head, too little and too close to the scalp to stand for the air.
    all_air = K.air(head_w, excluded=np.zeros_like(zero_w), above=above)
    zero_in_air = (float(np.count_nonzero(work[all_air] <= 0)) / int(all_air.sum())
                   if all_air.any() else 1.0)
    facts["air_zero_share"] = round(zero_in_air, 3)
    if defaced:
        air_reason = ("The image is defaced " + (record or "(its face region is blank)")
                      + ": its air is not measured, because defacing replaced part of "
                      "it with zeros.")
    elif stripped:
        air_reason = "The image is skull stripped: there is no air to measure."
    elif candidate_air < 1000:
        air_reason = "The field of view leaves too little air around the head to measure."
    elif zero_in_air > A.zero_air_pct / 100.0:
        air_reason = ("The air is set to zero (by the scanner or a tool): there is no "
                      "noise left in it to measure.")
    else:
        air_reason = ""
    air_ok = not air_reason
    facts["air_measured"] = air_ok
    if air_reason:
        facts["air_reason"] = air_reason
    step("defacing", 3)

    # 4. Tissue classes and the bias field.
    from . import segment as S

    seg = None
    labels_n = None
    tpms = None
    if reg is not None:
        tpms = np.stack([R.to_image(tpl[k], tpl.affine, reg, work.shape, waff)
                         for k in ("csf", "gm", "wm")])
    if T.TISSUE_MODEL in got and tissue_suffix:
        # robust_tissue's grey and white matter; CSF is the rest of the brain.
        raw = got[T.TISSUE_MODEL]
        names = T.TISSUE_LABELS[T.TISSUE_MODEL]
        labels_n = np.zeros(shape, dtype=np.uint8)
        labels_n[brain_n] = 1
        for value, name in names.items():
            labels_n[(raw == value) & brain_n] = 2 if name == "gm" else 3
        fractions = np.stack([stats.block_mean((labels_n == k).astype(np.float32), factor)
                              for k in (1, 2, 3)])
        seg = S.from_classes(work, fractions, brain_w)
        engines["tissues"] = T.TISSUE_MODEL
    elif suffix in TISSUE_SUFFIXES and reg is not None:
        seg_mask = ndi.binary_erosion(brain_w, iterations=1) if brain_w.sum() > 5000 else brain_w
        seg = S.segment(work, seg_mask, tpms)
        if seg is not None:
            engines["tissues"] = "EM with template priors (numpy)"
    facts["engine"] = engines
    if tool_note:
        facts["tool_note"] = tool_note
    step("segmentation", 4)

    # 5. Measures at the image's resolution.
    rng = np.random.default_rng(20261007)
    zero_n = (data <= 0) & stats.block_expand(ndi.binary_dilation(zero_w, iterations=1),
                                              factor, shape)
    head_n = stats.block_expand(head_w, factor, shape)
    metrics: list[Metric] = []
    tissue_stats = {}
    if seg is not None:
        brain_n_idx = np.flatnonzero(brain_n if labels_n is not None else stats.block_expand(
            seg.posteriors.sum(axis=0) > 0, factor, shape))
        if brain_n_idx.size > SAMPLE:
            brain_n_idx = np.sort(rng.choice(brain_n_idx, SAMPLE, replace=False))
        ijk = np.unravel_index(brain_n_idx, shape)
        wijk = tuple(np.minimum(c // f, n - 1) for c, f, n in zip(ijk, factor, work.shape))
        raw_vals = data.ravel()[brain_n_idx]
        corrected = raw_vals / np.exp(seg.field_log[wijk])
        # The classes again at the image's own resolution: the working
        # grid's posteriors spread over its 1 mm voxels mix the tissues at
        # every boundary (a 2 mm block half grey matter, half white), and
        # each tissue's spread, so every SNR, came out about twice MRIQC's.
        # The working posteriors are the priors; the noise is the image's.
        from . import segment as S

        if labels_n is not None:
            # The model's own class at each voxel, held loosely: where the
            # intensity disagrees (a boundary voxel), the class follows it.
            lab = labels_n.ravel()[brain_n_idx].astype(int)
            prior = np.full((3, lab.size), 0.1)
            prior[np.clip(lab - 1, 0, 2), np.arange(lab.size)] = 0.8
        else:
            prior = np.clip(seg.posteriors[:, wijk[0], wijk[1], wijk[2]].astype(np.float64),
                            1e-3, 1.0)
        prior /= prior.sum(axis=0)
        post_n, _m, _sd = S.em(corrected.astype(np.float64), prior, seg.means.copy(),
                               seg.sds * np.sqrt(float(np.prod(factor))))
        for k, name in enumerate(("csf", "gm", "wm")):
            tissue_stats[name] = stats.weighted_stats(corrected, post_n[k])
        for name in ("csf", "gm", "wm"):
            s = tissue_stats[name]
            n = max(s["n"], 2.0)
            metrics.append(_metric(f"snr_{name}", s["median"] / (s["stdv"] * np.sqrt(n / (n - 1)))
                                   if s["stdv"] > 0 else None))
        metrics.append(_metric("snr_total", float(np.nanmean(
            [m.value for m in metrics if m.key.startswith("snr_") and m.value is not None]))))
        gm, wm = tissue_stats["gm"], tissue_stats["wm"]
        gap = abs(wm["median"] - gm["median"])
        metrics.append(_metric("cjv", (wm["mad"] + gm["mad"]) / gap if gap > 0 else None))
    else:
        why = _why_no_tissue(suffix, reg, thinnest)
        for key in ("snr_csf", "snr_gm", "snr_wm", "snr_total", "cjv"):
            metrics.append(_metric(key, None, why))

    # The air.
    if air_ok:
        air_n = stats.block_expand(air_w, factor, shape) & ~zero_n
        air_vals = data[air_n]
        sd_air = float(air_vals.std()) if air_vals.size else float("nan")
        if seg is not None and sd_air > 0:
            snrd = {k: RAYLEIGH * tissue_stats[k]["median"] / sd_air for k in ("csf", "gm", "wm")}
            metrics.append(_metric("snrd_wm", snrd["wm"]))
            metrics.append(_metric("snrd_total", float(np.mean(list(snrd.values())))))
            gm, wm = tissue_stats["gm"], tissue_stats["wm"]
            metrics.append(_metric("cnr", abs(wm["median"] - gm["median"]) / np.sqrt(
                sd_air ** 2 + gm["stdv"] ** 2 + wm["stdv"] ** 2)))
        else:
            why = "Needs the tissue classes." if seg is None else "The air has no spread."
            for key in ("snrd_wm", "snrd_total", "cnr"):
                metrics.append(_metric(key, None, why))
        metrics.append(_metric("efc", _efc(data[~zero_n])))
        head_energy = float(np.median(data[head_n] ** 2)) if head_n.any() else float("nan")
        outside = ~head_n & ~zero_n
        air_energy = float(np.median(data[outside] ** 2)) if outside.any() else 0.0
        metrics.append(_metric("fber", head_energy / air_energy if air_energy > 1e-6 else None,
                               "The air's energy is zero."))
        art_w = K.artefacts(work, air_w, head_w)
        metrics.append(_metric("qi1", 100.0 * float(art_w.sum()) / max(float(air_w.sum()), 1.0)))
        metrics.append(_metric("qi2", _qi2(air_vals), "Too little air to fit."))
        pe = _pe_axis(sidecar)
        if pe is not None:
            n_pe = work.shape[pe]
            ghost = np.roll(head_w, n_pe // 2, axis=pe) & air_w
            rest = air_w & ~ghost
            head_mean = float(work[head_w].mean()) if head_w.any() else 0.0
            if ghost.sum() > 50 and rest.sum() > 50 and head_mean > 0:
                ratio = (float(work[ghost].mean()) - float(work[rest].mean())) / head_mean
            else:
                ratio = None
            metrics.append(_metric("ghost_ratio", ratio, "Too little air where a ghost "
                                                         "would fall."))
            facts["phase_encoding_axis"] = "ijk"[pe]
        else:
            metrics.append(_metric("ghost_ratio", None, "The sidecar names no phase-encoding "
                                                        "direction."))
        facts["air_summary"] = stats.weighted_stats(air_vals, np.ones(air_vals.size))
    else:
        art_w = None
        for key in ("snrd_wm", "snrd_total", "cnr", "efc", "fber", "qi1", "qi2", "ghost_ratio"):
            metrics.append(_metric(key, None, air_reason))

    # Tissues.
    if seg is not None:
        # MRIQC's maximum is the ceiling it clips the image to first: the
        # 99.9th percentile of the non-zero image (its median-filtered copy).
        nonzero = data[data > 0]
        top = float(np.percentile(nonzero, 99.9)) if nonzero.size else float("nan")
        metrics.append(_metric("wm2max", tissue_stats["wm"]["median"] / top
                               if top > 0 else None))
        from . import bias as B

        inu = B.nonuniformity(seg.field_log, seg.posteriors.sum(axis=0) > 0)
        metrics.append(_metric("inu_range", inu["range"]))
        vol = seg.posteriors.reshape(3, -1).sum(axis=1)
        total = float(vol.sum()) or 1.0
        for k, name in enumerate(("csf", "gm", "wm")):
            metrics.append(_metric(f"fraction_{name}", float(vol[k]) / total))
        # At the image's own resolution: the share of brain voxels no class
        # holds for sure.
        metrics.append(_metric("partial_volume", float(np.mean(
            post_n.max(axis=0) < A.partial_volume_p))))
        for k, name in ((1, "gm"), (2, "wm")):
            if tpms is None:
                metrics.append(_metric(f"template_overlap_{name}", None,
                                       "No registration to the template."))
                continue
            p, q = seg.posteriors[k], tpms[k]
            hi = float(np.maximum(p, q).sum())
            metrics.append(_metric(f"template_overlap_{name}",
                                   float(np.minimum(p, q).sum()) / hi if hi > 0 else None))
    else:
        why = _why_no_tissue(suffix, reg, thinnest)
        for key in ("wm2max", "inu_range", "fraction_csf", "fraction_gm", "fraction_wm",
                    "partial_volume", "template_overlap_gm", "template_overlap_wm"):
            metrics.append(_metric(key, None, why))
    step("measures", 5)

    # Smoothness, inside the brain at the image's resolution.
    fw = _fwhm(data, brain_n, zooms) if brain_n.sum() > 1000 else [float("nan")] * 3
    for axis, name in enumerate("xyz"):
        metrics.append(_metric(f"fwhm_{name}", fw[axis]))
    metrics.append(_metric("fwhm_avg", float(np.nanmean(fw)) if np.isfinite(fw).any() else None))

    # Coverage and the header.
    if reg is not None:
        fov = R.outside_fov(reg, shape, affine)
        metrics.append(_metric("fov_cut", 100.0 * fov["fraction"]))
    else:
        fov = None
        metrics.append(_metric("fov_cut", None, f"A slab of {thinnest:.0f} mm: the template "
                                                "is not registered to it."))
    head_vals = data[head_n]
    top_value = float(head_vals.max()) if head_vals.size else 0.0
    sat = (100.0 * float(np.count_nonzero(head_vals >= top_value)) / head_vals.size
           if head_vals.size and top_value > 0 else None)
    metrics.append(_metric("saturation", sat))
    result.metrics = metrics
    step("coverage", 6)

    # 6. Findings.
    findings = result.findings
    if tool_note:
        findings.append(Finding(
            "approximate", "Approximate masks", "info",
            f"The methods chosen in Settings could not run ({tool_note}): the fast "
            "methods made the masks, so masks and tissue measures are approximate.",
            "brain"))
    if uncertain:
        findings.append(Finding(
            "registration", "Registration to the template is uncertain", "warning",
            (f"The template's brain agrees with the brain found at Dice "
             f"{facts['registration'].get('dice_with_brain', 0):.2f}. "
             if "dice_with_brain" in facts["registration"] else
             f"The image matched the template at a correlation of {reg.correlation:.2f}, "
             f"with {100 * brain_in_head:.0f} % of the template's brain inside the head. ")
            + "The face, the field-of-view check and the template overlap depend on it.",
            "brain"))
    if reg is None:
        findings.append(Finding(
            "slab", "A thin slab", "info",
            f"The image covers {thinnest:.0f} mm along its thinnest axis: too little of the "
            "head to register the template to. The brain mask is the head's inside, and "
            "the field of view and the face are not checked.", "brain"))
    elif fov["fraction"] > A.fov_warning_pct / 100.0:
        sides = ", ".join(f"{k} {100 * v:.1f} %" for k, v in sorted(
            fov["sides"].items(), key=lambda kv: -kv[1]) if v >= 0.001)
        if slab:
            # A slab covers part of the brain on purpose: coverage, not a fault.
            findings.append(Finding(
                "fov_cut", "Part of the brain is covered", "info",
                f"The image is a slab of {thinnest:.0f} mm and covers "
                f"{100 * (1 - fov['fraction']):.0f} % of the brain ({sides} outside).",
                "brain"))
        else:
            findings.append(Finding(
                "fov_cut", "The field of view cuts the brain",
                "error" if fov["fraction"] > A.fov_error_pct / 100.0 else "warning",
                f"{100 * fov['fraction']:.1f} % of the brain lies outside the image "
                f"({sides}).", "brain"))
    if defaced:
        findings.append(Finding("defaced", "Defaced", "info",
                                (f"Defaced {record}. " if record else
                                 "The face region is blank, as defacing leaves it. ")
                                + "The air was not measured.", None))
    elif not stripped and blank < 0.1 and float(face.sum()) > 50:
        findings.append(Finding(
            "face", "A face is visible", "warning",
            "The sidecar records no defacing and the face region holds tissue. "
            "Deface the image before sharing the dataset.", "face"))
    if stripped:
        findings.append(Finding("skull_stripped", "Skull stripped", "info",
                                "Next to nothing outside the brain holds signal: the "
                                "image holds the brain only.", None))
    if air_ok and art_w is not None:
        qi1 = result.value("qi1") or 0.0
        if qi1 > A.air_artefact_pct:
            findings.append(Finding(
                "air_artefacts", "Structured signal in the air", "warning",
                f"{qi1:.2f} % of the air above the face holds signal well above its "
                "noise: ghosting, ringing, motion or wrap-around.", "artefacts"))
        pe = _pe_axis(sidecar)
        if pe is not None and art_w.any():
            n_pe = work.shape[pe]
            edge = max(2, int(0.1 * n_pe))
            idx = np.arange(n_pe)
            slab = (idx < edge) | (idx >= n_pe - edge)
            shape_b = [1, 1, 1]
            shape_b[pe] = n_pe
            slab_mask = np.broadcast_to(slab.reshape(shape_b), work.shape) & air_w
            touches = bool(np.take(head_w, [0, 1, n_pe - 2, n_pe - 1], axis=pe).any())
            share = float(art_w[slab_mask].mean()) if slab_mask.any() else 0.0
            if touches and share > 0.02:
                findings.append(Finding(
                    "wrap", "Wrap-around suspected", "warning",
                    f"The head reaches the edge of the field of view along the phase-"
                    f"encoding axis ({'ijk'[pe]}) and {100 * share:.1f} % of the air at "
                    "both ends of that axis holds structured signal: part of the head "
                    "folded over.", "artefacts"))
    if sat is not None and sat > A.saturation_pct:
        findings.append(Finding("saturation", "Saturated voxels", "warning",
                                f"{sat:.2f} % of the head is at the image's maximum: "
                                "bright tissue was clipped.", None))
    anis = float(zooms.max() / max(zooms.min(), 1e-6))
    if anis > 2.0:
        findings.append(Finding(
            "anisotropy", "Thick voxels", "info",
            f"The voxels are {anis:.1f} times longer along one axis "
            f"({' x '.join(f'{z:.2f}' for z in zooms)} mm): tissue measures are less "
            "reliable.", None))
    if header is not None:
        findings += _header_findings(header)
    if seg is None and tissue_suffix and reg is not None:
        findings.append(Finding("segmentation", "No tissue classes", "warning",
                                "The brain could not be segmented: the tissue measures are "
                                "missing.", None))
    step("findings", 7)

    # Maps for the viewer, on the working grid.
    maps = result.maps
    if engines["brain"] == "mindgrab":
        maps["brain"] = QCMap("brain", "Brain mask", brain_n.astype(np.uint8), affine, "mask",
                              "The brain as mindgrab finds it: where the tissue measures "
                              "are taken.", colour="red")
    else:
        maps["brain"] = QCMap("brain", "Brain mask", brain_w.astype(np.uint8), waff, "mask",
                              "The template's brain, registered to the image (approximate): "
                              "where the tissue measures are taken.", colour="red")
    maps["head"] = QCMap("head", "Head mask", head_w.astype(np.uint8), waff, "mask",
                         "Air and head are separated here.", colour="blue")
    if air_ok:
        maps["air"] = QCMap("air", "Air measured", air_w.astype(np.uint8), waff, "mask",
                            "The air the noise and artefact measures use: outside the head, "
                            "above the face and neck, without zero fill.", colour="green")
        if art_w is not None:
            maps["artefacts"] = QCMap("artefacts", "Artefacts in the air",
                                      art_w.astype(np.uint8), waff, "mask",
                                      "Air voxels far above the air's noise (QI1).",
                                      colour="warm")
    if seg is not None:
        if labels_n is not None:
            maps["tissues"] = QCMap("tissues", "Tissue classes", labels_n, affine, "labels",
                                    "Grey and white matter as the tissue model finds them; "
                                    "CSF is the rest of the brain.",
                                    labels={1: "CSF", 2: "Grey matter", 3: "White matter"})
        else:
            maps["tissues"] = QCMap("tissues", "Tissue classes", seg.labels(seg.posteriors.sum(
                axis=0) > 0), waff, "labels", "CSF, grey matter and white matter as found "
                "(approximate).", labels={1: "CSF", 2: "Grey matter", 3: "White matter"})
        field = np.exp(seg.field_log).astype(np.float32)
        field[seg.posteriors.sum(axis=0) <= 0] = 0.0
        maps["bias"] = QCMap("bias", "Bias field", field, waff, "field",
                             "The intensity non-uniformity found, as a multiplier (1 is "
                             "none).", colour="blue2red")
    if not defaced and not stripped and face.any():
        maps["face"] = QCMap("face", "Face region", face.astype(np.uint8), waff, "mask",
                             "Where a face is expected, from the template.", colour="cool")
    facts["seconds"] = round(time.perf_counter() - t_start, 2)
    facts["timings"] = timings
    if progress is not None:
        progress(8, 8)
    return result


def _header_findings(header) -> list[Finding]:
    out: list[Finding] = []
    try:
        q, qcode = header.get_qform(coded=True)
        s, scode = header.get_sform(coded=True)
    except Exception:  # noqa: BLE001 - not a NIfTI header
        return out
    if qcode and scode and q is not None and s is not None:
        if np.abs(np.asarray(q) - np.asarray(s)).max() > 1e-3:
            out.append(Finding(
                "qform_sform", "The two orientations disagree", "warning",
                "The header's qform and sform describe different positions: tools that "
                "read one or the other place the image differently.", None))
    aff = s if scode else q
    if aff is not None:
        lin = np.asarray(aff)[:3, :3]
        axes = lin / np.maximum(np.linalg.norm(lin, axis=0), 1e-9)
        tilt = float(np.degrees(np.arccos(np.clip(np.abs(axes).max(axis=0), -1, 1))).max())
        if tilt > 5.0:
            out.append(Finding("oblique", "Oblique acquisition", "info",
                               f"The voxel axes are tilted up to {tilt:.0f} degrees from "
                               "the scanner's axes.", None))
    return out


def check_file(path: Path, *, root: Optional[Path] = None, cancel=None,
               progress=None, config: Optional[QcConfig] = None) -> QCResult:
    """Read ``path`` and its sidecar (with inheritance) and check it."""
    import nibabel as nib

    from ..viz import bids as VB

    path = Path(path)
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    root = root or VB.dataset_root(path)
    sidecar = VB.inherited_sidecar(path, root)
    suffix = VB.suffix_of(path.name)
    return check(np.asarray(data, dtype=np.float32), img.affine, suffix=suffix,
                 sidecar=sidecar, path=str(path), header=img.header, cancel=cancel,
                 progress=progress, config=config)


__all__ = ["METRICS", "SUFFIXES", "TISSUE_SUFFIXES", "check", "check_file", "defacing_record"]

"""Overlays: one image drawn over another, and the look each kind opens with.

Any image can go over any other: the overlay is resampled at the WORLD
position of every pixel of the base's grid (:func:`views.layer_values`), so
a 2 mm statistical map lands on a 1 mm T1, and an MRS voxel on its anatomy,
wherever their grids sit.

What an overlay IS decides how it should look, and getting that wrong makes
it useless on first sight (an atlas through a grey colour map is noise; a
z-map drawn without a threshold hides the anatomy under it). So the data is
looked at once, on the worker that read it:

``labels``       a segmentation or atlas: one colour per region, named from
                 its ``_dseg.tsv`` (else numbered), half transparent
``mask``         two values, one of them zero: one colour, half transparent
``probability``  values in [0, 1]: a heat map, the lowest fifth hidden
``two_tailed``   values both sides of zero (a z- or t-map): warm for positive,
                 cool for negative, |value| below the threshold hidden
``image``        anything else (a mean, another contrast): a heat map at half
                 opacity, background hidden, for checking a registration

Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .bids import suffix_of
from .data import labels as L
from .compute.shapes import GridBox
from .data.volume import VolumeSource, open_volume
from .scene import VolumeDisplay

Kind = Literal["labels", "mask", "probability", "two_tailed", "image"]


@dataclass
class Overlay:
    source: VolumeSource
    display: VolumeDisplay
    kind: Kind
    #: A sentence for the status bar ("an atlas of 104 regions").
    note: str = ""
    #: The layer's name, when not the file's.
    name: str = ""


def _sample(src: VolumeSource) -> np.ndarray:
    raw = src.raw_frame(0)
    if raw is None:
        return np.empty(0, dtype=np.float32)
    flat = np.asarray(raw).reshape(-1)
    if flat.size > 2_000_000:
        flat = flat[:: flat.size // 2_000_000]
    return flat


def piecewise_constant(src: VolumeSource) -> float:
    """The share of non-zero voxels equal to their neighbour along x.

    What makes a label image one: regions of one value. An 8-bit anatomical
    image has as few distinct values as an atlas (256 at most), but almost
    no neighbour pair in it is equal, while in a segmentation nearly all are.
    """
    raw = src.raw_frame(0)
    if raw is None:
        return 0.0
    f = np.asarray(raw)
    if f.ndim != 3 or f.shape[0] < 2:
        return 0.0
    f = f[:, ::2, ::2]
    a, b = f[1:], f[:-1]
    inside = a != 0
    n = int(inside.sum())
    return float(((a == b) & inside).sum()) / n if n else 0.0


def classify(src: VolumeSource, path: Path) -> tuple[Kind, Optional[np.ndarray]]:
    """What ``src`` holds (its first frame), and its distinct values when it
    is a label image."""
    if src.is_rgb:
        return "image", None
    suffix = suffix_of(Path(path).name)
    raw = _sample(src)
    values = src.scale(raw)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "image", None
    if suffix == "probseg":
        return "probability", None
    distinct = L.distinct_values(finite)
    if distinct is not None:
        nonzero = distinct[distinct != 0]
        if suffix == "mask" or (distinct.size <= 2 and nonzero.size == 1):
            return "mask", distinct
        if suffix == "dseg" or (nonzero.size >= 2 and piecewise_constant(src) > 0.7):
            return "labels", distinct
    lo, hi = float(finite.min()), float(finite.max())
    if lo >= 0.0 and hi <= 1.0:
        return "probability", None
    if lo < 0.0 < hi:
        # Both tails must be real, not a rounding error under a positive map.
        neg = float(np.percentile(-finite[finite < 0], 99)) if np.any(finite < 0) else 0.0
        if neg > 0.05 * hi:
            return "two_tailed", None
    return "image", None


def _robust_nonzero(values: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[float, float]:
    v = values[np.isfinite(values) & (values != 0)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = (float(x) for x in np.percentile(v, (lo_pct, hi_pct)))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def display_for(src: VolumeSource, path: Path) -> tuple[VolumeDisplay, Kind, str]:
    """The look an overlay opens with, its kind, and a sentence about it."""
    kind, distinct = classify(src, path)
    values = src.scale(_sample(src))
    if kind == "labels":
        table_path = L.table_path_for(Path(path))
        table = L.read_label_tsv(table_path) if table_path is not None else None
        named = table is not None
        if table is None:
            table = L.generated_table(distinct if distinct is not None else [])
        n = len([k for k in table.labels if k != 0])
        note = (f"{n} regions, named from {table_path.name}" if named
                else f"{n} labels (no lookup table beside it, so numbered)")
        return (VolumeDisplay(label_table=table, opacity=0.5, interpolation="nearest"),
                kind, note)
    if kind == "mask":
        top = float(np.max(distinct)) if distinct is not None and distinct.size else 1.0
        return (VolumeDisplay(colormap="red", window=(top * 0.5, top), opacity=0.5,
                              threshold_mode="hide_below", interpolation="nearest",
                              gamma=1.0),
                kind, "a mask")
    if kind == "probability":
        return (VolumeDisplay(colormap="hot", window=(0.2, 1.0), opacity=0.7,
                              threshold_mode="hide_below", gamma=1.0),
                kind, "a probability map, below 0.2 hidden")
    if kind == "two_tailed":
        mags = np.abs(values[np.isfinite(values) & (values != 0)])
        if mags.size:
            thr, sat = (float(x) for x in np.percentile(mags, (90, 99.5)))
        else:
            thr, sat = 1.0, 2.0
        if not sat > thr:
            sat = thr + 1.0
        return (VolumeDisplay(colormap="warm", colormap_negative="cool",
                              window=(thr, sat), window_negative=(thr, sat),
                              threshold_mode="hide_below", gamma=1.0),
                kind, f"two-tailed, |value| below {thr:.3g} hidden")
    # The window is taken inside the head: over every non-zero voxel the
    # background noise drags the low end down and all tissue saturates.
    from .compute.qc import brain_mask

    finite = np.where(np.isfinite(values), values, 0.0)
    head = finite[brain_mask(finite)]
    lo, hi = (_robust_nonzero(head, 5.0, 98.0) if head.size
              else _robust_nonzero(values, 2.0, 99.5))
    return (VolumeDisplay(colormap="hot", window=(lo, hi), opacity=0.5,
                          threshold_mode="hide_below"),
            kind, "drawn at half opacity")


def mrs_voxel(path: Path) -> Overlay:
    """Where a spectrum was measured: the MRS file's voxel (or, for MRSI,
    its whole grid) as an outlined box, to draw on the anatomy.

    Built one voxel larger than the MRS grid on every side, ones inside and
    zeros around, on the file's own affine shifted by that voxel. Sampled
    nearest-neighbour, the edge then falls exactly on the voxel's faces, at
    whatever angle the voxel was placed.
    """
    import nibabel as nib

    from .data.volume import array_volume

    img = nib.load(str(path))
    shape = tuple(int(n) for n in img.shape[:3])
    data = np.zeros(tuple(n + 2 for n in shape), dtype=np.float32)
    data[1:-1, 1:-1, 1:-1] = 1.0
    shift = np.eye(4)
    shift[:3, 3] = -1.0
    affine = np.asarray(img.affine, dtype=float) @ shift
    src = array_volume(data, affine, path=Path(path), name="MRS voxel")
    # The box itself, which the views draw exactly; the array above is what
    # the readout and a sampled view fall back on.
    src.box = GridBox(np.asarray(img.affine, dtype=float), shape)
    display = VolumeDisplay(colormap="green", window=(0.5, 1.0), threshold_mode="hide_below",
                            interpolation="nearest", outline_px=2.0, gamma=1.0)
    n = int(np.prod(shape))
    note = "the spectroscopy voxel" if n == 1 else f"the spectroscopy grid ({n} voxels)"
    return Overlay(source=src, display=display, kind="mask", note=note,
                   name=f"Voxel of {Path(path).name}")


def deface_preview(path: Path, engine_id: str = "") -> Overlay:
    """What a defacing engine would blank in ``path``, as a red overlay on the
    untouched original. Runs the engine (seconds): on a worker."""
    from ..deface import DEFAULT_ENGINE_ID, engine
    from ..deface.preview import removed_mask, summary
    from .data.volume import array_volume

    engine_id = engine_id or DEFAULT_ENGINE_ID
    mask, head, affine = removed_mask(Path(path), engine_id=engine_id)
    src = array_volume(mask, affine, path=Path(path), name="Removed by defacing")
    display = VolumeDisplay(colormap="red", window=(0.5, 1.0), threshold_mode="hide_below",
                            interpolation="nearest", opacity=0.45, gamma=1.0)
    label = engine(engine_id).label
    return Overlay(source=src, display=display, kind="mask",
                   note=f"{label}: {summary(mask, head)}. Nothing was changed.",
                   name="What defacing would remove")


def open_overlay(path, *, budget_bytes: Optional[int] = None, cancel=None) -> Overlay:
    """Read ``path`` whole and decide its look. Runs on a worker. A
    spectroscopy file is drawn as its voxel."""
    from .data.formats import is_mrs_path

    path = Path(path)
    if is_mrs_path(path):
        return mrs_voxel(path)
    src = open_volume(path)
    src.stream(cancel=cancel, budget_bytes=budget_bytes)
    display, kind, note = display_for(src, path)
    return Overlay(source=src, display=display, kind=kind, note=note)


def candidate_hint(path, base_affine=None, base_shape=None) -> str:
    """What an image would be drawn as over the open one, from its NAME and
    HEADER alone (a picker must not read every image): a sentence for the
    overlay picker. Its content decides in the end (:func:`display_for`)."""
    import re

    from .data.formats import is_mrs_path

    path = Path(path)
    name = path.name.lower()
    if is_mrs_path(path):
        return "Spectroscopy: drawn as the outline of its voxel, exactly."
    try:
        import nibabel as nib

        img = nib.load(str(path))
        shape = tuple(int(v) for v in img.shape)
        affine = np.asarray(img.affine, dtype=float)
    except Exception:  # noqa: BLE001 - a hint must not fail on a header
        return ""
    if "_dseg" in name:
        what = "Segmentation: each label in its own colour, named when a dseg.tsv describes it."
    elif "_probseg" in name:
        what = "Probability map: a heat map, faint where the probability is low."
    elif "_mask" in name:
        what = "Mask: a translucent region with its outline."
    elif re.search(r"(^|_)stat-|_statmap|zstat|tstat", name):
        what = "Statistical map: positive and negative tails in their own colours."
    elif len(shape) > 3 and shape[3] > 1:
        what = (f"Series of {shape[3]} volumes: drawn at the volume on screen; its time "
                "course can be plotted.")
    else:
        what = "Image: drawn in a heat colour map at half opacity."
    grid = ""
    if base_affine is not None and base_shape is not None:
        same = (tuple(shape[:3]) == tuple(base_shape[:3])
                and np.allclose(affine, np.asarray(base_affine, dtype=float), atol=1e-3))
        grid = (" Same voxel grid as the open image." if same else
                " Different voxel grid: resampled in scanner space to the open image.")
    dims = " x ".join(str(v) for v in shape)
    return f"{dims}. {what}{grid}"


def computed_overlay(src: VolumeSource, path: Path) -> Overlay:
    """An overlay for a source built in memory (a QC map)."""
    display, kind, note = display_for(src, path)
    return Overlay(source=src, display=display, kind=kind, note=note)


__all__ = ["Kind", "Overlay", "candidate_hint", "classify", "computed_overlay",
           "display_for", "mrs_voxel",
           "open_overlay", "piecewise_constant"]

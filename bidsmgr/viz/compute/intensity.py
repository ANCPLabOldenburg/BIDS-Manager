"""From data values to colours: windows, colour maps, compositing.

Every 2-D view draws through :func:`colorize` and :func:`composite`. They are
vectorised end to end (one ``np.take`` per layer), because they run on every
repaint: a 256 x 256 slice takes about a millisecond.

Values are in the data's own units. A window is ``(lo, hi)``; a value is
normalised to ``(v - lo) / (hi - lo)``, clipped, bent by the gamma and looked
up in the colour map. NaN means "no data here" (outside an overlay's volume)
and is always transparent.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import colormaps
from ..scene import LabelTable, VolumeDisplay


def normalise(values: np.ndarray, lo: float, hi: float, gamma: float = 1.0) -> np.ndarray:
    """``values`` to 0..1 through the window, then the display gamma."""
    span = float(hi) - float(lo)
    if not np.isfinite(span) or span == 0:
        span = 1.0
    out = (np.asarray(values, dtype=np.float32) - np.float32(lo)) / np.float32(span)
    np.clip(out, 0.0, 1.0, out=out)
    if gamma and abs(gamma - 1.0) > 1e-6:
        np.power(out, np.float32(1.0 / gamma), out=out)
    return out


def _lut_index(norm: np.ndarray) -> np.ndarray:
    idx = np.nan_to_num(norm, nan=0.0) * 255.0 + 0.5
    return idx.astype(np.uint8)


def colorize(values: np.ndarray, display: VolumeDisplay, *, is_base: bool,
             window: Optional[tuple[float, float]] = None) -> np.ndarray:
    """One layer's slice as RGBA ``uint8`` (rows, cols, 4).

    ``values`` is (rows, cols) for a scalar image or (rows, cols, C) for a
    colour image (already display values, 0-1 or 0-255).
    """
    rows, cols = values.shape[:2]
    alpha = float(np.clip(display.opacity, 0.0, 1.0))

    # -- colour images: show the colours, only the opacity applies ----------
    if values.ndim == 3:
        rgb = np.asarray(values[..., :3], dtype=np.float32)
        finite = np.isfinite(rgb).all(axis=-1)
        rgb = np.nan_to_num(rgb)
        top = float(rgb.max()) if rgb.size else 0.0
        scale = 255.0 if top <= 1.0 + 1e-6 else 255.0 / max(top, 1e-6)
        out = np.empty((rows, cols, 4), dtype=np.uint8)
        out[..., :3] = np.clip(rgb * scale, 0, 255).astype(np.uint8)
        out[..., 3] = np.where(finite, int(round(alpha * 255)), 0).astype(np.uint8)
        return out

    vals = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(vals)

    # -- label images (atlases): one colour per integer ----------------------
    if display.label_table is not None:
        out = _colorize_labels(vals, finite, display.label_table, alpha)
        if display.outline_px > 0:
            regions = np.where(finite, np.round(vals), 0).astype(np.int64)
            out[..., 3] = np.where(edges(regions, display.outline_px), out[..., 3], 0)
        return out

    lo, hi = window if window is not None else (display.window or (0.0, 1.0))
    norm = normalise(vals, lo, hi, display.gamma)
    table = colormaps.lut(display.colormap, display.invert)
    out = np.take(table, _lut_index(norm), axis=0)

    a = np.full((rows, cols), alpha, dtype=np.float32)
    mode = display.threshold_mode
    if not is_base or mode != "range":
        below = vals < np.float32(lo)
        if mode == "hide_below":
            a[below] = 0.0
        elif mode == "translucent_below":
            # Fade toward the threshold rather than vanish: how close a
            # sub-threshold voxel came is the information.
            span = max(float(hi) - float(lo), 1e-6)
            frac = np.clip(1.0 - (np.float32(lo) - vals) / np.float32(span), 0.0, 1.0)
            a = np.where(below, alpha * 0.45 * frac, a)

    # -- the negative tail of a two-tailed map -------------------------------
    if display.colormap_negative:
        neg_lo, neg_hi = display.window_negative or (abs(float(lo)), abs(float(hi)))
        neg = vals < 0
        if np.any(neg):
            mag = -vals
            neg_norm = normalise(mag, neg_lo, neg_hi, display.gamma)
            neg_table = colormaps.lut(display.colormap_negative, display.invert)
            neg_rgba = np.take(neg_table, _lut_index(neg_norm), axis=0)
            out[neg] = neg_rgba[neg]
            neg_alpha = np.full((rows, cols), alpha, dtype=np.float32)
            if mode == "hide_below":
                neg_alpha[mag < np.float32(neg_lo)] = 0.0
            elif mode == "translucent_below":
                span = max(float(neg_hi) - float(neg_lo), 1e-6)
                frac = np.clip(1.0 - (np.float32(neg_lo) - mag) / np.float32(span), 0.0, 1.0)
                neg_alpha = np.where(mag < neg_lo, alpha * 0.45 * frac, neg_alpha)
            a = np.where(neg, neg_alpha, a)

    a[~finite] = 0.0
    if display.outline_px > 0 and not is_base:
        # Only the edge of what is shown: a cluster's extent without hiding
        # the anatomy inside it.
        a = np.where(edges((a > 0).astype(np.int64), display.outline_px), a, 0.0)
    out[..., 3] = (a * 255.0 + 0.5).astype(np.uint8)
    return out


def _colorize_labels(vals: np.ndarray, finite: np.ndarray, table: LabelTable,
                     alpha: float) -> np.ndarray:
    rows, cols = vals.shape
    out = np.zeros((rows, cols, 4), dtype=np.uint8)
    ints = np.where(finite, np.round(vals), 0).astype(np.int64)
    top = int(ints.max()) if ints.size else 0
    palette = np.zeros((max(top, 0) + 1, 4), dtype=np.uint8)
    for value, rgb in table.colors.items():
        if 0 < value <= top:
            palette[value, :3] = rgb
            palette[value, 3] = int(round(alpha * 255))
    # Labels without a colour still show, in a stable colour of their own.
    for value in range(1, top + 1):
        if palette[value, 3] == 0 and (value in table.labels or not table.colors):
            palette[value, :3] = stable_colour(value)
            palette[value, 3] = int(round(alpha * 255))
    safe = np.clip(ints, 0, top)
    out[:] = palette[safe]
    out[~finite | (ints <= 0)] = 0
    return out


def stable_colour(value: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(value * 7919 + 17)
    rgb = rng.integers(60, 240, 3)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def composite(layers: list[np.ndarray]) -> np.ndarray:
    """Alpha-blend RGBA layers bottom to top onto an opaque black canvas."""
    if not layers:
        raise ValueError("nothing to composite")
    rows, cols = layers[0].shape[:2]
    acc = np.zeros((rows, cols, 3), dtype=np.float32)
    for rgba in layers:
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        acc = acc * (1.0 - a) + rgba[..., :3].astype(np.float32) * a
    out = np.empty((rows, cols, 4), dtype=np.uint8)
    out[..., :3] = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def edges(regions: np.ndarray, width: float = 1.0) -> np.ndarray:
    """The pixels of each non-zero region that touch a different value
    (4-neighbourhood), ``width`` pixels deep. Region ids are integers; for a
    mask, 1 inside and 0 out."""
    r = np.asarray(regions)
    e = np.zeros(r.shape, dtype=bool)
    dy = r[1:, :] != r[:-1, :]
    dx = r[:, 1:] != r[:, :-1]
    e[1:, :] |= dy
    e[:-1, :] |= dy
    e[:, 1:] |= dx
    e[:, :-1] |= dx
    inside = r != 0
    e &= inside
    for _ in range(max(0, int(round(width)) - 1)):
        grown = e.copy()
        grown[1:, :] |= e[:-1, :]
        grown[:-1, :] |= e[1:, :]
        grown[:, 1:] |= e[:, :-1]
        grown[:, :-1] |= e[:, 1:]
        e = grown & inside
    return e


def outline(mask: np.ndarray) -> np.ndarray:
    """The boundary pixels of a boolean mask (4-neighbourhood)."""
    m = np.asarray(mask, dtype=bool)
    edge = np.zeros_like(m)
    edge[1:, :] |= m[1:, :] != m[:-1, :]
    edge[:-1, :] |= m[1:, :] != m[:-1, :]
    edge[:, 1:] |= m[:, 1:] != m[:, :-1]
    edge[:, :-1] |= m[:, 1:] != m[:, :-1]
    return edge & m


__all__ = ["colorize", "composite", "edges", "normalise", "outline", "stable_colour"]

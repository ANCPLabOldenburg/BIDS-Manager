"""Colour bars: what each layer's colours MEAN, as data.

A colour without a scale is a picture, not a measurement. Every visible
scalar layer gets a bar (top-most first): its name and what its values are
(``VolumeSource.quantity``: "tSNR (mean / SD)"), the colour map over its
window, round-number ticks, the threshold marked when values below the
window are hidden or faint, and the negative tail of a two-tailed map as a
mirrored bar on the left. Atlases get none (their colours are names, shown
on hover); colour images get none (their colours are directions).

Qt-free: the canvas only paints what :func:`bars_for` describes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

#: At most this many bars stacked under a view.
MAX_BARS = 3


@dataclass(frozen=True)
class BarSpec:
    title: str
    colormap: str
    invert: bool
    lo: float
    hi: float
    ticks: tuple[tuple[float, str], ...] = ()
    #: Where values start to be hidden (or faint), in data units.
    threshold: Optional[float] = None
    #: A two-tailed map's negative tail: (colour map, lo, hi) as magnitudes.
    negative: Optional[tuple[str, float, float]] = None
    gamma: float = 1.0
    extra: dict = field(default_factory=dict)


def nice_step(span: float, target: int = 4) -> float:
    """A round step (1, 2, 2.5 or 5 times a power of ten) giving about
    ``target`` intervals over ``span``."""
    if not span > 0 or not math.isfinite(span):
        return 1.0
    raw = span / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def nice_floor(x: float) -> float:
    """The largest 1, 2 or 5 times a power of ten not above ``x``."""
    if not x > 0 or not math.isfinite(x):
        return 1.0
    mag = 10.0 ** math.floor(math.log10(x))
    for m in (5.0, 2.0, 1.0):
        if m * mag <= x * (1 + 1e-9):
            return m * mag
    return mag


def fmt(value: float, step: float) -> str:
    """A tick label with as many decimals as the step needs."""
    if step >= 1 or step <= 0:
        text = f"{value:.0f}"
    else:
        text = f"{value:.{min(6, int(math.ceil(-math.log10(step) - 1e-9)))}f}"
    return "0" if text in ("-0", "-0.0", "-0.00") else text


def nice_ticks(lo: float, hi: float, target: int = 4) -> tuple[tuple[float, str], ...]:
    """Round-number ticks inside ``[lo, hi]`` (the ends always shown)."""
    if not hi > lo:
        return ((lo, fmt(lo, 1.0)),)
    step = nice_step(hi - lo, target)
    first = math.ceil(lo / step) * step
    values = []
    v = first
    while v <= hi + step * 1e-9:
        values.append(v)
        v += step
    out = [(float(x), fmt(x, step)) for x in values]
    # The ends, unless a round tick sits right on them.
    if not out or out[0][0] - lo > step * 0.25:
        out.insert(0, (lo, f"{lo:.4g}"))
    if hi - out[-1][0] > step * 0.25:
        out.append((hi, f"{hi:.4g}"))
    return tuple(out)


def bars_for(store) -> list[BarSpec]:
    """A bar for each visible scalar layer, top-most first (at most
    :data:`MAX_BARS`)."""
    from . import views

    out: list[BarSpec] = []
    for layer in reversed(store.scene.layers):
        if layer.kind != "volume" or not layer.visible:
            continue
        d = layer.display
        src = views.source_of(store, layer)
        if src is None or src.is_rgb or d.label_table is not None or views.is_shape(src):
            continue
        window = d.window
        if window is None:
            window = src.robust_range(views.frame_of(store, layer, src)) or (0.0, 1.0)
        lo, hi = float(window[0]), float(window[1])
        name = layer.name or layer.id
        quantity = getattr(src, "quantity", "") or ""
        title = f"{quantity}  ·  {name}" if quantity else name
        threshold = lo if d.threshold_mode in ("hide_below", "translucent_below") else None
        negative = None
        if d.colormap_negative:
            nlo, nhi = d.window_negative or (abs(lo), abs(hi))
            negative = (d.colormap_negative, float(nlo), float(nhi))
        out.append(BarSpec(title=title, colormap=d.colormap, invert=d.invert, lo=lo, hi=hi,
                           ticks=nice_ticks(lo, hi), threshold=threshold, negative=negative,
                           gamma=float(d.gamma)))
        if len(out) >= MAX_BARS:
            break
    return out


__all__ = ["BarSpec", "MAX_BARS", "bars_for", "fmt", "nice_floor", "nice_step", "nice_ticks"]

"""Colour maps: names, and 256-entry lookup tables.

The tables come from NiiVue's colour-map files (BSD-2, vendored as data in
``bidsmgr/vendor/niivue_colormaps``). Each file holds a few control points;
:func:`lut` interpolates them to 256 RGBA entries once and caches the result,
so colouring a slice is a single array index.

A few maps (``ct_*``) carry a suggested window in Hounsfield units, which
:func:`suggested_window` exposes so picking "CT bones" can also pick a
sensible window.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Optional

import numpy as np

_PACKAGE = "bidsmgr.vendor.niivue_colormaps"

#: Shown first in pickers: the maps people reach for.
FAVOURITES = (
    "gray", "hot", "warm", "winter", "cool", "red", "green", "blue",
    "viridis", "plasma", "inferno", "magma", "jet", "bone", "copper",
)


@lru_cache(maxsize=1)
def _files() -> dict[str, object]:
    root = resources.files(_PACKAGE)
    out = {}
    for entry in root.iterdir():
        name = entry.name
        if name.endswith(".json") and not name.startswith("_"):
            out[name[:-5]] = entry
    return out


@lru_cache(maxsize=None)
def _spec(name: str) -> dict:
    entry = _files().get(name)
    if entry is None:
        raise KeyError(name)
    return json.loads(entry.read_text(encoding="utf-8"))


def names() -> list[str]:
    """Every colour map, favourites first, the rest alphabetically."""
    available = set(_files())
    first = [n for n in FAVOURITES if n in available]
    rest = sorted(available - set(first))
    return first + rest


def exists(name: str) -> bool:
    return name in _files()


@lru_cache(maxsize=128)
def lut(name: str, invert: bool = False) -> np.ndarray:
    """A (256, 4) uint8 RGBA table for ``name``.

    Unknown names fall back to gray rather than raising: a scene saved with a
    colour map a later version renamed should still open.
    """
    try:
        spec = _spec(name)
    except KeyError:
        spec = _spec("gray")
    n = min(len(spec["R"]), len(spec["G"]), len(spec["B"]))
    idx = np.asarray(spec.get("I") or np.linspace(0, 255, n), dtype=float)
    # Two upstream files (bcgwhw, bcgwhw_dark) list more positions than
    # colours; use the positions that have one.
    n = min(n, len(idx))
    idx = idx[:n]
    out = np.empty((256, 4), dtype=np.uint8)
    x = np.arange(256, dtype=float)
    for c, key in enumerate(("R", "G", "B")):
        out[:, c] = np.clip(np.round(np.interp(x, idx, spec[key][:n])), 0, 255)
    out[:, 3] = 255
    if invert:
        out[:, :3] = out[::-1, :3]
    out.setflags(write=False)
    return out


def suggested_window(name: str) -> Optional[tuple[float, float]]:
    """A colour map's own window (CT maps carry one in Hounsfield units)."""
    try:
        spec = _spec(name)
    except KeyError:
        return None
    if "min" in spec and "max" in spec:
        lo, hi = float(spec["min"]), float(spec["max"])
        # Some maps carry a placeholder (0, 0); a zero-width window is not a
        # suggestion.
        return (lo, hi) if hi > lo else None
    return None


def swatch(name: str, width: int = 128) -> np.ndarray:
    """A (1, width, 4) strip, for pickers and colour bars."""
    table = lut(name)
    pos = np.linspace(0, 255, width).astype(int)
    return table[pos][None, :, :]


__all__ = ["FAVOURITES", "exists", "lut", "names", "suggested_window", "swatch"]

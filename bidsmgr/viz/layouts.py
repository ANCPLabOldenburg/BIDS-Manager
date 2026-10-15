"""Which layout a file opens in, decided by what kind of file it is.

A BOLD run is looked at as a time course, a T1 as anatomy, a PET series as
uptake over time. One remembered layout for everything meant that opening a
BOLD after a T1 hid its graph and opening a T1 after a BOLD showed an empty
one. So each KIND of image has a preset, matched from its BIDS datatype and
suffix, and remembers the layout the user last gave it
(``VizSettings.layout_state``); a kind never arranged opens in its preset.

What a layout holds is the arrangement, never the look: the view mode, the
plane, whether the graph is open and how it is drawn. Display conventions
(RAS, radiological, labels) are the user's, the same for every kind.

The second half places views: :func:`tiles_for` says which views a mode
shows (from the scene's :class:`~bidsmgr.viz.scene.LayoutState`, which the
user edits), :func:`tile_rects` where each goes. Pure functions, so the
geometry is tested without a display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class LayoutPreset:
    id: str
    title: str
    #: Matched against the file: ``datatype`` / ``suffix`` (sets of names),
    #: ``is_4d`` (True / False). Absent keys match anything.
    match: Mapping[str, Any] = field(default_factory=dict)
    #: The arrangement a kind opens in before the user has given it one.
    #: ``mode`` "" means the best this machine can show.
    view: Mapping[str, Any] = field(default_factory=dict)


#: First match wins, so the specific kinds come before the general ones.
PRESETS: tuple[LayoutPreset, ...] = (
    LayoutPreset(
        "mri.func", "Functional MRI",
        match={"suffix": {"bold", "cbv", "phase", "sbref"}, "is_4d": True},
        view={"mode": "multi", "graph_visible": True, "graph": {"x_axis": "auto"}},
    ),
    LayoutPreset(
        "pet.dynamic", "Dynamic PET",
        match={"datatype": {"pet"}, "is_4d": True},
        view={"mode": "", "graph_visible": True, "graph": {"x_axis": "auto"}},
    ),
    LayoutPreset("pet", "PET", match={"datatype": {"pet"}}, view={"mode": ""}),
    LayoutPreset("mri.dwi", "Diffusion", match={"datatype": {"dwi"}},
                 view={"mode": "multi", "graph_visible": False}),
    LayoutPreset("mri.perf", "Perfusion", match={"datatype": {"perf"}, "is_4d": True},
                 view={"mode": "multi", "graph_visible": True}),
    LayoutPreset("mri.fmap", "Field maps", match={"datatype": {"fmap"}},
                 view={"mode": "multi", "graph_visible": False}),
    LayoutPreset("mri.anat", "Anatomy", match={"datatype": {"anat"}},
                 view={"mode": "", "graph_visible": False}),
    LayoutPreset("volume.4d", "Other 4-D series", match={"is_4d": True},
                 view={"mode": "multi", "graph_visible": True}),
    LayoutPreset("volume", "Other images", view={"mode": "", "graph_visible": False}),
)

PRESET_BY_ID = {p.id: p for p in PRESETS}

#: The keys a layout remembers (the arrangement, not the look).
LAYOUT_KEYS = ("mode", "plane", "graph_visible", "graph", "layout", "mosaic",
               "mosaic_build")


def _matches(preset: LayoutPreset, datatype: str, suffix: str, is_4d: bool) -> bool:
    m = preset.match
    if "datatype" in m and datatype not in m["datatype"]:
        return False
    if "suffix" in m and suffix not in m["suffix"]:
        return False
    if "is_4d" in m and bool(m["is_4d"]) != bool(is_4d):
        return False
    return True


def match(datatype: str = "", suffix: str = "", is_4d: bool = False) -> LayoutPreset:
    """The preset for a file of this kind."""
    for preset in PRESETS:
        if _matches(preset, datatype, suffix, is_4d):
            return preset
    return PRESETS[-1]


def arrangement(state: Mapping[str, Any]) -> dict:
    """The part of a view state a layout remembers."""
    return {k: state[k] for k in LAYOUT_KEYS if k in state}


def opening_view(preset: LayoutPreset, remembered: Optional[Mapping[str, Any]],
                 default_mode: str) -> dict:
    """What a file of this kind opens in: the user's remembered arrangement,
    else the preset's, with an automatic mode resolved to ``default_mode``."""
    view = dict(preset.view)
    if remembered:
        view.update(arrangement(remembered))
    if not view.get("mode"):
        view["mode"] = default_mode
    return view


# ---------------------------------------------------------------------------
# Tiles: which views a mode shows, and where each goes
# ---------------------------------------------------------------------------

#: A tile is a plane or the 3-D render.
RENDER = "render"

Rect = tuple[float, float, float, float]


def tiles_for(mode: str, plane: str, layout, render_ok: bool) -> tuple[list[str], str]:
    """``(tiles, hero)``: the views ``mode`` shows, in order, and which of
    them is the large one ("" when none is). Mosaic is not tiled."""
    planes = list(dict.fromkeys(layout.planes)) or ["sagittal", "coronal", "axial"]
    if mode == "single":
        return [plane], ""
    if mode == "3d":
        return [RENDER] if render_ok else [plane], ""
    if mode == "combo":
        return planes + ([RENDER] if render_ok else []), ""
    if mode == "hero":
        hero = layout.hero or plane
        if hero == RENDER and not render_ok:
            hero = plane
        rest = [p for p in ("sagittal", "coronal", "axial") if p != hero]
        if render_ok and hero != RENDER:
            rest.append(RENDER)
        return [hero] + rest, hero
    return planes, ""


def _grid_shape(n: int, arrangement: str) -> tuple[int, int]:
    """(rows, columns) for ``n`` tiles in a fixed arrangement."""
    if n <= 1:
        return 1, 1
    if arrangement == "row":
        return 1, n
    if arrangement == "column":
        return n, 1
    cols = int(math.ceil(math.sqrt(n)))
    return int(math.ceil(n / cols)), cols


def _cells(n: int, rows: int, cols: int, x: float, y: float, w: float, h: float,
           gap: float) -> list[Rect]:
    cw = (w - gap * (cols - 1)) / cols
    ch = (h - gap * (rows - 1)) / rows
    return [(x + (k % cols) * (cw + gap), y + (k // cols) * (ch + gap), cw, ch)
            for k in range(n)]


def _fitted_area(cw: float, ch: float, aspect: float) -> float:
    """Area of an image of width/height ``aspect`` fitted in a cell."""
    if cw <= 0 or ch <= 0:
        return 0.0
    return min(cw * cw / aspect, ch * ch * aspect) if aspect > 0 else cw * ch


def choose_arrangement(n: int, width: float, height: float, aspect: float = 1.0,
                       gap: float = 2.0) -> str:
    """The arrangement that makes ``n`` images of shape ``aspect`` largest."""
    best, best_area = "row", -1.0
    for name in ("row", "column", "grid"):
        rows, cols = _grid_shape(n, name)
        cw = (width - gap * (cols - 1)) / cols
        ch = (height - gap * (rows - 1)) / rows
        area = _fitted_area(cw, ch, aspect)
        if area > best_area + 1e-9:
            best, best_area = name, area
    return best


def tile_rects(n: int, width: float, height: float, *, arrangement: str = "auto",
               aspect: float = 1.0, hero: bool = False, hero_fraction: float = 0.62,
               hero_side: str = "left", gap: float = 2.0) -> list[Rect]:
    """Rectangles ``(x, y, w, h)`` for ``n`` tiles in a ``width`` x
    ``height`` page. With ``hero`` the FIRST tile takes ``hero_fraction`` of
    the width (side ``left``) or height (``top``) and the rest share the
    remainder in a column or row."""
    if n <= 0:
        return []
    if hero and n > 1:
        f = min(max(float(hero_fraction), 0.2), 0.9)
        if hero_side == "top":
            big = (0.0, 0.0, width, height * f - gap / 2)
            rest = _cells(n - 1, 1, n - 1, 0.0, height * f + gap / 2, width,
                          height * (1 - f) - gap / 2, gap)
        else:
            big = (0.0, 0.0, width * f - gap / 2, height)
            rest = _cells(n - 1, n - 1, 1, width * f + gap / 2, 0.0,
                          width * (1 - f) - gap / 2, height, gap)
        return [big] + rest
    if arrangement == "auto":
        arrangement = choose_arrangement(n, width, height, aspect, gap)
    rows, cols = _grid_shape(n, arrangement)
    return _cells(n, rows, cols, 0.0, 0.0, width, height, gap)


__all__ = [
    "LAYOUT_KEYS", "LayoutPreset", "PRESETS", "PRESET_BY_ID", "RENDER", "arrangement",
    "choose_arrangement", "match", "opening_view", "tile_rects", "tiles_for",
]

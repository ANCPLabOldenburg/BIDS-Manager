"""Headless 2-D rendering: what a slice view shows, as pixels or a PNG.

The library's render surface, for a tool that wants a figure and no window
(``fsleyes render``, ``freeview -ss``, NiiVue's ``saveScene``):

    from bidsmgr.viz import render2d

    store = render2d.open_store("sub-01_T1w.nii.gz", overlays=["sub-01_dseg.nii.gz"])
    store.run("layer.set", colormap="gray", window=(0, 900))
    rgba = render2d.render(store, planes=("sagittal", "coronal", "axial"))
    render2d.write_png("sub-01.png", rgba)

or, in one call, ``render2d.render_file(path, "out.png", ...)``.

Qt-free and display-free: no Qt, no GL, no Pillow. :func:`slice_rgba` is
the SAME composition the slice canvas draws (layer order, colour maps,
windows, opacity, labels, the opaque base), so a scripted figure and the
screen cannot disagree; the canvas calls it too. Every scene command works
on the store first (a colour map, a window, a cursor, a frame), which is
also how a viewer's "show command line" reproduces a view.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from . import views
from .compute import geometry, intensity
from .scene import Cursor, Scene, SourceRef, VolumeDisplay, VolumeLayer
from .store import SceneStore

PLANES = ("sagittal", "coronal", "axial")
#: Pixels between two panels of one figure.
GAP_PX = 4
#: The crosshair's colour when one is drawn (the viewer's default).
CROSSHAIR = (79, 195, 247, 255)


@dataclass
class SliceImage:
    """One plane of a scene, as the canvas draws it before placement."""

    #: (rows, cols, 4) uint8, one pixel per grid pixel (not square in mm
    #: when the voxels are not).
    rgba: np.ndarray
    grid: geometry.SliceGrid
    #: Whether the base layer is drawn smooth (linear) or blocky (nearest).
    smooth: bool


# ---------------------------------------------------------------------------
# The composition (shared with the slice canvas)
# ---------------------------------------------------------------------------


def slice_rgba(store: SceneStore, plane: str, *, transparent: bool = False,
               grid: Optional[geometry.SliceGrid] = None) -> Optional[SliceImage]:
    """Every visible volume layer of ``store`` cut on ``plane``, coloured
    and composited bottom to top. None while the base layer's frame is not
    read yet (or there is no volume).

    A lone base layer is made opaque over black unless ``transparent``, so
    empty space is the canvas colour exactly as a composite would leave it.
    """
    grid = grid if grid is not None else views.grid(store, plane)
    if grid is None:
        return None
    rgba_layers = []
    smooth = True
    first = True
    for layer in store.scene.layers:
        if layer.kind != "volume" or not layer.visible:
            continue
        if views.is_shape(views.source_of(store, layer)):
            # Drawn as geometry on top, exactly (see compute.shapes).
            continue
        values = views.layer_values(store, layer, grid)
        if values is None:
            if first:
                return None
            continue
        window = layer.display.window
        if window is None:
            src = views.source_of(store, layer)
            window = (src.robust_range(views.frame_of(store, layer, src))
                      if src is not None else (0.0, 1.0))
        rgba_layers.append(intensity.colorize(values, layer.display, is_base=first,
                                              window=window))
        if first:
            smooth = layer.display.interpolation != "nearest"
        first = False
    if not rgba_layers:
        return None
    buf = intensity.composite(rgba_layers) if len(rgba_layers) > 1 else rgba_layers[0].copy()
    if len(rgba_layers) == 1 and not transparent:
        alpha = buf[..., 3:4].astype(np.float32) / 255.0
        buf[..., :3] = (buf[..., :3].astype(np.float32) * alpha + 0.5).astype(np.uint8)
        buf[..., 3] = 255
    return SliceImage(np.ascontiguousarray(buf), grid, smooth)


# ---------------------------------------------------------------------------
# Opening without a viewer
# ---------------------------------------------------------------------------


def open_store(path, *, overlays: Iterable = (), budget_bytes: Optional[int] = None,
               display: Optional[dict] = None) -> SceneStore:
    """A store holding ``path`` read whole, as the viewer opens it (cursor at
    the centre), with ``overlays`` over it, each in the look its contents
    call for (an atlas in labels, a statistical map in hot, an MRS file as
    its voxel). Blocking: call it on a worker from a GUI."""
    from .data.volume import open_volume
    from .overlays import open_overlay

    path = Path(path)
    src = open_volume(path)
    src.stream(budget_bytes=budget_bytes)
    store = SceneStore()
    store.sources = {"vol0": src}
    scene = Scene()
    scene.sources = {"vol0": SourceRef(id="vol0", path=str(path), kind="volume")}
    scene.layers = [VolumeLayer(id="base", source="vol0", name=path.name,
                                display=VolumeDisplay.model_validate(display or {}))]
    scene.cursor = Cursor(world=tuple(float(v) for v in src.center_world()))
    store.replace_scene(scene)
    for n, extra in enumerate(overlays, start=1):
        overlay = open_overlay(Path(extra), budget_bytes=budget_bytes)
        sid = f"ovl{n}"
        store.sources[sid] = overlay.source
        store.run("layer.add", id=f"overlay{n}", source=sid,
                  name=overlay.name or overlay.source.path.name,
                  display=overlay.display.model_dump(),
                  path="" if overlay.source.in_memory else str(overlay.source.path))
    return store


# ---------------------------------------------------------------------------
# Pixels
# ---------------------------------------------------------------------------


def _resample(rgba: np.ndarray, out_rows: int, out_cols: int, smooth: bool) -> np.ndarray:
    """``rgba`` stretched to ``(out_rows, out_cols)``, pixel centres kept."""
    rows, cols = rgba.shape[:2]
    r = (np.arange(out_rows) + 0.5) * rows / out_rows - 0.5
    c = (np.arange(out_cols) + 0.5) * cols / out_cols - 0.5
    if not smooth:
        ri = np.clip(np.round(r).astype(int), 0, rows - 1)
        ci = np.clip(np.round(c).astype(int), 0, cols - 1)
        return rgba[ri][:, ci]
    r = np.clip(r, 0, rows - 1)
    c = np.clip(c, 0, cols - 1)
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    r1 = np.minimum(r0 + 1, rows - 1)
    c1 = np.minimum(c0 + 1, cols - 1)
    fr = (r - r0)[:, None, None]
    fc = (c - c0)[None, :, None]
    a = rgba.astype(np.float32)
    top = a[r0][:, c0] * (1 - fc) + a[r0][:, c1] * fc
    bot = a[r1][:, c0] * (1 - fc) + a[r1][:, c1] * fc
    return np.clip(top * (1 - fr) + bot * fr + 0.5, 0, 255).astype(np.uint8)


def panel(store: SceneStore, plane: str, *, height_px: Optional[int] = None,
          px_per_mm: float = 2.0, crosshair: bool = False,
          transparent: bool = False) -> Optional[np.ndarray]:
    """One plane as square-millimetre pixels: ``height_px`` tall, or
    ``px_per_mm`` when no height is given. None while loading."""
    img = slice_rgba(store, plane, transparent=transparent)
    if img is None:
        return None
    w_mm, h_mm = img.grid.extent_mm
    if height_px is not None:
        px_per_mm = height_px / max(h_mm, 1e-9)
    out_rows = max(1, int(round(h_mm * px_per_mm)))
    out_cols = max(1, int(round(w_mm * px_per_mm)))
    out = _resample(img.rgba, out_rows, out_cols, img.smooth)
    out = draw_shapes(store, img.grid, out)
    world = store.scene.cursor.world
    if crosshair and world is not None:
        col, row, _d = img.grid.world_to_pixel(world)
        rows, cols = img.grid.shape
        if np.isfinite(col) and np.isfinite(row):
            x = int(round((col + 0.5) / cols * out_cols - 0.5))
            y = int(round((row + 0.5) / rows * out_rows - 0.5))
            out = out.copy()
            if 0 <= x < out_cols:
                out[:, x] = CROSSHAIR
            if 0 <= y < out_rows:
                out[y, :] = CROSSHAIR
    return out


#: How much of a shape's colour fills it (its rim is drawn at full opacity).
SHAPE_FILL = 0.16


def draw_shapes(store: SceneStore, grid: geometry.SliceGrid, out: np.ndarray) -> np.ndarray:
    """Every visible shape layer (the MRS voxel) over ``out``, which shows
    ``grid`` at any size: each pixel's world position is tested against the
    box, so the edge is exact at the output's own resolution."""
    found = views.shape_layers(store)
    if not found:
        return out
    from .compute.shapes import outline_mask

    out_rows, out_cols = out.shape[:2]
    rows, cols = grid.shape
    yy = (np.arange(out_rows) + 0.5) * rows / out_rows - 0.5
    xx = (np.arange(out_cols) + 0.5) * cols / out_cols - 0.5
    cc, rr = np.meshgrid(xx, yy)
    world = (grid.origin[None, None, :] + cc[..., None] * grid.col_vec[None, None, :]
             + rr[..., None] * grid.row_vec[None, None, :])
    out = out.astype(np.float32)
    for layer, src in found:
        inside = src.box.contains(world)
        if not inside.any():
            continue
        r, g, b, a = views.shape_colour(layer)
        colour = np.array([r, g, b], dtype=np.float32)
        alpha = a / 255.0
        width = max(1, int(round(layer.display.outline_px or 2.0)))
        rim = outline_mask(inside, width)
        fill = inside & ~rim
        for mask, k in ((fill, SHAPE_FILL * alpha), (rim, alpha)):
            if k > 0 and mask.any():
                out[mask, :3] = out[mask, :3] * (1.0 - k) + colour * k
                out[mask, 3] = np.maximum(out[mask, 3], 255.0 * k)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def render(store: SceneStore, *, planes: Sequence[str] = ("axial",),
           height_px: Optional[int] = None, px_per_mm: float = 2.0,
           crosshair: bool = False, transparent: bool = False,
           background: tuple[int, int, int, int] = (0, 0, 0, 255)) -> np.ndarray:
    """``planes`` side by side, one height, as (rows, cols, 4) uint8.

    Without ``height_px`` every panel is drawn at ``px_per_mm``, so the
    three planes of one image keep their true proportions. Raises
    ``ValueError`` for an unknown plane and ``RuntimeError`` when the base
    image has nothing to draw."""
    for plane in planes:
        if plane not in PLANES:
            raise ValueError(f"unknown plane {plane!r} (one of {', '.join(PLANES)})")
    panels = [panel(store, plane, height_px=height_px, px_per_mm=px_per_mm,
                    crosshair=crosshair, transparent=transparent) for plane in planes]
    if any(p is None for p in panels):
        raise RuntimeError("nothing to draw: the base image is not read")
    rows = max(p.shape[0] for p in panels)
    cols = sum(p.shape[1] for p in panels) + GAP_PX * (len(panels) - 1)
    fill = (0, 0, 0, 0) if transparent else background
    out = np.empty((rows, cols, 4), dtype=np.uint8)
    out[:] = fill
    x = 0
    for p in panels:
        y = (rows - p.shape[0]) // 2
        out[y:y + p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + GAP_PX
    return out


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def encode_png(rgba: np.ndarray) -> bytes:
    """An 8-bit RGBA PNG of ``rgba`` (rows, cols, 4), with nothing but zlib."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    rows, cols = rgba.shape[:2]
    raw = np.zeros((rows, cols * 4 + 1), dtype=np.uint8)
    raw[:, 1:] = rgba.reshape(rows, cols * 4)          # filter byte 0 per row

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", cols, rows, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))


def write_png(path, rgba: np.ndarray) -> Path:
    path = Path(path)
    path.write_bytes(encode_png(rgba))
    return path


def render_file(path, out, *, overlays: Iterable = (), planes: Sequence[str] = ("axial",),
                commands: Iterable[tuple[str, dict]] = (), height_px: Optional[int] = None,
                px_per_mm: float = 2.0, crosshair: bool = False,
                transparent: bool = False) -> Path:
    """Open ``path`` (and ``overlays``), run ``commands`` (``(id, params)``
    pairs, the viewer's own), render ``planes`` and write ``out`` as PNG."""
    store = open_store(path, overlays=overlays)
    for command_id, params in commands:
        store.run(command_id, **dict(params))
    rgba = render(store, planes=planes, height_px=height_px, px_per_mm=px_per_mm,
                  crosshair=crosshair, transparent=transparent)
    return write_png(out, rgba)


__all__ = ["PLANES", "SliceImage", "draw_shapes", "encode_png", "open_store", "panel",
           "render",
           "render_file", "slice_rgba", "write_png"]

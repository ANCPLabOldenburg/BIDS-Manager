"""Questions every volume view asks of the store, answered once.

Which volume defines the grid, where the cursor is in its voxels, what grid a
plane draws, what values a layer has on it, what to print in the readout.
Canvases and commands both call these, so the answer to "which slice is the
axial view showing" cannot differ between the thing drawing it and the thing
moving it.

Qt-free.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from .compute import geometry
from .scene import PLANE_AXIS, VolumeLayer

if TYPE_CHECKING:  # pragma: no cover
    from .data.volume import VolumeSource
    from .store import SceneStore


def source_of(store: "SceneStore", layer: Optional[VolumeLayer]) -> Optional["VolumeSource"]:
    if layer is None:
        return None
    return store.sources.get(layer.source)


def base(store: "SceneStore") -> tuple[Optional[VolumeLayer], Optional["VolumeSource"]]:
    """The base layer and its source (the volume whose grid views cut)."""
    layer = store.scene.base_layer()
    return layer, source_of(store, layer)


def orientation(src: "VolumeSource") -> geometry.Orientation:
    cached = src._range_cache.get("_ornt")
    if cached is None:
        cached = geometry.orientation_of(src.affine)
        src._range_cache["_ornt"] = cached
    return cached


def cursor_world(store: "SceneStore") -> Optional[np.ndarray]:
    world = store.scene.cursor.world
    if world is not None:
        return np.asarray(world, dtype=float)
    _layer, src = base(store)
    return None if src is None else src.center_world()


def cursor_voxel(store: "SceneStore") -> Optional[tuple[int, int, int]]:
    """The base volume's voxel under the cursor (clamped into the volume)."""
    _layer, src = base(store)
    world = cursor_world(store)
    if src is None or world is None:
        return None
    return src.clamp_voxel(src.world_to_voxel(world))


def grid(store: "SceneStore", plane: str) -> Optional[geometry.SliceGrid]:
    _layer, src = base(store)
    world = cursor_world(store)
    if src is None or world is None:
        return None
    d = store.scene.display
    return geometry.grid_for(
        plane, affine=src.affine, spatial=src.spatial, zooms=src.zooms3,
        orientation=orientation(src), cursor_world=world, space=d.space,
        ras=d.ras, radiological=d.radiological,
    )


def _is_series(src) -> bool:
    return src is not None and bool(src.is_4d) and not bool(src.is_rgb)


def series_layers(store: "SceneStore") -> list[VolumeLayer]:
    """Every layer that is a 4-D series, top-most first."""
    out = []
    for layer in reversed(store.scene.layers):
        if layer.kind == "volume" and _is_series(source_of(store, layer)):
            out.append(layer)
    return out


def series_layer(store: "SceneStore") -> tuple[Optional[VolumeLayer], Optional["VolumeSource"]]:
    """The 4-D layer the graph and the volume controls follow: the one the
    graph names, else the base image when it is a series, else the
    top-most visible 4-D overlay (a BOLD drawn over a T1)."""
    scene = store.scene
    if scene.graph.layer:
        layer = scene.layer(scene.graph.layer)
        src = source_of(store, layer)
        if layer is not None and _is_series(src):
            return layer, src
    base_layer, base_src = base(store)
    if _is_series(base_src):
        return base_layer, base_src
    for layer in series_layers(store):
        if layer.visible:
            return layer, source_of(store, layer)
    return None, None


def is_shape(src) -> bool:
    """Whether a source is drawn as geometry (the MRS voxel's box)."""
    return getattr(src, "box", None) is not None


def shape_layers(store: "SceneStore", *, in_3d: bool = False) -> list[tuple[VolumeLayer, "VolumeSource"]]:
    """The visible layers that are shapes, bottom to top, with their
    sources; ``in_3d`` keeps only those the render should draw."""
    out = []
    for layer in store.scene.layers:
        if layer.kind != "volume" or not layer.visible or (in_3d and not layer.in_3d):
            continue
        src = source_of(store, layer)
        if is_shape(src):
            out.append((layer, src))
    return out


def shape_colour(layer: VolumeLayer) -> tuple[int, int, int, int]:
    """The colour a shape layer is drawn in: its colour map at the top of
    its window, with its opacity."""
    from .compute import intensity

    d = layer.display
    window = d.window or (0.0, 1.0)
    rgba = intensity.colorize(np.array([[float(window[1])]], dtype=np.float32),
                              d.model_copy(update={"threshold_mode": "range"}),
                              is_base=False, window=tuple(window))
    r, g, b = (int(v) for v in rgba[0, 0, :3])
    return r, g, b, int(round(255 * float(np.clip(d.opacity, 0.0, 1.0))))


def frame_of(store: "SceneStore", layer: VolumeLayer, src: "VolumeSource") -> int:
    return max(0, min(int(layer.frame), src.n_frames - 1))


def layer_values(
    store: "SceneStore", layer: VolumeLayer, grid_: geometry.SliceGrid,
) -> Optional[np.ndarray]:
    """``layer``'s values on ``grid_`` (rows, cols[, C]); None while loading.

    The base layer on its own voxel grid is a pure array slice, scaled only
    where it is drawn. Anything else (an overlay, or the world-aligned grid)
    is resampled at the world positions of the grid's pixels.
    """
    src = source_of(store, layer)
    if src is None:
        return None
    t = frame_of(store, layer, src)
    raw = src.raw_frame(t)
    if raw is None:
        return None
    base_layer, _ = base(store)
    if grid_.kind == "voxel" and base_layer is not None and layer.id == base_layer.id:
        return src.scale(geometry.slice_voxel_grid(raw, grid_))
    order = 0 if (layer.display.interpolation == "nearest"
                  or layer.display.label_table is not None) else 1
    return geometry.sample_grid(src.scale(raw), src.inv_affine, grid_, order=order)


def voxel_in(src: "VolumeSource", world) -> Optional[tuple[int, int, int]]:
    """The voxel of ``src`` at a world point, or None when outside it."""
    vox = src.world_to_voxel(world)
    idx = np.round(vox).astype(int)
    if np.any(idx < 0) or np.any(idx >= np.asarray(src.spatial)):
        return None
    return int(idx[0]), int(idx[1]), int(idx[2])


def readout(store: "SceneStore") -> str:
    """"x y z mm | voxel (i, j, k) = value" for the base layer, plus one
    value per visible overlay. The mm is the scanner's; the indices are the
    FILE's own, not a reoriented copy's, so they match any other tool."""
    world = cursor_world(store)
    layer, src = base(store)
    if world is None or src is None or layer is None:
        return ""
    parts = [f"{world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f} mm"]
    for lay in store.scene.layers:
        if lay.kind != "volume" or not lay.visible:
            continue
        s = source_of(store, lay)
        if s is None:
            continue
        vox = voxel_in(s, world)
        if vox is None:
            continue
        value = s.value_at(vox, frame_of(store, lay, s))
        if value is None:
            text = "..."
        elif s.is_rgb:
            text = "[" + ", ".join(f"{v:.3g}" for v in np.atleast_1d(value)) + "]"
        else:
            label = None
            table = lay.display.label_table
            if table is not None:
                label = table.labels.get(int(round(value)))
            text = f"{value:.4g}" + (f" {label}" if label else "")
        name = "voxel" if lay is layer else (lay.name or "overlay")
        # A computed source's indices are its own array's, not a file's.
        where = "" if s.in_memory else f" ({vox[0]}, {vox[1]}, {vox[2]})"
        parts.append(f"{name}{where} = {text}")
    return "  |  ".join(parts)


def slice_count(store: "SceneStore", plane: str) -> int:
    """How many positions a plane can step through (base voxel grid)."""
    _layer, src = base(store)
    if src is None:
        return 0
    ornt = orientation(src)
    return src.spatial[ornt.data_of[PLANE_AXIS[plane]]]


def slice_index(store: "SceneStore", plane: str) -> int:
    vox = cursor_voxel(store)
    _layer, src = base(store)
    if vox is None or src is None:
        return 0
    return vox[orientation(src).data_of[PLANE_AXIS[plane]]]


__all__ = [
    "base", "cursor_voxel", "cursor_world", "frame_of", "grid", "layer_values",
    "orientation", "readout", "series_layer", "series_layers", "slice_count",
    "slice_index", "source_of", "voxel_in",
]

"""Overlays for the 3-D render: the scene's layers as ONE colour volume.

The 2-D views colour each layer with :func:`intensity.colorize` and blend
them with :func:`intensity.composite`. The 3-D render must show the same
thing, so this module does the same two steps on the whole volume: every
visible overlay is resampled onto the BASE image's render box (the texture's
space, so the shader needs no second geometry), coloured with the layer's own
window, colour map, opacity, threshold mode and label table, and blended in
layer order into one RGBA volume. The shader samples that volume at every ray
step; what is drawn in 3-D is therefore what the slices draw, by
construction.

Resolution follows the data, not the base: a 3 mm BOLD over a 1 mm T1 is
resampled at ~3 mm (its own detail), capped at :data:`MAX_DIM` per axis, so
an atlas never costs a 16 M-voxel upload. Outlines are a 2-D idea (the edge
of a region on a slice); in 3-D the region is drawn filled.

Runs on a worker. Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import geometry, intensity
from ..scene import VolumeDisplay

#: Longest axis of the overlay volume, in voxels. 192^3 RGBA is 28 MB.
MAX_DIM = 192


@dataclass(frozen=True)
class OverlayInput:
    """One overlay as the worker needs it: its data, where it is, its look."""

    key: tuple                      # identifies the resampled values (cache)
    values: np.ndarray              # (X, Y, Z[, C]) raw, as stored
    inv_affine: np.ndarray          # world -> this overlay's voxel
    zooms: tuple[float, float, float]
    display: VolumeDisplay
    window: tuple[float, float]
    #: Raw to data units, applied AFTER the resample (a linear map commutes
    #: with interpolation, and the resampled array is the smaller one).
    slope: float = 1.0
    inter: float = 0.0


def canonical_affine(affine: np.ndarray, spatial: tuple[int, int, int],
                     ornt: geometry.Orientation) -> np.ndarray:
    """The affine of the RAS-ordered, RAS-directed copy of a volume.

    :func:`bidsmgr.gui.viz.canvases.render.canonical_volume` transposes a
    frame to RAS order and flips the reversed axes; this is the affine of
    THAT array, so canonical voxel ``c`` lands where the file's voxel it came
    from lands.
    """
    a = np.asarray(affine, dtype=float)
    perm = np.zeros((4, 4), dtype=float)
    perm[3, 3] = 1.0
    for ras in range(3):
        d = ornt.data_of[ras]
        if ornt.sign[d] < 0:
            perm[d, ras] = -1.0
            perm[d, 3] = float(spatial[d] - 1)
        else:
            perm[d, ras] = 1.0
    return a @ perm


def overlay_dims(base_dims: tuple[int, int, int], base_spacing: tuple[float, float, float],
                 zooms: tuple[float, float, float], cap: int = MAX_DIM) -> tuple[int, int, int]:
    """Voxels per axis for an overlay texture over the base's box: the
    overlay's own resolution, never finer than the base, never over ``cap``."""
    out = []
    finest = min(float(z) for z in zooms if z > 0) if any(z > 0 for z in zooms) else 1.0
    for n, sp in zip(base_dims, base_spacing):
        extent = float(n) * float(sp)
        wanted = int(np.ceil(extent / max(finest, 1e-6)))
        out.append(int(max(1, min(n, wanted, cap))))
    return out[0], out[1], out[2]


def texture_to_world(canon_affine: np.ndarray, base_dims, dims) -> np.ndarray:
    """Affine from an overlay-texture index (x, y, z) to world mm.

    Texture voxel ``j`` of ``M`` spans the same box as base voxels; its
    centre sits at canonical base index ``(j + 0.5) / M * N - 0.5``.
    """
    m = np.eye(4, dtype=float)
    for ax in range(3):
        n, M = float(base_dims[ax]), float(dims[ax])
        m[ax, ax] = n / M
        m[ax, 3] = 0.5 * n / M - 0.5
    return np.asarray(canon_affine, dtype=float) @ m


def resample_to_box(values: np.ndarray, inv_affine: np.ndarray, tex_to_world: np.ndarray,
                    dims: tuple[int, int, int], *, order: int = 1) -> np.ndarray:
    """``values`` (X, Y, Z[, C]) sampled on the overlay texture grid, NaN
    outside the overlay. One ``affine_transform`` call per channel (C code),
    no coordinate arrays."""
    from scipy.ndimage import affine_transform

    tex_to_vox = np.asarray(inv_affine, dtype=float) @ np.asarray(tex_to_world, dtype=float)
    matrix = tex_to_vox[:3, :3]
    offset = tex_to_vox[:3, 3]

    def one(chan: np.ndarray) -> np.ndarray:
        return affine_transform(
            np.asarray(chan, dtype=np.float32), matrix, offset=offset,
            output_shape=tuple(int(d) for d in dims), order=order,
            mode="constant", cval=np.nan, prefilter=False,
        )

    if values.ndim == 3:
        return one(values)
    return np.stack([one(values[..., c]) for c in range(values.shape[-1])], axis=-1)


def colorize_volume(values: np.ndarray, display: VolumeDisplay,
                    window: tuple[float, float]) -> np.ndarray:
    """``intensity.colorize`` over a whole (X, Y, Z[, C]) volume -> RGBA
    (X, Y, Z, 4). The colouring is elementwise, so the volume is coloured as
    one tall image; only the outline (a slice's edge) is left out."""
    x, y, z = values.shape[:3]
    flat = values.reshape((x, y * z) + values.shape[3:])
    look = display if display.outline_px == 0 else display.model_copy(update={"outline_px": 0.0})
    rgba = intensity.colorize(flat, look, is_base=False, window=window)
    return rgba.reshape(x, y, z, 4)


def composite_volumes(layers: list[np.ndarray]) -> np.ndarray:
    """Blend RGBA volumes bottom to top, keeping the alpha (the shader
    composites the result over the lit base), as ``intensity.composite``
    does for a slice but without the opaque canvas under it."""
    if not layers:
        raise ValueError("nothing to composite")
    rgb = np.zeros(layers[0].shape[:3] + (3,), dtype=np.float32)
    alpha = np.zeros(layers[0].shape[:3], dtype=np.float32)
    for rgba in layers:
        a = rgba[..., 3].astype(np.float32) / 255.0
        rgb = rgb * (1.0 - a)[..., None] + rgba[..., :3].astype(np.float32) * a[..., None]
        alpha = alpha + a * (1.0 - alpha)
    out = np.empty(layers[0].shape[:3] + (4,), dtype=np.uint8)
    # Straight (not premultiplied) colour: divide the blend by its coverage.
    safe = np.maximum(alpha, 1e-6)[..., None]
    out[..., :3] = np.clip(rgb / safe + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def build_overlay_volume(
    canon_affine: np.ndarray, base_dims: tuple[int, int, int],
    base_spacing: tuple[float, float, float], overlays: list[OverlayInput],
    *, cache: Optional[dict] = None, cancel=None, cap: int = MAX_DIM,
) -> tuple[Optional[np.ndarray], tuple[int, int, int], dict]:
    """The overlays blended into one RGBA volume over the base's box.

    Returns ``(rgba (X, Y, Z, 4) or None, dims, cache)``. ``cache`` maps an
    overlay's ``key`` plus the dims to its resampled values, so changing an
    opacity re-colours without re-resampling (the costly half).
    """
    if not overlays:
        return None, (1, 1, 1), {}
    finest = min((min(o.zooms) for o in overlays), default=1.0)
    dims = overlay_dims(base_dims, base_spacing, (finest, finest, finest), cap)
    to_world = texture_to_world(canon_affine, base_dims, dims)
    kept: dict = {}
    coloured = []
    for ov in overlays:
        if cancel is not None and cancel.cancelled:
            return None, dims, kept
        order = 0 if (ov.display.interpolation == "nearest"
                      or ov.display.label_table is not None) else 1
        ckey = (ov.key, dims, order)
        values = (cache or {}).get(ckey)
        if values is None:
            values = resample_to_box(ov.values, ov.inv_affine, to_world, dims, order=order)
            if ov.slope != 1.0 or ov.inter != 0.0:
                values = values * np.float32(ov.slope) + np.float32(ov.inter)
        kept[ckey] = values
        coloured.append(colorize_volume(values, ov.display, ov.window))
    return composite_volumes(coloured), dims, kept


__all__ = [
    "MAX_DIM", "OverlayInput", "build_overlay_volume", "canonical_affine",
    "colorize_volume", "composite_volumes", "overlay_dims", "resample_to_box",
    "texture_to_world",
]

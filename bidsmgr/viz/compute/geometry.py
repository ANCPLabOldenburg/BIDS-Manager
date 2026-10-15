"""Where every pixel of a 2-D view is in the world, and back.

One idea carries the whole 2-D viewer: a :class:`SliceGrid` maps the pixels
of a view to millimetres in the scanner. Everything follows from it.

* Clicking: pixel to world. Drawing the crosshair: world to pixel.
* The aspect ratio: a pixel is ``|col_vec|`` mm wide and ``|row_vec|`` mm
  tall, so a 1 x 1 x 5 mm scan is drawn five times taller per voxel without
  anyone having to remember to.
* Overlays: any image is sampled at the world positions of the base image's
  pixels, whatever its own grid, resolution or storage order.
* Oblique scans: in ``world`` space the grid is aligned to the scanner's axes
  and the image is resampled onto it, so a tilted acquisition draws upright.

Two kinds of grid exist.

``voxel`` grids are the base image's own voxel planes, turned to the nearest
anatomical axes (the old ``as_closest_canonical`` view, without copying the
volume). Exact, and a pure array slice, so drawing them is as fast as it gets.

``world`` grids step along the scanner's x, y, z at the base image's finest
voxel size and are resampled.

Screen conventions (one place, used everywhere): the horizontal grows to the
right toward the plane's first RAS axis (neurological: subject's left on the
image's left); the vertical grows UP toward its second axis; rows count DOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..scene import PLANE_AXIS, PLANE_HV

#: Letters at the negative and positive end of each RAS axis.
AXIS_LETTERS = (("L", "R"), ("P", "A"), ("I", "S"))


@dataclass(frozen=True)
class Orientation:
    """How a volume's storage axes relate to RAS.

    ``ras_of[d]`` is the RAS axis (0 x, 1 y, 2 z) that data axis ``d`` runs
    along, and ``sign[d]`` is +1 when the index grows toward the positive end
    (R, A, S), -1 when it grows toward L, P or I. ``data_of[r]`` inverts it.
    """

    ras_of: tuple[int, int, int]
    sign: tuple[int, int, int]
    data_of: tuple[int, int, int]


def orientation_of(affine: np.ndarray) -> Orientation:
    """The closest anatomical orientation of an affine (nibabel's rule)."""
    import nibabel as nib

    ornt = nib.orientations.io_orientation(np.asarray(affine, dtype=float))
    ras_of: list[Optional[int]] = [None, None, None]
    sign = [1, 1, 1]
    for d, (ras, flip) in enumerate(ornt):
        if not np.isnan(ras):
            ras_of[d] = int(ras)
            sign[d] = 1 if flip > 0 else -1
    # A degenerate axis (a 2-D image's padding) gets whichever RAS axis is
    # still free, so the mapping stays a permutation.
    free = [r for r in range(3) if r not in ras_of]
    for d in range(3):
        if ras_of[d] is None:
            ras_of[d] = free.pop(0)
    data_of = [0, 0, 0]
    for d, r in enumerate(ras_of):
        data_of[int(r)] = d
    return Orientation(
        (int(ras_of[0]), int(ras_of[1]), int(ras_of[2])),
        (sign[0], sign[1], sign[2]),
        (data_of[0], data_of[1], data_of[2]),
    )


@dataclass(frozen=True)
class SliceGrid:
    """The pixels of one 2-D view, placed in the world.

    Pixel (row r, column c) has its CENTRE at
    ``origin + c * col_vec + r * row_vec`` (millimetres). ``normal`` points
    out of the plane; ``depth`` is the world coordinate of the plane along it.
    """

    plane: str
    shape: tuple[int, int]            # (rows, cols)
    origin: np.ndarray
    col_vec: np.ndarray
    row_vec: np.ndarray
    normal: np.ndarray
    #: ``voxel`` grids carry how to slice the base volume directly.
    kind: str = "voxel"
    #: For voxel grids: the data axis held fixed, its index, and how the two
    #: remaining data axes map to columns and rows (with reversal flags).
    slice_axis: int = 0
    slice_index: int = 0
    col_axis: int = 0
    row_axis: int = 0
    col_reversed: bool = False
    row_reversed: bool = False

    # -- sizes ----------------------------------------------------------
    @property
    def pixel_mm(self) -> tuple[float, float]:
        """(width, height) of one pixel in millimetres."""
        return float(np.linalg.norm(self.col_vec)), float(np.linalg.norm(self.row_vec))

    @property
    def extent_mm(self) -> tuple[float, float]:
        w, h = self.pixel_mm
        return self.shape[1] * w, self.shape[0] * h

    # -- mapping --------------------------------------------------------
    def pixel_to_world(self, col: float, row: float) -> np.ndarray:
        return self.origin + col * self.col_vec + row * self.row_vec

    def world_to_pixel(self, world) -> tuple[float, float, float]:
        """(col, row, distance-from-plane in mm) of a world point."""
        m = np.column_stack([self.col_vec, self.row_vec, self.normal])
        try:
            c, r, d = np.linalg.solve(m, np.asarray(world, dtype=float)[:3] - self.origin)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), float("nan")
        return float(c), float(r), float(d)

    def world_points(self) -> np.ndarray:
        """World position of every pixel centre, shape (rows, cols, 3)."""
        rows, cols = self.shape
        c = np.arange(cols, dtype=float)
        r = np.arange(rows, dtype=float)
        cc, rr = np.meshgrid(c, r)
        return (
            self.origin[None, None, :]
            + cc[..., None] * self.col_vec[None, None, :]
            + rr[..., None] * self.row_vec[None, None, :]
        )

    def key(self) -> tuple:
        """Hashable identity, for caching what was drawn on this grid."""
        return (
            self.plane, self.kind, self.shape, self.slice_axis, self.slice_index,
            self.col_reversed, self.row_reversed,
            tuple(np.round(self.origin, 4)), tuple(np.round(self.col_vec, 5)),
            tuple(np.round(self.row_vec, 5)),
        )


def screen_flips(plane: str, orientation: Orientation, *, ras: bool,
                 radiological: bool) -> tuple[bool, bool]:
    """Whether the drawn slice is mirrored horizontally / vertically.

    Relative to the canonical neurological view. ``ras=False`` keeps a file's
    stored-reversed axes reversed (the old "RAS off"); radiological mirrors
    left and right on the two planes that show them.
    """
    h_ras, v_ras = PLANE_HV[plane]
    flip_h = False
    flip_v = False
    if not ras:
        if orientation.sign[orientation.data_of[h_ras]] < 0:
            flip_h = True
        if orientation.sign[orientation.data_of[v_ras]] < 0:
            flip_v = True
    if radiological and h_ras == 0:
        flip_h = not flip_h
    return flip_h, flip_v


def voxel_grid(
    plane: str,
    affine: np.ndarray,
    spatial: tuple[int, int, int],
    orientation: Orientation,
    slice_index: int,
    *,
    flip_h: bool = False,
    flip_v: bool = False,
) -> SliceGrid:
    """The base image's own voxel plane, turned to the nearest anatomy."""
    affine = np.asarray(affine, dtype=float)
    ras_axis = PLANE_AXIS[plane]
    h_ras, v_ras = PLANE_HV[plane]
    s_axis = orientation.data_of[ras_axis]
    c_axis = orientation.data_of[h_ras]
    r_axis = orientation.data_of[v_ras]
    n_cols = spatial[c_axis]
    n_rows = spatial[r_axis]
    # Columns grow toward +h on screen: reverse the data axis if it grows
    # toward -h. Rows grow DOWN, i.e. toward -v: reverse if it grows toward +v.
    col_rev = orientation.sign[c_axis] < 0
    row_rev = orientation.sign[r_axis] > 0
    if flip_h:
        col_rev = not col_rev
    if flip_v:
        row_rev = not row_rev
    s = max(0, min(int(slice_index), spatial[s_axis] - 1))

    idx0 = [0.0, 0.0, 0.0]
    idx0[s_axis] = s
    idx0[c_axis] = (n_cols - 1) if col_rev else 0
    idx0[r_axis] = (n_rows - 1) if row_rev else 0
    origin = affine[:3, :3] @ np.array(idx0) + affine[:3, 3]
    col_vec = affine[:3, c_axis] * (-1.0 if col_rev else 1.0)
    row_vec = affine[:3, r_axis] * (-1.0 if row_rev else 1.0)
    normal = np.cross(col_vec, row_vec)
    n = np.linalg.norm(normal)
    normal = normal / n if n > 0 else np.array([0.0, 0.0, 1.0])
    return SliceGrid(
        plane=plane, shape=(n_rows, n_cols), origin=origin, col_vec=col_vec,
        row_vec=row_vec, normal=normal, kind="voxel", slice_axis=s_axis,
        slice_index=s, col_axis=c_axis, row_axis=r_axis,
        col_reversed=col_rev, row_reversed=row_rev,
    )


def world_bounds(affine: np.ndarray, spatial: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The world bounding box of a volume's voxel CENTRES."""
    corners = np.array(
        [[i, j, k] for i in (0, spatial[0] - 1) for j in (0, spatial[1] - 1)
         for k in (0, spatial[2] - 1)],
        dtype=float,
    )
    world = corners @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def world_grid(
    plane: str,
    affine: np.ndarray,
    spatial: tuple[int, int, int],
    zooms: tuple[float, float, float],
    depth_mm: float,
    *,
    flip_h: bool = False,
    flip_v: bool = False,
    max_pixels: int = 1024,
) -> SliceGrid:
    """A plane of the scanner's own axes covering the volume, resampled."""
    lo, hi = world_bounds(affine, spatial)
    step = max(min(z for z in zooms if z > 0), 1e-3)
    ras_axis = PLANE_AXIS[plane]
    h_ras, v_ras = PLANE_HV[plane]
    span_h = hi[h_ras] - lo[h_ras]
    span_v = hi[v_ras] - lo[v_ras]
    # Cap the resolution: a 0.2 mm in-plane scan of a whole head would ask
    # for a few thousand pixels per side, which no screen shows.
    step = max(step, max(span_h, span_v) / max_pixels)
    n_cols = max(1, int(np.floor(span_h / step)) + 1)
    n_rows = max(1, int(np.floor(span_v / step)) + 1)
    col_vec = np.zeros(3)
    row_vec = np.zeros(3)
    col_vec[h_ras] = step * (-1.0 if flip_h else 1.0)
    row_vec[v_ras] = -step * (-1.0 if flip_v else 1.0)
    origin = np.zeros(3)
    origin[h_ras] = hi[h_ras] if flip_h else lo[h_ras]
    origin[v_ras] = lo[v_ras] if flip_v else hi[v_ras]
    origin[ras_axis] = depth_mm
    normal = np.zeros(3)
    normal[ras_axis] = 1.0
    return SliceGrid(
        plane=plane, shape=(n_rows, n_cols), origin=origin, col_vec=col_vec,
        row_vec=row_vec, normal=normal, kind="world",
    )


def sample_grid(
    frame: np.ndarray,
    inv_affine: np.ndarray,
    grid: SliceGrid,
    *,
    order: int = 1,
    cval: float = np.nan,
) -> np.ndarray:
    """Values of ``frame`` at every pixel centre of ``grid`` (rows, cols).

    ``frame`` is (X, Y, Z) or (X, Y, Z, C); pixels outside the volume are
    ``cval`` (NaN marks "no data here", which the colouring draws as empty).
    """
    from scipy.ndimage import map_coordinates

    pts = grid.world_points().reshape(-1, 3)
    vox = pts @ np.asarray(inv_affine)[:3, :3].T + np.asarray(inv_affine)[:3, 3]
    coords = vox.T
    rows, cols = grid.shape
    if frame.ndim == 3:
        out = map_coordinates(
            np.asarray(frame, dtype=np.float32), coords, order=order,
            mode="constant", cval=cval, prefilter=False,
        )
        return out.reshape(rows, cols)
    chans = []
    for c in range(frame.shape[3]):
        chans.append(
            map_coordinates(
                np.asarray(frame[..., c], dtype=np.float32), coords, order=order,
                mode="constant", cval=cval, prefilter=False,
            ).reshape(rows, cols)
        )
    return np.stack(chans, axis=-1)


def slice_voxel_grid(frame: np.ndarray, grid: SliceGrid) -> np.ndarray:
    """The base image's own plane for a ``voxel`` grid: a pure array slice."""
    sl = [slice(None)] * 3
    sl[grid.slice_axis] = grid.slice_index
    plane2d = frame[tuple(sl)]
    # Remaining axes keep their order; put rows first, columns second.
    remaining = [a for a in range(3) if a != grid.slice_axis]
    if remaining.index(grid.row_axis) != 0:
        plane2d = np.swapaxes(plane2d, 0, 1)
    if grid.col_reversed:
        plane2d = plane2d[:, ::-1]
    if grid.row_reversed:
        plane2d = plane2d[::-1, :]
    return plane2d


def plane_labels(plane: str, grid: SliceGrid) -> dict[str, str]:
    """Anatomical letters at the four edges of a view, read off the grid.

    Derived from the vectors themselves rather than a table, so a flip, the
    native storage order or a radiological view can never leave a letter
    pointing the wrong way.
    """
    def letter(vec: np.ndarray) -> str:
        axis = int(np.argmax(np.abs(vec)))
        return AXIS_LETTERS[axis][1 if vec[axis] > 0 else 0]

    right = letter(grid.col_vec)
    left = letter(-grid.col_vec)
    bottom = letter(grid.row_vec)
    top = letter(-grid.row_vec)
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def grid_for(
    plane: str,
    *,
    affine: np.ndarray,
    spatial: tuple[int, int, int],
    zooms: tuple[float, float, float],
    orientation: Orientation,
    cursor_world,
    space: str,
    ras: bool,
    radiological: bool,
) -> SliceGrid:
    """The grid a view should draw for the cursor's current position."""
    flip_h, flip_v = screen_flips(plane, orientation, ras=ras, radiological=radiological)
    if space == "world":
        depth = float(np.asarray(cursor_world)[PLANE_AXIS[plane]])
        if not ras:
            # "Native" order only means something on the image's own grid.
            flip_h, flip_v = screen_flips(plane, orientation, ras=True,
                                          radiological=radiological)
        return world_grid(plane, affine, spatial, zooms, depth,
                          flip_h=flip_h, flip_v=flip_v)
    inv = np.linalg.inv(np.asarray(affine, dtype=float))
    vox = inv[:3, :3] @ np.asarray(cursor_world, dtype=float)[:3] + inv[:3, 3]
    s_axis = orientation.data_of[PLANE_AXIS[plane]]
    return voxel_grid(plane, affine, spatial, orientation,
                      int(round(float(vox[s_axis]))), flip_h=flip_h, flip_v=flip_v)


def snap_to_voxel(affine: np.ndarray, spatial, world) -> np.ndarray:
    """The world position of the voxel centre nearest ``world``."""
    affine = np.asarray(affine, dtype=float)
    inv = np.linalg.inv(affine)
    vox = inv[:3, :3] @ np.asarray(world, dtype=float)[:3] + inv[:3, 3]
    idx = np.clip(np.round(vox), 0, np.asarray(spatial) - 1)
    return affine[:3, :3] @ idx + affine[:3, 3]


def obliquity_deg(affine: np.ndarray) -> float:
    """How far the voxel axes are from the scanner's (0 = not oblique)."""
    a = np.asarray(affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(a, axis=0)
    norms[norms == 0] = 1.0
    cosines = np.max(np.abs(a / norms), axis=0)
    return float(np.degrees(np.arccos(np.clip(cosines.min(), -1.0, 1.0))))


def shear_deg(affine: np.ndarray) -> float:
    """The largest departure from right angles between voxel axes."""
    a = np.asarray(affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(a, axis=0)
    norms[norms == 0] = 1.0
    u = a / norms
    worst = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            ang = np.degrees(np.arccos(np.clip(abs(float(u[:, i] @ u[:, j])), 0, 1)))
            worst = max(worst, 90.0 - ang)
    return float(worst)


__all__ = [
    "AXIS_LETTERS", "Orientation", "SliceGrid", "grid_for", "obliquity_deg",
    "orientation_of", "plane_labels", "sample_grid", "screen_flips",
    "shear_deg", "slice_voxel_grid", "snap_to_voxel", "voxel_grid",
    "world_bounds", "world_grid",
]


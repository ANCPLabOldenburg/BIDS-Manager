"""Shapes that are drawn as geometry, not as pixels: a voxel grid's box.

Where a spectrum was measured is a cuboid in the scanner: the MRS file's
voxel (or the whole grid of an MRSI acquisition), placed by its affine at
whatever angle it was planned. Drawn as an image, the box can only be as
precise as the pixels it is sampled onto: on a 1 x 2 mm anatomy a 20 mm
voxel came out 22 mm tall with stepped edges, and in 3-D, sampled at its own
20 mm resolution, it was a blur up to 10 mm off. So it is kept as the box it
is, and every view computes it exactly:

* a 2-D slice draws :func:`section`, the polygon the slice plane cuts out
  of the box, in screen coordinates;
* the 3-D ray caster intersects each ray with the box analytically
  (:meth:`GridBox.unit_from_world` maps the scene into the box's own
  coordinates, where it is the unit cube);
* the headless renderer tests every output pixel with :meth:`GridBox.contains`.

Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The twelve edges of a box, as pairs of corner indices (corner ``k`` has
#: bit 0, 1, 2 set for the far end along axis 0, 1, 2).
EDGES = ((0, 1), (2, 3), (4, 5), (6, 7),       # along axis 0
         (0, 2), (1, 3), (4, 6), (5, 7),       # along axis 1
         (0, 4), (1, 5), (2, 6), (3, 7))       # along axis 2


@dataclass(frozen=True)
class GridBox:
    """The outer box of a voxel grid: ``shape`` voxels whose CENTRES the
    4 x 4 ``affine`` places in the world (millimetres, RAS)."""

    affine: np.ndarray
    shape: tuple[int, int, int]

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    def corners(self) -> np.ndarray:
        """(8, 3) world corners: the outer FACES of the grid, half a voxel
        beyond its first and last centres."""
        n = np.asarray(self.shape, dtype=float)
        out = np.empty((8, 3))
        for k in range(8):
            idx = np.array([(n[a] - 0.5) if (k >> a) & 1 else -0.5 for a in range(3)])
            out[k] = (self.affine @ np.append(idx, 1.0))[:3]
        return out

    def centre(self) -> np.ndarray:
        idx = (np.asarray(self.shape, dtype=float) - 1.0) / 2.0
        return (self.affine @ np.append(idx, 1.0))[:3]

    def size_mm(self) -> np.ndarray:
        """Edge lengths in millimetres along the grid's three axes."""
        cols = np.linalg.norm(self.affine[:3, :3], axis=0)
        return cols * np.asarray(self.shape, dtype=float)

    def unit_from_world(self) -> np.ndarray:
        """4 x 4 world -> box coordinates, where the box is [0, 1]^3."""
        n = np.asarray(self.shape, dtype=float)
        to_unit = np.eye(4)
        to_unit[:3, :3] = np.diag(1.0 / n)
        to_unit[:3, 3] = 0.5 / n
        return to_unit @ np.linalg.inv(self.affine)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Which of the (..., 3) world points lie inside the box."""
        pts = np.asarray(points, dtype=float)
        flat = pts.reshape(-1, 3)
        m = self.unit_from_world()
        u = flat @ m[:3, :3].T + m[:3, 3]
        inside = np.all((u >= 0.0) & (u <= 1.0), axis=1)
        return inside.reshape(pts.shape[:-1])


def section(box: GridBox, point, normal) -> np.ndarray:
    """The polygon a plane cuts out of ``box``: (K, 3) world points in order
    around it, K in 3..6; (0, 3) when the plane misses the box."""
    n = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm == 0.0:
        return np.empty((0, 3))
    n = n / norm
    c = box.corners()
    s = (c - np.asarray(point, dtype=float)) @ n
    span = float(np.abs(box.size_mm()).max()) or 1.0
    eps = 1e-9 * span
    pts = [c[k] for k in range(8) if abs(s[k]) <= eps]
    for a, b in EDGES:
        if (s[a] < -eps and s[b] > eps) or (s[a] > eps and s[b] < -eps):
            f = s[a] / (s[a] - s[b])
            pts.append(c[a] + f * (c[b] - c[a]))
    if len(pts) < 3:
        return np.empty((0, 3))
    pts = np.asarray(pts)
    # Order around the centroid, in a basis of the plane.
    centroid = pts.mean(axis=0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    d = pts - centroid
    order = np.argsort(np.arctan2(d @ v, d @ u))
    pts = pts[order]
    # Coincident points (a plane through a corner meets three edges there).
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-6 * span:
            keep.append(i)
    pts = pts[keep]
    if len(pts) > 2 and np.linalg.norm(pts[0] - pts[-1]) <= 1e-6 * span:
        pts = pts[:-1]
    return pts if len(pts) >= 3 else np.empty((0, 3))


def outline_mask(inside: np.ndarray, width: int = 2) -> np.ndarray:
    """The rim of a 2-D boolean region, ``width`` pixels deep, inside it."""
    inside = np.asarray(inside, dtype=bool)
    rim = np.zeros_like(inside)
    core = inside.copy()
    for _ in range(max(1, int(width))):
        shrunk = core.copy()
        shrunk[1:, :] &= core[:-1, :]
        shrunk[:-1, :] &= core[1:, :]
        shrunk[:, 1:] &= core[:, :-1]
        shrunk[:, :-1] &= core[:, 1:]
        shrunk[0, :] = shrunk[-1, :] = False
        shrunk[:, 0] = shrunk[:, -1] = False
        rim |= core & ~shrunk
        core = shrunk
    return rim


__all__ = ["EDGES", "GridBox", "outline_mask", "section"]

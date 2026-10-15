"""Head motion of a functional run: six rigid parameters per volume and the
framewise displacement (FD) derived from them.

READ when fMRIPrep's confounds exist for the run (beside it, or under
``derivatives/<pipeline>/`` at the same place): those come from a full
registration and are what an analysis will censor by, so the viewer shows
the same numbers rather than its own.

ESTIMATED otherwise, quickly, so a run can be judged before anything is
run on it: every volume is registered rigidly to the first steady one by
inverse-compositional Gauss-Newton (Baker and Matthews 2004), on the run's
own grid (voxels finer than about 3.5 mm averaged down to it) and on the
15,000 points where the reference's edges are strongest. The reference's
gradient and the 6 x 6 normal matrix are computed once; each volume then
costs a few trilinear resamplings, started from the previous volume's
answer: about 4 ms a volume.

Measured against fMRIPrep's own FD (2026-10-06, the SEGREGATION study):
on a participant who MOVES (three runs, FD up to 2.95 mm) FD correlates at
r 0.94 to 0.98, and of the volumes fMRIPrep puts above 0.5 mm it finds
every one in two runs and 11 of 13 in the third, with a few more of its
own. On a LOW-motion participant (mean FD 0.06 mm, motion near the noise)
the six parameter traces still agree (|r| 0.83 to 1.00), FD correlates at
r 0.54 to 0.88 and reads about 1.4 times fMRIPrep's. Registering at 6 mm
instead was worse (r 0.52 to 0.85, 1.6 times), and smoothing first made
both worse. A screening estimate, said so wherever it is shown, never the
registration an analysis uses.

FD is Power et al. (2012): the sum of the absolute frame-to-frame changes
of the three translations and of the three rotations taken as arc length
on a 50 mm sphere (the radius of a head).

Qt-free; runs on a worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

from ...qc.config import DEFAULT

# FD's sphere radius (Power 2012: 50 mm), the FD threshold worth a look
# (fMRIPrep's and Power 2012's 0.5 mm) and the voxel the estimate registers
# at (3.5 mm) are settings: ``bidsmgr.qc.config.QcBold``.
_BOLD = DEFAULT.bold
#: How many of the reference's strongest edges the estimate registers.
ESTIMATE_POINTS = 15_000

#: How fMRIPrep names its confounds table (current, then before 1.2).
CONFOUND_NAMES = ("desc-confounds_timeseries.tsv", "desc-confounds_regressors.tsv")
#: Entities a derivative adds to the run's name and its confounds do not have.
_DERIVED_ENTITIES = ("space", "res", "den", "desc", "cohort")
#: Column names: fMRIPrep 1.2 and later, then the earlier spelling.
_COLUMNS = (
    ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "framewise_displacement"),
    ("X", "Y", "Z", "RotX", "RotY", "RotZ", "FramewiseDisplacement"),
)


class Cancelled(Exception):
    pass


@dataclass
class Motion:
    """Six parameters per volume: translations (mm) then rotations (radians),
    and FD (mm; NaN for the first volume, which has nothing before it)."""

    params: np.ndarray
    fd: np.ndarray
    source: Literal["confounds", "estimated"]
    #: The confounds table, when read from one.
    path: Optional[Path] = None

    def describe(self) -> str:
        if self.source == "confounds":
            name = self.path.name if self.path is not None else "the confounds"
            return f"from {name}"
        return ("estimated here (rigid, quick): a screening estimate, not the "
                "registration an analysis uses")


# ---------------------------------------------------------------------------
# Reading fMRIPrep's
# ---------------------------------------------------------------------------


def _run_stem(path: Path) -> Optional[str]:
    """The run's name without the entities a derivative adds and without
    its suffix: ``sub-01_task-rest_run-1`` for any of its BOLD files."""
    name = Path(path).name.split(".")[0]
    parts = name.split("_")
    if len(parts) < 2 or "-" in parts[-1]:
        return None
    kept = [p for p in parts[:-1]
            if "-" in p and p.split("-", 1)[0] not in _DERIVED_ENTITIES]
    return "_".join(kept) or None


def confounds_for(path: Path, root: Optional[Path] = None) -> Optional[Path]:
    """fMRIPrep's confounds table for the run at ``path``: beside it, else
    in any ``derivatives/<pipeline>/`` of ``root`` at the same place."""
    path = Path(path)
    stem = _run_stem(path)
    if stem is None:
        return None
    names = [f"{stem}_{tail}" for tail in CONFOUND_NAMES]
    for name in names:
        if (path.parent / name).is_file():
            return path.parent / name
    if root is None:
        return None
    try:
        rel = path.parent.resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return None
    for pipeline in sorted((Path(root) / "derivatives").glob("*")):
        for name in names:
            candidate = pipeline / rel / name
            if candidate.is_file():
                return candidate
    return None


def framewise_displacement(params: np.ndarray,
                           radius: float = _BOLD.fd_radius_mm) -> np.ndarray:
    """Power's FD from ``params`` (n x 6: mm, then radians); NaN first."""
    p = np.asarray(params, dtype=float)
    d = np.abs(np.diff(p, axis=0))
    fd = d[:, :3].sum(axis=1) + radius * d[:, 3:].sum(axis=1)
    return np.concatenate([[np.nan], fd])


def read_confounds(path: Path, radius: float = _BOLD.fd_radius_mm) -> Motion:
    """The motion columns of an fMRIPrep confounds table. Its FD column is
    used as it is when ``radius`` is fMRIPrep's 50 mm; another radius
    recomputes FD from the parameters."""
    import pandas as pd

    table = pd.read_csv(path, sep="\t", na_values=["n/a"])
    for cols in _COLUMNS:
        if all(c in table.columns for c in cols[:6]):
            params = table[list(cols[:6])].to_numpy(dtype=float)
            if cols[6] in table.columns and abs(radius - 50.0) < 1e-9:
                fd = table[cols[6]].to_numpy(dtype=float)
                fd[0] = np.nan
            else:
                fd = framewise_displacement(np.nan_to_num(params), radius)
            return Motion(np.nan_to_num(params), fd, "confounds", Path(path))
    raise ValueError(f"{Path(path).name} has no motion parameters (trans_x ... rot_z)")


# ---------------------------------------------------------------------------
# Estimating
# ---------------------------------------------------------------------------


def _down(volume: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Block means of ``factor`` voxels (a cheap low-pass and a smaller grid)."""
    v = np.asarray(volume, dtype=np.float32)
    fx, fy, fz = (int(f) for f in factor)
    sx, sy, sz = (s // f * f for s, f in zip(v.shape[:3], (fx, fy, fz)))
    v = v[:sx, :sy, :sz]
    if fx == fy == fz == 1:
        return v
    return v.reshape(sx // fx, fx, sy // fy, fy, sz // fz, fz).mean(axis=(1, 3, 5))


def _rotation(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of the rotation vector ``w`` (Rodrigues)."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(w, dtype=float) / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def _rotation_vector(r: np.ndarray) -> np.ndarray:
    """The rotation vector of matrix ``r`` (inverse of :func:`_rotation`)."""
    cos = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos))
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if theta < 1e-9:
        return v / 2.0
    return v * theta / (2.0 * np.sin(theta))


class RigidReference:
    """One reference volume, ready to register others to it rigidly.

    Built once (its edges, their gradients and the 6 x 6 normal matrix), then
    :meth:`register` costs a few trilinear resamplings per volume. The
    registration :func:`estimate` runs for a BOLD series, and the one the
    diffusion check runs against each shell's own mean (``bidsmgr.qc.dwi``):
    a b=1000 volume and a b0 do not share a contrast, so each is registered
    to its own kind.

    ``reference`` is on the grid :func:`working_grid` returns for the series
    (block means of ``factor`` voxels); volumes passed to :meth:`register`
    must be on the same grid. Parameters are in mm about the grid's centre:
    three translations, then a rotation vector.
    """

    def __init__(self, reference: np.ndarray, spacing, *, points: int = ESTIMATE_POINTS,
                 mask: Optional[np.ndarray] = None) -> None:
        from scipy import ndimage

        ref = np.asarray(reference, dtype=np.float32)
        self.spacing = np.asarray(spacing, dtype=float)
        shape = np.asarray(ref.shape, dtype=float)
        self.centre = (shape - 1.0) / 2.0
        if mask is None:
            finite = ref[np.isfinite(ref)]
            top = float(np.percentile(finite, 98)) if finite.size else 0.0
            mask = ref > 0.2 * top
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("no voxel is bright enough to be in the head")
        # The edge of the head is where motion shows: keep a voxel around it.
        mask = ndimage.binary_dilation(mask, iterations=1)
        grads = np.gradient(ref.astype(np.float64), *self.spacing)
        g = np.stack([gr[mask] for gr in grads], axis=1)           # N x 3
        idx = np.argwhere(mask).astype(float)
        if points and len(idx) > points:
            # The strongest edges carry the registration; flat tissue adds
            # time, not information.
            keep = np.argpartition(-(g * g).sum(axis=1), points)[:points]
            keep.sort()
            g, idx = g[keep], idx[keep]
        self.x = (idx - self.centre) * self.spacing                # mm, centred
        self.values = ref[tuple(idx.astype(int).T)].astype(np.float64)
        self.sd = np.concatenate([g, np.cross(self.x, g)], axis=1)  # N x 6
        hessian = self.sd.T @ self.sd
        try:
            self.h_inv = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            self.h_inv = np.linalg.pinv(hessian)
        self.mean = float(self.values.mean()) or 1.0

    def register(self, frame: np.ndarray, rot: Optional[np.ndarray] = None,
                 trans: Optional[np.ndarray] = None, *, iterations: int = 12
                 ) -> tuple[np.ndarray, np.ndarray]:
        """``(rotation matrix, translation mm)`` taking this reference's
        points onto ``frame``, started from ``rot``/``trans`` (the previous
        volume's answer, usually)."""
        from scipy import ndimage

        rot = np.eye(3) if rot is None else np.array(rot, dtype=float)
        trans = np.zeros(3) if trans is None else np.array(trans, dtype=float)
        for _ in range(iterations):
            pts = self.x @ rot.T + trans
            coords = (pts / self.spacing + self.centre).T
            warped = ndimage.map_coordinates(frame, coords, order=1, mode="nearest",
                                             prefilter=False).astype(np.float64)
            # Each volume on the reference's intensity scale: a global
            # signal change is not motion.
            w_mean = float(warped.mean()) or 1.0
            err = warped * (self.mean / w_mean) - self.values
            dp = self.h_inv @ (self.sd.T @ err)
            d_rot = _rotation(dp[3:])
            rot = rot @ d_rot.T
            trans = trans - rot @ dp[:3]
            if np.max(np.abs(dp[:3])) < 1e-4 and np.max(np.abs(dp[3:])) < 1e-6:
                break
        return rot, trans


def working_grid(zooms, spatial, target_mm: float = _BOLD.motion_grid_mm) -> np.ndarray:
    """The block factor per axis that brings voxels finer than ``target_mm``
    up to about it, never below about eight voxels an axis (a gradient needs
    neighbours)."""
    zooms = np.asarray(zooms, dtype=float)
    factor = np.maximum(1, np.round(target_mm / np.maximum(zooms, 1e-3))).astype(int)
    return np.minimum(factor, np.maximum(1, np.asarray(spatial) // 8)).astype(int)


def estimate(src, *, skip: int = 0, cancel: Optional[Callable[[], bool]] = None,
             progress: Optional[Callable[[int, int], None]] = None,
             target_mm: float = _BOLD.motion_grid_mm, iterations: int = 12,
             points: int = ESTIMATE_POINTS, radius: float = _BOLD.fd_radius_mm) -> Motion:
    """Register every volume rigidly to volume ``skip`` (see the module)."""
    n = src.loaded_frames
    if n < 2:
        raise ValueError("motion needs at least two volumes")
    factor = working_grid(src.zooms3, src.spatial, target_mm)
    spacing = np.asarray(src.zooms3, dtype=float) * factor
    ref_index = int(min(max(skip, 0), n - 1))
    reference = RigidReference(_down(src.scale(np.asarray(src.raw_frame(ref_index))), factor),
                               spacing, points=points)

    params = np.zeros((n, 6))
    rot = np.eye(3)
    trans = np.zeros(3)
    order = list(range(ref_index, n)) + list(range(ref_index - 1, -1, -1))
    for count, t in enumerate(order):
        if cancel is not None and cancel():
            raise Cancelled()
        if t == ref_index - 1:
            # Walking back from the reference: start from it again.
            rot, trans = np.eye(3), np.zeros(3)
        frame = _down(src.scale(np.asarray(src.raw_frame(t))), factor)
        rot, trans = reference.register(frame, rot, trans, iterations=iterations)
        params[t, :3] = trans
        params[t, 3:] = _rotation_vector(rot)
        if progress is not None and (count % 16 == 0 or count == n - 1):
            progress(count + 1, n)
    return Motion(params, framewise_displacement(params, radius), "estimated")


def motion_for(src, path: Optional[Path] = None, root: Optional[Path] = None, *,
               skip: int = 0, config=None, cancel: Optional[Callable[[], bool]] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> Motion:
    """fMRIPrep's motion for the run when it has the run's length and
    ``config`` (a ``QcConfig``) allows it, else the estimate."""
    cfg = config or DEFAULT
    bold = cfg.bold
    found = (confounds_for(path, root)
             if path is not None and cfg.methods.bold_motion == "confounds" else None)
    if found is not None:
        try:
            got = read_confounds(found, bold.fd_radius_mm)
            if len(got.fd) == src.loaded_frames:
                return got
        except (ValueError, OSError, KeyError):
            pass
    return estimate(src, skip=skip, cancel=cancel, progress=progress,
                    target_mm=bold.motion_grid_mm, radius=bold.fd_radius_mm)


__all__ = [
    "CONFOUND_NAMES", "ESTIMATE_POINTS", "Cancelled",
    "Motion", "RigidReference", "confounds_for", "estimate", "framewise_displacement",
    "motion_for", "read_confounds", "working_grid",
]

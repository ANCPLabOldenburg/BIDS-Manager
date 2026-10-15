"""Spectra: a NIfTI-MRS file as a source.

The FID is kept with its higher dimensions and their NIfTI-MRS TAGS
(``dim_5`` = ``DIM_COIL``, ``dim_6`` = ``DIM_DYN``, ``DIM_EDIT``, ...),
instead of flattening everything past the points axis into one "repeats"
axis: coils are COMBINED, dynamics are averaged or picked, edit conditions
are picked or SUBTRACTED (the edit-on minus edit-off difference is what a
MEGA-PRESS acquisition is for). A dimension with no tag is treated as
dynamics.

The water reference of the same acquisition (``_mrsref``) is found beside
it, so it can be overlaid.

Qt-free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ...fixups.mrs_header import read_mrs_header
from ..compute import mrs as M

log = logging.getLogger(__name__)

#: Tags the selection understands; anything else counts as dynamics.
COIL = "DIM_COIL"
DYN = "DIM_DYN"
EDIT = "DIM_EDIT"


@dataclass
class SpectrumSource:
    path: Path
    #: (points, *higher) complex128.
    fid: np.ndarray
    #: One tag per axis after the points axis.
    dims: list[str]
    dwell: float
    spectrometer_mhz: float
    nucleus: str
    header: dict = field(default_factory=dict)
    shape: tuple = ()
    reference: Optional["SpectrumSource"] = None
    #: Where the voxel sits in the scanner (the NIfTI affine), when known.
    affine: Optional[np.ndarray] = None

    def voxel_centre_world(self) -> Optional[np.ndarray]:
        """The centre of the measured voxel (of the grid, for MRSI), in mm."""
        if self.affine is None or len(self.shape) < 3:
            return None
        centre = (np.asarray(self.shape[:3], dtype=float) - 1.0) / 2.0
        return (self.affine @ np.append(centre, 1.0))[:3]

    @property
    def n_points(self) -> int:
        return int(self.fid.shape[0])

    def size(self, tag: str) -> int:
        n = 1
        for i, t in enumerate(self.dims):
            if t == tag or (tag == DYN and t not in (COIL, EDIT)):
                n *= int(self.fid.shape[1 + i])
        return n

    @property
    def n_dynamics(self) -> int:
        return self.size(DYN)

    @property
    def n_coils(self) -> int:
        return self.size(COIL)

    @property
    def n_edits(self) -> int:
        return self.size(EDIT)

    def select(self, *, dynamic: int = -1, edit: int = -1) -> np.ndarray:
        """One FID: coils combined, dynamics averaged (``-1``) or picked,
        edit conditions averaged (``-1``), picked, or ``-2`` for the
        difference of the first two (edit on minus edit off)."""
        data = self.fid
        tags = list(self.dims)
        if COIL in tags:
            data = M.coil_combine(data, axis=1 + tags.index(COIL))
            tags.remove(COIL)
        if EDIT in tags:
            ax = 1 + tags.index(EDIT)
            n = data.shape[ax]
            if edit == -2 and n >= 2:
                data = np.take(data, 0, axis=ax) - np.take(data, 1, axis=ax)
            elif edit >= 0:
                data = np.take(data, min(edit, n - 1), axis=ax)
            else:
                data = data.mean(axis=ax)
            tags.remove(EDIT)
        # Everything left is dynamics: flatten, then average or pick.
        data = data.reshape(data.shape[0], -1)
        if dynamic >= 0:
            return data[:, min(dynamic, data.shape[1] - 1)]
        return data.mean(axis=1)

    def describe(self) -> str:
        bits = [f"{self.nucleus} spectrum, {self.n_points:,} points",
                f"{1.0 / self.dwell:.0f} Hz wide at {self.spectrometer_mhz:.2f} MHz"]
        if self.n_coils > 1:
            bits.append(f"{self.n_coils} coils combined")
        if self.n_edits > 1:
            bits.append(f"{self.n_edits} edit conditions")
        bits.append(f"{self.n_dynamics} repeat(s)")
        return ", ".join(bits)


def read_mrs(path, *, with_reference: bool = True) -> Optional[SpectrumSource]:
    """``path`` as a spectrum, or None when it is not NIfTI-MRS (no
    extension-44 header, real data, no dwell time)."""
    import nibabel as nib

    try:
        img = nib.load(str(path))
        header = read_mrs_header(path)
        if header is None:
            return None
        data = np.asarray(img.dataobj)
        if not np.iscomplexobj(data):
            log.debug("%s has an MRS header but real data", Path(path).name)
            return None
        if data.ndim < 4:
            return None
        # Single voxel: (x, y, z, points, d5, d6, d7) -> (points, d5, ...).
        # A multi-voxel (MRSI) file shows its first voxel here.
        fid = np.asarray(data[0, 0, 0], dtype=np.complex128)
        dims = []
        for k in range(5, 5 + fid.ndim - 1):
            dims.append(str(header.get(f"dim_{k}") or DYN).upper())
        zooms = img.header.get_zooms()
        dwell = float(zooms[3]) if len(zooms) > 3 else 0.0
        if not np.isfinite(dwell) or dwell <= 0:
            dwell = float(header.get("DwellTime") or 0.0)
        if dwell <= 0:
            log.debug("%s states no dwell time", Path(path).name)
            return None
        nuclei = header.get("ResonantNucleus") or ["1H"]
        nucleus = str(nuclei[0]).upper().replace(" ", "")
        freqs = header.get("SpectrometerFrequency") or [0.0]
        src = SpectrumSource(
            path=Path(path), fid=fid, dims=dims, dwell=dwell,
            spectrometer_mhz=float(freqs[0]), nucleus=nucleus, header=header,
            shape=tuple(int(x) for x in data.shape),
            affine=np.asarray(img.affine, dtype=float),
        )
    except Exception as exc:  # noqa: BLE001 - not every .nii.gz is MRS
        log.debug("could not read %s as NIfTI-MRS: %s", path, exc)
        return None
    if with_reference:
        ref = reference_for(Path(path))
        if ref is not None:
            src.reference = read_mrs(ref, with_reference=False)
    return src


def reference_for(path: Path) -> Optional[Path]:
    """The water reference of the same acquisition: the same name with the
    suffix ``mrsref``, else the only ``_mrsref`` file in the folder."""
    from ..bids import full_ext, stem_of

    path = Path(path)
    stem = stem_of(path)
    if stem.endswith("_mrsref"):
        return None
    ext = full_ext(path)
    base = stem.rsplit("_", 1)[0] if "_" in stem else stem
    exact = path.with_name(f"{base}_mrsref{ext}")
    if exact.is_file():
        return exact
    try:
        refs = sorted(p for p in path.parent.iterdir()
                      if stem_of(p).endswith("_mrsref") and full_ext(p) in (".nii", ".nii.gz"))
    except OSError:
        return None
    return refs[0] if len(refs) == 1 else None


__all__ = ["COIL", "DYN", "EDIT", "SpectrumSource", "read_mrs", "reference_for"]

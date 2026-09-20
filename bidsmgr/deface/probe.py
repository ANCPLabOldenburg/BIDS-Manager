"""Read just enough of a NIfTI header to decide whether to deface it.

The question is always the same: is this a single 3-D volume? A registration
defacer produces meaningless output on a time series, so a BOLD, a DWI or a
multi-echo run has to be refused rather than silently mangled.

Reading is deliberately shallow. ``dim`` sits in the first 56 bytes of a
NIfTI-1 header, so this decompresses a few hundred bytes and stops, instead of
loading a 200 MB series to look at eight integers. On a dataset of any size
that difference is the difference between a dialog that opens and one that
hangs.

nibabel is a dependency and could do this. It is not used here on purpose: it
reads more than is needed, and this runs once per candidate file while a dialog
is waiting to appear.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

# NIfTI-1 puts sizeof_hdr at offset 0 and it must read 348. NIfTI-2 uses 540
# and a different layout; we recognise it in order to say so, not to parse it.
NIFTI1_SIZEOF_HDR = 348
NIFTI2_SIZEOF_HDR = 540

# dim[0..7] are int16 at offset 40. Reading through dim[4] needs 50 bytes;
# take a little more so a short read is obviously a short read.
_HEADER_BYTES = 128
_DIM_OFFSET = 40


class NotNifti(ValueError):
    """The file is not a NIfTI-1 image, or is too short to be one."""


@dataclass(frozen=True)
class Dimensions:
    """What the header says about shape."""

    rank: int          # dim[0]
    dim1: int
    dim2: int
    dim3: int
    dim4: int

    @property
    def is_3d(self) -> bool:
        """One volume, so a registration defacer can work on it.

        ``rank`` alone is not enough. Plenty of 3-D images are written with
        ``dim[0] = 4`` and ``dim[4] = 1``, which is legal and means one volume.
        Refusing those would refuse a large share of real anatomical data.
        """
        return self.rank <= 3 or self.dim4 <= 1

    @property
    def shape(self) -> tuple[int, ...]:
        dims = (self.dim1, self.dim2, self.dim3, self.dim4)
        return dims[: max(1, min(self.rank, 4))]


def _read_head(path: Path) -> bytes:
    """The first bytes of the file, decompressing only if it is gzipped."""
    path = Path(path)
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as fh:
            return fh.read(_HEADER_BYTES)
    with open(path, "rb") as fh:
        return fh.read(_HEADER_BYTES)


def read_dimensions(path: Path) -> Dimensions:
    """Shape from the header. Raises :class:`NotNifti` if it is not one."""
    raw = _read_head(path)
    if len(raw) < _DIM_OFFSET + 10:
        raise NotNifti(f"{path} is too short to hold a NIfTI header")

    little = struct.unpack_from("<i", raw, 0)[0]
    big = struct.unpack_from(">i", raw, 0)[0]
    if little == NIFTI1_SIZEOF_HDR:
        endian = "<"
    elif big == NIFTI1_SIZEOF_HDR:
        endian = ">"
    elif NIFTI2_SIZEOF_HDR in (little, big):
        raise NotNifti(
            f"{path} is NIfTI-2, which this cannot read. Convert it to "
            "NIfTI-1, or deface it with another tool."
        )
    else:
        raise NotNifti(f"{path} does not start with a NIfTI-1 header")

    dims = struct.unpack_from(f"{endian}5h", raw, _DIM_OFFSET)
    return Dimensions(
        rank=int(dims[0]),
        dim1=int(dims[1]),
        dim2=int(dims[2]),
        dim3=int(dims[3]),
        dim4=int(dims[4]),
    )


def is_single_volume(path: Path) -> bool:
    """True when the file is one 3-D volume. False for anything unreadable.

    Fails closed. A file this cannot parse is not a file to deface blind.
    """
    try:
        return read_dimensions(path).is_3d
    except (NotNifti, OSError, struct.error):
        return False


__all__ = ["Dimensions", "NotNifti", "is_single_volume", "read_dimensions"]

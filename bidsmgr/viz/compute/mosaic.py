"""The mosaic grammar: many slices in one figure, from one line of text.

NiiVue's syntax (BSD-2), so a mosaic line written for NiiVue works here:

* ``A`` / ``C`` / ``S`` choose axial, coronal or sagittal for what follows;
* a number is a slice position in millimetres along that plane's axis;
* ``;`` starts a new row;
* ``X`` draws, on the following tiles, where the other tiles cut;
* ``L-`` hides the slice-position labels (``L+`` shows them again);
* ``H 0.3`` overlaps neighbouring tiles by 30 percent;
* ``R`` makes the following numbers 3-D renders. The CPU mosaic has no
  renderer, so render tiles are skipped and :func:`parse` reports them.

Pure functions; :mod:`bidsmgr.gui.viz.canvases.mosaic` draws the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_PLANE = {"A": "axial", "C": "coronal", "S": "sagittal"}


@dataclass(frozen=True)
class Tile:
    plane: str
    mm: float
    cross: bool = False


@dataclass
class Mosaic:
    rows: list[list[Tile]] = field(default_factory=list)
    labels: bool = True
    overlap: float = 0.0
    skipped_renders: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def tiles(self) -> list[Tile]:
        return [t for row in self.rows for t in row]


def parse(text: str) -> Mosaic:
    """Parse a mosaic line. Unknown tokens are reported, never fatal."""
    out = Mosaic()
    row: list[Tile] = []
    plane = "axial"
    render = False
    cross = False
    tokens = (text or "").replace(";", " ; ").split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        up = tok.upper()
        if up == ";":
            if row:
                out.rows.append(row)
            row = []
            render = False
            cross = False
        elif up in _PLANE:
            plane = _PLANE[up]
            render = False
        elif up == "R":
            render = True
        elif up == "X":
            cross = True
        elif up in ("L-", "L+", "L"):
            out.labels = up != "L-"
        elif up in ("H", "V"):
            if i + 1 < len(tokens):
                try:
                    if up == "H":
                        out.overlap = max(0.0, min(0.9, float(tokens[i + 1])))
                    i += 1
                except ValueError:
                    out.errors.append(f"{tok} needs a number")
        else:
            try:
                mm = float(tok)
            except ValueError:
                out.errors.append(f"not understood: {tok}")
            else:
                if render:
                    out.skipped_renders += 1
                else:
                    row.append(Tile(plane=plane, mm=mm, cross=cross))
        i += 1
    if row:
        out.rows.append(row)
    return out


# ---------------------------------------------------------------------------
# Building a line from what a figure needs
# ---------------------------------------------------------------------------

_LETTER = {"axial": "A", "coronal": "C", "sagittal": "S"}
#: How far the brain reaches below the top of the head, in mm (cerebrum and
#: cerebellum): where an axial fit stops when the image includes the neck.
BRAIN_HEIGHT_MM = 140.0
#: The plane a reference slice is cut in, for each plane of the grid: the
#: one that shows the grid's slices as lines across the head.
REFERENCE = {"axial": "sagittal", "coronal": "sagittal", "sagittal": "axial"}


def head_extent(values, affine, axis: int, *, fraction: float = 0.08) -> tuple[float, float]:
    """Where the head is along world ``axis`` (0 x, 1 y, 2 z), in mm: the
    2nd to 98th percentile of the world positions of voxels brighter than
    ``fraction`` of the image's robust maximum. A strided sample, so it is
    instant on a 0.6 mm T1; the percentiles keep a stray bright voxel (an
    artefact, a vitamin E marker) from stretching the range."""
    import numpy as np

    v = np.asarray(values)
    step = max(1, int(round((v.size / 400_000) ** (1.0 / 3.0))))
    sub = v[::step, ::step, ::step].astype(np.float64)
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        raise ValueError("the image has no finite values")
    top = float(np.percentile(finite, 99.5))
    idx = np.argwhere(np.nan_to_num(sub) > fraction * top)
    if idx.size == 0:
        raise ValueError("nothing in the image is bright enough to be a head")
    vox = idx * step
    a = np.asarray(affine, dtype=float)
    world = vox @ a[:3, :3].T + a[:3, 3]
    lo, hi = np.percentile(world[:, axis], (2.0, 98.0))
    if axis == 2 and hi - lo > BRAIN_HEIGHT_MM + 10.0:
        # A head with its neck: the brain is the top of it. A mosaic of the
        # neck's slices is not what an axial figure is for.
        lo = hi - BRAIN_HEIGHT_MM
    return float(lo), float(hi)


def positions(start: float, end: float, n: int) -> list[float]:
    """``n`` slice positions evenly across ``[start, end]``, each at the
    centre of its share (the outermost do not sit on the edge of the head,
    where they would show a sliver of scalp)."""
    if n <= 0:
        return []
    step = (end - start) / n
    return [round(start + (k + 0.5) * step, 1) for k in range(n)]


def build_line(plane: str, rows: int, cols: int, start: float, end: float, *,
               labels: bool = True, reference: bool = False, overlap: float = 0.0,
               reference_mm: float = 0.0) -> str:
    """The grammar line of a ``rows`` x ``cols`` grid of ``plane`` slices
    across ``[start, end]`` mm, with an optional reference slice first."""
    letter = _LETTER[plane]
    mm = positions(start, end, rows * cols)
    parts = []
    if not labels:
        parts.append("L-")
    if overlap > 0:
        parts.append(f"H {overlap:g}")
    if reference:
        parts.append(f"{_LETTER[REFERENCE[plane]]} X {reference_mm:g} ;")
    for r in range(rows):
        row = mm[r * cols:(r + 1) * cols]
        parts.append(letter + " " + " ".join(f"{v:g}" for v in row))
        if r < rows - 1:
            parts.append(";")
    return " ".join(parts)


__all__ = ["REFERENCE", "Mosaic", "Tile", "build_line", "head_extent", "parse", "positions"]

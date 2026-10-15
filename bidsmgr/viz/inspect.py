"""An image's header beside its sidecar, with every disagreement said.

Two descriptions of one acquisition travel together in BIDS: the NIfTI
header, written by the converter from the scanner's geometry, and the JSON
sidecar, written from the scanner's protocol. Tools read one or the other,
so when they disagree (a repetition time in the header that is not the
sidecar's, more slice times than slices, b-values for volumes that are not
there) results differ by which tool was used, silently.

:func:`inspect` lays the two side by side as :class:`HeaderRow` s, each
flagged where they disagree and naming the validator rules it answers, so a
validation finding can open the inspector on the row that explains it.

Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Status = Literal["ok", "info", "warning", "error"]

#: Validator rules the inspector can show the evidence for.
HEADER_RULES: frozenset[str] = frozenset({
    "NIFTI_DIMENSION", "NIFTI_PIXDIM", "NIFTI_PIXDIM_PET", "NIFTI_UNIT",
    "SFORM_AND_QFORM_IN_IMAGE_HEADER_ARE_ZERO", "REPETITION_TIME_MISMATCH",
    "SLICETIMING_ELEMENTS", "SLICETIMING_VALUES_GREATER_THAN_REPETITION_TIME",
    "VOLUME_COUNT_MISMATCH", "PET_FRAME_CONSISTENCY",
    "PET_FRAME_CONSISTENCY_FRAME_TIMES_START", "BVAL_MULTIPLE_ROWS",
})

#: NIfTI extension codes worth a name (the registry is long; these appear).
_EXTENSIONS = {
    2: "DICOM", 4: "AFNI", 6: "comment", 14: "MIND", 18: "Caret", 32: "CIfTI",
    40: "dcm2niix", 44: "NIfTI-MRS",
}

#: How far the header's repetition time may sit from the sidecar's before it
#: is a different number rather than a rounding (the validator's tolerance).
TR_TOLERANCE_S = 1e-3


@dataclass(frozen=True)
class HeaderRow:
    group: str
    field: str
    header: str
    sidecar: str = ""
    status: Status = "ok"
    note: str = ""
    #: Validator rule ids this row is the evidence for.
    rules: tuple[str, ...] = ()


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def axis_codes(affine: np.ndarray) -> str:
    """"RAS", "LPS", ...: where each voxel axis points (the nearest)."""
    try:
        import nibabel as nib

        return "".join(nib.orientations.aff2axcodes(affine))
    except Exception:  # noqa: BLE001 - a degenerate affine
        return "?"


def obliquity_degrees(affine: np.ndarray) -> float:
    """How far the voxel grid is turned from the scanner's axes, at worst."""
    rot = np.asarray(affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(rot, axis=0)
    if np.any(norms == 0):
        return 0.0
    cosines = np.max(np.abs(rot / norms), axis=0)
    return float(np.degrees(np.max(np.arccos(np.clip(cosines, -1.0, 1.0)))))


def _geometry(src, f) -> list[HeaderRow]:
    rows = []
    shape = " x ".join(str(n) for n in f.shape)
    rows.append(HeaderRow("Geometry", "Dimensions", shape, rules=("NIFTI_DIMENSION",),
                          note="" if len(f.shape) >= 3 else "a 2-D image"))
    zooms = [float(z) for z in f.zooms[:3]]
    bad = [z for z in zooms if not np.isfinite(z) or z <= 0]
    rows.append(HeaderRow(
        "Geometry", "Voxel size", " x ".join(_fmt(z) for z in zooms) + " " + (f.space_unit or ""),
        status="error" if bad else "ok",
        note="a voxel size of zero or less: distances in this image mean nothing" if bad else "",
        rules=("NIFTI_PIXDIM", "NIFTI_PIXDIM_PET")))
    units_ok = f.space_unit in ("mm", "meter", "micron")
    time_ok = len(f.shape) < 4 or f.time_unit in ("sec", "msec", "usec")
    note = []
    if not units_ok:
        note.append("no spatial unit: millimetres are assumed")
    if not time_ok:
        note.append("no time unit: seconds are assumed")
    rows.append(HeaderRow(
        "Geometry", "Units", f"{f.space_unit or 'unknown'}" + (
            f", {f.time_unit or 'unknown'}" if len(f.shape) >= 4 else ""),
        status="ok" if units_ok and time_ok else "warning", note="; ".join(note),
        rules=("NIFTI_UNIT",)))
    codes = {0: "unknown", 1: "scanner", 2: "aligned", 3: "Talairach", 4: "MNI", 5: "template"}
    none = f.qform_code == 0 and f.sform_code == 0
    rows.append(HeaderRow(
        "Geometry", "Orientation",
        f"{axis_codes(src.affine)}; sform {f.sform_code} ({codes.get(f.sform_code, '?')}), "
        f"qform {f.qform_code} ({codes.get(f.qform_code, '?')})",
        status="error" if none else "ok",
        note=("neither sform nor qform is set: no tool can say where this image "
              "is, or which side is left") if none else "",
        rules=("SFORM_AND_QFORM_IN_IMAGE_HEADER_ARE_ZERO",)))
    tilt = obliquity_degrees(src.affine)
    rows.append(HeaderRow(
        "Geometry", "Obliquity", f"{tilt:.2f} degrees",
        status="info" if tilt > 0.01 else "ok",
        note=("the voxel grid is turned from the scanner's axes; World space "
              "draws it upright") if tilt > 0.01 else ""))
    return rows


def _data(src, f) -> list[HeaderRow]:
    rows = []
    scaling = ""
    if f.slope != 1.0 or f.inter != 0.0:
        scaling = f", stored x {_fmt(f.slope)} + {_fmt(f.inter)}"
    rows.append(HeaderRow("Data", "Data type", f"{f.disk_dtype}{scaling}"))
    if f.cal_max > f.cal_min:
        rows.append(HeaderRow("Data", "Display range", f"{_fmt(f.cal_min)} to {_fmt(f.cal_max)}",
                              note="the range the writer suggests for display"))
    if f.intent_code:
        rows.append(HeaderRow("Data", "Intent", f"{f.intent_name or f.intent_code}"))
    if f.descrip:
        rows.append(HeaderRow("Data", "Description", f.descrip))
    if f.extension_codes:
        names = ", ".join(_EXTENSIONS.get(c, str(c)) for c in f.extension_codes)
        rows.append(HeaderRow("Data", "Extensions", names))
    return rows


def _slices_along(src, f) -> tuple[int, int]:
    """(axis, count) of the acquired slices: the header's slice dimension,
    else the third axis (BIDS assumes it when ``dim_info`` is unset)."""
    axis = f.slice_dim if 0 <= f.slice_dim < 3 else 2
    return axis, int(src.spatial[axis])


def _timing(src, f, sidecar: dict, bids) -> list[HeaderRow]:
    rows = []
    n = int(src.n_frames_total or src.n_frames)
    rows.append(HeaderRow("Time", "Volumes", str(n),
                          note="read the first " + str(src.n_frames) + " only (memory)"
                          if src.truncated else ""))
    tr_json = sidecar.get("RepetitionTime")
    tr_head = src.header_tr
    if isinstance(tr_json, (int, float)) or tr_head:
        side = f"{_fmt(float(tr_json))} s" if isinstance(tr_json, (int, float)) else "absent"
        head = f"{_fmt(tr_head)} s" if tr_head else "not set"
        status: Status = "ok"
        note = ""
        if isinstance(tr_json, (int, float)) and tr_head:
            if abs(float(tr_json) - tr_head) > TR_TOLERANCE_S:
                status = "error"
                note = ("tools that read the header and tools that read the sidecar "
                        "will place every event of this run differently")
        elif isinstance(tr_json, (int, float)) and not tr_head:
            status, note = "warning", "the header carries no repetition time"
        rows.append(HeaderRow("Time", "Repetition time", head, side, status, note,
                              rules=("REPETITION_TIME_MISMATCH",)))
    timing = sidecar.get("SliceTiming")
    if isinstance(timing, list) and timing:
        try:
            times = np.asarray([float(t) for t in timing], dtype=float)
        except (TypeError, ValueError):
            times = np.asarray([], dtype=float)
        axis, count = _slices_along(src, f)
        side = (f"{len(times)} times, {_fmt(float(times.min()))} to {_fmt(float(times.max()))} s"
                if times.size else "not numbers")
        status = "ok"
        note = ""
        rules: tuple[str, ...] = ("SLICETIMING_ELEMENTS",)
        if times.size != count:
            status = "error"
            note = (f"{times.size} slice times for {count} slices along axis "
                    f"{'xyz'[axis]}: slice-timing correction will misplace them")
        elif isinstance(tr_json, (int, float)) and times.size and times.max() >= float(tr_json):
            status = "error"
            note = "a slice is timed at or after the next volume starts"
            rules = ("SLICETIMING_VALUES_GREATER_THAN_REPETITION_TIME",)
        head = f"{count} slices along {'xyz'[axis]}"
        if f.slice_duration:
            head += f", {_fmt(f.slice_duration)} s each"
        rows.append(HeaderRow("Time", "Slice timing", head, side, status, note, rules))
    frames = getattr(bids, "frame_starts", None) if bids is not None else None
    if frames is not None:
        key = "FrameTimesStart" if "FrameTimesStart" in sidecar else "VolumeTiming"
        bad = len(frames) != n
        rows.append(HeaderRow(
            "Time", key, f"{n} volumes", f"{len(frames)} times",
            "error" if bad else "ok",
            f"{len(frames)} times for {n} volumes" if bad else "",
            rules=("PET_FRAME_CONSISTENCY", "PET_FRAME_CONSISTENCY_FRAME_TIMES_START")))
    return rows


def _diffusion(src, bids) -> list[HeaderRow]:
    bvals = getattr(bids, "bvals", None) if bids is not None else None
    if bvals is None:
        return []
    n = int(src.n_frames_total or src.n_frames)
    bad = len(bvals) != n
    shells = sorted({int(round(b / 50.0) * 50) for b in bvals})
    return [HeaderRow(
        "Diffusion", "b-values", f"{n} volumes",
        f"{len(bvals)} values; shells {', '.join(str(s) for s in shells)}",
        "error" if bad else "ok",
        f"{len(bvals)} b-values for {n} volumes" if bad else "",
        rules=("VOLUME_COUNT_MISMATCH",))]


def inspect(src, bids=None) -> list[HeaderRow]:
    """Every row for ``src`` (a :class:`~.data.volume.VolumeSource`), with
    its BIDS context (sidecar, b-values, frame times) when known."""
    f = src.facts
    sidecar: dict = dict(getattr(bids, "sidecar", {}) or {})
    rows = _geometry(src, f) + _data(src, f)
    if len(f.shape) >= 4 and int(np.prod(f.shape[3:])) > 1:
        rows += _timing(src, f, sidecar, bids)
    rows += _diffusion(src, bids)
    return rows


def rows_for_rule(rows: list[HeaderRow], rule: str) -> list[HeaderRow]:
    return [r for r in rows if rule in r.rules]


def worst(rows: list[HeaderRow]) -> Optional[Status]:
    order = {"error": 3, "warning": 2, "info": 1, "ok": 0}
    found = [r.status for r in rows if r.status != "ok"]
    return max(found, key=order.get) if found else None


__all__ = ["HEADER_RULES", "HeaderRow", "Status", "TR_TOLERANCE_S", "axis_codes",
           "inspect", "obliquity_degrees", "rows_for_rule", "worst"]

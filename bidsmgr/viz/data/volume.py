"""A volume opened for viewing: the header now, the voxels as they stream.

Reading a 4-D series the obvious way, ``img.get_fdata()``, costs twice:
float64 doubles or quadruples the memory of an int16 scan (a 1224-volume BOLD
run is 679 MB on disk and 2.7 GB as float64), and nothing can be shown until
the whole file is decompressed. Worse, gzip cannot seek backwards, so reading
"frame 1000" on demand decompresses everything before it, every time: 2 s per
frame on that run, which makes playback impossible.

So a :class:`VolumeSource` reads the header at once, then STREAMS the voxels
front to back on a worker into one array in the file's own dtype, frame by
frame. Frame 0 is drawable within milliseconds; a time course reads only the
frames already in. Scaling (``scl_slope`` / ``scl_inter``) is applied to the
few voxels a view actually draws, never to the whole series.

No file handle stays open after the read. An open handle on Windows stops the
Editor from renaming or deleting the file, which is why this is not a memory
map even for uncompressed files: an uncompressed read is disk-speed anyway.

Qt-free: the worker that calls :meth:`VolumeSource.stream` lives in
``bidsmgr/workers``; everything here is plain numpy and nibabel.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from ..compute.shapes import GridBox

log = logging.getLogger(__name__)

#: NIfTI intent codes that mean "each voxel is a colour", not a time series.
_INTENT_RGB = {2003, 2004}          # NIFTI_INTENT_RGB_VECTOR, RGBA_VECTOR
_INTENT_VECTOR = 1007               # NIFTI_INTENT_VECTOR

#: Frames read per chunk while streaming. Small enough that the first chunk
#: lands in tens of milliseconds, large enough that the Python loop is not the
#: cost.
_CHUNK_BYTES = 8 * 1024 * 1024


class StreamCancelled(Exception):
    """The read was abandoned because the viewer moved on."""


@dataclass(frozen=True)
class HeaderFacts:
    """What the header says, as plain values (for the header inspector)."""

    shape: tuple[int, ...]
    zooms: tuple[float, ...]
    disk_dtype: str
    space_unit: str
    time_unit: str
    qform_code: int
    sform_code: int
    intent_code: int
    intent_name: str
    descrip: str
    slice_code: int
    slice_duration: float
    slice_start: int
    slice_end: int
    toffset: float
    cal_min: float
    cal_max: float
    slope: float
    inter: float
    extension_codes: tuple[int, ...]
    format: str
    #: The axis the slices were acquired along (``dim_info``), -1 unknown.
    slice_dim: int = -1


@dataclass
class VolumeSource:
    """One image on disk, read lazily. Build with :func:`open_volume`."""

    path: Path
    #: The image's affine as the file states it (sform, else qform).
    affine: np.ndarray
    #: The three spatial dimensions (a 2-D image is padded with a 1).
    spatial: tuple[int, int, int]
    #: Voxel size in mm along the three spatial axes.
    zooms3: tuple[float, float, float]
    #: Drawable frames: a 4-D series has its 4th dimension; 5-D is flattened
    #: (time fastest); a colour image is ONE frame whatever its layout.
    n_frames: int
    #: Raw frames stored on disk (``n_frames`` times the channels for an
    #: intent-coded colour image, whose channels are separate raw frames).
    n_raw: int
    disk_dtype: np.dtype
    slope: float
    inter: float
    #: Colour image: frames are ``(X, Y, Z, C)`` instead of ``(X, Y, Z)``.
    is_rgb: bool
    channels: int
    #: How colour channels are stored: ``structured`` (one record per voxel,
    #: RGB24) or ``planes`` (one raw frame per channel, intent-coded).
    rgb_layout: str
    #: Repetition time from the header (seconds), when it declares one.
    header_tr: Optional[float]
    facts: HeaderFacts
    #: Byte offset of the voxel data in the file, and whether the streaming
    #: reader can use it (NIfTI-1/2). Other formats load in one call.
    vox_offset: int = 0
    streamable: bool = True
    #: Frames the FILE holds. ``n_frames`` drops below it when a series is
    #: larger than the memory budget and only its start is read.
    n_frames_total: int = 0
    #: True when the memory budget stopped the read short of the last frame.
    truncated: bool = False
    #: Built from an array (a QC map, a drawn box), not read from a file:
    #: everything is already in memory and there is nothing to stream.
    in_memory: bool = False
    #: What the values ARE, with their unit, for a colour bar ("tSNR (mean
    #: / SD)"); "" when unknown.
    quantity: str = ""
    #: What a computed source knows about itself (a quality map's summary,
    #: how it was made, how to read it).
    notes: dict = field(default_factory=dict)
    #: b-value of every volume of a diffusion series (from the ``.bval``
    #: beside it), when the dataset gives them and they match the volumes.
    bvals: Optional[np.ndarray] = None
    #: A source that IS a shape (the MRS voxel): its exact box, which every
    #: view draws as geometry rather than sampling the array onto pixels.
    box: Optional["GridBox"] = None
    # -- load state (written by the streaming thread) ---------------------
    _raw: Optional[np.ndarray] = field(default=None, repr=False)
    loaded_raw: int = 0
    error: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _range_cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def inv_affine(self) -> np.ndarray:
        cached = self._range_cache.get("_inv")
        if cached is None:
            cached = np.linalg.inv(self.affine)
            self._range_cache["_inv"] = cached
        return cached

    def world_to_voxel(self, world) -> np.ndarray:
        """Continuous voxel coordinates of a world point."""
        p = np.array([*list(world)[:3], 1.0], dtype=float)
        return (self.inv_affine @ p)[:3]

    def voxel_to_world(self, voxel) -> np.ndarray:
        p = np.array([*list(voxel)[:3], 1.0], dtype=float)
        return (self.affine @ p)[:3]

    def center_world(self) -> np.ndarray:
        """The world position of the middle voxel (where a view opens)."""
        mid = [s // 2 for s in self.spatial]
        return self.voxel_to_world(mid)

    def clamp_voxel(self, voxel) -> tuple[int, int, int]:
        out = []
        for axis, dim in enumerate(self.spatial):
            v = int(round(float(voxel[axis])))
            out.append(max(0, min(v, dim - 1)))
        return (out[0], out[1], out[2])

    # ------------------------------------------------------------------
    # Load state
    # ------------------------------------------------------------------

    @property
    def is_4d(self) -> bool:
        return self.n_frames > 1

    @property
    def loaded_frames(self) -> int:
        """How many DRAWABLE frames are already in memory."""
        if self._raw is None:
            return 0
        if self.is_rgb and self.rgb_layout == "planes":
            # Every channel of a frame has to be in before it is drawable.
            per = self.channels
            frames_per_channel = self.n_raw // max(per, 1)
            if self.loaded_raw >= self.n_raw:
                return self.n_frames
            # Channels are stored one after the other (all of channel 0 first).
            done_channels = self.loaded_raw // max(frames_per_channel, 1)
            return self.n_frames if done_channels >= per else 0
        return min(self.loaded_raw, self.n_frames)

    @property
    def fully_loaded(self) -> bool:
        return self._raw is not None and self.loaded_raw >= self.n_raw

    def frame_ready(self, t: int) -> bool:
        return 0 <= t < self.loaded_frames

    @property
    def resident_bytes(self) -> int:
        return 0 if self._raw is None else int(self._raw.nbytes)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def stream(
        self,
        *,
        cancel: Optional[Callable[[], bool]] = None,
        progress: Optional[Callable[[int, int], None]] = None,
        budget_bytes: Optional[int] = None,
    ) -> None:
        """Read every raw frame, front to back. Runs on a worker thread.

        ``progress(loaded_raw, n_raw)`` is called after each chunk, so the
        caller can draw frame 0 long before the last frame is in. Raises
        :class:`StreamCancelled` when ``cancel()`` turns true.

        ``budget_bytes`` caps what a scalar series may hold: when the frames
        do not fit, the FIRST ones that do are read, ``n_frames`` becomes
        that count and ``truncated`` says so. Better a usable start of the
        run than a machine swapping itself to a halt.
        """
        if self.in_memory:
            return
        if budget_bytes and self.streamable and not self.is_rgb and self.n_raw > 1:
            frame_bytes = int(np.prod(self.spatial)) * self.disk_dtype.itemsize
            fit = max(1, int(budget_bytes) // max(frame_bytes, 1))
            if fit < self.n_raw:
                self.n_raw = fit
                self.n_frames = fit
                self.truncated = True
        if self.streamable:
            self._stream_nifti(cancel, progress)
        else:
            self._load_whole(cancel, progress)

    def _allocate(self) -> np.ndarray:
        x, y, z = self.spatial
        # (raw frame, Z, Y, X) in C order: each raw frame is then exactly the
        # Fortran-ordered block the file holds, so a chunk is one memcpy, and
        # ``raw[t].T`` is the (X, Y, Z) frame with no copy.
        return np.empty((self.n_raw, z, y, x), dtype=self.disk_dtype)

    def _stream_nifti(self, cancel, progress) -> None:
        from nibabel.openers import ImageOpener

        raw = self._allocate()
        frame_items = int(np.prod(self.spatial))
        frame_bytes = frame_items * self.disk_dtype.itemsize
        per_chunk = max(1, _CHUNK_BYTES // max(frame_bytes, 1))
        with self._lock:
            self._raw = raw
            self.loaded_raw = 0
        with ImageOpener(str(self.path), "rb") as fobj:
            fobj.seek(self.vox_offset)
            t = 0
            while t < self.n_raw:
                if cancel is not None and cancel():
                    raise StreamCancelled(str(self.path))
                n = min(per_chunk, self.n_raw - t)
                want = n * frame_bytes
                buf = fobj.read(want)
                if len(buf) < want:
                    raise OSError(
                        f"{self.path.name} ends early: expected {self.n_raw} "
                        f"frames of {frame_bytes} bytes, file holds {t} and "
                        f"{len(buf)} bytes"
                    )
                block = np.frombuffer(buf, dtype=self.disk_dtype, count=n * frame_items)
                raw[t:t + n] = block.reshape(n, *raw.shape[1:])
                t += n
                self.loaded_raw = t
                if progress is not None:
                    progress(t, self.n_raw)

    def _load_whole(self, cancel, progress) -> None:
        """Formats without a known byte layout: one nibabel read."""
        import nibabel as nib

        img = nib.load(str(self.path))
        data = np.asanyarray(img.dataobj)
        if cancel is not None and cancel():
            raise StreamCancelled(str(self.path))
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[:, :, None]
        # Flatten everything past the third axis into frames (time fastest),
        # then move frames to the front and reverse the spatial axes to match
        # the (frame, Z, Y, X) layout of the streamed path.
        x, y, z = self.spatial
        flat = data.reshape((x, y, z, -1), order="F")
        raw = np.ascontiguousarray(np.moveaxis(flat, 3, 0).transpose(0, 3, 2, 1))
        if raw.dtype != self.disk_dtype:
            self.disk_dtype = raw.dtype
        with self._lock:
            self._raw = raw
            self.loaded_raw = raw.shape[0]
        if progress is not None:
            progress(self.loaded_raw, self.n_raw)

    def release(self) -> None:
        """Drop the voxels (the header facts stay)."""
        with self._lock:
            self._raw = None
            self.loaded_raw = 0
        self._range_cache.clear()

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def raw_frame(self, t: int) -> Optional[np.ndarray]:
        """Frame ``t`` in the file's own dtype as an (X, Y, Z[, C]) view.

        ``None`` while it is still being read. No copy is made for a scalar
        image; a colour image built from planes is stacked.
        """
        if not self.frame_ready(t):
            return None
        raw = self._raw
        if raw is None:
            return None
        if self.is_rgb and self.rgb_layout == "planes":
            per_channel = self.n_raw // self.channels
            planes = [raw[t + c * per_channel].T for c in range(self.channels)]
            return np.stack(planes, axis=-1)
        frame = raw[t].T
        if self.is_rgb and self.rgb_layout == "structured":
            from numpy.lib import recfunctions as rfn

            return rfn.structured_to_unstructured(frame)
        return frame

    def scale(self, values: np.ndarray) -> np.ndarray:
        """Raw values to data units, as float32."""
        out = np.asarray(values, dtype=np.float32)
        if self.slope != 1.0 or self.inter != 0.0:
            out = out * np.float32(self.slope) + np.float32(self.inter)
        return out

    def frame(self, t: int) -> Optional[np.ndarray]:
        """Frame ``t`` in data units (float32). Converts the WHOLE frame:
        draw code should slice :meth:`raw_frame` and :meth:`scale` the slice."""
        raw = self.raw_frame(t)
        return None if raw is None else self.scale(raw)

    def voxel_series(self, voxel) -> np.ndarray:
        """The values of one voxel across every frame read so far."""
        raw = self._raw
        if raw is None or self.is_rgb:
            return np.empty(0, dtype=np.float32)
        i, j, k = self.clamp_voxel(voxel)
        n = self.loaded_frames
        return self.scale(raw[:n, k, j, i])

    def value_at(self, voxel, t: int):
        """The value (or colour vector) at a voxel of frame ``t``, or None."""
        raw = self.raw_frame(t)
        if raw is None:
            return None
        i, j, k = self.clamp_voxel(voxel)
        v = raw[i, j, k]
        if self.is_rgb:
            return np.asarray(v, dtype=np.float32)
        return float(self.scale(np.asarray(v)))

    def _sample(self, t: int) -> Optional[np.ndarray]:
        """About 500,000 finite values of frame ``t`` in data units, taken by
        striding the VIEW on each spatial axis and copying only the sample:
        a frame is a transposed view, and flattening it first copied every
        voxel (100 ms for a 0.65 mm T1, on the GUI thread). Cached per frame.
        For a colour image, one row per voxel."""
        key = ("sample", t)
        if key in self._range_cache:
            return self._range_cache[key]
        raw = self.raw_frame(t)
        if raw is None:
            return None
        arr = np.asarray(raw)
        if arr.size > 500_000:
            step = int(np.ceil((arr.size / 500_000) ** (1.0 / 3.0)))
            arr = arr[(slice(None, None, step),) * min(arr.ndim, 3)]
        flat = arr.reshape(-1, arr.shape[-1]) if self.is_rgb else arr.reshape(-1)
        vals = self.scale(flat)
        vals = vals[np.isfinite(vals).all(axis=-1)] if self.is_rgb else vals[np.isfinite(vals)]
        self._range_cache[key] = vals
        return vals

    def robust_range(self, t: int, *, lo_pct: float = 1.0, hi_pct: float = 99.0):
        """A display window for frame ``t``: percentiles of a strided sample.

        Strided so a large frame stays instant; cached per frame. ``None``
        while the frame is not in yet.
        """
        key = ("range", t, lo_pct, hi_pct)
        if key in self._range_cache:
            return self._range_cache[key]
        vals = self._sample(t)
        if vals is None:
            return None
        if vals.size == 0:
            out = (0.0, 1.0)
        elif self.is_rgb:
            hi = float(np.max(vals))
            out = (0.0, 255.0 if hi > 1.0 + 1e-6 else 1.0)
        else:
            lo, hi = np.percentile(vals, (lo_pct, hi_pct))
            if not hi > lo:
                hi = lo + 1.0
            out = (float(lo), float(hi))
        self._range_cache[key] = out
        return out

    def data_range(self, t: int) -> Optional[tuple[float, float]]:
        """Smallest and largest value of frame ``t`` (of the sample)."""
        vals = self._sample(t)
        if vals is None:
            return None
        if vals.size == 0:
            return (0.0, 1.0)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        return (lo, hi if hi > lo else lo + 1.0)

    def histogram(self, t: int, bins: int = 96) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """``(counts, edges)`` of frame ``t`` over its data range, from the
        strided sample (what a window control draws behind its handles).
        The zero bin of a masked image is left in: the control draws counts
        on a log scale, so it does not swamp the rest."""
        key = ("hist", t, bins)
        if key in self._range_cache:
            return self._range_cache[key]
        vals = self._sample(t)
        rng = self.data_range(t)
        if vals is None or rng is None or self.is_rgb:
            return None
        counts, edges = np.histogram(vals, bins=bins, range=rng)
        out = (counts.astype(np.float64), edges)
        self._range_cache[key] = out
        return out

    def series_range(self, *, samples: int = 8) -> Optional[tuple[float, float]]:
        """A window that suits the whole series (a few frames sampled).

        One frame is a poor guide for PET, whose first frames hold almost no
        counts; a window from them would saturate everything after.
        """
        n = self.loaded_frames
        if n == 0:
            return None
        picks = sorted({int(round(v)) for v in np.linspace(0, n - 1, min(samples, n))})
        lows, highs = [], []
        for t in picks:
            r = self.robust_range(t)
            if r is not None:
                lows.append(r[0])
                highs.append(r[1])
        if not lows:
            return None
        return (float(np.median(lows)), float(np.max(highs)))


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def _unit_names(header) -> tuple[str, str]:
    try:
        space, time = header.get_xyzt_units()
        return str(space or ""), str(time or "")
    except Exception:  # noqa: BLE001 - not every format has units
        return "", ""


def _num(header, key, default=0.0):
    try:
        value = header[key]
        value = np.asarray(value).item()
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def _facts(img, fmt: str) -> HeaderFacts:
    header = img.header
    space_unit, time_unit = _unit_names(header)
    try:
        slope, inter = header.get_slope_inter()
    except Exception:  # noqa: BLE001
        slope, inter = None, None
    try:
        exts = tuple(int(e.get_code()) for e in getattr(header, "extensions", []))
    except Exception:  # noqa: BLE001
        exts = ()
    try:
        intent_name = str(header.get_intent()[0])
    except Exception:  # noqa: BLE001
        intent_name = ""
    try:
        descrip = header["descrip"].tobytes().split(b"\x00", 1)[0].decode(
            "latin-1", errors="replace",
        )
    except Exception:  # noqa: BLE001
        descrip = ""
    try:
        slice_dim = header.get_dim_info()[2]
        slice_dim = -1 if slice_dim is None else int(slice_dim)
    except Exception:  # noqa: BLE001 - not every format has dim_info
        slice_dim = -1
    return HeaderFacts(
        shape=tuple(int(s) for s in img.shape),
        zooms=tuple(float(z) for z in header.get_zooms()),
        disk_dtype=str(header.get_data_dtype()),
        space_unit=space_unit,
        time_unit=time_unit,
        qform_code=int(_num(header, "qform_code")),
        sform_code=int(_num(header, "sform_code")),
        intent_code=int(_num(header, "intent_code")),
        intent_name=intent_name,
        descrip=descrip,
        slice_code=int(_num(header, "slice_code")),
        slice_duration=_num(header, "slice_duration"),
        slice_start=int(_num(header, "slice_start")),
        slice_end=int(_num(header, "slice_end")),
        toffset=_num(header, "toffset"),
        cal_min=_num(header, "cal_min"),
        cal_max=_num(header, "cal_max"),
        slope=1.0 if slope is None or not np.isfinite(slope) else float(slope),
        inter=0.0 if inter is None or not np.isfinite(inter) else float(inter),
        extension_codes=exts,
        format=fmt,
        slice_dim=slice_dim,
    )


def open_volume(path: Path) -> VolumeSource:
    """Read a volume's header. Cheap: no voxel is touched."""
    import nibabel as nib

    path = Path(path)
    img = nib.load(str(path))
    header = img.header
    fmt = type(img).__name__
    facts = _facts(img, fmt)
    shape = tuple(int(s) for s in img.shape)
    if len(shape) < 2:
        raise ValueError(f"{path.name} is not an image (shape {shape})")
    spatial = (shape + (1, 1))[:3]
    disk_dtype = np.dtype(header.get_data_dtype())
    extra = shape[3:]
    n_raw = int(np.prod(extra)) if extra else 1

    zooms = tuple(float(z) for z in header.get_zooms())
    zooms3 = tuple(
        (z if z and np.isfinite(z) and z > 0 else 1.0) for z in (zooms + (1.0, 1.0, 1.0))[:3]
    )

    structured = disk_dtype.fields is not None
    intent = facts.intent_code
    is_rgb = False
    channels = 1
    rgb_layout = ""
    n_frames = n_raw
    if structured:
        channels = len(disk_dtype.fields)
        is_rgb = channels in (3, 4)
        rgb_layout = "structured"
        n_frames = n_raw if not is_rgb else max(1, n_raw)
    elif len(extra) >= 2 and extra[-1] in (3, 4) and (
        intent in _INTENT_RGB or intent == _INTENT_VECTOR
    ):
        # NIfTI stores a vector image with the components along the 5th
        # dimension, so each channel is a separate raw frame.
        channels = int(extra[-1])
        is_rgb = True
        rgb_layout = "planes"
        n_frames = int(np.prod(extra[:-1]))

    tr = None
    if len(shape) >= 4 and len(zooms) >= 4 and zooms[3] and zooms[3] > 0:
        _space, time_unit = _unit_names(header)
        scale = {"msec": 1e-3, "usec": 1e-6}.get(time_unit, 1.0)
        tr = float(zooms[3]) * scale

    # Where the voxels start, read from the array PROXY: the header object a
    # loaded image carries has its ``vox_offset`` reset to 0 (nibabel
    # recomputes it at save time), so trusting it would read the header
    # bytes as voxels. Only a Fortran-ordered single-file NIfTI is streamed.
    vox_offset = 0
    streamable = False
    proxy = getattr(img, "dataobj", None)
    try:
        if (
            isinstance(img, (nib.Nifti1Image, nib.Nifti2Image))
            and getattr(proxy, "order", "F") == "F"
            and getattr(proxy, "offset", None) is not None
        ):
            vox_offset = int(proxy.offset)
            disk_dtype = np.dtype(proxy.dtype)
            streamable = True
    except Exception:  # noqa: BLE001
        streamable = False
    # Scaling lives on the proxy too: nibabel moves scl_slope / scl_inter off
    # the loaded header copy, which reads (nan, nan) and would draw a scaled
    # image in raw units.
    slope, inter = facts.slope, facts.inter
    try:
        p_slope = float(getattr(proxy, "slope", slope))
        p_inter = float(getattr(proxy, "inter", inter))
        if np.isfinite(p_slope) and p_slope != 0:
            slope = p_slope
        if np.isfinite(p_inter):
            inter = p_inter
    except (TypeError, ValueError):
        pass
    facts = dataclasses.replace(facts, slope=slope, inter=inter)

    return VolumeSource(
        path=path,
        affine=np.asarray(img.affine, dtype=float),
        spatial=spatial,
        zooms3=zooms3,
        n_frames=max(1, n_frames),
        n_frames_total=max(1, n_frames),
        n_raw=max(1, n_raw),
        disk_dtype=disk_dtype,
        slope=slope,
        inter=inter,
        is_rgb=is_rgb,
        channels=channels,
        rgb_layout=rgb_layout,
        header_tr=tr,
        facts=facts,
        vox_offset=vox_offset,
        streamable=streamable,
    )


def array_volume(data: np.ndarray, affine, *, path: Path, name: str = "") -> VolumeSource:
    """A source over an array already in memory, in data units.

    ``data`` is (X, Y, Z) or (X, Y, Z, T) on the voxel grid ``affine``
    describes. What a computed map (mean, SD, temporal SNR) or a drawn box
    (an MRS voxel) needs to be a layer like any file. ``path`` is the file it
    was derived from, so the footer and a saved scene can say so.
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"an array volume is 3-D or 4-D, not {arr.ndim}-D")
    x, y, z, t = (int(v) for v in arr.shape)
    # (frame, Z, Y, X) in C order: the layout the streamed reader produces.
    raw = np.ascontiguousarray(np.moveaxis(arr, 3, 0).transpose(0, 3, 2, 1))
    affine = np.asarray(affine, dtype=float)
    zooms3 = tuple(float(v) or 1.0 for v in np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))
    facts = HeaderFacts(
        shape=(x, y, z) if t == 1 else (x, y, z, t), zooms=zooms3,
        disk_dtype=str(raw.dtype), space_unit="mm", time_unit="", qform_code=0,
        sform_code=0, intent_code=0, intent_name="", descrip=name, slice_code=0,
        slice_duration=0.0, slice_start=0, slice_end=0, toffset=0.0, cal_min=0.0,
        cal_max=0.0, slope=1.0, inter=0.0, extension_codes=(), format="computed",
    )
    src = VolumeSource(
        path=Path(path), affine=affine, spatial=(x, y, z), zooms3=zooms3,
        n_frames=t, n_frames_total=t, n_raw=t, disk_dtype=raw.dtype, slope=1.0,
        inter=0.0, is_rgb=False, channels=1, rgb_layout="", header_tr=None,
        facts=facts, streamable=False, in_memory=True,
    )
    src._raw = raw
    src.loaded_raw = t
    return src


def memory_budget_bytes(setting_mb: int = 0) -> int:
    """What a 4-D series may hold: the user's setting, else half of the
    memory that is free right now (4 GB when that cannot be measured)."""
    if setting_mb and setting_mb > 0:
        return int(setting_mb) * 1024 * 1024
    try:
        import psutil

        return int(psutil.virtual_memory().available // 2)
    except Exception:  # noqa: BLE001 - a guess beats refusing to open
        return 4 * 1024 ** 3


__all__ = [
    "HeaderFacts", "StreamCancelled", "VolumeSource", "array_volume",
    "memory_budget_bytes", "open_volume",
]

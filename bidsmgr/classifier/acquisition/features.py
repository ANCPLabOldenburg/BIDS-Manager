"""What a DICOM series says about how it was acquired.

Read during the scan, from the scan's own header pass, never from dcm2niix:

* :func:`file_facts` runs on EVERY file in the pass that already opens it
  (``inventory/mri_dicom._read_one``) and keeps only what varies across a
  series and is cheap: echo time, b-value and direction, orientation, the
  acquisition and temporal indices, the image type.
* :func:`series_facts` runs on ONE file per series and keeps what is the
  same across it, including what only the vendor's private header has: the
  Siemens CSA header (phase-encoding polarity, slice times, mosaic size,
  b-value) and the Siemens protocol (multiband factor, shim currents,
  magnetisation transfer), classic or XA enhanced.
* :func:`build` folds both into :class:`SeriesFeatures`, the plain data the
  rulebook scores.

Missing is missing: a field a vendor does not write stays ``None`` and
counts as no evidence, never as evidence against. Nothing identifying is
kept (no names, no dates, no UIDs but the series').
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

#: The phase-encoding direction labels of the ``dir`` entity, by the
#: patient axis (LPS) and the direction the encoding runs.
_DIR_LABELS = {("x", 1): "RL", ("x", -1): "LR", ("y", 1): "AP", ("y", -1): "PA",
               ("z", 1): "IS", ("z", -1): "SI"}


def _get(ds, name: str, default=None):
    try:
        value = getattr(ds, name, default)
    except Exception:  # noqa: BLE001 - a malformed element is no value
        return default
    return default if value in (None, "") else value


def _float(value) -> Optional[float]:
    try:
        if isinstance(value, (list, tuple)) or (hasattr(value, "__iter__")
                                                and not isinstance(value, (str, bytes))):
            value = list(value)[0]
        out = float(value)
    except (TypeError, ValueError, IndexError):
        return None
    return out if math.isfinite(out) else None


def _tokens(value) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [t for t in re.split(r"[\\\s]+", value.upper()) if t]
    try:
        return [str(v).strip().upper() for v in value if str(v).strip()]
    except TypeError:
        return [str(value).strip().upper()]


def _functional(ds, seq: str, attr: str):
    """``attr`` from the enhanced functional groups (shared first, then the
    first frame), for files that keep their parameters there."""
    for groups in ("SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"):
        items = _get(ds, groups)
        if not items:
            continue
        inner = _get(items[0], seq)
        if inner:
            value = _get(inner[0], attr)
            if value is not None:
                return value
    return None


# ---------------------------------------------------------------------------
# Per file (cheap)
# ---------------------------------------------------------------------------


def _bvalue(ds) -> Optional[float]:
    for value in (_get(ds, "DiffusionBValue"),
                  _functional(ds, "MRDiffusionSequence", "DiffusionBValue")):
        if value is not None:
            return _float(value)
    tags = {(0x0019, 0x100C): None,          # Siemens
            (0x0043, 0x1039): "ge",          # GE: first value, minus the 1e9 flag
            (0x2001, 0x1003): None}          # Philips
    for tag, kind in tags.items():
        if tag in ds:
            try:
                raw = ds[tag].value
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("ascii", "ignore").split("\\")
                value = _float(raw)
                if value is not None and kind == "ge":
                    value = value % 1e9
                if value is not None:
                    return value
            except Exception:  # noqa: BLE001
                continue
    return None


def _contrast_agent(ds) -> str:
    """The contrast agent the header names ("24ML MULTIHANCE"), or ""."""
    agent = str(_get(ds, "ContrastBolusAgent", "") or "").strip()
    if not agent:
        seq = _get(ds, "ContrastBolusAgentSequence")
        if seq:
            agent = str(_get(seq[0], "CodeMeaning", "") or "").strip()
    if agent.lower() in ("none", "no", "n/a", "0", "-"):
        return ""
    return agent


def _pixel_spacing(ds) -> Optional[float]:
    """The in-plane voxel size in mm (the mean of the two spacings)."""
    value = _get(ds, "PixelSpacing") or _functional(ds, "PixelMeasuresSequence",
                                                     "PixelSpacing")
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return round(sum(values) / len(values), 3) if values else None


def _bvector(ds) -> Optional[tuple]:
    value = _get(ds, "DiffusionGradientOrientation") or \
        _functional(ds, "MRDiffusionSequence", "DiffusionGradientDirectionSequence")
    if value is not None and hasattr(value, "__len__") and len(value) and \
            hasattr(value[0], "DiffusionGradientOrientation"):
        value = value[0].DiffusionGradientOrientation
    if value is None and (0x0019, 0x100E) in ds:      # Siemens
        try:
            value = ds[(0x0019, 0x100E)].value
        except Exception:  # noqa: BLE001
            value = None
    try:
        vec = tuple(round(float(v), 2) for v in value)
    except (TypeError, ValueError):
        return None
    return vec if len(vec) == 3 and any(abs(v) > 1e-3 for v in vec) else None


def file_facts(ds) -> dict:
    """What one file of a series says that may differ from its neighbours."""
    te = _get(ds, "EchoTime")
    if te is None:
        te = _functional(ds, "MREchoSequence", "EffectiveEchoTime")
    orientation = _get(ds, "ImageOrientationPatient")
    position = _get(ds, "ImagePositionPatient")
    if orientation is None:
        orientation = _functional(ds, "PlaneOrientationSequence", "ImageOrientationPatient")
    if position is None:
        position = _functional(ds, "PlanePositionSequence", "ImagePositionPatient")
    try:
        orient = tuple(round(float(v), 1) for v in orientation) if orientation else None
    except (TypeError, ValueError):
        orient = None
    try:
        pos = tuple(round(float(v), 1) for v in position) if position else None
    except (TypeError, ValueError):
        pos = None
    return {
        "te": _float(te),
        "echo": _float(_get(ds, "EchoNumbers")),
        "b": _bvalue(ds),
        "g": _bvector(ds),
        "orient": orient,
        "pos": pos,
        "acq": _float(_get(ds, "AcquisitionNumber")),
        "temporal": _float(_get(ds, "TemporalPositionIdentifier")),
        "frames": _float(_get(ds, "NumberOfFrames")),
        "image_type": _tokens(_get(ds, "ImageType")),
    }


# ---------------------------------------------------------------------------
# Per series (one file, the private headers too)
# ---------------------------------------------------------------------------


def parse_ascconv(text: str) -> dict[str, str]:
    """The Siemens protocol (``### ASCCONV BEGIN`` ... ``END``) as
    ``{key: value}``."""
    found = re.search(r"### ASCCONV BEGIN.*?###(.*?)### ASCCONV END ###", text, re.S)
    out: dict[str, str] = {}
    if not found:
        return out
    for line in found.group(1).splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key.strip()] = value.split("#", 1)[0].strip().strip('"')
    return out


def _siemens_private(ds) -> tuple[dict, dict]:
    """``(csa image header, protocol)`` of a Siemens file, classic or XA."""
    csa: dict = {}
    protocol: dict = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from nibabel.nicom import csareader
    except Exception:  # noqa: BLE001
        csareader = None
    if csareader is not None:
        for tag, target in (((0x0029, 0x1010), "image"), ((0x0029, 0x1020), "series")):
            if tag not in ds:
                continue
            try:
                header = csareader.read(ds[tag].value)
            except Exception:  # noqa: BLE001 - not every blob is a CSA header
                continue
            for name, entry in (header.get("tags") or {}).items():
                items = [i for i in entry.get("items", []) if i != ""]
                if not items:
                    continue
                if target == "series" and name == "MrPhoenixProtocol":
                    protocol.update(parse_ascconv(str(items[0])))
                elif target == "image":
                    csa[name] = items if len(items) > 1 else items[0]
    if not protocol:
        # XA: the protocol raw, in a private element of the shared groups.
        for elem in ds.iterall():
            if elem.tag.group % 2 == 1 and isinstance(elem.value, (bytes, bytearray)) \
                    and b"### ASCCONV BEGIN" in elem.value:
                protocol.update(parse_ascconv(bytes(elem.value).decode("latin-1", "replace")))
                break
    return csa, protocol


def _xa_polarity(ds) -> Optional[float]:
    """Siemens XA keeps PhaseEncodingDirectionPositive in a private element
    of its frames (SIEMENS MR SDI 02, 0021,xx1C)."""
    try:
        for elem in ds.iterall():
            if elem.tag.group == 0x0021 and (elem.tag.element & 0xFF) == 0x1C and \
                    elem.tag.element > 0x1000 and \
                    "SDI" in str(getattr(elem, "private_creator", "") or ""):
                return _float(elem.value)
    except Exception:  # noqa: BLE001
        return None
    return None


def _shim(protocol: dict) -> Optional[tuple]:
    keys = ["sGRADSPEC.asGPAData[0].lOffsetX", "sGRADSPEC.asGPAData[0].lOffsetY",
            "sGRADSPEC.asGPAData[0].lOffsetZ"] + [f"sGRADSPEC.alShimCurrent[{i}]"
                                                  for i in range(5)]
    values = [protocol.get(k) for k in keys]
    if not any(v not in (None, "") for v in values):
        return None
    out = []
    for v in values:
        try:
            out.append(int(float(v)) if v not in (None, "") else 0)
        except ValueError:
            out.append(0)
    return tuple(out)


def series_facts(ds) -> dict:
    """What one file says about its whole series (constant across it)."""
    manufacturer = str(_get(ds, "Manufacturer", "") or "")
    facts: dict[str, Any] = {
        "manufacturer": manufacturer,
        "model": str(_get(ds, "ManufacturerModelName", "") or ""),
        "field_strength": _float(_get(ds, "MagneticFieldStrength")),
        "software": str(_get(ds, "SoftwareVersions", "") or ""),
        "description": str(_get(ds, "SeriesDescription", "") or ""),
        "dicom_modality": str(_get(ds, "Modality", "") or "").strip().upper(),
        "sop_class": str(_get(ds, "SOPClassUID", "") or ""),
        "protocol": str(_get(ds, "ProtocolName", "") or ""),
        "scanning_sequence": _tokens(_get(ds, "ScanningSequence")),
        "sequence_variant": _tokens(_get(ds, "SequenceVariant")),
        "scan_options": _tokens(_get(ds, "ScanOptions")),
        "sequence_name": str(_get(ds, "SequenceName", "") or
                             _get(ds, "PulseSequenceName", "") or ""),
        "acquisition_type": str(_get(ds, "MRAcquisitionType", "") or
                                _functional(ds, "MRFOVGeometrySequence",
                                            "MRAcquisitionType") or "").upper(),
        "echo_pulse": str(_get(ds, "EchoPulseSequence", "") or "").upper(),
        "epi_flag": str(_get(ds, "EchoPlanarPulseSequence", "") or "").upper(),
        "tr": _float(_get(ds, "RepetitionTime") or
                     _functional(ds, "MRTimingAndRelatedParametersSequence",
                                 "RepetitionTime")),
        "ti": _float(_get(ds, "InversionTime") or
                     _functional(ds, "MRModifierSequence", "InversionTimes")),
        "flip": _float(_get(ds, "FlipAngle") or
                       _functional(ds, "MRTimingAndRelatedParametersSequence",
                                   "FlipAngle")),
        "pe_axis": str(_get(ds, "InPlanePhaseEncodingDirection", "") or
                       _functional(ds, "MRFOVGeometrySequence",
                                   "InPlanePhaseEncodingDirection") or "").upper(),
        "comments": str(_get(ds, "ImageComments", "") or ""),
        "contrast_agent": _contrast_agent(ds),
        "echo_train_length": _float(_get(ds, "EchoTrainLength") or
                                    _functional(ds, "MRTimingAndRelatedParametersSequence",
                                                "EchoTrainLength")),
        "pixel_spacing": _pixel_spacing(ds),
        "rows": _float(_get(ds, "Rows")),
        "columns": _float(_get(ds, "Columns")),
        "slice_thickness": _float(_get(ds, "SliceThickness")),
        "n_frames": _float(_get(ds, "NumberOfFrames")),
        "temporal_positions": _float(_get(ds, "NumberOfTemporalPositions")),
        "pe_positive": None, "mosaic_images": None, "slice_timing": False,
        "multiband": None, "shim": None, "mt": False, "csa_b": None,
        "enhanced_tes": [], "enhanced_bs": [], "enhanced_volumes": None,
        "enhanced_directions": 0, "protocol_echoes": None, "asl_mode": None,
    }
    if "siemens" in manufacturer.lower():
        csa, protocol = _siemens_private(ds)
        polarity = csa.get("PhaseEncodingDirectionPositive")
        if polarity not in (None, ""):
            facts["pe_positive"] = _float(polarity)
        facts["mosaic_images"] = _float(csa.get("NumberOfImagesInMosaic"))
        facts["slice_timing"] = "MosaicRefAcqTimes" in csa
        facts["csa_b"] = _float(csa.get("B_value"))
        if facts["pe_positive"] is None:
            facts["pe_positive"] = _xa_polarity(ds)
        mb = _float(protocol.get("sSliceAcceleration.lMultiBandFactor"))
        if not mb or mb <= 1:
            # CMRR sequences keep it in their own memory block; the image
            # comment says it ("Unaliased MB4/PE3/LB").
            m = re.search(r"\bmb ?(\d+)", (facts["comments"] + " " + facts["sequence_name"]
                                            + " " + facts["description"]).lower())
            mb = _float(m.group(1)) if m else mb
        facts["multiband"] = mb
        facts["shim"] = _shim(protocol)
        # The labelling the protocol ran: 1 none, 2 pulsed, 4 pseudo-continuous.
        facts["asl_mode"] = _float(protocol.get("sAsl.ulMode"))
        # The echoes the protocol acquired: a phase difference image is
        # reconstructed from two and carries only one echo time itself.
        facts["protocol_echoes"] = _float(protocol.get("lContrasts"))
        mtc = protocol.get("sPrepPulses.ucMTC") or protocol.get("sPrepPulses.ucMtcMode")
        facts["mt"] = mtc not in (None, "", "0", "0x0", "0x1") or \
            "MTC" in facts["sequence_variant"] or "MT" in facts["scan_options"]
        if facts["pe_axis"] == "" and protocol.get("sKSpace.ucPhaseEncodingDirection"):
            facts["pe_axis"] = "COL"
    else:
        facts["mt"] = "MT" in facts["scan_options"] or "MTC" in facts["sequence_variant"]
    if "philips" in manufacturer.lower() and facts["epi_flag"] != "YES":
        # Classic Philips headers name no EPI readout; the EPI factor says it.
        try:
            factor = _float(ds[(0x2001, 0x1013)].value) if (0x2001, 0x1013) in ds else None
        except Exception:  # noqa: BLE001
            factor = None
        if factor and factor > 1:
            facts["epi_flag"] = "YES"
    # An enhanced file holds the whole series: what varies is in its frames.
    frames = _get(ds, "PerFrameFunctionalGroupsSequence")
    if frames:
        tes, bs, temporal, diffusion = set(), set(), set(), set()
        for item in frames:
            echo = _get(item, "MREchoSequence")
            if echo:
                te = _float(_get(echo[0], "EffectiveEchoTime"))
                if te is not None:
                    tes.add(round(te, 2))
            diff = _get(item, "MRDiffusionSequence")
            if diff:
                b = _float(_get(diff[0], "DiffusionBValue"))
                if b is not None:
                    bs.add(round(b / 5) * 5)
                vec = _get(diff[0], "DiffusionGradientDirectionSequence")
                vec = _get(vec[0], "DiffusionGradientOrientation") if vec else None
                try:
                    vec = tuple(round(float(v), 2) for v in vec) if vec is not None else None
                except (TypeError, ValueError):
                    vec = None
                diffusion.add((round(b or 0), vec))
            content = _get(item, "FrameContentSequence")
            if content:
                t = _get(content[0], "TemporalPositionIndex")
                if t is not None:
                    temporal.add(int(t))
        facts["enhanced_tes"] = sorted(tes)
        facts["enhanced_bs"] = sorted(bs)
        # A diffusion series counts its volumes by (b, direction), not time.
        facts["enhanced_volumes"] = max(len(temporal), len(diffusion)) or None
        facts["enhanced_directions"] = len({v for _b, v in diffusion
                                            if v and any(abs(x) > 1e-3 for x in v)})
    return facts


# ---------------------------------------------------------------------------
# The series
# ---------------------------------------------------------------------------


@dataclass
class SeriesFeatures:
    """How a series was acquired, as the classifier reads it."""

    description: str = ""
    protocol: str = ""
    #: The DICOM Modality ("MR", "PT", "CT"); the classifier names MR only.
    dicom_modality: str = ""
    #: An MR spectroscopy series (its SOP class or image type says so): not
    #: an image, and not the classifier's to name.
    is_spectroscopy: bool = False
    manufacturer: str = ""
    model: str = ""
    field_strength: Optional[float] = None
    n_files: int = 0
    n_volumes: Optional[int] = None
    n_slices: Optional[int] = None
    n_orientations: int = 0
    acquisition_type: str = ""                     # "2D" | "3D" | ""
    is_epi: bool = False
    is_gre: bool = False
    is_se: bool = False
    is_ir: bool = False
    is_mosaic: bool = False
    is_enhanced: bool = False
    sequence_name: str = ""
    image_type: list[str] = field(default_factory=list)
    is_derived: bool = False
    part: str = ""                                 # "mag" | "phase" | "real" | "imag" | ""
    echo_times: list[float] = field(default_factory=list)   # ms, distinct
    protocol_echoes: Optional[int] = None          # echoes the protocol acquired (Siemens)
    repetition_time: Optional[float] = None        # s
    inversion_time: Optional[float] = None         # s
    flip_angle: Optional[float] = None             # degrees
    b_values: list[float] = field(default_factory=list)     # s/mm2, distinct
    n_directions: int = 0
    pe_axis: str = ""                              # "ROW" | "COL"
    pe_positive: Optional[float] = None
    pe_dir: str = ""                               # BIDS dir label, "AP", ...
    multiband: Optional[float] = None
    has_slice_timing: bool = False
    shim: Optional[tuple] = None
    mt: bool = False
    #: The contrast agent the header names ("24ML MULTIHANCE"); "" for none.
    contrast_agent: str = ""
    echo_train_length: Optional[int] = None
    #: Siemens ``sAsl.ulMode``: 1 no labelling, 2 pulsed, 4 pseudo-continuous.
    asl_mode: Optional[int] = None
    sequence_variant: list[str] = field(default_factory=list)   # SK, SP, MP, SS, ...
    scan_options: list[str] = field(default_factory=list)       # IR, FS, FC, PFP, ...
    rows: Optional[float] = None
    columns: Optional[float] = None
    slice_thickness: Optional[float] = None
    pixel_spacing: Optional[float] = None          # mm, in plane

    def to_dict(self) -> dict:
        out = asdict(self)
        out["shim"] = list(self.shim) if self.shim else None
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "SeriesFeatures":
        known = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
        if known.get("shim"):
            known["shim"] = tuple(known["shim"])
        return cls(**known)

    @property
    def max_b(self) -> float:
        return max(self.b_values) if self.b_values else 0.0

    @property
    def first_te(self) -> Optional[float]:
        return self.echo_times[0] if self.echo_times else None


#: The DICOM SOP class of MR Spectroscopy Storage.
MR_SPECTROSCOPY_SOP = "1.2.840.10008.5.1.4.1.1.4.2"


def _part(image_type: list[str]) -> str:
    for token, part in (("P", "phase"), ("PHASE", "phase"), ("R", "real"), ("REAL", "real"),
                        ("I", "imag"), ("IMAGINARY", "imag"), ("M", "mag"),
                        ("MAGNITUDE", "mag")):
        if token in image_type[2:4] or (token in ("PHASE", "MAGNITUDE", "REAL", "IMAGINARY")
                                        and token in image_type):
            return part
    return ""


def phase_direction(axis: str, positive: Optional[float], orientation) -> str:
    """The BIDS ``dir`` label of a phase-encoding axis ("ROW"/"COL") with
    its polarity (Siemens: positive along the image axis), in the patient's
    frame (LPS) given by the row and column cosines."""
    axis = {"COLUMN": "COL"}.get(axis, axis)
    if axis not in ("ROW", "COL") or positive is None or not orientation \
            or len(orientation) != 6:
        return ""
    vector = orientation[:3] if axis == "ROW" else orientation[3:]
    k = max(range(3), key=lambda i: abs(vector[i]))
    sign = (1 if vector[k] > 0 else -1) * (1 if positive >= 0.5 else -1)
    return _DIR_LABELS[("xyz"[k], sign)]


def build(files: Iterable[dict], series: dict, *, n_files: int) -> SeriesFeatures:
    """The series' features from its files' facts and one file's series
    facts."""
    files = [f for f in files if f]
    tes = sorted({round(f["te"], 2) for f in files if f.get("te") is not None} |
                 set(series.get("enhanced_tes") or []))
    bs = sorted({round(f["b"] / 5) * 5 for f in files if f.get("b") is not None} |
                set(series.get("enhanced_bs") or []))
    if not bs and series.get("csa_b") is not None:
        bs = [round(series["csa_b"] / 5) * 5]
    directions = {f["g"] for f in files if f.get("g")}
    orients = {f["orient"] for f in files if f.get("orient")}
    positions = {f["pos"] for f in files if f.get("pos")}
    image_type: list[str] = []
    for f in files:
        for token in f.get("image_type") or []:
            if token not in image_type:
                image_type.append(token)
    first_type = (files[0].get("image_type") or []) if files else []
    parts = {_part(f.get("image_type") or []) for f in files} or {_part(first_type)}
    mosaic = "MOSAIC" in image_type
    enhanced = bool(series.get("n_frames")) and (series.get("n_frames") or 0) > 1
    # Volumes: a mosaic is one volume per file; an enhanced file counts its
    # temporal positions; otherwise files over slices (positions).
    n_slices = None
    if mosaic:
        n_volumes = len({(f.get("acq"), f.get("temporal")) for f in files}) or n_files
        n_slices = int(series["mosaic_images"]) if series.get("mosaic_images") else None
    elif enhanced:
        # Each file holds some volumes (XA writes one per file; a fully
        # enhanced series, all of them in one).
        per_file = series.get("enhanced_volumes") or 1
        # One echo per file (an XA multi-echo series): the echoes are not volumes.
        file_echoes = len({round(f["te"], 2) for f in files if f.get("te") is not None})
        n_volumes = per_file * max(1, n_files // max(1, file_echoes))
        n_slices = int(series["n_frames"] // max(1, per_file)) if series.get("n_frames") \
            else None
    else:
        n_slices = len(positions) or None
        echoes = max(1, len(tes))
        n_volumes = max(1, int(round(n_files / max(1, (n_slices or n_files) * echoes)))) \
            if n_slices else None
        if n_volumes and len(bs) > 1:
            n_volumes = max(n_volumes, len(bs))
    seq = [s.upper() for s in series.get("scanning_sequence") or []]
    seq_name = series.get("sequence_name", "").lower()
    is_epi = "EP" in seq or series.get("epi_flag") == "YES" or \
        seq_name.lstrip("*").startswith(("ep", "epfid", "epse")) or "epi" in seq_name
    orient_first = next(iter(orients)) if len(orients) == 1 else \
        (files[0].get("orient") if files else None)
    rows, columns = series.get("rows"), series.get("columns")
    if mosaic and series.get("mosaic_images"):
        # A mosaic packs the slices into one image: the matrix is one tile.
        tiles = math.ceil(math.sqrt(series["mosaic_images"]))
        rows = rows / tiles if rows else rows
        columns = columns / tiles if columns else columns
    feats = SeriesFeatures(
        description=series.get("description", ""),
        dicom_modality=series.get("dicom_modality", ""),
        is_spectroscopy=(series.get("sop_class") == MR_SPECTROSCOPY_SOP
                         or "SPECTROSCOPY" in image_type),
        protocol=series.get("protocol", ""),
        manufacturer=series.get("manufacturer", ""),
        model=series.get("model", ""),
        field_strength=series.get("field_strength"),
        n_files=n_files,
        n_volumes=int(n_volumes) if n_volumes else None,
        n_slices=n_slices,
        n_orientations=len(orients),
        acquisition_type=series.get("acquisition_type", ""),
        is_epi=bool(is_epi),
        is_gre="GR" in seq or series.get("echo_pulse") == "GRADIENT",
        is_se="SE" in seq or series.get("echo_pulse") == "SPIN",
        is_ir="IR" in seq or series.get("ti") is not None and (series.get("ti") or 0) > 0,
        is_mosaic=mosaic,
        is_enhanced=enhanced,
        sequence_name=series.get("sequence_name", ""),
        image_type=image_type,
        is_derived=bool(first_type) and first_type[0] == "DERIVED",
        # A series holding magnitude AND phase files (an SBRef often does) is
        # neither: dcm2niix writes the two apart, and naming it one would be
        # wrong for the other.
        part=(parts.pop() if len(parts) == 1 else ""),
        echo_times=tes,
        protocol_echoes=int(series["protocol_echoes"]) if series.get("protocol_echoes") else None,
        repetition_time=(series["tr"] / 1000.0) if series.get("tr") else None,
        inversion_time=(series["ti"] / 1000.0) if series.get("ti") else None,
        flip_angle=series.get("flip"),
        b_values=bs,
        n_directions=max(len(directions), int(series.get("enhanced_directions") or 0)),
        pe_axis={"COLUMN": "COL"}.get(series.get("pe_axis", ""), series.get("pe_axis", "")),
        pe_positive=series.get("pe_positive"),
        pe_dir=phase_direction(series.get("pe_axis", ""), series.get("pe_positive"),
                               orient_first),
        multiband=series.get("multiband"),
        has_slice_timing=bool(series.get("slice_timing")),
        shim=series.get("shim"),
        mt=bool(series.get("mt")),
        contrast_agent=str(series.get("contrast_agent") or ""),
        echo_train_length=int(series["echo_train_length"])
        if series.get("echo_train_length") else None,
        asl_mode=int(series["asl_mode"]) if series.get("asl_mode") else None,
        sequence_variant=list(series.get("sequence_variant") or []),
        scan_options=list(series.get("scan_options") or []),
        pixel_spacing=series.get("pixel_spacing"),
        rows=rows,
        columns=columns,
        slice_thickness=series.get("slice_thickness"),
    )
    return feats


__all__ = ["SeriesFeatures", "build", "file_facts", "parse_ascconv", "phase_direction",
           "series_facts"]

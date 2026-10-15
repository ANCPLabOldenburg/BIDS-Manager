"""A DICOM header, read for people: grouped, decoded, marked.

What the DICOM pane (``gui/widgets/dicom_pane``) shows, computed here without
Qt so it can run on a worker and be tested on its own.

* **Grouped by module**, not by tag number: Patient, Study, Series,
  Equipment, Acquisition (the MR parameters), Image, Private, File meta, and
  the rest. Sequences unfold into their items.
* **Identifiers marked** from the one definition (``fixups/identifiers``)
  plus the patient module's other identity fields, so a header is never
  copied into an email without the person seeing what it carries.
* **Siemens private headers decoded** (the CSA image and series headers, and
  the ASCII protocol inside the series header), through nibabel's CSA reader,
  which BIDS Manager already depends on.
* **Enhanced multi-frame files** get a frame table (position, b-value, echo
  time, temporal index) from their functional groups.
* **What BIDS Manager reads**: each element the scanner (and the classifier)
  uses is marked with what it is used for.

Never writes; never reads pixel data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..fixups.identifiers import IDENTIFYING_KEYS

#: Patient-module fields that identify a person, beyond the shared set.
_MORE_IDENTIFYING = frozenset({
    "OtherPatientIDs", "OtherPatientIDsSequence", "OtherPatientNames", "PatientAddress",
    "PatientTelephoneNumbers", "PatientMotherBirthName", "PatientBirthName",
    "MilitaryRank", "BranchOfService", "MedicalRecordLocator", "PatientComments",
    "PatientBirthTime", "IssuerOfPatientID",
})
IDENTIFYING = IDENTIFYING_KEYS | _MORE_IDENTIFYING

#: What the scanner reads, and what for.
READ_BY_SCAN: dict[str, str] = {
    "PatientID": "who the person is (subject identity)",
    "PatientName": "who the person is (subject identity)",
    "PatientSex": "participants.tsv",
    "PatientAge": "participants.tsv",
    "PatientBirthDate": "the age, when the date is genuine",
    "StudyInstanceUID": "sessions",
    "StudyDate": "sessions",
    "StudyTime": "sessions",
    "StudyDescription": "the study's name",
    "SeriesInstanceUID": "one row per series",
    "SeriesDescription": "the series' name (and the name rules)",
    "SeriesNumber": "the series' order",
    "SeriesTime": "PET time zero",
    "AcquisitionTime": "fieldmap links, run order",
    "AcquisitionDateTime": "fieldmap links, run order",
    "Modality": "MRI, PET or CT",
    "ImageType": "magnitude or phase, derived images, fieldmaps",
    "SOPClassUID": "spectroscopy and non-image objects",
    "Rows": "whether the series holds an image",
    "Columns": "whether the series holds an image",
    "VolumeLocalizationTechnique": "spectroscopy",
    "ResonantNucleus": "spectroscopy",
    "Units": "PET",
    "DecayCorrection": "PET",
    "ReconstructionMethod": "PET",
    "ConvolutionKernel": "PET",
    "AttenuationCorrectionMethod": "PET",
    "RandomsCorrectionMethod": "PET",
    "ScatterCorrectionMethod": "PET",
    "ActualFrameDuration": "PET frames",
    "FrameReferenceTime": "PET frames",
    "RadiopharmaceuticalInformationSequence": "PET tracer and dose",
}

_STUDY = frozenset({
    "StudyInstanceUID", "StudyDate", "StudyTime", "StudyID", "AccessionNumber",
    "StudyDescription", "ReferringPhysicianName", "ProcedureCodeSequence",
    "RequestedProcedureDescription", "PerformedProcedureStepDescription",
    "PerformedProcedureStepStartDate", "PerformedProcedureStepStartTime",
    "PerformedProcedureStepID", "RequestingPhysician", "AdmittingDiagnosesDescription",
})
_SERIES = frozenset({
    "Modality", "SeriesInstanceUID", "SeriesNumber", "SeriesDescription", "SeriesDate",
    "SeriesTime", "ProtocolName", "BodyPartExamined", "PatientPosition", "OperatorsName",
    "PerformingPhysicianName", "Laterality", "FrameOfReferenceUID", "PositionReferenceIndicator",
})
_EQUIPMENT = frozenset({
    "Manufacturer", "ManufacturerModelName", "StationName", "InstitutionName",
    "InstitutionAddress", "InstitutionalDepartmentName", "DeviceSerialNumber",
    "SoftwareVersions", "DateOfLastCalibration", "TimeOfLastCalibration",
})
_ACQUISITION = frozenset({
    "AcquisitionDate", "AcquisitionTime", "AcquisitionDateTime", "AcquisitionNumber",
    "ContentDate", "ContentTime", "AcquisitionDuration",
})
_IMAGE = frozenset({
    "ImageType", "SOPClassUID", "SOPInstanceUID", "InstanceNumber", "InstanceCreationDate",
    "InstanceCreationTime", "SpecificCharacterSet",
})

MODULES = ("Patient", "Study", "Series", "Equipment", "Acquisition", "Image", "Private",
           "File meta", "Other")


@dataclass
class Element:
    """One element (or a sequence item) of a header."""

    tag: str                       # "(0018,0080)", "" for items and decoded entries
    keyword: str
    name: str
    vr: str
    value: str
    children: list["Element"] = field(default_factory=list)
    identifies: bool = False
    read_for: str = ""             # what BIDS Manager reads it for, "" when it does not

    def flatten(self) -> Iterable["Element"]:
        yield self
        for child in self.children:
            yield from child.flatten()


@dataclass
class DicomHeader:
    path: Path
    summary: dict[str, str]
    flags: list[str]
    modules: dict[str, list[Element]]
    frames: list[dict[str, str]] = field(default_factory=list)

    def elements(self) -> Iterable[Element]:
        for items in self.modules.values():
            for element in items:
                yield from element.flatten()

    def find(self, keyword: str) -> Optional[Element]:
        return next((e for e in self.elements() if e.keyword == keyword), None)


def _module_of(elem) -> str:
    group = elem.tag.group
    keyword = elem.keyword or ""
    if group == 0x0002:
        return "File meta"
    if group % 2 == 1:
        return "Private"
    if group == 0x0010:
        return "Patient"
    if keyword in _STUDY or group == 0x0032:
        return "Study"
    if keyword in _SERIES:
        return "Series"
    if keyword in _EQUIPMENT:
        return "Equipment"
    if group in (0x0018, 0x0019) or keyword in _ACQUISITION:
        return "Acquisition"
    if group in (0x0020, 0x0028) or keyword in _IMAGE:
        return "Image"
    return "Other"


def _text(value, limit: int = 2000) -> str:
    from pydicom.multival import MultiValue

    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value):,} bytes>"
    if isinstance(value, (list, tuple, MultiValue)):
        text = " \\ ".join(_text(v, limit) for v in value)
    else:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + " ..."


def _element(elem) -> Element:
    keyword = elem.keyword or ""
    tag = f"({elem.tag.group:04X},{elem.tag.element:04X})"
    name = elem.name or "Private tag"
    if name == "Private tag data":
        # Unknown to the dictionary: who wrote it says more than "data".
        creator = getattr(elem, "private_creator", None)
        name = f"Private ({creator})" if creator else "Private tag"
    out = Element(tag=tag, keyword=keyword, name=name, vr=str(elem.VR), value="",
                  identifies=keyword in IDENTIFYING,
                  read_for=READ_BY_SCAN.get(keyword, ""))
    if elem.VR == "SQ":
        items = list(elem.value or [])
        out.value = f"{len(items)} item(s)"
        for i, item in enumerate(items, start=1):
            child = Element(tag="", keyword="", name=f"Item {i}", vr="", value="")
            child.children = [_element(e) for e in item]
            child.identifies = any(c.identifies for c in child.children)
            out.children.append(child)
        out.identifies = out.identifies or any(c.identifies for c in out.children)
    else:
        out.value = _text(elem.value)
        raw = elem.value
        if elem.tag.group % 2 == 1 and isinstance(raw, (bytes, bytearray)) and \
                b"### ASCCONV BEGIN" in raw:
            # The Siemens protocol, raw (XA keeps it in a private sequence of
            # the shared functional groups; the classic CSA header is
            # decoded in read_header).
            protocol = _protocol(bytes(raw).decode("latin-1", "replace"))
            if protocol:
                out.children = protocol
                out.value = f"Siemens protocol ({len(protocol)} entries)"
    return out


def _csa_children(raw: bytes) -> list[Element]:
    """The entries of a Siemens CSA header, and the ASCII protocol if any."""
    import warnings

    try:
        with warnings.catch_warnings():
            # nibabel calls its DICOM readers experimental, on import.
            warnings.simplefilter("ignore")
            from nibabel.nicom import csareader

        csa = csareader.read(raw)
    except Exception:  # noqa: BLE001 - not every private blob is a CSA header
        return []
    out = []
    for name, entry in sorted((csa.get("tags") or {}).items()):
        items = [i for i in entry.get("items", []) if i != ""]
        if not items:
            continue
        if name == "MrPhoenixProtocol":
            protocol = _protocol(str(items[0]))
            if protocol:
                node = Element(tag="", keyword=name, name="Protocol (ASCCONV)", vr="",
                               value=f"{len(protocol)} entries", children=protocol)
                out.append(node)
            continue
        out.append(Element(tag="", keyword=name, name=name, vr=str(entry.get("vr", "")),
                           value=_text(items)))
    return out


_ASCCONV = re.compile(r"### ASCCONV BEGIN.*?###(.*?)### ASCCONV END ###", re.S)


def _protocol(text: str) -> list[Element]:
    """The Siemens ASCII protocol as ``key = value`` entries."""
    found = _ASCCONV.search(text)
    body = found.group(1) if found else ""
    out = []
    for line in body.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.split("#", 1)[0].strip().strip('"')
        if key:
            out.append(Element(tag="", keyword=key, name=key, vr="", value=value))
    return out


def _frames(ds) -> list[dict[str, str]]:
    """One row per frame of an enhanced multi-frame image."""
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if not per_frame:
        return []

    def first(item, seq: str, attr: str):
        s = getattr(item, seq, None)
        return getattr(s[0], attr, None) if s else None

    rows = []
    for i, item in enumerate(per_frame, start=1):
        pos = first(item, "PlanePositionSequence", "ImagePositionPatient")
        rows.append({
            "frame": str(i),
            "position": " \\ ".join(f"{float(v):.2f}" for v in pos) if pos else "",
            "b-value": _text(first(item, "MRDiffusionSequence", "DiffusionBValue") or ""),
            "echo time": _text(first(item, "MREchoSequence", "EffectiveEchoTime") or ""),
            "temporal index": _text(
                first(item, "FrameContentSequence", "TemporalPositionIndex") or ""),
            "stack position": _text(
                first(item, "FrameContentSequence", "InStackPositionNumber") or ""),
        })
    return rows


def _when(ds) -> str:
    """When the image was acquired, ``2026-03-01 10:00:05``, from the
    acquisition date and time (else the series' or the content's)."""
    stamp = str(getattr(ds, "AcquisitionDateTime", "") or "")
    date, time = stamp[:8], stamp[8:]
    for d, tm in (("AcquisitionDate", "AcquisitionTime"), ("SeriesDate", "SeriesTime"),
                  ("ContentDate", "ContentTime")):
        if date:
            break
        date = str(getattr(ds, d, "") or "")
        time = str(getattr(ds, tm, "") or "")
    out = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 and date.isdigit() else date
    digits = time.split(".")[0]
    if len(digits) >= 4 and digits.isdigit():
        clock = ":".join(digits[i:i + 2] for i in range(0, min(len(digits), 6), 2))
        out = f"{out} {clock}".strip()
    return out


def read_header(path: Path) -> DicomHeader:
    """Read ``path``'s header (no pixels) into a :class:`DicomHeader`."""
    import pydicom
    from pydicom.uid import UID

    path = Path(path)
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    modules: dict[str, list[Element]] = {m: [] for m in MODULES}
    for elem in ds.file_meta if getattr(ds, "file_meta", None) else []:
        modules["File meta"].append(_element(elem))
    siemens = "siemens" in str(getattr(ds, "Manufacturer", "")).lower()
    for elem in ds:
        if elem.tag.group == 0x7FE0:
            continue
        element = _element(elem)
        if siemens and elem.tag.group == 0x0029 and elem.VR in ("OB", "UN") and \
                isinstance(elem.value, (bytes, bytearray)):
            element.children = _csa_children(bytes(elem.value))
            if element.children:
                element.value = f"Siemens CSA header ({len(element.children)} entries)"
        modules[_module_of(elem)].append(element)
    sop = str(getattr(ds, "SOPClassUID", ""))
    try:
        sop_name = UID(sop).name if sop else ""
    except Exception:  # noqa: BLE001
        sop_name = sop
    image_type = [str(x).upper() for x in (getattr(ds, "ImageType", None) or [])]
    flags = []
    if "Enhanced" in sop_name or getattr(ds, "PerFrameFunctionalGroupsSequence", None):
        flags.append("Enhanced multi-frame")
    if image_type[:1] == ["DERIVED"]:
        flags.append("Derived")
    if "Rows" not in ds or "Columns" not in ds:
        flags.append("Non-image object")
    strength = getattr(ds, "MagneticFieldStrength", "")
    summary = {
        "Modality": str(getattr(ds, "Modality", "")),
        "Scanner": " ".join(str(getattr(ds, k, "")) for k in
                            ("Manufacturer", "ManufacturerModelName")).strip(),
        "Field strength": f"{float(strength):g} T" if strength not in ("", None) else "",
        "Series": " · ".join(x for x in (str(getattr(ds, "SeriesNumber", "") or ""),
                                         str(getattr(ds, "SeriesDescription", "") or "")) if x),
        "Acquired": _when(ds),
        "Object": sop_name,
    }
    return DicomHeader(path=path, summary={k: v for k, v in summary.items() if v},
                       flags=flags, modules={k: v for k, v in modules.items() if v},
                       frames=_frames(ds))


#: Elements that differ between any two images of a scan (identities,
#: instance numbers, clocks, positions). Comparing two series, they are
#: noise; the compare window hides them unless asked.
ROUTINE = frozenset({
    "SOPInstanceUID", "MediaStorageSOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID",
    "FrameOfReferenceUID", "ReferencedSOPInstanceUID", "IrradiationEventUID",
    "InstanceNumber", "AcquisitionNumber", "SeriesNumber",
    "InstanceCreationDate", "InstanceCreationTime", "ContentDate", "ContentTime",
    "AcquisitionDate", "AcquisitionTime", "AcquisitionDateTime", "SeriesDate", "SeriesTime",
    "PerformedProcedureStepStartTime", "PerformedProcedureStepID", "FrameAcquisitionDateTime",
    "FrameReferenceDateTime", "SliceLocation", "ImagePositionPatient", "WindowCenter",
    "WindowWidth", "LargestImagePixelValue", "SmallestImagePixelValue",
    "WindowCenterWidthExplanation", "TimeAfterStart", "ICE_Dims",
})


@dataclass(frozen=True)
class Difference:
    """One element that differs between two headers."""

    module: str
    #: Where it is: the element's name, after the sequences and items that
    #: hold it ("Shared Functional Groups Sequence > Item 1 > Echo Time").
    where: str
    tag: str
    a: Optional[str]               # None: not in the first header
    b: Optional[str]
    identifies: bool = False
    routine: bool = False


def _flat(header: DicomHeader) -> dict[tuple, tuple[str, str, Element]]:
    """Every element by its path: ``{key: (module, where, element)}``."""
    out: dict[tuple, tuple[str, str, Element]] = {}

    def walk(module: str, e: Element, key: tuple, names: tuple) -> None:
        key = key + (e.tag or e.keyword or e.name,)
        names = names + (e.name,)
        out[(module, *key)] = (module, " > ".join(names), e)
        for child in e.children:
            walk(module, child, key, names)

    for module, items in header.modules.items():
        for e in items:
            walk(module, e, (), ())
    return out


def compare(a: DicomHeader, b: DicomHeader) -> list[Difference]:
    """Every element, at any depth, that differs between ``a`` and ``b`` or
    is in one only, in module and header order. A sequence differing only
    inside is reported by what differs inside it."""
    fa, fb = _flat(a), _flat(b)
    order = {m: i for i, m in enumerate(MODULES)}
    keys = list(fa) + [k for k in fb if k not in fa]
    rows = []
    for key in sorted(keys, key=lambda k: order.get(k[0], len(order))):
        ea, eb = fa.get(key), fb.get(key)
        ref = (ea or eb)[2]
        if ref.children and ea is not None and eb is not None:
            continue                       # what differs inside is listed
        va = ea[2].value if ea is not None else None
        vb = eb[2].value if eb is not None else None
        if va == vb:
            continue
        module, where = (ea or eb)[0], (ea or eb)[1]
        rows.append(Difference(module=module, where=where, tag=ref.tag, a=va, b=vb,
                               identifies=ref.identifies,
                               routine=ref.keyword in ROUTINE or ref.keyword.endswith("UID")))
    return rows


def export(header: DicomHeader, *, identifiers: bool = False) -> dict:
    """The header as plain data (JSON-ready). Identifying elements are left
    out unless ``identifiers``."""
    def one(e: Element) -> Optional[dict]:
        if e.identifies and not identifiers and not e.children:
            return None
        node = {"tag": e.tag, "keyword": e.keyword, "name": e.name, "vr": e.vr,
                "value": e.value}
        kids = [x for x in (one(c) for c in e.children) if x is not None]
        if kids:
            node["children"] = kids
        return node

    return {
        "file": header.path.name,
        "summary": header.summary,
        "flags": header.flags,
        "modules": {m: [x for x in (one(e) for e in items) if x is not None]
                    for m, items in header.modules.items()},
        "frames": header.frames,
    }


def export_json(header: DicomHeader, *, identifiers: bool = False) -> str:
    return json.dumps(export(header, identifiers=identifiers), indent=2, ensure_ascii=False)


def export_text(header: DicomHeader, *, identifiers: bool = False) -> str:
    """The header as indented text, module by module, the way the pane
    shows it. Identifying values are left out unless ``identifiers``."""
    lines = [f"DICOM header: {header.path.name}", ""]
    lines += [f"{k}: {v}" for k, v in header.summary.items()]
    if header.flags:
        lines.append("Flags: " + ", ".join(header.flags))

    def one(e: Element, depth: int) -> None:
        pad = "  " * depth
        value = e.value
        if e.identifies and not e.children and not identifiers:
            value = "(identifying, left out)"
        where = " ".join(x for x in (e.tag, e.vr) if x)
        head = f"{pad}{e.name}" + (f" {where}" if where else "")
        lines.append(f"{head}: {value}" if value else head)
        for child in e.children:
            one(child, depth + 1)

    for module, items in header.modules.items():
        lines += ["", f"[{module}]"]
        for e in items:
            one(e, 1)
    if header.frames:
        lines += ["", f"[Frames] {len(header.frames)}"]
        cols = list(header.frames[0])
        lines.append("  " + "\t".join(cols))
        lines += ["  " + "\t".join(row.get(c, "") for c in cols) for row in header.frames]
    return "\n".join(lines) + "\n"


def count_identifying(header: DicomHeader) -> int:
    """How many values in the header identify the participant."""
    return sum(1 for e in header.elements() if e.identifies and not e.children and e.value)


__all__ = ["ROUTINE", "DicomHeader", "Difference", "Element", "IDENTIFYING", "MODULES",
           "READ_BY_SCAN", "compare", "count_identifying", "export", "export_json",
           "export_text", "read_header"]

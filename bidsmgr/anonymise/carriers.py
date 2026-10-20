"""Where identity is carried, and how each carrier is cleaned.

One function per kind of content. Each takes the content and returns the new
content and the changes, so a plan can show them before anything is written
and an operation can write them as one undoable step. Binary carriers (a
gzip or NIfTI header, an EDF header, a photo) are cleaned by editing bytes:
the recorded data are never decoded and re-encoded, so they stay identical.

Qt-free.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import re
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..fixups.identifiers import IDENTIFYING_KEYS, PROVENANCE_PATH_KEYS
from .policy import AnonymisePolicy
from .state import State, is_shifted


@dataclass(frozen=True)
class Change:
    """One thing that changes in one file."""

    what: str          # the field, column or header part
    before: str
    after: str         # "" when it is removed
    reason: str
    #: The policy category it belongs to (``policy.CATEGORIES``), or
    #: ``policy.RECORD`` for the note saying what was done.
    category: str = ""


@dataclass(frozen=True)
class Review:
    """Something that may identify a person and is not changed automatically:
    free text only a person can judge."""

    what: str
    value: str
    reason: str


# ---------------------------------------------------------------------------
# Dates and times in text
# ---------------------------------------------------------------------------

_DATETIME = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$")
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DICOM_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def parse_date(value: str) -> Optional[_dt.date]:
    """The calendar date a value states, if it states one."""
    for pattern in (_DATETIME, _DATE, _DICOM_DATE):
        m = pattern.match(value.strip()) if isinstance(value, str) else None
        if m:
            try:
                return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
    return None


def shift_text(value: str, days_back: int, *, midnight: bool) -> str:
    """``value`` with its date moved back ``days_back`` days (and its time of
    day set to midnight when asked), in the same spelling it came in."""
    text = value.strip()
    m = _DATETIME.match(text)
    if m:
        moved = _dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                             int(m.group(4)), int(m.group(5)), int(m.group(6)))
        moved -= _dt.timedelta(days=days_back)
        if midnight:
            moved = moved.replace(hour=0, minute=0, second=0)
            return moved.strftime("%Y-%m-%dT%H:%M:%S") + (m.group(8) or "")
        return moved.strftime("%Y-%m-%dT%H:%M:%S") + (m.group(7) or "") + (m.group(8) or "")
    for pattern, fmt in ((_DATE, "%Y-%m-%d"), (_DICOM_DATE, "%Y%m%d")):
        m = pattern.match(text)
        if m:
            moved = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return (moved - _dt.timedelta(days=days_back)).strftime(fmt)
    return value


def _needs_shift(value: str, policy: AnonymisePolicy, days_back: int) -> bool:
    date = parse_date(value)
    return (date is not None and policy.dates == "shift" and days_back > 0
            and not is_shifted(date))


# ---------------------------------------------------------------------------
# JSON sidecars
# ---------------------------------------------------------------------------

#: What "Names and IDs" removes: they name a person. Public so a caller can
#: tell a birth date (removed) from a date that is shifted.
IDENTITY_KEYS = IDENTIFYING_KEYS | frozenset({
    "PatientComments", "OtherPatientIDs", "PatientAddress", "PatientTelephoneNumbers",
    "PatientMotherBirthName", "MilitaryRank", "EthnicGroup",
})
_INSTITUTION_ADDRESS = frozenset({"InstitutionAddress", "InstitutionalDepartmentAddress"})
_INSTITUTION_ALL = _INSTITUTION_ADDRESS | frozenset({"InstitutionName",
                                                    "InstitutionalDepartmentName"})
_DEVICES = {"DeviceSerialNumber": "device", "StationName": "station"}
#: The scanner's own clock readings. Not BIDS fields; removed under Strict.
#: BIDS's own time fields (PET TimeZero and the like) are kept: they are
#: required, and other timings are stated relative to them.
_SCANNER_TIMES = frozenset({"AcquisitionTime", "SeriesTime", "StudyTime", "ContentTime",
                            "AcquisitionDate", "SeriesDate", "StudyDate", "ContentDate"})
#: Free text a person may have typed a name into.
_FREE_TEXT = frozenset({"ProtocolName", "SeriesDescription", "ImageComments",
                        "StudyDescription", "RequestedProcedureDescription",
                        "PerformedProcedureStepDescription", "AcquisitionComments"})


def clean_json(data: dict, policy: AnonymisePolicy, state: State, days_back: int,
               *, method: Optional[str] = None) -> tuple[dict, list[Change], list[Review]]:
    """A sidecar with its identity removed. ``method`` is written into
    ``DeidentificationMethod`` (an MRI/PET field) when given."""
    out: dict = {}
    changes: list[Change] = []
    reviews: list[Review] = []
    # What goes, and the category each removal belongs to.
    remove: dict[str, str] = {}
    if policy.identity:
        remove.update(dict.fromkeys(IDENTITY_KEYS, "identity"))
    if policy.traces:
        remove.update(dict.fromkeys(PROVENANCE_PATH_KEYS, "traces"))
    if policy.institution == "address":
        remove.update(dict.fromkeys(_INSTITUTION_ADDRESS, "site"))
    elif policy.institution == "all":
        remove.update(dict.fromkeys(_INSTITUTION_ALL, "site"))
    if policy.times == "midnight":
        remove.update(dict.fromkeys(_SCANNER_TIMES, "dates"))
    for key, value in data.items():
        shown = json.dumps(value, ensure_ascii=False)[:80]
        if key in remove and key in _INSTITUTION_ALL and value != "n/a":
            # The standard RECOMMENDS these: "n/a" says "not given" and keeps
            # the sidecar as valid as it was, where removing them added a
            # missing-field warning to every image.
            changes.append(Change(key, shown, '"n/a"', "the site's identity", remove[key]))
            out[key] = "n/a"
            continue
        if key in remove and key in _INSTITUTION_ALL:
            out[key] = value
            continue
        if key in remove:
            if key in _SCANNER_TIMES:
                why = "the scanner's clock"
            elif key in PROVENANCE_PATH_KEYS:
                why = "a path on the converting machine"
            else:
                why = "names a person"
            changes.append(Change(key, shown, "", why, remove[key]))
            continue
        if key in _DEVICES and policy.devices == "pseudonymise" and isinstance(value, str) \
                and value and not value.startswith(_DEVICES[key] + "-"):
            new = state.label(_DEVICES[key], value)
            changes.append(Change(key, shown, new, "identifies the machine", "devices"))
            out[key] = new
            continue
        if key.endswith("UID") and policy.uids == "pseudonymise" and isinstance(value, str) \
                and value and not value.startswith("2.25."):
            new = state.uid(value)
            changes.append(Change(key, shown, new, "a UID can encode the scanner and time",
                                  "uids"))
            out[key] = new
            continue
        if isinstance(value, str) and _needs_shift(value, policy, days_back):
            new = shift_text(value, days_back, midnight=policy.times == "midnight")
            changes.append(Change(key, shown, json.dumps(new), "a date", "dates"))
            out[key] = new
            continue
        if isinstance(value, str) and policy.times == "midnight" and _DATETIME.match(value) \
                and not value[11:19] == "00:00:00":
            new = shift_text(value, 0, midnight=True)
            changes.append(Change(key, shown, json.dumps(new), "a time of day", "dates"))
            out[key] = new
            continue
        if key in _FREE_TEXT and isinstance(value, str) and value.strip():
            reviews.append(Review(key, value, "free text: check it names no one"))
        out[key] = value
    if method and changes:
        # An array of strings, as the schema types it.
        current = out.get("DeidentificationMethod")
        listed = ([current] if isinstance(current, str) and current
                  else list(current) if isinstance(current, list) else [])
        if not any("BIDS Manager" in str(item) for item in listed):
            new = listed + [method]
            out["DeidentificationMethod"] = new
            changes.append(Change("DeidentificationMethod", json.dumps(current or []),
                                  json.dumps(new), "says what was done", "record"))
    return out, changes, reviews


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

#: participants.tsv columns that are the person (removed whatever the policy).
IDENTITY_COLUMNS = frozenset({
    "given_name", "family_name", "name", "patient_name", "patient_id", "patientid",
    "birth_date", "birthdate", "date_of_birth", "dob", "birthday", "address", "email",
    "phone", "telephone",
})


def clean_participants(header: list[str], rows: list[dict], policy: AnonymisePolicy,
                       in_scope=None) -> tuple[list[str], list[dict], list[Change]]:
    """participants.tsv cleaned. ``in_scope(participant_id)`` limits it to
    some rows: an identity column is then blanked (n/a) in those rows rather
    than removed, since the others still hold their values."""
    changes: list[Change] = []
    mine = [r for r in rows if in_scope is None or in_scope(r.get("participant_id", ""))]
    whole = len(mine) == len(rows)
    columns = [c for c in header if c.lower() in IDENTITY_COLUMNS] if policy.identity else []
    gone = columns if whole else []
    blanked = [] if whole else columns
    for col in gone:
        changes.append(Change(f"column {col}", f"{sum(1 for r in rows if r.get(col))} values",
                              "", "identifies the participant", "identity"))
    header = [c for c in header if c not in gone]
    out = []
    for row in rows:
        ours = in_scope is None or in_scope(row.get("participant_id", ""))
        row = {k: v for k, v in row.items() if k not in gone}
        who = row.get("participant_id", "?")
        if ours:
            for col in blanked:
                if row.get(col, "n/a") not in ("", "n/a"):
                    changes.append(Change(f"{col} of {who}", row[col], "n/a",
                                          "identifies the participant", "identity"))
                    row[col] = "n/a"
        age = row.get("age")
        try:
            value = float(age)
        except (TypeError, ValueError):
            value = None
        if ours and value is not None and policy.ages != "keep":
            new = min(value, 89.0)
            if policy.ages == "bins":
                new = 5 * int(new // 5)
            text = (str(int(new)) if float(new).is_integer() else str(new))
            if text != str(age):
                changes.append(Change(f"age of {who}", str(age), text,
                                      "an exact age, or one over 89, identifies", "ages"))
                row["age"] = text
        out.append(row)
    return header, out, changes


def clean_acq_times(rows: list[dict], policy: AnonymisePolicy, days_back: int,
                    in_scope=None) -> tuple[list[dict], list[Change]]:
    """``acq_time`` of a scans or sessions table, shifted with its sidecars;
    ``in_scope(row)`` limits it to some rows (a sessions table, some
    sessions)."""
    changes: list[Change] = []
    out = []
    for row in rows:
        value = row.get("acq_time", "")
        midnight = policy.times == "midnight"
        if in_scope is not None and not in_scope(row):
            out.append(row)
            continue
        if isinstance(value, str) and (_needs_shift(value, policy, days_back)
                                       or (midnight and _DATETIME.match(value)
                                           and value[11:19] != "00:00:00")):
            shift = days_back if _needs_shift(value, policy, days_back) else 0
            new = shift_text(value, shift, midnight=midnight)
            key = row.get("filename") or row.get("session_id") or "row"
            changes.append(Change(f"acq_time of {key}", value, new, "a date", "dates"))
            row = dict(row, acq_time=new)
        out.append(row)
    return out, changes


_VMRK_SEGMENT = re.compile(r"^(Mk\d+=New Segment,[^\r\n]*?,)(\d{8})(\d{6,12})", re.MULTILINE)


def clean_vmrk(text: str, policy: AnonymisePolicy,
               days_back: int) -> tuple[str, list[Change]]:
    """A BrainVision marker file's "New Segment" date."""
    changes: list[Change] = []

    def sub(m: re.Match) -> str:
        date = parse_date(m.group(2))
        if date is None or (is_shifted(date) and policy.times != "midnight"):
            return m.group(0)
        moved = date if is_shifted(date) else date - _dt.timedelta(days=days_back)
        clock = "0" * len(m.group(3)) if policy.times == "midnight" else m.group(3)
        new = m.group(1) + moved.strftime("%Y%m%d") + clock
        if new != m.group(0):
            changes.append(Change("New Segment date", m.group(2) + m.group(3),
                                  moved.strftime("%Y%m%d") + clock, "a date", "dates"))
        return new

    return _VMRK_SEGMENT.sub(sub, text), changes


# ---------------------------------------------------------------------------
# Binary headers, edited in place of the bytes
# ---------------------------------------------------------------------------

def gzip_body_offset(head: bytes) -> tuple[int, int, bool]:
    """``(offset of the compressed body, flags, has a name/comment/time)`` of
    a gzip member whose first bytes are ``head``."""
    if len(head) < 10 or head[:2] != b"\x1f\x8b":
        raise ValueError("not gzip")
    flags = head[3]
    mtime = struct.unpack("<I", head[4:8])[0]
    pos = 10
    if flags & 0x04:                                   # FEXTRA
        size = struct.unpack("<H", head[pos:pos + 2])[0]
        pos += 2 + size
    for bit in (0x08, 0x10):                           # FNAME, FCOMMENT
        if flags & bit:
            pos = head.index(b"\x00", pos) + 1
    if flags & 0x02:                                   # FHCRC
        pos += 2
    return pos, flags, bool(mtime or flags & 0x18)


def gzip_header_change(path: Path) -> Optional[Change]:
    """What the gzip header says that it should not (a time, a name)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
        _offset, flags, says = gzip_body_offset(head)
    except (OSError, ValueError, IndexError, struct.error):
        return None
    if not says:
        return None
    mtime = struct.unpack("<I", head[4:8])[0]
    parts = []
    if mtime:
        parts.append(_dt.datetime.fromtimestamp(mtime, _dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M"))
    if flags & 0x08:
        start = 10 + (2 + struct.unpack("<H", head[10:12])[0] if flags & 0x04 else 0)
        parts.append(head[start:head.index(b"\x00", start)].decode("latin-1"))
    return Change("gzip header", ", ".join(parts), "", "the time it was written, its "
                  "original name", "traces")


def write_clean_gzip(src: Path, dst: Path) -> None:
    """``src`` with a header that states no time and no name: the compressed
    body copied byte for byte (its checksum covers the data, not the header)."""
    with open(src, "rb") as fh:
        head = fh.read(4096)
        offset, flags, _says = gzip_body_offset(head)
        extra = head[10:10 + 2 + struct.unpack("<H", head[10:12])[0]] if flags & 0x04 else b""
        with open(dst, "wb") as out:
            out.write(b"\x1f\x8b\x08" + bytes([flags & 0x04]) + b"\x00\x00\x00\x00"
                      + head[8:10] + extra)
            fh.seek(offset)
            shutil.copyfileobj(fh, out, 1 << 20)


_DESCRIP_TIME = re.compile(rb";?Time=[0-9.]+")


def nifti_descrip_change(path: Path) -> Optional[tuple[bytes, bytes]]:
    """``(descrip, cleaned)`` when the NIfTI header's text states a time of
    day (dcm2niix writes ``Time=101530.500``)."""
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rb") as fh:
            header = fh.read(348)
    except (OSError, EOFError, zlib.error):
        # Damaged or truncated: left alone (the validator reports it), never
        # fatal to the rest of the dataset.
        return None
    if len(header) < 348:
        return None
    descrip = header[148:228]
    cleaned = _DESCRIP_TIME.sub(b"", descrip.rstrip(b"\x00")).lstrip(b";")
    if cleaned == descrip.rstrip(b"\x00"):
        return None
    return descrip.rstrip(b"\x00"), cleaned


def write_clean_nifti(src: Path, dst: Path) -> None:
    """``src`` with ``Time=`` taken out of its header text; the image data
    byte-identical. A ``.nii.gz`` is decompressed and compressed again, with
    a header that states no time and no name."""
    gz = src.name.endswith(".gz")
    opener = gzip.open if gz else open
    with opener(src, "rb") as fh:
        header = bytearray(fh.read(348))
        found = nifti_descrip_change(src)
        if found:
            header[148:228] = found[1].ljust(80, b"\x00")[:80]
        if gz:
            with open(dst, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                                       mtime=0, compresslevel=6) as out:
                out.write(bytes(header))
                shutil.copyfileobj(fh, out, 1 << 20)
        else:
            with open(dst, "wb") as out:
                out.write(bytes(header))
                shutil.copyfileobj(fh, out, 1 << 20)


#: The EDF+ convention for an anonymised header: the oldest date EDF can
#: hold, and an X for every unknown subfield.
EDF_DATE = b"01.01.85"


def edf_header_changes(path: Path, policy: AnonymisePolicy) -> tuple[Optional[bytes], list[Change]]:
    """The first 256 bytes of an EDF/BDF file with the patient, the recording
    line and the start date anonymised, or ``(None, [])`` when already so."""
    try:
        with open(path, "rb") as fh:
            head = bytearray(fh.read(256))
    except OSError:
        return None, []
    if len(head) < 256:
        return None, []
    changes: list[Change] = []
    patient = bytes(head[8:88]).decode("latin-1").strip()
    recording = bytes(head[88:168]).decode("latin-1").strip()
    plus = recording.startswith("Startdate")
    anon_patient = "X X X X"
    if policy.identity and patient and patient != anon_patient:
        head[8:88] = anon_patient.encode("latin-1").ljust(80)
        changes.append(Change("patient field", patient, anon_patient, "names the participant",
                              "identity"))
    anon_recording = "Startdate X X X X" if plus else "X"
    if policy.identity and recording and recording != anon_recording:
        head[88:168] = anon_recording.encode("latin-1").ljust(80)
        changes.append(Change("recording field", recording, anon_recording,
                              "the date, the technician and the equipment", "identity"))
    date = bytes(head[168:176])
    if policy.dates == "shift" and date != EDF_DATE:
        head[168:176] = EDF_DATE
        changes.append(Change("start date", date.decode("latin-1"), EDF_DATE.decode(),
                              "a date; EDF holds none before 1985, so the EDF+ "
                              "anonymous date", "dates"))
    clock = bytes(head[176:184])
    if policy.times == "midnight" and clock != b"00.00.00":
        head[176:184] = b"00.00.00"
        changes.append(Change("start time", clock.decode("latin-1"), "00.00.00",
                              "a time of day", "dates"))
    return (bytes(head) if changes else None), changes


def write_edf_header(src: Path, dst: Path, header: bytes) -> None:
    with open(src, "rb") as fh, open(dst, "wb") as out:
        fh.seek(len(header))
        out.write(header)
        shutil.copyfileobj(fh, out, 1 << 20)


# ---------------------------------------------------------------------------
# Photographs: metadata blocks removed, pixels untouched
# ---------------------------------------------------------------------------

_JPEG_DROP = {0xE1, 0xED, 0xFE}           # APP1 (EXIF, XMP), APP13 (IPTC), COM
_PNG_DROP = {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"}


def clean_photo(data: bytes) -> Optional[bytes]:
    """A JPEG or PNG without its metadata blocks, or ``None`` if there are
    none (or it is neither)."""
    if data[:2] == b"\xff\xd8":
        out = bytearray(data[:2])
        pos = 2
        dropped = False
        while pos + 4 <= len(data) and data[pos] == 0xFF:
            marker = data[pos + 1]
            if marker == 0xDA:                 # start of scan: the rest is image
                break
            size = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            segment = data[pos:pos + 2 + size]
            if marker in _JPEG_DROP:
                dropped = True
            else:
                out += segment
            pos += 2 + size
        if not dropped:
            return None
        return bytes(out + data[pos:])
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        out = bytearray(data[:8])
        pos = 8
        dropped = False
        while pos + 8 <= len(data):
            size = struct.unpack(">I", data[pos:pos + 4])[0]
            kind = data[pos + 4:pos + 8]
            chunk = data[pos:pos + 12 + size]
            if kind in _PNG_DROP:
                dropped = True
            else:
                out += chunk
            pos += 12 + size
        return bytes(out) if dropped else None
    return None


def png_ok(data: bytes) -> bool:
    """Every chunk's CRC still right (used by the tests)."""
    pos = 8
    while pos + 8 <= len(data):
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        body = data[pos + 4:pos + 8 + size]
        crc = struct.unpack(">I", data[pos + 8 + size:pos + 12 + size])[0]
        if zlib.crc32(body) & 0xFFFFFFFF != crc:
            return False
        pos += 12 + size
    return True


__all__ = ["Change", "EDF_DATE", "IDENTITY_COLUMNS", "IDENTITY_KEYS", "Review",
           "clean_acq_times",
           "clean_json", "clean_participants", "clean_photo", "clean_vmrk",
           "edf_header_changes", "gzip_body_offset", "gzip_header_change",
           "nifti_descrip_change", "parse_date", "png_ok", "shift_text",
           "write_clean_gzip", "write_clean_nifti", "write_edf_header"]

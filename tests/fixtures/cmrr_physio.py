"""Synthetic Siemens CMRR physiological logs, and the DICOM that carries them.

Shaped on the multimodal sample's real ``_PhysioLog`` (EJA_1 logs, 2026-10-22):
an ECG log with four leads interleaved line by line at 400 Hz, a pulse log at
200 Hz, a respiration log at 50 Hz, and the acquisition log (``_Info``) with a
row per slice of every volume. Two things the real one has and a naive reader
gets wrong are built in: a dropped sample (``gap_at``), and the clock
correction the scanner applies once early in a run, where every channel's
ticks step back by one (``clock_fix_at``).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

#: Ticks (2.5 ms) between two volumes: TR 0.85 s.
TR_TICKS = 340
START = 14882104


def _header(kind: str, sample_time: int | None) -> list[str]:
    lines = ["UUID        = 08b19a26-bf5b-4d0b-90aa-85b2893c9c75",
             "ScanDate    = 20230914_102000", "LogVersion  = EJA_1",
             f"LogDataType = {kind}"]
    if sample_time is not None:
        lines.append(f"SampleTime  = {sample_time}")
    return lines + ["", "ACQ_TIME_TICS  CHANNEL  VALUE  SIGNAL", ""]


def _signal(kind: str, channel: str, n: int, *, step: int, clock_fix_at: int | None,
            gap_at: int | None) -> list[str]:
    lines = []
    tick = START
    for i in range(n):
        if gap_at is not None and i == gap_at:
            tick += step                     # one sample never logged
            continue
        lines.append(f"     {tick}     {channel}   {1000 + (i % 97)} ")
        tick += step
        if clock_fix_at is not None and i == clock_fix_at:
            tick -= 1                        # the scanner corrects its clock
    return lines


def ecg_log(seconds: float = 10.0, *, leads: int = 4, clock_fix_at: int | None = 40,
            gap_at: int | None = 100) -> str:
    n = int(seconds * 400)
    lines = _header("ECG", 1)
    per_lead = [_signal("ECG", f"ECG{k}", n, step=1, clock_fix_at=clock_fix_at,
                        gap_at=gap_at if k == 1 else None) for k in range(1, leads + 1)]
    # Interleaved by time, as the scanner writes them.
    for rows in zip(*per_lead):
        lines.extend(rows)
    return "\n".join(lines + ["", f"FirstTime   = {START}"]) + "\n"


def single_log(kind: str, rate: int, seconds: float = 10.0, *,
               clock_fix_at: int | None = 20, gap_at: int | None = None) -> str:
    step = int(400 / rate)
    n = int(seconds * rate)
    lines = _header(kind, step) + _signal(kind, kind, n, step=step,
                                          clock_fix_at=clock_fix_at, gap_at=gap_at)
    return "\n".join(lines) + "\n"


def info_log(volumes: int = 10, *, first: int = START + 1000, slices: int = 4,
             last_slices: int | None = None) -> str:
    """The acquisition log; ``last_slices`` stops the run during its last
    volume after that many slices, as an operator stopping a long run does."""
    lines = ["UUID        = 08b19a26-bf5b-4d0b-90aa-85b2893c9c75",
             "ScanDate    = 20230914_102000", "LogVersion  = EJA_1",
             "LogDataType = ACQUISITION_INFO", f"NumSlices   = {slices}",
             f"NumVolumes  = {volumes}", "NumEchoes   = 1", "",
             "VOLUME   SLICE   ACQ_START_TICS  ACQ_FINISH_TICS  ECHO", ""]
    for v in range(volumes):
        start = first + v * TR_TICKS
        acquired = last_slices if (last_slices is not None and v == volumes - 1) else slices
        for s in range(acquired):
            # Slice-interleaved: the volume's start is its EARLIEST slice,
            # which is not the first row listed (slice 1 starts first).
            offset = 0 if s == 1 else 5 * (s + 1)
            lines.append(f"     {v}      {s}     {start + offset}     {start + offset + 17}     0")
    return "\n".join(lines) + "\n"


def write_logs(folder: Path, *, seconds: float = 12.0, volumes: int = 10,
               leads: int = 4) -> list[Path]:
    """Four ``.log`` files in ``folder``; returns them."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "Physio_ECG.log": ecg_log(seconds, leads=leads),
        "Physio_PULS.log": single_log("PULS", 200, seconds),
        "Physio_RESP.log": single_log("RESP", 50, seconds, clock_fix_at=None),
        "Physio_Info.log": info_log(volumes),
    }
    out = []
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
        out.append(folder / name)
    return out


def physio_dicom(path: Path, *, seconds: float = 12.0, volumes: int = 10,
                 leads: int = 4) -> Path:
    """A ``_PhysioLog`` DICOM carrying the four logs in ``7fe1,1010``, laid
    out as the CMRR sequence does (one 1024-byte header per log, padded so
    every log takes ``AcquisitionNumber * 1024`` bytes)."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    logs = [
        ("Physio_ECG.log", ecg_log(seconds, leads=leads)),
        ("Physio_PULS.log", single_log("PULS", 200, seconds)),
        ("Physio_RESP.log", single_log("RESP", 50, seconds, clock_fix_at=None)),
        ("Physio_Info.log", info_log(volumes)),
    ]
    encoded = [(name.encode(), text.encode()) for name, text in logs]
    rows = math.ceil((1024 + max(len(t) for _n, t in encoded)) / 1024)
    blob = bytearray()
    for name, text in encoded:
        block = bytearray(rows * 1024)
        block[0:4] = len(text).to_bytes(4, sys.byteorder)
        block[4:8] = len(name).to_bytes(4, sys.byteorder)
        block[8:8 + len(name)] = name
        block[1024:1024 + len(text)] = text
        blob += block

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.3.12.2.1107.5.9.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = "task-rest_bold_PhysioLog"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "RAWDATA", "PHYSIO"]
    ds.AcquisitionNumber = rows
    ds.add_new((0x7FE1, 0x0010), "LO", "SIEMENS CSA NON-IMAGE")
    ds.add_new((0x7FE1, 0x1010), "OB", bytes(blob))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(str(path), ds, enforce_file_format=True)
    return Path(path)

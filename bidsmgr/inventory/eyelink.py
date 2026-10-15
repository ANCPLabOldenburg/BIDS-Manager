"""Reading SR Research EyeLink recordings: ``.edf`` and ``.asc``.

An EyeLink ``.edf`` is NOT European Data Format: it is SR Research's own
binary, which starts with the text ``SR_RESEARCH_1000FILE`` and a short
preamble (date, tracker, camera, serial number). Only SR Research's EDF
Access API reads the rest. Two routes reach it, tried in order:

1. **eyelinkio** (a dependency of BIDS Manager, BSD-3): ships the API for
   macOS, Linux (x86_64) and Windows, so no SDK is needed. On Windows the
   bundled DLL needs the Microsoft Visual C++ 2008 runtime, which the
   Windows installer adds; on Linux it exists for x86_64 only. Only its
   low-level bindings are used, because its ``read_edf`` numbers the samples
   from zero at the sampling rate and so drops the pauses between trials (a
   48 s gap in its own test file): a ``timestamp`` column written from it
   would be wrong by the length of every pause before it.
2. **edf2asc**, from SR Research's EyeLink Developers Kit, when it is on
   the PATH: the file is converted to ASC text, which is then read like an
   ``.asc`` file.

An ``.asc`` file (already converted) is read directly. Either way the
result is one :class:`EyeRecording` with the tracker's own clock
(milliseconds), samples per eye, fixations, saccades, blinks and messages.
Qt-free.
"""

from __future__ import annotations

import ctypes as ct
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

SIGNATURE = b"SR_RESEARCH"
#: The EyeLink float meaning "no data" (gaze lost, eye closed).
MISSING_FLOAT = 1e8 - 1
EYES = ("left", "right")


def is_eyelink_edf(path) -> bool:
    """True when ``path`` is an EyeLink EDF (by its first bytes, not its name:
    a European Data Format ``.edf`` starts with ``0`` and spaces)."""
    try:
        with open(path, "rb") as fh:
            return fh.read(len(SIGNATURE)) == SIGNATURE
    except OSError:
        return False


def is_eyelink_asc(path) -> bool:
    """True when ``path`` is an EyeLink ASC file (``edf2asc`` output)."""
    p = Path(path)
    if p.suffix.lower() != ".asc":
        return False
    try:
        with open(p, "r", encoding="latin-1", errors="replace") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    return "** CONVERTED FROM" in head or "SR_RESEARCH" in head or "EYELINK" in head.upper()


@dataclass
class EyeEvent:
    kind: str            # "fixation" | "saccade" | "blink"
    eye: str             # "left" | "right"
    start: float         # tracker ms
    end: float           # tracker ms
    x: Optional[float] = None       # fixation: average; saccade: start
    y: Optional[float] = None
    x_end: Optional[float] = None   # saccade end
    y_end: Optional[float] = None
    peak_velocity: Optional[float] = None


@dataclass
class EyeRecording:
    """One EyeLink recording, on the tracker's clock."""

    path: Path
    reader: str                                  # "eyelinkio" | "edf2asc" | "asc"
    info: dict = field(default_factory=dict)     # preamble: date, version, source, ...
    sfreq: float = 0.0
    eyes: tuple[str, ...] = ()
    pupil: str = ""                              # "area" | "diameter"
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))   # ms
    gaze: dict[str, np.ndarray] = field(default_factory=dict)        # eye -> (n, 3)
    events: list[EyeEvent] = field(default_factory=list)
    messages: list[tuple[float, str]] = field(default_factory=list)
    blocks: list[tuple[float, float]] = field(default_factory=list)  # recording blocks
    #: The heuristic filter levels (file, link) of ``!MODE RECORD``.
    filters: tuple[str, str] = ("", "")

    @property
    def duration_s(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return float(self.times[-1] - self.times[0]) / 1000.0 + 1.0 / (self.sfreq or 1000.0)

    def messages_matching(self, text: str) -> list[float]:
        return [t for t, m in self.messages if text in m]


# ---------------------------------------------------------------------------
# What can read an EDF here
# ---------------------------------------------------------------------------


def _bindings():
    """eyelinkio's ctypes bindings and constants, or ``(None, None, why)``."""
    try:
        from eyelinkio.edf import _defines as defines
        from eyelinkio.edf import _edf2py as api
    except ImportError:
        return None, None, "the eyelinkio package is not installed"
    except OSError as exc:  # the bundled library does not load on this machine
        return None, None, f"eyelinkio cannot load SR Research's EDF reader here ({exc})"
    return api, defines, ""


def edf2asc_path() -> Optional[str]:
    """``edf2asc`` from SR Research's Developers Kit, when on the PATH."""
    for name in ("edf2asc", "edf2asc64", "edf2asc.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def edf_reader() -> tuple[str, str]:
    """``(reader, why_not)``: which route reads an EyeLink EDF here
    (``"eyelinkio"`` or ``"edf2asc"``), or ``""`` and why none can."""
    api, _defines, why = _bindings()
    if api is not None:
        return "eyelinkio", ""
    if edf2asc_path():
        return "edf2asc", ""
    return "", f"{why}. {remedy()}"


#: Microsoft's download of the Visual C++ 2008 SP1 runtime (x64) that the
#: Windows build of SR Research's EDF reader is linked against
#: (``Microsoft.VC90.CRT`` 9.0.21022.8 in its manifest).
VC2008_X64_URL = ("https://download.microsoft.com/download/5/D/8/"
                  "5D8C65CB-C849-4025-8E95-C3966CAFD8AE/vcredist_x64.exe")


def remedy(platform: Optional[str] = None, machine: Optional[str] = None) -> str:
    """What makes an EyeLink ``.edf`` readable on this machine, in a sentence."""
    import platform as _platform
    import sys

    platform = platform or sys.platform
    machine = (machine or _platform.machine()).lower()
    alternative = ("Or install SR Research's EyeLink Developers Kit (edf2asc on the "
                   "PATH), or export the recording as .asc in EyeLink Data Viewer: an "
                   ".asc file needs no reader.")
    if platform.startswith("win"):
        return ("SR Research's EDF reader for Windows needs the Microsoft Visual C++ 2008 "
                f"SP1 Redistributable (x64), which Windows does not include: install it "
                f"from {VC2008_X64_URL} and restart BIDS Manager. " + alternative)
    if platform.startswith("linux") and machine not in ("x86_64", "amd64"):
        return (f"SR Research publishes its EDF reader for Linux on x86_64 only, and this "
                f"machine is {machine}. " + alternative)
    return "Reinstall BIDS Manager (it brings the eyelinkio package). " + alternative


# ---------------------------------------------------------------------------
# The preamble (no reader needed)
# ---------------------------------------------------------------------------


def _extent(lo: float, hi: float) -> int:
    """Pixels between two display coordinates. EyeLink's are inclusive
    (``0 0 1919 1079`` is 1920 x 1080), but a tracker set up with the
    screen's size itself (``0 0 1280 1024``) is common: a round size wins."""
    n = int(round(hi - lo))
    return n if n % 16 == 0 and (n + 1) % 16 != 0 else n + 1


def _preamble_fields(lines) -> dict:
    out: dict = {}
    keys = {"DATE": "date", "VERSION": "version", "SOURCE": "source", "CAMERA": "camera",
            "SERIAL NUMBER": "serial", "CAMERA_CONFIG": "camera_config", "TYPE": "type"}
    for raw in lines:
        line = raw.lstrip("*").strip()
        if not line:
            continue
        if line.startswith("EYELINK") and "tracker" not in out and ":" not in line:
            out["tracker"] = line            # "EYELINK II CL v5.15 Jan 24 2018"
            continue
        if line.startswith("DISPLAY_COORDS"):
            nums = [float(n) for n in line.split()[1:5]]
            if len(nums) == 4:
                out["display"] = (_extent(nums[0], nums[2]), _extent(nums[1], nums[3]))
            continue
        key, sep, value = line.partition(":")
        if sep and key.strip() in keys:
            out[keys[key.strip()]] = value.strip()
    return out


def read_preamble(path) -> dict:
    """The text an EyeLink EDF (or ASC) starts with, without any reader:
    date, version, source, tracker, camera, serial number, display."""
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            head = fh.read(8192)
    except OSError:
        return {}
    if p.suffix.lower() == ".asc":
        lines = [ln for ln in head.decode("latin-1", "replace").splitlines()
                 if ln.startswith("**")]
    else:
        text = head.split(b"ENDP:")[0].decode("latin-1", "replace")
        lines = text.splitlines()[1:]
        # DISPLAY_COORDS follows the preamble as the first message.
        m = re.search(rb"DISPLAY_COORDS\s+[\d.\s-]+", head)
        if m:
            lines.append(m.group(0).decode("latin-1"))
    return _preamble_fields(lines)


# ---------------------------------------------------------------------------
# EDF through eyelinkio's bindings
# ---------------------------------------------------------------------------


def _ascii_copy(path: Path):
    """The API takes a char* path: a non-ASCII path is read from an ASCII
    copy in a temporary folder (the caller removes it)."""
    try:
        str(path).encode("ascii")
        return path, None
    except UnicodeEncodeError:
        tmp = Path(tempfile.mkdtemp(prefix="bidsmgr-eyelink-"))
        target = tmp / "recording.edf"
        shutil.copy2(path, target)
        return target, tmp


def _message_text(event) -> str:
    raw = ct.string_at(ct.byref(event.message[0]), event.message.contents.len + 1)[2:]
    return raw.decode("utf-8", "replace").replace("\x00", "").strip()


def read_edf_api(path, *, samples: bool = True) -> EyeRecording:
    """Read an EyeLink EDF with eyelinkio's bindings, keeping the tracker's
    clock. ``samples=False`` reads only events and messages (a probe)."""
    api, defines, why = _bindings()
    if api is None:
        raise RuntimeError(why)
    path = Path(path)
    readable, tmp = _ascii_copy(path)
    ev = defines.event_constants
    rec = EyeRecording(path=path, reader="eyelinkio", info=read_preamble(path))
    times: list[float] = []
    cols: dict[int, tuple[list, list, list]] = {0: ([], [], []), 1: ([], [], [])}
    eye_index: Optional[int] = None
    block_start: Optional[float] = None
    err = ct.c_int(1)
    edf = api.edf_open_file(os.fsencode(str(readable)), 2, 1, 1 if samples else 0,
                            ct.byref(err))
    try:
        if not edf or err.value != 0:
            raise OSError(f"SR Research's EDF reader could not open {path.name} "
                          f"(error {err.value})")
        kinds = {"ENDFIX": "fixation", "ENDSACC": "saccade", "ENDBLINK": "blink"}
        while True:
            code = api.edf_get_next_data(edf)
            name = ev.get(code)
            if name == "NO_PENDING_ITEMS" or code == 0:
                break
            if name == "SAMPLE_TYPE":
                s = api.edf_get_float_data(edf).contents.fs
                times.append(float(s.time))
                for i in (0, 1):
                    gx, gy, pa = cols[i]
                    gx.append(s.gx[i])
                    gy.append(s.gy[i])
                    pa.append(s.pa[i])
            elif name == "RECORDING_INFO":
                r = api.edf_get_float_data(edf).contents.rec
                if r.state:                                  # recording started
                    rec.sfreq = float(r.sample_rate)
                    eye_index = int(r.eye) - 1               # 0 left, 1 right, 2 both
                    rec.pupil = "diameter" if int(r.pupil_type) == 1 else "area"
                    block_start = float(r.time)
                elif block_start is not None:
                    rec.blocks.append((block_start, float(r.time)))
                    block_start = None
            elif name in kinds:
                e = api.edf_get_float_data(edf).contents.fe
                eye = EYES[int(e.eye)] if int(e.eye) in (0, 1) else "left"
                event = EyeEvent(kind=kinds[name], eye=eye, start=float(e.sttime),
                                 end=float(e.entime))
                if name == "ENDFIX":
                    event.x, event.y = _value(e.gavx), _value(e.gavy)
                elif name == "ENDSACC":
                    event.x, event.y = _value(e.gstx), _value(e.gsty)
                    event.x_end, event.y_end = _value(e.genx), _value(e.geny)
                    event.peak_velocity = _value(e.pvel)
                rec.events.append(event)
            elif name == "MESSAGEEVENT":
                e = api.edf_get_float_data(edf).contents.fe
                rec.messages.append((float(e.sttime), _message_text(e)))
            elif name in ("INPUTEVENT", "BUTTONEVENT"):
                e = api.edf_get_float_data(edf).contents.fe
                word, value = (("INPUT", e.input) if name == "INPUTEVENT"
                               else ("BUTTON", e.buttons))
                if e.sttime:           # a port read before recording started has no time
                    rec.messages.append((float(e.sttime), f"{word} {int(value)}"))
    finally:
        if edf:
            api.edf_close_file(edf)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    rec.eyes = {0: ("left",), 1: ("right",), 2: ("left", "right")}.get(eye_index, ())
    _finish(rec, times, {EYES[i]: cols[i] for i in (0, 1)})
    return rec


def _value(v: float) -> Optional[float]:
    v = float(v)
    return None if v >= MISSING_FLOAT or v <= -32768 else v


def _finish(rec: EyeRecording, times: list[float], cols: dict) -> None:
    """Arrays from the lists, missing values as NaN, the mode messages read."""
    rec.times = np.asarray(times, dtype=np.float64)
    for eye in rec.eyes:
        gx, gy, pa = cols[eye]
        arr = np.column_stack([np.asarray(gx, dtype=np.float64),
                               np.asarray(gy, dtype=np.float64),
                               np.asarray(pa, dtype=np.float64)]) if gx else np.zeros((0, 3))
        if arr.size:
            arr[:, :2][(arr[:, :2] >= MISSING_FLOAT) | (arr[:, :2] <= -32768)] = np.nan
        rec.gaze[eye] = arr
    for _t, text in rec.messages:
        if text.startswith("!MODE RECORD"):
            parts = text.split()
            if len(parts) >= 6:
                rec.filters = (parts[4], parts[5])        # file, link filter level
        elif text.startswith(("DISPLAY_COORDS", "GAZE_COORDS")) and "display" not in rec.info:
            nums = [float(n) for n in text.split()[1:5]]
            if len(nums) == 4:
                rec.info["display"] = (_extent(nums[0], nums[2]), _extent(nums[1], nums[3]))


# ---------------------------------------------------------------------------
# ASC (edf2asc output)
# ---------------------------------------------------------------------------


def _num(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        return float("nan")


def read_asc(path, *, source: Optional[Path] = None, reader: str = "asc") -> EyeRecording:
    """Read an EyeLink ASC file. ``source`` is the EDF it was made from."""
    path = Path(path)
    rec = EyeRecording(path=Path(source) if source else path, reader=reader)
    header: list[str] = []
    times: list[float] = []
    cols = {"left": ([], [], []), "right": ([], [], [])}
    sample_eyes: tuple[str, ...] = ()
    block_start: Optional[float] = None
    kinds = {"EFIX": "fixation", "ESACC": "saccade", "EBLINK": "blink"}
    eye_of = {"L": "left", "R": "right"}
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            if not line or line[0] in "\r\n":
                continue
            c = line[0]
            if c.isdigit():
                parts = line.split()
                if not parts:
                    continue
                times.append(_num(parts[0]))
                values = parts[1:]
                for k, eye in enumerate(sample_eyes or ("left",)):
                    trio = values[3 * k: 3 * k + 3]
                    gx, gy, pa = (_num(v) if v != "." else float("nan") for v in
                                  (trio + [".", ".", "."])[:3])
                    cols[eye][0].append(gx)
                    cols[eye][1].append(gy)
                    cols[eye][2].append(pa)
                continue
            if line.startswith("**"):
                header.append(line.rstrip("\n"))
                continue
            parts = line.split()
            word = parts[0]
            if word == "MSG" and len(parts) >= 2:
                text = line.split(None, 2)[2].strip() if len(parts) > 2 else ""
                rec.messages.append((_num(parts[1]), text))
            elif word in kinds and len(parts) >= 4:
                event = EyeEvent(kind=kinds[word], eye=eye_of.get(parts[1], "left"),
                                 start=_num(parts[2]), end=_num(parts[3]))
                if word == "EFIX" and len(parts) >= 7:
                    event.x, event.y = _num(parts[5]), _num(parts[6])
                elif word == "ESACC" and len(parts) >= 9:
                    event.x, event.y = _num(parts[5]), _num(parts[6])
                    event.x_end, event.y_end = _num(parts[7]), _num(parts[8])
                    if len(parts) >= 11:
                        event.peak_velocity = _num(parts[10])
                rec.events.append(event)
            elif word == "START" and len(parts) >= 2:
                block_start = _num(parts[1])
            elif word == "END" and len(parts) >= 2 and block_start is not None:
                rec.blocks.append((block_start, _num(parts[1])))
                block_start = None
            elif word == "SAMPLES":
                eyes = tuple(e for e, tag in (("left", "LEFT"), ("right", "RIGHT"))
                             if tag in parts)
                sample_eyes = eyes or sample_eyes
                if "RATE" in parts:
                    rec.sfreq = _num(parts[parts.index("RATE") + 1])
            elif word == "PUPIL" and len(parts) >= 2:
                rec.pupil = "diameter" if parts[1].upper() == "DIAMETER" else "area"
            elif word in ("INPUT", "BUTTON") and len(parts) >= 3:
                rec.messages.append((_num(parts[1]), f"{word} {' '.join(parts[2:])}"))
    rec.info = _preamble_fields(header)
    rec.eyes = sample_eyes
    _finish(rec, times, cols)
    return rec


def _via_edf2asc(path: Path) -> EyeRecording:
    exe = edf2asc_path()
    if not exe:
        raise RuntimeError("edf2asc is not on the PATH")
    tmp = Path(tempfile.mkdtemp(prefix="bidsmgr-edf2asc-"))
    try:
        copy = tmp / "recording.edf"
        shutil.copy2(path, copy)
        proc = subprocess.run([exe, "-y", str(copy)], capture_output=True, text=True,
                              timeout=600)
        asc = copy.with_suffix(".asc")
        if not asc.is_file():
            raise RuntimeError(f"edf2asc made no ASC file (exit {proc.returncode}): "
                               f"{(proc.stderr or proc.stdout or '').strip()[-300:]}")
        return read_asc(asc, source=path, reader="edf2asc")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def read_recording(path, *, samples: bool = True) -> EyeRecording:
    """An EyeLink ``.edf`` or ``.asc``, by whichever reader is here."""
    path = Path(path)
    if is_eyelink_asc(path):
        return read_asc(path)
    if not is_eyelink_edf(path):
        raise ValueError(f"{path.name} is not an EyeLink recording")
    reader, why = edf_reader()
    if reader == "eyelinkio":
        return read_edf_api(path, samples=samples)
    if reader == "edf2asc":
        return _via_edf2asc(path)
    raise RuntimeError(why)


# ---------------------------------------------------------------------------
# What the messages say about the calibration and the set-up
# ---------------------------------------------------------------------------

_VALIDATION = re.compile(
    r"!CAL VALIDATION (?P<type>\S+) \S+ (?P<eye>LEFT|RIGHT)\s+\S+\s+ERROR\s+"
    r"(?P<avg>[\d.]+) avg\.\s+(?P<max>[\d.]+) max")
_CALIBRATION = re.compile(r"CALIBRATION \((?P<type>[^,)]+),(?P<method>[^)]+)\) FOR (?P<eye>\w+)")


def calibration_facts(rec: EyeRecording, eye: str) -> dict:
    """What the calibration and validation messages say for ``eye``, in BIDS
    terms (``CalibrationType``, ``CalibrationCount``, ``EyeTrackingMethod``,
    ``AverageCalibrationError``, ``MaximalCalibrationError``,
    ``PupilFitMethod``)."""
    out: dict = {}
    count = 0
    for _t, text in rec.messages:
        m = _CALIBRATION.search(text)
        if m and m.group("eye").lower() == eye:
            count += 1
            out["CalibrationType"] = m.group("type")
            out["EyeTrackingMethod"] = m.group("method")
        m = _VALIDATION.search(text)
        if m and m.group("eye").lower() == eye:
            out["AverageCalibrationError"] = float(m.group("avg"))
            out["MaximalCalibrationError"] = float(m.group("max"))
        if text.startswith("ELCL_PROC"):
            method = text.split()[1].upper() if len(text.split()) > 1 else ""
            fit = {"CENTROID": "centre-of-mass", "ELLIPSE": "ellipse"}.get(method)
            if fit:
                out["PupilFitMethod"] = fit
    if count:
        out["CalibrationCount"] = count
    return out


__all__ = ["EYES", "EyeEvent", "EyeRecording", "SIGNATURE", "calibration_facts",
           "edf2asc_path", "edf_reader", "is_eyelink_asc", "is_eyelink_edf",
           "read_asc", "read_edf_api", "read_preamble", "read_recording"]

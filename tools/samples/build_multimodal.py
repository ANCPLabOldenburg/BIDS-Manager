#!/usr/bin/env python3
"""Build the multimodal sample datasets from the published version 2.

    python tools/samples/build_multimodal.py v3 SOURCE_V2 OUT
    python tools/samples/build_multimodal.py multimodal2 SOURCE_V2 OUT

``SOURCE_V2`` is the unpacked ``BIDS_Manager_multimodal_sample_v2`` folder.
Each command writes ``OUT/<name>/`` and ``OUT/<name>.zip``. Run it with an
environment that has BIDS Manager installed (pydicom, mne, eyelinkio).

**Multimodal sample, version 3.** Version 2 with ONE change: the BOLD run was
shortened to its first 60 of 100 volumes to keep the download small, and its
physiological log still covered all 100, so the log's trigger marked 40
volumes the image does not have and its tables ran 34 s past the image's end.
The log is cut to the volumes the run's image holds (see ``trim_physio``).
Everything else, including the subject identities the course teaches to
reconcile, is version 2 unchanged.

**Multimodal 2, version 1.** Version 3 made coherent in identity and time, as
one participant would be recorded in a real multimodal study:

* every recording carries the MRI's study code, ``OL_3844`` (male, born
  1993, 30 years old at the MRI, 66 kg, 1.73 m), in the field its format has
  for it: the DICOM patient module, the EDF+ patient field, the FIF subject;
* four visits: 14 Sep 2023, MRI and spectroscopy in one session, with the eye
  tracker recording during the resting BOLD run; 21 Sep, EEG; 28 Sep, MEG;
  5 Oct, PET at another site (whole days shifted, times of day kept);
* the spectroscopy, acquired in another session, is moved into the MRI's
  study (study UID, frame of reference, series numbers, times right after
  the last MRI series);
* ``participants.csv`` agrees with the headers (age 30), and the PET dose
  sheet holds realistic FDG values consistent with the injected activity the
  PET header records (the version 2 sheet implied 181 kg of tracer).

Zips are written the same way every time: sorted entries, one fixed
timestamp, no macOS metadata (``__MACOSX/``, ``._*``, ``.DS_Store``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path

STAMP = (2026, 1, 1, 0, 0, 0)
TICK = 0.0025                                   # one CMRR scanner tick, seconds
JUNK = {".DS_Store", "Thumbs.db"}

V3 = "BIDS_Manager_multimodal_sample_v3"
MULTIMODAL2 = "BIDS_Manager_multimodal_2_sample_v1"


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def copy_sample(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*JUNK, "._*"))


def dicoms(folder: Path):
    import pydicom

    for path in sorted(folder.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            try:
                yield path, pydicom.dcmread(str(path))
            except Exception:  # noqa: BLE001 - not a DICOM
                continue


def write_zip(folder: Path) -> Path:
    archive = folder.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(folder.rglob("*")):
            if path.name in JUNK or path.name.startswith("._") or "__MACOSX" in path.parts:
                continue
            name = f"{folder.name}/{path.relative_to(folder).as_posix()}"
            if path.is_dir():
                info = zipfile.ZipInfo(name + "/", STAMP)
                info.external_attr = 0o40755 << 16
                zf.writestr(info, b"")
                continue
            info = zipfile.ZipInfo(name, STAMP)
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as fh:
                zf.writestr(info, fh.read(), compresslevel=9)
    return archive


def _tm(seconds: float) -> str:
    """Seconds of the day as a DICOM TM with microseconds."""
    seconds %= 86400
    micro = int(round(seconds * 1e6))
    whole, frac = divmod(micro, 1_000_000)
    return f"{whole // 3600:02d}{whole % 3600 // 60:02d}{whole % 60:02d}.{frac:06d}"


def _tm_seconds(text: str) -> float:
    text = str(text).strip()
    whole, _dot, frac = text.partition(".")
    return int(whole[:2]) * 3600 + int(whole[2:4]) * 60 + int(whole[4:6]) + (
        float("0." + frac) if frac else 0.0)


# ---------------------------------------------------------------------------
# Version 3: the physiological log cut to the run's image
# ---------------------------------------------------------------------------


def _logs(ds) -> tuple[list[tuple[bytes, str]], int]:
    """``([(block header, text)], rows)`` of a CMRR physio DICOM: each log is
    a block of ``rows * 1024`` bytes, its length and name in the first 1024
    and the text after (the layout ``converter.physio_cmrr`` reads)."""
    data = bytes(ds[0x7FE1, 0x1010].value)
    rows = int(ds.AcquisitionNumber)
    waves = (len(data) // rows) // 1024
    length = len(data) // waves
    out = []
    for w in range(waves):
        block = data[w * length:(w + 1) * length]
        size = int.from_bytes(block[0:4], "little")
        out.append((block[:1024], block[1024:1024 + size].decode("utf-8")))
    return out, rows


def _encode(logs: list[tuple[bytes, str]]) -> tuple[bytes, int]:
    texts = [text.encode("utf-8") for _head, text in logs]
    rows = math.ceil((1024 + max(len(t) for t in texts)) / 1024)
    blob = bytearray()
    for (head, _text), text in zip(logs, texts):
        block = bytearray(rows * 1024)
        block[0:1024] = head
        block[0:4] = len(text).to_bytes(4, "little")
        block[1024:1024 + len(text)] = text
        blob += block
    return bytes(blob), rows


def _info_volumes(text: str) -> dict[int, tuple[int, int, int]]:
    """``{volume: (first start tick, last finish tick, slice rows)}`` of an
    acquisition (Info) log, first echo."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and all(p.isdigit() for p in parts[:5]):
            rows.append(tuple(int(p) for p in parts[:5]))
    echo = min((r[4] for r in rows), default=0)
    out: dict[int, list[int]] = {}
    for volume, _slice, start, finish, e in rows:
        if e != echo:
            continue
        v = out.setdefault(volume, [start, finish, 0])
        v[0], v[1], v[2] = min(v[0], start), max(v[1], finish), v[2] + 1
    return {k: tuple(v) for k, v in out.items()}


def trim_physio(mri: Path) -> dict:
    """Cut every CMRR physiological log in ``mri`` to the volumes its run's
    image holds. The run is the series whose description is the log's
    without ``_PhysioLog``; its volume count is the number of image files
    (one per volume on these scanners). The signals keep what was recorded
    up to the end of the last volume kept plus the same tail the scanner
    logged after the original last volume, ``NumVolumes`` and ``LastTime``
    say the new length, and the log's own time (when the scanner wrote it,
    0.7 s after the run ended) moves with the end of the run."""
    import pydicom

    series: dict[str, list[Path]] = {}
    logs = []
    for path, ds in dicoms(mri):
        description = str(ds.get("SeriesDescription", ""))
        series.setdefault(description, []).append(path)
        if description.endswith("_PhysioLog"):
            logs.append(path)
    report = {}
    for path in logs:
        ds = pydicom.dcmread(str(path))
        run = str(ds.SeriesDescription)[: -len("_PhysioLog")]
        n = len(series.get(run, []))
        blocks, _rows = _logs(ds)
        info = next(text for _h, text in blocks if "ACQUISITION_INFO" in text)
        volumes = _info_volumes(info)
        if not n or n >= len(volumes):
            report[path.name] = f"left as it is ({len(volumes)} volumes, image {n})"
            continue
        last_kept = sorted(volumes)[n - 1]
        last_logged = int(re.search(r"LastTime\s*=\s*(\d+)", info).group(1))
        tail = last_logged - max(v[1] for v in volumes.values())
        cutoff = volumes[last_kept][1] + tail
        trimmed = []
        new_last = 0
        for head, text in blocks:
            lines = text.splitlines(keepends=True)
            kept = []
            for line in lines:
                parts = line.split()
                if "ACQUISITION_INFO" in text:
                    if len(parts) >= 5 and all(p.isdigit() for p in parts[:5]):
                        if int(parts[0]) > last_kept:
                            continue
                    elif line.lstrip().startswith("NumVolumes"):
                        line = re.sub(r"=\s*\d+", f"= {n}", line)
                elif len(parts) >= 3 and parts[0].isdigit():
                    if int(parts[0]) > cutoff:
                        continue
                    new_last = max(new_last, int(parts[0]))
                kept.append(line)
            trimmed.append((head, "".join(kept)))
        trimmed = [(head, re.sub(r"(LastTime\s*=\s*)\d+", rf"\g<1>{new_last}", text))
                   for head, text in trimmed]
        blob, rows = _encode(trimmed)
        written = _tm_seconds(ds.SeriesTime) - max(v[1] for v in volumes.values()) * TICK
        ds[0x7FE1, 0x1010].value = blob
        ds.AcquisitionNumber = rows
        ds.SeriesTime = _tm(volumes[last_kept][1] * TICK + written)
        if (0x7FE1, 0x0000) in ds:              # a group length the new size invalidates
            del ds[0x7FE1, 0x0000]
        ds.save_as(str(path), enforce_file_format=True)
        report[path.name] = (f"{len(volumes)} -> {n} volumes, signals to tick {new_last} "
                             f"({(new_last - cutoff + tail) * TICK:+.4f} s after the last "
                             f"volume), written at {ds.SeriesTime}")
    return report


def build_v3(source: Path, out: Path) -> Path:
    target = out / V3
    copy_sample(source, target)
    print("physio:", json.dumps(trim_physio(target / "raw/sub-001/mri"), indent=1))
    manifest = json.loads((target / "manifest.json").read_text())
    manifest = {("task-rest physio log, cut to the run's 60 volumes"
                 if k == "task-rest physio log" else k): v for k, v in manifest.items()}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    readme = (target / "README.md").read_text()
    readme = readme.replace("multimodal sample dataset (version 2)",
                            "multimodal sample dataset (version 3)")
    readme = readme.replace(
        "│                   the physiological log recorded during that BOLD run\n",
        "│                   the physiological log recorded during that BOLD run,\n"
        "│                   cut to the same 60 volumes\n")
    readme = readme.replace(
        "and the\nMEG recording to its first 30 seconds; nothing else in any file was changed.",
        "and the\nMEG recording to its first 30 seconds. The BOLD run's physiological log was "
        "cut\nto the same 60 volumes (version 3; version 2 kept all 100, so its trigger\n"
        "marked 40 volumes the image does not have). Nothing else in any file was changed.")
    for old, new in (
        # Measured on the zip: 311.4 MB of files (what Finder shows), 124.9 MB zipped.
        ("About 305 MB unpacked, 126 MB as a zip.", "About 311 MB unpacked, 125 MB as a zip."),
        ("- **Physiological recordings split by signal.** The physio log of the BOLD run\n"
         "  becomes one `_physio.tsv.gz` per signal (cardiac, ECG, respiratory, trigger).",
         "- **Physiological recordings, one table per sampling rate.** The physio log of\n"
         "  the BOLD run becomes `_recording-ecg_` (the four ECG leads, 400 Hz),\n"
         "  `_recording-cardiac_` (pulse, 200 Hz) and `_recording-respiratory_physio.tsv.gz`\n"
         "  (breathing, 50 Hz), each with the scanner trigger, which marks the run's 60\n"
         "  volumes."),
        ("Converted and validated with BIDS Manager 1.4.1 on 2026-10-13: one error, in\n"
         "the PET scan's row of its scans table (a time without a date, which BIDS\n"
         "Manager writes for this scanner because its header holds no acquisition date);\n"
         "the warnings are the TODO placeholders for what no file records (authors,\n"
         "licence, task descriptions).",
         "Converted and validated with BIDS Manager 1.5.2 on 2026-10-26, the dose sheet\n"
         "imported and the scan's proposals accepted as they are: 69 files and four\n"
         "errors, two PET reconstruction fields no scanner records and the eye tracker's\n"
         "`StimulusPresentation` (one per eye); the warnings are the TODO placeholders\n"
         "for what no file records (authors, licence, task descriptions)."),
        ("- **Subject identity to reconcile.** Each source states who the participant is\n"
         "  in its own way: the MRI, the spectroscopy and the PET carry patient\n"
         "  identifiers in their DICOM headers, the EEG, MEG and eye tracking take theirs\n"
         "  from the folder. The scan therefore proposes more than one subject, and\n"
         "  several sessions; setting every row to one subject and one session is one\n"
         "  edit per column in the table, which is why the workflow puts a table between\n"
         "  the scan and the conversion.\n",
         "- **Subject identity to reconcile.** Each source states who the participant is\n"
         "  in its own way: the MRI, the spectroscopy and the PET carry patient\n"
         "  identifiers in their DICOM headers (three different ones), the EEG and MEG\n"
         "  take theirs from the folder, and the eye tracker, whose file names no\n"
         "  participant, gets a number of its own. The scan therefore proposes four\n"
         "  subjects, and several sessions; setting every row to one subject and one\n"
         "  session is one edit per column in the table, which is why the workflow puts\n"
         "  a table between the scan and the conversion.\n"),
    ):
        if old not in readme:
            raise SystemExit(f"README of version 2 changed; expected:\n{old}")
        readme = readme.replace(old, new)
    (target / "README.md").write_text(readme)
    return target


# ---------------------------------------------------------------------------
# Multimodal 2: one participant, four visits
# ---------------------------------------------------------------------------

#: The participant, as the MRI headers state them.
CODE = "OL_3844"
BIRTH = dt.date(1993, 1, 1)
VISIT_EEG = dt.datetime(2023, 9, 21, 14, 0, 0)
VISIT_MEG = dt.datetime(2023, 9, 28, 9, 30, 0)
VISIT_PET = dt.date(2023, 10, 5)
#: The eye tracker starts before the resting BOLD run (10:20:12) and records
#: through it (its 200 s cover the run's 51 s).
EYE_START = dt.datetime(2023, 9, 14, 10, 19, 30)
#: FDG at injection: a molar activity of 300 GBq/umol, a molar mass of
#: 181.15 g/mol. With the 75.85 MBq the PET header records, 0.0458 ug.
FDG_MOLAR_ACTIVITY_BQ_PER_UMOL = 300e9
FDG_MOLAR_MASS_UG_PER_UMOL = 181.15

_IDENTITY = ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge",
             "PatientWeight", "PatientSize")
_STUDY = ("StudyInstanceUID", "StudyID", "StudyDate", "StudyTime", "StudyDescription",
          "AccessionNumber", "ReferringPhysicianName", "FrameOfReferenceUID",
          "PerformedProcedureStepStartDate", "PerformedProcedureStepStartTime")


def _anchor(mri: Path):
    """The MRI's first image: whose identity and study the others take."""
    for _path, ds in dicoms(mri):
        if "PhysioLog" not in str(ds.get("SeriesDescription", "")):
            return ds
    raise SystemExit("no MRI image in the sample")


def _walk(ds):
    for el in ds:
        if el.VR == "SQ":
            for item in el.value:
                yield from _walk(item)
        else:
            yield el


def mrs_into_the_mri_session(mrs: Path, mri: Path) -> dict:
    """The spectroscopy as one more series of the MRI's session: its
    identity, study and frame of reference, numbered after the last MRI
    series, every time moved so it starts two minutes after the last MRI
    acquisition (its two series keep their order and spacing)."""
    anchor = _anchor(mri)
    last = 0.0
    numbers = set()
    for _path, ds in dicoms(mri):
        numbers.add(int(ds.get("SeriesNumber") or 0))
        if ds.get("AcquisitionTime"):
            last = max(last, _tm_seconds(ds.AcquisitionTime))
    items = list(dicoms(mrs))
    first = min(_tm_seconds(ds.AcquisitionTime) for _p, ds in items)
    shift = (last + 120.0) - first
    # After the last acquired series; the scanner numbers its reports 99.
    top = max(n for n in numbers if n < 99)
    renumber = {old: top + 1 + i for i, old in enumerate(
        sorted({int(ds.SeriesNumber) for _p, ds in items}))}
    for path, ds in items:
        for el in _walk(ds):
            if el.VR == "TM" and el.value and el.keyword not in ("StudyTime",
                                                                  "PerformedProcedureStepStartTime"):
                el.value = _tm(_tm_seconds(el.value) + shift)
        for key in _IDENTITY + _STUDY:
            if key in anchor:
                setattr(ds, key, anchor.data_element(key).value)
        ds.SeriesNumber = renumber[int(ds.SeriesNumber)]
        ds.save_as(str(path), enforce_file_format=True)
    return {"series": renumber, "starts": _tm(last + 120.0)}


def pet_to_the_participant(pet: Path, mri: Path) -> dict:
    """The PET as the same participant scanned at another site: the MRI's
    identity (its own study, as a second hospital would have), every date in
    the file (standard and private, in sequences too) moved by the same
    whole number of days to the PET visit, times of day kept."""
    anchor = _anchor(mri)
    items = list(dicoms(pet))
    old = dt.datetime.strptime(str(items[0][1].StudyDate), "%Y%m%d").date()
    days = (VISIT_PET - old).days

    def move(text: str) -> str:
        day = dt.datetime.strptime(text[:8], "%Y%m%d").date() + dt.timedelta(days=days)
        return day.strftime("%Y%m%d") + text[8:]

    for path, ds in items:
        for el in _walk(ds):
            if el.VR in ("DA", "DT") and isinstance(el.value, str) and re.match(r"\d{8}", el.value):
                el.value = move(el.value)
        for key in _IDENTITY:
            if key in anchor:
                setattr(ds, key, anchor.data_element(key).value)
        ds.StudyDescription = "BRAIN^FDG"
        ds.SeriesDescription = str(ds.SeriesDescription).replace("unif", "brain")
        ds.save_as(str(path), enforce_file_format=True)
    return {"days moved": days, "study date": move(old.strftime("%Y%m%d"))}


#: Month and day names as the formats spell them, whatever the machine's
#: locale (strftime would write "Okt" on a German Mac).
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _edf_date(day: dt.date) -> str:
    """EDF+ dates: ``21-SEP-2023``."""
    return f"{day.day:02d}-{_MONTHS[day.month - 1].upper()}-{day.year}"


def _edf_field(text: str, width: int) -> bytes:
    return text.ljust(width)[:width].encode("ascii")


def eeg_to_the_visit(eeg: Path) -> dict:
    """The EDF+ header of each run: the participant (``code sex birthdate
    name``, the name unknown), the visit's date, and start times one after
    the other (rest, then motor imagery, 2 minutes apart)."""
    out = {}
    start = VISIT_EEG
    for path in sorted(eeg.glob("*.edf"), key=lambda p: ("rest" not in p.name, p.name)):
        data = bytearray(path.read_bytes())
        data[8:88] = _edf_field(f"{CODE} M {_edf_date(BIRTH)} X", 80)
        data[88:168] = _edf_field(f"Startdate {_edf_date(start.date())} X X BCI2000", 80)
        data[168:176] = _edf_field(start.strftime("%d.%m.%y"), 8)
        data[176:184] = _edf_field(start.strftime("%H.%M.%S"), 8)
        path.write_bytes(bytes(data))
        out[path.name] = start.isoformat()
        start += dt.timedelta(minutes=2)
    return out


def meg_to_the_visit(meg: Path) -> dict:
    """The FIF's measurement date and subject, through MNE, in the format it
    was stored in."""
    import mne

    out = {}
    for path in sorted(meg.glob("*.fif")):
        raw = mne.io.read_raw_fif(str(path), preload=True, verbose="error")
        raw.set_meas_date(VISIT_MEG.replace(tzinfo=dt.timezone.utc))
        raw.info["subject_info"] = {"his_id": CODE, "sex": 1, "hand": 1, "birthday": BIRTH}
        fmt = raw.orig_format
        tmp = path.with_name(path.stem + ".part_raw.fif")
        raw.save(str(tmp), fmt=fmt, overwrite=True, verbose="error")
        tmp.replace(path)
        out[path.name] = VISIT_MEG.isoformat()
    return out


def eyetracking_during_the_bold(eye: Path) -> dict:
    """The EyeLink file's recording date (``DATE: Www Mmm dd hh:mm:ss yyyy``
    in its preamble), replaced in place at the same length."""
    out = {}
    pattern = re.compile(rb"DATE: ((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
                         rb"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [ 0-9]\d "
                         rb"\d\d:\d\d:\d\d \d{4})")
    e = EYE_START
    stamp = (f"{_DAYS[e.weekday()]} {_MONTHS[e.month - 1]} {e.day:2d} "
             f"{e:%H:%M:%S} {e.year}").encode("ascii")
    for path in sorted(eye.glob("*.edf")):
        data = bytearray(path.read_bytes())
        found = pattern.search(bytes(data[:4096]))
        if found is None or len(found.group(1)) != len(stamp):
            raise SystemExit(f"{path.name}: no recording date where EyeLink writes it")
        data[found.start(1):found.end(1)] = stamp
        path.write_bytes(bytes(data))
        out[path.name] = EYE_START.isoformat()
    return out


def coherent_tables(target: Path, mri: Path) -> dict:
    anchor = _anchor(mri)
    age = int(str(anchor.PatientAge).rstrip("Y"))
    (target / "participants.csv").write_text(
        "participant_id,sex,age,handedness,group\n"
        f"sub-001,M,{age},right,control\n")
    pet_dose = None
    for _path, ds in dicoms(target / "raw/sub-001/pet"):
        pet_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        break
    mass_ug = pet_dose / FDG_MOLAR_ACTIVITY_BQ_PER_UMOL * FDG_MOLAR_MASS_UG_PER_UMOL
    specific = pet_dose / (mass_ug * 1e-6)                   # Bq/g
    sheet = target / "dose_sheet.csv"
    lines = sheet.read_text().splitlines()
    fixed = []
    for line in lines:
        key = line.split(",", 1)[0]
        if key == "InjectedMass":
            line = f"InjectedMass,{mass_ug:.4f},ug,Mass of the radiolabelled compound that was injected"
        elif key == "SpecificRadioactivity":
            line = (f"SpecificRadioactivity,{specific:.6g},Bq/g,Activity per unit mass of the "
                    "tracer")
        elif key == "ModeOfAdministration":
            line = "ModeOfAdministration,bolus,,How the tracer entered the body"
        fixed.append(line)
    sheet.write_text("\n".join(fixed) + "\n")
    return {"age": age, "injected mass ug": round(mass_ug, 4), "specific Bq/g": f"{specific:.4g}"}


def build_multimodal2(source: Path, out: Path) -> Path:
    v3 = build_v3(source, out)
    target = out / MULTIMODAL2
    copy_sample(v3, target)
    root = target / "raw/sub-001"
    steps = {
        "mrs": mrs_into_the_mri_session(root / "mrs", root / "mri"),
        "pet": pet_to_the_participant(root / "pet", root / "mri"),
        "eeg": eeg_to_the_visit(root / "eeg"),
        "meg": meg_to_the_visit(root / "meg"),
        "eye tracking": eyetracking_during_the_bold(root / "eyetracking"),
        "tables": coherent_tables(target, root / "mri"),
    }
    print(json.dumps(steps, indent=1, default=str))
    readme = Path(__file__).with_name("README_multimodal_2.md")
    shutil.copyfile(readme, target / "README.md")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("variant", choices=("v3", "multimodal2"))
    parser.add_argument("source", type=Path, help="the unpacked version 2 sample")
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    folder = (build_v3 if args.variant == "v3" else build_multimodal2)(args.source, args.out)
    archive = write_zip(folder)
    print(f"{folder}\n{archive}  {archive.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

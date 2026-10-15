"""Keep the source data in the dataset's ``sourcedata/`` (optional).

BIDS reserves ``sourcedata/`` for the data as it was before conversion. With
the setting on, each subject's source files are copied there during the
conversion, for the modalities chosen, either as they are or as one archive
per series::

    sourcedata/sub-<label>/[ses-<label>/]<modality>/<series>/<files>
    sourcedata/sub-<label>/[ses-<label>/]<modality>/<series>.tar.gz
    sourcedata/sub-<label>/manifest.tsv

``<series>`` is the scanner's series number and description for a DICOM
series (``008_bold_run1``), made safe for every OS
(``util/paths.safe_path_component``); a recording keeps its own name and
brings the files that belong to it (BrainVision ``.vmrk``/``.eeg``, EEGLAB
``.fdt``, the parts of a split FIF, a CTF ``.ds`` folder). The manifest lists
every source file, where it went, its size and its SHA-256: provenance, and
how a second conversion knows a file is already there.

The copy is staged beside the subject's conversion and committed with it,
and follows ``--on-existing``: a file already there with the same checksum is
left alone; a different one is kept (``skip``) or backed up and replaced
(``update``, ``replace``). A file another program holds open (Windows) is
reported and left out, never fatal. Archives are fewer files, compress DICOM
well and keep paths short on Windows.

**Identifiers.** A DICOM header carries the participant's name, birth date
and more. A copy in ``sourcedata/`` is not anonymised, so a dataset with it
is not ready to share (:data:`IDENTIFIERS_NOTE`). Qt-free.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..util.cancel import OperationCancelled, is_cancelled
from ..util.paths import long_path, safe_path_component

log = logging.getLogger(__name__)

#: How the source data is kept: not at all, as it is, or one archive per series.
FORMS = ("none", "copy", "archive")
#: The modalities that can be kept, in the order the settings list them.
MODALITIES = ("mri", "pet", "eeg", "meg", "nirs", "physio", "eyetrack")
LABELS = {"mri": "MRI (DICOM)", "pet": "PET (DICOM, ECAT, blood)", "eeg": "EEG",
          "meg": "MEG", "nirs": "NIRS", "physio": "Physiology", "eyetrack": "Eye tracking"}
SOURCEDATA = "sourcedata"
MANIFEST = "manifest.tsv"
MANIFEST_COLUMNS = ("source", "destination", "size", "sha256")
#: What a user must know before turning it on.
IDENTIFIERS_NOTE = (
    "The copies are not anonymised: DICOM headers carry the participant's name, birth "
    "date and other identifiers. A dataset with sourcedata/ is not ready to share as it is.")

_CHUNK = 1 << 20


@dataclass(frozen=True)
class SourcedataOptions:
    """Whether, and how, source data is kept."""

    form: str = "none"
    modalities: frozenset[str] = frozenset(MODALITIES)

    def __post_init__(self) -> None:
        if self.form not in FORMS:
            raise ValueError(f"sourcedata form {self.form!r}: expected one of {FORMS}")
        unknown = set(self.modalities) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown sourcedata modalities: {', '.join(sorted(unknown))}")

    @property
    def enabled(self) -> bool:
        return self.form != "none" and bool(self.modalities)

    @classmethod
    def parse(cls, form: str, modalities: str | Iterable[str] = MODALITIES
              ) -> "SourcedataOptions":
        """From a form and a comma list (CLI, settings)."""
        if isinstance(modalities, str):
            modalities = [m.strip().lower() for m in modalities.split(",") if m.strip()]
        return cls(form=(form or "none").strip().lower(), modalities=frozenset(modalities))


@dataclass
class Item:
    """One series or recording to keep."""

    kind: str                 # the modality folder (MODALITIES)
    destination: str          # relative to sourcedata/sub-<label>/, POSIX
    files: list[tuple[Path, str]] = field(default_factory=list)  # (source, name inside)
    archive: bool = False
    #: The folder the files sit in inside an archive (a series' name; a
    #: recording's files sit at the top).
    folder: str = ""

    @property
    def size(self) -> int:
        return sum(_size(p) for p, _name in self.files)


@dataclass(frozen=True)
class Kept:
    """One source file and where it went (a manifest row)."""

    source: str
    destination: str
    size: int
    sha256: str


@dataclass
class Report:
    """What keeping a subject's source data did."""

    added: int = 0
    unchanged: int = 0
    replaced: int = 0
    kept_existing: int = 0
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.added} added"]
        if self.unchanged:
            parts.append(f"{self.unchanged} already there")
        if self.replaced:
            parts.append(f"{self.replaced} replaced")
        if self.kept_existing:
            parts.append(f"{self.kept_existing} different but kept")
        if self.failed:
            parts.append(f"{len(self.failed)} could not be copied")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# What belongs where
# ---------------------------------------------------------------------------


def kind_for(modality: str, datatype: str, suffix: str) -> str:
    """The modality folder of a recording, from what the inventory says:
    the datatype where it names a modality (a PET series the dcm2niix probe
    classified keeps ``modality`` "mri"), else the modality."""
    if (modality or "").lower() == "eyetrack":
        return "eyetrack"           # its files are physio too, but it is an eye tracker
    if suffix == "physio":
        return "physio"
    by_datatype = {"eeg": "eeg", "ieeg": "eeg", "meg": "meg", "nirs": "nirs", "pet": "pet",
                   "eyetrack": "eyetrack"}
    if datatype in by_datatype:
        return by_datatype[datatype]
    modality = (modality or "").lower()
    if modality == "ieeg":
        return "eeg"
    return modality if modality in MODALITIES else "mri"


def kind_of(task) -> str:
    """The modality folder a task's source data goes to."""
    return kind_for(str(getattr(task, "modality", "") or ""), task.datatype, task.suffix)


def _size(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size
    except OSError:
        return 0


def _vhdr_parts(vhdr: Path) -> list[str]:
    """The data and marker files a BrainVision header names, and the same
    stem's (the header can be wrong after a rename)."""
    names = []
    try:
        for line in vhdr.read_text(encoding="latin-1", errors="replace").splitlines():
            key, _, value = line.partition("=")
            if key.strip().lower() in ("datafile", "markerfile") and value.strip():
                names.append(value.strip())
    except OSError:
        pass
    return names + [vhdr.stem + ".eeg", vhdr.stem + ".vmrk"]


def recording_files(path: Path) -> list[Path]:
    """A recording and the files that belong to it, as they exist (names
    matched without regard to case, as Windows and macOS do). A
    folder-shaped recording (CTF ``.ds``, EGI ``.mff``) is the folder."""
    p = Path(path)
    if p.is_dir():
        return [p]
    name = p.name.lower()
    wanted: list[str] = []
    split = None
    if name.endswith(".vhdr"):
        wanted = _vhdr_parts(p)
    elif name.endswith(".set"):
        wanted = [p.stem + ".fdt"]
    elif name.endswith((".fif", ".fif.gz")):
        tail = ".fif.gz" if name.endswith(".fif.gz") else ".fif"
        split = (p.name[: -len(tail)].lower() + "-", tail)
    try:
        here = sorted(os.listdir(p.parent))
    except OSError:
        here = []
    lower = {n.lower(): n for n in here}
    out = [p] if p.exists() else []
    for n in wanted:
        real = lower.get(n.lower())
        if real is not None and real.lower() != name:
            out.append(p.parent / real)
    if split is not None:
        prefix, tail = split
        for n in here:
            low = n.lower()
            if low.startswith(prefix) and low.endswith(tail) and \
                    low[len(prefix):-len(tail)].isdigit():
                out.append(p.parent / n)
    seen: set[str] = set()
    unique = []
    for f in out:
        if f.name.lower() not in seen:
            seen.add(f.name.lower())
            unique.append(f)
    return unique


def series_name(task) -> str:
    """The folder (or archive) name of a DICOM series: its series number
    and description as the scanner wrote them."""
    number = description = ""
    first = next((Path(f) for f in task.source_files), None)
    if first is not None:
        try:
            import pydicom

            ds = pydicom.dcmread(str(first), stop_before_pixels=True, force=True,
                                 specific_tags=["SeriesNumber", "SeriesDescription",
                                                "ProtocolName"])
            number = str(getattr(ds, "SeriesNumber", "") or "").strip()
            description = str(getattr(ds, "SeriesDescription", "") or
                              getattr(ds, "ProtocolName", "") or "").strip()
        except Exception:  # noqa: BLE001 - a name is all that is lost
            pass
    if number.isdigit():
        number = f"{int(number):03d}"
    name = "_".join(x for x in (number, description) if x) or task.basename
    return safe_path_component(name)


def _unique(name: str, taken: set[str]) -> str:
    stem, dot, ext = name.partition(".")
    out, n = name, 2
    while out.lower() in taken:
        out = f"{stem}_{n}{dot}{ext}"
        n += 1
    taken.add(out.lower())
    return out


def plan(tasks, options: SourcedataOptions) -> list[Item]:
    """What a subject's tasks keep, and where (nothing when it is off)."""
    if not options.enabled:
        return []
    items: list[Item] = []
    archive = options.form == "archive"
    used: dict[str, set[str]] = {}
    for task in tasks:
        kind = kind_of(task)
        if kind not in options.modalities:
            continue
        base = "/".join(x for x in (
            f"ses-{safe_path_component(task.session)}" if task.session else "", kind) if x)
        taken = used.setdefault(base, set())
        if task.series_uid:
            series = _unique(series_name(task), taken)
            files: list[tuple[Path, str]] = []
            names: set[str] = set()
            for f in task.source_files:
                f = Path(f)
                files.append((f, _unique(safe_path_component(f.name), names)))
            dest = f"{base}/{series}" + (".tar.gz" if archive else "")
            folder = series
        else:
            folder = ""
            src = Path(task.source_files[0])
            files = [(f, f.name) for f in recording_files(src)]
            if not files:
                continue
            if archive:
                dest = f"{base}/{_unique(safe_path_component(src.name), taken)}.tar.gz"
            else:
                dest = base
                files = [(f, _unique(n, taken)) for f, n in files]
        if kind == "pet":
            # A PET run's blood data (PMOD exports) is source data too.
            for suffix, path in getattr(task, "companion_files", ()) or ():
                if suffix == "blood" and Path(path).is_file():
                    files.append((Path(path), _unique(Path(path).name, taken)))
        items.append(Item(kind=kind, destination=dest, files=files, archive=archive,
                          folder=folder))
    return items


def estimate(items: Iterable[Item]) -> tuple[int, int]:
    """``(files, bytes)`` the items add."""
    n = size = 0
    for item in items:
        for path, _name in item.files:
            if path.is_dir():
                n += sum(1 for f in path.rglob("*") if f.is_file())
            else:
                n += 1
            size += _size(path)
    return n, size


def space_problem(root: Path, needed: int) -> str:
    """Why ``needed`` bytes will not fit under ``root`` ("" when they will).
    Archives usually take less; the check assumes they do not."""
    probe = Path(root)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return ""
    margin = 256 * 1024 * 1024
    if needed + margin <= free:
        return ""
    return (f"Keeping the source data needs {human_size(needed)} and {probe} has "
            f"{human_size(free)} free.")


def human_size(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


# ---------------------------------------------------------------------------
# Copying into the staging area
# ---------------------------------------------------------------------------


class _Hashing:
    """A file read through, hashed as it goes (for the archive)."""

    def __init__(self, fh) -> None:
        self._fh = fh
        self.sha = hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        data = self._fh.read(n)
        self.sha.update(data)
        return data


def sha256_of(path: Path) -> str:
    sha = hashlib.sha256()
    with open(long_path(path), "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            sha.update(block)
    return sha.hexdigest()


def _copy_hashing(src: Path, dst: Path) -> str:
    """Copy ``src`` to ``dst`` (times kept) and return its SHA-256."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    with open(long_path(src), "rb") as fin, open(long_path(dst), "wb") as fout:
        for block in iter(lambda: fin.read(_CHUNK), b""):
            sha.update(block)
            fout.write(block)
    shutil.copystat(long_path(src), long_path(dst))
    return sha.hexdigest()


def _expand(files: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """A folder-shaped recording as its files, each with its path inside."""
    out = []
    for path, name in files:
        if path.is_dir():
            for f in sorted(p for p in path.rglob("*") if p.is_file()):
                out.append((f, f"{name}/{f.relative_to(path).as_posix()}"))
        else:
            out.append((path, name))
    return out


def stage(items: list[Item], staging: Path, *, cancel_check: Optional[Callable] = None,
          progress: Optional[Callable[[int, int], None]] = None) -> tuple[list[Kept], list[str]]:
    """Copy or archive ``items`` into ``staging`` (the subject's folder in
    the staging area). Returns the manifest rows and the files that could
    not be read."""
    rows: list[Kept] = []
    failed: list[str] = []
    total = sum(len(_expand(i.files)) for i in items)
    done = 0
    for item in items:
        files = _expand(item.files)
        if item.archive:
            target = staging / item.destination
            target.parent.mkdir(parents=True, exist_ok=True)
            inside = f"{item.folder}/" if item.folder else ""
            with tarfile.open(long_path(target), "w:gz") as tar:
                for src, name in files:
                    if is_cancelled(cancel_check):
                        raise OperationCancelled()
                    try:
                        info = tar.gettarinfo(long_path(src), arcname=f"{inside}{name}")
                        with open(long_path(src), "rb") as fh:
                            reader = _Hashing(fh)
                            tar.addfile(info, reader)
                        rows.append(Kept(str(src), f"{item.destination}!{inside}{name}",
                                         info.size, reader.sha.hexdigest()))
                    except OSError as exc:
                        failed.append(f"{src}: {exc.strerror or exc}")
                    done += 1
                    if progress:
                        progress(done, total)
            continue
        for src, name in files:
            if is_cancelled(cancel_check):
                raise OperationCancelled()
            rel = f"{item.destination}/{name}"
            try:
                sha = _copy_hashing(src, staging / rel)
                rows.append(Kept(str(src), rel, _size(src), sha))
            except OSError as exc:
                failed.append(f"{src}: {exc.strerror or exc}")
            done += 1
            if progress:
                progress(done, total)
    return rows, failed


# ---------------------------------------------------------------------------
# Committing beside the subject
# ---------------------------------------------------------------------------


def read_manifest(path: Path) -> dict[str, Kept]:
    """The manifest's rows by destination (``{}`` when there is none)."""
    if not path.is_file():
        return {}
    out = {}
    with open(long_path(path), encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                out[row["destination"]] = Kept(row["source"], row["destination"],
                                               int(row["size"] or 0), row["sha256"])
            except (KeyError, ValueError):
                continue
    return out


def write_manifest(path: Path, rows: Iterable[Kept]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(long_path(path), "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(MANIFEST_COLUMNS)
        for r in sorted(rows, key=lambda r: r.destination):
            writer.writerow((r.source, r.destination, r.size, r.sha256))


def commit(staging: Path, target: Path, rows: list[Kept], *, on_existing: str,
           backup_root: Path, stamp: str) -> Report:
    """Move the staged copy into ``target`` (``sourcedata/sub-<label>``),
    merging with what is there and its manifest."""
    report = Report()
    old = read_manifest(target / MANIFEST)
    merged = dict(old)
    # A staged FILE is a manifest destination, or an archive holding several.
    by_file: dict[str, list[Kept]] = {}
    for r in rows:
        by_file.setdefault(r.destination.split("!", 1)[0], []).append(r)
    for rel, members in sorted(by_file.items()):
        src = staging / rel
        dst = target / rel
        if not src.is_file():
            continue
        if dst.exists():
            same = all(old.get(m.destination) and old[m.destination].sha256 == m.sha256
                       for m in members)
            if not same and len(members) == 1 and "!" not in members[0].destination:
                try:
                    same = sha256_of(dst) == members[0].sha256
                except OSError:
                    same = False
            if same:
                report.unchanged += len(members)
                for m in members:
                    merged[m.destination] = m
                continue
            if on_existing not in ("update", "replace"):
                report.kept_existing += len(members)
                continue
            backup = backup_root / f"sourcedata_{target.name}_{stamp}" / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(long_path(dst), long_path(backup))
            report.replaced += len(members)
        else:
            report.added += len(members)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(long_path(src), long_path(dst))
        for m in members:
            merged[m.destination] = m
    if merged != old:
        write_manifest(target / MANIFEST, merged.values())
    return report


def keep_subject(tasks, options: SourcedataOptions, *, staging: Path, target: Path,
                 on_existing: str, backup_root: Path, stamp: str,
                 cancel_check: Optional[Callable] = None) -> Optional[Report]:
    """Stage and commit one subject's source data (``None`` when off or
    there is nothing to keep). Called after the subject itself committed,
    from the conversion's per-subject block."""
    items = plan(tasks, options)
    if not items:
        return None
    rows, failed = stage(items, staging, cancel_check=cancel_check)
    report = commit(staging, target, rows, on_existing=on_existing,
                    backup_root=backup_root, stamp=stamp)
    report.failed = failed
    return report


__all__ = ["FORMS", "IDENTIFIERS_NOTE", "LABELS", "MANIFEST", "MODALITIES", "SOURCEDATA",
           "Item", "Kept", "Report", "SourcedataOptions", "commit", "estimate", "human_size",
           "keep_subject", "kind_for", "kind_of", "plan", "read_manifest", "recording_files",
           "series_name", "sha256_of", "space_problem", "stage", "write_manifest"]

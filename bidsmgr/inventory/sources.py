"""Where a scan's source data is, and whether it is still there.

A scan records where it read from twice: the folder that was scanned (the
version's ``raw_root``), and the absolute path of every DICOM it read
(``files_by_uid``, beside the inventory). The rows themselves hold paths
RELATIVE to the scanned folder: ``source_folder`` for a series, and
``source_file`` for a recording or an ECAT file.

Data gets moved: to another disk, a server, a renamed folder. The absolute
paths then go stale and the scanned folder is gone. Saying where the folder
went (a relink) records the new location, and every path is re-based onto
it, for the preview and for the conversion alike (:class:`ScanSources`).

The record can also be wrong. Until 2026-10-09, picking a folder to scan
next while a scan table was open wrote that folder into the OPEN scan's
record, so an older scan named a newer scan's folder. The absolute DICOM
paths still say where its data really was, so they are believed first
(:func:`scanned_root`, :func:`resolve`).

When the scanned folder holds the files directly, the scanner writes the
folder's own NAME as ``source_folder`` (``mri_dicom``, ``pet_ecat``), so
``<root>/<source_folder>`` is not a folder: :meth:`ScanSources.folder`
knows the convention.

Qt-free; nothing here writes anything.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Optional

#: Rows sampled to judge whether a scan's data is still there. Enough to
#: notice a moved folder, few enough to stay instant on a network drive.
SAMPLE_ROWS = 24


def files_by_uid_path(inventory: Path) -> Path:
    """The ``files_by_uid`` sidecar the scan wrote beside ``inventory``."""
    inventory = Path(inventory)
    return inventory.with_suffix(inventory.suffix + ".files_by_uid.json.gz")


def load_files_by_uid(inventory: Path) -> dict[str, list[str]]:
    """The scan's absolute DICOM paths by series, ``{}`` when it wrote none
    (a scan of recordings only) or the sidecar cannot be read."""
    path = files_by_uid_path(inventory)
    if not path.is_file():
        return {}
    try:
        with gzip.open(path, "rb") as fh:
            data = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): [str(p) for p in v] for k, v in data.items()} if isinstance(data, dict) \
        else {}


def _posix(rel: str) -> str:
    # A TSV written on Windows by an older version can carry backslashes.
    return (rel or "").strip().replace("\\", "/").strip("/")


def _text(value) -> str:
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return ""
    return str(value).strip()


def _rows(df) -> Iterable[tuple[str, str, str]]:
    """``(series_uid, source_folder, source_file)`` of every row."""
    cols = getattr(df, "columns", ())
    columns = [df[c].tolist() if c in cols else [""] * len(df)
               for c in ("series_uid", "source_folder", "source_file")]
    for uid, folder, source_file in zip(*columns):
        yield _text(uid), _text(folder), _text(source_file)


def scanned_root(df, files_by_uid: Mapping[str, list[str]], *,
                 recorded: Optional[str | Path] = None,
                 label: Optional[str] = None) -> Optional[Path]:
    """The folder the scan read, worked out from where its DICOMs were.

    Each series' first file sits in ``<root>/<source_folder>``, so the root
    is that file's folder with ``source_folder`` taken off the end; when the
    files sat in the root itself, ``source_folder`` is the root's own name
    and the folder IS the root. Rows vote; a tie (every series in one folder
    of that name) goes to the ``recorded`` root, then to the one named
    ``label`` (the version's source label), then to the shorter path. With
    no DICOM to go by, the ``recorded`` root is all there is.
    """
    votes: Counter[Path] = Counter()
    seen = 0
    for uid, folder, _file in _rows(df):
        if not uid or seen >= 400:
            continue
        files = files_by_uid.get(uid.split("|")[0]) or []
        if not files:
            continue
        seen += 1
        parent = Path(files[0]).parent
        parts = PurePosixPath(_posix(folder)).parts if _posix(folder) else ()
        if not parts:
            votes[parent] += 1
            continue
        n = len(parts)
        if len(parent.parts) > n and tuple(parent.parts[-n:]) == tuple(parts):
            votes[Path(*parent.parts[:-n])] += 1
        if n == 1 and parent.name == parts[0]:
            votes[parent] += 1
    rec = Path(recorded) if recorded else None
    if not votes:
        return rec
    top = max(votes.values())
    best = [root for root, n in votes.items() if n == top]
    if rec is not None and rec in best:
        return rec
    named = [root for root in best if label and root.name == label]
    if named:
        return named[0]
    return min(best, key=lambda root: (len(root.parts), str(root)))


@dataclass(frozen=True)
class ScanSources:
    """Where one scan's data is now, and how to find each row's files."""

    #: Where the data is now (the scanned folder, or where it was relinked).
    root: Optional[Path]
    #: Where it was when scanned; the DICOM paths are under it.
    scanned: Optional[Path] = None
    files_by_uid: Mapping[str, list[str]] = field(default_factory=dict)

    def rebase(self, path: str | Path) -> Path:
        """``path`` as it would be under :attr:`root` (unchanged when it is
        not under the scanned folder, or nothing moved)."""
        p = Path(path)
        if self.root is None or self.scanned is None or self.root == self.scanned:
            return p
        try:
            return self.root / p.relative_to(self.scanned)
        except ValueError:
            return p

    def series_files(self, series_uid: str) -> list[Path]:
        """Every DICOM of the series (``a|b`` for a fieldmap's two), where
        they are now. One existence check per series decides whether its
        paths need re-basing."""
        out: list[Path] = []
        for uid in (u for u in (series_uid or "").split("|") if u):
            files = [Path(f) for f in self.files_by_uid.get(uid, [])]
            if files and not files[0].exists():
                files = [self.rebase(f) for f in files]
            out.extend(files)
        return out

    def folder(self, source_folder: str) -> Optional[Path]:
        """The folder a row's files are in, when it exists."""
        if self.root is None:
            return None
        rel = _posix(source_folder)
        if rel and Path(rel).is_absolute():
            candidate = self.rebase(rel)
            return candidate if candidate.is_dir() else None
        candidate = self.root / rel if rel else self.root
        if candidate.is_dir():
            return candidate
        # The files were in the scanned folder itself, and the scanner wrote
        # that folder's name.
        names = {self.root.name} | ({self.scanned.name} if self.scanned else set())
        if rel in names and self.root.is_dir():
            return self.root
        return None

    def file(self, source_file: str) -> Optional[Path]:
        """A row's recording (or ECAT file), when it exists."""
        rel = _posix(source_file)
        if not rel:
            return None
        p = Path(rel)
        if p.is_absolute():
            for candidate in (p, self.rebase(p)):
                if candidate.exists():
                    return candidate
            return None
        if self.root is None:
            return None
        candidate = self.root / rel
        return candidate if candidate.exists() else None

    def relocated_files_by_uid(self) -> dict[str, list[str]]:
        """``files_by_uid`` with every path where it is now, for the
        conversion (which reads the absolute paths)."""
        if self.root is None or self.scanned is None or self.root == self.scanned:
            return dict(self.files_by_uid)
        out: dict[str, list[str]] = {}
        for uid, files in self.files_by_uid.items():
            if files and not Path(files[0]).exists():
                out[uid] = [str(self.rebase(f)) for f in files]
            else:
                out[uid] = list(files)
        return out


@dataclass(frozen=True)
class SourceCheck:
    """What a look at a sample of a scan's rows found."""

    #: The rows looked at (those with a file to look for) and how many of
    #: them were there.
    checked: int
    found: int
    #: The folder the scan RECORDED, when the files say otherwise (the
    #: record was wrong and is to be corrected).
    corrected_from: Optional[Path] = None

    @property
    def state(self) -> str:
        """``ok``, ``partial`` (some missing), ``missing`` (none there) or
        ``unknown`` (nothing to look for)."""
        if self.checked == 0:
            return "unknown"
        if self.found == self.checked:
            return "ok"
        return "missing" if self.found == 0 else "partial"


def _same_series(path: Path, uid: str) -> bool:
    """Whether the DICOM at ``path`` belongs to ``uid`` (one tag read)."""
    try:
        import pydicom

        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True,
                             specific_tags=["SeriesInstanceUID"])
    except Exception:  # noqa: BLE001 - unreadable is "not this series"
        return False
    return str(getattr(ds, "SeriesInstanceUID", "")) == uid


def check(sources: ScanSources, df, *, limit: int = SAMPLE_ROWS,
          verify: bool = False) -> SourceCheck:
    """Look for a sample of the rows' files (one per row, spread over the
    table). ``verify`` also reads each sampled DICOM's series UID, for a
    folder the user picked: a folder with the same layout and file names
    from another session must not pass."""
    rows = [r for r in _rows(df) if r[0] or r[2]]
    if not rows:
        return SourceCheck(0, 0)
    step = max(1, len(rows) // max(1, limit))
    found = checked = 0
    for uid, _folder, source_file in rows[::step][:limit]:
        if uid:
            first = uid.split("|")[0]
            files = sources.files_by_uid.get(first) or []
            if not files:
                continue
            checked += 1
            path = Path(files[0])
            if not path.exists():
                path = sources.rebase(path)
            if path.exists() and (not verify or _same_series(path, first)):
                found += 1
        else:
            checked += 1
            if sources.file(source_file) is not None:
                found += 1
    return SourceCheck(checked, found)


def resolve(df, files_by_uid: Mapping[str, list[str]], *,
            recorded: Optional[str | Path], label: Optional[str] = None,
            limit: int = SAMPLE_ROWS) -> tuple[ScanSources, SourceCheck]:
    """Where a scan's data is, believing its files over its record.

    Tries the folder the DICOM paths point at, then the recorded one (a
    relink); the first that holds the sampled files wins. When neither does,
    the data has moved: the folder reported is the recorded one if it is
    gone (that is where it was last seen), and the scanned one if the
    recorded folder exists but holds none of this scan (a wrong record).
    """
    rec = Path(recorded) if recorded else None
    scanned = scanned_root(df, files_by_uid, recorded=rec, label=label)
    tried: list[tuple[ScanSources, SourceCheck]] = []
    for root in dict.fromkeys(r for r in (scanned, rec) if r is not None):
        sources = ScanSources(root, scanned, files_by_uid)
        result = check(sources, df, limit=limit)
        if result.state in ("ok", "unknown"):
            wrong = rec if (rec is not None and root != rec and result.state == "ok") else None
            return sources, SourceCheck(result.checked, result.found, wrong)
        tried.append((sources, result))
    best = max(tried, key=lambda t: t[1].found, default=None)
    if best is not None and best[1].found:
        return best
    if rec is not None and not rec.exists():
        root = rec
    else:
        root = scanned or rec
    sources = ScanSources(root, scanned, files_by_uid)
    return sources, (best[1] if best is not None else SourceCheck(0, 0))


def locate(df, files_by_uid: Mapping[str, list[str]], folder: Path, *,
           scanned: Optional[Path], limit: int = SAMPLE_ROWS) -> tuple[ScanSources, SourceCheck]:
    """Whether ``folder`` (picked by the user) holds this scan's data,
    reading the series UID of each sampled DICOM to be sure."""
    sources = ScanSources(Path(folder), scanned or Path(folder), files_by_uid)
    return sources, check(sources, df, limit=limit, verify=True)


def describe(check_: SourceCheck, root: Optional[Path]) -> str:
    """One sentence for the user about where the data is."""
    where = str(root) if root is not None else "the folder that was scanned"
    if check_.state == "missing":
        return (f"The source data is no longer at {where}: it was moved, renamed or is on "
                "a drive that is not connected. The preview and the conversion need it.")
    if check_.state == "partial":
        return (f"Only {check_.found} of {check_.checked} recordings checked are still at "
                f"{where}; part of the source data was moved or deleted.")
    return f"The source data is at {where}."


__all__ = [
    "SAMPLE_ROWS", "ScanSources", "SourceCheck", "check", "describe", "files_by_uid_path",
    "load_files_by_uid", "locate", "resolve", "scanned_root",
]

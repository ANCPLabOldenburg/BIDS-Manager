"""Write the sidecars a dataset's recordings are missing, from what is left.

A sidecar deleted by accident, or a dataset that arrives with none, leaves the
recordings themselves. This rebuilds each sidecar from every source that is
still there, in the order of how much it knows, and never overwrites a value
a sidecar already holds:

1. **The source data**, when the dataset is a BIDS Manager project that still
   knows where it was converted from. The recording's row is converted again
   into a temporary folder by the conversion's own code, so the sidecar is the
   one the conversion wrote: every field dcm2niix, bidsphysio, the ECAT reader
   or the eye-tracking converter read out of the original files.
2. **The recording's own header** (:mod:`bidsmgr.editor.headers`): the NIfTI
   timing fields, the NIfTI-MRS header extension, and for EEG and MEG what
   MNE-BIDS reads from the recording, which is everything it wrote.
3. **What the user answered** in the project (metadata templates, dataset and
   per-recording values, cells typed in the scan table), applied by the
   metadata step's own function, so an answer wins over the header exactly as
   it does in a conversion.
4. **The links** other sidecars hold to this one: a fieldmap's
   ``B0FieldIdentifier`` and ``IntendedFor`` from the other images of the same
   fieldmap, a run's ``B0FieldSource`` from the fieldmaps intended for it, and
   a phase difference's two echo times from its magnitude images.
5. **The file name**: ``TaskName``, and the ``TODO`` placeholders the chosen
   scope asks for, by the metadata step's own function.

Identifiers are removed as the conversion removes them, and a dataset that was
anonymised has the new fields anonymised the same way (the same per-participant
date shift and pseudonyms), so restoring a sidecar cannot put a face, a name or
a date back.

A field that a sidecar higher up already states (BIDS inheritance) is left to
it, and shown, so a dataset that keeps ``RepetitionTime`` at the top does not
gain a copy in every file. Where the source data or the user's answers say
otherwise, the value is written into the recording's own sidecar, where it
wins for that recording alone; a header that disagrees is only shown, because
a header can be approximate. A ``TODO`` is not an answer: a real value replaces
it, a placeholder does not.

Writing is one step of the Editor's history.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from .headers import (
    READABLE_EXTENSIONS,
    ROUNDED_TE,
    datatypes_for,
    default_suffix,
    echo_time,
    extension,
    guess_kind,
    read_header,
)
from .inheritance import applies_to, entities_of, suffix_of

log = logging.getLogger(__name__)

# -- where a field came from ---------------------------------------------------

SOURCE = "source"
HEADER = "header"
ANSWERS = "answers"
LINKS = "links"
FILENAME = "filename"
ANONYMISED = "anonymised"
PLACEHOLDER = "placeholder"

#: Every origin in the order of how much it knows, with its name for a person.
ORIGINS: dict[str, str] = {
    SOURCE: "Source data",
    HEADER: "File header",
    ANSWERS: "Your answers",
    LINKS: "Other sidecars",
    FILENAME: "File name",
    ANONYMISED: "Anonymisation",
    PLACEHOLDER: "Placeholder",
}

#: What each origin means, for a tooltip.
ORIGIN_HELP: dict[str, str] = {
    SOURCE: "Read from the data this recording was converted from, by converting "
            "it again with the conversion's own code.",
    HEADER: "Read from the recording file itself.",
    ANSWERS: "What you answered in this project: a metadata template, a dataset "
             "or per-recording value, or a cell of the scan table.",
    LINKS: "Taken from the other sidecars that link to this one, such as the "
           "fieldmap intended for this run.",
    FILENAME: "Taken from the file name, as the metadata step does.",
    ANONYMISED: "Written because this dataset was anonymised.",
    PLACEHOLDER: "Not known: a TODO for you to fill, as the metadata step marks it.",
}

#: Fields a fieldmap shares with the other images of the same fieldmap.
_FIELDMAP_SHARED = ("B0FieldIdentifier", "IntendedFor")

#: The images a fieldmap is made of.
_FIELDMAP_SUFFIXES = frozenset({
    "magnitude", "magnitude1", "magnitude2", "phase1", "phase2", "phasediff",
    "fieldmap", "epi", "m0scan", "TB1DAM", "TB1EPI", "TB1AFI", "TB1TFL", "TB1RFM",
    "TB1SRGE", "TB1map", "RB1COR", "RB1map",
})

#: Entities that tell the images of one fieldmap apart.
_WITHIN_FIELDMAP = ("dir", "part", "echo", "flip", "inv", "mt")

_SPLIT = re.compile(r"_split-[0-9]+")
_TODO = "TODO"


def _is_todo(value) -> bool:
    return value == _TODO or value == [_TODO]


# ---------------------------------------------------------------------------
# What a plan is
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewField:
    """One field the sidecar gains."""

    name: str
    value: object
    origin: str
    #: The placeholder this value replaces, when the sidecar held ``TODO``.
    replaces_todo: bool = False
    #: The sidecar higher up that states another value, which this one
    #: overrides for this recording (the nearest sidecar wins), and its value.
    overrides: str = ""
    overridden: object = None


@dataclass(frozen=True)
class Inherited:
    """A field left to a sidecar higher up, which already states it."""

    name: str
    value: object
    #: That sidecar, dataset-relative.
    source: str
    #: What this recording's own data says, when it says something else.
    found: object = None
    differs: bool = False


@dataclass(frozen=True)
class Rejected:
    """A value found and not written: the selected BIDS version rejects it."""

    name: str
    value: object
    origin: str
    problem: str


@dataclass
class SidecarPlan:
    """What one recording's sidecar would become."""

    recording: str              # dataset-relative POSIX path
    sidecar: str                # dataset-relative POSIX path of the JSON
    datatype: str
    suffix: str
    #: Whether the sidecar is there already (it is then only added to).
    exists: bool
    added: list[NewField] = field(default_factory=list)
    inherited: list[Inherited] = field(default_factory=list)
    #: How many fields the sidecar already had.
    kept: int = 0
    #: What was read, one line each ("the source DICOM series", ...).
    read: list[str] = field(default_factory=list)
    #: Why a source gave less than it might have.
    notes: list[str] = field(default_factory=list)
    #: Required fields no source could give, and no placeholder marks.
    still_missing: list[str] = field(default_factory=list)
    #: The whole sidecar as it would be written.
    document: dict = field(default_factory=dict)
    #: Values found and not written, because the BIDS version rejects them.
    rejected: list[Rejected] = field(default_factory=list)
    #: What the file is recorded as, and how that was decided.
    kind: Optional["Kind"] = None

    @property
    def changes(self) -> bool:
        return bool(self.added)

    def by_origin(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.added:
            out[f.origin] = out.get(f.origin, 0) + 1
        return out


@dataclass
class SidecarsPlan:
    """What writing sidecars would do to a dataset."""

    root: Path
    files: list[SidecarPlan] = field(default_factory=list)
    #: ``(recording, why)`` for recordings left alone.
    skipped: list[tuple[str, str]] = field(default_factory=list)
    #: The anonymisation applied to new fields ("" when the dataset is not).
    anonymised: str = ""
    #: Whether the data the dataset was converted from was looked for (only
    #: in a project that converted it).
    used_source: bool = False
    #: The BIDS version the fields and values follow.
    bids_version: str = ""

    def writes(self) -> list[SidecarPlan]:
        return [f for f in self.files if f.changes]


# ---------------------------------------------------------------------------
# Which files are recordings, and where their sidecar is
# ---------------------------------------------------------------------------


def _session_root(root: Path, path: Path) -> Optional[Path]:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    if not parts or not parts[0].startswith("sub-"):
        return None
    if len(parts) > 2 and parts[1].startswith("ses-"):
        return root / parts[0] / parts[1]
    return root / parts[0]


def is_recording(root: Path, path: Path) -> bool:
    """Whether ``path`` is a recording a sidecar describes.

    What the scans table lists (by the schema's rule, so every datatype the
    standard defines), plus the physiological and eye-tracking tables, whose
    sidecars are required, and less the events tables, whose sidecar is a
    description of columns no header knows.
    """
    from ..metadata.tables import is_scans_entry

    session = _session_root(Path(root), Path(path))
    if session is None:
        return False
    rel = Path(path).relative_to(session).as_posix()
    if rel.count("/") != 1:
        return False
    name = Path(path).name
    if name.endswith(("_physio.tsv.gz", "_physioevents.tsv.gz")):
        return True
    if suffix_of(name) == "events":
        return False
    return is_scans_entry(rel, is_dir=Path(path).is_dir())


def sidecar_path(recording: Path) -> Path:
    """The sidecar of ``recording``: its name with ``.json``, and without the
    ``split`` entity (the parts of one split recording share one sidecar)."""
    recording = Path(recording)
    ext = extension(recording.name)
    stem = recording.name[: -len(ext)] if ext else recording.name
    return recording.with_name(_SPLIT.sub("", stem) + ".json")


def is_readable(path: Path) -> bool:
    """A file whose sidecar can be written from what is in it, wherever it is
    and whatever it is called: an image, a recording MNE reads, a BIDS
    physiological or eye-tracking table."""
    path = Path(path)
    name = path.name
    if extension(name) in READABLE_EXTENSIONS:
        return path.exists()
    return name.endswith(("_physio.tsv.gz", "_physioevents.tsv.gz")) and path.is_file()


def _hidden(root: Path, path: Path) -> bool:
    try:
        parts = Path(path).relative_to(root).parts
    except ValueError:
        return True
    return not parts or any(part.startswith(".") for part in parts)


def is_candidate(root: Path, path: Path) -> bool:
    """A recording of the dataset (its BIDS name and place), or any file
    BIDS Manager can read a sidecar out of, inside the dataset and not in a
    hidden folder."""
    root, path = Path(root), Path(path)
    if _hidden(root, path):
        return False
    return is_recording(root, path) or is_readable(path)


def recording_for(sidecar: Path, root: Optional[Path] = None) -> Optional[Path]:
    """The recording a sidecar belongs to (the first, for a split one). A
    ``.bval`` shares the name and is not it."""
    sidecar = Path(sidecar)
    try:
        siblings = sorted(sidecar.parent.iterdir())
    except OSError:
        return None
    for path in siblings:
        if path == sidecar or path.name.endswith(".json") or sidecar_path(path) != sidecar:
            continue
        if is_readable(path) or (root is not None and is_recording(root, path)):
            return path
    return None


#: Folders at the top of a dataset that hold no recordings of its own. Walked
#: when picked (a derivative's sidecar can be written too), never as part of
#: "the whole dataset".
NOT_THE_DATASETS = frozenset({"derivatives", "sourcedata", "code", "stimuli", "log"})


def recordings(root: Path, targets: Optional[Iterable[Path]] = None) -> list[Path]:
    """Every file a sidecar can be written for under ``targets`` (files or
    folders; the whole dataset when none), in path order. BIDS names and
    places are not required: a recording called ``t1_mprage.nii`` in a folder
    of its own gets its sidecar beside it. A sidecar picked stands for its
    recording. The whole dataset leaves out what ``.bidsignore`` hides and
    the folders that are not its own recordings (``derivatives/``,
    ``sourcedata/``, ...); a folder or file picked is taken as it is."""
    from ..vendor.bidsval.files.bidsignore import load_bidsignore

    root = Path(root)
    picked = [Path(t) for t in targets] if targets else []
    whole = not picked or root in picked
    found: set[Path] = set()
    for start in [root] if whole else picked:
        if start.is_file() and start.name.endswith(".json"):
            rec = recording_for(start, root)
            if rec is not None and is_candidate(root, rec):
                found.add(rec)
            continue
        if start != root and is_candidate(root, start):
            found.add(start)
            continue
        if not start.is_dir() or (start != root and _hidden(root, start)):
            continue
        for dirpath, dirnames, filenames in os.walk(start):
            here = Path(dirpath)
            dirnames[:] = sorted(
                n for n in dirnames
                if not n.startswith(".")
                and not (whole and here == root and n in NOT_THE_DATASETS))
            for name in list(dirnames):
                # A recording that is a folder (CTF's .ds): not walked into.
                if is_candidate(root, here / name):
                    found.add(here / name)
                    dirnames.remove(name)
            for name in filenames:
                if is_candidate(root, here / name):
                    found.add(here / name)
    if not whole:
        return sorted(found)
    ignore = load_bidsignore(root)
    return sorted(p for p in found if not ignore.match(p.relative_to(root).as_posix()))


def _datatype(root: Path, path: Path) -> str:
    session = _session_root(root, path)
    return path.relative_to(session).parts[0] if session else ""


# ---------------------------------------------------------------------------
# What kind of recording a file is
# ---------------------------------------------------------------------------

#: How a file's kind was decided, in the order of how sure it is.
KIND_BIDS = "bids"        # its BIDS name and place
KIND_NAME = "name"        # a BIDS suffix in its name
KIND_FOLDER = "folder"    # the datatype folder it is in
KIND_GUESS = "guess"      # read from the file
KIND_CHOSEN = "chosen"    # the user said so


@dataclass(frozen=True)
class Kind:
    """What a file is recorded as: its datatype and suffix, which decide the
    fields the standard declares for it, and how that was decided."""

    datatype: str
    suffix: str
    how: str
    why: str = ""

    @property
    def bids(self) -> bool:
        return self.how == KIND_BIDS


def kind_choices(path: Path) -> list[tuple[str, list[str]]]:
    """The datatypes a file of this format may be recorded as, each with the
    suffixes the selected BIDS version gives it (less the tables and masks,
    which are not recordings)."""
    from .. import schema as schema_mod

    known = set(schema_mod.list_datatypes())
    out = []
    for datatype in datatypes_for(path):
        if datatype not in known:
            continue
        suffixes = [s for s in schema_mod.list_suffixes(datatype) if s not in _NOT_RECORDINGS]
        if suffixes:
            out.append((datatype, suffixes))
    return out


#: Suffixes that never name a recording read out of a file.
_NOT_RECORDINGS = frozenset({
    "physio", "physioevents", "stim", "events", "channels", "electrodes", "coordsystem",
    "photo", "headshape", "markers", "defacemask", "scans", "sessions",
})


def kind_of(root: Path, path: Path, chosen: Optional[tuple[str, str]] = None) -> Kind:
    """What ``path`` is recorded as. Its BIDS name and place when it has
    them; else ``chosen``; else a suffix in its name, the datatype folder it
    is in, or what the file itself shows."""
    root, path = Path(root), Path(path)
    if is_recording(root, path):
        return Kind(_datatype(root, path), suffix_of(path.name), KIND_BIDS,
                    "its BIDS name and place")
    if chosen:
        return Kind(chosen[0], chosen[1], KIND_CHOSEN, "as you chose")
    choices = kind_choices(path)
    if path.name.endswith(("_physio.tsv.gz", "_physioevents.tsv.gz")):
        suffix = "physioevents" if path.name.endswith("_physioevents.tsv.gz") else "physio"
        folder = path.parent.name
        return Kind(folder if folder in ("func", "beh", "anat", "dwi", "perf", "pet", "eeg",
                                         "meg", "nirs", "motion") else "beh",
                    suffix, KIND_NAME, f"the _{suffix} in its name")
    stem = path.name[: -len(extension(path.name))] if extension(path.name) else path.name
    token = stem.rsplit("_", 1)[-1]
    holding = [d for d, suffixes in choices if token in suffixes]
    if holding:
        datatype = path.parent.name if path.parent.name in holding else holding[0]
        return Kind(datatype, token, KIND_NAME, f"the _{token} in its name")
    folder = path.parent.name
    if folder in dict(choices):
        guessed = guess_kind(path)
        suffix = guessed[1] if guessed[0] == folder else default_suffix(folder)
        if suffix in dict(choices)[folder]:
            return Kind(folder, suffix, KIND_FOLDER,
                        f"the {folder}/ folder it is in; {suffix} assumed")
    datatype, suffix, why = guess_kind(path)
    return Kind(datatype, suffix, KIND_GUESS, why)


# ---------------------------------------------------------------------------
# The sidecars that already describe a recording
# ---------------------------------------------------------------------------


class _Listing:
    """The JSON files of each folder, read once per plan."""

    def __init__(self) -> None:
        self._names: dict[Path, list[Path]] = {}
        self._data: dict[Path, Optional[dict]] = {}

    def jsons(self, folder: Path) -> list[Path]:
        if folder not in self._names:
            try:
                self._names[folder] = sorted(
                    p for p in folder.iterdir() if p.name.endswith(".json") and p.is_file())
            except OSError:
                self._names[folder] = []
        return self._names[folder]

    def load(self, path: Path) -> Optional[dict]:
        if path not in self._data:
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                data = None
            self._data[path] = data if isinstance(data, dict) else None
        return self._data[path]


def _describing(root: Path, recording: Path, listing: _Listing) -> list[Path]:
    """Every sidecar that applies to ``recording``, nearest first."""
    out: list[Path] = []
    folder = recording.parent
    while True:
        out.extend(j for j in listing.jsons(folder) if applies_to(j, recording))
        if folder == root or root not in folder.parents:
            break
        folder = folder.parent
    return out


# ---------------------------------------------------------------------------
# The project: what each file was converted from, and what was answered
# ---------------------------------------------------------------------------


class _Project:
    """What a BIDS Manager project knows about its files. Read lazily: a
    dataset that is not a project, or a plan that does not need it, costs
    nothing."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._tasks: Optional[dict[str, dict]] = None
        self._versions: Optional[list] = None
        self._loaded: dict[str, tuple] = {}
        self._answers = None
        self.converted: dict[str, dict[str, dict]] = {}

    # -- which conversion made a file -------------------------------------

    def task_for(self, name: str) -> Optional[dict]:
        """The conversion record of the row that wrote the file ``name``."""
        if self._tasks is None:
            self._tasks = self._read_provenance()
        return self._tasks.get(name)

    def _read_provenance(self) -> dict[str, dict]:
        own: dict[str, dict] = {}
        others: dict[str, dict] = {}
        folder = self.root / ".bidsmgr" / "provenance"
        for path in sorted(folder.glob("sub-*.json")) if folder.is_dir() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            renames = {r.get("from"): r.get("to") for r in data.get("fieldmap_renames", [])
                       if isinstance(r, dict)}
            tasks = [t for t in data.get("tasks", [])
                     if isinstance(t, dict) and t.get("success", True) and t.get("row_id")]
            for task in tasks:
                base = str(task.get("basename", ""))
                for output in task.get("outputs", []) or []:
                    name = renames.get(output) or output
                    # A recording's own outputs first: an MNE-BIDS task lists
                    # every file of its folder, its neighbours' included.
                    if base and name.startswith((base + "_", base + ".")):
                        own[name] = task
                    else:
                        others.setdefault(name, task)
        return {**others, **own}

    # -- the scan the row is in -------------------------------------------

    def _scan_versions(self) -> list:
        if self._versions is None:
            try:
                from ..project.workspace import list_versions

                self._versions = list(reversed(list_versions(self.root)))
            except Exception:  # noqa: BLE001 - not a project
                self._versions = []
        return self._versions

    def _version(self, version) -> tuple:
        """``(rows by id, sources, spec)`` of one scan version."""
        key = version.version_id
        if key not in self._loaded:
            from ..inventory import sources as src
            from ..project.orchestration import row_id, version_dataframe
            from ..recording_meta import load_spec, scaffold_sidecar_path

            try:
                df = version_dataframe(version)
                rows = {row_id(df, i): df.loc[i] for i in df.index}
                files_by_uid = src.load_files_by_uid(Path(version.inventory))
                found, _seen = src.resolve(df, files_by_uid, recorded=version.raw_root,
                                           label=version.source_label)
            except Exception:  # noqa: BLE001 - an unreadable scan is no scan
                log.debug("scan %s unreadable", key, exc_info=True)
                rows, found = {}, None
            spec = None
            try:
                scaffold = scaffold_sidecar_path(Path(version.inventory))
                if scaffold.exists():
                    spec = load_spec(scaffold)
            except Exception:  # noqa: BLE001 - answers are a bonus
                log.debug("no answers in %s", key, exc_info=True)
            self._loaded[key] = (rows, found, spec)
        return self._loaded[key]

    def row(self, row_id: str):
        """``(row, sources, spec)`` for ``row_id``, newest scan first."""
        for version in self._scan_versions():
            rows, found, spec = self._version(version)
            if row_id in rows:
                return rows[row_id], found, spec
        return None

    def answers(self):
        """``(spec, row id by name, stated cells by row id)`` of the newest scan."""
        if self._answers is None:
            from ..fixups.sidecar_schema import _rows_from_inventory

            spec, by_name, cells = None, {}, {}
            versions = self._scan_versions()
            if versions:
                rows, _found, spec = self._version(versions[0])
                if rows:
                    import pandas as pd

                    by_name, cells = _rows_from_inventory(pd.DataFrame(list(rows.values())))
            self._answers = (spec, by_name, cells)
        return self._answers

    def converted_here(self) -> bool:
        """Whether this project converted anything (its provenance)."""
        return (self.root / ".bidsmgr" / "provenance").is_dir()


def _from_source(project: _Project, recording: Path, sidecar_name: str,
                 work: Path) -> tuple[Optional[dict], str, str]:
    """``(the sidecar the conversion wrote, what was read, why not)``."""
    task = project.task_for(recording.name) or project.task_for(sidecar_name)
    if task is None:
        return None, "", ("BIDS Manager has no record of converting this file (it was "
                          "converted elsewhere, or renamed since)")
    if str(task.get("datatype", "")) in ("eeg", "meg", "ieeg", "nirs"):
        # The recording in the dataset IS the source format: its header is
        # read directly, below, which is what converting it again would do.
        return None, "", ""
    row_id = str(task["row_id"])
    if row_id not in project.converted:
        found = project.row(row_id)
        if found is None:
            project.converted[row_id] = {"": "the scan that converted it is no longer in "
                                              "this project"}
        else:
            row, sources, spec = found
            if sources is None or sources.root is None:
                project.converted[row_id] = {"": "the scan does not say where its data is"}
            else:
                from ..cli.convert import TaskOptions, sidecars_for_row

                row = row.copy()
                # Named as it was converted, so the outputs carry the names the
                # dataset has, whatever the scan table has been edited to since.
                row["bids_name"] = str(task.get("basename", "")) or row.get("bids_name", "")
                row["datatype"] = str(task.get("datatype", "")) or row.get("datatype", "")
                if task.get("suffix"):
                    row["bids_guess_suffix"] = str(task["suffix"])
                # And converted as it was: the volumes removed then, the
                # physiology's layout, not what the table or settings say now.
                row["drop_volumes"] = ""
                options = TaskOptions(
                    drop_volumes=int(task.get("drop_volumes") or 0),
                    physio_layout=str(task.get("physio_layout") or "by_rate"),
                    physio_align=bool(task.get("physio_align", False)))
                try:
                    project.converted[row_id] = sidecars_for_row(
                        row, work / f"row{len(project.converted)}", sources, spec=spec,
                        options=options)
                except Exception as exc:  # noqa: BLE001 - reported per file
                    log.debug("source conversion failed for %s", row_id, exc_info=True)
                    project.converted[row_id] = {"": _short(exc)}
    produced = project.converted[row_id]
    if "" in produced:
        return None, "", f"the source data could not be used: {produced['']}"
    doc = produced.get(sidecar_name)
    if doc is None and len(produced) == 1:
        doc = next(iter(produced.values()))
    if doc is None:
        return None, "", "converting the source data again wrote no sidecar of this name"
    return dict(doc), "the data it was converted from, converted again", ""


def _short(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return (text[-1] if text else type(exc).__name__)[:200]


# ---------------------------------------------------------------------------
# The links other sidecars hold
# ---------------------------------------------------------------------------


def _without(entities: dict, keys: Iterable[str]) -> dict:
    drop = set(keys)
    return {k: v for k, v in entities.items() if k not in drop}


def _fieldmap_links(recording: Path, own_sidecar: Path, listing: _Listing) -> dict:
    """What the other images of the same fieldmap say it is and is for."""
    mine = _without(entities_of(recording.name), _WITHIN_FIELDMAP)
    out: dict = {}
    for other in listing.jsons(recording.parent):
        if other == own_sidecar or suffix_of(other.name) not in _FIELDMAP_SUFFIXES:
            continue
        if _without(entities_of(other.name), _WITHIN_FIELDMAP) != mine:
            continue
        data = listing.load(other) or {}
        for key in _FIELDMAP_SHARED:
            if key not in out and key in data and not _is_todo(data[key]):
                out[key] = data[key]
    return out


def _echo_times(recording: Path) -> dict:
    """A phase difference's two echo times, from its magnitude images."""
    out: dict = {}
    for key, suffix in (("EchoTime1", "magnitude1"), ("EchoTime2", "magnitude2")):
        stem = recording.name[: -len(extension(recording.name))]
        image = recording.with_name(stem[: -len("phasediff")] + suffix
                                    + extension(recording.name))
        te = echo_time(image) if image.exists() else None
        if te is not None:
            out[key] = te
    if "EchoTime2" not in out:
        te = echo_time(recording)
        if te is not None and te != out.get("EchoTime1"):
            out["EchoTime2"] = te
    return out if len(out) == 2 else {}


def _names(root: Path, recording: Path) -> set[str]:
    """Every spelling a reference to ``recording`` may use."""
    rel = recording.relative_to(root).as_posix()
    subject = rel.split("/", 1)[1] if "/" in rel else rel
    return {rel, f"bids::{rel}", subject}


def _fieldmaps_for(root: Path, recording: Path, listing: _Listing) -> list:
    """The ``B0FieldIdentifier`` of every fieldmap intended for ``recording``."""
    session = _session_root(root, recording)
    if session is None:
        return []
    names = _names(root, recording)
    found: list = []
    for fmap in listing.jsons(session / "fmap"):
        data = listing.load(fmap) or {}
        targets = data.get("IntendedFor")
        targets = [targets] if isinstance(targets, str) else (targets or [])
        if not any(str(t) in names for t in targets):
            continue
        ident = data.get("B0FieldIdentifier")
        for value in ident if isinstance(ident, list) else [ident]:
            if isinstance(value, str) and value and not _is_todo(value) \
                    and value not in found:
                found.append(value)
    return found


def _physio_derived(datatype: str, name: str, found: dict) -> dict:
    """What the conversion's physio step derives for this sidecar: run on a
    copy in a temporary folder, so it is the same rule."""
    from ..fixups.physio_sidecar import enrich_physio_sidecars

    with tempfile.TemporaryDirectory(prefix="bidsmgr-physio-", ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "sub-x" / datatype / name
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(found), encoding="utf-8")
        enrich_physio_sidecars(Path(tmp) / "sub-x")
        data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k not in found}


def _declares(datatype: str, suffix: str, name: str, field_name: str,
              root: Path) -> bool:
    from .. import schema as schema_mod

    try:
        specs = schema_mod.sidecar_fields(datatype, suffix, root,
                                          entities=schema_mod.name_entities(name))
    except (KeyError, ValueError, AttributeError):
        return False
    return any(s.name == field_name for s in specs)


# ---------------------------------------------------------------------------
# Anonymisation the dataset had
# ---------------------------------------------------------------------------


_METHOD = re.compile(r"BIDS Manager \S* ?anonymisation \((?P<name>[a-z]+)\)")


def _anonymisation(root: Path, listing: _Listing):
    """``(policy, state)`` the dataset was anonymised with, or ``None``.

    The project remembers the date shift and the pseudonym secret, not the
    choices; those are read from the record anonymising leaves in the
    sidecars. A custom choice is not recorded, so it is matched by Strict,
    which removes the most.
    """
    from ..anonymise import state as anon_state
    from ..anonymise.policy import PRESETS, STRICT

    if not (root / anon_state.STATE_FILE).exists():
        return None
    name = ""
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [n for n in dirnames if not n.startswith(".")
                       and (here != root or n.startswith("sub-"))]
        for fname in filenames:
            if not fname.endswith(".json"):
                continue
            method = (listing.load(here / fname) or {}).get("DeidentificationMethod")
            for item in method if isinstance(method, list) else [method]:
                m = _METHOD.search(str(item or ""))
                if m:
                    name = m.group("name")
                    break
            if name:
                break
        if name:
            break
    policy = PRESETS.get(name) or PRESETS[STRICT]
    return policy, anon_state.load(root)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _schema_problems(data: dict) -> dict[str, str]:
    """``{field: why}`` for every value the BIDS version in force rejects:
    the validator's own check, over every definition of the name (a field
    whose name has several is valid if any accepts it)."""
    from .. import schema as schema_mod
    from ..vendor.bidsval.rules.values import format_patterns, validate_value
    from ..vendor.bidsval.schema import introspect

    schema = schema_mod.get_schema()
    by_name = introspect.metadata_by_name(schema)
    formats = format_patterns(schema)
    out: dict[str, str] = {}
    for name, value in data.items():
        definitions = by_name.get(name)
        if not definitions:
            continue
        problems = [validate_value(value, d, formats) for d in definitions]
        if all(problems):
            out[name] = f"{name} {problems[0][0]}"
    return out


def _repair(data: dict, datatype: str, suffix: str) -> None:
    """The conversion's repairs and its identifier removal, on one dict."""
    from ..fixups.identifiers import strip_identifiers
    from ..fixups.sidecar_schema import (
        drop_stringified_nulls,
        repair_array_types,
        repair_datetimes,
        repair_key_names,
    )

    repair_key_names(data, datatype, suffix)
    repair_array_types(data, datatype, suffix)
    repair_datetimes(data, datatype, suffix)
    drop_stringified_nulls(data)
    strip_identifiers(data)


def _carry_origins(data: dict, origins: dict[str, str]) -> None:
    """Keep each field's origin when a repair renamed it (case only)."""
    lowered = {k.lower(): v for k, v in origins.items()}
    for key in data:
        if key not in origins and key.lower() in lowered:
            origins[key] = lowered[key.lower()]


def plan_sidecars(
    root: Path,
    targets: Optional[Iterable[Path]] = None,
    *,
    complete: bool = False,
    use_source: bool = True,
    fill_scope: str = "none",
    kinds: Optional[dict[str, tuple[str, str]]] = None,
    converted: Optional[dict] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> SidecarsPlan:
    """What writing the sidecars of ``targets`` (the whole dataset when
    ``None``) would do. Reads; never writes into the dataset.

    ``complete`` also adds to sidecars that are there; without it a recording
    with a sidecar is left alone. ``use_source`` converts the project's
    source data again where it can be found (slower, and the most complete).
    ``fill_scope`` is the metadata step's placeholder scope (``"none"``,
    ``"required"``, ``"recommended"``, ``"optional"``). ``kinds`` says what a
    file outside BIDS naming is recorded as (dataset-relative path ->
    ``(datatype, suffix)``); the rest are read from their name or the file.

    Fields and values follow the BIDS version in force
    (:func:`bidsmgr.schema.set_active_version`): a value that version would
    reject is never written, and is shown with the reason.

    ``converted`` keeps what converting the source data again gave, by scan
    row, between plans: a dialog that plans again after every choice converts
    each row once.

    ``progress(done, total)`` is called per recording; ``cancel()`` is polled
    between them and stops the plan (returned as far as it got).
    """
    from ..metadata.engine import mark_sidecar, scope_levels

    from .. import schema as schema_mod

    root = Path(root)
    plan = SidecarsPlan(root, bids_version=schema_mod.bids_version())
    kinds = dict(kinds or {})
    listing = _Listing()
    project = _Project(root)
    if converted is not None:
        project.converted = converted
    levels = scope_levels(fill_scope) or None
    anonymised = _anonymisation(root, listing)
    if anonymised is not None:
        plan.anonymised = anonymised[0].label
    use_source = use_source and project.converted_here()
    plan.used_source = use_source

    todo = recordings(root, targets)
    total = len(todo)
    with tempfile.TemporaryDirectory(prefix="bidsmgr-sidecars-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        for i, recording in enumerate(todo):
            if cancel is not None and cancel():
                break
            if progress is not None:
                progress(i, total)
            rel = recording.relative_to(root).as_posix()
            try:
                kind = kind_of(root, recording, kinds.get(rel))
                if not kind.datatype:
                    plan.skipped.append((rel, kind.why))
                    continue
                one = _plan_one(root, recording, kind, listing, project, work,
                                complete=complete, use_source=use_source,
                                levels=levels, anonymised=anonymised,
                                mark_sidecar=mark_sidecar)
            except Exception as exc:  # noqa: BLE001 - one file, reported
                log.warning("could not plan the sidecar of %s", rel, exc_info=True)
                plan.skipped.append((rel, f"could not be read: {_short(exc)}"))
                continue
            if isinstance(one, str):
                plan.skipped.append((rel, one))
            else:
                plan.files.append(one)
        if progress is not None:
            progress(total, total)
    return plan


def _plan_one(root: Path, recording: Path, kind: Kind, listing: _Listing,
              project: _Project, work: Path, *, complete: bool, use_source: bool,
              levels, anonymised, mark_sidecar):
    """One recording's :class:`SidecarPlan`, or why it is left alone."""
    from ..fixups.sidecar_schema import _row_id_for, state_into

    datatype, suffix = kind.datatype, kind.suffix
    own_path = sidecar_path(recording)
    # Inheritance is a BIDS rule about BIDS names: outside them a sidecar
    # describes the one file it is named after.
    describing = (_describing(root, recording, listing) if kind.bids
                  else [own_path] if own_path.exists() else [])
    own = listing.load(own_path) if own_path.exists() else None
    if own_path.exists() and own is None:
        return "its sidecar is not valid JSON: fix or delete it first"
    beside = [p for p in describing if p != own_path and p.parent == recording.parent]
    if not complete and (own is not None or beside):
        return "has a sidecar" if own is not None else \
            f"described by {beside[0].name}"
    inherited = [(p, listing.load(p) or {}) for p in describing if p != own_path]

    one = SidecarPlan(recording.relative_to(root).as_posix(),
                      own_path.relative_to(root).as_posix(), datatype, suffix,
                      exists=own is not None, kept=len(own or {}), kind=kind)
    found: dict = {}
    origins: dict[str, str] = {}

    # 1. The source data.
    task = None
    if use_source:
        doc, read, why = _from_source(project, recording, own_path.name, work)
        task = project.task_for(recording.name) or project.task_for(own_path.name)
        if doc:
            found.update(doc)
            origins.update(dict.fromkeys(doc, SOURCE))
            one.read.append(read)
        elif why:
            one.notes.append(why[:1].upper() + why[1:] + ".")

    # 2. The recording's own header.
    header = read_header(recording, datatype, suffix)
    added = {k: v for k, v in header.fields.items() if k not in found}
    if added:
        one.read.append(header.reader)
        found.update(added)
        origins.update(dict.fromkeys(added, HEADER))
    if header.note and (added or not found):
        one.notes.append(header.note)

    _repair(found, datatype, suffix)
    _carry_origins(found, origins)

    # 3. What the user answered.
    spec, by_name, cells = project.answers()
    if spec is not None:
        row_id = str(task["row_id"]) if task else _row_id_for(own_path, by_name)
        before = dict(found)
        state_into(found, own_path.name, datatype, suffix, spec,
                   row_id=row_id, row_values=cells.get(row_id))
        for key, value in found.items():
            if key not in before or before[key] != value:
                origins[key] = ANSWERS

    # 4. The links other sidecars hold.
    links: dict = {}
    if datatype == "fmap":
        links.update(_fieldmap_links(recording, own_path, listing))
        if suffix == "phasediff" and not {"EchoTime1", "EchoTime2"} & set(found):
            echoes = _echo_times(recording)
            if echoes:
                links.update(echoes)
                one.read.append("the echo times of its magnitude images")
                one.notes.append(ROUNDED_TE)
    elif _declares(datatype, suffix, recording.name, "B0FieldSource", root):
        sources = _fieldmaps_for(root, recording, listing)
        if sources:
            links["B0FieldSource"] = sources[0] if len(sources) == 1 else sources
    for key, value in links.items():
        if key not in found:
            found[key] = value
            origins[key] = HEADER if key.startswith("EchoTime") else LINKS

    # What a physio table's name gives (TaskName, PhysioType), by the
    # conversion's own physio step.
    if suffix in ("physio", "physioevents"):
        for key, value in _physio_derived(datatype, own_path.name, found).items():
            if key not in found:
                found[key] = value
                origins[key] = FILENAME

    # Anonymised like the rest of the dataset.
    if anonymised is not None and found:
        from ..anonymise.carriers import clean_json

        policy, state = anonymised
        participant = recording.relative_to(root).parts[0]
        image = extension(recording.name) in (".nii", ".nii.gz")
        method = None
        if image:
            from ..anonymise.plan import _method

            method = _method(policy)
        cleaned, changes, _reviews = clean_json(found, policy, state,
                                                state.days_back(participant, None),
                                                method=method)
        found = cleaned
        if "DeidentificationMethod" in found and \
                "DeidentificationMethod" not in origins:
            origins["DeidentificationMethod"] = ANONYMISED

    # Nothing the selected BIDS version would reject: shown, not written.
    for key, problem in _schema_problems(found).items():
        one.rejected.append(Rejected(key, found.pop(key), origins.get(key, HEADER), problem))

    # 5. The file name and the placeholders, on everything the recording will
    # have, so a placeholder is never written for a field stated anywhere.
    effective: dict = {}
    for _path, data in reversed(inherited):
        effective.update({k: v for k, v in data.items() if not _is_todo(v)})
    effective.update({k: v for k, v in (own or {}).items() if not _is_todo(v)})
    effective.update(found)
    marks = mark_sidecar(dict(effective), own_path.name, datatype, suffix, levels,
                         bids_root=root)
    for key, value in marks.derived.items():
        found.setdefault(key, value)
        origins.setdefault(key, FILENAME)
    for key, value in marks.todo_added.items():
        if key not in found:
            found[key] = value
            origins[key] = PLACEHOLDER
    one.still_missing = [n for n in marks.missing_required
                         if n not in found and n not in marks.todo_added]

    # What is written: never over a value, never over an inherited one.
    document = dict(own or {})
    for key, value in found.items():
        mine = (own or {}).get(key, _MISSING)
        if mine is not _MISSING and not _is_todo(mine):
            continue
        origin = origins.get(key, HEADER)
        if mine is not _MISSING and origin == PLACEHOLDER:
            continue
        stated = next(((p, d[key]) for p, d in inherited
                       if key in d and not _is_todo(d[key])), None)
        overrides, overridden = "", None
        if stated is not None:
            source, value_there = stated
            differs = value_there != value and origin != PLACEHOLDER
            if not (differs and origin in _AUTHORITATIVE):
                one.inherited.append(Inherited(key, value_there,
                                               source.relative_to(root).as_posix(),
                                               value if differs else None, differs))
                continue
            # The data this recording was converted from, or the user, says
            # otherwise: written here, where it wins for this recording only.
            overrides, overridden = source.relative_to(root).as_posix(), value_there
            one.notes.append(
                f"{key} is {json.dumps(value_there)} in {overrides}; the "
                f"{ORIGINS[origin].lower()} say {json.dumps(value)}, so it is written "
                "here, where it holds for this recording only.")
        document[key] = value
        one.added.append(NewField(key, value, origin, replaces_todo=mine is not _MISSING,
                                  overrides=overrides, overridden=overridden))
    one.document = document
    return one


_MISSING = object()

#: Origins that speak for this recording exactly: where one disagrees with a
#: sidecar higher up, its value is written into the recording's own sidecar.
#: A header read can be approximate (an echo time to two figures), so where
#: it disagrees the difference is shown and the inherited value left.
_AUTHORITATIVE = frozenset({SOURCE, ANSWERS})


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def apply_sidecars(root: Path, plans: Iterable[SidecarPlan], *, op=None) -> list[Path]:
    """Write the planned sidecars, as one step of the Editor's history (or
    into ``op``, a running operation). Returns what was written."""
    from ..project.operations import begin_operation

    root = Path(root)
    chosen = [p for p in plans if p.changes]
    if not chosen:
        return []
    written: list[Path] = []

    def write(o) -> None:
        for plan in chosen:
            path = root / plan.sidecar
            o.write_json(path, plan.document)
            written.append(path)

    if op is not None:
        write(op)
    else:
        n = len(chosen)
        with begin_operation(root, f"Write {n} sidecar{'s' if n != 1 else ''} "
                                   "from the recordings") as o:
            write(o)
    return written


__all__ = [
    "ANONYMISED", "ANSWERS", "FILENAME", "HEADER", "Inherited", "KIND_BIDS", "KIND_CHOSEN",
    "KIND_FOLDER", "KIND_GUESS", "KIND_NAME", "Kind", "LINKS", "NOT_THE_DATASETS", "NewField",
    "ORIGINS", "ORIGIN_HELP", "PLACEHOLDER", "Rejected", "SOURCE", "SidecarPlan",
    "SidecarsPlan", "apply_sidecars", "is_candidate", "is_readable", "is_recording",
    "kind_choices", "kind_of", "plan_sidecars", "recording_for", "recordings", "sidecar_path",
]

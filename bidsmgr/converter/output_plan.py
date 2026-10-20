"""What a conversion will write, before it runs.

The Filter panel's BIDS preview shows the dataset the conversion and the
metadata step will produce: every data file AND what comes with it (a sidecar,
``.bval``/``.bvec``, MNE-BIDS' channel and event tables, an eye tracker's two
files per eye, the files attached to a row, the ``*_scans.tsv`` of each
session, ``dataset_description.json``, ``participants.tsv``, the companion
stubs and the citation file when those settings are on, a ``sourcedata/``
copy), and what each file will roughly contain, so a user sees the result of
their settings and their answers before converting.

Two halves:

* :func:`plan_output` names the files, the way each converter names them
  (dcm2niix, MNE-BIDS, the physio and eye-tracking writers, the metadata
  step). A file a converter writes only sometimes is
  :attr:`PlannedFile.conditional` with the reason ("when the recording has
  events").
* :func:`approximate` predicts one file's content. Every value says where it
  comes from (:data:`SOURCES`).

It is a prediction, but not a second description of the conversion: every
step that changes what a file holds is the conversion's own function, run on
the predicted content. What the user stated goes through
:func:`~bidsmgr.fixups.sidecar_schema.state_into`, the placeholders through
:func:`~bidsmgr.metadata.engine.mark_sidecar`, the dataset description through
:func:`~bidsmgr.metadata.engine.describe_dataset`, the participants table
through :func:`~bidsmgr.metadata.engine.participant_rows`, the companion stubs
through :func:`~bidsmgr.fixups.associations.missing_for`. Qt-free: the GUI
calls it on a worker thread.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

#: Where a value in a predicted file comes from.
SOURCES = {
    "scan": "read from the header during the scan",
    "converter": "read by the converter at conversion (not known yet)",
    "settings": "written by BIDS Manager from your settings and the table",
    "stated": "what you stated: the metadata templates, per-file answers and dataset metadata",
    "todo": "a TODO placeholder the Write TODO setting writes",
}
#: Shown for a value the converter reads at conversion.
UNKNOWN = "…"

_FILE_BASED_EEG = {".edf": ".edf", ".bdf": ".bdf", ".set": ".set", ".vhdr": ".vhdr"}
_MEG_FORMATS = {".fif": ".fif", ".fif.gz": ".fif", ".ds": ".ds", ".con": ".con", ".sqd": ".sqd",
                ".kdf": ".kdf"}
_MNE_DATATYPES = ("eeg", "ieeg", "meg", "nirs")


@dataclass(frozen=True)
class PlanOptions:
    """The settings that change what is written."""

    dataset: str = ""
    #: The dataset's folder when it exists already (a project): the levels of
    #: a few fields depend on what is in it.
    root: str = ""
    #: Where the inventory is: the files the dataset metadata names (a
    #: participants spreadsheet, phenotype tables) are relative to it.
    inventory_dir: str = ""
    run_metadata: bool = True
    #: Which fields the metadata step leaves a TODO in: "none", "required",
    #: "recommended" or "optional" (``AppSettings.placeholder_scope``).
    fill_scope: str = "recommended"
    #: The metadata step also writes the missing companion files (stubs).
    companions: bool = False
    #: The metadata step also writes CITATION.cff from the description.
    citation: bool = False
    #: The metadata step writes the ``*_scans.tsv`` and ``*_sessions.tsv``
    #: tables (``metadata.tables``).
    tables: bool = True
    intended_for: str = "uri"            # "uri" | "relative"
    b0_field: bool = True
    deface: bool = False
    deface_engine: str = ""
    sourcedata: str = "none"             # "none" | "copy" | "archive"
    sourcedata_modalities: tuple[str, ...] = ()
    force_edf: bool = False
    bids_version: str = ""
    version: str = ""                    # BIDS Manager's
    dcm2niix_version: str = ""


@dataclass(frozen=True)
class PlannedFile:
    """One file the conversion or the metadata step will write."""

    path: str                      # relative to the dataset root, POSIX
    origin: str                    # who writes it
    rows: tuple[int, ...] = ()     # the inventory rows behind it
    conditional: str = ""          # why it may not be written; "" = always
    role: str = "data"             # data | sidecar | table | dataset | text | archive
    #: A file that will be written but is not valid BIDS, and why.
    warning: str = ""
    #: Written under another name than the table's, and why.
    renamed: str = ""
    #: A stub the companion setting writes: "events" | "channels" | "sidecar".
    stub: str = ""


@dataclass
class OutputPlan:
    files: list[PlannedFile] = field(default_factory=list)

    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    def find(self, path: str) -> Optional[PlannedFile]:
        return next((f for f in self.files if f.path == path), None)

    def of_row(self, row: int) -> list[PlannedFile]:
        return [f for f in self.files if row in f.rows]


@dataclass(frozen=True)
class FilePreview:
    """A file's predicted content: ``fields`` for a JSON file (key, value,
    source), ``header`` and ``rows`` for a table, ``text`` for anything else,
    and what it cannot say in ``note``."""

    kind: str                                      # "json" | "tsv" | "text" | "binary" | "folder"
    fields: tuple[tuple[str, object, str], ...] = ()
    header: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    text: str = ""
    note: str = ""

    def json_text(self) -> str:
        """The JSON as it would read, ``…`` for what is not known yet."""
        return json.dumps({k: v for k, v, _s in self.fields}, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Reading the table
# ---------------------------------------------------------------------------


def _cell(df, i, col: str) -> str:
    if col not in df.columns:
        return ""
    v = df.at[i, col]
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v).strip()


def _included(df, included: Optional[Sequence[bool]]):
    if included is not None:
        return [bool(v) for v in included]
    if "include" not in df.columns:
        return [True] * len(df)
    return [str(v).strip() == "1" for v in df["include"].tolist()]


def _split_name(name: str) -> dict[str, str]:
    out = {}
    for token in name.split("_")[:-1]:
        key, sep, value = token.partition("-")
        if sep:
            out[key] = value
    return out


def _ext(path: str) -> str:
    low = path.lower()
    for double in (".fif.gz", ".nii.gz", ".tsv.gz"):
        if low.endswith(double):
            return double
    dot = low.rfind(".")
    return low[dot:] if dot >= 0 else ""


def _folder(subject: str, session: str, datatype: str) -> str:
    parts = [subject] + ([session] if session else []) + ([datatype] if datatype else [])
    return "/".join(parts)


def _safe(name: str) -> str:
    try:
        from ..util.paths import safe_path_component

        return safe_path_component(name)
    except Exception:  # noqa: BLE001 - only a name
        return re.sub(r"[^0-9A-Za-z._-]+", "_", name) or "series"


def _row_id(df, i) -> str:
    from ..project.orchestration import row_id

    return row_id(df, i)


# ---------------------------------------------------------------------------
# What each converter writes
# ---------------------------------------------------------------------------


#: The signals a Siemens physio log can hold; bidsphysio writes one pair of
#: files per signal (``recording-<signal>``) when they were sampled apart.
_PHYSIO_SIGNALS = ("cardiac", "ecg", "respiratory", "trigger")


#: Suffixes whose echoes are the point of the series (a fieldmap's two
#: magnitudes): never split by echo.
_ECHO_SUFFIXES = {"magnitude", "magnitude1", "magnitude2", "phasediff", "phase1", "phase2",
                  "fieldmap", "epi"}
_SPLIT_WARNING = ("dcm2niix splits a series with several echoes into one file per echo "
                  "(_e1, _e2, ...), names BIDS does not allow; the validator will list them as "
                  "not included. Name each echo with the echo entity, or leave the series out.")


def _mne_basename(name: str) -> str:
    """The name MNE-BIDS writes for a table name.

    The backend hands it only the subject, session, task and run, the run as
    an integer, so ``run-01`` comes out ``run-1`` and any other entity
    (``acq-``, ``rec-``, ...) is not written.
    """
    ents = _split_name(name)
    parts = [f"{key}-{ents[key]}" for key in ("sub", "ses", "task") if ents.get(key)]
    run = ents.get("run", "")
    if run:
        try:
            parts.append(f"run-{int(float(run))}")
        except ValueError:
            pass
    return "_".join(parts + [name.rsplit("_", 1)[-1]])


def _renamed_reason(name: str, written: str) -> str:
    if written == name:
        return ""
    lost = [k for k in _split_name(name) if k not in ("sub", "ses", "task", "run")]
    bits = []
    if lost:
        bits.append(", ".join(f"{k}-{_split_name(name)[k]}" for k in lost) + " left out")
    run = _split_name(name).get("run", "")
    if run and _split_name(written).get("run", "") != run:
        bits.append(f"run-{run} written as run-{_split_name(written).get('run', '')}")
    return (f"MNE-BIDS writes this recording as {written} ({'; '.join(bits)}): it takes only "
            "the subject, session, task and run from the name. Answers aimed at the name in "
            "the table, and files attached to the row, do not reach it.")


def _files_for_row(df, i, two_echoes: bool, force_edf: bool,
                   diffusion: bool = False, echoes: int = 1) -> list[PlannedFile]:
    name = _cell(df, i, "bids_name")
    datatype = _cell(df, i, "datatype")
    subject = _cell(df, i, "participant_id") or f"sub-{_split_name(name).get('sub', '')}"
    session = _cell(df, i, "session")
    if session and not session.startswith("ses-"):
        session = f"ses-{session}"
    folder = _folder(subject, session, datatype)
    base = f"{folder}/{name}"
    suffix = name.rsplit("_", 1)[-1]
    modality = _cell(df, i, "modality").lower()
    source = _cell(df, i, "source_file")
    rows = (int(i),)
    out: list[PlannedFile] = []

    def add(path, origin, role="data", conditional="", warning="", renamed=""):
        out.append(PlannedFile(path=path, origin=origin, rows=rows, conditional=conditional,
                               role=role, warning=warning, renamed=renamed))

    if modality == "eyetrack":
        stem = base.rsplit("_", 1)[0]
        stem = re.sub(r"_recording-eye[12]", "", stem)
        for eye, cond in (("eye1", ""), ("eye2", "when the file records both eyes")):
            for kind in ("physio", "physioevents"):
                add(f"{stem}_recording-{eye}_{kind}.tsv.gz", "BIDS Manager (eye tracking)",
                    "data" if kind == "physio" else "table", cond)
                add(f"{stem}_recording-{eye}_{kind}.json", "BIDS Manager (eye tracking)",
                    "sidecar", cond)
        return out

    uid = _cell(df, i, "series_uid")
    if uid:                                              # a DICOM series
        if suffix == "physio":
            stem = base[: -len("_physio")]
            add(f"{base}.tsv.gz", "bidsphysio", conditional="when the log holds a single signal")
            add(f"{base}.json", "bidsphysio", "sidecar", "when the log holds a single signal")
            for signal in _PHYSIO_SIGNALS:
                cond = f"when the log holds a {signal} signal (one file per signal)"
                add(f"{stem}_recording-{signal}_physio.tsv.gz", "bidsphysio", conditional=cond)
                add(f"{stem}_recording-{signal}_physio.json", "bidsphysio", "sidecar", cond)
            return out
        origin = "dcm2niix"
        if echoes >= 2 and suffix not in _ECHO_SUFFIXES:
            for k in range(1, echoes + 1):
                add(f"{base}_e{k}.nii.gz", origin, warning=_SPLIT_WARNING)
                add(f"{base}_e{k}.json", origin, "sidecar", warning=_SPLIT_WARNING)
            return out
        add(f"{base}.nii.gz", origin)
        add(f"{base}.json", origin, "sidecar")
        if suffix == "dwi" or (suffix == "epi" and diffusion):
            add(f"{base}.bval", origin, "table")
            add(f"{base}.bvec", origin, "table")
        if suffix == "magnitude1":
            stem = base[: -len("magnitude1")]
            if two_echoes:
                add(f"{stem}magnitude2.nii.gz", origin)
                add(f"{stem}magnitude2.json", origin, "sidecar")
            if "|" in uid:                     # the magnitude and its phase, joined
                add(f"{stem}phasediff.nii.gz", origin)
                add(f"{stem}phasediff.json", origin, "sidecar")
        return out

    ext = _ext(source)
    if datatype == "pet":                               # ECAT
        add(f"{base}.nii.gz", "BIDS Manager (nibabel)")
        add(f"{base}.json", "BIDS Manager (nibabel)", "sidecar")
        return out
    if datatype in _MNE_DATATYPES:
        origin = "MNE-BIDS"
        written = _mne_basename(name)
        renamed = _renamed_reason(name, written)
        base = f"{folder}/{written}"
        if datatype == "meg":
            data_ext = _MEG_FORMATS.get(ext, ext or ".fif")
        elif force_edf and datatype != "nirs":
            data_ext = ".edf"
        elif datatype == "nirs":
            data_ext = ".snirf"
        else:
            data_ext = _FILE_BASED_EEG.get(ext, ".vhdr")
        add(f"{base}{data_ext}", origin, renamed=renamed)
        if data_ext == ".vhdr":
            add(f"{base}.eeg", origin)
            add(f"{base}.vmrk", origin)
        if data_ext == ".set" and ext == ".set":
            add(f"{base}.fdt", origin, conditional="when the EEGLAB file keeps its data apart")
        add(f"{base}.json", origin, "sidecar", renamed=renamed)
        stem = base.rsplit("_", 1)[0]
        # Positions describe the session, not a run: no task, no run.
        named = _split_name(written)
        place = "_".join(f"{k}-{named[k]}" for k in ("sub", "ses", "acq") if k in named)
        place = f"{folder}/{place}"
        add(f"{stem}_channels.tsv", origin, "table")
        add(f"{stem}_events.tsv", origin, "table",
            "when the recording has events or annotations")
        add(f"{stem}_events.json", origin, "sidecar",
            "when the recording has events or annotations")
        if datatype in ("eeg", "ieeg"):
            add(f"{place}_electrodes.tsv", origin, "table",
                "when electrode positions are known (a montage or digitised positions)")
            add(f"{place}_coordsystem.json", origin, "sidecar",
                "when electrode positions are known")
        if datatype == "meg":
            add(f"{place}_coordsystem.json", origin, "sidecar",
                "when the recording carries head digitisation")
        return out
    add(f"{base}{ext}", "BIDS Manager")
    return out


def _attached(df, i, files: list[PlannedFile]) -> list[PlannedFile]:
    """The files the user attached to row ``i``, where the conversion puts
    them: beside the recording, under its name with their own suffix.

    The copier finds the recording by the table's name, so a recording
    MNE-BIDS wrote under another name receives none of them (its data file
    says so).
    """
    from ..fixups.blood import parse_blood_role
    from ..fixups.companion import ALLOWED_COMPANION_SUFFIXES, parse_companions, src_extension

    pairs = parse_companions(_cell(df, i, "companion_files"))
    if not pairs:
        return []
    table = _cell(df, i, "bids_name")
    anchor = next((f for f in files if f.path.rsplit("/", 1)[-1] == f"{table}.json"), None)
    if anchor is None:
        return []
    folder = anchor.path.rsplit("/", 1)[0]
    prefix = table.rsplit("_", 1)[0] if "_" in table else table
    if _cell(df, i, "modality").lower() == "eyetrack":
        from .backends.eyelink import match_of

        prefix = match_of(table)
    rows = (int(i),)
    out: dict[str, PlannedFile] = {}
    for suffix, src in pairs:
        role = parse_blood_role(str(suffix))
        if role is not None:
            label = "manual" if role[1] == "manual" else "autosampler"
            for ext, kind in ((".tsv", "table"), (".json", "sidecar")):
                path = f"{folder}/{prefix}_recording-{label}_blood{ext}"
                out[path] = PlannedFile(path, "pet2bids (blood)", rows, role=kind,
                                        conditional="as pet2bids names the recording")
            continue
        if suffix not in ALLOWED_COMPANION_SUFFIXES:
            continue
        path = f"{folder}/{prefix}_{suffix}{src_extension(Path(src))}"
        out[path] = PlannedFile(path, "BIDS Manager (your file, copied)", rows,
                                role="sidecar" if path.endswith(".json") else "table")
    return list(out.values())


def _features_of(features: Optional[Mapping], uid: str):
    """A series' features; a row that joins two series (``uid|uid``) has
    the first one's."""
    if not features or not uid:
        return None
    return features.get(uid) or features.get(uid.split("|", 1)[0])


def _two_echoes(features: Mapping, uid: str) -> bool:
    feats = _features_of(features, uid)
    return bool(feats is not None and len(getattr(feats, "echo_times", []) or []) >= 2)


def _diffusion(features: Mapping, uid: str) -> bool:
    """A series acquired with diffusion encoding (its reference gets a
    ``.bval``/``.bvec`` of its own)."""
    feats = _features_of(features, uid)
    if feats is None:
        return False
    # GE stores a b-value of 0 on a plain EPI too: only a positive one counts.
    return any((b or 0) > 0 for b in (getattr(feats, "b_values", None) or [])) or \
        "DIFFUSION" in (getattr(feats, "image_type", []) or [])


def _echoes(features: Mapping, uid: str) -> int:
    feats = _features_of(features, uid)
    return len(getattr(feats, "echo_times", []) or []) if feats is not None else 1


def _metadata_file(options: PlanOptions, raw: str) -> Optional[Path]:
    """A file the dataset metadata names, resolved the way the metadata step
    resolves it (relative to the inventory)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute() and options.inventory_dir:
        path = Path(options.inventory_dir) / path
    return path


def plan_output(df, included: Optional[Sequence[bool]] = None,
                options: PlanOptions = PlanOptions(),
                features: Optional[Mapping] = None, spec=None,
                first_files: Optional[Mapping[str, str]] = None) -> OutputPlan:
    """Every file the conversion and the metadata step will write.

    ``spec`` is the dataset metadata (phenotype tables, a participants file);
    ``first_files`` the first DICOM of each series, by uid, which names its
    ``sourcedata/`` folder (read only when that copy is on).
    """
    flags = _included(df, included)
    by_path: dict[str, PlannedFile] = {}
    members: list[tuple[tuple[str, str], int, list]] = []
    sessions: dict[tuple[str, str], list[int]] = defaultdict(list)
    subjects: dict[str, list[int]] = defaultdict(list)
    keep = set(options.sourcedata_modalities or ())

    def put(f: PlannedFile) -> None:
        # A later file at the same path replaces the earlier one: an events
        # table the user attached is written over MNE-BIDS'.
        by_path[f.path] = f

    for pos, i in enumerate(df.index):
        if not flags[pos]:
            continue
        name, datatype = _cell(df, i, "bids_name"), _cell(df, i, "datatype")
        if not name or not datatype or datatype.startswith("derivatives"):
            continue
        uid = _cell(df, i, "series_uid")
        files = _files_for_row(df, i, _two_echoes(features or {}, uid), options.force_edf,
                               _diffusion(features or {}, uid), _echoes(features or {}, uid))
        for f in files + _attached(df, i, files):
            put(f)
        subject = _cell(df, i, "participant_id")
        session = _cell(df, i, "session")
        members.append(((subject, session), int(i), files))
        subjects[subject].append(int(i))
        if options.sourcedata in ("copy", "archive"):
            from .sourcedata import kind_for, series_name

            kind = kind_for(_cell(df, i, "modality"), datatype, name.rsplit("_", 1)[-1])
            if not keep or kind in keep:
                ses = f"/{session}" if session else ""
                if uid:
                    first = (first_files or {}).get(uid.split("|", 1)[0], "")
                    series = series_name(SimpleNamespace(
                        source_files=[first] if first else [], basename=name))
                    dest = f"sourcedata/{subject}{ses}/{kind}/{series}"
                    dest += ".tar.gz" if options.sourcedata == "archive" else "/"
                else:
                    src = _cell(df, i, "source_file").replace("\\", "/").rsplit("/", 1)[-1]
                    dest = f"sourcedata/{subject}{ses}/{kind}/{src}"
                    dest += ".tar.gz" if options.sourcedata == "archive" else ""
                put(PlannedFile(dest, "BIDS Manager (sourcedata)", (int(i),),
                                role="archive" if dest.endswith(".tar.gz") else "data"))

    # Which rows are rows of a scans table: decided per folder, over every
    # file planned there, by the metadata step's own rule.
    listed = _listed([f for _key, _i, files in members for f in files])
    for key, i, files in members:
        if any(f.path in listed for f in files):
            sessions[key].append(i)

    for (subject, session), rows in sessions.items():
        folder = _folder(subject, session, "")
        label = f"{subject}_{session}" if session else subject
        # MNE-BIDS writes the scans table of a session with EEG or MEG during
        # the conversion; for anything else the metadata step writes it.
        mne = any(_cell(df, r, "datatype") in _MNE_DATATYPES
                  and not _cell(df, r, "series_uid") for r in rows)
        if mne or (options.run_metadata and options.tables):
            put(PlannedFile(f"{folder}/{label}_scans.tsv",
                            "MNE-BIDS" if mne else "BIDS Manager (metadata)",
                            tuple(rows), role="table"))
    # A subject with sessions gets a sessions table from the metadata step.
    if options.run_metadata and options.tables:
        with_sessions: dict[str, list[int]] = defaultdict(list)
        for (subject, session), rows in sessions.items():
            if session:
                with_sessions[subject].extend(rows)
        for subject, rows in with_sessions.items():
            put(PlannedFile(f"{subject}/{subject}_sessions.tsv",
                            "BIDS Manager (metadata)", tuple(rows), role="table"))
    if options.sourcedata in ("copy", "archive"):
        for subject, rows in subjects.items():
            put(PlannedFile(f"sourcedata/{subject}/manifest.tsv",
                            "BIDS Manager (sourcedata)", tuple(rows), role="table"))

    put(PlannedFile("dataset_description.json", "BIDS Manager", role="dataset"))
    put(PlannedFile(".bidsignore", "BIDS Manager", role="text"))
    if options.run_metadata:
        meta_origin = "BIDS Manager (metadata)"
        put(PlannedFile("participants.tsv", meta_origin, role="dataset"))
        put(PlannedFile("participants.json", meta_origin, role="dataset"))
        put(PlannedFile("README", meta_origin, role="text",
                        conditional="unless the dataset already has one"))
        put(PlannedFile("CHANGES", meta_origin, role="text",
                        conditional="unless the dataset already has one"))
        if spec is not None and getattr(spec, "phenotype_files", None):
            from ..metadata.phenotype import measure_name

            for raw in spec.phenotype_files:
                path = _metadata_file(options, raw)
                if path is None:
                    continue
                measure = measure_name(path.stem)
                put(PlannedFile(f"phenotype/{measure}.tsv", meta_origin, role="table"))
                put(PlannedFile(f"phenotype/{measure}.json", meta_origin, role="sidecar"))
        if options.citation:
            put(PlannedFile("CITATION.cff", "BIDS Manager (citation)", role="text",
                            conditional="unless the dataset already has one"))

    plan = OutputPlan(list(by_path.values()))
    if options.run_metadata and options.companions:
        _add_companion_stubs(plan)
    plan.files.sort(key=lambda f: _sort_key(f.path))
    return plan


def _add_companion_stubs(plan: OutputPlan) -> None:
    """The companion files the metadata step generates: what the standard
    associates with a recording and nobody else writes, by the rule the
    step itself applies (:func:`~bidsmgr.fixups.associations.missing_for`)."""
    from ..fixups.associations import missing_for

    def in_tree(f: PlannedFile) -> bool:
        return f.path.startswith("sub-") and not f.path.endswith("/")

    certain = [f.path for f in plan.files if not f.conditional and in_tree(f)]
    maybe = {f.path: f.conditional for f in plan.files if f.conditional and in_tree(f)}
    index = {f.path: n for n, f in enumerate(plan.files)}
    rows_of = {f.path: f.rows for f in plan.files}
    try:
        missing = missing_for(certain)
        # A recording written only sometimes (the second eye) has its
        # companions only then.
        missing += [m for m in missing_for(certain + list(maybe))
                    if m[2] in maybe and m[0] not in index]
    except Exception:  # noqa: BLE001 - a schema question failed: plan without
        return
    for target, kind, source, _derivable in missing:
        if target in index and source in maybe:
            continue
        if source in maybe:
            plan.files.append(PlannedFile(
                target, "BIDS Manager (companion files)", rows_of.get(source, ()),
                role="sidecar" if kind == "sidecar" else "table", stub=kind,
                conditional=maybe[source]))
            index[target] = len(plan.files) - 1
            continue
        if target in index:
            # A converter writes it only sometimes (MNE-BIDS' events table):
            # when it does not, the stub does, so the file is always there.
            f = plan.files[index[target]]
            plan.files[index[target]] = replace(
                f, conditional="", stub=kind,
                origin=f"{f.origin}, or a TODO stub from BIDS Manager when it writes none")
            continue
        plan.files.append(PlannedFile(
            target, "BIDS Manager (companion files)", rows_of.get(source, ()),
            role="sidecar" if kind == "sidecar" else "table", stub=kind))
        index[target] = len(plan.files) - 1


def _listed(files) -> set[str]:
    """The planned paths that are rows of ``*_scans.tsv``: the metadata
    step's own rule (``metadata.tables.list_recordings``), folder by folder,
    so the two cannot drift."""
    from ..metadata.tables import list_recordings

    by_folder: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if f.conditional or f.warning:
            continue
        folder, _sep, name = f.path.rstrip("/").rpartition("/")
        by_folder[folder].append(name)
    out: set[str] = set()
    for folder, names in by_folder.items():
        datatype = folder.rsplit("/", 1)[-1]
        entries = [(n, n.endswith(_DIRECTORY_RECORDINGS)) for n in names]
        out.update(f"{folder}/{n}" for n in list_recordings(datatype, entries))
    return out


#: Extensions of recordings that are folders (CTF, MEF, OME-Zarr).
_DIRECTORY_RECORDINGS = (".ds", ".mefd", ".ome.zarr")


def _sort_key(path: str) -> tuple:
    parts = path.split("/")
    top = parts[0]
    rank = 0 if len(parts) == 1 else (2 if top == "sourcedata" else 1)
    return (rank, [tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", part))
                   for part in parts])


# ---------------------------------------------------------------------------
# What a file will contain
# ---------------------------------------------------------------------------


class _Fields:
    """A predicted JSON object, each key with where its value comes from."""

    def __init__(self) -> None:
        self.data: dict[str, object] = {}
        self.source: dict[str, str] = {}

    def put(self, key: str, value, source: str) -> None:
        self.data[key] = value
        self.source[key] = source

    def apply(self, step, source: str) -> None:
        """Run ``step`` on the object; whatever it set or changed is ``source``."""
        before = {k: _dump(v) for k, v in self.data.items()}
        step(self.data)
        for key, value in self.data.items():
            if before.get(key) != _dump(value):
                self.source[key] = source
        for key in [k for k in self.source if k not in self.data]:
            del self.source[key]

    def preview(self, note: str = "") -> FilePreview:
        return FilePreview("json", fields=tuple(
            (k, v, self.source.get(k, "converter")) for k, v in self.data.items()), note=note)


def _dump(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def approximate(df, planned: PlannedFile, options: PlanOptions = PlanOptions(),
                features: Optional[Mapping] = None, spec=None,
                included: Optional[Sequence[bool]] = None) -> FilePreview:
    """``planned``'s predicted content (see :class:`FilePreview`)."""
    path = planned.path
    name = path.rsplit("/", 1)[-1]
    flags = _included(df, included)
    if path.endswith("/"):
        return FilePreview("folder", note="A copy of the series' DICOM files, as they are.")
    if path == "dataset_description.json":
        return _description(options, spec).preview(
            note="Authors moves to CITATION.cff when the citation file is written."
            if options.run_metadata and options.citation else "")
    if path == "CITATION.cff":
        return _citation(options, spec)
    if path == "participants.tsv":
        return _participants_tsv(df, flags, options, spec)
    if path == "participants.json":
        return _participants_json(df, flags, options, spec)
    if path == "README":
        return FilePreview("text", text=_readme_text(options.dataset))
    if path == ".bidsignore":
        try:
            from ..cli._scaffold import BIDSIGNORE_LINES

            lines = list(BIDSIGNORE_LINES)
        except Exception:  # noqa: BLE001
            lines = [".bidsmgr/", ".tmp_bidsmgr/"]
        return FilePreview("text", text="\n".join(lines) + "\n",
                           note="BIDS Manager's own folders, so a validator skips them.")
    if path == "CHANGES":
        return FilePreview("text", text="1.0.0 <the date of the metadata step>\n  - Initial release.\n")
    if path.startswith("phenotype/"):
        return FilePreview("tsv" if name.endswith(".tsv") else "json",
                           note="The measure table you set, with participant_id first, and a "
                                "description of its columns.")
    if name.endswith("_scans.tsv"):
        return _scans_tsv(df, planned, features, options)
    if name.endswith("_sessions.tsv"):
        return _sessions_tsv(df, planned)
    if name == "manifest.tsv":
        return FilePreview("tsv", header=("source", "destination", "size", "sha256"),
                           note="One row per source file kept, with its checksum.")
    if planned.stub and "MNE-BIDS" not in planned.origin:
        return _stub(planned, options, spec, df)
    if not planned.rows:
        return FilePreview("binary")
    row = planned.rows[0]
    if name.endswith(".json"):
        return _sidecar(df, row, planned, options, features, spec)
    if name.endswith("_channels.tsv"):
        n = _cell(df, row, "n_channels")
        return FilePreview(
            "tsv", header=("name", "type", "units", "low_cutoff", "high_cutoff", "description",
                           "sampling_frequency", "status", "status_description"),
            note=f"One row per channel{f' ({n} channels)' if n else ''}, with the types the "
                 "scan found and the ones you set.")
    if name.endswith("_events.tsv"):
        return _events_tsv(df, row, spec, planned)
    if name.endswith("_electrodes.tsv"):
        return FilePreview("tsv", header=("name", "x", "y", "z", "size"),
                           note="One row per electrode, from the montage or digitised positions.")
    if name.endswith((".bval", ".bvec")):
        feats = _features_of(features, _cell(df, row, "series_uid"))
        bvals = getattr(feats, "b_values", []) if feats else []
        n = getattr(feats, "n_directions", 0) if feats else 0
        what = ("b-values" if name.endswith(".bval") else "gradient directions (three rows: x, y, z)")
        return FilePreview("text", text=f"{UNKNOWN}\n",
                           note=f"The {what} of every volume, written by dcm2niix."
                                + (f" The scan found b = {', '.join(f'{b:g}' for b in bvals)}"
                                   f"{f' and {n} directions' if n else ''}." if bvals else ""))
    if name.endswith("_physioevents.tsv.gz"):
        return FilePreview("tsv", header=("onset", "duration", "trial_type", "message"),
                           note="Fixations, saccades, blinks and the tracker's messages.")
    if name.endswith("_blood.tsv"):
        return FilePreview("tsv", note="The blood curves you attached, one column per measure, "
                                       "converted by pet2bids.")
    if planned.origin.startswith("BIDS Manager (your file"):
        return FilePreview("binary", note="Your file, copied as it is.")
    if name.endswith("_physio.tsv.gz") or name.endswith(".tsv.gz"):
        return FilePreview("tsv", note="Samples, one column per signal, without a header "
                                       "(the columns are named in the JSON).")
    return FilePreview("binary", note=f"Written by {planned.origin}.")


def _levels(options: PlanOptions) -> Optional[frozenset]:
    """The requirement levels the metadata step writes placeholders for, or
    ``None`` when there is no metadata step at all."""
    if not options.run_metadata:
        return None
    from ..metadata.engine import scope_levels

    return scope_levels(options.fill_scope)


@lru_cache(maxsize=1)
def _fresh_dataset() -> str:
    """A dataset with nothing in it yet, for the levels that depend on one.

    ``Authors`` is recommended only while a dataset has no CITATION.cff, and
    at the metadata step a new dataset has none; asking with no dataset at all
    leaves such levels undecided. Kept for the life of the process.
    """
    import atexit
    import shutil
    import tempfile

    folder = tempfile.mkdtemp(prefix="bidsmgr-preview-")
    atexit.register(shutil.rmtree, folder, True)
    Path(folder, "dataset_description.json").write_text(
        json.dumps({"Name": "preview", "BIDSVersion": "1.10.0", "DatasetType": "raw"}),
        encoding="utf-8")
    return folder


def _root(options: PlanOptions) -> Path:
    if options.root and Path(options.root, "dataset_description.json").is_file():
        return Path(options.root)
    return Path(_fresh_dataset())


def _description(options: PlanOptions, spec) -> _Fields:
    """``dataset_description.json`` after the conversion, the metadata step
    and the citation file, each by its own function."""
    out = _Fields()
    # The conversion's scaffold (cli/_scaffold.ensure_dataset_description).
    out.put("Name", options.dataset or UNKNOWN, "settings")
    out.put("BIDSVersion", options.bids_version or UNKNOWN, "settings")
    out.put("DatasetType", "raw", "settings")
    out.put("GeneratedBy", [{
        "Name": "bidsmgr", "Version": options.version or UNKNOWN,
        "Description": "dcm2niix-direct backend",
        "Container": {"Type": "binary", "Tag": options.dcm2niix_version or UNKNOWN},
    }], "settings")
    if not options.run_metadata:
        return out
    from ..metadata.engine import describe_dataset, mark_dataset_description, \
        stated_dataset_fields
    from ..metadata.types import DatasetMetadata

    stated = stated_dataset_fields(spec) if spec is not None else {}
    meta = DatasetMetadata(**stated)
    if meta.name in (None, "", "Untitled BIDS Dataset"):
        meta = meta.model_copy(update={"name": options.dataset or UNKNOWN})
    extras = dict(getattr(getattr(spec, "dataset_description", None), "extra", None) or {})
    merged = describe_dataset(dict(out.data), meta, "bidsmgr", extras)

    def step(data: dict) -> None:
        data.clear()
        data.update(merged)

    out.apply(step, "stated")
    for key in ("Name", "BIDSVersion", "DatasetType", "GeneratedBy"):
        if key in out.data and key not in stated and key != "Name":
            out.source[key] = "settings"
    if "name" not in stated:
        out.source["Name"] = "settings"
    levels = _levels(options)
    root = _root(options)
    out.apply(lambda data: mark_dataset_description(
        data, levels, bids_root=root, citation=(root / "CITATION.cff").is_file()), "todo")
    if options.citation:
        # The citation file takes Authors out of the description (stating it
        # in both is an error); the rest it copies and leaves.
        out.data.pop("Authors", None)
        out.source.pop("Authors", None)
    return out


def _citation(options: PlanOptions, spec) -> FilePreview:
    from ..fixups.citation import build_citation

    description = _description(replace(options, citation=False), spec).data
    try:
        text = build_citation(description)
    except Exception as exc:  # noqa: BLE001 - a preview of one file
        return FilePreview("text", note=f"Written from the dataset description ({exc}).")
    return FilePreview("text", text=text,
                       note="Written from the dataset description, after the metadata step.")


def _subjects(df, flags) -> list[str]:
    seen: list[str] = []
    for pos, i in enumerate(df.index):
        if flags[pos] and _cell(df, i, "bids_name"):
            s = _cell(df, i, "participant_id")
            if s and s not in seen:
                seen.append(s)
    return sorted(seen, key=lambda s: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)])


def _participants(df, flags, options: PlanOptions, spec) -> tuple[list[str], list[dict], dict]:
    """``participants.tsv`` as the metadata step writes a new one: the
    inventory's demographics, the participants file over them, columns
    nobody answered left out."""
    import pandas as pd

    from ..metadata.demographics import load_participants_table, merge_demographics
    from ..metadata.engine import demographics_by_participant, participant_rows
    from ..metadata.phenotype import load_sidecar_dictionary

    table = df.copy()
    for col in table.columns:
        table[col] = table[col].map(lambda v: "" if v is None or (isinstance(v, float) and v != v)
                                    else str(v))
    lookup = demographics_by_participant(table)
    codebook: dict = {}
    source = _metadata_file(options, getattr(spec, "participants_file", "") or "")
    if source is not None and source.is_file():
        try:
            lookup = merge_demographics(lookup, load_participants_table(source))
            codebook = load_sidecar_dictionary(source)
        except Exception:  # noqa: BLE001 - the file is the user's; show what we can
            pass
    columns, rows = participant_rows(_subjects(df, flags), lookup)
    frame = pd.DataFrame(rows, columns=columns)
    keep = ["participant_id"] + [c for c in columns[1:] if (frame[c].astype(str) != "n/a").any()]
    return keep, [{c: r[c] for c in keep} for r in rows], codebook


def _participants_tsv(df, flags, options: PlanOptions, spec) -> FilePreview:
    columns, rows, _codebook = _participants(df, flags, options, spec)
    note = ("Age, sex and handedness from the DICOM headers or the participants file you set; "
            "a column nobody answered is left out.")
    if {"given_name", "family_name", "patient_id"} & set(columns):
        note += (" The names and patient IDs come from the DICOM headers: they identify the "
                 "participants, so remove them before sharing the dataset.")
    return FilePreview("tsv", header=tuple(columns),
                       rows=tuple(tuple(r[c] for c in columns) for r in rows), note=note)


def _participants_json(df, flags, options: PlanOptions, spec) -> FilePreview:
    from ..metadata.engine import participants_json

    columns, _rows, codebook = _participants(df, flags, options, spec)
    return FilePreview("json", fields=tuple(
        (c, value, "stated" if c in codebook else "settings")
        for c, value in participants_json(columns, codebook).items()))


def _readme_text(dataset: str) -> str:
    try:
        from ..metadata.engine import readme_text

        return readme_text(dataset or "<dataset>")
    except Exception:  # noqa: BLE001
        return f"# {dataset}\n"


def _iso_time(df, i) -> str:
    """The recording's time as the scans table writes it: MNE-BIDS in UTC
    with microseconds and a Z, the metadata step as dcm2niix states it."""
    date = re.sub(r"\D", "", _cell(df, i, "study_date"))
    time = _cell(df, i, "acq_time")
    rec = _cell(df, i, "recording_time")
    if rec and re.match(r"\d{4}-\d{2}-\d{2}T", rec):
        stamp = rec[:26].split("+")[0].rstrip("Z")
        if _cell(df, i, "datatype") in _MNE_DATATYPES and not _cell(df, i, "series_uid"):
            whole, _dot, frac = stamp.partition(".")
            return f"{whole}.{(frac + '000000')[:6]}Z"
        return stamp
    if len(date) >= 8 and time:
        whole, _dot, frac = time.replace(":", "").partition(".")
        whole = re.sub(r"\D", "", whole)
        if len(whole) >= 6:
            stamp = f"{date[:4]}-{date[4:6]}-{date[6:8]}T{whole[:2]}:{whole[2:4]}:{whole[4:6]}"
            return stamp + (f".{(frac + '000000')[:6]}" if frac.isdigit() else "")
    # Not read by the scan: the converter takes it from the recording.
    return UNKNOWN


def _scans_tsv(df, planned: PlannedFile, features, options: PlanOptions) -> FilePreview:
    """``filename`` relative to the session folder, and the time."""
    by_row = {}
    for i in planned.rows:
        uid = _cell(df, i, "series_uid")
        by_row[i] = _files_for_row(df, i, _two_echoes(features or {}, uid), options.force_edf,
                                   _diffusion(features or {}, uid), _echoes(features or {}, uid))
    # As the metadata step lists them, by its own rule: the recordings, not
    # their sidecars, gradient tables or side files.
    listed = _listed([f for files in by_row.values() for f in files])
    rows = []
    for i, files in by_row.items():
        datatype = _cell(df, i, "datatype")
        for f in files:
            if f.path in listed:
                rows.append((f"{datatype}/{f.path.rsplit('/', 1)[-1]}", _iso_time(df, i)))
    rows.sort()
    return FilePreview("tsv", header=("filename", "acq_time"), rows=tuple(rows),
                       note="One row per recording of the session, its time from the scan. "
                            "For a series of many images dcm2niix writes the first volume's "
                            "time, which can be a second or two apart.")


def _sessions_tsv(df, planned: PlannedFile) -> FilePreview:
    """``session_id`` and the session's earliest recording."""
    earliest: dict[str, str] = {}
    for i in planned.rows:
        session = _cell(df, i, "session")
        stamp = _iso_time(df, i)
        if not session:
            continue
        if stamp == UNKNOWN:
            earliest.setdefault(session, UNKNOWN)
        elif earliest.get(session, UNKNOWN) == UNKNOWN or stamp < earliest[session]:
            earliest[session] = stamp
    rows = tuple(sorted(earliest.items()))
    return FilePreview("tsv", header=("session_id", "acq_time"), rows=rows,
                       note="One row per session, its time the earliest recording's.")


#: The columns MNE-BIDS describes in an events.json, in its words.
_MNE_EVENT_COLUMNS = {
    "onset": {"Description": "Onset (in seconds) of the event from the beginning of the first "
                             "datapoint. Negative onsets account for events before the first "
                             "stored data point.", "Units": "s"},
    "duration": {"Description": "Duration of the event in seconds from onset. Must be zero, "
                                "positive, or 'n/a' if unavailable. A zero value indicates an "
                                "impulse event. ", "Units": "s"},
    "sample": {"Description": "The event onset time in number of sampling points.First sample "
                              "is 0."},
    "value": {"Description": "The event code (also known as trigger code or event ID) "
                             "associated with the event."},
}


def _event_map(df, row, spec) -> dict:
    """The labels the user gave a recording's event codes: its task's map,
    else the one for every task (the enrichment's own resolution)."""
    if spec is None:
        return {}
    from ..recording_meta import resolve_effective

    task = _split_name(_cell(df, row, "bids_name")).get("task") or None
    try:
        eff = resolve_effective(spec, _row_id(df, row), task, _cell(df, row, "datatype"))
    except Exception:  # noqa: BLE001 - a malformed spec shows no labels
        return {}
    # A code with no label is relabelled by nothing.
    return {code: label for code, label in (eff.event_map or {}).items() if str(label).strip()}


def _events_tsv(df, row, spec, planned: PlannedFile) -> FilePreview:
    from ..fixups.associations import events_stub

    header = ("onset", "duration", "trial_type", "value", "sample")
    if planned.origin.startswith("BIDS Manager (your file"):
        return FilePreview("tsv", note="Your events table, copied as it is: it replaces the one "
                                       "MNE-BIDS writes from the recording.")
    labels = [] if planned.renamed else list(_event_map(df, row, spec).items())
    rows = tuple((UNKNOWN, UNKNOWN, str(label), str(code), UNKNOWN) for code, label in labels[:6])
    note = ("One row per event or annotation in the recording, with the labels you gave "
            "event codes." if labels else "One row per event or annotation in the recording.")
    if planned.stub:
        note += (" When the recording has none, the companion setting writes a stub instead: "
                 + " | ".join(events_stub().strip().splitlines()) + ".")
    return FilePreview("tsv", header=header, rows=rows, note=note)


def _stub(planned: PlannedFile, options: PlanOptions, spec, df) -> FilePreview:
    """A companion file the metadata step generates, as it generates it."""
    from ..fixups.associations import events_stub, sidecar_stub

    note = ("Generated because the standard associates it with a recording and nothing else "
            "writes it. Validation reports each TODO until it is filled in.")
    if planned.stub == "events":
        lines = events_stub().strip().splitlines()
        return FilePreview("tsv", header=tuple(lines[0].split("\t")),
                           rows=tuple(tuple(line.split("\t")) for line in lines[1:]),
                           note=note + " A TODO onset is deliberately invalid: an empty table "
                                       "would read as a recording with no events.")
    if planned.stub == "channels":
        return FilePreview("tsv", header=("name", "type", "units"),
                           note="Read from the recording: one row per channel with its type "
                                "and units, or a TODO row when the recording cannot be read.")
    parts = planned.path.split("/")
    datatype = parts[-2] if len(parts) >= 2 else ""
    stem = parts[-1][: -len(".json")]
    suffix = stem.rsplit("_", 1)[-1]
    fields = _Fields()
    try:
        data = json.loads(sidecar_stub(datatype, suffix))
    except Exception:  # noqa: BLE001
        data = {"Description": "TODO"}
    for key, value in data.items():
        fields.put(key, value, "todo")
    return fields.preview(note=note)


def _seconds(ms) -> Optional[float]:
    try:
        return round(float(ms) / 1000.0, 6)
    except (TypeError, ValueError):
        return None


def _pe_direction(feats) -> Optional[str]:
    axis = getattr(feats, "pe_axis", "")
    if axis not in ("ROW", "COL"):
        return None
    letter = "i" if axis == "ROW" else "j"
    positive = getattr(feats, "pe_positive", None)
    if positive is None:
        return None
    return letter if positive else f"{letter}-"


def _links(df, row, options: PlanOptions, features) -> list[tuple[str, object, str]]:
    """IntendedFor / B0FieldIdentifier / B0FieldSource as the conversion
    writes them: which runs by its rule (``plan_links``), the identifiers in
    its order (``group_key``, :func:`fieldmap_identifier`), and the opposite
    run's SBRef joining a one-direction EPI reference."""
    from ..fixups.intended_for import fieldmap_identifier, group_key, plan_links

    shims = {}
    for uid, feats in (features or {}).items():
        if getattr(feats, "shim", None):
            shims[uid] = feats.shim
    try:
        links, _why = plan_links(df, shims=shims)
    except Exception:  # noqa: BLE001
        return []

    def where(i) -> tuple[str, str]:
        return (_cell(df, i, "participant_id"), _cell(df, i, "session"))

    def included(i) -> bool:
        return _cell(df, i, "include") not in ("0", "false", "False")

    # Every fieldmap acquisition of a session, in the conversion's order,
    # those that correct no run included: that is how it counts.
    keys: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i in df.index:
        if included(i) and _cell(df, i, "datatype") == "fmap":
            key = group_key(_cell(df, i, "bids_name"))
            if key not in keys[where(i)]:
                keys[where(i)].append(key)

    def ident(fm) -> str:
        session = _cell(df, fm, "session").replace("ses-", "")
        ordered = sorted(keys[where(fm)])
        key = group_key(_cell(df, fm, "bids_name"))
        return fieldmap_identifier(session, ordered.index(key) + 1 if key in ordered else 1)

    out: list[tuple[str, object, str]] = []
    name = _cell(df, row, "bids_name")
    is_phase_part = bool(re.search(r"_part-(phase|real|imag)", name))
    if row in links and links[row] and not is_phase_part:
        subject = _cell(df, row, "participant_id")
        names = []
        for t in links[row]:
            datatype, target = _cell(df, t, "datatype"), _cell(df, t, "bids_name")
            session = _cell(df, t, "session")
            parts = ([session] if session else []) + [datatype, f"{target}.nii.gz"]
            rel = "/".join(parts)
            names.append(rel if options.intended_for == "relative" else f"bids::{subject}/{rel}")
        out.append(("IntendedFor", names, "settings"))
        if options.b0_field:
            out.append(("B0FieldIdentifier", ident(row), "settings"))
    elif options.b0_field:
        sources = sorted({ident(fm) for fm, targets in links.items() if row in targets
                          and not re.search(r"_part-(phase|real|imag)",
                                            _cell(df, fm, "bids_name"))})
        if sources:
            out.append(("B0FieldSource", sources[0] if len(sources) == 1 else sources, "settings"))
        partner_of = _partners(df, links, features)
        if row in partner_of:
            out.append(("B0FieldIdentifier", ident(partner_of[row]), "settings"))
    return out


def _partners(df, links: Mapping[int, list[int]], features) -> dict[int, int]:
    """``{SBRef or run row: fieldmap row}`` for each EPI reference of one
    phase-encoding direction: the nearest of its runs phase-encoded the other
    way gives it the second direction (its SBRef when it has one)."""
    def pe(i):
        feats = _features_of(features, _cell(df, i, "series_uid"))
        if feats is None or getattr(feats, "pe_axis", "") not in ("ROW", "COL"):
            return None
        if getattr(feats, "pe_positive", None) is None:
            return None
        return getattr(feats, "pe_axis"), bool(getattr(feats, "pe_positive"))

    def seconds(i) -> Optional[float]:
        from ..inventory._time import parse_time_seconds

        return parse_time_seconds(_cell(df, i, "acq_time"))

    out: dict[int, int] = {}
    for fm, targets in links.items():
        if not _cell(df, fm, "bids_name").endswith("_epi") or not targets:
            continue
        mine = pe(fm)
        if mine is None:
            continue
        when = seconds(fm)
        best = None
        for run in targets:
            other = pe(run)
            if other is None or other != (mine[0], not mine[1]):
                continue
            t = seconds(run)
            gap = abs(t - when) if t is not None and when is not None else float("inf")
            if best is None or gap < best[0]:
                best = (gap, run)
        if best is None:
            continue
        run = best[1]
        sbref = _cell(df, run, "bids_name").rsplit("_", 1)[0] + "_sbref"
        partner = next((i for i in df.index if _cell(df, i, "bids_name") == sbref
                        and _cell(df, i, "include") not in ("0", "false", "False")), run)
        out[int(partner)] = int(fm)
    return out


#: How dcm2niix names a manufacturer, by what the DICOM header says.
_MANUFACTURERS = (("SIEMENS", "Siemens"), ("GE ", "GE"), ("GE_", "GE"), ("GEMS", "GE"),
                  ("PHILIPS", "Philips"),
                  ("TOSHIBA", "Toshiba"), ("CANON", "Canon"), ("UIH", "UIH"),
                  ("UNITED IMAGING", "UIH"), ("HITACHI", "Hitachi"), ("BRUKER", "Bruker"))


def _manufacturer(raw: str) -> str:
    upper = f"{raw.upper()} "
    for token, name in _MANUFACTURERS:
        if upper.startswith(token) or token.strip() == upper.strip():
            return name
    return raw


def _mri_fields(df, row, feats, out: _Fields, suffix: str = "") -> None:
    """What dcm2niix writes from the header, in its own forms: the
    manufacturer by its short name, multi-valued codes joined by a backslash,
    a factor of 1 left out."""
    def add(key, value, source="scan"):
        if value not in (None, "", [], ()):
            out.put(key, value, source)

    add("Modality", "MR")
    add("MagneticFieldStrength", getattr(feats, "field_strength", None))
    add("Manufacturer", _manufacturer(getattr(feats, "manufacturer", "") or ""))
    add("ManufacturersModelName", getattr(feats, "model", ""))
    out.put("SoftwareVersions", UNKNOWN, "converter")
    add("MRAcquisitionType", getattr(feats, "acquisition_type", ""))
    add("SeriesDescription", getattr(feats, "description", "") or _cell(df, row, "sequence"))
    add("ProtocolName", getattr(feats, "protocol", ""))
    scanning = [x for x, on in (("EP", getattr(feats, "is_epi", False)),
                                ("GR", getattr(feats, "is_gre", False)),
                                ("SE", getattr(feats, "is_se", False)),
                                ("IR", getattr(feats, "is_ir", False))) if on]
    add("ScanningSequence", "\\".join(scanning))
    add("SequenceVariant", "\\".join(getattr(feats, "sequence_variant", []) or []))
    add("ScanOptions", "\\".join(getattr(feats, "scan_options", []) or []))
    add("SequenceName", getattr(feats, "sequence_name", ""))
    # dcm2niix adds its own words (MAGNITUDE, PHASE) to the header's.
    out.put("ImageType", UNKNOWN, "converter")
    add("SliceThickness", getattr(feats, "slice_thickness", None))
    # A fieldmap row writes several files: the first magnitude carries the
    # first echo, the second and the phase difference the second, and the
    # phase difference names both.
    echoes = sorted(getattr(feats, "echo_times", []) or [])
    if echoes:
        add("EchoTime", _seconds(echoes[-1] if suffix in ("magnitude2", "phasediff")
                                 else echoes[0]))
    if len(echoes) >= 2 and suffix == "phasediff":
        add("EchoTime1", _seconds(echoes[0]))
        add("EchoTime2", _seconds(echoes[-1]))
    add("RepetitionTime", getattr(feats, "repetition_time", None))
    add("InversionTime", getattr(feats, "inversion_time", None))
    add("FlipAngle", getattr(feats, "flip_angle", None))
    if (getattr(feats, "echo_train_length", None) or 0) > 1:
        add("EchoTrainLength", feats.echo_train_length)
    if (getattr(feats, "multiband", None) or 0) > 1:
        add("MultibandAccelerationFactor", feats.multiband)
    add("ContrastBolusAgent", getattr(feats, "contrast_agent", ""))
    if getattr(feats, "is_epi", False):
        # Its sign depends on how the vendor stores the polarity; dcm2niix
        # works it out at conversion.
        out.put("PhaseEncodingDirection", UNKNOWN, "converter")
        out.put("EffectiveEchoSpacing", UNKNOWN, "converter")
        out.put("TotalReadoutTime", UNKNOWN, "converter")
    if getattr(feats, "has_slice_timing", False):
        out.put("SliceTiming", UNKNOWN, "converter")
    out.put("ConversionSoftware", "dcm2niix", "settings")


#: Fields the schema declares for a datatype that its converter never
#: writes: MNE-BIDS counts ECoG and sEEG channels only for iEEG.
_NOT_WRITTEN = {"eeg": ("ECOGChannelCount", "SEEGChannelCount"),
                "meg": ("ECOGChannelCount", "SEEGChannelCount")}


def _dropped(value) -> bool:
    """A value the conversion's repair step removes: dcm2niix's literal "None"
    for a field the DICOM does not state."""
    return isinstance(value, str) and value.strip().lower() == "none"


def _converter_answers(df, row, datatype: str, suffix: str, spec, out: _Fields) -> str:
    """The fields the converter writes by itself, beside those predicted.

    Where the scan ran a probe conversion (the default), it recorded what the
    converter wrote for each kind of file: those values are exact and win
    over the scan's own reading of the first file. The header facts of each
    recording fill what is still open. A file the row writes under another
    suffix (a fieldmap's second magnitude) takes the probed kind's fields, not
    its values. Without a probe, the fields a conversion usually fills
    (measured, :mod:`~bidsmgr.metadata.derivable`) are shown as unknown, so a
    field the converter will answer is not predicted a TODO.
    """
    from .. import schema
    from ..recording_meta import VARIES

    def applies(key: str) -> bool:
        try:
            return schema.field_applies(key, datatype, suffix)
        except Exception:  # noqa: BLE001
            return False

    probed = getattr(spec, "converter_preview", {}) or {}
    rows = getattr(spec, "row_preview", {}) or {}
    row_suffix = _cell(df, row, "bids_name").rsplit("_", 1)[-1]
    # dcm2niix's probe output is exact; for a recording MNE-BIDS reads, the
    # preview holds the scan's own estimates, which fill gaps but do not
    # replace what MNE-BIDS computes (its duration from the samples).
    exact = bool(_cell(df, row, "series_uid"))
    own = probed.get(f"{datatype}/{suffix}")
    if own is not None:
        for key, value in own.items():
            if _dropped(value) or not applies(key) or key in _NOT_WRITTEN.get(datatype, ()):
                continue
            if value == VARIES or value in (None, "", [], {}):
                if key not in out.data:
                    out.put(key, UNKNOWN, "converter")
            elif exact or key not in out.data:
                out.put(key, value, "scan")
    if suffix == row_suffix:
        for key, value in (rows.get(_row_id(df, row), {}) or {}).items():
            if key in out.data or _dropped(value) or not applies(key) \
                    or key in _NOT_WRITTEN.get(datatype, ()):
                continue
            out.put(key, UNKNOWN if value == VARIES else value,
                    "converter" if value == VARIES else "scan")
    if own is not None:
        return ""
    sibling = probed.get(f"{datatype}/{row_suffix}")
    if sibling is not None:
        for key in sibling:
            if key not in out.data and applies(key):
                out.put(key, UNKNOWN, "converter")
        return ""
    from ..metadata.derivable import derived_fields

    added = False
    for key in sorted(derived_fields(datatype)):
        if key in out.data or key in ("IntendedFor", "TaskName") or not applies(key):
            continue
        out.put(key, UNKNOWN, "converter")
        added = True
    return ("Fields marked … are what the converter usually reads from the recording; a scan "
            "with a probe conversion knows exactly which.") if added else ""


def _pet_fields(df, row, feats, out: _Fields) -> None:
    """What dcm2niix and the PET fixup write for a PET image, before the
    fixup drops the DICOM keys BIDS does not define."""
    def scan(key, value):
        if value not in (None, "", [], ()):
            out.put(key, value, "scan")

    scan("Manufacturer", getattr(feats, "manufacturer", "") if feats else "")
    scan("ManufacturersModelName", getattr(feats, "model", "") if feats else "")
    out.put("SoftwareVersions", UNKNOWN, "converter")
    out.put("Units", UNKNOWN, "converter")
    tracer = _cell(df, row, "tracer_suggestion")
    out.put("TracerName", tracer or UNKNOWN, "scan" if tracer else "converter")
    nuclide = _cell(df, row, "radionuclide_suggestion")
    out.put("TracerRadionuclide", nuclide or UNKNOWN, "scan" if nuclide else "converter")
    dose = _cell(df, row, "injected_dose_suggestion")
    match = re.match(r"\s*([0-9.]+)\s*(\S+)?", dose)
    if match:
        scan("InjectedRadioactivity", float(match.group(1)))
        scan("InjectedRadioactivityUnits", match.group(2) or "")
    # The PET fixup names the reconstruction method in its own words.
    out.put("ReconMethodName", UNKNOWN, "converter")
    for key in ("FrameTimesStart", "FrameDuration", "ImageDecayCorrected",
                "ImageDecayCorrectionTime", "AttenuationCorrection"):
        out.put(key, UNKNOWN, "converter")
    # TimeZero is derived from SeriesTime, not left to the converter's clock.
    out.put("TimeZero", UNKNOWN, "converter")
    out.put("ScanStart", 0, "settings")
    out.put("ConversionSoftware", "dcm2niix" if _cell(df, row, "series_uid") else "nibabel",
            "settings")


def _eeg_fields(df, row, renamed: bool, spec, out: _Fields) -> None:
    """The keys MNE-BIDS writes, with what it writes when the recording does
    not say: "n/a" where BIDS requires a key it cannot answer."""
    datatype = _cell(df, row, "datatype")
    task = _split_name(_cell(df, row, "bids_name")).get("task", "")
    fif = _ext(_cell(df, row, "source_file")) in (".fif", ".fif.gz")
    if task:
        out.put("TaskName", task, "settings")
    out.put("Manufacturer", UNKNOWN if datatype == "meg" else "n/a", "converter")
    # The line frequency the conversion hands MNE-BIDS (the table's cell, else
    # the dataset metadata); a FIF header carries its own; else "n/a".
    line = _cell(df, row, "line_freq") or _stated_line_freq(df, row, spec)
    try:
        out.put("PowerLineFrequency", float(line), "settings")
    except ValueError:
        out.put("PowerLineFrequency", UNKNOWN if fif else "n/a", "converter")
    sfreq = _cell(df, row, "sfreq")
    out.put("SamplingFrequency", float(sfreq) if sfreq else UNKNOWN,
            "scan" if sfreq else "converter")
    out.put("SoftwareFilters", UNKNOWN if fif else "n/a", "converter")
    out.put("RecordingDuration", UNKNOWN, "converter")
    out.put("RecordingType", "continuous", "converter")
    if datatype in ("eeg", "ieeg"):
        for key in ("EEGReference", "EEGGround", "EEGPlacementScheme"):
            out.put(key, "n/a", "converter")
    if datatype == "meg":
        out.put("DewarPosition", "n/a", "converter")
    # The conversion's enrichment writes the task protocol, on the file it
    # finds by the table's name.
    if spec is not None and task and not renamed:
        from ..recording_meta import resolve_effective

        try:
            proto = resolve_effective(spec, _row_id(df, row), task, datatype).task_protocol
        except Exception:  # noqa: BLE001
            proto = None
        if proto is not None:
            if proto.task_description:
                out.put("TaskDescription", proto.task_description, "stated")
            if proto.instructions:
                out.put("Instructions", proto.instructions, "stated")


def _stated_line_freq(df, row, spec) -> str:
    """The line frequency the dataset metadata gives a recording, if any."""
    if spec is None:
        return ""
    from ..recording_meta import resolve_effective

    try:
        acq = resolve_effective(spec, _row_id(df, row), None, _cell(df, row, "datatype")).acquisition
    except Exception:  # noqa: BLE001
        return ""
    value = getattr(acq, "power_line_freq", None)
    return "" if value in (None, "") else str(value)


def _deface(df, row, datatype: str, suffix: str, options: PlanOptions, features,
            out: _Fields) -> str:
    """The defacing record a face-bearing single-volume image gets."""
    from ..deface.select import FACE_BEARING_DATATYPES

    if not options.deface or datatype not in FACE_BEARING_DATATYPES:
        return ""
    feats = _features_of(features, _cell(df, row, "series_uid"))
    volumes = getattr(feats, "n_volumes", None) if feats is not None else None
    if volumes is not None and volumes > 1:
        return ""
    from ..deface import engines, status

    try:
        engine = engines.engine(options.deface_engine or engines.DEFAULT_ENGINE_ID)
    except KeyError:
        return ""
    out.apply(lambda data: data.update(status.record(data, engine)), "settings")
    return "" if volumes == 1 else "The defacing record is written when the image is one volume."


def _sidecar(df, row, planned: PlannedFile, options: PlanOptions, features, spec) -> FilePreview:
    name = planned.path.rsplit("/", 1)[-1][:-len(".json")]
    parts = planned.path.split("/")
    datatype = parts[-2] if len(parts) >= 2 else _cell(df, row, "datatype")
    suffix = name.rsplit("_", 1)[-1]
    uid = _cell(df, row, "series_uid")
    feats = _features_of(features, uid)
    out = _Fields()
    notes: list[str] = []

    if suffix in ("channels", "events", "coordsystem", "electrodes"):
        if suffix == "events":
            for key, value in _MNE_EVENT_COLUMNS.items():
                out.put(key, dict(value), "converter")
            trial = {"Description": "The type, category, or name of the event."}
            levels = {} if planned.renamed else _event_map(df, row, spec)
            if levels:
                trial["Levels"] = {str(label): f"trigger code {code}"
                                   for code, label in levels.items()}
            out.put("trial_type", trial, "stated" if levels else "converter")
        elif suffix == "coordsystem":
            meg = datatype == "meg"
            out.put("MEGCoordinateSystem" if meg else "EEGCoordinateSystem", UNKNOWN, "converter")
            out.put("MEGCoordinateUnits" if meg else "EEGCoordinateUnits", UNKNOWN, "converter")
    elif suffix in ("physio", "physioevents"):
        if "recording-eye" in name and suffix == "physioevents":
            from .backends.eyelink import EVENTS_SIDECAR

            for key, value in EVENTS_SIDECAR.items():
                out.put(key, value, "settings")
        elif "recording-eye" in name:
            out.put("SamplingFrequency", UNKNOWN, "converter")
            out.put("StartTime", UNKNOWN, "converter")
            out.put("Columns", ["timestamp", "x_coordinate", "y_coordinate", "pupil_size"],
                    "settings")
            out.put("PhysioType", "eyetrack", "settings")
            out.put("RecordedEye", "left" if "eye1" in name else "right", "scan")
            out.put("SampleCoordinateSystem", "gaze-on-screen", "settings")
            out.put("Manufacturer", "SR Research", "settings")
            out.put("ManufacturersModelName", UNKNOWN, "converter")
        else:
            out.put("SamplingFrequency", UNKNOWN, "converter")
            out.put("StartTime", UNKNOWN, "converter")
            out.put("Columns", UNKNOWN, "converter")
            notes.append("One pair of files per signal (recording-<signal>) when they were "
                         "sampled at different rates.")
        # The physio fixup names the task the recording belongs to.
        task = _split_name(name).get("task", "")
        if task:
            out.put("TaskName", task, "settings")
    elif datatype in _MNE_DATATYPES:
        _eeg_fields(df, row, bool(planned.renamed), spec, out)
        _converter_answers(df, row, datatype, suffix, spec, out)
    elif datatype == "pet":
        _pet_fields(df, row, feats, out)
        note = _converter_answers(df, row, datatype, suffix, spec, out)
        if note:
            notes.append(note)
    else:
        if feats is not None:
            _mri_fields(df, row, feats, out, suffix)
        else:
            out.put("Modality", "MR", "scan")
            out.put("SeriesDescription", _cell(df, row, "sequence"), "scan")
        note = _converter_answers(df, row, datatype, suffix, spec, out)
        if note:
            notes.append(note)
        if datatype == "mrs" and out.data.get("InversionTime") == 0:
            # The MRS fixup drops it: a spectroscopy sequence with none.
            out.data.pop("InversionTime")
            out.source.pop("InversionTime", None)
        if datatype in ("fmap", "func", "dwi", "perf"):
            for key, value, source in _links(df, row, options, features):
                out.put(key, value, source)

    if suffix not in ("channels", "events", "coordsystem", "electrodes", "physio", "physioevents"):
        note = _deface(df, row, datatype, suffix, options, features, out)
        if note:
            notes.append(note)
    if datatype == "pet":
        from ..fixups.pet_sidecar import NON_BIDS_KEYS

        out.apply(lambda data: [data.pop(k, None) for k in NON_BIDS_KEYS], "settings")

    # The metadata step: what the user stated, then the placeholders.
    if options.run_metadata:
        from ..fixups.sidecar_schema import state_into, stated_cells
        from ..metadata.engine import mark_sidecar

        if spec is not None:
            cells = stated_cells({c: _cell(df, row, c) for c in df.columns})
            out.apply(lambda data: state_into(data, f"{name}.json", datatype, suffix, spec,
                                              row_id=_row_id(df, row), row_values=cells),
                      "stated")
        levels = _levels(options)
        marks = mark_sidecar(out.data, f"{name}.json", datatype, suffix, levels,
                             bids_root=_root(options))
        for key in marks.derived:
            out.source[key] = "settings"
        for key in marks.todo_added:
            out.source[key] = "todo"
    elif spec is not None:
        notes.append("Your metadata answers reach the files in the metadata step, which is off.")
    if planned.renamed:
        notes.append("The conversion's enrichment finds the recording by the table's name, so "
                     "the event labels and task description you set do not reach this file.")
    return out.preview(note=" ".join(notes))


__all__ = [
    "FilePreview",
    "OutputPlan",
    "PlanOptions",
    "PlannedFile",
    "SOURCES",
    "UNKNOWN",
    "approximate",
    "plan_output",
]

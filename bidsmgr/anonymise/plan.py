"""Anonymise a BIDS dataset: plan every change, then make them as one step.

Works on any BIDS dataset, converted here or not, every datatype: the
carriers are kinds of FILE (a JSON sidecar, a table, a gzip or NIfTI header,
an EDF header, a BrainVision marker file, a FIF recording, a photograph),
not modalities. What it cannot rewrite safely it lists rather than skips
silently, with the reason.

Two passes. The first finds each participant's earliest date, which decides
how far their dates move (one number of whole weeks for all of them, so
intervals survive; remembered in ``.bidsmgr/project/anonymise.json`` so a
later conversion of the same participant moves with it). The second plans
each file. Applying writes through an editor operation when given one (the
Editor: one undoable step), directly otherwise (a conversion).

Running it again changes nothing more: a date already in 1925 or before is
left, a pseudonym is recognised, a removed key stays removed.

Qt-free.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import carriers as c
from .policy import AnonymisePolicy
from .scope import Scope
from .state import STATE_FILE, State, is_shifted, load

#: Not walked into: the tool's own state, and the folders that are not the
#: raw dataset (each listed in the plan as not changed).
_TOOL = {".bidsmgr", ".tmp_bidsmgr", ".git", ".datalad"}
_NOT_RAW = {"sourcedata": "the original files, identity and all: remove it, or keep it "
                          "out of what you share",
            "derivatives": "outputs of other tools: anonymise each derivative dataset "
                           "on its own",
            "code": "scripts, which may name people or paths"}
#: Recordings whose format this cannot rewrite safely, and why.
_CANNOT = {
    ".set": "EEGLAB: the header is a MATLAB file; open it in EEGLAB to clear the "
            "subject and date fields",
    ".fdt": "",
    ".snirf": "SNIRF (HDF5): its metadata are not rewritten here",
    ".nwb": "NWB (HDF5): its metadata are not rewritten here",
    ".ds": "CTF: the header files are not rewritten here",
    ".con": "KIT/Yokogawa: the header is not rewritten here",
    ".sqd": "KIT/Yokogawa: the header is not rewritten here",
    ".mefd": "MEF3: the session metadata are not rewritten here",
    ".ome.zarr": "OME-Zarr: the XML metadata are not rewritten here",
    ".ome.tif": "OME-TIFF: the XML metadata are not rewritten here",
    ".ome.btf": "OME-TIFF: the XML metadata are not rewritten here",
    ".tif": "TIFF: its tags are not rewritten here",
    ".mrk": "", ".pos": "",
}
_DESCRIPTIVE = {"dataset_description.json", "CITATION.cff", "README", "README.md",
                "CHANGES", "LICENSE", ".bidsignore"}
_PHOTO = (".jpg", ".jpeg", ".png")
_EDF = (".edf", ".bdf")
_FIF = (".fif", ".fif.gz")


@dataclass
class FilePlan:
    path: Path
    kind: str
    changes: list[c.Change]
    reviews: list[c.Review] = field(default_factory=list)
    content: Optional[str] = None
    writer: Optional[Callable[[Path, Path], None]] = None


@dataclass
class AnonymisePlan:
    root: Path
    policy: AnonymisePolicy
    state: State
    scope: Scope = field(default_factory=Scope)
    files: list[FilePlan] = field(default_factory=list)
    reviews: list[tuple[Path, c.Review]] = field(default_factory=list)
    cannot: list[tuple[str, str]] = field(default_factory=list)   # (path, why)
    #: Files that could not be rewritten when applied (damaged, unreadable),
    #: and why: left as they were, the rest written.
    failed: list[tuple[str, str]] = field(default_factory=list)
    state_changed: bool = False

    def grouped_reviews(self) -> list[tuple[str, str, str, list[Path]]]:
        """The values to read by hand, once each: ``(what, value, why,
        files)``. The same protocol name in forty sidecars is one question."""
        groups: dict[tuple[str, str], tuple[str, list[Path]]] = {}
        for path, review in self.reviews:
            key = (review.what, review.value)
            groups.setdefault(key, (review.reason, []))[1].append(path)
        return [(what, value, why, paths)
                for (what, value), (why, paths) in sorted(groups.items())]

    @property
    def n_changes(self) -> int:
        return sum(len(f.changes) for f in self.files)

    def by_category(self) -> dict[str, int]:
        """How many changes each category makes."""
        out: dict[str, int] = {}
        for planned in self.files:
            for change in planned.changes:
                out[change.category] = out.get(change.category, 0) + 1
        return out

    def rel(self, path: Path) -> str:
        try:
            return Path(path).relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def summary(self) -> str:
        if not self.files:
            return "Nothing to change: the dataset states nothing this policy removes."
        return (f"{self.n_changes:,} change{'s' if self.n_changes != 1 else ''} in "
                f"{len(self.files):,} file{'s' if len(self.files) != 1 else ''}")


# ---------------------------------------------------------------------------
# Walking
# ---------------------------------------------------------------------------

def _walk(root: Path, plan: AnonymisePlan) -> list[Path]:
    """Every file to consider; folders that are not the raw dataset, and
    folder recordings, noted as not changed."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        keep = []
        for name in sorted(dirnames):
            if name in _TOOL or name.startswith("."):
                continue
            if base == root and name in _NOT_RAW:
                plan.cannot.append((name + "/", _NOT_RAW[name]))
                continue
            lowered = name.lower()
            ext = next((e for e in _CANNOT if lowered.endswith(e)), None)
            if ext and _CANNOT[ext]:
                plan.cannot.append((plan.rel(base / name), _CANNOT[ext]))
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            if not name.startswith("."):
                out.append(base / name)
    return out


def _participant(root: Path, path: Path) -> Optional[str]:
    try:
        first = path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return None
    return first if first.startswith("sub-") else None


def _ext(name: str) -> str:
    lowered = name.lower()
    for double in (".nii.gz", ".tsv.gz", ".fif.gz", ".ome.tif", ".ome.btf"):
        if lowered.endswith(double):
            return double
    return Path(lowered).suffix


def _read_json(path: Path) -> tuple[Optional[dict], str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
        data = json.loads(text)
    except (OSError, ValueError):
        return None, ""
    return (data if isinstance(data, dict) else None), text


def _dump_like(data: dict, original: str) -> str:
    """JSON in the indentation the file had."""
    indent = 2
    for line in original.splitlines()[1:3]:
        stripped = len(line) - len(line.lstrip(" "))
        if stripped:
            indent = stripped
            break
    return json.dumps(data, indent=indent, ensure_ascii=False) + "\n"


def _fif_info(path: Path):
    try:
        import mne

        return mne.io.read_info(path, verbose="error")
    except Exception:  # noqa: BLE001 - an unreadable FIF is reported, not fatal
        return None


# ---------------------------------------------------------------------------
# Pass one: each participant's earliest date
# ---------------------------------------------------------------------------

def _span(root: Path, files: list[Path]) -> dict[str, tuple[_dt.date, _dt.date]]:
    """``{participant: (earliest, latest)}`` over the dates that will be
    SHIFTED: a birth date is removed, not shifted, and must not decide how far
    the rest moves (it did: a 1951 birth date left a 2025 scan in 1976)."""
    from ..metadata.tables import read_table

    found: dict[str, tuple[_dt.date, _dt.date]] = {}

    def note(participant: Optional[str], value) -> None:
        if participant is None or not isinstance(value, str):
            return
        date = c.parse_date(value)
        if date is None or is_shifted(date):
            return
        low, high = found.get(participant, (date, date))
        found[participant] = (min(low, date), max(high, date))

    for path in files:
        participant = _participant(root, path)
        if participant is None:
            continue
        ext = _ext(path.name)
        if ext == ".json":
            data, _text = _read_json(path)
            for key, value in (data or {}).items():
                if key not in c.IDENTITY_KEYS:
                    note(participant, value)
        elif path.name.endswith(("_scans.tsv", "_sessions.tsv")):
            for row in read_table(path)[1]:
                note(participant, row.get("acq_time", ""))
        elif ext == ".vmrk":
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in c._VMRK_SEGMENT.finditer(text):
                note(participant, m.group(2))
        elif ext in _FIF:
            info = _fif_info(path)
            when = info["meas_date"] if info is not None else None
            if when is not None:
                note(participant, when.date().isoformat())
    return found


# ---------------------------------------------------------------------------
# Pass two: each file
# ---------------------------------------------------------------------------

def plan_anonymise(root: Path, policy: AnonymisePolicy, *,
                   scope: Optional[Scope] = None,
                   method: Optional[str] = None) -> AnonymisePlan:
    """Every change ``policy`` makes to the dataset at ``root``, nothing written.

    ``scope`` limits it to the dataset files and the subjects and sessions it
    names (a table listing participants or sessions changes only in their
    rows); ``method`` is the text written into ``DeidentificationMethod`` of
    MRI and PET image sidecars.
    """
    root = Path(root)
    scope = scope or Scope()
    state = load(root)
    plan = AnonymisePlan(root, policy, state, scope)
    known = set(state.offsets)
    every = _walk(root, plan)
    files = every if scope.whole else [f for f in every if scope.has_path(root, f)]
    participants = sorted({p for p in (_participant(root, f) for f in files) if p}
                          | ({p for p in _listed_participants(root)
                              if scope.has_subject(p)} if scope.top else set()))
    # A participant's offset is chosen from ALL their dates, whatever part
    # of them is anonymised now: a later run over the rest moves with it.
    spans = _span(root, [f for f in every if _participant(root, f) in set(participants)])
    days: dict[str, int] = {}
    for participant in participants:
        low, high = spans.get(participant, (None, None))
        days[participant] = state.days_back(participant, low, high)
    plan.state_changed = set(state.offsets) != known
    method = method or _method(policy)

    for path in files:
        participant = _participant(root, path)
        back = days.get(participant, 0) if participant else 0
        planned = _plan_file(root, path, policy, state, back, method, plan)
        if planned is not None and planned.changes:
            plan.files.append(planned)
        if planned is not None:
            plan.reviews.extend((path, r) for r in planned.reviews)
    for participant in days:
        label = participant[len("sub-"):]
        if any(ch.isalpha() for ch in label) and sum(ch.isalpha() for ch in label) >= 3 \
                and label != "emptyroom":
            plan.reviews.append((root / participant, c.Review(
                "participant label", participant,
                "a label made of letters can be a name or an ID from the source")))
    return plan


def _listed_participants(root: Path) -> list[str]:
    """The participants participants.tsv lists (rows without a folder too)."""
    from ..metadata.tables import read_table

    table = root / "participants.tsv"
    if not table.is_file():
        return []
    return [r.get("participant_id", "") for r in read_table(table)[1]
            if r.get("participant_id", "").startswith("sub-")]


def _method(policy: AnonymisePolicy) -> str:
    import bidsmgr

    version = getattr(bidsmgr, "__version__", "")
    return (f"BIDS Manager {version} anonymisation ({policy.name}): "
            + "; ".join(line[:1].lower() + line[1:] for line in policy.describe()[:3]))


def _plan_file(root: Path, path: Path, policy: AnonymisePolicy, state: State, back: int,
               method: str, plan: AnonymisePlan) -> Optional[FilePlan]:
    from ..metadata.tables import read_table, table_text

    name = path.name
    ext = _ext(name)
    if path.parent == root and name in _DESCRIPTIVE:
        return None
    if ext in _CANNOT:
        why = _CANNOT[ext]
        if why:
            plan.cannot.append((plan.rel(path), why))
        return None
    if ext == ".json":
        data, text = _read_json(path)
        if data is None:
            return None
        if path.parent == root and name == "participants.json":
            return _plan_participants_json(root, path, data, text, policy, plan.scope)
        image = any((path.parent / (name[:-5] + e)).exists() for e in (".nii.gz", ".nii"))
        new, changes, reviews = c.clean_json(
            data, policy, state, back, method=method if image else None)
        return FilePlan(path, "json", changes, reviews,
                        content=_dump_like(new, text) if changes else None)
    if path.parent == root and name == "participants.tsv":
        header, rows = read_table(path)
        scope = plan.scope
        header, rows, changes = c.clean_participants(
            header, rows, policy,
            in_scope=None if scope.subjects is None else scope.has_subject)
        return FilePlan(path, "tsv", changes,
                        content=table_text(header, rows) if changes else None)
    if name.endswith(("_scans.tsv", "_sessions.tsv")):
        header, rows = read_table(path)
        if "acq_time" not in header:
            return None
        in_scope = None
        if name.endswith("_sessions.tsv") and plan.scope.sessions is not None:
            subject = path.parent.name
            in_scope = (lambda row, s=subject:
                        plan.scope.has_session(s, row.get("session_id", "")))
        rows, changes = c.clean_acq_times(rows, policy, back, in_scope)
        return FilePlan(path, "tsv", changes,
                        content=table_text(header, rows) if changes else None)
    if ext == ".vmrk":
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        new, changes = c.clean_vmrk(text, policy, back)
        return FilePlan(path, "text", changes, content=new if changes else None)
    if ext in _EDF:
        header, changes = c.edf_header_changes(path, policy)
        if header is None:
            return None
        return FilePlan(path, "edf", changes,
                        writer=lambda src, dst, h=header: c.write_edf_header(src, dst, h))
    if ext in (".nii.gz", ".nii"):
        changes = []
        descrip = c.nifti_descrip_change(path) if policy.times == "midnight" else None
        if descrip:
            changes.append(c.Change("NIfTI header text", descrip[0].decode("latin-1"),
                                    descrip[1].decode("latin-1"), "a time of day", "dates"))
        gz = c.gzip_header_change(path) if ext == ".nii.gz" and policy.traces else None
        if gz:
            changes.append(gz)
        if not changes:
            return None
        writer = c.write_clean_nifti if descrip else c.write_clean_gzip
        return FilePlan(path, "nifti", changes, writer=writer)
    if ext == ".tsv.gz" and policy.traces:
        gz = c.gzip_header_change(path)
        return FilePlan(path, "gzip", [gz], writer=c.write_clean_gzip) if gz else None
    if ext in _FIF:
        return _plan_fif(path, policy, back, plan)
    if ext in _PHOTO and policy.photos:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        cleaned = c.clean_photo(data)
        if cleaned is None:
            return None
        return FilePlan(path, "photo", [c.Change("photo metadata", "EXIF / text blocks", "",
                                                 "can hold a date, a place, a device",
                                                 "photos")],
                        writer=lambda src, dst, b=cleaned: Path(dst).write_bytes(b))
    return None


def _plan_participants_json(root: Path, path: Path, data: dict, text: str,
                            policy: AnonymisePolicy, scope: Scope) -> FilePlan:
    """The descriptions of the columns participants.tsv loses: only when it
    loses them (Names and IDs on, every row in scope)."""
    whole = scope.subjects is None or set(_listed_participants(root)) <= set(scope.subjects)
    gone = ([k for k in data if k.lower() in c.IDENTITY_COLUMNS]
            if policy.identity and whole else [])
    changes = [c.Change(f"{k} (description)", "described", "", "its column is removed",
                        "identity") for k in gone]
    new = {k: v for k, v in data.items() if k not in gone}
    return FilePlan(path, "json", changes, content=_dump_like(new, text) if changes else None)


def _plan_fif(path: Path, policy: AnonymisePolicy, back: int,
              plan: AnonymisePlan) -> Optional[FilePlan]:
    if not policy.identity:
        # MNE anonymises a FIF as a whole: names, devices and dates together.
        plan.cannot.append((plan.rel(path), "a FIF recording is anonymised by MNE as a "
                                            "whole, names and dates together: turn on "
                                            "Names and IDs to anonymise it"))
        return None
    if "_split-" in path.name:
        plan.cannot.append((plan.rel(path), "a FIF split across files: anonymise it with "
                                            "MNE before it is split"))
        return None
    info = _fif_info(path)
    if info is None:
        plan.cannot.append((plan.rel(path), "this FIF could not be read"))
        return None
    changes = []
    when = info["meas_date"]
    shift = policy.dates == "shift" and back and when is not None \
        and not is_shifted(when.date())
    if shift or (when is not None and policy.times == "midnight"
                 and (when.hour, when.minute, when.second) != (0, 0, 0)):
        moved = when - _dt.timedelta(days=back if shift else 0)
        if policy.times == "midnight":
            moved = moved.replace(hour=0, minute=0, second=0, microsecond=0)
        changes.append(c.Change("measurement date", when.isoformat(), moved.isoformat(),
                                "a date", "dates"))
    subject = info.get("subject_info") or {}
    personal = {k: v for k, v in subject.items()
                if k in ("first_name", "middle_name", "last_name", "birthday", "his_id",
                         "weight", "height") and not _fif_anonymous(v)}
    if personal:
        changes.append(c.Change("subject_info", ", ".join(sorted(personal)), "",
                                "names the participant", "identity"))
    if info.get("experimenter") and not _fif_anonymous(info["experimenter"]):
        changes.append(c.Change("experimenter", str(info["experimenter"]), "",
                                "names a person", "identity"))
    if not changes:
        return None
    midnight = policy.times == "midnight"
    days = back if policy.dates == "shift" else 0
    return FilePlan(path, "fif", changes,
                    writer=lambda src, dst, d=days, m=midnight: _write_fif(src, dst, d, m))


def _fif_anonymous(value) -> bool:
    """Empty, or what MNE's own anonymisation writes in its place: the word
    ``mne_anonymize``, an id of 0, a birthday moved back with the dates."""
    if value in (None, "", 0, "0", "mne_anonymize"):
        return True
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.year <= 1925
    return False


def _write_fif(src: Path, dst: Path, days_back: int, midnight: bool) -> None:
    """The recording through MNE's own anonymisation, saved in FIF again.
    ``days_back`` 0 keeps its dates (MNE moves them by exactly that)."""
    import mne

    raw = mne.io.read_raw_fif(src, preload=False, allow_maxshield=True, verbose="error")
    when = raw.info["meas_date"]
    # None would move the date to 2000-01-01 (MNE's default): a date already
    # shifted, or kept, is moved by 0 days instead.
    if when is None:
        daysback = None
    elif is_shifted(when.date()) or not days_back:
        daysback = 0
    else:
        daysback = days_back
    raw.anonymize(daysback=daysback, keep_his=False, verbose="error")
    if midnight and raw.info["meas_date"] is not None:
        raw.set_meas_date(raw.info["meas_date"].replace(hour=0, minute=0, second=0,
                                                         microsecond=0))
    raw.save(dst, overwrite=True, verbose="error")


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def apply_anonymise(plan: AnonymisePlan, *, op=None,
                    only: Optional[set[str]] = None) -> list[Path]:
    """Make ``plan``'s changes. Through ``op`` (an editor operation: one
    undoable step) when given, directly otherwise. ``only`` limits it to those
    files (paths relative to the dataset root). Returns the files written."""
    written: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="bidsmgr-anon-") as tmp:
        for i, planned in enumerate(plan.files):
            if only is not None and plan.rel(planned.path) not in only:
                continue
            if planned.content is not None:
                if op is not None:
                    op.write_text(planned.path, planned.content)
                else:
                    _replace_text(planned.path, planned.content)
            elif planned.writer is not None:
                # Same name in its own folder: MNE reads the kind of FIF from it.
                staging = Path(tmp) / str(i)
                staging.mkdir()
                dst = staging / planned.path.name
                try:
                    planned.writer(planned.path, dst)
                except Exception as exc:  # noqa: BLE001 - one damaged file, not all
                    plan.failed.append((plan.rel(planned.path), str(exc) or type(exc).__name__))
                    continue
                if op is not None:
                    op.replace_from(planned.path, dst)
                else:
                    shutil.copystat(planned.path, dst)
                    os.replace(dst, planned.path)
            else:
                continue
            written.append(planned.path)
    if written or plan.state_changed:
        target = plan.root / STATE_FILE
        if op is not None:
            op.write_text(target, plan.state.to_json())
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            _replace_text(target, plan.state.to_json())
    return written


def _replace_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.anon.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def anonymise(root: Path, policy: AnonymisePolicy, *, op=None,
              scope: Optional[Scope] = None) -> AnonymisePlan:
    """Plan and apply in one call (a conversion's post step, the CLI)."""
    plan = plan_anonymise(root, policy, scope=scope)
    apply_anonymise(plan, op=op)
    return plan


__all__ = ["AnonymisePlan", "FilePlan", "anonymise", "apply_anonymise", "plan_anonymise"]

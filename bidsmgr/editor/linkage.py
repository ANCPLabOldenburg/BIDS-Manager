"""The links between files, and the six sidecar fields that carry them.

BIDS has six fields whose value is not data but a POINTER at another file in
the same dataset. BIDS Manager wrote one of them automatically and left the
other five to be typed by hand into a JSON editor:

==========================  ===========================================
``IntendedFor``             the runs a fieldmap corrects
``B0FieldIdentifier``       the newer pairing mechanism, by name
``B0FieldSource``           the other half of it
``AssociatedEmptyRoom``     the MEG empty-room recording
``Sources`` / ``RawSources``  what a derivative was made from
``AnatomicalImage``         the anatomical an MRS voxel sits on
==========================  ===========================================

Treating them one at a time is what produced that gap, so this module treats
them as one kind of thing. A link has a source file, a field, and a list of
targets; the operations are read them, check them, propose them and write
them. Which fields a given file may carry is a question for the schema, and
which files a field may point AT is a question the dataset answers.

Two properties that are easy to lose and worth stating:

**A link is written as a ``bids::`` URI.** Not a relative path. The URI form
is the one ``rename.PATH_FIELDS`` already follows, so a link written here
survives a later rename, a session being added, or a subject being fused,
without this module knowing any of that happened.

**A broken link is invisible.** The validator checks ``IntendedFor`` and says
nothing about the other five, so a pointer at a file somebody deleted last
month sits there looking fine. :func:`broken_links` is the only thing in the
tool that will tell you.

Qt-free: the dialog, the CLI and the tests all drive this.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .. import schema as schema_mod
from ..project.operations import begin_operation
from .rename import PATH_FIELDS, walk_dataset

log = logging.getLogger(__name__)


# The fields, in the order a dialog should offer them: the one people come
# for first, then the rest. ``RawSources`` is the deprecated spelling of
# ``Sources`` and is listed so an existing dataset that uses it can be read
# and repaired, not so anybody writes a new one.
LINK_FIELDS: tuple[str, ...] = (
    "IntendedFor",
    "AssociatedEmptyRoom",
    "Sources",
    "RawSources",
    "AnatomicalImage",
)

# What each field is FOR, in one sentence, for the dialog.
FIELD_HELP: dict[str, str] = {
    "IntendedFor": (
        "The images this fieldmap can correct. Written on the fieldmap, "
        "pointing at the functional or diffusion runs."
    ),
    "AssociatedEmptyRoom": (
        "The empty-room recording measured for this MEG session, used to "
        "estimate the sensor noise."
    ),
    "Sources": (
        "The files this one was made from. Written on a derivative, "
        "pointing back at the raw data."
    ),
    "RawSources": (
        "The deprecated spelling of Sources. Shown so an existing dataset "
        "can be read and repaired; new links should use Sources."
    ),
    "AnatomicalImage": (
        "The anatomical image this spectroscopy voxel was placed on."
    ),
}

# Which datatypes a field is worth offering on. The schema knows which
# sidecars declare each field, but a link is a product decision as much as a
# schema one: offering ``Sources`` on every file in the dataset is true and
# useless. ``None`` means every datatype.
_FIELD_DATATYPES: dict[str, Optional[frozenset[str]]] = {
    "IntendedFor": frozenset({"fmap", "perf", "micr"}),
    "AssociatedEmptyRoom": frozenset({"meg"}),
    "Sources": None,
    "RawSources": None,
    "AnatomicalImage": frozenset({"mrs"}),
}

# What each field may point AT. Same reasoning: the standard is broad and a
# useful candidate list is narrow.
_TARGET_DATATYPES: dict[str, Optional[frozenset[str]]] = {
    "IntendedFor": frozenset({"func", "dwi", "perf", "asl"}),
    "AssociatedEmptyRoom": frozenset({"meg"}),
    "Sources": None,
    "RawSources": None,
    "AnatomicalImage": frozenset({"anat"}),
}

_IMAGE_EXTS = (".nii.gz", ".nii")
_RECORDING_EXTS = _IMAGE_EXTS + (
    ".fif", ".fif.gz", ".edf", ".bdf", ".set", ".vhdr", ".con", ".sqd", ".ds",
)


# --------------------------------------------------------------------------
# Reading


@dataclass(frozen=True)
class Link:
    """One field on one file, and where it points."""

    source: Path            # the .json holding the field
    field: str
    targets: tuple[str, ...]        # as written, usually bids:: URIs
    resolved: tuple[Optional[Path], ...]   # None where nothing is there

    @property
    def broken(self) -> tuple[str, ...]:
        return tuple(
            t for t, r in zip(self.targets, self.resolved) if r is None
        )


def sidecar_for(path: Path) -> Path:
    """The editable ``.json`` for a data file, or the file if it is one."""
    name = path.name
    if name.endswith(".json"):
        return path
    for ext in _RECORDING_EXTS:
        if name.endswith(ext):
            return path.with_name(name[: -len(ext)] + ".json")
    return path.with_suffix(".json")


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _datatype_of(root: Path, path: Path) -> Optional[str]:
    """The datatype folder a file sits in, or ``None`` outside one."""
    try:
        parts = path.resolve().relative_to(Path(root).resolve()).parts
    except ValueError:
        return None
    known = set(schema_mod.list_datatypes())
    for part in reversed(parts[:-1]):
        if part in known:
            return part
    return None


def fields_for(root: Path, path: Path) -> list[str]:
    """The link fields worth offering on ``path``.

    Filtered by datatype rather than by what the schema permits, because the
    standard permits ``Sources`` almost everywhere and a list of six on every
    file teaches the reader nothing.
    """
    datatype = _datatype_of(root, path)
    out = []
    for name in LINK_FIELDS:
        allowed = _FIELD_DATATYPES.get(name)
        if allowed is None or (datatype and datatype in allowed):
            out.append(name)
    return out


def resolve(root: Path, target: str) -> Optional[Path]:
    """The file a written target names, or ``None`` if nothing is there.

    Understands the three spellings a dataset may hold: a ``bids::`` URI, a
    named-scheme ``bids:rawdata:`` URI, and the legacy subject-relative path
    that ``IntendedFor`` used before URIs existed.
    """
    root = Path(root)
    text = str(target).strip()
    if not text:
        return None

    body = text
    if body.startswith("bids:"):
        body = body.split(":", 2)[-1]
        candidate = root / body
        return candidate if candidate.is_file() else None

    # Legacy: relative to the SUBJECT folder, so it has to be tried under
    # each one. Cheap, because it stops at the first hit.
    candidate = root / body
    if candidate.is_file():
        return candidate
    for subject in sorted(root.glob("sub-*")):
        candidate = subject / body
        if candidate.is_file():
            return candidate
    return None


def read_links(root: Path, path: Path) -> list[Link]:
    """Every link currently written on ``path``."""
    root = Path(root)
    sidecar = sidecar_for(Path(path))
    data = _load(sidecar)
    out: list[Link] = []
    for name in LINK_FIELDS:
        if name not in data:
            continue
        raw = data[name]
        targets = tuple(
            str(v) for v in (raw if isinstance(raw, list) else [raw]) if str(v)
        )
        out.append(Link(
            source=sidecar,
            field=name,
            targets=targets,
            resolved=tuple(resolve(root, t) for t in targets),
        ))
    return out


def to_uri(root: Path, target: Path) -> str:
    """``bids::`` URI for a file, which is how a link is written."""
    rel = Path(target).resolve().relative_to(Path(root).resolve())
    return "bids::" + rel.as_posix()


def data_file_for(sidecar: Path) -> Path:
    """The recording a ``.json`` describes, or the sidecar if there is none.

    The reverse of :func:`sidecar_for`, and the one a person wants to SEE: a
    link belongs to ``sub-001_phasediff.nii.gz``, and saying so is clearer
    than naming the ``.json`` that happens to store it. Falls back to the
    sidecar for a JSON that describes no single file (an inherited one at the
    subject level, say), because naming a file that is not there would be
    worse than naming the sidecar.
    """
    sidecar = Path(sidecar)
    if not sidecar.name.endswith(".json"):
        return sidecar
    stem = sidecar.name[: -len(".json")]
    for ext in _RECORDING_EXTS:
        candidate = sidecar.with_name(stem + ext)
        if candidate.exists():
            return candidate
    return sidecar


# --------------------------------------------------------------------------
# What carries a field, and how it is doing

# The states a link can be in. Short on purpose: they are read in a column,
# and a sentence in a column is a sentence nobody reads.
OK = "ok"
BROKEN = "points at a missing file"
UNSET = "not set"
DIFFERS = "differs from the times"
IMPLIED = "the times imply one"


@dataclass(frozen=True)
class SourceRow:
    """One file, one field, and how that field is currently doing."""

    path: Path                  # the recording, for display
    sidecar: Path               # where the field is actually written
    field: str
    targets: tuple[str, ...]            # as written
    resolved: tuple[Optional[Path], ...]
    proposed: Optional[tuple[Path, ...]]   # what the acquisition times imply
    reason: str = ""

    @property
    def status(self) -> str:
        if any(r is None for r in self.resolved):
            return BROKEN
        if not self.targets:
            return IMPLIED if self.proposed else UNSET
        if self.proposed is not None:
            written = {str(r) for r in self.resolved if r is not None}
            if written != {str(p) for p in self.proposed}:
                return DIFFERS
        return OK

    @property
    def live(self) -> tuple[Path, ...]:
        """The targets that are actually there."""
        return tuple(r for r in self.resolved if r is not None)


def restricted_to(field: str) -> Optional[frozenset[str]]:
    """The datatypes ``field`` is worth offering on, or ``None`` for any.

    ``None`` is the interesting answer: it means the standard allows the
    field almost anywhere, so a dialog listing every file that COULD carry
    it would list the whole dataset.
    """
    return _FIELD_DATATYPES.get(field)


def may_carry(root: Path, path: Path, field: str) -> bool:
    """Whether ``field`` is worth offering on ``path``."""
    allowed = _FIELD_DATATYPES.get(field)
    if allowed is None:
        return True
    return _datatype_of(Path(root), Path(path)) in allowed


def sources(
    root: Path, field: str, *, prefix: str = "", only_set: bool = False,
) -> list[SourceRow]:
    """Every file in ``prefix`` that may carry ``field``, with its state.

    ``prefix`` is a dataset-relative path (``sub-001``, ``sub-001/ses-pre``)
    or empty for the whole dataset, matching ``values.Scope``.

    ``only_set`` restricts the answer to files that already carry the field.
    That is the right default for ``Sources``, which the standard allows
    almost anywhere: a list of every recording in the dataset is true and
    unusable. For a field the standard confines to one datatype, the full
    list is short and showing the files with NOTHING set is the point.
    """
    root = Path(root)
    out: list[SourceRow] = []
    # One proposal pass per (subject, session), not per file: the rule is
    # computed for a whole folder at a time, so asking it once per fieldmap
    # walked the same directory forty times.
    cache: dict[tuple, list] = {}
    for path in walk_dataset(root):
        if not path.name.endswith(_RECORDING_EXTS):
            continue
        try:
            key = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if prefix and not key.startswith(prefix + "/"):
            continue
        if not may_carry(root, path, field):
            continue

        sidecar = sidecar_for(path)
        written: tuple[str, ...] = ()
        for link in read_links(root, sidecar):
            if link.field == field:
                written = link.targets
                break
        if only_set and not written:
            continue

        proposal = _cached_proposal(root, path, field, cache)
        out.append(SourceRow(
            path=path,
            sidecar=sidecar,
            field=field,
            targets=written,
            resolved=tuple(resolve(root, t) for t in written),
            proposed=tuple(proposal.targets) if proposal else None,
            reason=proposal.reason if proposal else "",
        ))
    return sorted(out, key=lambda r: r.path)


def _cached_proposal(
    root: Path, path: Path, field: str, cache: dict,
) -> Optional["Proposal"]:
    """:func:`propose`, with the per-folder sweep done once.

    The rule is evaluated for a whole (subject, session) at a time and
    returns a pair per fieldmap acquisition, so the answer for every
    fieldmap in a folder comes out of one call.
    """
    if field != "IntendedFor":
        return None
    from ..fixups.intended_for import suggest_intended_for

    sidecar = sidecar_for(Path(path))
    subject = _subject_of(root, sidecar)
    session = _session_of(root, sidecar)
    if not subject:
        return None

    key = (subject, session)
    if key not in cache:
        scope = root / subject
        if session:
            scope = scope / session
        pairs, why = suggest_intended_for(
            scope, subject[len("sub-"):],
            session[len("ses-"):] if session else None,
        )
        cache[key] = [(pairs, why)]
    pairs, why = cache[key][0]

    for members, uris in pairs:
        if sidecar in members:
            resolved = [resolve(root, u) for u in uris]
            return Proposal(
                targets=tuple(p for p in resolved if p is not None),
                reason=why or "",
            )
    return None


def incoming(root: Path) -> dict[Path, list[tuple[Path, str]]]:
    """For every file, what points AT it, and through which field.

    The reverse direction, and the half of the question nothing else here
    answers: not "what does this fieldmap correct" but "is this run
    corrected by anything at all". Built in one sweep because it cannot be
    answered from a single file.
    """
    root = Path(root)
    out: dict[Path, list[tuple[Path, str]]] = {}
    for path in walk_dataset(root):
        if not path.name.endswith(".json"):
            continue
        for link in read_links(root, path):
            for target in link.resolved:
                if target is not None:
                    out.setdefault(target, []).append(
                        (data_file_for(path), link.field)
                    )
    return out


# --------------------------------------------------------------------------
# Candidates and proposals


def candidates(root: Path, path: Path, field: str) -> list[Path]:
    """The files in this dataset that ``field`` could legally point at.

    Scoped to the source file's own subject and session. A fieldmap does not
    correct another participant's run, and an empty-room recording belongs to
    the session it was measured in, so offering the whole dataset would be a
    list nobody can read with the wrong answers in it.
    """
    root = Path(root)
    path = Path(path)
    allowed = _TARGET_DATATYPES.get(field)
    subject = _subject_of(root, path)
    session = _session_of(root, path)

    out: list[Path] = []
    for candidate in walk_dataset(root):
        if not candidate.name.endswith(_RECORDING_EXTS):
            continue
        if candidate == path or candidate == sidecar_for(path):
            continue
        if _subject_of(root, candidate) != subject:
            continue
        if session and _session_of(root, candidate) != session:
            continue
        datatype = _datatype_of(root, candidate)
        if allowed is not None and datatype not in allowed:
            continue
        out.append(candidate)
    return sorted(out)


def _subject_of(root: Path, path: Path) -> Optional[str]:
    return _first_part(root, path, "sub-")


def _session_of(root: Path, path: Path) -> Optional[str]:
    return _first_part(root, path, "ses-")


def _first_part(root: Path, path: Path, prefix: str) -> Optional[str]:
    try:
        parts = Path(path).resolve().relative_to(Path(root).resolve()).parts
    except ValueError:
        return None
    for part in parts:
        if part.startswith(prefix):
            return part
    return None


@dataclass(frozen=True)
class Proposal:
    """What the tool thinks a link should say, and why."""

    targets: tuple[Path, ...]
    reason: str


def propose(root: Path, path: Path, field: str) -> Optional[Proposal]:
    """The link the tool would write, with the rule stated in words.

    Only ``IntendedFor`` has a rule worth applying automatically, and it is
    the SAME rule the conversion applies: a fieldmap covers the runs acquired
    after it and before the next one. It is imported rather than restated, so
    the proposal and the conversion cannot disagree.

    The reason is returned with it on purpose. "Fieldmap 2 covers runs 2 and
    3 because it was acquired at 09:54" is how somebody notices that the rule
    is wrong for their protocol, which is the whole point of having a tool
    rather than a fixup.
    """
    if field != "IntendedFor":
        return None

    from ..fixups.intended_for import suggest_intended_for

    root = Path(root)
    sidecar = sidecar_for(Path(path))
    subject = _subject_of(root, sidecar)
    session = _session_of(root, sidecar)
    if not subject:
        return None

    scope = root / subject
    if session:
        scope = scope / session

    pairs, why = suggest_intended_for(
        scope, subject[len("sub-"):],
        session[len("ses-"):] if session else None,
    )
    for members, uris in pairs:
        if sidecar in members:
            resolved = [resolve(root, u) for u in uris]
            return Proposal(
                targets=tuple(p for p in resolved if p is not None),
                reason=why or "",
            )
    return None


# --------------------------------------------------------------------------
# Writing


def plan_write(
    root: Path, path: Path, field: str, targets: Iterable[Path],
) -> tuple[Path, list[str]]:
    """The sidecar that would change, and the value it would take."""
    root = Path(root)
    sidecar = sidecar_for(Path(path))
    uris = [to_uri(root, Path(t)) for t in targets]
    return sidecar, uris


def apply_links(
    root: Path, edits: Iterable[tuple[Path, str, list[str]]], *,
    label: Optional[str] = None,
) -> int:
    """Write ``(sidecar, field, uris)`` triples. Returns files changed.

    One operation for the whole batch, so a linkage pass over forty
    fieldmaps is one entry in the history and one undo.

    An empty list REMOVES the key rather than writing ``[]``. An empty list
    is a claim that the file points at nothing, which is a different and
    wronger statement than not saying.
    """
    root = Path(root)
    edits = list(edits)
    if not edits:
        return 0

    changed = 0
    with begin_operation(root, label or f"Link {len(edits)} file(s)") as op:
        for sidecar, field, uris in edits:
            data = _load(sidecar)
            before = data.get(field)
            if uris:
                data[field] = list(uris)
            else:
                data.pop(field, None)
            if data.get(field) == before and (field in data) == (before is not None):
                continue
            op.write_json(sidecar, data)
            changed += 1
    return changed


# --------------------------------------------------------------------------
# Dataset-wide


@dataclass
class BrokenLink:
    """A pointer at a file that is not there."""

    source: Path
    field: str
    target: str


def broken_links(root: Path) -> list[BrokenLink]:
    """Every link in the dataset that resolves to nothing.

    The validator reports this for none of the five fields other than
    ``IntendedFor``, so without this pass a pointer left behind by a manual
    delete is invisible until somebody's pipeline fails on it.
    """
    root = Path(root)
    out: list[BrokenLink] = []
    for path in walk_dataset(root):
        if not path.name.endswith(".json"):
            continue
        for link in read_links(root, path):
            for target in link.broken:
                out.append(BrokenLink(path, link.field, target))
    return out


#: How each field reads as a rule, for a dialog that has to state one.
FIELD_RULE: dict[str, str] = {
    "IntendedFor": (
        "Written on a fieldmap, pointing at the functional or diffusion runs "
        "in the SAME subject and session that it can correct."
    ),
    "AssociatedEmptyRoom": (
        "Written on an MEG recording, pointing at the empty-room measurement "
        "from the same session."
    ),
    "Sources": (
        "Written on a derivative, pointing back at the files it was made "
        "from."
    ),
    "RawSources": (
        "The deprecated spelling of Sources. Read and repaired here; new "
        "links should use Sources."
    ),
    "AnatomicalImage": (
        "Written on a spectroscopy recording, pointing at the anatomical "
        "image its voxel was placed on."
    ),
}


__all__ = [
    "BROKEN",
    "BrokenLink",
    "DIFFERS",
    "FIELD_HELP",
    "FIELD_RULE",
    "IMPLIED",
    "LINK_FIELDS",
    "Link",
    "OK",
    "PATH_FIELDS",
    "Proposal",
    "SourceRow",
    "UNSET",
    "apply_links",
    "broken_links",
    "candidates",
    "data_file_for",
    "fields_for",
    "incoming",
    "may_carry",
    "plan_write",
    "propose",
    "read_links",
    "resolve",
    "restricted_to",
    "sidecar_for",
    "sources",
    "to_uri",
]

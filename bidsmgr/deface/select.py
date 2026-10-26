"""Which images in a dataset can be defaced, and why the rest cannot.

The second half matters as much as the first. A user who is told "4 files will
be defaced" and not told that three more were skipped has been handed a dataset
with a face in it and no reason to suspect it. So every image this walks past
comes back either as a candidate or as a skip with a reason, and the dialog
shows both.

What is a candidate:

* a NIfTI image of the head (not a mask, a segmentation or a defacemask),
* in a datatype that can show a face,
* that is a single 3-D volume,
* under a subject, not under ``sourcedata/`` or a dot folder;
* under ``derivatives/<pipeline>/sub-*`` only when the caller includes
  derivatives: a preprocessed T1w carries the face of the scan it came from,
  so a dataset shared with its derivatives needs them defaced too.

Qt-free, and it reads nothing but headers and sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from .engines import Engine
from .probe import NotNifti, read_dimensions
from .status import defaced_by_others, defaced_by_us, read_sidecar, sidecar_for

# Datatypes whose images can contain a face.
#
# anat is the obvious one. pet is here because a PET/MR study converts its MR
# half as ordinary anat, but a PET image reconstructed on a CT-derived
# attenuation map can carry facial structure of its own, and a user who asked
# to remove faces should be offered it rather than have us decide.
#
# func, dwi and fmap are NOT here. They contain a face too, in principle, but
# they are 4-D or low-resolution, an affine defacer does nothing useful to
# them, and defacing an EPI is not how anybody removes a face from a study.
FACE_BEARING_DATATYPES = frozenset({"anat", "pet"})

# Folders whose contents are never defaced in place.
#
# sourcedata holds what fed the conversion, code and stimuli are not images
# of anybody. Dot folders are tool state, ours included. ``derivatives`` is
# its own case: excluded unless the caller asks for it (``include_derivatives``),
# because a derivative defaced without its source, or the other way round, is
# a choice the user should make knowingly.
EXCLUDED_TOP_LEVEL = frozenset({"sourcedata", "code", "stimuli"})
DERIVATIVES = "derivatives"

# Suffixes that a datatype allows but that are not an image of the head: a
# defacemask is the mask itself, the rest are not NIfTI images at all. A
# derivative folder adds its own (mask, dseg, probseg), which the schema's raw
# suffix list does not name, so they are refused by not being on it.
NOT_HEAD_IMAGES = frozenset({
    "defacemask", "physio", "physioevents", "stim", "blood", "events",
})

NIFTI_SUFFIXES = (".nii.gz", ".nii")


class Skip(str, Enum):
    """Why an image is not a candidate. The value is shown to a user."""

    FOUR_D = "not a single 3-D volume"
    UNREADABLE = "not a readable NIfTI-1 image"
    DATATYPE = "not a datatype that shows a face"
    LOCATION = "not raw subject data"
    DERIVATIVE = "in derivatives/ (Include derivatives to do it too)"
    NOT_HEAD = "not an image of the head (a mask, a segmentation, a map)"
    STRIPPED = "already skull stripped: no face left"
    ALREADY = "already defaced by BIDS Manager"


@dataclass(frozen=True)
class Candidate:
    """An image that can be defaced."""

    path: Path
    relative: str           # POSIX, dataset-relative. For display and keys.
    datatype: str
    shape: tuple[int, ...]
    size_bytes: int
    # The derivative pipeline the image belongs to (``fmriprep``), or ``""``
    # for raw data. A dialog groups by it; the skull strip names its output
    # after it.
    pipeline: str = ""
    # Set when another tool has recorded deidentification on this image. Not a
    # refusal: they may have done something different, or something partial.
    foreign_methods: tuple[str, ...] = ()
    # Set when WE defaced it before. Re-defacing is allowed and starts from
    # what is on disk, so the dialog has to say so.
    previous_engine: Optional[str] = None


@dataclass(frozen=True)
class Skipped:
    """An image that will not be defaced, and why."""

    path: Path
    relative: str
    reason: Skip
    detail: str = ""


@dataclass(frozen=True)
class Selection:
    """The answer to "what would defacing this do?"."""

    candidates: tuple[Candidate, ...]
    skipped: tuple[Skipped, ...]

    @property
    def total_bytes(self) -> int:
        return sum(c.size_bytes for c in self.candidates)

    def __bool__(self) -> bool:
        return bool(self.candidates)


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in NIFTI_SUFFIXES)


def _relative(root: Path, path: Path) -> str:
    """POSIX, always. It becomes a dict key and a line in a dialog.

    See CROSS_PLATFORM_RULES 1.1: a path that goes into data or into a key is
    POSIX on every platform, or a Windows run keys on ``sub-01\\anat\\...``
    and nothing matches.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _datatype_of(root: Path, path: Path) -> str:
    """The datatype folder an image sits in, or ``""``."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return ""
    # <sub>/[<ses>/]<datatype>/<file>, also inside derivatives/<pipeline>/
    return parts[-2] if len(parts) >= 2 else ""


def pipeline_of(root: Path, path: Path) -> str:
    """``fmriprep`` for ``derivatives/fmriprep/...``; ``""`` for raw data."""
    try:
        parts = Path(path).relative_to(root).parts
    except ValueError:
        return ""
    if len(parts) >= 2 and parts[0] == DERIVATIVES:
        return parts[1]
    return ""


def _suffix_of(name: str) -> str:
    for ext in NIFTI_SUFFIXES:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return name.rsplit("_", 1)[-1] if "_" in name else ""


def _head_suffixes(datatype: str) -> frozenset[str]:
    """The suffixes of ``datatype`` that name an image of the head."""
    from .. import schema

    try:
        known = schema.list_suffixes(datatype)
    except Exception:  # noqa: BLE001 - an unknown datatype has none
        known = []
    return frozenset(s for s in known if s not in NOT_HEAD_IMAGES)


def _location(root: Path, path: Path, include_derivatives: bool) -> Optional[Skip]:
    """Why the place an image sits in rules it out, or ``None``."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return Skip.LOCATION
    if not parts:
        return Skip.LOCATION
    if any(p.startswith(".") for p in parts[:-1]):
        return Skip.LOCATION
    if parts[0] == DERIVATIVES:
        # derivatives/<pipeline>/sub-*/.../<file>: the pipeline's own subject
        # data. Anything else in a pipeline folder (its sourcedata, its
        # figures) is not an image of a participant's head.
        if len(parts) < 4 or not parts[2].startswith("sub-"):
            return Skip.LOCATION
        return None if include_derivatives else Skip.DERIVATIVE
    if parts[0] in EXCLUDED_TOP_LEVEL:
        return Skip.LOCATION
    return None if parts[0].startswith("sub-") else Skip.LOCATION


def inspect(root: Path, path: Path, *,
            include_derivatives: bool = False) -> Candidate | Skipped:
    """Classify one image."""
    root = Path(root)
    path = Path(path)
    rel = _relative(root, path)

    where = _location(root, path, include_derivatives)
    if where is not None:
        return Skipped(path, rel, where)

    datatype = _datatype_of(root, path)
    if datatype not in FACE_BEARING_DATATYPES:
        return Skipped(
            path, rel, Skip.DATATYPE,
            f"in {datatype}/" if datatype else "",
        )

    suffix = _suffix_of(path.name)
    if suffix and suffix not in _head_suffixes(datatype):
        return Skipped(path, rel, Skip.NOT_HEAD, f"_{suffix}")

    try:
        dims = read_dimensions(path)
    except (NotNifti, OSError) as exc:
        return Skipped(path, rel, Skip.UNREADABLE, str(exc))

    if not dims.is_3d:
        return Skipped(
            path, rel, Skip.FOUR_D, f"{dims.dim4} volumes",
        )

    sidecar = read_sidecar(sidecar_for(path))
    ours = defaced_by_us(sidecar)
    if ours is not None and ours.is_strip:
        # Our own skull strip: only the brain is left, so there is no face to
        # remove and no skull to strip.
        return Skipped(path, rel, Skip.STRIPPED, ours.label)
    theirs = defaced_by_others(sidecar)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    return Candidate(
        path=path,
        relative=rel,
        datatype=datatype,
        shape=dims.shape,
        size_bytes=size,
        foreign_methods=tuple(theirs),
        previous_engine=ours.id if ours else None,
        pipeline=pipeline_of(root, path),
    )


def derivative_pipelines(root: Path) -> list[Path]:
    """Every ``derivatives/<pipeline>/`` folder that holds subject data."""
    base = Path(root) / DERIVATIVES
    try:
        folders = sorted(p for p in base.iterdir()
                         if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []
    return [p for p in folders if any(c.is_dir() for c in p.glob("sub-*"))]


def has_derivatives(root: Path) -> bool:
    """Whether the dataset has a derivative pipeline with subject data."""
    return bool(derivative_pipelines(root))


def _counterparts(root: Path, starts: list[Path]) -> list[Path]:
    """For each raw place in ``starts`` (the dataset, a subject, a session,
    a datatype folder), the same place in every derivative pipeline: what
    "this subject, derivatives included" means."""
    pipelines = derivative_pipelines(root)
    out: list[Path] = []
    for start in starts:
        try:
            rel = start.relative_to(root)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] in (DERIVATIVES, *EXCLUDED_TOP_LEVEL):
            continue
        if start.is_file():
            continue
        for pipe in pipelines:
            twin = pipe / rel
            if twin.is_dir():
                out.append(twin)
    return out


def walk(root: Path, targets: Optional[Iterable[Path]] = None, *,
         include_derivatives: bool = False) -> Selection:
    """Classify every image under ``root``, or under each of ``targets``.

    ``targets`` may name files or folders; a folder is walked. Passing none
    walks the whole dataset, which is what the Tools menu does.
    ``include_derivatives`` makes images under ``derivatives/<pipeline>/``
    candidates too, and widens a raw scope to the same subject or session in
    every pipeline.
    """
    root = Path(root)
    if targets is None:
        roots: list[Path] = [root]
    else:
        roots = [Path(t) for t in targets]
    if include_derivatives:
        roots = roots + [t for t in _counterparts(root, roots) if t not in roots]

    seen: set[Path] = set()
    found: list[Path] = []
    for start in roots:
        if start.is_file():
            if _is_nifti(start) and start not in seen:
                seen.add(start)
                found.append(start)
            continue
        if not start.is_dir():
            continue
        for path in sorted(start.rglob("*")):
            if not path.is_file() or not _is_nifti(path):
                continue
            if path in seen:
                continue
            seen.add(path)
            found.append(path)

    candidates: list[Candidate] = []
    skipped: list[Skipped] = []
    for path in found:
        result = inspect(root, path, include_derivatives=include_derivatives)
        if isinstance(result, Candidate):
            candidates.append(result)
        else:
            skipped.append(result)

    candidates.sort(key=lambda c: (c.pipeline != "", c.pipeline, c.relative))
    skipped.sort(key=lambda s: s.relative)
    return Selection(tuple(candidates), tuple(skipped))


def targets_in_derivatives(root: Path, targets: Optional[Iterable[Path]]) -> bool:
    """Whether any of ``targets`` lies inside ``derivatives/``: a right-click
    on a derivative image means derivatives are wanted."""
    if not targets:
        return False
    for t in targets:
        try:
            parts = Path(t).relative_to(root).parts
        except ValueError:
            continue
        if parts and parts[0] == DERIVATIVES:
            return True
    return False


def summarise(selection: Selection, eng: Engine) -> str:
    """One line for a log or a CLI. Says what will happen and what will not."""
    n = len(selection.candidates)
    parts = [f"{n} image{'' if n == 1 else 's'} to deface with {eng.label}"]
    if selection.skipped:
        parts.append(f"{len(selection.skipped)} skipped")
    redo = sum(1 for c in selection.candidates if c.previous_engine)
    if redo:
        parts.append(f"{redo} already defaced")
    return ", ".join(parts)


__all__ = [
    "Candidate",
    "DERIVATIVES",
    "EXCLUDED_TOP_LEVEL",
    "FACE_BEARING_DATATYPES",
    "Selection",
    "Skip",
    "Skipped",
    "derivative_pipelines",
    "has_derivatives",
    "inspect",
    "pipeline_of",
    "summarise",
    "targets_in_derivatives",
    "walk",
]

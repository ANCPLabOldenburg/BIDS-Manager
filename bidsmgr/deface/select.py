"""Which images in a dataset can be defaced, and why the rest cannot.

The second half matters as much as the first. A user who is told "4 files will
be defaced" and not told that three more were skipped has been handed a dataset
with a face in it and no reason to suspect it. So every image this walks past
comes back either as a candidate or as a skip with a reason, and the dialog
shows both.

What is a candidate:

* a NIfTI image,
* in a datatype that can show a face,
* that is a single 3-D volume,
* under a subject, not under ``derivatives/``, ``sourcedata/`` or a dot folder.

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
# derivatives and sourcedata hold data produced from, or feeding, the raw
# images; defacing a derivative without its source is how a dataset ends up
# internally inconsistent. Dot folders are tool state, ours included.
EXCLUDED_TOP_LEVEL = frozenset({"derivatives", "sourcedata", "code", "stimuli"})

NIFTI_SUFFIXES = (".nii.gz", ".nii")


class Skip(str, Enum):
    """Why an image is not a candidate. The value is shown to a user."""

    FOUR_D = "not a single 3-D volume"
    UNREADABLE = "not a readable NIfTI-1 image"
    DATATYPE = "not a datatype that shows a face"
    LOCATION = "not raw subject data"
    ALREADY = "already defaced by BIDS Manager"


@dataclass(frozen=True)
class Candidate:
    """An image that can be defaced."""

    path: Path
    relative: str           # POSIX, dataset-relative. For display and keys.
    datatype: str
    shape: tuple[int, ...]
    size_bytes: int
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
    # <sub>/[<ses>/]<datatype>/<file>
    return parts[-2] if len(parts) >= 2 else ""


def _in_excluded_location(root: Path, path: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    if not parts:
        return True
    if any(p.startswith(".") for p in parts[:-1]):
        return True
    if parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    return not parts[0].startswith("sub-")


def inspect(root: Path, path: Path) -> Candidate | Skipped:
    """Classify one image."""
    root = Path(root)
    path = Path(path)
    rel = _relative(root, path)

    if _in_excluded_location(root, path):
        return Skipped(path, rel, Skip.LOCATION)

    datatype = _datatype_of(root, path)
    if datatype not in FACE_BEARING_DATATYPES:
        return Skipped(
            path, rel, Skip.DATATYPE,
            f"in {datatype}/" if datatype else "",
        )

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
    )


def walk(root: Path, targets: Optional[Iterable[Path]] = None) -> Selection:
    """Classify every image under ``root``, or under each of ``targets``.

    ``targets`` may name files or folders; a folder is walked. Passing none
    walks the whole dataset, which is what the Tools menu does.
    """
    root = Path(root)
    if targets is None:
        roots: list[Path] = [root]
    else:
        roots = [Path(t) for t in targets]

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
        result = inspect(root, path)
        if isinstance(result, Candidate):
            candidates.append(result)
        else:
            skipped.append(result)

    candidates.sort(key=lambda c: c.relative)
    skipped.sort(key=lambda s: s.relative)
    return Selection(tuple(candidates), tuple(skipped))


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
    "EXCLUDED_TOP_LEVEL",
    "FACE_BEARING_DATATYPES",
    "Selection",
    "Skip",
    "Skipped",
    "inspect",
    "summarise",
    "walk",
]

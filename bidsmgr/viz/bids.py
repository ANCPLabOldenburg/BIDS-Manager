"""What a viewer knows about a file because it sits in a BIDS dataset.

A generic viewer sees an array. A viewer inside the dataset can also see the
sidecar beside it, the events of the same run, the fieldmaps that point at it,
the same scan in another session. :class:`BidsContext` gathers that once per
opened file, on a worker, as plain data, so the features built on it (a time
axis in seconds, events on the graph, b-values per volume, jumping to a
fieldmap's target) never touch the disk while the user is looking.

Qt-free. Relative paths that become data are POSIX (CROSS_PLATFORM_RULES 1.1).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

_ENTITY = re.compile(r"(?:^|_)([a-zA-Z0-9]+)-([a-zA-Z0-9]+)")
#: Extensions made of two suffixes. Everything else is its last suffix.
_DOUBLE_EXTS = (".nii.gz", ".tsv.gz", ".fif.gz")


def full_ext(name) -> str:
    """The extension, lower-case, with the double ones kept whole
    (``.nii.gz``, ``.tsv.gz``, ``.fif.gz``). THE one implementation: the
    volume, signal, spectrum and physio code all ask this."""
    low = Path(name).name.lower()
    for ext in _DOUBLE_EXTS:
        if low.endswith(ext):
            return ext
    return Path(low).suffix


def stem_of(path: Path) -> str:
    """The name without its (possibly double) extension."""
    ext = full_ext(path.name)
    return path.name[: len(path.name) - len(ext)] if ext else path.name


def parse_entities(name: str) -> dict[str, str]:
    stem = stem_of(Path(name))
    return {k: v for k, v in _ENTITY.findall(stem)}


def suffix_of(name: str) -> str:
    stem = stem_of(Path(name))
    return stem.rsplit("_", 1)[-1] if "_" in stem else ""


def sidecar_for(path: Path) -> Path:
    """The JSON sidecar that shares the file's name."""
    return path.with_name(stem_of(path) + ".json")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def inherited_sidecar(path: Path, root: Optional[Path]) -> dict:
    """The sidecar with BIDS inheritance applied (nearest file wins).

    Walks from the dataset root down to the file's folder and merges every
    ``.json`` whose entities are a subset of the file's and whose suffix
    matches, so ``task-rest_bold.json`` at the top supplies
    ``RepetitionTime`` to every run that does not override it.
    """
    ents = parse_entities(path.name)
    suffix = suffix_of(path.name)
    merged: dict[str, Any] = {}
    folders = [path.parent]
    if root is not None:
        try:
            rel = path.parent.resolve().relative_to(Path(root).resolve())
            folders = [Path(root)]
            cur = Path(root)
            for part in rel.parts:
                cur = cur / part
                folders.append(cur)
        except ValueError:
            folders = [path.parent]
    for folder in folders:
        try:
            candidates = sorted(folder.glob("*.json"))
        except OSError:
            continue
        for cand in candidates:
            if suffix_of(cand.name) != suffix:
                continue
            cents = parse_entities(cand.name)
            if all(ents.get(k) == v for k, v in cents.items()):
                merged.update(_read_json(cand))
    exact = sidecar_for(path)
    if exact.is_file():
        merged.update(_read_json(exact))
    return merged


@dataclass
class BidsContext:
    """Plain facts about one file's place in the dataset."""

    path: Path
    root: Optional[Path]
    entities: dict[str, str] = field(default_factory=dict)
    suffix: str = ""
    datatype: str = ""
    sidecar: dict = field(default_factory=dict)
    #: Seconds per volume, from the sidecar (``RepetitionTime``).
    tr: Optional[float] = None
    #: Onset of each frame in seconds when frames are unevenly spaced: PET
    #: (``FrameTimesStart``) or a sparse BOLD acquisition (``VolumeTiming``).
    frame_starts: Optional[np.ndarray] = None
    frame_durations: Optional[np.ndarray] = None
    #: b-values per volume (DWI), from the ``.bval`` beside the image.
    bvals: Optional[np.ndarray] = None
    events_path: Optional[Path] = None
    #: The run's events (``viz.data.events.Event``), read with the rest.
    events: list = field(default_factory=list)
    physio_paths: list[Path] = field(default_factory=list)
    #: ``IntendedFor`` targets (for a fieldmap), resolved to paths.
    intended_for: list[Path] = field(default_factory=list)
    #: The fieldmaps whose ``IntendedFor`` names this file.
    fieldmaps: list[Path] = field(default_factory=list)
    #: Entity -> (previous, next) file differing only in that entity.
    neighbours: dict[str, tuple[Optional[Path], Optional[Path]]] = field(default_factory=dict)

    @property
    def rel(self) -> str:
        """The dataset-relative path, POSIX."""
        if self.root is None:
            return self.path.name
        try:
            return self.path.resolve().relative_to(Path(self.root).resolve()).as_posix()
        except ValueError:
            return self.path.name

    def frame_times(self, n_frames: int) -> Optional[np.ndarray]:
        """Real time of every frame, in seconds, when known.

        PET frames are unevenly spaced (seconds early, minutes late), so a
        graph against frame index misdraws the kinetics exactly where they
        matter. Mid-frame times when durations are known, else starts.
        """
        if self.frame_starts is not None and len(self.frame_starts) == n_frames:
            if self.frame_durations is not None and len(self.frame_durations) == n_frames:
                return self.frame_starts + self.frame_durations / 2.0
            return self.frame_starts
        if self.tr and self.tr > 0 and n_frames > 1:
            return np.arange(n_frames, dtype=float) * self.tr
        return None


def _float_list(value) -> Optional[np.ndarray]:
    if not isinstance(value, list) or len(value) < 1:
        return None
    try:
        arr = np.asarray([float(v) for v in value], dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if np.all(np.isfinite(arr)) else None


def bids_context(path: Path, root: Optional[Path] = None) -> BidsContext:
    """Gather the BIDS facts for ``path``. Reads a few small files."""
    path = Path(path)
    ctx = BidsContext(path=path, root=Path(root) if root else None)
    ctx.entities = parse_entities(path.name)
    ctx.suffix = suffix_of(path.name)
    ctx.datatype = path.parent.name if path.parent.name in {
        "anat", "func", "dwi", "fmap", "perf", "pet", "meg", "eeg", "ieeg",
        "beh", "mrs", "nirs", "motion", "micr",
    } else ""
    try:
        ctx.sidecar = inherited_sidecar(path, ctx.root)
    except Exception:  # noqa: BLE001 - context is a bonus, never a blocker
        log.debug("could not read the sidecar of %s", path, exc_info=True)
        ctx.sidecar = {}
    sc = ctx.sidecar
    tr = sc.get("RepetitionTime")
    if isinstance(tr, (int, float)) and tr > 0:
        ctx.tr = float(tr)
    # FrameTimesStart is a PET field; VolumeTiming is how BIDS times a BOLD
    # run whose volumes are not evenly spaced. Each is read only where the
    # standard defines it, so a stray key cannot redraw an unrelated graph.
    if ctx.suffix == "pet" or ctx.datatype == "pet":
        starts = _float_list(sc.get("FrameTimesStart"))
        if starts is not None and len(starts) >= 2:
            ctx.frame_starts = starts
            ctx.frame_durations = _float_list(sc.get("FrameDuration"))
    else:
        timing = _float_list(sc.get("VolumeTiming"))
        if timing is not None and len(timing) >= 2:
            ctx.frame_starts = timing
    # Siblings of the same run, matched on the run's entities (a glob on the
    # name would give run 1 the physio of run 10 too).
    from .data.events import events_sibling, read_events_tsv, run_base

    stem = stem_of(path)
    ctx.events_path = events_sibling(path)
    if ctx.events_path is not None:
        ctx.events = read_events_tsv(ctx.events_path)
    mine = run_base(path)
    try:
        ctx.physio_paths = sorted(
            p for p in path.parent.glob("*_physio.tsv*")
            if full_ext(p) in (".tsv", ".tsv.gz") and run_base(p) == mine
        )
    except OSError:
        ctx.physio_paths = []
    bval = path.with_name(stem + ".bval")
    if bval.is_file():
        try:
            vals = np.loadtxt(bval, dtype=float).ravel()
            ctx.bvals = vals
        except (OSError, ValueError):
            ctx.bvals = None
    intended = sc.get("IntendedFor")
    if isinstance(intended, str):
        intended = [intended]
    if isinstance(intended, list) and ctx.root is not None:
        sub = ctx.entities.get("sub")
        for ref in intended:
            if not isinstance(ref, str):
                continue
            if ref.startswith("bids::"):
                target = Path(ctx.root).joinpath(*ref[len("bids::"):].split("/"))
            elif sub:
                target = Path(ctx.root).joinpath(f"sub-{sub}", *ref.split("/"))
            else:
                continue
            ctx.intended_for.append(target)
    # The dataset around the file: what is next along each entity, and which
    # fieldmaps were meant for it. Globs, so on the worker, once per file.
    try:
        ctx.neighbours = neighbours(path, ctx.root)
        if ctx.datatype != "fmap":
            ctx.fieldmaps = fieldmaps_for(path, ctx.root)
    except OSError:
        log.debug("could not look around %s", path, exc_info=True)
    return ctx


#: Entities a viewer steps along, in menu order.
NAVIGABLE: tuple[str, ...] = ("run", "echo", "ses", "sub")


def dataset_root(path: Path) -> Optional[Path]:
    """The folder holding ``dataset_description.json`` above ``path``, else
    the parent of its ``sub-`` folder."""
    path = Path(path)
    for parent in path.parents:
        if (parent / "dataset_description.json").is_file():
            return parent
    sub = next((p for p in path.parents if p.name.startswith("sub-")), None)
    return sub.parent if sub is not None else None


def _natural(value: str):
    return (0, int(value), "") if value.isdigit() else (1, 0, value)


def siblings_along(path: Path, entity: str, root: Optional[Path] = None) -> list[Path]:
    """Every file differing from ``path`` ONLY in ``entity`` (the same suffix,
    extension and every other entity), ``path`` included, sorted by the
    entity's value (numbers as numbers: run 10 after run 9)."""
    path = Path(path)
    ents = parse_entities(path.name)
    if entity not in ents:
        return []
    base = Path(root) if root is not None else dataset_root(path)
    if base is None:
        base = path.parent
    try:
        rel = path.relative_to(base).as_posix()
    except ValueError:
        base, rel = path.parent, path.name
    token = f"{entity}-{ents[entity]}"
    pattern = re.sub(rf"(?<![A-Za-z0-9]){re.escape(token)}(?=[_/.]|$)", f"{entity}-*", rel)
    suffix, ext = suffix_of(path.name), full_ext(path)
    dir_entities = [p.split("-", 1)[0] for p in Path(rel).parent.parts if "-" in p]
    found = []
    try:
        candidates = list(base.glob(pattern))
    except (OSError, ValueError):
        return [path]
    for cand in candidates:
        cents = parse_entities(cand.name)
        if suffix_of(cand.name) != suffix or full_ext(cand) != ext or set(cents) != set(ents):
            continue
        if any(cents[k] != ents[k] for k in ents if k != entity):
            continue
        # A folder named for an entity must agree with the file's own value.
        parts = set(cand.relative_to(base).parent.parts)
        if any(f"{k}-{cents[k]}" not in parts for k in dir_entities if k in cents):
            continue
        found.append(cand)
    if path not in found:
        found.append(path)
    found.sort(key=lambda p: _natural(parse_entities(p.name)[entity]))
    return found


def neighbours(path: Path, root: Optional[Path] = None
               ) -> dict[str, tuple[Optional[Path], Optional[Path]]]:
    """For each entity of :data:`NAVIGABLE` that ``path`` has, the previous
    and next file along it (None at either end)."""
    path = Path(path)
    out: dict[str, tuple[Optional[Path], Optional[Path]]] = {}
    for entity in NAVIGABLE:
        sibs = siblings_along(path, entity, root)
        if len(sibs) <= 1:
            continue
        i = sibs.index(path)
        out[entity] = (sibs[i - 1] if i > 0 else None,
                       sibs[i + 1] if i + 1 < len(sibs) else None)
    return out


def fieldmaps_for(path: Path, root: Optional[Path] = None) -> list[Path]:
    """The fieldmap images whose ``IntendedFor`` names ``path``, in any of
    the spellings BIDS allows (subject-relative, or a ``bids::`` URI)."""
    path = Path(path)
    base = Path(root) if root is not None else dataset_root(path)
    sub_dir = next((p for p in path.parents if p.name.startswith("sub-")), None)
    if base is None or sub_dir is None:
        return []
    try:
        wanted = {path.relative_to(sub_dir).as_posix(),
                  "bids::" + path.relative_to(base).as_posix()}
    except ValueError:
        return []
    out = []
    for sidecar in sorted(sub_dir.glob("**/fmap/*.json")):
        refs = _read_json(sidecar).get("IntendedFor")
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list) or not wanted.intersection(r for r in refs
                                                                 if isinstance(r, str)):
            continue
        for ext in (".nii.gz", ".nii"):
            image = sidecar.with_name(stem_of(sidecar) + ext)
            if image.is_file():
                out.append(image)
                break
    return out


#: b-values within this of each other are one shell (scanners write 995 or
#: 1005 for a nominal 1000, and 5 for a b=0).
SHELL_STEP = 50.0


def shells_of(bvals) -> np.ndarray:
    """Each volume's shell: its b-value rounded to the nearest 50."""
    b = np.asarray(bvals, dtype=float)
    return (np.round(b / SHELL_STEP) * SHELL_STEP).astype(int)


#: Anatomical images to show something on, in order of preference.
ANATOMY_PREFERENCE = ("T1w", "MPRAGE", "T2w", "FLAIR", "PDw", "T2starw", "inplaneT1")


def anatomical_for(path: Path) -> Optional[Path]:
    """The subject's anatomical image to show ``path`` on (an MRS voxel, a
    statistical map): the same session's first, else any session's; T1w
    before other contrasts. None outside a ``sub-`` folder."""
    path = Path(path)
    sub_dir = next((p for p in path.parents if p.name.startswith("sub-")), None)
    if sub_dir is None:
        return None
    ses_dir = next((p for p in path.parents
                    if p.name.startswith("ses-") and p.parent == sub_dir), None)
    folders = ([ses_dir / "anat"] if ses_dir is not None else []) + [sub_dir / "anat"]
    try:
        folders += sorted(sub_dir.glob("ses-*/anat"))
    except OSError:
        pass
    seen: set[Path] = set()
    for folder in folders:
        if folder in seen or not folder.is_dir():
            continue
        seen.add(folder)
        images = [p for p in folder.iterdir() if full_ext(p) in (".nii", ".nii.gz")]
        for suffix in ANATOMY_PREFERENCE:
            hits = sorted(p for p in images if suffix_of(p.name) == suffix)
            if hits:
                return hits[0]
    return None


__all__ = [
    "ANATOMY_PREFERENCE", "BidsContext", "NAVIGABLE", "SHELL_STEP", "anatomical_for",
    "bids_context", "dataset_root", "fieldmaps_for", "full_ext", "inherited_sidecar",
    "neighbours", "parse_entities", "shells_of", "sidecar_for", "siblings_along",
    "stem_of", "suffix_of",
]

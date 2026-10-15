"""Checking a dataset: which images, one image by its path, many at once.

Shared by ``bidsmgr-qc`` and the Editor's Quality check. Images are checked
in separate processes (joblib), each writing its own JSON; the group tables
are written once at the end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

from . import anat, report
from .config import DEFAULT, QcConfig
from .types import Finding, QCResult

#: Datatypes the check knows.
DATATYPES = ("anat", "dwi")


def kind_of(path: Path) -> Optional[str]:
    """"anat", "dwi", or None when the check does not apply to ``path``."""
    from ..viz import bids as VB

    name = Path(path).name
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        return None
    suffix = VB.suffix_of(name)
    if suffix == "dwi":
        return "dwi"
    if suffix in anat.SUFFIXES:
        return "anat"
    return None


def find_images(root: Path, *, datatypes: Iterable[str] = DATATYPES,
                participants: Optional[Iterable[str]] = None) -> list[Path]:
    """Every anatomical and diffusion image of the dataset (not under
    ``derivatives``, ``sourcedata`` or hidden folders), sorted."""
    root = Path(root)
    wanted = set(datatypes)
    subs = {p if p.startswith("sub-") else f"sub-{p}" for p in participants or ()}
    out = []
    for path in root.rglob("sub-*/**/*.nii*"):
        rel = path.relative_to(root).parts
        if any(part.startswith(".") or part in ("derivatives", "sourcedata", "code")
               for part in rel[:-1]):
            continue
        if subs and rel[0] not in subs:
            continue
        kind = kind_of(path)
        if kind is None or kind not in wanted or path.parent.name not in wanted:
            continue
        out.append(path)
    return sorted(out)


def check_path(path: Path, root: Optional[Path] = None, *, cancel=None, progress=None,
               config: Optional[QcConfig] = None) -> QCResult:
    """The check ``path`` calls for; a result with an error finding when it
    cannot be read."""
    path = Path(path)
    kind = kind_of(path)
    try:
        if kind == "dwi":
            from . import dwi

            return dwi.check_file(path, root=root, cancel=cancel, progress=progress,
                                  config=config)
        if kind == "anat":
            return anat.check_file(path, root=root, cancel=cancel, progress=progress,
                                   config=config)
    except (OSError, ValueError, MemoryError) as exc:
        res = QCResult(path=str(path), kind=kind or "", suffix="")
        res.findings.append(Finding("unreadable", "Could not be checked", "error",
                                    f"{type(exc).__name__}: {exc}", None))
        return res
    raise ValueError(f"{path.name}: no quality check for this kind of image")


def _one(path: str, root: str, config: dict) -> dict:
    """Worker (another process): check and save one image, return its JSON.
    The configuration travels as plain data."""
    res = check_path(Path(path), Path(root), config=QcConfig.model_validate(config))
    report.save(res, Path(root))
    return report.to_json(res, root=Path(root))


def run(root: Path, paths: list[Path], *, jobs: int = 1, config: Optional[QcConfig] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None) -> list[dict]:
    """Check ``paths`` (``jobs`` processes), save each, write the group
    tables, and return every result as JSON."""
    from joblib import Parallel, delayed

    root = Path(root)
    plain = (config or DEFAULT).model_dump(mode="json")
    report.write_description(root)
    rows: list[dict] = []
    total = len(paths)
    if jobs <= 1:
        for i, p in enumerate(paths):
            if cancel is not None and cancel():
                break
            rows.append(_one(str(p), str(root), plain))
            if progress is not None:
                progress(i + 1, total, Path(p).name)
    else:
        gen = Parallel(n_jobs=jobs, return_as="generator")(
            delayed(_one)(str(p), str(root), plain) for p in paths)
        for i, row in enumerate(gen):
            rows.append(row)
            if progress is not None:
                progress(i + 1, total, Path(str(row.get("path", ""))).name)
            if cancel is not None and cancel():
                break
    # The group: every result in the derivative, not only this run's.
    all_rows = report.load_all(root)
    report.write_group(root, all_rows)
    return rows


__all__ = ["DATATYPES", "check_path", "find_images", "kind_of", "run"]

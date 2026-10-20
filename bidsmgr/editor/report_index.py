"""From a path to its verdict in a validation report, in one lookup.

The Editor asked "which verdict is this file's" in three places (the
validation pane, the sidecar form twice per click, Fix and Highlight), each
by walking every verdict in the report and resolving every path on the way:
about eight ``lstat`` calls per verdict. On a 21,663-file report that was 3.4
seconds to click a file late in the report, and a folder, which matches
nothing, paid for the whole walk.

One index per report, keyed by the path relative to the dataset root (POSIX),
built the first time it is asked for. The root is resolved ONCE, so a tree
that spells it ``/var/...`` and a report that spells it ``/private/var/...``
still meet. A caller that changes ``report.files`` (a one-file re-validation
merged in) calls :func:`forget`.

Qt-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .types import FileVerdict, ValidationReport

# id(report) -> (report, {relative posix path: verdict}). Holding the report
# keeps its id from being reused while the entry lives; a few are kept, the
# Editor never has more than one report at a time.
_INDEXES: dict[int, tuple[ValidationReport, dict[str, FileVerdict]]] = {}
_KEEP = 4

_RESOLVED: dict[str, str] = {}


def _resolved_root(root: Path) -> str:
    key = str(root)
    found = _RESOLVED.get(key)
    if found is None:
        try:
            found = str(Path(root).resolve())
        except OSError:
            found = key
        _RESOLVED[key] = found
    return found


def relative(root: Path, path: Path) -> Optional[str]:
    """``path`` relative to ``root`` as a POSIX string, or ``None`` outside it.

    Lexically (``relative_to`` reads no disk and knows each platform's
    separators), against the root as given and as resolved; the path itself
    is resolved only when neither matches (a symlink inside the tree).
    """
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    for base in (Path(root), Path(_resolved_root(root))):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    try:
        return path.resolve().relative_to(_resolved_root(root)).as_posix()
    except (OSError, ValueError):
        return None


def index(report: ValidationReport, root: Optional[Path] = None) -> dict[str, FileVerdict]:
    """``{relative posix path: verdict}`` for ``report``, made once."""
    hit = _INDEXES.get(id(report))
    if hit is not None and hit[0] is report:
        return hit[1]
    root = Path(root) if root is not None else report.bids_root
    out: dict[str, FileVerdict] = {}
    for verdict in report.files:
        key = (relative(root, verdict.path) if verdict.path.is_absolute() and root is not None
               else verdict.path.as_posix())
        if key is not None:
            out.setdefault(key, verdict)
    if len(_INDEXES) >= _KEEP:
        _INDEXES.pop(next(iter(_INDEXES)))
    _INDEXES[id(report)] = (report, out)
    return out


def find_verdict(report: Optional[ValidationReport], root: Optional[Path],
                 path: Optional[Path]) -> Optional[FileVerdict]:
    """The verdict for ``path`` (absolute or relative to ``root``), or None."""
    if report is None or root is None or path is None:
        return None
    rel = relative(Path(root), Path(path))
    if rel is None:
        return None
    return index(report, root).get(rel)


def forget(report: ValidationReport) -> None:
    """``report.files`` changed: the next lookup builds the index again."""
    _INDEXES.pop(id(report), None)


__all__ = ["find_verdict", "forget", "index", "relative"]

"""Write ``CITATION.cff`` from what the dataset already says about itself.

BIDS 1.11 recognises ``CITATION.cff`` at the dataset root and bidsval validates
it. Every fact it needs is already in ``dataset_description.json``, whose keys
the BIDS schema defines, so this reads them rather than asking the user a
second time.

The field model and the mapping live in :mod:`bidsmgr.editor.cff`, which the
Editor's citation form also renders. One description of the format, two
surfaces, so a field added there appears in both.

Two things this will not do. It will not invent an author list: a citation
whose authors are wrong is worse than no citation, so a dataset with no
``Authors`` gets a ``TODO`` that validation keeps reporting. And it will not
overwrite a ``CITATION.cff`` somebody wrote by hand unless told to.

Qt-free.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from ..editor.cff import (
    FILENAME,
    MOVED_FIELDS,
    dumps,
    from_dataset_description,
)

log = logging.getLogger(__name__)


def citation_path(root: Path) -> Path:
    return Path(root) / FILENAME


def build_citation(description: dict[str, Any]) -> str:
    """Render a ``CITATION.cff`` body from a parsed dataset description."""
    return dumps(from_dataset_description(description))


def write_citation(root: Path, *, overwrite: bool = False) -> Optional[Path]:
    """Generate ``CITATION.cff`` and move the fields it now owns into it.

    Returns the path, or ``None`` if a citation file was already there and
    ``overwrite`` was not asked for. One reversible operation, so undo takes
    back both the new file and the edit to the description.
    """
    from ..project.operations import begin_operation

    root = Path(root)
    target = citation_path(root)
    if target.exists() and not overwrite:
        log.info("CITATION.cff already exists; not overwriting")
        return None
    dd = root / "dataset_description.json"
    try:
        description = json.loads(dd.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s: %s", dd, exc)
        description = {}
    if not isinstance(description, dict):
        description = {}
    # Only what the standard forbids duplicating leaves the description. The
    # rest is rendered into the citation file AND kept where it was, so
    # writing a citation never silently deletes an answer.
    remaining = {k: v for k, v in description.items() if k not in MOVED_FIELDS}
    moved = sorted(set(description) & set(MOVED_FIELDS))
    with begin_operation(root, "Write CITATION.cff") as op:
        op.write_text(target, build_citation(description))
        if moved:
            op.write_json(dd, remaining)
    if moved:
        log.info(
            "moved %s from dataset_description.json into CITATION.cff "
            "(the standard forbids stating them in both)", ", ".join(moved),
        )
    return target


def fields_moved_by_citation(root: Path) -> list[str]:
    """Which fields writing the citation file would take out of the
    description, so a caller can say so before doing it."""
    try:
        description = json.loads(
            (Path(root) / "dataset_description.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    if not isinstance(description, dict):
        return []
    return sorted(set(description) & set(MOVED_FIELDS))


__all__ = [
    "MOVED_FIELDS",
    "build_citation",
    "citation_path",
    "fields_moved_by_citation",
    "write_citation",
]

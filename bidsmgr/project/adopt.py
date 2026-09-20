"""Bring a dataset BIDS Manager did not convert under its management.

The project model assumes BIDS Manager wrote the dataset: a scan version, an
event log, a baseline it can return to. A dataset that arrived from somewhere
else has none of that, so the Editor could change its files with no way back.

Adopting is the missing first step. It creates the same ``.bidsmgr/`` bundle a
converted dataset has, and records a **manifest** of the tree as found: every
path with its size and a hash. Without that baseline, "revert" has nothing to
revert to for a file BIDS Manager never wrote, and the promise of reversible
editing would be true only for edits made after the tool arrived.

Two things this deliberately does not do:

**It does not modify the dataset.** Nothing outside ``.bidsmgr/`` is touched.
An adopted dataset validates exactly as it did before, because dot-directories
are ignored by every BIDS tool.

**It does not claim the dataset was converted here.** The event records that
this project was adopted, and the manifest is the state at adoption rather
than the output of a conversion. A reader of the history can tell the two
apart.

Qt-free.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MANIFEST = Path(".bidsmgr") / "adopted_manifest.json"

# Only the first megabyte is hashed. A manifest exists to notice that a file
# changed, not to prove it did not; hashing 800 MB of BOLD to find that out
# would make adopting a real dataset take minutes for no added certainty.
_HASH_BYTES = 1024 * 1024


@dataclass
class AdoptResult:
    """What adopting found and wrote."""

    root: Path
    files: int
    bytes_seen: int
    already_managed: bool
    manifest_path: Path


class NotABidsDataset(ValueError):
    """The folder does not look like a BIDS dataset."""


def looks_like_bids(root: Path) -> bool:
    """A dataset description, or at least one ``sub-*`` directory.

    Deliberately loose. A dataset missing its description is exactly the kind
    of thing a user wants to open in order to fix, so refusing it would block
    the repair.
    """
    root = Path(root)
    if (root / "dataset_description.json").is_file():
        return True
    return any(p.is_dir() and p.name.startswith("sub-") for p in root.iterdir())


def is_managed(root: Path) -> bool:
    """Does this dataset already carry a BIDS Manager project bundle?"""
    return (Path(root) / ".bidsmgr" / "project").is_dir()


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            h.update(fh.read(_HASH_BYTES))
    except OSError:
        return ""
    return h.hexdigest()[:16]


def build_manifest(root: Path) -> dict:
    """Record the tree as it is now: path, size, and a partial hash."""
    root = Path(root)
    entries: dict[str, dict] = {}
    total = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".bidsmgr" in p.parts:
            continue
        try:
            rel = p.relative_to(root).as_posix()
            size = p.stat().st_size
        except (ValueError, OSError):
            continue
        entries[rel] = {"size": size, "sha256_1m": _digest(p)}
        total += size
    return {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": (
            "State of the dataset when BIDS Manager adopted it. Files listed "
            "here were not written by this tool."
        ),
        "files": entries,
        "total_bytes": total,
    }


def adopt(root: Path, *, name: Optional[str] = None) -> AdoptResult:
    """Create the project bundle and the baseline manifest for ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise NotABidsDataset(f"{root} is not a directory")
    if not looks_like_bids(root):
        raise NotABidsDataset(
            f"{root} has no dataset_description.json and no sub-* folder, so "
            "it does not look like a BIDS dataset."
        )

    # The bundle lives INSIDE the dataset, which is the whole point and also
    # the one thing that can make adoption impossible. Say so before reading
    # the tree rather than after.
    if not os.access(root, os.W_OK):
        raise NotABidsDataset(
            f"{root} is not writable, so the tracking folder cannot be "
            "created there. A read-only or cloud-mounted dataset has to be "
            "copied somewhere writable first."
        )

    already = is_managed(root)
    manifest = build_manifest(root)

    if not already:
        # The same bundle a converted dataset gets, so everything downstream
        # (versions, event log, undo) works without knowing the difference.
        #
        # Deliberately NOT ``cli.create.open_or_create_workspace``: that also
        # scaffolds README, .bidsignore and dataset_description.json, which is
        # right for a dataset being created and wrong for one being adopted.
        # Adopting must leave the dataset byte-identical, or the baseline it
        # records is a baseline of a tree this tool already changed.
        from ..cli._scaffold import project_bundle_dir
        from .project import Project

        Project.create(
            project_bundle_dir(root),
            name=name or root.name,
            description=(
                "Adopted by BIDS Manager. The dataset was not converted here; "
                "see .bidsmgr/adopted_manifest.json for its state at adoption."
            ),
        )

    # A dataset under git gains an untracked folder. Offering to ignore it is
    # not this function's job, but noticing is: the caller shows it.
    _ensure_git_ignored(root)

    target = root / MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    return AdoptResult(
        root=root,
        files=len(manifest["files"]),
        bytes_seen=manifest["total_bytes"],
        already_managed=already,
        manifest_path=target,
    )


def _ensure_git_ignored(root: Path) -> bool:
    """Add ``.bidsmgr/`` to a git working tree's ignore list, once.

    Adopting a dataset that is under version control otherwise leaves it dirty
    with a folder the user did not create, which reads as the tool having
    changed their data. Only touched when a ``.git`` directory is actually
    there, and never duplicated.
    """
    if not (root / ".git").exists():
        return False
    ignore = root / ".gitignore"
    line = ".bidsmgr/"
    try:
        existing = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        if any(row.strip() == line for row in existing.splitlines()):
            return False
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        ignore.write_text(
            existing + prefix
            + "# BIDS Manager keeps its curation history here.\n"
            + line + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        log.debug("could not update %s: %s", ignore, exc)
        return False
    return True


def read_manifest(root: Path) -> Optional[dict]:
    path = Path(root) / MANIFEST
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def changed_since_adoption(root: Path) -> list[str]:
    """Files that differ from the manifest, plus ones added since.

    What "revert" would have to deal with, and a plain answer to "what has
    this tool done to my dataset".
    """
    manifest = read_manifest(root)
    if manifest is None:
        return []
    root = Path(root)
    recorded = manifest.get("files", {})
    out: list[str] = []
    seen: set[str] = set()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".bidsmgr" in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        seen.add(rel)
        was = recorded.get(rel)
        if was is None:
            out.append(rel)
            continue
        try:
            if p.stat().st_size != was.get("size"):
                out.append(rel)
                continue
        except OSError:
            continue
        if _digest(p) != was.get("sha256_1m"):
            out.append(rel)
    out.extend(sorted(set(recorded) - seen))
    return sorted(out)


__all__ = [
    "AdoptResult",
    "MANIFEST",
    "NotABidsDataset",
    "adopt",
    "build_manifest",
    "changed_since_adoption",
    "is_managed",
    "looks_like_bids",
    "read_manifest",
]

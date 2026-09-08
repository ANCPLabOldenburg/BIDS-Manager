"""Copy already-curated companion files into the BIDS tree (no conversion).

For task logs / behavioural / stimulus data that the user has already curated
into a BIDS-shaped file, the per-row "companion files" links are copied into the
staged datatype directory under the recording's BIDS name. This is a "place and
name" step, not a converter: the file's content is the user's responsibility and
is checked by ``bidsmgr-validate`` afterwards. The set of attachable suffixes is
deliberately constrained (so this never becomes a generic file copier).

An attached ``events`` companion overwrites the events.tsv mne-bids generated
from the recording's triggers (the curated file wins, per the agreed precedence).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .blood import is_blood_role

log = logging.getLogger(__name__)

# BIDS companion suffixes a row may link. Constrained on purpose.
ALLOWED_COMPANION_SUFFIXES: frozenset[str] = frozenset(
    {"events", "beh", "stim", "physio", "channels", "electrodes"}
)


def _src_extension(path: Path) -> str:
    """Return the source file's extension, preserving ``.tsv.gz`` etc."""
    name = path.name.lower()
    if name.endswith(".tsv.gz"):
        return ".tsv.gz"
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix


def attach_companion_files(subject_staging_dir: Path, tasks) -> int:
    """Copy each task's linked companion files into the staged tree.

    Returns the number of files copied. Best-effort: a missing source, an
    unsupported suffix, or a copy error is logged and skipped, never raised.
    """
    if not subject_staging_dir.is_dir():
        return 0

    n_copied = 0
    for task in tasks:
        companions = getattr(task, "companion_files", ()) or ()
        if not companions:
            continue
        basename = getattr(task, "basename", "") or ""
        if not basename:
            continue
        # Anchor on the staged datatype sidecar so we land beside the recording.
        sidecar = next(iter(subject_staging_dir.rglob(f"{basename}.json")), None)
        if sidecar is None:
            continue
        prefix = basename.rsplit("_", 1)[0] if "_" in basename else basename

        for suffix, src in companions:
            if is_blood_role(str(suffix)):
                # A blood curve is converted, not copied. fixups/blood.py owns it.
                continue
            if suffix not in ALLOWED_COMPANION_SUFFIXES:
                log.warning(
                    "companion: unsupported suffix %r for %s; skipped", suffix, basename,
                )
                continue
            src_path = Path(src)
            if not src_path.is_file():
                log.warning("companion: source not found %s (%s); skipped", src_path, suffix)
                continue
            ext = _src_extension(src_path)
            dest = sidecar.with_name(f"{prefix}_{suffix}{ext}")
            try:
                shutil.copyfile(src_path, dest)
                n_copied += 1
                log.info("companion: copied %s -> %s", src_path.name, dest.name)
            except OSError as exc:
                log.warning("companion: copy failed %s: %s", src_path, exc)

    return n_copied


__all__ = ["attach_companion_files", "ALLOWED_COMPANION_SUFFIXES"]

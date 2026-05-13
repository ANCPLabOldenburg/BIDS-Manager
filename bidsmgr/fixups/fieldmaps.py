"""Post-conversion fmap suffix mapping.

dcm2niix splits a single fieldmap series into multiple files and labels
them with its own tokens (``_e1``, ``_e2``, ``_ph``, …). BIDS expects
those to appear as the canonical fmap suffixes (``magnitude1``,
``magnitude2``, ``phasediff``, ``phase1``, ``phase2``). The classifier
already proposed a basename ending in one of the BIDS suffixes; this
fixup walks the per-subject staging tree and rewrites the dcm2niix
tokens in place so each file lands at its final BIDS path.

This is the dcm2niix-direct counterpart of v0.2.5
``post_conv_renamer.process_fmap_dir`` — the rule shape is the same, but
v0.2.5 worked off HeuDiConv's outputs (``echo-1`` / ``echo-2``) while we
work off dcm2niix's tokens.

Example renames (the file extension can be ``.nii.gz`` / ``.nii`` /
``.json`` / ``.bval`` / ``.bvec``)::

    sub-001_magnitude1_e1.nii.gz → sub-001_magnitude1.nii.gz
    sub-001_magnitude1_e2.nii.gz → sub-001_magnitude2.nii.gz
    sub-001_magnitude1_ph.nii.gz → sub-001_phasediff.nii.gz
    sub-001_fmap_e1.nii.gz       → sub-001_magnitude1.nii.gz
    sub-001_magnitude1_e1_ph.nii.gz → sub-001_phase1.nii.gz
    sub-001_magnitude1_e2_ph.nii.gz → sub-001_phase2.nii.gz
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# Order matters: longer tokens (``_e1_ph``, ``_e2_ph``) must match before
# their shorter prefixes (``_e1``, ``_e2``).
_TAIL_RE = re.compile(
    r"(?:_(?P<existing>magnitude[12]|phasediff|phase[12]|fieldmap|epi|fmap))?"
    r"(?P<token>_e1_ph|_e2_ph|_e1|_e2|_ph)"
    r"(?P<ext>\.(?:nii(?:\.gz)?|json|bval|bvec))$"
)

_TOKEN_TO_BIDS_SUFFIX: dict[str, str] = {
    "_e1": "magnitude1",
    "_e2": "magnitude2",
    "_ph": "phasediff",
    "_e1_ph": "phase1",
    "_e2_ph": "phase2",
}


def rename_for_fmap_token(name: str) -> Optional[str]:
    """Return the BIDS-renamed filename, or ``None`` if no token is present.

    Files already named with a canonical BIDS fmap suffix (no dcm2niix
    token) return ``None`` — they're in the right place already.
    """
    m = _TAIL_RE.search(name)
    if not m:
        return None
    head = name[: m.start()]
    bids_suffix = _TOKEN_TO_BIDS_SUFFIX[m.group("token")]
    ext = m.group("ext")
    return f"{head}_{bids_suffix}{ext}"


def apply_fieldmap_renames(subject_staging_dir: Path) -> dict[Path, Path]:
    """Walk every ``fmap/`` directory under ``subject_staging_dir`` and rename.

    Searches both ``<staging>/fmap/`` (no-session layout) and
    ``<staging>/ses-*/fmap/`` (session layout). Matches ``.nii``,
    ``.nii.gz``, ``.json``, ``.bval``, and ``.bvec`` siblings.

    Returns
    -------
    dict[Path, Path]
        Mapping ``old_path -> new_path`` for every file actually renamed.
        Empty dict if no fmap dir exists or nothing matched.
    """
    if not subject_staging_dir.is_dir():
        return {}

    rename_map: dict[Path, Path] = {}

    fmap_dirs = [d for d in subject_staging_dir.rglob("fmap") if d.is_dir()]
    if not fmap_dirs:
        return rename_map

    for fmap_dir in fmap_dirs:
        # Sort for deterministic rename order in tests / logs.
        for src in sorted(fmap_dir.iterdir()):
            if not src.is_file():
                continue
            new_name = rename_for_fmap_token(src.name)
            if not new_name or new_name == src.name:
                continue
            dst = fmap_dir / new_name
            if dst.exists() and dst != src:
                # Don't clobber: a real file already sits at the target.
                # This is unusual (means dcm2niix produced both the
                # tokened and un-tokened outputs) — keep the existing
                # one and warn so the user can investigate.
                log.warning(
                    "fmap rename: refusing to overwrite existing %s with %s",
                    dst, src,
                )
                continue
            src.rename(dst)
            rename_map[src] = dst
            log.info("fmap rename: %s → %s", src.name, dst.name)

    return rename_map


__all__ = ["apply_fieldmap_renames", "rename_for_fmap_token"]

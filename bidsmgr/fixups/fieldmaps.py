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

import json
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

_DATA_EXTS = (".nii.gz", ".nii", ".json", ".bval", ".bvec")


def _sidecar_for(path: Path) -> Path:
    """The JSON dcm2niix wrote beside ``path`` (or ``path`` itself)."""
    name = path.name
    for ext in _DATA_EXTS:
        if name.endswith(ext):
            return path.with_name(name[: -len(ext)] + ".json")
    return path.with_suffix(".json")


def _read_sidecar(path: Path) -> dict:
    """Parse a sidecar. Unreadable or absent reads as empty.

    A rename must not fail because a sidecar is malformed: the filename
    token is still there to fall back on.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _is_phase(sidecar: dict) -> bool:
    kinds = {str(v).upper() for v in sidecar.get("ImageType", [])}
    return bool({"P", "PHASE"} & kinds)


def _is_magnitude(sidecar: dict) -> bool:
    kinds = {str(v).upper() for v in sidecar.get("ImageType", [])}
    return bool({"M", "MAGNITUDE"} & kinds)


def fmap_suffix_from_sidecar(sidecar: dict) -> Optional[str]:
    """The BIDS fmap suffix the sidecar's own numbers imply, or ``None``.

    dcm2niix's FILENAME token says which echo a file came from. It is not a
    BIDS suffix and the two do not line up: a Siemens ``gre_field_mapping``
    acquires two echoes and reconstructs ONE phase image from the pair, and
    dcm2niix names that ``_e2_ph`` because it belongs to the second echo
    while calling it ``phasediff`` in the sidecar it writes beside it.
    Reading only the token turned every such fieldmap into ``phase2``.

    That is wrong twice over. The file is a phase DIFFERENCE, not the phase
    at echo 2; and ``phase2`` without a ``phase1`` is not one of the four
    fieldmap forms the standard defines, so the result described a dataset
    that cannot exist.

    So decide from the facts instead. The standard says a ``phasediff``
    sidecar carries ``EchoTime1`` and ``EchoTime2`` while ``phase1`` and
    ``phase2`` each carry a single ``EchoTime``, which makes the presence
    of that pair a definitive answer rather than an inference. Only when
    the numbers say nothing does the caller fall back to the token.

    This is deliberately not "believe ``BidsGuess``". dcm2niix's guess is
    consulted by :func:`rename_for_fmap_token` as a tie-breaker, and only
    after it has been checked against the schema's own list of fieldmap
    suffixes.
    """
    if not sidecar:
        return None

    if _is_phase(sidecar):
        if "EchoTime1" in sidecar and "EchoTime2" in sidecar:
            return "phasediff"
        echo = sidecar.get("EchoNumber")
        if isinstance(echo, (int, float)) and int(echo) in (1, 2):
            return f"phase{int(echo)}"
        return None

    if _is_magnitude(sidecar):
        echo = sidecar.get("EchoNumber")
        if isinstance(echo, (int, float)) and int(echo) in (1, 2):
            return f"magnitude{int(echo)}"
        return None

    return None


def _guessed_suffix(sidecar: dict) -> Optional[str]:
    """The suffix out of dcm2niix's ``BidsGuess``, if the schema knows it.

    ``BidsGuess`` is a pair, ``["fmap", "_acq-fm2_phasediff"]``. Only the
    suffix is taken, and only if it is one the standard actually defines
    for ``fmap``, so a future dcm2niix that invents a token cannot rename a
    file to something no validator will accept.
    """
    guess = sidecar.get("BidsGuess")
    if not (isinstance(guess, (list, tuple)) and len(guess) == 2):
        return None
    suffix = str(guess[1]).rsplit("_", 1)[-1]
    try:
        from .. import schema as schema_mod

        if suffix in set(schema_mod.list_suffixes("fmap")):
            return suffix
    except Exception:  # noqa: BLE001 - a hint must never break a rename
        return None
    return None


def rename_for_fmap_token(
    name: str, sidecar: Optional[dict] = None
) -> Optional[str]:
    """Return the BIDS-renamed filename, or ``None`` if no token is present.

    Files already named with a canonical BIDS fmap suffix (no dcm2niix
    token) return ``None`` — they're in the right place already.

    ``sidecar`` is the JSON dcm2niix wrote beside the file. When it is
    given, what it says about echo times decides the suffix and the
    filename token is only the fallback. Callers that have no sidecar get
    the old token-only behaviour.
    """
    m = _TAIL_RE.search(name)
    if not m:
        return None
    head = name[: m.start()]
    ext = m.group("ext")

    bids_suffix = (
        fmap_suffix_from_sidecar(sidecar or {})
        or _guessed_suffix(sidecar or {})
        or _TOKEN_TO_BIDS_SUFFIX[m.group("token")]
    )
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
        files = [p for p in sorted(fmap_dir.iterdir()) if p.is_file()]

        # PLAN FIRST, then move. The suffix is decided by the sidecar, and
        # the sidecar is one of the files being renamed: renaming
        # ``..._e2_ph.json`` before ``..._e2_ph.nii.gz`` leaves the image
        # with no sidecar to read, so it falls back to the token and the
        # pair lands under two different suffixes. Every sidecar is read
        # while they are all still where they were.
        plan: list[tuple[Path, str]] = []
        for src in files:
            new_name = rename_for_fmap_token(
                src.name, _read_sidecar(_sidecar_for(src))
            )
            if new_name and new_name != src.name:
                plan.append((src, new_name))

        for src, new_name in plan:
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

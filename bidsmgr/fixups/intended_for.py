"""Populate ``IntendedFor`` in fmap JSON sidecars.

Per BIDS spec, every fmap sidecar should declare which functional runs
it's intended to correct. We use the BIDS URI form (spec ≥ 1.9):

    "IntendedFor": ["bids::sub-001/ses-pre/func/sub-001_ses-pre_task-rest_bold.nii.gz"]

Algorithm (ported from v0.2.5
``BIDS-Manager/bids_manager/post_conv_renamer._update_intended_for``):

1. Per (subject, session) folder containing both ``fmap/`` AND ``func/``.
2. Read ``AcquisitionTime`` from every fmap and bold JSON.
3. **Time-based binding** (preferred): each fieldmap binds to all
   functional runs acquired *after* its own time and *before* the next
   fieldmap. This matches the operator's mental model — "I just shimmed,
   here are the runs I'm about to record."
4. **Fallback** (any timing missing): link every fmap to every run.
5. Emit ``IntendedFor`` as ``bids::sub-X[/ses-Y]/func/<filename>``.

Fieldmaps that share an acquisition (``magnitude1`` / ``magnitude2`` /
``phasediff``) are grouped by their stripped stem so the same
``IntendedFor`` list is written into all members of the group.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from ..inventory._time import parse_time_seconds

log = logging.getLogger(__name__)


_FMAP_SUFFIX_STRIP_RE = re.compile(
    r"_(magnitude[12]|phasediff|phase[12]|fieldmap|epi)$"
)


def populate_intended_for(
    subject_staging_dir: Path,
    subject: str,
    session: Optional[str] = None,
) -> int:
    """Update ``IntendedFor`` in every fmap sidecar under ``subject_staging_dir``.

    Parameters
    ----------
    subject_staging_dir
        The per-subject staging tree (``<bids_root>/.tmp_bidsmgr/sub-<id>/``).
    subject
        Subject label *without* the ``sub-`` prefix (e.g. ``"001"``). Used
        to build the BIDS URI.
    session
        Optional session label *without* ``ses-`` prefix. Currently unused
        — sessions are discovered from the staging layout — but kept for
        callers that already know the value and want to assert.

    Returns
    -------
    int
        Number of fmap JSON sidecars that were updated.
    """
    del session  # discovered from layout; kept for API symmetry with the orchestrator
    if not subject_staging_dir.is_dir():
        return 0

    n_updated = 0
    # Two layouts: ``<staging>/{fmap,func}`` (no-session) or
    # ``<staging>/ses-<label>/{fmap,func}`` (with sessions).
    session_dirs = [d for d in subject_staging_dir.glob("ses-*") if d.is_dir()]
    if session_dirs:
        for ses_dir in sorted(session_dirs):
            ses_label = ses_dir.name[len("ses-"):]  # strip "ses-"
            n_updated += _process_one_root(ses_dir, subject, ses_label)
    else:
        n_updated += _process_one_root(subject_staging_dir, subject, None)
    return n_updated


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def suggest_intended_for(
    root: Path, subject: str, session: Optional[str],
) -> tuple[list[tuple[list[Path], list[str]]], Optional[str]]:
    """What ``IntendedFor`` SHOULD say under ``root``, without writing it.

    Returns one ``(sidecars, uris)`` pair per fieldmap acquisition, plus the
    reason the assignment is the way it is, or ``None`` when there is
    nothing to say.

    Split out of :func:`populate_intended_for` so the Editor's linkage tool
    proposes exactly what the conversion would have written, rather than a
    second implementation of the same rule that can disagree with it.
    """
    fmap_dir = root / "fmap"
    func_dir = root / "func"
    if not (fmap_dir.is_dir() and func_dir.is_dir()):
        return [], None

    timed_runs, runs_missing_time = _collect_func_runs(func_dir)
    if not timed_runs and not runs_missing_time:
        return [], None

    timed_fmaps, fmap_missing_time = _collect_fieldmap_groups(fmap_dir)
    if not timed_fmaps and not fmap_missing_time:
        return [], None

    if runs_missing_time or fmap_missing_time:
        # Fallback: every run is a candidate for every fieldmap. Say so, or
        # the user cannot tell this apart from a protocol where one
        # fieldmap genuinely covers everything.
        all_runs = sorted(
            [img for img, _ in timed_runs] + runs_missing_time,
            key=lambda p: p.name,
        )
        uris = [
            _format_bids_uri(subject, session, "func", p.name) for p in all_runs
        ]
        groups = [members for _, members, _ in timed_fmaps] + fmap_missing_time
        nameless = [p.name for p in runs_missing_time] + [
            p.name for g in fmap_missing_time for p in g
        ]
        why = (
            "every run is offered to every fieldmap, because "
            f"{len(nameless)} file(s) carry no readable acquisition time "
            f"({', '.join(sorted(nameless)[:3])}"
            f"{', ...' if len(nameless) > 3 else ''})"
        )
        return [(g, uris) for g in groups], why

    # All timing present: a fieldmap covers the runs acquired after it and
    # before the next one.
    timed_runs.sort(key=lambda item: item[1])
    timed_fmaps.sort(key=lambda item: item[2])

    out: list[tuple[list[Path], list[str]]] = []
    for idx, (_key, members, fmap_time) in enumerate(timed_fmaps):
        next_time = (
            timed_fmaps[idx + 1][2] if idx + 1 < len(timed_fmaps) else float("inf")
        )
        uris = [
            _format_bids_uri(subject, session, "func", img.name)
            for img, acq in timed_runs
            if fmap_time <= acq < next_time
        ]
        out.append((members, uris))
    why = (
        f"{len(timed_fmaps)} fieldmap(s) and {len(timed_runs)} run(s) carry "
        "acquisition times, so each fieldmap takes the runs acquired after "
        "it and before the next one"
    )
    return out, why


def _process_one_root(
    root: Path, subject: str, session: Optional[str],
) -> int:
    """Update fmap sidecars under ``root`` (a sub-X or sub-X/ses-Y folder)."""
    n_updated = 0
    for members, uris in suggest_intended_for(root, subject, session)[0]:
        n_updated += _write_intended_for([members], uris)
    return n_updated


def _collect_func_runs(
    func_dir: Path,
) -> tuple[list[tuple[Path, float]], list[Path]]:
    """Return ``([(image, acq_time), …], [image_without_time, …])``."""
    timed: list[tuple[Path, float]] = []
    missing: list[Path] = []

    for image in sorted(func_dir.iterdir()):
        if not image.is_file():
            continue
        name = image.name
        # Only BOLD images participate in IntendedFor; skip references,
        # SBRef, aborted runs, and anything that's not a NIfTI.
        if not name.endswith((".nii", ".nii.gz")):
            continue
        if not _is_bold(name):
            continue

        meta = _load_json(_matching_json(image))
        acq_time = parse_time_seconds(
            meta.get("AcquisitionTime")
            or meta.get("acq_time")
            or meta.get("AcquisitionDateTime")
        )
        if acq_time is None:
            missing.append(image)
        else:
            timed.append((image, acq_time))
    return timed, missing


def _collect_fieldmap_groups(
    fmap_dir: Path,
) -> tuple[list[tuple[str, list[Path], float]], list[list[Path]]]:
    """Group fmap JSONs by acquisition; return timed + untimed groups.

    Members of the same group (e.g. magnitude1 + magnitude2 + phasediff
    of one fieldmap acquisition) get the same ``IntendedFor`` list.
    """
    grouped: dict[str, dict] = {}
    for sidecar in sorted(fmap_dir.glob("*.json")):
        key = _fieldmap_group_key(sidecar)
        meta = _load_json(sidecar)
        acq_time = parse_time_seconds(
            meta.get("AcquisitionTime")
            or meta.get("acq_time")
            or meta.get("AcquisitionDateTime")
        )
        slot = grouped.setdefault(key, {"members": [], "time": None})
        slot["members"].append(sidecar)
        if slot["time"] is None and acq_time is not None:
            slot["time"] = acq_time

    timed: list[tuple[str, list[Path], float]] = []
    untimed: list[list[Path]] = []
    for key, slot in grouped.items():
        if slot["time"] is None:
            untimed.append(slot["members"])
        else:
            timed.append((key, slot["members"], float(slot["time"])))
    return timed, untimed


def _fieldmap_group_key(json_path: Path) -> str:
    """Strip BIDS fmap suffix from the stem so siblings of one acquisition group."""
    stem = json_path.name
    if stem.endswith(".json"):
        stem = stem[: -len(".json")]
    return _FMAP_SUFFIX_STRIP_RE.sub("", stem)


def _format_bids_uri(
    subject: str, session: Optional[str], datatype: str, filename: str,
) -> str:
    """Return ``bids::sub-X[/ses-Y]/<datatype>/<filename>``."""
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.append(datatype)
    parts.append(filename)
    return "bids::" + "/".join(parts)


def _is_bold(name: str) -> bool:
    """Return True if ``name`` looks like a BOLD acquisition we should bind to.

    Excludes single-band reference (``_sbref``) and aborted runs
    (``_bold_aborted`` from v0.2.5; not produced by dcm2niix-direct but
    cheap to keep guarded).
    """
    lower = name.lower()
    if "_sbref" in lower:
        return False
    if "_bold_aborted" in lower:
        return False
    # Strip extension(s) and check the trailing BIDS suffix.
    stem = re.sub(r"\.nii(\.gz)?$", "", lower)
    return stem.endswith("_bold")


def _matching_json(image: Path) -> Path:
    """Return the JSON sidecar path for a NIfTI image."""
    if image.suffix == ".gz" and image.name.endswith(".nii.gz"):
        return image.with_suffix("").with_suffix(".json")
    return image.with_suffix(".json")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return {}


def _write_intended_for(
    groups: list[list[Path]], intended_for: list[str],
) -> int:
    """Write ``IntendedFor`` into each sidecar in ``groups``. Returns count written."""
    n = 0
    for group in groups:
        for sidecar in group:
            meta = _load_json(sidecar)
            meta["IntendedFor"] = intended_for
            try:
                with open(sidecar, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=4)
                    fh.write("\n")
                n += 1
                log.info(
                    "IntendedFor: wrote %d entries into %s",
                    len(intended_for), sidecar.name,
                )
            except OSError as exc:
                log.warning("could not write %s: %s", sidecar, exc)
    return n


__all__ = ["populate_intended_for"]

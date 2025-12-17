#!/usr/bin/env python3
"""
post_fmap_rename.py — Fieldmap Renamer (PyCharm-friendly)
---------------------------------------------------------
This script renames fieldmap files in a BIDS dataset so that:
  - echo-1 → _magnitude1
  - echo-2 → _magnitude2
  - plain _fmap → _phasediff
It also **removes** the trailing `_fmap` from the filenames and moves any
``_rep-<n>`` suffix to the end (e.g. ``magnitude1_rep-2``).
Both ``.nii``/``.nii.gz`` images and their JSON sidecars are handled.

After renaming, each fieldmap JSON gains an ``IntendedFor`` field listing
all functional runs in the same subject/session. This allows fMRIPrep and
other BIDS apps to correctly associate fieldmaps with their target EPI
images.

Usage in PyCharm:
  1. Open this script in PyCharm.
  2. Set the BIDS_ROOT path below to your dataset directory.
  3. Run this script (e.g., click ▶️ in the editor).

No CLI arguments required.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Configuration: EDIT this path to point to your BIDS dataset
# -----------------------------------------------------------------------------
BIDS_ROOT = Path("/path/to/your/BIDS_dataset")

# -----------------------------------------------------------------------------
# Rename rules based on filename patterns
# -----------------------------------------------------------------------------
RENAME_RULES = [
    # echo-1 → magnitude1
    (re.compile(r"echo[-_]?1", re.I), "magnitude1"),
    # echo-2 → magnitude2
    (re.compile(r"echo[-_]?2", re.I), "magnitude2"),
]
# Match plain '_fmap' before .nii, .nii.gz or .json
FMAP_SUFFIX_RE = re.compile(r"_fmap(?=(\.nii(?:\.gz)?|\.json)$)", re.I)
BOLD_RENAME_RULES = [
    # heudiconv sometimes emits ``_bold1`` and ``_bold2`` suffixes that are not
    # BIDS compliant.  ``_bold1`` is the real run, ``_bold2`` is a residual
    # aborted run that we want to keep but clearly mark as aborted.
    (re.compile(r"_bold1(?=(\.nii(?:\.gz)?|\.json)$)", re.I), "_bold"),
    (re.compile(r"_bold2(?=(\.nii(?:\.gz)?|\.json)$)", re.I), "_bold_aborted"),
]


def _move_rep_suffix(name: str) -> str:
    """Ensure ``_rep-N`` appears after magnitude/phase suffix."""
    name = re.sub(r"(_rep-\d+)(_magnitude[12])", r"\2\1", name)
    name = re.sub(r"(_rep-\d+)(_phasediff)", r"\2\1", name)
    return name

# -----------------------------------------------------------------------------
# Process a single fmap directory
# -----------------------------------------------------------------------------
def process_fmap_dir(fmap_dir: Path) -> None:
    """Rename fieldmap files within ``fmap_dir`` according to BIDS rules."""
    for file in sorted(fmap_dir.iterdir()):
        if not file.is_file():
            continue
        name = file.name
        # apply echo rules
        for pattern, replacement in RENAME_RULES:
            if pattern.search(name) and name.lower().endswith(('.nii', '.nii.gz', '.json')):
                # replace echo tag
                interim = pattern.sub(replacement, name)
                # remove trailing _fmap before extension
                new_name = FMAP_SUFFIX_RE.sub('', interim)
                new_name = _move_rep_suffix(new_name)
                file.rename(fmap_dir / new_name)
                print(f"Renamed: {name} → {new_name}")
                break
        else:
            # apply phase rule for plain fmap (no echo)
            if name.lower().endswith(('.nii', '.nii.gz', '.json')) and '_fmap' in name and not any(rep in name.lower() for rep in ['magnitude1', 'magnitude2']):
                # replace _fmap with _phasediff
                new_name = name.replace('_fmap', '_phasediff')
                new_name = _move_rep_suffix(new_name)
                file.rename(fmap_dir / new_name)
                print(f"Renamed: {name} → {new_name}")


def _rename_bold_variants(func_dir: Path) -> None:
    """Normalise heudiconv ``_bold1``/``_bold2`` outputs in a ``func`` folder."""

    for file in sorted(func_dir.iterdir()):
        if not file.is_file():
            continue

        name = file.name
        if not name.lower().endswith((".nii", ".nii.gz", ".json")):
            continue

        for pattern, replacement in BOLD_RENAME_RULES:
            if not pattern.search(name):
                continue

            new_name = pattern.sub(replacement, name)
            new_path = func_dir / new_name

            # Avoid overwriting an existing file; keep the original if there is
            # a naming collision to preserve data rather than risk loss.
            if new_path.exists():
                print(
                    f"Skipping rename of {name} because {new_path.name} already exists."
                )
                break

            file.rename(new_path)
            print(f"Renamed: {name} → {new_name}")
            break


def rename_bold_sequences(bids_root: Path) -> None:
    """Apply BOLD renaming across all ``func`` directories in ``bids_root``."""

    for sub in bids_root.glob("sub-*"):
        if not sub.is_dir():
            continue

        session_dirs = [ses for ses in sub.glob("ses-*") if ses.is_dir()]
        roots = session_dirs or [sub]

        for root in roots:
            func_dir = root / "func"
            if func_dir.is_dir():
                _rename_bold_variants(func_dir)

# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def post_fmap_rename(bids_root: Path) -> None:
    """Walk ``bids_root`` and apply :func:`process_fmap_dir` to each ``fmap`` folder."""
    if not bids_root.is_dir():
        print(f"Error: '{bids_root}' is not a directory", file=sys.stderr)
        return
    fmap_dirs = list(bids_root.rglob('fmap'))
    if not fmap_dirs:
        print(f"No 'fmap' directories found under {bids_root}")
    else:
        for fmap_dir in fmap_dirs:
            process_fmap_dir(fmap_dir)

    # Normalise BOLD outputs before computing ``IntendedFor`` so that aborted
    # runs are clearly marked and excluded from associations.
    rename_bold_sequences(bids_root)

    # After renaming, populate ``IntendedFor`` in the fieldmap sidecars so
    # downstream tools know which functional runs they apply to.
    add_intended_for(bids_root)

    # Finally, refresh filenames recorded in ``*_scans.tsv`` to match the new
    # fieldmap file names.
    update_scans_tsv(bids_root)


def _parse_acq_time(value: Optional[str]) -> Optional[float]:
    """Convert various acquisition time representations to sortable seconds.

    The GUI exposes acquisition metadata as strings that may either follow the
    DICOM ``HHMMSS(.ffffff)`` convention, be colon separated, or be expressed as
    ISO date-times.  Returning a float keeps ordering stable while remaining
    agnostic to the exact formatting originally stored in the JSON/TSV files.
    """

    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    # ``datetime.fromisoformat`` gracefully handles values containing dates,
    # optional timezone offsets, or fractional seconds.
    cleaned = text.rstrip("Z")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        # Normalise plain ``HHMMSS`` or ``HHMMSS.ffffff`` strings by inserting
        # the missing colons so that ``strptime`` can understand them.
        if re.fullmatch(r"\d{6}(?:\.\d+)?", cleaned):
            cleaned = f"{cleaned[0:2]}:{cleaned[2:4]}:{cleaned[4:]}"

        for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue
        else:
            return None

        # Anchor the time to seconds from midnight for consistent ordering.
        return (
            dt.hour * 3600
            + dt.minute * 60
            + dt.second
            + dt.microsecond / 1_000_000
        )

    # When a date is present we can rely on the absolute timestamp for
    # ordering.  ``timestamp`` requires ``tzinfo``; treat naïve datetimes as
    # local without conversion which is sufficient for relative comparisons.
    if dt.tzinfo:
        return dt.timestamp()
    return (
        dt.hour * 3600
        + dt.minute * 60
        + dt.second
        + dt.microsecond / 1_000_000
    )


def _load_json_metadata(path: Path) -> Dict:
    """Return the JSON metadata stored in ``path`` if available."""

    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _matching_json(image: Path) -> Path:
    """Return the JSON sidecar corresponding to ``image`` (nii or nii.gz)."""

    if image.suffix == ".gz" and image.name.endswith(".nii.gz"):
        return image.with_suffix("").with_suffix(".json")
    return image.with_suffix(".json")


def _fieldmap_group_key(json_path: Path) -> str:
    """Derive a stable group identifier for fieldmap components.

    Fieldmap acquisitions typically produce ``magnitude1``, ``magnitude2`` and a
    phase-derived file.  Grouping by the file stem without those suffixes lets us
    assign the same ``IntendedFor`` list to all three sidecars.
    """

    stem = json_path.stem
    stem = re.sub(r"_magnitude[12]$", "", stem)
    stem = re.sub(r"_phasediff$", "", stem)
    stem = re.sub(r"_phase[12]$", "", stem)
    return stem


def _format_intended_for(path: Path, bids_root: Path) -> str:
    """Convert ``path`` into the ``bids::`` reference expected by the GUI."""

    rel = path.relative_to(bids_root).as_posix()
    return f"bids::{rel}"


def _collect_func_runs(func_dir: Path) -> Tuple[List[Tuple[Path, float]], List[Path]]:
    """Gather BOLD images and their acquisition times.

    Returns
    -------
    tuple
        A tuple containing:
        1. A list of ``(image_path, acq_time_seconds)`` pairs for runs that
           possess timing information.
        2. A list of image paths lacking timing data so the caller can decide on
           a fallback strategy.
    """

    timed_runs: List[Tuple[Path, float]] = []
    missing_time: List[Path] = []

    for image in sorted(func_dir.glob("*.nii*")):
        name_lower = image.name.lower()
        if "ref" in name_lower or "_bold_aborted" in name_lower:
            # Skip reference volumes and aborted runs.
            continue

        if not name_lower.endswith("bold.nii") and not name_lower.endswith("bold.nii.gz"):
            # Skip reference volumes and non-BOLD acquisitions.
            continue

        meta = _load_json_metadata(_matching_json(image))
        acq_time = _parse_acq_time(
            meta.get("acq_time")
            or meta.get("AcquisitionTime")
            or meta.get("AcquisitionDateTime")
        )

        if acq_time is None:
            missing_time.append(image)
        else:
            timed_runs.append((image, acq_time))

    return timed_runs, missing_time


def _collect_fieldmaps(
    fmap_dir: Path
) -> Tuple[List[Tuple[str, List[Path], float]], List[List[Path]]]:
    """Group fieldmap JSON files by acquisition and capture timing data."""

    grouped: Dict[str, Dict[str, object]] = {}
    missing: List[List[Path]] = []

    for sidecar in sorted(fmap_dir.glob("*.json")):
        key = _fieldmap_group_key(sidecar)
        meta = _load_json_metadata(sidecar)
        acq_time = _parse_acq_time(
            meta.get("acq_time")
            or meta.get("AcquisitionTime")
            or meta.get("AcquisitionDateTime")
        )

        group = grouped.setdefault(key, {"members": [], "time": None})
        group["members"].append(sidecar)
        if group["time"] is None and acq_time is not None:
            group["time"] = acq_time

    timed_groups: List[Tuple[str, List[Path], float]] = []

    for key, info in grouped.items():
        members: List[Path] = info["members"]  # type: ignore[assignment]
        time_val = info.get("time")
        if time_val is None:
            missing.append(members)
        else:
            timed_groups.append((key, members, float(time_val)))

    return timed_groups, missing


def _update_intended_for(root: Path, bids_root: Path) -> None:
    """Add ``IntendedFor`` entries to fieldmap JSONs under ``root``."""
    # ``root`` points to either ``sub-<id>`` or ``sub-<id>/ses-<id>``
    # within the BIDS dataset. The function expects ``fmap`` and ``func``
    # directories side-by-side inside this folder.
    fmap_dir = root / "fmap"
    func_dir = root / "func"

    # Skip if either directory does not exist (e.g. no functional runs).
    if not (fmap_dir.is_dir() and func_dir.is_dir()):
        return

    timed_runs, runs_missing_time = _collect_func_runs(func_dir)
    if not timed_runs and not runs_missing_time:
        return

    timed_fmaps, fmap_missing_time = _collect_fieldmaps(fmap_dir)
    if not timed_fmaps and not fmap_missing_time:
        return

    # Decide whether we can honour the acquisition-based association.  If any
    # run or fieldmap lacks timing data we fall back to the legacy behaviour of
    # linking every fieldmap to every run.  This preserves previous
    # functionality rather than making incorrect assumptions.
    if runs_missing_time or fmap_missing_time:
        all_runs = sorted([image for image, _ in timed_runs] + runs_missing_time)
        rel_paths = [
            _format_intended_for(image, bids_root)
            for image in all_runs
        ]

        for group in [members for _, members, _ in timed_fmaps] + fmap_missing_time:
            for sidecar in group:
                meta = _load_json_metadata(sidecar)
                meta["IntendedFor"] = rel_paths
                with open(sidecar, "w", encoding="utf-8") as handle:
                    json.dump(meta, handle, indent=4)
                    handle.write("\n")
                print(f"Updated IntendedFor in {sidecar.relative_to(bids_root)}")
        return

    # With reliable timing information, assign each fieldmap to the functional
    # runs acquired after it until the next fieldmap occurs.
    timed_runs.sort(key=lambda item: item[1])
    timed_fmaps.sort(key=lambda item: item[2])

    run_queue: List[Tuple[Path, float]] = list(timed_runs)

    for idx, (_, members, fmap_time) in enumerate(timed_fmaps):
        next_time = timed_fmaps[idx + 1][2] if idx + 1 < len(timed_fmaps) else float("inf")
        intended = [
            _format_intended_for(image, bids_root)
            for image, acq in run_queue
            if fmap_time <= acq < next_time
        ]

        for sidecar in members:
            meta = _load_json_metadata(sidecar)
            meta["IntendedFor"] = intended
            with open(sidecar, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=4)
                handle.write("\n")
            print(f"Updated IntendedFor in {sidecar.relative_to(bids_root)}")


def add_intended_for(bids_root: Path) -> None:
    """Populate ``IntendedFor`` in all fieldmap JSONs."""
    # Walk through all subjects and sessions in the dataset. ``_update_intended_for``
    # handles the actual JSON editing for each folder.
    for sub in bids_root.glob("sub-*"):
        if not sub.is_dir():
            continue
        sessions = [s for s in sub.glob("ses-*") if s.is_dir()]
        if sessions:
            for ses in sessions:
                _update_intended_for(ses, bids_root)
        else:
            _update_intended_for(sub, bids_root)


def _rename_in_scans(tsv: Path, bids_root: Path) -> None:
    """Update file names in a single ``*_scans.tsv`` if needed."""
    import pandas as pd

    df = pd.read_csv(tsv, sep="\t")
    if "filename" not in df.columns:
        return

    changed = False
    for idx, fname in enumerate(df["filename"]):
        path = Path(fname)
        new_name = path.name

        # Keep ``*_scans.tsv`` in sync with BOLD renaming so the references
        # match the files on disk.
        for pattern, replacement in BOLD_RENAME_RULES:
            if pattern.search(new_name) and new_name.lower().endswith((".nii", ".nii.gz", ".json")):
                new_name = pattern.sub(replacement, new_name)
                break

        if "fmap" not in path.parts:
            if new_name != path.name:
                candidate = tsv.parent / path.parent / new_name
                if candidate.exists():
                    df.at[idx, "filename"] = (path.parent / new_name).as_posix()
                    changed = True
            continue
        
        for pattern, replacement in RENAME_RULES:
            if pattern.search(new_name) and new_name.lower().endswith((".nii", ".nii.gz", ".json")):
                interim = pattern.sub(replacement, new_name)
                new_name = FMAP_SUFFIX_RE.sub("", interim)
                new_name = _move_rep_suffix(new_name)
                break
        else:
            if new_name.lower().endswith((".nii", ".nii.gz", ".json")) and "_fmap" in new_name and not any(rep in new_name.lower() for rep in ["magnitude1", "magnitude2"]):
                new_name = new_name.replace("_fmap", "_phasediff")
                new_name = _move_rep_suffix(new_name)

        if new_name != path.name:
            candidate = tsv.parent / path.parent / new_name
            if candidate.exists():
                df.at[idx, "filename"] = (path.parent / new_name).as_posix()
                changed = True

    if changed:
        df.to_csv(tsv, sep="\t", index=False)
        print(f"Updated {tsv.relative_to(bids_root)}")


def update_scans_tsv(bids_root: Path) -> None:
    """Refresh filenames inside all ``*_scans.tsv`` files."""
    for sub in bids_root.glob("sub-*"):
        if not sub.is_dir():
            continue
        sessions = [s for s in sub.glob("ses-*") if s.is_dir()]
        roots = sessions or [sub]
        for root in roots:
            for tsv in root.glob("*_scans.tsv"):
                _rename_in_scans(tsv, bids_root)

# -----------------------------------------------------------------------------
# Run immediately when executed
# -----------------------------------------------------------------------------
def main() -> None:
    """CLI wrapper around :func:`post_fmap_rename`."""

    import argparse

    parser = argparse.ArgumentParser(description="Rename BIDS fieldmap files")
    parser.add_argument('bids_root', help='Path to BIDS dataset root')
    args = parser.parse_args()

    bids_root = Path(args.bids_root)
    print(f"Starting fieldmap rename in: {bids_root}")
    post_fmap_rename(bids_root)
    print("Done.")


if __name__ == '__main__':
    main()


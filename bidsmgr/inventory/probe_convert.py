"""Probe-convert: dcm2niix one *detected sequence* at a time and inspect
the real outputs to verify / improve BIDS name predictions.

The probe runs **per inventory row** (one ``SeriesInstanceUID`` per
invocation). It does not convert whole DICOM source folders at once —
that would (a) re-process scout / report / discard series we never
intend to keep, and (b) hide the ownership of each output file in the
shared per-folder dump. Instead, for every row that the user is likely
to commit to the BIDS root we:

1. Build a per-series staging directory of symlinks (one per source
   DICOM file belonging to that ``SeriesInstanceUID``).
2. Invoke dcm2niix against the staging directory.
3. Walk the output and group every produced file by SeriesInstanceUID
   (read from the JSON sidecar, since the filename token ``%j`` may be
   suffixed with multi-echo / phase markers).

The work directory uses a hidden ``.tmp`` convention by default
(architecture.md §7) — the staging trees are scratch, the user shouldn't
need to look at them and they shouldn't sit alongside the user's outputs
under a visible name.

Outputs lay out as::

    <work_root>/
      <bids_id>/
        <series_uid>/
          <series_uid>.nii.gz
          <series_uid>.json
          ... etc.

so a researcher can inspect per-subject per-series outputs side by side
with the originating row in the inventory TSV.

Reference: architecture.md §7 (converter backends), improvement_plan.md
§ post-conversion auditing.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from joblib import Parallel, delayed

from bidsmgr.util.paths import long_path

from ..classifier.dcm2niix_bidsguess import find_dcm2niix
from .types import InventoryRow

log = logging.getLogger(__name__)


@dataclass
class ProbeFileStats:
    """Output statistics for one DICOM series (one ``SeriesInstanceUID``)."""

    series_uid: str
    output_files: list[str] = field(default_factory=list)
    n_files: int = 0
    n_nifti: int = 0
    n_volumes_max: int = 0
    extensions: list[str] = field(default_factory=list)
    sidecar_count: int = 0


def _run_dcm2niix_full(
    dicom_dir: Path,
    output_dir: Path,
    *,
    dcm2niix_bin: Optional[Path] = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess:
    """Invoke dcm2niix in full conversion mode against ``dicom_dir``.

    Flags:

    * ``-b y``  : write JSON sidecars
    * ``-ba n`` : do not anonymize sidecars (keeps SeriesInstanceUID)
    * ``-z y``  : gzip the NIfTIs
    * ``-f %j`` : filename = SeriesInstanceUID (with multi-echo / phase
                  splits dcm2niix appends ``_e1``, ``_e2``, ``_ph`` etc.)
    """

    binary = str(dcm2niix_bin or find_dcm2niix())
    cmd = [
        binary,
        "-b", "y",
        "-ba", "n",
        "-z", "y",
        "-o", str(output_dir),
        "-f", "%j",
        str(dicom_dir),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


_BIDS_EXT_PRIORITY = (".nii.gz", ".nii", ".bval", ".bvec", ".tsv", ".tsv.gz")


def _strip_bids_ext(name: str) -> tuple[str, str]:
    for ext in _BIDS_EXT_PRIORITY:
        if name.endswith(ext):
            return name[: -len(ext)], ext
    if "." in name:
        head, tail = name.rsplit(".", 1)
        return head, "." + tail
    return name, ""


def _read_dim4(path: Path) -> int:
    """Return time dimension of a NIfTI file, or 0 if not 4D / unreadable."""
    try:
        import nibabel as nib  # type: ignore
    except ImportError:
        return 0
    try:
        img = nib.load(str(path))
        if img.ndim >= 4:
            return int(img.shape[3])
        return 1 if img.ndim == 3 else 0
    except Exception as exc:
        log.debug("nibabel could not read %s: %s", path, exc)
        return 0


def collect_probe_stats(directory: Path) -> dict[str, ProbeFileStats]:
    """Walk ``directory``, group all dcm2niix outputs by SeriesInstanceUID.

    Returns ``{series_uid: ProbeFileStats}``. Files whose JSON sidecar
    doesn't carry a SeriesInstanceUID are skipped.
    """

    by_uid: dict[str, ProbeFileStats] = {}

    for json_path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.debug("could not read sidecar %s: %s", json_path, exc)
            continue
        uid = data.get("SeriesInstanceUID")
        if not uid:
            continue

        stats = by_uid.setdefault(uid, ProbeFileStats(series_uid=uid))
        stats.output_files.append(json_path.name)
        stats.sidecar_count += 1

        stem = json_path.name[: -len(".json")]

        for ext in _BIDS_EXT_PRIORITY:
            companion = directory / f"{stem}{ext}"
            if companion.exists():
                stats.output_files.append(companion.name)
                if ext in (".nii.gz", ".nii"):
                    stats.n_nifti += 1
                    d4 = _read_dim4(companion)
                    if d4 > stats.n_volumes_max:
                        stats.n_volumes_max = d4

    for stats in by_uid.values():
        stats.n_files = len(stats.output_files)
        ext_set: set[str] = set()
        for fname in stats.output_files:
            _, ext = _strip_bids_ext(fname)
            if ext:
                ext_set.add(ext)
        stats.extensions = sorted(ext_set)

    return by_uid


def _stage_series(
    series_uid: str,
    file_paths: list[str],
    staging_dir: Path,
) -> int:
    """Symlink every DICOM of one SeriesInstanceUID into ``staging_dir``.

    Returns the number of files staged. Existing entries are removed and
    re-linked so re-running the probe is idempotent.

    Windows notes (mirrors ``dcm2niix_direct._stage_dicoms``):

    * ``os.symlink`` requires ``SeCreateSymbolicLinkPrivilege`` (granted
      by Developer Mode) or Administrator on Windows. When the syscall
      fails with ``WinError 1314`` this function switches the rest of
      the series to ``shutil.copyfile`` so non-admin Windows users still
      get a working probe pass.
    * Staged filenames are short, zero-padded sequential names rather
      than the source name. Vendor Siemens/GE DICOMs are ~80 chars and
      easily push the full Win32 path past the 260-char ``MAX_PATH``
      ceiling inside a deep staging tree, which makes dcm2niix exit
      ``rc=2`` without a useful error. dcm2niix orders frames by DICOM
      tags (InstanceNumber etc.), not by filename, so renaming on stage
      is safe and removes the need for collision handling.
    """

    if staging_dir.exists():
        shutil.rmtree(long_path(staging_dir), ignore_errors=False)
    staging_dir.mkdir(parents=True)
    n = 0
    use_copy_fallback = False
    for idx, fp in enumerate(file_paths):
        src = Path(fp)
        if not src.exists():
            continue
        link_name = f"{idx:06d}{src.suffix or '.dcm'}"
        link = staging_dir / link_name
        src_resolved = src.resolve()
        if use_copy_fallback:
            shutil.copyfile(long_path(src_resolved), long_path(link))
            n += 1
            continue
        try:
            os.symlink(long_path(src_resolved), long_path(link))
        except FileExistsError:
            os.unlink(long_path(link))
            os.symlink(long_path(src_resolved), long_path(link))
        except OSError as exc:
            # WinError 1314: SeCreateSymbolicLinkPrivilege missing.
            # Switch to copy-mode for the rest of this series.
            if getattr(exc, "winerror", None) == 1314:
                use_copy_fallback = True
                shutil.copyfile(long_path(src_resolved), long_path(link))
            else:
                raise
        n += 1
    return n


def _should_probe_row(row: InventoryRow) -> bool:
    """Decide whether a row is worth converting in the probe pass.

    Skips low-level / discard rows where dcm2niix conversion either
    consistently fails or produces non-BIDS output that the user wouldn't
    convert anyway. Concretely: scout / localizer / report / physio
    sequences identified by the legacy fine-modality field. The CLI
    flow can pass the planner verdict here once the planner exists.
    """

    if row.modality != "mri" or not row.series_uid:
        return False
    fine = (row.fine_modality or "").lower()
    skip_modalities = {"scout", "report", "physio"}
    if fine in skip_modalities:
        return False
    return True


def _probe_one_series(
    uid: str,
    files: list[str],
    series_workdir: Path,
    dcm2niix_bin: str,
) -> tuple[str, Optional[ProbeFileStats]]:
    """Worker: stage one series's DICOMs, run dcm2niix, parse outputs.

    Pure function — accepts only picklable args so it can run inside a
    joblib worker process. Logs go through the standard ``log`` handle of
    this module; the worker's stdout/stderr are merged by joblib.
    """

    staging_dir = series_workdir / "_dicoms"
    n = _stage_series(uid, files, staging_dir)
    if n == 0:
        log.warning("probe: staging directory empty for UID %s", uid)
        return uid, None

    output_dir = series_workdir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    proc = _run_dcm2niix_full(staging_dir, output_dir, dcm2niix_bin=Path(dcm2niix_bin))
    if proc.returncode != 0:
        log.info(
            "probe: dcm2niix returncode=%s for UID %s; stderr tail=%s",
            proc.returncode, uid, proc.stderr[-500:],
        )
    stats = collect_probe_stats(output_dir)

    if uid not in stats:
        log.info(
            "probe: UID %s produced no sidecars (dcm2niix may have rejected "
            "the series; staging=%s)", uid, staging_dir,
        )
        return uid, None
    return uid, stats[uid]


def probe_rows(
    rows: Iterable[InventoryRow],
    work_root: Path,
    files_by_uid: dict[str, list[str]],
    *,
    dcm2niix_bin: Optional[Path] = None,
    n_jobs: int = 1,
) -> dict[str, ProbeFileStats]:
    """Run a per-series dcm2niix conversion for every row worth probing.

    Parameters
    ----------
    rows
        Inventory rows produced by the scanner. Each row's
        ``series_uid`` is used to locate its source DICOMs in
        ``files_by_uid``.
    work_root
        Hidden temp directory (``.tmp`` by convention) where per-series
        staging directories and dcm2niix outputs are written.
    files_by_uid
        Mapping ``SeriesInstanceUID -> list[absolute_dicom_path]``,
        produced by ``mri_dicom.scan_dicoms_long`` and stashed on
        ``df.attrs["files_by_uid"]``.
    n_jobs
        Number of dcm2niix invocations to run concurrently (joblib).
        ``1`` (default) runs serially; ``-1`` uses all CPUs. Each series
        is independent (separate staging dir + separate output dir) so
        parallelism scales linearly with available cores.

    Returns ``{series_uid: ProbeFileStats}`` — only series that were
    actually probed and produced sidecars.
    """

    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    bin_path = str(dcm2niix_bin or find_dcm2niix())

    # Build the task list deterministically so the cache layout under
    # ``work_root`` is stable across re-runs.
    tasks: list[tuple[str, list[str], Path]] = []
    seen_uids: set[str] = set()
    for row in rows:
        if not _should_probe_row(row):
            continue
        uid = row.series_uid or ""
        if not uid or uid in seen_uids:
            continue
        seen_uids.add(uid)
        files = files_by_uid.get(uid, [])
        if not files:
            log.debug("probe: no source files recorded for UID %s", uid)
            continue
        subject_dir = (row.subject_hint or "unknown")
        series_workdir = work_root / f"sub-{subject_dir}" / uid
        tasks.append((uid, files, series_workdir))

    if not tasks:
        return {}

    log.info(
        "probe: %d series queued for conversion (n_jobs=%s)", len(tasks), n_jobs,
    )

    # joblib's loky backend forks workers; the worker function and
    # arguments must be picklable. Path / str / list[str] all are. nibabel
    # is imported lazily inside collect_probe_stats so cold workers don't
    # pay the import cost when no NIfTI 4D check is needed.
    results = Parallel(n_jobs=n_jobs)(
        delayed(_probe_one_series)(uid, files, series_workdir, bin_path)
        for uid, files, series_workdir in tasks
    )

    out: dict[str, ProbeFileStats] = {}
    for uid, stats in results:
        if stats is not None:
            out[uid] = stats
    return out


__all__ = [
    "ProbeFileStats",
    "collect_probe_stats",
    "probe_rows",
]

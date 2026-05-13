"""Default MRI converter backend: invoke ``dcm2niix`` directly.

The orchestrator (``cli/convert.py``) drives parallelism — one
``ConvertTask`` per call, joblib pool sized by ``-j``. This backend is
synchronous and stateless; ``convert(task, staging_dir)`` runs one
dcm2niix invocation and reports back what landed on disk.

Layout written by ``convert``::

    <staging_dir>/
      _dicoms_<series_uid>/      (symlinks to the source DICOMs)
      <datatype>/
        <basename>.nii.gz
        <basename>.json
        <basename>.bval           (DWI only)
        <basename>.bvec           (DWI only)
        <basename>_e1.nii.gz      (fmap multi-echo: renamed by fixups/fieldmaps)
        <basename>_e2.nii.gz
        <basename>_ph.nii.gz

The fmap suffix mapping (``_e1`` / ``_e2`` / ``_ph`` →
``magnitude1`` / ``magnitude2`` / ``phasediff``) is applied by
``fixups/fieldmaps.py`` *after* this backend returns, before the
atomic per-subject commit. ``IntendedFor`` lives in
``fixups/intended_for.py`` for the same reason.

Reference: architecture.md §7. Decision log §13: dcm2niix invoked
directly, no wrapper layer.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from ...classifier.dcm2niix_bidsguess import find_dcm2niix
from ...util.paths import long_path
from ..types import ConvertResult, ConvertTask

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 1800
_STDERR_TAIL_BYTES = 4096


class Dcm2niixDirect:
    """Per-series dcm2niix backend.

    Constructed once by the registry; called once per task. The binary
    path is resolved at construction (so a missing ``dcm2niix`` fails
    fast, before any subject-level work begins).
    """

    name = "dcm2niix_direct"

    def __init__(self, dcm2niix_bin: Optional[Path] = None) -> None:
        self._bin = Path(dcm2niix_bin) if dcm2niix_bin else find_dcm2niix()

    @property
    def binary(self) -> Path:
        return self._bin

    def can_handle(self, task: ConvertTask) -> bool:
        if task.suffix == "physio":
            # Physio rows route to ``PhysioDcmBackend`` (bidsphysio
            # wrapper). dcm2niix can't produce ``_physio.tsv.gz``.
            return False
        if task.datatype in {"eeg", "meg", "ieeg", "nirs"}:
            # EEG/MEG/iEEG/NIRS rows route to ``MneBidsBackend``.
            # dcm2niix doesn't read raw electrophysiology formats.
            return False
        return bool(task.source_files) and bool(task.basename)

    def convert(self, task: ConvertTask, staging_dir: Path) -> ConvertResult:
        """Run dcm2niix for one series; report what landed in staging.

        Failure modes captured in ``ConvertResult.error``:

        * ``"empty staging"`` — none of ``task.source_files`` exist.
        * ``"dcm2niix failed: rc=<n>"`` — dcm2niix exited non-zero.
        * ``"missing expected output: <ext>"`` — required extension
          (``.nii.gz`` or ``.json``) absent after a 0-exit run.

        Multi-output cases (fmap, multi-echo) are *not* failures here —
        the extra files are reported in ``staged_files`` and the
        downstream fmap fixup deals with the suffix mapping.
        """

        t0 = time.monotonic()
        try:
            return self._convert_inner(task, staging_dir, t0)
        except subprocess.TimeoutExpired as exc:
            return ConvertResult(
                task=task, success=False,
                error=f"dcm2niix timed out after {exc.timeout}s",
                duration_s=time.monotonic() - t0,
            )
        except Exception as exc:
            log.exception("dcm2niix_direct: unexpected error for %s", task.basename)
            return ConvertResult(
                task=task, success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - t0,
            )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _convert_inner(
        self, task: ConvertTask, staging_dir: Path, t0: float,
    ) -> ConvertResult:
        # Per-series symlink fan-out. Sibling staging dirs (one per
        # series) live alongside the datatype tree so nothing collides
        # when multiple tasks for the same subject run in parallel.
        # The dir name is a short hash of ``task.series_uid`` rather
        # than the UID itself: (a) fmap rows carry a ``|``-joined pair
        # of UIDs which is an illegal Windows path character, and
        # (b) a raw UID is ~64 chars (130 for a pair), which pushes
        # deep Windows BIDS trees past the 260-char ``MAX_PATH`` limit
        # and trips dcm2niix with ``rc=2``.
        dicoms_dir = staging_dir / _safe_dicoms_dirname(task.series_uid)
        n_staged = _stage_dicoms(task.source_files, dicoms_dir)
        if n_staged == 0:
            return ConvertResult(
                task=task, success=False,
                error="empty staging: no source DICOMs were accessible",
                duration_s=time.monotonic() - t0,
            )

        output_dir = staging_dir / task.datatype
        output_dir.mkdir(parents=True, exist_ok=True)

        proc = _run_dcm2niix(
            self._bin, dicoms_dir, output_dir, task.basename,
        )
        stderr_tail = (proc.stderr or "")[-_STDERR_TAIL_BYTES:]

        # Best-effort cleanup of the per-series symlink dir; not fatal.
        shutil.rmtree(dicoms_dir, ignore_errors=True)

        if proc.returncode != 0:
            return ConvertResult(
                task=task, success=False,
                error=f"dcm2niix failed: rc={proc.returncode}",
                dcm2niix_returncode=proc.returncode,
                dcm2niix_stderr_tail=stderr_tail,
                duration_s=time.monotonic() - t0,
            )

        staged = _collect_outputs(output_dir, task.basename)

        missing = _missing_expected(staged, task.expected_outputs, task.basename, output_dir)
        if missing:
            return ConvertResult(
                task=task, staged_files=tuple(staged), success=False,
                error=f"missing expected output(s): {', '.join(missing)}",
                dcm2niix_returncode=proc.returncode,
                dcm2niix_stderr_tail=stderr_tail,
                duration_s=time.monotonic() - t0,
            )

        return ConvertResult(
            task=task, staged_files=tuple(staged), success=True,
            dcm2niix_returncode=proc.returncode,
            dcm2niix_stderr_tail=stderr_tail,
            duration_s=time.monotonic() - t0,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (picklable for joblib workers; reused by tests)
# ---------------------------------------------------------------------------


def _safe_dicoms_dirname(series_uid: str) -> str:
    """Return the per-series staging dir name as ``_dicoms_<hash>``.

    The raw ``series_uid`` is unsuitable as a directory component:

    * fmap rows collapse a magnitude/phase pair into one row whose
      ``series_uid`` is the two UIDs joined by ``|`` — illegal in
      Windows path components.
    * A single UID is ~64 chars and a pair ~130; combined with deep
      Windows BIDS trees this pushes the full staging path past the
      260-char ``MAX_PATH`` limit and dcm2niix exits ``rc=2``.

    A 12-hex SHA-1 prefix is unique in practice (one subject's batch
    has on the order of 10² series — collision probability ≈ 2⁻⁴⁰)
    and keeps the dir at 20 chars including the ``_dicoms_`` prefix.
    """
    digest = hashlib.sha1(series_uid.encode("utf-8")).hexdigest()[:12]
    return f"_dicoms_{digest}"


def _stage_dicoms(source_files, staging_dir: Path) -> int:
    """Symlink (or copy on Windows w/o symlink-priv) the source DICOMs.

    Returns the number of files actually staged. The destination path
    is funnelled through :func:`long_path` so that deep Windows BIDS
    trees do not hit the 260-char ``MAX_PATH`` limit at the symlink
    syscall — pathlib's ``Path`` machinery sometimes still routes
    through narrow-string APIs on older interpreters.

    On Windows, creating a symbolic link requires either the
    ``SeCreateSymbolicLinkPrivilege`` (granted in Developer Mode) or
    an Administrator process. When ``os.symlink`` raises
    ``OSError(WinError 1314)`` we fall back to a plain file copy so
    end users without those privileges still get a working batch.
    """
    if staging_dir.exists():
        shutil.rmtree(long_path(staging_dir), ignore_errors=False)
    staging_dir.mkdir(parents=True)
    n = 0
    use_copy_fallback = False
    for idx, fp in enumerate(source_files):
        src = Path(fp)
        if not src.exists():
            continue
        # Short sequential names: vendor DICOM filenames are commonly
        # ~80 chars (full SeriesInstanceUID + instance suffix), which
        # pushes the full Win32 path past MAX_PATH inside deep staging
        # trees and makes dcm2niix exit ``rc=2`` without a useful error.
        # dcm2niix orders frames by DICOM tags (InstanceNumber etc.),
        # not by filename, so renaming on stage is safe. The zero-padded
        # counter also makes collision handling unnecessary.
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
            # Windows error 1314: "A required privilege is not held by
            # the client." (SeCreateSymbolicLinkPrivilege missing.)
            # Switch this series to copy-mode and stage this file via
            # plain copy. Other OSErrors propagate so we see them.
            if getattr(exc, "winerror", None) == 1314:
                use_copy_fallback = True
                shutil.copyfile(long_path(src_resolved), long_path(link))
            else:
                raise
        n += 1
    return n


def _run_dcm2niix(
    binary: Path,
    dicom_dir: Path,
    output_dir: Path,
    basename: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess:
    """Invoke dcm2niix to write ``<basename>.nii.gz`` etc. into ``output_dir``.

    Flags: ``-b y`` (write JSON), ``-ba n`` (don't anonymise — keeps
    SeriesInstanceUID for provenance / dedup), ``-z y`` (gzip),
    ``-f <basename>`` (we control the name; do not use ``%j`` here),
    and ``--terse`` is intentionally NOT passed so debugging stderr
    survives.
    """
    cmd = [
        str(binary),
        "-b", "y",
        "-ba", "n",
        "-z", "y",
        "-o", str(output_dir),
        "-f", basename,
        str(dicom_dir),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _collect_outputs(output_dir: Path, basename: str) -> list[Path]:
    """Return every file dcm2niix produced for this basename, sorted.

    dcm2niix occasionally writes additional siblings (``_e1``, ``_e2``,
    ``_ph`` for fmap; ``.bval`` / ``.bvec`` for DWI). Glob ``<basename>*``
    catches them all; deterministic sorting makes the result stable.
    """
    return sorted(output_dir.glob(f"{basename}*"))


def _missing_expected(
    staged: list[Path],
    expected_exts: tuple[str, ...],
    basename: str,
    output_dir: Path,
) -> list[str]:
    """Return any required extension that didn't materialise.

    For fmap multi-output, the *exact* ``<basename>.nii.gz`` may be absent
    while ``<basename>_e1.nii.gz`` etc. exist — that's fine, the fixup
    will rename them. Only flag missing if NEITHER the exact nor any
    suffixed sibling matches the extension.
    """
    missing: list[str] = []
    for ext in expected_exts:
        exact = output_dir / f"{basename}{ext}"
        if exact.exists():
            continue
        # Any sibling with this extension counts (e.g. <basename>_e1.nii.gz).
        if any(p.name.endswith(ext) for p in staged):
            continue
        missing.append(ext)
    return missing


__all__ = ["Dcm2niixDirect"]

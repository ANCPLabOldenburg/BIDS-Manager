"""Siemens CMRR physio backend — wraps ``bidsphysio.dcm2bidsphysio``.

CMRR sequences emit a single ``_PhysioLog.dcm`` file alongside the
imaging series. ``bidsphysio.dcm2bidsphysio.dcm2bids`` parses that
DICOM and returns a ``PhysioData`` object whose ``save_to_bids``
method writes BIDS-spec ``_physio.tsv.gz`` + ``_physio.json`` pairs
(one pair per distinct sampling rate — the BIDS spec requires the
JSON's ``SamplingFrequency`` be single-valued, so multi-rate
recordings split into ``_recording-<rate>_physio.tsv.gz`` siblings).

This backend's :meth:`can_handle` is intentionally narrow: it claims
only rows whose ``suffix == "physio"``. Other physio formats
(BIOPAC ``.acq``, raw Philips ``.puls``/``.resp``/``.ecg``) need
their own backends; they fall through to a clear "no backend
matched" warning rather than getting silently mis-converted.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Optional

from ..types import ConvertResult, ConvertTask

log = logging.getLogger(__name__)


class PhysioDcmBackend:
    """Convert Siemens CMRR physio DICOMs via ``bidsphysio``.

    Stateless; safe to instantiate once per ``run_convert`` and reuse
    across tasks.
    """

    name = "physio_dcm"

    def can_handle(self, task: ConvertTask) -> bool:
        if task.suffix != "physio":
            return False
        if not task.source_files or not task.basename:
            return False
        # Heuristic: at least one source file mentions physio in its
        # name. Keeps the backend from claiming non-CMRR rows that
        # accidentally classified to suffix=physio.
        for fp in task.source_files:
            name_lower = fp.name.lower()
            if "physio" in name_lower or "_physiolog" in name_lower:
                return True
        # If the suffix is physio but no name hint matches, still try —
        # some scanner exports drop the PhysioLog token. Worst case the
        # bidsphysio call fails and ``convert`` records the error.
        return True

    def convert(self, task: ConvertTask, staging_dir: Path) -> ConvertResult:
        t0 = time.monotonic()
        try:
            return self._convert_inner(task, staging_dir, t0)
        except Exception as exc:
            log.exception("physio_dcm: unexpected error for %s", task.basename)
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
        # bidsphysio's pkg_resources warning is cosmetic — silence it
        # for the duration of this call so user-facing output stays clean.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*pkg_resources.*",
            )
            try:
                from bidsphysio.dcm2bids.dcm2bidsphysio import dcm2bids as _bp_dcm2bids
            except ImportError as exc:
                return ConvertResult(
                    task=task, success=False,
                    error=(
                        "bidsphysio is not installed; cannot convert physio "
                        f"row {task.basename!r}: {exc}"
                    ),
                    duration_s=time.monotonic() - t0,
                )

            # bidsphysio expects a list of file paths (or a single path).
            source_paths = [str(p) for p in task.source_files if p.exists()]
            if not source_paths:
                return ConvertResult(
                    task=task, success=False,
                    error="empty staging: no source physio DICOMs were accessible",
                    duration_s=time.monotonic() - t0,
                )

            try:
                physio = _bp_dcm2bids(source_paths)
            except Exception as exc:
                return ConvertResult(
                    task=task, success=False,
                    error=f"bidsphysio.dcm2bids failed: {type(exc).__name__}: {exc}",
                    duration_s=time.monotonic() - t0,
                )

            output_dir = staging_dir / task.datatype
            output_dir.mkdir(parents=True, exist_ok=True)
            bids_prefix = output_dir / task.basename

            try:
                # save_to_bids writes one (.tsv.gz, .json) pair per
                # sampling rate; bidsphysio appends ``_recording-<rate>``
                # automatically when there's more than one rate.
                physio.save_to_bids(str(bids_prefix))
            except Exception as exc:
                return ConvertResult(
                    task=task, success=False,
                    error=f"PhysioData.save_to_bids failed: {type(exc).__name__}: {exc}",
                    duration_s=time.monotonic() - t0,
                )

        staged = sorted(output_dir.glob(f"{task.basename}*"))
        if not staged:
            return ConvertResult(
                task=task, success=False,
                error="bidsphysio produced no output files",
                duration_s=time.monotonic() - t0,
            )

        return ConvertResult(
            task=task, staged_files=tuple(staged), success=True,
            duration_s=time.monotonic() - t0,
        )


__all__ = ["PhysioDcmBackend"]

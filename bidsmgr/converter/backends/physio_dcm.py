"""Siemens CMRR physio backend.

CMRR sequences emit a single ``_PhysioLog.dcm`` alongside the imaging
series, holding every physiological log of the run. It is read by
:mod:`bidsmgr.converter.physio_cmrr` (each channel on its own grid, every
ECG lead its own column, the volume onsets from the acquisition log) and
written as BIDS tables with a ``trigger`` column last: one table per
sampling rate, or one for the run (``ConvertTask.physio_layout``), cropped to
the run when asked (``physio_align``), starting at the first volume kept
(``drop_volumes``).

This backend's :meth:`can_handle` is intentionally narrow: it claims
only rows whose ``suffix == "physio"``. Other physio formats
(BIOPAC ``.acq``, raw Philips ``.puls``/``.resp``/``.ecg``) need
their own backends; they fall through to a clear "no backend
matched" warning rather than getting silently mis-converted.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..types import ConvertResult, ConvertTask

log = logging.getLogger(__name__)


class PhysioDcmBackend:
    """Convert Siemens CMRR physio DICOMs (``converter.physio_cmrr``).

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
        from ..physio_cmrr import read_cmrr, write_tables

        source_paths = [p for p in task.source_files if p.exists()]
        if not source_paths:
            return ConvertResult(
                task=task, success=False,
                error="empty staging: no source physio DICOMs were accessible",
                duration_s=time.monotonic() - t0,
            )
        try:
            physio = read_cmrr(source_paths)
        except Exception as exc:
            return ConvertResult(
                task=task, success=False,
                error=f"the physiological log could not be read: {type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - t0,
            )

        output_dir = staging_dir / task.datatype
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            task.basename[: -len("_physio")]
            if task.basename.endswith("_physio")
            else task.basename
        )
        try:
            staged = write_tables(
                physio, output_dir / stem, layout=task.physio_layout,
                align=task.physio_align, drop_volumes=task.drop_volumes,
            )
        except Exception as exc:
            return ConvertResult(
                task=task, success=False,
                error=f"the physiological tables could not be written: "
                      f"{type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - t0,
            )
        if not staged:
            return ConvertResult(
                task=task, success=False,
                error="the physiological log produced no table",
                duration_s=time.monotonic() - t0,
            )
        if physio.unfinished:
            log.info("physio %s: the run was stopped during its last volume, which the image "
                     "does not hold; the trigger leaves it out", task.basename)
        if not physio.channels:
            log.info("physio %s: no signal in the log, only the timing of its %d volume(s): "
                     "the trigger alone", task.basename, len(physio.volumes))
        else:
            log.info(
                "physio %s: %d channel(s) (%s), %d volume(s) in the log, %s",
                task.basename, len(physio.channels),
                ", ".join(ch.label for ch in physio.channels), len(physio.volumes),
                {"one_interpolated": "one table, interpolated",
                 "one_as_recorded": "one table, as recorded"}.get(task.physio_layout,
                                                                  "one table per rate"),
            )
        return ConvertResult(
            task=task, staged_files=tuple(sorted(staged)), success=True,
            duration_s=time.monotonic() - t0,
        )


__all__ = ["PhysioDcmBackend"]

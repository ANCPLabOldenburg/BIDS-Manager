"""Backend registry — selects a converter for a given task.

Reference: architecture.md §7. The registry is in-tree for v1
(decisions log §13: "in-tree registry; entry-points later"). Each
backend declares ``can_handle(task: ConvertTask) -> bool``; the
orchestrator asks each in priority order and the first match wins.

Today's roster (priority order — narrower matches first):

* ``PhysioDcmBackend`` — claims rows whose ``suffix == "physio"`` that
  look like Siemens CMRR ``_PhysioLog.dcm`` outputs. Wraps
  ``bidsphysio.dcm2bidsphysio``.
* ``EcatDirect`` — claims PET rows whose source carries the ECAT
  ``MATRIX7x`` signature (Siemens HRRT and older CTI scanners, which
  dcm2niix cannot read). Uses nibabel, no subprocess.
* ``Dcm2niixDirect`` — broad MRI fallback. Claims everything else,
  including PET rows that came from DICOM.

Future: ``mne_bids`` (EEG/MEG/iEEG), ``passthrough`` (already-BIDS).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from ..inventory.types import Modality
from .types import ConvertResult, ConvertTask


@runtime_checkable
class ConverterBackend(Protocol):
    """Per-series conversion contract.

    The orchestrator drives parallelism (one ``ConvertTask`` per
    invocation, joblib pool sized by ``-j``); the backend just runs the
    underlying tool and reports the staged output files.
    """

    name: str

    def can_handle(self, task: ConvertTask) -> bool: ...

    def convert(self, task: ConvertTask, staging_dir: Path) -> ConvertResult: ...


def default_backends(
    *,
    dcm2niix_bin: Optional[Path] = None,
) -> list[ConverterBackend]:
    """Return the priority-ordered list of registered backends.

    Narrower matches go first. Lazy imports keep modules with heavy
    optional deps (bidsphysio's pkg_resources warning, mne, dcm2niix
    discovery) out of the import path until actually used.

    Order: ``PhysioDcmBackend`` (suffix=physio narrow match) →
    ``EcatDirect`` (datatype=pet AND an ECAT signature) →
    ``MneBidsBackend`` (datatype ∈ eeg/meg/ieeg/nirs) →
    ``Dcm2niixDirect`` (broad MRI fallback, which is also the DICOM PET
    path). Each upstream backend's
    ``can_handle`` declines the others' datatypes so the dispatch is
    unambiguous.

    EEG/MEG ``line_freq`` / ``montage`` are not backend-level knobs: they
    are resolved per row by the convert verb (the inventory cell, else the
    recording-metadata dataset default) and carried on each
    :class:`ConvertTask`.
    """
    from .backends.dcm2niix_direct import Dcm2niixDirect
    from .backends.ecat_direct import EcatDirect
    from .backends.mne_bids import MneBidsBackend
    from .backends.physio_dcm import PhysioDcmBackend

    return [
        PhysioDcmBackend(),
        EcatDirect(),
        MneBidsBackend(),
        Dcm2niixDirect(dcm2niix_bin=dcm2niix_bin),
    ]


def dispatch(
    backends: list[ConverterBackend], task: ConvertTask,
) -> ConverterBackend:
    """Pick the first backend whose ``can_handle(task)`` returns ``True``.

    Raises ``LookupError`` when no backend matches — the orchestrator
    converts that into a per-task ``ConvertResult`` so a single
    unmatched row doesn't abort the whole subject.
    """
    for backend in backends:
        if backend.can_handle(task):
            return backend
    raise LookupError(
        f"no backend can handle task {task.basename!r} "
        f"(suffix={task.suffix!r})"
    )


def select_backend(
    modality: Modality, *, dcm2niix_bin: Optional[Path] = None,
) -> ConverterBackend:
    """Modality-keyed selector kept for backward compatibility.

    Returns the *primary* backend for the modality. Most callers should
    use :func:`default_backends` + :func:`dispatch` instead so that
    physio rows route to the physio backend.
    """
    if modality == "mri":
        from .backends.dcm2niix_direct import Dcm2niixDirect
        return Dcm2niixDirect(dcm2niix_bin=dcm2niix_bin)
    raise NotImplementedError(f"No converter backend for modality={modality!r}")


__all__ = [
    "ConverterBackend", "default_backends", "dispatch", "select_backend",
]

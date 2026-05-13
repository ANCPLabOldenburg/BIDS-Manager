"""Backend registry — selects a converter for a given task.

Reference: architecture.md §7. The registry is in-tree for v1
(decisions log §13: "in-tree registry; entry-points later"). Each
backend declares ``can_handle(task: ConvertTask) -> bool``; the
orchestrator asks each in priority order and the first match wins.

Today's roster (priority order — narrower matches first):

* ``PhysioDcmBackend`` — claims rows whose ``suffix == "physio"`` that
  look like Siemens CMRR ``_PhysioLog.dcm`` outputs. Wraps
  ``bidsphysio.dcm2bidsphysio``.
* ``Dcm2niixDirect`` — broad MRI fallback. Claims everything else.

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
    line_freq: Optional[float] = 50.0,
    montage: Optional[str] = None,
) -> list[ConverterBackend]:
    """Return the priority-ordered list of registered backends.

    Narrower matches go first. Lazy imports keep modules with heavy
    optional deps (bidsphysio's pkg_resources warning, mne, dcm2niix
    discovery) out of the import path until actually used.

    Order: ``PhysioDcmBackend`` (suffix=physio narrow match) →
    ``MneBidsBackend`` (datatype ∈ eeg/meg/ieeg/nirs) →
    ``Dcm2niixDirect`` (broad MRI fallback). Each upstream backend's
    ``can_handle`` declines the others' datatypes so the dispatch is
    unambiguous.

    EEG/MEG-specific overrides (``line_freq``, ``montage``) flow into
    the :class:`MneBidsBackend` constructor so the user can supply them
    at the CLI level (``bidsmgr-convert --line-freq 60 --montage
    biosemi64``).
    """
    from .backends.dcm2niix_direct import Dcm2niixDirect
    from .backends.mne_bids import MneBidsBackend
    from .backends.physio_dcm import PhysioDcmBackend

    return [
        PhysioDcmBackend(),
        MneBidsBackend(line_freq=line_freq, montage=montage),
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

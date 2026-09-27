"""Pure-data types passed to converter backends.

Reference: architecture.md §7. ``ConvertTask`` is what the CLI orchestrator
hands a backend; ``ConvertResult`` is what the backend hands back. Backends
never decide BIDS names — ``basename`` and ``datatype`` come from the
schema engine via the inventory TSV's ``bids_name`` column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConvertTask(BaseModel):
    """One unit of conversion work — one input series → one set of outputs.

    The ``source_files`` field carries the input paths the backend should
    consume. Its concrete shape varies by modality:

    * MRI (DICOM): potentially many files — every DICOM in the series.
    * Physio (Siemens CMRR): one ``_PhysioLog.dcm``.
    * EEG/MEG/iEEG: one recording file (or one folder for ``.ds``/``.mff``).

    Multi-output conversion cases (fmap mag1+mag2+phasediff, DWI nii.gz+
    json+bval+bvec, multi-rate physio) are still a single task — backends
    report whatever files actually landed on disk via ``staged_files``.

    For backwards compatibility the constructor accepts the old
    ``source_dicom_files`` keyword and copies it into ``source_files``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    row_id: str
    series_uid: str
    source_files: tuple[Path, ...]
    dataset: str
    bids_root: Path
    subject: str
    session: Optional[str] = None
    datatype: str
    suffix: str
    entities: dict[str, str] = Field(default_factory=dict)
    basename: str
    expected_outputs: tuple[str, ...] = (".nii.gz", ".json")
    repetition_type: str = ""

    # When True (default), the MRI backend drops dcm2niix "residual"
    # secondary outputs -- the derived single-volume duplicates dcm2niix
    # splits off a single input series and names by gluing a collision
    # letter onto the basename (e.g. ``..._bold`` -> ``..._bolda``) or with
    # an ``_Eq_<n>`` / ``_ROI`` / ``_i<instance>`` marker. They are not real
    # acquired images and have no valid BIDS suffix. Legitimate multi-output
    # (fmap ``_e1``/``_e2``/``_ph``, complex ``_real``/``_imaginary``,
    # DWI ``.bval``/``.bvec``) is never affected. Set False to keep them.
    skip_residuals: bool = True

    # EEG/MEG-specific: per-row knobs the inventory TSV carries. The CLI
    # resolves these (cell, else recording-metadata default) before building
    # the task. ``line_freq`` / ``montage`` are applied during the write;
    # ``eeg_reference`` / ``eeg_ground`` are consumed by the post-write
    # sidecar-enrichment fixup, where they override the spec value.
    line_freq: Optional[float] = None
    montage: Optional[str] = None
    eeg_reference: Optional[str] = None
    eeg_ground: Optional[str] = None

    # When True, EEG / iEEG recordings are re-encoded to EDF on write
    # (``mne_bids.write_raw_bids(format="EDF")``) instead of kept in their
    # source format. Lets a study harmonise to a single BIDS-native format,
    # and is the way a non-BIDS-native but mne-readable source (GDF, EGI, ...)
    # becomes convertible. MEG / NIRS ignore it (EDF is an EEG / iEEG format).
    force_edf: bool = False

    # Already-curated companion files to COPY into the BIDS tree on convert
    # (not converted): ``((suffix, source_path), ...)`` where suffix is a
    # BIDS companion suffix (events / beh / stim / physio / channels / ...).
    # The file is named ``<entity_prefix>_<suffix><ext>`` next to the
    # recording; an attached ``events`` replaces the auto-generated events.tsv.
    companion_files: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_source_dicom_files(cls, data: Any) -> Any:
        """Accept ``source_dicom_files`` (old name) as an alias.

        Lets older test code construct tasks with the previous keyword
        without immediately blowing up. Internally we always use
        ``source_files``.
        """
        if isinstance(data, dict) and "source_files" not in data:
            legacy = data.pop("source_dicom_files", None)
            if legacy is not None:
                data["source_files"] = legacy
        return data

    @property
    def source_dicom_files(self) -> tuple[Path, ...]:
        """Deprecated alias for :attr:`source_files`. Kept for compat."""
        return self.source_files


class ConvertResult(BaseModel):
    """What the backend reports back to the orchestrator.

    ``staged_files`` is the list of files actually present after the
    backend ran (still in staging — they haven't been atomically moved
    into the live BIDS tree yet). ``error`` is non-None iff ``success``
    is False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: ConvertTask
    staged_files: tuple[Path, ...] = ()
    success: bool = False
    error: Optional[str] = None
    dcm2niix_returncode: Optional[int] = None
    dcm2niix_stderr_tail: str = ""
    duration_s: Optional[float] = None


__all__ = ["ConvertTask", "ConvertResult"]

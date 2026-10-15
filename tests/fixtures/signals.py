"""Small, genuine recordings written on the fly for the viewer tests.

Shared by the Qt-free tests of ``bidsmgr.viz`` and the GUI tests of the
viewer, so both exercise the same files.
"""

from __future__ import annotations

import gzip
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np


def write_physio(root: Path, name: str, rows: list[list[float]], meta: dict,
                 *, datatype: str = "func") -> Path:
    """A ``*_physio.tsv.gz`` (no header row, as BIDS requires) and its sidecar."""
    folder = root / "sub-001" / datatype
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write("\t".join(f"{v:g}" for v in row) + "\n")
    sidecar = folder / name.replace(".tsv.gz", ".json")
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    return path


def write_fif(folder: Path, name: str = "sub-01_task-rest_eeg.fif", *,
              sfreq: float = 128.0, seconds: float = 3.0,
              stim_at: tuple[int, ...] = (), first_samp: int = 0) -> Path:
    """A five-channel FIF (three EEG, one EOG, one stim).

    ``stim_at``: sample indices (from the start of the DATA) where the stim
    channel steps to 5 for ten samples. ``first_samp``: where the recording's
    first sample sits on the acquisition clock, as a real MEG system writes
    it (the defect this guards drew every stim event that far late).
    """
    import mne

    n = int(sfreq * seconds)
    data = np.random.default_rng(2).standard_normal((5, n)) * 1e-5
    data[4] = 0.0
    for s in stim_at:
        data[4, s:s + 10] = 5.0
    info = mne.create_info(["Fz", "Cz", "Pz", "EOG", "STI"], sfreq,
                           ["eeg", "eeg", "eeg", "eog", "stim"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.RawArray(data, info, first_samp=first_samp, verbose=False)
        raw.save(path, overwrite=True, verbose=False)
    return path


#: A 3 T 1H spectrometer and a 2000 Hz spectral width.
SF_MHZ = 123.26
DWELL = 0.0005


def fid_at(ppm_shift: float, n: int = 2048, *, decay_s: float = 0.15,
           amplitude: complex = 1.0) -> np.ndarray:
    """A decaying complex sinusoid that transforms to one peak at ``ppm``."""
    hz = -(ppm_shift - 4.65) * SF_MHZ
    t = np.arange(n) * DWELL
    return amplitude * np.exp(2j * np.pi * hz * t) * np.exp(-t / decay_s)


def write_mrs(folder: Path, name: str, fid: np.ndarray,
              header: Optional[dict] = None) -> Path:
    """A genuine NIfTI-MRS file: ``fid`` is ``(points, *higher)``, the header
    goes in extension 44 and the dwell time in the fourth pixdim."""
    import nibabel as nib

    from bidsmgr.fixups.mrs_header import MRS_EXTENSION_CODE

    folder.mkdir(parents=True, exist_ok=True)
    data = np.asarray(fid, dtype=np.complex64).reshape((1, 1, 1) + fid.shape)
    img = nib.Nifti2Image(data, np.eye(4))
    zooms = [1.0, 1.0, 1.0, DWELL] + [1.0] * (data.ndim - 4)
    img.header.set_zooms(zooms)
    meta = {"SpectrometerFrequency": [SF_MHZ], "ResonantNucleus": ["1H"]}
    meta.update(header or {})
    img.header.extensions.append(
        nib.nifti1.Nifti1Extension(MRS_EXTENSION_CODE, json.dumps(meta).encode()))
    path = folder / name
    nib.save(img, str(path))
    return path

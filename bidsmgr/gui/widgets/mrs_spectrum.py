"""The maths a NIfTI-MRS file has to go through before it means anything.

Qt-free, so it can be tested without a QApplication and run on a worker.

An MRS file is not an image. The NIfTI data block holds a **free induction
decay**: a complex signal in the TIME domain, which is what the scanner
actually measured. Nobody reads that. What a spectroscopist reads is its
Fourier transform plotted against **chemical shift in parts per million**,
with the axis running backwards, because that is the convention every
textbook, every fitting package and every paper uses.

Getting from one to the other needs three numbers that live in the
NIfTI-MRS header inside the file, not in the image:

``dwell time``             the spacing of the samples, which sets the
                           spectral width as ``1 / dwell``
``SpectrometerFrequency``  MHz, which converts Hz to ppm
``ResonantNucleus``        which sets where zero ppm sits by convention

Two processing steps are offered because a raw transform is genuinely hard
to read, and both are standard rather than inventions:

**Line broadening** multiplies the FID by a decaying exponential before the
transform. It trades resolution for signal-to-noise, and every MRS package
applies a few Hz by default.

**Zero-order phase** rotates the complex spectrum. An FID is rarely
perfectly phased as acquired, and an unphased real part shows peaks that
dip below the baseline, which reads as an artefact rather than as a phase.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

#: NIfTI extension code the NIfTI-MRS standard reserves for its header.
MRS_EXTENSION_CODE = 44

#: Where zero ppm sits, per nucleus, by convention. For 1H the reference is
#: the water resonance at body temperature; the others are referenced to
#: their own standards and need no offset here.
NUCLEUS_REFERENCE_PPM: dict[str, float] = {
    "1H": 4.65,
    "13C": 0.0,
    "31P": 0.0,
    "19F": 0.0,
    "23NA": 0.0,
}

#: Chemical shifts of the metabolites a 1H brain spectrum is read for, in
#: ppm, with the name a spectroscopist would use.
#:
#: This is the thing that makes the viewer worth opening. A spectrum without
#: them is a row of unlabelled bumps; with them, the question "is the NAA
#: where it should be" is answered by looking. Only the principal peak of
#: each is marked: a second line three pixels away labels nothing.
METABOLITES_1H: tuple[tuple[str, float, str], ...] = (
    ("Lip/MM", 0.90, "Lipids and macromolecules"),
    ("Lac", 1.33, "Lactate (doublet)"),
    ("Ala", 1.47, "Alanine"),
    ("NAA", 2.01, "N-acetylaspartate"),
    ("GABA", 2.28, "Gamma-aminobutyric acid"),
    ("Glx", 2.35, "Glutamate + glutamine"),
    ("NAAG", 2.60, "N-acetylaspartylglutamate"),
    ("Cr", 3.03, "Creatine + phosphocreatine"),
    ("Cho", 3.22, "Choline-containing compounds"),
    ("mI", 3.56, "Myo-inositol"),
    ("Glx2", 3.75, "Glutamate + glutamine"),
    ("Cr2", 3.91, "Creatine (methylene)"),
    ("H2O", 4.70, "Residual water"),
)

#: The window a 1H brain spectrum is conventionally shown in. Outside it
#: there is water on one side and lipid on the other, and neither is what
#: the scan was run for.
DEFAULT_PPM_RANGE_1H: tuple[float, float] = (0.2, 4.2)


def read_mrs(path: Path) -> Optional[dict]:
    """Everything needed to plot ``path``, or ``None`` if it is not MRS.

    Returns the FID as ``(points, dynamics)`` complex, plus the header and
    the three numbers the transform needs. ``None`` means the file carries
    no NIfTI-MRS header, which is how the caller decides whether to offer
    this viewer at all.
    """
    try:
        import nibabel as nib

        img = nib.load(str(path))
        header = None
        for ext in img.header.extensions:
            if ext.get_code() != MRS_EXTENSION_CODE:
                continue
            text = ext.get_content().decode("utf-8", "replace").rstrip("\x00")
            candidate = json.loads(text)
            if isinstance(candidate, dict):
                header = candidate
                break
        if header is None:
            return None

        data = np.asarray(img.dataobj)
        if not np.iscomplexobj(data):
            # A NIfTI-MRS file is complex by definition. Anything else is a
            # file that merely carries the extension.
            log.debug("%s has an MRS header but real data", path.name)
            return None

        # (x, y, z, t, ...) -> (t, everything else). A single-voxel scan has
        # one spatial position and any number of dynamics, coils or edits;
        # they all collapse into the second axis and the caller decides how
        # to combine them.
        if data.ndim < 4:
            return None
        points = data.shape[3]
        fid = data.reshape(-1, points, *data.shape[4:])
        fid = np.moveaxis(fid.reshape(-1, points, int(np.prod(data.shape[4:]) or 1)), 1, 0)
        fid = fid.reshape(points, -1).astype(np.complex128)

        zooms = img.header.get_zooms()
        dwell = float(zooms[3]) if len(zooms) > 3 else 0.0
        if not np.isfinite(dwell) or dwell <= 0:
            dwell = float(header.get("DwellTime") or 0.0)
        if dwell <= 0:
            log.debug("%s states no dwell time; cannot build a ppm axis", path.name)
            return None

        nuclei = header.get("ResonantNucleus") or ["1H"]
        nucleus = str(nuclei[0]).upper().replace(" ", "")
        freqs = header.get("SpectrometerFrequency") or [0.0]
        spectrometer_mhz = float(freqs[0])

        return {
            "fid": fid,
            "dwell": dwell,
            "spectrometer_mhz": spectrometer_mhz,
            "nucleus": nucleus,
            "header": header,
            "shape": tuple(int(x) for x in data.shape),
            "n_dynamics": int(fid.shape[1]),
        }
    except Exception as exc:  # noqa: BLE001 - not every .nii.gz is MRS
        log.debug("could not read %s as NIfTI-MRS: %s", path, exc)
        return None


def is_mrs_path(path) -> bool:
    """True when the Editor should open ``path`` in the MRS viewer.

    Decided by the DATATYPE folder and the schema's own suffix list, not by
    opening the file: routing a click must not read a volume off disk, and
    a 500 MB BOLD and a 32 KB spectrum look the same from the filename
    until you do.
    """
    p = Path(path)
    name = p.name
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        return False
    if p.parent.name != "mrs":
        return False
    try:
        from ... import schema

        suffixes = set(schema.list_suffixes("mrs"))
    except Exception:  # noqa: BLE001
        suffixes = {"svs", "mrsi", "unloc", "mrsref"}
    stem = name.split(".")[0]
    return stem.rsplit("_", 1)[-1] in suffixes


def combine(fid: np.ndarray, mode: str = "mean", index: int = 0) -> np.ndarray:
    """Collapse the dynamics axis to one FID.

    ``mean`` is what a spectroscopist wants almost always: the repeats exist
    to be averaged, and that is where the signal-to-noise comes from.
    ``single`` shows one repeat, which is how a corrupted average gets
    found.
    """
    if fid.ndim == 1:
        return fid
    if mode == "single":
        column = max(0, min(int(index), fid.shape[1] - 1))
        return fid[:, column]
    return fid.mean(axis=1)


def apodize(fid: np.ndarray, dwell: float, line_broadening_hz: float) -> np.ndarray:
    """Multiply the FID by a decaying exponential. Zero is a no-op.

    Trades resolution for signal-to-noise, which is the trade every MRS
    package makes by default, because an unbroadened in-vivo spectrum is
    dominated by noise at the tail of the FID where there is no longer any
    signal to transform.
    """
    if not line_broadening_hz:
        return fid
    t = np.arange(fid.shape[0]) * dwell
    return fid * np.exp(-np.pi * float(line_broadening_hz) * t)


def spectrum(
    fid: np.ndarray,
    dwell: float,
    spectrometer_mhz: float,
    nucleus: str = "1H",
    *,
    line_broadening_hz: float = 0.0,
    phase_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ppm, hz, complex spectrum)`` from one FID.

    The ppm axis is NEGATED before the reference is added, which is the step
    that is easy to get wrong and impossible to notice if you do: a spectrum
    plotted with the sign the other way round is a perfectly plausible
    picture in which every metabolite is on the wrong side of the water.
    """
    data = apodize(np.asarray(fid, dtype=np.complex128), dwell, line_broadening_hz)
    n = data.shape[0]
    spec = np.fft.fftshift(np.fft.fft(data))
    hz = np.fft.fftshift(np.fft.fftfreq(n, d=dwell))
    if phase_deg:
        spec = spec * np.exp(1j * np.deg2rad(float(phase_deg)))
    reference = NUCLEUS_REFERENCE_PPM.get(nucleus.upper(), 0.0)
    if spectrometer_mhz > 0:
        ppm = -hz / spectrometer_mhz + reference
    else:
        # No spectrometer frequency means no ppm axis. Give back Hz in its
        # place rather than a silently meaningless scale.
        ppm = hz
    return ppm, hz, spec


def part(spec: np.ndarray, which: str) -> np.ndarray:
    """The displayed component of a complex spectrum."""
    if which == "real":
        return spec.real
    if which == "imaginary":
        return spec.imag
    if which == "phase":
        return np.angle(spec, deg=True)
    return np.abs(spec)


def metabolites_for(nucleus: str) -> tuple[tuple[str, float, str], ...]:
    """The labelled reference peaks for a nucleus, or none.

    Only 1H has a table here, because only 1H has one worth drawing from
    memory. A 31P or 13C spectrum gets an unlabelled axis rather than 1H
    labels in the wrong places.
    """
    return METABOLITES_1H if nucleus.upper() == "1H" else ()


def stagger(
    shifts: list[float], span: float, *, rows: int = 4, min_gap: float = 0.055,
) -> list[int]:
    """Which row each label should sit on so none covers its neighbour.

    ``shifts`` in ppm, ``span`` the ppm width currently on screen. Returns a
    row index per label, 0 being the highest.

    Metabolites are not evenly spaced: creatine at 3.03 and choline at 3.22
    are a fifth of a ppm apart, and glutamate at 3.75 sits beside creatine's
    second peak at 3.91. Drawn on one line they overlap into an unreadable
    smear exactly where a 1H spectrum is busiest.

    The test is in FRACTION OF THE VISIBLE SPAN, not in ppm, because a
    label's width in PIXELS does not change with the zoom while the pane's
    does not either: what changes is how much ppm a pixel is worth. So two
    names 0.19 ppm apart collide across the whole 4 ppm window and have
    room to spare zoomed into half a ppm, and the rows have to be recomputed
    on every range change rather than decided once.

    Each label takes the highest row far enough from the last one already on
    it, which is the standard greedy pass and is stable, so a label does not
    jump rows as a neighbour scrolls past.
    """
    if not shifts:
        return []
    width = (span if span > 0 else 1.0) * float(min_gap)
    last: list[float] = [-1e9] * max(1, rows)
    out: list[int] = []
    for shift in shifts:
        for row in range(len(last)):
            if abs(shift - last[row]) >= width:
                last[row] = shift
                out.append(row)
                break
        else:
            # Everything is crowded. Put it on the row whose last label is
            # furthest away rather than dropping it: a missing label is a
            # metabolite the reader cannot find.
            row = max(range(len(last)), key=lambda r: abs(shift - last[r]))
            last[row] = shift
            out.append(row)
    return out


def default_ppm_range(nucleus: str, ppm: np.ndarray) -> tuple[float, float]:
    """The window to open at: conventional for 1H, the data's own otherwise."""
    if nucleus.upper() == "1H":
        return DEFAULT_PPM_RANGE_1H
    return float(np.min(ppm)), float(np.max(ppm))


__all__ = [
    "DEFAULT_PPM_RANGE_1H",
    "METABOLITES_1H",
    "MRS_EXTENSION_CODE",
    "NUCLEUS_REFERENCE_PPM",
    "apodize",
    "combine",
    "default_ppm_range",
    "is_mrs_path",
    "metabolites_for",
    "part",
    "read_mrs",
    "spectrum",
    "stagger",
]

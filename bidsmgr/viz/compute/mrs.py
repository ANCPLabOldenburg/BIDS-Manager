"""The maths a NIfTI-MRS file has to go through before it means anything.

An MRS file is not an image. The NIfTI data block holds a **free induction
decay**: a complex signal in the TIME domain, which is what the scanner
actually measured. Nobody reads that. What a spectroscopist reads is its
Fourier transform plotted against **chemical shift in parts per million**,
with the axis running backwards, because that is the convention every
textbook, every fitting package and every paper uses.

Getting from one to the other needs three numbers from the NIfTI-MRS header
inside the file: the dwell time (``1 / dwell`` is the spectral width), the
spectrometer frequency (MHz, Hz to ppm) and the nucleus (where zero ppm sits).

Processing offered, all standard rather than inventions:

* **line broadening**: an exponential on the FID before the transform,
  trading resolution for signal-to-noise;
* **zero-order phase**: one rotation of the whole complex spectrum;
* **first-order phase**: a rotation growing linearly with frequency, which a
  delayed acquisition start produces. Given as a time (ms), the delay it
  undoes: ``phase(f) = phase0 - 360 * f * delay``
  (a delay multiplies each line by ``exp(+2 pi i f delay)``; this takes it
  back out);
* **coil combination**: each coil's FID turned to the phase of its first
  point and weighted by its amplitude there, then summed. A plain average
  of unphased coils cancels the signal it is meant to add up.

Qt-free.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: Where zero ppm sits, per nucleus, by convention. For 1H the reference is
#: the water resonance at body temperature.
NUCLEUS_REFERENCE_PPM: dict[str, float] = {
    "1H": 4.65, "13C": 0.0, "31P": 0.0, "19F": 0.0, "23NA": 0.0,
}

#: Chemical shifts of the metabolites a 1H brain spectrum is read for, in
#: ppm. Only the principal peak of each: a second line three pixels away
#: labels nothing.
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

#: The window a 1H brain spectrum is conventionally shown in.
DEFAULT_PPM_RANGE_1H: tuple[float, float] = (0.2, 4.2)


def combine(fid: np.ndarray, mode: str = "mean", index: int = 0) -> np.ndarray:
    """Collapse a repeats axis (the last) to one FID: ``mean`` (the repeats
    exist to be averaged) or ``single`` (how a corrupted repeat is found)."""
    if fid.ndim == 1:
        return fid
    if mode == "single":
        column = max(0, min(int(index), fid.shape[-1] - 1))
        return fid[..., column]
    return fid.mean(axis=-1)


def coil_combine(fid: np.ndarray, axis: int = 1) -> np.ndarray:
    """Combine receive coils: phase each coil to its first point, weight by
    its amplitude there, sum, normalise by the weights.

    ``fid`` has points along axis 0 and coils along ``axis``.
    """
    data = np.moveaxis(np.asarray(fid, dtype=np.complex128), axis, -1)
    first = data[0]                                    # (..., coils)
    weight = np.abs(first)
    phase = np.exp(-1j * np.angle(first))
    total = weight.sum(axis=-1)
    total = np.where(total > 0, total, 1.0)
    return (data * (weight * phase)[None, ...]).sum(axis=-1) / total[None, ...]


def apodize(fid: np.ndarray, dwell: float, line_broadening_hz: float) -> np.ndarray:
    """Multiply the FID by a decaying exponential. Zero is a no-op."""
    if not line_broadening_hz:
        return fid
    t = np.arange(fid.shape[0]) * dwell
    shape = (-1,) + (1,) * (fid.ndim - 1)
    return fid * np.exp(-np.pi * float(line_broadening_hz) * t).reshape(shape)


def spectrum(
    fid: np.ndarray,
    dwell: float,
    spectrometer_mhz: float,
    nucleus: str = "1H",
    *,
    line_broadening_hz: float = 0.0,
    phase_deg: float = 0.0,
    phase1_ms: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ppm, hz, complex spectrum)`` from one FID.

    The ppm axis is NEGATED before the reference is added, which is the step
    that is easy to get wrong and impossible to notice if you do: a spectrum
    plotted with the sign the other way round is a perfectly plausible
    picture in which every metabolite is on the wrong side of the water.
    """
    data = apodize(np.asarray(fid, dtype=np.complex128), dwell, line_broadening_hz)
    n = data.shape[0]
    spec = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    hz = np.fft.fftshift(np.fft.fftfreq(n, d=dwell))
    if phase_deg or phase1_ms:
        phi = np.deg2rad(float(phase_deg)) - 2.0 * np.pi * hz * (float(phase1_ms) / 1000.0)
        spec = spec * np.exp(1j * phi).reshape((-1,) + (1,) * (spec.ndim - 1))
    reference = NUCLEUS_REFERENCE_PPM.get(nucleus.upper(), 0.0)
    if spectrometer_mhz > 0:
        ppm = -hz / spectrometer_mhz + reference
    else:
        # No spectrometer frequency, no ppm axis: Hz rather than nonsense.
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
    """Labelled reference peaks for 1H only: 1H labels on a 31P spectrum
    would be in the wrong places."""
    return METABOLITES_1H if nucleus.upper() == "1H" else ()


def stagger(shifts: list[float], span: float, *, rows: int = 4,
            min_gap: float = 0.055) -> list[int]:
    """Which row each label sits on so none covers its neighbour.

    The test is in FRACTION OF THE VISIBLE SPAN: a label's width in pixels
    does not change with zoom, how much ppm a pixel is worth does. So rows
    are recomputed on every range change. Greedy and stable: a label does
    not jump rows as a neighbour scrolls past.
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
            # Everything is crowded: the row whose last label is furthest
            # away, never dropped (a missing label is a lost metabolite).
            row = max(range(len(last)), key=lambda r: abs(shift - last[r]))
            last[row] = shift
            out.append(row)
    return out


def default_ppm_range(nucleus: str, ppm: np.ndarray) -> tuple[float, float]:
    """The window to open at: conventional for 1H, the data's own otherwise.
    Clipped to the data: a 7 T spectrum 2000 Hz wide starts at 1.3 ppm, and
    asking a view for 0.2 to 4.2 slid it to 1.3 to 5.3, water and all."""
    lo, hi = float(np.min(ppm)), float(np.max(ppm))
    if nucleus.upper() == "1H":
        a, b = DEFAULT_PPM_RANGE_1H
        a, b = max(a, lo), min(b, hi)
        if b - a > 0.5:
            return a, b
    return lo, hi


def peak_ppm(ppm: np.ndarray, values: np.ndarray,
             window: Optional[tuple[float, float]] = None) -> Optional[float]:
    """Where the strongest point of ``values`` sits, inside ``window``."""
    mask = np.ones_like(ppm, dtype=bool)
    if window is not None:
        lo, hi = sorted(window)
        mask = (ppm >= lo) & (ppm <= hi)
    if not mask.any():
        return None
    i = int(np.argmax(np.where(mask, values, -np.inf)))
    return float(ppm[i])


# ---------------------------------------------------------------------------
# Reading a spectrum: the height to show it at, its phase, its quality
# ---------------------------------------------------------------------------

#: The residual water band of a 1H spectrum, left out when the height is
#: fitted (Osprey and Gannet fit up to about 4.25 ppm): a water peak fifty
#: times NAA would otherwise flatten every metabolite.
WATER_BAND_1H: tuple[float, float] = (4.4, 5.0)
#: Where the zero-order phase is judged (the singlets NAA, Cr, Cho; spant).
PHASE_WINDOW_1H: tuple[float, float] = (1.8, 4.0)
#: Where NAA is looked for, for the SNR and its line width (FID-A).
NAA_WINDOW_1H: tuple[float, float] = (1.9, 2.1)


def y_range(x: np.ndarray, y: np.ndarray, x0: float, x1: float, *,
            exclude: Optional[tuple[float, float]] = None, margin: float = 0.08,
            robust: bool = False) -> Optional[tuple[float, float]]:
    """The height that shows ``y`` over the visible ``[x0, x1]``, leaving
    out ``exclude`` (the water band) when enough points remain. ``robust``
    takes the 0.5 and 99.5 percentiles instead of the extremes. None when
    nothing is visible."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    vis = (x >= lo_x) & (x <= hi_x) & np.isfinite(y)
    if not vis.any():
        return None
    keep = vis
    if exclude is not None:
        out = vis & ~((x >= exclude[0]) & (x <= exclude[1]))
        if out.sum() >= 8:
            keep = out
    v = y[keep]
    if robust and v.size > 20:
        lo, hi = (float(q) for q in np.percentile(v, (0.5, 99.5)))
    else:
        lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if not span > 0:
        span = abs(hi) or 1.0
    return lo - margin * span, hi + margin * span


def zero_filled(ppm: np.ndarray, spec: np.ndarray, factor: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """The same spectrum on a ``factor`` times finer grid (the FID padded
    with zeros). A peak between two bins is read at the nearest one, where
    its dispersion is not zero: a phase error of up to atan(2 pi delta /
    width), 17 degrees for a narrow line, that only a finer grid removes.
    ``spec`` is what :func:`spectrum` returns (fftshifted)."""
    n = spec.shape[0]
    if n < 4 or factor <= 1:
        return ppm, spec
    fid = np.fft.ifft(np.fft.ifftshift(spec, axes=0), axis=0)
    padded = np.zeros((n * factor,) + fid.shape[1:], dtype=np.complex128)
    padded[:n] = fid
    fine = np.fft.fftshift(np.fft.fft(padded, axis=0), axes=0)
    # Both grids start at -1 / (2 dwell): the fine one steps 1/factor as far.
    step = (ppm[1] - ppm[0]) / factor
    return ppm[0] + step * np.arange(n * factor), fine


def auto_phase0(ppm: np.ndarray, spec: np.ndarray,
                window: tuple[float, float] = PHASE_WINDOW_1H) -> float:
    """The zero-order phase (degrees) that makes the peaks in ``window``
    upright and absorptive: the phase at which the real part reaches its
    highest point there. At the top of a Lorentzian the dispersion is zero,
    so its real height is ``A cos(error)``, largest exactly when the phase
    is right; an integral criterion is biased by the dispersion tails of a
    peak near the window's edge (NAA at 2.01 against 1.8). ``spec``
    unphased. A 0.5-degree search, then 0.05 around the best."""
    ppm, spec = zero_filled(np.asarray(ppm, dtype=float),
                            np.asarray(spec, dtype=np.complex128))
    lo, hi = sorted(window)
    mask = (ppm >= lo) & (ppm <= hi)
    if mask.sum() < 4:
        mask = np.ones_like(ppm, dtype=bool)
    seg = spec[mask]

    def best(trials: np.ndarray) -> float:
        heights = (seg[None, :] * np.exp(1j * np.deg2rad(trials))[:, None]).real.max(axis=1)
        return float(trials[int(np.argmax(heights))])

    coarse = best(np.arange(-180.0, 180.0, 0.5))
    fine = best(coarse + np.arange(-0.5, 0.55, 0.05))
    return float((fine + 180.0) % 360.0 - 180.0)


def _detrended(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.size < 4:
        return y - y.mean()
    coeff = np.polyfit(x, y, 2)
    return y - np.polyval(coeff, x)


def noise_band(ppm: np.ndarray, nucleus: str = "1H", width: float = 0.8) -> tuple[float, float]:
    """Where the noise is measured: -2 to 0 ppm when the spectrum reaches it
    (FID-A's band, empty of metabolites); else the edge of the spectrum
    farthest from 0 to 5 ppm (a 7 T spectrum 2000 Hz wide spans 1.3 to 8)."""
    lo, hi = float(np.min(ppm)), float(np.max(ppm))
    if nucleus.upper() == "1H" and lo <= -1.8:
        return -2.0, 0.0
    below = (lo, min(lo + width, 0.0))
    above = (max(hi - width, 9.0 if hi > 9.5 else hi - width), hi)
    if below[1] - below[0] >= width * 0.5:
        return below
    return above


def qc_metrics(ppm: np.ndarray, spec: np.ndarray, spectrometer_mhz: float,
               nucleus: str = "1H") -> dict:
    """What a spectroscopist checks first, from the PHASED, unbroadened
    spectrum: ``snr`` (the NAA peak above its local baseline over the SD of
    the detrended noise band: FID-A's definition, with the baseline taken
    out because raw data rarely sits on zero), ``naa_ppm`` (where NAA sits:
    a shift means a mis-referenced scan), ``fwhm_hz`` (its line width at half
    its height above the baseline) and ``noise_band``. 1H only; empty
    otherwise."""
    if nucleus.upper() != "1H":
        return {}
    ppm = np.asarray(ppm, dtype=float)
    real = np.asarray(spec).real.astype(float)
    order = np.argsort(ppm)
    xs, ys = ppm[order], real[order]
    lo, hi = NAA_WINDOW_1H
    sig = (xs >= lo) & (xs <= hi)
    if sig.sum() < 2:
        return {}
    j = int(np.flatnonzero(sig)[np.argmax(ys[sig])])
    peak = ys[j]
    # The local baseline: the low tenth of the 0.6 ppm around NAA.
    near = (xs >= xs[j] - 0.3) & (xs <= xs[j] + 0.3)
    base = float(np.percentile(ys[near], 10)) if near.sum() > 4 else 0.0
    height = peak - base
    nb = noise_band(ppm, nucleus)
    noise = (xs >= nb[0]) & (xs <= nb[1])
    out: dict = {"noise_band": nb, "naa_ppm": float(xs[j])}
    if noise.sum() >= 8:
        sd = float(np.std(_detrended(xs[noise], ys[noise])))
        out["snr"] = float(height / sd) if sd > 0 else float("inf")
    if height > 0:
        half = base + height / 2.0
        left = j
        while left > 0 and ys[left] > half:
            left -= 1
        right = j
        while right < ys.size - 1 and ys[right] > half:
            right += 1
        if ys[left] <= half < ys[left + 1] and ys[right] <= half < ys[right - 1]:
            xl = np.interp(half, [ys[left], ys[left + 1]], [xs[left], xs[left + 1]])
            xr = np.interp(half, [ys[right], ys[right - 1]], [xs[right], xs[right - 1]])
            out["fwhm_hz"] = float(abs(xr - xl) * spectrometer_mhz)
    return out


def describe_qc(qc: dict) -> str:
    """"SNR 45 · NAA 6.1 Hz wide at 2.01 ppm"."""
    parts = []
    if "snr" in qc:
        parts.append(f"SNR {qc['snr']:.0f}")
    if "fwhm_hz" in qc:
        parts.append(f"NAA {qc['fwhm_hz']:.1f} Hz wide")
    if "naa_ppm" in qc:
        parts.append(f"at {qc['naa_ppm']:.2f} ppm")
    return " · ".join(parts)


__all__ = [
    "DEFAULT_PPM_RANGE_1H", "METABOLITES_1H", "NAA_WINDOW_1H", "NUCLEUS_REFERENCE_PPM",
    "PHASE_WINDOW_1H", "WATER_BAND_1H", "apodize", "auto_phase0", "coil_combine",
    "combine", "default_ppm_range", "describe_qc", "metabolites_for", "noise_band",
    "part", "peak_ppm", "qc_metrics", "spectrum", "stagger", "y_range", "zero_filled",
]

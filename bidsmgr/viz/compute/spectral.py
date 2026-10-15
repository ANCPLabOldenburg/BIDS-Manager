"""The power spectrum: ONE Welch estimate, used by every viewer.

The MEG/EEG viewer and the Converter's per-row button used to compute the
spectrum two different ways (Welch over the samples, and ``Raw.compute_psd``,
which silently drops stim and misc channels), so the same file gave two
different answers depending on where it was asked. Now both call
:func:`psd`.

From the SAMPLES rather than through ``Raw.compute_psd``: that refuses a stim
channel whatever the picks, and a physio recording of nothing but a trigger
came back as "picks yielded no channels". A spectrum is a question about
numbers, not about what kind of sensor produced them.

Qt-free; an FFT, so it runs on a QThread (guard 8b), never a pool.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .filters import FilterSpec, apply, validate

#: Highest frequency shown by default: above it a neural or physiological
#: spectrum is noise floor, and a 5 kHz MEG spectrum squeezes everything
#: interesting into the first pixels.
DEFAULT_FMAX = 150.0
#: Welch window. Four seconds where the recording allows: respiration lives
#: near 0.3 Hz and a shorter window puts it in the first bin.
WINDOW_S = 4.0


def welch(data: np.ndarray, sfreq: float, *, seconds: float = WINDOW_S,
          fmax: Optional[float] = DEFAULT_FMAX) -> tuple[np.ndarray, np.ndarray]:
    """``(freqs, power)`` with power ``(channels, freqs)``. Gaps count as
    zeros, which is what the filter saw."""
    from scipy.signal import welch as _welch

    data = np.nan_to_num(np.atleast_2d(np.asarray(data, dtype=np.float64)))
    n = data.shape[-1]
    nperseg = int(min(n, max(256, round(sfreq * seconds))))
    freqs, power = _welch(data, fs=sfreq, nperseg=max(1, nperseg), axis=-1)
    if fmax is not None:
        keep = freqs <= min(sfreq / 2.0, float(fmax))
        freqs, power = freqs[keep], power[:, keep]
    return freqs, np.atleast_2d(power)


def psd(src, indices: Optional[list[int]] = None, *,
        spec: Optional[FilterSpec] = None, fmax: Optional[float] = DEFAULT_FMAX) -> dict:
    """The spectrum of ``src``'s channels ``indices`` (all when None).

    ``spec``: the filter to apply first, or None for the raw signal. A
    spectrum of the filtered signal shows what the filter did; of the raw
    one, what the recording holds. The caller chooses and the result says
    which it is.
    """
    if indices is None:
        indices = list(range(len(src.ch_names)))
    data = src.read(indices, 0, src.n_times)
    filtered = False
    messages: list[str] = []
    if spec is not None and spec.active:
        spec = validate(spec, src.sfreq)
        data, messages = apply(data, src.sfreq, spec)
        filtered = True
    freqs, power = welch(data, src.sfreq, fmax=fmax)
    return {
        "freqs": freqs,
        "data": power,
        "ch_names": [src.ch_names[i] for i in indices],
        "ch_types": [src.ch_types[i] for i in indices],
        "filtered": filtered,
        "filter": spec.describe() if filtered else "",
        "messages": messages,
        "title": f"Power spectral density: {src.path.name}",
    }


def to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 1e-30))


__all__ = ["DEFAULT_FMAX", "WINDOW_S", "psd", "to_db", "welch"]

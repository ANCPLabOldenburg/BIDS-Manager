"""The whole recording at a glance: how active it is, bin by bin.

What the overview bar under the traces draws behind its events and its view
rectangle (mne-qt-browser's overview, AcqKnowledge's navigator): for each of
a few hundred time bins, the root mean square of up to 32 signal channels,
each robustly normalised (median removed, divided by its median absolute
deviation) so one loud channel cannot set the profile, averaged, and scaled
to 0..1 by the 99th percentile. An artefact, a dropout, a flat stretch shows
as a spike, a gap or a dip before anyone pages to it.

Qt-free. Runs on a worker: it reads every sample of the channels it uses.
"""

from __future__ import annotations

import numpy as np

#: Bins across the recording.
BINS = 800
#: Channels averaged, at most (evenly chosen).
MAX_CHANNELS = 32
#: Types that are not signals: their activity says nothing about the data.
NOT_SIGNAL = frozenset({"stim", "syst", "ias", "chpi", "exci", "misc"})


def activity(src, bins: int = BINS, *, cancel=None) -> dict:
    """``{"profile": (bins,) 0..1, "gaps": (bins,) bool, "bins": bins}``."""
    idx = [i for i, t in enumerate(src.ch_types) if t not in NOT_SIGNAL]
    if not idx:
        idx = list(range(len(src.ch_types)))
    if len(idx) > MAX_CHANNELS:
        idx = [idx[k] for k in np.linspace(0, len(idx) - 1, MAX_CHANNELS).astype(int)]
    n = src.n_times
    bins = max(1, min(int(bins), n))
    edges = np.linspace(0, n, bins + 1).astype(int)
    total = np.zeros(bins)
    used = np.zeros(bins)
    gaps = np.zeros(bins, dtype=bool)
    for ch in idx:
        if cancel is not None and getattr(cancel, "cancelled", False):
            raise RuntimeError("cancelled")
        x = src.read([ch], 0, n)[0]
        mask = src.gap_mask([ch], 0, n)
        missing = mask[0] if mask is not None and mask.shape[-1] == n else np.zeros(n, bool)
        good = x[~missing]
        if good.size < 2:
            continue
        med = float(np.median(good[:: max(1, good.size // 100_000)]))
        mad = float(np.median(np.abs(good[:: max(1, good.size // 100_000)] - med))) * 1.4826
        if not mad > 0:
            continue
        z = np.where(missing, np.nan, (x - med) / mad)
        sq = np.where(np.isfinite(z), z * z, 0.0)
        cnt = np.isfinite(z).astype(float)
        s = np.add.reduceat(sq, edges[:-1])
        c = np.add.reduceat(cnt, edges[:-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            rms = np.sqrt(np.where(c > 0, s / c, np.nan))
        ok = np.isfinite(rms)
        total[ok] += rms[ok]
        used[ok] += 1
        gaps |= c == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(used > 0, total / used, 0.0)
    top = float(np.percentile(profile[used > 0], 99)) if (used > 0).any() else 1.0
    profile = np.clip(profile / (top or 1.0), 0.0, 1.0)
    return {"profile": profile, "gaps": gaps & (used == 0), "bins": bins}


__all__ = ["BINS", "MAX_CHANNELS", "activity"]

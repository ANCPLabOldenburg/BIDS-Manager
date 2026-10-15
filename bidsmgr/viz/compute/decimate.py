"""Drawing a long trace in a narrow pane without losing a single feature.

Qt-free; called on every redraw of a traces or spectrum canvas.
"""

from __future__ import annotations

import numpy as np


def peak_decimate(times, values, pixels: int):
    """Reduce a trace to about two points per pixel, KEEPING THE EXTREMES.

    A pane is on the order of a thousand pixels wide. Asking Qt to stroke a
    forty-minute recording sample by sample is a million points into a
    thousand columns, a thousand of them per column, and all but two of
    those thousand land on a pixel another one already covered: the work is
    real and the result is identical. That is what made Fit all freeze.

    Taking every n-th sample would be wrong, not just lossy: the peak of an
    R wave or a trigger one sample wide falls between the samples kept and
    the feature DISAPPEARS, which is worse than slow because it is quietly
    incorrect. So each column keeps its MINIMUM and its MAXIMUM, which is
    the standard answer and preserves the envelope exactly: what is drawn
    covers the same pixels the full trace would have.

    A gap stays a gap. A column is NaN only when every sample in it is,
    because a column holding one dropped sample among a thousand good ones
    still has a range worth drawing, and propagating the NaN would open a
    hole the recording does not have.
    """
    n = int(np.asarray(values).size)
    target = max(256, int(pixels) * 2)
    if n <= target:
        return times, values
    buckets = target // 2
    usable = (n // buckets) * buckets
    if usable < buckets * 2:
        return times, values

    block = np.asarray(values[:usable], dtype=float).reshape(buckets, -1)
    # fmin / fmax skip NaN and give NaN only for a column that is ALL NaN,
    # which is exactly the rule above: one pass each, no mask, no copy, no
    # all-NaN warning. The masked version cost 10 ms a channel on a
    # four-million-sample physio run, on every redraw.
    lo = np.fmin.reduce(block, axis=1)
    hi = np.fmax.reduce(block, axis=1)

    t = np.asarray(times[:usable], dtype=float).reshape(buckets, -1)
    out_t = np.empty(buckets * 2, dtype=float)
    out_y = np.empty(buckets * 2, dtype=float)
    out_t[0::2] = t[:, 0]
    out_t[1::2] = t[:, -1]
    out_y[0::2] = lo
    out_y[1::2] = hi
    # Whatever the reshape could not cover, at most one bucket's worth, so
    # the trace still reaches the right-hand edge.
    if usable < n:
        out_t = np.concatenate([out_t, np.asarray(times[usable:], dtype=float)])
        out_y = np.concatenate([out_y, np.asarray(values[usable:], dtype=float)])
    return out_t, out_y


__all__ = ["peak_decimate"]

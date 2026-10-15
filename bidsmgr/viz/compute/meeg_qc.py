"""A quick quality check of an MEG or EEG recording, computed when asked.

The signal counterpart of the 4-D volume's QC rows: what a reviewer looks
for before analysis, so the viewer can say WHICH channels to doubt and
WHERE in the recording to look, before a full pipeline such as MEEGqc is
run. The measures are MEEGqc's and MNE's: the standard deviation (STD) and
the peak-to-peak amplitude (PtP) of each channel in short segments, MNE's
muscle detector, and line noise against its neighbourhood.

EVERY measure is taken within ONE channel type. Magnetometers (T),
gradiometers (T/m) and EEG (V) measure different things in different units,
so a channel is only ever compared with channels of its own type, and a
segment is judged type by type: "20 % of the gradiometers are off here" is a
statement; "20 % of all channels" mixing gradiometers and EEG is not. No
fixed amplitude threshold suits every cap, system and montage either:
everything is RELATIVE, to the channel's own level and to its type's.

Measured after a zero-phase high-pass (``highpass_hz``, 1 Hz by default,
applied in the frequency domain on padded chunks):
slow drifts differ from sensor to sensor and, left in, decided which
magnetometer looked noisy on a resting recording (17 of 102 flagged).

Per channel, against the other channels of its type:

* **noisy**: its level, log STD or log PtP, more than ``noisy_z`` robust
  standard deviations above its type's (a spread of at least a tenth of a
  decade is assumed, so "noisy" means at least about twice the type's level)
  AND it does not follow the other channels of its type: the 98th
  percentile of its absolute correlation with them (the PREP pipeline's
  measure) below ``min_correlation``. Without that second condition the
  check flagged 16 of 102 magnetometers of a raw resting recording that
  MNE's ``find_bad_channels_maxwell`` passes: every one correlated at 0.79
  to 1.00 with the others, because what made them loud was the room's
  field, which they share;
* **uncorrelated**: at a normal level but not following its type;
* **flat**: its level below ``flat_ratio`` of its type's median;
* **line noise**: power at the line frequency more than ``line_db`` dB above
  its type's typical, after the ratio to its neighbourhood.

A channel's level is its QUIET level (the 20th percentile over segments):
blinks, frequent enough to lift the median of the frontal channels, are not
a reason to drop them, while a truly noisy channel is noisy even then.

Per segment (``segment_s`` seconds), per type:

* **off**: the share of the type's channels whose STD or PtP in the segment
  is more than ``segment_factor`` times their own usual level (or less than
  its inverse): movement, a jump, a loose reference;
* **muscle**: the muscle band's share of the power, z-scored over time per
  channel and averaged over the type. MNE's ``annotate_muscle_zscore``
  z-scores the band's own envelope; the SHARE is used here so that a
  broadband burst (movement, a loose electrode), which raises every band
  alike, is reported as channels off and not as muscle.

A segment is FLAGGED when one type has ``segment_share`` of its channels off,
or a mean muscle z above ``muscle_z``. The parameters are
:class:`~bidsmgr.viz.settings.MeegQcSettings`, set by the user.

Qt-free. Runs on a worker: it reads every sample of every data channel.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..settings import MeegQcSettings

#: The least spread (decades) the channels of a type are taken to have.
MIN_LOG_SD = 0.1
#: The percentile of a channel's segment levels taken as its level.
QUIET_PERCENTILE = 20.0
#: Channels a quality check is about (not triggers, not housekeeping).
DATA_TYPES = frozenset({"eeg", "mag", "grad", "seeg", "ecog", "dbs"})
#: How each type is named on screen.
TYPE_LABELS = {"mag": "MEG mag", "grad": "MEG grad", "eeg": "EEG", "seeg": "SEEG",
               "ecog": "ECoG", "dbs": "DBS"}
#: Values (channels x samples) read and filtered at once: about 80 MB.
CHUNK_VALUES = 10_000_000
#: Segments the channel correlations are measured in, at most (spread
#: evenly over the recording).
CORRELATION_SEGMENTS = 120

HELP = {
    "off": "The share of this type's channels whose STD or peak-to-peak range in the "
           "segment is beyond their usual level by the factor set (movement, a jump, a "
           "loose reference). The dashed line is the share that flags a segment.",
    "muscle": "The muscle band's share of the power, z-scored over time per channel and "
              "averaged over this type (after MNE's muscle detector, as a share so that "
              "movement is not called muscle). Above the dashed line is likely muscle: "
              "jaw clench, swallowing, frowning.",
    "channels": "Each channel is compared ONLY with the channels of its own type "
                "(magnetometers, gradiometers and EEG measure different things in "
                "different units). Noisy: its STD or peak-to-peak well above its type's "
                "AND it does not follow the other channels of its type (a large signal "
                "the others share is the brain, the heart or the room, not a broken "
                "sensor). Uncorrelated: it does not follow them at a normal level. Flat: "
                "far below its type. Line noise: power at the line frequency well above "
                "its type's typical.",
    "reading": "A quick check, relative to THIS recording: a uniformly noisy recording "
               "flags nothing, and the thresholds are yours to set (QC settings). Use it "
               "to decide where to look before a full pipeline such as MEEGqc.",
}


def _robust(values: np.ndarray) -> tuple[float, float]:
    """Median and robust standard deviation (1.4826 x MAD)."""
    v = values[np.isfinite(values)]
    if not v.size:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, (mad if mad > 0 else (float(np.std(v)) or 1.0))


def _log_z(levels: np.ndarray) -> np.ndarray:
    """Robust z of log levels within one type (spread at least MIN_LOG_SD)."""
    logs = np.log10(np.maximum(levels, 1e-30))
    med, sd = _robust(logs)
    return (logs - med) / max(sd, MIN_LOG_SD)


def _projector(src, names: list[str]) -> Optional[np.ndarray]:
    """The recording's SSP projectors over ``names`` as ``U``, an
    orthonormal basis of the projection vectors (the projector is
    ``I - U U^T``, as MNE builds it), or None when none is stored. Raw MEG
    carries them to remove the room's field."""
    raw = getattr(src, "raw", None)
    info = getattr(raw, "info", None)
    projs = list(info.get("projs") or []) if info is not None else []
    vectors = []
    for proj in projs:
        try:
            data = proj["data"]
            where = {name: j for j, name in enumerate(data["col_names"])}
            rows = np.atleast_2d(np.asarray(data["data"], dtype=float))
        except (KeyError, TypeError, ValueError):
            continue
        for row in rows:
            v = np.array([row[where[name]] if name in where else 0.0 for name in names])
            norm = float(np.linalg.norm(v))
            if norm > 0:
                vectors.append(v / norm)
    if not vectors:
        return None
    u, sv, _vt = np.linalg.svd(np.asarray(vectors).T, full_matrices=False)
    return u[:, sv > sv.max() * 1e-6]


def _highpass(x: np.ndarray, sfreq: float, hz: float) -> np.ndarray:
    """Zero-phase high-pass of each row: the magnitude response of a
    forward-backward 4th-order Butterworth (``1 / (1 + (hz / f)^8)``) applied
    in the frequency domain, several times cheaper than filtering twice in
    time. Each row's end-to-end line is taken out first so the transform
    does not see a step where the end wraps round to the start."""
    n = x.shape[1]
    if n < 4:
        return x
    ramp = np.linspace(0.0, 1.0, n)
    x = x - x[:, :1] - (x[:, -1:] - x[:, :1]) * ramp
    freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
    with np.errstate(divide="ignore", over="ignore"):
        gain = 1.0 / (1.0 + (hz / np.maximum(freqs, 1e-12)) ** 8)
    gain[0] = 0.0
    return np.fft.irfft(np.fft.rfft(x, axis=1) * gain, n=n, axis=1)


def _follow(x: np.ndarray) -> np.ndarray:
    """How well each row follows the others: the 98th percentile of its
    absolute correlation with them (the PREP pipeline's measure)."""
    z = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt((z * z).sum(axis=1, keepdims=True))
    sd[sd == 0] = np.inf
    z = z / sd
    c = np.abs(z @ z.T)
    # The row's own 1 is replaced by 0: one low value among the row's moves
    # its 98th percentile by nothing that matters, and a plain percentile is
    # a hundred times faster than a NaN-aware one.
    np.fill_diagonal(c, 0.0)
    return np.percentile(c, 98, axis=1)


def type_label(ch_type: str) -> str:
    return TYPE_LABELS.get(ch_type, ch_type.upper())


def quality(src, *, settings: Optional[MeegQcSettings] = None,
            line_freq: Optional[float] = None, exclude: Optional[set] = None,
            cancel=None, progress=None) -> dict:
    """The quality check of ``src`` (a :class:`~bidsmgr.viz.data.signal.SignalSource`).

    ``exclude``: channel names left out (those already marked bad).
    Raises ``ValueError`` when there is nothing to check."""
    s = settings if settings is not None else MeegQcSettings()
    if not (s.use_std or s.use_ptp):
        raise ValueError("choose at least one measure: STD or peak-to-peak")
    exclude = set(exclude or ())
    picks = [i for i, t in enumerate(src.ch_types)
             if t in DATA_TYPES and src.ch_names[i] not in exclude]
    if not picks:
        raise ValueError("no MEG or EEG channel to check")
    # Only the types asked for (none of them in this recording: all of them,
    # rather than nothing).
    wanted = [i for i in picks if src.ch_types[i] in set(s.types)] if s.types else []
    picks = wanted or picks
    sfreq = float(src.sfreq)
    seg = max(8, int(round(s.segment_s * sfreq)))
    n_seg = int(src.n_times // seg)
    if n_seg < 3:
        raise ValueError(f"the recording is too short for {s.segment_s:g} s segments")
    freqs = np.fft.rfftfreq(seg, 1.0 / sfreq)
    window = np.hanning(seg).astype(np.float32)
    lo_band, hi_band = sorted((float(s.muscle_low), float(s.muscle_high)))
    muscle_ok = bool(s.muscle and sfreq / 2.0 > hi_band and hi_band > lo_band)
    band = (freqs >= lo_band) & (freqs <= hi_band) if muscle_ok else None
    line = None
    if line_freq and 0 < float(line_freq) < sfreq / 2.0 - 6.0:
        lf = float(line_freq)
        line = ((np.abs(freqs - lf) <= 0.6),
                ((np.abs(freqs - lf) >= 1.5) & (np.abs(freqs - lf) <= 6.0)))
    highpass = (float(s.highpass_hz) if s.highpass_hz and 0 < float(s.highpass_hz) < sfreq / 2.0
                else 0.0)
    # Room for the filter to settle at each chunk's edges.
    pad = int(min(10.0, 4.0 / highpass) * sfreq) if highpass else 0
    names = [src.ch_names[i] for i in picks]
    types = [src.ch_types[i] for i in picks]
    by_type: dict[str, np.ndarray] = {}
    for k, t in enumerate(types):
        by_type.setdefault(t, []).append(k)
    by_type = {t: np.asarray(rows) for t, rows in by_type.items()}
    projector = _projector(src, names) if s.apply_proj else None
    n = len(picks)
    std = np.full((n, n_seg), np.nan)
    ptp = np.full((n, n_seg), np.nan)
    muscle = np.full((n, n_seg), np.nan) if muscle_ok else None
    power_sum = np.zeros((n, freqs.size)) if line is not None else None
    # How well each channel follows the others of its type, in a bounded
    # sample of segments (the cost is channels squared per segment).
    corr_every = max(1, n_seg // CORRELATION_SEGMENTS)
    corr = np.full((n, n_seg), np.nan)
    chunk = max(1, int(CHUNK_VALUES // max(n * seg, 1)))
    for c0 in range(0, n_seg, chunk):
        if cancel is not None and getattr(cancel, "cancelled", False):
            raise RuntimeError("cancelled")
        c1 = min(n_seg, c0 + chunk)
        a0, a1 = c0 * seg, c1 * seg
        r0, r1 = max(0, a0 - pad), min(int(src.n_times), a1 + pad)
        x = src.read(picks, r0, r1)
        if projector is not None:
            # x - U (U^T x): a few vectors, never the channels-squared matrix.
            x -= projector @ (projector.T @ x)
        if highpass:
            x = _highpass(np.nan_to_num(x), sfreq, highpass)
        x = x[:, a0 - r0:a1 - r0]
        mask = src.gap_mask(picks, a0, a1)
        gaps = bool(mask is not None and mask.shape == x.shape and mask.any())
        if gaps:
            x[mask] = np.nan
        segs = x.reshape(n, c1 - c0, seg)
        cols = slice(c0, c1)
        if gaps:
            with np.errstate(invalid="ignore"):
                std[:, cols] = np.nanstd(segs, axis=2)
                ptp[:, cols] = np.nanmax(segs, axis=2) - np.nanmin(segs, axis=2)
        else:
            # Without gaps the plain reductions, several times faster.
            std[:, cols] = segs.std(axis=2)
            ptp[:, cols] = np.ptp(segs, axis=2)
        if muscle_ok or line is not None:
            mean = (np.nanmean(segs, axis=2, keepdims=True) if gaps
                    else segs.mean(axis=2, keepdims=True))
            centred = (segs - mean).astype(np.float32)
            if gaps:
                centred = np.nan_to_num(centred)
            power = np.abs(np.fft.rfft(centred * window, axis=2)) ** 2
            if muscle_ok:
                # The band's SHARE of the power above 1 Hz: a broadband burst
                # (movement, a loose electrode) raises every band alike and
                # is not muscle; muscle tilts the spectrum upwards.
                total = power[..., freqs >= 1.0].sum(axis=2) + 1e-30
                muscle[:, cols] = np.log10(power[..., band].sum(axis=2) / total + 1e-30)
            if power_sum is not None:
                power_sum += power.sum(axis=1)
        for i in range(c0, c1):
            if i % corr_every:
                continue
            for rows in by_type.values():
                if rows.size >= 3:
                    corr[rows, i] = _follow(np.nan_to_num(segs[rows, i - c0]))
        if progress is not None:
            progress(c1, n_seg)
    line_db = np.full(n, np.nan)
    if line is not None:
        mean_power = power_sum / n_seg
        at, around = line
        line_db = 10.0 * np.log10((mean_power[:, at].mean(axis=1) + 1e-30)
                                  / (np.median(mean_power[:, around], axis=1) + 1e-30))
    follows = np.nanmedian(corr, axis=1)

    quiet_std = np.nanpercentile(std, QUIET_PERCENTILE, axis=1)
    quiet_ptp = np.nanpercentile(ptp, QUIET_PERCENTILE, axis=1)
    std_z = np.zeros(n)
    ptp_z = np.zeros(n)
    flat = np.zeros(n, dtype=bool)
    line_excess = np.full(n, np.nan)
    # -- segments, per type ---------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        own_std = np.nanmedian(std, axis=1, keepdims=True)
        own_ptp = np.nanmedian(ptp, axis=1, keepdims=True)
        f = float(s.segment_factor)
        off = np.zeros((n, n_seg), dtype=bool)
        if s.use_std:
            r = std / own_std
            off |= (r > f) | (r < 1.0 / f)
        if s.use_ptp:
            r = ptp / own_ptp
            off |= (r > f) | (r < 1.0 / f)
    per_type: dict[str, dict] = {}
    flagged = np.zeros(n_seg, dtype=bool)
    reasons_seg: list[list[str]] = [[] for _ in range(n_seg)]
    kinds_seg: list[set] = [set() for _ in range(n_seg)]
    for t, rows in by_type.items():
        # Channels, against their type only.
        std_z[rows] = _log_z(quiet_std[rows])
        ptp_z[rows] = _log_z(quiet_ptp[rows])
        if s.use_std:
            flat[rows] |= quiet_std[rows] <= s.flat_ratio * float(np.nanmedian(quiet_std[rows]))
        if s.use_ptp:
            flat[rows] |= quiet_ptp[rows] <= s.flat_ratio * float(np.nanmedian(quiet_ptp[rows]))
        if line is not None:
            line_med, _sd = _robust(line_db[rows])
            line_excess[rows] = line_db[rows] - line_med
        # Segments: a flat channel is off everywhere and says nothing about when.
        live = rows[~flat[rows]]
        share = (off[live].mean(axis=0) if live.size else np.zeros(n_seg))
        muscle_t = None
        if muscle is not None and live.size:
            zs = np.empty((live.size, n_seg))
            for j, k in enumerate(live):
                med, sd = _robust(muscle[k])
                zs[j] = (muscle[k] - med) / sd
            muscle_t = np.nanmean(zs, axis=0)
        label = type_label(t)
        t_flag = share >= s.segment_share
        for i in np.flatnonzero(t_flag):
            reasons_seg[i].append(f"{label}: {share[i] * 100:.0f} % of channels off")
            # Which measure: a jump moves the peak-to-peak range far more
            # than the standard deviation.
            if s.use_ptp and s.use_std:
                bad = live[off[live, i]]
                ratio_std = np.nanmedian(std[bad, i] / own_std[bad, 0])
                ratio_ptp = np.nanmedian(ptp[bad, i] / own_ptp[bad, 0])
                kinds_seg[i].add("jump" if ratio_ptp > 1.5 * ratio_std else "noise")
            else:
                kinds_seg[i].add("noise")
        if muscle_t is not None:
            m_flag = muscle_t > s.muscle_z
            for i in np.flatnonzero(m_flag):
                reasons_seg[i].append(f"{label}: muscle z {muscle_t[i]:.1f}")
                kinds_seg[i].add("muscle")
            t_flag = t_flag | m_flag
        flagged |= t_flag
        with np.errstate(invalid="ignore", divide="ignore"):
            std_rel = (np.nanmedian(std[live] / own_std[live], axis=0) if live.size
                       else np.full(n_seg, np.nan))
            ptp_rel = (np.nanmedian(ptp[live] / own_ptp[live], axis=0) if live.size
                       else np.full(n_seg, np.nan))
            channel_map = np.clip(np.log2(std[rows] / own_std[rows]), -3.0, 3.0)
        per_type[t] = {"label": label, "n": int(rows.size), "share": share,
                       "muscle": muscle_t, "flagged": t_flag, "std": std_rel,
                       "ptp": ptp_rel, "map": channel_map.astype(np.float32),
                       "names": [names[k] for k in rows]}

    channels = []
    for k in range(n):
        reasons = []
        loud = (s.use_std and std_z[k] > s.noisy_z) or (s.use_ptp and ptp_z[k] > s.noisy_z)
        # A channel that follows the others of its type is picking up real
        # fields (the brain, the heart, the room), however large; one that
        # does not, at a normal level, is broken in a quieter way.
        alone = bool(s.min_correlation > 0 and np.isfinite(follows[k])
                     and follows[k] < s.min_correlation)
        if flat[k]:
            reasons.append("flat")
        elif loud and (alone or s.min_correlation <= 0 or not np.isfinite(follows[k])):
            reasons.append("noisy")
        elif alone:
            reasons.append("uncorrelated")
        if np.isfinite(line_excess[k]) and line_excess[k] > s.line_db:
            reasons.append("line noise")
        channels.append({
            "name": names[k], "type": types[k], "label": type_label(types[k]),
            "std_z": float(std_z[k]), "ptp_z": float(ptp_z[k]),
            "std": float(quiet_std[k]), "ptp": float(quiet_ptp[k]),
            "line_db": float(line_excess[k]) if np.isfinite(line_excess[k]) else None,
            "off_share": float(np.mean(off[k])),
            "follows": float(follows[k]) if np.isfinite(follows[k]) else None,
            "loud": bool(loud),
            "reasons": reasons,
        })

    times = src.start_time + np.arange(n_seg) * (seg / sfreq)
    parts = []
    for t, info in per_type.items():
        rows = by_type[t]
        counted = {r: sum(r in channels[k]["reasons"] for k in rows)
                   for r in ("noisy", "flat", "uncorrelated", "line noise")}
        parts.append(f"{info['label']}: {counted['noisy']} noisy, {counted['flat']} flat, "
                     f"{counted['uncorrelated']} uncorrelated, {counted['line noise']} with "
                     f"line noise of {info['n']}")
    n_flag = int(flagged.sum())
    summary = ("; ".join(parts) + f". {n_flag} of {n_seg} segments flagged "
               f"({100.0 * n_flag / n_seg:.0f} %, {n_flag * seg / sfreq:.0f} s)")
    return {
        "segment_s": seg / sfreq, "times": times, "flagged": flagged,
        "segment_reasons": reasons_seg, "segment_kinds": kinds_seg, "types": per_type,
        "channels": channels,
        "suggested_bads": [c["name"] for c in channels
                           if {"noisy", "flat", "uncorrelated"} & set(c["reasons"])],
        "projected": projector is not None,
        "summary": summary, "help": HELP,
        "line_freq": float(line_freq) if line is not None else None,
        "muscle_checked": muscle_ok, "settings": s.model_dump(),
        "start_time": float(src.start_time),
    }


#: The plots the QC panel can show, per channel type, in their default
#: order: (id, title, unit, what it shows).
METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("off", "channels off", "%",
     "The share of this type's channels whose STD or peak-to-peak range in each segment "
     "is beyond their usual level by the factor set. The dashed line is the share that "
     "flags a segment."),
    ("muscle", "muscle", "z",
     "The muscle band's share of the power, z-scored over time per channel and averaged "
     "over the type. Above the dashed line is likely muscle."),
    ("std", "STD", "x its level",
     "The median, over this type's channels, of each segment's standard deviation over "
     "the channel's own usual level: 1 is typical, the dashed line the factor that marks "
     "a channel off."),
    ("ptp", "peak-to-peak", "x its level",
     "The same for the peak-to-peak range, which a jump or a pop moves far more than the "
     "standard deviation."),
    ("map", "channel map", "",
     "Every channel of this type (rows) in every segment (columns): its STD over its own "
     "level, on a log2 scale from a quarter (dark) to four times (light). A bright row is "
     "a channel that is often off; a bright column, a moment many channels were. Click a "
     "cell to see that channel in that segment on the traces."),
)
METRIC_IDS = tuple(m[0] for m in METRICS)


def tracks(result: dict, metrics=("off", "muscle"), types=None) -> list[dict]:
    """The QC as tracks (``gui.viz.canvases.tracks``) on the recording's own
    time axis (seconds from its start): one per chosen metric and channel
    type, never a metric of two types on one axis. ``types`` None: all."""
    s = result.get("settings") or {}
    step = float(result["segment_s"])
    start = float(result.get("start_time", 0.0))
    x = np.asarray(result["times"], dtype=float) - start + step / 2.0
    meta = {m[0]: m for m in METRICS}
    out = []
    for metric in metrics:
        if metric not in meta:
            continue
        _mid, name, unit, note = meta[metric]
        for t, info in result["types"].items():
            if types and t not in types:
                continue
            label = info["label"]
            flagged = x[np.asarray(info["flagged"], dtype=bool)]
            track = {"id": f"{metric}:{t}", "title": f"{label}: {name}", "unit": unit,
                     "kind": "line", "x": x, "ticks": flagged, "colour": t,
                     "movable": True, "closable": True, "note": note, "near": step / 2.0,
                     "fmt": "{:.3g}"}
            if metric == "off":
                share = np.asarray(info["share"], dtype=float) * 100.0
                limit = float(s.get("segment_share", 0.2)) * 100.0
                track.update(ys=[share], rule=limit,
                             y_range=(0.0, max(float(np.nanmax(share)) if share.size else 0.0,
                                               limit * 1.2) * 1.05),
                             summary=(f"{flagged.size} segments flagged, at most "
                                      f"{float(np.nanmax(share)) if share.size else 0:.0f} % "
                                      f"of {info['n']} channels off"))
            elif metric == "muscle":
                if info["muscle"] is None:
                    continue
                m = np.asarray(info["muscle"], dtype=float)
                track.update(ys=[m], rule=float(s.get("muscle_z", 4.0)),
                             summary=f"highest {float(np.nanmax(m)):.1f} z")
            elif metric in ("std", "ptp"):
                v = np.asarray(info[metric], dtype=float)
                track.update(ys=[v], rule=float(s.get("segment_factor", 4.0)),
                             summary=f"highest {float(np.nanmax(v)):.2f} x")
            else:
                image = np.asarray(info["map"], dtype=np.float32)
                track.update(kind="image", image=image,
                             image_x=(float(x[0] - step / 2.0), float(x[-1] + step / 2.0)),
                             levels=(-2.0, 2.0), rows=list(info["names"]),
                             value_name="log2 of STD over its level", fmt="{:+.2f}",
                             height=2.5, ticks=np.empty(0),
                             summary=(f"{info['n']} channels, light = louder than usual; "
                                      "click a cell to go there"))
            out.append(track)
    return out


def bad_label(kinds: set) -> str:
    """The BAD_ label a flagged segment gets, by why it was flagged."""
    if "muscle" in kinds:
        return "BAD_muscle"
    if kinds == {"jump"}:
        return "BAD_jump"
    return "BAD_noise"


def flagged_segments(result: dict) -> list[tuple[float, float]]:
    """``(onset, duration)`` of every flagged segment, neighbours merged."""
    out: list[list[float]] = []
    step = float(result["segment_s"])
    for t, bad in zip(result["times"], result["flagged"]):
        if not bad:
            continue
        if out and abs(out[-1][0] + out[-1][1] - t) < 1e-9:
            out[-1][1] += step
        else:
            out.append([float(t), step])
    return [(a, b) for a, b in out]


__all__ = ["DATA_TYPES", "HELP", "METRICS", "METRIC_IDS", "TYPE_LABELS", "bad_label",
           "flagged_segments", "quality", "tracks", "type_label"]

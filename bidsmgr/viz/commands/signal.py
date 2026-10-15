"""Commands for signals: time, channels, scale, filters, events.

Times are seconds of RECORDING time (0 at the first sample); events are in
run time and are converted with the source's ``start_time``.
"""

from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING

from . import command
from ..compute.filters import FilterSpec, validate
from ..scene import Span

if TYPE_CHECKING:  # pragma: no cover
    from ..store import SceneStore

#: What a recording opens showing: enough to see the shape of a cardiac or
#: respiratory cycle, short enough that a long recording does not draw a
#: million points before anyone asked.
DEFAULT_WIDTH_S = 10.0
#: Traces on screen before the scroll bar takes over.
DEFAULT_COUNT = 20
#: A page step: most of a window, so the next page shares some context.
PAGE_FRACTION = 0.8


def source(store: "SceneStore"):
    layer = store.scene.first("signal")
    if layer is None:
        return None
    return store.sources.get(layer.source)


def _clamp_t0(store, t0: float) -> float:
    src = source(store)
    tr = store.scene.traces
    if src is None:
        return max(0.0, float(t0))
    return float(min(max(0.0, t0), max(0.0, src.duration - tr.width)))


def _shown(store) -> list[int]:
    src = source(store)
    tr = store.scene.traces
    return [] if src is None else src.picks_for(tr.ch_type, tr.picks)


def opening_state(src) -> dict:
    """The traces state a recording opens in. EVERYTHING measured in the
    recording's own terms starts from its default: a ten-second window
    carried from a ten-second file onto a forty-minute one opens it zoomed
    into its first 0.4 percent, which reads as a broken load."""
    return dict(t0=0.0, width=min(DEFAULT_WIDTH_S, max(0.1, src.duration or 0.1)),
                ch_type="all", picks=None,
                count=max(1, min(DEFAULT_COUNT, len(src.ch_names))), offset=0,
                scale=1.0, normalize=False, hp=None, lp=None, notch=None,
                events=False, event_source="auto", bads=None, butterfly=False,
                bad_spans=None, annotate=False, selected_span=None)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


@command("time.set", "Go to a time", category="Time")
def time_set(store: "SceneStore", t0: float) -> set[str]:
    new = _clamp_t0(store, t0)
    if new == store.scene.traces.t0:
        return set()
    store.scene.traces.t0 = new
    return {"traces.time"}


@command("time.page", "Page through time", category="Time")
def time_page(store: "SceneStore", n: float = 1.0) -> set[str]:
    tr = store.scene.traces
    return time_set(store, tr.t0 + float(n) * PAGE_FRACTION * tr.width)


@command("time.start", "Go to the start", category="Time")
def time_start(store: "SceneStore") -> set[str]:
    return time_set(store, 0.0)


@command("time.end", "Go to the end", category="Time")
def time_end(store: "SceneStore") -> set[str]:
    src = source(store)
    return time_set(store, src.duration if src is not None else 0.0)


@command("time.width", "Set the time window", category="Time")
def time_width(store: "SceneStore", seconds: float) -> set[str]:
    src = source(store)
    tr = store.scene.traces
    hi = max(0.1, src.duration) if src is not None else 3600.0
    new = float(min(max(0.1, seconds), hi))
    if new == tr.width:
        return set()
    tr.width = new
    tr.t0 = _clamp_t0(store, tr.t0)
    return {"traces.time"}


@command("time.zoom", "Zoom time", category="Time")
def time_zoom(store: "SceneStore", factor: float) -> set[str]:
    """Widen (factor > 1) or narrow the window about its centre."""
    tr = store.scene.traces
    centre = tr.t0 + tr.width / 2.0
    changed = time_width(store, tr.width * float(factor))
    if changed:
        tr.t0 = _clamp_t0(store, centre - tr.width / 2.0)
    return changed


@command("time.fit", "Show the whole recording", category="Time")
def time_fit(store: "SceneStore") -> set[str]:
    src = source(store)
    if src is None:
        return set()
    changed = time_width(store, src.duration)
    return changed | time_set(store, 0.0)


@command("cursor.time", "Place the time cursor", category="Time")
def cursor_time(store: "SceneStore", t: Optional[float] = None) -> set[str]:
    """A vertical line at ``t`` (seconds of RUN time); None removes it.
    A click on the traces places it, with every channel's value there."""
    new = None if t is None else float(t)
    if store.scene.cursor.time == new:
        return set()
    store.scene.cursor.time = new
    return {"cursor.time"}


# ---------------------------------------------------------------------------
# Channels and scale
# ---------------------------------------------------------------------------


@command("traces.type", "Show one channel type", category="Channels", undoable=True)
def traces_type(store: "SceneStore", ch_type: str) -> set[str]:
    src = source(store)
    if src is not None and ch_type not in ("all", "mag+grad", *src.available_types):
        raise ValueError(f"this recording has no {ch_type!r} channels")
    tr = store.scene.traces
    if tr.ch_type == ch_type:
        return set()
    tr.ch_type = ch_type
    tr.offset = 0
    return {"traces.channels"}


@command("traces.pick", "Pick channels", category="Channels", undoable=True)
def traces_pick(store: "SceneStore", names: Optional[list[str]] = None) -> set[str]:
    """Show only ``names`` (None: every channel)."""
    src = source(store)
    tr = store.scene.traces
    if names is not None and src is not None:
        known = set(src.ch_names)
        names = [n for n in names if n in known]
        if len(set(names)) == len(src.ch_names):
            names = None
    if names == tr.picks:
        return set()
    tr.picks = names
    tr.offset = 0
    return {"traces.channels"}


@command("traces.focus", "Outline a channel over a stretch", category="Channels")
def traces_focus(store: "SceneStore", channel: Optional[str] = None, onset: float = 0.0,
                 duration: float = 0.0) -> set[str]:
    """Outline ``channel`` from ``onset`` (seconds of run time) for
    ``duration``; no channel removes the outline."""
    from ..scene import TraceFocus

    new = None if channel is None else TraceFocus(channel=channel, onset=float(onset),
                                                  duration=max(0.0, float(duration)))
    if store.scene.traces.focus == new:
        return set()
    store.scene.traces.focus = new
    return {"traces.look"}


@command("traces.go_to", "Go to a channel at a moment", category="Channels")
def traces_go_to(store: "SceneStore", channel: str, onset: float,
                 duration: float = 0.0) -> set[str]:
    """Show ``channel`` from ``onset`` (seconds of run time, as the time
    cursor) for ``duration``, as a click on a QC channel map asks: the
    channel brought on screen (every type shown when its own was filtered
    out), the stretch centred in the window (from its start when it is
    longer than the window), the time cursor at its start and the stretch
    outlined on the channel."""
    src = source(store)
    if src is None:
        return set()
    if channel not in src.ch_names:
        raise ValueError(f"no channel called {channel!r}")
    tr = store.scene.traces
    changed: set[str] = set()
    index = src.ch_names.index(channel)
    if index not in _shown(store):
        if tr.picks is not None and channel not in tr.picks:
            tr.picks = None
        if index not in _shown(store):
            tr.ch_type = "all"
        changed.add("traces.channels")
    pool = _shown(store)
    k = pool.index(index)
    if not tr.offset <= k < tr.offset + tr.count:
        tr.offset = max(0, min(k - tr.count // 2, len(pool) - tr.count))
        changed.add("traces.channels")
    start = float(onset) - float(src.start_time)
    length = max(0.0, float(duration))
    t0 = start if length >= tr.width else start + length / 2.0 - tr.width / 2.0
    changed |= time_set(store, t0)
    changed |= cursor_time(store, float(onset))
    changed |= traces_focus(store, channel, float(onset), length)
    return changed


@command("traces.count", "Traces on screen", category="Channels")
def traces_count(store: "SceneStore", n: int) -> set[str]:
    tr = store.scene.traces
    n = max(1, min(int(n), 500))
    if n == tr.count:
        return set()
    tr.count = n
    tr.offset = min(tr.offset, max(0, len(_shown(store)) - n))
    return {"traces.channels"}


@command("traces.scroll", "Scroll channels", category="Channels")
def traces_scroll(store: "SceneStore", n: int = 1) -> set[str]:
    tr = store.scene.traces
    total = len(_shown(store))
    new = max(0, min(tr.offset + int(n), max(0, total - tr.count)))
    if new == tr.offset:
        return set()
    tr.offset = new
    return {"traces.channels"}


@command("traces.page_channels", "Page through channels", category="Channels")
def traces_page_channels(store: "SceneStore", n: int = 1) -> set[str]:
    return traces_scroll(store, int(n) * store.scene.traces.count)


@command("traces.scale", "Trace amplitude", category="Channels")
def traces_scale(store: "SceneStore", value: Optional[float] = None,
                 factor: Optional[float] = None) -> set[str]:
    tr = store.scene.traces
    new = tr.scale if value is None else float(value)
    if factor is not None:
        new *= float(factor)
    new = max(0.01, min(new, 1000.0))
    if new == tr.scale:
        return set()
    tr.scale = new
    return {"traces.look"}


@command("traces.option", "Trace display option", category="Channels")
def traces_option(store: "SceneStore",
                  field: Literal["page_scale", "remove_dc", "clip", "butterfly"],
                  value: Optional[bool] = None) -> set[str]:
    """Flip (or set) a display option: per-page scale, DC removal,
    clipping, butterfly."""
    tr = store.scene.traces
    new = (not getattr(tr, field)) if value is None else bool(value)
    if new == getattr(tr, field):
        return set()
    setattr(tr, field, new)
    return {"traces.look"}


def bad_channels(store: "SceneStore") -> set:
    """The channels marked bad now: this session's list, else the
    recording's own."""
    tr = store.scene.traces
    if tr.bads is not None:
        return set(tr.bads)
    src = source(store)
    return set(getattr(src, "bads", set()) or set()) if src is not None else set()


@command("channels.toggle_bad", "Mark a channel bad or good", category="Channels",
         undoable=True)
def channels_toggle_bad(store: "SceneStore", name: str) -> set[str]:
    """A click on a channel's name: bad if it was good, good if bad. Kept in
    the session; writing it to the dataset's ``_channels.tsv`` is a separate,
    deliberate step."""
    src = source(store)
    if src is not None and name not in src.ch_names:
        raise ValueError(f"no channel called {name!r}")
    bads = bad_channels(store)
    bads ^= {name}
    store.scene.traces.bads = sorted(bads)
    return {"traces.look"}


# ---------------------------------------------------------------------------
# Bad segments (annotation mode)
# ---------------------------------------------------------------------------

#: Labels offered for a bad segment; any ``BAD_<reason>`` is accepted.
BAD_LABELS = ("BAD_", "BAD_muscle", "BAD_eye", "BAD_movement", "BAD_flat", "BAD_noise",
              "BAD_jump")


def bad_label(text: str) -> str:
    """``text`` as a bad-segment label: ``BAD_`` and a reason. Only BAD
    labels are segments to leave out; anything else is an event."""
    text = "".join(ch for ch in str(text).strip() if ch.isalnum() or ch in "_-")
    rest = text[3:].lstrip("_") if text.upper().startswith("BAD") else text
    return "BAD_" + rest


def bad_spans(store: "SceneStore") -> list[Span]:
    """The bad segments now: this session's list, else the recording's own."""
    tr = store.scene.traces
    if tr.bad_spans is not None:
        return list(tr.bad_spans)
    src = source(store)
    if src is None:
        return []
    return [Span(onset=float(e.onset), duration=float(e.duration), label=str(e.label) or "BAD_")
            for e in getattr(src, "bad_spans", [])]


def spans_changed(store: "SceneStore") -> bool:
    """Whether the session's bad segments differ from the recording's."""
    tr = store.scene.traces
    if tr.bad_spans is None:
        return False
    src = source(store)
    own = [(round(e.onset, 6), round(e.duration, 6), e.label)
           for e in getattr(src, "bad_spans", [])] if src is not None else []
    now = [(round(sp.onset, 6), round(sp.duration, 6), sp.label) for sp in tr.bad_spans]
    return sorted(own) != sorted(now)


def _set_spans(store: "SceneStore", spans: list[Span]) -> None:
    store.scene.traces.bad_spans = sorted(spans, key=lambda sp: (sp.onset, sp.duration))


def _clamp_span(store: "SceneStore", onset: float, duration: float) -> tuple[float, float]:
    src = source(store)
    start = float(getattr(src, "start_time", 0.0) or 0.0) if src is not None else 0.0
    end = start + (float(src.duration) if src is not None else onset + duration)
    lo = max(start, min(onset, onset + duration))
    hi = min(end, max(onset, onset + duration))
    step = 1.0 / float(src.sfreq) if src is not None and src.sfreq else 1e-3
    if hi - lo < step:
        raise ValueError("a bad segment needs a duration: drag across the traces")
    return lo, hi - lo


@command("annotate.mode", "Annotation mode", category="Annotation")
def annotate_mode(store: "SceneStore", value: Optional[bool] = None) -> set[str]:
    """In annotation mode a drag across the traces marks a bad segment
    (instead of scrolling), and segments can be moved, resized, relabelled
    and deleted."""
    tr = store.scene.traces
    new = (not tr.annotate) if value is None else bool(value)
    if new == tr.annotate:
        return set()
    tr.annotate = new
    if not new:
        tr.selected_span = None
    return {"traces.annotations"}


@command("annotate.label", "Label for new bad segments", category="Annotation")
def annotate_label(store: "SceneStore", label: str) -> set[str]:
    new = bad_label(label)
    if new == store.scene.traces.annotate_label:
        return set()
    store.scene.traces.annotate_label = new
    return {"traces.annotations"}


@command("annotate.add", "Mark a bad segment", category="Annotation", undoable=True)
def annotate_add(store: "SceneStore", onset: float, duration: float,
                 label: Optional[str] = None) -> set[str]:
    """A segment ``[onset, onset + duration)`` (seconds of run time) marked
    bad, with ``label`` (default: the annotation label)."""
    lo, length = _clamp_span(store, float(onset), float(duration))
    span = Span(onset=lo, duration=length,
                label=bad_label(label) if label else store.scene.traces.annotate_label)
    spans = bad_spans(store)
    if span not in spans:
        # The same segment twice says nothing more, and saved twice it is
        # two identical rows in the table.
        spans.append(span)
    _set_spans(store, spans)
    store.scene.traces.selected_span = store.scene.traces.bad_spans.index(span)
    return {"traces.annotations"}


@command("annotate.add_many", "Mark segments bad", category="Annotation", undoable=True)
def annotate_add_many(store: "SceneStore", segments: list[tuple[float, float]],
                      label: str = "BAD_") -> set[str]:
    """Many segments at once (``(onset, duration)`` pairs; the quality
    check's suggestions), merged with each other where they touch."""
    label = bad_label(label)
    merged: list[list[float]] = []
    for onset, duration in sorted((float(a), float(b)) for a, b in segments):
        if merged and onset <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], onset + duration)
        else:
            merged.append([onset, onset + duration])
    added = []
    for lo, hi in merged:
        try:
            a, d = _clamp_span(store, lo, hi - lo)
        except ValueError:
            continue
        added.append(Span(onset=a, duration=d, label=label))
    have = bad_spans(store)
    added = [sp for sp in added if sp not in have]
    if not added:
        return set()
    _set_spans(store, have + added)
    return {"traces.annotations"}


def _span_index(store: "SceneStore", index: Optional[int]) -> int:
    spans = bad_spans(store)
    i = store.scene.traces.selected_span if index is None else int(index)
    if i is None or not 0 <= i < len(spans):
        raise ValueError("no bad segment is selected")
    return i


@command("annotate.set", "Change a bad segment", category="Annotation", undoable=True)
def annotate_set(store: "SceneStore", index: Optional[int] = None,
                 onset: Optional[float] = None, duration: Optional[float] = None,
                 label: Optional[str] = None) -> set[str]:
    """Move, resize or relabel a segment (the selected one by default)."""
    i = _span_index(store, index)
    spans = bad_spans(store)
    old = spans[i]
    new_onset = old.onset if onset is None else float(onset)
    new_duration = old.duration if duration is None else float(duration)
    lo, length = _clamp_span(store, new_onset, new_duration)
    new = Span(onset=lo, duration=length, label=bad_label(label) if label else old.label)
    if new == old:
        return set()
    spans[i] = new
    _set_spans(store, spans)
    store.scene.traces.selected_span = store.scene.traces.bad_spans.index(new)
    return {"traces.annotations"}


@command("annotate.remove", "Delete a bad segment", category="Annotation", undoable=True)
def annotate_remove(store: "SceneStore", index: Optional[int] = None) -> set[str]:
    """Delete a segment (the selected one by default)."""
    i = _span_index(store, index)
    spans = bad_spans(store)
    del spans[i]
    _set_spans(store, spans)
    store.scene.traces.selected_span = None
    return {"traces.annotations"}


@command("annotate.select", "Select a bad segment", category="Annotation")
def annotate_select(store: "SceneStore", index: Optional[int] = None) -> set[str]:
    spans = bad_spans(store)
    new = None if index is None or not 0 <= int(index) < len(spans) else int(index)
    if new == store.scene.traces.selected_span:
        return set()
    store.scene.traces.selected_span = new
    return {"traces.annotations"}


@command("channels.set_bad", "Mark channels bad or good", category="Channels",
         undoable=True)
def channels_set_bad(store: "SceneStore", names: list[str], bad: bool = True) -> set[str]:
    """Several channels at once (the quality check's suggestions)."""
    src = source(store)
    known = set(src.ch_names) if src is not None else set(names)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"no channel called {unknown[0]!r}")
    bads = bad_channels(store)
    new = (bads | set(names)) if bad else (bads - set(names))
    if new == bads:
        return set()
    store.scene.traces.bads = sorted(new)
    return {"traces.look"}


@command("traces.qc_view", "QC plots", category="Channels")
def traces_qc_view(store: "SceneStore", metrics: Optional[list[str]] = None,
                   types: Optional[list[str]] = None, order: Optional[list[str]] = None,
                   hidden: Optional[list[str]] = None,
                   scroll: Optional[bool] = None,
                   beside: Optional[bool] = None) -> set[str]:
    """Which QC plots are shown under the traces, for which channel types, in
    what order, and how tall."""
    from ..compute.meeg_qc import METRIC_IDS

    tr = store.scene.traces
    changed = False
    if metrics is not None:
        unknown = [m for m in metrics if m not in METRIC_IDS]
        if unknown:
            raise ValueError(f"no QC plot called {unknown[0]!r}")
    for name, value in (("qc_metrics", metrics), ("qc_types", types), ("qc_order", order),
                        ("qc_hidden", hidden)):
        if value is not None:
            value = list(dict.fromkeys(str(v) for v in value))
            if getattr(tr, name) != value:
                setattr(tr, name, value)
                changed = True
    if scroll is not None and tr.qc_scroll != bool(scroll):
        tr.qc_scroll = bool(scroll)
        changed = True
    if beside is not None and tr.qc_beside != bool(beside):
        tr.qc_beside = bool(beside)
        changed = True
    return {"traces.qc_view"} if changed else set()


@command("traces.quality", "QC", category="Channels")
def traces_quality(store: "SceneStore", value: Optional[bool] = None) -> set[str]:
    """Show the quality check (computed on a worker the first time)."""
    tr = store.scene.traces
    new = (not tr.quality) if value is None else bool(value)
    if new == tr.quality:
        return set()
    tr.quality = new
    return {"traces.quality"}


@command("traces.normalize", "Normalise each channel", category="Channels")
def traces_normalize(store: "SceneStore", value: Optional[bool] = None) -> set[str]:
    tr = store.scene.traces
    new = (not tr.normalize) if value is None else bool(value)
    if new == tr.normalize:
        return set()
    tr.normalize = new
    return {"traces.look"}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


@command("traces.filter", "Filter the signal", category="Filters", undoable=True)
def traces_filter(store: "SceneStore", hp: Optional[float] = None,
                  lp: Optional[float] = None, notch: Optional[float] = None) -> set[str]:
    """Zero-phase high-pass, low-pass and notch (Hz; 0 or None: off).
    A cut-off at or above the Nyquist frequency is REFUSED with the reason."""
    src = source(store)
    spec = FilterSpec(hp or None, lp or None, notch or None)
    if src is not None:
        spec = validate(spec, src.sfreq)
    tr = store.scene.traces
    if (tr.hp, tr.lp, tr.notch) == (spec.hp, spec.lp, spec.notch):
        return set()
    tr.hp, tr.lp, tr.notch = spec.hp, spec.lp, spec.notch
    return {"traces.filter"}


@command("traces.reset_filters", "Remove the filters", category="Filters", undoable=True)
def traces_reset_filters(store: "SceneStore") -> set[str]:
    return traces_filter(store)


@command("traces.reset", "Reset the trace view", category="Channels", undoable=True)
def traces_reset(store: "SceneStore") -> set[str]:
    src = source(store)
    if src is None:
        return set()
    state = opening_state(src)
    state["events"] = store.scene.traces.events
    state["event_source"] = store.scene.traces.event_source
    # Bad channels are a judgement about the data, not a way of looking at
    # it: a reset of the view keeps them.
    state["bads"] = store.scene.traces.bads
    state["bad_spans"] = store.scene.traces.bad_spans
    state["annotate"] = store.scene.traces.annotate
    changed = False
    for key, value in state.items():
        if getattr(store.scene.traces, key) != value:
            setattr(store.scene.traces, key, value)
            changed = True
    if store.scene.traces.focus is not None:
        store.scene.traces.focus = None
        changed = True
    paths = {"traces.time", "traces.channels", "traces.look", "traces.filter"} if changed else set()
    if store.scene.cursor.time is not None:
        store.scene.cursor.time = None
        paths.add("cursor.time")
    return paths


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _jump_to_first_event(store) -> set[str]:
    """Events often start well into a recording (an MEG run's first trigger
    can be 100 s in), so turning them on at t=0 looked like nothing
    happened. When none is on screen, go to the first."""
    src = source(store)
    tr = store.scene.traces
    if src is None:
        return set()
    events = src.events(tr.event_source)
    if not events:
        return set()
    lo, hi = tr.t0, tr.t0 + tr.width
    rec = [e.onset - src.start_time for e in events]
    if any(lo <= t <= hi for t in rec):
        return set()
    return time_set(store, min(rec) - tr.width * 0.1)


@command("events.show", "Show events", category="Events")
def events_show(store: "SceneStore", value: Optional[bool] = None) -> set[str]:
    tr = store.scene.traces
    new = (not tr.events) if value is None else bool(value)
    if new == tr.events:
        return set()
    tr.events = new
    changed = {"traces.events"}
    if new:
        changed |= _jump_to_first_event(store)
    return changed


@command("events.source", "Choose where events come from", category="Events")
def events_source(store: "SceneStore",
                  which: Literal["auto", "events.tsv", "stim", "annotations"] = "auto"
                  ) -> set[str]:
    tr = store.scene.traces
    if tr.event_source == which:
        return set()
    tr.event_source = which
    changed = {"traces.events"}
    if tr.events:
        changed |= _jump_to_first_event(store)
    return changed


__all__ = ["DEFAULT_COUNT", "DEFAULT_WIDTH_S", "PAGE_FRACTION", "bad_channels",
           "opening_state", "source"]

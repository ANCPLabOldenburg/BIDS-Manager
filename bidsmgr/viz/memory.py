"""What the viewers remember between files, windows and sessions.

The user's arrangement of a viewer is effort: a layout tuned, a 3-D look
dialled in, a filter and an amplitude that suit a lab's recordings. All of it
outlives the file on screen. What does NOT carry over is what belongs to one
file: where the crosshair is, its window in the image's own units, the volume
or the second shown, the bad channels and segments, a spectrum's phase.

Three memories, kept in :class:`~bidsmgr.viz.settings.VizSettings`:

* per KIND of image (a layout preset id: ``mri.func``, ``mri.anat``...) the
  arrangement: :data:`bidsmgr.viz.layouts.LAYOUT_KEYS`;
* for every image the LOOK: :data:`VOLUME_LOOK_KEYS` (display conventions,
  the 3-D effect and its parameters, the clip planes);
* per kind of signal (``meeg``, ``physio``) the trace options and filters
  (:data:`TRACE_KEYS`), and the spectrum's processing and display options
  (:data:`SPECTRUM_KEYS`).

Restoring is validated: a filter the new recording cannot take, a time span
longer than it, a channel type it does not have are dropped, never applied.

Qt-free.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .scene import ClipPlane, Display, RenderState, Scene, SpectrumState, TracesState

#: The look every image shares: display conventions, the 3-D effect and its
#: parameters, the clip planes.
VOLUME_LOOK_KEYS = ("display", "render", "clips")
#: The trace options a signal viewer keeps for the next recording.
TRACE_KEYS = ("ch_type", "count", "scale", "width", "normalize", "page_scale", "remove_dc",
              "clip", "butterfly", "events", "event_source", "hp", "lp", "notch", "quality",
              "qc_metrics", "qc_types", "qc_order", "qc_hidden", "qc_scroll", "qc_beside")
#: The spectrum options kept for the next file (never its phase: that is
#: the file's).
SPECTRUM_KEYS = ("domain", "part", "lb_hz", "exclude_water", "metabolites",
                 "standard_window", "reference")


# -- volumes -------------------------------------------------------------------

def volume_look(scene: Scene) -> dict:
    """The look of ``scene`` as plain data."""
    return scene.model_dump(mode="json", include=set(VOLUME_LOOK_KEYS))


def apply_volume_look(scene: Scene, look: Optional[Mapping[str, Any]]) -> Scene:
    """``scene`` with a remembered look (validated; a bad entry is skipped)."""
    if not look:
        return scene
    if isinstance(look.get("display"), Mapping):
        try:
            scene.display = Display.model_validate({**scene.display.model_dump(),
                                                    **look["display"]})
        except ValueError:
            pass
    if isinstance(look.get("render"), Mapping):
        try:
            scene.render = RenderState.model_validate(look["render"])
        except ValueError:
            pass
    if isinstance(look.get("clips"), list):
        try:
            scene.clips = [ClipPlane.model_validate(c) for c in look["clips"]]
        except ValueError:
            pass
    return scene


# -- signals -------------------------------------------------------------------

def trace_prefs(traces: TracesState) -> dict:
    return traces.model_dump(mode="json", include=set(TRACE_KEYS))


def restore_traces(opening: Mapping[str, Any], remembered: Optional[Mapping[str, Any]],
                   src, *, qc_on_open: bool = False) -> dict:
    """The traces state a recording opens in: ``opening`` (its own
    defaults) with the remembered options that suit it. QC is carried over
    only when the user asked for QC on opening (``QcSettings.on_open``):
    otherwise every recording opens with it off."""
    state = dict(opening)
    if not remembered:
        return state
    for key in TRACE_KEYS:
        if key in remembered:
            state[key] = remembered[key]
    if not qc_on_open:
        state["quality"] = opening.get("quality", False)
    duration = float(getattr(src, "duration", 0.0) or 0.0)
    if duration > 0:
        state["width"] = min(float(state.get("width") or opening["width"]), max(0.1, duration))
    n = len(getattr(src, "ch_names", []) or [])
    if n:
        state["count"] = max(1, min(int(state.get("count") or 1), n))
    available = set(getattr(src, "available_types", []) or [])
    if state.get("ch_type") not in ({"all", "mag+grad"} | available):
        state["ch_type"] = "all"
    if state.get("event_source") not in ("auto", *getattr(src, "event_sources", lambda: [])()):
        state["event_source"] = "auto"
    # A filter this recording cannot take (a cut-off at or above its Nyquist
    # frequency, a band too low for its length) is dropped, not applied.
    try:
        from .compute.filters import FilterSpec, validate

        spec = FilterSpec(state.get("hp"), state.get("lp"), state.get("notch"))
        if spec.active:
            validate(spec, float(src.sfreq))
    except Exception:  # noqa: BLE001 - any refusal means: no filter
        state["hp"] = state["lp"] = state["notch"] = None
    try:
        return TracesState.model_validate(state).model_dump()
    except ValueError:
        return dict(opening)


# -- spectra -------------------------------------------------------------------

def spectrum_prefs(spectrum: SpectrumState) -> dict:
    return spectrum.model_dump(mode="json", include=set(SPECTRUM_KEYS))


def restore_spectrum(state: SpectrumState, remembered: Optional[Mapping[str, Any]]
                     ) -> SpectrumState:
    if not remembered:
        return state
    merged = {**state.model_dump(), **{k: remembered[k] for k in SPECTRUM_KEYS if k in remembered}}
    try:
        return SpectrumState.model_validate(merged)
    except ValueError:
        return state


# -- forgetting ---------------------------------------------------------------

#: Every field of :class:`~bidsmgr.viz.settings.VizSettings` that is MEMORY
#: (how things were left) rather than a preference the user set in Settings.
MEMORY_FIELDS = ("layout_state", "layout_sizes", "volume_look", "traces_state",
                 "spectrum_state", "inspector_sections")
#: What "Restore every viewer default" keeps: the user's own work.
KEPT_ON_RESET = ("keymap", "mousemap", "view_presets")


def carry_memory(new, current) -> None:
    """Copy the memory of ``current`` into ``new`` (a settings copy edited
    in a dialog must not roll back views changed while it was open)."""
    for name in MEMORY_FIELDS:
        value = getattr(current, name)
        setattr(new, name, value.copy() if hasattr(value, "copy") else value)


def forget(settings) -> None:
    """Clear every remembered view: each kind opens in its preset again."""
    for name in MEMORY_FIELDS:
        setattr(settings, name, type(getattr(settings, name))())


def remembered_count(settings) -> dict:
    """What is remembered, for the Settings page: counts per memory."""
    return {
        "kinds": len(settings.layout_state), "sizes": len(settings.layout_sizes),
        "look": bool(settings.volume_look), "signals": len(settings.traces_state),
        "spectrum": bool(settings.spectrum_state),
    }


def defaults(settings):
    """A settings object with every default restored, keeping the user's own
    work (shortcuts, mouse map, saved presets)."""
    fresh = type(settings)()
    for name in KEPT_ON_RESET:
        value = getattr(settings, name)
        setattr(fresh, name, value.copy() if hasattr(value, "copy") else value)
    return fresh


__all__ = ["KEPT_ON_RESET", "MEMORY_FIELDS", "SPECTRUM_KEYS", "TRACE_KEYS",
           "VOLUME_LOOK_KEYS", "apply_volume_look", "carry_memory", "defaults", "forget",
           "remembered_count",
           "restore_spectrum", "restore_traces", "spectrum_prefs", "trace_prefs",
           "volume_look"]

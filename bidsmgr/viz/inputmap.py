"""What the mouse does, per kind of canvas, as data.

A gesture is ``<button>`` (``left``, ``right``, ``middle``) or ``wheel``,
optionally prefixed by modifiers (``shift+``, ``ctrl+``, ``alt+``) and, for
the wheel, a held key (``h+wheel``). Each gesture maps to a TOOL id that the
canvas knows how to run. The defaults reproduce the viewer's gestures as they
were; a user can rebind any of them (``VizSettings.mousemap``), the way
NiiVue's ``mouseEventConfig`` allows.

Lookup is most-specific first: ``ctrl+shift+left`` before ``shift+left``
before ``left``, so adding a modifier binding never breaks the plain one.

Pure data.
"""

from __future__ import annotations

from typing import Mapping, Optional

#: Tools a 2-D slice canvas can run, by what drives them: a drag or a wheel.
SLICE_DRAG_TOOLS = {
    "crosshair": "Move the crosshair",
    "pan": "Pan (when zoomed)",
    "zoom": "Zoom",
    "window": "Window level and width",
    "window_box": "Fit the window to a box",
    "none": "Nothing",
}
SLICE_WHEEL_TOOLS = {
    "slice": "Step through slices",
    "frame": "Step through volumes (4-D)",
    "zoom": "Zoom",
    "none": "Nothing",
}
SLICE_TOOLS = {**SLICE_DRAG_TOOLS, **SLICE_WHEEL_TOOLS}

#: Tools the 3-D render canvas can run.
RENDER_DRAG_TOOLS = {
    "orbit": "Rotate (a click without a drag moves the crosshair)",
    "pan": "Pan",
    "zoom": "Zoom",
    "clip_tilt": "Tilt the clip plane (when on)",
    "pick": "Move the crosshair to the surface under the pointer",
    "none": "Nothing",
}
RENDER_WHEEL_TOOLS = {
    "zoom": "Zoom",
    "clip_push": "Move the clip plane (when on)",
    "none": "Nothing",
}
RENDER_TOOLS = {**RENDER_DRAG_TOOLS, **RENDER_WHEEL_TOOLS}

#: Tools of a traces canvas (MEG, EEG, physio).
TRACES_DRAG_TOOLS = {
    "scrub": "Drag the recording along",
    "none": "Nothing",
}
TRACES_WHEEL_TOOLS = {
    "channels": "Scroll through channels",
    "time": "Scroll through time",
    "zoom_time": "Zoom time",
    "scale": "Change the amplitude",
    "none": "Nothing",
}
TRACES_TOOLS = {**TRACES_DRAG_TOOLS, **TRACES_WHEEL_TOOLS}

#: Every gesture the Settings page offers to rebind, per canvas.
GESTURES = {
    "slice": ("left", "right", "middle", "shift+left", "ctrl+left", "alt+left",
              "wheel", "shift+wheel", "ctrl+wheel", "alt+wheel", "hwheel", "h+wheel"),
    "render": ("left", "right", "middle", "shift+left", "ctrl+left", "alt+left",
               "wheel", "shift+wheel", "ctrl+wheel", "hwheel"),
    "traces": ("left", "right", "middle", "wheel", "shift+wheel", "ctrl+wheel",
               "alt+wheel", "hwheel"),
}

DEFAULT_MOUSEMAP: dict[str, str] = {
    # 2-D slices
    "slice:left": "crosshair",
    "slice:right": "window",
    "slice:middle": "pan",
    "slice:shift+left": "window_box",
    "slice:ctrl+left": "pan",
    "slice:wheel": "slice",
    "slice:hwheel": "frame",
    "slice:h+wheel": "frame",
    "slice:shift+wheel": "frame",
    "slice:ctrl+wheel": "zoom",
    # 3-D render
    "render:left": "orbit",
    "render:right": "pan",
    "render:middle": "pan",
    "render:shift+left": "clip_tilt",
    "render:wheel": "zoom",
    "render:shift+wheel": "clip_push",
    "render:hwheel": "clip_push",
    # Traces: the old view's gestures (drag scrubs, wheel scrolls channels)
    # plus time on a horizontal scroll and zoom on Ctrl.
    "traces:left": "scrub",
    "traces:wheel": "channels",
    "traces:hwheel": "time",
    "traces:shift+wheel": "time",
    "traces:ctrl+wheel": "zoom_time",
    "traces:alt+wheel": "scale",
}


def tools_for(canvas: str, gesture_text: str) -> dict[str, str]:
    """The tools a gesture can be bound to: drag tools for a button, wheel
    tools for a scroll."""
    wheel = gesture_text.endswith("wheel")
    if canvas == "render":
        return RENDER_WHEEL_TOOLS if wheel else RENDER_DRAG_TOOLS
    if canvas == "traces":
        return TRACES_WHEEL_TOOLS if wheel else TRACES_DRAG_TOOLS
    return SLICE_WHEEL_TOOLS if wheel else SLICE_DRAG_TOOLS


def binding(canvas: str, gesture_text: str,
            overrides: Optional[Mapping[str, str]] = None) -> str:
    """The tool EXACTLY bound to one gesture (no fallback), "none" if unbound."""
    key = f"{canvas}:{gesture_text}"
    if overrides and key in overrides:
        return overrides[key]
    return DEFAULT_MOUSEMAP.get(key, "none")


_MOD_ORDER = ("ctrl", "alt", "shift", "h")


def gesture(button: str, mods: set[str]) -> str:
    """Canonical gesture text from a button and a set of modifier names."""
    parts = [m for m in _MOD_ORDER if m in mods]
    return "+".join(parts + [button])


def lookup(canvas: str, button: str, mods: set[str],
           overrides: Optional[Mapping[str, str]] = None) -> str:
    """The tool for a gesture, most specific binding first."""
    table = dict(DEFAULT_MOUSEMAP)
    if overrides:
        table.update(overrides)
    present = [m for m in _MOD_ORDER if m in mods]
    # All subsets of the held modifiers, largest first.
    subsets: list[list[str]] = [[]]
    for m in present:
        subsets += [s + [m] for s in subsets]
    subsets.sort(key=len, reverse=True)
    for subset in subsets:
        key = f"{canvas}:" + "+".join(
            [m for m in _MOD_ORDER if m in subset] + [button]
        )
        if key in table:
            return table[key]
    return "none"


__all__ = [
    "DEFAULT_MOUSEMAP", "GESTURES", "RENDER_DRAG_TOOLS", "RENDER_TOOLS",
    "RENDER_WHEEL_TOOLS", "SLICE_DRAG_TOOLS", "SLICE_TOOLS", "SLICE_WHEEL_TOOLS",
    "TRACES_DRAG_TOOLS", "TRACES_TOOLS", "TRACES_WHEEL_TOOLS",
    "binding", "gesture", "lookup", "tools_for",
]

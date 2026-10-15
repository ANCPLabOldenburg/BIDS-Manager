"""Presets: how the viewer looks, carried from one image to another.

A preset (Save > Save the look as a preset) is the view without the image:
the layout, the display, the time course with its QC plots, the 3-D look.
It is applied to whatever image is open, and an image takes only what it
has. A T1w has no time series, so a preset saved on a BOLD gives it the
layout and the look and leaves the time course, its QC and where that panel
sits alone. A preset saved on a T1w says nothing about a time course, so
applying it to a BOLD keeps the BOLD's own.

Before this, a BOLD preset applied to a T1w switched the time course on in
a viewer that could not show it (its action read disabled and checked), and
the next BOLD opened with it.

A scene (:mod:`scenes`) is applied through :func:`applicable` too: its image
is its own, but the file may have changed since.

Qt-free.
"""

from __future__ import annotations

from typing import Any, Mapping

#: What a preset leaves out: where you were (the cursor, each plane's zoom
#: and pan belong to one image), what is not a view at all, and the signal
#: and spectrum viewers' state (a preset is the image viewer's).
EXCLUDE = {"sources", "layers", "cursor", "views", "measurements", "schema_version",
           "traces", "spectrum"}

#: The parts of a view that exist only for an image with a time series.
SERIES_KEYS = ("graph", "graph_visible")
#: ... and, in the layout, where the time-course panel sits.
SERIES_LAYOUT_KEYS = ("graph",)


def snapshot(scene, *, has_series: bool) -> dict[str, Any]:
    """The view of ``scene`` as a preset: without what its image lacks."""
    state = scene.model_dump(mode="json", exclude=EXCLUDE)
    if not has_series:
        return _without_series(state)
    # The graph can name the layer it follows; a layer id belongs to one
    # image, and on another it names nothing (or the wrong layer).
    state["graph"] = {**state["graph"], "layer": ""}
    return state


def applicable(state: Mapping[str, Any], scene, *, has_series: bool) -> dict[str, Any]:
    """``state`` as the image in ``scene`` takes it: without the time
    course when it has no time series, and with any layout field the state
    does not set kept as the scene has it (validated alone, a partial
    layout would put every missing field back to its default)."""
    out = dict(state) if has_series else _without_series(state)
    layout = out.get("layout")
    if isinstance(layout, Mapping):
        out["layout"] = {**scene.layout.model_dump(mode="json"), **layout}
    return out


def _without_series(state: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in state.items() if k not in SERIES_KEYS}
    layout = out.get("layout")
    if isinstance(layout, Mapping):
        out["layout"] = {k: v for k, v in layout.items() if k not in SERIES_LAYOUT_KEYS}
    return out


__all__ = ["EXCLUDE", "SERIES_KEYS", "SERIES_LAYOUT_KEYS", "applicable", "snapshot"]

"""Every editable property of a layer, described once, as data.

A property is a field of :class:`~bidsmgr.viz.scene.VolumeDisplay` (or of
the layer itself) with what a control needs to edit it: a kind, a range and
step, a unit, the choices of a choice, a help text, and WHEN it applies
(``when`` in the action-expression language: ``scalar``, ``overlay``,
``two_tailed``). The Layers panel is generated from this table, so adding a
property to the display model and a line here gives it a control, a tooltip
and a command, with no widget code.

Every property is changed through ``layer.set``, so it is undoable, linkable
between two viewers, and scriptable.

Pure data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Prop:
    key: str
    title: str
    kind: Literal["colormap", "window", "float", "bool", "choice"]
    #: Where the value lives: the layer's ``display`` or the layer itself.
    target: Literal["display", "layer"] = "display"
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.01
    unit: str = ""
    choices: tuple[tuple[Any, str], ...] = field(default_factory=tuple)
    help: str = ""
    #: When the control is offered (expression over the layer context).
    when: str = ""
    #: Which section shows it: ``display`` (how values become colours) or
    #: ``overlay`` (what only a layer drawn over another needs).
    group: str = "display"


LAYER_PROPS: tuple[Prop, ...] = (
    Prop("visible", "Visible", "bool", target="layer",
         help="Hide the layer without removing it."),
    Prop("in_3d", "Show in 3-D", "bool", group="overlay", target="layer", when="overlay",
         help="Draw this overlay in the 3-D rendering as well as on the slices."),
    Prop("colormap", "Colour map", "colormap", when="scalar && !labels",
         help="How values are turned into colours."),
    Prop("window", "Window", "window", when="scalar && !labels && !shape",
         help="The display range, in the data's own units: values at or below the low end "
              "take the first colour of the map, at or above the high end the last "
              "(window and level)."),
    Prop("gamma", "Gamma", "float", lo=0.1, hi=10.0, step=0.05,
         when="scalar && !labels && !shape",
         help="The brightness curve inside the window: above 1 lifts the mid-tones, below "
              "1 darkens them."),
    Prop("invert", "Invert colour map", "bool", when="scalar && !labels && !shape",
         help="Run the colour map backwards."),
    Prop("opacity", "Opacity", "float", lo=0.0, hi=1.0, step=0.05, unit="",
         help="0 is invisible; 1 hides the layers below completely."),
    Prop("interpolation", "Interpolation", "choice", when="!shape",
         choices=(("linear", "Linear (smooth)"), ("nearest", "Nearest neighbour (voxels)")),
         help="How the image is drawn between voxel centres: linear blends neighbouring "
              "voxels; nearest neighbour shows each voxel as a square, as the data is."),
    Prop("threshold_mode", "Values below the window", "choice",
         when="scalar && !labels && !shape",
         choices=(("range", "Darkest colour"), ("hide_below", "Hidden (threshold)"),
                  ("translucent_below", "Translucent")),
         help="How values below the window's low end are drawn: in the map's first colour "
              "(an anatomical image), hidden (a thresholded statistical map), or "
              "translucent, so a cluster that nearly reached the threshold still shows."),
    Prop("colormap_negative", "Negative colour map", "colormap",
         when="scalar && !labels && !shape",
         help="Draw values below zero with their own colour map (a two-tailed statistic). "
              "Empty: no separate negative tail."),
    Prop("window_negative", "Negative window", "window", when="two_tailed && !labels",
         help="Threshold and saturation of the negative tail, as magnitudes."),
    Prop("outline_px", "Outline width", "float", group="overlay", lo=0.0, hi=5.0, step=1.0,
         when="overlay",
         help="Draw only the edge of each region, this many pixels wide, so the image under "
              "it stays visible; 0 fills it. For the MRS voxel, the width of its box."),
)

PROP_BY_KEY = {p.key: p for p in LAYER_PROPS}


def layer_context(layer, src) -> dict[str, Any]:
    """The facts a property's ``when`` is evaluated against."""
    is_rgb = bool(getattr(src, "is_rgb", False))
    return {
        "scalar": not is_rgb,
        "rgb": is_rgb,
        "overlay": bool(getattr(layer, "id", "base") != "base"),
        "two_tailed": bool(getattr(getattr(layer, "display", None), "colormap_negative", "")),
        # An atlas is coloured by its table: no colour map, window or gamma.
        "labels": getattr(getattr(layer, "display", None), "label_table", None) is not None,
        # A shape (the MRS voxel) is drawn as geometry: a colour, an opacity,
        # an outline; no window, gamma or interpolation.
        "shape": getattr(src, "box", None) is not None,
    }


__all__ = ["LAYER_PROPS", "PROP_BY_KEY", "Prop", "layer_context"]

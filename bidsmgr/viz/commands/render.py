"""Commands for the 3-D render: camera, effect, parameters, clip planes."""

from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING

import numpy as np

from . import command
from .. import render3d
from ..scene import Camera, ClipPlane

if TYPE_CHECKING:  # pragma: no cover
    from ..store import SceneStore

MAX_CLIPS = 6

#: Commands that act on ONE clip plane (they take ``index``). A host aims
#: them at the plane its controls are editing; the rest act on every plane.
PLANE_COMMANDS = frozenset({"clip.set", "clip.toggle", "clip.axis", "clip.invert",
                            "clip.nudge", "clip.tilt"})


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


@command("render.camera", "Set the 3-D camera", category="3-D")
def render_camera(store: "SceneStore", az: Optional[float] = None,
                  el: Optional[float] = None, dist: Optional[float] = None,
                  target: Optional[tuple[float, float, float]] = None) -> set[str]:
    cam = store.scene.render.camera
    new = Camera(
        az=cam.az if az is None else float(az),
        el=cam.el if el is None else float(np.clip(el, -1.55, 1.55)),
        dist=cam.dist if dist is None else float(np.clip(dist, 0.5, 12.0)),
        target=cam.target if target is None else tuple(map(float, target)),
    )
    if new == cam:
        return set()
    store.scene.render.camera = new
    return {"render.camera"}


@command("render.orbit", "Rotate the 3-D view", category="3-D")
def render_orbit(store: "SceneStore", d_az: float, d_el: float) -> set[str]:
    cam = store.scene.render.camera
    return render_camera(store, az=cam.az + d_az, el=cam.el + d_el)


@command("render.pan", "Pan the 3-D view", category="3-D")
def render_pan(store: "SceneStore", dx: float, dy: float) -> set[str]:
    """Pan by screen pixels (scaled by the distance, as before)."""
    cam = store.scene.render.camera
    r, u = render3d.camera_basis(cam.az, cam.el, cam.dist, cam.target)
    scale = 0.0022 * cam.dist
    target = np.asarray(cam.target, dtype=float) + (-dx * r + dy * u) * scale
    return render_camera(store, target=tuple(float(v) for v in target))


@command("render.zoom", "Zoom the 3-D view", category="3-D")
def render_zoom(store: "SceneStore", steps: float) -> set[str]:
    cam = store.scene.render.camera
    return render_camera(store, dist=cam.dist * (0.9 ** float(steps)))


@command("render.reset_view", "Reset the 3-D camera", category="3-D")
def render_reset_view(store: "SceneStore") -> set[str]:
    if store.scene.render.camera == Camera():
        return set()
    store.scene.render.camera = Camera()
    return {"render.camera"}


@command("render.preset_view", "Look from a side", category="3-D")
def render_preset_view(store: "SceneStore",
                       side: Literal["left", "right", "front", "back", "top", "bottom"]) -> set[str]:
    az_el = {
        "right": (np.pi / 2, 0.0), "left": (-np.pi / 2, 0.0),
        "front": (0.0, 0.0), "back": (np.pi, 0.0),
        "top": (0.0, 1.55), "bottom": (0.0, -1.55),
    }[side]
    return render_camera(store, az=az_el[0], el=az_el[1], target=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Look
# ---------------------------------------------------------------------------


@command("render.effect", "Choose the 3-D effect", category="3-D", undoable=True)
def render_effect(store: "SceneStore", effect: str) -> set[str]:
    if effect not in render3d.EFFECTS:
        raise ValueError(f"unknown effect {effect!r}")
    if store.scene.render.effect == effect:
        return set()
    store.scene.render.effect = effect
    return {"render.effect"}


@command("render.param", "Change a 3-D parameter", category="3-D", undoable=True)
def render_param(store: "SceneStore", key: str, value: float,
                 effect: Optional[str] = None) -> set[str]:
    """Set one parameter of an effect (slider units). The two thresholds
    keep their gap, pushing the other one as the old panel did."""
    if key not in render3d.PARAM_BY_KEY:
        raise ValueError(f"unknown 3-D parameter {key!r}")
    if key in render3d.SHARED_PARAMS:
        rs = store.scene.render
        value = render3d.clamp_param(key, float(value))
        if render3d.values_for(rs).get(key) == value:
            return set()
        rs.shared[key] = value
        return {"render.params"}
    eff = effect or store.scene.render.effect
    overrides = store.scene.render.params.setdefault(eff, {})
    current = render3d.effective(eff, overrides)
    value = render3d.clamp_param(key, float(value))
    if current.get(key) == value:
        return set()
    overrides[key] = value
    gap = render3d.THRESH_GAP
    if key == "lo" and current["hi"] < value + gap:
        overrides["hi"] = min(1000.0, value + gap)
    elif key == "hi" and current["lo"] > value - gap:
        overrides["lo"] = max(0.0, value - gap)
    return {"render.params"}


@command("render.use_colormap", "Colour the 3-D view with the 2-D colour map",
         category="3-D", undoable=True)
def render_use_colormap(store: "SceneStore", value: Optional[bool] = None) -> set[str]:
    rs = store.scene.render
    new = (not rs.use_colormap) if value is None else bool(value)
    if new == rs.use_colormap:
        return set()
    rs.use_colormap = new
    return {"render.params"}


@command("render.reset_params", "Reset this effect's parameters",
         category="3-D", undoable=True)
def render_reset_params(store: "SceneStore", keys: Optional[list[str]] = None) -> set[str]:
    """Reset the current effect's own parameters to its baseline. With
    ``keys`` (one section of the controls), those keys only, the ones every
    effect shares (see-through, overlays) included; without, the effect's own
    and never the shared ones. The clip plane, the camera and every other
    effect are left alone."""
    rs = store.scene.render
    eff = rs.effect
    wanted = set(render3d.EFFECT_PARAMS[eff]) if keys is None else set(keys)
    overrides = rs.params.get(eff) or {}
    keep = {k: v for k, v in overrides.items() if k not in wanted}
    shared = (dict(rs.shared) if keys is None
              else {k: v for k, v in rs.shared.items() if k not in wanted})
    if keep == overrides and shared == rs.shared:
        return set()
    rs.params[eff] = keep
    rs.shared = shared
    return {"render.params"}


@command("render.reset_all", "Reset every effect's parameters",
         category="3-D", undoable=True)
def render_reset_all(store: "SceneStore") -> set[str]:
    rs = store.scene.render
    if not rs.params and not rs.shared:
        return set()
    rs.params = {}
    rs.shared = {}
    return {"render.params"}


# ---------------------------------------------------------------------------
# Clip planes
# ---------------------------------------------------------------------------


def _clip(store: "SceneStore", index: int) -> ClipPlane:
    clips = store.scene.clips
    while len(clips) <= index:
        clips.append(ClipPlane())
    return clips[index]


@command("clip.set", "Set a clip plane", category="3-D", undoable=True)
def clip_set(store: "SceneStore", index: int = 0, active: Optional[bool] = None,
             az: Optional[float] = None, el: Optional[float] = None,
             pos: Optional[float] = None, thick: Optional[float] = None,
             flip: Optional[bool] = None) -> set[str]:
    if not 0 <= index < MAX_CLIPS:
        raise ValueError(f"clip plane index must be 0..{MAX_CLIPS - 1}")
    clip = _clip(store, index)
    before = clip.model_copy()
    if active is not None:
        clip.active = bool(active)
    if az is not None:
        clip.az = float(az) % 360.0
    if el is not None:
        clip.el = float(np.clip(el, -90.0, 90.0))
    if pos is not None:
        clip.pos = float(np.clip(pos, 0.0, 1.0))
    if thick is not None:
        clip.thick = float(np.clip(thick, 0.0, 1.0))
    if flip is not None:
        clip.flip = bool(flip)
    return set() if clip == before else {"clips"}


@command("clip.toggle", "Turn the clip plane on or off", category="3-D")
def clip_toggle(store: "SceneStore", index: int = 0) -> set[str]:
    clip = _clip(store, index)
    return clip_set(store, index=index, active=not clip.active)


@command("clip.axis", "Cut along an anatomical plane", category="3-D")
def clip_axis(store: "SceneStore", plane: Literal["axial", "sagittal", "coronal"],
              index: int = 0) -> set[str]:
    az, el = {"axial": (0.0, 90.0), "sagittal": (90.0, 0.0), "coronal": (0.0, 0.0)}[plane]
    return clip_set(store, index=index, active=True, az=az, el=el, flip=False)


@command("clip.invert", "Cut from the other side", category="3-D")
def clip_invert(store: "SceneStore", index: int = 0) -> set[str]:
    clip = _clip(store, index)
    return clip_set(store, index=index, active=True, flip=not clip.flip)


@command("clip.nudge", "Move the clip plane", category="3-D")
def clip_nudge(store: "SceneStore", delta: float, index: int = 0) -> set[str]:
    clip = _clip(store, index)
    return clip_set(store, index=index, pos=clip.pos + float(delta))


@command("clip.tilt", "Tilt the clip plane", category="3-D")
def clip_tilt(store: "SceneStore", d_az: float, d_el: float, index: int = 0) -> set[str]:
    clip = _clip(store, index)
    return clip_set(store, index=index, az=clip.az + d_az, el=clip.el + d_el)


#: Plane angles (az, el) that face each anatomical direction: the cut
#: removes what lies on that side.
_FACING = {"front": (0.0, 0.0), "back": (180.0, 0.0), "right": (90.0, 0.0),
           "left": (270.0, 0.0), "top": (0.0, 90.0), "bottom": (0.0, -90.0)}
#: Fraction of the box a crop keeps on each side of the centre.
_CROP = 0.69

#: name -> (planes as (facing, pos), cut away, through the crosshair)
CLIP_PRESETS: dict[str, tuple[tuple[tuple[str, float], ...], bool, bool]] = {
    "none": ((), False, False),
    # Half the head off, at the crosshair: one slice face on the solid.
    "one": ((("front", 0.5),), False, True),
    # The front-top quarter out: an axial and a coronal face.
    "wedge": ((("front", 0.5), ("top", 0.5)), True, True),
    # The octant facing the camera out: the three slices the 2-D views show,
    # meeting at the crosshair.
    "corner": ((("front", 0.5), ("top", 0.5), ("right", 0.5)), True, True),
    # Crops: what lies outside the planes goes.
    "four": (tuple((f, _CROP) for f in ("front", "back", "right", "left")), False, False),
    "box": (tuple((f, _CROP) for f in _FACING), False, False),
}


@command("clip.preset", "Use a set of clip planes", category="3-D", undoable=True)
def clip_preset(store: "SceneStore",
                preset: Literal["none", "one", "wedge", "corner", "four", "box"]) -> set[str]:
    """Replace the clip planes with a named arrangement (up to six), with
    the way they combine and whether they follow the crosshair."""
    planes, cut_away, at_cursor = CLIP_PRESETS[preset]
    new = [ClipPlane(active=True, az=_FACING[f][0], el=_FACING[f][1], pos=pos)
           for f, pos in planes] or [ClipPlane()]
    rs = store.scene.render
    if new == store.scene.clips and (rs.cut_away, rs.cut_at_cursor) == (cut_away, at_cursor):
        return set()
    store.scene.clips = new
    rs.cut_away = cut_away
    rs.cut_at_cursor = at_cursor
    return {"clips"}


@command("clip.mode", "How the clip planes cut", category="3-D", undoable=True)
def clip_mode(store: "SceneStore", cut_away: Optional[bool] = None,
              at_cursor: Optional[bool] = None) -> set[str]:
    """``cut_away``: remove only where every plane cuts (a wedge, a corner)
    instead of wherever any does. ``at_cursor``: every plane through the
    crosshair."""
    rs = store.scene.render
    before = (rs.cut_away, rs.cut_at_cursor)
    if cut_away is not None:
        rs.cut_away = bool(cut_away)
    if at_cursor is not None:
        rs.cut_at_cursor = bool(at_cursor)
    return set() if (rs.cut_away, rs.cut_at_cursor) == before else {"clips"}


@command("clip.toggle_at_cursor", "Cut through the crosshair", category="3-D")
def clip_toggle_at_cursor(store: "SceneStore") -> set[str]:
    return clip_mode(store, at_cursor=not store.scene.render.cut_at_cursor)


__all__: list[str] = []

"""Commands for volumes: the cursor, frames, views, display, windows."""

from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING

import numpy as np

from . import command
from .. import views
from ..compute import geometry
from ..scene import PLANE_AXIS, PLANES, SourceRef, VolumeDisplay, VolumeLayer

if TYPE_CHECKING:  # pragma: no cover
    from ..store import SceneStore

Plane = Literal["sagittal", "coronal", "axial"]
Mode = Literal["single", "multi", "3d", "combo", "hero", "mosaic"]


def _layer(store: "SceneStore", layer: Optional[str]) -> Optional[VolumeLayer]:
    if layer:
        found = store.scene.layer(layer)
        return found if found is not None and found.kind == "volume" else None
    return store.scene.base_layer()


def _series(store: "SceneStore", layer: Optional[str]) -> Optional[VolumeLayer]:
    """The layer a time command acts on: the one named, else the series the
    graph follows (a 4-D overlay over a 3-D image), else the base."""
    if layer:
        return _layer(store, layer)
    found, _src = views.series_layer(store)
    return found if found is not None else store.scene.base_layer()


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


@command("cursor.set_world", "Move the crosshair to a position",
         category="Navigate")
def cursor_set_world(store: "SceneStore", x: float, y: float, z: float,
                     snap: bool = True) -> set[str]:
    """Move the crosshair to a world position (mm).

    ``snap`` moves it to the centre of the base volume's nearest voxel, which
    is what a click on a slice wants: the readout then names one voxel, and
    the other two planes cut exactly through it.
    """
    world = np.array([x, y, z], dtype=float)
    _layer_, src = views.base(store)
    if src is not None and snap and store.scene.display.space == "voxel":
        world = geometry.snap_to_voxel(src.affine, src.spatial, world)
    new = (float(world[0]), float(world[1]), float(world[2]))
    if store.scene.cursor.world == new:
        return set()
    store.scene.cursor.world = new
    return {"cursor"}


@command("cursor.set_voxel", "Move the crosshair to a voxel", category="Navigate")
def cursor_set_voxel(store: "SceneStore", i: int, j: int, k: int) -> set[str]:
    """Move the crosshair to a voxel of the base volume (file indices)."""
    _layer_, src = views.base(store)
    if src is None:
        return set()
    w = src.voxel_to_world(src.clamp_voxel((i, j, k)))
    return cursor_set_world(store, float(w[0]), float(w[1]), float(w[2]), snap=False)


@command("cursor.step", "Step through slices", category="Navigate")
def cursor_step(store: "SceneStore", plane: Plane, n: int = 1) -> set[str]:
    """Move the crosshair ``n`` slices along the axis ``plane`` cuts.

    Positive is toward the anatomical positive end (right, front, top) in
    both spaces, so a scroll means the same thing whatever order the file
    stores its slices in.
    """
    _layer_, src = views.base(store)
    world = views.cursor_world(store)
    if src is None or world is None or not n:
        return set()
    ras_axis = PLANE_AXIS[plane]
    if store.scene.display.space == "world":
        step = min(z for z in src.zooms3 if z > 0)
        new = np.array(world, dtype=float)
        new[ras_axis] += n * step
        lo, hi = geometry.world_bounds(src.affine, src.spatial)
        new[ras_axis] = float(np.clip(new[ras_axis], lo[ras_axis], hi[ras_axis]))
        return cursor_set_world(store, *map(float, new), snap=False)
    ornt = views.orientation(src)
    d = ornt.data_of[ras_axis]
    vox = list(src.clamp_voxel(src.world_to_voxel(world)))
    vox[d] = max(0, min(vox[d] + n * ornt.sign[d], src.spatial[d] - 1))
    return cursor_set_voxel(store, *vox)


@command("cursor.set_slice", "Go to a slice", category="Navigate")
def cursor_set_slice(store: "SceneStore", plane: Plane, index: int) -> set[str]:
    """Put the crosshair on slice ``index`` of the plane (file index)."""
    _layer_, src = views.base(store)
    vox = views.cursor_voxel(store)
    if src is None or vox is None:
        return set()
    d = views.orientation(src).data_of[PLANE_AXIS[plane]]
    new = list(vox)
    new[d] = int(index)
    return cursor_set_voxel(store, *new)


@command("cursor.center", "Centre the crosshair", category="Navigate")
def cursor_center(store: "SceneStore") -> set[str]:
    _layer_, src = views.base(store)
    if src is None:
        return set()
    w = src.center_world()
    return cursor_set_world(store, float(w[0]), float(w[1]), float(w[2]), snap=False)


# ---------------------------------------------------------------------------
# Frames (4-D)
# ---------------------------------------------------------------------------


@command("frame.set", "Go to a volume of a 4-D series", category="Navigate")
def frame_set(store: "SceneStore", frame: int, layer: Optional[str] = None) -> set[str]:
    lay = _series(store, layer)
    if lay is None:
        return set()
    src = views.source_of(store, lay)
    n = src.n_frames if src is not None else 1
    frame = max(0, min(int(frame), n - 1))
    if lay.frame == frame:
        return set()
    lay.frame = frame
    return {f"layer:{lay.id}.frame"}


@command("frame.step", "Step through volumes", category="Navigate")
def frame_step(store: "SceneStore", n: int = 1, layer: Optional[str] = None,
               wrap: bool = False) -> set[str]:
    lay = _series(store, layer)
    if lay is None:
        return set()
    src = views.source_of(store, lay)
    total = src.n_frames if src is not None else 1
    if total <= 1:
        return set()
    target = lay.frame + int(n)
    if wrap:
        target %= total
    return frame_set(store, target, layer=lay.id)


def _shells(store: "SceneStore", layer: Optional[str]):
    lay = _series(store, layer)
    src = views.source_of(store, lay)
    b = getattr(src, "bvals", None) if src is not None else None
    if lay is None or b is None or len(b) < src.n_frames or src.n_frames <= 1:
        return None, None
    from ..bids import shells_of

    return lay, shells_of(b[: src.n_frames])


@command("frame.same_shell", "Next volume of this shell", category="Navigate")
def frame_same_shell(store: "SceneStore", n: int = 1, layer: Optional[str] = None) -> set[str]:
    """Step through the volumes of the shell on screen (every b=1000
    direction, say), wrapping at the end: how a bad direction is found."""
    lay, shells = _shells(store, layer)
    if lay is None:
        return set()
    t = max(0, min(lay.frame, len(shells) - 1))
    same = np.flatnonzero(shells == shells[t])
    if same.size <= 1:
        return set()
    k = int(np.searchsorted(same, t))
    return frame_set(store, int(same[(k + int(n)) % same.size]), layer=lay.id)


@command("frame.shell", "Go to another shell", category="Navigate")
def frame_shell(store: "SceneStore", n: int = 1, layer: Optional[str] = None) -> set[str]:
    """The first volume of the next (``n`` > 0) or previous shell, by
    b-value, wrapping."""
    lay, shells = _shells(store, layer)
    if lay is None:
        return set()
    order = sorted(set(int(s) for s in shells))
    if len(order) <= 1:
        return set()
    t = max(0, min(lay.frame, len(shells) - 1))
    target = order[(order.index(int(shells[t])) + int(n)) % len(order)]
    return frame_set(store, int(np.flatnonzero(shells == target)[0]), layer=lay.id)


# ---------------------------------------------------------------------------
# View layout and display options
# ---------------------------------------------------------------------------


@command("view.mode", "Choose the view layout", category="View")
def view_mode(store: "SceneStore", mode: Mode) -> set[str]:
    if store.scene.mode == mode:
        return set()
    store.scene.mode = mode
    return {"mode"}


@command("view.toggle_mode", "Toggle a view layout", category="View")
def view_toggle_mode(store: "SceneStore", mode: Mode) -> set[str]:
    """Switch to ``mode``, or back to a single slice if already there."""
    return view_mode(store, "single" if store.scene.mode == mode else mode)


@command("view.plane", "Show one plane", category="View")
def view_plane(store: "SceneStore", plane: Plane, single: bool = False) -> set[str]:
    """Set the single view's plane; ``single`` also leaves any multi view."""
    changed: set[str] = set()
    if single and store.scene.mode != "single":
        store.scene.mode = "single"
        changed.add("mode")
    if store.scene.plane != plane:
        store.scene.plane = plane
        changed.add("plane")
    return changed


_DISPLAY_FLAGS = ("ras", "radiological", "labels", "colorbar", "ruler", "crosshair")


@command("view.flag", "Toggle a display option", category="View")
def view_flag(store: "SceneStore",
              flag: Literal["ras", "radiological", "labels", "colorbar", "ruler", "crosshair"],
              value: Optional[bool] = None) -> set[str]:
    current = getattr(store.scene.display, flag)
    new = (not current) if value is None else bool(value)
    if new == current:
        return set()
    setattr(store.scene.display, flag, new)
    return {f"display.{flag}"}


@command("view.space", "Draw on the voxel grid or in world space", category="View")
def view_space(store: "SceneStore", space: Optional[Literal["voxel", "world"]] = None) -> set[str]:
    current = store.scene.display.space
    new = space or ("world" if current == "voxel" else "voxel")
    if new == current:
        return set()
    store.scene.display.space = new
    return {"display.space"}


@command("view.graph", "Show the time-course graph", category="View")
def view_graph(store: "SceneStore", value: Optional[bool] = None) -> set[str]:
    new = (not store.scene.graph_visible) if value is None else bool(value)
    if new == store.scene.graph_visible:
        return set()
    store.scene.graph_visible = new
    return {"graph_visible"}


@command("view.mosaic_text", "Set the mosaic line", category="View",
         undoable=True)
def view_mosaic_text(store: "SceneStore", text: str) -> set[str]:
    """Set the mosaic grammar line by hand (see ``viz.compute.mosaic``): the
    builder then leaves it alone until it is asked to build again."""
    text = " ".join(str(text).split())
    if store.scene.mosaic == text:
        return set()
    store.scene.mosaic = text
    store.scene.mosaic_build.custom = True
    return {"mosaic"}


def mosaic_extent(store: "SceneStore", plane: str) -> Optional[tuple[float, float]]:
    """The head's extent along ``plane``'s world axis (mm), cached on the
    source; None without an image."""
    from ..compute import mosaic as M

    _layer, src = views.base(store)
    if src is None:
        return None
    t = views.frame_of(store, _layer, src) if _layer is not None else 0
    key = ("mosaic-extent", plane, t)
    cache = getattr(src, "_range_cache", None)
    if cache is not None and key in cache:
        return cache[key]
    raw = src.raw_frame(t)
    if raw is None:
        return None
    try:
        found = M.head_extent(src.scale(raw), src.affine, PLANE_AXIS[plane])
    except ValueError:
        return None
    if cache is not None:
        cache[key] = found
    return found


def _built_line(store: "SceneStore") -> Optional[str]:
    from ..compute import mosaic as M

    b = store.scene.mosaic_build
    fit = mosaic_extent(store, b.plane)
    if fit is None and (b.start is None or b.end is None):
        return None
    start = b.start if b.start is not None else fit[0]
    end = b.end if b.end is not None else fit[1]
    if end <= start:
        start, end = end, start
    world = views.cursor_world(store)
    ref_axis = PLANE_AXIS[M.REFERENCE[b.plane]]
    ref_mm = float(world[ref_axis]) if world is not None else 0.0
    return M.build_line(b.plane, b.rows, b.cols, start, end, labels=b.labels,
                        reference=b.reference, overlap=b.overlap,
                        reference_mm=round(ref_mm, 1))


@command("mosaic.set", "Build the mosaic", category="View", undoable=True)
def mosaic_set(store: "SceneStore", plane: Optional[Plane] = None, rows: Optional[int] = None,
               cols: Optional[int] = None, start: Optional[float] = None,
               end: Optional[float] = None, labels: Optional[bool] = None,
               reference: Optional[bool] = None, overlap: Optional[float] = None,
               crop: Optional[bool] = None, fit: bool = False) -> set[str]:
    """A grid of one plane's slices: ``rows`` x ``cols`` across ``start``
    to ``end`` mm (``fit``: across the head, found in the image). The
    builder takes over from a line written by hand."""
    b = store.scene.mosaic_build
    before = b.model_copy()
    if plane is not None and plane != b.plane:
        b.plane = plane
        b.start = b.end = None          # another axis: fit it again
    for name, value in (("rows", rows), ("cols", cols), ("labels", labels),
                        ("reference", reference), ("overlap", overlap), ("crop", crop)):
        if value is not None:
            setattr(b, name, value)
    if start is not None:
        b.start = float(start)
    if end is not None:
        b.end = float(end)
    if fit:
        b.start = b.end = None
    b.custom = False
    store.scene.mosaic_build = type(b).model_validate(b.model_dump())
    line = _built_line(store)
    changed = set() if store.scene.mosaic_build == before else {"mosaic"}
    if line is not None and line != store.scene.mosaic:
        store.scene.mosaic = line
        changed.add("mosaic")
    return changed


@command("mosaic.refresh", "Fit the mosaic to the image", category="View")
def mosaic_refresh(store: "SceneStore") -> set[str]:
    """Build the line again for the image on screen (a new file, a new
    head), unless it was written by hand."""
    if store.scene.mosaic_build.custom:
        return set()
    line = _built_line(store)
    if line is None or line == store.scene.mosaic:
        return set()
    store.scene.mosaic = line
    return {"mosaic"}


@command("layout.set", "Arrange the views", category="View", undoable=True)
def layout_set(store: "SceneStore",
               arrangement: Optional[Literal["auto", "row", "column", "grid"]] = None,
               planes: Optional[list[Plane]] = None, hero: Optional[str] = None,
               hero_fraction: Optional[float] = None,
               hero_side: Optional[Literal["left", "top"]] = None,
               graph: Optional[Literal["bottom", "right"]] = None,
               quality: Optional[Literal["right", "bottom"]] = None) -> set[str]:
    """Change how the multi-view layouts are arranged."""
    lay = store.scene.layout
    before = lay.model_copy(deep=True)
    if arrangement is not None:
        lay.arrangement = arrangement
    if planes is not None:
        clean = list(dict.fromkeys(p for p in planes if p in PLANES))
        if not clean:
            raise ValueError("a layout needs at least one plane")
        lay.planes = clean
    if hero is not None:
        if hero not in ("", "render", *PLANES):
            raise ValueError(f"no view called {hero!r}")
        lay.hero = hero
    if hero_fraction is not None:
        lay.hero_fraction = float(np.clip(hero_fraction, 0.3, 0.85))
    if hero_side is not None:
        lay.hero_side = hero_side
    if graph is not None:
        lay.graph = graph
    if quality is not None:
        lay.quality = quality
    return set() if lay == before else {"layout"}


@command("layout.reset", "Restore the layout's defaults", category="View", undoable=True)
def layout_reset(store: "SceneStore") -> set[str]:
    """Arrangement, planes shown, large view and graph placement as
    installed."""
    from ..scene import LayoutState

    if store.scene.layout == LayoutState():
        return set()
    store.scene.layout = LayoutState()
    return {"layout"}


@command("display.reset", "Restore the slice views' defaults", category="View",
         undoable=True)
def display_reset(store: "SceneStore") -> set[str]:
    """Orientation labels, RAS, radiological, world space, colour bar and
    crosshair as installed."""
    from ..scene import Display

    if store.scene.display == Display():
        return set()
    store.scene.display = Display()
    return {"display.all"}


@command("layer.reset", "Restore a layer's look", category="Layers", undoable=True)
def layer_reset(store: "SceneStore", layer: Optional[str] = None,
                display: Optional[dict] = None) -> set[str]:
    """Replace a layer's whole look with ``display`` (the caller's default:
    the user's defaults for the base image, the look its content calls for
    for an overlay)."""
    lay = _layer(store, layer)
    if lay is None:
        return set()
    new = VolumeDisplay.model_validate(display or {})
    # What the look is ABOUT stays: an atlas keeps its table.
    new.label_table = lay.display.label_table
    if new == lay.display:
        return set()
    lay.display = new
    return {"layers", f"layer:{lay.id}", f"layer:{lay.id}.display"}


@command("view.zoom", "Zoom a 2-D view", category="View")
def view_zoom(store: "SceneStore", plane: Plane, factor: float,
              about: Optional[tuple[float, float]] = None) -> set[str]:
    """Multiply a view's zoom. ``about`` (screen mm from the centre) stays put."""
    state = store.scene.views[plane]
    new_zoom = float(np.clip(state.zoom * factor, 1.0, 40.0))
    if new_zoom == state.zoom:
        return set()
    if about is not None:
        # Keep the point under the mouse where it is: the pan is in screen
        # millimetres at zoom 1, so scale it about that point.
        ratio = new_zoom / state.zoom
        px, py = state.pan
        ax, ay = about
        state.pan = (ax - (ax - px) * ratio, ay - (ay - py) * ratio)
    state.zoom = new_zoom
    if new_zoom == 1.0:
        state.pan = (0.0, 0.0)
    return {f"views:{plane}"}


@command("view.pan", "Pan a 2-D view", category="View")
def view_pan(store: "SceneStore", plane: Plane, dx: float, dy: float) -> set[str]:
    state = store.scene.views[plane]
    if state.zoom <= 1.0:
        return set()
    state.pan = (state.pan[0] + dx, state.pan[1] + dy)
    return {f"views:{plane}"}


@command("view.reset", "Reset zoom and pan", category="View")
def view_reset(store: "SceneStore", plane: Optional[Plane] = None) -> set[str]:
    changed = set()
    for p in ([plane] if plane else PLANES):
        state = store.scene.views[p]
        if state.zoom != 1.0 or state.pan != (0.0, 0.0):
            state.zoom = 1.0
            state.pan = (0.0, 0.0)
            changed.add(f"views:{p}")
    return changed


@command("graph.set", "Graph options", category="View")
def graph_set(store: "SceneStore", scope: Optional[int] = None,
              dot: Optional[int] = None, mark_neighbors: Optional[bool] = None,
              scaling: Optional[Literal["raw", "percent", "demean"]] = None,
              x_axis: Optional[Literal["auto", "frames", "seconds"]] = None,
              events: Optional[bool] = None, physio: Optional[bool] = None,
              qc: Optional[bool] = None, layer: Optional[str] = None,
              qc_rows: Optional[list[str]] = None,
              tracks_mode: Optional[Literal["fit", "scroll"]] = None) -> set[str]:
    g = store.scene.graph
    changed = False
    if qc_rows is not None:
        from ..compute.qc import QC_ROW_IDS

        unknown = [r for r in qc_rows if r not in QC_ROW_IDS]
        if unknown:
            raise ValueError(f"no QC plot called {unknown[0]!r}")
        # In the user's order, each once.
        qc_rows = list(dict.fromkeys(qc_rows))
    if layer:
        found = store.scene.layer(layer)
        if found is None or found.kind != "volume":
            raise ValueError(f"no layer called {layer!r}")
    for name, value, clamp in (
        ("scope", scope, (1, 4)), ("dot", dot, (1, 20)),
        ("mark_neighbors", mark_neighbors, None), ("scaling", scaling, None),
        ("x_axis", x_axis, None), ("events", events, None), ("physio", physio, None),
        ("qc", qc, None), ("layer", layer, None), ("qc_rows", qc_rows, None),
        ("tracks_mode", tracks_mode, None),
    ):
        if value is None:
            continue
        if clamp is not None:
            value = max(clamp[0], min(int(value), clamp[1]))
        if getattr(g, name) != value:
            setattr(g, name, value)
            changed = True
    return {"graph"} if changed else set()


# ---------------------------------------------------------------------------
# Layer display
# ---------------------------------------------------------------------------


_DISPLAY_FIELDS = {
    "colormap", "colormap_negative", "window", "window_negative", "gamma",
    "invert", "opacity", "threshold_mode", "outline_px", "interpolation",
}


@command("layer.set", "Change how a layer is drawn", category="Layers",
         undoable=True)
def layer_set(store: "SceneStore", layer: Optional[str] = None,
              colormap: Optional[str] = None,
              colormap_negative: Optional[str] = None,
              window: Optional[tuple[float, float]] = None,
              window_negative: Optional[tuple[float, float]] = None,
              gamma: Optional[float] = None, invert: Optional[bool] = None,
              opacity: Optional[float] = None,
              threshold_mode: Optional[Literal["range", "hide_below", "translucent_below"]] = None,
              outline_px: Optional[float] = None,
              interpolation: Optional[Literal["linear", "nearest"]] = None,
              visible: Optional[bool] = None, in_3d: Optional[bool] = None,
              name: Optional[str] = None) -> set[str]:
    """Patch a layer's display, NiiVue's ``setVolume`` in spirit."""
    lay = _layer(store, layer)
    if lay is None:
        return set()
    changed: set[str] = set()
    values = dict(colormap=colormap, colormap_negative=colormap_negative,
                  window=window, window_negative=window_negative, gamma=gamma,
                  invert=invert, opacity=opacity, threshold_mode=threshold_mode,
                  outline_px=outline_px, interpolation=interpolation)
    for key, value in values.items():
        if value is None:
            continue
        if key == "window":
            lo, hi = float(value[0]), float(value[1])
            if hi < lo:
                lo, hi = hi, lo
            if hi == lo:
                hi = lo + 1e-6
            value = (lo, hi)
        if key == "gamma":
            value = float(np.clip(value, 0.1, 10.0))
        if key == "opacity":
            value = float(np.clip(value, 0.0, 1.0))
        if getattr(lay.display, key) != value:
            setattr(lay.display, key, value)
            changed.add(f"layer:{lay.id}.display")
    for key, value in (("visible", visible), ("in_3d", in_3d), ("name", name)):
        if value is not None and getattr(lay, key) != value:
            setattr(lay, key, value)
            changed.add(f"layer:{lay.id}")
    return changed


@command("layer.toggle", "Toggle a layer option", category="Layers",
         undoable=True)
def layer_toggle(store: "SceneStore",
                 field: Literal["invert", "visible", "in_3d", "nearest"],
                 layer: Optional[str] = None) -> set[str]:
    lay = _layer(store, layer)
    if lay is None:
        return set()
    if field == "invert":
        return layer_set(store, layer=lay.id, invert=not lay.display.invert)
    if field == "nearest":
        new = "linear" if lay.display.interpolation == "nearest" else "nearest"
        return layer_set(store, layer=lay.id, interpolation=new)
    return layer_set(store, layer=lay.id, **{field: not getattr(lay, field)})


@command("layer.add", "Add a layer", category="Layers", undoable=True)
def layer_add(store: "SceneStore", id: str, source: str, name: str = "",
              display: Optional[dict] = None, path: str = "", origin: str = "",
              visible: bool = True, in_3d: bool = True) -> set[str]:
    """Put a volume over the others. Its source must already be in the
    store (reading it is the caller's work, on a worker); ``path`` records
    where it came from, so a saved scene can open it again."""
    if store.scene.layer(id) is not None:
        raise ValueError(f"there is already a layer called {id!r}")
    if source not in store.sources:
        raise ValueError(f"layer.add: no source {source!r} has been loaded")
    if path and source not in store.scene.sources:
        store.scene.sources[source] = SourceRef(id=source, path=path, kind="volume")
    store.scene.layers.append(VolumeLayer(
        id=id, source=source, name=name or id,
        display=VolumeDisplay.model_validate(display or {}),
        in_3d=in_3d, origin=origin, visible=visible,
    ))
    return {"layers", f"layer:{id}"}


def _overlay_index(store: "SceneStore", layer: str) -> int:
    base = store.scene.base_layer()
    for i, lay in enumerate(store.scene.layers):
        if lay.id == layer:
            if base is not None and lay.id == base.id:
                raise ValueError("the base image is not an overlay: open another file instead")
            return i
    raise ValueError(f"no layer called {layer!r}")


@command("layer.remove", "Remove an overlay", category="Layers", undoable=True)
def layer_remove(store: "SceneStore", layer: str) -> set[str]:
    """Take an overlay off. The base image stays: it is the file that is
    open, and the grid every view cuts."""
    del store.scene.layers[_overlay_index(store, layer)]
    return {"layers", f"layer:{layer}"}


@command("layer.move", "Move an overlay up or down", category="Layers", undoable=True)
def layer_move(store: "SceneStore", layer: str, by: int = 1) -> set[str]:
    """Raise (``by`` > 0) or lower an overlay among the overlays. Never below
    the base image, which is drawn first by definition."""
    i = _overlay_index(store, layer)
    layers = store.scene.layers
    base = store.scene.base_layer()
    lowest = next((k + 1 for k, lay in enumerate(layers) if base is not None and lay.id == base.id), 0)
    j = int(np.clip(i + int(by), lowest, len(layers) - 1))
    if j == i:
        return set()
    layers.insert(j, layers.pop(i))
    return {"layers"}


@command("window.robust", "Window to the robust range", category="Layers",
         undoable=True)
def window_robust(store: "SceneStore", layer: Optional[str] = None,
                  series: bool = False) -> set[str]:
    """Set the window to the 1st-99th percentile (of the frame, or of a
    sample of the whole series)."""
    lay = _layer(store, layer)
    src = views.source_of(store, lay)
    if lay is None or src is None:
        return set()
    rng = src.series_range() if series else src.robust_range(views.frame_of(store, lay, src))
    if rng is None:
        return set()
    return layer_set(store, layer=lay.id, window=rng)


@command("window.full", "Window to the full range", category="Layers",
         undoable=True)
def window_full(store: "SceneStore", layer: Optional[str] = None) -> set[str]:
    lay = _layer(store, layer)
    src = views.source_of(store, lay)
    if lay is None or src is None:
        return set()
    rng = src.robust_range(views.frame_of(store, lay, src), lo_pct=0.0, hi_pct=100.0)
    if rng is None:
        return set()
    return layer_set(store, layer=lay.id, window=rng)


@command("window.level_width", "Adjust window level and width",
         category="Layers")
def window_level_width(store: "SceneStore", d_level: float, d_width: float,
                       layer: Optional[str] = None) -> set[str]:
    """Shift the level and stretch the width, both as fractions of the
    current width (a mouse drag sends small fractions)."""
    lay = _layer(store, layer)
    if lay is None or lay.display.window is None:
        return set()
    lo, hi = lay.display.window
    width = max(hi - lo, 1e-9)
    level = (lo + hi) / 2.0 + d_level * width
    width = max(width * (1.0 + d_width), 1e-9)
    return layer_set(store, layer=lay.id,
                     window=(level - width / 2.0, level + width / 2.0))


@command("window.fit_box", "Fit the window to a box", category="Layers",
         undoable=True)
def window_fit_box(store: "SceneStore", plane: Plane,
                   c0: float, r0: float, c1: float, r1: float,
                   layer: Optional[str] = None) -> set[str]:
    """Window to the range inside a rectangle of a view (grid pixels)."""
    lay = _layer(store, layer)
    grid_ = views.grid(store, plane)
    if lay is None or grid_ is None:
        return set()
    values = views.layer_values(store, lay, grid_)
    if values is None or values.ndim != 2:
        return set()
    rows, cols = values.shape
    a0, a1 = sorted((int(np.floor(c0)), int(np.ceil(c1))))
    b0, b1 = sorted((int(np.floor(r0)), int(np.ceil(r1))))
    a0, a1 = max(0, a0), min(cols, max(a1, a0 + 1))
    b0, b1 = max(0, b0), min(rows, max(b1, b0 + 1))
    box = values[b0:b1, a0:a1]
    box = box[np.isfinite(box)]
    if box.size < 2:
        return set()
    lo, hi = np.percentile(box, (1.0, 99.0))
    if hi <= lo:
        # One value carries no contrast to fit; inventing a width would put
        # everything in the box at one end of the colour map.
        return set()
    return layer_set(store, layer=lay.id, window=(float(lo), float(hi)))


__all__: list[str] = []

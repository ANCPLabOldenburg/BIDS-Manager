"""Keeping two viewers in step: comparison and before/after views.

The old comparison synced a 15-key dictionary through two signals and three
loop guards, and still missed the contrast. Here a link copies scene STATE
from one store to the other whenever the first changes, in world
millimetres for the cursor, so two images of different shapes, resolutions or
storage orders still point at the same anatomy.

What is shared is chosen per link (``cursor``, ``view``, ``render``,
``window``, ``frame``). A copy runs commands on the other store with a guard,
so it never echoes back.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...viz.scene import Display, GraphState, RenderState, SliceViewState


def apply_view_state(store, state: Mapping[str, Any]) -> None:
    """Adopt another viewer's view (mode, planes, display, graph, 3-D)."""
    scene = store.scene
    changed: set[str] = set()
    if "mode" in state and state["mode"] != scene.mode:
        scene.mode = state["mode"]
        changed.add("mode")
    if "plane" in state and state["plane"] != scene.plane:
        scene.plane = state["plane"]
        changed.add("plane")
    if "display" in state:
        new = Display.model_validate(state["display"])
        if new != scene.display:
            scene.display = new
            changed.add("display.all")
    if "views" in state:
        for plane, v in state["views"].items():
            new = SliceViewState.model_validate(v)
            if scene.views.get(plane) != new:
                scene.views[plane] = new
                changed.add(f"views:{plane}")
    if "graph" in state:
        new = GraphState.model_validate(state["graph"])
        if new != scene.graph:
            scene.graph = new
            changed.add("graph")
    if "graph_visible" in state and state["graph_visible"] != scene.graph_visible:
        scene.graph_visible = bool(state["graph_visible"])
        changed.add("graph_visible")
    if "render" in state:
        new = RenderState.model_validate(state["render"])
        if new != scene.render:
            if new.camera != scene.render.camera:
                changed.add("render.camera")
            if new.effect != scene.render.effect:
                changed.add("render.effect")
            if new.params != scene.render.params or new.shared != scene.render.shared:
                changed.add("render.params")
            if (new.cut_away, new.cut_at_cursor) != (scene.render.cut_away,
                                                     scene.render.cut_at_cursor):
                changed.add("clips")
            scene.render = new
    if "clips" in state:
        from ...viz.scene import ClipPlane

        new = [ClipPlane.model_validate(c) for c in state["clips"]]
        if new != scene.clips:
            scene.clips = new
            changed.add("clips")
    if "mosaic" in state and state["mosaic"] != scene.mosaic:
        scene.mosaic = state["mosaic"]
        changed.add("mosaic")
    if "mosaic_build" in state:
        from ...viz.scene import MosaicBuild

        new = MosaicBuild.model_validate(state["mosaic_build"])
        if new != scene.mosaic_build:
            scene.mosaic_build = new
            changed.add("mosaic")
    if "layout" in state:
        # Ignored before: a linked viewer and a saved preset both lost the
        # arrangement, the planes shown and the large view.
        from ...viz.scene import LayoutState

        new = LayoutState.model_validate(state["layout"])
        if new != scene.layout:
            scene.layout = new
            changed.add("layout")
    cursor = (state.get("cursor") or {}).get("world") if isinstance(state.get("cursor"), Mapping) else None
    if cursor is not None:
        changed |= store.run("cursor.set_world", x=cursor[0], y=cursor[1], z=cursor[2], snap=True)
    if changed:
        store.changed(changed)


class ViewerLink:
    """Mirror changes from viewer ``a`` to viewer ``b`` and back."""

    def __init__(self, a, b, *, cursor: bool = True, view: bool = True,
                 render: bool = True, window: bool = False, frame: bool = True) -> None:
        self.a, self.b = a, b
        self.cursor, self.view, self.render = cursor, view, render
        self.window, self.frame = window, frame
        self._busy = False
        self.enabled = True
        a.qstore.changed.connect(lambda paths: self._mirror(a, b, paths))
        b.qstore.changed.connect(lambda paths: self._mirror(b, a, paths))

    def sync_now(self, source=None) -> None:
        """Make ``b`` match ``a`` (or the given side) entirely."""
        src = source or self.a
        dst = self.b if src is self.a else self.a
        self._mirror(src, dst, frozenset({"scene", "cursor", "layer:base.frame",
                                          "layer:base.display"}))

    def _mirror(self, src, dst, paths) -> None:
        if self._busy or not self.enabled:
            return
        if dst.store.scene.base_layer() is None or src.store.scene.base_layer() is None:
            return
        self._busy = True
        try:
            s = src.store.scene
            state: dict[str, Any] = {}
            whole = "scene" in paths
            if self.view and (whole or paths & {"mode", "plane", "graph", "graph_visible",
                                                "mosaic", "layout"}
                              or any(p.startswith(("display", "views:", "graph.")) for p in paths)):
                state.update(s.model_dump(mode="json", include={
                    "mode", "plane", "display", "views", "graph", "graph_visible", "mosaic",
                    "layout", "mosaic_build"}))
            if self.render and (whole or any(p.startswith(("render", "clips")) for p in paths)):
                state.update(s.model_dump(mode="json", include={"render", "clips"}))
            if self.cursor and (whole or "cursor" in paths) and s.cursor.world is not None:
                state["cursor"] = {"world": list(s.cursor.world)}
            if state:
                apply_view_state(dst.store, state)
            base_src = s.base_layer()
            if self.frame and (whole or any(p.endswith(".frame") for p in paths)):
                dst.store.run("frame.set", frame=base_src.frame)
            if self.window and (whole or any(p.endswith(".display") for p in paths)):
                d = base_src.display
                params = dict(gamma=d.gamma, colormap=d.colormap, invert=d.invert)
                if d.window is not None:
                    params["window"] = d.window
                dst.store.run("layer.set", **params)
            dst.qstore.flush()
        finally:
            self._busy = False


def link(a, b, **kwargs) -> ViewerLink:
    return ViewerLink(a, b, **kwargs)


__all__ = ["ViewerLink", "apply_view_state", "link"]

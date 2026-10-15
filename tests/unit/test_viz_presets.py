"""Presets carry what their image has, and an image takes what it has."""

from __future__ import annotations

from bidsmgr.viz import presets
from bidsmgr.viz.scene import Scene


def _bold_look() -> Scene:
    scene = Scene()
    scene.mode = "hero"
    scene.graph_visible = True
    scene.graph.qc = True
    scene.graph.qc_rows = ["fd", "carpet"]
    scene.graph.layer = "overlay-2"
    scene.layout.graph = "right"
    scene.layout.arrangement = "column"
    return scene


def test_a_preset_is_a_look_not_a_place_nor_another_viewer():
    state = presets.snapshot(_bold_look(), has_series=True)
    for key in ("cursor", "views", "layers", "sources", "traces", "spectrum"):
        assert key not in state
    assert state["graph"]["qc_rows"] == ["fd", "carpet"]
    assert state["graph"]["layer"] == "", "a layer id belongs to one image"


def test_saved_without_a_time_series_it_says_nothing_about_one():
    state = presets.snapshot(_bold_look(), has_series=False)
    assert "graph" not in state and "graph_visible" not in state
    assert "graph" not in state["layout"]
    assert state["layout"]["arrangement"] == "column"


def test_an_image_without_a_time_series_takes_only_the_rest():
    state = presets.snapshot(_bold_look(), has_series=True)
    t1w = Scene()
    t1w.layout.graph = "bottom"
    got = presets.applicable(state, t1w, has_series=False)
    assert "graph" not in got and "graph_visible" not in got
    assert got["mode"] == "hero" and got["layout"]["arrangement"] == "column"
    assert got["layout"]["graph"] == "bottom", "the image's own placement stays"


def test_a_series_keeps_its_own_placement_when_the_preset_has_none():
    state = presets.snapshot(_bold_look(), has_series=False)
    bold = Scene()
    bold.layout.graph = "right"
    got = presets.applicable(state, bold, has_series=True)
    assert got["layout"]["graph"] == "right"
    assert "graph_visible" not in got

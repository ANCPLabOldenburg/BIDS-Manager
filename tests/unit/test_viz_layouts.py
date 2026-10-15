"""Which layout a file opens in, by kind."""

from __future__ import annotations

import pytest

from bidsmgr.viz import layouts as L


@pytest.mark.parametrize("datatype, suffix, is_4d, expected", [
    ("func", "bold", True, "mri.func"),
    ("func", "sbref", True, "mri.func"),
    ("func", "bold", False, "volume"),        # a 3-D "bold" is not a run
    ("pet", "pet", True, "pet.dynamic"),
    ("pet", "pet", False, "pet"),
    ("dwi", "dwi", True, "mri.dwi"),
    ("perf", "asl", True, "mri.perf"),
    ("fmap", "phasediff", False, "mri.fmap"),
    ("anat", "T1w", False, "mri.anat"),
    ("", "", True, "volume.4d"),
    ("", "", False, "volume"),
])
def test_files_match_their_kind(datatype, suffix, is_4d, expected) -> None:
    assert L.match(datatype, suffix, is_4d).id == expected


def test_every_preset_has_a_unique_id_and_the_last_matches_anything() -> None:
    ids = [p.id for p in L.PRESETS]
    assert len(ids) == len(set(ids))
    assert L.PRESETS[-1].match == {}


def test_a_remembered_arrangement_wins_over_the_preset() -> None:
    preset = L.PRESET_BY_ID["mri.func"]
    view = L.opening_view(preset, None, "multi")
    assert view["graph_visible"] and view["mode"] == "multi"
    view = L.opening_view(preset, {"mode": "single", "graph_visible": False,
                                   "display": {"radiological": True}}, "multi")
    assert view["mode"] == "single" and not view["graph_visible"]
    assert "display" not in view            # the look is never part of a layout


def test_an_automatic_mode_takes_the_machine_default() -> None:
    assert L.opening_view(L.PRESET_BY_ID["mri.anat"], None, "combo")["mode"] == "combo"


# ---------------------------------------------------------------------------
# Tiles: which views, and where
# ---------------------------------------------------------------------------

from bidsmgr.viz.scene import LayoutState  # noqa: E402


class TestTiles:
    def test_each_mode_shows_its_views(self):
        lay = LayoutState()
        assert L.tiles_for("single", "axial", lay, True) == (["axial"], "")
        assert L.tiles_for("multi", "axial", lay, True) == (["sagittal", "coronal", "axial"], "")
        assert L.tiles_for("3d", "axial", lay, True) == (["render"], "")
        assert L.tiles_for("combo", "axial", lay, True)[0][-1] == "render"
        tiles, hero = L.tiles_for("hero", "coronal", lay, True)
        assert hero == "coronal" and tiles[0] == "coronal" and "render" in tiles

    def test_without_a_gpu_the_render_is_left_out(self):
        lay = LayoutState(hero="render")
        assert L.tiles_for("3d", "axial", lay, False) == (["axial"], "")
        assert "render" not in L.tiles_for("combo", "axial", lay, False)[0]
        tiles, hero = L.tiles_for("hero", "sagittal", lay, False)
        assert hero == "sagittal" and "render" not in tiles

    def test_the_user_chooses_the_planes_and_their_order(self):
        lay = LayoutState(planes=["axial", "axial", "coronal"])
        assert L.tiles_for("multi", "axial", lay, True)[0] == ["axial", "coronal"]

    def test_the_render_can_be_the_large_view(self):
        tiles, hero = L.tiles_for("hero", "axial", LayoutState(hero="render"), True)
        assert hero == "render" and tiles == ["render", "sagittal", "coronal", "axial"]


class TestRects:
    def test_fixed_arrangements(self):
        assert [r[2] for r in L.tile_rects(3, 298, 100, arrangement="row", gap=2)] == [98.0] * 3
        assert [r[3] for r in L.tile_rects(3, 100, 298, arrangement="column", gap=2)] == [98.0] * 3
        grid = L.tile_rects(4, 202, 202, arrangement="grid", gap=2)
        assert {(r[0], r[1]) for r in grid} == {(0, 0), (102, 0), (0, 102), (102, 102)}

    def test_auto_picks_what_makes_the_images_largest(self):
        assert L.choose_arrangement(3, 1500, 400) == "row"
        assert L.choose_arrangement(3, 400, 1500) == "column"
        assert L.choose_arrangement(4, 800, 800) == "grid"

    def test_auto_minds_the_images_shape(self):
        # Wide images in a wide window still stack when stacking fits them better.
        assert L.choose_arrangement(2, 900, 700, aspect=3.0) == "column"

    def test_the_hero_takes_its_share(self):
        rects = L.tile_rects(4, 1000, 600, hero=True, hero_fraction=0.6, gap=0)
        assert rects[0] == (0.0, 0.0, 600.0, 600.0)
        assert all(r[0] == 600.0 and r[2] == 400.0 for r in rects[1:])
        top = L.tile_rects(3, 1000, 600, hero=True, hero_fraction=0.5, hero_side="top", gap=0)
        assert top[0] == (0.0, 0.0, 1000.0, 300.0) and top[1][1] == 300.0

    def test_nothing_overlaps_and_everything_fits(self):
        for n in range(1, 6):
            for arr in ("auto", "row", "column", "grid"):
                rects = L.tile_rects(n, 640, 480, arrangement=arr)
                assert len(rects) == n
                for x, y, w, h in rects:
                    assert x >= 0 and y >= 0 and x + w <= 640 + 1e-6 and y + h <= 480 + 1e-6


class TestLayoutCommand:
    def test_set_and_undo(self):
        from bidsmgr.viz.store import SceneStore

        store = SceneStore()
        store.run("layout.set", arrangement="column", planes=["axial", "coronal"])
        assert store.scene.layout.arrangement == "column"
        assert store.scene.layout.planes == ["axial", "coronal"]
        store.undo()
        assert store.scene.layout.planes == ["sagittal", "coronal", "axial"]

    def test_refusals_and_clamps(self):
        import pytest

        from bidsmgr.viz.store import SceneStore

        store = SceneStore()
        with pytest.raises(ValueError):
            store.run("layout.set", planes=[])
        with pytest.raises(ValueError):
            store.run("layout.set", hero="graph")
        store.run("layout.set", hero_fraction=0.99)
        assert store.scene.layout.hero_fraction == 0.85

    def test_a_layout_is_remembered_per_kind(self):
        assert "layout" in L.LAYOUT_KEYS


class TestGestures:
    def test_a_drag_is_one_undo_step(self):
        from bidsmgr.viz.store import SceneStore

        store = SceneStore()
        store.run("layout.set", hero_fraction=0.4)
        with store.gesture():
            for f in (0.45, 0.5, 0.55, 0.6, 0.65):
                store.run("layout.set", hero_fraction=f)
        assert store.scene.layout.hero_fraction == 0.65
        store.undo()
        assert store.scene.layout.hero_fraction == 0.4, "the whole drag, in one step"
        store.undo()
        assert store.scene.layout.hero_fraction == 0.62

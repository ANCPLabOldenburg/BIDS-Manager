"""Overlays in the 3-D render: the same colours as the slices, in the right place."""

from __future__ import annotations

import numpy as np
import pytest

from bidsmgr.viz import render3d
from bidsmgr.viz.compute import geometry, intensity, overlay3d
from bidsmgr.viz.scene import ClipPlane, LabelTable, VolumeDisplay


def _canonical(frame, affine):
    """What the render uploads: the frame in RAS order and direction."""
    ornt = geometry.orientation_of(affine)
    order = list(ornt.data_of)
    vol = np.transpose(frame, order)
    for ras in range(3):
        if ornt.sign[ornt.data_of[ras]] < 0:
            vol = np.flip(vol, axis=ras)
    return vol, ornt


@pytest.mark.parametrize("affine", [
    np.diag([2.0, 2.0, 2.0, 1.0]),
    np.diag([-1.0, 1.0, 3.0, 1.0]),
    # Stored as (y, z, x), x reversed: a sagittal acquisition.
    np.array([[0, 0, -1.5, 40], [1.0, 0, 0, -20], [0, 2.0, 0, 5], [0, 0, 0, 1]]),
])
def test_the_canonical_affine_puts_every_voxel_where_the_file_does(affine):
    rng = np.random.default_rng(0)
    frame = rng.random((4, 5, 6)).astype(np.float32)
    vol, ornt = _canonical(frame, affine)
    canon = overlay3d.canonical_affine(affine, frame.shape, ornt)
    inv = np.linalg.inv(affine)
    for c in [(0, 0, 0), (vol.shape[0] - 1, 1, 2), (1, vol.shape[1] - 1, vol.shape[2] - 1)]:
        world = canon @ np.array([*c, 1.0])
        d = np.round(inv @ world)[:3].astype(int)
        assert vol[c] == frame[tuple(d)]


def test_a_texture_voxel_centre_is_a_base_voxel_centre_when_sizes_match():
    canon = np.diag([2.0, 2.0, 2.0, 1.0])
    to_world = overlay3d.texture_to_world(canon, (10, 10, 10), (10, 10, 10))
    assert np.allclose(to_world, canon)


def test_a_coarser_texture_spans_the_same_box():
    canon = np.eye(4)
    to_world = overlay3d.texture_to_world(canon, (8, 8, 8), (4, 4, 4))
    # Texture voxel 0 covers base voxels 0-1; its centre is at 0.5.
    assert np.allclose(to_world @ [0, 0, 0, 1], [0.5, 0.5, 0.5, 1])
    assert np.allclose(to_world @ [3, 3, 3, 1], [6.5, 6.5, 6.5, 1])


def test_overlay_resolution_follows_the_data_and_has_a_ceiling():
    # A 3 mm map over a 1 mm 180-voxel box: sampled at 3 mm.
    assert overlay3d.overlay_dims((180, 180, 180), (1, 1, 1), (3, 3, 3)) == (60, 60, 60)
    # Never finer than the base, never over the cap.
    assert overlay3d.overlay_dims((40, 40, 40), (2, 2, 2), (0.5, 0.5, 0.5)) == (40, 40, 40)
    assert max(overlay3d.overlay_dims((512, 512, 300), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))) \
        == overlay3d.MAX_DIM


def test_resampling_lands_an_overlay_in_world_space():
    """A 2 mm overlay on a 1 mm base: its one bright voxel lands where it is."""
    over = np.zeros((5, 5, 5), np.float32)
    over[2, 3, 1] = 1.0
    over_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    canon = np.eye(4)
    dims = (10, 10, 10)
    to_world = overlay3d.texture_to_world(canon, dims, dims)
    out = overlay3d.resample_to_box(over, np.linalg.inv(over_affine), to_world, dims, order=0)
    assert out[4, 6, 2] == 1.0
    assert np.nansum(out) >= 1.0
    # Outside the overlay's own volume there is no data, not zero.
    assert np.isnan(out[9, 9, 9])


def test_volume_colours_are_the_slice_colours():
    rng = np.random.default_rng(1)
    vals = rng.normal(0, 2, (6, 7, 8)).astype(np.float32)
    look = VolumeDisplay(colormap="warm", colormap_negative="cool", window=(1.0, 3.0),
                         window_negative=(1.0, 3.0), threshold_mode="hide_below",
                         opacity=0.6, gamma=1.0)
    rgba = overlay3d.colorize_volume(vals, look, (1.0, 3.0))
    for z in (0, 4, 7):
        expect = intensity.colorize(vals[:, :, z], look, is_base=False, window=(1.0, 3.0))
        assert np.array_equal(rgba[:, :, z], expect)


def test_an_outlined_region_is_drawn_filled_in_3d():
    vals = np.zeros((6, 6, 6), np.float32)
    vals[1:5, 1:5, 1:5] = 1.0
    look = VolumeDisplay(colormap="green", window=(0.5, 1.0), threshold_mode="hide_below",
                         outline_px=2.0)
    rgba = overlay3d.colorize_volume(vals, look, (0.5, 1.0))
    assert rgba[2, 2, 2, 3] > 0, "the middle of the box is drawn"


def test_labels_keep_their_table_colours():
    vals = np.zeros((4, 4, 4), np.float32)
    vals[1, 1, 1] = 7
    table = LabelTable(labels={7: "Thalamus"}, colors={7: (10, 200, 30)})
    rgba = overlay3d.colorize_volume(vals, VolumeDisplay(label_table=table, opacity=0.5),
                                     (0.0, 1.0))
    assert tuple(rgba[1, 1, 1, :3]) == (10, 200, 30)
    assert rgba[0, 0, 0, 3] == 0


def test_layers_blend_bottom_to_top_and_keep_their_coverage():
    red = np.zeros((2, 1, 1, 4), np.uint8)
    red[..., 0], red[..., 3] = 255, 255
    blue = np.zeros((2, 1, 1, 4), np.uint8)
    blue[0, ..., 2], blue[0, ..., 3] = 255, 128
    out = overlay3d.composite_volumes([red, blue])
    assert out[1, 0, 0, 0] == 255 and out[1, 0, 0, 3] == 255     # only red
    assert out[0, 0, 0, 2] > 100 and out[0, 0, 0, 0] > 100       # blue over red
    empty = np.zeros((1, 1, 1, 4), np.uint8)
    assert overlay3d.composite_volumes([empty])[0, 0, 0, 3] == 0


def _input(values, affine, look, window, key="a", slope=1.0, inter=0.0):
    return overlay3d.OverlayInput(
        key=(key,), values=values, inv_affine=np.linalg.inv(affine),
        zooms=tuple(float(z) for z in np.abs(np.diag(affine)[:3])), display=look,
        window=window, slope=slope, inter=inter)


def test_the_overlay_volume_reuses_its_resample_when_only_the_look_changes():
    vals = np.zeros((8, 8, 8), np.float32)
    vals[2:6, 2:6, 2:6] = 1.0
    look = VolumeDisplay(colormap="red", window=(0.5, 1.0), threshold_mode="hide_below")
    canon = np.eye(4)
    rgba, dims, cache = overlay3d.build_overlay_volume(
        canon, (8, 8, 8), (1, 1, 1), [_input(vals, np.eye(4), look, (0.5, 1.0))])
    assert dims == (8, 8, 8) and rgba[3, 3, 3, 3] == 255 and rgba[0, 0, 0, 3] == 0
    faint = look.model_copy(update={"opacity": 0.25})
    sentinel = {k: np.full_like(v, 1.0) for k, v in cache.items()}
    again, _d, _c = overlay3d.build_overlay_volume(
        canon, (8, 8, 8), (1, 1, 1), [_input(vals, np.eye(4), faint, (0.5, 1.0))],
        cache=sentinel)
    # The cached values (all ones) were used: every voxel is drawn now.
    assert again[0, 0, 0, 3] == pytest.approx(64, abs=1)


def test_the_scale_is_applied_after_the_resample():
    raw = np.full((4, 4, 4), 10, np.int16)
    look = VolumeDisplay(colormap="gray", window=(0.0, 100.0), threshold_mode="hide_below")
    rgba, _d, cache = overlay3d.build_overlay_volume(
        np.eye(4), (4, 4, 4), (1, 1, 1),
        [_input(raw, np.eye(4), look, (0.0, 100.0), slope=5.0, inter=-10.0)])
    (values,) = cache.values()
    assert np.nanmax(values) == pytest.approx(40.0)


def test_nothing_to_draw_is_none():
    assert overlay3d.build_overlay_volume(np.eye(4), (4, 4, 4), (1, 1, 1), [])[0] is None


# ---------------------------------------------------------------------------
# Clip planes, rays and the depth pick
# ---------------------------------------------------------------------------


def test_only_active_planes_cut_and_at_most_six():
    clips = [ClipPlane(active=i % 2 == 0, az=10 * i) for i in range(14)]
    planes = render3d.clip_plane_uniforms(clips)
    assert len(planes) == render3d.MAX_CLIP_PLANES == 6
    assert render3d.clip_plane_uniforms([ClipPlane()]) == []


def _camera(az=0.0, el=0.0, dist=1.9):
    eye = render3d.camera_eye(az, el, dist, (0, 0, 0))
    view = render3d.look_at(eye, np.zeros(3, np.float32), np.array([0, 0, 1], np.float32))
    proj = render3d.perspective(45.0, 1.0, 0.01, 20.0)
    return np.linalg.inv(proj @ view)


def test_the_centre_ray_points_at_the_box():
    ro, rd = render3d.ray_from_ndc(_camera(), 0.0, 0.0)
    assert np.isclose(np.linalg.norm(rd), 1.0)
    hit = render3d.intersect_box(ro, rd, (0.5, 0.5, 0.5))
    assert hit is not None and hit[1] > hit[0] > 0
    assert render3d.intersect_box(ro, -rd, (0.5, 0.5, 0.5)) is None


def _cube(n=32, lo=8, hi=24):
    vol = np.zeros((n, n, n), np.uint8)
    vol[lo:hi, lo:hi, lo:hi] = 255
    return vol


def test_a_click_lands_on_the_near_face_of_what_is_seen():
    # Camera at az=0, el=0 looks from +y (anterior) toward the centre.
    ro, rd = render3d.ray_from_ndc(_camera(), 0.0, 0.0)
    tex = render3d.pick_depth(_cube(), (0.5, 0.5, 0.5), ro, rd, lo=0.1, hi=0.42, density=1.0)
    assert tex is not None
    assert tex[1] == pytest.approx(24 / 32, abs=0.04), "the face toward the camera"
    assert tex[0] == pytest.approx(0.5, abs=0.03) and tex[2] == pytest.approx(0.5, abs=0.03)


def test_a_click_beside_the_object_finds_nothing():
    ro, rd = render3d.ray_from_ndc(_camera(), 0.95, 0.95)
    assert render3d.pick_depth(_cube(), (0.5, 0.5, 0.5), ro, rd,
                               lo=0.1, hi=0.42, density=1.0) is None


def test_a_cut_lets_the_click_through_to_what_is_behind_it():
    ro, rd = render3d.ray_from_ndc(_camera(), 0.0, 0.0)
    # Cut away the front half (y > 0.5 in texture space).
    cut = render3d.clip_plane_uniforms([ClipPlane(active=True, az=0, el=0, pos=0.5)])
    tex = render3d.pick_depth(_cube(), (0.5, 0.5, 0.5), ro, rd, lo=0.1, hi=0.42,
                              density=1.0, clips=cut)
    assert tex is not None and tex[1] < 0.55


def test_a_projection_picks_the_brightest_point():
    vol = np.zeros((16, 16, 16), np.uint8)
    vol[8, 4, 8] = 255
    vol[8, 12, 8] = 100
    ro, rd = render3d.ray_from_ndc(_camera(), 0.0, 0.0)
    tex = render3d.pick_depth(vol, (0.5, 0.5, 0.5), ro, rd, lo=0.05, hi=0.5,
                              density=1.0, surface=False, steps=1024)
    assert tex is not None and tex[1] == pytest.approx(4.5 / 16, abs=0.04)


def test_texture_and_world_coordinates_round_trip():
    canon = np.array([[0.9, 0.1, 0, -30], [0, 1.1, 0, 12], [0, 0, 2.0, 4], [0, 0, 0, 1]])
    dims = (40, 50, 30)
    world = render3d.world_of_texcoord(canon, dims, (0.2, 0.5, 0.75))
    back = render3d.texcoord_of_world(np.linalg.inv(canon), dims, world)
    assert np.allclose(back, (0.2, 0.5, 0.75))
    # Voxel centres sit at (i + 0.5) / n, as the GPU samples them.
    assert np.allclose(render3d.texcoord_of_world(np.linalg.inv(canon), dims,
                                                  canon[:3, 3]), (0.5 / 40, 0.5 / 50, 0.5 / 30))


# ---------------------------------------------------------------------------
# How planes combine, cutting through the crosshair, shared parameters
# ---------------------------------------------------------------------------


def _planes(*facings, pos=0.5):
    from bidsmgr.viz.commands.render import _FACING

    return render3d.clip_plane_uniforms(
        [ClipPlane(active=True, az=_FACING[f][0], el=_FACING[f][1], pos=pos) for f in facings])


def test_cropping_removes_what_any_plane_cuts_and_cutting_away_what_all_do():
    corners = np.array([[x, y, z] for x in (0.25, 0.75) for y in (0.25, 0.75)
                        for z in (0.25, 0.75)])
    planes = _planes("front", "top", "right")
    crop = render3d.cut_mask(corners, planes)
    away = render3d.cut_mask(corners, planes, cut_away=True)
    assert crop.sum() == 7, "cropping keeps one octant"
    assert away.sum() == 1 and away[-1], "cutting away removes the right-front-top one"


def test_planes_through_the_crosshair_keep_their_angle():
    clip = ClipPlane(active=True, az=0, el=0, pos=0.9)
    ((normal, depth, _t),) = render3d.clip_plane_uniforms([clip], cursor_tex=(0.5, 0.3, 0.5))
    assert np.allclose(normal, (0, 1, 0), atol=1e-6)
    assert depth == pytest.approx(-0.2), "the plane moved to the crosshair"
    tilted = ClipPlane(active=True, az=30, el=20)
    ((n2, d2, _t2),) = render3d.clip_plane_uniforms([tilted], cursor_tex=(0.6, 0.4, 0.7))
    assert d2 == pytest.approx(float(np.dot(n2, np.array([0.1, -0.1, 0.2]))))


def test_the_pick_cuts_away_as_the_render_does():
    eye = render3d.camera_eye(0.0, 0.0, 1.9, (0, 0, 0))
    view = render3d.look_at(eye, np.zeros(3, np.float32), np.array([0, 0, 1], np.float32))
    inv = np.linalg.inv(render3d.perspective(45.0, 1.0, 0.01, 20.0) @ view)
    ro, rd = render3d.ray_from_ndc(inv, 0.0, 0.0)
    vol = _cube()
    # Front AND top: the centre ray (z = 0.5 exactly) is on the boundary of
    # "top"; move the top plane up so the ray passes under it.
    planes = _planes("front") + _planes("top", pos=0.6)
    crop = render3d.pick_depth(vol, (0.5, 0.5, 0.5), ro, rd, lo=0.1, hi=0.42,
                               density=1.0, clips=planes)
    away = render3d.pick_depth(vol, (0.5, 0.5, 0.5), ro, rd, lo=0.1, hi=0.42,
                               density=1.0, clips=planes, cut_away=True)
    assert crop[1] < 0.55, "cropped: the front half is gone"
    assert away[1] == pytest.approx(24 / 32, abs=0.04), "cut away: nothing here is cut"


def test_overlay_parameters_are_shared_by_every_effect():
    from bidsmgr.viz.store import SceneStore

    store = SceneStore()
    store.run("render.param", key="seethrough", value=0)
    store.run("render.param", key="density", value=33)
    store.run("render.effect", effect="Realistic")
    values = render3d.values_for(store.scene.render)
    assert values["seethrough"] == 0, "switching effect kept the see-through"
    assert values["density"] != 33, "but not the other effect's look"
    store.run("render.reset_params")
    assert render3d.values_for(store.scene.render)["seethrough"] == 0
    store.run("render.reset_all")
    assert render3d.values_for(store.scene.render)["seethrough"] == \
        render3d.PARAM_BY_KEY["seethrough"].default


def test_the_presets_say_how_they_cut():
    from bidsmgr.viz.store import SceneStore

    store = SceneStore()
    store.run("clip.preset", preset="corner")
    rs = store.scene.render
    assert rs.cut_away and rs.cut_at_cursor and len(store.scene.clips) == 3
    store.run("clip.preset", preset="box")
    assert not rs.cut_away and not rs.cut_at_cursor
    assert len(render3d.clip_plane_uniforms(store.scene.clips)) == 6
    store.run("clip.preset", preset="none")
    assert render3d.clip_plane_uniforms(store.scene.clips) == []
    store.run("clip.toggle_at_cursor")
    assert store.scene.render.cut_at_cursor


# ---------------------------------------------------------------------------
# The 3-D view coloured by the 2-D colour map
# ---------------------------------------------------------------------------


def test_the_quantisation_states_its_range():
    vol = np.linspace(-50.0, 950.0, 1000, dtype=np.float32).reshape(10, 10, 10)
    u8, (lo, hi) = render3d.normalize_to_u8(vol)
    assert u8.dtype == np.uint8 and lo < hi
    # Level k stands for lo + k / 255 * (hi - lo).
    k = int(u8[5, 5, 5])
    assert lo + k / 255 * (hi - lo) == pytest.approx(float(vol[5, 5, 5]), abs=(hi - lo) / 255)


def test_the_transfer_table_is_the_slice_colouring():
    look = VolumeDisplay(colormap="hot", window=(100.0, 800.0), gamma=1.0,
                         threshold_mode="hide_below")
    table = render3d.transfer_lut(look, (0.0, 1000.0), (100.0, 800.0))
    assert table.shape == (256, 4) and table.dtype == np.uint8
    values = np.linspace(0.0, 1000.0, 256, dtype=np.float32)[None, :]
    expect = intensity.colorize(values, look, is_base=True, window=(100.0, 800.0))[0]
    assert np.array_equal(table, expect)
    assert table[0, 3] == 0, "below the window: hidden, as on the slices"
    assert table[-1, 3] == 255 and table[-1, 0] > 200


def test_an_inverted_map_reverses_the_table():
    look = VolumeDisplay(colormap="gray", window=(0.0, 1.0), gamma=1.0)
    plain = render3d.transfer_lut(look, (0.0, 1.0), (0.0, 1.0))
    inverted = render3d.transfer_lut(look.model_copy(update={"invert": True}), (0.0, 1.0), (0.0, 1.0))
    assert plain[0, 0] < plain[-1, 0] and inverted[0, 0] > inverted[-1, 0]


def test_the_option_is_a_command():
    from bidsmgr.viz.store import SceneStore

    store = SceneStore()
    assert store.scene.render.use_colormap
    store.run("render.use_colormap", value=False)
    assert not store.scene.render.use_colormap
    store.run("render.use_colormap")
    assert store.scene.render.use_colormap

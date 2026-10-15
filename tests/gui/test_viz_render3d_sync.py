"""What the slices show, the 3-D render shows: overlays, the crosshair, a
click that moves it, and up to six clip planes.

Under Qt's ``offscreen`` platform there is no OpenGL, so the GPU gate is
forced open and what is checked is the state the render is handed (the
overlay volume, the crosshair position, the clip planes) and the pick, which
runs on the CPU. The last section draws real pixels and skips, saying why,
where there is no OpenGL context (run it on the native platform).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("OpenGL")

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402

from bidsmgr.gui.viz import Viewer  # noqa: E402
from bidsmgr.viz import render3d  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def gpu(monkeypatch):
    from bidsmgr.gui.viz.canvases import render

    monkeypatch.setattr(render, "gpu_available", lambda: True)


def _save(path: Path, arr, affine=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, np.eye(4) if affine is None else affine), str(path))
    return path


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "Study"
    anat = root / "sub-01" / "anat"
    t1 = np.zeros((32, 32, 32), dtype=np.float32)
    t1[8:24, 8:24, 8:24] = 500.0
    _save(anat / "sub-01_T1w.nii.gz", t1)
    # A 2 mm mask over the front half of the cube.
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[4:12, 6:12, 4:12] = 1
    _save(anat / "sub-01_desc-front_mask.nii.gz", mask, np.diag([2.0, 2.0, 2.0, 1.0]))
    return root


def _t1(root):
    return root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"


def _mask(root):
    return root / "sub-01" / "anat" / "sub-01_desc-front_mask.nii.gz"


def _viewer(qtbot) -> Viewer:
    v = Viewer(kind="volume")
    qtbot.addWidget(v)
    v.resize(1000, 640)
    v.show()
    qtbot.waitExposed(v)
    return v


def _open(qtbot, v, path, root):
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(path, root)
    v.qstore.flush()
    return v


def _add(qtbot, v, path) -> str:
    with qtbot.waitSignal(v.overlay_added, timeout=20_000) as added:
        assert v.add_overlay(path)
    v.qstore.flush()
    return added.args[0]


def _in_3d(qtbot, v, mode="3d"):
    v.run("view.mode", mode=mode)
    v.qstore.flush()
    render = v.presenter.render_canvas
    qtbot.waitUntil(render.has_volume, timeout=10_000)
    return render


def _panel(v):
    from tests.gui.test_viz_render3d import _Panel3D

    return _Panel3D(v)


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------


class TestOverlaysInTheRender:
    def test_an_overlay_added_in_2d_is_in_the_render(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        v.run("view.mode", mode="multi")
        layer_id = _add(qtbot, v, _mask(ds))
        assert v.scene.layer(layer_id).in_3d
        render = _in_3d(qtbot, v)
        qtbot.waitUntil(render.has_overlay, timeout=10_000)
        vol = render._ov_pending                     # (z, y, x, 4), as uploaded
        # Sampled at the mask's own 2 mm, not the T1's 1 mm.
        assert vol.shape == (16, 16, 16, 4)
        # Drawn red where the mask is (world 8..23 x, 12..23 y, 8..23 z) ...
        assert vol[8, 9, 8, 3] > 0 and vol[8, 9, 8, 0] > vol[8, 9, 8, 1]
        # ... and nowhere else.
        assert vol[8, 2, 8, 3] == 0 and vol[1, 1, 1, 3] == 0

    def test_an_overlay_added_in_3d_arrives_without_leaving_it(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        render = _in_3d(qtbot, v)
        assert not render.has_overlay()
        _add(qtbot, v, _mask(ds))
        qtbot.waitUntil(render.has_overlay, timeout=10_000)

    def test_hidden_or_kept_out_of_3d_it_leaves_the_render(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        layer_id = _add(qtbot, v, _mask(ds))
        render = _in_3d(qtbot, v)
        qtbot.waitUntil(render.has_overlay, timeout=10_000)
        v.run("layer.set", layer=layer_id, in_3d=False)
        v.qstore.flush()
        qtbot.waitUntil(lambda: not render.has_overlay(), timeout=5000)
        v.run("layer.set", layer=layer_id, in_3d=True)
        v.qstore.flush()
        qtbot.waitUntil(render.has_overlay, timeout=10_000)
        v.run("layer.set", layer=layer_id, visible=False)
        v.qstore.flush()
        qtbot.waitUntil(lambda: not render.has_overlay(), timeout=5000)
        v.run("layer.set", layer=layer_id, visible=True)
        v.qstore.flush()
        qtbot.waitUntil(render.has_overlay, timeout=10_000)
        # The render's own switch turns every overlay off at once.
        v.run("render.param", key="layers", value=0)
        v.qstore.flush()
        qtbot.waitUntil(lambda: not render.has_overlay(), timeout=5000)

    def test_a_new_look_reaches_the_render(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        layer_id = _add(qtbot, v, _mask(ds))
        render = _in_3d(qtbot, v)
        qtbot.waitUntil(render.has_overlay, timeout=10_000)
        before = int(render._ov_pending[8, 9, 8, 3])
        v.run("layer.set", layer=layer_id, opacity=0.1)
        v.qstore.flush()
        qtbot.waitUntil(lambda: int(render._ov_pending[8, 9, 8, 3]) < before, timeout=10_000)
        v.run("layer.set", layer=layer_id, colormap="green")
        v.qstore.flush()
        qtbot.waitUntil(lambda: render._ov_pending[8, 9, 8, 1]
                        > render._ov_pending[8, 9, 8, 0], timeout=10_000)

    def test_removing_the_overlay_empties_the_render(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        layer_id = _add(qtbot, v, _mask(ds))
        render = _in_3d(qtbot, v)
        qtbot.waitUntil(render.has_overlay, timeout=10_000)
        v.run("layer.remove", layer=layer_id)
        v.qstore.flush()
        qtbot.waitUntil(lambda: not render.has_overlay(), timeout=5000)

    def test_the_layers_panel_offers_in_3d_for_an_overlay_only(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        layer_id = _add(qtbot, v, _mask(ds))
        v.presenter.set_inspector(True, remember=False)
        insp = v.presenter.inspector
        box = insp.section("overlay").control("in_3d")
        v.ctx.select_layer(layer_id)
        assert insp.section("overlay").isVisibleTo(insp)
        assert box.isVisibleTo(insp) and box.isChecked()
        box.setChecked(False)
        assert not v.scene.layer(layer_id).in_3d
        v.ctx.select_layer(v.scene.base_layer().id)
        assert not insp.section("overlay").isVisibleTo(insp)

    def test_a_saved_scene_keeps_an_overlay_out_of_3d(self, qtbot, ds, gpu):
        from bidsmgr.viz import scenes

        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        layer_id = _add(qtbot, v, _mask(ds))
        v.run("layer.set", layer=layer_id, in_3d=False)
        data = scenes.snapshot(v.scene, v.store.sources, ds, "flat")
        assert data["overlays"][0]["in_3d"] is False


# ---------------------------------------------------------------------------
# The crosshair and the pick
# ---------------------------------------------------------------------------


def _mouse(kind, x, y, button=Qt.MouseButton.LeftButton):
    return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button, button,
                       Qt.KeyboardModifier.NoModifier)


def _click(render, x, y, to=None):
    to = to or (x, y)
    render.mousePressEvent(_mouse(QMouseEvent.Type.MouseButtonPress, x, y))
    if to != (x, y):
        render.mouseMoveEvent(_mouse(QMouseEvent.Type.MouseMove, *to))
    render.mouseReleaseEvent(_mouse(QMouseEvent.Type.MouseButtonRelease, *to))


class TestCrosshairAndPick:
    def test_the_crosshair_is_where_the_cursor_is(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        render = _in_3d(qtbot, v)
        v.run("cursor.set_world", x=5.0, y=20.0, z=11.0)
        tex, width, _rgb = render._crosshair_uniforms()
        assert np.allclose(tex, (5.5 / 32, 20.5 / 32, 11.5 / 32))
        assert width > 0
        # The same width on screen whatever the zoom: twice as close, half
        # as wide in the box.
        v.run("render.camera", dist=v.scene.render.camera.dist / 2)
        assert render._crosshair_uniforms()[1] == pytest.approx(width / 2)
        v.trigger("view.crosshair")
        assert not v.scene.display.crosshair
        assert render._crosshair_uniforms()[0] is None

    def test_moving_the_cursor_redraws_without_uploading(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        render = _in_3d(qtbot, v)
        key = render._uploaded_key
        v.run("cursor.set_world", x=10.0, y=10.0, z=10.0)
        v.qstore.flush()
        assert render._wanted_key is None and render._uploaded_key == key
        assert not v.jobs.running("render-volume")

    def test_a_click_moves_the_crosshair_to_the_surface(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        render = _in_3d(qtbot, v)
        before = np.asarray(v.scene.cursor.world or (15.5, 15.5, 15.5))
        _click(render, render.width() / 2, render.height() / 2)
        world = np.asarray(v.scene.cursor.world)
        assert not np.allclose(world, before)
        # On the cube (voxels 8..23), on its surface, not in its middle.
        assert np.all(world >= 7.0) and np.all(world <= 24.0)
        assert np.max(np.minimum(np.abs(world - 7.5), np.abs(world - 23.5))) < 9
        assert np.min(np.minimum(np.abs(world - 7.5), np.abs(world - 23.5))) < 1.5

    def test_a_drag_turns_the_view_and_leaves_the_crosshair(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        render = _in_3d(qtbot, v)
        before = v.scene.cursor.world
        az = v.scene.render.camera.az
        cx, cy = render.width() / 2, render.height() / 2
        _click(render, cx, cy, to=(cx + 40, cy))
        assert v.scene.render.camera.az != az
        assert v.scene.cursor.world == before

    def test_a_click_on_nothing_leaves_the_crosshair(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        render = _in_3d(qtbot, v)
        before = v.scene.cursor.world
        _click(render, 2, 2)
        assert v.scene.cursor.world == before

    def test_a_cut_lets_the_click_through(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        render = _in_3d(qtbot, v)
        cx, cy = render.width() / 2, render.height() / 2
        _click(render, cx, cy)
        near = np.asarray(v.scene.cursor.world)
        # Cut along the camera's line of sight, through the middle.
        cam = v.scene.render.camera
        v.run("clip.set", active=True, az=float(np.degrees(cam.az)) % 360,
              el=float(np.degrees(cam.el)), pos=0.5)
        v.run("cursor.center")
        _click(render, cx, cy)
        far = np.asarray(v.scene.cursor.world)
        assert np.linalg.norm(far - near) > 3.0


# ---------------------------------------------------------------------------
# Six clip planes
# ---------------------------------------------------------------------------


class TestClipPlanes:
    def test_the_panel_edits_the_chosen_plane(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        _in_3d(qtbot, v)
        panel = _panel(v)
        panel.clip_choice.setCurrentIndex(2)
        assert v.ctx.active_clip == 2
        panel.clip_box.setChecked(True)
        panel.clip_el.setValue(45)
        v.qstore.flush()
        clips = v.scene.clips
        assert len(clips) == 3 and clips[2].active and clips[2].el == 45
        assert not clips[0].active
        panel.clip_choice.setCurrentIndex(0)
        assert not panel.clip_box.isChecked() and not panel.clip_el.isEnabled()

    def test_the_keys_act_on_the_chosen_plane(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        _in_3d(qtbot, v)
        _panel(v).clip_choice.setCurrentIndex(1)
        v.trigger("clip.toggle")
        v.trigger("clip.axial")
        v.qstore.flush()
        assert v.scene.clips[1].active and v.scene.clips[1].el == 90
        assert not v.scene.clips[0].active
        assert v.action("clip.toggle").isChecked()

    def test_an_arrangement_cuts_with_every_plane(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        _in_3d(qtbot, v)
        panel = _panel(v)
        panel.clip_preset.setCurrentIndex(panel.clip_preset.findData("box"))
        v.qstore.flush()
        assert len(render3d.clip_plane_uniforms(v.scene.clips)) == 6
        panel.clip_preset.setCurrentIndex(panel.clip_preset.findData("wedge"))
        assert len(render3d.clip_plane_uniforms(v.scene.clips)) == 2
        panel.clip_preset.setCurrentIndex(panel.clip_preset.findData("none"))
        assert render3d.clip_plane_uniforms(v.scene.clips) == []


# ---------------------------------------------------------------------------
# Real pixels, on a real OpenGL context
# ---------------------------------------------------------------------------


@pytest.fixture
def gl_context(qapp):
    from PyQt6.QtGui import QOffscreenSurface, QOpenGLContext

    from bidsmgr.gui.viz.canvases.render import request_gl_format

    surface = QOffscreenSurface()
    surface.setFormat(request_gl_format())
    surface.create()
    if not surface.isValid():
        pytest.skip("the Qt platform plugin cannot create an offscreen surface")
    context = QOpenGLContext()
    context.setFormat(request_gl_format())
    if not context.create() or not context.makeCurrent(surface):
        pytest.skip("no OpenGL 3.3 core context on this machine")
    context.doneCurrent()
    yield context


@pytest.fixture
def shell(tmp_path: Path) -> Path:
    """A closed box (a skull) with a mask inside it (a structure)."""
    root = tmp_path / "Shell"
    anat = root / "sub-01" / "anat"
    vol = np.zeros((40, 40, 40), dtype=np.float32)
    vol[4:36, 4:36, 4:36] = 1000.0
    vol[7:33, 7:33, 7:33] = 0.0
    _save(anat / "sub-01_T1w.nii.gz", vol)
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[14:26, 14:26, 14:26] = 1
    _save(anat / "sub-01_mask.nii.gz", mask)
    return root


def _pixels(render):
    image = render.grab_image()
    assert not image.isNull()
    from PyQt6.QtGui import QImage

    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    return np.frombuffer(ptr, np.uint8).reshape(image.height(), image.width(), 4)[..., :3].astype(int)


def _red(px) -> int:
    return int(np.count_nonzero((px[..., 0] > px[..., 1] + 60) & (px[..., 0] > px[..., 2] + 60)))


def _cyan(px) -> int:
    return int(np.count_nonzero((px[..., 2] > px[..., 0] + 50) & (px[..., 1] > px[..., 0] + 30)))


class TestRealPixels:
    def _ready(self, qtbot, shell):
        v = _open(qtbot, _viewer(qtbot), shell / "sub-01" / "anat" / "sub-01_T1w.nii.gz", shell)
        layer_id = _add(qtbot, v, shell / "sub-01" / "anat" / "sub-01_mask.nii.gz")
        v.run("layer.set", layer=layer_id, opacity=1.0)
        render = _in_3d(qtbot, v)
        qtbot.waitUntil(render.gl_ok, timeout=5000)
        qtbot.waitUntil(render.has_overlay, timeout=10_000)
        qtbot.waitUntil(lambda: bool(render._ov_tex), timeout=10_000)
        v.trigger("view.crosshair")          # off: these count overlay pixels
        return v, render

    def test_an_overlay_inside_is_hidden_by_depth_and_shown_by_a_cut(
            self, qtbot, gl_context, shell):
        v, render = self._ready(qtbot, shell)
        v.run("render.param", key="seethrough", value=0)
        v.qstore.flush()
        hidden = _red(_pixels(render))
        cam = v.scene.render.camera
        v.run("clip.set", active=True, az=float(np.degrees(cam.az)) % 360,
              el=float(np.degrees(cam.el)), pos=0.5)
        v.qstore.flush()
        cut = _red(_pixels(render))
        assert hidden == 0, "a box hides what is inside it"
        assert cut > 200, "cut open, the structure inside shows in its colour"

    def test_see_through_lays_the_inside_over_the_surface(self, qtbot, gl_context, shell):
        v, render = self._ready(qtbot, shell)
        v.run("render.param", key="seethrough", value=0)
        v.qstore.flush()
        none = _red(_pixels(render))
        v.run("render.param", key="seethrough", value=100)
        v.qstore.flush()
        full = _red(_pixels(render))
        assert none == 0 and full > 200

    def test_every_effect_draws_the_overlay(self, qtbot, gl_context, shell):
        v, render = self._ready(qtbot, shell)
        for effect in ("Standard", "MIP", "X-ray", "Glass", "Jelly"):
            v.run("render.effect", effect=effect)
            v.run("render.param", key="seethrough", value=100)
            v.qstore.flush()
            assert _red(_pixels(render)) > 100, effect

    def test_the_crosshair_is_drawn(self, qtbot, gl_context, shell):
        v, render = self._ready(qtbot, shell)
        v.run("layer.set", layer=v.scene.layers[-1].id, in_3d=False)
        v.qstore.flush()
        qtbot.waitUntil(lambda: not render.has_overlay(), timeout=5000)
        without = _cyan(_pixels(render))
        v.trigger("view.crosshair")
        v.qstore.flush()
        with_it = _cyan(_pixels(render))
        assert with_it > without + 50


# ---------------------------------------------------------------------------
# The cut follows the crosshair; linked renders share how they cut
# ---------------------------------------------------------------------------


class TestCutAtTheCrosshair:
    def test_the_corner_key_cuts_at_the_crosshair_and_follows_it(self, qtbot, ds, gpu):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        render = _in_3d(qtbot, v)
        v.run("cursor.set_world", x=10.0, y=12.0, z=20.0)
        v.trigger("clip.corner")
        v.qstore.flush()
        assert v.scene.render.cut_away and v.scene.render.cut_at_cursor
        assert v.action("clip.at_cursor").isChecked()
        planes = render._planes()
        tex = render.cursor_texcoord()
        for normal, depth, _thick in planes:
            assert depth == pytest.approx(float(np.dot(normal, tex - 0.5)), abs=1e-6)
        v.run("cursor.set_world", x=20.0, y=5.0, z=8.0)
        moved = render._planes()
        assert [d for _n, d, _t in moved] != [d for _n, d, _t in planes]
        panel = _panel(v)
        assert panel.cut_at_cursor_box.isChecked() and panel.cut_away_box.isChecked()
        assert not panel.clip_pos.isEnabled(), "the crosshair sets the depth"
        v.trigger("clip.at_cursor")
        v.qstore.flush()
        assert not v.scene.render.cut_at_cursor and not panel.cut_at_cursor_box.isChecked()

    def test_linked_renders_cut_alike(self, qtbot, ds, gpu):
        from bidsmgr.gui.viz.link import link

        a = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        b = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        lk = link(a, b)
        a.run("clip.preset", preset="corner")
        a.run("render.param", key="seethrough", value=70)
        a.qstore.flush()
        lk.sync_now(a)
        assert b.scene.render.cut_away and b.scene.render.cut_at_cursor
        assert render3d.values_for(b.scene.render)["seethrough"] == 70
        assert len(b.scene.clips) == 3


class TestColourMapIn3D:
    def test_the_render_takes_the_colour_map(self, qtbot, gl_context, shell):
        v = _open(qtbot, _viewer(qtbot), shell / "sub-01" / "anat" / "sub-01_T1w.nii.gz", shell)
        render = _in_3d(qtbot, v)
        qtbot.waitUntil(render.gl_ok, timeout=5000)
        v.trigger("view.crosshair")
        v.run("layer.set", colormap="hot", window=(0.0, 1000.0))
        v.qstore.flush()
        warm = _red(_pixels(render))
        v.run("render.use_colormap", value=False)
        v.qstore.flush()
        plain = _red(_pixels(render))
        assert warm > 200, "hot: a red-to-yellow render"
        assert plain == 0, "the effect's own grey shading"

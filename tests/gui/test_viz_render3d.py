"""The 3-D view: its wiring, its controls, and (where a GPU exists) its pixels.

Replaces the tests of the old ``nifti_gl_view`` and the 3-D half of the old
NIfTI pane. Under Qt's ``offscreen`` platform there is no OpenGL, so the GPU
gate is forced open for the wiring tests: what is under test is what the gate
protects, and there is no point testing it against a gate that is shut. The
last section asks for a REAL context and skips, saying why, where none exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("OpenGL")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent, QWheelEvent  # noqa: E402

from bidsmgr.gui.viz import Viewer  # noqa: E402
from bidsmgr.viz import render3d  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def gpu(monkeypatch):
    from bidsmgr.gui.viz.canvases import render

    monkeypatch.setattr(render, "gpu_available", lambda: True)


def _write(path: Path, arr, affine=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, np.eye(4) if affine is None else affine), str(path))
    return path


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    _write(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz",
           np.arange(10 * 12 * 8, dtype=np.float32).reshape(10, 12, 8))
    _write(root / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz",
           np.random.default_rng(0).random((6, 6, 4, 3), dtype=np.float32))
    return root


def _t1(root):
    return root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"


def _bold(root):
    return root / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz"


def _viewer(qtbot) -> Viewer:
    viewer = Viewer(kind="volume")
    qtbot.addWidget(viewer)
    viewer.resize(1000, 640)
    viewer.show()
    qtbot.waitExposed(viewer)
    return viewer


def _open(qtbot, viewer, path, root=None):
    with qtbot.waitSignal(viewer.loaded, timeout=20_000):
        viewer.set_file(path, root)
    viewer.qstore.flush()
    return viewer


def _in_3d(qtbot, viewer, mode="3d"):
    viewer.run("view.mode", mode=mode)
    viewer.qstore.flush()
    render = viewer.presenter.render_canvas
    qtbot.waitUntil(render.has_volume, timeout=10_000)
    return render


class _Units:
    """A number control seen in the slider units the presets are written in."""

    def __init__(self, control, scale: float) -> None:
        self.control = control
        self.scale = scale

    def value(self) -> int:
        return int(round(self.control.value() * self.scale))

    def setValue(self, v) -> None:  # noqa: N802 - the old QSlider spelling
        self.control.type_value(v / self.scale)

    def isEnabled(self) -> bool:  # noqa: N802
        return self.control.isEnabled()


class _Row:
    def __init__(self, section, key: str) -> None:
        self.section, self.key = section, key

    def isHidden(self) -> bool:  # noqa: N802
        return self.section.isHidden() or not self.section.row_shown(self.key)


class _Panel3D:
    """The 3-D sections of the controls column, by the names these tests
    used for the 3-D panel they replaced."""

    def __init__(self, viewer) -> None:
        insp = viewer.presenter.inspector
        assert insp is not None, "the controls column is open"
        self.insp = insp
        look = insp.section("3d.look")
        self.effect_combo = look.effect
        self._widgets, self._rows = {}, {}
        for p in render3d.PARAMS:
            sec = insp.section(f"3d.{p.group}")
            ctl = sec.control(p.key)
            self._rows[p.key] = _Row(sec, p.key)
            self._widgets[p.key] = _Units(ctl, p.scale or 1.0) if p.kind == "slider" else ctl
        clip = insp.section("3d.clip")
        self.clip_box = clip.enabled
        self.clip_choice = clip.choice
        self.clip_preset = clip.preset
        self.clip_az = _Units(clip.az, 1)
        self.clip_el = _Units(clip.el, 1)
        self.clip_pos = _Units(clip.pos, 10)       # percent -> the old 0..1000
        self.cut_at_cursor_box = clip.at_cursor
        self.cut_away_box = clip.cut_away

    def isVisible(self) -> bool:  # noqa: N802
        return self.insp.section("3d.look").isVisible()


def _panel(viewer) -> _Panel3D:
    return _Panel3D(viewer)


# ---------------------------------------------------------------------------
# The gate and the format
# ---------------------------------------------------------------------------


def test_the_gl_format_is_33_core(qapp) -> None:
    from PyQt6.QtGui import QSurfaceFormat

    from bidsmgr.gui.viz.canvases.render import request_gl_format

    fmt = request_gl_format()
    assert (fmt.majorVersion(), fmt.minorVersion()) == (3, 3)
    assert fmt.profile() == QSurfaceFormat.OpenGLContextProfile.CoreProfile


def test_the_gate_answers_a_bool_and_the_cube_builds(qapp) -> None:
    from bidsmgr.gui.viz.canvases.render import gpu_available

    assert isinstance(gpu_available(), bool)
    assert render3d.cube_geometry().shape == (36, 8)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_3d_is_offered_once_a_volume_is_open(qtbot, ds, gpu) -> None:
    viewer = _viewer(qtbot)
    assert not viewer.action("view.3d").isEnabled()
    _open(qtbot, viewer, _t1(ds), ds)
    assert viewer.action("view.3d").isEnabled()
    _open(qtbot, viewer, _bold(ds), ds)
    assert viewer.action("view.3d").isEnabled()


def test_the_first_volume_opens_planes_with_the_render(qtbot, ds, gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    assert viewer.scene.mode == "combo"


def test_the_3d_button_shows_the_render_and_its_panel(qtbot, ds, gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    viewer.run("view.mode", mode="single")
    viewer.action("view.3d").trigger()
    viewer.qstore.flush()
    render = viewer.presenter.render_canvas
    qtbot.waitUntil(render.has_volume, timeout=10_000)
    assert viewer.scene.mode == "3d"
    assert viewer.canvases("render") == [render]
    assert viewer.canvases("slice") == []
    assert _panel(viewer).isVisible(), "the 3-D sections of the controls column"
    # Stepping slices means nothing with no slice on screen.
    assert not viewer.action("slice.next").isEnabled()
    # The planes stay one click away: they leave 3-D.
    assert viewer.action("view.multi").isEnabled()
    viewer.action("view.3d").trigger()
    viewer.qstore.flush()
    assert viewer.scene.mode == "single"
    assert viewer.canvases("render") == []


def test_one_layout_at_a_time(qtbot, ds, gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    for action, mode in (("view.multi", "multi"), ("view.3d", "3d"),
                         ("view.combo", "combo"), ("view.multi", "multi")):
        viewer.action(action).trigger()
        viewer.qstore.flush()
        assert viewer.scene.mode == mode
        assert viewer.presenter.layout_button.text() == {
            "multi": "Multi-planar", "3d": "3-D", "combo": "Planes + 3-D"}[mode]
        checked = {a for a in ("view.multi", "view.3d", "view.combo")
                   if viewer.action(a).isChecked()}
        assert checked == {action}


def test_combo_keeps_three_planes_and_the_render(qtbot, ds, gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    render = _in_3d(qtbot, viewer, "combo")
    assert sorted(c.plane for c in viewer.canvases("slice")) == ["axial", "coronal", "sagittal"]
    assert viewer.canvases("render") == [render]
    assert viewer.action("slice.next").isEnabled()


def test_a_plane_key_leaves_3d(qtbot, ds, gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    _in_3d(qtbot, viewer)
    viewer.trigger("view.sagittal")
    viewer.qstore.flush()
    assert (viewer.scene.mode, viewer.scene.plane) == ("single", "sagittal")


def test_a_new_volume_reaches_the_render(qtbot, ds, gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
    render = _in_3d(qtbot, viewer)
    first = render._uploaded_key
    viewer.trigger("frame.next")
    viewer.qstore.flush()
    qtbot.waitUntil(lambda: render._uploaded_key not in (None, first), timeout=10_000)
    assert render._uploaded_key[1] == 1


def test_closing_the_file_clears_the_render(qtbot, ds, gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    render = _in_3d(qtbot, viewer)
    viewer.set_file(None, None)
    assert not render.has_volume()
    assert not viewer.action("view.3d").isEnabled()


def test_ras_and_radiological_reach_the_render(qtbot, ds, gpu) -> None:
    """The render mirrors with the slices: one extra L/R mirror, constant,
    because a look-at view is the mirror image of a neurological slice."""
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    render = _in_3d(qtbot, viewer)
    assert render.display_flip() == (-1.0, 1.0, 1.0)
    viewer.trigger("view.radiological")
    assert render.display_flip() == (1.0, 1.0, 1.0)


def test_a_reversed_file_mirrors_only_with_ras_off(qtbot, tmp_path, gpu) -> None:
    path = _write(tmp_path / "sub-01_T1w.nii.gz", np.ones((6, 6, 6), np.float32),
                  affine=np.diag([2.0, 2.0, -2.0, 1.0]))
    viewer = _open(qtbot, _viewer(qtbot), path, tmp_path)
    render = _in_3d(qtbot, viewer)
    assert render.display_flip() == (-1.0, 1.0, 1.0)
    viewer.trigger("view.ras")
    assert render.display_flip() == (-1.0, 1.0, -1.0)


def test_colour_fa_renders_as_colour(qtbot, tmp_path, gpu) -> None:
    rgb = np.zeros((8, 9, 10), dtype=[("R", "u1"), ("G", "u1"), ("B", "u1")])
    rgb["R"][2:6] = 220
    rgb["G"][:, 3:7] = 180
    path = _write(tmp_path / "sub-01_colFA.nii.gz", rgb, np.diag([2.0, 2.0, 2.0, 1.0]))
    viewer = _open(qtbot, _viewer(qtbot), path, tmp_path)
    assert viewer.action("view.3d").isEnabled()
    render = _in_3d(qtbot, viewer)
    assert render._is_rgb


def test_the_render_background_is_black(qtbot, ds, gpu) -> None:
    """The same surround as the slices, in both themes."""
    from bidsmgr.gui import theme_manager
    from bidsmgr.gui.viz.bridge import ThemeHub

    try:
        ThemeHub.instance().publish(theme_manager.LIGHT)
        assert ThemeHub.instance().theme.background == "#000000"
    finally:
        ThemeHub.instance().publish(theme_manager.DARK)


def test_the_controls_column_is_opaque(qtbot, ds, gpu) -> None:
    """The black image canvas showed through the controls' scroll gutter as
    a stripe: every layer of the column must be styled by name."""
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    _in_3d(qtbot, viewer)
    insp = viewer.presenter.inspector
    assert insp.objectName() == "viz-inspector"
    scroll = viewer.presenter.side
    assert scroll.objectName() == "viz-side-scroll"
    assert scroll.viewport().objectName() == "viz-side-viewport"
    qss = (Path(__file__).resolve().parents[2] / "bidsmgr" / "gui" / "theme.qss").read_text()
    for name in ("viz-side-scroll", "viz-side-viewport", "viz-inspector", "viz-section",
                 "viz-section-body"):
        assert f"#{name}" in qss, f"{name} has no stylesheet rule"
    for section in insp.sections.values():
        assert section.testAttribute(Qt.WidgetAttribute.WA_StyledBackground), section.key


# ---------------------------------------------------------------------------
# The control panel runs commands; the scene is the state
# ---------------------------------------------------------------------------


@pytest.fixture
def panel3d(qtbot, ds, gpu):
    viewer = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
    _in_3d(qtbot, viewer)
    return viewer, _panel(viewer)


def _effective(viewer) -> dict:
    rs = viewer.scene.render
    return render3d.effective(rs.effect, rs.params.get(rs.effect))


def test_the_effect_selector(panel3d) -> None:
    viewer, panel = panel3d
    panel.effect_combo.setCurrentText("MIP")
    assert viewer.scene.render.effect == "MIP"
    assert render3d.EFFECT_FX["MIP"] == 4


def test_presets_apply_their_values(panel3d) -> None:
    viewer, panel = panel3d
    for name in ("Jelly", "Skull", "Juicy shiny", "Juicy shiny 2", "Realistic"):
        panel.effect_combo.setCurrentText(name)
        viewer.qstore.flush()
        values = _effective(viewer)
        for key, value in render3d.EFFECT_PRESET[name].items():
            assert values[key] == pytest.approx(float(value)), (name, key)
        assert panel._widgets["density"].value() == render3d.EFFECT_PRESET[name]["density"]


def test_thresholds_cannot_invert(panel3d) -> None:
    """lo dragged above hi used to paint the whole box black."""
    viewer, panel = panel3d
    panel._widgets["hi"].setValue(300)
    panel._widgets["lo"].setValue(900)
    viewer.qstore.flush()
    values = _effective(viewer)
    assert values["hi"] >= values["lo"] + render3d.THRESH_GAP
    assert panel._widgets["hi"].value() > panel._widgets["lo"].value()


def test_only_the_parameters_an_effect_uses_are_shown(panel3d) -> None:
    viewer, panel = panel3d
    for name in ("Glass", "MIP", "Opacity peeling"):
        panel.effect_combo.setCurrentText(name)
        viewer.qstore.flush()
        used = render3d.EFFECT_PARAMS[name]
        for key, row in panel._rows.items():
            assert row.isHidden() == (key not in used), (name, key)


def test_the_cut_face_slice_is_on_for_surface_effects_only(panel3d) -> None:
    viewer, panel = panel3d
    for name in render3d.EFFECTS:
        panel.effect_combo.setCurrentText(name)
        viewer.qstore.flush()
        on = _effective(viewer)["overlay"] >= 0.5
        assert on == (name in render3d.SLICE_DEFAULT_ON), name


def test_each_effect_keeps_its_own_parameters(panel3d) -> None:
    viewer, panel = panel3d
    density = panel._widgets["density"]

    def choose(name):
        panel.effect_combo.setCurrentText(name)
        viewer.qstore.flush()

    choose("Jelly")
    density.setValue(88)
    choose("Skull")
    assert density.value() == render3d.EFFECT_PRESET["Skull"]["density"]
    density.setValue(11)
    choose("Jelly")
    assert density.value() == 88
    viewer.run("render.reset_params")
    viewer.qstore.flush()
    assert density.value() == render3d.EFFECT_PRESET["Jelly"]["density"]
    choose("Skull")
    assert density.value() == 11
    viewer.run("render.reset_all")
    viewer.qstore.flush()
    assert density.value() == render3d.EFFECT_PRESET["Skull"]["density"]


def test_reset_leaves_the_clip_and_other_effects_alone(panel3d) -> None:
    viewer, panel = panel3d
    panel.effect_combo.setCurrentText("Glass")
    panel.clip_box.setChecked(True)
    panel._widgets["specular"].setValue(95)
    viewer.run("render.param", key="brighten", value=260, effect="Matte")
    viewer.run("render.reset_params")
    viewer.qstore.flush()
    assert viewer.scene.render.effect == "Glass"
    assert viewer.scene.clips[0].active
    assert _effective(viewer)["specular"] == render3d.PARAM_BY_KEY["specular"].default
    assert viewer.scene.render.params["Matte"]["brighten"] == 260


def test_quality_starts_at_half_for_every_effect(panel3d) -> None:
    viewer, panel = panel3d
    assert render3d.QUALITY_DEFAULT == render3d.QUALITY_MAX // 2
    assert not any("quality" in p for p in render3d.EFFECT_PRESET.values())
    for name in render3d.EFFECTS:
        panel.effect_combo.setCurrentText(name)
        viewer.qstore.flush()
        assert panel._widgets["quality"].value() == render3d.QUALITY_DEFAULT


def test_every_slider_shows_its_value(panel3d) -> None:
    """The numbers used to hide behind a "Show values" box, in internal
    units: now every slider has its number beside it, in the shader's."""
    viewer, panel = panel3d
    panel.effect_combo.setCurrentText("Jelly")
    viewer.qstore.flush()
    density = panel._widgets["density"].control
    assert density.spin.value() == pytest.approx(
        render3d.EFFECT_PRESET["Jelly"]["density"] / 100.0)
    density.type_value(0.5)
    viewer.qstore.flush()
    assert render3d.values_for(viewer.scene.render)["density"] == pytest.approx(50)
    assert density.slider.value() == density._to_pos(0.5)


def test_a_drag_on_a_3d_slider_is_one_undo_step(panel3d) -> None:
    viewer, panel = panel3d
    density = panel._widgets["density"].control
    before = render3d.values_for(viewer.scene.render)["density"]
    density.pressed.emit()
    for pos in (300, 400, 500, 600):
        density.slider.setValue(pos)
    density.released.emit()
    viewer.qstore.flush()
    assert render3d.values_for(viewer.scene.render)["density"] != before
    viewer.store.undo()
    assert render3d.values_for(viewer.scene.render)["density"] == before


def test_the_effect_order_is_the_menu_order(qapp) -> None:
    order = render3d.EFFECTS
    assert order.index("Juicy shiny") == order.index("Matte") + 1
    assert order.index("Juicy shiny 2") == order.index("Juicy shiny") + 1
    assert order.index("Realistic") == order.index("Juicy shiny 2") + 1
    assert order.index("Jelly") == order.index("X-ray") + 1
    assert order.index("Skull") == order.index("Jelly") + 1
    assert render3d.EFFECT_FX["Realistic"] not in (
        render3d.EFFECT_FX["Standard"], render3d.EFFECT_FX["Matte"])


# ---------------------------------------------------------------------------
# The clip plane: panel, keys and gestures all reach one state
# ---------------------------------------------------------------------------


def test_the_clip_controls(panel3d) -> None:
    viewer, panel = panel3d
    assert not viewer.scene.clips[0].active
    assert not panel.clip_az.isEnabled()
    panel.clip_box.setChecked(True)
    viewer.qstore.flush()
    assert viewer.scene.clips[0].active and panel.clip_az.isEnabled()
    panel.clip_el.setValue(90)
    clip = viewer.scene.clips[0]
    assert render3d.clip_normal_from(clip.az, clip.el)[2] > 0.99
    panel.clip_el.setValue(0)
    panel.clip_az.setValue(0)
    clip = viewer.scene.clips[0]
    assert render3d.clip_normal_from(clip.az, clip.el)[1] == pytest.approx(1.0, abs=1e-3)
    panel.clip_box.setChecked(False)
    viewer.qstore.flush()
    assert not viewer.scene.clips[0].active and not panel.clip_az.isEnabled()


def test_the_slicer_keys(panel3d) -> None:
    viewer, panel = panel3d
    viewer.trigger("clip.toggle")
    viewer.qstore.flush()
    assert viewer.scene.clips[0].active and panel.clip_box.isChecked()
    viewer.trigger("clip.axial")
    viewer.qstore.flush()
    assert (panel.clip_az.value(), panel.clip_el.value()) == (0, 90)
    viewer.trigger("clip.sagittal")
    clip = viewer.scene.clips[0]
    assert abs(render3d.clip_normal_from(clip.az, clip.el)[0]) > 0.99
    viewer.trigger("clip.coronal")
    clip = viewer.scene.clips[0]
    assert abs(render3d.clip_normal_from(clip.az, clip.el)[1]) > 0.99
    viewer.trigger("clip.invert")
    assert viewer.scene.clips[0].flip
    viewer.run("clip.tilt", d_az=20, d_el=-5)
    viewer.run("clip.nudge", delta=0.05)
    viewer.qstore.flush()
    clip = viewer.scene.clips[0]
    assert panel.clip_az.value() == int(round(clip.az))
    assert panel.clip_pos.value() == int(round(clip.pos * 1000))
    viewer.trigger("clip.toggle")
    assert not viewer.scene.clips[0].active


def test_the_slicer_keys_are_the_documented_ones(panel3d) -> None:
    viewer, _panel_ = panel3d
    keys = {a: viewer.action_manager.keys_for(a)
            for a in ("clip.toggle", "clip.axial", "clip.sagittal", "clip.coronal", "clip.invert")}
    assert keys == {"clip.toggle": ["Shift+Z"], "clip.axial": ["Shift+A"],
                    "clip.sagittal": ["Shift+S"], "clip.coronal": ["Shift+C"],
                    "clip.invert": ["Shift+X"]}


# ---------------------------------------------------------------------------
# Gestures on the render
# ---------------------------------------------------------------------------


def _mouse(kind, x, y, button=Qt.MouseButton.LeftButton, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button, button, mods)


def _wheel(ay, ax, shift=False):
    return QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(ax, ay),
                       Qt.MouseButton.NoButton,
                       Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier,
                       Qt.ScrollPhase.NoScrollPhase, False)


def test_dragging_right_turns_the_head_right(panel3d) -> None:
    """Left-right drag used to be inverted."""
    viewer, _ = panel3d
    render = viewer.presenter.render_canvas
    az0 = viewer.scene.render.camera.az
    render.mousePressEvent(_mouse(QMouseEvent.Type.MouseButtonPress, 0, 0))
    render.mouseMoveEvent(_mouse(QMouseEvent.Type.MouseMove, 10, 0))
    render.mouseReleaseEvent(_mouse(QMouseEvent.Type.MouseButtonRelease, 10, 0))
    assert viewer.scene.render.camera.az > az0


def test_a_right_drag_pans_and_reset_recentres(panel3d) -> None:
    viewer, _ = panel3d
    render = viewer.presenter.render_canvas
    right = Qt.MouseButton.RightButton
    render.mousePressEvent(_mouse(QMouseEvent.Type.MouseButtonPress, 0, 0, right))
    render.mouseMoveEvent(_mouse(QMouseEvent.Type.MouseMove, 30, 10, right))
    assert not np.allclose(viewer.scene.render.camera.target, 0.0)
    viewer.trigger("render.reset_view")
    assert np.allclose(viewer.scene.render.camera.target, 0.0)


def test_the_wheel_zooms_and_with_a_cut_moves_it(panel3d) -> None:
    """Shift+wheel arrives as a horizontal scroll on X11 and macOS; both move
    an active cut, and without one a horizontal scroll still zooms."""
    viewer, _ = panel3d
    render = viewer.presenter.render_canvas
    d0 = viewer.scene.render.camera.dist
    render.wheelEvent(_wheel(120, 0))
    assert viewer.scene.render.camera.dist / d0 == pytest.approx(0.9)
    viewer.run("clip.set", active=True)
    p = viewer.scene.clips[0].pos
    render.wheelEvent(_wheel(120, 0, shift=True))
    assert viewer.scene.clips[0].pos == pytest.approx(p + 0.02)
    p = viewer.scene.clips[0].pos
    render.wheelEvent(_wheel(0, 120))
    assert viewer.scene.clips[0].pos == pytest.approx(p + 0.02)
    viewer.run("clip.set", active=False)
    d0 = viewer.scene.render.camera.dist
    render.wheelEvent(_wheel(0, 120))
    assert viewer.scene.render.camera.dist != d0


def test_shift_drag_tilts_an_active_cut_only(panel3d) -> None:
    viewer, _ = panel3d
    render = viewer.presenter.render_canvas
    shift = Qt.KeyboardModifier.ShiftModifier
    az0 = viewer.scene.render.camera.az
    render.mousePressEvent(_mouse(QMouseEvent.Type.MouseButtonPress, 0, 0, mods=shift))
    render.mouseMoveEvent(_mouse(QMouseEvent.Type.MouseMove, 10, 0, mods=shift))
    render.mouseReleaseEvent(_mouse(QMouseEvent.Type.MouseButtonRelease, 10, 0))
    assert viewer.scene.render.camera.az != az0        # no cut: it orbits
    viewer.run("clip.set", active=True)
    caz = viewer.scene.clips[0].az
    render.mousePressEvent(_mouse(QMouseEvent.Type.MouseButtonPress, 0, 0, mods=shift))
    render.mouseMoveEvent(_mouse(QMouseEvent.Type.MouseMove, 10, 0, mods=shift))
    render.mouseReleaseEvent(_mouse(QMouseEvent.Type.MouseButtonRelease, 10, 0))
    assert viewer.scene.clips[0].az != caz


# ---------------------------------------------------------------------------
# Upload preparation (worker side, no GL)
# ---------------------------------------------------------------------------


def test_colour_volumes_keep_one_window_across_channels(qapp) -> None:
    """Windowing channels separately would recolour colour-FA's directions."""
    vol = np.zeros((2, 2, 2, 3), np.float32)
    vol[..., 0] = 1.0
    vol[..., 1] = 0.5
    u8 = render3d.rgb_to_u8(vol)
    assert u8[..., 0].max() == 255
    assert abs(int(u8[..., 1].max()) - 128) <= 1
    assert u8[..., 2].max() == 0


def test_the_upload_is_in_ras_order(qapp, tmp_path) -> None:
    """The texture's axes are the render box's: a reversed file is turned."""
    from bidsmgr.gui.viz.canvases.render import prepare_upload
    from bidsmgr.viz.data.volume import open_volume

    data = np.zeros((4, 5, 6), np.float32)
    data[0, 0, 0] = 10.0
    path = _write(tmp_path / "x.nii.gz", data, np.diag([-1.0, 1.0, 1.0, 1.0]))
    src = open_volume(path)
    src.stream()
    u8, spacing, rng = prepare_upload(src.raw_frame(0), src, False)
    assert u8.shape == (4, 5, 6) and spacing == (1.0, 1.0, 1.0)
    # Voxel i=0 is the subject's RIGHT-most (x grows left on disk): RAS puts it last.
    assert u8[-1, 0, 0] == u8.max() and u8[0, 0, 0] < u8.max()


def test_the_cube_letters_follow_the_mirror(qapp) -> None:
    from bidsmgr.gui.viz.canvases.render import make_cube_atlas

    a0 = make_cube_atlas(16, (1.0, 1.0, 1.0))
    a1 = make_cube_atlas(16, (-1.0, 1.0, 1.0))
    assert a0.shape == a1.shape and not np.array_equal(a0, a1)


# ---------------------------------------------------------------------------
# Real pixels, on a real OpenGL context
# ---------------------------------------------------------------------------


@pytest.fixture
def gl_context(qapp):
    """A real OpenGL 3.3 core context, or a skip saying why not."""
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
    yield context
    context.doneCurrent()


def test_the_gate_agrees_with_reality(gl_context) -> None:
    from bidsmgr.gui.viz.canvases.render import gpu_available

    assert gpu_available()


def test_a_volume_renders_something(qtbot, gl_context, tmp_path) -> None:
    """A shader that fails to compile fails silently; only pixels prove it ran."""
    volume = np.zeros((32, 32, 32), dtype=np.float32)
    volume[8:24, 8:24, 8:24] = 1.0
    path = _write(tmp_path / "cube.nii.gz", volume)
    viewer = _open(qtbot, _viewer(qtbot), path, tmp_path)
    render = _in_3d(qtbot, viewer)
    qtbot.waitUntil(render.gl_ok, timeout=5000)
    image = render.grab_image()
    assert not image.isNull()
    brightest = max(image.pixelColor(x, y).lightness()
                    for x in range(0, image.width(), 4) for y in range(0, image.height(), 4))
    assert brightest > 10, "the volume rendered to an empty image"

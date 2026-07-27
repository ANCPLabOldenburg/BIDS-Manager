"""Tests for the NIfTI viewer's 3-D GPU raycast integration.

These exercise the *wiring* — the 3D toolbar toggle, mutual exclusion
with Multi view, control enable/disable, and the data path into the GL
view — without requiring a live OpenGL context. The ``RaycastGLWidget``
is created (a ``QOpenGLWidget``) but never shown, so ``initializeGL`` /
rendering don't run; the volume is stashed as ``pending`` and observed
via :meth:`RaycastGLWidget.has_volume`. This keeps the suite green under
the headless ``offscreen`` Qt platform used in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("OpenGL")

from bidsmgr.gui.widgets.nifti_viewer_pane import NiftiViewerPane
from bidsmgr.gui.widgets.nifti_gl_view import (
    Nifti3DView,
    RaycastGLWidget,
    request_gl_format,
)


pytestmark = pytest.mark.gui


def _write_nifti(path: Path, arr: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(path))


@pytest.fixture
def bids_root_with_nifti(tmp_path: Path) -> Path:
    root = tmp_path / "Studyname"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    arr_3d = np.arange(10 * 12 * 8, dtype=np.float32).reshape(10, 12, 8)
    _write_nifti(anat / "sub-01_ses-01_T1w.nii.gz", arr_3d)
    func = root / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    arr_4d = np.random.default_rng(0).random((6, 6, 4, 3), dtype=np.float32)
    _write_nifti(func / "sub-01_ses-01_task-rest_bold.nii.gz", arr_4d)
    return root


def _t1(root: Path) -> Path:
    return root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"


def _bold(root: Path) -> Path:
    return root / "sub-01" / "ses-01" / "func" / "sub-01_ses-01_task-rest_bold.nii.gz"


def _load_and_wait(pane, path, root, qtbot, timeout_ms: int = 5000):
    with qtbot.waitSignal(pane.loaded, timeout=timeout_ms):
        pane.set_file(path, root)


# ---------------------------------------------------------------------------
# request_gl_format
# ---------------------------------------------------------------------------


def test_request_gl_format_is_33_core(qapp) -> None:
    from PyQt6.QtGui import QSurfaceFormat

    fmt = request_gl_format()
    assert fmt.majorVersion() == 3 and fmt.minorVersion() == 3
    assert fmt.profile() == QSurfaceFormat.OpenGLContextProfile.CoreProfile


# ---------------------------------------------------------------------------
# 3D button enablement
# ---------------------------------------------------------------------------


def test_3d_button_disabled_before_load(qapp, isolated_settings) -> None:
    pane = NiftiViewerPane()
    assert pane._td_btn.isEnabled() is False


def test_3d_button_enabled_for_3d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    assert pane._td_btn.isEnabled() is True


def test_3d_button_enabled_for_4d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _bold(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    assert pane._td_btn.isEnabled() is True


# ---------------------------------------------------------------------------
# Toggle behaviour
# ---------------------------------------------------------------------------


def test_3d_toggle_switches_stack_and_disables_2d_controls(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)

    pane._td_btn.click()
    qapp.processEvents()
    assert pane._three_d is True
    # The GL page was created and is now current.
    assert pane._gl_page_index is not None
    assert pane._image_stack.currentIndex() == pane._gl_page_index
    # 2-D-only controls greyed out.
    for btn in (pane._sa_btn, pane._co_btn, pane._ax_btn):
        assert not btn.isEnabled()
    assert not pane._slice_slider.isEnabled()
    assert not pane._bright_slider.isEnabled()
    assert not pane._contrast_slider.isEnabled()
    # Multi view stays live so the user can switch straight to it.
    assert pane._tri_btn.isEnabled()

    # Toggle off restores the single-pane view + controls.
    pane._td_btn.click()
    qapp.processEvents()
    assert pane._three_d is False
    assert pane._image_stack.currentIndex() == 0
    assert pane._sa_btn.isEnabled()
    assert pane._slice_slider.isEnabled()
    assert pane._bright_slider.isEnabled()


def test_3d_toggle_feeds_volume_to_gl_view(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    pane._td_btn.click()
    qapp.processEvents()
    assert pane._gl is not None
    # Volume was handed to the GL widget (stashed as pending until first paint).
    assert pane._gl.has_volume() is True


def test_ras_and_radiological_drive_shared_render(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    """The RAS and Radiological toggles fold into one display flip that reaches
    both the 2-D labels and the shared 3-D render."""
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    pane._td_btn.click()
    qapp.processEvents()
    from bidsmgr.gui.widgets.nifti_viewer_pane import _AXIS_AXIAL
    assert pane._ras_btn.isChecked() is True
    assert pane._radio_btn.isChecked() is False

    # The 3-D flip carries a constant extra L/R mirror vs the 2-D flip so the
    # rendered volume's left/right lines up with the 2-D slices.
    def _aligned():
        d = pane._display_flip()
        assert pane._gl_flip() == (-d[0], d[1], d[2])
        assert np.allclose(pane._gl._flip, pane._gl_flip())
    _aligned()
    assert pane._display_labels(0)["left"] == "P"          # sagittal unaffected
    assert pane._display_labels(_AXIS_AXIAL)["left"] == "L"  # neurological

    pane._radio_btn.setChecked(True)
    qapp.processEvents()
    _aligned()                                              # still aligned
    assert pane._display_flip()[0] == -1.0                  # 2-D mirrored L/R
    assert pane._display_labels(_AXIS_AXIAL)["left"] == "R"  # labels swap too


def test_colour_fa_is_3d_capable(
    qapp, qtbot, bids_root_with_nifti, isolated_settings, tmp_path,
) -> None:
    """Colour-FA volumes render in 3-D (tinted), like MRIcroGL.

    They used to be excluded: the raycaster only understood a scalar density
    field. The shader now takes the RGB vector length as the density and the
    hue as the surface colour.
    """
    import nibabel as nib

    rgb = np.zeros((8, 9, 10), dtype=[("R", "u1"), ("G", "u1"), ("B", "u1")])
    rgb["R"][2:6] = 220
    rgb["G"][:, 3:7] = 180
    path = tmp_path / "sub-01_ses-01_colFA.nii.gz"
    nib.save(nib.Nifti1Image(rgb, np.diag([2.0, 2.0, 2.0, 1.0])), str(path))

    pane = NiftiViewerPane()
    _load_and_wait(pane, path, tmp_path, qtbot)
    assert pane._is_rgb is True
    assert pane._is_3d_capable is True
    if not pane._gpu_ok:
        return
    assert pane._td_btn.isEnabled() is True

    pane._td_btn.click()
    qapp.processEvents()
    assert pane._gl.has_volume() is True
    assert pane._gl.is_rgb_volume() is True      # uploaded as colour


def test_first_scan_opens_in_default_view(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    """The first volume of a session opens in Multi-Planar 3D with a GPU (plain
    Multi-Planar without one); later scans keep whatever view the user is in."""
    pane = NiftiViewerPane()
    pane._gpu_ok = True
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    assert pane._combo_view is True                 # landed in Multi-Planar 3D

    # The user picks a different view; a later scan must not override it.
    pane._set_view_mode("single")
    qapp.processEvents()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    assert pane._combo_view is False
    assert pane._tri_view is False                  # user's choice persisted


def test_first_scan_default_view_without_gpu(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    """Without a GPU the default is the 2-D three-plane Multi-Planar view."""
    pane = NiftiViewerPane()
    pane._gpu_ok = False
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    assert pane._tri_view is True
    assert pane._combo_view is False


def test_3d_and_multiview_mutually_exclusive(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)

    # Multi view on, then 3D on → Multi view turns off.
    pane._tri_btn.click()
    qapp.processEvents()
    assert pane._tri_view is True
    pane._td_btn.click()
    qapp.processEvents()
    assert pane._three_d is True
    assert pane._tri_view is False
    assert pane._image_stack.currentIndex() == pane._gl_page_index

    # Multi view on again → 3D turns off.
    pane._tri_btn.click()
    qapp.processEvents()
    assert pane._tri_view is True
    assert pane._three_d is False
    assert pane._image_stack.currentIndex() == 1


def test_volume_slider_repushes_in_3d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    """Changing the 4-D volume while 3-D is open re-feeds the GL view."""
    pane = NiftiViewerPane()
    _load_and_wait(pane, _bold(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    pane._td_btn.click()
    qapp.processEvents()
    # Moving the volume slider should not raise and keeps a volume bound.
    pane._vol_slider.setValue(2)
    qapp.processEvents()
    assert pane._gl.has_volume() is True


def test_set_file_none_clears_3d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    pane._td_btn.click()
    qapp.processEvents()
    assert pane._three_d is True

    pane.set_file(None, None)
    qapp.processEvents()
    assert pane._three_d is False
    assert pane._td_btn.isEnabled() is False
    assert pane._image_stack.currentIndex() == 0


def test_orientation_shortcut_exits_3d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    from bidsmgr.gui.widgets.nifti_viewer_pane import _AXIS_SAGITTAL

    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    pane._td_btn.click()
    qapp.processEvents()
    assert pane._three_d is True
    # Pressing a plane shortcut leaves 3-D and shows that plane in 2-D.
    pane._shortcut_orientation(_AXIS_SAGITTAL)
    qapp.processEvents()
    assert pane._three_d is False
    assert pane._orientation == _AXIS_SAGITTAL
    assert pane._image_stack.currentIndex() == 0


# ---------------------------------------------------------------------------
# Nifti3DView unit
# ---------------------------------------------------------------------------


def test_nifti3dview_normalises_and_binds(qapp) -> None:
    from bidsmgr.gui.widgets.nifti_gl_view import EFFECTS, EFFECT_FX

    view = Nifti3DView()
    vol = np.linspace(0, 1000, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    view.set_volume(vol, (1.0, 1.0, 1.0))
    assert view.gl.has_volume() is True
    # The effect combo maps its label to the shader effect index.
    view.controls._effect.setCurrentIndex(EFFECTS.index("MIP"))
    assert view.gl.effect == EFFECT_FX["MIP"]


def test_effect_and_lighting_selectors(qapp) -> None:
    from bidsmgr.gui.widgets.nifti_gl_view import EFFECTS, EFFECT_FX, LIGHTINGS

    view = Nifti3DView()
    assert len(EFFECTS) >= 5 and len(LIGHTINGS) >= 5
    view.controls._effect.setCurrentIndex(EFFECTS.index("Glass"))
    assert view.gl.effect == EFFECT_FX["Glass"]
    view.controls._light.setCurrentIndex(3)    # a different matcap
    assert view.gl.matcap_name == LIGHTINGS[3]


def test_preset_effects_apply(qapp) -> None:
    """Jelly / Skull are Opacity-peeling presets that apply parameter values."""
    from bidsmgr.gui.widgets.nifti_gl_view import EFFECTS, EFFECT_FX, EFFECT_PRESET

    view = Nifti3DView()
    c = view.controls
    for name in ("Jelly", "Skull"):
        c._effect.setCurrentIndex(EFFECTS.index(name))
        assert view.gl.effect == EFFECT_FX[name]          # underlying shader fx
        preset = EFFECT_PRESET[name]
        assert c._den.value() == preset["density"]        # preset value applied
        assert abs(view.gl.density - preset["density"] / 100.0) < 1e-6


def test_threshold_sliders_cannot_invert(qapp) -> None:
    """Regression: lo dragged above hi used to paint the whole box black."""
    view = Nifti3DView()
    c = view.controls
    c._hi.setValue(300)
    c._lo.setValue(900)                        # shove lo past hi
    # The coupling keeps hi strictly above lo, so the shader window is valid.
    assert view.gl.thresh_hi > view.gl.thresh_lo
    assert c._hi.value() > c._lo.value()


def test_clip_controls_drive_gl(qapp) -> None:
    """MRIcroGL-style oblique clip: enable + azimuth/elevation/depth/thick."""
    view = Nifti3DView()
    c = view.controls
    assert view.gl.clip_active == 0
    assert not c._caz.isEnabled()              # az/el/depth/thick off until enabled
    c._clip_en.setChecked(True)
    assert view.gl.clip_active == 1
    assert c._caz.isEnabled() and c._cel.isEnabled()
    # Elevation 90° -> axial normal (0,0,1); az=0 el=0 -> coronal (0,1,0).
    c._cel.setValue(90)
    assert view.gl.clip_normal[2] > 0.99
    c._cel.setValue(0); c._caz.setValue(0)
    assert abs(view.gl.clip_normal[1] - 1.0) < 1e-3
    c._clip_en.setChecked(False)
    assert view.gl.clip_active == 0
    assert not c._caz.isEnabled()


def test_slicer_shortcuts(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    """Shift+Y/A/S/C/X drive the clip plane and stay in sync with the panel."""
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    pane._td_btn.click()
    qapp.processEvents()
    gl, c = pane._gl, pane._gl_controls

    pane._kbd_toggle_slicer()                       # Shift+Y
    assert gl.clip_active == 1 and c._clip_en.isChecked()
    pane._kbd_clip_axial()                          # Shift+A -> normal ~Z
    assert abs(gl.clip_normal[2]) > 0.99
    assert c._caz.value() == 0 and c._cel.value() == 90
    pane._kbd_clip_sagittal()                       # Shift+S -> normal ~X
    assert abs(gl.clip_normal[0]) > 0.99
    pane._kbd_clip_coronal()                        # Shift+C -> normal ~Y
    assert abs(gl.clip_normal[1]) > 0.99
    n = gl.clip_normal
    pane._kbd_clip_invert()                         # Shift+X -> flip
    assert gl.clip_flip
    assert all(abs(gl.clip_normal[i] + n[i]) < 1e-5 for i in range(3))
    # Shift+drag / Shift+scroll equivalents mirror into the sliders.
    gl.drag_clip_azel(20, -5)
    assert c._caz.value() == int(round(gl.clip_az))
    gl.nudge_clip_pos(0.05)
    assert c._cdepth.value() == int(round(gl.clip_pos * 1000))
    pane._kbd_toggle_slicer()                       # Shift+Y off
    assert gl.clip_active == 0


def test_effect_param_graying(qapp) -> None:
    """Per-effect the panel shows only the relevant parameter rows."""
    from bidsmgr.gui.widgets.nifti_gl_view import EFFECTS

    view = Nifti3DView()
    c = view.controls
    c._effect.setCurrentIndex(EFFECTS.index("Glass"))
    assert c._rows["brighten"].isHidden()      # Glass: no matcap brightness
    assert not c._rows["specular"].isHidden()  # Glass: Phong specular shown
    c._effect.setCurrentIndex(EFFECTS.index("MIP"))
    assert c._rows["density"].isHidden()        # MIP: nothing but window + quality
    c._effect.setCurrentIndex(EFFECTS.index("Opacity peeling"))
    assert not c._rows["peel"].isHidden()       # peeling: peel layers shown


def test_slice_overlay_per_effect_default(qapp) -> None:
    """The cut-face intensity slice defaults on for surface effects only."""
    from bidsmgr.gui.widgets.nifti_gl_view import EFFECTS, SLICE_DEFAULT_ON

    view = Nifti3DView()
    c = view.controls
    for name in SLICE_DEFAULT_ON:                 # Standard / Matte / Topography
        c._effect.setCurrentIndex(EFFECTS.index(name))
        assert view.gl.slice_overlay == 1
        assert not c._rows["overlaydepth"].isHidden()
    for name in ("Glass", "Shell", "Opacity peeling", "MIP"):
        c._effect.setCurrentIndex(EFFECTS.index(name))
        assert view.gl.slice_overlay == 0
        assert c._rows["overlaydepth"].isHidden()
    c._effect.setCurrentIndex(0)
    c._odepth.setValue(70)                        # overlay-depth slider drives gl
    assert abs(view.gl.slice_depth - 0.70) < 1e-6


def test_gpu_available_and_cube(qapp) -> None:
    from bidsmgr.gui.widgets import nifti_gl_view as M

    # Probe returns a bool and is cached; cube geometry/atlas build.
    assert isinstance(M.gpu_available(), bool)
    assert M._cube_geometry().shape == (36, 8)
    view = Nifti3DView()
    assert view.gl._show_cube is True
    view.set_show_cube(False)
    assert view.gl._show_cube is False


def test_volume_retained_for_context_loss(qapp) -> None:
    """Regression: detach/re-attach recreates the GL context; the widget must
    keep the volume so ``initializeGL`` can re-upload it (else 3-D goes blank).
    """
    w = RaycastGLWidget()
    vol = np.random.default_rng(0).random((8, 8, 8), dtype=np.float32)
    w.set_volume_float(vol, (1.0, 1.0, 1.0))
    assert w._volume is not None            # retained, not consumed on upload
    assert w.has_volume() is True


def test_pan_moves_target(qapp) -> None:
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    w = RaycastGLWidget()
    t0 = w._target.copy()
    press = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress, QPointF(0, 0), QPointF(0, 0),
        Qt.MouseButton.RightButton, Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mousePressEvent(press)
    assert w._drag == "pan"
    move = QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(30, 10), QPointF(30, 10),
        Qt.MouseButton.RightButton, Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mouseMoveEvent(move)
    assert not np.allclose(w._target, t0)      # pan shifted the camera target
    w.reset_view()
    assert np.allclose(w._target, 0.0)         # reset recentres it


def test_wheel_zoom_and_slice_nav(qapp) -> None:
    """Vertical wheel zooms; Shift+wheel (or a horizontal scroll — the X11 /
    macOS remap of Shift+wheel) navigates the slice while the clip is active."""
    from PyQt6.QtCore import QPointF, QPoint, Qt
    from PyQt6.QtGui import QWheelEvent

    def wheel(ay, ax, shift=False):
        return QWheelEvent(
            QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(ax, ay),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )

    w = RaycastGLWidget()
    d0 = w._dist
    w.wheelEvent(wheel(120, 0))                  # vertical -> zoom one notch
    assert abs(w._dist / d0 - 0.9) < 1e-6

    w.clip_active = 1
    p = w.clip_pos
    w.wheelEvent(wheel(120, 0, shift=True))      # Shift+vertical -> slice nav
    assert abs(w.clip_pos - (p + 0.02)) < 1e-6
    p = w.clip_pos
    w.wheelEvent(wheel(0, 120))                  # horizontal (Shift remap) -> slice nav
    assert abs(w.clip_pos - (p + 0.02)) < 1e-6

    w.clip_active = 0                            # horizontal without clip -> zoom
    d0 = w._dist
    w.wheelEvent(wheel(0, 120))
    assert w._dist != d0


def test_render_flip_and_cube_atlas(qapp) -> None:
    """set_flip mirrors the render along canonical axes, and the orientation
    cube swaps its letters (not its geometry) to match."""
    from bidsmgr.gui.widgets.nifti_gl_view import _make_cube_atlas

    w = RaycastGLWidget()
    assert np.allclose(w._flip, [1.0, 1.0, 1.0])   # neurological RAS default

    w.set_flip(-1, 1, 1)
    assert np.allclose(w._flip, [-1.0, 1.0, 1.0])
    w.set_flip(1, -1, -1)
    assert np.allclose(w._flip, [1.0, -1.0, -1.0])

    # Flipping an axis swaps that axis's letter pair in the atlas (L<->R here),
    # so a mirrored render still reads upright.
    a0 = _make_cube_atlas(16, (1.0, 1.0, 1.0))
    a1 = _make_cube_atlas(16, (-1.0, 1.0, 1.0))
    assert a0.shape == a1.shape
    assert not np.array_equal(a0, a1)


def test_native_flip_from_affine(qapp) -> None:
    """The native storage flip is (1,1,1) for RAS and flags reversed axes."""
    from bidsmgr.gui.widgets.nifti_viewer_pane import _native_flip_from_affine

    assert _native_flip_from_affine(np.diag([2, 2, 2, 1.0])) == (1.0, 1.0, 1.0)
    # LAS: L increases along +i -> the R/L axis is stored reversed.
    assert _native_flip_from_affine(np.diag([-2, 2, 2, 1.0])) == (-1.0, 1.0, 1.0)
    # RAI: S/I reversed.
    assert _native_flip_from_affine(np.diag([2, 2, -2, 1.0])) == (1.0, 1.0, -1.0)


def test_show_values_reflects_slider(qapp) -> None:
    """The 'Show values' option appends each slider's live value to its label,
    and updates when a preset changes the sliders."""
    from bidsmgr.gui.widgets.nifti_gl_view import EFFECT_PRESET

    view = Nifti3DView()
    c = view.controls
    chips = {base: chip for (_s, chip, base) in c._value_rows}
    sliders = {base: s for (s, _c, base) in c._value_rows}

    # Off: labels are the plain base text.
    assert chips["Threshold low"].text() == "Threshold low"

    c._values_en.setChecked(True)
    assert chips["Threshold low"].text().endswith(str(sliders["Threshold low"].value()))

    # A preset moves sliders; the label tracks the new value live.
    c._effect.setCurrentText("Jelly")
    assert chips["Density"].text().endswith(str(EFFECT_PRESET["Jelly"]["density"]))

    c._values_en.setChecked(False)
    assert chips["Density"].text() == "Density"


def test_reset_params_current_effect_only(qapp) -> None:
    """Reset parameters resets ONLY the current effect's params; the effect,
    clip and unrelated params are left alone."""
    from bidsmgr.gui.widgets.nifti_gl_view import DEFAULTS, EFFECTS, EFFECT_FX

    view = Nifti3DView()
    c = view.controls
    c._effect.setCurrentIndex(EFFECTS.index("Glass"))
    c._clip_en.setChecked(True)
    c._spec.setValue(95)          # a Glass parameter
    c._bright.setValue(260)       # NOT a Glass parameter
    c.reset_params()
    assert view.gl.effect == EFFECT_FX["Glass"]        # effect unchanged
    assert view.gl.clip_active == 1                    # clip left as-is
    assert c._spec.value() == DEFAULTS["specular"]     # Glass param reset
    assert c._bright.value() == 260                    # unrelated param untouched
    view.gl.refresh()             # no-op without a live context, must not raise


def test_effect_params_are_independent(qapp) -> None:
    """Each effect owns its parameters: tweaking one never leaks into another,
    and switching back restores what was left behind."""
    from bidsmgr.gui.widgets.nifti_gl_view import EFFECT_PRESET

    view = Nifti3DView()
    c = view.controls

    c._effect.setCurrentText("Jelly")
    assert c._den.value() == EFFECT_PRESET["Jelly"]["density"]   # starts at preset
    c._den.setValue(88)                                          # tweak Jelly

    c._effect.setCurrentText("Skull")
    # Skull starts from ITS preset, not Jelly's edited value.
    assert c._den.value() == EFFECT_PRESET["Skull"]["density"]
    c._den.setValue(11)                                          # tweak Skull

    c._effect.setCurrentText("Jelly")
    assert c._den.value() == 88                                  # Jelly remembered
    c._effect.setCurrentText("Skull")
    assert c._den.value() == 11                                  # Skull remembered

    # Resetting one effect leaves the others' values alone.
    c.reset_params()
    assert c._den.value() == EFFECT_PRESET["Skull"]["density"]
    c._effect.setCurrentText("Jelly")
    assert c._den.value() == 88

    # Reset-all clears every effect back to its baseline.
    c.reset_all_params()
    assert c._den.value() == EFFECT_PRESET["Jelly"]["density"]
    c._effect.setCurrentText("Skull")
    assert c._den.value() == EFFECT_PRESET["Skull"]["density"]


def test_juicy_shiny_effects(qapp) -> None:
    """Juicy shiny / Juicy shiny 2 are Standard (flat Phong) presets placed
    after Matte, and Jelly / Skull sit after X-ray."""
    from bidsmgr.gui.widgets.nifti_gl_view import (
        EFFECTS, EFFECT_FX, EFFECT_PRESET, SLICE_DEFAULT_ON,
    )

    assert EFFECTS.index("Juicy shiny") == EFFECTS.index("Matte") + 1
    assert EFFECTS.index("Juicy shiny 2") == EFFECTS.index("Juicy shiny") + 1
    assert EFFECTS.index("Jelly") == EFFECTS.index("X-ray") + 1
    assert EFFECTS.index("Skull") == EFFECTS.index("Jelly") + 1

    view = Nifti3DView()
    c = view.controls
    for name in ("Juicy shiny", "Juicy shiny 2"):
        assert EFFECT_FX[name] == EFFECT_FX["Standard"]
        assert name in SLICE_DEFAULT_ON
        c._effect.setCurrentText(name)
        preset = EFFECT_PRESET[name]
        assert c._lo.value() == preset["lo"]
        assert c._hi.value() == preset["hi"]
        assert c._den.value() == preset["density"]
        assert c._amb.value() == preset["ambient"]
        assert c._dif.value() == preset["diffuse"]
        assert c._spec.value() == preset["specular"]
        assert c._shin.value() == preset["shininess"]
        assert c._odepth.value() == preset["overlaydepth"]
        assert view.gl.effect == EFFECT_FX[name]
        assert view.gl.slice_overlay == 1           # cut-face slice on

    # The two are independent presets, not the same values.
    assert EFFECT_PRESET["Juicy shiny 2"] != EFFECT_PRESET["Juicy shiny"]


def test_quality_defaults_to_half(qapp) -> None:
    """Every effect starts at 50% quality (the user raises it when they want).
    Quality is a performance knob, so no preset pins it."""
    from bidsmgr.gui.widgets.nifti_gl_view import (
        DEFAULTS, EFFECTS, EFFECT_PRESET, QUALITY_MAX,
    )

    assert DEFAULTS["quality"] == QUALITY_MAX // 2
    assert not any("quality" in p for p in EFFECT_PRESET.values())

    view = Nifti3DView()
    c = view.controls
    for name in EFFECTS:
        c._effect.setCurrentText(name)
        assert c._q.value() == DEFAULTS["quality"]
        assert view.gl.steps == DEFAULTS["quality"]

    # Raising it is per-effect, like every other parameter.
    c._effect.setCurrentText("Jelly")
    c._q.setValue(QUALITY_MAX)
    c._effect.setCurrentText("Skull")
    assert c._q.value() == DEFAULTS["quality"]
    c._effect.setCurrentText("Jelly")
    assert c._q.value() == QUALITY_MAX


def test_rgb_volume_upload(qapp) -> None:
    """Colour volumes (colour-FA) upload as colour, not as a scalar field."""
    w = RaycastGLWidget()
    assert w.is_rgb_volume() is False

    rgb = np.zeros((4, 5, 6, 3), np.uint8)
    rgb[..., 0] = 200
    w.set_volume(rgb, (1.0, 1.0, 1.0))
    assert w.is_rgb_volume() is True

    # RGBA is accepted; alpha is dropped (the shader samples .rgb).
    rgba = np.zeros((4, 5, 6, 4), np.uint8)
    w.set_volume(rgba, (1.0, 1.0, 1.0))
    assert w._volume[0].shape == (4, 5, 6, 3)

    # A scalar volume still takes the scalar path.
    w.set_volume(np.zeros((4, 5, 6), np.uint8), (1.0, 1.0, 1.0))
    assert w.is_rgb_volume() is False

    for bad in (np.zeros((4, 5, 6, 2), np.uint8), np.zeros((4, 5), np.uint8)):
        with pytest.raises(ValueError):
            w.set_volume(bad, (1.0, 1.0, 1.0))


def test_rgb_volume_float_keeps_hue(qapp) -> None:
    """Colour channels share ONE window, so per-voxel hue survives.

    Windowing each channel separately would recolour the data — a colour-FA
    map's direction encoding (red = L-R, green = A-P, blue = S-I) would be lost.
    """
    w = RaycastGLWidget()
    vol = np.zeros((2, 2, 2, 3), np.float32)
    vol[..., 0] = 1.0        # red at full
    vol[..., 1] = 0.5        # green at half
    w.set_volume_float(vol, (1.0, 1.0, 1.0))
    stored = w._volume[0]
    assert stored.shape == (2, 2, 2, 3)
    assert stored[..., 0].max() == 255
    assert abs(int(stored[..., 1].max()) - 128) <= 1     # ratio preserved
    assert stored[..., 2].max() == 0


def test_realistic_effect(qapp) -> None:
    """Realistic is a living-tissue variant of Juicy shiny 2, sitting after it
    and driven by its own shader branch (not the flat-Phong one)."""
    from bidsmgr.gui.widgets.nifti_gl_view import (
        EFFECTS, EFFECT_FX, EFFECT_PRESET, SLICE_DEFAULT_ON,
    )

    assert EFFECTS.index("Realistic") == EFFECTS.index("Juicy shiny 2") + 1
    assert "Realistic" in SLICE_DEFAULT_ON
    # Its own shader branch — the tissue colour ramp can't live in Standard's.
    assert EFFECT_FX["Realistic"] not in (
        EFFECT_FX["Standard"], EFFECT_FX["Matte"],
    )

    preset, parent = EFFECT_PRESET["Realistic"], EFFECT_PRESET["Juicy shiny 2"]
    # Inherits the window / density from its parent...
    for key in ("lo", "hi", "density", "overlaydepth"):
        assert preset[key] == parent[key]
    # ...but leans on directional light instead of ambient, or a coloured
    # surface flattens into a featureless pink card.
    assert preset["ambient"] < parent["ambient"]
    assert preset["diffuse"] > parent["diffuse"]

    view = Nifti3DView()
    c = view.controls
    c._effect.setCurrentText("Realistic")
    assert view.gl.effect == EFFECT_FX["Realistic"]
    assert c._amb.value() == preset["ambient"]
    assert view.gl.slice_overlay == 1


def test_render_background_is_black(qapp) -> None:
    """The GL clear / empty-space colour is pure black, matching the 2-D
    canvas (#nifti-canvas in theme.qss)."""
    w = RaycastGLWidget()
    assert w._bg == (0.0, 0.0, 0.0)


def test_controls_column_is_opaque(qapp, isolated_settings) -> None:
    """Regression: the black image canvas showed through the 3-D controls
    scroll gutter as a stripe. Every layer of that column must be nameable by
    the stylesheet so it paints the panel colour."""
    from bidsmgr.gui.widgets.nifti_gl_view import Nifti3DControls, RaycastGLWidget

    # The panel used to borrow "sidecar-toolbar" — a QFrame rule that never
    # matched this QWidget, so it painted nothing.
    controls = Nifti3DControls(RaycastGLWidget())
    assert controls.objectName() == "nifti-3d-controls"

    pane = NiftiViewerPane()
    if not pane._gpu_ok:
        return                                    # 3-D pages are not built
    pane._ensure_gl()
    assert pane._gl_controls_scroll.objectName() == "nifti-ctrl-scroll"
    # The viewport can't be selected from a stylesheet, so it is named here.
    assert pane._gl_controls_scroll.viewport().objectName() == "nifti-ctrl-viewport"
    assert pane._ctrl_slot_3d.objectName() == "nifti-ctrl-slot"
    assert pane._ctrl_slot_combo.objectName() == "nifti-ctrl-slot"

    qss = (Path(__file__).resolve().parents[2]
           / "bidsmgr" / "gui" / "theme.qss").read_text()
    for name in ("nifti-ctrl-slot", "nifti-ctrl-scroll",
                 "nifti-ctrl-viewport", "nifti-3d-controls"):
        assert f"#{name}" in qss, f"{name} has no stylesheet rule"


def test_drag_right_increases_azimuth(qapp) -> None:
    """Regression: left-right drag was inverted."""
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    w = RaycastGLWidget()
    az0 = w._az
    w._last = QPoint(0, 0)
    ev = QMouseEvent(
        QMouseEvent.Type.MouseMove, QPointF(10, 0), QPointF(10, 0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mouseMoveEvent(ev)
    assert w._az > az0


# ---------------------------------------------------------------------------
# Ortho 3D (3 planes + render) combined mode
# ---------------------------------------------------------------------------


def test_ortho3d_enabled_for_3d(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    assert pane._quad_btn.isEnabled() is True


def test_ortho3d_toggle_shows_grid_and_keeps_slice_controls(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    # The first scan of a session lands in a GPU-dependent default view; start
    # from a known state so clicking the toggle below always turns it ON.
    pane._set_view_mode("single")
    qapp.processEvents()

    pane._quad_btn.click()
    qapp.processEvents()
    assert pane._combo_view is True
    assert pane._combo_page_index is not None
    assert pane._image_stack.currentIndex() == pane._combo_page_index
    # The render got the volume, and all three slice panels exist.
    assert pane._gl.has_volume() is True
    assert pane._gl_controls is not None
    assert len(pane._combo_labels) == 3
    # Slices are shown, so brightness/contrast/crosshair stay live...
    assert pane._bright_slider.isEnabled()
    assert pane._crosshair_swatch.isEnabled()
    # ...but the single-pane-only controls do not.
    assert not pane._sa_btn.isEnabled()
    assert not pane._slice_slider.isEnabled()


def test_three_3d_modes_mutually_exclusive(
    qapp, qtbot, bids_root_with_nifti, isolated_settings,
) -> None:
    pane = NiftiViewerPane()
    _load_and_wait(pane, _t1(bids_root_with_nifti), bids_root_with_nifti, qtbot)
    # Known starting point — see the session-default view above.
    pane._set_view_mode("single")
    qapp.processEvents()

    pane._quad_btn.click(); qapp.processEvents()
    assert pane._combo_view and not pane._three_d and not pane._tri_view

    pane._td_btn.click(); qapp.processEvents()
    assert pane._three_d and not pane._combo_view

    pane._tri_btn.click(); qapp.processEvents()
    assert pane._tri_view and not pane._three_d and not pane._combo_view

    pane._tri_btn.click(); qapp.processEvents()
    assert not (pane._tri_view or pane._three_d or pane._combo_view)
    assert pane._image_stack.currentIndex() == 0

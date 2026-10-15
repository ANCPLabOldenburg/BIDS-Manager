"""The store and its commands: the only way the scene changes."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from pydantic import ValidationError

from bidsmgr.viz import render3d, views
from bidsmgr.viz.commands import COMMANDS
from bidsmgr.viz.data.volume import open_volume
from bidsmgr.viz.scene import Cursor, Scene, SourceRef, VolumeLayer
from bidsmgr.viz.store import SceneStore, touches


def _store_with_volume(tmp_path: Path, data=None, affine=None) -> SceneStore:
    if data is None:
        data = np.random.default_rng(0).random((8, 9, 10, 5)).astype(np.float32)
    path = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), str(path))
    src = open_volume(path)
    src.stream()
    store = SceneStore()
    store.sources["v"] = src
    scene = Scene()
    scene.sources["v"] = SourceRef(id="v", path=str(path), kind="volume")
    scene.layers = [VolumeLayer(id="base", source="v", name="vol")]
    scene.cursor = Cursor(world=tuple(float(v) for v in src.center_world()))
    store.replace_scene(scene)
    return store


def test_a_command_reports_what_it_changed_and_a_no_op_reports_nothing() -> None:
    store = SceneStore()
    assert store.run("view.mode", mode="3d") == frozenset({"mode"})
    assert store.run("view.mode", mode="3d") == frozenset()


def test_parameters_are_validated() -> None:
    store = SceneStore()
    with pytest.raises(ValidationError):
        store.run("view.mode", mode="nonsense")
    with pytest.raises(KeyError):
        store.run("no.such.command")


def test_listeners_hear_paths_and_a_batch_notifies_once() -> None:
    store = SceneStore()
    heard: list = []
    store.subscribe(heard.append)
    with store.batch():
        store.run("view.mode", mode="single")
        store.run("view.plane", plane="coronal")
    assert heard == [frozenset({"mode", "plane"})]
    assert touches(heard[0], "plan")


def test_cursor_steps_toward_the_anatomical_end_whatever_the_storage(tmp_path: Path) -> None:
    """A file stored left-to-right reversed still steps toward +x for n > 0."""
    store = _store_with_volume(tmp_path, affine=np.diag([-1.0, 1.0, 1.0, 1.0]))
    before = store.scene.cursor.world
    store.run("cursor.step", plane="sagittal", n=1)
    after = store.scene.cursor.world
    assert after[0] == pytest.approx(before[0] + 1.0)


def test_cursor_clicks_snap_to_voxel_centres(tmp_path: Path) -> None:
    store = _store_with_volume(tmp_path, affine=np.diag([2.0, 2.0, 2.0, 1.0]))
    store.run("cursor.set_world", x=3.1, y=4.9, z=7.2)
    assert store.scene.cursor.world == pytest.approx((4.0, 4.0, 8.0))


def test_set_slice_and_set_voxel(tmp_path: Path) -> None:
    store = _store_with_volume(tmp_path)
    store.run("cursor.set_slice", plane="axial", index=2)
    assert views.slice_index(store, "axial") == 2
    store.run("cursor.set_voxel", i=1, j=2, k=3)
    assert views.cursor_voxel(store) == (1, 2, 3)


def test_frames_clamp_and_wrap(tmp_path: Path) -> None:
    store = _store_with_volume(tmp_path)
    store.run("frame.set", frame=99)
    assert store.scene.layers[0].frame == 4
    store.run("frame.step", n=1, wrap=True)
    assert store.scene.layers[0].frame == 0
    store.run("frame.step", n=-1)
    assert store.scene.layers[0].frame == 0


def test_toggle_mode_returns_to_a_single_slice() -> None:
    store = SceneStore()
    store.run("view.mode", mode="multi")
    store.run("view.toggle_mode", mode="multi")
    assert store.scene.mode == "single"


def test_plane_shortcut_leaves_a_multi_view() -> None:
    store = SceneStore()
    store.run("view.mode", mode="combo")
    store.run("view.plane", plane="sagittal", single=True)
    assert store.scene.mode == "single" and store.scene.plane == "sagittal"


def test_layer_patch_and_undo(tmp_path: Path) -> None:
    store = _store_with_volume(tmp_path)
    store.run("layer.set", colormap="hot", window=(5, 1))
    disp = store.scene.layers[0].display
    assert disp.colormap == "hot" and disp.window == (1.0, 5.0)
    store.run("cursor.step", plane="axial", n=1)
    cursor = store.scene.cursor.world
    assert store.undo()
    assert store.scene.layers[0].display.colormap == "gray"
    # Undo restores display state; it never jumps the crosshair back.
    assert store.scene.cursor.world == cursor
    assert store.redo()
    assert store.scene.layers[0].display.colormap == "hot"


def test_window_commands(tmp_path: Path) -> None:
    store = _store_with_volume(tmp_path)
    store.run("window.robust")
    lo, hi = store.scene.layers[0].display.window
    assert 0 <= lo < hi <= 1
    store.run("window.level_width", d_level=0.5, d_width=1.0)
    lo2, hi2 = store.scene.layers[0].display.window
    assert hi2 - lo2 == pytest.approx(2 * (hi - lo))
    store.run("window.full")
    lo3, hi3 = store.scene.layers[0].display.window
    assert lo3 <= lo and hi3 >= hi


def test_window_fit_box(tmp_path: Path) -> None:
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[2:5, 2:5, :] = 100.0
    store = _store_with_volume(tmp_path, data=data)
    # Axial rows run down from +y: rows 4-8 are j 5-1, so this box straddles
    # the bright square's edge.
    store.run("window.fit_box", plane="axial", c0=1, r0=4, c1=4, r1=8)
    lo, hi = store.scene.layers[0].display.window
    assert lo == pytest.approx(0.0) and hi == pytest.approx(100.0)
    # A box inside the square holds one value: nothing to fit, no change.
    assert store.run("window.fit_box", plane="axial", c0=2, r0=5, c1=4, r1=7) == frozenset()
    assert store.scene.layers[0].display.window == (lo, hi)


def test_zoom_keeps_the_point_under_the_mouse() -> None:
    store = SceneStore()
    store.run("view.zoom", plane="axial", factor=2.0, about=(10.0, 0.0))
    state = store.scene.views["axial"]
    assert state.zoom == 2.0
    assert state.pan == pytest.approx((-10.0, 0.0))
    store.run("view.reset")
    assert store.scene.views["axial"].zoom == 1.0


def test_render_thresholds_keep_their_gap() -> None:
    store = SceneStore()
    store.run("render.param", key="lo", value=900)
    values = render3d.effective("Standard", store.scene.render.params["Standard"])
    assert values["hi"] >= values["lo"] + render3d.THRESH_GAP


def test_effects_keep_their_own_parameters() -> None:
    store = SceneStore()
    store.run("render.param", key="density", value=250)
    store.run("render.effect", effect="Glass")
    store.run("render.effect", effect="Standard")
    assert store.scene.render.params["Standard"]["density"] == 250
    store.run("render.reset_params")
    assert "density" not in store.scene.render.params["Standard"]


def test_clip_commands() -> None:
    store = SceneStore()
    store.run("clip.axis", plane="sagittal")
    clip = store.scene.clips[0]
    assert clip.active and (clip.az, clip.el) == (90.0, 0.0)
    store.run("clip.invert")
    assert store.scene.clips[0].flip
    store.run("clip.nudge", delta=0.9)
    assert store.scene.clips[0].pos == 1.0
    store.run("clip.preset", preset="box")
    assert len(store.scene.clips) == 6
    with pytest.raises(ValueError):
        store.run("clip.set", index=7)


def test_camera_is_clamped() -> None:
    store = SceneStore()
    store.run("render.camera", el=3.0, dist=100.0)
    cam = store.scene.render.camera
    assert cam.el == pytest.approx(1.55) and cam.dist == pytest.approx(12.0)


def test_scene_serialises_and_round_trips(tmp_path: Path) -> None:
    store = _store_with_volume(tmp_path)
    store.run("layer.set", colormap="viridis")
    store.run("clip.preset", preset="wedge")
    data = store.scene.model_dump_json()
    assert Scene.model_validate_json(data) == store.scene


def test_every_command_has_a_title_and_a_category() -> None:
    for spec in COMMANDS.values():
        assert spec.title and spec.category, spec.id

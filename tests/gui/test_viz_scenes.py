"""Scenes saved with the dataset: save one, open it in another viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from bidsmgr.gui.viz import Viewer  # noqa: E402
from bidsmgr.viz import scenes  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "Study"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "x"}))
    t1 = np.zeros((16, 16, 8), np.float32)
    t1[2:14, 2:14, 1:7] = 400
    nib.save(nib.Nifti1Image(t1, np.eye(4)), str(anat / "sub-01_T1w.nii.gz"))
    atlas = np.zeros((16, 16, 8), np.int16)
    atlas[3:8, 3:12, :] = 5
    atlas[9:13, 3:12, :] = 9
    nib.save(nib.Nifti1Image(atlas, np.eye(4)), str(anat / "sub-01_dseg.nii.gz"))
    rng = np.random.default_rng(0)
    bold = (500 + rng.normal(0, 5, (16, 16, 8, 6))).astype(np.float32)
    nib.save(nib.Nifti1Image(bold, np.eye(4)), str(root / "sub-01" / "anat" / "sub-01_bold.nii.gz"))
    return root


def _viewer(qtbot) -> Viewer:
    v = Viewer(kind="volume")
    qtbot.addWidget(v)
    v.resize(900, 600)
    v.show()
    qtbot.waitExposed(v)
    return v


def _t1(root):
    return root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"


def test_a_scene_is_saved_in_the_dataset_and_opens_again(qtbot, ds):
    v = _viewer(qtbot)
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(_t1(ds), ds)
    with qtbot.waitSignal(v.overlay_added, timeout=20_000) as added:
        v.add_overlay(ds / "sub-01" / "anat" / "sub-01_dseg.nii.gz")
    v.run("layer.set", layer=added.args[0], opacity=0.8, outline_px=2.0)
    v.run("layer.set", colormap="bone")
    v.run("view.mode", mode="single")
    v.run("cursor.set_world", x=5.0, y=6.0, z=3.0, snap=False)
    v.qstore.flush()
    path = v.presenter.save_scene("hippocampus check")
    assert path == ds / ".bidsmgr" / "viz" / "scenes" / "hippocampus check.json"
    saved = json.loads(path.read_text())
    assert saved["base"] == "sub-01/anat/sub-01_T1w.nii.gz", "relative to the dataset, POSIX"
    assert saved["overlays"][0]["path"] == "sub-01/anat/sub-01_dseg.nii.gz"

    w = _viewer(qtbot)
    w.set_file(ds / "sub-01" / "anat" / "sub-01_bold.nii.gz", ds)   # something else first
    with qtbot.waitSignal(w.overlay_added, timeout=20_000):
        assert w.presenter.open_scene(path)
    w.qstore.flush()
    assert w.current_file() == _t1(ds)
    base, over = w.scene.layers
    assert base.display.colormap == "bone"
    assert over.name == "sub-01_dseg.nii.gz"
    assert over.display.opacity == pytest.approx(0.8)
    assert over.display.outline_px == pytest.approx(2.0)
    assert over.display.label_table is not None, "the atlas look came back whole"
    assert w.scene.mode == "single"
    assert tuple(w.scene.cursor.world) == pytest.approx((5.0, 6.0, 3.0))


def test_the_views_menu_lists_the_datasets_scenes(qtbot, ds):
    v = _viewer(qtbot)
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(_t1(ds), ds)
    v.presenter.save_scene("first look")
    menu = v.presenter.save_button.menu()
    v.presenter._fill_save_menu(menu)
    entries = {a.text(): a for a in menu.actions()}
    assert "Save a scene in this dataset..." in entries
    sub = entries["Open a scene of this dataset"].menu()
    assert [a.text() for a in sub.actions()] == ["first look"]


def test_a_quality_map_is_computed_again(qtbot, ds):
    v = _viewer(qtbot)
    bold = ds / "sub-01" / "anat" / "sub-01_bold.nii.gz"
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(bold, ds)
    with qtbot.waitSignal(v.overlay_added, timeout=20_000):
        v.trigger("qc.tsnr")
    path = v.presenter.save_scene("tsnr")
    assert json.loads(path.read_text())["overlays"][0]["origin"] == "qc:tsnr"
    w = _viewer(qtbot)
    with qtbot.waitSignal(w.overlay_added, timeout=30_000):
        w.presenter.open_scene(path)
    assert w.scene.layers[-1].origin == "qc:tsnr"
    assert w.scene.layers[-1].name.startswith("Temporal SNR of")


def test_a_scene_whose_image_is_gone_says_so(qtbot, ds):
    v = _viewer(qtbot)
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(_t1(ds), ds)
    path = v.presenter.save_scene("gone")
    _t1(ds).unlink()
    with qtbot.waitSignal(v.status_message, timeout=2000) as said:
        assert not v.presenter.open_scene(path)
    assert "gone" in said.args[0]


def test_a_scene_whose_overlay_is_gone_opens_and_names_it(qtbot, ds):
    v = _viewer(qtbot)
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(_t1(ds), ds)
    dseg = ds / "sub-01" / "anat" / "sub-01_dseg.nii.gz"
    with qtbot.waitSignal(v.overlay_added, timeout=20_000):
        v.add_overlay(dseg)
    path = v.presenter.save_scene("atlas")
    dseg.unlink()
    said = []
    v.status_message.connect(said.append)
    assert v.presenter.open_scene(path)
    qtbot.waitUntil(lambda: any("Opened the scene" in m for m in said), timeout=20_000)
    opened = next(m for m in said if "Opened the scene" in m)
    assert "sub-01/anat/sub-01_dseg.nii.gz" in opened and "gone" in opened
    assert not any("FileNotFoundError" in m for m in said)


class TestQtFree:
    def test_relative_paths_round_trip(self, tmp_path):
        root = tmp_path / "Study"
        p = root / "sub-01" / "anat" / "x.nii.gz"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
        rel = scenes.to_rel(root, p)
        assert rel == "sub-01/anat/x.nii.gz"
        assert scenes.from_rel(root, rel) == root / "sub-01" / "anat" / "x.nii.gz"

    def test_a_file_outside_the_dataset_keeps_its_absolute_path(self, tmp_path):
        outside = tmp_path / "elsewhere.nii.gz"
        outside.write_bytes(b"")
        rel = scenes.to_rel(tmp_path / "Study", outside)
        assert Path(rel).is_absolute()

    def test_list_and_load(self, tmp_path):
        scenes.save(tmp_path, {"schema": 1, "name": "b scene", "base": "x"})
        scenes.save(tmp_path, {"schema": 1, "name": "A scene", "base": "y"})
        assert [n for n, _p in scenes.list_scenes(tmp_path)] == ["A scene", "b scene"]
        assert scenes.load(scenes.scene_path(tmp_path, "A scene"))["base"] == "y"

    def test_a_newer_scene_is_refused_not_misread(self, tmp_path):
        path = scenes.save(tmp_path, {"schema": 99, "name": "future", "base": "x"})
        with pytest.raises(ValueError, match="newer"):
            scenes.load(path)

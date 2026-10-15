"""Moving through the dataset from the viewer: next run, subject; fieldmaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from bidsmgr.gui.viz import Viewer  # noqa: E402

pytestmark = pytest.mark.gui


def _img(path: Path, value: float, shape=(12, 12, 8)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.full(shape, value, np.float32), np.eye(4)), str(path))
    return path


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "Study"
    root.mkdir()
    (root / "dataset_description.json").write_text(json.dumps({"Name": "x"}))
    for sub in ("01", "02"):
        for run in ("1", "2"):
            _img(root / f"sub-{sub}" / "func" / f"sub-{sub}_task-x_run-{run}_bold.nii.gz",
                 float(run))
    fmap = _img(root / "sub-01" / "fmap" / "sub-01_phasediff.nii.gz", 3.0)
    fmap.with_name("sub-01_phasediff.json").write_text(json.dumps(
        {"IntendedFor": ["func/sub-01_task-x_run-1_bold.nii.gz"]}))
    return root


def _bold(root, sub="01", run="1"):
    return root / f"sub-{sub}" / "func" / f"sub-{sub}_task-x_run-{run}_bold.nii.gz"


def _viewer(qtbot) -> Viewer:
    v = Viewer(kind="volume")
    qtbot.addWidget(v)
    v.resize(900, 600)
    v.show()
    qtbot.waitExposed(v)
    return v


def _open(qtbot, v, path, root):
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(path, root)
    v.qstore.flush()


class TestNextRun:
    def test_it_opens_the_next_run_where_the_crosshair_was(self, qtbot, ds):
        v = _viewer(qtbot)
        _open(qtbot, v, _bold(ds), ds)
        v.run("cursor.set_world", x=3.0, y=4.0, z=2.0)
        v.qstore.flush()
        assert v.action("nav.run_next").isEnabled()
        assert not v.action("nav.run_prev").isEnabled()
        with qtbot.waitSignal(v.loaded, timeout=20_000):
            v.trigger("nav.run_next")
        v.qstore.flush()
        assert v.current_file() == _bold(ds, run="2")
        assert tuple(v.scene.cursor.world) == pytest.approx((3.0, 4.0, 2.0))
        assert v.action("nav.run_prev").isEnabled()

    def test_next_subject(self, qtbot, ds):
        v = _viewer(qtbot)
        _open(qtbot, v, _bold(ds), ds)
        assert v.action("nav.sub_next").shortcut().toString() == "Alt+Down"
        with qtbot.waitSignal(v.loaded, timeout=20_000):
            v.trigger("nav.sub_next")
        assert v.current_file() == _bold(ds, sub="02")

    def test_no_echo_no_echo_action(self, qtbot, ds):
        v = _viewer(qtbot)
        _open(qtbot, v, _bold(ds), ds)
        assert not v.action("nav.echo_next").isEnabled()

    def test_a_host_navigator_is_used(self, qtbot, ds):
        """The Editor moves through its tree, so everything else follows."""
        v = _viewer(qtbot)
        _open(qtbot, v, _bold(ds), ds)
        asked = []
        v.navigator = asked.append
        v.trigger("nav.run_next")
        assert asked == [_bold(ds, run="2")]


class TestFieldmaps:
    def _menu(self, v):
        menu = v.presenter.tools_button.menu()
        v.presenter._fill_tools_menu(menu)
        return {a.text(): a for a in menu.actions()}

    def test_an_image_lists_the_fieldmaps_meant_for_it(self, qtbot, ds):
        v = _viewer(qtbot)
        _open(qtbot, v, _bold(ds), ds)
        sub = self._menu(v)["Fieldmaps for this image"].menu()
        draw = [a for a in sub.actions() if a.text() == "Draw sub-01_phasediff.nii.gz over this"]
        assert draw
        with qtbot.waitSignal(v.overlay_added, timeout=20_000):
            draw[0].trigger()
        assert [lay.name for lay in v.scene.layers][-1] == "sub-01_phasediff.nii.gz"

    def test_a_run_without_one_has_no_entry(self, qtbot, ds):
        v = _viewer(qtbot)
        _open(qtbot, v, _bold(ds, run="2"), ds)
        assert "Fieldmaps for this image" not in self._menu(v)

    def test_a_fieldmap_opens_its_target_with_itself_over_it(self, qtbot, ds):
        v = _viewer(qtbot)
        fmap = ds / "sub-01" / "fmap" / "sub-01_phasediff.nii.gz"
        _open(qtbot, v, fmap, ds)
        sub = self._menu(v)["Images this fieldmap is for"].menu()
        (act,) = sub.actions()
        assert act.text() == "Open sub-01_task-x_run-1_bold.nii.gz with this over it"
        with qtbot.waitSignal(v.overlay_added, timeout=20_000):
            act.trigger()
        assert v.current_file() == _bold(ds)
        assert [lay.name for lay in v.scene.layers] == [
            "sub-01_task-x_run-1_bold.nii.gz", "sub-01_phasediff.nii.gz"]


class TestInTheEditor:
    def test_next_run_moves_the_tree(self, qtbot, ds):
        from bidsmgr.gui.editor_panel import EditorPanel

        ep = EditorPanel()
        qtbot.addWidget(ep)
        ep._set_root(ds, persist=False)
        viewer = ep._nifti_viewer
        with qtbot.waitSignal(viewer.loaded, timeout=20_000):
            ep.select_file_in_tree(_bold(ds))
        viewer.qstore.flush()
        with qtbot.waitSignal(viewer.loaded, timeout=20_000):
            viewer.trigger("nav.run_next")
        assert viewer.current_file() == _bold(ds, run="2")
        assert ep._tree_pane.selected_paths() == [_bold(ds, run="2")]

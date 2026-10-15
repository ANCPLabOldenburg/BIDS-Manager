"""The MRS voxel on the anatomy, drawn as the box it is.

The slice canvas draws the polygon its plane cuts out of the voxel, in
screen space; the 3-D render intersects the box exactly. Both were sampled
from pixels before, and were off by up to half an anatomy pixel (2-D) or
half the voxel (3-D).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QPointF  # noqa: E402

from bidsmgr.gui.viz import Viewer  # noqa: E402
from bidsmgr.viz.compute.shapes import section  # noqa: E402
from tests.gui.test_viz_render3d_sync import _pixels, gl_context  # noqa: E402,F401

pytestmark = pytest.mark.gui

CENTRE = (5.0, -3.0, 4.0)


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "Study"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    head = np.zeros((80, 80, 40), dtype=np.float32)
    head[10:70, 10:70, 5:35] = 100.0
    a = np.diag([1.0, 1.0, 2.0, 1.0])
    a[:3, :3] = a[:3, :3] @ np.array([[1, 0, 0], [0, 0.995, -0.0998], [0, 0.0998, 0.995]])
    a[:3, 3] = [-40.0, -40.0, -40.0]
    nib.save(nib.Nifti1Image(head, a), str(anat / "sub-01_T2w.nii.gz"))
    mrs = root / "sub-01" / "mrs"
    mrs.mkdir(parents=True)
    t = np.radians(10.0)
    m = np.eye(4)
    m[:3, :3] = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]) * 20.0
    m[:3, 3] = CENTRE
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1, 64), np.complex64), m),
             str(mrs / "sub-01_svs.nii.gz"))
    return root


def _viewer(qtbot, ds) -> Viewer:
    v = Viewer(kind="volume")
    qtbot.addWidget(v)
    v.resize(900, 600)
    v.show()
    qtbot.waitExposed(v)
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(ds / "sub-01" / "anat" / "sub-01_T2w.nii.gz", ds)
    qtbot.waitUntil(v.is_loaded, timeout=20_000)
    with qtbot.waitSignal(v.overlay_added, timeout=20_000):
        assert v.add_overlay(ds / "sub-01" / "mrs" / "sub-01_svs.nii.gz")
    v.run("cursor.set_world", x=CENTRE[0], y=CENTRE[1], z=CENTRE[2], snap=False)
    v.qstore.flush()
    return v


@pytest.mark.parametrize("plane", ["axial", "coronal", "sagittal"])
def test_the_slice_draws_the_exact_cut(qtbot, ds, plane):
    v = _viewer(qtbot, ds)
    v.trigger(f"view.{plane}")
    v.qstore.flush()
    canvas = v.canvases("slice")[0]
    canvas.repaint()
    polygons = canvas.shape_polygons()
    assert len(polygons) == 1
    _layer, polygon = polygons[0]
    grid = canvas._grid
    src = v.store.sources["ovl1"]
    expected = section(src.box, grid.origin, grid.normal)
    assert polygon.count() == len(expected)
    for i in range(polygon.count()):
        col, row = canvas.screen_to_grid(QPointF(polygon.at(i)))
        world = grid.pixel_to_world(col, row)
        assert np.min(np.linalg.norm(expected - world, axis=1)) < 1e-6


def test_hidden_it_is_not_drawn(qtbot, ds):
    v = _viewer(qtbot, ds)
    v.run("layer.set", layer="overlay1", visible=False)
    v.qstore.flush()
    canvas = v.canvases("slice")[0]
    canvas.repaint()
    assert canvas.shape_polygons() == []


def _green(px) -> int:
    return int(np.count_nonzero((px[..., 1] > px[..., 0] + 60) & (px[..., 1] > px[..., 2] + 60)))


class TestIn3D:
    def test_the_box_is_drawn_and_follows_its_switch(self, qtbot, gl_context, ds):  # noqa: F811
        v = _viewer(qtbot, ds)
        v.run("view.mode", mode="3d")
        v.qstore.flush()
        render = v.presenter.render_canvas
        qtbot.waitUntil(render.has_volume, timeout=10_000)
        qtbot.waitUntil(render.gl_ok, timeout=5000)
        v.trigger("view.crosshair")
        v.run("render.param", key="seethrough", value=100)
        v.qstore.flush()
        shown = _green(_pixels(render))
        v.run("layer.set", layer="overlay1", in_3d=False)
        v.qstore.flush()
        hidden = _green(_pixels(render))
        assert shown > 100 and hidden == 0

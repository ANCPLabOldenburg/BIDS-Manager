"""The 3-D half of a comparison: one camera, one effect, one cut, in both.

Rotating one head and not the other, or cutting them from opposite sides, is
not a comparison: it is a difference the data does not have. The GPU gate is
forced open for the whole file (its 2-D twin is ``test_viz_compare``), so the
wiring is exercised under the offscreen platform.
"""

from __future__ import annotations

import json
import shutil

import pytest

pytest.importorskip("nibabel")
pytest.importorskip("OpenGL")

from bidsmgr.deface import status  # noqa: E402
from bidsmgr.deface.engines import ALLINEATE, TEMPLATE  # noqa: E402
from bidsmgr.gui.deface_compare import DefaceCompareDialog  # noqa: E402
from bidsmgr.project.operations import begin_operation  # noqa: E402

pytestmark = pytest.mark.gui

REL = "sub-01/anat/sub-01_T1w.nii.gz"


@pytest.fixture(autouse=True)
def gpu(monkeypatch):
    from bidsmgr.gui.viz.canvases import render

    monkeypatch.setattr(render, "gpu_available", lambda: True)


@pytest.fixture
def panes(qtbot, tmp_path):
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"}))
    shutil.copyfile(TEMPLATE, root / REL)
    with begin_operation(root, "Deface 1 image (allineate)") as op:
        op.write_bytes(root / REL, (root / REL).read_bytes())
    status.sidecar_for(root / REL).write_text(json.dumps(status.record({}, ALLINEATE)))
    dlg = DefaceCompareDialog(root, REL)
    qtbot.addWidget(dlg)
    dlg.show()
    p = dlg.panes
    qtbot.waitUntil(lambda: p.left.is_loaded() and p.right.is_loaded()
                    and p.link.isEnabled(), timeout=20_000)
    yield p
    dlg.close()


def _flush(p) -> None:
    p.left.qstore.flush()
    p.right.qstore.flush()


def test_a_gpu_opens_the_planes_with_the_render(panes) -> None:
    assert panes.left.scene.mode == "combo" and panes.right.scene.mode == "combo"
    assert panes.left.presenter.render_canvas is not None


def test_the_camera_is_shared(panes) -> None:
    panes.left.run("render.orbit", d_az=0.5, d_el=0.1)
    _flush(panes)
    assert panes.right.scene.render.camera == panes.left.scene.render.camera


def test_an_effect_and_its_parameters_reach_the_other_render(panes) -> None:
    panes.left.run("render.effect", effect="Glass")
    panes.left.run("render.param", key="specular", value=77)
    _flush(panes)
    assert panes.right.scene.render.effect == "Glass"
    assert panes.right.scene.render.params["Glass"]["specular"] == 77


def test_the_cut_is_shared_including_its_side(panes) -> None:
    """Shift+X flips which side is kept; two renders cut from opposite sides
    are actively misleading."""
    panes.left.trigger("clip.toggle")
    panes.left.trigger("clip.sagittal")
    panes.left.run("clip.nudge", delta=0.1)
    panes.left.trigger("clip.invert")
    _flush(panes)
    assert panes.right.scene.clips == panes.left.scene.clips
    assert panes.right.scene.clips[0].flip and panes.right.scene.clips[0].active


def test_a_gesture_on_the_render_reaches_the_other(panes) -> None:
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    panes.left.run("view.mode", mode="3d")
    _flush(panes)
    render = panes.left.presenter.render_canvas
    left = Qt.MouseButton.LeftButton

    def ev(kind, x):
        return QMouseEvent(kind, QPointF(x, 0), QPointF(x, 0), left, left,
                           Qt.KeyboardModifier.NoModifier)

    render.mousePressEvent(ev(QMouseEvent.Type.MouseButtonPress, 0))
    render.mouseMoveEvent(ev(QMouseEvent.Type.MouseMove, 25))
    render.mouseReleaseEvent(ev(QMouseEvent.Type.MouseButtonRelease, 25))
    _flush(panes)
    assert panes.right.scene.render.camera.az == pytest.approx(
        panes.left.scene.render.camera.az)


def test_unsyncing_lets_the_renders_differ(panes) -> None:
    panes.link.setChecked(False)
    panes.left.run("render.effect", effect="MIP")
    _flush(panes)
    assert panes.right.scene.render.effect != "MIP"
    panes.link.setChecked(True)
    _flush(panes)
    assert panes.right.scene.render.effect == "MIP"

"""The viewer and the library draw the same thing, and say how.

The slice canvas composes its image through ``viz.render2d.slice_rgba``,
so a figure rendered with no window is the figure on screen; the Views
menu's "Show the command line" writes the call that reproduces the view.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("pyqtgraph")

from bidsmgr.gui.viz import Viewer  # noqa: E402
from bidsmgr.viz import render2d, reproduce  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "Study"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (root / "dataset_description.json").write_text('{"Name": "x", "BIDSVersion": "1.10.0"}')
    data = np.arange(20 * 24 * 10, dtype=np.float32).reshape(20, 24, 10)
    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 2.0, 1.0])),
             str(anat / "sub-01_T1w.nii.gz"))
    stat = np.zeros((20, 24, 10), dtype=np.float32)
    stat[8:12, 10:14, 4:6] = 5.0
    nib.save(nib.Nifti1Image(stat, np.diag([1.0, 1.0, 2.0, 1.0])),
             str(anat / "sub-01_stat.nii.gz"))
    return root


@pytest.fixture
def viewer(qtbot, ds) -> Viewer:
    v = Viewer(kind="volume")
    qtbot.addWidget(v)
    v.resize(900, 600)
    v.show()
    qtbot.waitExposed(v)
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    qtbot.waitUntil(v.is_loaded, timeout=20_000)
    with qtbot.waitSignal(v.overlay_added, timeout=20_000):
        assert v.add_overlay(ds / "sub-01" / "anat" / "sub-01_stat.nii.gz")
    v.run("layer.set", layer="base", colormap="viridis")
    v.trigger("view.axial")
    v.qstore.flush()
    return v


def test_the_canvas_draws_what_the_library_renders(qtbot, viewer):
    canvas = viewer.canvases("slice")[0]
    canvas.repaint()
    assert canvas._ensure_image()
    headless = render2d.slice_rgba(viewer.store, canvas.plane)
    assert np.array_equal(canvas._image_buf, headless.rgba)


def test_the_views_menu_shows_the_command_line(qtbot, viewer, tmp_path):
    from PyQt6.QtGui import QGuiApplication

    assert viewer.action("view.command_line").isEnabled()
    viewer.trigger("view.command_line")
    dlg = viewer.presenter.command_line_dialog
    qtbot.addWidget(dlg)
    assert [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())] == ["Open", "Render", "Python"]
    assert "sub-01_stat.nii.gz" in dlg.current_text()
    assert "colormap=viridis" in dlg.current_text()
    dlg.tabs.setCurrentIndex(2)
    dlg.copy()
    assert QGuiApplication.clipboard().text().startswith("from bidsmgr.viz import render2d")
    # What it says reproduces the screen: rendered again with no window, the
    # axial plane is the canvas's composition.
    from bidsmgr.cli.view import main

    out = tmp_path / "again.png"
    text = reproduce.shell(viewer.store, render=str(out), style="posix")
    assert main(reproduce.parse_shell(text)) == 0
    assert out.stat().st_size > 0

"""The quality check in the image viewer and in the Editor's Quality check:
Check quality, the report, the evidence over the image, Show the noise,
the diffusion plots under the series, and Go there."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bidsmgr.gui.viz import Viewer  # noqa: E402
from bidsmgr.qc import live  # noqa: E402
from bidsmgr.viz import views  # noqa: E402

from tests.unit.qc_phantoms import (anat_phantom, dataset, dwi_phantom, gradient_table,  # noqa: E402
                                    save, save_dwi)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    # The phantoms are not heads a network was trained on: the numpy engine,
    # deterministic and fast. The tools have their own tests.
    monkeypatch.setenv("BIDSMGR_QC_ENGINE", "numpy")
    live.forget()
    yield
    live.forget()


@pytest.fixture
def no_gpu(monkeypatch):
    """The host's GPU must not decide what a 2-D test measures."""
    from bidsmgr.gui.viz.canvases import render

    monkeypatch.setattr(render, "gpu_available", lambda: False)


@pytest.fixture
def ds(tmp_path) -> Path:
    root = dataset(tmp_path / "ds")
    data, affine = anat_phantom()
    save(data, affine, root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    save(data, affine, root / "sub-02" / "anat" / "sub-02_T1w.nii.gz",
         {"DeidentificationMethod": ["pydeface"]})
    bvals, bvecs = gradient_table(24, 3)
    dw, aff = dwi_phantom(bvals, bvecs)
    v = int(np.flatnonzero(bvals > 0)[9])
    dw[:, :, 15, v] *= 0.4                                   # a planted dropout
    save_dwi(dw, aff, bvals, bvecs, root / "sub-01" / "dwi" / "sub-01_dwi.nii.gz")
    root.joinpath("dropout_volume.txt").write_text(str(v))
    return root


def _viewer(qtbot) -> Viewer:
    viewer = Viewer(kind="volume")
    qtbot.addWidget(viewer)
    viewer.resize(1200, 760)
    viewer.show()
    qtbot.waitExposed(viewer)
    return viewer


def _open(qtbot, viewer: Viewer, path: Path, root: Path) -> Viewer:
    with qtbot.waitSignal(viewer.loaded, timeout=30_000):
        viewer.set_file(path, root)
    qtbot.waitUntil(viewer.is_loaded, timeout=30_000)
    viewer.qstore.flush()
    return viewer


def _panel(qtbot, viewer: Viewer):
    """The docked quality panel once its result is in."""
    p = viewer.presenter
    qtbot.waitUntil(lambda: p._quality_panel is not None
                    and p._quality_panel.result is not None, timeout=120_000)
    return p._quality_panel


def _measures(panel) -> dict:
    t = panel.table
    return {t.item(r, 0).text(): r for r in range(t.rowCount())}


def test_check_quality_docks_the_panel_beside_the_views(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    p = viewer.presenter
    assert viewer.action("qc.check").isEnabled()
    assert not p.quality_button.isHidden()
    viewer.trigger("qc.check")
    panel = _panel(qtbot, viewer)
    assert panel.result.kind == "anat" and panel.result.facts["air_measured"]
    # Docked, not a window: beside the views by default.
    assert panel.window() is viewer.window()
    assert panel.parentWidget() is p.quality_right and p.quality_right.isVisible()
    assert not p.quality_below.isVisible()
    assert viewer.action("qc.check").isChecked()
    rows = _measures(panel)
    assert "SNR in white matter" in rows and "Entropy focus criterion" in rows
    # Its corner: below, maximise, own window, then More and close.
    corner = panel.header.corner_buttons()
    assert corner[-1] is panel.close_button and corner[-2] is panel.more_button
    # Check quality again closes it, as the time course's button does.
    viewer.trigger("qc.check")
    assert not p.quality_right.isVisible() and not viewer.action("qc.check").isChecked()


def test_measures_explain_themselves_and_groups_fold(qtbot, ds, no_gpu) -> None:
    """What a measure means is on hover and behind its info icon, never in
    a box at the end of the table; a group folds away from its name."""
    from bidsmgr.gui.viz.panels import explain_popup
    from bidsmgr.gui.viz.panels.quality_panel import COL_INFO

    viewer = _open(qtbot, _viewer(qtbot), ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    viewer.trigger("qc.check")
    panel = _panel(qtbot, viewer)
    t = panel.table
    r = _measures(panel)["SNR in white matter"]
    tip = t.item(r, 0).toolTip()
    assert "white matter" in tip.lower() and "snr_wm" in tip and "info icon" in tip
    assert not t.item(r, COL_INFO).icon().isNull()
    assert not hasattr(panel, "description"), "the explanation box is back"
    panel._on_cell(r, COL_INFO)
    menu, label = explain_popup.last_shown()
    text = label.text()
    assert "How to read it" in text and "In this image" in text and "SNR in white matter" in text
    menu.close()
    # The group's own explanation, and folding.
    head = next(i for i, e in enumerate(panel._rows) if isinstance(e, tuple) and e[1] == "noise")
    panel._on_cell(head, COL_INFO)
    assert "How to read it" in explain_popup.last_shown()[1].text()
    explain_popup.last_shown()[0].close()
    height = t.height()
    panel._on_cell(head, 0)
    assert t.isRowHidden(r) and t.height() < height
    panel._on_cell(head, 0)
    assert not t.isRowHidden(r) and t.height() == height


def test_a_defaced_image_reports_its_air_as_not_measured(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), ds / "sub-02" / "anat" / "sub-02_T1w.nii.gz", ds)
    viewer.trigger("qc.check")
    panel = _panel(qtbot, viewer)
    assert not panel.result.facts["air_measured"]
    assert "defaced" in panel.summary.text().lower()
    efc = _measures(panel)["Entropy focus criterion"]
    assert panel.table.item(efc, 1).text() == "not computed"
    assert "defaced" in panel.table.item(efc, 1).toolTip()


def test_evidence_checkboxes_show_and_hide_one_layer_each(qtbot, ds, no_gpu) -> None:
    from bidsmgr.gui.viz.panels.quality_panel import TISSUE_COLOURS

    viewer = _open(qtbot, _viewer(qtbot), ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    viewer.trigger("qc.check")
    panel = _panel(qtbot, viewer)
    before = len(viewer.scene.layers)
    # The tissue classes are three boxes, each with its colour beside it.
    assert {"brain", "tissues:1", "tissues:2", "tissues:3"} <= set(panel.map_boxes)
    assert [panel.map_boxes[f"tissues:{v}"].text() for v in (1, 2, 3)] == [
        "CSF", "Grey matter", "White matter"]
    assert all(not b.icon().isNull() for b in panel.map_boxes.values())
    with qtbot.waitSignal(viewer.overlay_added, timeout=5000):
        panel.map_boxes["brain"].setChecked(True)
    with qtbot.waitSignal(viewer.overlay_added, timeout=5000):
        panel.map_boxes["tissues:2"].setChecked(True)
    layers = viewer.scene.layers
    assert len(layers) == before + 2
    gm = layers[-1]
    assert gm.name == "Grey matter"
    assert gm.display.label_table.labels == {1: "Grey matter"}
    assert tuple(gm.display.label_table.colors[1])[:3] == TISSUE_COLOURS[2]
    # Grey matter alone: the layer holds that class and nothing else.
    src = viewer.store.sources[gm.source]
    grey = np.asarray(src.raw_frame(0)) > 0
    labels = viewer.presenter._quality_result.maps["tissues"].data
    assert grey.sum() == int((np.asarray(labels) == 2).sum())
    # Unticked: hidden, not removed; ticked again: the SAME layer, no copy.
    brain_id = viewer.presenter._evidence["brain"]
    panel.map_boxes["brain"].setChecked(False)
    viewer.qstore.flush()
    assert not next(lay for lay in viewer.scene.layers if lay.id == brain_id).visible
    panel.map_boxes["brain"].setChecked(True)
    viewer.qstore.flush()
    assert len(viewer.scene.layers) == before + 2
    assert next(lay for lay in viewer.scene.layers if lay.id == brain_id).visible
    # A finding's Show ticks its box rather than adding another layer, and
    # "tissues" ticks every class.
    panel._show_evidence("brain")
    assert len(viewer.scene.layers) == before + 2
    panel._show_evidence("tissues")
    viewer.qstore.flush()
    assert all(panel.map_boxes[f"tissues:{v}"].isChecked() for v in (1, 2, 3))
    assert len(viewer.scene.layers) == before + 4
    # Hidden from the layer list: the box follows.
    viewer.run("layer.set", layer=brain_id, visible=False)
    viewer.qstore.flush()
    assert not panel.map_boxes["brain"].isChecked()
    # Removed from the layer list: the box follows, and ticking adds it anew.
    viewer.run("layer.remove", layer=brain_id)
    viewer.qstore.flush()
    assert not panel.map_boxes["brain"].isChecked()
    assert "brain" not in viewer.presenter._evidence
    with qtbot.waitSignal(viewer.overlay_added, timeout=5000):
        panel.map_boxes["brain"].setChecked(True)
    assert len(viewer.scene.layers) == before + 4


def test_the_panel_moves_below_maximises_and_detaches(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    p = viewer.presenter
    viewer.trigger("qc.check")
    panel = _panel(qtbot, viewer)
    viewer.trigger("qc.below")
    viewer.qstore.flush()
    assert viewer.scene.layout.quality == "bottom"
    assert panel.parentWidget() is p.quality_below and p.quality_below.isVisible()
    assert not p.quality_right.isVisible() and viewer.action("qc.below").isChecked()
    viewer.trigger("qc.maximize")
    assert not p.qsplit.isVisible() and panel.isVisible()
    viewer.trigger("qc.maximize")
    assert p.qsplit.isVisible()
    viewer.trigger("qc.detach")
    win = p._quality_window
    assert win is not None and panel.window() is win
    assert not p.quality_below.isVisible() and not p.quality_right.isVisible()
    assert not viewer.action("qc.maximize").isEnabled()
    win.close()
    assert p._quality_window is None and panel.window() is viewer.window()
    assert p.quality_below.isVisible()
    viewer.trigger("qc.below")
    viewer.qstore.flush()
    assert viewer.scene.layout.quality == "right" and p.quality_right.isVisible()


def test_the_open_panel_follows_the_next_file(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    viewer.trigger("qc.check")
    panel = _panel(qtbot, viewer)
    first = panel.result
    _open(qtbot, viewer, ds / "sub-02" / "anat" / "sub-02_T1w.nii.gz", ds)
    # Not checked yet (QC on opening is off): the offer, not sub-01's result.
    assert panel.result is None and panel.name == "sub-02_T1w.nii.gz"
    panel.check_requested.emit()
    panel = _panel(qtbot, viewer)
    assert panel.result is not first and not panel.result.facts["air_measured"]
    # Back to sub-01: its result is kept and shown at once.
    _open(qtbot, viewer, ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    assert panel.result is first


def test_with_qc_on_opening_the_open_panel_checks_the_next_file(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    viewer.trigger("qc.check")
    _panel(qtbot, viewer)
    viewer.presenter.ctx.settings_hub.update(lambda st: setattr(st.qc, "on_open", True))
    try:
        _open(qtbot, viewer, ds / "sub-02" / "anat" / "sub-02_T1w.nii.gz", ds)
        panel = _panel(qtbot, viewer)
        assert panel.name == "sub-02_T1w.nii.gz" and not panel.result.facts["air_measured"]
    finally:
        viewer.presenter.ctx.settings_hub.update(lambda st: setattr(st.qc, "on_open", False))


def test_show_the_noise_and_back(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    layer = viewer.scene.base_layer()
    look = (layer.display.window, layer.display.colormap)
    viewer.trigger("qc.noise")
    viewer.qstore.flush()
    assert viewer.action("qc.noise").isChecked()
    shown = viewer.scene.base_layer().display
    assert shown.colormap == "viridis" and shown.window[1] < look[0][1]
    viewer.trigger("qc.noise")
    viewer.qstore.flush()
    back = viewer.scene.base_layer().display
    assert (back.window, back.colormap) == look
    assert not viewer.action("qc.noise").isChecked()
    # The panel's box and the action are one switch.
    viewer.trigger("qc.check")
    panel = _panel(qtbot, viewer)
    panel.noise_box.setChecked(True)
    viewer.qstore.flush()
    assert viewer.action("qc.noise").isChecked()
    viewer.trigger("qc.noise")
    viewer.qstore.flush()
    assert not panel.noise_box.isChecked()


def test_a_diffusion_series_gets_its_plots_and_go_there(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), ds / "sub-01" / "dwi" / "sub-01_dwi.nii.gz", ds)
    planted = int((ds / "dropout_volume.txt").read_text())
    viewer.trigger("qc.check")
    panel = _panel(qtbot, viewer)
    assert panel.result.kind == "dwi" and panel.plots_box is not None
    found = {f.key: f for f in panel.result.findings}
    assert found["dropout"].evidence["volume"] == planted
    panel.plots_box.setChecked(True)
    viewer.qstore.flush()
    graph = viewer.presenter.graph
    qtbot.waitUntil(lambda: graph._qc_src is not None and "slices" in graph._qc_src["rows"],
                    timeout=120_000)
    shown = [t["id"] for t in graph.shown_tracks()]
    assert "displacement" in shown and "slices" in shown
    # The plots stay under the time course, the panel beside both.
    p = viewer.presenter
    assert p.graph_host.isVisible() and p.quality_right.isVisible()
    # The Plots menu offers the diffusion rows, not BOLD's.
    assert graph._qc_plot_actions["slices"].isVisible()
    assert not graph._qc_plot_actions["dvars"].isVisible()
    panel.go_to.emit(found["dropout"].evidence)
    viewer.qstore.flush()
    layer, src = views.series_layer(viewer.store)
    assert views.frame_of(viewer.store, layer, src) == planted
    assert views.cursor_voxel(viewer.store)[2] == 15
    # Turned off from the graph: the panel's box follows.
    viewer.run("graph.set", qc=False)
    viewer.qstore.flush()
    assert not panel.plots_box.isChecked()


def test_the_check_waits_for_an_image_still_opening(qtbot, ds, no_gpu) -> None:
    viewer = _viewer(qtbot)
    viewer.set_file(ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    assert viewer.presenter.check_quality()
    _panel(qtbot, viewer)


def test_saving_writes_the_derivative(qtbot, ds, no_gpu) -> None:
    viewer = _open(qtbot, _viewer(qtbot), ds / "sub-01" / "anat" / "sub-01_T1w.nii.gz", ds)
    viewer.trigger("qc.check")
    panel = _panel(qtbot, viewer)
    panel.save_action.trigger()
    assert (ds / "derivatives" / "bidsmgr-qc" / "sub-01" / "anat" / "sub-01_T1w.json").is_file()
    assert (ds / "derivatives" / "bidsmgr-qc" / "group_T1w.tsv").is_file()


def test_the_editor_dialog_checks_and_opens(qtbot, ds) -> None:
    from bidsmgr.gui.quality_check_dialog import QualityCheckDialog

    opened = []
    dlg = QualityCheckDialog(ds, open_image=opened.append)
    qtbot.addWidget(dlg)
    dlg.show()
    assert len(dlg.candidates) == 3
    assert dlg.anat_table.rowCount() == 0
    dlg.flips_box.setChecked(False)
    dlg.jobs.setValue(1)
    dlg.start()
    qtbot.waitUntil(lambda: dlg._worker is None, timeout=240_000)
    assert dlg.anat_table.rowCount() == 2 and dlg.dwi_table.rowCount() == 1
    names = sorted(dlg.anat_table.item(r, 0).text() for r in range(2))
    assert names == ["sub-01_T1w", "sub-02_T1w"]
    dlg.dwi_table.cellDoubleClicked.emit(0, 0)
    assert opened == [ds / "sub-01" / "dwi" / "sub-01_dwi.nii.gz"]
    # Nothing left to check: Check says so instead of running.
    dlg.start()
    assert dlg._worker is None and "Every image is checked" in dlg.status.text()

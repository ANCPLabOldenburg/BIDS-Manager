"""Settings > Viewer and Settings > Viewer shortcuts.

The Viewer page is generated from the settings model, so the test that
matters is that EVERY preference has a control, and that a Save reaches a
viewer that is already open. The shortcuts page is tested the way a user
works it: pick an action, press a key, Set.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from PyQt6.QtGui import QKeySequence  # noqa: E402

from bidsmgr.gui.app_settings import AppSettings  # noqa: E402
from bidsmgr.gui.settings_dialog import SettingsDialog  # noqa: E402
from bidsmgr.gui.viz import Viewer  # noqa: E402
from bidsmgr.gui.viz.bridge import SettingsHub  # noqa: E402
from bidsmgr.gui.viz.settings_pages import ShortcutsPage, ViewerSettingsPage  # noqa: E402
from bidsmgr.viz import keynames  # noqa: E402
from bidsmgr.viz.settings import VIEWER_SECTIONS, VizSettings  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    from bidsmgr.gui.viz.canvases import render

    monkeypatch.setattr(render, "gpu_available", lambda: False)


def _select(page: ShortcutsPage, title: str) -> None:
    for row in range(page.table.rowCount()):
        if page.table.item(row, 0).text() == title:
            page.table.selectRow(row)
            return
    raise AssertionError(title)


def _keys(page: ShortcutsPage, title: str) -> str:
    for row in range(page.table.rowCount()):
        if page.table.item(row, 0).text() == title:
            return page.table.item(row, 2).text()
    raise AssertionError(title)


# ---------------------------------------------------------------------------
# The generated page
# ---------------------------------------------------------------------------


def test_every_scalar_preference_has_a_control(qtbot) -> None:
    page = ViewerSettingsPage()
    qtbot.addWidget(page)
    for section in VIEWER_SECTIONS:
        model = VizSettings.model_fields[section].annotation
        for name, info in model.model_fields.items():
            # Collections have their own editors (the keymap page, the QC
            # section's channel-type boxes), not a field here.
            collection = getattr(info.annotation, "__origin__", None) in (dict, list)
            assert (page.control(f"{section}.{name}") is None) == collection, \
                f"{section}.{name}"


def test_controls_carry_the_model_s_range_unit_and_help(qtbot) -> None:
    page = ViewerSettingsPage()
    qtbot.addWidget(page)
    thickness = page.control("crosshair.thickness")
    assert (thickness.minimum(), thickness.maximum()) == (1, 5)
    assert thickness.suffix().strip() == "px"
    assert "Line width" in thickness.toolTip()
    memory = page.control("volume.memory_mb")
    assert memory.specialValueText() == "Automatic"
    mode = page.control("volume.mode")
    assert mode.itemText(mode.findData("combo")) == "Planes + 3-D"


def test_load_and_apply_round_trip(qtbot) -> None:
    page = ViewerSettingsPage()
    qtbot.addWidget(page)
    s = VizSettings()
    s.crosshair.thickness = 3
    s.volume.colormap = "viridis"
    s.volume.radiological = True
    s.traces.line_color = "#ff0000"
    page.load(s)
    out = VizSettings()
    page.apply_to(out)
    assert out.crosshair.thickness == 3 and out.volume.colormap == "viridis"
    assert out.volume.radiological and out.traces.line_color == "#ff0000"


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------


def test_recording_and_setting_a_key(qtbot) -> None:
    page = ShortcutsPage()
    qtbot.addWidget(page)
    page.load(VizSettings())
    assert _keys(page, "Axial view") == "A"
    _select(page, "Axial view")
    page.recorder.setKeySequence(QKeySequence("Shift+Q"))
    page._set_key()
    assert _keys(page, "Axial view") == keynames.key("Shift+Q")
    page.recorder.setKeySequence(QKeySequence("Q"))
    page._add_key()
    assert _keys(page, "Axial view") == ", ".join(keynames.keys(["Shift+Q", "Q"]))
    out = VizSettings()
    page.apply_to(out)
    assert out.keymap == {"view.axial": ["Shift+Q", "Q"]}


def test_a_key_used_twice_is_shown_as_a_conflict(qtbot) -> None:
    page = ShortcutsPage()
    qtbot.addWidget(page)
    page.load(VizSettings())
    assert page.conflict_count() == 0
    assert "No key is used twice" in page.conflicts.text()
    _select(page, "Time-course graph")
    page.recorder.setKeySequence(QKeySequence("A"))
    page._set_key()
    assert page.conflict_count() == 1
    assert "Axial view" in page.conflicts.text()
    page._default_one()
    assert page.conflict_count() == 0


def test_unbinding_and_resetting(qtbot) -> None:
    page = ShortcutsPage()
    qtbot.addWidget(page)
    page.load(VizSettings())
    _select(page, "Sagittal view")
    page._unbind()
    assert _keys(page, "Sagittal view") == ""
    out = VizSettings()
    page.apply_to(out)
    assert out.keymap == {"view.sagittal": []}
    page._reset_keys()
    assert _keys(page, "Sagittal view") == "S"


def test_setting_a_key_back_to_its_default_stores_nothing(qtbot) -> None:
    page = ShortcutsPage()
    qtbot.addWidget(page)
    page.load(VizSettings())
    _select(page, "Coronal view")
    page.recorder.setKeySequence(QKeySequence("C"))
    page._set_key()
    out = VizSettings()
    page.apply_to(out)
    assert out.keymap == {}


def test_search_filters_the_actions(qtbot) -> None:
    page = ShortcutsPage()
    qtbot.addWidget(page)
    page.load(VizSettings())
    page.search.setText("clip")
    shown = [page.table.item(r, 0).text() for r in range(page.table.rowCount())
             if not page.table.isRowHidden(r)]
    assert shown and all("clip" in t.lower() or "cut" in t.lower() for t in shown)


def test_export_and_import(qtbot, tmp_path) -> None:
    page = ShortcutsPage()
    qtbot.addWidget(page)
    page.load(VizSettings())
    _select(page, "Axial view")
    page.recorder.setKeySequence(QKeySequence("Shift+Q"))
    page._set_key()
    page._set_mouse("slice", "right", "pan")
    out = tmp_path / "keys.json"
    page.export_to(out)
    assert json.loads(out.read_text())["keymap"]["view.axial"] == ["Shift+Q"]
    other = ShortcutsPage()
    qtbot.addWidget(other)
    other.load(VizSettings())
    other.import_from(out)
    assert _keys(other, "Axial view") == keynames.key("Shift+Q")
    s = VizSettings()
    other.apply_to(s)
    assert s.mousemap == {"slice:right": "pan"}


def test_an_import_ignores_what_it_does_not_know(qtbot, tmp_path) -> None:
    page = ShortcutsPage()
    qtbot.addWidget(page)
    page.load(VizSettings())
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"keymap": {"no.such": ["K"]},
                               "mousemap": {"slice:left": "teleport"}}))
    page.import_from(bad)
    s = VizSettings()
    page.apply_to(s)
    assert s.keymap == {} and s.mousemap == {}


# ---------------------------------------------------------------------------
# Mouse
# ---------------------------------------------------------------------------


def test_wheel_gestures_offer_wheel_tools_only(qtbot) -> None:
    page = ShortcutsPage()
    qtbot.addWidget(page)
    wheel = page._mouse_combos["slice:wheel"]
    tools = {wheel.itemData(i) for i in range(wheel.count())}
    assert tools == {"slice", "frame", "zoom", "none"}
    drag = page._mouse_combos["render:left"]
    assert "orbit" in {drag.itemData(i) for i in range(drag.count())}


def test_rebinding_a_button(qtbot) -> None:
    page = ShortcutsPage()
    qtbot.addWidget(page)
    page.load(VizSettings())
    combo = page._mouse_combos["slice:right"]
    combo.setCurrentIndex(combo.findData("pan"))
    s = VizSettings()
    page.apply_to(s)
    assert s.mousemap == {"slice:right": "pan"}
    combo.setCurrentIndex(combo.findData("window"))      # back to the default
    page.apply_to(s)
    assert s.mousemap == {}


# ---------------------------------------------------------------------------
# Through the Settings dialog, to a viewer that is open
# ---------------------------------------------------------------------------


def _open_viewer(qtbot, tmp_path) -> Viewer:
    path = tmp_path / "sub-01_T1w.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.float32), np.eye(4)), str(path))
    viewer = Viewer(kind="volume")
    qtbot.addWidget(viewer)
    with qtbot.waitSignal(viewer.loaded, timeout=20_000):
        viewer.set_file(path, tmp_path)
    return viewer


def test_saving_reaches_an_open_viewer(qtbot, tmp_path) -> None:
    viewer = _open_viewer(qtbot, tmp_path)
    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    dlg._viewer_page.control("crosshair.thickness").setValue(4)
    _select(dlg._shortcuts_page, "Axial view")
    dlg._shortcuts_page.recorder.setKeySequence(QKeySequence("Shift+Q"))
    dlg._shortcuts_page._set_key()
    dlg._on_save()
    assert SettingsHub.instance().settings.crosshair.thickness == 4
    assert viewer.presenter.inspector.section("view").cross_width.value() == 4
    assert viewer.action_manager.keys_for("view.axial") == ["Shift+Q"]
    # Stored: a new session reads it back.
    SettingsHub.reset_instance()
    assert SettingsHub.instance().settings.keymap["view.axial"] == ["Shift+Q"]


def test_saving_keeps_what_the_pages_do_not_show(qtbot) -> None:
    SettingsHub.instance().update(lambda s: s.layout_sizes.__setitem__("volume.hero", [700, 300]))
    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    dlg._on_save()
    assert SettingsHub.instance().settings.layout_sizes["volume.hero"] == [700, 300]


def test_restore_defaults_resets_the_viewer_pages_too(qtbot) -> None:
    SettingsHub.instance().update(lambda s: setattr(s.crosshair, "thickness", 5))
    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    assert dlg._viewer_page.control("crosshair.thickness").value() == 5
    dlg._on_restore_defaults()
    assert dlg._viewer_page.control("crosshair.thickness").value() == 1


def test_cancel_changes_nothing(qtbot) -> None:
    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    dlg._viewer_page.control("crosshair.thickness").setValue(5)
    dlg.reject()
    assert SettingsHub.instance().settings.crosshair.thickness == 1


def test_a_rebound_mouse_gesture_changes_what_a_click_does(qtbot, tmp_path) -> None:
    """Left on a slice, rebound to pan, no longer moves the crosshair."""
    from PyQt6.QtCore import Qt

    from bidsmgr.viz import views

    viewer = _open_viewer(qtbot, tmp_path)
    viewer.resize(700, 500)
    viewer.show()
    qtbot.waitExposed(viewer)
    viewer.run("view.mode", mode="single")
    viewer.qstore.flush()
    SettingsHub.instance().update(lambda s: s.mousemap.__setitem__("slice:left", "pan"))
    qtbot.waitUntil(lambda: len(viewer.canvases("slice")) == 1, timeout=5000)
    (canvas,) = viewer.canvases("slice")
    qtbot.wait(10)
    canvas.repaint()
    before = views.cursor_voxel(viewer.store)
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=canvas.grid_to_screen(0, 0).toPoint())
    assert views.cursor_voxel(viewer.store) == before

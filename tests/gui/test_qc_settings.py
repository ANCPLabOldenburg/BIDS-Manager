"""Settings: the pages listed down the side, and Settings > Quality control.

The Quality control page is generated from ``bidsmgr.qc.config`` (through
the viewer settings), so the tests that matter: every setting has a
control, a dependent setting follows its switch, a group restores its own
defaults, and a Save reaches the checks. The dialog must shrink from the
sides: its page list turns to icons and nothing sets a floor.
"""

from __future__ import annotations

import pytest

from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.settings_dialog import SettingsDialog
from bidsmgr.gui.viz import fonts
from bidsmgr.gui.viz.bridge import SettingsHub
from bidsmgr.gui.viz.settings_pages import QualitySettingsPage
from bidsmgr.viz.settings import QC_SECTIONS, VizSettings, qc_config

pytestmark = pytest.mark.gui


def test_every_quality_setting_has_a_control(qtbot) -> None:
    page = QualitySettingsPage()
    qtbot.addWidget(page)
    for section in QC_SECTIONS:
        model = VizSettings.model_fields[section].annotation
        for name, info in model.model_fields.items():
            collection = getattr(info.annotation, "__origin__", None) in (dict, list)
            assert (page.control(f"{section}.{name}") is None) == collection, \
                f"{section}.{name}"


def test_controls_carry_the_configuration_s_range_and_help(qtbot) -> None:
    page = QualitySettingsPage()
    qtbot.addWidget(page)
    fd = page.control("qc_bold.fd_threshold_mm")
    assert (fd.minimum(), fd.maximum()) == (0.05, 5.0) and fd.suffix().strip() == "mm"
    assert "framewise displacement" in fd.toolTip()
    brain = page.control("qc_methods.brain")
    assert brain.itemText(brain.findData("mindgrab")).startswith("mindgrab")
    detrend = page.control("qc_bold.detrend")
    assert detrend.itemText(detrend.findData(2)) == "Quadratic"


def test_a_dependent_setting_follows_its_switch(qtbot) -> None:
    page = QualitySettingsPage()
    qtbot.addWidget(page)
    page.load(VizSettings())
    page.control("qc_bold.skip_nonsteady").setChecked(False)
    assert not page.control("qc_bold.nonsteady_z").isEnabled()
    page.control("qc_bold.skip_nonsteady").setChecked(True)
    assert page.control("qc_bold.nonsteady_z").isEnabled()
    page.control("qc_dwi.check_flips").setChecked(False)
    assert not page.control("qc_dwi.flip_margin_pct").isEnabled()


def test_a_group_restores_its_own_defaults(qtbot) -> None:
    page = QualitySettingsPage()
    qtbot.addWidget(page)
    page.load(VizSettings())
    page.control("qc_bold.fd_threshold_mm").setValue(0.2)
    page.control("qc_dwi.spike_z").setValue(11.0)
    page.restore_section("qc_bold")
    assert page.control("qc_bold.fd_threshold_mm").value() == 0.5
    assert page.control("qc_dwi.spike_z").value() == 11.0, "another group was reset"


def test_the_page_says_whether_the_tools_can_run(qtbot, monkeypatch) -> None:
    from bidsmgr.qc import tools

    monkeypatch.setattr(tools, "unavailable_reason", lambda: "niimath is not installed")
    page = QualitySettingsPage()
    qtbot.addWidget(page)
    assert "niimath is not installed" in page.tools_text.text()
    assert "fast methods" in page.tools_text.text()


def test_the_dialog_lists_its_pages_down_the_side(qtbot) -> None:
    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    titles = dlg.page_titles()
    assert titles[:3] == ["BIDS version", "Display", "System"]
    assert {"Scan rules", "Convert", "Validation", "Quality control", "Viewer",
            "Viewer shortcuts"} <= set(titles)
    dlg.show_page("Quality control")
    assert dlg.current_page() == "Quality control"
    assert dlg._stack.currentWidget().isAncestorOf(dlg._qc_page)


def test_saving_a_quality_setting_reaches_the_checks(qtbot) -> None:
    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    dlg._qc_page.control("qc_bold.fd_threshold_mm").setValue(0.3)
    dlg._qc_page.control("qc_methods.tissues").setCurrentIndex(
        dlg._qc_page.control("qc_methods.tissues").findData("em"))
    dlg._on_save()
    cfg = qc_config(SettingsHub.instance().settings)
    assert cfg.bold.fd_threshold_mm == 0.3 and cfg.methods.tissues == "em"
    # Restore defaults covers the page too.
    again = SettingsDialog(AppSettings.load())
    qtbot.addWidget(again)
    assert again._qc_page.control("qc_bold.fd_threshold_mm").value() == 0.3
    again._on_restore_defaults()
    assert again._qc_page.control("qc_bold.fd_threshold_mm").value() == 0.5


def test_the_dialog_shrinks_from_the_sides(qtbot) -> None:
    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    wide = dlg._nav.width()
    assert dlg._nav.item(0).text() == "BIDS version"
    # Every page lets the dialog go narrow: its content scrolls or wraps.
    floor = fonts.px(480)
    for title in dlg.page_titles():
        dlg.show_page(title)
        assert dlg.minimumSizeHint().width() <= floor, title
    dlg.resize(floor, dlg.height())
    qtbot.waitUntil(lambda: dlg._nav.item(0).text() == "", timeout=2000)
    assert dlg._nav.width() < wide / 2, "the page list did not turn to icons"
    assert dlg._nav.item(0).toolTip() == "BIDS version"
    dlg.resize(fonts.px(900), dlg.height())
    qtbot.waitUntil(lambda: dlg._nav.item(0).text() == "BIDS version", timeout=2000)

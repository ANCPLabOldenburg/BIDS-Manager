"""The "What's new" dialog: shown once per release, painted, theme-aware."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bidsmgr.gui import whats_new as wn


def test_shown_until_this_releases_news_has_been_seen() -> None:
    assert wn.should_show("")             # a first install
    assert wn.should_show("1.4")          # an update from an older release
    assert not wn.should_show(wn.NEWS_FOR)
    assert not wn.should_show(f" {wn.NEWS_FOR} ")


def test_maybe_show_opens_it_once(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QWidget

    from bidsmgr.gui.app_settings import AppSettings

    host = QWidget()
    qtbot.addWidget(host)
    first = wn.maybe_show(host)
    assert isinstance(first, wn.WhatsNewDialog)
    assert first.parent() is host, "a parentless dialog would be a stray window"
    assert AppSettings.whats_new_seen() == wn.NEWS_FOR
    first.reject()
    assert wn.maybe_show(host) is None, "shown again after it was seen"


def test_building_the_window_never_opens_it(qtbot, isolated_settings) -> None:
    """The dialog is scheduled by main(), so tests that build a window get none."""
    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager
    from PyQt6.QtWidgets import QApplication

    theme = ThemeManager(QApplication.instance())
    theme.apply("dark")
    win = MainWindow(theme)
    qtbot.addWidget(win)
    qtbot.wait(700)
    assert not win.findChildren(wn.WhatsNewDialog)


def test_the_particles_gather_into_the_logo_and_stay() -> None:
    parts = wn.Particles(per_kind=60)
    assert parts.n == 60 * len(wn.KINDS)
    start = parts.at(0.0)
    assert np.isfinite(start).all()
    assert np.allclose(start, parts.figure_points(0.0))
    settled = parts.at(wn.SETTLED)
    assert np.allclose(settled, parts.target, atol=1e-9), "not all particles arrived"
    later = parts.at(wn.SETTLED + 30.0)
    assert np.abs(later - parts.target).max() < 0.01, "the logo drifts instead of breathing"
    mid = parts.at(wn.GATHER_AT + wn.GATHER / 2)
    assert np.isfinite(mid).all()


def test_the_figures_labels_fade_in_hold_and_go() -> None:
    assert wn.Particles.figures_alpha(0.0) == 0.0
    assert wn.Particles.figures_alpha(wn.APPEAR) == 1.0
    assert wn.Particles.figures_alpha(wn.GATHER_AT - 0.01) == 1.0
    assert wn.Particles.figures_alpha(wn.SETTLED) == 0.0


def test_seven_distinct_colours_in_every_theme() -> None:
    """Seven kinds of data, seven colours a reader can tell apart, each
    standing out from the theme's background, in all nineteen themes."""
    from bidsmgr.gui import theme_manager as tm
    from bidsmgr.gui.tones import contrast, oklch
    from bidsmgr.viz.theme import VizTheme

    def lab(c):
        light, chroma, hue = oklch(c)
        return light, chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue))

    for theme_id, palette in tm.PALETTES.items():
        theme = VizTheme.from_palette(palette, theme_id)
        tokens = {**theme.tokens, **{f"series{i + 1}": theme.series(i) for i in range(6)}}
        colours = wn.seven_colours(tokens)
        assert len(colours) == 7 and len(set(colours)) == 7, theme_id
        closest = min(math.dist(lab(a), lab(b))
                      for i, a in enumerate(colours) for b in colours[i + 1:])
        assert closest >= 0.06, f"{theme_id}: two kinds share a colour ({closest:.3f})"
        assert min(contrast(c, palette["bg"]) for c in colours) >= 3.0, theme_id


@pytest.mark.parametrize("theme_id", ["dark", "light", "hc-dark"])
def test_the_dialog_paints_and_stops_when_hidden(qtbot, isolated_settings, theme_id) -> None:
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.theme_manager import ThemeManager

    ThemeManager(QApplication.instance()).apply(theme_id)
    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.canvas.is_running()
    for t in (0.0, 1.0, wn.GATHER_AT + 0.7, wn.SETTLED + 3.0):
        dialog.canvas.set_time(t)
        assert not dialog.canvas.grab().isNull()
    dialog.hide()
    assert not dialog.canvas.is_running(), "the animation keeps running behind a closed dialog"


def test_the_buttons_open_the_documentation(qtbot, isolated_settings, monkeypatch) -> None:
    opened = []
    monkeypatch.setattr(wn.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    dialog._buttons["docs"].click()
    dialog._buttons["notes"].click()
    assert opened == [wn.DOCS_URL, wn.NOTES_URL]
    assert wn.DOCS_URL.startswith("https://ancplaboldenburg.github.io/bids_manager_documentation/")
    assert wn.NOTES_URL.endswith("/updates/#v150")
    assert dialog._buttons["continue"].isDefault()
    dialog._buttons["continue"].click()
    assert dialog.result() == dialog.DialogCode.Accepted


def test_the_acquisition_parameter_naming_is_said_to_be_for_mri(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QLabel

    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    texts = " ".join(l.text() for l in dialog.findChildren(QLabel))
    assert "MRI series named from their acquisition parameters" in texts


def test_show_dialog_deletes_the_dialog_when_closed(qtbot, isolated_settings) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    dialog = wn.show_dialog(host)
    assert dialog.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.reject()


def test_the_classifier_settings_credit_the_swiftcat_team(qtbot, isolated_settings) -> None:
    """Settings > Classification says the parameters are for MRI, and credits
    the SWIFTCAT team, behind an info icon at the top of the page."""
    from bidsmgr.classifier.acquisition.config import ClassifierConfig
    from bidsmgr.gui import classifier_rules
    from bidsmgr.gui.viz.panels import explain_popup

    page = classifier_rules.ClassifierSettingsPage(("general",), root=ClassifierConfig,
                                                   hint="A hint.")
    qtbot.addWidget(page)
    page.about_button.click()
    _menu, label = explain_popup.last_shown()
    text = label.text()
    for needed in CREDIT_WORDS:
        assert needed in text, needed
    _menu.close()


#: What the note must say, wherever it is opened.
CREDIT_WORDS = ("For MRI only", "SWIFTCAT", "Marta Loureiro", "Rita G. Nunes",
                "Rita Reis Nunes", "LaSEEB", "University of Lisbon", "Patricia Clement",
                "UZ Gent", "dcm2niix", "early implementation", "works well overall",
                "clinical and heterogeneous data", "stay tuned")


def test_the_whats_new_dialog_opens_the_same_credit(qtbot, isolated_settings) -> None:
    """The note behind Settings > Classification's info icon is in the What's
    new dialog too, under the new features."""
    from bidsmgr.gui.viz.panels import explain_popup

    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    dialog.credit_button.click()
    menu, label = explain_popup.last_shown()
    for needed in CREDIT_WORDS:
        assert needed in label.text(), needed
    menu.close()

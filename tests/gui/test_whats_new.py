"""The "What's new" dialog: the news of the releases a user has not had,
one page each, painted in the theme's colours.

Which news a start shows (the version the user comes from), the walk through
them, the release menu and Home's; the 1.5.0 page (the seven kinds of data
gathering into the logo, the SWIFTCAT credit); and the 1.5.1 and 1.5.2 page
(a scene per feature, its words and values taken from the code that owns
them).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from bidsmgr.gui import news_scenes as ns
from bidsmgr.gui import whats_new as wn


# ---------------------------------------------------------------------------
# Which news, for whom
# ---------------------------------------------------------------------------


def test_versions_are_compared_as_numbers() -> None:
    assert wn.version_tuple("1.5.2") == (1, 5, 2)
    assert wn.version_tuple("1.5") == (1, 5, 0)
    assert wn.version_tuple("1.5.2.dev3+g1234") == (1, 5, 2)
    assert wn.version_tuple("1.10.0") > wn.version_tuple("1.9.9")
    assert [n.key for n in wn.NEWS] == sorted((n.key for n in wn.NEWS), key=wn.version_tuple,
                                              reverse=True), "newest first"


def test_the_version_a_user_comes_from() -> None:
    previous = wn.previous_version
    assert previous(last="1.5.2", seen="1.5", used_before=True) == "1.5.2", "the record wins"
    assert previous(last="", seen="1.5", used_before=True) == "1.5.0", \
        "the 1.5 news was shown by 1.5.0 and 1.5.1: at the latest 1.5.0"
    assert previous(last="", seen="", used_before=True) == "1.4", \
        "settings without a record: a release from before What's new"
    assert previous(last="", seen="", used_before=False) is None, "a first installation"


@pytest.mark.parametrize("previous, keys", [
    (None, ["1.5.2"]),                 # a first installation: the newest alone
    ("1.4", ["1.5.2", "1.5.0"]),       # everything since
    ("1.4.1", ["1.5.2", "1.5.0"]),
    ("1.5.0", ["1.5.2"]),
    ("1.5.1", ["1.5.2"]),
    ("1.5.2", []),                     # nothing new: no dialog
    ("1.6.0", []),                     # a newer version that ran here first
])
def test_the_news_due(previous, keys) -> None:
    assert [n.key for n in wn.due(previous)] == keys


def test_a_page_leaves_out_the_releases_the_user_had() -> None:
    news = wn.NEWS[0]
    assert wn.releases_new_to(news, None) == ("1.5.1", "1.5.2")
    assert wn.releases_new_to(news, "1.5.0") == ("1.5.1", "1.5.2")
    assert wn.releases_new_to(news, "1.5.1") == ("1.5.2",)


def test_a_first_start_shows_the_newest_once(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QWidget

    from bidsmgr import __version__
    from bidsmgr.gui.app_settings import AppSettings

    host = QWidget()
    qtbot.addWidget(host)
    assert wn.read_previous() is None, "isolated settings are a first installation"
    first = wn.maybe_show(host)
    assert isinstance(first, wn.WhatsNewDialog)
    assert first.parent() is host, "a parentless dialog would be a stray window"
    assert first.news().key == "1.5.2"
    assert first._buttons["continue"].text() == "Continue", "one news: nothing to walk to"
    assert AppSettings.last_version() == __version__
    first.reject()
    assert wn.maybe_show(host) is None, "shown again after it was seen"


def test_coming_from_14_walks_through_every_news(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QWidget

    from bidsmgr.gui.app_settings import AppSettings

    AppSettings.remember_guided_mode(False)        # any setting: it ran here before
    assert wn.read_previous() == "1.4"
    host = QWidget()
    qtbot.addWidget(host)
    dialog = wn.maybe_show(host)
    assert dialog.news().key == "1.5.2"
    page = dialog.page()
    assert isinstance(page, wn.ScenesPage)
    assert [s.version for s in page.stage.scenes].count("1.5.1") == 3, "both releases listed"
    main = dialog._buttons["continue"]
    assert main.text() == "Next: what 1.5.0 brought"
    main.click()
    assert dialog.news().key == "1.5.0" and isinstance(dialog.page(), wn.GatherPage)
    assert main.text() == "Continue"
    main.click()
    assert dialog.result() == dialog.DialogCode.Accepted


def test_coming_from_150_shows_the_new_releases(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QLabel, QWidget

    from bidsmgr.gui.app_settings import AppSettings

    AppSettings.remember_whats_new_seen("1.5")      # what 1.5.0 and 1.5.1 recorded
    host = QWidget()
    qtbot.addWidget(host)
    dialog = wn.maybe_show(host)
    groups = [label.text() for label in dialog.page().findChildren(QLabel)
              if label.objectName() == "whatsnew-group"]
    assert groups == ["New in 1.5.2", "New in 1.5.1"]
    assert dialog._buttons["continue"].text() == "Continue", "1.5.0's news is not due"
    dialog.reject()


def test_coming_from_151_leaves_its_features_out(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QLabel, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    dialog = wn.maybe_show(host, "1.5.1")
    page = dialog.page()
    groups = [label.text() for label in page.findChildren(QLabel)
              if label.objectName() == "whatsnew-group"]
    assert groups == ["New in 1.5.2"]
    assert {s.version for s in page.stage.scenes} == {"1.5.2"}
    titles = [label.text() for label in page.findChildren(QLabel)
              if label.objectName() == "whatsnew-lede"]
    assert titles == ["What version 1.5.2 brings. Pick one to see it again."]
    dialog.reject()


def test_building_the_window_never_opens_it(qtbot, isolated_settings) -> None:
    """The dialog is scheduled by main(), so tests that build a window get none."""
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager

    theme = ThemeManager(QApplication.instance())
    theme.apply("dark")
    win = MainWindow(theme)
    qtbot.addWidget(win)
    qtbot.wait(700)
    assert not win.findChildren(wn.WhatsNewDialog)


def test_the_release_menu_opens_any_news(qtbot, isolated_settings, monkeypatch) -> None:
    """Every news is in the menu, the shown one ticked; Release notes opens
    the shown one's card on the Updates page."""
    opened = []
    monkeypatch.setattr(wn.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    actions = dialog.release_menu.actions()
    assert [a.text() for a in actions] == ["What's new in 1.5.1 and 1.5.2", "What's new in 1.5.0"]
    assert actions[0].isChecked() and not actions[1].isChecked()
    assert dialog.release_button.text() == "What's new in 1.5.1 and 1.5.2"
    dialog._buttons["notes"].click()
    actions[1].trigger()
    assert dialog.news().key == "1.5.0" and actions[1].isChecked()
    dialog._buttons["notes"].click()
    dialog._buttons["docs"].click()
    assert opened == [wn.DOCS_URL + "updates/#v152", wn.DOCS_URL + "updates/#v150", wn.DOCS_URL]
    assert wn.DOCS_URL.startswith("https://ancplaboldenburg.github.io/bids_manager_documentation/")
    assert dialog._buttons["continue"].text() == "Continue", "opened by hand: no walk"
    assert dialog._buttons["continue"].isDefault()


def test_show_dialog_opens_on_the_news_asked(qtbot, isolated_settings) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    dialog = wn.show_dialog(host, start="1.5.0")
    assert dialog.news().key == "1.5.0"
    assert dialog.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.reject()


# ---------------------------------------------------------------------------
# The 1.5.1 and 1.5.2 news: what it says
# ---------------------------------------------------------------------------


def test_it_lists_what_151_and_152_brought() -> None:
    keys = [s.key for s in ns.SCENES]
    assert keys == ["guide", "physio", "dummy", "names", "sidecars", "derivatives",
                    "undo", "anonymise", "ignore"]
    versions = [s.version for s in ns.SCENES]
    assert versions == sorted(versions, reverse=True), "newest release first, grouped"
    assert set(versions) == set(wn.NEWS[0].releases)
    for scene in ns.SCENES:
        for words in (scene.title, scene.text, scene.shows):
            assert words and "—" not in words and "–" not in words, scene.key
        assert 6.0 <= scene.duration <= 12.0, scene.key
    assert "—" not in ns.ALSO and "Windows" in ns.ALSO


def test_no_scene_shows_a_persons_name() -> None:
    """A tool that keeps names out of datasets shows none: the patient name
    field holds a study code, as MRI sites type there."""
    import inspect

    source = inspect.getsource(ns)
    for name in ("Müller", "Muller", "Anna", "Smith"):
        assert name not in source, name
    assert '("PatientName", "MOT_017")' in source


def test_the_words_shown_are_the_apps_own() -> None:
    """A scene shows what the app does, in its words: the guide's real first
    question, the menu entries and the anonymiser's real output."""
    import inspect

    from bidsmgr.anonymise.policy import CATEGORIES, PURPOSES
    from bidsmgr.editor.sidecars import ORIGINS
    from bidsmgr.gui.guide_steps import find_task
    from bidsmgr.gui.widgets import bids_tree_pane

    source = inspect.getsource(ns)
    facts = {"project": True, "rows": 5, "tables": 1, "row_clicks": 0, "raw_picks": 1}
    review = find_task("converter", "review", facts)
    assert review.label in source and len(review.steps) == 4
    for step in review.steps[:3]:
        assert step.title in source
    for origin in ORIGINS.values():
        if origin != "Anonymisation":
            assert origin in source, origin
    titles = {c.key: c.title for c in CATEGORIES}
    for key in ("identity", "dates", "site", "devices"):
        assert titles[key] in source, key
    for _key, heading in PURPOSES[:3]:
        assert heading.upper() in source, heading
    tree = inspect.getsource(bids_tree_pane)
    for entry in ("Write its sidecar from the recording...", "Ignore in validation"):
        assert entry in tree and entry in source, entry
    for entry in ("Remove faces...", "Remove the skull...", "Every .txt file"):
        assert entry in source, entry


def test_the_anonymised_values_are_what_the_anonymiser_writes(tmp_path) -> None:
    """The scene's sidecar, anonymised with the Recommended preset: the same
    fields go, the same kinds of values come."""
    import json
    import re

    from bidsmgr.anonymise.plan import plan_anonymise
    from bidsmgr.anonymise.policy import policy_for

    (tmp_path / "dataset_description.json").write_text('{"Name": "t", "BIDSVersion": "1.10.0"}')
    folder = tmp_path / "sub-01" / "anat"
    folder.mkdir(parents=True)
    (folder / "sub-01_T1w.json").write_text(json.dumps({
        "PatientName": "MOT_017", "PatientID": "OL_0007",
        "AcquisitionDateTime": "2026-03-14T09:12:00", "InstitutionName": "University of Oldenburg",
        "InstitutionAddress": "Ammerländer Heerstr. 114", "DeviceSerialNumber": "167023",
        "StationName": "AWP167023", "RepetitionTime": 2.3}))
    plan = plan_anonymise(tmp_path, policy_for("recommended"))
    changes = {c.what: c for f in plan.files for c in f.changes}
    assert changes["PatientName"].after == "" and changes["PatientID"].after == ""
    assert re.fullmatch(r'"19(0[2-9]|1\d|2[0-5])-\d\d-\d\dT09:12:00"',
                        changes["AcquisitionDateTime"].after)
    assert changes["InstitutionAddress"].after == '"n/a"'
    assert "InstitutionName" not in changes, "Recommended keeps the institution's name"
    assert changes["DeviceSerialNumber"].after.startswith("device-")
    assert changes["StationName"].after.startswith("station-")
    assert "RepetitionTime" not in changes


def test_the_shifted_date_keeps_the_weekday() -> None:
    from datetime import date

    assert date(2026, 3, 14).weekday() == date(1905, 3, 11).weekday()
    assert (date(2026, 3, 14) - date(1905, 3, 11)).days % 7 == 0


def test_the_subject_labels_and_entities_shown_are_what_the_scan_makes() -> None:
    from bidsmgr.inventory import name_entities, subject_label

    assert subject_label.curate("OL_0007") == "OL0007"
    assert subject_label.curate("MOT_017") == "MOT017"
    assert name_entities.from_name("t1_mprage_acq-highres") == {"acq": "highres"}
    assert subject_label.SOURCES[subject_label.PATIENT_ID] == "Patient ID"
    assert subject_label.SOURCES[subject_label.FAMILY_NAME] == "Family name"


def test_the_stripped_derivative_is_named_as_the_tool_names_it(tmp_path) -> None:
    from bidsmgr.deface.derivatives import PIPELINE, derivative_path

    image = tmp_path / "derivatives" / "fmriprep" / "sub-01" / "anat" / \
        "sub-01_desc-preproc_T1w.nii.gz"
    out = derivative_path(tmp_path, image)
    assert out is not None and PIPELINE == "bidsmgr-skullstrip"
    assert out.relative_to(tmp_path).as_posix() == \
        "derivatives/bidsmgr-skullstrip/sub-01/anat/sub-01_desc-preprocbrain_T1w.nii.gz"


# ---------------------------------------------------------------------------
# The 1.5.1 and 1.5.2 news: how it plays
# ---------------------------------------------------------------------------


def _scenes_dialog(qtbot, size=None):
    dialog = wn.WhatsNewDialog()
    qtbot.addWidget(dialog)
    if size:
        dialog.resize(*size)
    return dialog, dialog.page()


def test_a_scene_fades_in_holds_and_fades_out() -> None:
    assert wn.scene_alpha(0.0, 8.0) == 0.0
    assert wn.scene_alpha(wn.FADE, 8.0) == 1.0
    assert wn.scene_alpha(4.0, 8.0) == 1.0
    assert wn.scene_alpha(8.0, 8.0) == 0.0
    assert ns.swap(0.0) == (1.0, 0.0) and ns.swap(1.0) == (0.0, 1.0)
    old, new = ns.swap(0.5)
    assert old == new == 0.0, "the old text has gone before the new one comes"


@pytest.mark.parametrize("theme_id", ["dark", "light", "hc-dark"])
def test_every_scene_paints_at_every_moment(qtbot, isolated_settings, theme_id) -> None:
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.theme_manager import ThemeManager

    ThemeManager(QApplication.instance()).apply(theme_id)
    dialog, page = _scenes_dialog(qtbot, (1000, 600))
    dialog.show()
    qtbot.waitExposed(dialog)
    assert page.stage.is_running()
    for i, scene in enumerate(page.stage.scenes):
        page.stage.set_scene(i, by_user=True)
        for k in range(9):
            page.stage.set_time(scene.duration * k / 8)
            assert not page.stage.grab().isNull()
    dialog.show_news("1.5.0")
    assert not page.stage.is_running(), "a page not shown stops its animation"
    dialog.hide()
    assert not dialog.page().canvas.is_running(), "the animation keeps running behind a closed dialog"


def test_a_frame_is_cheap(qtbot, isolated_settings) -> None:
    """At 60 frames a second a frame has 16 ms; every scene stays far below."""
    dialog, page = _scenes_dialog(qtbot, (1000, 600))
    dialog.show()
    qtbot.waitExposed(dialog)
    for i, scene in enumerate(page.stage.scenes):
        page.stage.set_scene(i, by_user=True)
        page.stage.set_time(0.5)
        page.stage.repaint()           # warm the fonts and the template
        start = time.perf_counter()
        for k in range(20):
            page.stage.set_time(scene.duration * k / 20)
            page.stage.repaint()
        per_frame = (time.perf_counter() - start) / 20 * 1000
        assert per_frame < 12.0, f"{scene.key}: {per_frame:.1f} ms a frame"


def test_the_stage_keeps_the_scenes_proportions(qtbot) -> None:
    stage = wn.FeatureStage()
    qtbot.addWidget(stage)
    assert stage.hasHeightForWidth()
    for width in (400, 600, 900):
        height = stage.heightForWidth(width)
        pad, below = stage._pads()
        assert math.isclose((width - 2 * pad) / (height - 2 * pad - below), ns.W / ns.H,
                            rel_tol=0.01)


def test_scenes_hand_over_until_the_user_picks_one(qtbot, isolated_settings, monkeypatch) -> None:
    _dialog, page = _scenes_dialog(qtbot)
    stage = page.stage
    assert stage.index() == 0 and stage.is_auto()
    monkeypatch.setattr(stage, "time", lambda: 99.0)
    stage._tick()
    assert stage.index() == 1, "a finished scene hands over to the next"
    assert page.feature_buttons[1].isChecked()
    assert page.scene_title.text() == stage.scenes[1].title
    assert page.scene_text.text() == stage.scenes[1].text
    page.feature_buttons[4].click()
    assert stage.index() == 4 and not stage.is_auto()
    stage._tick()
    assert stage.index() == 4, "a scene the user picked plays again"
    monkeypatch.undo()
    stage.set_scene(len(stage.scenes) - 1)
    stage.set_scene(stage.index() + 1)
    assert stage.index() == 0, "after the last, the first"


def test_arrow_keys_pick_the_previous_or_next(qtbot, isolated_settings) -> None:
    from PyQt6.QtCore import Qt

    dialog, page = _scenes_dialog(qtbot)
    dialog.show()
    qtbot.waitExposed(dialog)
    dialog._buttons["continue"].setFocus()
    qtbot.keyClick(dialog, Qt.Key.Key_Right)
    assert page.stage.index() == 1 and not page.stage.is_auto()
    qtbot.keyClick(dialog, Qt.Key.Key_Left)
    qtbot.keyClick(dialog, Qt.Key.Key_Left)
    assert page.stage.index() == len(page.stage.scenes) - 1
    assert page.feature_buttons[-1].isChecked()


def test_the_stage_says_what_it_shows(qtbot, isolated_settings) -> None:
    _dialog, page = _scenes_dialog(qtbot)
    for i, scene in enumerate(page.stage.scenes):
        page.stage.set_scene(i)
        assert scene.title in page.stage.accessibleName()
        assert page.stage.accessibleDescription() == scene.shows


def test_the_features_are_grouped_by_release(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QLabel

    _dialog, page = _scenes_dialog(qtbot)
    groups = [label.text() for label in page.findChildren(QLabel)
              if label.objectName() == "whatsnew-group"]
    assert groups == ["New in 1.5.2", "New in 1.5.1"]
    assert [b.text() for b in page.feature_buttons] == [s.title for s in ns.SCENES]
    assert all(b.toolTip() for b in page.feature_buttons)


def test_the_text_fits_at_a_larger_font(qtbot, isolated_settings) -> None:
    """At 130 percent the feature names and the release menu still fit."""
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.theme_manager import ThemeManager

    theme = ThemeManager(QApplication.instance())
    theme.set_font_scale(1.3)
    theme.apply("dark")
    try:
        dialog = wn.WhatsNewDialog(news=list(wn.NEWS), previous="1.4", walk=True)
        qtbot.addWidget(dialog)
        dialog.resize(1300, 780)
        page = dialog.page()
        dialog.show()
        qtbot.waitExposed(dialog)
        main = dialog._buttons["continue"]
        assert main.text() == "Next: what 1.5.0 brought"
        for button in [*page.feature_buttons, dialog.release_button, main]:
            assert button.width() >= button.sizeHint().width() - 1, button.text()
            assert button.mapTo(dialog, button.rect().topRight()).x() <= dialog.width()
    finally:
        theme.set_font_scale(1.0)
        theme.apply("dark")


# ---------------------------------------------------------------------------
# The 1.5.0 news: the seven kinds of data gathering into the logo
# ---------------------------------------------------------------------------


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
def test_the_150_page_paints_and_stops_when_hidden(qtbot, isolated_settings, theme_id) -> None:
    from PyQt6.QtWidgets import QApplication

    from bidsmgr.gui.theme_manager import ThemeManager

    ThemeManager(QApplication.instance()).apply(theme_id)
    dialog = wn.WhatsNewDialog(start="1.5.0")
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    canvas = dialog.page().canvas
    assert canvas.is_running()
    for t in (0.0, 1.0, wn.GATHER_AT + 0.7, wn.SETTLED + 3.0):
        canvas.set_time(t)
        assert not canvas.grab().isNull()
    dialog.hide()
    assert not canvas.is_running(), "the animation keeps running behind a closed dialog"


def test_the_150_page_says_what_it_brought(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QLabel

    dialog = wn.WhatsNewDialog(start="1.5.0")
    qtbot.addWidget(dialog)
    texts = " ".join(label.text() for label in dialog.page().findChildren(QLabel))
    assert "BIDS Manager 1.5.0" in texts
    assert "MRI series named from their acquisition parameters" in texts


#: What the SWIFTCAT note must say, wherever it is opened.
CREDIT_WORDS = ("For MRI only", "SWIFTCAT", "Marta Loureiro", "Rita G. Nunes",
                "Rita Reis Nunes", "LaSEEB", "University of Lisbon", "Patricia Clement",
                "UZ Gent", "dcm2niix", "early implementation", "works well overall",
                "clinical and heterogeneous data", "stay tuned")


def test_the_150_page_opens_the_swiftcat_credit(qtbot, isolated_settings) -> None:
    """The note behind Settings > Classification's info icon is in the 1.5.0
    news too, under the new features."""
    from bidsmgr.gui.viz.panels import explain_popup

    dialog = wn.WhatsNewDialog(start="1.5.0")
    qtbot.addWidget(dialog)
    dialog.page().credit_button.click()
    menu, label = explain_popup.last_shown()
    for needed in CREDIT_WORDS:
        assert needed in label.text(), needed
    menu.close()


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
    menu, label = explain_popup.last_shown()
    for needed in CREDIT_WORDS:
        assert needed in label.text(), needed
    menu.close()

"""Tests for the project-first Welcome tab + lifecycle wiring (Phase D)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.cli._scaffold import project_bundle_dir
from bidsmgr.cli.create import open_or_create_workspace
from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.main_window import MainWindow
from bidsmgr.gui.theme_manager import ThemeManager
from bidsmgr.gui.welcome_panel import WelcomePanel
from bidsmgr.project import Project

pytestmark = pytest.mark.gui


def test_create_project_scaffolds_and_emits(qtbot, isolated_settings, tmp_path) -> None:
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    captured: list = []
    panel.project_opened.connect(lambda proj, root: captured.append((proj, root)))

    root = panel.create_project(tmp_path, "My Study")

    # Slugified folder under the chosen parent.
    assert root == tmp_path / "My-Study"
    assert (root / "dataset_description.json").exists()
    assert project_bundle_dir(root).exists()
    dd = json.loads((root / "dataset_description.json").read_text())
    assert dd["Name"] == "My Study"  # human name preserved
    # Signal fired with a Project + the dataset root; recent list updated.
    assert len(captured) == 1 and isinstance(captured[0][0], Project)
    assert str(root) in AppSettings.load().recent_projects


def test_open_project_adopts_existing(qtbot, isolated_settings, tmp_path) -> None:
    # A dataset created out of band, no .bidsmgr bundle yet.
    root = tmp_path / "external"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "External", "BIDSVersion": "1.11.1"})
    )
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    captured: list = []
    panel.project_opened.connect(lambda proj, r: captured.append((proj, r)))

    panel.open_project(root)

    assert project_bundle_dir(root).exists()  # adopted: bundle added
    assert json.loads((root / "dataset_description.json").read_text())["Name"] == "External"
    assert len(captured) == 1


def test_delete_project_from_disk(qtbot, isolated_settings, tmp_path) -> None:
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    root = panel.create_project(tmp_path, "Doomed")  # also adds to recent
    assert root.exists() and str(root) in AppSettings.load().recent_projects

    ok = panel.delete_project(root, from_disk=True)
    assert ok
    assert not root.exists()                                   # folder gone
    assert str(root) not in AppSettings.load().recent_projects  # forgotten


def test_remove_from_list_keeps_folder(qtbot, isolated_settings, tmp_path) -> None:
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    root = panel.create_project(tmp_path, "Keep")

    panel.delete_project(root, from_disk=False)
    assert root.exists()                                        # folder kept
    assert str(root) not in AppSettings.load().recent_projects  # just forgotten


def test_delete_refuses_non_dataset_folder(qtbot, isolated_settings, tmp_path) -> None:
    # A plain folder (not a BIDS dataset / BM project) must never be rmtree'd.
    plain = tmp_path / "not_a_dataset"
    plain.mkdir()
    (plain / "important.txt").write_text("keep me")
    AppSettings.remember_recent_project(plain)

    panel = WelcomePanel()
    qtbot.addWidget(panel)
    ok = panel.delete_project(plain, from_disk=True)
    assert ok is False
    assert plain.exists() and (plain / "important.txt").exists()  # untouched


def test_mainwindow_lands_on_welcome_and_binds_on_open(
    qapp, isolated_settings, tmp_path,
) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    # Lands on Welcome (index 2) with no project.
    assert win.stack.currentIndex() == 2

    # Simulate the user creating a project from the Welcome tab.
    root = tmp_path / "study"
    proj = open_or_create_workspace(root, name="Study")
    win._on_project_opened(proj, root)
    qapp.processEvents()

    # Switches to the Converter, binds the project, and soft-locks the output.
    assert win.stack.currentIndex() == 0
    assert win.converter._project is proj
    assert win.converter._bids_root == root
    assert win.converter._bids_pathbar.change_button.isEnabled() is False


def test_theme_swap_repaints_welcome(qapp, isolated_settings, monkeypatch) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    calls: list = []
    original = win.welcome.repaint_for_palette

    def _spy(pal: dict) -> None:
        calls.append(pal)
        return original(pal)

    monkeypatch.setattr(win.welcome, "repaint_for_palette", _spy)

    theme.toggle()  # dark -> light
    theme.toggle()  # light -> dark

    # The Welcome panel is in the palette cascade (runs the unpolish/polish
    # dance) so it never keeps stale white/black surfaces.
    assert len(calls) == 2


def test_resources_card_lists_links(qtbot, isolated_settings) -> None:
    from PyQt6.QtWidgets import QLabel

    panel = WelcomePanel()
    qtbot.addWidget(panel)
    links = [
        l for l in panel.findChildren(QLabel) if l.objectName() == "welcome-link"
    ]
    # 3 resource links + 4 sample datasets + 1 update-notes link = 8 clickable
    # links, all external.
    assert len(links) == 8
    assert all(l.openExternalLinks() for l in links)
    hrefs = " ".join(l.text() for l in links)
    assert "bids_manager_documentation" in hrefs       # docs site
    assert "github.com/ANCPLabOldenburg" in hrefs       # source
    assert "cloud.uol.de" in hrefs                       # sample data
    assert "updates.html" in hrefs                       # update notes
    # Raw URLs never show as the visible text — friendly labels only.
    assert ">Documentation website<" in hrefs
    assert ">MEG Elekta sample dataset<" in hrefs


def test_recent_rows_carry_name_and_path(qtbot, isolated_settings, tmp_path) -> None:
    from bidsmgr.gui.welcome_panel import (
        _RECENT_MISSING_ROLE,
        _RECENT_NAME_ROLE,
        _RECENT_PATH_ROLE,
    )

    panel = WelcomePanel()
    qtbot.addWidget(panel)
    # A real project (named) plus a now-missing one.
    root = panel.create_project(tmp_path, "Fancy Name")
    AppSettings.remember_recent_project(tmp_path / "gone")
    panel.refresh_recent()

    by_path = {
        panel._recent.item(i).data(_RECENT_PATH_ROLE): panel._recent.item(i)
        for i in range(panel._recent.count())
    }
    real = by_path[str(root)]
    assert real.data(_RECENT_NAME_ROLE) == "Fancy Name"      # dataset Name, not slug
    assert real.data(_RECENT_MISSING_ROLE) is False
    missing = by_path[str(tmp_path / "gone")]
    assert missing.data(_RECENT_MISSING_ROLE) is True


def test_open_button_is_accent_styled(qtbot, isolated_settings) -> None:
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    # The blue-font open button keeps its dedicated object name (styled in QSS).
    assert panel._open_btn.objectName() == "welcome-open-btn"


def test_create_blocks_spaces_and_suggests_underscores(
    qtbot, isolated_settings, tmp_path, monkeypatch,
) -> None:
    from PyQt6.QtWidgets import QMessageBox

    shown: list = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda *a, **k: shown.append(a),
    )
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    created: list = []
    panel.project_opened.connect(lambda p, r: created.append(r))

    panel._location_edit.setText(str(tmp_path))
    panel._name_edit.setText("My Study")
    panel._on_inline_create()

    # Nothing created; user was warned; the field is pre-filled with underscores.
    assert created == []
    assert shown, "expected an information dialog about spaces"
    assert panel._name_edit.text() == "My_Study"
    assert not (tmp_path / "My Study").exists()


def test_parse_qcolor_handles_rgba_and_hex() -> None:
    from bidsmgr.gui.welcome_panel import _parse_qcolor

    c = _parse_qcolor("rgba(88,166,255,0.12)")
    assert c.isValid()
    assert (c.red(), c.green(), c.blue()) == (88, 166, 255)
    assert c.alpha() == int(round(0.12 * 255))  # not 0/black
    h = _parse_qcolor("#11161d")
    assert (h.red(), h.green(), h.blue()) == (0x11, 0x16, 0x1d)


def test_project_switcher_shows_and_switches(qapp, isolated_settings, tmp_path) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    # Hidden until a project is open.
    assert win._header._project_btn.isHidden()

    a = open_or_create_workspace(tmp_path / "StudyA", name="Study A")
    AppSettings.remember_recent_project(tmp_path / "StudyA")
    b = open_or_create_workspace(tmp_path / "StudyB", name="Study B")
    AppSettings.remember_recent_project(tmp_path / "StudyB")

    win._on_project_opened(a, tmp_path / "StudyA")
    qapp.processEvents()
    assert not win._header._project_btn.isHidden()
    assert "Study A" in win._header._project_btn.text()

    # The dropdown lists the current project header + the other recent.
    win._header._rebuild_project_menu()
    assert len(win._header._project_menu.actions()) >= 3

    # Switching to a recent rebinds both views + relabels the switcher.
    win._on_switch_project(tmp_path / "StudyB")
    qapp.processEvents()
    assert "Study B" in win._header._project_btn.text()
    assert win.converter._bids_root == tmp_path / "StudyB"


def test_home_pill_returns_to_welcome(qapp, isolated_settings) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    # Go to Converter, then Home pill back to Welcome.
    win._header._converter_btn.setChecked(True)
    win._header._pill_group.idClicked.emit(0)
    qapp.processEvents()
    assert win.stack.currentIndex() == 0

    win._header._welcome_btn.setChecked(True)
    win._header._pill_group.idClicked.emit(2)
    qapp.processEvents()
    assert win.stack.currentIndex() == 2

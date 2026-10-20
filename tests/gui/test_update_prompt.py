"""What the update prompt says and does when a release needs a newer Python.

Before, a release the environment's Python could not run was offered anyway:
the update reported success, changed nothing, and the prompt came back at
every start. Now an installer installation is told the update moves it to
the newer Python; any other installation is offered only what it can run,
is left in peace at startup, and is told why on "Check updates".
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QMessageBox, QWidget

from bidsmgr.gui import update_widgets, updater

BLOCKED = updater.Latest(version="1.6.0", newest="1.7.0", needs_python="3.13")


@pytest.fixture
def window(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def prompts(monkeypatch):
    shown = []
    monkeypatch.setattr(update_widgets, "_prompt_and_launch_update",
                        lambda parent, current, plan: shown.append(plan))
    return shown


def _installer_installation(monkeypatch, present: bool):
    found = updater.Installation(manifest=None, uv=None) if present else None
    monkeypatch.setattr(updater, "find_installation", lambda: found)


def test_startup_offers_an_installer_installation_the_move(window, prompts, monkeypatch):
    _installer_installation(monkeypatch, True)
    monkeypatch.setattr(updater, "installed_version", lambda: "1.5.0")
    update_widgets._on_startup_check_done(window, BLOCKED)
    assert prompts == [updater.UpdatePlan(version="1.7.0", move_to_python="3.13")]


def test_startup_offers_any_other_installation_what_it_can_run(window, prompts, monkeypatch):
    _installer_installation(monkeypatch, False)
    monkeypatch.setattr(updater, "installed_version", lambda: "1.5.0")
    update_widgets._on_startup_check_done(window, BLOCKED)
    assert prompts == [updater.UpdatePlan(version="1.6.0")]


def test_startup_stays_quiet_when_nothing_can_be_installed(window, prompts, monkeypatch):
    _installer_installation(monkeypatch, False)
    monkeypatch.setattr(updater, "installed_version", lambda: "1.6.0")
    update_widgets._on_startup_check_done(window, BLOCKED)
    assert prompts == []


def test_check_updates_explains_why_the_newest_release_is_not_offered(window, monkeypatch):
    _installer_installation(monkeypatch, False)
    monkeypatch.setattr(updater, "installed_version", lambda: "1.6.0")
    told = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda parent, title, text: told.append(text))

    class Checker:
        def __init__(self, parent):
            self.finished = self

        def connect(self, slot):
            self.slot = slot

        def start(self):
            self.slot(BLOCKED)

    monkeypatch.setattr(update_widgets, "_UpdateChecker", Checker)
    update_widgets.check_for_updates_interactive(window)
    assert len(told) == 1
    assert "1.7.0 needs Python 3.13" in told[0]
    assert "You have the newest release this installation can run (1.6.0)" in told[0]


def test_the_prompt_says_the_update_moves_python_and_yes_asks_for_the_move(window, monkeypatch):
    texts, launched = [], []

    def fake_exec(box):
        texts.append(box.text())
        box.chosen = next(b for b in box.buttons() if b.text() == "Yes")
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box.chosen)
    monkeypatch.setattr(update_widgets, "_launch_helper_and_quit",
                        lambda parent, move: launched.append(move))
    update_widgets._prompt_and_launch_update(
        window, "1.5.0", updater.UpdatePlan(version="1.7.0", move_to_python="3.13"))
    assert "needs Python 3.13 or newer, so the update also moves this installation" in texts[0]
    assert "PyPI:        1.7.0" in texts[0]
    assert launched == ["3.13"]


def test_an_ordinary_update_says_nothing_about_python(window, monkeypatch):
    texts, launched = [], []

    def fake_exec(box):
        texts.append(box.text())
        box.chosen = next(b for b in box.buttons() if b.text() == "Yes")
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: box.chosen)
    monkeypatch.setattr(update_widgets, "_launch_helper_and_quit",
                        lambda parent, move: launched.append(move))
    update_widgets._prompt_and_launch_update(window, "1.5.0", updater.UpdatePlan("1.6.0"))
    assert "Python" not in texts[0]
    assert launched == [""]

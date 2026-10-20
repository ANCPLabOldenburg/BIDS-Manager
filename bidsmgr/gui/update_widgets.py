"""Bottom-right version label + Check-updates button + startup checker.

Public surface
==============

* :func:`attach_update_widgets` adds the label + button to a status bar.
* :func:`run_startup_check` fires a background QThread shortly after
  the main window paints; on success it pops a non-blocking dialog if a
  new version is available. Any failure (no internet, SSL error, etc.)
  is logged at debug level and silently dropped.
* :func:`check_for_updates_interactive` is the manual "Check updates"
  click handler: shows a busy cursor, fetches in a worker thread,
  prompts the user, and (on confirmation) hands off to the detached
  helper before quitting the GUI.

Failure policy
==============

The whole module is wrapped to be **fail-safe at GUI startup**. A
broken network, missing ``packaging``, etc. must never propagate into
the main window's ``__init__``. ``run_startup_check`` swallows every
exception its worker raises.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Optional

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QWidget,
)

from . import updater

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class _LatestVersionWorker(QObject):
    """Fetches what PyPI offers off the GUI thread.

    Emits :pyattr:`finished` with an :class:`updater.Latest`, or ``None``
    on failure. Never raises into the event loop.
    """

    finished = pyqtSignal(object)

    def run(self) -> None:
        latest = None
        try:
            latest = updater.fetch_latest()
        except Exception as exc:
            log.debug("update check worker failed: %s", exc)
        self.finished.emit(latest)


class _UpdateChecker(QObject):
    """One-shot PyPI version fetch on a worker QThread.

    This object **lives on the main thread** (parented to the GUI
    window). The worker emits its ``finished`` signal from the worker
    thread; because the receiver lives on the main thread, Qt's default
    AutoConnection becomes a QueuedConnection and our :meth:`_on_worker_done`
    slot runs back on the main thread, where it is safe to touch
    QMessageBox / QApplication.restoreOverrideCursor.

    Connecting the worker's signal to a plain Python callable would
    fall back to DirectConnection (no QObject receiver, no thread
    affinity) and the callback would run on the worker thread, which
    is the bug pattern that made the previous implementation hang on
    QMessageBox.exec().
    """

    finished = pyqtSignal(object)  # updater.Latest, or None on any failure

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._worker = _LatestVersionWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        # Cross-thread emit, main-thread slot → auto-queued.
        self._worker.finished.connect(self._on_worker_done)
        self._worker.finished.connect(self._thread.quit)
        # Cleanup is wired to the **thread**'s ``finished`` signal — not
        # the worker's — so by the time any of these run, the thread's
        # event loop has actually exited. Calling ``self.deleteLater()``
        # from ``_on_worker_done`` would destroy the QThread child while
        # it is still running and segfault the app a few seconds after
        # launch ("QThread: Destroyed while thread is still running").
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self.deleteLater)

    def start(self) -> None:
        self._thread.start()

    def _on_worker_done(self, latest) -> None:
        # Just re-emit. Do not deleteLater here — see the comment in
        # __init__ about the QThread destruction race.
        self.finished.emit(latest)


# ---------------------------------------------------------------------------
# Bottom-right widgets
# ---------------------------------------------------------------------------


def attach_update_widgets(
    status_bar: QStatusBar,
    parent_window: QWidget,
) -> tuple[QLabel, QPushButton]:
    """Add "v1.2.3" label + "Check updates" button to the bottom-right.

    Uses :meth:`QStatusBar.addPermanentWidget` so they sit on the right
    side of the status bar, opposite the existing status text.
    """
    current = updater.installed_version()
    version_label = QLabel(f"v{current}")
    version_label.setToolTip(f"Installed bids-manager version ({current}).")
    version_label.setObjectName("update-version-label")

    check_btn = QPushButton("Check updates")
    check_btn.setObjectName("update-check-btn")
    check_btn.setToolTip("Check PyPI for a newer bids-manager release.")
    check_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    check_btn.clicked.connect(
        lambda: check_for_updates_interactive(parent_window, check_btn)
    )

    status_bar.addPermanentWidget(version_label)
    status_bar.addPermanentWidget(check_btn)
    return version_label, check_btn


# ---------------------------------------------------------------------------
# Manual "Check updates" flow
# ---------------------------------------------------------------------------


def check_for_updates_interactive(
    parent: QWidget,
    button: Optional[QPushButton] = None,
) -> None:
    """Manual update-check entry point. Wired to the bottom-right button.

    Shows a wait cursor while the PyPI fetch runs in a worker thread;
    pops a result dialog on completion (up-to-date / new version /
    couldn't reach PyPI). On user confirmation, kicks the detached
    helper and quits the GUI.
    """
    if button is not None:
        button.setEnabled(False)
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _on_done(latest: Optional[updater.Latest]) -> None:
        QApplication.restoreOverrideCursor()
        if button is not None:
            button.setEnabled(True)

        if latest is None:
            QMessageBox.warning(
                parent, "Check updates",
                "Could not reach PyPI to check for updates.\n\n"
                "Check your internet connection and try again.",
            )
            return

        current = updater.installed_version()
        plan = _plan(latest, current)
        if plan is None:
            if latest.needs_python and updater.is_newer(latest.newest, current):
                QMessageBox.information(parent, "Check updates",
                                        python_notice(latest, current))
            else:
                QMessageBox.information(
                    parent, "Check updates",
                    f"You are up to date.\n\nInstalled: {current}\nPyPI: {latest.newest}",
                )
            return

        _prompt_and_launch_update(parent, current, plan)

    checker = _UpdateChecker(parent)
    checker.finished.connect(_on_done)
    checker.start()


# ---------------------------------------------------------------------------
# Startup check
# ---------------------------------------------------------------------------


def run_startup_check(window: QWidget, delay_ms: int = 2500) -> None:
    """Fire a delayed background update check after the window paints.

    Pops a non-blocking dialog only if a strictly newer version is on
    PyPI **and** the user hasn't already chosen "Skip this version" for
    that exact version. Silent on every failure.

    Skipped in three cases:

    1. Editable installs (``pip install -e .``) — devs don't want their
       working tree replaced by a PyPI build.
    2. Running under pytest (``PYTEST_CURRENT_TEST`` set) — the worker
       thread would otherwise outlive the test and crash teardown.
    3. ``BIDSMGR_NO_UPDATE_CHECK=1`` — explicit opt-out for users on
       air-gapped machines or institutional networks that block PyPI.
    """
    if os.environ.get("BIDSMGR_NO_UPDATE_CHECK") == "1":
        log.debug("skipping startup update check: BIDSMGR_NO_UPDATE_CHECK=1")
        return
    if "PYTEST_CURRENT_TEST" in os.environ:
        log.debug("skipping startup update check: running under pytest")
        return
    if updater.is_editable_install():
        log.debug("skipping startup update check: editable install detected")
        return

    def _start_thread() -> None:
        try:
            checker = _UpdateChecker(window)
            checker.finished.connect(
                lambda latest: _on_startup_check_done(window, latest)
            )
            checker.start()
        except Exception as exc:
            log.debug("startup update check failed to start: %s", exc)

    # Delay so the main window has time to paint first.
    QTimer.singleShot(max(0, delay_ms), _start_thread)


def _on_startup_check_done(window: QWidget, latest: Optional[updater.Latest]) -> None:
    """Decide whether to nag the user about an available update.

    Silent when there is nothing this installation can install, including
    a newer release that needs a Python it does not have and cannot move
    to: asking at every start about an update that cannot happen helps
    nobody. "Check updates" explains that case.
    """
    if latest is None:
        # Silent on any failure: no network, no PyPI, etc.
        return

    current = updater.installed_version()
    plan = _plan(latest, current)
    if plan is None:
        return

    _prompt_and_launch_update(window, current, plan)


def _plan(latest: updater.Latest, current: str) -> Optional[updater.UpdatePlan]:
    return updater.plan_update(
        latest, current, can_move_python=updater.find_installation() is not None)


def python_notice(latest: updater.Latest, current: str) -> str:
    """Why the newest release is not offered: it needs a newer Python than
    this installation has, and only an installer installation can move."""
    return (
        f"You have the newest release this installation can run ({current}).\n\n"
        f"BIDS Manager {latest.newest} needs Python {latest.needs_python} or newer, "
        f"and this installation runs Python {platform.python_version()}. To use it, "
        "install BIDS Manager with the installer from the documentation, or in an "
        f"environment with Python {latest.needs_python} or newer."
    )


# ---------------------------------------------------------------------------
# Shared confirmation prompt
# ---------------------------------------------------------------------------


# Where the release notes live, and how a version becomes an anchor on that
# page: 1.2.6 is ``#v126``, 1.2.4.2 is ``#v1242``. Dropping the dots is the
# convention the documentation already uses.
_UPDATES_URL = (
    "https://ancplaboldenburg.github.io/bids_manager_documentation/updates/"
)


def release_notes_url(version: str) -> str:
    """The release-notes page, anchored at ``version`` when that is possible.

    A version that is not the usual dotted digits (a dev build, a local
    install) still gets the page, just not the anchor. Better to open the notes
    at the top than to send someone to a fragment that does not exist.
    """
    anchor = version.replace(".", "").strip()
    if anchor and anchor.isdigit():
        return f"{_UPDATES_URL}#v{anchor}"
    return _UPDATES_URL


def _prompt_and_launch_update(
    parent: QWidget,
    current: str,
    plan: updater.UpdatePlan,
) -> None:
    """Confirmation dialog → detached helper → GUI quit.

    Three buttons. ``What changed`` opens the release notes for the version on
    offer and leaves the dialog up, because deciding whether to take an update
    is exactly the moment somebody wants to read what is in it, and an update
    closes the application to install itself. ``Yes`` updates, ``No`` defers.

    The default button is ``No`` so an accidental Enter / Space while the main
    window has focus cannot trigger a self-terminating update. The dialog
    reopens on the next launch if the user is still on an older version.

    When the release needs a newer Python, the dialog says the update moves
    the installation to it, which takes longer than an ordinary update.
    """
    latest = plan.version
    move = ""
    if plan.move_to_python:
        move = (
            f"BIDS Manager {latest} needs Python {plan.move_to_python} or newer, "
            "so the update also moves this installation to it. That takes a "
            "few minutes longer. Your settings and datasets are not touched.\n\n"
        )
    while True:
        msg = QMessageBox(parent)
        msg.setWindowTitle("Update available")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(
            f"A newer bids-manager release is available.\n\n"
            f"Installed: {current}\n"
            f"PyPI:        {latest}\n\n"
            f"{move}"
            "Update now? The GUI will close, install the update, and reopen."
        )
        msg.setInformativeText(
            "Not sure? Read what changed in this release first."
        )
        btn_notes = msg.addButton("What changed", QMessageBox.ButtonRole.ActionRole)
        btn_yes = msg.addButton("Yes", QMessageBox.ButtonRole.AcceptRole)
        btn_no = msg.addButton("No", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_no)
        msg.setEscapeButton(btn_no)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked is btn_notes:
            # Opening the notes is not an answer to the question, so ask it
            # again rather than making the user wait for the next launch.
            QDesktopServices.openUrl(QUrl(release_notes_url(latest)))
            continue
        if clicked is btn_yes:
            _launch_helper_and_quit(parent, plan.move_to_python)
        return


def _launch_helper_and_quit(parent: QWidget, move_to_python: str = "") -> None:
    """Spawn the update helper, then close the application cleanly."""
    ok = updater.launch_update_helper(restart=True, move_to_python=move_to_python)
    if not ok:
        QMessageBox.critical(
            parent, "Update failed",
            "Could not start the update helper. Please try again, run the "
            "BIDS Manager installer again, or run "
            "`pip install --upgrade bids-manager`.",
        )
        return
    # Give the helper a moment to actually start before we exit.
    QTimer.singleShot(200, QApplication.instance().quit)


__all__ = [
    "attach_update_widgets",
    "check_for_updates_interactive",
    "python_notice",
    "release_notes_url",
    "run_startup_check",
]

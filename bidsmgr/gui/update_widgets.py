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
    """Fetches the latest PyPI version off the GUI thread.

    Emits :pyattr:`finished` with the latest version string or an empty
    string on failure. Never raises into the event loop.
    """

    finished = pyqtSignal(str)

    def run(self) -> None:
        latest = ""
        try:
            value = updater.fetch_latest_pypi()
            if value:
                latest = value
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

    finished = pyqtSignal(str)  # latest version, or "" on any failure

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

    def _on_worker_done(self, latest: str) -> None:
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

    def _on_done(latest: str) -> None:
        QApplication.restoreOverrideCursor()
        if button is not None:
            button.setEnabled(True)

        if not latest:
            QMessageBox.warning(
                parent, "Check updates",
                "Could not reach PyPI to check for updates.\n\n"
                "Check your internet connection and try again.",
            )
            return

        current = updater.installed_version()
        if not updater.is_newer(latest, current):
            QMessageBox.information(
                parent, "Check updates",
                f"You are up to date.\n\nInstalled: {current}\nPyPI: {latest}",
            )
            return

        _prompt_and_launch_update(parent, current, latest)

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


def _on_startup_check_done(window: QWidget, latest: str) -> None:
    """Decide whether to nag the user about an available update."""
    if not latest:
        # Silent on any failure: no network, no PyPI, etc.
        return

    current = updater.installed_version()
    if not updater.is_newer(latest, current):
        return

    _prompt_and_launch_update(window, current, latest)


# ---------------------------------------------------------------------------
# Shared confirmation prompt
# ---------------------------------------------------------------------------


# Where the release notes live, and how a version becomes an anchor on that
# page: 1.2.6 is ``#v126``, 1.2.4.2 is ``#v1242``. Dropping the dots is the
# convention the documentation already uses.
_UPDATES_URL = (
    "https://ancplaboldenburg.github.io/bids_manager_documentation/updates.html"
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
    latest: str,
) -> None:
    """Confirmation dialog → detached helper → GUI quit.

    Three buttons. ``What changed`` opens the release notes for the version on
    offer and leaves the dialog up, because deciding whether to take an update
    is exactly the moment somebody wants to read what is in it, and an update
    closes the application to install itself. ``Yes`` updates, ``No`` defers.

    The default button is ``No`` so an accidental Enter / Space while the main
    window has focus cannot trigger a self-terminating update. The dialog
    reopens on the next launch if the user is still on an older version.
    """
    while True:
        msg = QMessageBox(parent)
        msg.setWindowTitle("Update available")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(
            f"A newer bids-manager release is available.\n\n"
            f"Installed: {current}\n"
            f"PyPI:        {latest}\n\n"
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
            _launch_helper_and_quit(parent)
        return


def _launch_helper_and_quit(parent: QWidget) -> None:
    """Spawn the update helper, then close the application cleanly."""
    ok = updater.launch_update_helper(restart=True)
    if not ok:
        QMessageBox.critical(
            parent, "Update failed",
            "Could not start the update helper. Please try again or "
            "run `pip install --upgrade bids-manager` manually.",
        )
        return
    # Give the helper a moment to actually start before we exit.
    QTimer.singleShot(200, QApplication.instance().quit)


__all__ = [
    "attach_update_widgets",
    "check_for_updates_interactive",
    "release_notes_url",
    "run_startup_check",
]

"""Shared fixtures for the GUI test suite.

Defines :func:`isolated_settings` — sandbox ``QSettings`` per-test so
the GUI's persistence layer doesn't leak the real user's
preferences into tests (or vice versa).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from PyQt6.QtCore import QCoreApplication, QSettings


@pytest.fixture
def isolated_settings(tmp_path: Path) -> Iterator[None]:
    """Redirect ``QSettings`` into a per-test INI file.

    Forces the IniFormat default (macOS otherwise uses native plist
    and ignores ``setPath``) and points it at ``tmp_path``. The
    org/app names also get swapped so a leaked value from outside
    the sandbox cannot poison the test.
    """
    orig_org = QCoreApplication.organizationName()
    orig_app = QCoreApplication.applicationName()
    orig_default = QSettings.defaultFormat()

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    QCoreApplication.setOrganizationName("bidsmgr-tests")
    QCoreApplication.setApplicationName("bidsmgr-tests")
    # Ensure an empty starting state.
    QSettings().clear()
    QSettings().sync()
    try:
        yield
    finally:
        QSettings().clear()
        QSettings().sync()
        QCoreApplication.setOrganizationName(orig_org)
        QCoreApplication.setApplicationName(orig_app)
        QSettings.setDefaultFormat(orig_default)

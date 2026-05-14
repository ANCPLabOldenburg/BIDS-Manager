"""GUI entry point for the ``bidsmgr`` console script.

Usage::

    bidsmgr [--theme dark|light] [--project PATH]

* ``--theme``  selects the initial palette (defaults to ``dark``).
* ``--project`` opens or creates a ``*.bidsmgr`` project bundle. If the
  path doesn't exist, a fresh project is created at that location;
  otherwise it is opened and any prior overrides are applied to the
  inventory when one is loaded.

The CLI side of the workflow stays available — ``bidsmgr-scan``,
``-rebuild``, ``-convert``, ``-metadata``, ``-validate`` are unchanged.
The GUI is a convenience layer over the same engine.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr",
        description="Schema-driven BIDS converter / curator (GUI).",
    )
    parser.add_argument(
        "--theme", choices=("dark", "light"), default=None,
        help=(
            "Initial color theme. If omitted, the last theme the user "
            "selected in-app is restored (default: dark on first run)."
        ),
    )
    parser.add_argument(
        "--project", type=Path, default=None,
        help=(
            "Open (or create if absent) a `.bidsmgr` project bundle. "
            "Edits made in the GUI append to its event log."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    args = parser.parse_args(argv)

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    # Import Qt + GUI lazily so the ``--help`` path doesn't require
    # PyQt to be available. On Linux, make sure libxcb-cursor0 is
    # reachable (Qt 6.5+ refuses to load the xcb plugin without it).
    from .util.qt_platform import prepare as _prepare_qt_platform
    _prepare_qt_platform()

    from PyQt6.QtWidgets import QApplication

    from .gui.main_window import MainWindow
    from .gui.theme_manager import ThemeManager
    from .project import Project, ProjectError

    project = None
    if args.project is not None:
        path = args.project
        if path.exists():
            try:
                project = Project.open(path)
            except ProjectError as exc:
                print(f"could not open project {path}: {exc}", file=sys.stderr)
                return 2
        else:
            project = Project.create(path, name=path.stem)
            print(f"created new project at {path}")

    app = QApplication(sys.argv)
    # QSettings keys these to find the right per-user config file on
    # macOS / Linux / Windows. Setting them once here means every
    # ``QSettings()`` constructed in the GUI picks the same INI / plist
    # / registry location.
    app.setOrganizationName("bidsmgr")
    app.setApplicationName("bidsmgr")
    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(13)
    app.setFont(font)

    # Brand icon for the title bar / taskbar / alt-tab on Linux and
    # Windows. macOS reads its Dock and Spotlight icons from the
    # ``.app`` bundle the installer builds; the call here is still
    # safe (Qt no-ops where a native bundle already supplies an icon).
    from .gui.app_icon import set_app_icon
    set_app_icon(app)

    # Honor the persisted theme if the user didn't pass --theme.
    from .gui.app_settings import AppSettings
    initial_theme = args.theme or AppSettings.load().theme

    theme = ThemeManager(app)
    theme.apply(initial_theme)

    win = MainWindow(theme, project=project)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

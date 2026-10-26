"""GUI entry point for the ``bidsmgr`` console script.

Usage::

    bidsmgr [--theme THEME] [--project PATH]

* ``--theme``  selects the initial theme: dark, dim, nord, hc-dark, light,
  paper or hc-light (defaults to the last one chosen, else ``dark``).
* ``--project`` opens (or creates / adopts) a BIDS dataset project at the
  given directory and lands in the Converter bound to it - the same
  project-first flow as the Welcome tab's Open / Create. The output is locked
  to the dataset and the header project switcher appears.

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


def _theme_ids() -> list[str]:
    from .gui.theme_manager import theme_ids

    return theme_ids()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsmgr",
        description="Schema-driven BIDS converter / curator (GUI).",
    )
    parser.add_argument(
        "--theme", choices=_theme_ids(), default=None,
        help=(
            "Initial color theme. If omitted, the last theme the user "
            "selected in-app is restored (default: dark on first run)."
        ),
    )
    parser.add_argument(
        "--project", type=Path, default=None,
        help=(
            "Open (or create / adopt) a BIDS dataset project at this directory "
            "and land in the Converter bound to it (same as the Welcome tab's "
            "Open / Create). The output is locked to the dataset."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    args = parser.parse_args(argv)

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    # Qt and the GUI are imported lazily so ``--help`` works without PyQt.
    # ``--project`` is a BIDS dataset directory (the project-first model):
    # open or create/adopt the bundle at <dir>/.bidsmgr/project and bind it
    # below, through the same flow the Welcome tab uses.
    project = None
    bids_root = None
    if args.project is not None:
        from .cli.create import open_or_create_workspace
        bids_root = Path(args.project)
        try:
            project = open_or_create_workspace(bids_root)
        except Exception as exc:
            print(f"could not open project {bids_root}: {exc}", file=sys.stderr)
            return 2

    from .gui.bootstrap import create_application

    app, theme = create_application(args.theme)

    # The version this computer last ran, read before this session writes any
    # setting: it decides which What's new a start shows (``greet``).
    from .gui import whats_new

    previous = whats_new.read_previous()

    # Warm the schema on a background thread while the user is still
    # choosing a folder. Answering "which sidecar fields apply here" is a
    # walk of the standard's rule tree, cached for the life of the process,
    # and the first walk was being paid on the GUI thread the moment a scan
    # finished: 393 ms of dead window with the spinner already stopped.
    # (A pool is fine here: this is a rule-tree walk, not scipy; guard 8b.)
    from PyQt6.QtCore import QThreadPool

    from . import schema as _schema

    QThreadPool.globalInstance().start(_schema.warm_caches)

    # Which BIDS version this session speaks. Set before any window exists, so
    # the first form built already asks the right questions. Until this, the
    # setting reached the validator alone: a dataset could be checked against
    # one version while being filled in against another.
    from .gui.app_settings import AppSettings
    from .schema import set_active_version
    set_active_version(AppSettings.load().validate_schema_version)

    from .gui.main_window import MainWindow

    win = MainWindow(theme)
    # Bind the --project dataset through the standard open-project flow so the
    # Converter is set_project'd (output locked), the Editor points at the root,
    # and the header project switcher appears - identical to a Welcome open.
    if project is not None and bids_root is not None:
        win._on_project_opened(project, bids_root)
    win.show()

    # What this release brings, the first time it opens; then, once a day, a
    # request for a star on GitHub. From here and not from the window, so a
    # test that builds a window gets neither.
    from PyQt6.QtCore import QTimer

    QTimer.singleShot(600, lambda: greet(win, previous))
    return app.exec()


def greet(window, previous="") -> None:
    """The dialogs a start may show, one after the other: What's new for the
    releases a user coming from ``previous`` has not had (read from the
    settings when not given), then the star request if it is due today
    (after What's new closes, when both are due)."""
    from PyQt6.QtCore import QTimer

    from .gui import star_request, whats_new

    news = whats_new.maybe_show(window, previous)
    if news is None:
        star_request.maybe_show(window)
    else:
        news.finished.connect(
            lambda _result: QTimer.singleShot(400, lambda: star_request.maybe_show(window)))


if __name__ == "__main__":
    raise SystemExit(main())

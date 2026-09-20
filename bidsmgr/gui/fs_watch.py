"""Let go of the directory watches while the filesystem is being changed.

Windows only, and it has to be: on Windows a watched directory cannot be
renamed.

``QFileSystemWatcher`` is implemented with ``FindFirstChangeNotification``,
which opens a real handle on each directory it watches, and an open handle
makes ``MoveFile`` fail with ``ERROR_ACCESS_DENIED`` (WinError 5). inotify and
FSEvents watch an inode without holding it open, so macOS and Linux rename a
watched directory without noticing, which is why this was invisible there.

Measured, on a directory watched by a live ``QFileSystemWatcher``::

    rmdir  an empty watched directory   -> OK
    rename a watched directory          -> PermissionError [WinError 5]
    rename it after ``removePaths``     -> OK

Only the rename is refused, which is what made the symptom so odd to read: a
subject rename moved every file correctly (files are not watched), failed on
the one call that would have moved the folder, recorded it among ``errors`` and
carried on. The new tree was built file by file and the old, now-empty tree
stayed on disk. "Renaming a folder leaves the previous folder behind."

So the rule is: anything that renames a directory inside the dataset runs
inside :func:`watchers_released`. Not a retry loop — the handle is held for as
long as the watch is, so retrying only fails more slowly.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from PyQt6.QtCore import QFileSystemWatcher
from PyQt6.QtWidgets import QApplication

log = logging.getLogger(__name__)


def live_watchers() -> list[QFileSystemWatcher]:
    """Every ``QFileSystemWatcher`` currently owned by a window.

    The panes build theirs as ``QFileSystemWatcher(self)``, so they hang off
    widgets in the window tree and are reachable from the top-level widgets.
    Collected rather than registered because a registry is one more thing to
    keep in step with the panes, and a missed registration would be a silent
    return of exactly this bug.
    """
    app = QApplication.instance()
    if app is None:
        return []
    found: list[QFileSystemWatcher] = []
    seen: set[int] = set()
    for window in app.topLevelWidgets():
        for watcher in window.findChildren(QFileSystemWatcher):
            if id(watcher) not in seen:
                seen.add(id(watcher))
                found.append(watcher)
    return found


@contextmanager
def watchers_released() -> Iterator[None]:
    """Drop every directory watch for the duration of the block.

    A no-op off Windows: there is nothing to release, and releasing it would
    mean rebuilding watches the platform never needed dropped.

    Paths are restored afterwards, minus the ones that no longer exist —
    which is the normal case here, since the point of the block is usually
    that some of them were just renamed away. The panes re-register their
    own watches on the refresh that follows.
    """
    if os.name != "nt":
        yield
        return

    released: list[tuple[QFileSystemWatcher, list[str]]] = []
    for watcher in live_watchers():
        paths = list(watcher.directories()) + list(watcher.files())
        if not paths:
            continue
        try:
            watcher.removePaths(paths)
        except RuntimeError:          # the C++ object went away
            continue
        released.append((watcher, paths))

    if released:
        log.debug(
            "released %d watched path(s) across %d watcher(s)",
            sum(len(p) for _w, p in released), len(released),
        )
    try:
        yield
    finally:
        for watcher, paths in released:
            alive = [p for p in paths if os.path.exists(p)]
            if not alive:
                continue
            try:
                watcher.addPaths(alive)
            except RuntimeError:
                pass


__all__ = ["live_watchers", "watchers_released"]

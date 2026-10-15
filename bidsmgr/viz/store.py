"""The store: one scene, its sources, and the only way to change them.

:class:`SceneStore` holds a :class:`~bidsmgr.viz.scene.Scene` and the opened
sources it refers to, runs commands against them, keeps undo history and tells
subscribers which paths changed. Qt-free: the GUI wraps it in a ``QObject``
that turns notifications into a signal, and nothing else here knows a GUI
exists.

Notifications carry PATHS, not values: ``"cursor"``, ``"layer:<id>.display"``,
``"render.camera"``. A canvas subscribes once and redraws only when a path it
cares about changed, so dragging the 3-D camera does not resample a slice.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
from typing import Any, Callable, Iterator, Optional

from .commands import COMMANDS, spec as command_spec
from .scene import Scene

log = logging.getLogger(__name__)

Listener = Callable[[frozenset], None]

#: How many undo steps are kept.
UNDO_DEPTH = 100


def touches(changed: frozenset, *prefixes: str) -> bool:
    """Whether any changed path starts with one of ``prefixes``."""
    for path in changed:
        for prefix in prefixes:
            if path == prefix or path.startswith(prefix):
                return True
    return False


class SceneStore:
    """One viewer's state, and the single gate it changes through."""

    def __init__(self, scene: Optional[Scene] = None) -> None:
        self.scene: Scene = scene or Scene()
        #: Opened sources by id (``VolumeSource`` and friends). Not part of the
        #: serialised scene: the scene names files, the store holds them open.
        self.sources: dict[str, Any] = {}
        #: Bumped on every change. Workers stamp their results with it so a
        #: result computed for a scene that has since changed can be dropped.
        self.version = 0
        self._listeners: dict[int, Listener] = {}
        self._ids = itertools.count(1)
        self._undo: list[tuple[str, dict]] = []
        self._redo: list[tuple[str, dict]] = []
        self._batch_depth = 0
        self._pending: set[str] = set()
        # An open gesture (a slider drag): its first undoable change is ONE
        # undo step, the rest of the drag adds none.
        self._gesture_depth = 0
        self._gesture_recorded = False

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def run(self, command_id: str, /, **params: Any) -> frozenset:
        """Run a command; returns the paths it changed (empty: a no-op)."""
        spec = command_spec(command_id)
        record = spec.undoable and not (self._gesture_depth and self._gesture_recorded)
        before = self.scene.model_dump(mode="python") if record else None
        changed = spec.fn(self, **params) or set()
        changed = frozenset(changed)
        if changed:
            if before is not None:
                self._undo.append((command_id, before))
                del self._undo[:-UNDO_DEPTH]
                self._redo.clear()
                if self._gesture_depth:
                    self._gesture_recorded = True
            self.changed(changed)
        return changed

    # ------------------------------------------------------------------
    # Gestures: a drag is one undo step
    # ------------------------------------------------------------------

    def begin_gesture(self) -> None:
        """Start a continuous change (a slider pressed): until
        :meth:`end_gesture`, only the first change is recorded for undo, so
        undoing a drag undoes the drag, and a drag cannot push the user's
        real history out of the undo list."""
        if self._gesture_depth == 0:
            self._gesture_recorded = False
        self._gesture_depth += 1

    def end_gesture(self) -> None:
        self._gesture_depth = max(0, self._gesture_depth - 1)

    @contextlib.contextmanager
    def gesture(self) -> Iterator[None]:
        self.begin_gesture()
        try:
            yield
        finally:
            self.end_gesture()

    def available(self) -> list[str]:
        return sorted(COMMANDS)

    # ------------------------------------------------------------------
    # Notification
    # ------------------------------------------------------------------

    def subscribe(self, listener: Listener) -> int:
        token = next(self._ids)
        self._listeners[token] = listener
        return token

    def unsubscribe(self, token: int) -> None:
        self._listeners.pop(token, None)

    def changed(self, paths) -> None:
        """Announce changed paths (commands call this through :meth:`run`;
        sources call it directly when more of a file has been read)."""
        paths = set(paths)
        if not paths:
            return
        self.version += 1
        if self._batch_depth:
            self._pending |= paths
            return
        self._emit(frozenset(paths))

    def _emit(self, paths: frozenset) -> None:
        for listener in list(self._listeners.values()):
            try:
                listener(paths)
            except Exception:  # noqa: BLE001 - one bad listener must not stop the rest
                log.exception("viz listener failed for %s", sorted(paths))

    @contextlib.contextmanager
    def batch(self) -> Iterator[None]:
        """Run several commands, notify once at the end."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._pending:
                pending, self._pending = frozenset(self._pending), set()
                self._emit(pending)

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        command_id, before = self._undo.pop()
        self._redo.append((command_id, self.scene.model_dump(mode="python")))
        self._restore(before)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        command_id, after = self._redo.pop()
        self._undo.append((command_id, self.scene.model_dump(mode="python")))
        self._restore(after)
        return True

    def _restore(self, data: dict) -> None:
        # Sources and the cursor are not part of an undo step: undoing a
        # colour-map change must not also jump the crosshair back.
        cursor = self.scene.cursor
        sources = self.scene.sources
        self.scene = Scene.model_validate(data)
        self.scene.cursor = cursor
        self.scene.sources = sources
        self.changed({"scene"})

    # ------------------------------------------------------------------
    # Whole-state exchange
    # ------------------------------------------------------------------

    def replace_scene(self, scene: Scene) -> None:
        self.scene = scene
        self._undo.clear()
        self._redo.clear()
        self.changed({"scene"})


__all__ = ["SceneStore", "touches", "UNDO_DEPTH"]

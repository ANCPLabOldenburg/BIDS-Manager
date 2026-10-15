"""Where the Qt-free library meets the Qt event loop.

* :class:`QtStore` wraps a :class:`~bidsmgr.viz.store.SceneStore` and re-emits
  its notifications as a signal, coalesced: several commands in one event-loop
  turn reach a canvas as one ``changed`` emission, so a drag that runs three
  commands per mouse move repaints once.
* :class:`JobRunner` starts :class:`~bidsmgr.workers.viz.VizJob` threads, one
  per tag, cancelling the previous job with the same tag, and keeps them alive
  until they finish. ``stop_all`` waits for them before the owner is destroyed:
  destroying a RUNNING QThread aborts the process.
* :class:`ThemeHub` and :class:`SettingsHub` are process-wide broadcasters.
  Every viewer listens to them, so a viewer inside a dialog follows a theme
  swap or a changed preference exactly like the Editor's does. The old
  dialogs (comparison, defacing, the spectrum window) missed both.
"""

from __future__ import annotations

import contextlib
import logging
import weakref
from typing import Any, Callable, Iterator, Optional

from PyQt6 import sip
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from ...viz.settings import VizSettings
from ...viz.store import SceneStore
from ...viz.theme import VizTheme, default_theme

log = logging.getLogger(__name__)


def connect_while_alive(signal, receiver: QObject,
                        slot: Callable[..., Any]) -> None:
    """Call ``slot(receiver, *args)`` on ``signal`` while ``receiver`` exists.

    The theme and settings hubs live for the whole process; the canvases
    listening to them do not (a comparison dialog is closed, a viewer is
    rebuilt). A lambda connected to a hub has no Qt receiver, so PyQt never
    disconnects it: the next theme swap called into a deleted widget, and the
    lambda kept the whole dead viewer in memory.

    The receiver is held WEAKLY and checked at each emission; the first
    emission that finds it gone disconnects. Deliberately not done through
    ``receiver.destroyed``: a Python callable on ``destroyed`` runs inside the
    C++ destructor, where PyQt may already have released it, and that
    segfaulted (measured: ``PyQtSlot::call`` dereferencing 0x24 during a
    deferred delete). ``slot`` gets the receiver as its first argument so it
    need not capture it.
    """
    ref = weakref.ref(receiver)

    def call(*args):
        obj = ref()
        if obj is None or sip.isdeleted(obj):
            try:
                signal.disconnect(call)
            except (TypeError, RuntimeError):
                pass
            return
        slot(obj, *args)

    signal.connect(call)


class QtStore(QObject):
    """A SceneStore whose notifications arrive as a coalesced Qt signal."""

    #: frozenset of changed paths since the last emission.
    changed = pyqtSignal(object)

    def __init__(self, store: Optional[SceneStore] = None, parent=None) -> None:
        super().__init__(parent)
        self.store = store or SceneStore()
        self._pending: set[str] = set()
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(0)
        self._flush_timer.timeout.connect(self._flush)
        self._token = self.store.subscribe(self._on_store)

    @property
    def scene(self):
        return self.store.scene

    def run(self, command_id: str, /, **params: Any) -> frozenset:
        return self.store.run(command_id, **params)

    def _on_store(self, paths: frozenset) -> None:
        self._pending |= set(paths)
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def flush(self) -> None:
        """Deliver pending notifications now (tests, and before a grab)."""
        if self._flush_timer.isActive():
            self._flush_timer.stop()
        self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        paths, self._pending = frozenset(self._pending), set()
        self.changed.emit(paths)

    def detach(self) -> None:
        self.store.unsubscribe(self._token)


class JobRunner(QObject):
    """Runs viz callables on QThreads, one live job per tag.

    The jobs are not its children: :mod:`bidsmgr.workers.viz` keeps them
    alive until they finish, and a job whose runner is gone cancels itself.
    So a viewer destroyed mid-read can never take the process with it, and
    ``stop_all`` is a courtesy (stop NOW, and wait) rather than a duty.
    """

    done = pyqtSignal(str, int, object)
    failed = pyqtSignal(str, int, str)
    progressed = pyqtSignal(str, int, int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._jobs: dict[str, Any] = {}
        #: Cancelled jobs still winding down (``busy`` and ``stop_all``).
        self._retired: list[Any] = []

    def start(self, tag: str, generation: int, fn: Callable[..., Any],
              *args: Any, **kwargs: Any):
        from ...workers.viz import VizJob

        self.cancel(tag)
        job = VizJob(tag, generation, fn, *args, owner=self, **kwargs)
        job.done.connect(self.done)
        job.failed.connect(self.failed)
        job.progressed.connect(self.progressed)
        self._jobs[tag] = job
        job.start()
        return job

    def running(self, tag: str) -> bool:
        job = self._jobs.get(tag)
        return job is not None and job.isRunning()

    def busy(self) -> bool:
        """Whether any job, current or cancelled and still winding down,
        is running."""
        self._retired = [j for j in self._retired if j.isRunning()]
        return bool(self._retired) or any(j.isRunning() for j in self._jobs.values())

    def cancel(self, tag: str) -> None:
        job = self._jobs.pop(tag, None)
        if job is not None:
            job.cancel()
            if job.isRunning():
                self._retired.append(job)

    def cancel_all(self) -> None:
        for tag in list(self._jobs):
            self.cancel(tag)

    def stop_all(self, timeout_ms: int = 5000) -> None:
        """Cancel everything and wait for it to stop."""
        jobs = [*self._jobs.values(), *self._retired]
        self._jobs.clear()
        self._retired.clear()
        for job in jobs:
            job.cancel()
        for job in jobs:
            if job.isRunning():
                job.wait(timeout_ms)


class ThemeHub(QObject):
    """The current :class:`VizTheme`, broadcast on every palette change."""

    changed = pyqtSignal(object)
    _instance: Optional["ThemeHub"] = None

    def __init__(self) -> None:
        super().__init__()
        self.theme: VizTheme = default_theme()

    @classmethod
    def instance(cls) -> "ThemeHub":
        if cls._instance is None:
            cls._instance = ThemeHub()
            try:
                from .. import theme_manager

                cls._instance.theme = VizTheme.from_palette(theme_manager.CUR())
            except Exception:  # noqa: BLE001 - a default theme is fine
                pass
        return cls._instance

    def publish(self, palette: dict, name: str = "") -> None:
        if not name:
            from .. import theme_manager

            name = next((k for k, p in theme_manager.PALETTES.items() if p is palette or p == palette),
                        "dark")
        self.theme = VizTheme.from_palette(palette, name)
        self.changed.emit(self.theme)


class SettingsHub(QObject):
    """The current :class:`VizSettings`, loaded once, broadcast on change."""

    changed = pyqtSignal(object)
    _instance: Optional["SettingsHub"] = None

    def __init__(self) -> None:
        super().__init__()
        from .settings_store import load_viz_settings

        try:
            self.settings: VizSettings = load_viz_settings()
        except Exception:  # noqa: BLE001 - defaults are always usable
            log.debug("could not load viewer settings; using defaults", exc_info=True)
            self.settings = VizSettings()

    @classmethod
    def instance(cls) -> "SettingsHub":
        if cls._instance is None:
            cls._instance = SettingsHub()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Forget the cached settings (tests switch QSettings files)."""
        cls._instance = None

    #: While > 0, changes are applied and broadcast but not written to disk.
    _suspended = 0

    @contextlib.contextmanager
    def suspended(self) -> Iterator[None]:
        """Changes inside take effect but are not saved: a one-off, such as
        ``bidsmgr-view --layout mosaic`` drawing a QC figure, must not
        become the layout the GUI opens in."""
        self._suspended += 1
        try:
            yield
        finally:
            self._suspended -= 1

    def update(self, mutate: Callable[[VizSettings], None], *, persist: bool = True) -> None:
        """Change settings through ``mutate(settings)``, store, broadcast."""
        new = self.settings.model_copy(deep=True)
        mutate(new)
        new = VizSettings.model_validate(new.model_dump())
        self.settings = new
        if persist and not self._suspended:
            from .settings_store import save_viz_settings

            try:
                save_viz_settings(new)
            except Exception:  # noqa: BLE001 - a preference is not worth a crash
                log.debug("could not store viewer settings", exc_info=True)
        self.changed.emit(new)

    def replace(self, settings: VizSettings, *, persist: bool = True) -> None:
        """Adopt a whole settings object (an import, or Restore defaults)."""
        self.settings = VizSettings.model_validate(settings.model_dump())
        if persist and not self._suspended:
            from .settings_store import save_viz_settings

            try:
                save_viz_settings(self.settings)
            except Exception:  # noqa: BLE001
                log.debug("could not store viewer settings", exc_info=True)
        self.changed.emit(self.settings)


__all__ = ["JobRunner", "QtStore", "SettingsHub", "ThemeHub"]

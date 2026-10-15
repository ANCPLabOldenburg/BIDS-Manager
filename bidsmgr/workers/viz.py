"""The one worker the visualisation library uses.

:class:`VizJob` runs any Qt-free callable from :mod:`bidsmgr.viz` on a
``QThread``. Never a ``QThreadPool``: scipy's FFT dies with a bus error on a
pooled thread whose slot is reused (CLAUDE.md guard 8b), and the library runs
scipy filters, spectra and resampling. Never a widget import either: a worker
runs viz code and hands back plain data.

Three things the older loaders lacked:

* a CANCEL TOKEN the callable can check between chunks, so cancelling a
  1000-frame read actually stops it instead of only muting its result;
* a GENERATION number the caller stamps on the job, so a result that arrives
  for a file that has since been reopened is dropped even though the path is
  the same;
* SURVIVING ITS OWNER. A job has no Qt parent: a module registry holds it
  until it finishes. Destroying a viewer while a read was in flight used to
  destroy a RUNNING QThread, and Qt aborts the process for that ("QThread:
  Destroyed while thread is still running"), so every host had to remember to
  stop the viewer first. Now the owner is held weakly, the token reports
  "cancelled" once the owner is gone, the read stops at its next chunk, and
  the thread ends on its own. An ``atexit`` hook waits for anything still
  running when the interpreter shuts down.
"""

from __future__ import annotations

import atexit
import logging
import threading
import traceback
import weakref
from typing import Any, Callable, Optional

from PyQt6 import sip
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

#: Every job that has started and not yet been reaped. Holding them here,
#: rather than through a Qt parent, is what lets an owner die first.
_LIVE: set["VizJob"] = set()


class CancelToken:
    """A flag a long computation polls. Thread-safe.

    Also reports cancelled once ``owner_alive()`` says the job's owner is
    gone, so abandoned work stops without anyone having to cancel it.
    """

    def __init__(self, owner_alive: Optional[Callable[[], bool]] = None) -> None:
        self._event = threading.Event()
        self._owner_alive = owner_alive

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        if self._event.is_set():
            return True
        if self._owner_alive is not None and not self._owner_alive():
            self._event.set()
            return True
        return False

    def __call__(self) -> bool:
        return self.cancelled


class VizJob(QThread):
    """Runs ``fn(*args, cancel=token, progress=emit, **kwargs)`` off the GUI.

    ``fn`` receives the token as ``cancel`` and a ``progress(done, total)``
    callable when it declares those keyword arguments. Signals carry the
    job's ``tag`` and ``generation`` so one handler can serve many jobs.
    """

    #: (tag, generation, result)
    done = pyqtSignal(str, int, object)
    #: (tag, generation, message)
    failed = pyqtSignal(str, int, str)
    #: (tag, generation, done, total)
    progressed = pyqtSignal(str, int, int, int)

    def __init__(self, tag: str, generation: int, fn: Callable[..., Any],
                 *args: Any, owner=None, **kwargs: Any) -> None:
        # Deliberately no Qt parent: see the module docstring.
        super().__init__(None)
        self.tag = tag
        self.generation = generation
        self._owner = weakref.ref(owner) if owner is not None else None
        self.token = CancelToken(self._owner_alive if owner is not None else None)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        _LIVE.add(self)
        self.finished.connect(self._reap)

    def _owner_alive(self) -> bool:
        owner = self._owner() if self._owner is not None else None
        if owner is None:
            return False
        try:
            return not sip.isdeleted(owner)
        except RuntimeError:
            return False

    def cancel(self) -> None:
        self.token.cancel()

    def run(self) -> None:  # noqa: D401 - QThread entry point
        import inspect

        kwargs = dict(self._kwargs)
        try:
            params = inspect.signature(self._fn).parameters
        except (TypeError, ValueError):
            params = {}
        if "cancel" in params:
            kwargs["cancel"] = self.token
        if "progress" in params:
            kwargs["progress"] = self._emit_progress
        try:
            result = self._fn(*self._args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            if self.token.cancelled:
                return
            log.debug("viz job %s failed:\n%s", self.tag, traceback.format_exc())
            self.failed.emit(self.tag, self.generation, f"{type(exc).__name__}: {exc}")
            return
        if self.token.cancelled:
            return
        self.done.emit(self.tag, self.generation, result)

    def _emit_progress(self, done: int, total: int) -> None:
        if not self.token.cancelled:
            self.progressed.emit(self.tag, self.generation, int(done), int(total))

    def _reap(self) -> None:
        # ``finished`` is emitted while the thread is still winding down;
        # destroying the QThread before it has fully ended aborts. Wait (a
        # formality by now), then let go.
        self.wait(2000)
        self._fn = None
        self._args = ()
        self._kwargs = {}
        # Released on the NEXT turn of the event loop: dropping the last
        # reference here would delete the object inside the delivery of its
        # own event.
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(0, lambda job=self: _LIVE.discard(job))


def live_jobs() -> int:
    """How many jobs are running or not yet reaped (tests, diagnostics)."""
    return len(_LIVE)


@atexit.register
def _wait_for_live_jobs() -> None:
    """Interpreter shutdown destroys whatever is left; a running QThread
    destroyed that way aborts. Cancel and wait instead."""
    for job in list(_LIVE):
        try:
            job.cancel()
            job.wait(5000)
        except RuntimeError:
            pass
    _LIVE.clear()


__all__ = ["CancelToken", "VizJob", "live_jobs"]

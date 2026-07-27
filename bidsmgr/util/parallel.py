"""Backend selection for the joblib pools used by scan / probe / convert.

joblib's default ``loky`` backend runs each task in a worker *process*, which
is what we want for DICOM header reads and dcm2niix probes: they are CPU- and
IO-heavy and sidestep the GIL entirely.

Some environments cannot run it. The known case is **Windows with Python
3.14**, where loky's Windows spawn path starts workers that die during
bootstrap and joblib reports::

    joblib.externals.loky.process_executor.TerminatedWorkerError: A worker
    process managed by the executor was unexpectedly terminated. ...

which takes the whole scan down. joblib 1.5.3 / loky 3.5.6 are the newest
releases at the time of writing and both still carry the bug, so there is no
version to upgrade to.

Rather than trying to enumerate the affected (OS, Python) pairs — we only ever
confirmed one, and hard-coding a guess would both punish machines that are
fine and miss ones that are not — the policy is **detect, don't predict**:

1. Always start with the fast process backend.
2. If a pool dies, fall back to threads for that run *and remember it*, so the
   rest of the session goes straight to threads instead of paying the failed
   start-up again. The memory is per-process, so a fixed joblib picks the fast
   path back up on the next launch with no code change here.

Threads are slower for this work (the GIL is only released around IO and inside
pydicom / subprocess calls) but they are correct, use less memory than a
process pool, and keep the UI responsive.

Set ``BIDSMGR_PARALLEL_BACKEND=threading`` to skip the process backend
entirely — useful for confirming a support hunch without a code change.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Iterable, Optional

from joblib import Parallel

log = logging.getLogger(__name__)

#: Set to ``"threading"`` to force the safe backend for the whole process.
BACKEND_ENV_VAR = "BIDSMGR_PARALLEL_BACKEND"

#: Fallback backend: in-process threads, no worker spawning to go wrong.
SAFE_BACKEND = "threading"

# Flipped to True the first time a process pool dies in this interpreter, so
# the failed start-up is paid once per session rather than once per pool
# (scan, probe and convert each build their own).
_process_pool_broken = False
_lock = threading.Lock()


def _forced_backend() -> Optional[str]:
    value = os.environ.get(BACKEND_ENV_VAR, "").strip()
    return value or None


def process_backend_disabled() -> bool:
    """Whether the process backend has been ruled out for this session."""
    with _lock:
        return _process_pool_broken


def _disable_process_backend() -> None:
    global _process_pool_broken
    with _lock:
        _process_pool_broken = True


def reset_backend_state() -> None:
    """Forget a previous pool failure (test helper)."""
    global _process_pool_broken
    with _lock:
        _process_pool_broken = False


def preferred_backend() -> Optional[str]:
    """Backend for :class:`joblib.Parallel`, or ``None`` for joblib's default.

    ``None`` means loky (worker processes) and is the normal answer on every
    platform. ``"threading"`` is returned only after a process pool has
    actually died in this session, or when the environment variable forces it.
    """
    forced = _forced_backend()
    if forced:
        return forced
    return SAFE_BACKEND if process_backend_disabled() else None


def _causes(exc: BaseException) -> Iterable[BaseException]:
    """``exc`` plus its ``__cause__`` / ``__context__`` chain (bounded)."""
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen and len(seen) < 10:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _is_dead_worker_error(exc: BaseException) -> bool:
    """Whether ``exc`` means the pool itself died rather than a task failing.

    Matched by class name so joblib's private executor module (and its vendored
    loky) needn't be imported, and so equivalents from other joblib versions
    are still recognised. A task raising is NOT this: that error belongs to the
    caller and is re-raised untouched.
    """
    names = {type(e).__name__ for e in _causes(exc)}
    return bool(names & {
        "TerminatedWorkerError",     # worker segfaulted / was killed
        "BrokenProcessPool",         # pool unusable
        "BrokenExecutor",
        "WorkerInterrupt",
        "ProcessTerminatedError",
    })


def run_parallel(
    tasks: Callable[[], Iterable],
    *,
    n_jobs: int,
    consume: Callable[[Any], Any],
    backend: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Run ``tasks`` through :class:`joblib.Parallel`, falling back to threads.

    ``tasks`` is a *factory* returning a fresh iterable of ``delayed(...)``
    calls: the first attempt consumes the iterable, so a retry needs its own.
    ``consume`` receives whatever ``Parallel`` returns (a list, or a generator
    when ``return_as="generator"``) and drives the iteration — that is what
    lets callers poll for cancellation while results stream in.

    If the pool dies, the run is retried on threads and the process backend is
    disabled for the rest of the session. The retry restarts from the
    beginning: pools normally die as they start up, so little is lost, and a
    silently partial result would be worse than a slow complete one.

    Only pool-death counts. An exception raised by a task — or
    ``OperationCancelled`` from ``consume`` — propagates untouched.
    """
    chosen = backend if backend is not None else preferred_backend()
    if chosen != SAFE_BACKEND and process_backend_disabled():
        # A pool already died this session; don't pay for it again.
        chosen = SAFE_BACKEND

    try:
        return consume(Parallel(n_jobs=n_jobs, backend=chosen, **kwargs)(tasks()))
    except Exception as exc:
        if chosen == SAFE_BACKEND or not _is_dead_worker_error(exc):
            raise
        _disable_process_backend()
        log.warning(
            "Parallel worker processes died (%s); falling back to the %r "
            "backend for the rest of this session. This is slower but avoids "
            "the failure. Known to affect Windows with Python 3.14.",
            type(exc).__name__, SAFE_BACKEND,
        )

    return consume(
        Parallel(n_jobs=n_jobs, backend=SAFE_BACKEND, **kwargs)(tasks())
    )

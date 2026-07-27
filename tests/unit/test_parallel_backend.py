"""Backend selection / fallback for the joblib pools (bidsmgr.util.parallel).

Regression cover for the Windows + Python 3.14 failure, where loky's worker
processes die during bootstrap and joblib raises ``TerminatedWorkerError``,
taking a whole DICOM scan down with it.

The policy under test is *detect, don't predict*: always try the fast process
backend, and only downgrade to threads once a pool has actually died — then
remember that for the rest of the session.
"""

from __future__ import annotations

import pytest
from joblib import delayed

from bidsmgr.util import parallel as P


def _square(i: int) -> int:
    return i * i


@pytest.fixture(autouse=True)
def _clean_backend_state(monkeypatch: pytest.MonkeyPatch):
    """Each test starts with the process backend considered healthy."""
    monkeypatch.delenv(P.BACKEND_ENV_VAR, raising=False)
    P.reset_backend_state()
    yield
    P.reset_backend_state()


class _DeadPool:
    """Stand-in for Parallel whose process pools always die."""

    def __init__(self, seen: list, real):
        self._seen, self._real = seen, real

    def __call__(self, n_jobs=1, backend=None, **kw):
        self._seen.append(backend)
        outer = self

        class _P:
            def __call__(self, tasks):
                from joblib.externals.loky.process_executor import (
                    TerminatedWorkerError,
                )
                if backend != P.SAFE_BACKEND:
                    raise TerminatedWorkerError("worker died")
                return outer._real(n_jobs=1, backend=P.SAFE_BACKEND, **kw)(tasks)

        return _P()


class TestBackendSelection:
    def test_process_backend_is_the_default_everywhere(self) -> None:
        """No OS/version guessing — the fast backend is always tried first."""
        assert P.process_backend_disabled() is False
        assert P.preferred_backend() is None          # None == joblib's loky

    def test_env_var_can_force_the_safe_backend(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(P.BACKEND_ENV_VAR, "threading")
        assert P.preferred_backend() == "threading"

    def test_downgrades_only_after_a_real_failure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert P.preferred_backend() is None
        monkeypatch.setattr(P, "Parallel", _DeadPool([], P.Parallel))
        P.run_parallel(
            lambda: (delayed(_square)(i) for i in range(3)),
            n_jobs=2, consume=list,
        )
        assert P.process_backend_disabled() is True
        assert P.preferred_backend() == "threading"


class TestRunParallel:
    def test_runs_and_collects(self) -> None:
        out = P.run_parallel(
            lambda: (delayed(_square)(i) for i in range(6)),
            n_jobs=2, consume=list,
        )
        assert out == [0, 1, 4, 9, 16, 25]

    def test_generator_mode(self) -> None:
        """return_as="generator" is how the scan polls for cancellation."""
        out = P.run_parallel(
            lambda: (delayed(_square)(i) for i in range(4)),
            n_jobs=2, consume=lambda gen: list(gen),
            return_as="generator",
        )
        assert out == [0, 1, 4, 9]

    def test_dead_pool_retries_on_threads(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list = []
        monkeypatch.setattr(P, "Parallel", _DeadPool(seen, P.Parallel))
        out = P.run_parallel(
            lambda: (delayed(_square)(i) for i in range(4)),
            n_jobs=2, consume=list,
        )
        assert out == [0, 1, 4, 9]                 # complete, not partial
        assert seen == [None, "threading"]         # tried fast, then fell back

    def test_failure_is_remembered_across_runs(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The dead-pool cost is paid once per session, not once per pool."""
        seen: list = []
        monkeypatch.setattr(P, "Parallel", _DeadPool(seen, P.Parallel))
        for _ in range(3):
            P.run_parallel(
                lambda: (delayed(_square)(i) for i in range(2)),
                n_jobs=2, consume=list,
            )
        # First run probes and falls back; later runs go straight to threads.
        assert seen == [None, "threading", "threading", "threading"]

    def test_task_errors_are_not_swallowed(self) -> None:
        """Only a dead pool triggers the retry — real failures propagate."""
        def boom(_i: int) -> None:
            raise RuntimeError("task failed")

        with pytest.raises(RuntimeError, match="task failed"):
            P.run_parallel(
                lambda: (delayed(boom)(i) for i in range(2)),
                n_jobs=1, consume=list,
            )
        # A failing task must not be mistaken for a broken pool.
        assert P.process_backend_disabled() is False

    def test_consume_errors_propagate(self) -> None:
        """Cancellation is raised from consume() and must not be retried."""
        class Cancelled(Exception):
            pass

        def cancel(_gen):
            raise Cancelled("cancelled by user")

        with pytest.raises(Cancelled):
            P.run_parallel(
                lambda: (delayed(_square)(i) for i in range(2)),
                n_jobs=1, consume=cancel,
            )
        assert P.process_backend_disabled() is False

    def test_threading_backend_does_not_retry(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Already on threads, a failure must surface rather than loop."""
        from joblib.externals.loky.process_executor import TerminatedWorkerError

        calls: list = []

        class AlwaysDead:
            def __init__(self, n_jobs=1, backend=None, **kw):
                calls.append(backend)

            def __call__(self, tasks):
                raise TerminatedWorkerError("worker died")

        monkeypatch.setattr(P, "Parallel", AlwaysDead)
        with pytest.raises(TerminatedWorkerError):
            P.run_parallel(
                lambda: (delayed(_square)(i) for i in range(2)),
                n_jobs=2, consume=list, backend="threading",
            )
        assert calls == ["threading"]

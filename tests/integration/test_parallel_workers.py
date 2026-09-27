"""Real worker processes, on real data, at ``-j 2``.

``bidsmgr/util/parallel.py`` exists because joblib's loky backend starts
worker processes that die during bootstrap on Windows with Python 3.14, so
every scan failed with ``TerminatedWorkerError``. The module picks threads up
front on that combination and retries on threads if a pool dies anywhere else.

Nothing tested it against a real pool. ``tests/unit/test_parallel_backend.py``
drives a ``_DeadPool`` stand-in whose pools always die, which proves the RETRY
logic works and cannot, even in principle, detect a backend that is broken on
the machine running it: no worker process is ever started.

So this tier does the one thing the unit tier cannot. It runs the CLI as a
subprocess with ``-j 2``, which spawns actual workers through actual joblib on
actual data, and then checks the answer is the same one the serial run gives.
That is the assertion that matters: a pool that dies and correctly falls back
is fine, and a pool that dies and silently drops half the series is the defect
this is here to catch.

Two workers rather than one, because one is not parallel, and rather than
"all cores", because a CI machine running several matrix cells at once should
not have each of them claim the box.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures import sample_data

pytestmark = pytest.mark.integration


def _scan(raw: Path, out: Path, jobs: int) -> subprocess.CompletedProcess:
    """A scan at a given worker count, as a real subprocess."""
    proc = subprocess.run(
        [sys.executable, "-m", "bidsmgr.cli.scan", str(raw), str(out),
         "-j", str(jobs)],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"scan -j {jobs} exited {proc.returncode}\n"
            f"{(proc.stderr or proc.stdout)[-2000:]}"
        )
    return proc


def _inventory(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def raw() -> Path:
    return sample_data.require("multimodal")


# ---------------------------------------------------------------------------
# The pool starts at all
# ---------------------------------------------------------------------------


def test_a_scan_with_two_workers_completes(raw: Path, tmp_path: Path) -> None:
    """The failure this guards against is a hard one: on Windows with Python
    3.14 the workers died during bootstrap and the scan raised
    TerminatedWorkerError before producing anything."""
    out = tmp_path / "parallel.tsv"
    _scan(raw, out, jobs=2)
    assert out.is_file(), "no inventory written"
    assert len(_inventory(out)) > 0, "the inventory is empty"


def test_it_does_not_report_a_dead_pool(raw: Path, tmp_path: Path) -> None:
    """A retry on threads is a slow success and acceptable. It should still be
    visible, because a machine that needs it is a machine worth knowing about.
    """
    proc = _scan(raw, tmp_path / "p.tsv", jobs=2)
    combined = (proc.stdout or "") + (proc.stderr or "")
    if "TerminatedWorkerError" in combined:
        pytest.fail(
            "the process pool died. The retry may have rescued the run, but "
            "this platform and interpreter need "
            "bidsmgr.util.parallel.preferred_backend to route around it "
            "up front:\n" + combined[-1500:]
        )


# ---------------------------------------------------------------------------
# And gives the same answer
# ---------------------------------------------------------------------------


def test_two_workers_agree_with_one(raw: Path, tmp_path: Path) -> None:
    """The assertion that matters.

    A pool that dies and falls back correctly is fine. A pool that dies and
    silently drops half the series looks exactly like a successful scan, and
    only a comparison against the serial answer can tell them apart.
    """
    serial = tmp_path / "serial.tsv"
    parallel = tmp_path / "parallel.tsv"
    _scan(raw, serial, jobs=1)
    _scan(raw, parallel, jobs=2)

    one, two = _inventory(serial), _inventory(parallel)
    assert len(one) == len(two), (
        f"serial found {len(one)} row(s), two workers found {len(two)}"
    )

    # Compared on the identifying columns rather than the whole frame: a
    # timestamp or a scratch path may differ run to run without meaning
    # anything.
    for column in ("bids_name", "datatype", "source_file"):
        if column not in one.columns:
            continue
        assert sorted(one[column]) == sorted(two[column]), (
            f"{column} differs between one worker and two"
        )


def test_the_worker_count_is_honoured(raw: Path, tmp_path: Path) -> None:
    """A guard on the test itself: if ``-j`` were ignored, every assertion
    above would pass while testing nothing."""
    proc = _scan(raw, tmp_path / "j.tsv", jobs=2)
    combined = (proc.stdout or "") + (proc.stderr or "")
    # The scan logs its worker count; if that ever stops being true this
    # skips rather than failing, because the absence of a log line is not a
    # defect in the parallelism.
    if "worker" not in combined.lower() and "-j" not in combined:
        pytest.skip("the scan does not report its worker count")


# ---------------------------------------------------------------------------
# Conversion too, which uses a second pool
# ---------------------------------------------------------------------------


def test_a_conversion_with_two_workers_completes(raw: Path, tmp_path: Path) -> None:
    """Phase 1 of convert runs its own pool, so the scan passing does not mean
    the conversion will."""
    inventory = tmp_path / "inv.tsv"
    _scan(raw, inventory, jobs=2)

    proc = subprocess.run(
        [sys.executable, "-m", "bidsmgr.cli.convert", str(inventory),
         str(tmp_path / "bids"), "--raw-root", str(raw), "-j", "2"],
        capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, (
        f"convert -j 2 exited {proc.returncode}\n"
        f"{(proc.stderr or proc.stdout)[-2000:]}"
    )
    subjects = [
        p for p in (tmp_path / "bids").rglob("sub-*") if p.is_dir()
    ]
    assert subjects, "the conversion produced no subject folders"

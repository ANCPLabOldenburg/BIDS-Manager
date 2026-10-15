"""Fixtures every test gets.

Operation logs (``util/oplog``) go to the dataset's ``.bidsmgr/logs``, or,
for a run with no dataset, to ``~/.bidsmgr/logs``. A test is such a run, so
without this every scan and conversion in the suite left a file in the
home folder of whoever ran it.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _logs_stay_in_the_test(tmp_path: Path, monkeypatch) -> None:
    from bidsmgr.util import oplog

    real = oplog.log_dir
    monkeypatch.setattr(oplog, "log_dir",
                        lambda root: real(root) if root is not None else tmp_path / "home_logs")

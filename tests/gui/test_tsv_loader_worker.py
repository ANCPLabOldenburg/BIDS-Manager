"""The TSV loader worker: a table is parsed off the GUI thread."""

from __future__ import annotations

from pathlib import Path

import pytest

from bidsmgr.workers import TsvLoaderWorker

pytestmark = pytest.mark.gui


def test_tsv_loader_worker(qtbot, tmp_path: Path) -> None:
    p = tmp_path / "big.tsv"
    p.write_text("a\tb\n1\t2\n3\t4\n")
    w = TsvLoaderWorker(p, 5000)
    with qtbot.waitSignal(w.finished_with_data, timeout=5000) as blocker:
        w.start()
    header, rows, total, path = blocker.args
    assert header == ["a", "b"]
    assert rows == [["1", "2"], ["3", "4"]]
    assert total == 2
    assert path == p
    w.wait()

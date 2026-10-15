"""Every operation writes its log file (util/oplog)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from bidsmgr.util import oplog


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_an_operation_writes_header_records_and_footer(tmp_path):
    root = tmp_path / "ds"
    (root / ".bidsmgr").mkdir(parents=True)
    with oplog.operation_log("convert", root=root, details={"inventory": "inv.tsv"}) as path:
        logging.getLogger("bidsmgr.cli.convert").info("sub-001: 3 files")
        logging.getLogger("bidsmgr.cli.convert").warning("sub-002: no sidecar")
    assert path.parent == root / ".bidsmgr" / "logs" and path.name.endswith("_convert.log")
    text = _lines(path)
    assert "BIDS Manager" in text[0] and ": convert" in text[0]
    assert any("inventory: inv.tsv" in line for line in text)
    assert any("sub-001: 3 files" in line for line in text), "INFO reaches the file"
    assert "convert finished" in text[-1] and "1 warning(s), 0 error(s)" in text[-1]
    assert ":" not in path.name, "no colons: Windows file names"


def test_the_terminal_is_not_made_louder(tmp_path):
    stream = io.StringIO()
    console = logging.StreamHandler(stream)
    root_logger = logging.getLogger()
    root_logger.addHandler(console)
    old = root_logger.level
    root_logger.setLevel(logging.WARNING)
    try:
        with oplog.operation_log("scan", root=None, path=tmp_path / "x.log"):
            logging.getLogger("bidsmgr.cli.scan").info("quiet detail")
            logging.getLogger("bidsmgr.cli.scan").warning("loud problem")
    finally:
        root_logger.removeHandler(console)
        root_logger.setLevel(old)
    printed = stream.getvalue()
    assert "loud problem" in printed and "quiet detail" not in printed
    assert "quiet detail" in (tmp_path / "x.log").read_text(encoding="utf-8")


def test_a_failure_is_recorded_and_raised(tmp_path):
    with pytest.raises(RuntimeError):
        with oplog.operation_log("validate", path=tmp_path / "v.log"):
            raise RuntimeError("dcm2niix said no")
    assert "stopped by RuntimeError: dcm2niix said no" in _lines(tmp_path / "v.log")[-1]


def test_gui_records_stay_out(tmp_path):
    with oplog.operation_log("scan", path=tmp_path / "s.log"):
        logging.getLogger("bidsmgr.gui.viz.viewer").warning("a viewer computing beside")
        logging.getLogger("bidsmgr.inventory.mri_dicom").warning("the scan's own")
    text = (tmp_path / "s.log").read_text(encoding="utf-8")
    assert "the scan's own" in text and "a viewer computing" not in text


def test_disabled_writes_nothing(tmp_path):
    with oplog.operation_log("scan", root=tmp_path, enabled=False) as path:
        logging.getLogger("bidsmgr").warning("x")
    assert path is None and not (tmp_path / ".bidsmgr").exists()


def test_old_logs_are_pruned(tmp_path, monkeypatch):
    monkeypatch.setattr(oplog, "KEEP", 3)
    folder = oplog.log_dir(tmp_path)
    folder.mkdir(parents=True)
    for i in range(5):
        (folder / f"2026010{i}T000000000Z_scan.log").write_text("x", encoding="utf-8")
    with oplog.operation_log("scan", root=tmp_path):
        pass
    assert len(list(folder.glob("*.log"))) == 3
    assert oplog.latest(tmp_path, "scan").name > "20260104"


def test_a_non_ascii_dataset_path(tmp_path):
    root = tmp_path / "Ünïcödé datos"
    with oplog.operation_log("metadata", root=root, details={"target": str(root)}) as path:
        logging.getLogger("bidsmgr").info("ñandú")
    assert "ñandú" in path.read_text(encoding="utf-8")


def test_the_dataset_is_found_from_a_path_inside_it(tmp_path):
    root = tmp_path / "ds"
    staging = root / ".bidsmgr" / "project" / ".scan_staging"
    staging.mkdir(parents=True)
    assert oplog.dataset_root_for(staging) == root
    assert oplog.dataset_root_for(tmp_path / "elsewhere") is None


def test_the_cli_writes_a_log_and_takes_the_flags(tmp_path):
    """bidsmgr-validate on a tiny dataset: a log by default, at --log-file
    when given, none with --no-log-file."""
    from bidsmgr.cli import validate

    root = tmp_path / "ds"
    (root / ".bidsmgr").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        '{"Name": "x", "BIDSVersion": "1.11.1"}', encoding="utf-8")
    validate.main([str(root), "--no-report"]) if "--no-report" in _help(validate) else \
        validate.main([str(root)])
    assert oplog.latest(root, "validate") is not None
    explicit = tmp_path / "mine.log"
    validate.main([str(root), "--log-file", str(explicit)])
    assert explicit.is_file()
    before = len(oplog.history(root))
    validate.main([str(root), "--no-log-file"])
    assert len(oplog.history(root)) == before


def _help(module) -> str:
    import contextlib

    buf = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
        module.main(["--help"])
    return buf.getvalue()

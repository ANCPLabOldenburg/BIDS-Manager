"""Where a scan's source data is, and whether it is still there.

The record a scan keeps (its ``raw_root``) was overwritten by the next folder
picked to scan, so an older scan named a newer scan's folder; the absolute
DICOM paths the scan wrote are believed first. Data that was moved is found
again by saying where it went, and every path is re-based onto that.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pydicom")

from bidsmgr.inventory import sources  # noqa: E402
from bidsmgr.inventory.probe_convert import series_files  # noqa: E402
from tests.fixtures.dicoms import write_mr_series  # noqa: E402


def _scan(root: Path, layout: dict[str, str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Write one series per ``uid -> folder under root`` and the inventory
    rows plus ``files_by_uid`` a scan of ``root`` would have made (a series
    directly in the root has the root's NAME as its folder)."""
    rows, files_by_uid = [], {}
    for uid, rel in layout.items():
        folder = root / rel if rel else root
        write_mr_series(folder, uid, description=f"s{uid[-1]}")
        files_by_uid[uid] = series_files(folder, uid)
        rows.append({"series_uid": uid, "source_folder": rel or root.name, "source_file": ""})
    return pd.DataFrame(rows), files_by_uid


def test_the_scanned_folder_is_read_from_where_the_files_were(tmp_path):
    root = tmp_path / "study"
    df, fbu = _scan(root, {"1.2.3.1": "sub01/DICOM", "1.2.3.2": "sub02/DICOM"})
    assert sources.scanned_root(df, fbu) == root


def test_files_directly_in_the_scanned_folder(tmp_path):
    """The scanner writes the folder's own name as ``source_folder``, so
    ``root/source_folder`` is not a folder; the preview used to give up."""
    root = tmp_path / "OL_0003"
    df, fbu = _scan(root, {"1.2.3.1": ""})
    assert df.loc[0, "source_folder"] == "OL_0003"
    found, seen = sources.resolve(df, fbu, recorded=root, label="OL_0003")
    assert seen.state == "ok" and found.root == root
    assert found.folder("OL_0003") == root, "the flat convention"


def test_a_wrong_record_is_corrected_by_the_files(tmp_path):
    """Measured on six real projects: an older scan's record named the next
    scan's folder."""
    first, second = tmp_path / "Shari_data", tmp_path / "Shayam_data"
    df, fbu = _scan(first, {"1.2.3.1": "sub01/DICOM"})
    _scan(second, {"1.2.3.9": "sub01/DICOM"})
    found, seen = sources.resolve(df, fbu, recorded=second, label="Shari_data")
    assert seen.state == "ok" and found.root == first
    assert seen.corrected_from == second


def test_moved_data_is_reported_then_found_where_it_went(tmp_path):
    old, new = tmp_path / "raw", tmp_path / "elsewhere" / "raw_moved"
    df, fbu = _scan(old, {"1.2.3.1": "sub01/DICOM", "1.2.3.2": ""})
    new.parent.mkdir()
    shutil.move(str(old), str(new))
    found, seen = sources.resolve(df, fbu, recorded=old, label="raw")
    assert seen.state == "missing" and found.root == old, "where it was last seen"
    assert "no longer at" in sources.describe(seen, found.root)

    found, seen = sources.locate(df, fbu, new, scanned=found.scanned)
    assert seen.state == "ok"
    files = found.series_files("1.2.3.1")
    assert files and all(f.exists() and new in f.parents for f in files)
    assert found.folder("sub01/DICOM") == new / "sub01" / "DICOM"
    assert found.folder("raw") == new, "flat rows follow the move too"
    relocated = found.relocated_files_by_uid()
    assert all(Path(f).exists() for f in relocated["1.2.3.2"])

    # A relink is recorded as the root; resolving again finds it there.
    again, seen = sources.resolve(df, fbu, recorded=new, label="raw")
    assert seen.state == "ok" and again.root == new and seen.corrected_from is None


def test_a_folder_that_only_looks_the_same_is_not_accepted(tmp_path):
    """Same layout and file names, another session: the series UID of each
    sampled DICOM is read before a picked folder is believed."""
    old, other = tmp_path / "raw", tmp_path / "other"
    df, fbu = _scan(old, {"1.2.3.1": "sub01"})
    write_mr_series(other / "sub01", "9.9.9.1", description="s1")
    shutil.rmtree(old)
    _found, seen = sources.locate(df, fbu, other, scanned=old)
    assert seen.checked == 1 and seen.found == 0


def test_a_fieldmap_row_brings_both_of_its_series(tmp_path):
    root = tmp_path / "raw"
    _df, fbu = _scan(root, {"1.2.3.1": "fm", "1.2.3.2": "fm"})
    got = sources.ScanSources(root, root, fbu).series_files("1.2.3.1|1.2.3.2")
    assert len(got) == len(fbu["1.2.3.1"]) + len(fbu["1.2.3.2"])


def test_recordings_are_found_relative_to_the_root(tmp_path):
    rec = tmp_path / "raw" / "sub-01" / "eeg" / "rest.edf"
    rec.parent.mkdir(parents=True)
    rec.write_bytes(b"0")
    df = pd.DataFrame([{"series_uid": "", "source_folder": "sub-01/eeg",
                        "source_file": "sub-01/eeg/rest.edf"}])
    found, seen = sources.resolve(df, {}, recorded=tmp_path / "raw")
    assert seen.state == "ok" and found.file("sub-01/eeg/rest.edf") == rec
    # A path written on Windows by an older version.
    assert found.file("sub-01\\eeg\\rest.edf") == rec
    rec.unlink()
    assert sources.resolve(df, {}, recorded=tmp_path / "raw")[1].state == "missing"


def test_nothing_recorded_is_unknown_not_moved():
    df = pd.DataFrame([{"series_uid": "", "source_folder": "", "source_file": "a.edf"}])
    found, seen = sources.resolve(df, {}, recorded=None)
    assert found.root is None and seen.state == "unknown"
    assert sources.resolve(pd.DataFrame(columns=["series_uid"]), {}, recorded=None)[1].state \
        == "unknown"


def test_the_files_by_uid_record_is_read(tmp_path):
    import gzip
    import json

    inv = tmp_path / "inventory.tsv"
    assert sources.load_files_by_uid(inv) == {}
    with gzip.open(sources.files_by_uid_path(inv), "wb") as fh:
        fh.write(json.dumps({"1.2": ["/a/b.dcm"]}).encode("utf-8"))
    assert sources.load_files_by_uid(inv) == {"1.2": ["/a/b.dcm"]}

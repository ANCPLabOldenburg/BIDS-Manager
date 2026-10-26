"""DICOMs on a network share, on Windows: staging, classification, messages.

What failed on a share, and what each test pins:

- staging LINKED every DICOM, and Windows does not follow a link stored on a
  share (its "remote to remote" and "remote to local" link rules are off by
  default), so a project kept on a share handed dcm2niix a folder of links it
  could not open: ``rc=2`` "Unable to find any DICOM images". On Windows a
  share on either side is now copied, and any error creating a link switches
  the rest of the series to copies (only "no privilege", 1314, did before);
- link targets came from ``Path.resolve``, which on Windows turns ``Z:\\data``
  into ``\\\\server\\share\\data`` (or, on DFS, into whichever server answered);
  so did ``long_path_for_tree``, which could also prefix a path twice;
- a dcm2niix timeout or a read error during the scan's classification or
  probe ended the whole scan; it is now that folder's or that series' problem;
- a folder on a share that dcm2niix read nothing from is tried again from a
  local copy;
- a mapped drive invisible to this process (started as administrator) gave
  "NotADirectoryError: Z:\\data"; it now says why.

The Windows conditions are simulated, so these run on every platform. The
simulation swaps the ``os`` the module under test sees, never ``os.name``
itself: ``pathlib`` reads that to choose between ``PosixPath`` and
``WindowsPath``, and patching it globally breaks ``Path()`` for the session.
"""

from __future__ import annotations

import json
import ntpath
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bidsmgr.classifier import dcm2niix_bidsguess
from bidsmgr.inventory import probe_convert
from bidsmgr.inventory.types import InventoryRow
from bidsmgr.util import paths as paths_mod
from bidsmgr.util import staging


def _windows(monkeypatch, drive_types=None) -> None:
    """Make ``util.paths`` see Windows; ``drive_types`` maps ``"Z:\\"`` to
    what ``GetDriveTypeW`` answers (4 is a network drive, 3 a local disk)."""
    monkeypatch.setattr(paths_mod, "os", SimpleNamespace(
        name="nt", path=ntpath, fspath=os.fspath))
    table = drive_types or {}
    monkeypatch.setattr(paths_mod, "_drive_type", lambda root: table.get(root.upper(), 3))


#: The folder standing for a network share in the simulated tests.
SHARE = "netshare"


def _on_share(path) -> bool:
    return SHARE in Path(path).parts


def _dicoms(folder: Path, n: int = 3) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = folder / f"1.3.12.2.1107.5.2.43.{i:04d}.dcm"
        p.write_bytes(b"DICM" + bytes([i]))
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# What a network path is
# ---------------------------------------------------------------------------


def test_network_paths_as_windows_sees_them(monkeypatch) -> None:
    _windows(monkeypatch, {"Z:\\": 4})
    assert paths_mod.is_network_path("\\\\server\\share\\study")
    assert paths_mod.is_network_path("\\\\?\\UNC\\server\\share\\study")
    assert paths_mod.is_network_path("\\\\?\\unc\\server\\share")
    assert paths_mod.is_network_path("Z:\\study\\sub-01")
    assert paths_mod.is_network_path("z:/study")
    assert not paths_mod.is_network_path("C:\\study")
    assert not paths_mod.is_network_path("\\\\?\\C:\\study")


def test_nothing_is_a_network_path_off_windows(tmp_path) -> None:
    if os.name == "nt":
        pytest.skip("checks the macOS and Linux answer")
    assert not paths_mod.is_network_path(tmp_path)
    assert not paths_mod.is_network_path("//server/share/study")


# ---------------------------------------------------------------------------
# long_path_for_tree keeps the drive the user picked and never prefixes twice
# ---------------------------------------------------------------------------


def test_a_mapped_drive_stays_a_drive(monkeypatch) -> None:
    _windows(monkeypatch, {"Z:\\": 4})
    assert paths_mod.long_path_for_tree("Z:\\data\\sub-01\\..\\sub-02") == \
        "\\\\?\\Z:\\data\\sub-02"
    assert paths_mod.long_path_for_tree("Z:/data/sub-01") == "\\\\?\\Z:\\data\\sub-01"


def test_a_unc_path_takes_the_unc_form_once(monkeypatch) -> None:
    _windows(monkeypatch)
    assert paths_mod.long_path_for_tree("\\\\server\\share\\data") == \
        "\\\\?\\UNC\\server\\share\\data"
    for already in ("\\\\?\\UNC\\server\\share\\data", "\\\\?\\C:\\data", "\\\\.\\C:\\data"):
        assert paths_mod._prefixed(already) == already


def test_the_tree_form_does_not_resolve(monkeypatch) -> None:
    """``Path.resolve`` is what turned a mapped drive into a UNC path."""
    _windows(monkeypatch)
    monkeypatch.setattr(Path, "resolve", lambda self, *a, **k: pytest.fail("resolved"))
    paths_mod.long_path_for_tree("Z:\\data")


# ---------------------------------------------------------------------------
# A drive this process cannot see
# ---------------------------------------------------------------------------


def test_a_missing_mapped_drive_is_explained(monkeypatch) -> None:
    _windows(monkeypatch)
    message = paths_mod.missing_folder_message("Q:\\study\\raw")
    assert message.startswith("no such folder: Q:\\study\\raw")
    assert "Drive Q: is not visible" in message
    assert "administrator" in message and "\\\\server\\share" in message


def test_off_windows_it_names_the_folder(tmp_path) -> None:
    if os.name == "nt":
        pytest.skip("checks the macOS and Linux answer")
    assert paths_mod.missing_folder_message(tmp_path / "gone") == \
        f"no such folder: {tmp_path / 'gone'}"


def test_the_scan_says_it_before_scanning(tmp_path) -> None:
    from bidsmgr.cli.scan import run_scan

    with pytest.raises(NotADirectoryError, match="no such folder"):
        run_scan(tmp_path / "gone", tmp_path / "inv.tsv", write_log=False)


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def test_every_file_arrives_under_a_short_name(tmp_path) -> None:
    files = _dicoms(tmp_path / "src", 4)
    folder = tmp_path / "stage"
    assert staging.stage_files(files + [tmp_path / "ghost.dcm"], folder) == 4
    staged = sorted(folder.iterdir())
    assert [p.name for p in staged] == ["000000.dcm", "000001.dcm", "000002.dcm", "000003.dcm"]
    assert sorted(p.read_bytes() for p in staged) == sorted(p.read_bytes() for p in files)
    # Staging again starts from an empty folder.
    assert staging.stage_files(files[:2], folder) == 2
    assert len(list(folder.iterdir())) == 2


@pytest.mark.skipif(os.name == "nt", reason="a Windows account may not create links")
def test_links_where_links_work_and_they_are_not_resolved(tmp_path) -> None:
    real = tmp_path / "real"
    _dicoms(real, 2)
    (tmp_path / "via").symlink_to(real, target_is_directory=True)
    files = sorted((tmp_path / "via").iterdir())
    folder = tmp_path / "stage"
    staging.stage_files(files, folder)
    for staged, source in zip(sorted(folder.iterdir()), files):
        assert staged.is_symlink()
        # The path the scan recorded, not the one the file system resolves
        # it to: on Windows that is a mapped drive's UNC path.
        assert os.readlink(staged) == os.path.abspath(source)


def test_any_link_error_switches_to_copies(tmp_path, monkeypatch, caplog) -> None:
    """A share can refuse a link with errors other than "no privilege"."""
    calls = []

    def refuse(src, dst, *a, **k):
        calls.append(src)
        raise OSError(50, "The request is not supported")

    monkeypatch.setattr(staging.os, "symlink", refuse)
    files = _dicoms(tmp_path / "src", 3)
    with caplog.at_level("INFO"):
        assert staging.stage_files(files, tmp_path / "stage") == 3
    assert len(calls) == 1, "one refusal is enough to copy the rest"
    staged = sorted((tmp_path / "stage").iterdir())
    assert not any(p.is_symlink() for p in staged)
    assert [p.read_bytes() for p in staged] == [p.read_bytes() for p in files]
    assert "copying the rest of the series" in caplog.text


def test_a_share_on_either_side_is_copied(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(staging, "is_network_path", _on_share)
    monkeypatch.setattr(staging.os, "symlink",
                        lambda *a, **k: pytest.fail("linked across a share"))
    # The DICOMs on a share, the staging folder local.
    staging.stage_files(_dicoms(tmp_path / SHARE / "raw", 2), tmp_path / "local")
    # The DICOMs local, the staging folder (the project) on a share.
    staging.stage_files(_dicoms(tmp_path / "raw", 2), tmp_path / SHARE / "stage")
    assert len(list((tmp_path / "local").iterdir())) == 2
    assert len(list((tmp_path / SHARE / "stage").iterdir())) == 2


def test_staging_for_a_share_happens_locally(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(staging, "is_network_path", _on_share)
    wanted = tmp_path / SHARE / "project" / "_dicoms"
    local = staging.staging_folder(wanted)
    try:
        assert local != wanted and not _on_share(local)
        assert local.name.startswith("bidsmgr-dicoms-")
    finally:
        local.rmdir()
    assert staging.staging_folder(tmp_path / "here") == tmp_path / "here"


# ---------------------------------------------------------------------------
# The conversion and the probe stage on a share's behalf, and clean up
# ---------------------------------------------------------------------------


def test_the_conversion_stages_locally_for_a_project_on_a_share(tmp_path, monkeypatch) -> None:
    from bidsmgr.converter.backends import dcm2niix_direct as backend
    from bidsmgr.converter.types import ConvertTask

    monkeypatch.setattr(staging, "is_network_path", _on_share)
    seen = {}

    def fake_run(cmd, **kwargs):
        folder = Path(cmd[-1])
        seen["folder"], seen["files"] = folder, sorted(p.name for p in folder.iterdir())
        out = Path(cmd[cmd.index("-o") + 1])
        name = cmd[cmd.index("-f") + 1]
        (out / f"{name}.nii.gz").write_bytes(b"\x1f\x8b")
        (out / f"{name}.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    binary = tmp_path / "dcm2niix"
    binary.write_text("")
    task = ConvertTask(
        row_id="r", series_uid="1.2.3", source_dicom_files=tuple(_dicoms(tmp_path / "raw")),
        dataset="d", bids_root=tmp_path / SHARE / "bids", subject="001",
        datatype="anat", suffix="T1w", entities={"sub": "001"}, basename="sub-001_T1w",
        expected_outputs=(".nii.gz", ".json"))
    result = backend.Dcm2niixDirect(dcm2niix_bin=binary).convert(
        task, tmp_path / SHARE / "bids" / ".tmp_bidsmgr" / "sub-001")
    assert result.success, result.error
    assert not _on_share(seen["folder"])
    assert seen["files"] == ["000000.dcm", "000001.dcm", "000002.dcm"]
    assert not seen["folder"].exists(), "the local staging is removed"
    assert all(_on_share(p) for p in result.staged_files)


def test_a_probe_series_that_fails_is_skipped_not_the_scan(tmp_path, monkeypatch, caplog) -> None:
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="dcm2niix", timeout=1800)

    monkeypatch.setattr(probe_convert, "_run_dcm2niix_full", timeout)
    files = [str(p) for p in _dicoms(tmp_path / "raw")]
    with caplog.at_level("WARNING"):
        assert probe_convert._probe_one_series("1.2.3", files, tmp_path / "w", "dcm2niix") == \
            ("1.2.3", None)
    assert "skipped" in caplog.text

    def unreadable(*a, **k):
        raise PermissionError(13, "the file is locked by another program")

    monkeypatch.setattr(probe_convert, "stage_files", unreadable)
    assert probe_convert._probe_one_series("1.2.4", files, tmp_path / "w2", "dcm2niix") == \
        ("1.2.4", None)


def test_the_probe_removes_its_local_staging(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(staging, "is_network_path", _on_share)
    seen = {}

    def fake_full(folder, out, **k):
        seen["folder"] = folder
        seen["n"] = len(list(folder.iterdir()))
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(probe_convert, "_run_dcm2niix_full", fake_full)
    files = [str(p) for p in _dicoms(tmp_path / "raw")]
    probe_convert._probe_one_series("1.2.3", files, tmp_path / SHARE / ".tmp" / "s", "x")
    assert seen["n"] == 3 and not _on_share(seen["folder"])
    assert not seen["folder"].exists()


# ---------------------------------------------------------------------------
# Classification reads the user's own folder: errors stay in that folder
# ---------------------------------------------------------------------------


def _sidecar(out_dir: Path, uid: str) -> None:
    (out_dir / "1.json").write_text(json.dumps(
        {"BidsGuess": ["anat", "_acq-mprage_T1w"], "SeriesInstanceUID": uid}))


def test_a_timeout_keeps_what_was_written(tmp_path, monkeypatch, caplog) -> None:
    def slow(dicom_dir, out_dir, **k):
        _sidecar(Path(out_dir), "1.2.3")
        raise subprocess.TimeoutExpired(cmd="dcm2niix", timeout=600)

    monkeypatch.setattr(dcm2niix_bidsguess, "_run_dcm2niix_sidecars", slow)
    rows = [InventoryRow(modality="mri", source=tmp_path, series_uid="1.2.3"),
            InventoryRow(modality="mri", source=tmp_path, series_uid="1.2.4")]
    with caplog.at_level("WARNING"):
        out = dcm2niix_bidsguess.classify_dicom_folder(tmp_path, rows, dcm2niix_bin=Path("x"))
    assert [c.suffix for c in out] == ["T1w"]
    assert "could not classify" in caplog.text and "1 sidecar(s)" in caplog.text


def test_an_unreadable_folder_is_that_folders_problem(tmp_path, monkeypatch) -> None:
    def denied(*a, **k):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(dcm2niix_bidsguess, "_run_dcm2niix_sidecars", denied)
    rows = [InventoryRow(modality="mri", source=tmp_path, series_uid="1.2.3")]
    assert dcm2niix_bidsguess.classify_dicom_folder(tmp_path, rows, dcm2niix_bin=Path("x")) == []


def test_a_missing_dcm2niix_is_still_the_scans_problem(tmp_path, monkeypatch) -> None:
    """Said once by the caller ("BidsGuess skipped"), not once per folder."""
    def none(*a, **k):
        raise FileNotFoundError("dcm2niix not found")

    monkeypatch.setattr(dcm2niix_bidsguess, "find_dcm2niix", none)
    rows = [InventoryRow(modality="mri", source=tmp_path, series_uid="1.2.3")]
    with pytest.raises(FileNotFoundError):
        dcm2niix_bidsguess.classify_dicom_folder(tmp_path, rows)


def test_a_share_that_gave_nothing_is_read_from_a_local_copy(tmp_path, monkeypatch, caplog) -> None:
    raw = tmp_path / SHARE / "raw"
    _dicoms(raw / "series1", 2)
    _dicoms(raw / "series1" / "a" / "b" / "c" / "d" / "e" / "too_deep", 1)
    monkeypatch.setattr(dcm2niix_bidsguess, "is_network_path", _on_share)
    seen = []

    def by_place(dicom_dir, out_dir, **k):
        dicom_dir = Path(dicom_dir)
        seen.append((dicom_dir, sorted(p.name for p in dicom_dir.iterdir())))
        if not _on_share(dicom_dir):
            _sidecar(Path(out_dir), "1.2.3")
        return subprocess.CompletedProcess([], 2, "", "Unable to find any DICOM images")

    monkeypatch.setattr(dcm2niix_bidsguess, "_run_dcm2niix_sidecars", by_place)
    rows = [InventoryRow(modality="mri", source=raw, series_uid="1.2.3")]
    with caplog.at_level("WARNING"):
        out = dcm2niix_bidsguess.classify_dicom_folder(raw, rows, dcm2niix_bin=Path("x"))
    assert [c.suffix for c in out] == ["T1w"]
    assert seen[0][0] == raw
    local, names = seen[1]
    # Every file dcm2niix would have read (five folders deep), none deeper.
    assert names == ["000000.dcm", "000001.dcm"]
    assert not local.exists(), "the local copy is removed"
    assert "trying again from a local copy" in caplog.text


def test_a_local_folder_is_not_copied(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dcm2niix_bidsguess, "_run_dcm2niix_sidecars",
                        lambda d, o, **k: calls.append(d) or
                        subprocess.CompletedProcess([], 2, "", ""))
    rows = [InventoryRow(modality="mri", source=tmp_path, series_uid="1.2.3")]
    dcm2niix_bidsguess.classify_dicom_folder(tmp_path, rows, dcm2niix_bin=Path("x"))
    assert calls == [tmp_path]

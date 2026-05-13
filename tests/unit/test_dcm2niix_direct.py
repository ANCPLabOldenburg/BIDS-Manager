"""Unit tests for ``converter/backends/dcm2niix_direct.py``.

dcm2niix is mocked at the ``subprocess.run`` boundary: each test installs
a fake that writes the output files the real binary would write, then
returns a synthetic ``CompletedProcess``. This keeps tests fast and lets
us exercise success/failure/multi-output paths without real DICOMs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

import pytest

from bidsmgr.converter.backends import dcm2niix_direct as backend_mod
from bidsmgr.converter.backends.dcm2niix_direct import (
    Dcm2niixDirect,
    _collect_outputs,
    _missing_expected,
    _safe_dicoms_dirname,
    _stage_dicoms,
)
from bidsmgr.converter.types import ConvertTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dicoms(tmp_path: Path, n: int = 3) -> tuple[Path, ...]:
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for i in range(n):
        p = src / f"IMG{i:04d}.dcm"
        p.write_bytes(b"DICM")
        paths.append(p)
    return tuple(paths)


def _make_task(
    tmp_path: Path,
    *,
    basename: str = "sub-001_T1w",
    datatype: str = "anat",
    suffix: str = "T1w",
    expected: tuple[str, ...] = (".nii.gz", ".json"),
    n_dicoms: int = 3,
) -> ConvertTask:
    return ConvertTask(
        row_id="r1",
        series_uid="1.2.3.4",
        source_dicom_files=_make_dicoms(tmp_path, n_dicoms),
        dataset="study",
        bids_root=tmp_path / "bids" / "study",
        subject="001",
        session=None,
        datatype=datatype,
        suffix=suffix,
        entities={"sub": "001"},
        basename=basename,
        expected_outputs=expected,
    )


def _backend(tmp_path: Path) -> Dcm2niixDirect:
    """Backend instance with a fake binary path (no real dcm2niix call)."""
    fake_bin = tmp_path / "fake_dcm2niix"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    return Dcm2niixDirect(dcm2niix_bin=fake_bin)


def _fake_dcm2niix(
    *, write_files: list[str], returncode: int = 0, stderr: str = ""
) -> Callable[..., subprocess.CompletedProcess]:
    """Return a ``subprocess.run`` replacement that writes ``write_files``.

    Each entry is a basename suffix appended to ``-f <basename>``; the
    helper rewrites them into the ``-o`` directory.
    """

    def _runner(cmd, **kwargs):
        # Cmd is: [bin, -b, y, -ba, n, -z, y, -o, OUTDIR, -f, BASENAME, INDIR]
        out_idx = cmd.index("-o") + 1
        f_idx = cmd.index("-f") + 1
        out_dir = Path(cmd[out_idx])
        basename = cmd[f_idx]
        for suffix in write_files:
            target = out_dir / f"{basename}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if suffix.endswith(".json"):
                target.write_text("{}")
            else:
                target.write_bytes(b"\x1f\x8b\x08\x00")  # gzip magic; fake nii.gz
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout="", stderr=stderr)

    return _runner


# ---------------------------------------------------------------------------
# _safe_dicoms_dirname
# ---------------------------------------------------------------------------


class TestSafeDicomsDirname:
    """The per-series staging dir name must be portable + short.

    Regression for Windows ``WinError 123`` (illegal ``|`` in fmap-pair
    series_uids) and dcm2niix ``rc=2`` triggered by hitting MAX_PATH
    with raw UID-named staging dirs on deep BIDS trees.
    """

    _FORBIDDEN = '<>:"/\\|?*'

    def test_replaces_pipe_with_safe_chars(self) -> None:
        # fmap pair: two UIDs joined by '|' — the original Windows
        # crash trigger.
        joined = (
            "1.3.12.2.1107.5.2.43.66080.2025052611251937202010812.0.0.0"
            "|"
            "1.3.12.2.1107.5.2.43.66080.2025052611251937202710813.0.0.0"
        )
        name = _safe_dicoms_dirname(joined)
        assert "|" not in name
        assert not any(c in name for c in self._FORBIDDEN)

    def test_dirname_stays_short(self) -> None:
        # Long UID — must still produce a short dir name.
        long_uid = "1.3.12.2.1107.5.2.43.66080.2025052611205648833107050.0.0.0"
        name = _safe_dicoms_dirname(long_uid)
        assert len(name) <= 24, f"dir name too long: {name!r}"

    def test_deterministic(self) -> None:
        assert _safe_dicoms_dirname("1.2.3") == _safe_dicoms_dirname("1.2.3")

    def test_unique_per_uid(self) -> None:
        assert _safe_dicoms_dirname("a") != _safe_dicoms_dirname("b")


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_handles_well_formed_task(self, tmp_path: Path) -> None:
        b = _backend(tmp_path)
        assert b.can_handle(_make_task(tmp_path)) is True

    def test_rejects_empty_source_files(self, tmp_path: Path) -> None:
        b = _backend(tmp_path)
        task = _make_task(tmp_path).model_copy(update={"source_files": ()})
        assert b.can_handle(task) is False


# ---------------------------------------------------------------------------
# convert — success paths
# ---------------------------------------------------------------------------


class TestConvertSuccess:
    def test_basename_passthrough(self, tmp_path: Path, monkeypatch) -> None:
        """The backend writes a file at exactly the given basename."""
        b = _backend(tmp_path)
        task = _make_task(tmp_path, basename="sub-007_acq-mprage_T1w")
        monkeypatch.setattr(
            backend_mod.subprocess, "run",
            _fake_dcm2niix(write_files=[".nii.gz", ".json"]),
        )

        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)

        assert result.success is True
        assert result.error is None
        names = sorted(p.name for p in result.staged_files)
        assert names == ["sub-007_acq-mprage_T1w.json", "sub-007_acq-mprage_T1w.nii.gz"]
        # Files are in <staging>/<datatype>/
        for p in result.staged_files:
            assert p.parent == staging / "anat"

    def test_dwi_bval_bvec_collected(self, tmp_path: Path, monkeypatch) -> None:
        b = _backend(tmp_path)
        task = _make_task(
            tmp_path, basename="sub-001_dwi", datatype="dwi", suffix="dwi",
            expected=(".nii.gz", ".json", ".bval", ".bvec"),
        )
        monkeypatch.setattr(
            backend_mod.subprocess, "run",
            _fake_dcm2niix(write_files=[".nii.gz", ".json", ".bval", ".bvec"]),
        )

        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)

        assert result.success is True
        names = sorted(p.name for p in result.staged_files)
        assert names == [
            "sub-001_dwi.bval", "sub-001_dwi.bvec",
            "sub-001_dwi.json", "sub-001_dwi.nii.gz",
        ]

    def test_fmap_multi_output_collected_as_success(self, tmp_path: Path, monkeypatch) -> None:
        """fmap series produce ``_e1``/``_e2``/``_ph`` siblings, not the bare basename.

        That counts as success at this layer — the fmap fixup renames the
        suffixes downstream.
        """
        b = _backend(tmp_path)
        task = _make_task(
            tmp_path, basename="sub-001_fmap", datatype="fmap", suffix="fmap",
            # A hypothetical fmap task with .nii.gz/.json expected — both
            # extensions are present (just on suffixed siblings).
            expected=(".nii.gz", ".json"),
        )
        monkeypatch.setattr(
            backend_mod.subprocess, "run",
            _fake_dcm2niix(write_files=[
                "_e1.nii.gz", "_e1.json",
                "_e2.nii.gz", "_e2.json",
                "_ph.nii.gz", "_ph.json",
            ]),
        )

        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)

        assert result.success is True, result.error
        names = sorted(p.name for p in result.staged_files)
        assert names == [
            "sub-001_fmap_e1.json", "sub-001_fmap_e1.nii.gz",
            "sub-001_fmap_e2.json", "sub-001_fmap_e2.nii.gz",
            "sub-001_fmap_ph.json", "sub-001_fmap_ph.nii.gz",
        ]


# ---------------------------------------------------------------------------
# convert — failure paths
# ---------------------------------------------------------------------------


class TestConvertFailures:
    def test_empty_staging_marks_failure(self, tmp_path: Path) -> None:
        b = _backend(tmp_path)
        # All source files point at non-existent paths.
        task = _make_task(tmp_path).model_copy(
            update={"source_files": (tmp_path / "nope" / "x.dcm",)},
        )
        result = b.convert(task, tmp_path / "staging")
        assert result.success is False
        assert "empty staging" in (result.error or "")

    def test_dcm2niix_nonzero_returncode_marks_failure(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        b = _backend(tmp_path)
        task = _make_task(tmp_path)
        monkeypatch.setattr(
            backend_mod.subprocess, "run",
            _fake_dcm2niix(write_files=[], returncode=2, stderr="dcm2niix: bad header"),
        )

        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)

        assert result.success is False
        assert result.dcm2niix_returncode == 2
        assert "rc=2" in (result.error or "")
        assert "bad header" in result.dcm2niix_stderr_tail

    def test_missing_expected_output_marks_failure(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """rc=0 but no .nii.gz written → flagged."""
        b = _backend(tmp_path)
        task = _make_task(tmp_path)  # expects .nii.gz + .json
        monkeypatch.setattr(
            backend_mod.subprocess, "run",
            _fake_dcm2niix(write_files=[".json"]),  # no .nii.gz
        )

        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)

        assert result.success is False
        assert "missing expected output" in (result.error or "")
        assert ".nii.gz" in (result.error or "")

    def test_subprocess_timeout_marks_failure(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        b = _backend(tmp_path)
        task = _make_task(tmp_path)

        def raise_timeout(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

        monkeypatch.setattr(backend_mod.subprocess, "run", raise_timeout)

        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)

        assert result.success is False
        assert "timed out" in (result.error or "")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestStageDicoms:
    def test_symlinks_every_file(self, tmp_path: Path) -> None:
        files = _make_dicoms(tmp_path, n=4)
        staging = tmp_path / "stage"

        n = _stage_dicoms(files, staging)
        assert n == 4
        # Staging renames each input to a zero-padded sequential name so
        # deep Windows paths stay under MAX_PATH; dcm2niix orders frames
        # by DICOM tags, not by filename, so this is safe.
        assert sorted(p.name for p in staging.iterdir()) == [
            "000000.dcm", "000001.dcm", "000002.dcm", "000003.dcm",
        ]
        # Each entry is a symlink resolving back to the source.
        for p in staging.iterdir():
            assert p.is_symlink()

    def test_skips_missing_files(self, tmp_path: Path) -> None:
        files = list(_make_dicoms(tmp_path, n=2))
        files.append(tmp_path / "ghost.dcm")  # does not exist
        n = _stage_dicoms(tuple(files), tmp_path / "stage")
        assert n == 2

    def test_idempotent_rerun(self, tmp_path: Path) -> None:
        files = _make_dicoms(tmp_path, n=2)
        staging = tmp_path / "stage"
        _stage_dicoms(files, staging)
        # Running again should rebuild the staging dir cleanly.
        n = _stage_dicoms(files, staging)
        assert n == 2


class TestCollectOutputs:
    def test_globs_basename_and_sorts(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        for name in ["sub-1_T1w.json", "sub-1_T1w.nii.gz", "other.json", "sub-1_T1w_e1.nii.gz"]:
            (out / name).write_text("x")

        result = _collect_outputs(out, "sub-1_T1w")
        names = [p.name for p in result]
        assert names == ["sub-1_T1w.json", "sub-1_T1w.nii.gz", "sub-1_T1w_e1.nii.gz"]
        assert "other.json" not in names


class TestMissingExpected:
    def test_exact_match_satisfies(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        nii = out / "sub-1_T1w.nii.gz"
        js = out / "sub-1_T1w.json"
        nii.write_text("")
        js.write_text("")
        missing = _missing_expected(
            [nii, js], (".nii.gz", ".json"), "sub-1_T1w", out,
        )
        assert missing == []

    def test_suffixed_sibling_satisfies(self, tmp_path: Path) -> None:
        """fmap case — ``<basename>_e1.nii.gz`` satisfies ``.nii.gz`` requirement."""
        out = tmp_path / "out"
        out.mkdir()
        sib = out / "sub-1_fmap_e1.nii.gz"
        sib_json = out / "sub-1_fmap_e1.json"
        sib.write_text("")
        sib_json.write_text("")
        missing = _missing_expected(
            [sib, sib_json], (".nii.gz", ".json"), "sub-1_fmap", out,
        )
        assert missing == []

    def test_neither_exact_nor_suffix_flags_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        js = out / "sub-1_T1w.json"
        js.write_text("")
        missing = _missing_expected(
            [js], (".nii.gz", ".json"), "sub-1_T1w", out,
        )
        assert missing == [".nii.gz"]

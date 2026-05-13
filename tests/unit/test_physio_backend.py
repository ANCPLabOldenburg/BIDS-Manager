"""Unit tests for ``PhysioDcmBackend`` and the registry dispatch.

bidsphysio's ``dcm2bidsphysio.dcm2bids`` is mocked; we don't read real
DICOM physio files in unit tests. Real-data coverage lives in the
gated ``tests/real_data`` suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from bidsmgr.converter import default_backends, dispatch
from bidsmgr.converter.backends import physio_dcm as physio_mod
from bidsmgr.converter.backends.dcm2niix_direct import Dcm2niixDirect
from bidsmgr.converter.backends.physio_dcm import PhysioDcmBackend
from bidsmgr.converter.types import ConvertTask


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_physio_dicoms(tmp_path: Path) -> tuple[Path, ...]:
    folder = tmp_path / "src"
    folder.mkdir()
    fp = folder / "ses-pre_task-mb_bold_PhysioLog.dcm"
    fp.write_bytes(b"DICM")
    return (fp,)


def _make_task(
    tmp_path: Path,
    *,
    suffix: str = "physio",
    datatype: str = "func",
    basename: str = "sub-001_task-mb_physio",
    files: Sequence[Path] | None = None,
) -> ConvertTask:
    if files is None:
        files = _make_physio_dicoms(tmp_path)
    return ConvertTask(
        row_id="r1",
        series_uid="1.2.3",
        source_dicom_files=tuple(files),
        dataset="study",
        bids_root=tmp_path / "bids" / "study",
        subject="001",
        session=None,
        datatype=datatype,
        suffix=suffix,
        entities={"sub": "001"},
        basename=basename,
        expected_outputs=(".tsv.gz", ".json"),
    )


class _FakePhysioData:
    """Mimics bidsphysio.base.bidsphysio.PhysioData.save_to_bids."""

    def __init__(self, write_files: Sequence[str]) -> None:
        self._write_files = list(write_files)

    def save_to_bids(self, bids_prefix: str) -> None:
        prefix = Path(bids_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix in self._write_files:
            target = prefix.parent / f"{prefix.name}{suffix}"
            if suffix.endswith(".json"):
                target.write_text("{}")
            elif suffix.endswith(".tsv.gz"):
                target.write_bytes(b"\x1f\x8b")  # gzip magic
            else:
                target.write_bytes(b"")


def _patch_dcm2bids(monkeypatch, factory_or_exc) -> None:
    """Patch the lazy import of bidsphysio's ``dcm2bids`` callable.

    ``factory_or_exc`` can be:
    * a callable taking ``source_paths`` and returning ``_FakePhysioData``
    * an Exception instance to raise from the call
    """
    import importlib
    import sys

    fake_module = type(sys.modules["bidsmgr"])("bidsphysio.dcm2bids.dcm2bidsphysio")

    def _runner(source_paths):
        if isinstance(factory_or_exc, BaseException):
            raise factory_or_exc
        return factory_or_exc(source_paths)

    fake_module.dcm2bids = _runner

    parent_pkg = type(sys.modules["bidsmgr"])("bidsphysio.dcm2bids")
    parent_pkg.dcm2bidsphysio = fake_module
    root_pkg = type(sys.modules["bidsmgr"])("bidsphysio")
    root_pkg.dcm2bids = parent_pkg

    monkeypatch.setitem(sys.modules, "bidsphysio.dcm2bids.dcm2bidsphysio", fake_module)
    monkeypatch.setitem(sys.modules, "bidsphysio.dcm2bids", parent_pkg)
    monkeypatch.setitem(sys.modules, "bidsphysio", root_pkg)


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_handles_physio_with_physiolog_filename(self, tmp_path: Path) -> None:
        b = PhysioDcmBackend()
        task = _make_task(tmp_path)
        assert b.can_handle(task) is True

    def test_rejects_non_physio_suffix(self, tmp_path: Path) -> None:
        b = PhysioDcmBackend()
        task = _make_task(tmp_path, suffix="bold", basename="sub-001_task-mb_bold")
        assert b.can_handle(task) is False

    def test_rejects_empty_source_files(self, tmp_path: Path) -> None:
        b = PhysioDcmBackend()
        task = _make_task(tmp_path).model_copy(update={"source_files": ()})
        assert b.can_handle(task) is False


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


class TestRegistryDispatch:
    def test_default_backends_priority_order(self) -> None:
        backends = default_backends()
        # Priority: physio (narrowest), mne-bids (eeg/meg/ieeg/nirs),
        # dcm2niix-direct (broad MRI fallback).
        assert [b.name for b in backends] == [
            "physio_dcm", "mne_bids", "dcm2niix_direct",
        ]

    def test_physio_task_routes_to_physio_backend(self, tmp_path: Path) -> None:
        backends = default_backends()
        task = _make_task(tmp_path)
        chosen = dispatch(backends, task)
        assert isinstance(chosen, PhysioDcmBackend)

    def test_mri_task_routes_to_dcm2niix(self, tmp_path: Path) -> None:
        backends = default_backends()
        # Build a "normal" T1 task — physio backend declines, dcm2niix accepts.
        src = tmp_path / "src" / "T1"
        src.mkdir(parents=True)
        (src / "IMG0001.dcm").write_bytes(b"DICM")
        task = ConvertTask(
            row_id="r1", series_uid="1.2.3.T1",
            source_dicom_files=(src / "IMG0001.dcm",),
            dataset="study",
            bids_root=tmp_path / "bids" / "study",
            subject="001", session=None,
            datatype="anat", suffix="T1w",
            entities={"sub": "001"},
            basename="sub-001_T1w",
            expected_outputs=(".nii.gz", ".json"),
        )
        chosen = dispatch(backends, task)
        assert isinstance(chosen, Dcm2niixDirect)

    def test_dcm2niix_declines_physio_rows(self, tmp_path: Path) -> None:
        """Even called directly, dcm2niix backend must say it can't
        handle physio (no nii.gz to produce)."""
        b = Dcm2niixDirect()
        task = _make_task(tmp_path)
        assert b.can_handle(task) is False


# ---------------------------------------------------------------------------
# convert (success / failure paths) — bidsphysio mocked
# ---------------------------------------------------------------------------


class TestConvertSuccess:
    def test_writes_tsvgz_and_json_pair(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        b = PhysioDcmBackend()
        task = _make_task(tmp_path)
        _patch_dcm2bids(
            monkeypatch,
            lambda paths: _FakePhysioData([".tsv.gz", ".json"]),
        )

        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)

        assert result.success is True, result.error
        names = sorted(p.name for p in result.staged_files)
        assert names == [
            "sub-001_task-mb_physio.json",
            "sub-001_task-mb_physio.tsv.gz",
        ]
        # Files land under <staging>/<datatype>/.
        for p in result.staged_files:
            assert p.parent == staging / "func"

    def test_multi_sampling_rate_writes_recording_pairs(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When the recording has multiple sampling rates, bidsphysio
        writes ``_recording-<rate>_physio.tsv.gz`` siblings."""
        b = PhysioDcmBackend()
        task = _make_task(tmp_path)
        # Simulate ECG@400Hz + RESP@50Hz — bidsphysio appends recording entity.
        _patch_dcm2bids(
            monkeypatch,
            lambda paths: _FakePhysioData([
                "_recording-400Hz_physio.tsv.gz",
                "_recording-400Hz_physio.json",
                "_recording-50Hz_physio.tsv.gz",
                "_recording-50Hz_physio.json",
            ]),
        )
        staging = tmp_path / "staging"
        staging.mkdir()
        # Adjust task basename so save_to_bids uses ``sub-001_task-mb`` as
        # the prefix (bidsphysio chooses to append _physio itself in that
        # mode). We test the file collection logic accepts whatever the
        # save_to_bids call produced.
        task = task.model_copy(update={"basename": "sub-001_task-mb"})
        result = b.convert(task, staging)

        assert result.success is True
        names = sorted(p.name for p in result.staged_files)
        assert names == [
            "sub-001_task-mb_recording-400Hz_physio.json",
            "sub-001_task-mb_recording-400Hz_physio.tsv.gz",
            "sub-001_task-mb_recording-50Hz_physio.json",
            "sub-001_task-mb_recording-50Hz_physio.tsv.gz",
        ]


class TestConvertFailure:
    def test_empty_staging_is_failure(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        b = PhysioDcmBackend()
        # All source files non-existent.
        task = _make_task(tmp_path).model_copy(
            update={"source_files": (tmp_path / "nope.dcm",)},
        )
        _patch_dcm2bids(
            monkeypatch,
            lambda paths: _FakePhysioData([".tsv.gz", ".json"]),
        )
        result = b.convert(task, tmp_path / "staging")
        assert result.success is False
        assert "empty staging" in (result.error or "")

    def test_dcm2bids_exception_is_recorded(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        b = PhysioDcmBackend()
        task = _make_task(tmp_path)
        _patch_dcm2bids(monkeypatch, RuntimeError("malformed PMU log"))
        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)
        assert result.success is False
        assert "bidsphysio.dcm2bids failed" in (result.error or "")
        assert "malformed PMU log" in (result.error or "")

    def test_no_outputs_produced_is_failure(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        b = PhysioDcmBackend()
        task = _make_task(tmp_path)
        _patch_dcm2bids(
            monkeypatch,
            lambda paths: _FakePhysioData([]),  # save_to_bids writes nothing
        )
        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)
        assert result.success is False
        assert "produced no output files" in (result.error or "")

"""Unit tests for ``PhysioDcmBackend`` and the registry dispatch.

The conversion reads a synthetic CMRR physio DICOM laid out as the scanner
writes it (``tests/fixtures/cmrr_physio.py``); the reader and writer have their
own tests (``test_physio_cmrr.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


from bidsmgr.converter import default_backends, dispatch
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
        # Priority: eye tracking (an EyeLink file shares .edf with EEG, so it
        # must be claimed first), physio, ecat (pet rows with an ECAT
        # signature), mne-bids (eeg/meg/ieeg/nirs), dcm2niix-direct (broad
        # MRI fallback, and the DICOM PET path).
        assert [b.name for b in backends] == [
            "eyelink", "physio_dcm", "ecat_direct", "mne_bids", "dcm2niix_direct",
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
    """Real reading: a synthetic CMRR physio DICOM (four ECG leads, pulse,
    respiration, the acquisition log), as the scanner lays it out."""

    def _dicom_task(self, tmp_path: Path, **update) -> ConvertTask:
        from tests.fixtures.cmrr_physio import physio_dicom

        dcm = physio_dicom(tmp_path / "src" / "ses-pre_task-mb_bold_PhysioLog.dcm")
        return _make_task(tmp_path, files=(dcm,)).model_copy(update=update)

    def test_one_table_per_rate_with_the_trigger_last(self, tmp_path: Path) -> None:
        task = self._dicom_task(tmp_path)
        staging = tmp_path / "staging"
        staging.mkdir()
        result = PhysioDcmBackend().convert(task, staging)

        assert result.success is True, result.error
        names = sorted(p.name for p in result.staged_files)
        assert names == [
            "sub-001_task-mb_recording-cardiac_physio.json",
            "sub-001_task-mb_recording-cardiac_physio.tsv.gz",
            "sub-001_task-mb_recording-ecg_physio.json",
            "sub-001_task-mb_recording-ecg_physio.tsv.gz",
            "sub-001_task-mb_recording-respiratory_physio.json",
            "sub-001_task-mb_recording-respiratory_physio.tsv.gz",
        ]
        for path in result.staged_files:
            assert path.parent == staging / "func"
            if path.suffix == ".json":
                assert json.loads(path.read_text())["Columns"][-1] == "trigger"

    def test_the_task_chooses_one_table_aligned_to_the_run(self, tmp_path: Path) -> None:
        task = self._dicom_task(tmp_path, physio_layout="one_interpolated", physio_align=True,
                                drop_volumes=2)
        staging = tmp_path / "staging"
        staging.mkdir()
        result = PhysioDcmBackend().convert(task, staging)

        assert result.success is True, result.error
        assert sorted(p.name for p in result.staged_files) == [
            "sub-001_task-mb_physio.json", "sub-001_task-mb_physio.tsv.gz"]
        side = json.loads((staging / "func/sub-001_task-mb_physio.json").read_text())
        assert 0 <= side["StartTime"] < 1 / side["SamplingFrequency"] + 1e-9
        assert "2 volumes removed" in side["trigger"]["Description"]


class TestConvertFailure:
    def test_empty_staging_is_failure(self, tmp_path: Path) -> None:
        b = PhysioDcmBackend()
        # All source files non-existent.
        task = _make_task(tmp_path).model_copy(
            update={"source_files": (tmp_path / "nope.dcm",)},
        )
        result = b.convert(task, tmp_path / "staging")
        assert result.success is False
        assert "empty staging" in (result.error or "")

    def test_an_unreadable_log_is_recorded(self, tmp_path: Path) -> None:
        # The fixture's "DICM" stub is not a DICOM: the reason is reported.
        b = PhysioDcmBackend()
        task = _make_task(tmp_path)
        staging = tmp_path / "staging"
        staging.mkdir()
        result = b.convert(task, staging)
        assert result.success is False
        assert "the physiological log could not be read" in (result.error or "")

    def test_a_log_with_only_volume_timings_writes_the_trigger_alone(
            self, tmp_path: Path) -> None:
        # Logging on, no sensor attached: the volume timings are what the
        # log holds, and they are written (1.5.2 first wrote nothing).
        from tests.fixtures.cmrr_physio import info_log

        log = tmp_path / "src" / "Physio_Info.log"
        log.parent.mkdir(parents=True)
        log.write_text(info_log(5))
        task = _make_task(tmp_path, files=(log,))
        staging = tmp_path / "staging"
        staging.mkdir()
        result = PhysioDcmBackend().convert(task, staging)
        assert result.success is True, result.error
        names = sorted(Path(p).name for p in result.staged_files)
        assert len(names) == 2 and all(n.endswith(("_physio.json", "_physio.tsv.gz"))
                                       for n in names)
        sidecar = next(Path(p) for p in result.staged_files if str(p).endswith(".json"))
        meta = json.loads(sidecar.read_text())
        assert meta["Columns"] == ["trigger"] and meta["SamplingFrequency"] == 400.0

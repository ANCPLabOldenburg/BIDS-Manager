from __future__ import annotations

from textwrap import dedent
from pathlib import Path

import pandas as pd
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from bids_manager.run_heudiconv_from_heuristic import (
    convert_physio_series,
    load_heuristic_module,
)


def _write_dicom(path: Path, series_uid: str, description: str) -> None:
    """Create a minimal DICOM file containing the provided metadata."""

    meta = Dataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(path.name, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = description
    ds.PatientName = "Test^Patient"
    ds.save_as(path)


def _write_simple_heuristic(path: Path) -> None:
    path.write_text(
        dedent(
            """
            from typing import Tuple

            def create_key(template: str,
                           outtype: Tuple[str, ...] = ("nii.gz",),
                           annotation_classes=None):
                return template, outtype, annotation_classes

            SID_MAP = {"Subject": "sub-001"}

            key_physio = create_key("sub-001/func/sub-001_task-rest_run-01_physio")

            def infotodict(seqinfo):
                key_physio_list = []
                info = {key_physio: key_physio_list}
                for s in seqinfo:
                    if (
                        s.series_description == "PHYSIO"
                        and s.dcm_dir_name == "PHYSIO"
                        and getattr(s, "series_uid", "") == "123"
                    ):
                        key_physio_list.append(s.series_id)
                return info
            """
        ),
        encoding="utf-8",
    )


def test_convert_physio_series_invokes_bidsphysio(tmp_path, monkeypatch):
    heur_path = tmp_path / "heuristic_physio.py"
    _write_simple_heuristic(heur_path)
    module = load_heuristic_module(heur_path)

    raw_root = tmp_path / "raw"
    bids_out = tmp_path / "bids"
    physio_dir = raw_root / "Subject" / "PHYSIO"
    physio_dir.mkdir(parents=True)
    physio_file = physio_dir / "physio.dcm"
    _write_dicom(physio_file, "123", "PHYSIO")

    df = pd.DataFrame(
        [
            {
                "StudyDescription": "Example Study",
                "source_folder": "Subject/PHYSIO",
                "sequence": "PHYSIO",
                "series_uid": "123",
                "modality": "physio",
                "include": 1,
            }
        ]
    )

    calls: list[tuple[str, str]] = []

    def fake_dcm2bids(src: str, prefix: str) -> None:
        calls.append((src, prefix))

    monkeypatch.setattr(
        "bids_manager.run_heudiconv_from_heuristic.dcm2bidsphysio.dcm2bids",
        fake_dcm2bids,
    )

    convert_physio_series(raw_root, bids_out, module, df)

    assert calls == [
        (
            str(physio_file),
            str(bids_out / "sub-001/func/sub-001_task-rest_run-01_physio"),
        )
    ]


def test_convert_physio_series_prefers_matching_uid(tmp_path, monkeypatch):
    heur_path = tmp_path / "heuristic_physio.py"
    _write_simple_heuristic(heur_path)
    module = load_heuristic_module(heur_path)

    raw_root = tmp_path / "raw"
    bids_out = tmp_path / "bids"
    physio_dir = raw_root / "Subject" / "PHYSIO"
    physio_dir.mkdir(parents=True)

    unrelated = physio_dir / "other.dcm"
    target = physio_dir / "physio.dcm"
    _write_dicom(unrelated, "999", "OTHER")
    _write_dicom(target, "123", "PHYSIO")

    df = pd.DataFrame(
        [
            {
                "StudyDescription": "Example Study",
                "source_folder": "Subject/PHYSIO",
                "sequence": "PHYSIO",
                "series_uid": "123",
                "modality": "physio",
                "include": 1,
            }
        ]
    )

    calls: list[tuple[str, str]] = []

    def fake_dcm2bids(src: str, prefix: str) -> None:
        calls.append((src, prefix))

    monkeypatch.setattr(
        "bids_manager.run_heudiconv_from_heuristic.dcm2bidsphysio.dcm2bids",
        fake_dcm2bids,
    )

    convert_physio_series(raw_root, bids_out, module, df)

    assert calls, "Physio conversion should have been invoked"
    assert calls[0][0] == str(target)


def test_convert_physio_series_respects_include_flag(tmp_path, monkeypatch):
    heur_path = tmp_path / "heuristic_physio.py"
    _write_simple_heuristic(heur_path)
    module = load_heuristic_module(heur_path)

    raw_root = tmp_path / "raw"
    bids_out = tmp_path / "bids"
    physio_dir = raw_root / "Subject" / "PHYSIO"
    physio_dir.mkdir(parents=True)
    _write_dicom(physio_dir / "physio.dcm", "123", "PHYSIO")

    df = pd.DataFrame(
        [
            {
                "StudyDescription": "Example Study",
                "source_folder": "Subject/PHYSIO",
                "sequence": "PHYSIO",
                "series_uid": "123",
                "modality": "physio",
                "include": 0,
            }
        ]
    )

    fake = lambda src, prefix: (_ for _ in ()).throw(AssertionError("should not run"))

    monkeypatch.setattr(
        "bids_manager.run_heudiconv_from_heuristic.dcm2bidsphysio.dcm2bids",
        fake,
    )

    convert_physio_series(raw_root, bids_out, module, df)


def test_convert_physio_series_skips_when_uid_missing(tmp_path, monkeypatch):
    heur_path = tmp_path / "heuristic_physio.py"
    _write_simple_heuristic(heur_path)
    module = load_heuristic_module(heur_path)

    raw_root = tmp_path / "raw"
    bids_out = tmp_path / "bids"
    physio_dir = raw_root / "Subject" / "PHYSIO"
    physio_dir.mkdir(parents=True)
    _write_dicom(physio_dir / "first.dcm", "111", "OTHER")
    _write_dicom(physio_dir / "second.dcm", "222", "OTHER")

    df = pd.DataFrame(
        [
            {
                "StudyDescription": "Example Study",
                "source_folder": "Subject/PHYSIO",
                "sequence": "PHYSIO",
                "series_uid": "999",
                "modality": "physio",
                "include": 1,
            }
        ]
    )

    calls: list[tuple[str, str]] = []

    def fake_dcm2bids(src: str, prefix: str) -> None:
        calls.append((src, prefix))

    monkeypatch.setattr(
        "bids_manager.run_heudiconv_from_heuristic.dcm2bidsphysio.dcm2bids",
        fake_dcm2bids,
    )

    convert_physio_series(raw_root, bids_out, module, df)

    assert calls == []

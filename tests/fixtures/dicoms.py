"""A small MR series dcm2niix really converts (every geometry tag it needs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_mr_series(folder: Path, series_uid: str, *, n_slices: int = 4, size: int = 8,
                    description: str = "t1_test", value: int = 100) -> list[Path]:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    folder.mkdir(parents=True, exist_ok=True)
    study = generate_uid()
    frame = generate_uid()
    out = []
    for k in range(n_slices):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = MRImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        path = folder / f"{description}_{k:03d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.Modality = "MR"
        ds.Manufacturer = "SIEMENS"
        ds.PatientID = "P1"
        ds.PatientName = "Test^Subject"
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame
        ds.SeriesNumber = 3
        ds.SeriesDescription = description
        ds.ProtocolName = description
        ds.InstanceNumber = k + 1
        ds.AcquisitionNumber = 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
        ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = "20260101"
        ds.StudyTime = ds.SeriesTime = ds.AcquisitionTime = "120000.000000"
        ds.ImagePositionPatient = [0.0, 0.0, float(k) * 2.0]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.0
        ds.SpacingBetweenSlices = 2.0
        ds.MagneticFieldStrength = 3.0
        ds.EchoTime = 3.0
        ds.RepetitionTime = 2000.0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = size
        ds.Columns = size
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        pixels = np.full((size, size), value + k, dtype=np.uint16)
        ds.PixelData = pixels.tobytes()
        ds.save_as(str(path), enforce_file_format=True)
        out.append(path)
    return out

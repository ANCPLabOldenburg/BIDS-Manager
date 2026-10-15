"""Converting one DICOM series for a look before the real conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")

from bidsmgr.inventory.probe_convert import preview_series, series_files  # noqa: E402
from tests.fixtures.dicoms import write_mr_series  # noqa: E402


def test_only_the_series_asked_for_is_collected(tmp_path):
    folder = tmp_path / "raw" / "session1"
    write_mr_series(folder, "1.2.3.1", description="t1")
    write_mr_series(folder / "sub", "1.2.3.2", description="t2")
    (folder / "notes.txt").write_text("not a DICOM")
    found = series_files(folder, "1.2.3.1")
    assert len(found) == 4 and all("t1_" in Path(f).name for f in found)
    assert len(series_files(folder, "1.2.3.2")) == 4, "found in a subfolder too"


def test_the_series_converts_into_the_work_folder(tmp_path):
    folder = tmp_path / "raw"
    write_mr_series(folder, "1.2.3.1", n_slices=4, size=8)
    write_mr_series(folder, "1.2.3.9", description="other")
    work = tmp_path / "work"
    images = preview_series(series_files(folder, "1.2.3.1"), work)
    assert len(images) == 1
    assert images[0].parent == work / "out"
    img = nib.load(str(images[0]))
    assert sorted(img.shape) == [4, 8, 8]
    data = np.asarray(img.dataobj)
    assert data.min() >= 100 and data.max() <= 103, "only this series' slices"


def test_an_unknown_series_says_so(tmp_path):
    folder = tmp_path / "raw"
    write_mr_series(folder, "1.2.3.1")
    with pytest.raises(ValueError, match="no DICOM of this series"):
        preview_series(series_files(folder, "9.9.9"), tmp_path / "work")

"""One inventory row, converted for a look the way the conversion will.

Before this the preview ran plain dcm2niix on one series UID: a fieldmap
(two series in one row) found nothing, a physiology log made no image, a
spectrum opened in the volume viewer, an EEG row had no preview, and a series
lying directly in the scanned folder was not found at all. Now the row goes
through the conversion's own task builder and backends, and each file it
makes is one a viewer opens.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pydicom")

from bidsmgr.cli.convert import preview_row  # noqa: E402
from bidsmgr.inventory import sources  # noqa: E402
from bidsmgr.inventory.probe_convert import series_files  # noqa: E402
from bidsmgr.viz.data.formats import kind_of  # noqa: E402
from tests.fixtures import data_root, sample_data  # noqa: E402
from tests.fixtures.dicoms import write_mr_series  # noqa: E402


def _bold_row(**over) -> dict:
    row = {
        "participant_id": "sub-001", "session": "", "include": 1, "modality": "mri",
        "datatype": "func", "bids_name": "sub-001_task-rest_bold",
        "bids_guess_suffix": "bold", "entities": "", "series_uid": "1.2.3.1",
        "source_folder": "s1", "source_file": "", "issues": "",
    }
    row.update(over)
    return row


def _sources(root: Path, *uids: str) -> sources.ScanSources:
    return sources.ScanSources(root, root, {u: series_files(root, u) for u in uids})


def test_a_series_converts_under_the_name_it_will_have(tmp_path):
    raw = tmp_path / "raw"
    write_mr_series(raw / "s1", "1.2.3.1", description="rest")
    files = preview_row(_bold_row(), tmp_path / "work", _sources(raw, "1.2.3.1"))
    assert [f.name for f in files] == ["sub-001_task-rest_bold.nii.gz"]
    assert kind_of(files[0]) == "volume" and (tmp_path / "work") in files[0].parents


def test_moved_data_previews_from_where_it_went(tmp_path):
    raw, moved = tmp_path / "raw", tmp_path / "moved"
    write_mr_series(raw / "s1", "1.2.3.1", description="rest")
    fbu = {"1.2.3.1": series_files(raw, "1.2.3.1")}
    shutil.move(str(raw), str(moved))
    df = pd.DataFrame([_bold_row()])
    found, seen = sources.locate(df, fbu, moved, scanned=raw)
    assert seen.state == "ok"
    assert preview_row(_bold_row(), tmp_path / "work", found)


def test_a_series_the_plan_does_not_convert_is_still_shown(tmp_path):
    """A scout has no BIDS name, so no conversion task: plain dcm2niix."""
    raw = tmp_path / "raw"
    write_mr_series(raw / "s1", "1.2.3.1", description="localizer")
    row = _bold_row(datatype="", bids_name="", bids_guess_suffix="", include=0)
    files = preview_row(row, tmp_path / "work", _sources(raw, "1.2.3.1"))
    assert len(files) == 1 and kind_of(files[0]) == "volume"


def test_a_recording_is_shown_as_it_is(tmp_path):
    pytest.importorskip("mne")
    from tests.fixtures.signals import write_fif

    write_fif(tmp_path / "raw" / "sub-01", "rest_eeg.fif")
    row = _bold_row(series_uid="", datatype="eeg", source_file="sub-01/rest_eeg.fif",
                    bids_name="sub-001_task-rest_eeg")
    work = tmp_path / "work"
    files = preview_row(row, work, sources.ScanSources(tmp_path / "raw", tmp_path / "raw"))
    assert files == [tmp_path / "raw" / "sub-01" / "rest_eeg.fif"]
    assert kind_of(files[0]) == "signal" and not any(work.glob("**/*.fif")), "not copied"


def test_missing_data_says_where_it_looked(tmp_path):
    row = _bold_row(series_uid="", datatype="eeg", source_file="sub-01/gone.edf")
    with pytest.raises(FileNotFoundError, match="gone.edf"):
        preview_row(row, tmp_path / "work", sources.ScanSources(tmp_path, tmp_path))


# ---------------------------------------------------------------------------
# Real data: every kind of row the user found unpreviewable
# ---------------------------------------------------------------------------


def _scanned(raw: Path, tmp_path: Path):
    from bidsmgr.cli.scan import run_scan

    inv = tmp_path / "inventory.tsv"
    df = run_scan(raw, inv, n_jobs=2).fillna("")
    found, seen = sources.resolve(df, sources.load_files_by_uid(inv), recorded=raw,
                                  label=raw.name)
    assert seen.state == "ok"
    return df, found


def _preview(df, found, tmp_path, **match) -> list[Path]:
    rows = df
    for col, value in match.items():
        rows = rows[rows[col] == value]
    assert len(rows), f"no row with {match}"
    return preview_row(rows.iloc[0].to_dict(), tmp_path / "work", found)


def test_real_fieldmap_pet_eeg_and_meg_rows(tmp_path):
    raw = sample_data.require("multimodal")
    df, found = _scanned(raw, tmp_path)
    fmap = sorted(f.name.split("_")[-1] for f in _preview(df, found, tmp_path / "f",
                                                          datatype="fmap"))
    assert fmap == ["magnitude1.nii.gz", "magnitude2.nii.gz", "phasediff.nii.gz"]
    (pet,) = _preview(df, found, tmp_path / "p", datatype="pet")
    assert kind_of(pet) == "volume"
    for datatype in ("eeg", "meg"):
        (rec,) = _preview(df, found, tmp_path / datatype, datatype=datatype)
        assert kind_of(rec) == "signal" and raw in rec.parents


@pytest.mark.skipif(not data_root.have("MRI", "MRI_sample_data_neuroimging_unit_oldenburg_2",
                                       "OL_0003"),
                    reason=data_root.why_missing("MRI"))
def test_real_physio_and_a_series_lying_in_the_scanned_folder(tmp_path):
    raw = data_root.dataset("MRI", "MRI_sample_data_neuroimging_unit_oldenburg_2", "OL_0003")
    df, found = _scanned(raw, tmp_path)
    assert set(df["source_folder"]) == {"OL_0003"}, "every series is in the root itself"
    (physio,) = _preview(df, found, tmp_path / "ph", bids_guess_suffix="physio")
    assert physio.name.endswith("_physio.tsv.gz") and kind_of(physio) == "physio"


@pytest.mark.skipif(not data_root.have("MRS"), reason=data_root.why_missing("MRS"))
def test_real_spectroscopy_opens_as_a_spectrum(tmp_path):
    df, found = _scanned(data_root.dataset("MRS"), tmp_path)
    (spectrum,) = _preview(df, found, tmp_path / "m", datatype="mrs")
    assert kind_of(spectrum) == "spectrum"

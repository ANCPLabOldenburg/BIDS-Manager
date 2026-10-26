"""The coherent multimodal sample: one participant, four visits.

Multimodal sample 2 holds the same recordings as the multimodal sample, made
coherent: every file names the participant ``OL_3844`` in the field its format
has for it (or, for the eye tracker, lies in that participant's folder), and
every file was made on the day of its visit. What it exists to show is that
the scan then proposes ONE participant for everything, so that is pinned
here, with the dates each modality's own reader gives.

Fetched once and cached outside the checkout (``tests/fixtures/sample_data``);
skipped when the machine is offline and has no cache. A scan only, no
conversion: the conversion of these recordings is the multimodal sample's
test.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from tests.fixtures import sample_data

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def scanned(tmp_path_factory) -> pd.DataFrame:
    raw = sample_data.require("multimodal2")
    out = tmp_path_factory.mktemp("multimodal2") / "inventory.tsv"
    proc = subprocess.run([sys.executable, "-m", "bidsmgr.cli.scan", str(raw), str(out)],
                          capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, (proc.stderr or proc.stdout)[-2000:]
    return pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)


def test_every_recording_is_one_participant(scanned) -> None:
    """The DICOM series by the patient fields they share, EEG and MEG by
    their folder, the eye tracker by its folder because everything else in
    it is that participant (``cli.scan._join_folder_subject``)."""
    assert set(scanned["participant_id"]) == {"sub-001"}, \
        scanned[["modality", "participant_id", "source_folder"]].to_string()
    assert set(scanned["modality"]) >= {"mri", "pet", "eeg", "meg", "eyetrack"}


def test_each_modality_is_on_its_visit(scanned) -> None:
    days = {m: set(scanned.loc[scanned["modality"] == m, "study_date"]) - {""}
            for m in ("pet", "eyetrack")}
    assert days["pet"] == {"20231005"}
    assert days["eyetrack"] == {"20230914"}, "recorded during the resting BOLD run"
    mri = scanned[(scanned["modality"] == "mri") & (scanned["include"] == "1")]
    assert set(mri["study_date"]) == {"20230914"}, "MRI and spectroscopy: one visit"

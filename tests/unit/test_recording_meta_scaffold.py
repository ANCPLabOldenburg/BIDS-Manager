"""Tests for the scan-time recording-metadata scaffold writer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bidsmgr.cli.scan import _write_recording_meta_scaffold
from bidsmgr.recording_meta import load_spec, scaffold_sidecar_path


def _eeg_df(event_codes, manufacturer="", model=""):
    return pd.DataFrame([
        {
            "source_file": "sub-001/rec.edf",
            "datatype": "eeg",
            "_event_codes": json.dumps(event_codes),
            "_manufacturer": manufacturer,
            "_model": model,
        },
        {
            "source_file": "sub-002/rec.edf",
            "datatype": "eeg",
            "_event_codes": json.dumps(event_codes),
            "_manufacturer": manufacturer,
            "_model": model,
        },
    ])


def test_scaffold_written_with_blank_event_labels(tmp_path):
    tsv = tmp_path / "inv.tsv"
    df = _eeg_df(["T1", "T0", "T2"], manufacturer="Elekta", model="TRIUX")
    out = _write_recording_meta_scaffold(df, tsv)
    assert out == scaffold_sidecar_path(tsv)
    spec = load_spec(out)
    # Codes seeded sorted, labels blank for the user to fill.
    assert spec.event_maps["*"] == {"T0": "", "T1": "", "T2": ""}
    # Device fields are NOT seeded - manufacturer is a read-only suggestion and
    # mne-bids fills the MEG sidecar value, so seeding would wrongly overwrite it.
    assert spec.defaults.manufacturer is None
    assert spec.defaults.amplifier_model is None


def test_scaffold_not_overwritten(tmp_path):
    tsv = tmp_path / "inv.tsv"
    sidecar = scaffold_sidecar_path(tsv)
    sidecar.write_text('{"schema_version": 1, "event_maps": {"*": {"T0": "rest"}}}', encoding="utf-8")
    out = _write_recording_meta_scaffold(_eeg_df(["T0"]), tsv)
    assert out is None  # preserved
    assert load_spec(sidecar).event_maps["*"]["T0"] == "rest"  # user label intact


def test_no_scaffold_when_nothing_detected(tmp_path):
    tsv = tmp_path / "inv.tsv"
    out = _write_recording_meta_scaffold(_eeg_df([], manufacturer="", model=""), tsv)
    assert out is None
    assert not scaffold_sidecar_path(tsv).exists()


def test_no_scaffold_for_mri_only(tmp_path):
    tsv = tmp_path / "inv.tsv"
    mri = pd.DataFrame([{"series_uid": "1.2.3", "source_file": ""}])
    assert _write_recording_meta_scaffold(mri, tsv) is None


def test_manufacturer_alone_does_not_seed(tmp_path):
    """A manufacturer with no event codes seeds NOTHING (manufacturer is a
    read-only suggestion now, not a scaffold default)."""
    tsv = tmp_path / "inv.tsv"
    out = _write_recording_meta_scaffold(_eeg_df([], manufacturer="BrainProducts"), tsv)
    assert out is None
    assert not scaffold_sidecar_path(tsv).exists()

"""Tests for the phenotype/ table importer."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from bidsmgr.metadata.phenotype import write_phenotype


def _report():
    return SimpleNamespace(files_written=[], warnings=[])


def test_writes_measure_tsv_and_json(tmp_path):
    bids_root = tmp_path / "ds"
    bids_root.mkdir()
    src = tmp_path / "edinburgh.tsv"
    src.write_text(
        "participant_id\tehi_total\tehi_decile\n"
        "sub-001\t80\t9\n"
        "sub-002\t-40\t2\n",
        encoding="utf-8",
    )
    report = _report()
    write_phenotype(bids_root, [src], report)

    tsv = bids_root / "phenotype" / "edinburgh.tsv"
    js = bids_root / "phenotype" / "edinburgh.json"
    assert tsv.exists() and js.exists()
    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    assert list(df["participant_id"]) == ["sub-001", "sub-002"]
    data_dict = json.loads(js.read_text())
    # participant_id is the index, not described; measures are.
    assert "participant_id" not in data_dict
    assert set(data_dict) == {"ehi_total", "ehi_decile"}
    assert tsv in report.files_written and js in report.files_written


def test_none_is_noop(tmp_path):
    bids_root = tmp_path / "ds"
    bids_root.mkdir()
    write_phenotype(bids_root, None, _report())
    assert not (bids_root / "phenotype").exists()


def test_skips_table_without_participant_id(tmp_path):
    bids_root = tmp_path / "ds"
    bids_root.mkdir()
    src = tmp_path / "bad.tsv"
    src.write_text("subject\tscore\nx\t1\n", encoding="utf-8")
    report = _report()
    write_phenotype(bids_root, [src], report)
    assert not (bids_root / "phenotype" / "bad.tsv").exists()
    assert any("participant_id" in w for w in report.warnings)


def test_missing_file_warns(tmp_path):
    bids_root = tmp_path / "ds"
    bids_root.mkdir()
    report = _report()
    write_phenotype(bids_root, [tmp_path / "nope.tsv"], report)
    assert any("not found" in w for w in report.warnings)


def test_multiple_instruments(tmp_path):
    bids_root = tmp_path / "ds"
    bids_root.mkdir()
    a = tmp_path / "bdi.tsv"
    a.write_text("participant_id\tbdi_total\nsub-001\t12\n", encoding="utf-8")
    b = tmp_path / "stai.csv"
    b.write_text("participant_id,stai_state\nsub-001,40\n", encoding="utf-8")
    write_phenotype(bids_root, [a, b], _report())
    assert (bids_root / "phenotype" / "bdi.tsv").exists()
    assert (bids_root / "phenotype" / "stai.tsv").exists()  # .csv read, .tsv written


def test_phenotype_auto_discovered_from_scaffold(tmp_path):
    """run_metadata picks up phenotype tables listed in the inventory scaffold."""
    from bidsmgr.metadata.engine import run_metadata
    from bidsmgr.recording_meta import RecordingMetaSpec, dump_spec, scaffold_sidecar_path

    bids_root = tmp_path / "ds"
    (bids_root / "sub-001").mkdir(parents=True)
    inv = tmp_path / "inv.tsv"
    inv.write_text("participant_id\nsub-001\n", encoding="utf-8")
    pheno = tmp_path / "edinburgh.tsv"
    pheno.write_text("participant_id\tehi\nsub-001\t80\n", encoding="utf-8")

    spec = RecordingMetaSpec(phenotype_files=[str(pheno)])
    scaffold_sidecar_path(inv).write_text(dump_spec(spec), encoding="utf-8")

    run_metadata(bids_root, inventory_tsv=inv, write_report=False)
    assert (bids_root / "phenotype" / "edinburgh.tsv").exists()
    assert (bids_root / "phenotype" / "edinburgh.json").exists()


def test_codebook_sidecar_merges_into_measure_json(tmp_path):
    """A sibling <measure>.json codebook supplies real Descriptions / Levels /
    Units (and MeasurementToolMetadata); auto-fill covers the rest."""
    bids_root = tmp_path / "ds"
    bids_root.mkdir()
    src = tmp_path / "bdi.tsv"
    src.write_text(
        "participant_id\tbdi_total\tseverity\nsub-001\t24\tmoderate\n",
        encoding="utf-8",
    )
    (tmp_path / "bdi.json").write_text(json.dumps({
        "MeasurementToolMetadata": {"Description": "Beck Depression Inventory II"},
        "bdi_total": {"Description": "BDI-II total score", "Units": "points"},
        "severity": {"Description": "band", "Levels": {"moderate": "20-28"}},
    }), encoding="utf-8")

    write_phenotype(bids_root, [src], _report())

    dd = json.loads((bids_root / "phenotype" / "bdi.json").read_text())
    assert dd["MeasurementToolMetadata"]["Description"] == "Beck Depression Inventory II"
    assert dd["bdi_total"]["Units"] == "points"
    assert dd["severity"]["Levels"]["moderate"] == "20-28"


def test_no_codebook_falls_back_to_column_name(tmp_path):
    bids_root = tmp_path / "ds"
    bids_root.mkdir()
    src = tmp_path / "stai.tsv"
    src.write_text("participant_id\tstai_state\nsub-001\t40\n", encoding="utf-8")
    write_phenotype(bids_root, [src], _report())
    dd = json.loads((bids_root / "phenotype" / "stai.json").read_text())
    assert dd["stai_state"] == {"Description": "stai_state"}  # auto-filled

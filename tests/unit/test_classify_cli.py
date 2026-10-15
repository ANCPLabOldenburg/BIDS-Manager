"""``bidsmgr-classify``: an inventory named again from its cached features."""

from __future__ import annotations

import io
import json

import pandas as pd

from bidsmgr.cli.classify import main, run_classify
from bidsmgr.classifier.acquisition import cache
from bidsmgr.classifier.acquisition.features import SeriesFeatures


def _inventory(tmp_path):
    rows = [
        {"participant_id": "sub-001", "session": "", "modality": "mri", "include": "1",
         "series_uid": "1.1", "sequence": "bold_rest", "acq_time": "100000", "datatype": "func",
         "bids_guess_datatype": "func", "bids_guess_suffix": "bold",
         "entities": json.dumps({"subject": "001", "task": "rest"}),
         "bids_name": "sub-001_task-rest_bold", "bids_path": "", "issues": ""},
        {"participant_id": "sub-001", "session": "", "modality": "mri", "include": "1",
         "series_uid": "1.2", "sequence": "bold_rest_rev", "acq_time": "100500",
         "datatype": "func", "bids_guess_datatype": "func", "bids_guess_suffix": "bold",
         "entities": json.dumps({"subject": "001", "task": "rest", "run": "2"}),
         "bids_name": "sub-001_task-rest_run-2_bold", "bids_path": "", "issues": ""},
    ]
    tsv = tmp_path / "inv.tsv"
    pd.DataFrame(rows).to_csv(tsv, sep="\t", index=False)
    base = dict(rows=64.0, columns=64.0, n_slices=30, n_orientations=1, acquisition_type="2D",
                is_epi=True, is_gre=True, echo_times=[30.0], repetition_time=0.8)
    cache.write(tsv, {"1.1": SeriesFeatures(description="bold_rest", n_volumes=300,
                                            pe_dir="PA", multiband=4, **base),
                      "1.2": SeriesFeatures(description="bold_rest_rev", n_volumes=4,
                                            pe_dir="AP", multiband=4, **base)})
    return tsv


def test_the_report_names_the_reference_and_says_why(tmp_path) -> None:
    tsv = _inventory(tmp_path)
    out = io.StringIO()
    n = run_classify(tsv, out=out)
    text = out.getvalue()
    assert n == 1
    assert "-> bold_rest_rev" in text and "fmap/epi" in text
    assert "opposite to a run" in text
    assert pd.read_csv(tsv, sep="\t", dtype=str).iloc[1]["datatype"] == "func", \
        "nothing written without --apply"


def test_apply_writes_the_new_name(tmp_path) -> None:
    tsv = _inventory(tmp_path)
    assert main([str(tsv), "--apply"]) == 0
    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    assert df.iloc[1]["bids_name"] == "sub-001_dir-AP_run-2_epi"
    assert df.iloc[0]["bids_name"] == "sub-001_task-rest_bold"
    assert float(df.iloc[1]["classification_confidence"]) > 0.8


def test_shadow_writes_only_the_report(tmp_path) -> None:
    tsv = _inventory(tmp_path)
    main([str(tsv), "--apply", "--mode", "shadow"])
    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    assert df.iloc[1]["datatype"] == "func" and df.iloc[1]["classification_evidence"]


def test_print_config(capsys) -> None:
    assert main(["--print-config"]) == 0
    assert json.loads(capsys.readouterr().out)["general"]["mode"] == "applied"


def test_without_features_it_says_to_rescan(tmp_path) -> None:
    tsv = tmp_path / "inv.tsv"
    pd.DataFrame([{"series_uid": "1"}]).to_csv(tsv, sep="\t", index=False)
    out = io.StringIO()
    assert run_classify(tsv, out=out) == 0 and "scan the raw data again" in out.getvalue()


def test_print_rulebook_and_broken_rules_refused(tmp_path, capsys) -> None:
    assert main(["--print-rulebook"]) == 0
    book = json.loads(capsys.readouterr().out)
    assert "anat/T1w" in book["candidates"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rulebook": {"candidates": {"anat/T1w": {"evidence": [
        {"source": "parameters", "feature": "nope", "weight": 1}]}}}}))
    import pytest

    with pytest.raises(SystemExit):
        main(["--print-rulebook", "--config", str(bad)])
    assert "no feature named 'nope'" in capsys.readouterr().err

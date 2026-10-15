"""Every vendor DICOM test set scans, and the conversion's files are the
ones the BIDS preview predicts.

Gated on ``BIDS_MANAGER_REAL_MRI_DATA``; the data is
``$BIDSMGR_TEST_DATA/MRI/vendor_test_data`` (dcm2niix validation sets and
clinical studies from the NCI Imaging Data Commons, organised by manufacturer;
see its README). The scan is checked on all 52 sets; the conversion, being
slower, on a handful that cover the companions (gradient tables, fieldmap
parts, physio, spectroscopy, old clinical scanners).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures.data_root import dataset

ROOT = dataset("MRI") / "vendor_test_data"

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not os.environ.get("BIDS_MANAGER_REAL_MRI_DATA"),
                       reason="BIDS_MANAGER_REAL_MRI_DATA not set"),
    pytest.mark.skipif(not ROOT.exists(), reason=f"vendor test data missing at {ROOT}"),
]


def _sets() -> list[Path]:
    return sorted(p.parent for p in ROOT.glob("*/*/*/DICOM")) if ROOT.exists() else []


@pytest.mark.parametrize("folder", _sets(), ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_every_set_scans(folder: Path, tmp_path: Path) -> None:
    from bidsmgr.cli.scan import run_scan

    df = run_scan(folder / "DICOM", tmp_path / "inv.tsv", n_jobs=2)
    assert len(df), "every set holds at least one series"
    included = df[df["include"].astype(str) == "1"]
    if folder.parent.parent.name == "NOT_MRI":
        assert included.empty, "BIDS has no CT: nothing is set to convert"
    else:
        assert len(included), "something to convert"


_CONVERTED = ("dcm_qa_nih", "dcm_qa_me", "dcm_qa_uih", "dcm_qa_dti", "dcm_qa_spec", "dcm_qa_ge",
              "ICDC-Glioma_Vista_1.0T", "ICDC-Glioma_MAGNETOM_EXPERT")


@pytest.mark.parametrize("name", _CONVERTED)
def test_the_preview_names_what_the_conversion_writes(name: str, tmp_path: Path) -> None:
    from bidsmgr.classifier.acquisition import cache
    from bidsmgr.cli.convert import run_convert
    from bidsmgr.cli.scan import run_scan
    from bidsmgr.converter.output_plan import PlanOptions, plan_output

    folder = next((p for p in _sets() if p.name == name), None)
    if folder is None:
        pytest.skip(f"{name} not in the vendor test data")
    inv = tmp_path / "inv.tsv"
    run_scan(folder / "DICOM", inv, n_jobs=2, dataset=name)
    run_convert(inv, tmp_path / "bids", n_jobs=2)
    df = pd.read_csv(inv, sep="\t", dtype=str, keep_default_na=False)
    plan = plan_output(df, options=PlanOptions(dataset=name, run_metadata=False),
                       features=cache.read(inv))
    root = tmp_path / "bids" / name
    written = {p.relative_to(root).as_posix() for p in root.rglob("*")
               if p.is_file() and ".bidsmgr" not in p.parts}
    sure = {f.path for f in plan.files if not f.conditional}
    maybe = {f.path for f in plan.files if f.conditional}
    assert sure - written == set(), "predicted and not written"
    assert written - sure - maybe == set(), "written and not predicted"

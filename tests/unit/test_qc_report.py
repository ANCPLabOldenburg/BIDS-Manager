"""Quality results on disk (the BIDS derivative), the group tables, finding
the images of a dataset, and ``bidsmgr-qc``."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from bidsmgr.qc import report, run
from bidsmgr.qc.types import Finding, Metric, QCResult

from tests.unit.qc_phantoms import (anat_phantom, dataset, dwi_phantom, gradient_table, save,
                                    save_dwi)


def _result(path, value, suffix="T1w") -> QCResult:
    return QCResult(path=str(path), kind="anat", suffix=suffix,
                    metrics=[Metric("snr_wm", "SNR in white matter", value, fmt="{:.2f}"),
                             Metric("efc", "EFC", None, why_missing="defaced")],
                    findings=[Finding("face", "A face is visible", "warning", "m", "face")],
                    facts={"zooms": [1.0, 1.0, 1.0], "defaced": False},
                    tracks={"fd": {"title": "FD", "ys": [np.array([0.0, np.nan, 0.2])]}})


def test_json_round_trip(tmp_path) -> None:
    root = dataset(tmp_path / "ds")
    image = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    res = _result(image, 12.5)
    data = report.to_json(res, root=root)
    assert data["path"] == "sub-01/anat/sub-01_T1w.nii.gz"
    assert data["metrics"] == {"snr_wm": 12.5, "efc": None}
    assert data["metric_info"]["efc"]["why_missing"] == "defaced"
    assert data["tracks"]["fd"]["ys"][0] == [0.0, None, 0.2]
    back = report.from_json(json.loads(json.dumps(data)))
    assert back.value("snr_wm") == 12.5 and back.metric("efc").why_missing == "defaced"
    assert np.isnan(back.tracks["fd"]["ys"][0][1])


def test_save_and_load_in_the_derivative(tmp_path) -> None:
    root = dataset(tmp_path / "ds")
    image = root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    written = report.save(_result(image, 9.0), root)
    assert written == root / "derivatives" / "bidsmgr-qc" / "sub-01" / "anat" / "sub-01_T1w.json"
    desc = json.loads((root / "derivatives" / "bidsmgr-qc" / "dataset_description.json")
                      .read_text())
    assert desc["DatasetType"] == "derivative"
    assert desc["GeneratedBy"][0]["Name"] == "BIDS Manager"
    assert report.load(root, image).value("snr_wm") == 9.0
    assert report.load(root, root / "sub-02" / "anat" / "sub-02_T1w.nii.gz") is None


def test_group_z_is_within_suffix_and_acq() -> None:
    rows = [{"path": f"sub-0{i}/anat/sub-0{i}_T1w.nii.gz", "suffix": "T1w",
             "metrics": {"snr_wm": v}} for i, v in enumerate([10, 11, 10.5, 30], 1)]
    rows.append({"path": "sub-01/anat/sub-01_acq-fast_T1w.nii.gz", "suffix": "T1w",
                 "metrics": {"snr_wm": 30}})
    z = report.group_z(rows)
    assert z["sub-04/anat/sub-04_T1w.nii.gz"]["snr_wm"] > report.GROUP_Z
    assert abs(z["sub-01/anat/sub-01_T1w.nii.gz"]["snr_wm"]) < 2
    # Alone in its acq: no group, no z.
    assert "sub-01/anat/sub-01_acq-fast_T1w.nii.gz" not in z


def test_protocol_differences_are_found() -> None:
    rows = [{"path": f"p{i}", "suffix": "T1w", "facts": {"zooms": [1, 1, 1]}, "metrics": {}}
            for i in range(4)]
    rows[3]["facts"]["zooms"] = [1, 1, 2]
    sidecars = {f"p{i}": {"RepetitionTime": 2.3} for i in range(4)}
    sidecars["p2"]["RepetitionTime"] = 1.9
    found = report.protocol_findings(rows, sidecars)
    assert set(found) == {"p2", "p3"}
    assert "RepetitionTime 1.9" in found["p2"][0]["message"]


def test_find_images_skips_derivatives_and_other_suffixes(tmp_path) -> None:
    root = dataset(tmp_path / "ds")
    for rel in ("sub-01/anat/sub-01_T1w.nii.gz", "sub-01/anat/sub-01_T1map.nii.gz",
                "sub-01/dwi/sub-01_dwi.nii.gz", "sub-01/func/sub-01_task-x_bold.nii.gz",
                "derivatives/x/sub-01/anat/sub-01_T1w.nii.gz",
                "sub-02/anat/sub-02_FLAIR.nii.gz"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"")
    found = [p.relative_to(root).as_posix() for p in run.find_images(root)]
    assert found == ["sub-01/anat/sub-01_T1w.nii.gz", "sub-01/dwi/sub-01_dwi.nii.gz",
                     "sub-02/anat/sub-02_FLAIR.nii.gz"]
    assert [p.name for p in run.find_images(root, participants=["02"])] == ["sub-02_FLAIR.nii.gz"]
    assert [p.name for p in run.find_images(root, datatypes=["dwi"])] == ["sub-01_dwi.nii.gz"]


@pytest.mark.parametrize("jobs", [1, 2])
def test_the_command_checks_a_dataset(tmp_path, capsys, jobs) -> None:
    from bidsmgr.cli import qc as cli

    root = dataset(tmp_path / "ds")
    data, affine = anat_phantom()
    save(data, affine, root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    save(data, affine, root / "sub-02" / "anat" / "sub-02_T1w.nii.gz",
         {"DeidentificationMethod": ["pydeface"]})
    bvals, bvecs = gradient_table(20, 3)
    dw, aff = dwi_phantom(bvals, bvecs)
    save_dwi(dw, aff, bvals, bvecs, root / "sub-01" / "dwi" / "sub-01_dwi.nii.gz")
    code = cli.main([str(root), "-j", str(jobs), "--no-flip-check", "--fast"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Checked 3 image(s)" in out
    base = root / "derivatives" / "bidsmgr-qc"
    assert (base / "sub-01" / "anat" / "sub-01_T1w.json").is_file()
    assert (base / "sub-01" / "dwi" / "sub-01_dwi.json").is_file()
    with (base / "group_T1w.tsv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert [r["path"] for r in rows] == ["sub-01/anat/sub-01_T1w.nii.gz",
                                        "sub-02/anat/sub-02_T1w.nii.gz"]
    # The defaced image has no air measures; the other has.
    assert rows[1]["efc"] == "n/a" and rows[0]["efc"] != "n/a"
    assert (base / "group_dwi.tsv").is_file()


def test_the_command_refuses_what_is_not_a_dataset(tmp_path, capsys) -> None:
    from bidsmgr.cli import qc as cli

    assert cli.main([str(tmp_path)]) == 2
    assert "not a BIDS dataset" in capsys.readouterr().err

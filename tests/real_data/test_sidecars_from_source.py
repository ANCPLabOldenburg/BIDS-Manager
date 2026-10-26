"""Sidecars written again from what is left, on a real project.

The multimodal sample (MRI, fieldmaps, diffusion, physio, spectroscopy, PET,
EEG, MEG, eye tracking) is made a project, scanned and converted the way the
GUI does it. Then:

* every sidecar is deleted and written again from the source data: each field
  the conversion wrote comes back with the same value. The links between
  sidecars (``IntendedFor``, ``B0Field*``) cannot, because every sidecar that
  held them is gone; and ``TaskName`` is the one field added (the metadata
  step's, from the file name);
* one fieldmap's sidecar and one run's are deleted: both come back exactly,
  links included, from the sidecars still there;
* from the headers alone, with no source data, no field is written that
  disagrees with what the conversion wrote, beyond the echo time's two
  significant figures.

Gated on ``BIDS_MANAGER_REAL_MRI_DATA``; the published multimodal sample is
downloaded once and cached (``tests/fixtures/sample_data``), as for the BIDS
preview's real-data test.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from tests.fixtures import sample_data

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not os.environ.get("BIDS_MANAGER_REAL_MRI_DATA"),
                       reason="BIDS_MANAGER_REAL_MRI_DATA not set"),
]


def _sample() -> Path:
    """The published multimodal sample, downloaded once and cached
    (``tests/fixtures/sample_data``): the folder holding ``raw/`` and its
    spreadsheets. Skips when it cannot be had."""
    return sample_data.require("multimodal").parent

#: Fields that link one sidecar to another: gone when every sidecar is.
LINKS = {"IntendedFor", "B0FieldIdentifier", "B0FieldSource"}


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    from bidsmgr.cli import convert, create, scan

    sample = _sample()
    ds = tmp_path_factory.mktemp("sidecars") / "study"
    assert create.main([str(ds)]) == 0
    assert scan.main([str(sample / "raw"), "--project", str(ds), "-j", "2"]) == 0
    assert convert.main(["--project", str(ds)]) == 0
    assert (ds / ".bidsmgr" / "provenance").is_dir()
    return ds


def _copy(project: Path, tmp_path: Path) -> Path:
    root = tmp_path / "study"
    shutil.copytree(project, root, symlinks=True)
    return root


def _sidecars(root: Path) -> dict[str, dict]:
    from bidsmgr.editor.sidecars import recordings, sidecar_path

    out = {}
    for rec in recordings(root):
        side = sidecar_path(rec)
        if side.exists():
            out[side.relative_to(root).as_posix()] = json.loads(side.read_text())
    return out


def test_every_sidecar_comes_back_from_the_source_data(project, tmp_path) -> None:
    from bidsmgr.editor.sidecars import apply_sidecars, plan_sidecars

    root = _copy(project, tmp_path)
    before = _sidecars(root)
    assert len(before) >= 20, sorted(before)
    for rel in before:
        (root / rel).unlink()

    plan = plan_sidecars(root, use_source=True, fill_scope="none")
    assert plan.used_source and not plan.skipped
    apply_sidecars(root, plan.writes())
    after = _sidecars(root)
    assert sorted(after) == sorted(before)

    wrong = []
    for rel, was in before.items():
        now = after[rel]
        for key, value in was.items():
            if key in LINKS:
                continue
            if now.get(key, "<absent>") != value:
                wrong.append(f"{rel}: {key} was {value!r}, now {now.get(key, '<absent>')!r}")
        extra = set(now) - set(was) - {"TaskName"}
        if extra:
            wrong.append(f"{rel}: added {sorted(extra)}")
    assert not wrong, "\n".join(wrong[:30])


def test_a_lost_fieldmap_and_run_come_back_exactly(project, tmp_path) -> None:
    from bidsmgr.editor.sidecars import apply_sidecars, plan_sidecars

    root = _copy(project, tmp_path)
    before = _sidecars(root)
    lost = [rel for rel in before
            if rel.endswith(("_phasediff.json", "_task-rest_bold.json"))]
    assert len(lost) == 2, sorted(before)
    for rel in lost:
        (root / rel).unlink()
    plan = plan_sidecars(root, use_source=True, fill_scope="none")
    assert sorted(p.sidecar for p in plan.writes()) == sorted(lost)
    apply_sidecars(root, plan.writes())
    after = _sidecars(root)
    for rel in lost:
        was, now = before[rel], after[rel]
        assert {k: v for k, v in now.items() if k != "TaskName"} == was, rel


def test_headers_alone_never_contradict_the_conversion(project, tmp_path) -> None:
    from bidsmgr.editor.sidecars import plan_sidecars

    root = _copy(project, tmp_path)
    before = _sidecars(root)
    for rel in before:
        (root / rel).unlink()
    plan = plan_sidecars(root, use_source=False, fill_scope="none")
    assert not plan.used_source
    wrong = []
    for one in plan.files:
        was = before[one.sidecar]
        for field in one.added:
            if field.name not in was or field.origin == "filename":
                continue
            ok = was[field.name] == field.value
            if not ok and field.name.startswith("EchoTime"):
                ok = float(f"{was[field.name] * 1000:.2g}") / 1000 == field.value
            if not ok and isinstance(field.value, (list, float)):
                ok = json.dumps(was[field.name]) == json.dumps(field.value) or \
                    abs(float(json.dumps(field.value).strip("[]").split(",")[0])
                        - float(json.dumps(was[field.name]).strip("[]").split(",")[0])) < 1e-4
            if not ok:
                wrong.append(f"{one.sidecar}: {field.name} {was[field.name]!r} vs "
                             f"{field.value!r} ({field.origin})")
    assert not wrong, "\n".join(wrong[:30])

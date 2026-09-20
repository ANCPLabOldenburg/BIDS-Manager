"""Curated metadata must survive a second conversion of the same subject.

The scenario, which happens to real users: convert a subject, spend an
afternoon in the Editor filling in the fields no scanner records, then convert
that subject again because a series was missing. Before this, the second pass
decided at file level: replace and the afternoon was gone, skip and the fix the
second pass brought never arrived.

The rule these tests pin down is one sentence: a value a person stated wins, a
value only the converter stated loses. Everything else follows from it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bidsmgr.cli.convert import _merge_commit
from bidsmgr.metadata import preserve


# ---------------------------------------------------------------------------
# The merge rule, on its own
# ---------------------------------------------------------------------------


def test_a_curated_value_beats_a_freshly_converted_one() -> None:
    merged, kept = preserve.merge_sidecar(
        {"InstitutionName": "Oldenburg", "EchoTime": 0.03},
        {"InstitutionName": "", "EchoTime": 0.031},
    )
    assert merged["InstitutionName"] == "Oldenburg"
    assert merged["EchoTime"] == 0.03
    # Only the genuine disagreement is reported. Keeping a stated value over a
    # blank one is not one, and logging it would bury the cases that are.
    assert kept == ["EchoTime"]


def test_a_field_only_the_fresh_pass_knows_is_added() -> None:
    """New knowledge nobody has contradicted."""
    merged, kept = preserve.merge_sidecar(
        {"EchoTime": 0.03}, {"EchoTime": 0.03, "FlipAngle": 9},
    )
    assert merged["FlipAngle"] == 9
    assert kept == []


def test_a_field_only_the_existing_file_has_is_kept() -> None:
    """The converter never had an opinion, so it cannot be overruling one."""
    merged, _ = preserve.merge_sidecar(
        {"InstitutionalDepartmentName": "Neuropsychology"}, {"EchoTime": 0.03},
    )
    assert merged["InstitutionalDepartmentName"] == "Neuropsychology"


def test_a_todo_is_not_an_opinion() -> None:
    """TODO is what BIDS Manager writes when nothing could answer a field."""
    merged, kept = preserve.merge_sidecar(
        {"Manufacturer": "TODO", "ManufacturersModelName": "  todo  "},
        {"Manufacturer": "Siemens", "ManufacturersModelName": "Prisma"},
    )
    assert merged["Manufacturer"] == "Siemens"
    assert merged["ManufacturersModelName"] == "Prisma"
    assert kept == []


def test_zero_and_false_are_answers() -> None:
    """A placeholder check that swallowed falsy values would discard real
    metadata: 0 Hz notch and False are things a person can mean."""
    merged, kept = preserve.merge_sidecar(
        {"NotchFilter": 0, "DwellTime": False},
        {"NotchFilter": 50, "DwellTime": True},
    )
    assert merged["NotchFilter"] == 0
    assert merged["DwellTime"] is False
    assert sorted(kept) == ["DwellTime", "NotchFilter"]


def test_field_order_follows_the_curated_file() -> None:
    merged, _ = preserve.merge_sidecar(
        {"B": 2, "A": 1}, {"A": 9, "C": 3},
    )
    assert list(merged) == ["B", "A", "C"]


# ---------------------------------------------------------------------------
# Through the file layer
# ---------------------------------------------------------------------------


def test_a_merge_that_cannot_be_done_correctly_is_not_done(
    tmp_path: Path,
) -> None:
    """Half-writing a file is worse than falling back to a plain replace."""
    existing = tmp_path / "a.json"
    fresh = tmp_path / "b.json"
    existing.write_text("{not json")
    fresh.write_text(json.dumps({"EchoTime": 0.03}))
    assert preserve.merge_sidecar_files(existing, fresh) is None
    assert existing.read_text() == "{not json", "must be left untouched"


def test_only_sidecars_and_scans_tables_are_merged(tmp_path: Path) -> None:
    """Merging an arbitrary table without knowing which column identifies a
    row would silently interleave two datasets."""
    assert preserve.is_mergeable(tmp_path / "sub-01_T1w.json")
    assert preserve.is_mergeable(tmp_path / "sub-01_scans.tsv")
    assert not preserve.is_mergeable(tmp_path / "sub-01_events.tsv")
    assert not preserve.is_mergeable(tmp_path / "sub-01_T1w.nii.gz")


def _write_tsv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        return list(r.fieldnames), list(r)


def test_a_scans_table_gains_rows_without_losing_curated_cells(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "sub-01_scans.tsv"
    fresh = tmp_path / "fresh.tsv"
    _write_tsv(existing, ["filename", "acq_time", "comments"], [
        {"filename": "anat/sub-01_T1w.nii.gz", "acq_time": "2026-01-01T09:00:00",
         "comments": "subject moved"},
    ])
    _write_tsv(fresh, ["filename", "acq_time"], [
        {"filename": "anat/sub-01_T1w.nii.gz", "acq_time": "n/a"},
        {"filename": "func/sub-01_task-rest_bold.nii.gz",
         "acq_time": "2026-01-01T09:20:00"},
    ])
    assert preserve.merge_scans_files(existing, fresh) == 2

    fields, rows = _read_tsv(existing)
    assert fields == ["filename", "acq_time", "comments"]
    assert rows[0]["acq_time"] == "2026-01-01T09:00:00"   # n/a never wins
    assert rows[0]["comments"] == "subject moved"         # curated column kept
    assert rows[1]["filename"] == "func/sub-01_task-rest_bold.nii.gz"
    assert rows[1]["comments"] == "n/a"                   # new row, no comment


def test_a_scans_table_without_its_key_column_is_left_alone(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "sub-01_scans.tsv"
    fresh = tmp_path / "fresh.tsv"
    _write_tsv(existing, ["something_else"], [{"something_else": "x"}])
    _write_tsv(fresh, ["filename"], [{"filename": "y"}])
    assert preserve.merge_scans_files(existing, fresh) is None


# ---------------------------------------------------------------------------
# Through the commit, which is where a user meets it
# ---------------------------------------------------------------------------


def _stage(tmp_path: Path, files: dict[str, str]) -> Path:
    s = tmp_path / "staging" / "sub-001"
    for rel, text in files.items():
        p = s / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return s


def _existing(tmp_path: Path, files: dict[str, str]) -> Path:
    t = tmp_path / "out" / "sub-001"
    for rel, text in files.items():
        p = t / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return t


CURATED = {
    "anat/sub-001_T1w.json": json.dumps({
        "EchoTime": 0.03,
        "InstitutionName": "University of Oldenburg",
        "Manufacturer": "TODO",
    }),
}
FRESH = {
    "anat/sub-001_T1w.json": json.dumps({
        "EchoTime": 0.031,
        "Manufacturer": "Siemens",
        "FlipAngle": 9,
    }),
    "anat/sub-001_T1w.nii.gz": "NEW IMAGE",
}


@pytest.mark.parametrize("policy", ["update", "replace"])
def test_re_converting_keeps_the_afternoon_of_curation(
    tmp_path: Path, policy: str,
) -> None:
    staging = _stage(tmp_path, FRESH)
    target = _existing(tmp_path, {
        **CURATED, "anat/sub-001_T1w.nii.gz": "OLD IMAGE",
    })
    _merge_commit(staging, target, on_existing=policy)

    sidecar = json.loads((target / "anat" / "sub-001_T1w.json").read_text())
    assert sidecar["InstitutionName"] == "University of Oldenburg"  # curated
    assert sidecar["EchoTime"] == 0.03                             # curated
    assert sidecar["Manufacturer"] == "Siemens"   # was a TODO, not an opinion
    assert sidecar["FlipAngle"] == 9              # new knowledge

    # The image itself is data, not metadata: the fresh one wins outright.
    assert (target / "anat" / "sub-001_T1w.nii.gz").read_text() == "NEW IMAGE"


def test_turning_preservation_off_lets_the_fresh_pass_win(
    tmp_path: Path,
) -> None:
    staging = _stage(tmp_path, FRESH)
    target = _existing(tmp_path, CURATED)
    _merge_commit(
        staging, target, on_existing="replace", preserve_curation=False,
    )
    sidecar = json.loads((target / "anat" / "sub-001_T1w.json").read_text())
    assert "InstitutionName" not in sidecar
    assert sidecar["EchoTime"] == 0.031


def test_the_pre_merge_sidecar_is_always_recoverable(tmp_path: Path) -> None:
    """A merge is still a change to a file somebody curated, so the backup has
    to hold what was there before it, not after."""
    staging = _stage(tmp_path, FRESH)
    target = _existing(tmp_path, CURATED)
    _merge_commit(staging, target, on_existing="replace")

    backups = list((target.parent / ".bidsmgr" / "backup").glob("sub-001_*"))
    assert len(backups) == 1
    saved = json.loads(
        (backups[0] / "anat" / "sub-001_T1w.json").read_text()
    )
    assert saved["Manufacturer"] == "TODO"    # exactly the pre-merge state
    assert "FlipAngle" not in saved


def test_skip_still_means_skip(tmp_path: Path) -> None:
    """Preservation must not turn the safe-by-default policy into a writer."""
    staging = _stage(tmp_path, FRESH)
    target = _existing(tmp_path, CURATED)
    before = (target / "anat" / "sub-001_T1w.json").read_text()
    _merge_commit(staging, target, on_existing="skip")
    assert (target / "anat" / "sub-001_T1w.json").read_text() == before


def test_a_brand_new_subject_is_unaffected(tmp_path: Path) -> None:
    staging = _stage(tmp_path, FRESH)
    target = tmp_path / "out" / "sub-001"
    added, replaced, kept = _merge_commit(staging, target, on_existing="update")
    assert (added, replaced, kept) == (2, 0, 0)


def test_the_merge_is_reported_not_silent(tmp_path: Path, caplog) -> None:
    """A tool that keeps your value over the converter's must say so."""
    import logging

    staging = _stage(tmp_path, FRESH)
    target = _existing(tmp_path, CURATED)
    with caplog.at_level(logging.INFO, logger="bidsmgr.cli.convert"):
        _merge_commit(staging, target, on_existing="replace")
    assert any("curated field" in r.getMessage() for r in caplog.records)

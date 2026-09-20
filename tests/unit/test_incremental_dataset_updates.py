"""Converting into a dataset that already exists, again and again.

Nobody converts a study once. Subjects arrive over months, a modality is added
halfway through, a session is redone. Each pass has to leave the three files
that describe the dataset as a whole correct for everything in it, not just for
what this pass brought:

* ``participants.tsv`` (and its ``.json``) has to gain the new subjects and
  keep every value and column already there, including ones the user added by
  hand;
* the per-subject ``*_scans.tsv`` has to gain the new files;
* ``dataset_description.json`` has to keep the Name, licence and authorship
  somebody chose.

This was believed to work and had never been checked against a genuinely
incremental run. These tests are that check, written before anything was
changed so a regression here means a real one.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.cli.convert import _merge_commit
from bidsmgr.metadata.engine import run_metadata


def _subject(root: Path, sub: str, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / sub / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A dataset after one conversion pass, with one subject in it."""
    root = tmp_path / "ds"
    root.mkdir()
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "Sleep and memory", "BIDSVersion": "1.10.0",
        "Authors": ["A Person"], "License": "CC0",
    }))
    _subject(root, "sub-01", {
        "anat/sub-01_T1w.nii.gz": "img",
        "anat/sub-01_T1w.json": json.dumps({"EchoTime": 0.03}),
    })
    run_metadata(root, write_report=False)
    return root


def _participants(root: Path) -> pd.DataFrame:
    # ``keep_default_na=False`` because pandas reads BIDS's own "n/a" as NaN,
    # which would make an intact table look like a table full of holes.
    return pd.read_csv(
        root / "participants.tsv", sep="\t", dtype=str, keep_default_na=False,
    )


# ---------------------------------------------------------------------------
# participants.tsv
# ---------------------------------------------------------------------------


def test_a_second_pass_adds_the_new_subject(dataset: Path) -> None:
    assert list(_participants(dataset)["participant_id"]) == ["sub-01"]

    _subject(dataset, "sub-02", {"anat/sub-02_T1w.nii.gz": "img"})
    run_metadata(dataset, write_report=False)

    assert sorted(_participants(dataset)["participant_id"]) == [
        "sub-01", "sub-02",
    ]


def test_a_second_pass_keeps_a_hand_edited_cell(dataset: Path) -> None:
    """The commonest way an incremental tool destroys work: rewriting a table
    it did not fully author."""
    table = _participants(dataset)
    table["age"] = ["31"]
    table.to_csv(dataset / "participants.tsv", sep="\t", index=False)

    _subject(dataset, "sub-02", {"anat/sub-02_T1w.nii.gz": "img"})
    run_metadata(dataset, write_report=False)

    after = _participants(dataset).set_index("participant_id")
    assert after.at["sub-01", "age"] == "31"


def test_a_second_pass_keeps_a_column_the_user_invented(dataset: Path) -> None:
    table = _participants(dataset)
    table["group"] = ["control"]
    table.to_csv(dataset / "participants.tsv", sep="\t", index=False)

    _subject(dataset, "sub-02", {"anat/sub-02_T1w.nii.gz": "img"})
    run_metadata(dataset, write_report=False)

    after = _participants(dataset).set_index("participant_id")
    assert "group" in after.columns
    assert after.at["sub-01", "group"] == "control"
    assert after.at["sub-02", "group"] == "n/a"


def test_the_participants_sidecar_is_written_beside_the_table(
    dataset: Path,
) -> None:
    _subject(dataset, "sub-02", {"anat/sub-02_T1w.nii.gz": "img"})
    run_metadata(dataset, write_report=False)
    described = json.loads((dataset / "participants.json").read_text())
    for column in _participants(dataset).columns:
        if column == "participant_id":
            continue
        assert column in described, f"{column} is undescribed"


# ---------------------------------------------------------------------------
# dataset_description.json
# ---------------------------------------------------------------------------


def test_a_second_pass_does_not_rename_the_dataset(dataset: Path) -> None:
    """A metadata run with no --name must not overwrite a chosen Name."""
    _subject(dataset, "sub-02", {"anat/sub-02_T1w.nii.gz": "img"})
    run_metadata(dataset, write_report=False)
    described = json.loads((dataset / "dataset_description.json").read_text())
    assert described["Name"] == "Sleep and memory"
    assert described["Authors"] == ["A Person"]
    assert described["License"] == "CC0"


def test_the_generator_is_recorded_once_per_version(dataset: Path) -> None:
    """GeneratedBy is a list, so a tool that appends blindly grows it by one
    entry per conversion and the provenance becomes noise."""
    for _ in range(3):
        run_metadata(dataset, write_report=False)
    described = json.loads((dataset / "dataset_description.json").read_text())
    names = [
        (g.get("Name"), g.get("Version"))
        for g in described.get("GeneratedBy", [])
    ]
    assert len(names) == len(set(names)), names


# ---------------------------------------------------------------------------
# *_scans.tsv, which the conversion writes rather than the metadata step
# ---------------------------------------------------------------------------


def _scans(root: Path, sub: str):
    path = root / sub / f"{sub}_scans.tsv"
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_a_new_session_lands_beside_the_existing_one(tmp_path: Path) -> None:
    """The incremental case the merge commit exists for."""
    target = tmp_path / "out" / "sub-01"
    (target / "ses-pre" / "anat").mkdir(parents=True)
    (target / "ses-pre" / "anat" / "sub-01_ses-pre_T1w.nii.gz").write_text("a")

    staging = tmp_path / "staging" / "sub-01"
    (staging / "ses-post" / "anat").mkdir(parents=True)
    (staging / "ses-post" / "anat" / "sub-01_ses-post_T1w.nii.gz").write_text("b")

    added, replaced, kept = _merge_commit(staging, target, on_existing="skip")
    assert (added, replaced, kept) == (1, 0, 0)
    assert (target / "ses-pre" / "anat" / "sub-01_ses-pre_T1w.nii.gz").exists()
    assert (target / "ses-post" / "anat" / "sub-01_ses-post_T1w.nii.gz").exists()


def test_a_scans_table_gains_the_new_pass_rows(tmp_path: Path) -> None:
    target = tmp_path / "out" / "sub-01"
    target.mkdir(parents=True)
    (target / "sub-01_scans.tsv").write_text(
        "filename\tacq_time\n"
        "anat/sub-01_T1w.nii.gz\t2026-01-01T09:00:00\n"
    )
    staging = tmp_path / "staging" / "sub-01"
    staging.mkdir(parents=True)
    (staging / "sub-01_scans.tsv").write_text(
        "filename\tacq_time\n"
        "func/sub-01_task-rest_bold.nii.gz\t2026-02-01T10:00:00\n"
    )

    _merge_commit(staging, target, on_existing="update")

    names = [r["filename"] for r in _scans(tmp_path / "out", "sub-01")]
    assert names == [
        "anat/sub-01_T1w.nii.gz",
        "func/sub-01_task-rest_bold.nii.gz",
    ], "the second pass must not drop the first pass's rows"


def test_a_second_modality_does_not_disturb_the_first(dataset: Path) -> None:
    """Adding EEG to an MRI dataset months later."""
    _subject(dataset, "sub-01", {
        "eeg/sub-01_task-rest_eeg.edf": "eeg",
        "eeg/sub-01_task-rest_eeg.json": json.dumps({
            "PowerLineFrequency": 50, "EEGReference": "Cz",
        }),
    })
    run_metadata(dataset, write_report=False)

    assert (dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz").exists()
    sidecar = json.loads(
        (dataset / "sub-01" / "anat" / "sub-01_T1w.json").read_text()
    )
    assert sidecar["EchoTime"] == 0.03
    eeg = json.loads(
        (dataset / "sub-01" / "eeg" / "sub-01_task-rest_eeg.json").read_text()
    )
    assert eeg["EEGReference"] == "Cz"

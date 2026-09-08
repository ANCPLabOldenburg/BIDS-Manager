"""One project's answers must not reach another's files.

Projects are normally kept side by side, so the parent of one is the home of
all of them. The step that runs after a conversion is given that parent, because
that is where convert puts each dataset, and the metadata engine walks every
BIDS root it finds there. So a dataset name typed into one project's template
was written into every neighbouring project.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bidsmgr.cli._scaffold import ensure_dataset_description
from bidsmgr.cli.metadata import run_metadata_cli
from bidsmgr.recording_meta import (
    RecordingMetaSpec,
    dataset_description_from_bids,
    dump_spec,
    scaffold_sidecar_path,
)


def _two_projects(tmp_path: Path) -> Path:
    """Two datasets side by side, as anyone would keep them."""
    for name in ("StudyA", "StudyB"):
        root = tmp_path / name
        (root / "sub-001" / "anat").mkdir(parents=True)
        (root / "sub-001" / "anat" / "sub-001_T1w.json").write_text("{}")
        ensure_dataset_description(root, name=name)
    return tmp_path


def _state_a_name(tmp_path: Path, project: str, name: str) -> Path:
    tsv = tmp_path / project / "inv.tsv"
    pd.DataFrame([{
        "include": "1", "proposed_basename": "sub-001_T1w", "series_uid": "1.2.3",
    }]).to_csv(tsv, sep="\t", index=False)
    spec = RecordingMetaSpec()
    dataset_description_from_bids(spec.dataset_description, {"Name": name})
    scaffold_sidecar_path(tsv).write_text(dump_spec(spec))
    return tsv


def _name_of(tmp_path: Path, project: str) -> str:
    path = tmp_path / project / "dataset_description.json"
    return json.loads(path.read_text())["Name"]


def test_a_name_stated_in_one_project_stays_there(tmp_path: Path) -> None:
    parent = _two_projects(tmp_path)
    tsv = _state_a_name(parent, "StudyA", "Study A, the real one")

    run_metadata_cli(parent, inventory_tsv=tsv, datasets=["StudyA"], write_report=False)

    assert _name_of(parent, "StudyA") == "Study A, the real one"
    assert _name_of(parent, "StudyB") == "StudyB", (
        "a neighbouring project must not be renamed by this one"
    )


def test_without_the_filter_it_reaches_them_all(tmp_path: Path) -> None:
    """The behaviour this guards against, pinned so the reason is legible.

    A deliberate ``bidsmgr-metadata <folder>`` over a folder of datasets is
    still allowed to touch all of them: that is what the command means when a
    caller does not say otherwise. What must never happen is the GUI's
    post-convert step doing it by accident.
    """
    parent = _two_projects(tmp_path)
    tsv = _state_a_name(parent, "StudyA", "Study A, the real one")

    run_metadata_cli(parent, inventory_tsv=tsv, write_report=False)
    assert _name_of(parent, "StudyB") == "Study A, the real one"


def test_naming_a_dataset_that_is_the_target_itself_works(tmp_path: Path) -> None:
    """The GUI names the project folder while targeting its parent, but a
    caller may target the dataset directly."""
    parent = _two_projects(tmp_path)
    tsv = _state_a_name(parent, "StudyA", "Named directly")

    run_metadata_cli(
        parent / "StudyA", inventory_tsv=tsv, datasets=["StudyA"], write_report=False,
    )
    assert _name_of(parent, "StudyA") == "Named directly"
    assert _name_of(parent, "StudyB") == "StudyB"

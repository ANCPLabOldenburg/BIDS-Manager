"""Repairs that act on a whole dataset, at conversion time or afterwards.

The rule these are all tested against: a repair must never make the dataset
worse, and must never make an unfinished file look finished.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.fixups import associations as assoc
from bidsmgr.fixups.citation import (
    MOVED_FIELDS,
    build_citation,
    citation_path,
    write_citation,
)
from bidsmgr.fixups.dataset_fixups import run_dataset_fixups
from bidsmgr.project.adopt import (
    NotABidsDataset,
    adopt,
    changed_since_adoption,
    is_managed,
    looks_like_bids,
)
from bidsmgr.project.operations import read_log, undo_last


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "sub-01" / "func").mkdir(parents=True)
    (root / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz").write_bytes(b"")
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "Study", "BIDSVersion": "1.10.0",
        "Authors": ["Lopez Vilaret, Karel", "Rieger, Jochem"],
        "License": "CC0-1.0",
    }))
    return root


# ---------------------------------------------------------------------------
# Companion files
# ---------------------------------------------------------------------------


def test_a_task_recording_wants_events_and_a_sidecar(dataset: Path) -> None:
    kinds = {m.kind for m in assoc.find_missing(dataset)}
    assert kinds == {"events", "sidecar"}


def test_a_generated_sidecar_names_every_required_field(dataset: Path) -> None:
    """Derived fields are included: no conversion is going to run for a file
    that has no sidecar at all, so the stub must name them."""
    missing = [m for m in assoc.find_missing(dataset) if m.kind == "sidecar"]
    assoc.generate(dataset, missing)
    written = json.loads(missing[0].target.read_text())
    assert {"RepetitionTime", "TaskName"} <= set(written)
    assert set(written.values()) == {"TODO"}


def test_a_generated_events_table_is_deliberately_invalid(
    dataset: Path,
) -> None:
    """A valid but empty events table is indistinguishable from a recording
    that genuinely had no events, and would pass quietly forever."""
    missing = [m for m in assoc.find_missing(dataset) if m.kind == "events"]
    assoc.generate(dataset, missing)
    text = missing[0].target.read_text()
    assert text.startswith("onset\tduration")
    assert "TODO" in text


def test_generation_is_one_undo(dataset: Path) -> None:
    created, failed = assoc.generate(dataset, assoc.find_missing(dataset))
    assert created and not failed
    assert len(read_log(dataset)) == 1
    undo_last(dataset)
    assert not any(p.exists() for p in created)


# ---------------------------------------------------------------------------
# CITATION.cff
# ---------------------------------------------------------------------------


def _parsed(description: dict) -> dict:
    """Build a citation and read it back, so the assertions are about what
    the file MEANS rather than how the serialiser spells it."""
    import yaml

    return yaml.safe_load(build_citation(description))


def test_a_citation_splits_names_and_strips_a_doi_resolver() -> None:
    data = _parsed({
        "Name": "ds",
        "Authors": ["Lopez Vilaret, Karel", "Jochem Rieger"],
        "DatasetDOI": "https://doi.org/10.18112/openneuro.ds1.v1",
    })
    assert data["authors"][0] == {
        "family-names": "Lopez Vilaret", "given-names": "Karel",
    }
    # No comma: the last word is taken as the family name.
    assert data["authors"][1]["family-names"] == "Rieger"
    assert data["doi"] == "10.18112/openneuro.ds1.v1"


def test_a_dataset_with_no_authors_gets_a_todo_not_an_invention() -> None:
    data = _parsed({"Name": "ds"})
    assert data["authors"] == [{"family-names": "TODO", "given-names": "TODO"}]


def test_punctuation_in_the_title_survives_a_round_trip() -> None:
    """Written through PyYAML rather than by hand, so a title with a quote
    and a colon in it is not a corner case."""
    title = 'A "pilot": study, with commas'
    assert _parsed({"Name": title})["title"] == title


def test_writing_a_citation_moves_the_fields_it_owns(dataset: Path) -> None:
    """BIDS treats the citation file as the single source for these. Leaving
    them in both files is an error, not a duplicate."""
    write_citation(dataset)
    described = json.loads((dataset / "dataset_description.json").read_text())
    for field in MOVED_FIELDS:
        assert field not in described
    assert citation_path(dataset).exists()


def test_an_existing_citation_is_not_overwritten(dataset: Path) -> None:
    citation_path(dataset).write_text("hand written\n")
    assert write_citation(dataset) is None
    assert citation_path(dataset).read_text() == "hand written\n"


def test_writing_a_citation_is_one_undo(dataset: Path) -> None:
    write_citation(dataset)
    undo_last(dataset)
    assert not citation_path(dataset).exists()
    described = json.loads((dataset / "dataset_description.json").read_text())
    assert "Authors" in described


# ---------------------------------------------------------------------------
# The shared entry point
# ---------------------------------------------------------------------------


def test_nothing_runs_unless_asked(dataset: Path) -> None:
    report = run_dataset_fixups(dataset)
    assert not report.did_anything
    assert not citation_path(dataset).exists()
    assert read_log(dataset) == []


def test_both_repairs_run_and_report(dataset: Path) -> None:
    report = run_dataset_fixups(
        dataset, generate_companions=True, write_citation_file=True,
    )
    assert report.companions_written
    assert report.citation_written
    assert any("companion" in line for line in report.lines())
    assert any("CITATION.cff" in line for line in report.lines())


def test_one_repair_failing_does_not_stop_the_other(
    dataset: Path, monkeypatch,
) -> None:
    """A dataset that got its citation file and not its events tables is
    better off than one that got neither."""
    def _boom(*_a, **_k):
        raise RuntimeError("no")

    monkeypatch.setattr(
        "bidsmgr.fixups.associations.find_missing", _boom,
    )
    report = run_dataset_fixups(
        dataset, generate_companions=True, write_citation_file=True,
    )
    assert report.failed
    assert report.citation_written


# ---------------------------------------------------------------------------
# Adoption
# ---------------------------------------------------------------------------


def test_a_folder_that_is_not_bids_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "nope"
    empty.mkdir()
    assert not looks_like_bids(empty)
    with pytest.raises(NotABidsDataset):
        adopt(empty)


def test_a_dataset_with_only_subject_folders_is_accepted(
    tmp_path: Path,
) -> None:
    """A dataset missing its description is exactly what a user wants to open
    in order to fix, so refusing it would block the repair."""
    root = tmp_path / "ds"
    (root / "sub-01").mkdir(parents=True)
    assert looks_like_bids(root)


def test_adopting_records_a_baseline_and_changes_nothing_else(
    dataset: Path,
) -> None:
    before = {
        p.relative_to(dataset).as_posix(): p.read_bytes()
        for p in dataset.rglob("*") if p.is_file()
    }
    result = adopt(dataset)
    assert is_managed(dataset)
    assert result.files == len(before)
    after = {
        p.relative_to(dataset).as_posix(): p.read_bytes()
        for p in dataset.rglob("*")
        if p.is_file() and ".bidsmgr" not in p.parts
    }
    assert after == before, "adoption must not touch the dataset itself"


def test_status_reports_what_changed_since_adoption(dataset: Path) -> None:
    adopt(dataset)
    assert changed_since_adoption(dataset) == []
    (dataset / "dataset_description.json").write_text('{"Name": "other"}')
    (dataset / "NEW.txt").write_text("hello")
    changed = changed_since_adoption(dataset)
    assert "dataset_description.json" in changed
    assert "NEW.txt" in changed


# ---------------------------------------------------------------------------
# The CFF model the writer and the Editor form share
# ---------------------------------------------------------------------------


def test_a_person_round_trips_through_both_name_shapes() -> None:
    from bidsmgr.editor.cff import join_person, split_person

    for name in ("Lopez Vilaret, Karel", "Rieger, Jochem"):
        assert join_person(split_person(name)) == name


def test_a_single_word_name_is_kept_whole() -> None:
    from bidsmgr.editor.cff import split_person

    assert split_person("Cher") == {"family-names": "Cher"}


def test_the_required_keys_are_the_ones_bidsval_checks() -> None:
    from bidsmgr.editor.cff import REQUIRED, missing_required

    assert set(REQUIRED) == {"cff-version", "message", "title"}
    assert missing_required({"cff-version": "1.2.0"}) == ["message", "title"]


def test_an_unknown_key_survives_a_write(tmp_path: Path) -> None:
    """Somebody who hand-wrote a CFF key the model does not offer must not
    lose it by opening the file."""
    from bidsmgr.editor.cff import dumps, load

    data = {
        "cff-version": "1.2.0", "message": "cite", "title": "ds",
        "preferred-citation": {"type": "article"},
    }
    p = tmp_path / "CITATION.cff"
    p.write_text(dumps(data))
    assert load(p)["preferred-citation"] == {"type": "article"}


def test_fields_are_ordered_required_first() -> None:
    from bidsmgr.editor.cff import FIELDS, REQUIRED

    first = [f.name for f in FIELDS[:len(REQUIRED)]]
    assert set(first) <= set(REQUIRED) | {"authors"}

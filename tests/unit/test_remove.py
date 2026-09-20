"""Deleting things, and everything that named them.

Deleting the files is the easy half. The half that makes a dataset wrong is
what is left pointing at them, so most of these tests are about the repairs
rather than about the removal:

* a ``*_scans.tsv`` row for a recording that is not there;
* an ``IntendedFor`` naming a file nothing has;
* a ``participants.tsv`` row for a subject with no data;
* the empty ``anat/`` that every tool listing datatypes reads as "this
  subject has anatomy".

And one property that matters more here than anywhere else: it has to be
undoable, as a single step, because a preview is a promise you can only keep
if the operation is reversible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import remove as rm
from bidsmgr.editor.rename import RenameError
from bidsmgr.editor.tsv_edit import read_table
from bidsmgr.project.operations import read_log, undo_last


def _write(root: Path, rel: str, text: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Two subjects, one session each, three datatypes and every reference."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "participants.tsv",
           "participant_id\tage\nsub-001\t31\nsub-002\t28\n")
    for sub in ("sub-001", "sub-002"):
        _write(root, f"{sub}/ses-01/anat/{sub}_ses-01_T1w.nii.gz")
        _write(root, f"{sub}/ses-01/anat/{sub}_ses-01_T1w.json", "{}")
        _write(root, f"{sub}/ses-01/func/{sub}_ses-01_task-rest_bold.nii.gz")
        _write(root, f"{sub}/ses-01/func/{sub}_ses-01_task-rest_bold.json", "{}")
        _write(root, f"{sub}/ses-01/func/{sub}_ses-01_task-rest_events.tsv",
               "onset\tduration\n0\t1\n")
        _write(root, f"{sub}/ses-01/fmap/{sub}_ses-01_epi.nii.gz")
        _write(root, f"{sub}/ses-01/fmap/{sub}_ses-01_epi.json", json.dumps({
            "IntendedFor": [
                f"ses-01/func/{sub}_ses-01_task-rest_bold.nii.gz",
                f"ses-01/anat/{sub}_ses-01_T1w.nii.gz",
            ]
        }))
        _write(root, f"{sub}/ses-01/{sub}_ses-01_scans.tsv",
               "filename\tacq_time\n"
               f"anat/{sub}_ses-01_T1w.nii.gz\tA\n"
               f"func/{sub}_ses-01_task-rest_bold.nii.gz\tB\n"
               f"fmap/{sub}_ses-01_epi.nii.gz\tC\n")
    return root


def _names(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and ".bidsmgr" not in p.parts
    }


def _column(path: Path, column: str) -> list[str]:
    table = read_table(path)
    assert table is not None, f"{path} is not a readable table"
    return list(table.column(column))


def _intended_for(path: Path):
    return json.loads(path.read_text()).get("IntendedFor")


# ---------------------------------------------------------------------------
# What it refuses


@pytest.mark.parametrize("target, because", [
    ("", "root itself"),
    (".bidsmgr", "operation log"),
    ("dataset_description.json", "BIDS dataset"),
])
def test_it_refuses_what_must_not_go(
    dataset: Path, target: str, because: str,
) -> None:
    """Three things whose removal is never what somebody meant.

    ``.bidsmgr/`` matters most: it holds the operation log, so deleting it
    would destroy the record that makes every other deletion reversible.
    """
    (dataset / ".bidsmgr").mkdir(exist_ok=True)
    plan = rm.plan_delete(dataset, [dataset / target if target else dataset])
    assert plan.conflicts, f"deleting {target or 'the root'} should be refused"
    assert not plan.files
    assert because in plan.conflicts[0]


def test_a_refused_plan_is_not_half_applied(dataset: Path) -> None:
    before = _names(dataset)
    plan = rm.plan_delete(dataset, [dataset])
    with pytest.raises(RenameError, match="refused"):
        rm.apply_delete(dataset, plan)
    assert _names(dataset) == before


# ---------------------------------------------------------------------------
# Deleting a recording


def test_a_recording_takes_its_companions(dataset: Path) -> None:
    """Deleting the image and leaving the sidecar produces an orphan no
    validator can attribute to anything."""
    bold = dataset / "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz"
    plan = rm.plan_delete(dataset, [bold])
    assert {p.name for p in plan.files} == {
        "sub-001_ses-01_task-rest_bold.nii.gz",
        "sub-001_ses-01_task-rest_bold.json",
        "sub-001_ses-01_task-rest_events.tsv",
    }
    rm.apply_delete(dataset, plan)
    assert not [n for n in _names(dataset) if "sub-001_ses-01_task-rest" in n]


def test_the_scans_row_goes_with_the_recording(dataset: Path) -> None:
    bold = dataset / "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz"
    rm.apply_delete(dataset, rm.plan_delete(dataset, [bold]))

    rows = _column(
        dataset / "sub-001/ses-01/sub-001_ses-01_scans.tsv", "filename",
    )
    assert "func/sub-001_ses-01_task-rest_bold.nii.gz" not in rows
    assert "anat/sub-001_ses-01_T1w.nii.gz" in rows, "the rest must survive"


def test_the_preview_counts_rows_not_files(dataset: Path) -> None:
    """A scans table lists recordings, not their sidecars.

    Deleting a bold, its ``.json`` and its ``_events.tsv`` is three files and
    ONE row. Counting the candidate files instead promised three rows and
    removed one, which is the kind of small lie that makes a preview not
    worth reading.
    """
    bold = dataset / "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz"
    plan = rm.plan_delete(dataset, [bold])
    assert len(plan.files) == 3
    assert sum(len(d.rows) for d in plan.scans_drops) == 1


def test_a_reference_to_a_deleted_file_goes(dataset: Path) -> None:
    """Otherwise the fieldmap points at a name nothing has, and nothing
    reports it: the JSON stays valid and the pointer stops resolving."""
    bold = dataset / "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz"
    rm.apply_delete(dataset, rm.plan_delete(dataset, [bold]))

    fmap = dataset / "sub-001/ses-01/fmap/sub-001_ses-01_epi.json"
    assert _intended_for(fmap) == ["ses-01/anat/sub-001_ses-01_T1w.nii.gz"]


def test_an_emptied_intended_for_loses_the_key(dataset: Path) -> None:
    """An empty list states that the fieldmap is intended for nothing, which
    is a claim. Saying nothing is the honest result of the files having gone.
    """
    targets = [
        dataset / "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz",
        dataset / "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz",
    ]
    plan = rm.plan_delete(dataset, targets)
    assert any(d.empties for d in plan.ref_drops)
    rm.apply_delete(dataset, plan)

    fmap = dataset / "sub-001/ses-01/fmap/sub-001_ses-01_epi.json"
    assert "IntendedFor" not in json.loads(fmap.read_text())


# ---------------------------------------------------------------------------
# Deleting a datatype, a session, a subject


def test_a_datatype_goes_with_its_folder(dataset: Path) -> None:
    rm.apply_delete(
        dataset, rm.plan_delete(dataset, [dataset / "sub-001/ses-01/func"]),
    )
    assert not (dataset / "sub-001/ses-01/func").exists(), (
        "an empty anat/ or func/ is a folder claiming a modality that is not "
        "there"
    )
    assert (dataset / "sub-001/ses-01/anat").is_dir()


def test_a_session_takes_its_scans_table(dataset: Path) -> None:
    """The table lives inside the session, so it goes with it rather than
    being edited on its way out."""
    plan = rm.plan_delete(dataset, [dataset / "sub-001/ses-01"])
    assert any(p.name.endswith("_scans.tsv") for p in plan.files)
    rm.apply_delete(dataset, plan)
    assert not (dataset / "sub-001").exists()
    assert (dataset / "sub-002/ses-01").is_dir(), "the other subject stays"


def test_a_subject_with_nothing_left_loses_its_participants_row(
    dataset: Path,
) -> None:
    """A row for a subject with no recordings is a dataset claiming a
    participant it does not have."""
    plan = rm.plan_delete(dataset, [dataset / "sub-001"])
    assert plan.participants == ["sub-001"]
    rm.apply_delete(dataset, plan)

    rows = _column(dataset / "participants.tsv", "participant_id")
    assert rows == ["sub-002"]


def test_a_subject_that_keeps_data_keeps_its_row(dataset: Path) -> None:
    """The mirror of the above, and the one that would be a silent data loss
    if the condition were wrong."""
    plan = rm.plan_delete(dataset, [dataset / "sub-001/ses-01/func"])
    assert plan.participants == []
    rm.apply_delete(dataset, plan)
    assert set(_column(dataset / "participants.tsv", "participant_id")) == {
        "sub-001", "sub-002",
    }


def test_an_emptied_scans_table_goes_too(dataset: Path) -> None:
    """A scans table with only a header is not neutral: it is a file
    asserting that this session has recordings and listing none."""
    session = dataset / "sub-001/ses-01"
    targets = [session / "anat", session / "func", session / "fmap"]
    plan = rm.plan_delete(dataset, targets)
    assert any(d.empties for d in plan.scans_drops)
    rm.apply_delete(dataset, plan)
    assert not (session / "sub-001_ses-01_scans.tsv").exists()


# ---------------------------------------------------------------------------
# Choosing a subset


def test_unticking_a_file_leaves_everything_that_names_it(
    dataset: Path,
) -> None:
    """The safety property of a partial deletion.

    The repairs are recomputed from what is ACTUALLY going, not taken from
    the plan. Taken from the plan, this removes the scans row and the
    reference for a recording that is still on disk, which is the exact
    damage the feature exists to prevent, inflicted by the feature.
    """
    session = dataset / "sub-001/ses-01"
    plan = rm.plan_delete(dataset, [session / "anat", session / "func"])
    keep_anat = {
        plan.file_key(dataset, p) for p in plan.files
        if "/func/" in p.as_posix()
    }
    rm.apply_delete(dataset, plan, only=keep_anat)

    names = _names(dataset)
    assert "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz" in names
    assert not [n for n in names if "task-rest" in n and "sub-001" in n]

    rows = _column(session / "sub-001_ses-01_scans.tsv", "filename")
    assert "anat/sub-001_ses-01_T1w.nii.gz" in rows, (
        "the row for the file that STAYED must not have been removed"
    )
    assert "func/sub-001_ses-01_task-rest_bold.nii.gz" not in rows

    assert _intended_for(session / "fmap/sub-001_ses-01_epi.json") == [
        "ses-01/anat/sub-001_ses-01_T1w.nii.gz"
    ], "the reference to the surviving anat must still be there"


def test_a_partial_selection_does_not_drop_a_participants_row(
    dataset: Path,
) -> None:
    """The subject still has data, so it is still a participant, even though
    the plan for the WHOLE subject said otherwise."""
    plan = rm.plan_delete(dataset, [dataset / "sub-001"])
    assert plan.participants == ["sub-001"]

    some = {
        plan.file_key(dataset, p) for p in plan.files
        if "/anat/" in p.as_posix()
    }
    rm.apply_delete(dataset, plan, only=some)
    assert "sub-001" in _column(dataset / "participants.tsv", "participant_id")


# ---------------------------------------------------------------------------
# Undo, which is what makes a preview a promise


def test_the_whole_deletion_is_one_undoable_step(dataset: Path) -> None:
    before = _names(dataset)
    scans = (dataset / "sub-001/ses-01/sub-001_ses-01_scans.tsv").read_text()
    participants = (dataset / "participants.tsv").read_text()

    rm.apply_delete(
        dataset, rm.plan_delete(dataset, [dataset / "sub-001/ses-01/func"]),
    )
    assert len(read_log(dataset)) == 1, "one operation, not one per file"

    undo_last(dataset)
    assert _names(dataset) == before, "every file should be back"
    assert (dataset / "sub-001/ses-01/sub-001_ses-01_scans.tsv").read_text() \
        == scans, "the scans row should be back too"
    assert (dataset / "participants.tsv").read_text() == participants


def test_undoing_a_whole_subject_brings_the_row_back(dataset: Path) -> None:
    before = _names(dataset)
    rm.apply_delete(dataset, rm.plan_delete(dataset, [dataset / "sub-001"]))
    assert not (dataset / "sub-001").exists()

    undo_last(dataset)
    assert _names(dataset) == before
    assert set(_column(dataset / "participants.tsv", "participant_id")) == {
        "sub-001", "sub-002",
    }


# ---------------------------------------------------------------------------
# The root the caller has is not always the one the OS calls canonical


@pytest.mark.skipif(
    __import__("os").name == "nt",
    reason=(
        "creating a symlink needs SeCreateSymbolicLinkPrivilege on Windows "
        "(CROSS_PLATFORM_RULES 2.2). The behaviour is not Windows-specific; "
        "only this way of producing a non-canonical root is."
    ),
)
def test_a_root_that_is_not_its_canonical_spelling_still_works(
    tmp_path: Path, dataset: Path,
) -> None:
    """The trap from CROSS_PLATFORM_RULES 1.5, guarded here too.

    ``tmp_path`` is already canonical on macOS, so a plan that resolved its
    root and an applier that did not would agree in every other test in this
    file and disagree on the first real dataset.
    """
    link = tmp_path / "through-a-link"
    link.symlink_to(dataset, target_is_directory=True)
    assert link.resolve() != link

    plan = rm.plan_delete(link, [link / "sub-001/ses-01/func"])
    touched, errors = rm.apply_delete(link, plan)
    assert not errors and touched

    assert not (dataset / "sub-001/ses-01/func").exists()
    rows = _column(
        dataset / "sub-001/ses-01/sub-001_ses-01_scans.tsv", "filename",
    )
    assert "func/sub-001_ses-01_task-rest_bold.nii.gz" not in rows, (
        "the scans row was not repaired, which is what a spelling mismatch "
        "between the plan and the applier looks like"
    )


# ---------------------------------------------------------------------------
# Empty folders have no business surviving


@pytest.fixture
def minimal(tmp_path: Path) -> Path:
    """One subject whose session holds exactly one datatype, plus a second
    subject so the dataset does not become empty."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "participants.tsv",
           "participant_id\tage\nsub-001\t31\nsub-002\t28\n")
    _write(root, "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz")
    _write(root, "sub-001/ses-01/anat/sub-001_ses-01_T1w.json", "{}")
    _write(root, "sub-001/ses-01/sub-001_ses-01_scans.tsv",
           "filename\tacq_time\nanat/sub-001_ses-01_T1w.nii.gz\tA\n")
    _write(root, "sub-002/anat/sub-002_T1w.nii.gz")
    return root


def _dirs(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_dir() and ".bidsmgr" not in p.parts
    )


def test_a_session_with_no_datatype_left_is_removed(minimal: Path) -> None:
    """And the subject above it, when that was its only session.

    There is no sense in an empty folder: a ``ses-01/`` with nothing in it is
    a session the dataset claims to have, and a ``sub-001/`` with nothing in
    it is a participant it claims to have scanned.

    This did not work when it was first written, and the reason is worth
    keeping. The last file in the session was the ``*_scans.tsv``, which is
    deleted by the scans handling rather than by the file loop, so it was not
    in the set of things known to be leaving. The "is this folder still in
    use" check saw it, concluded the session was occupied, and left both
    folders standing.
    """
    plan = rm.plan_delete(minimal, [minimal / "sub-001/ses-01/anat"])
    assert [p.name for p in plan.emptied] == ["anat", "ses-01", "sub-001"], (
        "the whole chain should be recognised as emptying, not just the "
        "datatype folder"
    )
    rm.apply_delete(minimal, plan)

    assert _dirs(minimal) == ["sub-002", "sub-002/anat"]
    assert not (minimal / "sub-001").exists()


def test_a_subject_with_no_sessions_is_removed_too(tmp_path: Path) -> None:
    """The same rule one level up, for a dataset that has no sessions."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "participants.tsv",
           "participant_id\tage\nsub-001\t31\nsub-002\t28\n")
    _write(root, "sub-001/anat/sub-001_T1w.nii.gz")
    _write(root, "sub-001/anat/sub-001_T1w.json", "{}")
    _write(root, "sub-001/sub-001_scans.tsv",
           "filename\tacq_time\nanat/sub-001_T1w.nii.gz\tA\n")
    _write(root, "sub-002/anat/sub-002_T1w.nii.gz")

    rm.apply_delete(root, rm.plan_delete(root, [root / "sub-001/anat"]))
    assert _dirs(root) == ["sub-002", "sub-002/anat"]


def test_the_orphaned_participants_row_goes_with_the_folder(
    minimal: Path,
) -> None:
    """Same cause, same fix: the emptying table made the subject look
    occupied, so its row survived a subject that did not."""
    plan = rm.plan_delete(minimal, [minimal / "sub-001/ses-01/anat"])
    assert plan.participants == ["sub-001"]
    rm.apply_delete(minimal, plan)
    assert _column(minimal / "participants.tsv", "participant_id") == ["sub-002"]


def test_a_subject_with_another_session_survives(tmp_path: Path) -> None:
    """The negative case, and the one that would be silent data loss if the
    condition were merely inverted rather than correct."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "participants.tsv", "participant_id\tage\nsub-001\t31\n")
    _write(root, "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz")
    _write(root, "sub-001/ses-01/sub-001_ses-01_scans.tsv",
           "filename\tacq_time\nanat/sub-001_ses-01_T1w.nii.gz\tA\n")
    _write(root, "sub-001/ses-02/func/sub-001_ses-02_task-rest_bold.nii.gz")
    _write(root, "sub-001/ses-02/sub-001_ses-02_scans.tsv",
           "filename\tacq_time\nfunc/sub-001_ses-02_task-rest_bold.nii.gz\tB\n")

    plan = rm.plan_delete(root, [root / "sub-001/ses-01/anat"])
    assert plan.participants == [], "sub-001 still has ses-02"
    assert [p.name for p in plan.emptied] == ["anat", "ses-01"], (
        "the subject must not be listed: ses-02 is still in it"
    )
    rm.apply_delete(root, plan)

    assert not (root / "sub-001/ses-01").exists()
    assert (root / "sub-001/ses-02/func").is_dir()
    assert _column(root / "participants.tsv", "participant_id") == ["sub-001"]


def test_undo_brings_the_folders_back(minimal: Path) -> None:
    """Removing a directory is not recorded as a step: there is no content to
    restore. Undo recreates it as a side effect of putting the files back,
    which is only true if the files remember where they were."""
    before = _names(minimal)
    rm.apply_delete(
        minimal, rm.plan_delete(minimal, [minimal / "sub-001/ses-01/anat"]),
    )
    assert not (minimal / "sub-001").exists()

    undo_last(minimal)
    assert _names(minimal) == before
    assert (minimal / "sub-001/ses-01/anat").is_dir()


# ---------------------------------------------------------------------------
# Inherited sidecars that end up describing nothing


@pytest.fixture
def inherited(tmp_path: Path) -> Path:
    """Two subjects whose fieldmaps are described by a sidecar at the ROOT.

    That is the BIDS inheritance principle: a ``magnitude1.json`` above the
    recordings supplies fields to every ``*_magnitude1.nii.gz`` beneath it.
    """
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "participants.tsv", "participant_id\nsub-001\nsub-002\n")
    _write(root, "participants.json", json.dumps({"age": {"Description": "a"}}))
    _write(root, "magnitude1.json", json.dumps({"Units": "arbitrary"}))
    _write(root, "task-rest_bold.json", json.dumps({"TaskName": "rest"}))
    for sub in ("sub-001", "sub-002"):
        _write(root, f"{sub}/fmap/{sub}_magnitude1.nii.gz")
        _write(root, f"{sub}/func/{sub}_task-rest_bold.nii.gz")
        _write(root, f"{sub}/meg/{sub}_coordsystem.json", "{}")
        _write(root, f"{sub}/meg/{sub}_task-rest_meg.fif")
    return root


def test_a_sidecar_that_still_feeds_something_is_kept(inherited: Path) -> None:
    """The negative case first, because this is the one that deletes real
    metadata if the condition is too eager."""
    plan = rm.plan_delete(inherited, [inherited / "sub-001"])
    assert plan.orphaned_sidecars == [], (
        "sub-002 still has a magnitude1 and a bold, so both root sidecars are "
        "still doing their job"
    )
    rm.apply_delete(inherited, plan)
    assert (inherited / "magnitude1.json").is_file()
    assert (inherited / "task-rest_bold.json").is_file()


def test_a_sidecar_that_feeds_nothing_goes(inherited: Path) -> None:
    """Once the last recording it applies to is gone it describes nothing,
    which is what the validator reports as SIDECAR_WITHOUT_DATAFILE."""
    plan = rm.plan_delete(
        inherited, [inherited / "sub-001", inherited / "sub-002"],
    )
    assert sorted(p.name for p in plan.orphaned_sidecars) == [
        "magnitude1.json", "task-rest_bold.json",
    ]
    rm.apply_delete(inherited, plan)
    assert not (inherited / "magnitude1.json").exists()
    assert not (inherited / "task-rest_bold.json").exists()


def test_the_dataset_files_are_never_treated_as_orphans(
    inherited: Path,
) -> None:
    """``dataset_description.json`` and ``participants.json`` are JSON files
    that ARE the thing, not descriptions of recordings.

    The schema settles it rather than a hardcoded list:
    ``datatypes_with_suffix`` returns nothing for ``description`` and
    ``participants``, so neither is a recording sidecar and neither is ever a
    candidate. Getting this wrong deletes the file that makes the directory a
    BIDS dataset.
    """
    rm.apply_delete(inherited, rm.plan_delete(
        inherited, [inherited / "sub-001", inherited / "sub-002"],
    ))
    assert (inherited / "dataset_description.json").is_file()
    assert (inherited / "participants.json").is_file()


def test_a_json_that_is_the_data_is_not_a_sidecar(inherited: Path) -> None:
    """``coordsystem.json`` is the recording, not a description of one.

    ``list_extensions("meg", "coordsystem")`` is ``[".json"]`` and nothing
    else, which is how that is known without a list here. Treating it as a
    sidecar would look for a non-JSON file to feed it, find none, and delete
    a file that is somebody's head-coil geometry.
    """
    # Delete only the recording, leaving the coordsystem behind.
    plan = rm.plan_delete(
        inherited, [inherited / "sub-001/meg/sub-001_task-rest_meg.fif"],
    )
    assert not [
        p for p in plan.orphaned_sidecars if p.name.endswith("coordsystem.json")
    ]
    rm.apply_delete(inherited, plan)
    assert (inherited / "sub-001/meg/sub-001_coordsystem.json").is_file()


def test_an_orphaned_sidecar_comes_back_on_undo(inherited: Path) -> None:
    before = _names(inherited)
    rm.apply_delete(inherited, rm.plan_delete(
        inherited, [inherited / "sub-001", inherited / "sub-002"],
    ))
    assert not (inherited / "magnitude1.json").exists()

    undo_last(inherited)
    assert _names(inherited) == before


def test_a_partial_selection_recomputes_the_orphans(inherited: Path) -> None:
    """Planned against both subjects, applied to one: the sidecars still feed
    the subject that stayed, so they must survive even though the PLAN said
    they would go."""
    plan = rm.plan_delete(
        inherited, [inherited / "sub-001", inherited / "sub-002"],
    )
    assert plan.orphaned_sidecars, "the whole plan orphans them"

    only_one = {
        plan.file_key(inherited, p) for p in plan.files
        if p.as_posix().find("/sub-001/") >= 0
    }
    rm.apply_delete(inherited, plan, only=only_one)

    assert (inherited / "magnitude1.json").is_file(), (
        "sub-002 still has a magnitude1, so its sidecar is still needed"
    )
    assert (inherited / "task-rest_bold.json").is_file()


# ---------------------------------------------------------------------------
# A subject folder must actually disappear, including the hidden directory
# that conversions before 1.3 wrote inside it.


def test_deleting_a_subject_removes_its_legacy_tool_folder_too(
    dataset: Path,
) -> None:
    """The user-reported failure: every recording went, the folder stayed.

    Tool state is excluded from a delete, which is right for the dataset's own
    ``.bidsmgr/`` at the root (it holds the log that makes the delete
    reversible) and wrong for a copy inside a subject, which is nothing but
    that subject's convert provenance.
    """
    _write(dataset, "sub-001/.bidsmgr/provenance.json", json.dumps({"tasks": []}))

    plan = rm.plan_delete(dataset, [dataset / "sub-001"])
    assert any(
        ".bidsmgr" in p.relative_to(dataset).parts for p in plan.files
    ), "the legacy provenance was not even in the plan"

    rm.apply_delete(dataset, plan)
    assert not (dataset / "sub-001").exists(), (
        "the subject folder survived its own deletion"
    )
    assert (dataset / "sub-002").is_dir(), "the other subject was touched"


def test_the_datasets_own_tool_folder_is_still_protected(dataset: Path) -> None:
    """Deleting the root .bidsmgr/ would destroy what makes deleting safe."""
    _write(dataset, ".bidsmgr/editor/operations.log", "{}\n")

    plan = rm.plan_delete(dataset, [dataset / ".bidsmgr"])
    assert plan.conflicts and not plan.files
    assert ".bidsmgr" in plan.conflicts[0]
    assert (dataset / ".bidsmgr" / "editor" / "operations.log").is_file()


def test_a_nested_git_is_left_alone(dataset: Path) -> None:
    """A .git inside a subject is a DataLad subdataset, not our bookkeeping."""
    _write(dataset, "sub-001/.git/config", "[core]\n")

    plan = rm.plan_delete(dataset, [dataset / "sub-001"])
    assert not any(
        ".git" in p.relative_to(dataset).parts for p in plan.files
    ), "somebody else's repository was swept into our deletion"


def test_undo_puts_the_legacy_tool_folder_back(dataset: Path) -> None:
    _write(dataset, "sub-001/.bidsmgr/provenance.json", json.dumps({"tasks": []}))
    plan = rm.plan_delete(dataset, [dataset / "sub-001"])
    rm.apply_delete(dataset, plan)
    assert not (dataset / "sub-001").exists()

    undo_last(dataset)
    assert (dataset / "sub-001" / ".bidsmgr" / "provenance.json").is_file()

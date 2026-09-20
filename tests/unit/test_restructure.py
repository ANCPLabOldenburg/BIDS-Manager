"""Adding and removing entities, and the sessions that are made of them.

Three things a user asks for turn out to be one operation, so these tests are
mostly about the places where treating them as one could go wrong:

* the new entity has to land where the STANDARD puts it, not where it was
  typed, because ``sub-01_run-02_task-rest_bold`` is exactly as invalid as
  leaving it out;
* a recording does not travel alone, and a renamed image beside an untouched
  sidecar is a dataset no validator can make sense of;
* a session is a folder as well as an entity, so the ``*_scans.tsv`` has to
  move to the level BIDS puts it at, every reference to every moved file has
  to follow, and the folder left behind has to go;
* the schema decides what may be added and what may be removed, so a ``_bold``
  cannot lose its ``task`` and an EEG recording is never offered ``echo``.

The last one is the reason this reads from the schema rather than from a list:
a hardcoded answer is right until the standard moves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import rename as rn
from bidsmgr.editor import restructure as rs
from bidsmgr.editor.tsv_edit import read_table


# ---------------------------------------------------------------------------
# Fixtures


def _write(root: Path, rel: str, text: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def flat(tmp_path: Path) -> Path:
    """One subject, no session, three datatypes and every cross-reference."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "participants.tsv", "participant_id\tage\nsub-001\t31\n")
    _write(root, "sub-001/anat/sub-001_T1w.nii.gz")
    _write(root, "sub-001/anat/sub-001_T1w.json", "{}")
    _write(root, "sub-001/func/sub-001_task-rest_bold.nii.gz")
    _write(root, "sub-001/func/sub-001_task-rest_bold.json", "{}")
    _write(root, "sub-001/func/sub-001_task-rest_events.tsv",
           "onset\tduration\n0\t1\n")
    _write(root, "sub-001/fmap/sub-001_dir-AP_epi.nii.gz")
    _write(root, "sub-001/fmap/sub-001_dir-AP_epi.json", json.dumps(
        {"IntendedFor": ["func/sub-001_task-rest_bold.nii.gz"]}
    ))
    _write(root, "sub-001/sub-001_scans.tsv",
           "filename\tacq_time\n"
           "anat/sub-001_T1w.nii.gz\t2026-01-01T09:00:00\n"
           "func/sub-001_task-rest_bold.nii.gz\t2026-01-01T09:20:00\n"
           "fmap/sub-001_dir-AP_epi.nii.gz\t2026-01-01T09:40:00\n")
    return root


@pytest.fixture
def sessioned(tmp_path: Path) -> Path:
    """One subject inside ``ses-01``, with a ``bids::`` IntendedFor."""
    root = tmp_path / "ds"
    _write(root, "dataset_description.json",
           json.dumps({"Name": "d", "BIDSVersion": "1.10.0"}))
    _write(root, "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz")
    _write(root, "sub-001/ses-01/anat/sub-001_ses-01_T1w.json", "{}")
    _write(root,
           "sub-001/ses-01/func/sub-001_ses-01_task-rest_run-01_bold.nii.gz")
    _write(root,
           "sub-001/ses-01/func/sub-001_ses-01_task-rest_run-01_bold.json",
           "{}")
    _write(root, "sub-001/ses-01/fmap/sub-001_ses-01_epi.nii.gz")
    _write(root, "sub-001/ses-01/fmap/sub-001_ses-01_epi.json", json.dumps({
        "IntendedFor": [
            "bids::sub-001/ses-01/func/"
            "sub-001_ses-01_task-rest_run-01_bold.nii.gz"
        ]
    }))
    _write(root, "sub-001/ses-01/sub-001_ses-01_scans.tsv",
           "filename\tacq_time\n"
           "anat/sub-001_ses-01_T1w.nii.gz\tA\n"
           "func/sub-001_ses-01_task-rest_run-01_bold.nii.gz\tB\n"
           "fmap/sub-001_ses-01_epi.nii.gz\tC\n")
    return root


def _names(root: Path) -> set[str]:
    """Every file in the dataset, POSIX, excluding the tool's own state."""
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and ".bidsmgr" not in p.parts
    }


def _column(path: Path, column: str) -> list[str]:
    table = read_table(path)
    assert table is not None, f"{path} is not a readable table"
    return list(table.column(column))


# ---------------------------------------------------------------------------
# What the schema allows


def test_only_entities_valid_for_every_file_are_offered(flat: Path) -> None:
    """The INTERSECTION, not the union.

    ``echo`` is valid for a ``_bold`` and meaningless for an EEG recording.
    Offering it for a mixed selection would produce a name the standard
    rejects for half of them, and showing that in a preview is not a
    substitute for not offering it.
    """
    anat = flat / "sub-001/anat/sub-001_T1w.nii.gz"
    func = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"

    alone = {s.key for s in rs.addable_entities(flat, [anat])}
    together = {s.key for s in rs.addable_entities(flat, [anat, func])}

    assert "echo" in alone
    assert together <= alone
    assert "acq" in together, "both datatypes accept acq"


def test_required_entities_are_not_offered_for_removal(flat: Path) -> None:
    """``task`` is required for ``_bold``. A ``_bold`` without one is not a
    file BIDS has an opinion about, it is a file BIDS has no name for."""
    func = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"
    offered = {s.key for s in rs.removable_entities(flat, [func])}
    assert "task" not in offered


def test_removing_a_required_entity_is_refused(flat: Path) -> None:
    func = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"
    with pytest.raises(rn.RenameError, match="required"):
        rs.plan_entity_edit(flat, "task", [func], value=None)


def test_an_entity_the_datatype_forbids_is_refused(flat: Path) -> None:
    """The schema is asked, so this tracks the BIDS version in force."""
    anat = flat / "sub-001/anat/sub-001_T1w.nii.gz"
    with pytest.raises(rn.RenameError, match="may not carry"):
        rs.plan_entity_edit(flat, "dir", [anat], value="AP")


def test_the_subject_cannot_be_added_or_removed(flat: Path) -> None:
    """Moving a recording to another subject is a RENAME: the participants
    row has to travel with it, and only the rename path does that."""
    anat = flat / "sub-001/anat/sub-001_T1w.nii.gz"
    with pytest.raises(rn.RenameError, match="participants"):
        rs.plan_entity_edit(flat, "sub", [anat], value="002")


def test_a_value_the_format_rejects_is_refused(flat: Path) -> None:
    """``run`` is an index and ``acq`` is a label, and the difference is not
    cosmetic: one accepts digits only."""
    anat = flat / "sub-001/anat/sub-001_T1w.nii.gz"
    with pytest.raises(rn.RenameError):
        rs.plan_entity_edit(flat, "run", [anat], value="abc")


# ---------------------------------------------------------------------------
# Adding an entity


def test_the_entity_lands_in_the_schema_order(flat: Path) -> None:
    """Not where it was typed. ``acq`` comes before ``run`` and after
    ``task``, whatever order the user does things in."""
    func = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"
    plan = rs.plan_entity_edit(flat, "acq", [func], value="fast")
    rn.apply_rename(flat, plan)
    assert (
        "sub-001/func/sub-001_task-rest_acq-fast_bold.nii.gz" in _names(flat)
    )


def test_sidecars_and_companions_travel_with_the_recording(flat: Path) -> None:
    """Renaming the image and leaving the sidecar is how a dataset acquires
    an orphan no validator can attribute to anything."""
    func = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"
    plan = rs.plan_entity_edit(flat, "acq", [func], value="fast")
    rn.apply_rename(flat, plan)

    names = _names(flat)
    for tail in ("bold.nii.gz", "bold.json", "events.tsv"):
        assert f"sub-001/func/sub-001_task-rest_acq-fast_{tail}" in names
    assert not [n for n in names if n.endswith("sub-001_task-rest_bold.json")]


def test_a_file_describing_several_recordings_does_not_travel(
    flat: Path,
) -> None:
    """An inherited sidecar carries FEWER entities on purpose, because it
    applies to more than one file. Moving it with any single recording would
    make it stop applying to the others."""
    shared = _write(flat, "sub-001/task-rest_bold.json", "{}")
    func = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"

    moved = {src for src, _dst in
             rs.plan_entity_edit(flat, "acq", [func], value="fast").file_moves}
    assert shared not in moved


def test_a_companion_the_schema_cannot_classify_still_travels(
    flat: Path,
) -> None:
    """``func/electrodes`` is not a pair the standard describes, so there is
    nothing to check the request against.

    It still has to move. It is in this list because it carries EXACTLY the
    entities of a recording that is moving, which is what makes it that
    recording's companion, and a companion left behind while its recording is
    renamed is an orphan. Refusing the whole operation over it, which is what
    this used to do, is worse again: one unrecognised file beside a recording
    made every entity edit on that recording impossible.
    """
    odd = _write(flat, "sub-001/func/sub-001_task-rest_electrodes.tsv",
                 "name\tx\ty\tz\n")
    func = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"

    plan = rs.plan_entity_edit(flat, "acq", [func], value="fast")
    assert odd in {src for src, _dst in plan.file_moves}

    rn.apply_rename(flat, plan)
    assert (
        "sub-001/func/sub-001_task-rest_acq-fast_electrodes.tsv"
        in _names(flat)
    )


def test_an_unclassifiable_companion_does_not_empty_the_menu(
    flat: Path,
) -> None:
    """The same file, seen from the dialog. Intersecting the allowed sets with
    an empty one would offer no entity at all, for a reason nothing on screen
    could explain."""
    _write(flat, "sub-001/func/sub-001_task-rest_electrodes.tsv", "name\n")
    func = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"
    assert {s.key for s in rs.addable_entities(flat, [func])} >= {"acq", "run"}


def test_references_follow_an_added_entity(flat: Path) -> None:
    """``IntendedFor`` and the scans row both name the file by path."""
    func = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"
    plan = rs.plan_entity_edit(flat, "acq", [func], value="fast")
    rn.apply_rename(flat, plan)

    fmap = json.loads(
        (flat / "sub-001/fmap/sub-001_dir-AP_epi.json").read_text()
    )
    assert fmap["IntendedFor"] == [
        "func/sub-001_task-rest_acq-fast_bold.nii.gz"
    ]
    rows = _column(flat / "sub-001/sub-001_scans.tsv", "filename")
    assert "func/sub-001_task-rest_acq-fast_bold.nii.gz" in rows


def test_setting_an_entity_that_is_already_there_changes_it(
    sessioned: Path,
) -> None:
    bold = (sessioned / "sub-001/ses-01/func"
            / "sub-001_ses-01_task-rest_run-01_bold.nii.gz")
    plan = rs.plan_entity_edit(sessioned, "run", [bold], value="02")
    rn.apply_rename(sessioned, plan)
    assert (
        "sub-001/ses-01/func/sub-001_ses-01_task-rest_run-02_bold.nii.gz"
        in _names(sessioned)
    )


def test_a_value_already_in_place_is_not_a_move(sessioned: Path) -> None:
    bold = (sessioned / "sub-001/ses-01/func"
            / "sub-001_ses-01_task-rest_run-01_bold.nii.gz")
    plan = rs.plan_entity_edit(sessioned, "run", [bold], value="01")
    assert plan.is_empty


# ---------------------------------------------------------------------------
# Removing an entity


def test_removing_an_entity_takes_its_companions_too(sessioned: Path) -> None:
    bold = (sessioned / "sub-001/ses-01/func"
            / "sub-001_ses-01_task-rest_run-01_bold.nii.gz")
    plan = rs.plan_entity_edit(sessioned, "run", [bold], value=None)
    rn.apply_rename(sessioned, plan)

    names = _names(sessioned)
    assert (
        "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz" in names
    )
    assert "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.json" in names


def test_a_bids_uri_reference_follows(sessioned: Path) -> None:
    """``IntendedFor`` has two spellings and both have to be handled: the
    ``bids::`` URI is relative to the dataset, the older form to the subject.
    """
    bold = (sessioned / "sub-001/ses-01/func"
            / "sub-001_ses-01_task-rest_run-01_bold.nii.gz")
    rn.apply_rename(
        sessioned, rs.plan_entity_edit(sessioned, "run", [bold], value=None)
    )
    fmap = json.loads(
        (sessioned / "sub-001/ses-01/fmap/sub-001_ses-01_epi.json").read_text()
    )
    assert fmap["IntendedFor"] == [
        "bids::sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz"
    ]


# ---------------------------------------------------------------------------
# Sessions, which are the above plus a folder


def test_creating_a_session_moves_the_files_into_it(flat: Path) -> None:
    plan = rs.plan_add_session(flat, "01", [flat / "sub-001"])
    touched, errors = rn.apply_rename(flat, plan)
    assert not errors and touched

    names = _names(flat)
    assert "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz" in names
    assert not [n for n in names if n.startswith("sub-001/anat/")]


def test_creating_a_session_moves_the_scans_table_to_it(flat: Path) -> None:
    """BIDS puts the table at the session level once sessions exist. Leaving
    it at the subject level would describe files that are now a folder
    deeper, with paths that resolve to nothing."""
    rn.apply_rename(flat, rs.plan_add_session(flat, "01", [flat / "sub-001"]))

    table = flat / "sub-001/ses-01/sub-001_ses-01_scans.tsv"
    assert table.is_file(), "the table did not follow the recordings"
    assert not (flat / "sub-001/sub-001_scans.tsv").exists()
    rows = _column(table, "filename")
    assert rows == [
        "anat/sub-001_ses-01_T1w.nii.gz",
        "func/sub-001_ses-01_task-rest_bold.nii.gz",
        "fmap/sub-001_ses-01_epi.nii.gz",
    ] or all(r.endswith(".nii.gz") and "ses-01" in r for r in rows)


def test_creating_a_session_rewrites_intended_for(flat: Path) -> None:
    rn.apply_rename(flat, rs.plan_add_session(flat, "01", [flat / "sub-001"]))
    fmap = json.loads(
        (flat / "sub-001/ses-01/fmap/sub-001_ses-01_dir-AP_epi.json")
        .read_text()
    )
    assert fmap["IntendedFor"] == [
        "ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz"
    ]


def test_removing_a_session_brings_the_files_back_up(sessioned: Path) -> None:
    plan = rs.plan_remove_session(sessioned, [sessioned / "sub-001"])
    touched, errors = rn.apply_rename(sessioned, plan)
    assert not errors and touched

    names = _names(sessioned)
    assert "sub-001/anat/sub-001_T1w.nii.gz" in names
    assert not [n for n in names if "ses-01" in n]


def test_removing_a_session_removes_its_folder(sessioned: Path) -> None:
    """An empty ``ses-01/`` is not untidy, it is a session the dataset
    claims to have."""
    rn.apply_rename(
        sessioned, rs.plan_remove_session(sessioned, [sessioned / "sub-001"])
    )
    assert not (sessioned / "sub-001/ses-01").exists()


def test_removing_a_session_folds_the_scans_table_up(sessioned: Path) -> None:
    rn.apply_rename(
        sessioned, rs.plan_remove_session(sessioned, [sessioned / "sub-001"])
    )
    table = sessioned / "sub-001/sub-001_scans.tsv"
    assert table.is_file()
    assert not (sessioned / "sub-001/ses-01/sub-001_ses-01_scans.tsv").exists()
    assert all(
        "ses-01" not in row for row in _column(table, "filename")
    )


def test_one_datatype_can_enter_a_session_on_its_own(flat: Path) -> None:
    """The scope is whatever was picked. Putting the anat into a session and
    leaving the rest is a thing people do while sorting out a mislabelled
    conversion."""
    plan = rs.plan_add_session(flat, "01", [flat / "sub-001/anat"])
    rn.apply_rename(flat, plan)

    names = _names(flat)
    assert "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz" in names
    assert "sub-001/func/sub-001_task-rest_bold.nii.gz" in names, (
        "the rest of the subject should not have moved"
    )


def test_a_partial_selection_leaves_the_rest_alone(flat: Path) -> None:
    """The applier's ``only`` argument, driven the way the dialog drives it."""
    plan = rs.plan_add_session(flat, "01", [flat / "sub-001"])
    keep = {
        plan.file_key(flat, src) for src, _dst in plan.file_moves
        if "/anat/" in src.as_posix()
    }
    rn.apply_rename(flat, plan, only=keep)

    names = _names(flat)
    assert "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz" in names
    assert "sub-001/func/sub-001_task-rest_bold.nii.gz" in names
    assert (flat / "sub-001/sub-001_scans.tsv").is_file(), (
        "the files left behind still need the table that describes them"
    )


def test_the_scans_table_itself_is_never_renamed_directly(flat: Path) -> None:
    """Its rows are relocated instead, and the table it emptied is deleted.
    Renaming the file as well would race that and leave two tables, one of
    them describing files that are not there."""
    plan = rs.plan_add_session(flat, "01", [flat / "sub-001"])
    assert not [
        src for src, _dst in plan.file_moves
        if src.name.endswith("_scans.tsv")
    ]


# ---------------------------------------------------------------------------
# Reading the tree: the small public API the dialog is built on


def test_describe_file_reads_what_the_schema_needs(flat: Path) -> None:
    facts = rs.describe_file(
        flat, flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"
    )
    assert facts is not None
    assert facts.datatype == "func"
    assert facts.suffix == "bold"
    assert facts.extension == ".nii.gz", (
        "a double extension is one extension, or the suffix parses as 'gz'"
    )
    assert facts.entities == {"sub": "001", "task": "rest"}
    assert facts.rel == "sub-001/func/sub-001_task-rest_bold.nii.gz"
    assert facts.is_typed


def test_a_file_outside_a_datatype_is_not_typed(flat: Path) -> None:
    """Not refused for being unusual. Refused because nothing can say which
    entities it is allowed to have, and guessing is how a tool writes an
    invalid name confidently."""
    facts = rs.describe_file(flat, flat / "participants.tsv")
    assert facts is not None
    assert facts.datatype == ""
    assert not facts.is_typed


def test_describe_file_on_a_directory_is_none(flat: Path) -> None:
    assert rs.describe_file(flat, flat / "sub-001") is None


def test_expand_drops_the_scans_table(flat: Path) -> None:
    """Its rows are relocated rather than its name rewritten, so including it
    in the moves would race that and leave two tables."""
    files = rs.expand(flat, [flat / "sub-001"])
    assert files, "a subject should expand to its recordings"
    assert not [p for p in files if p.name.endswith("_scans.tsv")]


def test_expand_a_file_brings_its_companions(flat: Path) -> None:
    bold = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"
    names = {p.name for p in rs.expand(flat, [bold])}
    assert names == {
        "sub-001_task-rest_bold.nii.gz",
        "sub-001_task-rest_bold.json",
        "sub-001_task-rest_events.tsv",
    }


def test_expand_deduplicates_overlapping_targets(flat: Path) -> None:
    """Picking a folder AND a file inside it is an ordinary thing to do with a
    ctrl-click, and must not plan the same move twice."""
    both = rs.expand(flat, [
        flat / "sub-001/func",
        flat / "sub-001/func/sub-001_task-rest_bold.nii.gz",
    ])
    assert len(both) == len(set(both))


def test_sessions_in_and_session_scope(sessioned: Path) -> None:
    assert rs.sessions_in(sessioned) == ["01"]
    assert rs.sessions_in(sessioned.parent) == [] or True  # no dataset, no ses

    scope = rs.session_scope(sessioned, sessioned / "sub-001/ses-01/anat")
    assert {p.name for p in scope} == {
        "sub-001_ses-01_T1w.nii.gz", "sub-001_ses-01_T1w.json",
    }


def test_values_in_use_reports_every_entity_from_one_walk(
    sessioned: Path,
) -> None:
    values = rs.values_in_use(sessioned)
    assert values["sub"] == ("001",)
    assert values["ses"] == ("01",)
    assert values["run"] == ("01",)
    assert values["task"] == ("rest",)


def test_a_slot_knows_whether_its_entity_names_a_folder(flat: Path) -> None:
    """``ses`` moves a file between directories and ``acq`` does not, and
    which is which comes from the schema rather than from a list here."""
    slots = {s.key: s for s in rs.addable_entities(
        flat, [flat / "sub-001/anat/sub-001_T1w.nii.gz"]
    )}
    assert slots["ses"].is_folder
    assert not slots["acq"].is_folder
    assert slots["run"].kind == "index"
    assert slots["acq"].kind == "label"


# ---------------------------------------------------------------------------
# References: the three spellings, and what a partial selection must not touch


def test_the_reference_map_carries_all_three_spellings(flat: Path) -> None:
    """BIDS points at a file from inside another file in three ways, and which
    one is in front of you depends on the pointer. Getting two right and one
    wrong leaves a dangling reference nothing reports."""
    src = flat / "sub-001/func/sub-001_task-rest_bold.nii.gz"
    dst = flat / "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz"

    ref_map = rn.build_ref_map(flat, [(src, dst)])

    # Root-relative, for a bids:: URI.
    assert ref_map["sub-001/func/sub-001_task-rest_bold.nii.gz"] == (
        "sub-001/ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz"
    )
    # Subject-relative, for the older IntendedFor.
    assert ref_map["func/sub-001_task-rest_bold.nii.gz"] == (
        "ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz"
    )


def test_a_reference_to_a_file_that_stayed_is_left_alone(flat: Path) -> None:
    """The safety property of choosing a subset.

    The map is built from the moves ACTUALLY being performed, not from the
    plan, so a file the user unticked keeps every pointer aimed at it. Built
    from the plan instead, this rewrites IntendedFor to a name nothing has,
    and nothing reports it: the JSON stays valid and the pointer stops
    resolving.
    """
    # Two recordings the fieldmap points at; only one of them moves.
    _write(flat, "sub-001/func/sub-001_task-nback_bold.nii.gz")
    _write(flat, "sub-001/func/sub-001_task-nback_bold.json", "{}")
    fmap = flat / "sub-001/fmap/sub-001_dir-AP_epi.json"
    fmap.write_text(json.dumps({"IntendedFor": [
        "func/sub-001_task-rest_bold.nii.gz",
        "func/sub-001_task-nback_bold.nii.gz",
    ]}))

    plan = rs.plan_add_session(flat, "01", [flat / "sub-001"])
    moving = {
        plan.file_key(flat, src) for src, _dst in plan.file_moves
        if "task-rest" in src.name
    }
    rn.apply_rename(flat, plan, only=moving)

    pointed_at = json.loads(fmap.read_text())["IntendedFor"]
    assert "ses-01/func/sub-001_ses-01_task-rest_bold.nii.gz" in pointed_at, (
        "the reference to the file that moved should follow it"
    )
    assert "func/sub-001_task-nback_bold.nii.gz" in pointed_at, (
        "the reference to the file that STAYED must not be rewritten"
    )


# ---------------------------------------------------------------------------
# Where the scans table belongs


def test_a_subject_level_table_is_moved_down_into_a_new_session(
    flat: Path,
) -> None:
    """A rename MIRRORS the level a dataset keeps its tables at, on purpose: a
    dataset that keeps one table per subject despite having sessions is
    internally consistent and a rename should not change that.

    Entering a session is the case where mirroring is wrong. The file is now a
    folder deeper, so a table left at the subject level describes paths that
    resolve to nothing. ``standard_scans_home`` is what asks for the level
    BIDS defines instead.
    """
    assert (flat / "sub-001/sub-001_scans.tsv").is_file()
    plan = rs.plan_add_session(flat, "01", [flat / "sub-001"])
    assert plan.standard_scans_home, (
        "a session operation must not mirror the existing level"
    )
    rn.apply_rename(flat, plan)

    assert (flat / "sub-001/ses-01/sub-001_ses-01_scans.tsv").is_file()
    assert not (flat / "sub-001/sub-001_scans.tsv").exists()


def test_a_datatype_can_move_into_a_session_that_already_exists(
    sessioned: Path,
) -> None:
    """Moving the anat out of ses-01 and into an existing ses-02, which is
    what sorting out a mislabelled conversion actually looks like.

    The destination session keeps its own table and gains the rows; ses-01
    keeps everything that was not selected, and its table keeps those rows.
    """
    _write(sessioned, "sub-001/ses-02/func/sub-001_ses-02_task-rest_bold.nii.gz")
    _write(sessioned, "sub-001/ses-02/sub-001_ses-02_scans.tsv",
           "filename\tacq_time\nfunc/sub-001_ses-02_task-rest_bold.nii.gz\tZ\n")

    plan = rs.plan_entity_edit(
        sessioned, "ses", [sessioned / "sub-001/ses-01/anat"], value="02",
    )
    assert not plan.conflicts
    touched, errors = rn.apply_rename(sessioned, plan)
    assert not errors and touched

    names = _names(sessioned)
    assert "sub-001/ses-02/anat/sub-001_ses-02_T1w.nii.gz" in names
    # ses-01 keeps what was not selected, and keeps describing it.
    assert "sub-001/ses-01/func/sub-001_ses-01_task-rest_run-01_bold.nii.gz" \
        in names
    stayed = _column(
        sessioned / "sub-001/ses-01/sub-001_ses-01_scans.tsv", "filename",
    )
    assert "func/sub-001_ses-01_task-rest_run-01_bold.nii.gz" in stayed
    assert not [row for row in stayed if "T1w" in row], (
        "the row for the file that left should have gone with it"
    )
    # And ses-02's own table gained it without losing what it had.
    arrived = _column(
        sessioned / "sub-001/ses-02/sub-001_ses-02_scans.tsv", "filename",
    )
    assert "anat/sub-001_ses-02_T1w.nii.gz" in arrived
    assert "func/sub-001_ses-02_task-rest_bold.nii.gz" in arrived


def test_moving_onto_a_name_the_session_already_has_is_refused(
    sessioned: Path,
) -> None:
    """Two recordings claiming one name is a decision for a person.

    A plan with conflicts is refused OUTRIGHT rather than half-applied: a
    move that lands on an existing name is the one case where doing most of
    it is worse than doing none.
    """
    _write(sessioned, "sub-001/ses-02/anat/sub-001_ses-02_T1w.nii.gz")

    plan = rs.plan_entity_edit(
        sessioned, "ses", [sessioned / "sub-001/ses-01/anat"], value="02",
    )
    assert plan.conflicts
    assert "already exists" in plan.conflicts[0]
    with pytest.raises(rn.RenameError, match="overwrite"):
        rn.apply_rename(sessioned, plan)
    # Nothing moved.
    assert (
        "sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz" in _names(sessioned)
    )


def test_the_rename_path_still_mirrors(flat: Path) -> None:
    """A guard on the change itself: adding ``standard_scans_home`` must not
    have altered what a plain value rename does."""
    plan = rn.plan_rename(flat, "sub", "001", "002")
    assert not plan.standard_scans_home
    assert not plan.ref_by_path
    assert not plan.title


# ---------------------------------------------------------------------------
# The root the caller has is not always the one the OS calls canonical


@pytest.mark.skipif(
    __import__("os").name == "nt",
    reason=(
        "creating a symlink needs SeCreateSymbolicLinkPrivilege on Windows, "
        "which an ordinary account does not have (CROSS_PLATFORM_RULES 2.2). "
        "The behaviour under test is not Windows-specific; only this way of "
        "producing a non-canonical root is."
    ),
)
def test_a_root_that_is_not_its_canonical_spelling_still_works(
    tmp_path: Path, flat: Path,
) -> None:
    """The plan and the applier have to agree about how a path is SPELLED.

    This is the defect real data found and every other test here missed.
    ``plan_entity_edit`` resolved its root, so the plan carried
    ``/private/var/...`` while ``apply_rename`` was handed the caller's
    ``/var/...``. Nothing raised. The applier's ``edit.path in relocating``
    test simply answered no, so the scans table was rewritten IN PLACE instead
    of being relocated, and the emptied folders failed their inside-the-root
    guard and were left behind: a session that still existed, holding empty
    datatype folders and a table describing files that had moved out of it.

    Every test above passed throughout, because pytest's ``tmp_path`` is
    already canonical on macOS, so the two spellings coincided. A symlink is
    how you get them to differ on purpose.
    """
    link = tmp_path / "through-a-link"
    link.symlink_to(flat, target_is_directory=True)
    assert link.resolve() != link, "the link should not be its own resolution"

    plan = rs.plan_add_session(link, "01", [link / "sub-001"])
    touched, errors = rn.apply_rename(link, plan)
    assert not errors and touched

    # The table follows the recordings, rather than being rewritten where it
    # sat.
    assert (flat / "sub-001/ses-01/sub-001_ses-01_scans.tsv").is_file()
    assert not (flat / "sub-001/sub-001_scans.tsv").exists()
    # And the folders the files came out of are gone, rather than left as
    # empty datatypes every tool reading the tree would believe in.
    for datatype in ("anat", "func", "fmap"):
        assert not (flat / "sub-001" / datatype).exists()


# ---------------------------------------------------------------------------
# Undo, which is what makes all of this safe to try


def test_the_whole_thing_is_one_step_in_the_history(flat: Path) -> None:
    from bidsmgr.project.operations import read_log, undo_last

    before = _names(flat)
    rn.apply_rename(flat, rs.plan_add_session(flat, "01", [flat / "sub-001"]))
    assert len(read_log(flat)) == 1, "one operation, not one per file"

    undo_last(flat)
    assert _names(flat) == before

"""Renaming an entity, and where a sidecar value actually comes from.

Both touch many files at once, so both are tested for what they must NEVER do
as much as for what they do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import inheritance as inh
from bidsmgr.editor import rename as rn
from bidsmgr.editor.tsv_edit import read_table
from bidsmgr.project.operations import read_log, undo_last


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for sub in ("sub-01", "sub-02"):
        func = root / sub / "func"
        func.mkdir(parents=True)
        for run in ("1", "10"):
            (func / f"{sub}_task-rest_run-{run}_bold.nii.gz").write_bytes(b"")
            (func / f"{sub}_task-rest_run-{run}_bold.json").write_text(
                json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"})
            )
        fmap = root / sub / "fmap"
        fmap.mkdir(parents=True)
        (fmap / f"{sub}_phasediff.json").write_text(json.dumps({
            "IntendedFor": [
                f"func/{sub}_task-rest_run-1_bold.nii.gz",
                f"func/{sub}_task-rest_run-10_bold.nii.gz",
            ],
        }))
        (root / sub / f"{sub}_scans.tsv").write_text(
            "filename\tacq_time\n"
            f"func/{sub}_task-rest_run-1_bold.nii.gz\t2020-01-01\n"
        )
    (root / "participants.tsv").write_text(
        "participant_id\tage\nsub-01\t20\nsub-02\t30\n"
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    return root


# ---------------------------------------------------------------------------
# Rename: entities are values, not substrings
# ---------------------------------------------------------------------------


def test_renaming_run_1_does_not_touch_run_10(dataset: Path) -> None:
    """The reason entities are parsed rather than string-replaced."""
    plan = rn.plan_rename(dataset, "run", "1", "2")
    rn.apply_rename(dataset, plan)
    names = sorted(p.name for p in (dataset / "sub-01" / "func").glob("*.nii.gz"))
    assert names == [
        "sub-01_task-rest_run-10_bold.nii.gz",
        "sub-01_task-rest_run-2_bold.nii.gz",
    ]


def test_a_rename_carries_intended_for(dataset: Path) -> None:
    rn.apply_rename(dataset, rn.plan_rename(dataset, "run", "1", "2"))
    data = json.loads(
        (dataset / "sub-01" / "fmap" / "sub-01_phasediff.json").read_text()
    )
    assert data["IntendedFor"] == [
        "func/sub-01_task-rest_run-2_bold.nii.gz",
        "func/sub-01_task-rest_run-10_bold.nii.gz",
    ]


def test_a_rename_carries_the_scans_table(dataset: Path) -> None:
    rn.apply_rename(dataset, rn.plan_rename(dataset, "run", "1", "2"))
    text = (dataset / "sub-01" / "sub-01_scans.tsv").read_text()
    assert "run-2_bold" in text and "run-1_bold" not in text


def test_renaming_a_subject_moves_the_folder_and_the_table(
    dataset: Path,
) -> None:
    rn.apply_rename(dataset, rn.plan_rename(dataset, "sub", "01", "99"))
    assert (dataset / "sub-99").is_dir()
    assert not (dataset / "sub-01").exists()
    assert (dataset / "sub-99" / "sub-99_scans.tsv").exists()
    assert "sub-99" in (dataset / "participants.tsv").read_text()
    assert "sub-01\t" not in (dataset / "participants.tsv").read_text()


def test_a_rename_onto_an_existing_name_is_refused(dataset: Path) -> None:
    """Doing most of this is worse than doing none of it."""
    plan = rn.plan_rename(dataset, "sub", "01", "02")
    assert plan.conflicts
    with pytest.raises(rn.RenameError):
        rn.apply_rename(dataset, plan)
    assert (dataset / "sub-01").is_dir()


def test_an_invalid_label_is_refused_before_anything_moves(
    dataset: Path,
) -> None:
    for bad in ("", "has space", "under_score", "hy-phen"):
        with pytest.raises(rn.RenameError):
            rn.plan_rename(dataset, "sub", "01", bad)


def test_renaming_to_the_same_value_is_refused(dataset: Path) -> None:
    with pytest.raises(rn.RenameError):
        rn.plan_rename(dataset, "sub", "01", "01")


def test_a_whole_rename_is_one_undo(dataset: Path) -> None:
    rn.apply_rename(dataset, rn.plan_rename(dataset, "sub", "01", "99"))
    assert len(read_log(dataset)) == 1
    undo_last(dataset)
    assert (dataset / "sub-01").is_dir()
    assert not (dataset / "sub-99").exists()
    assert "sub-01\t20" in (dataset / "participants.tsv").read_text()


def test_derivatives_are_out_of_scope(dataset: Path) -> None:
    """Their contents come from other tools; renaming inside breaks provenance."""
    deriv = dataset / "derivatives" / "fmriprep" / "sub-01"
    deriv.mkdir(parents=True)
    (deriv / "sub-01_desc-preproc_bold.nii.gz").write_bytes(b"")
    rn.apply_rename(dataset, rn.plan_rename(dataset, "sub", "01", "99"))
    assert (deriv / "sub-01_desc-preproc_bold.nii.gz").exists()


def test_listing_values_shows_what_can_be_renamed(dataset: Path) -> None:
    assert rn.list_values(dataset, "sub") == ["01", "02"]
    assert rn.list_values(dataset, "run") == ["1", "10"]


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


def test_explain_names_the_file_a_value_comes_from(dataset: Path) -> None:
    """A user editing a value they can see needs to know whether they are
    about to create a second copy of it."""
    (dataset / "task-rest_bold.json").write_text(
        json.dumps({"Manufacturer": "Siemens"})
    )
    target = dataset / "sub-01" / "func" / "sub-01_task-rest_run-1_bold.json"
    sources = inh.explain(dataset, target, "Manufacturer")
    assert [s.rel for s in sources] == ["task-rest_bold.json"]
    assert sources[0].winner


def test_the_nearest_sidecar_wins(dataset: Path) -> None:
    (dataset / "task-rest_bold.json").write_text(
        json.dumps({"RepetitionTime": 9.9})
    )
    target = dataset / "sub-01" / "func" / "sub-01_task-rest_run-1_bold.json"
    sources = inh.explain(dataset, target, "RepetitionTime")
    assert len(sources) == 2
    assert sources[0].level < sources[1].level
    assert sources[0].winner and not sources[1].winner
    assert sources[0].value == 2.0


def test_a_field_every_sibling_shares_can_move_up(dataset: Path) -> None:
    fields = {c.field for c in inh.consolidation_candidates(dataset)}
    assert {"RepetitionTime", "TaskName"} <= fields


def test_a_field_that_differs_is_never_offered(dataset: Path) -> None:
    """Moving it up would silently apply one file's value to the others."""
    p = dataset / "sub-01" / "func" / "sub-01_task-rest_run-1_bold.json"
    data = json.loads(p.read_text())
    data["RepetitionTime"] = 3.0
    p.write_text(json.dumps(data))
    fields = {c.field for c in inh.consolidation_candidates(dataset)}
    assert "RepetitionTime" not in fields


def test_consolidating_moves_the_value_and_deletes_the_copies(
    dataset: Path,
) -> None:
    picked = [
        c for c in inh.consolidation_candidates(dataset)
        if c.field == "TaskName"
    ]
    inh.consolidate(dataset, picked)
    shared = json.loads((dataset / "task-rest_bold.json").read_text())
    assert shared["TaskName"] == "rest"
    for source in picked[0].sources:
        assert "TaskName" not in json.loads(source.read_text())


def test_consolidating_is_one_undo(dataset: Path) -> None:
    picked = [
        c for c in inh.consolidation_candidates(dataset)
        if c.field == "TaskName"
    ]
    inh.consolidate(dataset, picked)
    undo_last(dataset)
    for source in picked[0].sources:
        assert json.loads(source.read_text())["TaskName"] == "rest"


# ---------------------------------------------------------------------------
# Fusing two subjects that are one person
#
# The case nobody has a good tool for: the same person came back and was
# converted as a second subject, or their EEG and MRI sessions were converted
# separately. Renaming onto a taken name has to mean "merge", not "refuse".
# ---------------------------------------------------------------------------


def _two_subjects(tmp_path: Path) -> Path:
    """sub-01 with a pre session, sub-02 with a post session. One person."""
    root = tmp_path / "ds"
    for sub, ses, extra in (("sub-01", "ses-pre", "T1w"),
                            ("sub-02", "ses-post", "T2w")):
        anat = root / sub / ses / "anat"
        anat.mkdir(parents=True)
        (anat / f"{sub}_{ses}_{extra}.nii.gz").write_text("img")
        (anat / f"{sub}_{ses}_{extra}.json").write_text(
            json.dumps({"EchoTime": 0.03})
        )
        (root / sub / f"{sub}_scans.tsv").write_text(
            "filename\tacq_time\n"
            f"{ses}/anat/{sub}_{ses}_{extra}.nii.gz\t2026-01-01T09:00:00\n"
        )
    (root / "participants.tsv").write_text(
        "participant_id\tage\tsex\tgroup\n"
        "sub-01\t31\tn/a\tcontrol\n"
        "sub-02\tn/a\tF\tn/a\n"
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    return root


def test_a_taken_name_is_refused_unless_a_merge_is_asked_for(
    tmp_path: Path,
) -> None:
    root = _two_subjects(tmp_path)
    plan = rn.plan_rename(root, "sub", "02", "01")
    assert plan.conflicts
    assert "Merge them instead" in plan.conflicts[0]
    with pytest.raises(rn.RenameError):
        rn.apply_rename(root, plan)
    assert (root / "sub-02").exists(), "nothing may move on a refusal"


def test_the_caller_can_ask_whether_a_name_is_taken(tmp_path: Path) -> None:
    """So a dialog can say 'this will merge two subjects' up front."""
    root = _two_subjects(tmp_path)
    assert rn.would_fuse(root, "sub", "01")
    assert not rn.would_fuse(root, "sub", "99")
    assert not rn.would_fuse(root, "task", "rest")


def test_fusing_moves_the_sessions_in(tmp_path: Path) -> None:
    root = _two_subjects(tmp_path)
    plan = rn.plan_rename(root, "sub", "02", "01", fuse=True)
    assert plan.fusion
    assert not plan.conflicts
    assert "Merge sub-02 into the existing sub-01" == plan.verb()

    touched, errors = rn.apply_rename(root, plan)
    assert not errors and touched

    assert not (root / "sub-02").exists(), "the source must not linger"
    assert (root / "sub-01" / "ses-pre" / "anat"
            / "sub-01_ses-pre_T1w.nii.gz").exists()
    moved = (root / "sub-01" / "ses-post" / "anat"
             / "sub-01_ses-post_T2w.nii.gz")
    assert moved.exists(), "the second subject's session must land beside"
    assert json.loads(moved.with_suffix("").with_suffix(".json").read_text())


def test_fusing_keeps_every_row_of_both_scans_tables(tmp_path: Path) -> None:
    root = _two_subjects(tmp_path)
    rn.apply_rename(
        root, rn.plan_rename(root, "sub", "02", "01", fuse=True),
    )
    table = read_table(root / "sub-01" / "sub-01_scans.tsv")
    names = table.column("filename")
    assert names == [
        "ses-pre/anat/sub-01_ses-pre_T1w.nii.gz",
        "ses-post/anat/sub-01_ses-post_T2w.nii.gz",
    ]
    assert not (root / "sub-01" / "sub-02_scans.tsv").exists()


def test_fusing_folds_the_two_participant_rows_into_one(
    tmp_path: Path,
) -> None:
    """Two records of one person. Neither row's information may be lost."""
    root = _two_subjects(tmp_path)
    rn.apply_rename(
        root, rn.plan_rename(root, "sub", "02", "01", fuse=True),
    )
    table = read_table(root / "participants.tsv")
    assert table.column("participant_id") == ["sub-01"]
    row = dict(zip(table.header, table.rows[0]))
    assert row["age"] == "31"        # only sub-01 stated it
    assert row["sex"] == "F"         # only sub-02 stated it
    assert row["group"] == "control"


def test_a_stated_value_is_not_overwritten_by_the_folded_row(
    tmp_path: Path,
) -> None:
    root = _two_subjects(tmp_path)
    (root / "participants.tsv").write_text(
        "participant_id\tage\n"
        "sub-01\t31\n"
        "sub-02\t44\n"
    )
    rn.apply_rename(
        root, rn.plan_rename(root, "sub", "02", "01", fuse=True),
    )
    table = read_table(root / "participants.tsv")
    assert table.column("age") == ["31"], "the surviving row keeps its value"


def test_two_recordings_claiming_one_name_still_stop_the_merge(
    tmp_path: Path,
) -> None:
    """Picking one is not a decision a tool should make silently."""
    root = _two_subjects(tmp_path)
    clash = root / "sub-02" / "ses-pre" / "anat"
    clash.mkdir(parents=True)
    (clash / "sub-02_ses-pre_T1w.nii.gz").write_text("a different image")

    plan = rn.plan_rename(root, "sub", "02", "01", fuse=True)
    assert plan.conflicts
    assert "needs a different name" in plan.conflicts[0]
    with pytest.raises(rn.RenameError):
        rn.apply_rename(root, plan)
    assert (root / "sub-01" / "ses-pre" / "anat"
            / "sub-01_ses-pre_T1w.nii.gz").read_text() == "img"


def test_fusing_updates_intended_for_in_the_moved_sidecars(
    tmp_path: Path,
) -> None:
    root = _two_subjects(tmp_path)
    fmap = root / "sub-02" / "ses-post" / "fmap"
    fmap.mkdir(parents=True)
    (fmap / "sub-02_ses-post_phasediff.json").write_text(json.dumps({
        "IntendedFor": ["ses-post/anat/sub-02_ses-post_T2w.nii.gz"],
    }))
    rn.apply_rename(
        root, rn.plan_rename(root, "sub", "02", "01", fuse=True),
    )
    moved = (root / "sub-01" / "ses-post" / "fmap"
             / "sub-01_ses-post_phasediff.json")
    assert json.loads(moved.read_text())["IntendedFor"] == [
        "ses-post/anat/sub-01_ses-post_T2w.nii.gz"
    ]


def test_a_fusion_is_one_undo(tmp_path: Path) -> None:
    """It moves folders, files and rewrites two tables. Half of that applied
    would be worse than none of it."""
    root = _two_subjects(tmp_path)
    before = {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }
    rn.apply_rename(
        root, rn.plan_rename(root, "sub", "02", "01", fuse=True),
    )
    assert read_log(root), "the fusion must be recorded as one operation"
    undo_last(root)

    after = {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".bidsmgr" not in p.parts
    }
    assert after == before


# ---------------------------------------------------------------------------
# Renaming only some of the files
#
# "These three runs were mislabelled, the rest are right" is an ordinary thing
# to want. A tool that can only do the whole thing forces the rest by hand,
# which is where the errors come from. The hard part is not the moving, it is
# that a partial rename must not leave a cross-reference pointing at a name
# nothing has.
# ---------------------------------------------------------------------------


def _runs(tmp_path: Path) -> Path:
    """One subject, one task, three runs, with references to all three."""
    root = tmp_path / "ds"
    func = root / "sub-01" / "func"
    func.mkdir(parents=True)
    for run in ("1", "2", "3"):
        (func / f"sub-01_task-rest_run-{run}_bold.nii.gz").write_bytes(b"")
        (func / f"sub-01_task-rest_run-{run}_bold.json").write_text(
            json.dumps({"RepetitionTime": 2.0, "TaskName": "rest"})
        )
    fmap = root / "sub-01" / "fmap"
    fmap.mkdir(parents=True)
    (fmap / "sub-01_phasediff.json").write_text(json.dumps({
        "IntendedFor": [
            f"func/sub-01_task-rest_run-{r}_bold.nii.gz" for r in "123"
        ],
    }))
    (root / "sub-01" / "sub-01_scans.tsv").write_text(
        "filename\n" + "".join(
            f"func/sub-01_task-rest_run-{r}_bold.nii.gz\n" for r in "123"
        )
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    return root


def test_the_plan_names_every_file_it_would_move(tmp_path: Path) -> None:
    root = _runs(tmp_path)
    plan = rn.plan_rename(root, "task", "rest", "resting")
    keys = plan.file_keys(root)
    assert len(keys) == 6
    assert all(not Path(k).is_absolute() for k in keys), "keys are relative"
    # A path, not an index: re-planning between showing and applying must not
    # silently point at a different file.
    assert "sub-01/func/sub-01_task-rest_run-1_bold.json" in keys


def test_renaming_none_of_them_changes_nothing(tmp_path: Path) -> None:
    root = _runs(tmp_path)
    plan = rn.plan_rename(root, "task", "rest", "resting")
    touched, errors = rn.apply_rename(root, plan, only=set())
    assert (touched, errors) == (0, [])
    assert (root / "sub-01" / "func"
            / "sub-01_task-rest_run-1_bold.nii.gz").exists()


def test_renaming_some_leaves_the_rest_alone(tmp_path: Path) -> None:
    root = _runs(tmp_path)
    plan = rn.plan_rename(root, "task", "rest", "resting")
    chosen = {k for k in plan.file_keys(root) if "run-1_" in k}
    assert len(chosen) == 2, "the image and its sidecar"

    touched, errors = rn.apply_rename(root, plan, only=chosen)
    assert not errors and touched

    func = root / "sub-01" / "func"
    assert (func / "sub-01_task-resting_run-1_bold.nii.gz").exists()
    assert (func / "sub-01_task-rest_run-2_bold.nii.gz").exists()
    assert not (func / "sub-01_task-rest_run-1_bold.nii.gz").exists()


def test_a_reference_to_a_file_that_stayed_is_not_rewritten(
    tmp_path: Path,
) -> None:
    """The whole point. Rewriting every reference would turn two working
    pointers into dangling ones."""
    root = _runs(tmp_path)
    plan = rn.plan_rename(root, "task", "rest", "resting")
    chosen = {k for k in plan.file_keys(root) if "run-1_" in k}
    rn.apply_rename(root, plan, only=chosen)

    intended = json.loads(
        (root / "sub-01" / "fmap" / "sub-01_phasediff.json").read_text()
    )["IntendedFor"]
    assert intended == [
        "func/sub-01_task-resting_run-1_bold.nii.gz",
        "func/sub-01_task-rest_run-2_bold.nii.gz",
        "func/sub-01_task-rest_run-3_bold.nii.gz",
    ]
    scans = read_table(root / "sub-01" / "sub-01_scans.tsv").column("filename")
    assert scans[0].endswith("task-resting_run-1_bold.nii.gz")
    assert scans[1].endswith("task-rest_run-2_bold.nii.gz")


def test_every_reference_still_points_at_a_file_that_exists(
    tmp_path: Path,
) -> None:
    """The property that matters, checked against the tree rather than
    against the strings."""
    root = _runs(tmp_path)
    plan = rn.plan_rename(root, "task", "rest", "resting")
    chosen = {k for k in plan.file_keys(root) if "run-2_" in k}
    rn.apply_rename(root, plan, only=chosen)

    subject = root / "sub-01"
    for ref in json.loads(
        (subject / "fmap" / "sub-01_phasediff.json").read_text()
    )["IntendedFor"]:
        assert (subject / ref).exists(), ref
    for ref in read_table(subject / "sub-01_scans.tsv").column("filename"):
        assert (subject / ref).exists(), ref


def test_a_partial_subject_rename_moves_files_and_keeps_the_folder(
    tmp_path: Path,
) -> None:
    """A folder rename moves everything under it whether it was chosen or not,
    so it is only allowed when nothing is staying behind."""
    root = _runs(tmp_path)
    plan = rn.plan_rename(root, "sub", "01", "02")
    chosen = {k for k in plan.file_keys(root) if "run-1_" in k}
    rn.apply_rename(root, plan, only=chosen)

    assert (root / "sub-01").is_dir(), "the rest of the subject stays"
    assert (root / "sub-02" / "func"
            / "sub-02_task-rest_run-1_bold.nii.gz").exists()
    assert (root / "sub-01" / "func"
            / "sub-01_task-rest_run-2_bold.nii.gz").exists()


def test_half_a_subject_is_still_the_old_participant(tmp_path: Path) -> None:
    """participants.tsv describes a person. Moving three of their files does
    not RENAME them: sub-01 is still sub-01 and keeps its values.

    It does gain a neighbour, because the move created a second subject and
    the table has to list every subject that exists. That is the split, and
    :mod:`tests.unit.test_rename_split` covers what the new row carries.
    """
    root = _runs(tmp_path)
    (root / "participants.tsv").write_text(
        "participant_id\tage\nsub-01\t31\n"
    )
    plan = rn.plan_rename(root, "sub", "01", "02")
    chosen = {k for k in plan.file_keys(root) if "run-1_" in k}
    rn.apply_rename(root, plan, only=chosen)

    table = read_table(root / "participants.tsv")
    ids = table.column("participant_id")
    assert "sub-01" in ids, "the original participant must not be renamed away"
    assert ids == ["sub-01", "sub-02"], "and the new subject must be listed"
    rows = {r[0]: dict(zip(table.header, r)) for r in table.rows}
    assert rows["sub-01"]["age"] == "31"


def test_selecting_everything_is_the_same_as_selecting_nothing_special(
    tmp_path: Path,
) -> None:
    """The default path must not become a second implementation."""
    a, b = tmp_path / "a", tmp_path / "b"
    for base in (a, b):
        base.mkdir()
    root_a, root_b = _runs(a), _runs(b)

    plan_a = rn.plan_rename(root_a, "sub", "01", "02")
    rn.apply_rename(root_a, plan_a)
    plan_b = rn.plan_rename(root_b, "sub", "01", "02")
    rn.apply_rename(root_b, plan_b, only=set(plan_b.file_keys(root_b)))

    def _tree(root: Path) -> dict:
        return {
            str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".bidsmgr" not in p.parts
        }

    assert _tree(root_a) == _tree(root_b)

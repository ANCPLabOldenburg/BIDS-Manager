"""Every BIDS fact the Editor acts on must come from the active schema.

The rule the whole package is built on (CLAUDE.md guard 8): which entities
exist, what their values may look like, which datatypes carry which companion
file. All of those are facts about the standard, and a hand-kept copy of one
is a copy that silently goes stale.

These tests are written against the SCHEMA rather than against a list, so they
keep holding when the schema version changes. Where a specific value is
asserted it is one the hardcoded version got wrong, kept as a regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr import schema as sc
from bidsmgr.editor import rename as rn


# ---------------------------------------------------------------------------
# The entity set
# ---------------------------------------------------------------------------


def test_entity_keys_are_the_schema_s_own_short_names() -> None:
    keys = sc.entity_keys()
    assert keys[0] == "sub", "canonical filename order, subject first"
    assert "ses" in keys and "task" in keys and "acq" in keys
    assert len(keys) == len(sc.entity_order())
    assert len(set(keys)) == len(keys), "no key may appear twice"


def test_an_entity_can_be_looked_up_by_the_name_a_filename_uses() -> None:
    """The schema keys entities by their long name; filenames use the short
    one. Translating in every caller is how the two drift apart."""
    assert sc.entity_key_info("sub").display_name == "Subject"
    assert sc.entity_key_info("acq").display_name == "Acquisition"
    with pytest.raises(KeyError):
        sc.entity_key_info("notanentity")


def test_the_dialog_offers_every_entity_the_schema_defines() -> None:
    from bidsmgr.gui.rename_entity_dialog import entity_choices

    offered = [key for key, _label in entity_choices()]
    assert offered == list(sc.entity_keys())
    # The hand-kept list had nine of them.
    assert len(offered) > 9


def test_the_tree_labels_entities_from_the_schema() -> None:
    from bidsmgr.gui.widgets.bids_tree_pane import _renameable_entities

    got = dict(
        (key, label)
        for key, _value, label in _renameable_entities(
            Path("sub-01_ses-pre_task-rest_run-02_bold.nii.gz")
        )
    )
    assert got["sub"] == sc.entity_key_info("sub").display_name.lower()
    assert got["run"] == sc.entity_key_info("run").display_name.lower()


# ---------------------------------------------------------------------------
# What a value may be. The hardcoded rule was wrong in BOTH directions.
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_acq-highres_T1w.nii.gz").write_bytes(b"")
    func = root / "sub-01" / "func"
    func.mkdir(parents=True)
    (func / "sub-01_task-rest_run-01_bold.nii.gz").write_bytes(b"")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    return root


def test_a_label_may_contain_a_plus(dataset: Path) -> None:
    """The schema's label format is ``[0-9a-zA-Z+]+``. ``acq-6p+s2`` is valid
    BIDS and the hardcoded ``[A-Za-z0-9]+`` refused it."""
    assert sc.entity_key_info("acq").format.name == "label"
    plan = rn.plan_rename(dataset, "acq", "highres", "6p+s2")
    assert plan.file_moves
    assert plan.file_moves[0][1].name == "sub-01_acq-6p+s2_T1w.nii.gz"


def test_an_index_entity_refuses_letters(dataset: Path) -> None:
    """``run`` is an index, so ``run-abc`` is not valid, and the hardcoded
    rule accepted it."""
    assert sc.entity_key_info("run").format.name == "index"
    with pytest.raises(rn.RenameError) as exc:
        rn.plan_rename(dataset, "run", "01", "abc")
    assert "digits" in str(exc.value)


def test_the_refusal_says_what_the_standard_asks_for(dataset: Path) -> None:
    with pytest.raises(rn.RenameError) as exc:
        rn.plan_rename(dataset, "task", "rest", "two words")
    message = str(exc.value)
    assert "task" in message.lower()
    assert "label" in message


# ---------------------------------------------------------------------------
# Which folders are entities, and which are out of scope
# ---------------------------------------------------------------------------


def test_folder_entities_come_from_the_schema_s_directory_rules() -> None:
    assert rn.folder_entities() == sc.directory_entity_keys()
    assert "sub" in rn.folder_entities()
    assert "ses" in rn.folder_entities()
    assert "task" not in rn.folder_entities(), "a task is not a folder"


def test_out_of_scope_directories_are_the_schema_s_opaque_ones() -> None:
    skipped = rn._skip_top_level()
    for name in sc.opaque_directories():
        assert name in skipped, name
    # Plus the tool's own state, which is not the standard's business.
    assert ".bidsmgr" in skipped


def test_a_rename_does_not_reach_into_derivatives(dataset: Path) -> None:
    deriv = dataset / "derivatives" / "fmriprep" / "sub-01" / "anat"
    deriv.mkdir(parents=True)
    (deriv / "sub-01_acq-highres_T1w.nii.gz").write_bytes(b"")
    plan = rn.plan_rename(dataset, "acq", "highres", "hires")
    assert all("derivatives" not in str(src) for src, _ in plan.file_moves)


# ---------------------------------------------------------------------------
# Which datatypes carry which companion. The hardcoded sets dropped three.
# ---------------------------------------------------------------------------


def test_companion_datatypes_come_from_the_schema() -> None:
    from bidsmgr.fixups.associations import _datatypes_with

    assert _datatypes_with("events") == frozenset(
        sc.datatypes_with_suffix("events")
    )
    assert _datatypes_with("channels") == frozenset(
        sc.datatypes_with_suffix("channels")
    )


@pytest.mark.parametrize("datatype", ["emg", "motion", "mrs"])
def test_the_modalities_the_hardcoded_list_forgot_are_covered(
    datatype: str,
) -> None:
    """These three were missing from the hand-kept events set, so a whole
    modality was skipped by a check that reported itself as complete."""
    from bidsmgr.fixups.associations import _datatypes_with

    assert datatype in _datatypes_with("events")


@pytest.mark.parametrize("datatype", ["emg", "motion"])
def test_channels_covers_the_modalities_the_hardcoded_list_forgot(
    datatype: str,
) -> None:
    from bidsmgr.fixups.associations import _datatypes_with

    assert datatype in _datatypes_with("channels")


def test_a_datatype_with_no_events_is_not_offered_one() -> None:
    """The derivation must not become "everything"."""
    from bidsmgr.fixups.associations import _datatypes_with

    assert "anat" not in _datatypes_with("events")
    assert "dwi" not in _datatypes_with("channels")


def test_a_motion_recording_is_offered_its_companions(tmp_path: Path) -> None:
    """End to end through the fix-up, on a datatype the old list dropped."""
    from bidsmgr.fixups import associations as assoc

    root = tmp_path / "ds"
    motion = root / "sub-01" / "motion"
    motion.mkdir(parents=True)
    (motion / "sub-01_task-walk_tracksys-imu_motion.tsv").write_text("x\n")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    kinds = {m.kind for m in assoc.find_missing(root)}
    assert "events" in kinds
    assert "channels" in kinds

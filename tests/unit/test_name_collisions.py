"""Two recordings must never claim the same BIDS name.

When they do, the second overwrites the first and a scan disappears from the
dataset without anything being reported. Nothing checked, until three ECAT
phantoms all resolved to ``sub-014_pet``.

Three layers are tested here: numbering the clashes BIDS has an answer for,
reporting the ones it does not, and refusing to convert while any remain.

The split matters. A run is the right answer for two genuine repeats and the
wrong one for two different tasks, and nothing in the file can tell them apart.
So a run is assigned only where the schema allows it and nobody has already
stated one; anything else is left exactly as it is and shown in red.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bidsmgr.inventory.name_collisions import (
    assign_runs,
    count_collisions,
    describe_collisions,
    find_collisions,
)


def _row(**overrides) -> dict:
    base = {
        "dataset": "study",
        "include": "1",
        "proposed_datatype": "pet",
        "bids_guess_suffix": "pet",
        "proposed_basename": "sub-001_pet",
        "entities": json.dumps({"subject": "001"}, sort_keys=True),
        "proposed_issues": "",
        "source_file": "a.v",
        "run": "",
    }
    base.update(overrides)
    return base


def _frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def test_distinct_names_do_not_collide() -> None:
    df = _frame(_row(), _row(proposed_basename="sub-002_pet", source_file="b.v"))
    assert find_collisions(df) == {}


def test_the_same_name_twice_is_a_collision() -> None:
    df = _frame(_row(source_file="a.v"), _row(source_file="b.v"))
    assert len(find_collisions(df)) == 1


def test_an_excluded_row_cannot_collide() -> None:
    """It is never written, so it cannot overwrite anything."""
    df = _frame(_row(source_file="a.v"), _row(source_file="b.v", include="0"))
    assert find_collisions(df) == {}


def test_the_same_name_in_a_different_dataset_is_fine() -> None:
    """Sibling datasets are separate trees; only the full destination counts."""
    df = _frame(_row(), _row(dataset="other", source_file="b.v"))
    assert find_collisions(df) == {}


def test_a_row_with_no_name_yet_is_not_a_collision() -> None:
    """Not having been named is a different problem from being named twice."""
    df = _frame(_row(proposed_basename=""), _row(proposed_basename="", source_file="b.v"))
    assert find_collisions(df) == {}


# --------------------------------------------------------------------------
# numbering what BIDS has an answer for
# --------------------------------------------------------------------------


def test_clashing_rows_become_runs() -> None:
    """Three ECAT phantoms that all resolved to one name."""
    df = _frame(
        _row(source_file="c.v"), _row(source_file="a.v"), _row(source_file="b.v"),
    )
    assert assign_runs(df) == 3
    assert sorted(json.loads(e)["run"] for e in df["entities"]) == ["1", "2", "3"]
    assert find_collisions(df) == {}


def test_the_basename_is_rebuilt_to_carry_the_run() -> None:
    """A run written only into the entities JSON would rename no file."""
    df = _frame(_row(source_file="a.v"), _row(source_file="b.v"))
    assign_runs(df)
    assert all("run-" in name for name in df["proposed_basename"])


def test_numbering_is_deterministic() -> None:
    """A re-scan must not shuffle filenames under the user."""
    first = _frame(_row(source_file="c.v"), _row(source_file="a.v"))
    second = _frame(_row(source_file="a.v"), _row(source_file="c.v"))
    assign_runs(first)
    assign_runs(second)

    def by_source(df):
        return {
            r["source_file"]: json.loads(r["entities"])["run"]
            for _, r in df.iterrows()
        }

    assert by_source(first) == by_source(second)


def test_the_earliest_acquisition_is_run_one() -> None:
    df = _frame(
        _row(source_file="late.v", acq_time="2024-01-02T10:00:00"),
        _row(source_file="early.v", acq_time="2024-01-01T10:00:00"),
    )
    assign_runs(df)
    got = {r["source_file"]: json.loads(r["entities"])["run"] for _, r in df.iterrows()}
    assert got == {"early.v": "1", "late.v": "2"}


def test_nothing_is_numbered_when_nothing_clashes() -> None:
    df = _frame(_row(), _row(proposed_basename="sub-002_pet", source_file="b.v"))
    before = df.copy()
    assert assign_runs(df) == 0
    pd.testing.assert_frame_equal(df, before)


# --------------------------------------------------------------------------
# reporting what it cannot number, and never storing that
# --------------------------------------------------------------------------


def test_a_stated_run_is_never_overwritten() -> None:
    """Somebody set these by hand and they still clash.

    Renaming over a stated answer is the thing this module exists to prevent,
    so it is left alone and reported instead.
    """
    entities = json.dumps({"subject": "001", "run": "1"}, sort_keys=True)
    df = _frame(
        _row(source_file="a.v", entities=entities),
        _row(source_file="b.v", entities=entities),
    )
    assert assign_runs(df) == 0
    assert all(json.loads(e)["run"] == "1" for e in df["entities"])
    assert count_collisions(df) == 2


def test_a_datatype_that_cannot_carry_a_run_is_left_alone() -> None:
    """The schema decides which files may have a run, not us."""
    df = _frame(
        _row(proposed_datatype="anat", bids_guess_suffix="TB1TFL",
             proposed_basename="sub-001_TB1TFL", source_file="a.nii"),
        _row(proposed_datatype="anat", bids_guess_suffix="TB1TFL",
             proposed_basename="sub-001_TB1TFL", source_file="b.nii"),
    )
    assign_runs(df)
    assert count_collisions(df) == 2
    assert all("run-" not in n for n in df["proposed_basename"])


def test_every_clashing_row_is_counted() -> None:
    df = _frame(
        _row(source_file="a.v"), _row(source_file="b.v"), _row(source_file="c.v"),
    )
    assert count_collisions(df) == 3


def test_nothing_is_written_into_the_inventory() -> None:
    """The bug that made the red name look stuck.

    A note stamped into a cell is stale the moment an entity is edited, so the
    row stayed red after the user had already fixed it. The clash is a property
    of the table as it stands and is derived from it, never stored.
    """
    df = _frame(_row(source_file="a.v"), _row(source_file="b.v"))
    before = df.copy()
    count_collisions(df)
    pd.testing.assert_frame_equal(df, before)


def test_fixing_one_row_clears_the_clash_for_both() -> None:
    """Because it is recomputed rather than remembered."""
    df = _frame(_row(source_file="a.v"), _row(source_file="b.v"))
    assert count_collisions(df) == 2

    df.at[1, "proposed_basename"] = "sub-001_task-video_eeg"
    assert count_collisions(df) == 0
    assert find_collisions(df) == {}


# --------------------------------------------------------------------------
# refusing to write
# --------------------------------------------------------------------------


def test_a_clean_inventory_has_nothing_to_report() -> None:
    df = _frame(_row(), _row(proposed_basename="sub-002_pet", source_file="b.v"))
    assert describe_collisions(df) is None


def test_the_refusal_names_both_sources() -> None:
    """A hand-edited TSV can collide after the scan resolved everything.

    The message has to be actionable: which name, and which files.
    """
    df = _frame(_row(source_file="scan_a.v"), _row(source_file="scan_b.v"))
    message = describe_collisions(df)

    assert message is not None
    assert "sub-001_pet" in message
    assert "scan_a.v" in message
    assert "scan_b.v" in message


@pytest.mark.parametrize("include", ["0", "False", "false"])
def test_excluding_one_side_resolves_the_refusal(include: str) -> None:
    df = _frame(_row(source_file="a.v"), _row(source_file="b.v", include=include))
    assert describe_collisions(df) is None


# --------------------------------------------------------------------------
# the EEG-only scan path
# --------------------------------------------------------------------------


def _eeg_row(**overrides) -> dict:
    return _row(
        proposed_datatype="eeg",
        bids_guess_suffix="eeg",
        proposed_basename="sub-001_task-CLV002_eeg",
        entities=json.dumps(
            {"subject": "001", "task": "CLV002"}, sort_keys=True,
        ),
        modality="eeg",
        **overrides,
    )


def test_an_eeg_only_inventory_is_checked_too() -> None:
    """The reported bug.

    A workshop tree where every subject exports a rest and a video recording
    gives two rows with one name per subject. The check existed, but it ran on
    the MRI frame before the EEG rows were concatenated in, so a dataset with
    no MRI in it was never checked at all.
    """
    from bidsmgr.cli.scan import _finish_unified_frame

    df = _frame(
        _eeg_row(source_file="sub-001/rest/CLV002.set"),
        _eeg_row(source_file="sub-001/video/CLV002.set"),
    )
    _finish_unified_frame(df, [])

    # eeg/eeg may carry a run, so the scan resolves this one for the user.
    assert count_collisions(df) == 0
    assert all("run-" in n for n in df["proposed_basename"])


def test_an_excluded_row_is_not_a_clash() -> None:
    """Exclusions run first, because an excluded row is never written."""
    from bidsmgr.cli.scan import _finish_unified_frame

    df = _frame(
        _eeg_row(source_file="a.set"),
        _eeg_row(source_file="b.set", include="0"),
    )
    _finish_unified_frame(df, [])

    assert count_collisions(df) == 0


def test_both_scan_paths_share_one_tail() -> None:
    """A structural guard, because duplication is what caused the bug.

    ``run_scan`` writes the inventory from two places, one for a tree with MRI
    in it and one for a tree without. They had drifted: a check added to the
    second did nothing for the first. Keeping the final passes in a single
    helper is what stops that happening again, so the helper must stay the only
    caller of the passes it owns.
    """
    import inspect

    from bidsmgr.cli import scan as scan_mod

    source = inspect.getsource(scan_mod)
    body = inspect.getsource(scan_mod._finish_unified_frame)

    for pass_name in ("_apply_user_exclusions(", "assign_runs("):
        outside = source.count(pass_name) - body.count(pass_name)
        # One remaining mention is the definition or import, never a call.
        assert outside <= 1, (
            f"{pass_name} is called outside _finish_unified_frame; both scan "
            "write paths must share the same final passes"
        )

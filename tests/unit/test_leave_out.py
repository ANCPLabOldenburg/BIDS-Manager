"""Entities a user leaves out of the names a scan proposes.

Some labs never want ``acq-`` in a filename, or want ``run-`` only where there
are two runs. Leaving an entity out must never cost a recording: it stays where
the standard requires it, and where it is what tells two recordings apart
(otherwise one would overwrite the other at conversion).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bidsmgr.cli import scan as scan_cli
from bidsmgr.inventory.leave_out import apply_leave_out, offered, plan_leave_out
from bidsmgr.inventory.rebuild import rebuild_from_entities


def _row(entities: dict, datatype: str, suffix: str, *, include: int = 1,
         dataset: str = "ds", time: str = "") -> dict:
    return {"entities": json.dumps({"subject": "01", **entities}),
            "datatype": datatype, "bids_guess_suffix": suffix,
            "bids_name": "", "bids_path": "", "include": include, "dataset": dataset,
            "session": "", "task": "", "run": "", "acq_time": time,
            "source_file": "", "issues": ""}


def _frame(*rows: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    rebuild_from_entities(df, in_place=True)
    return df


def _names(df: pd.DataFrame) -> list[str]:
    return df["bids_name"].tolist()


def _leave_out(df: pd.DataFrame, *keys: str):
    plan = plan_leave_out(df, keys)
    apply_leave_out(df, plan)
    return plan


def test_what_can_be_left_out_comes_from_the_schema() -> None:
    keys = offered()
    assert {"acq", "rec", "ce", "dir", "run", "echo", "part", "chunk"} <= set(keys)
    assert not {"sub", "ses", "task"} & set(keys), "these name the recording or its folder"
    assert keys.index("acq") < keys.index("run"), "in filename order"


def test_an_entity_is_left_out_of_every_name_it_does_not_hold_apart() -> None:
    df = _frame(_row({"acquisition": "mprage"}, "anat", "T1w"),
                _row({"acquisition": "flair3d"}, "anat", "FLAIR"),
                _row({"task": "rest", "acquisition": "mb4"}, "func", "bold"))
    plan = _leave_out(df, "acq")
    assert _names(df) == ["sub-01_T1w", "sub-01_FLAIR", "sub-01_task-rest_bold"]
    assert plan.removed == {"acq": 3}
    assert json.loads(df.at[0, "entities"]) == {"subject": "01"}, "the entities say so too"


def test_two_acquisitions_are_never_made_into_one_name_or_into_runs() -> None:
    df = _frame(_row({"acquisition": "mprage"}, "anat", "T1w"),
                _row({"acquisition": "highres"}, "anat", "T1w"),
                _row({"acquisition": "x"}, "anat", "FLAIR"))
    plan = _leave_out(df, "acq")
    assert _names(df) == ["sub-01_acq-mprage_T1w", "sub-01_acq-highres_T1w", "sub-01_FLAIR"]
    assert plan.kept_apart == {"acq": 2}
    assert "kept it on 2 recordings where it tells two apart" in plan.summary()


def test_run_stays_only_where_there_are_two() -> None:
    df = _frame(_row({"task": "rest", "run": "1"}, "func", "bold"),
                _row({"task": "faces", "run": "1"}, "func", "bold"),
                _row({"task": "faces", "run": "2"}, "func", "bold"))
    _leave_out(df, "run")
    assert _names(df) == ["sub-01_task-rest_bold", "sub-01_task-faces_run-1_bold",
                          "sub-01_task-faces_run-2_bold"]
    assert df.at[0, "run"] == "", "the run column follows the name"


def test_a_required_entity_stays() -> None:
    df = _frame(_row({"echo": "1"}, "anat", "MEGRE"),
                _row({"echo": "1", "task": "rest"}, "func", "bold"))
    plan = _leave_out(df, "echo")
    assert _names(df) == ["sub-01_echo-1_MEGRE", "sub-01_task-rest_bold"]
    assert plan.kept_required == {"echo": 1}
    assert plan.summary() == "Left echo out of 1 name; kept it on 1 recording that requires it."


def test_only_kept_is_said_plainly() -> None:
    df = _frame(_row({"echo": "1"}, "anat", "MEGRE"))
    plan = plan_leave_out(df, ["echo"])
    assert not plan
    assert plan.summary() == "Kept echo on 1 recording that requires it."


def test_a_clash_that_was_there_before_is_not_held_against_it() -> None:
    """Two rows that already claim one name clash either way: the run pass
    and the red row state deal with them, not this."""
    df = _frame(_row({"acquisition": "a"}, "anat", "T1w"),
                _row({"acquisition": "a"}, "anat", "T1w"))
    plan = _leave_out(df, "acq")
    assert _names(df) == ["sub-01_T1w", "sub-01_T1w"]
    assert plan.kept_apart == {}


def test_a_left_out_row_does_not_hold_an_entity_on_another() -> None:
    df = _frame(_row({"acquisition": "a"}, "anat", "T1w"),
                _row({"acquisition": "b"}, "anat", "T1w", include=0))
    _leave_out(df, "acq")
    assert _names(df)[0] == "sub-01_T1w", "only included recordings can collide"


def test_another_dataset_is_another_place() -> None:
    df = _frame(_row({"acquisition": "a"}, "anat", "T1w", dataset="one"),
                _row({"acquisition": "b"}, "anat", "T1w", dataset="two"))
    _leave_out(df, "acq")
    assert _names(df) == ["sub-01_T1w", "sub-01_T1w"]


def test_entities_are_judged_one_after_another() -> None:
    """A row that keeps acq because it tells two T1w apart still loses rec."""
    df = _frame(_row({"acquisition": "a", "reconstruction": "norm"}, "anat", "T1w"),
                _row({"acquisition": "b", "reconstruction": "norm"}, "anat", "T1w"))
    _leave_out(df, "acq", "rec")
    assert _names(df) == ["sub-01_acq-a_T1w", "sub-01_acq-b_T1w"]


def test_nothing_to_do_changes_nothing() -> None:
    df = _frame(_row({}, "anat", "T1w"))
    before = df.copy()
    plan = plan_leave_out(df, ["acq"])
    assert not plan and plan.summary() == ""
    assert apply_leave_out(df, plan) == 0
    pd.testing.assert_frame_equal(df, before)


def test_a_key_that_cannot_be_left_out_is_ignored() -> None:
    df = _frame(_row({"task": "rest"}, "func", "bold"))
    assert not plan_leave_out(df, ["task", "sub"])


def test_the_scan_leaves_them_out_before_numbering_runs(capsys) -> None:
    """In the scan's last pass, so the run pass sees the final names."""
    df = _frame(_row({"acquisition": "a"}, "anat", "T1w"),
                _row({"acquisition": "b"}, "anat", "T1w"),
                _row({"acquisition": "x", "task": "rest"}, "func", "bold"))
    scan_cli._finish_unified_frame(df, None, None, ["acq"])
    assert _names(df) == ["sub-01_acq-a_T1w", "sub-01_acq-b_T1w", "sub-01_task-rest_bold"]
    assert "Left acq out of 1 name; kept it on 2 recordings" in capsys.readouterr().out


def test_the_cli_takes_several_and_refuses_what_names_the_recording(
        tmp_path, monkeypatch) -> None:
    calls = {}
    monkeypatch.setattr(scan_cli, "run_scan",
                        lambda *a, **kw: calls.update(kw) or pd.DataFrame())
    raw = tmp_path / "raw"
    raw.mkdir()
    out = str(tmp_path / "inv.tsv")
    assert scan_cli.main([str(raw), out, "--no-log-file",
                          "--drop-entity", "acq,rec", "--drop-entity", "run"]) == 0
    assert calls["drop_entities"] == ["acq", "rec", "run"]
    with pytest.raises(SystemExit):
        scan_cli.main([str(raw), out, "--drop-entity", "task"])

"""Unit tests for ``bidsmgr.inventory.rebuild`` and the ``bidsmgr-rebuild`` CLI.

The rebuild engine reconciles the inventory TSV's ``entities`` JSON
column with the derived display cells (``proposed_basename``, mirror
columns). Two directions:

* ``--from entities`` (default): JSON is the source of truth.
* ``--from columns``: cells are the source; JSON is regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.inventory.rebuild import (
    rebuild_from_columns,
    rebuild_from_entities,
)


def _make_row(
    *,
    entities: dict | None = None,
    proposed_basename: str = "",
    proposed_datatype: str = "",
    bids_guess_suffix: str = "",
    BIDS_name: str = "sub-001",
    session: str = "",
    task: str = "",
    run: str = "",
) -> dict:
    return {
        "entities": json.dumps(entities, sort_keys=True) if entities else "",
        "proposed_basename": proposed_basename,
        "Proposed BIDS name": proposed_basename,
        "proposed_datatype": proposed_datatype,
        "bids_guess_suffix": bids_guess_suffix,
        "BIDS_name": BIDS_name,
        "session": session,
        "task": task,
        "run": run,
    }


# ---------------------------------------------------------------------------
# rebuild_from_entities (default direction)
# ---------------------------------------------------------------------------


class TestRebuildFromEntities:
    def test_no_entities_column_is_no_op(self) -> None:
        df = pd.DataFrame([{"foo": "bar"}])
        out, report = rebuild_from_entities(df)
        assert out.equals(df)
        assert report.rows_updated == 0
        assert any("no 'entities' column" in w for w in report.warnings)

    def test_blank_entities_skips_row(self) -> None:
        df = pd.DataFrame([_make_row()])  # entities=""
        out, report = rebuild_from_entities(df)
        assert report.rows_updated == 0
        assert report.basename_changes == 0

    def test_malformed_json_is_warned_not_crashed(self) -> None:
        df = pd.DataFrame([{
            "entities": "{not valid json",
            "proposed_basename": "x",
            "proposed_datatype": "anat",
            "bids_guess_suffix": "T1w",
            "BIDS_name": "sub-001", "session": "", "task": "", "run": "",
        }])
        out, report = rebuild_from_entities(df)
        assert report.rows_updated == 0
        assert any("cannot parse entities JSON" in w for w in report.warnings)

    def test_basename_rebuilt_from_edited_entities(self) -> None:
        """User changed ``task`` from "restt" to "rest" in the JSON →
        ``proposed_basename`` regenerated."""
        df = pd.DataFrame([_make_row(
            entities={"subject": "001", "task": "rest"},
            proposed_basename="sub-001_task-restt_bold",   # the stale typo
            proposed_datatype="func", bids_guess_suffix="bold",
        )])
        out, report = rebuild_from_entities(df)
        assert report.basename_changes == 1
        assert out.iloc[0]["proposed_basename"] == "sub-001_task-rest_bold"
        assert out.iloc[0]["Proposed BIDS name"] == "sub-001_task-rest_bold"

    def test_mirror_cells_rebuilt(self) -> None:
        df = pd.DataFrame([_make_row(
            entities={"subject": "001", "session": "post", "task": "rest", "run": "2"},
            proposed_basename="sub-001_ses-post_task-rest_run-2_bold",
            proposed_datatype="func", bids_guess_suffix="bold",
            session="ses-pre",   # stale
            task="restt",        # stale
            run="1",             # stale
        )])
        out, report = rebuild_from_entities(df)
        # All three mirror cells get refreshed.
        assert out.iloc[0]["session"] == "ses-post"
        assert out.iloc[0]["task"] == "rest"
        assert out.iloc[0]["run"] == "2"
        # Three mirror changes recorded.
        assert report.mirror_changes == 3

    def test_no_change_when_already_consistent(self) -> None:
        df = pd.DataFrame([_make_row(
            entities={"subject": "001", "task": "rest"},
            proposed_basename="sub-001_task-rest_bold",
            proposed_datatype="func", bids_guess_suffix="bold",
            task="rest",
        )])
        out, report = rebuild_from_entities(df)
        assert report.rows_updated == 0
        assert report.basename_changes == 0


# ---------------------------------------------------------------------------
# rebuild_from_columns (reverse direction)
# ---------------------------------------------------------------------------


class TestRebuildFromColumns:
    def test_blank_entities_built_from_cells(self) -> None:
        """User had no entities JSON; we synthesise it from cells."""
        df = pd.DataFrame([_make_row(
            entities=None,   # blank cell
            BIDS_name="sub-001",
            session="ses-pre",
            task="rest",
            run="1",
            proposed_datatype="func", bids_guess_suffix="bold",
        )])
        out, report = rebuild_from_columns(df)
        result = json.loads(out.iloc[0]["entities"])
        assert result == {
            "subject": "001",
            "session": "pre",
            "task": "rest",
            "run": "1",
        }

    def test_existing_json_updated_from_edited_cells(self) -> None:
        """User edited the ``task`` cell in the spreadsheet; rebuild
        --from columns syncs the JSON."""
        df = pd.DataFrame([_make_row(
            entities={"subject": "001", "task": "restt"},
            proposed_basename="sub-001_task-restt_bold",
            proposed_datatype="func", bids_guess_suffix="bold",
            BIDS_name="sub-001",
            task="rest",   # user fixed the typo here
        )])
        out, report = rebuild_from_columns(df)
        result = json.loads(out.iloc[0]["entities"])
        assert result["task"] == "rest"
        # And the basename was refreshed too (rebuild_from_entities runs
        # as a secondary pass at the end).
        assert out.iloc[0]["proposed_basename"] == "sub-001_task-rest_bold"

    def test_empty_cell_removes_entity(self) -> None:
        """Clearing the ``run`` cell drops ``run`` from the JSON."""
        df = pd.DataFrame([_make_row(
            entities={"subject": "001", "task": "rest", "run": "2"},
            proposed_basename="sub-001_task-rest_run-2_bold",
            proposed_datatype="func", bids_guess_suffix="bold",
            BIDS_name="sub-001",
            task="rest",
            run="",   # cleared
        )])
        out, report = rebuild_from_columns(df)
        result = json.loads(out.iloc[0]["entities"])
        assert "run" not in result

    def test_malformed_json_repaired(self) -> None:
        df = pd.DataFrame([{
            "entities": "{garbage",
            "proposed_basename": "",
            "proposed_datatype": "anat", "bids_guess_suffix": "T1w",
            "BIDS_name": "sub-001",
            "session": "", "task": "T1", "run": "",
            "Proposed BIDS name": "",
        }])
        out, report = rebuild_from_columns(df)
        # JSON was rebuilt from cells.
        result = json.loads(out.iloc[0]["entities"])
        assert result["subject"] == "001"
        assert report.json_repaired == 1


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestRebuildCli:
    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        from bidsmgr.cli.rebuild import run_rebuild_cli

        tsv = tmp_path / "inv.tsv"
        df = pd.DataFrame([_make_row(
            entities={"subject": "001", "task": "rest"},
            proposed_basename="sub-001_task-restt_bold",  # stale
            proposed_datatype="func", bids_guess_suffix="bold",
        )])
        df.to_csv(tsv, sep="\t", index=False)
        before = tsv.read_text()

        rc = run_rebuild_cli(tsv, direction="entities", dry_run=True)
        assert rc == 0
        assert tsv.read_text() == before  # untouched

    def test_writes_back_when_not_dry_run(self, tmp_path: Path) -> None:
        from bidsmgr.cli.rebuild import run_rebuild_cli

        tsv = tmp_path / "inv.tsv"
        df = pd.DataFrame([_make_row(
            entities={"subject": "001", "task": "rest"},
            proposed_basename="sub-001_task-restt_bold",
            proposed_datatype="func", bids_guess_suffix="bold",
        )])
        df.to_csv(tsv, sep="\t", index=False)

        rc = run_rebuild_cli(tsv, direction="entities", dry_run=False)
        assert rc == 0

        rebuilt = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
        assert rebuilt.iloc[0]["proposed_basename"] == "sub-001_task-rest_bold"

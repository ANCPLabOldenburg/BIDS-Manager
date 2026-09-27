"""The physio sidecar, filled from the ACTIVE schema rather than a list.

The vendored ``bidsphysio`` writes ``Columns``, ``SamplingFrequency`` and
``StartTime``, and nothing BIDS has asked for since it was written. Every
physio file we produced drew the same two validator warnings.

What is checked here is that the gap is closed from facts, not guesses: a
field is written only when the schema in force declares it AND the value can
be derived from what is already on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.fixups.physio_sidecar import enrich_physio_sidecars


def _sidecar(root: Path, name: str, data: dict) -> Path:
    folder = root / "sub-001" / "func"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(data), encoding="utf-8")
    (folder / name.replace(".json", ".tsv.gz")).write_bytes(b"")
    return path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestWhatItFills:
    def test_task_name_comes_from_the_task_entity(self, tmp_path):
        path = _sidecar(
            tmp_path, "sub-001_task-rest_run-1_physio.json",
            {"Columns": ["trigger"], "SamplingFrequency": 1000.0,
             "StartTime": 0.0},
        )
        assert enrich_physio_sidecars(tmp_path) == 1
        assert _read(path)["TaskName"] == "rest"

    @pytest.mark.parametrize("columns,expected", [
        (["cardiac"], "generic"),
        (["respiratory"], "generic"),
        (["cardiac", "respiratory", "trigger"], "generic"),
        (["pulse_ox_raw"], "generic"),
        (["x_coordinate", "y_coordinate", "pupil_size"], "eyetrack"),
        (["x_coordinate", "trigger"], "eyetrack"),
    ])
    def test_the_columns_decide_the_type(self, tmp_path, columns, expected):
        """Only within the values the schema allows. The field is an enum
        (``generic`` / ``eyetrack``), not free text: an earlier version of
        this derived "cardiac" from the column name, which reads perfectly
        well and turned a missing RECOMMENDED field into a validation
        ERROR."""
        path = _sidecar(
            tmp_path, "sub-001_task-rest_physio.json",
            {"Columns": columns, "SamplingFrequency": 100.0, "StartTime": 0.0},
        )
        enrich_physio_sidecars(tmp_path)
        assert _read(path)["PhysioType"] == expected

    def test_it_only_ever_writes_a_value_the_schema_allows(self, tmp_path):
        from bidsmgr.fixups.physio_sidecar import _allowed_values

        allowed = _allowed_values("func", "PhysioType")
        assert allowed, "the schema stopped constraining PhysioType"
        path = _sidecar(
            tmp_path, "sub-001_task-rest_physio.json",
            {"Columns": ["cardiac"], "SamplingFrequency": 100.0,
             "StartTime": 0.0},
        )
        enrich_physio_sidecars(tmp_path)
        assert _read(path)["PhysioType"] in allowed

    def test_a_schema_with_no_enum_for_it_is_left_alone(self, tmp_path,
                                                        monkeypatch):
        from bidsmgr.fixups import physio_sidecar as ps

        monkeypatch.setattr(ps, "_allowed_values", lambda *_a: frozenset())
        path = _sidecar(
            tmp_path, "sub-001_physio.json",
            {"Columns": ["cardiac"], "SamplingFrequency": 100.0,
             "StartTime": 0.0},
        )
        enrich_physio_sidecars(tmp_path)
        assert "PhysioType" not in _read(path)

    def test_the_task_name_uses_the_standards_own_rule(self, tmp_path):
        """A label keeps only ``[0-9a-zA-Z+]``, so the name derived back out
        of one keeps exactly those."""
        path = _sidecar(
            tmp_path, "sub-001_task-nback2_physio.json",
            {"Columns": ["trigger"], "SamplingFrequency": 100.0,
             "StartTime": 0.0},
        )
        enrich_physio_sidecars(tmp_path)
        assert _read(path)["TaskName"] == "nback2"


class TestWhatItLeavesAlone:
    def test_it_never_overwrites(self, tmp_path):
        path = _sidecar(
            tmp_path, "sub-001_task-rest_physio.json",
            {"Columns": ["cardiac"], "SamplingFrequency": 100.0,
             "StartTime": 0.0, "TaskName": "a name somebody typed",
             "PhysioType": "generic"},
        )
        assert enrich_physio_sidecars(tmp_path) == 0
        data = _read(path)
        assert data["TaskName"] == "a name somebody typed"
        assert data["PhysioType"] == "generic"

    def test_it_invents_no_equipment_facts(self, tmp_path):
        """Manufacturer and Description are recommended and underivable.
        The metadata step marks them unanswered; this must not fill them."""
        path = _sidecar(
            tmp_path, "sub-001_task-rest_physio.json",
            {"Columns": ["cardiac"], "SamplingFrequency": 100.0,
             "StartTime": 0.0},
        )
        enrich_physio_sidecars(tmp_path)
        data = _read(path)
        assert "Manufacturer" not in data
        assert "Description" not in data

    def test_a_row_with_no_task_entity_gets_no_task_name(self, tmp_path):
        path = _sidecar(
            tmp_path, "sub-001_physio.json",
            {"Columns": ["cardiac"], "SamplingFrequency": 100.0,
             "StartTime": 0.0},
        )
        enrich_physio_sidecars(tmp_path)
        assert "TaskName" not in _read(path)

    def test_a_tree_with_no_physio_is_a_no_op(self, tmp_path):
        (tmp_path / "sub-001" / "anat").mkdir(parents=True)
        assert enrich_physio_sidecars(tmp_path) == 0

    def test_a_missing_tree_is_a_no_op(self, tmp_path):
        assert enrich_physio_sidecars(tmp_path / "nope") == 0

    def test_unreadable_json_is_skipped_not_raised(self, tmp_path):
        folder = tmp_path / "sub-001" / "func"
        folder.mkdir(parents=True)
        (folder / "sub-001_task-rest_physio.json").write_text("{not json")
        assert enrich_physio_sidecars(tmp_path) == 0


class TestItFollowsTheSchema:
    def test_a_field_the_schema_does_not_declare_is_not_written(
        self, tmp_path, monkeypatch,
    ):
        """The whole reason this is a fixup rather than a patch to the
        vendored writer: an older BIDS that has never heard of PhysioType
        must not be handed one."""
        from bidsmgr.fixups import physio_sidecar as ps

        monkeypatch.setattr(
            ps, "_declared_fields", lambda *_a: {"Columns", "SamplingFrequency"},
        )
        path = _sidecar(
            tmp_path, "sub-001_task-rest_physio.json",
            {"Columns": ["cardiac"], "SamplingFrequency": 100.0,
             "StartTime": 0.0},
        )
        assert enrich_physio_sidecars(tmp_path) == 0
        assert "PhysioType" not in _read(path)
        assert "TaskName" not in _read(path)

    def test_the_active_schema_does_declare_both(self, tmp_path):
        """If this fails, BIDS changed and the fixup should follow it."""
        from bidsmgr.fixups.physio_sidecar import _declared_fields

        declared = _declared_fields("func", "physio")
        assert {"PhysioType", "TaskName"} <= declared

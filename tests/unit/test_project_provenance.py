"""Unit tests for ``bidsmgr.project.provenance``."""

from __future__ import annotations

import json
from pathlib import Path

from bidsmgr.project.provenance import (
    SOURCE_CLASSIFIER,
    SOURCE_DICOM,
    SOURCE_USER,
    ProvenanceMap,
)


def test_empty_map_has_zero_entries() -> None:
    m = ProvenanceMap()
    assert len(m) == 0
    assert m.get("r1", "task") is None
    assert ("r1", "task") not in m


def test_set_and_get_roundtrip() -> None:
    m = ProvenanceMap()
    m.set("r1", "task", SOURCE_USER)
    entry = m.get("r1", "task")
    assert entry is not None
    assert entry.row_id == "r1"
    assert entry.field == "task"
    assert entry.source == SOURCE_USER
    assert entry.set_at.endswith("Z")


def test_set_overwrites_previous_source() -> None:
    m = ProvenanceMap()
    m.set("r1", "task", f"{SOURCE_CLASSIFIER}:dcm2niix_bidsguess")
    m.set("r1", "task", SOURCE_USER)
    assert m.get("r1", "task").source == SOURCE_USER
    assert len(m) == 1  # same key, one entry


def test_for_row_returns_all_fields_for_row() -> None:
    m = ProvenanceMap()
    m.set("r1", "task", SOURCE_USER)
    m.set("r1", "RepetitionTime", f"{SOURCE_DICOM}:(0018,0080)")
    m.set("r2", "task", SOURCE_USER)
    row1 = m.for_row("r1")
    assert set(row1.keys()) == {"task", "RepetitionTime"}
    assert m.for_row("absent") == {}


def test_save_load_roundtrip(tmp_path: Path) -> None:
    m = ProvenanceMap()
    m.set("r1", "task", SOURCE_USER, set_at="2026-01-01T00:00:00.000Z")
    m.set("r1", "RepetitionTime", f"{SOURCE_DICOM}:(0018,0080)",
          set_at="2026-01-01T00:00:01.000Z")
    m.set("r2", "task", f"{SOURCE_CLASSIFIER}:dcm2niix_bidsguess",
          set_at="2026-01-01T00:00:02.000Z")

    p = tmp_path / "prov.json"
    m.save(p)
    assert p.exists()

    loaded = ProvenanceMap.load(p)
    assert len(loaded) == 3
    assert loaded.get("r1", "task").source == SOURCE_USER
    assert loaded.get("r1", "task").set_at == "2026-01-01T00:00:00.000Z"
    assert loaded.get("r2", "task").source == f"{SOURCE_CLASSIFIER}:dcm2niix_bidsguess"


def test_load_absent_file_returns_empty_map(tmp_path: Path) -> None:
    m = ProvenanceMap.load(tmp_path / "does_not_exist.json")
    assert len(m) == 0


def test_save_is_atomic(tmp_path: Path) -> None:
    m = ProvenanceMap()
    m.set("r1", "task", SOURCE_USER)
    p = tmp_path / "prov.json"
    m.save(p)
    # Temp file from the atomic rewrite should not be left behind.
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_save_format_is_stable_for_diffs(tmp_path: Path) -> None:
    # Stable key ordering matters for clean diffs in version-controlled
    # provenance files (some users may track these alongside their TSVs).
    m = ProvenanceMap()
    m.set("r2", "task", SOURCE_USER, set_at="2026-01-01T00:00:00.000Z")
    m.set("r1", "task", SOURCE_USER, set_at="2026-01-01T00:00:00.000Z")
    p = tmp_path / "prov.json"
    m.save(p)
    raw = p.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    # Top-level keys must be sorted (r1 before r2).
    assert list(parsed.keys()) == ["r1", "r2"]

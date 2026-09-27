"""Tests for demographic normalization, participants-file import, and the
handedness column flowing into participants.tsv."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from bidsmgr.metadata.demographics import (
    load_participants_table,
    merge_demographics,
    normalize_handedness,
    normalize_sex,
)
from bidsmgr.metadata.engine import _write_participants


@pytest.mark.parametrize("raw,expected", [
    ("M", "M"), ("m", "M"), ("male", "M"), ("1", "M"),
    ("F", "F"), ("female", "F"), ("2", "F"),
    ("O", "O"), ("other", "O"),
    ("", ""), ("unknown", ""), ("xyz", ""),
])
def test_normalize_sex(raw, expected):
    assert normalize_sex(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("R", "R"), ("right", "R"), ("right-handed", "R"), ("1", "R"),
    ("L", "L"), ("left", "L"), ("2", "L"),
    ("A", "A"), ("ambidextrous", "A"), ("both", "A"), ("3", "A"),
    ("", ""), ("?", ""),
])
def test_normalize_handedness(raw, expected):
    assert normalize_handedness(raw) == expected


def test_load_participants_table_tsv(tmp_path):
    p = tmp_path / "participants.tsv"
    p.write_text(
        "participant_id\tage\tsex\thandedness\n"
        "sub-001\t24\tfemale\tleft\n"
        "002\t31\tmale\tright\n",
        encoding="utf-8",
    )
    table = load_participants_table(p)
    assert set(table) == {"sub-001", "sub-002"}  # bare id normalised to sub-
    assert table["sub-001"]["age"] == "24"
    assert table["sub-001"]["sex"] == "female"


def test_load_participants_table_requires_participant_id(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("name\tage\nx\t1\n", encoding="utf-8")
    assert load_participants_table(p) == {}


def test_merge_demographics_overlay_wins_nonblank():
    base = {"sub-001": {"sex": "M", "age": "20"}}
    overlay = {"sub-001": {"sex": "F", "age": ""}}  # blank age must not clobber
    out = merge_demographics(base, overlay)
    assert out["sub-001"]["sex"] == "F"
    assert out["sub-001"]["age"] == "20"


def _report():
    return SimpleNamespace(files_written=[], warnings=[])


def test_handedness_flows_from_inventory_to_participants(tmp_path):
    bids_root = tmp_path / "ds"
    (bids_root / "sub-001").mkdir(parents=True)
    inv = tmp_path / "inv.tsv"
    inv.write_text(
        "participant_id\tPatientSex\tHandedness\tPatientAge\n"
        "sub-001\tmale\tright\t29\n",
        encoding="utf-8",
    )
    _write_participants(bids_root, inv, _report())

    df = pd.read_csv(bids_root / "participants.tsv", sep="\t", dtype=str, keep_default_na=False)
    row = df.iloc[0]
    assert row["handedness"] == "R"  # normalised from "right"
    assert row["sex"] == "M"         # normalised from "male"
    assert row["age"] == "29"

    desc = json.loads((bids_root / "participants.json").read_text())
    assert desc["handedness"]["Levels"]["R"] == "right"


def test_participants_file_overrides_inventory(tmp_path):
    bids_root = tmp_path / "ds"
    (bids_root / "sub-001").mkdir(parents=True)
    inv = tmp_path / "inv.tsv"
    inv.write_text("participant_id\tHandedness\nsub-001\tleft\n", encoding="utf-8")
    pfile = tmp_path / "participants_src.tsv"
    pfile.write_text("participant_id\thandedness\nsub-001\tright\n", encoding="utf-8")

    _write_participants(bids_root, inv, _report(), participants_file=pfile)
    df = pd.read_csv(bids_root / "participants.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert df.iloc[0]["handedness"] == "R"  # spreadsheet ("right") wins over inventory ("left")

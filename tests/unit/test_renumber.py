"""New subject numbers for the people of a scan (inventory/renumber)."""

from __future__ import annotations

import pandas as pd

from bidsmgr.inventory.renumber import identity, plan_renumber, rows_of_person, taken_labels


def _row(pid: str, patient: str, date: str, time: str = "10:00:00") -> dict:
    return {"participant_id": pid, "PatientID": patient, "GivenName": "", "FamilyName": "",
            "study_date": date, "acq_time": time}


def test_each_person_gets_the_next_free_number_in_order_of_acquisition() -> None:
    df = pd.DataFrame([
        _row("sub-001", "P-B", "20260302"),
        _row("sub-001", "P-B", "20260302", "10:30:00"),        # same person, second series
        _row("sub-002", "P-A", "20260301"),
    ])
    people = plan_renumber(df, [0, 1, 2], taken={"001", "002"})
    assert [(p.identity, p.new, p.rows) for p in people] == [
        ("P-A", "003", (2,)), ("P-B", "004", (0, 1))]


def test_numbers_in_use_are_skipped_and_the_width_follows_the_dataset() -> None:
    df = pd.DataFrame([_row("sub-01", "X", "20260101")])
    assert plan_renumber(df, [0], taken={"0001", "0003"})[0].new == "0004"
    assert plan_renumber(df, [0], taken={"01", "02"}, width=2)[0].new == "03"
    assert plan_renumber(df, [0], taken=set())[0].new == "001", "at least three digits"


def test_what_is_taken(tmp_path) -> None:
    (tmp_path / "sub-007").mkdir()
    df = pd.DataFrame([_row("sub-001", "A", ""), _row("sub-002", "B", "")])
    other = pd.DataFrame({"participant_id": ["sub-009"]})
    assert taken_labels(df, [0], bids_root=tmp_path, others=[other]) == {"002", "007", "009"}


def test_a_recording_without_a_scanned_identity_is_its_label() -> None:
    df = pd.DataFrame([{"participant_id": "sub-OL4594", "PatientID": ""}])
    assert identity(df, 0) == "OL4594"


def test_all_rows_of_a_person() -> None:
    df = pd.DataFrame([_row("sub-001", "A", ""), _row("sub-002", "B", ""),
                       _row("sub-001", "A", "")])
    assert rows_of_person(df, 0) == [0, 2]

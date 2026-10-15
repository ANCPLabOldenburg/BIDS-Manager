"""The inventory counted by subject and kind of recording, and what differs.

Each finding is checked for what it says AND for when it must stay quiet: an
overview that flags everything in a mixed corpus is ignored by the first
person who opens it.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from bidsmgr.inventory import overview as ov_mod
from bidsmgr.inventory.overview import Kind, Visit, build_overview


def _row(sub: str, name: str, datatype: str, *, ses: str = "", include="1",
         modality: str = "mri", study_date: str = "", recording_time: str = "",
         check: bool = False, dataset: str = "ds") -> dict:
    prefix = f"{sub}_{ses}_" if ses else f"{sub}_"
    evidence = json.dumps({"chosen": f"{datatype}/x", "check": True}) if check else ""
    return {"participant_id": sub, "session": ses, "datatype": datatype,
            "bids_name": prefix + name, "include": include, "modality": modality,
            "study_date": study_date, "recording_time": recording_time,
            "classification_evidence": evidence, "dataset": dataset}


def _protocol(sub: str, *, ses: str = "", skip: tuple[str, ...] = (), **kw) -> list[dict]:
    """A T1w, a resting BOLD and a phase-difference fieldmap."""
    rows = [("T1w", "anat"), ("task-rest_bold", "func"), ("phasediff", "fmap")]
    return [_row(sub, name, dt, ses=ses, **kw) for name, dt in rows if name not in skip]


def _find(ov, prefix: str):
    hits = [f for f in ov.findings if f.key.startswith(prefix)]
    assert hits, f"no finding {prefix!r} in {[f.key for f in ov.findings]}"
    return hits[0]


def _keys(ov) -> list[str]:
    return [f.key for f in ov.findings]


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_the_headline_counts_are_what_a_user_could_count() -> None:
    rows = _protocol("sub-01", ses="ses-pre") + _protocol("sub-01", ses="ses-post")
    rows += _protocol("sub-02", ses="ses-pre")
    rows.append(_row("sub-02", "task-rest_eeg", "eeg", ses="ses-pre", modality="eeg"))
    rows.append(_row("sub-02", "T2w", "anat", ses="ses-pre", include="0"))
    ov = build_overview(pd.DataFrame(rows))
    assert (ov.subjects, ov.sessions, ov.recordings, ov.left_out) == (2, 3, 10, 1)
    assert ov.datatypes == 4
    assert ov.by_modality == {"mri": 9, "eeg": 1}
    assert ov.modality_line() == "9 MRI, 1 EEG"


def test_rows_columns_and_cells() -> None:
    rows = []
    for n in (2, 10, 1):
        sub = f"sub-{n}"
        rows += _protocol(sub)
    rows.append(_row("sub-1", "run-2_T1w", "anat"))     # a second run, same kind
    ov = build_overview(pd.DataFrame(rows))
    assert [v.label for v in ov.visits] == ["sub-1", "sub-2", "sub-10"], "natural order"
    assert [k.key for k in ov.kinds] == ["anat/T1w", "func/task-rest_bold", "fmap/phasediff"]
    assert ov.cell(0, 0).count == 2, "runs are counted in one cell"
    assert ov.cell(1, 2).count == 1
    assert ov.kinds[1] == Kind("func", "bold", ("task-rest",))
    assert ov.kinds[1].caption == "task-rest"
    assert ov.visits[0] == Visit("sub-1")
    # Every row of a visit, and the rows of each kind, point back at the table.
    assert sorted(ov.visit_rows[0]) == [6, 7, 8, 9]
    assert sorted(ov.kind_rows[0]) == [0, 3, 6, 9]


def test_include_flags_from_the_caller_win_and_the_default_is_the_conversions() -> None:
    rows = _protocol("sub-01")
    rows[0]["include"] = ""          # blank: the conversion writes only "1"
    df = pd.DataFrame(rows)
    assert build_overview(df).recordings == 2
    assert build_overview(df, included=[True, True, True]).recordings == 3
    assert build_overview(df.drop(columns=["include"])).recordings == 3, "no column: all"


# ---------------------------------------------------------------------------
# What is usual, and what differs from it
# ---------------------------------------------------------------------------


def _study(n: int, **odd) -> list[dict]:
    """``n`` subjects with the same protocol; ``odd`` maps a subject to the
    recordings it lacks."""
    rows = []
    for i in range(1, n + 1):
        sub = f"sub-{i:02d}"
        rows += _protocol(sub, skip=odd.get(sub.replace("-", "_"), ()))
    return rows


def test_a_missing_recording_is_named_with_how_many_others_have_it() -> None:
    ov = build_overview(pd.DataFrame(_study(8, sub_04=("phasediff",))))
    finding = _find(ov, "missing:fmap/phasediff")
    assert finding.text == "sub-04 has no fmap/phasediff; the other 7 have one."
    assert finding.severity == "warn"
    # Show selects what that subject HAS, to see what it got instead.
    assert set(finding.rows) == set(ov.visit_rows[3])
    cell = ov.cell(3, 2)
    assert (cell.state, cell.usual, cell.peers) == ("missing", 1, 8)


def test_a_recording_left_out_is_said_so_rather_than_missing() -> None:
    rows = _study(6)
    rows[5]["include"] = "0"          # sub-02's fieldmap
    ov = build_overview(pd.DataFrame(rows))
    finding = _find(ov, "left-out:fmap/phasediff")
    assert finding.text == "sub-02's fmap/phasediff is left out; the other 5 convert theirs."
    assert finding.rows == (5,)
    assert not any(k.startswith("missing:") for k in _keys(ov))


def test_more_or_fewer_than_the_others() -> None:
    rows = _study(6)
    rows += [_row("sub-03", "run-2_T1w", "anat"), _row("sub-05", "run-2_T1w", "anat")]
    ov = build_overview(pd.DataFrame(rows))
    finding = _find(ov, "extra:anat/T1w")
    assert finding.text == "sub-03 and sub-05 have 2 anat/T1w each; the other 4 have one."
    assert len(finding.rows) == 4


def test_one_label_spelled_two_ways_is_one_finding() -> None:
    rows = _study(5)
    rows[4]["bids_name"] = "sub-02_task-resting_bold"
    ov = build_overview(pd.DataFrame(rows))
    finding = _find(ov, "spelling:")
    assert finding.text == ("sub-02 has func/task-resting_bold where the other 4 have "
                            "func/task-rest_bold: the same task spelled two ways?")
    assert finding.rows == (4,)
    assert not any(k.startswith(("missing:", "rare:")) for k in _keys(ov)), \
        "the same difference is not reported three times"


def test_spelling_needs_a_similar_label() -> None:
    rows = _study(5)
    rows[4]["bids_name"] = "sub-02_task-faces_bold"
    ov = build_overview(pd.DataFrame(rows))
    assert not any(k.startswith("spelling:") for k in _keys(ov))
    assert any(k.startswith("missing:func/task-rest_bold") for k in _keys(ov))


def test_only_one_subject_having_something_needs_enough_subjects() -> None:
    rows = _study(5) + [_row("sub-03", "T2w", "anat")]
    assert not any(k.startswith("rare:") for k in _keys(build_overview(pd.DataFrame(rows)))), \
        "one of five is noise more often than news"
    rows = _study(10) + [_row("sub-07", "T2w", "anat")]
    finding = _find(build_overview(pd.DataFrame(rows)), "rare:anat/T2w")
    assert finding.text == "Only sub-07 has anat/T2w."
    assert finding.severity == "info"


def test_nothing_is_usual_without_enough_to_compare() -> None:
    ov = build_overview(pd.DataFrame(_study(2, sub_02=("phasediff",))))
    assert not ov.compared
    assert ov.cell(1, 2).usual is None
    assert not [f for f in ov.findings if f.key.startswith(("missing", "extra", "rare"))]


def test_nothing_is_usual_when_the_subjects_disagree() -> None:
    rows = _study(3) + _study(3)          # duplicates make every count 2...
    rows += [_row("sub-01", "run-3_T1w", "anat")]   # ...but 3, 2, 2 is still a usual 2
    ov = build_overview(pd.DataFrame(rows))
    assert ov.cell(0, 0).usual == 2
    rows = _study(5)
    rows += [_row("sub-01", "run-2_T1w", "anat"), _row("sub-02", "run-2_T1w", "anat"),
             _row("sub-03", "run-2_T1w", "anat")]
    assert build_overview(pd.DataFrame(rows)).cell(0, 0).usual == 2, "3 of 5 is 60 %: usual"
    rows.append(_row("sub-03", "run-3_T1w", "anat"))
    ov = build_overview(pd.DataFrame(rows))
    assert ov.cell(0, 0).usual is None, "counts 2, 2, 3, 1, 1: none held by 60 %"
    assert not any(f.key.startswith(("extra", "fewer")) for f in ov.findings)


def test_subjects_are_compared_within_their_modality() -> None:
    rows = _study(4)
    rows += [_row(f"sub-{i:02d}", "task-oddball_eeg", "eeg", modality="eeg") for i in (5, 6, 7)]
    ov = build_overview(pd.DataFrame(rows))
    assert not any(k.startswith("missing:anat") for k in _keys(ov)), \
        "an EEG-only subject does not lack an MRI recording"
    rows = _study(8)
    rows = [r for r in rows if r["participant_id"] != "sub-08"]
    rows.append(_row("sub-08", "task-oddball_eeg", "eeg", modality="eeg"))
    rows += [_row(f"sub-{i:02d}", "task-oddball_eeg", "eeg", modality="eeg") for i in range(1, 8)]
    ov = build_overview(pd.DataFrame(rows))
    finding = _find(ov, "no-modality:mri")
    assert finding.text == "sub-08 has no MRI recordings; the other 7 have some."


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_a_subject_without_a_session_among_subjects_with_sessions() -> None:
    rows = _protocol("sub-001", ses="ses-pre") + _protocol("sub-001", ses="ses-post")
    rows += _protocol("sub-002")
    ov = build_overview(pd.DataFrame(rows))
    finding = _find(ov, "no-session")
    assert finding.text == ("sub-002 has no session, while the other one has sessions "
                            "(ses-post and ses-pre).")
    assert len(finding.rows) == 3


def test_a_subject_with_recordings_both_with_and_without_a_session() -> None:
    rows = _protocol("sub-01", ses="ses-1") + [_row("sub-01", "T2w", "anat")]
    rows += _protocol("sub-02", ses="ses-1")
    finding = _find(build_overview(pd.DataFrame(rows)), "with-and-without-session")
    assert finding.text == "sub-01 has recordings both with and without a session."


def test_a_session_most_subjects_have() -> None:
    rows = []
    for i in range(1, 6):
        rows += _protocol(f"sub-{i}", ses="ses-pre")
        if i != 3:
            rows += _protocol(f"sub-{i}", ses="ses-post")
    finding = _find(build_overview(pd.DataFrame(rows)), "missing-session:ses-post")
    assert finding.text == "sub-3 has no ses-post; the other 4 have one."


def test_several_sessions_on_one_day() -> None:
    rows = [_row("sub-01", f"{n}_T1w".lstrip("_"), "anat", ses=f"ses-{n}", study_date="20121228")
            for n in (1, 2)]
    rows += [_row("sub-02", "task-x_eeg", "eeg", ses=s, modality="eeg",
                  recording_time="2009-08-12T16:15:00+00:00") for s in ("ses-a", "ses-b")]
    rows += [_row("sub-03", "T1w", "anat", ses=s, study_date=d)
             for s, d in (("ses-1", "20200101"), ("ses-2", "20200301"))]
    ov = build_overview(pd.DataFrame(rows))
    texts = [f.text for f in ov.findings if f.key.startswith("same-day")]
    assert texts == [
        "sub-01 has 2 sessions recorded on 2012-12-28 (ses-1 and ses-2). If they are one "
        "visit, give them one session label.",
        "sub-02 has 2 sessions recorded on 2009-08-12 (ses-a and ses-b). If they are one "
        "visit, give them one session label.",
    ]


def test_many_split_subjects_are_one_finding() -> None:
    rows = [_row(f"sub-{i}", "T1w", "anat", ses=s, study_date="20200101")
            for i in range(1, 6) for s in ("ses-1", "ses-2")]
    found = [f for f in build_overview(pd.DataFrame(rows)).findings if f.key.startswith("same-day")]
    assert len(found) == 1 and found[0].text.startswith("5 subjects each have several sessions")
    assert len(found[0].rows) == 10


# ---------------------------------------------------------------------------
# Labels, columns, the table as a whole
# ---------------------------------------------------------------------------


def test_labels_that_look_like_identifiers() -> None:
    rows = [_row(f"sub-{i:02d}", f"task-S{i:03d}R01_eeg", "eeg", modality="eeg")
            for i in range(1, 13)]
    ov = build_overview(pd.DataFrame(rows))
    finding = _find(ov, "id-like:task")
    assert finding.text == ("12 different task labels across 12 recordings: they look like "
                            "file names or IDs rather than tasks. The grid groups them by "
                            "datatype and suffix.")
    assert ov.level == "suffix" and [k.key for k in ov.kinds] == ["eeg/eeg"]
    assert ov.id_like == {"task": 12}
    explicit = build_overview(pd.DataFrame(rows), level="labels")
    assert explicit.level == "labels" and len(explicit.kinds) == 12


def test_the_automatic_level_stays_readable() -> None:
    rows = []
    for i in range(1, 4):
        rows += [_row(f"sub-{i}", f"task-t{t}_acq-a{a}_bold", "func")
                 for t in range(6) for a in range(8)]
    ov = build_overview(pd.DataFrame(rows))
    assert ov.level_columns == {"labels": 48, "task": 6, "suffix": 1}
    assert ov.level == "task", "48 columns is past reading at a glance"


def test_two_recordings_claiming_one_name() -> None:
    rows = _study(3)
    rows.append(dict(rows[0]))
    finding = _find(build_overview(pd.DataFrame(rows)), "same-name")
    assert finding.severity == "err"
    assert finding.text == "2 recordings would be written as sub-01_T1w: one would overwrite the other."
    assert finding.rows == (0, 9)


def test_a_recording_with_no_name_and_names_marked_to_check() -> None:
    rows = _study(3)
    rows[1]["bids_name"] = ""
    rows[4]["classification_evidence"] = json.dumps({"check": True})
    ov = build_overview(pd.DataFrame(rows))
    assert _find(ov, "unnamed").text == ("One recording set to convert has no BIDS name or "
                                         "datatype yet.")
    assert _find(ov, "to-check").text == ("One name is marked to check: the acquisition "
                                          "parameters did not settle it.")
    assert _find(ov, "to-check").rows == (4,)


def test_a_subject_with_nothing_to_convert() -> None:
    rows = _study(4)
    for r in rows:
        if r["participant_id"] == "sub-04":
            r["include"] = "0"
    ov = build_overview(pd.DataFrame(rows))
    finding = _find(ov, "nothing-to-convert")
    assert finding.text == "sub-04 has nothing to convert: its 3 recordings are left out."
    assert not any(k.startswith(("left-out:", "missing:")) for k in _keys(ov)), \
        "one finding, not one per kind"


def test_findings_are_ordered_by_severity() -> None:
    rows = _study(10, sub_04=("phasediff",)) + [_row("sub-07", "T2w", "anat")]
    rows.append(dict(rows[0]))
    order = [f.severity for f in build_overview(pd.DataFrame(rows)).findings]
    assert order == sorted(order, key=["err", "warn", "info"].index)


def test_names_are_listed_then_counted() -> None:
    assert ov_mod._listed(["a"]) == "a"
    assert ov_mod._listed(["a", "b"]) == "a and b"
    assert ov_mod._listed(list("abcde")) == "a, b, c, d and e"
    assert ov_mod._listed(list("abcdef")) == "a, b, c, d and 2 more"


def test_an_empty_or_unnamed_table() -> None:
    ov = build_overview(pd.DataFrame(columns=["participant_id", "bids_name"]))
    assert (ov.visits, ov.kinds, ov.findings) == ([], [], [])
    ov = build_overview(pd.DataFrame([{"participant_id": "sub-1", "bids_name": "",
                                       "include": "1"}]))
    assert ov.recordings == 1 and not ov.kinds and _find(ov, "unnamed")


@pytest.mark.parametrize("n_subjects", [400])
def test_a_large_table_is_counted_quickly(n_subjects: int) -> None:
    rows = []
    for i in range(n_subjects):
        for run in (1, 2):
            rows += _protocol(f"sub-{i:04d}", ses="ses-1")
            rows[-1]["bids_name"] = rows[-1]["bids_name"].replace("phasediff", f"run-{run}_phasediff")
    df = pd.DataFrame(rows)
    start = time.perf_counter()
    ov = build_overview(df)
    assert time.perf_counter() - start < 1.0, f"{len(df)} rows"
    assert len(ov.visits) == n_subjects

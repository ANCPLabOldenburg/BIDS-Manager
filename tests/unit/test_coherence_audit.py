"""Check coherence, after the 1.5.1 audit.

The audit ran the tool over 108 bids-examples datasets and our own output
and found it offering wrong things: a ``coordsystem.json`` called an orphan,
inherited sidecars called orphans, links to folder recordings called broken,
provenance into a raw dataset proposed for deletion, TODO authors written
over real ones, ``run-1`` to ``run-10`` called a width split, the MEG
empty-room subject called an uneven session, ``.bval`` rows, a renamed file
reported twice. One test per finding, and the new checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor import coherence
from bidsmgr.editor.coherence import Kind
from bidsmgr.metadata.tables import read_table


def _write(path: Path, content=b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content))
    elif isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


@pytest.fixture()
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    _write(root / "dataset_description.json", {"Name": "t", "BIDSVersion": "1.10.0"})
    _write(root / "participants.tsv", "participant_id\nsub-01\n")
    _write(root / "sub-01/anat/sub-01_T1w.nii.gz")
    _write(root / "sub-01/anat/sub-01_T1w.json", {"AcquisitionDateTime": "2026-01-01T09:00:00"})
    return root


def _kinds(root: Path) -> list[Kind]:
    return [f.kind for f in coherence.check(root)]


def _only(root: Path, kind: Kind):
    found = [f for f in coherence.check(root) if f.kind is kind]
    assert len(found) == 1, found
    return found[0]


# -- sidecars and companions ------------------------------------------------


def test_1_a_coordsystem_file_is_not_an_orphan_sidecar(ds: Path) -> None:
    _write(ds / "sub-01/eeg/sub-01_task-x_eeg.edf")
    _write(ds / "sub-01/eeg/sub-01_coordsystem.json", {"EEGCoordinateSystem": "CapTrak"})
    assert Kind.SIDECAR_ORPHAN not in _kinds(ds)


def test_2_an_inherited_sidecar_in_the_same_folder_is_not_one(ds: Path) -> None:
    func = ds / "sub-01/func"
    for run in (1, 2):
        _write(func / f"sub-01_task-x_run-{run}_bold.nii.gz")
    _write(func / "sub-01_task-x_bold.json", {"RepetitionTime": 2})
    for part in ("mag", "phase"):
        _write(func / f"sub-01_task-y_part-{part}_bold.nii.gz")
    _write(func / "sub-01_task-y_bold.json", {"RepetitionTime": 2})
    assert Kind.SIDECAR_ORPHAN not in _kinds(ds)
    _write(func / "sub-01_task-z_bold.json", {"RepetitionTime": 2})
    assert _only(ds, Kind.SIDECAR_ORPHAN).path.name == "sub-01_task-z_bold.json"


def test_3_folder_recordings_and_bidsignore_are_not_walked(ds: Path) -> None:
    bti = ds / "sub-01/meg/sub-01_task-x_meg"
    _write(bti / "c,rfDC")
    _write(bti / "config")
    _write(bti / "hs_file.json", {})
    _write(ds / "sub-01/meg/sub-01_task-x_meg.json", {"PowerLineFrequency": 50})
    _write(ds / "sub-01/anat/sub-01_T2w.json", {})
    (ds / ".bidsignore").write_text("**/*_T2w.json\n")
    assert Kind.SIDECAR_ORPHAN not in _kinds(ds)


def test_an_events_table_whose_run_is_gone_is_an_orphan(ds: Path) -> None:
    func = ds / "sub-01/func"
    _write(func / "sub-01_task-x_run-1_bold.nii.gz")
    _write(func / "sub-01_task-x_events.tsv", "onset\tduration\n")
    _write(func / "sub-01_task-x_recording-cardiac_physio.tsv.gz")
    assert Kind.COMPANION_ORPHAN not in _kinds(ds), "inherited by run-1, physio too"
    _write(func / "sub-01_task-gone_events.tsv", "onset\tduration\n")
    found = _only(ds, Kind.COMPANION_ORPHAN)
    assert found.path.name == "sub-01_task-gone_events.tsv"
    assert not found.safe, "deleting a file is never ticked for you"


def test_behaviour_alone_is_not_an_orphan(ds: Path) -> None:
    _write(ds / "sub-01/beh/sub-01_task-quiz_events.tsv", "onset\tduration\n")
    assert Kind.COMPANION_ORPHAN not in _kinds(ds)


# -- links ------------------------------------------------------------------


def test_4_a_link_to_a_folder_recording_is_not_broken(ds: Path) -> None:
    _write(ds / "sub-01/meg/sub-01_task-x_meg.ds/inner.meg4")
    _write(ds / "sub-01/meg/sub-01_task-x_meg.json",
           {"AssociatedEmptyRoom": "bids::sub-01/meg/sub-01_task-x_meg.ds"})
    assert Kind.LINK_BROKEN not in _kinds(ds)


def test_5_a_leading_slash_still_finds_the_file(ds: Path) -> None:
    _write(ds / "sub-01/func/sub-01_task-x_bold.nii.gz")
    _write(ds / "sub-01/fmap/sub-01_phasediff.nii.gz")
    _write(ds / "sub-01/fmap/sub-01_phasediff.json",
           {"IntendedFor": "/func/sub-01_task-x_bold.nii.gz"})
    assert Kind.LINK_BROKEN not in _kinds(ds)


def test_6_provenance_into_another_dataset_is_not_broken(ds: Path) -> None:
    desc = {"Name": "t", "BIDSVersion": "1.10.0", "DatasetType": "derivative",
            "DatasetLinks": {"raw": "https://openneuro.org/datasets/ds000001"}}
    _write(ds / "dataset_description.json", desc)
    _write(ds / "sub-01/anat/sub-01_T1w.json",
           {"Sources": ["bids:raw:sub-01/anat/sub-01_T1w.nii.gz"],
            "RawSources": ["sub-01/anat/sub-01_T1w.nii.gz"]})
    _write(ds / "sub-01/anat/sub-01_T1w.nii.gz")
    (ds / "sub-01/anat/sub-01_T1w.nii.gz").unlink()
    _write(ds / "sub-01/anat/sub-01_T1w.nii.gz")
    assert Kind.LINK_BROKEN not in _kinds(ds)


def test_9_and_11_a_string_intended_for_is_one_entry_and_the_key_goes(ds: Path) -> None:
    fmap = _write(ds / "sub-01/fmap/sub-01_phasediff.json",
                  {"IntendedFor": "bids::sub-01/func/sub-01_task-gone_bold.nii.gz"})
    _write(ds / "sub-01/fmap/sub-01_phasediff.nii.gz")
    found = _only(ds, Kind.LINK_BROKEN)
    assert found.changes == (("IntendedFor: bids::sub-01/func/sub-01_task-gone_bold.nii.gz",
                              "", 1),), "one entry, not one per character"
    coherence.apply(ds, [found])
    assert "IntendedFor" not in json.loads(fmap.read_text()), "removed, not left as []"


def test_a_fieldmap_naming_another_subjects_run(ds: Path) -> None:
    _write(ds / "sub-02/func/sub-02_task-x_bold.nii.gz")
    _write(ds / "sub-01/fmap/sub-01_phasediff.nii.gz")
    _write(ds / "sub-01/fmap/sub-01_phasediff.json",
           {"IntendedFor": ["bids::sub-02/func/sub-02_task-x_bold.nii.gz"]})
    found = _only(ds, Kind.LINK_OTHER_SUBJECT)
    assert "sub-02" in found.detail and not found.safe


def test_a_b0_field_source_no_fieldmap_identifies(ds: Path) -> None:
    _write(ds / "sub-01/func/sub-01_task-x_bold.nii.gz")
    _write(ds / "sub-01/func/sub-01_task-x_bold.json", {"B0FieldSource": "fmap0"})
    _write(ds / "sub-01/fmap/sub-01_phasediff.nii.gz")
    _write(ds / "sub-01/fmap/sub-01_phasediff.json", {"B0FieldIdentifier": "fmap1"})
    found = _only(ds, Kind.B0_SOURCE_UNMATCHED)
    assert not found.fixable and found.no_repair
    _write(ds / "sub-01/fmap/sub-01_phasediff.json", {"B0FieldIdentifier": ["fmap1", "fmap0"]})
    assert Kind.B0_SOURCE_UNMATCHED not in _kinds(ds)


# -- intended for -------------------------------------------------------------


def _fieldmap_session(root: Path, intended) -> Path:
    func = root / "sub-01/func"
    _write(func / "sub-01_task-x_bold.nii.gz")
    _write(func / "sub-01_task-x_bold.json", {"AcquisitionTime": "10:00:00"})
    fmap = root / "sub-01/fmap"
    _write(fmap / "sub-01_phasediff.nii.gz")
    meta = {"AcquisitionTime": "09:55:00", "EchoTime1": 0.0049, "EchoTime2": 0.0074}
    if intended is not None:
        meta["IntendedFor"] = intended
    return _write(fmap / "sub-01_phasediff.json", meta)


def test_10_a_different_spelling_of_the_same_file_is_not_a_difference(ds: Path) -> None:
    _fieldmap_session(ds, ["func/sub-01_task-x_bold.nii.gz"])
    assert Kind.INTENDED_FOR_DIFFERS not in _kinds(ds)


def test_12_the_intended_for_advice_is_never_ticked_and_skips_b0_links(ds: Path) -> None:
    sidecar = _fieldmap_session(ds, [])
    found = _only(ds, Kind.INTENDED_FOR_DIFFERS)
    assert not found.safe
    meta = json.loads(sidecar.read_text())
    meta["B0FieldIdentifier"] = "fmap0"
    sidecar.write_text(json.dumps(meta))
    assert Kind.INTENDED_FOR_DIFFERS not in _kinds(ds), "B0Field links are what pipelines use"


# -- names ------------------------------------------------------------------


def test_13_numbering_without_padding_is_not_a_width_split(ds: Path) -> None:
    runs = [_write(ds / f"sub-01/func/sub-01_task-x_run-{run}_bold.nii.gz")
            for run in range(1, 11)]
    assert Kind.ENTITY_WIDTH_SPLIT not in _kinds(ds)
    for path in runs:
        path.unlink()
    _write(ds / "sub-01/func/sub-01_task-x_run-01_bold.nii.gz")
    _write(ds / "sub-01/func/sub-01_task-x_run-2_bold.nii.gz")
    found = _only(ds, Kind.ENTITY_WIDTH_SPLIT)
    assert found.changes == (("run-2", "run-02", 1),)


def test_14_the_padding_refusal_names_the_entity_not_runs(ds: Path) -> None:
    for echo in ("1", "01"):
        _write(ds / f"sub-01/func/sub-01_task-x_echo-{echo}_bold.nii.gz")
    found = _only(ds, Kind.ENTITY_WIDTH_SPLIT)
    assert "two echo values" in found.no_repair


def test_15_task_name_case_is_not_a_mismatch_and_is_never_ticked(ds: Path) -> None:
    _write(ds / "sub-01/func/sub-01_task-rest_bold.nii.gz")
    _write(ds / "sub-01/func/sub-01_task-rest_bold.json", {"TaskName": "Rest"})
    assert Kind.TASK_NAME_MISMATCH not in _kinds(ds)
    _write(ds / "sub-01/func/sub-01_task-rest_bold.json", {"TaskName": "nback"})
    assert not _only(ds, Kind.TASK_NAME_MISMATCH).safe


def test_16_the_empty_room_subject_is_not_an_uneven_session(ds: Path) -> None:
    _write(ds / "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz")
    (ds / "sub-01/anat/sub-01_T1w.nii.gz").unlink()
    (ds / "sub-01/anat/sub-01_T1w.json").unlink()
    (ds / "sub-01/anat").rmdir()
    _write(ds / "sub-emptyroom/meg/sub-emptyroom_task-noise_meg.fif")
    assert Kind.SESSIONS_UNEVEN not in _kinds(ds)


def test_case_spellings_name_the_files(ds: Path) -> None:
    _write(ds / "sub-01/func/sub-01_task-Rest_bold.nii.gz")
    _write(ds / "sub-01/func/sub-01_task-rest_run-2_bold.nii.gz")
    found = _only(ds, Kind.ENTITY_CASE_SPLIT)
    assert len(found.files) == 2 and found.no_repair


# -- scans and sessions tables ----------------------------------------------


def _scans(root: Path, body: str) -> Path:
    return _write(root / "sub-01/sub-01_scans.tsv", "filename\tacq_time\n" + body)


def test_17_a_bval_row_is_one_to_drop_and_is_safe(ds: Path) -> None:
    _write(ds / "sub-01/dwi/sub-01_dwi.nii.gz")
    _write(ds / "sub-01/dwi/sub-01_dwi.bval", "0 1000\n")
    table = _scans(ds, "anat/sub-01_T1w.nii.gz\t2026-01-01T09:00:00\n"
                       "dwi/sub-01_dwi.nii.gz\t2026-01-01T09:10:00\n"
                       "dwi/sub-01_dwi.bval\t2026-01-01T09:10:00\n")
    found = _only(ds, Kind.SCANS_ROW_EXTRA)
    assert found.safe and "part of a recording" in found.detail
    coherence.apply(ds, [found])
    assert [r["filename"] for r in read_table(table)[1]] == [
        "anat/sub-01_T1w.nii.gz", "dwi/sub-01_dwi.nii.gz"]


def test_18_a_missing_row_gets_the_sidecars_time(ds: Path) -> None:
    table = _scans(ds, "")
    found = _only(ds, Kind.SCANS_ROW_MISSING)
    assert found.about == "sub-01/anat/sub-01_T1w.nii.gz", "named by the recording"
    assert found.safe
    coherence.apply(ds, [found])
    assert read_table(table)[1] == [
        {"filename": "anat/sub-01_T1w.nii.gz", "acq_time": "2026-01-01T09:00:00"}]


def test_19_a_file_renamed_outside_is_one_finding(ds: Path) -> None:
    table = _scans(ds, "anat/sub-01_acq-old_T1w.nii.gz\t2026-01-01T08:00:00\n")
    found = coherence.check(ds)
    assert [f.kind for f in found] == [Kind.SCANS_ROW_MOVED]
    coherence.apply(ds, found)
    assert read_table(table)[1] == [
        {"filename": "anat/sub-01_T1w.nii.gz", "acq_time": "2026-01-01T08:00:00"}], \
        "the row follows, its time kept"


def test_20_a_row_in_the_wrong_letter_case_is_reported(ds: Path) -> None:
    _scans(ds, "anat/sub-01_t1w.nii.gz\t2026-01-01T09:00:00\n")
    assert Kind.SCANS_ROW_MOVED in _kinds(ds), "even where the disk ignores case"


def test_blank_cells_are_filled_with_the_time_or_na(ds: Path) -> None:
    table = _scans(ds, "anat/sub-01_T1w.nii.gz\t\n")
    found = _only(ds, Kind.TABLE_BLANK_CELLS)
    coherence.apply(ds, [found])
    assert read_table(table)[1][0]["acq_time"] == "2026-01-01T09:00:00"


def test_the_parts_of_one_recording_must_share_a_time(ds: Path) -> None:
    for echo in (1, 2):
        _write(ds / f"sub-01/anat/sub-01_echo-{echo}_MEGRE.nii.gz")
    table = _scans(ds, "anat/sub-01_T1w.nii.gz\t2026-01-01T09:00:00\n"
                       "anat/sub-01_echo-1_MEGRE.nii.gz\t2026-01-01T09:10:00\n"
                       "anat/sub-01_echo-2_MEGRE.nii.gz\t2026-01-01T09:10:02\n")
    found = _only(ds, Kind.SCANS_TIMES_DIFFER)
    assert not found.safe
    coherence.apply(ds, [found])
    assert {r["acq_time"] for r in read_table(table)[1][1:]} == {"2026-01-01T09:10:00"}


def test_a_session_without_a_table_when_others_have_one(ds: Path) -> None:
    _scans(ds, "anat/sub-01_T1w.nii.gz\t2026-01-01T09:00:00\n")
    _write(ds / "sub-02/anat/sub-02_T1w.nii.gz")
    _write(ds / "participants.tsv", "participant_id\nsub-01\nsub-02\n")
    found = _only(ds, Kind.SCANS_TABLE_MISSING)
    assert found.safe
    coherence.apply(ds, [found])
    assert (ds / "sub-02/sub-02_scans.tsv").is_file()


def test_sessions_rows_follow_the_session_folders(ds: Path) -> None:
    for ses in ("pre", "post"):
        _write(ds / f"sub-01/ses-{ses}/anat/sub-01_ses-{ses}_T1w.nii.gz")
    table = _write(ds / "sub-01/sub-01_sessions.tsv",
                   "session_id\tgroup\nses-pre\tA\nses-gone\tA\n")
    kinds = _kinds(ds)
    assert Kind.SESSION_ROW_ORPHAN in kinds and Kind.SESSION_ROW_MISSING in kinds
    coherence.apply(ds, [f for f in coherence.check(ds) if f.safe])
    ids = [r["session_id"] for r in read_table(table)[1]]
    assert ids == ["ses-pre", "ses-post"]


def test_no_tables_at_all_is_a_finding_unless_tables_are_off(ds: Path) -> None:
    """What Tools, Scans and sessions tables would write, Check coherence
    reports (2026-10-19: a dataset with none was called coherent)."""
    _write(ds / "sub-02/ses-1/anat/sub-02_ses-1_T1w.nii.gz")
    _write(ds / "sub-02/ses-2/anat/sub-02_ses-2_T1w.nii.gz")
    found = {(f.kind, f.about) for f in coherence.check(ds)}
    assert (Kind.SCANS_TABLE_MISSING, "sub-01") in found
    assert (Kind.SCANS_TABLE_MISSING, "sub-02/ses-1") in found
    assert (Kind.SESSIONS_TABLE_MISSING, "sub-02") in found
    missing = [f for f in coherence.check(ds) if f.kind is Kind.SCANS_TABLE_MISSING]
    assert all(f.safe for f in missing), "creating a table loses nothing"
    assert "optional" in missing[0].detail
    tables = {Kind.SCANS_TABLE_MISSING, Kind.SESSIONS_TABLE_MISSING, Kind.SCANS_ROW_MISSING,
              Kind.SESSION_ROW_MISSING}
    assert not [f for f in coherence.check(ds, tables=False) if f.kind in tables], \
        "tables switched off in Settings: none is coherent"


@pytest.mark.parametrize("layout", ["none", "scans only", "half", "stale"])
def test_after_its_repairs_the_tables_tool_has_nothing_left(ds: Path, layout: str) -> None:
    """Whatever the tables tool would change is a finding, and repairing every
    finding leaves it nothing to do."""
    from bidsmgr.metadata.tables import plan_tables

    _write(ds / "sub-02/ses-1/anat/sub-02_ses-1_T1w.nii.gz")
    _write(ds / "sub-02/ses-1/beh/sub-02_ses-1_task-quiz_beh.tsv", "trial\n1\n")
    _write(ds / "sub-02/ses-2/func/sub-02_ses-2_task-x_bold.nii.gz")
    if layout in ("scans only", "half", "stale"):
        _scans(ds, "anat/sub-01_T1w.nii.gz\t2026-01-01T09:00:00\n")
    if layout == "half":
        _write(ds / "sub-02/sub-02_sessions.tsv", "session_id\tacq_time\nses-1\tn/a\n")
    if layout == "stale":
        _write(ds / "sub-02/ses-1/sub-02_ses-1_scans.tsv",
               "filename\tacq_time\nanat/sub-02_ses-1_T1w.nii.gz\t\nanat/gone.nii.gz\tn/a\n")
    assert plan_tables(ds), "the tool would change something"
    findings = coherence.check(ds)
    assert findings
    coherence.apply(ds, [f for f in findings if f.fixable])
    assert plan_tables(ds) == [], [e.summary() for e in plan_tables(ds)]


# -- participants and citation ------------------------------------------------


def test_8_a_phenotype_only_participant_is_not_an_orphan(ds: Path) -> None:
    _write(ds / "participants.tsv", "participant_id\tage\nsub-01\t30\nsub-02\t40\n")
    _write(ds / "phenotype/questionnaire.tsv", "participant_id\tscore\nsub-02\t7\n")
    assert Kind.PARTICIPANT_ORPHAN not in _kinds(ds)
    _write(ds / "participants.tsv", "participant_id\tage\nsub-01\t30\nsub-03\t40\n")
    found = _only(ds, Kind.PARTICIPANT_ORPHAN)
    assert not found.safe, "a row says things about a person; never ticked for you"


def test_21_no_participants_table_offers_to_create_it(ds: Path) -> None:
    (ds / "participants.tsv").unlink()
    found = _only(ds, Kind.PARTICIPANTS_TABLE_MISSING)
    assert found.safe and "create participants.tsv" in found.repair
    coherence.apply(ds, [found])
    assert read_table(ds / "participants.tsv")[1] == [{"participant_id": "sub-01"}]


def test_7_citation_compares_only_what_both_state(ds: Path) -> None:
    pytest.importorskip("yaml")
    from bidsmgr.editor import cff

    cff_path = ds / "CITATION.cff"
    cff_path.write_text(cff.dumps({"cff-version": "1.2.0", "message": "m", "title": "t",
                                   "authors": [{"family-names": "Doe",
                                                "given-names": "Jane"}]}))
    assert Kind.CITATION_DIVERGED not in _kinds(ds), "the description states no authors"
    _write(ds / "dataset_description.json",
           {"Name": "t", "BIDSVersion": "1.10.0", "Authors": ["John Smith"]})
    found = _only(ds, Kind.CITATION_DIVERGED)
    assert not found.safe and "Authors" in found.detail


# -- the walk -----------------------------------------------------------------


def test_22_the_tools_scratch_folder_is_not_walked(ds: Path) -> None:
    _write(ds / ".tmp_bidsmgr/sub-09/anat/sub-09_T1w.json", {})
    _write(ds / ".tmp_bidsmgr/sub-09/anat/sub-09_task-x_events.tsv", "onset\n")
    assert coherence.check(ds, tables=False) == []
    assert not [f for f in coherence.check(ds) if ".tmp_bidsmgr" in f"{f.path} {f.about}"]


def test_only_safe_repairs_are_safe(ds: Path) -> None:
    """Safe means nothing is lost: add a row, drop a row naming nothing."""
    safe_kinds = {Kind.SCANS_ROW_MOVED, Kind.SCANS_ROW_ORPHAN, Kind.SCANS_ROW_EXTRA,
                  Kind.SCANS_ROW_MISSING, Kind.SCANS_TABLE_MISSING, Kind.SESSION_ROW_ORPHAN,
                  Kind.SESSION_ROW_MISSING, Kind.SESSIONS_TABLE_MISSING,
                  Kind.TABLE_BLANK_CELLS, Kind.PARTICIPANTS_TABLE_MISSING,
                  Kind.PARTICIPANT_MISSING}
    _scans(ds, "anat/sub-01_gone_T1w.nii.gz\tn/a\n")
    _write(ds / "sub-02/anat/sub-02_T1w.nii.gz")
    _write(ds / "sub-01/anat/sub-01_T2w.json", {})
    for finding in coherence.check(ds):
        assert finding.safe == (finding.kind in safe_kinds), finding.kind

"""What the conversion will write, predicted before it runs.

The prediction was checked against real conversions of 49 vendor test sets
(dcm2niix QA, ICDC-Glioma, UPENN-GBM) and the multimodal sample: 39 matched
file for file, the rest differ only by vendor-specific files the conversion
does not rename. These tests pin the conventions that made it match.
"""

from __future__ import annotations

import json

import pandas as pd

from bidsmgr.classifier.acquisition.features import SeriesFeatures
from bidsmgr.converter.output_plan import UNKNOWN, PlanOptions, approximate, plan_output


def _row(sub, name, datatype, *, ses="", uid="", source="", modality="mri", include="1", **extra):
    row = {"participant_id": sub, "session": ses, "datatype": datatype,
           "bids_name": name, "bids_guess_suffix": name.rsplit("_", 1)[-1],
           "include": include, "modality": modality, "series_uid": uid, "source_file": source,
           "sequence": name, "acq_time": "", "study_date": "", "dataset": "ds"}
    row.update(extra)
    return row


def _paths(plan, *, sure: bool = True) -> set[str]:
    return {f.path for f in plan.files if (not f.conditional) == sure}


def test_an_mri_session_with_its_companions() -> None:
    rows = [
        _row("sub-01", "sub-01_ses-a_T1w", "anat", ses="ses-a", uid="u1"),
        _row("sub-01", "sub-01_ses-a_task-rest_bold", "func", ses="ses-a", uid="u2", task="rest"),
        _row("sub-01", "sub-01_ses-a_dir-AP_dwi", "dwi", ses="ses-a", uid="u3"),
        _row("sub-01", "sub-01_ses-a_dir-PA_epi", "fmap", ses="ses-a", uid="u4"),
        _row("sub-01", "sub-01_ses-a_magnitude1", "fmap", ses="ses-a", uid="u5|u6"),
        _row("sub-01", "sub-01_ses-a_localizer", "anat", ses="ses-a", uid="u7", include="0"),
    ]
    feats = {"u4": SeriesFeatures(image_type=["ORIGINAL", "PRIMARY", "DIFFUSION"]),
             "u5": SeriesFeatures(echo_times=[4.92, 7.38])}
    plan = plan_output(pd.DataFrame(rows), features=feats)
    sure = _paths(plan)
    folder = "sub-01/ses-a"
    for path in ("anat/sub-01_ses-a_T1w.nii.gz", "anat/sub-01_ses-a_T1w.json",
                 "func/sub-01_ses-a_task-rest_bold.nii.gz",
                 "dwi/sub-01_ses-a_dir-AP_dwi.bval", "dwi/sub-01_ses-a_dir-AP_dwi.bvec",
                 "fmap/sub-01_ses-a_dir-PA_epi.bval",
                 "fmap/sub-01_ses-a_magnitude2.nii.gz", "fmap/sub-01_ses-a_phasediff.json",
                 "sub-01_ses-a_scans.tsv"):
        assert f"{folder}/{path}" in sure, path
    assert not any("localizer" in p for p in sure), "a row left out writes nothing"
    assert {"dataset_description.json", ".bidsignore", "participants.tsv", "participants.json"} <= sure
    assert {"README", "CHANGES"} <= _paths(plan, sure=False), "written unless there is one"


def test_a_plain_epi_reference_gets_no_gradient_tables() -> None:
    """GE stores a b-value of 0 on a plain EPI: that is not diffusion."""
    rows = [_row("sub-01", "sub-01_dir-AP_epi", "fmap", uid="u1")]
    plan = plan_output(pd.DataFrame(rows), features={"u1": SeriesFeatures(b_values=[0.0])})
    assert not any(p.endswith((".bval", ".bvec")) for p in _paths(plan))


def test_a_series_with_several_echoes_is_split_and_said_so() -> None:
    rows = [_row("sub-01", "sub-01_task-x_bold", "func", uid="u1")]
    plan = plan_output(pd.DataFrame(rows), features={"u1": SeriesFeatures(echo_times=[12.0, 30.0])})
    split = [f for f in plan.files if "_e" in f.path]
    assert sorted(f.path.rsplit("/", 1)[-1] for f in split) == [
        "sub-01_task-x_bold_e1.json", "sub-01_task-x_bold_e1.nii.gz",
        "sub-01_task-x_bold_e2.json", "sub-01_task-x_bold_e2.nii.gz"]
    assert all("not included" in f.warning for f in split)
    assert "sub-01/func/sub-01_task-x_bold.nii.gz" not in _paths(plan)


def test_physio_is_one_file_per_signal_when_it_has_several() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_physio", "func", uid="p1")]
    plan = plan_output(pd.DataFrame(rows))
    maybe = _paths(plan, sure=False)
    for signal in ("cardiac", "ecg", "respiratory", "trigger"):
        assert f"sub-01/func/sub-01_task-rest_recording-{signal}_physio.tsv.gz" in maybe
    assert "sub-01/func/sub-01_task-rest_physio.tsv.gz" in maybe
    assert not any(p.endswith("_scans.tsv") for p in _paths(plan)), \
        "physio is not a row of the scans table"


def test_eeg_and_meg_as_mne_bids_writes_them() -> None:
    rows = [_row("sub-01", "sub-01_ses-1_task-rest_eeg", "eeg", ses="ses-1",
                 source="raw/rest.vhdr", modality="eeg"),
            _row("sub-01", "sub-01_ses-2_task-rest_meg", "meg", ses="ses-2",
                 source="raw/rest.fif", modality="meg")]
    plan = plan_output(pd.DataFrame(rows))
    sure, maybe = _paths(plan), _paths(plan, sure=False)
    for ext in (".vhdr", ".eeg", ".vmrk", ".json"):
        assert f"sub-01/ses-1/eeg/sub-01_ses-1_task-rest_eeg{ext}" in sure
    assert "sub-01/ses-1/eeg/sub-01_ses-1_task-rest_channels.tsv" in sure
    assert "sub-01/ses-1/eeg/sub-01_ses-1_task-rest_events.tsv" in maybe
    # Positions describe the session, not the run: no task in the name.
    assert "sub-01/ses-1/eeg/sub-01_ses-1_electrodes.tsv" in maybe
    assert "sub-01/ses-2/meg/sub-01_ses-2_coordsystem.json" in maybe
    assert "sub-01/ses-2/meg/sub-01_ses-2_task-rest_meg.fif" in sure
    assert {"sub-01/ses-1/sub-01_ses-1_scans.tsv", "sub-01/ses-2/sub-01_ses-2_scans.tsv"} <= sure


def test_force_edf_changes_the_eeg_file() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_eeg", "eeg", source="rest.cnt", modality="eeg")]
    assert "sub-01/eeg/sub-01_task-rest_eeg.vhdr" in _paths(plan_output(pd.DataFrame(rows)))
    plan = plan_output(pd.DataFrame(rows), options=PlanOptions(force_edf=True))
    assert "sub-01/eeg/sub-01_task-rest_eeg.edf" in _paths(plan)


def test_eye_tracking_writes_two_files_per_eye() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_recording-eye1_physio", "beh",
                 source="rec.edf", modality="eyetrack")]
    plan = plan_output(pd.DataFrame(rows))
    sure, maybe = _paths(plan), _paths(plan, sure=False)
    assert "sub-01/beh/sub-01_task-rest_recording-eye1_physio.tsv.gz" in sure
    assert "sub-01/beh/sub-01_task-rest_recording-eye1_physioevents.json" in sure
    assert "sub-01/beh/sub-01_task-rest_recording-eye2_physio.tsv.gz" in maybe
    # Eye tracking with nothing else recorded IS the subject's data, so the
    # metadata step lists it, and the preview says so (the second eye only
    # when it exists).
    scans = approximate(pd.DataFrame(rows), plan.find("sub-01/sub-01_scans.tsv"))
    assert [r[0] for r in scans.rows] == ["beh/sub-01_task-rest_recording-eye1_physio.tsv.gz"]


def _dicom(path, number: int, description: str) -> str:
    """A minimal DICOM header holding the series number and description."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = "1.2.3"
    ds = Dataset()
    ds.file_meta = meta
    ds.SeriesNumber = number
    ds.SeriesDescription = description
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


def test_the_metadata_step_and_sourcedata_follow_the_settings(tmp_path) -> None:
    rows = [_row("sub-01", "sub-01_T1w", "anat", uid="u1", sequence="t1_mprage")]
    off = plan_output(pd.DataFrame(rows), options=PlanOptions(run_metadata=False))
    assert "participants.tsv" not in _paths(off)
    assert not any(p.endswith("_scans.tsv") for p in _paths(off)), \
        "an MRI session's scans table is the metadata step's"
    eeg = [_row("sub-01", "sub-01_task-x_eeg", "eeg", source="r.edf", modality="eeg")]
    assert "sub-01/sub-01_scans.tsv" in _paths(plan_output(
        pd.DataFrame(eeg), options=PlanOptions(run_metadata=False))), "MNE-BIDS writes it"
    # The series folder is named as the conversion names it: the series
    # number and description from the DICOM, else the BIDS name.
    first = {"u1": _dicom(tmp_path / "1.dcm", 5, "t1 mprage")}
    copy = plan_output(pd.DataFrame(rows), options=PlanOptions(sourcedata="copy"),
                       first_files=first)
    assert "sourcedata/sub-01/mri/005_t1 mprage/" in _paths(copy)
    assert "sourcedata/sub-01/manifest.tsv" in _paths(copy)
    archive = plan_output(pd.DataFrame(rows), options=PlanOptions(sourcedata="archive"),
                          first_files=first)
    assert "sourcedata/sub-01/mri/005_t1 mprage.tar.gz" in _paths(archive)
    unread = plan_output(pd.DataFrame(rows), options=PlanOptions(sourcedata="copy"))
    assert "sourcedata/sub-01/mri/sub-01_T1w/" in _paths(unread)
    other = plan_output(pd.DataFrame(rows), options=PlanOptions(sourcedata="copy",
                                                                 sourcedata_modalities=("eeg",)))
    assert not any(p.startswith("sourcedata/sub-01/mri") for p in _paths(other))


# ---------------------------------------------------------------------------
# What a file will contain
# ---------------------------------------------------------------------------


def _fields(preview) -> dict:
    return {k: (v, s) for k, v, s in preview.fields}


def test_a_bold_sidecar_from_the_scan_the_table_and_the_settings() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_bold", "func", uid="b1"),
            _row("sub-01", "sub-01_phasediff", "fmap", uid="f1"),
            ]
    feats = {"b1": SeriesFeatures(manufacturer="SIEMENS", model="Prisma", field_strength=3.0,
                                  repetition_time=0.8, echo_times=[30.0], flip_angle=52.0,
                                  is_epi=True, pe_axis="COL", pe_positive=0.0,
                                  multiband=8.0, has_slice_timing=True)}
    df = pd.DataFrame(rows)
    plan = plan_output(df, features=feats)
    bold = plan.find("sub-01/func/sub-01_task-rest_bold.json")
    f = _fields(approximate(df, bold, PlanOptions(fill_scope="none"), feats))
    assert f["TaskName"] == ("rest", "settings")
    assert f["Manufacturer"] == ("Siemens", "scan"), "as dcm2niix names it"
    assert f["RepetitionTime"] == (0.8, "scan")
    assert f["EchoTime"] == (0.03, "scan"), "ms in the scan, seconds in BIDS"
    # The sign depends on how the vendor stores the polarity: dcm2niix's.
    assert f["PhaseEncodingDirection"] == (UNKNOWN, "converter")
    assert f["MultibandAccelerationFactor"] == (8.0, "scan")
    assert f["SliceTiming"] == (UNKNOWN, "converter")
    assert f["TotalReadoutTime"] == (UNKNOWN, "converter")
    assert f["B0FieldSource"][1] == "settings"
    assert not any(s == "todo" for _v, s in f.values())


def test_fieldmap_links_follow_the_settings() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_bold", "func", uid="b1", acq_time="100000"),
            _row("sub-01", "sub-01_phasediff", "fmap", uid="f1", acq_time="095000")]
    df = pd.DataFrame(rows)
    feats = {"b1": SeriesFeatures(), "f1": SeriesFeatures(echo_times=[4.92, 7.38])}
    plan = plan_output(df, features=feats)
    fmap = plan.find("sub-01/fmap/sub-01_phasediff.json")
    uri = _fields(approximate(df, fmap, PlanOptions(fill_scope="none"), feats))
    assert uri["IntendedFor"][0] == ["bids::sub-01/func/sub-01_task-rest_bold.nii.gz"]
    assert uri["B0FieldIdentifier"] == ("fmap1", "settings")
    assert uri["EchoTime1"] == (0.00492, "scan") and uri["EchoTime2"] == (0.00738, "scan")
    rel = _fields(approximate(df, fmap, PlanOptions(intended_for="relative", b0_field=False,
                                                    fill_scope="none"), feats))
    assert rel["IntendedFor"][0] == ["func/sub-01_task-rest_bold.nii.gz"]
    assert "B0FieldIdentifier" not in rel


def test_fill_scope_adds_the_placeholders_the_metadata_step_writes() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_eeg", "eeg", source="r.edf", modality="eeg",
                 sfreq="500", duration_sec="61.5", line_freq="50")]
    df = pd.DataFrame(rows)
    plan = plan_output(df)
    sidecar = plan.find("sub-01/eeg/sub-01_task-rest_eeg.json")
    on = _fields(approximate(df, sidecar, PlanOptions(fill_scope="recommended")))
    off = _fields(approximate(df, sidecar, PlanOptions(fill_scope="none")))
    assert on["SamplingFrequency"] == (500.0, "scan")
    assert on["PowerLineFrequency"] == (50.0, "settings")
    # MNE-BIDS computes the duration from the samples, and writes "n/a"
    # where BIDS requires a key the recording does not answer.
    assert on["RecordingDuration"] == (UNKNOWN, "converter")
    assert on["Manufacturer"] == ("n/a", "converter")
    todos = {k for k, (_v, s) in on.items() if s == "todo"}
    assert "TaskDescription" in todos and "Manufacturer" not in todos
    assert not any(s == "todo" for _v, s in off.values())


def test_dataset_files() -> None:
    rows = [_row("sub-02", "sub-02_T1w", "anat", uid="u1", PatientAge="034Y", PatientSex="F"),
            _row("sub-01", "sub-01_T1w", "anat", uid="u2", PatientSex="M")]
    df = pd.DataFrame(rows)
    plan = plan_output(df, options=PlanOptions(dataset="study", bids_version="1.11.1"))
    desc = _fields(approximate(df, plan.find("dataset_description.json"),
                               PlanOptions(dataset="study", bids_version="1.11.1", version="9.9")))
    assert desc["Name"] == ("study", "settings") and desc["BIDSVersion"][0] == "1.11.1"
    assert desc["License"] == ("TODO", "todo")
    # Recommended while the dataset has no CITATION.cff, which a new one has
    # not at the metadata step: the schema's rule, as the step applies it.
    assert desc["Authors"] == (["TODO"], "todo")
    assert "Funding" not in desc, "optional: not in the default fill scope"
    everything = _fields(approximate(df, plan.find("dataset_description.json"),
                                     PlanOptions(fill_scope="optional")))
    assert everything["Funding"] == (["TODO"], "todo")
    assert [g["Description"] for g in desc["GeneratedBy"][0]] == [
        "dcm2niix-direct backend", "metadata engine"]
    cited = _fields(approximate(df, plan.find("dataset_description.json"),
                                PlanOptions(citation=True)))
    assert "Authors" not in cited, "the citation file takes it"
    parts = approximate(df, plan.find("participants.tsv"))
    assert parts.header == ("participant_id", "age", "sex")
    assert parts.rows == (("sub-01", "n/a", "M"), ("sub-02", "34", "F"))
    codebook = _fields(approximate(df, plan.find("participants.json")))
    assert set(codebook) == {"participant_id", "age", "sex"}
    from bidsmgr.metadata.engine import readme_text

    assert approximate(df, plan.find("README")).text == readme_text("<dataset>")
    assert approximate(df, plan.find(".bidsignore")).text == ".bidsmgr/\n.tmp_bidsmgr/\n"


def test_the_scans_table() -> None:
    rows = [_row("sub-01", "sub-01_T1w", "anat", uid="u1", study_date="20230914",
                 acq_time="102520.255"),
            _row("sub-01", "sub-01_dwi", "dwi", uid="u2", study_date="20230914", acq_time="103842")]
    df = pd.DataFrame(rows)
    plan = plan_output(df)
    scans = approximate(df, plan.find("sub-01/sub-01_scans.tsv"))
    assert scans.header == ("filename", "acq_time")
    # One row per recording: the gradient tables belong to the image's row.
    assert scans.rows == (("anat/sub-01_T1w.nii.gz", "2023-09-14T10:25:20.255000"),
                          ("dwi/sub-01_dwi.nii.gz", "2023-09-14T10:38:42"))


def test_events_name_the_labels_given_to_event_codes() -> None:
    """The task's own map, else the one for every task, never another task's:
    the enrichment's resolution."""
    from bidsmgr.recording_meta import RecordingMetaSpec

    spec = RecordingMetaSpec(event_maps={"x": {"1": "left", "2": "right"},
                                         "y": {"9": "other"}, "*": {"5": "any"}})
    rows = [_row("sub-01", "sub-01_task-x_eeg", "eeg", source="r.edf", modality="eeg"),
            _row("sub-01", "sub-01_task-z_eeg", "eeg", source="s.edf", modality="eeg")]
    df = pd.DataFrame(rows)
    plan = plan_output(df)
    events = approximate(df, plan.find("sub-01/eeg/sub-01_task-x_events.tsv"), spec=spec)
    assert events.header[:3] == ("onset", "duration", "trial_type")
    assert [r[2] for r in events.rows] == ["left", "right"]
    other = approximate(df, plan.find("sub-01/eeg/sub-01_task-z_events.tsv"), spec=spec)
    assert [r[2] for r in other.rows] == ["any"]
    levels = _fields(approximate(df, plan.find("sub-01/eeg/sub-01_task-x_events.json"),
                                 spec=spec))
    assert levels["trial_type"][0]["Levels"] == {"left": "trigger code 1",
                                                "right": "trigger code 2"}


def test_json_text_reads_as_the_file_would() -> None:
    rows = [_row("sub-01", "sub-01_T1w", "anat", uid="u1")]
    df = pd.DataFrame(rows)
    plan = plan_output(df, features={"u1": SeriesFeatures(manufacturer="GE MEDICAL SYSTEMS")})
    preview = approximate(df, plan.find("sub-01/anat/sub-01_T1w.json"),
                          PlanOptions(fill_scope="none"), {"u1": SeriesFeatures(manufacturer="GE")})
    data = json.loads(preview.json_text())
    assert data["Modality"] == "MR" and data["Manufacturer"] == "GE"


def test_a_large_table_is_planned_quickly() -> None:
    import time

    rows = []
    for i in range(1500):
        rows.append(_row(f"sub-{i:04d}", f"sub-{i:04d}_task-rest_bold", "func", uid=f"u{i}"))
    start = time.perf_counter()
    plan = plan_output(pd.DataFrame(rows))
    assert time.perf_counter() - start < 2.0
    assert len(plan.files) > 4500


# ---------------------------------------------------------------------------
# The preview follows what the user stated and every setting that writes
# ---------------------------------------------------------------------------


def _spec(**fields):
    from bidsmgr.recording_meta import RecordingMetaSpec

    return RecordingMetaSpec(**fields)


def test_templates_and_per_file_answers_reach_the_sidecar() -> None:
    """What the metadata form states, for a kind of file, a task or one
    recording, appears in the prediction, marked as the user's, and a field
    it answers gets no TODO."""
    rows = [_row("sub-01", "sub-01_task-rest_bold", "func", uid="u1", task="rest"),
            _row("sub-02", "sub-02_task-rest_bold", "func", uid="u2", task="rest")]
    df = pd.DataFrame(rows)
    spec = _spec(sequence_templates={"func/bold": {"TaskDescription": "eyes open"},
                                     "func/bold@rest": {"Instructions": "look at the cross"}},
                 row_templates={"u2": {"TaskDescription": "eyes closed"}})
    plan = plan_output(df)
    one = _fields(approximate(df, plan.find("sub-01/func/sub-01_task-rest_bold.json"), spec=spec))
    two = _fields(approximate(df, plan.find("sub-02/func/sub-02_task-rest_bold.json"), spec=spec))
    assert one["TaskDescription"] == ("eyes open", "stated")
    assert one["Instructions"] == ("look at the cross", "stated")
    assert two["TaskDescription"] == ("eyes closed", "stated"), "one recording's own answer"
    # Without the metadata step the answers do not reach the files at all.
    off = _fields(approximate(df, plan.find("sub-01/func/sub-01_task-rest_bold.json"),
                              PlanOptions(run_metadata=False), spec=spec))
    assert "TaskDescription" not in off


def test_dataset_wide_answers_and_the_tables_cells_reach_eeg() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_eeg", "eeg", source="r.edf", modality="eeg",
                 eeg_reference="Cz")]
    df = pd.DataFrame(rows)
    spec = _spec(defaults={"power_line_freq": 60, "institution_name": "Oldenburg"})
    plan = plan_output(df)
    side = _fields(approximate(df, plan.find("sub-01/eeg/sub-01_task-rest_eeg.json"), spec=spec))
    assert side["PowerLineFrequency"][0] == 60.0, "handed to MNE-BIDS at conversion"
    assert side["EEGReference"] == ("Cz", "stated"), "the table's cell"
    assert side["InstitutionName"] == ("Oldenburg", "stated")
    assert "InstitutionName" not in {k for k, (_v, s) in side.items() if s == "todo"}


def test_the_dataset_description_carries_the_stated_answers() -> None:
    rows = [_row("sub-01", "sub-01_T1w", "anat", uid="u1")]
    df = pd.DataFrame(rows)
    spec = _spec(dataset_description={"name": "My study", "license": "CC0",
                                      "authors": ["Ada Lovelace", "Alan Turing"]})
    plan = plan_output(df)
    desc = _fields(approximate(df, plan.find("dataset_description.json"),
                               PlanOptions(dataset="folder"), spec=spec))
    assert desc["Name"] == ("My study", "stated")
    assert desc["License"] == ("CC0", "stated")
    assert desc["Authors"] == (["Ada Lovelace", "Alan Turing"], "stated")
    cff = approximate(df, plan_output(df, options=PlanOptions(citation=True)).find("CITATION.cff"),
                      PlanOptions(citation=True), spec=spec)
    assert "Lovelace" in cff.text and "cff-version" in cff.text


def test_companion_stubs_follow_the_setting() -> None:
    """With "Generate missing companion files" on, a BOLD run's events table
    is a TODO stub, and MNE-BIDS' sometimes-written one is always there."""
    rows = [_row("sub-01", "sub-01_task-rest_bold", "func", uid="u1", task="rest"),
            _row("sub-01", "sub-01_task-rest_sbref", "func", uid="u2", task="rest"),
            _row("sub-01", "sub-01_task-x_eeg", "eeg", source="r.edf", modality="eeg")]
    df = pd.DataFrame(rows)
    off = plan_output(df)
    assert "sub-01/func/sub-01_task-rest_events.tsv" not in _paths(off)
    on = plan_output(df, options=PlanOptions(companions=True))
    stub = on.find("sub-01/func/sub-01_task-rest_events.tsv")
    assert stub is not None and stub.stub == "events" and not stub.conditional
    assert sum(f.path == stub.path for f in on.files) == 1, "the BOLD and its SBRef share it"
    content = approximate(df, stub, PlanOptions(companions=True))
    assert content.header == ("onset", "duration", "trial_type")
    assert content.rows == (("TODO", "TODO", "TODO"),)
    eeg = on.find("sub-01/eeg/sub-01_task-x_events.tsv")
    assert not eeg.conditional and "stub" in eeg.origin
    assert not plan_output(df, options=PlanOptions(companions=True, run_metadata=False)) \
        .find(stub.path), "the companion step runs with the metadata step"


def test_attached_files_land_beside_their_recording(tmp_path) -> None:
    log = tmp_path / "behaviour.tsv"
    log.write_text("onset\tduration\n")
    attached = json.dumps([{"suffix": "events", "path": str(log)},
                           {"suffix": "beh", "path": str(tmp_path / "answers.tsv")}])
    rows = [_row("sub-01", "sub-01_task-x_eeg", "eeg", source="r.edf", modality="eeg",
                 companion_files=attached)]
    plan = plan_output(pd.DataFrame(rows))
    events = plan.find("sub-01/eeg/sub-01_task-x_events.tsv")
    assert events.origin.startswith("BIDS Manager (your file") and not events.conditional, \
        "the user's events table replaces the one MNE-BIDS writes"
    assert plan.find("sub-01/eeg/sub-01_task-x_beh.tsv") is not None


def test_mne_bids_names_are_shown_as_written() -> None:
    rows = [_row("sub-01", "sub-01_task-rest_acq-high_run-01_eeg", "eeg", source="r.edf",
                 modality="eeg", run="01")]
    plan = plan_output(pd.DataFrame(rows))
    data = plan.find("sub-01/eeg/sub-01_task-rest_run-1_eeg.edf")
    assert data is not None, "MNE-BIDS keeps subject, session, task and run only"
    assert "acq-high left out" in data.renamed and "run-1" in data.renamed
    assert not any("acq-high" in p for p in _paths(plan))


def test_fieldmaps_are_numbered_as_the_conversion_numbers_them() -> None:
    """In group-key order, counting a fieldmap that corrects nothing; the
    opposite run's SBRef joins a one-direction EPI reference."""
    pa = SeriesFeatures(pe_axis="COL", pe_positive=0.0)
    ap = SeriesFeatures(pe_axis="COL", pe_positive=1.0)
    rows = [
        # Corrects nothing (every run is nearer the EPI), but sorts first.
        _row("sub-01", "sub-01_acq-a_magnitude1", "fmap", uid="m1|p1", acq_time="100000"),
        _row("sub-01", "sub-01_acq-z_dir-PA_epi", "fmap", uid="e1", acq_time="100500"),
        _row("sub-01", "sub-01_task-rest_sbref", "func", uid="s1", task="rest",
             acq_time="101000"),
        _row("sub-01", "sub-01_task-rest_bold", "func", uid="b1", task="rest",
             acq_time="101005"),
    ]
    df = pd.DataFrame(rows)
    feats = {"e1": pa, "b1": ap, "s1": ap}
    plan = plan_output(df, features=feats)

    def links(path):
        got = _fields(approximate(df, plan.find(path), PlanOptions(fill_scope="none"), feats))
        return {k: v[0] for k, v in got.items()
                if k in ("IntendedFor", "B0FieldIdentifier", "B0FieldSource")}

    assert links("sub-01/fmap/sub-01_acq-a_magnitude1.json") == {}, "nothing to correct"
    assert links("sub-01/fmap/sub-01_acq-z_dir-PA_epi.json")["B0FieldIdentifier"] == "fmap2", \
        "the unused fieldmap still counts"
    assert links("sub-01/func/sub-01_task-rest_bold.json") == {"B0FieldSource": "fmap2"}
    assert links("sub-01/func/sub-01_task-rest_sbref.json") == {"B0FieldIdentifier": "fmap2"}, \
        "the SBRef gives the second direction"


def test_pet_shows_pet_fields_and_the_dose_you_stated() -> None:
    rows = [_row("sub-01", "sub-01_pet", "pet", uid="u1", modality="pet",
                 tracer_suggestion="FDG", radionuclide_suggestion="F18",
                 injected_dose_suggestion="75.85 MBq")]
    df = pd.DataFrame(rows)
    spec = _spec(pet_defaults={"injected_mass": 181149988.37, "injected_mass_units": "ug"})
    plan = plan_output(df)
    side = _fields(approximate(df, plan.find("sub-01/pet/sub-01_pet.json"),
                               features={"u1": SeriesFeatures(manufacturer="GE",
                                                              dicom_modality="PT")},
                               spec=spec))
    assert "Modality" not in side and "MagneticFieldStrength" not in side
    assert side["TracerName"] == ("FDG", "scan")
    assert side["InjectedRadioactivity"] == (75.85, "scan")
    assert side["InjectedMass"] == (181149988.37, "stated")


def test_defacing_records_itself_in_the_sidecar() -> None:
    rows = [_row("sub-01", "sub-01_T1w", "anat", uid="u1")]
    df = pd.DataFrame(rows)
    plan = plan_output(df)
    feats = {"u1": SeriesFeatures(n_volumes=1)}
    on = _fields(approximate(df, plan.find("sub-01/anat/sub-01_T1w.json"),
                             PlanOptions(deface=True), feats))
    assert on["DeidentificationMethodCodeSequence"][1] == "settings"
    off = _fields(approximate(df, plan.find("sub-01/anat/sub-01_T1w.json"), PlanOptions(), feats))
    assert "DeidentificationMethod" not in off


def test_participants_show_what_identifies_them(tmp_path) -> None:
    sheet = tmp_path / "participants.csv"
    sheet.write_text("participant_id,age,group\nsub-01,30,control\n")
    rows = [_row("sub-01", "sub-01_T1w", "anat", uid="u1", GivenName="Ada",
                 PatientSex="F")]
    df = pd.DataFrame(rows)
    spec = _spec(participants_file="participants.csv")
    plan = plan_output(df)
    table = approximate(df, plan.find("participants.tsv"),
                        PlanOptions(inventory_dir=str(tmp_path)), spec=spec)
    row = dict(zip(table.header, table.rows[0]))
    assert row["age"] == "30" and row["group"] == "control", "the participants file"
    assert row["given_name"] == "Ada"
    assert "identify the participants" in table.note

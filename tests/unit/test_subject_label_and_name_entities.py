"""Two scan options: subject labels from the DICOM identity, and entities the
sequence name decides.

Subject labels: numbers by default; or the patient ID, given or family name,
curated into a BIDS label (accents, spaces and punctuation go), numbered where
empty or anonymised, kept apart when two values match, and never taking a
number a curated label already uses.

Entities from names: ``key-value`` tokens of the series description (else the
protocol name), valid values only, allowed entities only, winning over the
scan's estimate for the entities chosen, the run numbering included.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from bidsmgr.inventory import name_entities, subject_label
from bidsmgr.inventory.mri_dicom import scan_dicoms_long

# ---------------------------------------------------------------------------
# Subject labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, label", [
    ("OL_0001", "OL0001"),
    ("Müller", "Muller"),
    ("Smith Jr.", "SmithJr"),
    ("  ", ""),
    ("éàç-123", "eac123"),
    (None, ""),
])
def test_values_are_curated_into_bids_labels(raw, label) -> None:
    assert subject_label.curate(raw) == label


def test_numbers_by_default() -> None:
    clusters = {"a": [("P1", "Ann", "Lee")], "b": [("P2", "Bob", "Ray")]}
    assert subject_label.assign(clusters) == {"a": "sub-001", "b": "sub-002"}


def test_a_field_gives_the_label_and_empty_ones_are_numbered() -> None:
    clusters = {
        "a": [("OL_0001", "Ann", "Lee")],
        "b": [("", "Bob", "Ray")],             # no patient ID
        "c": [("ANONYMOUS", "Cy", "Fox")],     # an anonymisation stamp
        "d": [("003", "Di", "May")],           # a curated label that is a number
    }
    labels = subject_label.assign(clusters, subject_label.PATIENT_ID)
    assert labels["a"] == "sub-OL0001" and labels["d"] == "sub-003"
    # Numbered, never with a number a curated label already uses.
    assert labels["b"] == "sub-001" and labels["c"] == "sub-002"
    assert len(set(labels.values())) == 4


def test_two_subjects_with_one_value_are_kept_apart() -> None:
    clusters = {"a": [("1", "Ann", "Smith")], "b": [("2", "Bea", "Smith")],
                "c": [("3", "Cy", "smith")]}
    labels = subject_label.assign(clusters, subject_label.FAMILY_NAME)
    assert sorted(labels.values()) == ["sub-Smith", "sub-Smith2", "sub-smith3"]


def test_the_first_usable_member_value_is_taken() -> None:
    clusters = {"a": [("", "", ""), ("P9", "Eve", "Ng")]}
    assert subject_label.assign(clusters, subject_label.GIVEN_NAME) == {"a": "sub-Eve"}


def test_an_unknown_source_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown subject label source"):
        subject_label.assign({"a": []}, "nickname")


def _dicom(path: Path, *, pid: str, name: str, desc: str, series: str,
           protocol: str = "", study: str = "") -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = pid
    ds.PatientName = name
    ds.Modality = "MR"
    ds.SeriesDescription = desc
    if protocol:
        ds.ProtocolName = protocol
    ds.SeriesInstanceUID = series
    ds.StudyInstanceUID = study or series + ".9"
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
    ds.Rows = ds.Columns = 4
    ds.BitsAllocated = 16
    ds.PixelData = b"\x00\x00" * 16
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def test_a_scan_takes_its_labels_from_the_chosen_field(tmp_path: Path) -> None:
    _dicom(tmp_path / "a/1.dcm", pid="OL_0001", name="Müller^Anna", desc="t1", series="1.1")
    _dicom(tmp_path / "b/1.dcm", pid="OL_0002", name="Lee^Bo", desc="t1", series="1.2")
    _dicom(tmp_path / "c/1.dcm", pid="ANONYMOUS", name="", desc="t1", series="1.3")
    ids = lambda df: sorted(set(df["participant_id"]))  # noqa: E731
    assert ids(scan_dicoms_long(tmp_path)) == ["sub-001", "sub-002", "sub-003"]
    assert ids(scan_dicoms_long(tmp_path, subject_from="patient_id")) == [
        "sub-001", "sub-OL0001", "sub-OL0002"]
    assert ids(scan_dicoms_long(tmp_path, subject_from="family_name")) == [
        "sub-001", "sub-Lee", "sub-Muller"]


def test_the_scan_command_line_hands_both_options_over(monkeypatch, tmp_path) -> None:
    from bidsmgr.cli import scan

    seen = {}
    monkeypatch.setattr(scan, "run_scan", lambda *a, **k: seen.update(k) or pd.DataFrame())
    scan.main([str(tmp_path), str(tmp_path / "inv.tsv"), "--subject-from", "patient-id",
               "--name-entity", "acq,task", "--name-entity", "run", "--no-log-file"])
    assert seen["subject_from"] == "patient_id"
    assert seen["name_entities"] == ["task", "acq", "run"]


def test_the_scan_command_line_refuses_an_unknown_entity(tmp_path, capsys) -> None:
    from bidsmgr.cli import scan

    with pytest.raises(SystemExit):
        scan.main([str(tmp_path), str(tmp_path / "inv.tsv"), "--name-entity", "sub",
                   "--no-log-file"])
    assert "not an entity a name can decide" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Entities from the sequence name
# ---------------------------------------------------------------------------


def test_tokens_are_read_and_checked() -> None:
    assert name_entities.from_name("func_task-nback_acq-mb4_run-02 dir-AP") == {
        "task": "nback", "acq": "mb4", "run": "02", "dir": "AP"}
    # Not valid for the entity: a run is a number, part has four values,
    # a label has no punctuation.
    assert name_entities.from_name("run-two_part-amplitude_acq-") == {}
    assert name_entities.from_name("x_part-mag") == {"part": "mag"}
    # The first text that states an entity wins.
    assert name_entities.from_name("acq-desc", "acq-proto_task-x") == {
        "acq": "desc", "task": "x"}


def test_only_the_chosen_entities_and_only_where_allowed() -> None:
    entities = {"task": "rest", "acquisition": "epfid2p2", "run": "1"}
    out, changed = name_entities.apply(entities, "func", "bold", ["acq"],
                                       "func_task-nback_acq-mb4_run-02")
    assert out == {"task": "rest", "acquisition": "mb4", "run": "1"}
    assert changed == {"acquisition": "mb4"}
    # dir is not an entity of a T1w: not added even when chosen.
    out, changed = name_entities.apply({}, "anat", "T1w", ["dir", "acq"], "T1w_dir-AP_acq-hr")
    assert out == {"acquisition": "hr"} and "direction" not in out


def test_parse_list_refuses_what_a_name_cannot_decide() -> None:
    assert name_entities.parse_list("acq, task,run") == ["task", "acq", "run"]
    with pytest.raises(ValueError, match="sub"):
        name_entities.parse_list("acq,sub")


def test_a_scan_lets_the_name_win(tmp_path: Path) -> None:
    from bidsmgr.cli.scan import run_scan

    for n in range(2):
        _dicom(tmp_path / f"raw/bold{n}/{n}.dcm", pid="P1", name="Lee^Bo",
               desc=f"func_task-nback_acq-mb4_run-0{n + 5}_bold", series=f"1.{n + 1}",
               study="1.7")
    _dicom(tmp_path / "raw/t1/0.dcm", pid="P1", name="Lee^Bo", desc="T1w",
           protocol="T1w_acq-highres", series="1.9", study="1.7")

    def names(**kw):
        df = run_scan(tmp_path / "raw", tmp_path / f"inv{len(kw)}.tsv",
                      skip_bids_guess=True, write_log=False, **kw)
        return sorted(df["bids_name"])

    plain = names()
    # Without the option the scan numbers the runs and reads no protocol name.
    assert not any("acq-highres" in n for n in plain)
    assert sorted(n.split("run-")[1].split("_")[0] for n in plain if "_bold" in n) == ["1", "2"]
    decided = names(name_entities=["acq", "run"])
    # The protocol name speaks where the description does not.
    assert any("acq-highres_T1w" in n for n in decided)
    bolds = [n for n in decided if n.endswith("_bold")]
    assert len(bolds) == 2 and all("acq-mb4" in n for n in bolds)
    # The operator's run numbers, not the scan's 1 and 2.
    assert sorted(n.split("run-")[1].split("_")[0] for n in bolds) == ["05", "06"]

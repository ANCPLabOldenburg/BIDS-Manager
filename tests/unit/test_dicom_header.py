"""A DICOM header read for people (inventory/dicom_header)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bidsmgr.inventory import dicom_header as dh
from tests.fixtures.dicoms import write_mr_series

_PROTOCOL = (b"### ASCCONV BEGIN object=MrProtDataImpl@MrProtocolData version=51130001 ###\n"
             b"lRepetitions = 49\n"
             b"alTR[0] = 850000\n"
             b'tSequenceFileName = "%SiemensSeq%\\\\epfid"  # the sequence\n'
             b"sKSpace.ucMultiSliceMode = 0x2\n"
             b"### ASCCONV END ###")


def _series(tmp_path: Path, **kw) -> Path:
    return write_mr_series(tmp_path / "s", "1.2.3.4", **kw)[0]


def _edit(path: Path, fn) -> Path:
    import pydicom

    ds = pydicom.dcmread(str(path))
    fn(ds)
    ds.save_as(str(path))
    return path


def test_grouped_by_module_with_a_summary(tmp_path) -> None:
    header = dh.read_header(_series(tmp_path, description="bold_run1"))
    assert list(header.modules)[:2] == ["Patient", "Study"]
    assert header.summary["Modality"] == "MR"
    assert header.summary["Series"] == "3 · bold_run1"
    assert "SIEMENS" in header.summary["Scanner"]
    assert header.summary["Acquired"] == "2026-01-01 12:00:00"
    assert header.find("SeriesDescription").read_for.startswith("the series' name")
    names = {e.keyword for e in header.modules["Series"]}
    assert {"SeriesDescription", "SeriesInstanceUID"} <= names
    assert all(e.tag.startswith("(0002,") for e in header.modules["File meta"])


def test_identifiers_are_marked_and_counted(tmp_path) -> None:
    header = dh.read_header(_series(tmp_path))
    assert header.find("PatientName").identifies
    assert header.find("PatientID").identifies
    assert not header.find("SeriesDescription").identifies
    assert dh.count_identifying(header) >= 2


def test_identifiers_are_left_out_of_exports_unless_asked(tmp_path) -> None:
    header = dh.read_header(_series(tmp_path))
    text = dh.export_text(header)
    assert "Test^Subject" not in text and "(identifying, left out)" in text
    assert "Test^Subject" in dh.export_text(header, identifiers=True)
    data = json.loads(dh.export_json(header))
    assert "Test^Subject" not in json.dumps(data)
    assert data["summary"]["Modality"] == "MR"
    assert "Test^Subject" in dh.export_json(header, identifiers=True)


def test_a_sequence_unfolds_into_items_and_marks_what_it_holds(tmp_path) -> None:
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    def add(ds) -> None:
        other = Dataset()
        other.PatientID = "OTHER-1"
        ds.OtherPatientIDsSequence = Sequence([other])

    header = dh.read_header(_edit(_series(tmp_path), add))
    seq = header.find("OtherPatientIDsSequence")
    assert seq.value == "1 item(s)"
    assert seq.children[0].name == "Item 1"
    assert seq.identifies and seq.children[0].identifies


def test_the_siemens_protocol_is_decoded_from_a_private_element(tmp_path) -> None:
    def add(ds) -> None:
        block = ds.private_block(0x0021, "SIEMENS MR SDS 01", create=True)
        block.add_new(0xFE, "OB", _PROTOCOL)

    header = dh.read_header(_edit(_series(tmp_path), add))
    private = [e for e in header.modules["Private"] if e.children]
    assert private, "the protocol element unfolds"
    protocol = {c.keyword: c.value for c in private[0].children}
    assert private[0].value == "Siemens protocol (4 entries)"
    assert protocol["alTR[0]"] == "850000"
    assert protocol["tSequenceFileName"] == "%SiemensSeq%\\\\epfid"
    assert protocol["sKSpace.ucMultiSliceMode"] == "0x2"


def test_an_enhanced_file_gets_a_frame_table(tmp_path) -> None:
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    def add(ds) -> None:
        frames = []
        for i in range(3):
            item = Dataset()
            pos = Dataset()
            pos.ImagePositionPatient = [0.0, 0.0, float(i)]
            item.PlanePositionSequence = Sequence([pos])
            echo = Dataset()
            echo.EffectiveEchoTime = 30.0
            item.MREchoSequence = Sequence([echo])
            content = Dataset()
            content.TemporalPositionIndex = i + 1
            item.FrameContentSequence = Sequence([content])
            frames.append(item)
        ds.PerFrameFunctionalGroupsSequence = Sequence(frames)

    header = dh.read_header(_edit(_series(tmp_path), add))
    assert "Enhanced multi-frame" in header.flags
    assert [f["frame"] for f in header.frames] == ["1", "2", "3"]
    assert header.frames[2]["position"] == "0.00 \\ 0.00 \\ 2.00"
    assert header.frames[0]["echo time"] == "30.0"
    assert header.frames[1]["temporal index"] == "2"
    assert "[Frames] 3" in dh.export_text(header)


def test_a_non_image_object_is_flagged(tmp_path) -> None:
    def strip(ds) -> None:
        for name in ("Rows", "Columns", "PixelData"):
            if name in ds:
                delattr(ds, name)

    header = dh.read_header(_edit(_series(tmp_path), strip))
    assert "Non-image object" in header.flags


def test_compare_reports_what_differs_at_any_depth(tmp_path) -> None:
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    a, b = write_mr_series(tmp_path / "s", "1.2.3.4", n_slices=2)

    def seq(te: float):
        def add(ds) -> None:
            echo = Dataset()
            echo.EffectiveEchoTime = te
            item = Dataset()
            item.MREchoSequence = Sequence([echo])
            ds.SharedFunctionalGroupsSequence = Sequence([item])
            ds.SeriesDescription = "one" if te < 20 else "two"
        return add

    ha = dh.read_header(_edit(a, seq(10.0)))
    hb = dh.read_header(_edit(b, seq(30.0)))
    rows = dh.compare(ha, hb)
    by = {r.where: r for r in rows}
    assert by["Series Description"].a == "one" and by["Series Description"].b == "two"
    deep = [r for r in rows if r.where.endswith("Effective Echo Time")]
    assert deep and deep[0].a == "10.0" and deep[0].b == "30.0"
    assert "Item 1" in deep[0].where
    # The holding sequence is not listed as well: what differs inside is.
    assert not any(r.where == "Shared Functional Groups Sequence" for r in rows)
    sop = by["SOP Instance UID"]
    assert sop.routine and not by["Series Description"].routine


def test_compare_lists_an_element_present_in_one_only(tmp_path) -> None:
    a, b = write_mr_series(tmp_path / "s", "1.2.3.4", n_slices=2)
    _edit(b, lambda ds: setattr(ds, "ProtocolName", "changed"))
    _edit(b, lambda ds: setattr(ds, "BodyPartExamined", "BRAIN"))
    rows = {r.where: r for r in dh.compare(dh.read_header(a), dh.read_header(b))}
    assert rows["Body Part Examined"].a is None and rows["Body Part Examined"].b == "BRAIN"


def test_pixels_are_never_read(tmp_path, monkeypatch) -> None:
    import pydicom

    calls = []
    real = pydicom.dcmread

    def spy(*args, **kw):
        calls.append(kw)
        return real(*args, **kw)

    monkeypatch.setattr(pydicom, "dcmread", spy)
    dh.read_header(_series(tmp_path))
    assert calls and all(c.get("stop_before_pixels") for c in calls)


def test_routing_is_by_content(tmp_path) -> None:
    from bidsmgr.viz.data.formats import document_kind

    first = _series(tmp_path)
    ima = first.with_name("IM_0001.IMA")
    bare = first.with_name("IM0002")
    uid_name = first.with_name("1.3.46.670589.28.1")
    for target in (ima, bare, uid_name):
        target.write_bytes(first.read_bytes())
        assert document_kind(target) == "dicom", target.name
    text = tmp_path / "notes"
    text.write_text("not a dicom, just text that is long enough " * 10)
    assert document_kind(text) != "dicom"


_REAL = os.environ.get("BIDS_MANAGER_REAL_MRI_DATA") == "1"
_RAW = Path("/Users/karelo/Development/datasets/BIDS_Manager/raw_data")


@pytest.mark.skipif(not _REAL, reason="real MRI data not enabled")
def test_real_siemens_headers_decode() -> None:
    from bidsmgr.inventory.mri_dicom import is_dicom_file

    found = 0
    for folder in sorted(_RAW.glob("**/")):
        files = [p for p in sorted(folder.iterdir())[:3] if p.is_file() and is_dicom_file(str(p))]
        if not files:
            continue
        header = dh.read_header(files[0])
        if "siemens" not in header.summary.get("Scanner", "").lower():
            continue
        decoded = [e for e in header.elements() if e.value.startswith(("Siemens CSA",
                                                                       "Siemens protocol"))]
        assert decoded, f"{files[0]}: no Siemens private header decoded"
        found += 1
        if found >= 5:
            break
    assert found, "no Siemens DICOM in the real data"

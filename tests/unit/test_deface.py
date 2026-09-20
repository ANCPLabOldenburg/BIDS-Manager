"""The defacing engine, without running a binary where that is avoidable.

Three layers are tested separately because they fail separately: the table of
engines (pure data), the header probe and the selector (pure functions over
files), and the dataset apply (which needs the real binary and is marked).
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import pytest

from bidsmgr.deface import engines, probe, select, status
from bidsmgr.deface.engines import ALLINEATE, ALLINEATE_AFNI, ALLINEATE_ROBUST


# ---------------------------------------------------------------- the table

def test_every_niimath_engine_builds_the_argv_niimath_expects():
    """Both kinds are the SAME flag with a different mask.

    niimath's own help says so: "skull-stripping: use -deface with a brain
    mask". The mask decides what survives, so a face mask removes the face
    and a brain mask keeps only the brain.
    """
    for eng in engines.ENGINES:
        if eng.backend != engines.BACKEND_NIIMATH:
            continue
        argv = eng.argv(Path("in.nii.gz"), Path("out.nii.gz"))
        assert argv[0] == "in.nii.gz"
        assert argv[-1] == "out.nii.gz"
        assert "-deface" in argv
        # template and mask, in that order, straight after -deface
        i = argv.index("-deface")
        assert argv[i + 1].endswith("avg152T1.nii.gz")
        expected = "avg152T1brainmask.nii.gz" if eng.is_strip else "avg152T1mask.nii.gz"
        assert argv[i + 2].endswith(expected)


def test_the_two_kinds_use_opposite_masks():
    """Getting this backwards would keep the face and delete the brain."""
    face = ALLINEATE.argv(Path("a"), Path("b"))
    brain = engines.STRIP_ATLAS.argv(Path("a"), Path("b"))
    assert face[face.index("-deface") + 2].endswith("avg152T1mask.nii.gz")
    assert brain[brain.index("-deface") + 2].endswith("avg152T1brainmask.nii.gz")


def test_a_deface_dropdown_is_not_offered_a_skull_stripper():
    """Offering one where the other is meant deletes the skull by accident."""
    assert engines.engine_ids(engines.KIND_DEFACE) == [
        "allineate", "allineate-robust", "allineate-afni",
    ]
    assert engines.engine_ids(engines.KIND_STRIP) == ["mindgrab", "strip-atlas"]
    assert all(not e.is_strip for e in engines.DEFACE_ENGINES)
    assert all(e.is_strip for e in engines.STRIP_ENGINES)


def test_the_sidecar_code_says_which_operation_it_was():
    """A stripped image and a defaced one must not record the same thing."""
    assert "DEFACE" in ALLINEATE.code_value
    assert "STRIP" in engines.MINDGRAB.code_value
    assert "skull strip" in engines.MINDGRAB.deid_entry()["CodeMeaning"]
    assert "deface" in ALLINEATE.deid_entry()["CodeMeaning"]


def test_robustfov_comes_before_deface_not_after():
    """Order is the whole point: crop first, then register what is left."""
    argv = ALLINEATE_ROBUST.argv(Path("in.nii.gz"), Path("out.nii.gz"))
    assert argv.index("-robustfov") < argv.index("-deface")


def test_the_cost_engine_passes_a_cost_and_the_others_do_not():
    assert "-cost" in ALLINEATE_AFNI.argv(Path("a"), Path("b"))
    assert "-cost" not in ALLINEATE.argv(Path("a"), Path("b"))


def test_code_values_are_unique_and_carry_the_revision():
    codes = [e.code_value for e in engines.ENGINES]
    assert len(set(codes)) == len(codes)
    assert all(c.endswith(engines.ENGINE_REVISION) for c in codes)


def test_unknown_engine_names_the_ones_that_exist():
    with pytest.raises(KeyError) as exc:
        engines.engine("pydeface")
    assert "allineate" in str(exc.value)


# ---------------------------------------------------------------- the probe

def _nifti(path: Path, dim, *, gz=True, endian="<", sizeof=348):
    """A 348-byte NIfTI-1 header with dim[] set. No image data needed."""
    raw = bytearray(352)
    struct.pack_into(f"{endian}i", raw, 0, sizeof)
    struct.pack_into(f"{endian}8h", raw, 40, *dim)
    data = bytes(raw)
    if gz:
        path.write_bytes(gzip.compress(data))
    else:
        path.write_bytes(data)
    return path


def test_a_three_dimensional_image_is_a_single_volume(tmp_path):
    p = _nifti(tmp_path / "a.nii.gz", (3, 224, 320, 320, 1, 1, 1, 1))
    assert probe.read_dimensions(p).is_3d
    assert probe.is_single_volume(p)


def test_rank_four_with_one_volume_is_still_a_single_volume(tmp_path):
    """The case a stricter gate gets wrong.

    Writing dim[0] = 4 and dim[4] = 1 is legal and means one volume. Real
    anatomical images do it, so refusing on rank alone refuses data that
    defaces perfectly well.
    """
    p = _nifti(tmp_path / "a.nii.gz", (4, 224, 320, 320, 1, 1, 1, 1))
    assert probe.read_dimensions(p).is_3d


def test_a_time_series_is_not(tmp_path):
    p = _nifti(tmp_path / "bold.nii.gz", (4, 64, 64, 36, 200, 1, 1, 1))
    dims = probe.read_dimensions(p)
    assert not dims.is_3d
    assert dims.dim4 == 200


def test_uncompressed_nifti_reads_too(tmp_path):
    p = _nifti(tmp_path / "a.nii", (3, 8, 8, 8, 1, 1, 1, 1), gz=False)
    assert probe.read_dimensions(p).shape == (8, 8, 8)


def test_big_endian_headers_read(tmp_path):
    p = _nifti(tmp_path / "be.nii.gz", (3, 9, 9, 9, 1, 1, 1, 1), endian=">")
    assert probe.read_dimensions(p).shape == (9, 9, 9)


def test_nifti2_is_refused_by_name(tmp_path):
    p = _nifti(tmp_path / "n2.nii.gz", (3, 8, 8, 8, 1, 1, 1, 1), sizeof=540)
    with pytest.raises(probe.NotNifti) as exc:
        probe.read_dimensions(p)
    assert "NIfTI-2" in str(exc.value)


def test_a_file_that_is_not_a_nifti_fails_closed(tmp_path):
    p = tmp_path / "notes.nii.gz"
    p.write_bytes(gzip.compress(b"this is not an image"))
    assert probe.is_single_volume(p) is False


# ------------------------------------------------------------- the sidecar

def test_recording_an_engine_writes_both_bids_fields():
    out = status.record({}, ALLINEATE)
    assert out[status.CODE_SEQUENCE][0]["CodingSchemeDesignator"] == "BIDSManager"
    assert out[status.METHOD][0].startswith("BIDS Manager")


def test_another_tools_entry_is_never_touched():
    theirs = {"CodingSchemeDesignator": "DCM", "CodeValue": "113101"}
    before = {status.CODE_SEQUENCE: [theirs]}

    after = status.record(before, ALLINEATE)
    assert theirs in after[status.CODE_SEQUENCE]
    assert len(after[status.CODE_SEQUENCE]) == 2

    reverted = status.clear(after)
    assert reverted[status.CODE_SEQUENCE] == [theirs]


def test_defacing_twice_replaces_our_entry_rather_than_stacking():
    once = status.record({}, ALLINEATE)
    twice = status.record(once, ALLINEATE_ROBUST)
    ours = [e for e in twice[status.CODE_SEQUENCE]
            if e["CodingSchemeDesignator"] == "BIDSManager"]
    assert len(ours) == 1
    assert status.defaced_by_us(twice) is ALLINEATE_ROBUST


def test_clearing_the_last_entry_removes_the_array(tmp_path):
    """Not an empty list.

    ``[]`` asserts that deidentification was considered and produced nothing,
    which is a different claim from a file that was never defaced.
    """
    out = status.clear(status.record({}, ALLINEATE))
    assert status.CODE_SEQUENCE not in out
    assert status.METHOD not in out


def test_an_unknown_code_value_still_reads_as_defaced_by_us():
    """A dataset defaced by a future version, opened by this one."""
    sidecar = {status.CODE_SEQUENCE: [{
        "CodingSchemeDesignator": "BIDSManager",
        "CodeValue": "BIDSMANAGER-DEFACE-SOMETHING-v9",
    }]}
    assert status.defaced_by_us(sidecar) is None       # engine unknown
    assert status.defaced_by_us_at_all(sidecar) is True  # but it was us


@pytest.mark.parametrize(
    "name, expected",
    [("sub-01_T1w.nii.gz", "sub-01_T1w.json"),
     ("sub-01_T1w.nii", "sub-01_T1w.json")],
)
def test_the_sidecar_is_found_through_a_compound_extension(name, expected):
    assert status.sidecar_for(Path("/d") / name).name == expected


# ------------------------------------------------------------- the selector

def _dataset(tmp_path):
    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "sub-01" / "func").mkdir(parents=True)
    (root / "derivatives" / "fmriprep" / "sub-01" / "anat").mkdir(parents=True)
    (root / ".bidsmgr").mkdir(parents=True)
    (root / "dataset_description.json").write_text("{}")
    return root


def test_the_selector_finds_anatomy_and_says_why_it_skipped_the_rest(tmp_path):
    root = _dataset(tmp_path)
    _nifti(root / "sub-01/anat/sub-01_T1w.nii.gz", (3, 8, 8, 8, 1, 1, 1, 1))
    _nifti(root / "sub-01/func/sub-01_task-x_bold.nii.gz",
           (4, 8, 8, 8, 40, 1, 1, 1))
    _nifti(root / "derivatives/fmriprep/sub-01/anat/sub-01_desc-p_T1w.nii.gz",
           (3, 8, 8, 8, 1, 1, 1, 1))

    sel = select.walk(root)
    assert [c.relative for c in sel.candidates] == ["sub-01/anat/sub-01_T1w.nii.gz"]

    reasons = {s.relative: s.reason for s in sel.skipped}
    assert reasons["sub-01/func/sub-01_task-x_bold.nii.gz"] is select.Skip.DATATYPE
    assert reasons[
        "derivatives/fmriprep/sub-01/anat/sub-01_desc-p_T1w.nii.gz"
    ] is select.Skip.LOCATION


def test_a_four_d_anatomical_is_skipped_as_four_d_not_silently(tmp_path):
    root = _dataset(tmp_path)
    _nifti(root / "sub-01/anat/sub-01_MEGRE.nii.gz", (4, 8, 8, 8, 12, 1, 1, 1))
    sel = select.walk(root)
    assert not sel.candidates
    assert sel.skipped[0].reason is select.Skip.FOUR_D
    assert "12 volumes" in sel.skipped[0].detail


def test_relative_paths_are_posix(tmp_path):
    """They become dict keys and dialog lines. See CROSS_PLATFORM_RULES 1.1."""
    root = _dataset(tmp_path)
    _nifti(root / "sub-01/anat/sub-01_T1w.nii.gz", (3, 8, 8, 8, 1, 1, 1, 1))
    rel = select.walk(root).candidates[0].relative
    assert "\\" not in rel
    assert rel == "sub-01/anat/sub-01_T1w.nii.gz"


def test_a_previously_defaced_image_is_still_a_candidate_and_says_so(tmp_path):
    root = _dataset(tmp_path)
    img = _nifti(root / "sub-01/anat/sub-01_T1w.nii.gz", (3, 8, 8, 8, 1, 1, 1, 1))
    status.sidecar_for(img).write_text(json.dumps(status.record({}, ALLINEATE)))

    cand = select.walk(root).candidates[0]
    assert cand.previous_engine == "allineate"


def test_another_tools_deidentification_is_surfaced_not_refused(tmp_path):
    root = _dataset(tmp_path)
    img = _nifti(root / "sub-01/anat/sub-01_T1w.nii.gz", (3, 8, 8, 8, 1, 1, 1, 1))
    status.sidecar_for(img).write_text(json.dumps({
        status.CODE_SEQUENCE: [
            {"CodingSchemeDesignator": "DCM", "CodeMeaning": "Basic Profile"},
        ]
    }))
    cand = select.walk(root).candidates[0]
    assert cand.foreign_methods == ("Basic Profile",)


def test_targets_narrow_the_walk(tmp_path):
    root = _dataset(tmp_path)
    (root / "sub-02" / "anat").mkdir(parents=True)
    _nifti(root / "sub-01/anat/sub-01_T1w.nii.gz", (3, 8, 8, 8, 1, 1, 1, 1))
    _nifti(root / "sub-02/anat/sub-02_T1w.nii.gz", (3, 8, 8, 8, 1, 1, 1, 1))

    sel = select.walk(root, [root / "sub-02"])
    assert [c.relative for c in sel.candidates] == ["sub-02/anat/sub-02_T1w.nii.gz"]


def test_dot_folders_are_never_walked(tmp_path):
    root = _dataset(tmp_path)
    (root / ".bidsmgr" / "editor" / "originals" / "ab" / "sub-01" / "anat").mkdir(
        parents=True
    )
    _nifti(
        root / ".bidsmgr/editor/originals/ab/sub-01/anat/sub-01_T1w.nii.gz",
        (3, 8, 8, 8, 1, 1, 1, 1),
    )
    sel = select.walk(root)
    assert not sel.candidates
    assert all(s.reason is select.Skip.LOCATION for s in sel.skipped)

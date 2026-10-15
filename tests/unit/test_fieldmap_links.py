"""Which runs a fieldmap corrects, and how it says so.

Measured 2026-10-12 with sdcflows (what fMRIPrep uses): BIDS-URI
IntendedFor is silently ignored by sdcflows 2.5 to 2.10 (fMRIPrep 23.1 to
24.1), subject-relative paths with the session and B0FieldIdentifier /
B0FieldSource are read by every version. So both are written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bidsmgr.fixups.intended_for import (
    Fieldmap, Run, bind, group_key, plan_links, populate_intended_for, serves_of,
)


def _pair(folder: Path, stem: str, **meta) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.nii.gz").write_bytes(b"\x1f\x8b")
    (folder / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder / f"{stem}.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_time_takes_the_last_fieldmap_before_and_else_the_first_after() -> None:
    fmaps = [Fieldmap("a", time=100.0), Fieldmap("b", time=300.0)]
    runs = [Run("early", "func", time=50.0), Run("mid", "func", time=200.0),
            Run("late", "func", time=400.0)]
    out, why = bind(fmaps, runs)
    assert out == {"a": ["early", "mid"], "b": ["late"]}
    assert "acquisition times" in why


def test_the_same_shim_beats_time() -> None:
    """Re-shimmed between two runs: the fieldmap measured at the run's own
    shim is the right one, even if another was acquired closer."""
    shim_a, shim_b = (1.0, 2.0, 3.0), (1.5, 2.0, 3.0)
    fmaps = [Fieldmap("a", time=100.0, shim=shim_a), Fieldmap("b", time=190.0, shim=shim_b)]
    out, why = bind(fmaps, [Run("r", "func", time=200.0, shim=shim_a)])
    assert out["a"] == ["r"] and out["b"] == []
    assert "shim" in why


def test_a_diffusion_reference_only_serves_diffusion() -> None:
    fmaps = [Fieldmap("dwiref", time=100.0, serves="dwi"), Fieldmap("gre", time=50.0)]
    runs = [Run("bold", "func", time=150.0), Run("dwi", "dwi", time=160.0)]
    out, _ = bind(fmaps, runs)
    assert out["dwiref"] == ["dwi"] and out["gre"] == ["bold"]


def test_no_times_offers_every_run_and_says_so() -> None:
    out, why = bind([Fieldmap("a"), Fieldmap("b")], [Run("r", "func")])
    assert out == {"a": ["r"], "b": ["r"]} and "lack of acquisition times" in why


def test_what_a_fieldmap_was_recorded_for() -> None:
    assert serves_of({"SequenceName": "*ep_b0"}) == "dwi"
    assert serves_of({}, has_gradients=True) == "dwi"
    assert serves_of({"PulseSequenceDetails": "%CustomerSeq%_cmrr_mbep2d_bold"}) == "func"
    assert serves_of({"SequenceName": "*fm2d2r"}) is None, "a gradient echo serves all"


# ---------------------------------------------------------------------------
# Writing it
# ---------------------------------------------------------------------------


def test_both_links_are_written_with_the_session(tmp_path) -> None:
    ses = tmp_path / "ses-pre"
    for s in ("magnitude1", "magnitude2", "phasediff"):
        _pair(ses / "fmap", f"sub-01_ses-pre_{s}", AcquisitionTime="10:00:00")
    _pair(ses / "func", "sub-01_ses-pre_task-rest_bold", AcquisitionTime="10:05:00")
    _pair(ses / "dwi", "sub-01_ses-pre_dwi", AcquisitionTime="10:20:00")
    n = populate_intended_for(tmp_path, subject="01", form="relative")
    assert n == 3
    meta = _read(ses / "fmap" / "sub-01_ses-pre_phasediff.json")
    assert meta["IntendedFor"] == ["ses-pre/dwi/sub-01_ses-pre_dwi.nii.gz",
                                   "ses-pre/func/sub-01_ses-pre_task-rest_bold.nii.gz"]
    assert meta["B0FieldIdentifier"] == "fmappre1"
    assert _read(ses / "func" / "sub-01_ses-pre_task-rest_bold.json")["B0FieldSource"] == \
        "fmappre1"
    assert _read(ses / "dwi" / "sub-01_ses-pre_dwi.json")["B0FieldSource"] == "fmappre1"


def test_uris_and_no_b0_field(tmp_path) -> None:
    _pair(tmp_path / "fmap", "sub-01_phasediff", AcquisitionTime="10:00:00")
    _pair(tmp_path / "func", "sub-01_task-rest_bold", AcquisitionTime="10:05:00")
    populate_intended_for(tmp_path, subject="01", b0_field=False)
    meta = _read(tmp_path / "fmap" / "sub-01_phasediff.json")
    assert meta["IntendedFor"] == ["bids::sub-01/func/sub-01_task-rest_bold.nii.gz"]
    assert "B0FieldIdentifier" not in meta
    assert "B0FieldSource" not in _read(tmp_path / "func" / "sub-01_task-rest_bold.json")


def test_the_users_choice_wins(tmp_path) -> None:
    _pair(tmp_path / "fmap", "sub-01_acq-a_phasediff", AcquisitionTime="10:00:00")
    _pair(tmp_path / "fmap", "sub-01_acq-b_phasediff", AcquisitionTime="11:00:00")
    _pair(tmp_path / "func", "sub-01_task-x_bold", AcquisitionTime="10:05:00")
    _pair(tmp_path / "func", "sub-01_task-y_bold", AcquisitionTime="11:05:00")
    populate_intended_for(tmp_path, subject="01", links={
        "sub-01_acq-a_phasediff.nii.gz": ["sub-01_task-x_bold.nii.gz",
                                          "sub-01_task-y_bold.nii.gz"]})
    a = _read(tmp_path / "fmap" / "sub-01_acq-a_phasediff.json")["IntendedFor"]
    b = _read(tmp_path / "fmap" / "sub-01_acq-b_phasediff.json")["IntendedFor"]
    assert len(a) == 2 and b == ["bids::sub-01/func/sub-01_task-y_bold.nii.gz"]
    sources = _read(tmp_path / "func" / "sub-01_task-y_bold.json")["B0FieldSource"]
    assert sources == ["fmap1", "fmap2"], "a run two fieldmaps claim lists both"


def test_one_acquisition_is_one_fieldmap() -> None:
    """Magnitude and phase, the parts of an EPI reference and both directions
    of a reversed-phase pair are one fieldmap; separate runs are not."""
    assert group_key("sub-01_acq-gre_magnitude1") == group_key("sub-01_acq-gre_phasediff")
    assert group_key("sub-01_acq-gre_fieldmap") == group_key("sub-01_acq-gre_magnitude")
    assert group_key("sub-01_acq-b0_dir-AP_epi") == group_key("sub-01_acq-b0_dir-PA_epi")
    assert group_key("sub-01_dir-AP_run-1_part-mag_epi") == \
        group_key("sub-01_dir-AP_run-1_part-phase_epi") == group_key("sub-01_dir-AP_run-1_epi")
    assert group_key("sub-01_dir-AP_run-1_epi") != group_key("sub-01_dir-AP_run-2_epi")
    assert group_key("sub-01_acq-a_phasediff") != group_key("sub-01_acq-b_phasediff")


def test_a_reversed_phase_pair_is_one_fieldmap(tmp_path) -> None:
    """MRI_sample_2: a b0 phase-encoded each way after the diffusion run.
    Both files carry the link and one identifier, so sdcflows estimates them
    together (they were two fieldmaps and only one was linked)."""
    _pair(tmp_path / "dwi", "sub-01_dir-AP_dwi", AcquisitionTime="11:47:20",
          PhaseEncodingDirection="j-")
    _pair(tmp_path / "fmap", "sub-01_acq-b0_dir-AP_epi", AcquisitionTime="11:47:50",
          PhaseEncodingDirection="j-")
    _pair(tmp_path / "fmap", "sub-01_acq-b0_dir-PA_epi", AcquisitionTime="11:48:22",
          PhaseEncodingDirection="j")
    assert populate_intended_for(tmp_path, subject="01") == 2
    for d in ("AP", "PA"):
        meta = _read(tmp_path / "fmap" / f"sub-01_acq-b0_dir-{d}_epi.json")
        assert meta["IntendedFor"] == ["bids::sub-01/dwi/sub-01_dir-AP_dwi.nii.gz"]
        assert meta["B0FieldIdentifier"] == "fmap1"
    dwi = _read(tmp_path / "dwi" / "sub-01_dir-AP_dwi.json")
    assert dwi["B0FieldSource"] == "fmap1"
    assert "B0FieldIdentifier" not in dwi, "two directions already: nothing joins"


def test_a_one_direction_reference_takes_its_run_as_the_other(tmp_path) -> None:
    """MRI_sample_1: a gradient-echo fieldmap for the BOLD and one PA b0 for
    the AP diffusion. sdcflows drops a one-direction group and, once the
    gradient echo's group works, never falls back to IntendedFor: measured
    on 2.5, 2.10 and 2.15, the diffusion went uncorrected. The run joins the
    reference's group, which gives it its second direction."""
    for s in ("magnitude1", "magnitude2", "phasediff"):
        _pair(tmp_path / "fmap", f"sub-01_{s}", AcquisitionTime="10:15:00")
    _pair(tmp_path / "func", "sub-01_task-rest_bold", AcquisitionTime="10:20:00",
          PhaseEncodingDirection="j-")
    _pair(tmp_path / "fmap", "sub-01_dir-PA_epi", AcquisitionTime="10:37:00",
          PhaseEncodingDirection="j", SequenceName="*ep_b0")
    _pair(tmp_path / "dwi", "sub-01_dir-AP_dwi", AcquisitionTime="10:44:00",
          PhaseEncodingDirection="j-")
    populate_intended_for(tmp_path, subject="01")
    epi = _read(tmp_path / "fmap" / "sub-01_dir-PA_epi.json")
    dwi = _read(tmp_path / "dwi" / "sub-01_dir-AP_dwi.json")
    assert dwi["B0FieldSource"] == epi["B0FieldIdentifier"]
    assert dwi["B0FieldIdentifier"] == epi["B0FieldIdentifier"], "the second direction"
    bold = _read(tmp_path / "func" / "sub-01_task-rest_bold.json")
    assert "B0FieldIdentifier" not in bold, "the gradient echo needs no partner"


def test_the_sbref_gives_the_second_direction_and_phase_images_stay_out(tmp_path) -> None:
    """Shayam: a CMRR reversed-phase reference saved as magnitude and phase,
    before a PA run saved the same way with its SBRef. The SBRef joins (one
    volume, like the reference); a phase image never feeds an estimator, so
    it carries no link sdcflows could pick up."""
    _pair(tmp_path / "fmap", "sub-01_dir-AP_run-1_part-mag_epi", AcquisitionTime="11:46:23",
          PhaseEncodingDirection="j-")
    _pair(tmp_path / "fmap", "sub-01_dir-AP_run-1_part-phase_epi", AcquisitionTime="11:46:25",
          PhaseEncodingDirection="j-")
    _pair(tmp_path / "func", "sub-01_task-x_dir-PA_run-1_sbref", AcquisitionTime="11:48:26",
          PhaseEncodingDirection="j")
    for part, when in (("mag", "11:50:59"), ("phase", "11:53:39")):
        _pair(tmp_path / "func", f"sub-01_task-x_dir-PA_run-1_part-{part}_bold",
              AcquisitionTime=when, PhaseEncodingDirection="j")
    populate_intended_for(tmp_path, subject="01")
    mag = _read(tmp_path / "fmap" / "sub-01_dir-AP_run-1_part-mag_epi.json")
    phase = _read(tmp_path / "fmap" / "sub-01_dir-AP_run-1_part-phase_epi.json")
    assert mag["B0FieldIdentifier"] == "fmap1" and len(mag["IntendedFor"]) == 2
    assert "IntendedFor" not in phase and "B0FieldIdentifier" not in phase
    sbref = _read(tmp_path / "func" / "sub-01_task-x_dir-PA_run-1_sbref.json")
    assert sbref["B0FieldIdentifier"] == "fmap1"
    run = _read(tmp_path / "func" / "sub-01_task-x_dir-PA_run-1_part-mag_bold.json")
    assert run["B0FieldSource"] == "fmap1" and "B0FieldIdentifier" not in run


def test_a_file_that_is_no_fieldmap_takes_no_runs(tmp_path) -> None:
    """dcm2niix's phase of a series mixing magnitude and phase keeps its
    ``_ph`` name; it must not become a fieldmap that takes the runs."""
    _pair(tmp_path / "fmap", "sub-01_dir-AP_epi", AcquisitionTime="10:00:00",
          PhaseEncodingDirection="j-")
    _pair(tmp_path / "fmap", "sub-01_dir-AP_epi_ph", AcquisitionTime="10:30:00")
    _pair(tmp_path / "func", "sub-01_task-x_bold", AcquisitionTime="10:40:00",
          PhaseEncodingDirection="j")
    assert populate_intended_for(tmp_path, subject="01") == 1
    assert "IntendedFor" not in _read(tmp_path / "fmap" / "sub-01_dir-AP_epi_ph.json")
    assert _read(tmp_path / "fmap" / "sub-01_dir-AP_epi.json")["IntendedFor"]


def test_no_run_the_other_way_leaves_the_reference_alone(tmp_path) -> None:
    _pair(tmp_path / "fmap", "sub-01_dir-AP_epi", AcquisitionTime="10:00:00",
          PhaseEncodingDirection="j-")
    _pair(tmp_path / "func", "sub-01_task-x_bold", AcquisitionTime="10:05:00",
          PhaseEncodingDirection="j-")
    populate_intended_for(tmp_path, subject="01")
    assert "B0FieldIdentifier" not in _read(tmp_path / "func" / "sub-01_task-x_bold.json")
    # Linking twice changes nothing (the second pass finds what it wrote).
    _pair(tmp_path / "func", "sub-01_task-y_bold", AcquisitionTime="10:10:00",
          PhaseEncodingDirection="j")
    populate_intended_for(tmp_path, subject="01")
    first = _read(tmp_path / "func" / "sub-01_task-y_bold.json")
    populate_intended_for(tmp_path, subject="01")
    assert _read(tmp_path / "func" / "sub-01_task-y_bold.json") == first
    assert first["B0FieldIdentifier"] == "fmap1"


# ---------------------------------------------------------------------------
# On the inventory, before converting
# ---------------------------------------------------------------------------


def _inv_row(uid, datatype, suffix, time, *, ses="", targets="", seq=""):
    return {"participant_id": "sub-01", "session": ses, "include": 1, "datatype": datatype,
            "bids_guess_suffix": suffix, "bids_name": f"sub-01_acq-{uid}_{suffix}", "acq_time": time,
            "series_uid": uid, "sequence": seq or suffix, "fieldmap_targets": targets,
            "source_file": ""}


def test_the_preview_uses_the_same_rule_and_the_users_choice() -> None:
    df = pd.DataFrame([
        _inv_row("f1", "fmap", "phasediff", "10:00:00"),
        _inv_row("b1", "func", "bold", "10:05:00"),
        _inv_row("f2", "fmap", "phasediff", "11:00:00"),
        _inv_row("b2", "func", "bold", "11:05:00"),
        _inv_row("s1", "func", "sbref", "11:04:00"),
    ])
    plan, reasons = plan_links(df)
    assert plan == {0: [1], 2: [3]}, "an SBRef is not a target"
    df.at[0, "fieldmap_targets"] = json.dumps(["b1", "b2"])
    plan, reasons = plan_links(df)
    assert plan[0] == [1, 3] and reasons[0] == "your own choice"


def test_a_users_choice_reaches_the_converted_files(tmp_path) -> None:
    """From row ids in the inventory to file names after the fieldmap
    renames (``cli.convert._user_fieldmap_links``)."""
    from bidsmgr.cli.convert import _user_fieldmap_links
    from bidsmgr.converter.types import ConvertResult, ConvertTask

    def task(uid, dt, suffix, targets=()):
        return ConvertTask(row_id=uid, series_uid=uid, source_files=(), dataset="d",
                           bids_root=tmp_path, subject="01", datatype=dt, suffix=suffix,
                           basename=f"sub-01_{suffix}", fieldmap_targets=targets)

    staged_fmap = tmp_path / "sub-01_magnitude1_e2_ph.nii.gz"
    final_fmap = tmp_path / "sub-01_phasediff.nii.gz"
    results = [
        ConvertResult(task=task("f1", "fmap", "phasediff", ("b1",)),
                      staged_files=(staged_fmap,), success=True),
        ConvertResult(task=task("b1", "func", "bold"),
                      staged_files=(tmp_path / "sub-01_task-x_bold.nii.gz",
                                    tmp_path / "sub-01_task-x_bold.json"), success=True),
    ]
    links = _user_fieldmap_links(results, {staged_fmap: final_fmap})
    assert links == {"sub-01_phasediff.nii.gz": ["sub-01_task-x_bold.nii.gz"]}


def test_the_preview_groups_an_acquisition_like_the_conversion() -> None:
    """A fieldmap written as two rows (Shayam's B0 map and its magnitude, a
    pair's two directions) is one fieldmap: both rows correct the same runs,
    and a choice on either is the acquisition's."""
    df = pd.DataFrame([
        {**_inv_row("m", "fmap", "magnitude", "11:26:55"), "bids_name": "sub-01_acq-gre_magnitude"},
        {**_inv_row("f", "fmap", "fieldmap", "11:26:55"), "bids_name": "sub-01_acq-gre_fieldmap"},
        _inv_row("b1", "func", "bold", "11:34:15"),
        {**_inv_row("ap", "fmap", "epi", "11:46:11"), "bids_name": "sub-01_dir-AP_epi"},
        {**_inv_row("pa", "fmap", "epi", "11:46:40"), "bids_name": "sub-01_dir-PA_epi"},
        _inv_row("b2", "func", "bold", "11:50:59"),
    ])
    plan, reasons = plan_links(df)
    assert plan[0] == plan[1] == [2]
    assert plan[3] == plan[4] == [5], "the pair, not one direction of it"
    df.at[1, "fieldmap_targets"] = json.dumps(["b1", "b2"])
    plan, reasons = plan_links(df)
    assert plan[0] == plan[1] == [2, 5] and reasons[0] == "your own choice"


def test_the_preview_matches_shims_from_the_scan() -> None:
    """Re-shimmed between two runs: the scan's features carry the shim, and
    the preview matches it the way the conversion matches the sidecars."""
    df = pd.DataFrame([
        _inv_row("f1", "fmap", "phasediff", "10:00:00"),
        _inv_row("f2", "fmap", "phasediff", "10:30:00"),
        _inv_row("b1", "func", "bold", "10:40:00"),
    ])
    assert plan_links(df)[0][1] == [2], "by time: the last before"
    plan, reasons = plan_links(df, shims={"f1": (1.0, 2.0), "f2": (3.0, 4.0), "b1": (1.0, 2.0)})
    assert plan[0] == [2] and plan[1] == [] and "shim" in reasons[0]

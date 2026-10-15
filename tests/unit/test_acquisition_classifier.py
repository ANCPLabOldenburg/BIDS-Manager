"""The parameter-based classifier (classifier/acquisition): features read
from DICOM, scoring, the session context, the decision against dcm2niix's
guess, and how it lands in a scan."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from bidsmgr.classifier.acquisition import config as cfgmod
from bidsmgr.classifier.acquisition import engine
from bidsmgr.classifier.acquisition import features as feat
from bidsmgr.classifier.acquisition.features import SeriesFeatures
from bidsmgr.classifier.types import Classification
from tests.fixtures.dicoms import write_mr_series


def _f(**kw) -> SeriesFeatures:
    base = dict(description="", rows=64.0, columns=64.0, n_slices=30, n_volumes=1,
                n_orientations=1, acquisition_type="2D")
    base.update(kw)
    return SeriesFeatures(**base)


T1 = _f(description="t1_mprage", is_gre=True, is_ir=True, acquisition_type="3D",
        echo_times=[2.3], repetition_time=2.3, inversion_time=0.9, flip_angle=8, n_slices=176)
FLAIR = _f(description="t2_space_dark_fluid", is_se=True, is_ir=True, acquisition_type="3D",
           echo_times=[390.0], repetition_time=5.0, inversion_time=1.8, flip_angle=120)
T2 = _f(description="t2_tse_tra", is_se=True, echo_times=[95.0], repetition_time=6.0,
        flip_angle=150)
BOLD = _f(description="cmrr_mbep2d_bold", is_epi=True, is_gre=True, n_volumes=300,
          echo_times=[30.0], repetition_time=0.8, multiband=6.0, pe_dir="PA", n_slices=72)
DWI = _f(description="ep2d_diff", is_epi=True, is_se=True, n_volumes=64,
         b_values=[0, 1000], n_directions=60, echo_times=[80.0], pe_dir="AP")


def _best(f: SeriesFeatures) -> str:
    return engine.score(f).finish().best


# ---------------------------------------------------------------------------
# Scoring one series
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("features, expected", [
    (T1, "anat/T1w"), (FLAIR, "anat/FLAIR"), (T2, "anat/T2w"),
    (BOLD, "func/bold"), (DWI, "dwi/dwi"),
])
def test_the_common_contrasts(features, expected) -> None:
    verdict = engine.score(features).finish()
    assert verdict.best == expected
    assert verdict.confidence > 0.8


def test_a_3d_phase_difference_is_not_a_map_in_hz() -> None:
    """Shayam's B0 map: a 3-D gradient echo whose phase difference image
    carries ONE echo time (values in +-4096, "phase difference images").
    The protocol acquired two; it was named a field map in Hz, and the
    magnitude series that goes with it (two echoes) was named ``magnitude``
    and converted as ``_magnitude_magnitude1``."""
    common = dict(is_gre=True, acquisition_type="3D", n_slices=44, repetition_time=0.4)
    phase = _f(description="B0_Map", part="phase", image_type=["ORIGINAL", "PRIMARY", "P"],
               echo_times=[3.06], protocol_echoes=2, **common)
    magnitude = _f(description="gre_1441B_B0_4.0mm", part="mag", echo_times=[2.04, 5.1],
                   image_type=["ORIGINAL", "PRIMARY", "M"], protocol_echoes=2, **common)
    assert _best(phase) == "fmap/phasediff"
    assert _best(magnitude) == "fmap/magnitude1"
    # Without the protocol's two echoes, one echo time stays a map in Hz.
    hz = _f(description="B0_Map", part="phase", image_type=["ORIGINAL", "PRIMARY", "P"],
            echo_times=[3.06], **common)
    assert _best(hz) == "fmap/fieldmap"


def test_parameters_win_over_a_wrong_name() -> None:
    """Named T2 by the operator, acquired as an inversion recovery with a
    long inversion: FLAIR."""
    assert _best(replace(FLAIR, description="t2_space")) == "anat/FLAIR"


def test_a_short_epi_series_is_not_a_run() -> None:
    verdict = engine.score(replace(BOLD, n_volumes=4, description="cmrr_mbep2d_REV")).finish()
    assert verdict.best == "fmap/epi"
    assert "func/bold" != verdict.best


def test_the_echo_time_separates_t2_from_proton_density() -> None:
    pd_like = replace(T2, description="tse", echo_times=[15.0])
    assert _best(pd_like) == "anat/PDw"
    between = replace(T2, description="hires t2", echo_times=[44.0], repetition_time=3.5)
    assert _best(between) == "anat/T2w", "a soft ramp: 44 ms leans T2 with the name"


def test_a_localizer_has_several_orientations() -> None:
    loc = _f(description="whatever", is_gre=True, n_orientations=3, n_slices=3,
             echo_times=[3.0], repetition_time=0.008)
    assert _best(loc) == "exclude/localizer"


def test_a_reformat_is_left_out() -> None:
    mpr = _f(description="t1_mpr_sag", is_derived=True,
             image_type=["DERIVED", "PRIMARY", "MPR", "RESAMPLED"], is_gre=True)
    assert _best(mpr) == "exclude/derived"


def test_no_opinion_on_what_is_not_an_image_or_is_a_diffusion_map() -> None:
    assert not engine.score(_f(rows=None, n_volumes=None)).finish().probabilities
    fa = replace(DWI, is_derived=True, image_type=["DERIVED", "PRIMARY", "DIFFUSION", "FA"])
    assert not engine.score(fa).finish().probabilities


def test_missing_evidence_counts_nothing() -> None:
    bare = _f(description="series 7")
    verdict = engine.score(bare).finish()
    assert verdict.confidence < 0.4


def test_every_line_is_explained() -> None:
    verdict = engine.score(BOLD).finish()
    lines = verdict.evidence_for("func/bold")
    assert lines and all(ln.text and ln.source in ("parameters", "names") for ln in lines)
    assert any("300 volumes" in ln.text for ln in lines)


def test_weights_scale_the_sources() -> None:
    cfg = cfgmod.ClassifierConfig()
    cfg.weights.names = 0.0
    named_only = _f(description="t1_mprage")
    assert engine.score(named_only, cfg).finish().confidence < 0.4


def test_the_ranges_are_settings() -> None:
    cfg = cfgmod.ClassifierConfig()
    cfg.series.run_min_volumes = 400
    verdict = engine.score(BOLD, cfg).finish()
    assert verdict.probabilities.get("func/bold", 0) < engine.score(BOLD).finish().probabilities[
        "func/bold"]


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


def _session(*items) -> list[engine.Series]:
    return [engine.Series(key=str(i), features=f, order=float(i)) for i, f in enumerate(items)]


def test_a_reversed_phase_reference_beside_its_run() -> None:
    rev = replace(BOLD, n_volumes=4, description="mbep2d_rest_rev", pe_dir="AP")
    session = _session(BOLD, rev)
    engine.classify_session(session)
    assert session[1].verdict.best == "fmap/epi"
    assert session[1].verdict.entities["direction"] == "AP"
    assert any(ln.source == "context" for ln in session[1].verdict.lines)


def test_an_sbref_just_before_a_multiband_run() -> None:
    sbref = replace(BOLD, n_volumes=1, description="mbep2d_bold")
    session = _session(sbref, BOLD)
    engine.classify_session(session)
    assert session[0].verdict.best == "func/sbref"


def test_two_inversions_make_an_mp2rage() -> None:
    inv = _f(description="mp2rage_INV1", is_gre=True, is_ir=True, acquisition_type="3D",
             echo_times=[2.5], repetition_time=5.0, inversion_time=0.9, flip_angle=4,
             sequence_name="*tfl3d1_16ns", n_slices=176)
    inv2 = replace(inv, description="mp2rage_INV2", inversion_time=2.75)
    session = _session(inv, inv2)
    engine.classify_session(session)
    assert [x.verdict.best for x in session] == ["anat/MP2RAGE", "anat/MP2RAGE"]
    assert [x.verdict.entities["inversion"] for x in session] == ["1", "2"]


def test_magnitude_and_phase_twins_are_parts() -> None:
    mag = replace(BOLD, part="mag")
    phase = replace(BOLD, part="phase")
    session = _session(mag, phase)
    engine.classify_session(session)
    assert [x.verdict.entities.get("part") for x in session] == ["mag", "phase"]


# ---------------------------------------------------------------------------
# What the classifier never names
# ---------------------------------------------------------------------------


_SVS = _f(description="svs_se", rows=1.0, columns=1.0, n_slices=1, n_volumes=64,
          is_se=True, echo_times=[30.0], repetition_time=4.0, dicom_modality="MR",
          image_type=["ORIGINAL", "PRIMARY", "SPECTROSCOPY", "NONE"], is_spectroscopy=True)
_PET = _f(description="[BR-DY_CTAC] Brain Dynamic", n_volumes=26, dicom_modality="PT",
          rows=256.0, columns=256.0, n_slices=109)


@pytest.mark.parametrize("features", [_SVS, _PET, replace(BOLD, dicom_modality="CT")])
def test_spectroscopy_pet_and_ct_are_not_scored(features) -> None:
    """REGRESSION 2026-10-12: a spectroscopy series (a 1 x 1 'image' with a
    spin echo's TE and TR) was named anat/PDw and func/bold, and five PET
    series anat/angio and func/bold, over dcm2niix's mrs and pet."""
    assert not engine.in_domain(features)
    assert engine.score(features).finish().probabilities == {}


def test_a_name_outside_the_rulebook_is_never_replaced() -> None:
    from bidsmgr.classifier.acquisition import scan as by_parameters

    keys = {"m": uuid4(), "p": uuid4(), "y": uuid4()}
    chosen = {
        "m": Classification(row_id=keys["m"], classifier="dcm2niix_bidsguess",
                            datatype="mrs", suffix="svs", confidence=0.85),
        "p": Classification(row_id=keys["p"], classifier="dcm2niix_bidsguess",
                            datatype="pet", suffix="pet", confidence=0.85),
        "y": Classification(row_id=keys["y"], classifier="sequence_dict",
                            datatype="func", suffix="physio", confidence=0.4),
    }
    session = _session(T2, BOLD, T1)
    engine.classify_session(session)
    outcomes = {}
    for key, x in zip(("m", "p", "y"), session):
        d = engine.decide(x.verdict, "", "")
        outcomes[key] = by_parameters.Outcome(x.verdict, d, x.features)
        assert d.decided_by == "parameters"
    out = by_parameters.apply(chosen, outcomes, keys)
    assert [(out[k].datatype, out[k].suffix) for k in "mpy"] == \
        [("mrs", "svs"), ("pet", "pet"), ("func", "physio")]


def test_spectroscopy_is_recognised_in_the_header(tmp_path) -> None:
    from bidsmgr.classifier.acquisition.features import MR_SPECTROSCOPY_SOP

    built = feat.build([{"image_type": ["ORIGINAL", "PRIMARY", "SPECTROSCOPY", "NONE"],
                         "te": 30.0}],
                       {"description": "svs_se", "dicom_modality": "MR",
                        "sop_class": MR_SPECTROSCOPY_SOP, "rows": 1, "columns": 1},
                       n_files=1)
    assert built.is_spectroscopy and built.dicom_modality == "MR"


# ---------------------------------------------------------------------------
# Contrast agents, perfusion, angiography (measured on UPENN-GBM and the
# dcm2niix QA sets, 2026-10-12)
# ---------------------------------------------------------------------------


_T1_POST = replace(T1, description="t1 axial stealth-post", contrast_agent="24ML MULTIHANCE",
                   sequence_variant=["SP", "MP"], acquisition_type="3D", pixel_spacing=0.98,
                   slice_thickness=1.0)
_DSC = _f(description="ep2d_perf BOLUS", is_epi=True, n_volumes=60, echo_times=[45.0],
          repetition_time=2.0, contrast_agent="24ML MULTIHANCE", sequence_name="*epfid2d1_128",
          image_type=["ORIGINAL", "PRIMARY", "PERFUSION", "NONE"], n_slices=15)
_TOF = _f(description="tof_3d_multi-slab", is_gre=True, acquisition_type="3D",
          echo_times=[3.6], repetition_time=0.022, flip_angle=18, sequence_name="*fl3d1r_t70",
          n_slices=160, pixel_spacing=0.4, slice_thickness=0.5)
_PCASL = _f(description="pcasl_2d", is_epi=True, n_volumes=102, echo_times=[14.0],
            repetition_time=2.54, sequence_name="epfid2d1_72", pe_dir="AP")
_ASL_3D = _f(description="pasl_3d", is_epi=True, n_volumes=12, echo_times=[39.6],
             repetition_time=3.6, asl_mode=4, sequence_name="*tgse3d1_22080",
             image_type=["ORIGINAL", "PRIMARY", "ASL", "NONE"], acquisition_type="3D")


def test_the_contrast_agent_names_the_ce_entity() -> None:
    v = engine.score(_T1_POST).finish()
    assert v.best == "anat/T1w" and v.entities["ceagent"] == "multihance"
    assert "MULTIHANCE" in v.entity_why["ceagent"]
    cfg = cfgmod.ClassifierConfig()
    cfg.contrast.entity = "fixed"
    assert engine.score(_T1_POST, cfg).finish().entities["ceagent"] == "gadolinium"
    cfg.contrast.entity = "none"
    assert "ceagent" not in engine.score(_T1_POST, cfg).finish().entities


def test_a_name_saying_post_contrast_is_marked_to_check() -> None:
    """Without the agent in the header, the name may say it, but a session
    called "post" must not."""
    named = replace(T1, description="t1_mprage_post_gd")
    v = engine.score(named).finish()
    assert v.entities["ceagent"] == "gadolinium" and v.entity_check
    assert engine.decide(v, "", "").check
    session = replace(T1, description="ses-post_acq-mprage_T1w")
    assert "ceagent" not in engine.score(session).finish().entities


def test_ce_reaches_a_name_dcm2niix_chose() -> None:
    from bidsmgr.classifier.acquisition import scan as by_parameters

    key = uuid4()
    v = engine.score(_T1_POST).finish()
    d = engine.Decision("anat/T1w", "dcm2niix", True, 0.3, {}, v.best, "anat/T1w")
    chosen = {"k": Classification(row_id=key, classifier="dcm2niix_bidsguess",
                                  datatype="anat", suffix="T1w", confidence=0.85)}
    out = by_parameters.apply(chosen, {"k": by_parameters.Outcome(v, d, _T1_POST)}, {"k": key})
    assert out["k"].candidate_entities["ceagent"] == "multihance"


def test_a_contrast_bolus_is_left_out_or_named_cbv() -> None:
    v = engine.score(_DSC).finish()
    assert v.best == "exclude/dsc" and v.confidence > 0.9
    cfg = cfgmod.ClassifierConfig()
    cfg.contrast.dsc = "cbv"
    v = engine.score(_DSC, cfg).finish()
    assert v.best == "func/cbv"
    d = engine.decide(v, "", "", cfg)
    assert d.name == "func/cbv" and d.entities["task"] == "dsc"
    assert d.entities["ceagent"] == "multihance"


def test_time_of_flight_is_an_angiogram_not_a_t1() -> None:
    assert _best(_TOF) == "anat/angio"
    flash = replace(_TOF, description="flash_3d", sequence_name="*fl3d1")
    assert _best(flash) != "anat/angio", "no flow compensation: not time of flight"


def test_asl_from_the_name_the_protocol_and_the_image_type() -> None:
    assert _best(_PCASL) == "perf/asl", "TE 14 ms is short for BOLD; the name says pCASL"
    assert _best(_ASL_3D) == "perf/asl"
    m0 = replace(_ASL_3D, description="pasl_3d_m0", n_volumes=2)
    assert _best(m0) == "perf/m0scan"
    pw = replace(_ASL_3D, description="Perfusion_Weighted", n_volumes=1, is_derived=True,
                 image_type=["DERIVED", "PRIMARY", "ASL", "SUBTRACTION"])
    assert _best(pw) == "exclude/derived"


def test_a_short_epi_with_no_opposite_partner_is_no_reference() -> None:
    """GE writes no phase-encoding polarity: a 3-volume fMRI QA series was
    named a reversed-phase reference."""
    short = replace(BOLD, description="Ax fMRI HB8 72sl int asc", n_volumes=3, pe_dir="")
    session = _session(short)
    engine.classify_session(session)
    assert any("not recorded" in ln.text for ln in session[0].verdict.lines)
    ap = replace(BOLD, n_volumes=3, pe_dir="AP", description="EPI PE=AP")
    pa = replace(ap, pe_dir="PA", description="EPI PE=PA")
    session = _session(ap, pa)
    engine.classify_session(session)
    assert [x.verdict.best for x in session] == ["fmap/epi", "fmap/epi"], "a pair"


def test_unsure_parameters_follow_the_name_when_dcm2niix_is_silent() -> None:
    d = engine.decide(_verdict({"fmap/epi": 0.62, "func/bold": 0.38}), "", "func/bold")
    assert (d.name, d.decided_by, d.check) == ("func/bold", "names", True)


def test_a_b0_map_series_keeps_dcm2niix_names() -> None:
    assert engine.same("fmap/fieldmap", "fmap/magnitude")


# ---------------------------------------------------------------------------
# Every number is a setting
# ---------------------------------------------------------------------------


def _cfg(path: str, value):
    cfg = cfgmod.ClassifierConfig()
    section, name = path.split(".")
    setattr(getattr(cfg, section), name, value)
    return cfg


_IR_3D = _f(is_gre=True, is_ir=True, acquisition_type="3D", echo_times=[2.5],
            repetition_time=4.0, inversion_time=0.9, flip_angle=8)
_FLASH = _f(is_gre=True, acquisition_type="3D", echo_times=[4.0], repetition_time=0.02,
            flip_angle=20)
_2D_T1 = _f(is_gre=True, echo_times=[5.0], repetition_time=0.5, flip_angle=70)
_MT = _f(is_gre=True, echo_times=[5.0], repetition_time=0.03, mt=True, flip_angle=10)
_THREE_ECHOES = _f(is_gre=True, acquisition_type="3D", echo_times=[5.0, 10.0, 15.0],
                   repetition_time=0.03)
_FMAP_ECHOES = _f(is_gre=True, echo_times=[4.9, 7.4], repetition_time=0.5, part="mag")
_LOCALIZER = _f(is_gre=True, n_slices=3, n_orientations=3, echo_times=[4.0],
                repetition_time=0.008)


@pytest.mark.parametrize("features, fact, path, moved", [
    (_IR_3D, "mp2rage_part", "anatomy.mp2rage_tr_min", 5.0),
    (_FLASH, "spoiled_3d", "gradient_echo.spoiled_3d_tr_max", 10.0),
    (_FLASH, "spoiled_3d", "gradient_echo.spoiled_3d_flip_max", 15.0),
    (_2D_T1, "spoiled_2d_t1", "gradient_echo.spoiled_2d_te_max", 4.0),
    (_2D_T1, "spoiled_2d_t1", "gradient_echo.spoiled_2d_tr_max", 0.4),
    (_2D_T1, "spoiled_2d_t1", "gradient_echo.spoiled_2d_flip_min", 80.0),
    (_MT, "mt_short_te", "gradient_echo.mt_te_max", 4.0),
    (_THREE_ECHOES, "megre", "echoes.multi_echo_min", 4),
    (_FMAP_ECHOES, "fieldmap_echoes", "echoes.fieldmap_te_max", 4.0),
    (_LOCALIZER, "few_slices", "series.localizer_max_slices", 2),
    (_TOF, "tof_angio", "gradient_echo.angio_tr_max", 15.0),
    (_TOF, "tof_angio", "gradient_echo.angio_flip_max", 15.0),
    (_T1_POST, "high_res_3d", "anatomy.high_res_max_mm", 0.8),
    (_DSC, "contrast_bolus", "contrast.dsc_min_volumes", 80),
])
def test_each_physical_threshold_is_a_setting(features, fact, path, moved) -> None:
    """The user's question of 2026-10-12: the physical and contrast ranges
    were numbers in the code. Each now moves its fact."""
    assert engine.derived(features)[fact], f"{fact} holds with the defaults"
    assert not engine.derived(features, _cfg(path, moved))[fact], f"{path} moves it"


def test_the_sbref_look_ahead_is_a_setting() -> None:
    sbref = replace(BOLD, n_volumes=1, description="mbep2d_bold")
    other = replace(T1, description="t1_between")
    session = _session(sbref, other, other, BOLD)
    engine.classify_session(session)
    assert session[0].verdict.best == "func/sbref"
    session = _session(sbref, other, other, BOLD)
    engine.classify_session(session, _cfg("series.sbref_lookahead", 1))
    assert not any(ln.source == "context" and ln.candidate == "func/sbref"
                   for ln in session[0].verdict.lines)


def test_no_setting_is_left_unread() -> None:
    """Every field of every section is read by the engine or the rulebook."""
    source = Path(engine.__file__).read_text(encoding="utf-8")
    book = engine.RULEBOOK.read_text(encoding="utf-8")
    scan_src = (Path(engine.__file__).parent / "scan.py").read_text(encoding="utf-8")
    cfg = cfgmod.ClassifierConfig()
    for section in ("anatomy", "gradient_echo", "echoes", "series", "weights", "general",
                    "contrast"):
        for name in type(getattr(cfg, section)).model_fields:
            used = f".{name}" in source or f"cfg:{section}.{name}" in book \
                or f".{name}" in scan_src
            assert used, f"{section}.{name} is a setting nothing reads"


def test_the_rules_can_be_changed_through_the_settings() -> None:
    """A changed rule replaces the shipped one; null removes a name."""
    t2 = engine.score(T2).finish()
    assert t2.best == "anat/T2w"
    cfg = cfgmod.ClassifierConfig(rulebook={"candidates": {"anat/T2w": None}})
    assert "anat/T2w" not in engine.score(T2, cfg).finish().probabilities
    louder = json.loads(json.dumps(engine.rulebook()["candidates"]["anat/PDw"]))
    louder["evidence"].append({"source": "names", "pattern": "t2 tse", "weight": 9.0,
                               "why": "our scanner's PD protocol"})
    cfg = cfgmod.ClassifierConfig(rulebook={"candidates": {"anat/PDw": louder}})
    assert engine.score(T2, cfg).finish().best == "anat/PDw"
    # Round trip through JSON, as the CLI and the GUI keep it.
    assert cfgmod.load_json(cfg.model_dump_json()).rulebook == cfg.rulebook


def test_broken_rules_say_what_is_wrong() -> None:
    book = {"candidates": {"anat/T2w": {"evidence": [
        {"source": "parameters", "feature": "no_such_fact", "weight": 1},
        {"source": "names", "pattern": "(", "weight": 1},
        {"source": "elsewhere", "pattern": "x", "weight": "a lot"}]},
        "T2w": {"evidence": []}}}
    problems = " | ".join(engine.rulebook_problems(book))
    for said in ("no feature named 'no_such_fact'", "does not compile", "source is one of",
                 "weight is not a number", "datatype/suffix"):
        assert said in problems
    assert engine.rulebook_problems(engine.rulebook()) == []


def test_only_what_differs_from_the_shipped_rules_is_kept() -> None:
    whole = json.loads(json.dumps(engine.rulebook()))
    assert engine.rulebook_delta(whole, full=True) == {}
    whole["candidates"]["anat/PDw"]["evidence"][0]["weight"] = 0.5
    del whole["candidates"]["anat/angio"]
    delta = engine.rulebook_delta(whole, full=True)
    assert set(delta["candidates"]) == {"anat/PDw", "anat/angio"}
    assert delta["candidates"]["anat/angio"] is None
    # A file holding only a change removes nothing else.
    only = {"candidates": {"anat/PDw": whole["candidates"]["anat/PDw"]}}
    assert set(engine.rulebook_delta(only, full=False)["candidates"]) == {"anat/PDw"}


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def _verdict(probabilities: dict) -> engine.Verdict:
    v = engine.Verdict()
    v.probabilities = probabilities
    return v


def test_agreement_is_confirmed() -> None:
    d = engine.decide(_verdict({"func/bold": 0.7, "fmap/epi": 0.3}), "func/bold", "func/bold")
    assert (d.name, d.decided_by, d.check) == ("func/bold", "agreement", False)


def test_sure_and_different_wins_and_says_so() -> None:
    d = engine.decide(_verdict({"fmap/epi": 0.95}), "func/bold", "")
    assert (d.name, d.decided_by, d.check) == ("fmap/epi", "parameters", False)
    assert "dcm2niix suggests func/bold" in d.note


def test_unsure_follows_the_name_and_asks() -> None:
    d = engine.decide(_verdict({"anat/T1w": 0.6}), "anat/T2starw", "anat/T2starw")
    assert (d.name, d.check) == ("anat/T2starw", True)
    d = engine.decide(_verdict({"anat/T1w": 0.6}), "anat/T2starw", "anat/T1w")
    assert (d.name, d.check) == ("anat/T1w", True)


def test_silent_leaves_it_to_the_other_opinion() -> None:
    d = engine.decide(_verdict({"anat/T1w": 0.2}), "anat/T2w", "")
    assert (d.name, d.decided_by, d.check) == ("anat/T2w", "dcm2niix", True)


def test_a_fieldmaps_outputs_are_one_name() -> None:
    assert engine.same("fmap/magnitude", "fmap/phasediff")
    assert engine.same("fmap/magnitude1", "fmap/magnitude")
    assert not engine.same("fmap/epi", "fmap/magnitude")


def test_thresholds_are_settings() -> None:
    cfg = cfgmod.ClassifierConfig()
    cfg.general.apply_at = 0.99
    d = engine.decide(_verdict({"fmap/epi": 0.95}), "func/bold", "", cfg)
    assert d.check


# ---------------------------------------------------------------------------
# Applying it to the chain's choices
# ---------------------------------------------------------------------------


def test_apply_replaces_keeps_and_respects_a_forced_rule() -> None:
    from bidsmgr.classifier.acquisition import scan as by_parameters

    ids = {k: uuid4() for k in "abcd"}
    chosen = {
        "a": Classification(row_id=ids["a"], classifier="sequence_dict", datatype="func",
                            suffix="bold", candidate_entities={"task": "rest"}, confidence=0.4),
        "b": Classification(row_id=ids["b"], classifier="user_hint", datatype="anat",
                            suffix="T2w", confidence=0.95),
        "c": Classification(row_id=ids["c"], classifier="dcm2niix_bidsguess", datatype="func",
                            suffix="bold", candidate_entities={"task": "rest"}, confidence=0.85),
    }
    rev = replace(BOLD, n_volumes=4, pe_dir="AP")

    def outcome(name, by, ents=None, f=None):
        verdict = _verdict({name: 0.97})
        verdict.entities = dict(ents or {})
        return by_parameters.Outcome(verdict, engine.Decision(name, by, False, 0.97,
                                                             dict(ents or {})), f)

    outcomes = {
        "a": outcome("fmap/epi", "parameters", {"direction": "AP"}, rev),
        "b": outcome("anat/FLAIR", "parameters"),
        "c": outcome("func/bold", "agreement", {"part": "mag"}),
        "d": outcome("exclude/localizer", "parameters"),
    }
    out = by_parameters.apply(chosen, outcomes, ids, forced={"b"})
    assert (out["a"].datatype, out["a"].suffix) == ("fmap", "epi")
    assert out["a"].candidate_entities == {"direction": "AP"}, "a run's task does not travel"
    assert out["b"].suffix == "T2w", "a forced rule stands"
    assert out["c"].classifier == "dcm2niix_bidsguess"
    assert out["c"].candidate_entities == {"task": "rest", "part": "mag"}
    assert out["d"].skip and out["d"].datatype == "discard"


# ---------------------------------------------------------------------------
# Features from DICOM, and the scan
# ---------------------------------------------------------------------------


def test_features_are_read_from_the_headers(tmp_path) -> None:
    import pydicom

    files = write_mr_series(tmp_path / "s", "1.2.3.4", n_slices=4, description="t1_mprage")
    for f in files:
        ds = pydicom.dcmread(str(f))
        ds.EchoTime, ds.RepetitionTime, ds.InversionTime, ds.FlipAngle = 2.3, 2300, 900, 8
        ds.ScanningSequence, ds.MRAcquisitionType = ["GR", "IR"], "3D"
        ds.save_as(str(f))
    facts = [feat.file_facts(pydicom.dcmread(str(f), stop_before_pixels=True)) for f in files]
    series = feat.series_facts(pydicom.dcmread(str(files[0]), stop_before_pixels=True))
    built = feat.build(facts, series, n_files=len(files))
    assert built.echo_times == [2.3] and built.repetition_time == 2.3
    assert built.inversion_time == 0.9 and built.is_ir and built.is_gre
    assert built.n_slices == 4 and built.n_volumes == 1 and built.acquisition_type == "3D"
    assert engine.score(built).finish().best == "anat/T1w"


def test_the_ascconv_protocol_is_parsed() -> None:
    text = ("### ASCCONV BEGIN ###\nsSliceAcceleration.lMultiBandFactor\t = 4\n"
            'tProtocolName = "bold"  # x\n### ASCCONV END ###')
    assert feat.parse_ascconv(text) == {"sSliceAcceleration.lMultiBandFactor": "4",
                                        "tProtocolName": "bold"}


def test_the_phase_encoding_direction() -> None:
    axial = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert feat.phase_direction("COL", 1.0, axial) == "AP"
    assert feat.phase_direction("COL", 0.0, axial) == "PA"
    assert feat.phase_direction("ROW", 1.0, axial) == "RL"
    assert feat.phase_direction("COLUMN", 0.0, axial) == "PA"
    assert feat.phase_direction("COL", None, axial) == ""


def test_the_features_round_trip_through_the_cache(tmp_path) -> None:
    from bidsmgr.classifier.acquisition import cache

    tsv = tmp_path / "inv.tsv"
    cache.write(tsv, {"1.2": replace(BOLD, shim=(1, 2, 3))})
    back = cache.read(tsv)
    assert back["1.2"] == replace(BOLD, shim=(1, 2, 3))
    assert cache.read(tmp_path / "missing.tsv") == {}


def test_the_config_clamps_and_round_trips() -> None:
    cfg = cfgmod.load_json(json.dumps({"general": {"apply_at": 5, "mode": "nonsense"},
                                       "series": {"run_min_volumes": 20}}))
    assert cfg.general.apply_at == 0.99 and cfg.general.mode == "applied"
    assert cfg.series.run_min_volumes == 20
    assert cfgmod.load_json(cfg.model_dump_json()) == cfg
    assert cfgmod.key(cfg) != cfgmod.key(cfgmod.DEFAULT)


def test_a_rulebook_delta_replaces_or_removes() -> None:
    book = engine.rulebook({"candidates": {"anat/angio": None,
                                           "anat/T1w": {"label": "x", "evidence": []}}})
    assert "anat/angio" not in book["candidates"]
    assert book["candidates"]["anat/T1w"]["evidence"] == []
    assert "anat/angio" in engine.rulebook()["candidates"], "the shipped book is untouched"


def _scan(raw: Path, tmp_path: Path, mode: str):
    from bidsmgr.cli.scan import run_scan

    cfg = cfgmod.ClassifierConfig()
    cfg.general.mode = mode
    out = tmp_path / mode
    out.mkdir()
    return run_scan(raw, out / "inv.tsv", skip_bids_guess=True, classifier_config=cfg)


def _bold_series(folder: Path, uid: str, *, volumes: int, description: str, pe: int):
    """A Siemens-like mosaic EPI series (one file per volume)."""
    import pydicom

    files = write_mr_series(folder, uid, n_slices=volumes, description=description)
    for k, f in enumerate(files):
        ds = pydicom.dcmread(str(f))
        ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND", "MOSAIC"]
        ds.ScanningSequence, ds.SequenceName = ["EP"], "epfid2d1_64"
        ds.EchoTime, ds.RepetitionTime, ds.FlipAngle = 30, 800, 60
        ds.AcquisitionNumber = k + 1
        ds.InPlanePhaseEncodingDirection = "COL"
        ds.ImagePositionPatient = [0, 0, 0]
        ds.save_as(str(f))
    return files


def test_a_scan_names_a_short_reference_and_reports_why(tmp_path) -> None:
    raw = tmp_path / "raw"
    _bold_series(raw / "run", "1.2.3.10", volumes=40, description="series_a", pe=0)
    _bold_series(raw / "ref", "1.2.3.11", volumes=3, description="series_b", pe=1)
    applied = _scan(raw, tmp_path, "applied").set_index("series_uid")
    shadow = _scan(raw, tmp_path, "shadow").set_index("series_uid")
    off = _scan(raw, tmp_path, "off").set_index("series_uid")
    assert applied.loc["1.2.3.10", "datatype"] == "func"
    assert float(applied.loc["1.2.3.10", "classification_confidence"]) > 0.8
    evidence = json.loads(applied.loc["1.2.3.10", "classification_evidence"])
    assert evidence["chosen"] == "func/bold" and evidence["for"]
    assert off.loc["1.2.3.10", "classification_confidence"] == ""
    # Shadow reports without naming.
    assert shadow.loc["1.2.3.11", "classification_confidence"] != ""
    assert shadow.loc["1.2.3.11", "datatype"] == off.loc["1.2.3.11", "datatype"]

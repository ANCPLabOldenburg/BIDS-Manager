"""Every quality measure, plot, map and group has an explanation, and the
explanations read the way the rest of the program does."""

from __future__ import annotations

import pytest

from bidsmgr.qc import anat, explain
from bidsmgr.viz.compute import qc

from tests.unit.qc_phantoms import anat_phantom


def test_every_anatomical_measure_is_explained() -> None:
    for key in anat.METRICS:
        assert explain.lookup(f"anat.{key}") is not None, key


@pytest.mark.parametrize("key", [
    "coverage_gap_b1000", "motion_b0", "fd_b0_max", "volumes_far", "fd_mean", "eddy_shift",
    "rotation_max", "drift", "fa_median", "md_median", "negative_eigen", "dropout_slices",
    "interleave_volumes", "spikes_ppm", "ndc", "sigma_b0", "sigma_pca", "snr_cc_b0",
    "snr_cc_b1000_along", "snr_cc_b2000_across", "noise_floor", "efc_b0", "efc_b1000",
    "fber_b0", "fber_b2000"])
def test_every_diffusion_measure_is_explained(key) -> None:
    assert explain.lookup(f"dwi.{key}") is not None, key


def test_the_exact_key_wins_over_a_pattern() -> None:
    assert explain.lookup("dwi.snr_cc_b0").title != explain.lookup("dwi.snr_cc_b1000_along").title
    assert explain.lookup("dwi.no_such_measure") is None


def test_every_plot_map_and_group_is_explained() -> None:
    from bidsmgr.gui.viz.panels.quality_panel import GROUPS

    for row, _t, _h in qc.QC_ROWS:
        assert explain.lookup(f"plot.bold.{row}") is not None, row
    for row, _t, _h in qc.DWI_QC_ROWS:
        assert explain.lookup(f"plot.dwi.{row}") is not None, row
    for which in ("mean", "sd", "tsnr"):
        assert explain.lookup(f"map.bold.{which}") is not None, which
    data, affine = anat_phantom()
    res = anat.check(data, affine, suffix="T1w")
    groups = {m.group for m in res.metrics}
    for g in groups:
        assert explain.lookup(f"group.anat.{g}") is not None, g
    known = {g for g, _t in GROUPS}
    assert groups <= known


def test_the_texts_follow_the_house_style() -> None:
    for key, e in explain.ENTRIES.items():
        for name in ("title", "short", "measures", "computed", "reading", "causes",
                     "caveats", "mriqc"):
            text = getattr(e, name)
            assert "—" not in text and "–" not in text, (key, name)
        assert e.short.count(". ") <= 2, key


def test_the_tissue_snr_explains_why_csf_reads_low() -> None:
    """The user's question: why CSF's SNR is low and white matter's high."""
    noise = explain.lookup("group.anat.noise").reading.lower()
    assert "upper bound" in noise and "partial volume" in noise
    csf = explain.lookup("anat.snr_csf")
    assert "darkest" in csf.reading.lower() and "normal" in csf.reading.lower()

"""The quality check's building blocks: statistics, template, registration,
masks, bias field and segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from bidsmgr.qc import bias, masks, register as R, stats
from bidsmgr.qc.templates import FILES, template

from tests.unit.qc_phantoms import anat_phantom


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_mad_is_the_sd_of_normal_data() -> None:
    x = np.random.default_rng(0).normal(5.0, 2.0, 200_000)
    assert stats.mad(x) == pytest.approx(2.0, rel=0.02)


def test_robust_z_ignores_one_outlier() -> None:
    x = np.r_[np.random.default_rng(1).normal(0, 1, 500), 50.0]
    z = stats.robust_z(x)
    assert z[-1] > 30 and np.abs(z[:-1]).max() < 5


def test_weighted_stats_of_a_hard_mask_are_the_plain_ones() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(10, 3, 10_000)
    w = (rng.random(10_000) > 0.5).astype(float)
    s = stats.weighted_stats(x, w)
    sel = x[w > 0]
    assert s["mean"] == pytest.approx(sel.mean())
    assert s["stdv"] == pytest.approx(sel.std())
    assert s["median"] == pytest.approx(np.median(sel), abs=0.05)
    assert s["n"] == pytest.approx(w.sum())


def test_otsu_splits_two_populations() -> None:
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(10, 1, 5000), rng.normal(100, 5, 5000)]
    assert 20 < stats.otsu(x) < 90


def test_block_mean_and_its_affine_agree() -> None:
    affine = np.array([[0.8, 0, 0, -10], [0, 0.8, 0, -20], [0, 0, 1.0, -30], [0, 0, 0, 1]])
    vol = np.arange(10 * 10 * 6, dtype=np.float32).reshape(10, 10, 6)
    f = stats.block_factor((0.8, 0.8, 1.0), 2.0)
    small = stats.block_mean(vol, f)
    a = stats.block_affine(affine, f)
    # The first block's centre in world mm is the mean of its voxels' centres.
    centres = [affine @ [i, j, k, 1] for i in range(f[0]) for j in range(f[1]) for k in range(f[2])]
    assert np.allclose(a @ [0, 0, 0, 1], np.mean(centres, axis=0))
    assert small[0, 0, 0] == pytest.approx(vol[:f[0], :f[1], :f[2]].mean())


def test_the_kde_is_a_density() -> None:
    grid = np.linspace(0, 110, 1000)
    d = stats.gaussian_kde_grid(np.random.default_rng(4).normal(50, 5, 5000), grid, 4.0)
    assert np.sum(d) * (grid[1] - grid[0]) == pytest.approx(1.0, rel=1e-6)
    assert grid[np.argmax(d)] == pytest.approx(50, abs=3)


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------


def test_the_template_is_one_grid_of_maps_in_0_1() -> None:
    tpl = template()
    assert set(FILES) == set(tpl.images)
    for key, img in tpl.images.items():
        assert img.shape == tpl.shape, key
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0 + 1e-6, key
    total = tpl["csf"] + tpl["gm"] + tpl["wm"]
    assert float(total[tpl["brain"] > 0.5].mean()) == pytest.approx(1.0, abs=0.15)


def test_the_template_ships_with_its_licence() -> None:
    from importlib import resources

    text = (resources.files("bidsmgr.qc.templates") / "PROVENANCE.md").read_text()
    assert "Louis Collins" in text and "McConnell" in text


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _moved(data, affine, shift_mm, angle_deg):
    """The same image with its world position moved: a known registration."""
    t = np.eye(4)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    t[:3, 3] = shift_mm
    return data, t @ affine, t


@pytest.mark.parametrize("contrast", ["T1w", "T2w"])
def test_registration_recovers_a_known_move(contrast) -> None:
    data, affine = anat_phantom(contrast=contrast)
    data, moved_affine, truth = _moved(data, affine, (12.0, -8.0, 6.0), 6.0)
    head = masks.head(data, exclude=masks.zero_fill(data))
    reg = R.register(data, moved_affine, head, contrast=contrast)
    # The template's brain centre lands where the move put it.
    tpl = template()
    centre = np.argwhere(tpl["brain"] > 0.5).mean(axis=0)
    c_world = tpl.affine @ np.r_[centre, 1]
    assert np.linalg.norm(reg.matrix @ c_world - truth @ c_world) < 3.0
    assert not reg.uncertain


def test_the_field_of_view_cut_is_measured() -> None:
    data, affine = anat_phantom()
    reg = R.Registration(matrix=np.eye(4), correlation=1.0)
    whole = R.outside_fov(reg, data.shape, affine)
    assert whole["fraction"] < 0.001
    cut = data[:, :, : data.shape[2] - 52]                 # the top 80 mm of the template gone
    out = R.outside_fov(reg, cut.shape, affine)
    assert out["fraction"] > 0.1
    assert max(out["sides"], key=out["sides"].get) == "top"


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def test_zero_fill_is_the_blank_border_not_dark_tissue() -> None:
    data, _ = anat_phantom(noise=0.0)
    data[:, :, :10] = 0.0                                   # padding at the bottom
    zf = masks.zero_fill(data)
    assert zf[:, :, :10].all()
    centre = tuple(s // 2 for s in data.shape)
    assert not zf[centre]


def test_the_head_mask_is_the_head() -> None:
    data, _ = anat_phantom()
    head = masks.head(data)
    truth = np.pad(template()["head"] > 0.5, 12)
    overlap = (head & truth).sum() / (head | truth).sum()
    assert overlap > 0.85


def test_artefacts_find_a_ghost_in_the_air_and_not_the_noise() -> None:
    data, _ = anat_phantom(noise=0.02)
    head = masks.head(data)
    air = masks.air(head, excluded=np.zeros_like(head))
    clean = masks.artefacts(data, air, head)
    ghost = data.copy()
    ghost[2:10, 2:10, -12:-4] += 300.0                      # a bright patch far out
    found = masks.artefacts(ghost, air, head)
    assert clean.sum() == 0
    assert found[2:10, 2:10, -12:-4].sum() > 100


# ---------------------------------------------------------------------------
# Bias field
# ---------------------------------------------------------------------------


def test_a_planted_bias_field_is_found() -> None:
    shape = (40, 40, 30)
    x = np.linspace(-1, 1, shape[0])[:, None, None]
    planted = (0.25 * x * np.ones(shape)).astype(np.float32)
    mask = np.ones(shape, dtype=np.float32)
    found = bias.fit_field(planted, mask)
    assert np.abs(found - planted).max() < 1e-3
    nu = bias.nonuniformity(found, mask > 0)
    assert nu["range"] == pytest.approx(np.exp(0.225) - np.exp(-0.225), rel=0.1)

"""The diffusion check on a phantom with planted defects: a broken table is
caught first, and a dropped slice, a spike, a moved volume, a drift and a
flipped b-vector are each found where they were planted."""

from __future__ import annotations

import numpy as np
import pytest

from bidsmgr.qc import dwi
from bidsmgr.qc.config import QcConfig
from bidsmgr.qc.series import Series

from tests.unit.qc_phantoms import dataset, dwi_phantom, gradient_table, save_dwi

#: The checks with and without the b-vector flip check; the numpy brain
#: mask (the phantoms are not heads a network was trained on).
FLIPS = QcConfig()
FLIPS.methods.dwi_brain = "median_otsu"
NO_FLIPS = FLIPS.model_copy(deep=True)
NO_FLIPS.dwi.check_flips = False


def _series(data, affine, bvals, bvecs, sidecar=None) -> Series:
    return Series(n=data.shape[3], shape=data.shape[:3], affine=affine,
                  frame=lambda i, d=data: np.asarray(d[..., int(i)], dtype=np.float32),
                  path="sub-01_dwi.nii.gz", sidecar=sidecar or {}, bvals=np.asarray(bvals),
                  bvecs=np.asarray(bvecs).T)


@pytest.fixture(scope="module")
def table():
    return gradient_table(n_dirs=30, n_b0=4)


@pytest.fixture(scope="module")
def clean(table):
    bvals, bvecs = table
    data, affine = dwi_phantom(bvals, bvecs)
    return data, affine, dwi.check(_series(data, affine, bvals, bvecs), config=NO_FLIPS)


# ---------------------------------------------------------------------------
# The gradient table
# ---------------------------------------------------------------------------


def test_a_table_that_does_not_match_the_volumes_is_an_error() -> None:
    bvals, bvecs = gradient_table(10, 2)
    found, usable = dwi.check_table(len(bvals) + 3, bvals, bvecs.T, [])
    assert not usable
    assert [f.key for f in found] == ["table_count"]
    assert found[0].level == "error"


def test_missing_gradient_files_are_errors(tmp_path) -> None:
    from bidsmgr.qc.series import read_gradients

    bvals, bvecs, problems = read_gradients(tmp_path / "sub-01_dwi.nii.gz")
    assert bvals is None and bvecs is None and len(problems) == 2
    found, usable = dwi.check_table(10, bvals, bvecs, problems)
    assert not usable and all(f.level == "error" for f in found)


def test_vectors_that_are_not_unit_are_reported() -> None:
    bvals, bvecs = gradient_table(10, 2)
    bvecs = bvecs.copy()
    bvecs[bvals > 0] *= 0.9
    found, usable = dwi.check_table(len(bvals), bvals, bvecs.T, [])
    assert "bvec_norm" in {f.key for f in found}
    assert np.allclose(np.linalg.norm(usable["bvecs"][bvals > 0], axis=1), 1.0)


def test_no_b0_is_an_error() -> None:
    bvals, bvecs = gradient_table(10, 0)
    found, _ = dwi.check_table(len(bvals), bvals, bvecs.T, [])
    assert "no_b0" in {f.key for f in found}


def test_coverage_and_duplicates() -> None:
    _b, g = gradient_table(30, 0)
    assert dwi.coverage_gap_deg(g) < 25
    assert dwi.coverage_gap_deg(np.tile([[1.0, 0.0, 0.0]], (10, 1))) > 80
    assert dwi.duplicates(np.r_[g, g[:3]]) == 3


# ---------------------------------------------------------------------------
# The tensor
# ---------------------------------------------------------------------------


def test_the_tensor_fit_recovers_a_known_tensor() -> None:
    bvals, bvecs = gradient_table(30, 3)
    d = np.diag([1.7e-3, 0.3e-3, 0.3e-3])
    s = 1000 * np.exp(-bvals * np.einsum("ni,ij,nj->n", bvecs, d, bvecs))
    fit = dwi.fit_tensor(np.tile(s, (5, 1)), bvals, bvecs)
    lam = np.array([1.7, 0.3, 0.3]) * 1e-3
    fa = np.sqrt(1.5 * ((lam - lam.mean()) ** 2).sum() / (lam ** 2).sum())
    assert fit["fa"][0] == pytest.approx(fa, abs=1e-3)
    assert abs(fit["e1"][0] @ [1, 0, 0]) == pytest.approx(1.0, abs=1e-4)


def test_mppca_finds_the_noise() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(125, 3)) @ rng.normal(size=(3, 40)) * 50.0
    noisy = signal + rng.normal(0, 7.0, signal.shape)
    assert dwi.mppca_sigma(noisy) == pytest.approx(7.0, rel=0.15)


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def test_a_clean_phantom_is_clean(clean) -> None:
    _d, _a, res = clean
    keys = {f.key for f in res.findings if f.level in ("warning", "error")}
    assert not keys & {"dropout", "interleave", "motion", "far", "drift"}, keys
    assert res.value("dropout_slices") == 0
    assert res.value("fa_median") is not None
    assert res.value("snr_cc_b0") is not None or res.facts.get("cc_voxels", 0) < 5
    assert {"displacement", "slices", "spikes"} <= set(res.tracks)
    assert {"brain", "fa", "md"} <= set(res.maps)


def test_a_dropped_slice_is_found_where_it_was_planted(table) -> None:
    bvals, bvecs = table
    data, affine = dwi_phantom(bvals, bvecs)
    v = int(np.flatnonzero(bvals > 0)[10])
    data[:, :, 15, v] *= 0.4
    res = dwi.check(_series(data, affine, bvals, bvecs), config=NO_FLIPS)
    found = {f.key: f for f in res.findings}
    assert "dropout" in found
    assert found["dropout"].evidence == {"volume": v, "slice": 15, "axis": 2}


def test_a_spike_is_counted(clean, table) -> None:
    bvals, bvecs = table
    data, affine = dwi_phantom(bvals, bvecs)
    v = int(np.flatnonzero(bvals > 0)[5])
    data[26:30, 10:14, 14:16, v] *= 6.0                   # in tissue, not CSF
    res = dwi.check(_series(data, affine, bvals, bvecs), config=NO_FLIPS)
    assert res.value("spikes_ppm") > clean[2].value("spikes_ppm") + 10


def test_a_moved_volume_is_out_of_place(table) -> None:
    from scipy import ndimage as ndi

    bvals, bvecs = table
    data, affine = dwi_phantom(bvals, bvecs)
    v = int(np.flatnonzero(bvals > 0)[12])
    data[..., v] = ndi.shift(data[..., v], (3.0, 0.0, 0.0), order=1)   # 6 mm
    res = dwi.check(_series(data, affine, bvals, bvecs), config=NO_FLIPS)
    disp = np.asarray(res.tracks["displacement"]["ys"][0])
    assert disp[v] > 4.0
    assert v in [int(x) for x in res.tracks["displacement"]["ticks"]]


def test_a_drift_is_measured(table) -> None:
    bvals, bvecs = table
    data, affine = dwi_phantom(bvals, bvecs)
    n = data.shape[3]
    data = data * (1.0 - 0.10 * np.arange(n) / (n - 1))[None, None, None, :]
    res = dwi.check(_series(data.astype(np.float32), affine, bvals, bvecs), config=NO_FLIPS)
    assert res.value("drift") == pytest.approx(-10.0, abs=2.5)
    assert any(f.key == "drift" for f in res.findings)


def test_a_flipped_bvec_is_suspected(table) -> None:
    bvals, bvecs = table
    data, affine = dwi_phantom(bvals, bvecs, shape=(64, 64, 40), noise=5.0, radius=26.0,
                               band=9.0)
    stored = bvecs.copy()
    stored[:, 0] *= -1.0
    res = dwi.check(_series(data, affine, bvals, stored), config=FLIPS)
    fc = res.facts["flip_check"]
    assert fc["best"] == "x reversed"
    assert any(f.key == "bvec_flip" for f in res.findings)
    good = dwi.check(_series(data, affine, bvals, bvecs), config=FLIPS)
    assert good.facts["flip_check"]["stored_is_best"]
    assert not any(f.key == "bvec_flip" for f in good.findings)


def test_the_air_is_not_measured_in_a_defaced_series(table) -> None:
    bvals, bvecs = table
    data, affine = dwi_phantom(bvals, bvecs)
    res = dwi.check(_series(data, affine, bvals, bvecs,
                            sidecar={"DeidentificationMethod": ["defaced"]}), config=NO_FLIPS)
    m = res.metric("efc_b0")
    assert m.value is None and "defaced" in m.why_missing


def test_the_check_of_a_file(tmp_path, table) -> None:
    bvals, bvecs = table
    data, affine = dwi_phantom(bvals, bvecs)
    root = dataset(tmp_path / "ds")
    path = save_dwi(data, affine, bvals, bvecs, root / "sub-01" / "dwi" / "sub-01_dwi.nii.gz")
    res = dwi.check_file(path, config=NO_FLIPS)
    assert res.suffix == "dwi" and res.facts["shells"] == {"0": 4, "1000": 30}

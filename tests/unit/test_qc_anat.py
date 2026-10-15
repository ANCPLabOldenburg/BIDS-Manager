"""The anatomical check on a phantom with planted defects: each measure
moves the right way, and the air is measured only when the image is not
defaced."""

from __future__ import annotations

import numpy as np
import pytest

from bidsmgr.qc import anat
from bidsmgr.qc.config import QcConfig, fast

from tests.unit.qc_phantoms import anat_phantom

AIR = ("snrd_wm", "snrd_total", "cnr", "efc", "fber", "qi1", "qi2", "ghost_ratio")


@pytest.fixture(scope="module")
def clean():
    data, affine = anat_phantom()
    return data, affine, anat.check(data, affine, suffix="T1w", sidecar={})


def test_a_clean_head_is_checked_whole(clean) -> None:
    _d, _a, res = clean
    assert res.facts["air_measured"]
    assert not res.facts["defaced"]
    for key in ("snr_wm", "cjv", "efc", "fber", "qi2", "fwhm_avg", "fraction_gm"):
        assert res.value(key) is not None, key
    assert res.value("fov_cut") < 0.5
    assert {"brain", "head", "air", "tissues", "bias"} <= set(res.maps)
    assert res.facts["seconds"] < 30


def test_the_air_is_not_measured_when_the_sidecar_records_defacing(clean) -> None:
    data, affine, _ = clean
    sidecar = {"DeidentificationMethod": ["pydeface"]}
    res = anat.check(data, affine, suffix="T1w", sidecar=sidecar)
    assert res.facts["defaced"]
    for key in AIR:
        m = res.metric(key)
        assert m.value is None and "defaced" in m.why_missing, key
    # What does not need the air still is.
    assert res.value("snr_wm") is not None
    assert any(f.key == "defaced" for f in res.findings)
    assert "air" not in res.maps


def test_a_blank_face_counts_as_defaced_without_a_record(clean) -> None:
    from bidsmgr.qc import register as R
    from bidsmgr.qc.templates import FACE_Y_MM, FACE_Z_MM, template

    data, affine, _ = clean
    tpl = template()
    ijk = np.indices(data.shape).reshape(3, -1)
    world = affine[:3, :3] @ ijk + affine[:3, 3:4]
    face = ((world[1] > FACE_Y_MM - 10) & (world[2] < FACE_Z_MM + 10)).reshape(data.shape)
    blanked = np.where(face, 0.0, data).astype(np.float32)
    res = anat.check(blanked, affine, suffix="T1w", sidecar={})
    assert res.facts["defaced"]
    assert not res.facts["air_measured"]
    del tpl, R


def test_a_visible_face_is_reported(clean) -> None:
    _d, _a, res = clean
    assert any(f.key == "face" for f in res.findings)


def test_noise_lowers_the_snr(clean) -> None:
    _d, _a, res = clean
    noisy, affine = anat_phantom(noise=0.08)
    worse = anat.check(noisy, affine, suffix="T1w", sidecar={})
    assert worse.value("snr_wm") < res.value("snr_wm")
    assert worse.value("cjv") > res.value("cjv")


def test_a_ghost_along_phase_encoding_is_measured() -> None:
    data, affine = anat_phantom()
    clean = anat.check(data, affine, suffix="T1w", sidecar={"PhaseEncodingDirection": "j"})
    ghosted = data + 0.15 * np.roll(data, data.shape[1] // 2, axis=1)
    worse = anat.check(ghosted.astype(np.float32), affine, suffix="T1w",
                       sidecar={"PhaseEncodingDirection": "j"})
    assert worse.value("ghost_ratio") > clean.value("ghost_ratio") + 0.02
    assert worse.facts["phase_encoding_axis"] == "j"


def test_a_planted_bias_field_raises_the_non_uniformity(clean) -> None:
    data, affine, res = clean
    x = np.linspace(-1, 1, data.shape[0])[:, None, None]
    biased = (data * np.exp(0.4 * x)).astype(np.float32)
    worse = anat.check(biased, affine, suffix="T1w", sidecar={})
    assert worse.value("inu_range") > res.value("inu_range") + 0.12


def test_a_cut_field_of_view_is_an_error() -> None:
    data, affine = anat_phantom()
    cut = data[:, :, : data.shape[2] - 50]
    res = anat.check(cut, affine, suffix="T1w", sidecar={})
    found = {f.key: f for f in res.findings}
    assert res.value("fov_cut") > 3.0
    assert found["fov_cut"].level in ("warning", "error")


def test_saturation_is_reported() -> None:
    data, affine = anat_phantom()
    top = np.percentile(data[data > 0], 97)
    clipped = np.minimum(data, top).astype(np.float32)
    res = anat.check(clipped, affine, suffix="T1w", sidecar={})
    assert res.value("saturation") > 0.5
    assert any(f.key == "saturation" for f in res.findings)


def test_a_flair_gets_no_tissue_measures_and_says_why() -> None:
    data, affine = anat_phantom(contrast="T2w")
    res = anat.check(data, affine, suffix="FLAIR", sidecar={})
    m = res.metric("snr_wm")
    assert m.value is None and "T1w and T2w" in m.why_missing
    assert res.value("efc") is not None


def test_a_thin_slab_is_not_registered() -> None:
    data, affine = anat_phantom()
    mid = data.shape[2] // 2
    slab = data[:, :, mid - 12: mid + 12]
    res = anat.check(slab, affine, suffix="T2starw", sidecar={})
    assert res.facts["slab"] and res.facts["registration"].get("skipped")
    assert any(f.key == "slab" for f in res.findings)
    assert res.value("fov_cut") is None


def test_the_check_of_a_file_reads_its_sidecar(tmp_path) -> None:
    from tests.unit.qc_phantoms import dataset, save

    root = dataset(tmp_path / "ds")
    data, affine = anat_phantom()
    path = save(data, affine, root / "sub-01" / "anat" / "sub-01_T1w.nii.gz",
                {"DeidentificationMethod": ["mri_deface"]})
    res = anat.check_file(path, config=fast(QcConfig()))
    assert res.suffix == "T1w"
    assert res.facts["defaced"] and "mri_deface" in res.facts["defacing_record"]

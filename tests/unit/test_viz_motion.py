"""Head motion and the per-volume QC rows: what fMRIPrep wrote is read, what
it did not is estimated, and the estimate recovers a motion we made."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import ndimage  # noqa: E402

from bidsmgr.qc.config import DEFAULT  # noqa: E402
from bidsmgr.viz.compute import motion, qc  # noqa: E402
from bidsmgr.viz.data.volume import array_volume  # noqa: E402


def _phantom(shape=(40, 40, 28)) -> np.ndarray:
    """A smooth head-like object with structure inside (blobs of several
    sizes), so a rigid motion changes the image everywhere."""
    rng = np.random.default_rng(3)
    vol = np.zeros(shape, dtype=np.float64)
    grid = np.indices(shape).astype(float)
    centre = (np.asarray(shape) - 1) / 2
    r = np.sqrt((((grid.T - centre) / (np.asarray(shape) * 0.38)) ** 2).sum(axis=-1)).T
    vol += 600.0 * (r < 1.0)
    for _ in range(25):
        c = centre + rng.uniform(-0.25, 0.25, 3) * np.asarray(shape)
        s = rng.uniform(1.5, 4.0)
        vol += rng.uniform(100, 400) * np.exp(-(((grid.T - c) ** 2).sum(axis=-1).T) / (2 * s * s))
    return ndimage.gaussian_filter(vol, 1.0)


def _moved(vol: np.ndarray, trans_mm, rot_vec, spacing: float) -> np.ndarray:
    """``vol`` with the object moved by rotation ``rot_vec`` about the grid
    centre and translation ``trans_mm``: frame(y) = vol(R^T (y - t))."""
    rot = motion._rotation(np.asarray(rot_vec, float))
    centre = (np.asarray(vol.shape) - 1) / 2.0
    m = rot.T
    offset = centre - m @ centre - m @ (np.asarray(trans_mm, float) / spacing)
    return ndimage.affine_transform(vol, m, offset=offset, order=3, mode="nearest")


def test_the_estimate_recovers_a_known_motion() -> None:
    spacing = 3.0
    base = _phantom()
    truth = np.array([
        [0, 0, 0, 0, 0, 0],
        [0.3, 0, 0, 0, 0, 0],
        [0.3, -0.4, 0.2, 0, 0, 0],
        [0.3, -0.4, 0.2, np.radians(0.5), 0, 0],
        [0.1, 0.2, -0.3, np.radians(0.5), np.radians(-0.4), np.radians(0.3)],
        [0, 0, 0, 0, 0, 0],
    ])
    frames = np.stack([_moved(base, p[:3], p[3:], spacing) for p in truth], axis=-1)
    src = array_volume(frames.astype(np.float32), np.diag([spacing] * 3 + [1.0]),
                       path=Path("bold.nii.gz"))
    got = motion.estimate(src)
    assert got.source == "estimated"
    assert np.abs(got.params[:, :3] - truth[:, :3]).max() < 0.06, "translation off by > 0.06 mm"
    assert np.degrees(np.abs(got.params[:, 3:] - truth[:, 3:])).max() < 0.06
    true_fd = motion.framewise_displacement(truth)
    assert np.allclose(got.fd[1:], true_fd[1:], atol=0.12)


def test_framewise_displacement_is_powers() -> None:
    p = np.zeros((3, 6))
    p[1, 0] = 0.2
    p[2, 0] = 0.2
    p[2, 3] = 0.01
    fd = motion.framewise_displacement(p)
    assert np.isnan(fd[0])
    assert fd[1] == pytest.approx(0.2)
    assert fd[2] == pytest.approx(0.5), "0.01 rad on a 50 mm sphere is 0.5 mm"


def _table(path: Path, header: str, n: int = 5) -> Path:
    cols = header.split("\t")
    rows = ["\t".join("n/a" if (i == 0 and "isplacement" in c) else f"{i * 0.1:g}"
                      for c in cols) for i in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


NEW = "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\tframewise_displacement"
OLD = "X\tY\tZ\tRotX\tRotY\tRotZ\tFramewiseDisplacement"


@pytest.mark.parametrize("header", [NEW, OLD])
def test_both_spellings_of_the_confounds_are_read(tmp_path, header) -> None:
    got = motion.read_confounds(_table(tmp_path / "c.tsv", header))
    assert got.source == "confounds" and got.params.shape == (5, 6)
    assert np.isnan(got.fd[0]) and got.fd[1] == pytest.approx(0.1)


def test_a_table_without_motion_is_refused(tmp_path) -> None:
    p = tmp_path / "c.tsv"
    p.write_text("global_signal\n1\n2\n")
    with pytest.raises(ValueError):
        motion.read_confounds(p)


def test_the_confounds_are_found_beside_and_under_derivatives(tmp_path) -> None:
    root = tmp_path / "ds"
    bold = root / "sub-01" / "ses-1" / "func" / "sub-01_ses-1_task-rest_run-2_bold.nii.gz"
    bold.parent.mkdir(parents=True)
    bold.write_bytes(b"")
    assert motion.confounds_for(bold, root) is None
    deriv = (root / "derivatives" / "fmriprep" / "sub-01" / "ses-1" / "func"
             / "sub-01_ses-1_task-rest_run-2_desc-confounds_timeseries.tsv")
    _table(deriv, NEW)
    assert motion.confounds_for(bold, root) == deriv
    beside = bold.parent / "sub-01_ses-1_task-rest_run-2_desc-confounds_timeseries.tsv"
    _table(beside, NEW)
    assert motion.confounds_for(bold, root) == beside, "the run's own folder first"
    # A preprocessed run names the same table: its space/desc are dropped.
    pre = bold.parent / "sub-01_ses-1_task-rest_run-2_space-MNI_desc-preproc_bold.nii.gz"
    assert motion.confounds_for(pre, root) == beside


def test_confounds_of_another_length_are_not_used(tmp_path) -> None:
    rng = np.random.default_rng(0)
    data = (_phantom((16, 16, 12))[..., None] + rng.normal(0, 1, (16, 16, 12, 5)))
    bold = tmp_path / "sub-01_task-x_bold.nii.gz"
    src = array_volume(data.astype(np.float32), np.diag([3.0, 3.0, 3.0, 1.0]), path=bold)
    _table(tmp_path / "sub-01_task-x_desc-confounds_timeseries.tsv", NEW, n=7)
    assert motion.motion_for(src, bold, tmp_path).source == "estimated"
    _table(tmp_path / "sub-01_task-x_desc-confounds_timeseries.tsv", NEW, n=5)
    assert motion.motion_for(src, bold, tmp_path).source == "confounds"


# ---------------------------------------------------------------------------
# Outliers, slice spikes, the carpet
# ---------------------------------------------------------------------------


def _series(n=40, spike_at=None, slice_spike=None):
    rng = np.random.default_rng(1)
    head = _phantom((20, 20, 12))
    data = head[..., None] + rng.normal(0, 4.0, head.shape + (n,))
    data += np.linspace(0, 30, n)            # slow drift: never an outlier
    if spike_at is not None:
        data[..., spike_at] += 150.0
    if slice_spike is not None:
        t, z = slice_spike
        data[:, :, z, t] += 120.0
    return array_volume(data.astype(np.float32), np.eye(4), path=Path("b.nii.gz"))


def test_a_spiked_volume_has_many_outlier_voxels() -> None:
    src = _series(spike_at=17)
    s = qc.sample_series(src)
    frac = qc.outlier_fraction(s["series"])
    assert int(np.nanargmax(frac)) == 17 and frac[17] > 0.5
    others = np.delete(frac, 17)
    limit = DEFAULT.bold.outlier_limit_pct / 100.0
    assert np.nanmax(others) < limit, "drift or noise counted as outliers"


def test_a_spike_in_one_slice_is_found_with_its_slice() -> None:
    src = _series(slice_spike=(9, 5))
    s = qc.sample_series(src)
    sp = qc.slice_spikes(s["slice_means"])
    assert list(sp["flagged"]) == [9] and list(sp["slice"]) == [5]
    # A change of the whole volume is not a slice spike.
    whole = qc.slice_spikes(qc.sample_series(_series(spike_at=9))["slice_means"])
    assert 9 not in list(whole["flagged"])


def test_the_carpet_runs_from_the_edge_inwards() -> None:
    src = _series(spike_at=12)
    s = qc.sample_series(src)
    img = qc.carpet(s["series"], s["depth"], rows=30)
    assert img.shape == (30, src.loaded_frames)
    assert np.nanmax(np.abs(img)) <= 3.0
    assert np.all(img[:, 12] > 1.0), "a spike is a band across every row"
    assert np.all(np.diff(np.sort(s["depth"])) >= 0)


def test_detrending_removes_a_quadratic_drift() -> None:
    u = np.linspace(0, 1, 50)[:, None]
    y = 3.0 + 2.0 * u + 5.0 * u ** 2
    assert np.abs(qc.detrend(y)).max() < 1e-9

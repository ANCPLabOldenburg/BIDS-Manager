"""Reading volumes: header now, voxels streamed in the file's own dtype."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bidsmgr.viz.data.volume import StreamCancelled, open_volume


def _save(tmp_path: Path, name: str, data, affine=None, *, slope=None, inter=None):
    img = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    if slope is not None:
        img.header.set_slope_inter(slope, inter or 0.0)
    path = tmp_path / name
    nib.save(img, str(path))
    return path


@pytest.mark.parametrize("ext", [".nii.gz", ".nii"])
def test_streamed_frames_equal_nibabel(tmp_path: Path, ext: str) -> None:
    data = np.random.default_rng(0).integers(0, 3000, (6, 7, 5, 9)).astype(np.int16)
    path = _save(tmp_path, "bold" + ext, data)
    src = open_volume(path)
    assert src.n_frames == 9 and src.is_4d and src.spatial == (6, 7, 5)
    assert src.loaded_frames == 0
    src.stream()
    assert src.fully_loaded
    ref = nib.load(str(path)).get_fdata()
    for t in range(9):
        np.testing.assert_array_equal(src.frame(t), ref[..., t])


def test_storage_stays_in_the_file_dtype(tmp_path: Path) -> None:
    """int16 on disk is held as int16: a quarter of float64."""
    data = np.zeros((10, 10, 10, 4), dtype=np.int16)
    src = open_volume(_save(tmp_path, "a.nii.gz", data))
    src.stream()
    assert src._raw.dtype == np.int16
    assert src.resident_bytes == data.nbytes
    assert src.facts.disk_dtype == "int16"


def test_scaling_is_applied_where_drawn(tmp_path: Path) -> None:
    data = np.arange(4 * 4 * 4, dtype=np.int16).reshape(4, 4, 4)
    path = _save(tmp_path, "s.nii.gz", data, slope=0.5, inter=10.0)
    src = open_volume(path)
    src.stream()
    np.testing.assert_allclose(src.frame(0), nib.load(str(path)).get_fdata())
    assert src.value_at((1, 2, 3), 0) == pytest.approx(float(data[1, 2, 3]) * 0.5 + 10.0)


def test_progress_reports_and_frame_zero_is_usable_early(tmp_path: Path) -> None:
    data = np.ones((40, 40, 40, 20), dtype=np.float32)
    src = open_volume(_save(tmp_path, "big.nii.gz", data))
    seen: list[tuple[int, int]] = []
    ready_at_first: list[bool] = []

    def progress(done, total):
        seen.append((done, total))
        ready_at_first.append(src.frame_ready(0))

    src.stream(progress=progress)
    assert seen[-1] == (20, 20)
    assert ready_at_first[0]


def test_cancelling_stops_the_read(tmp_path: Path) -> None:
    # Enough frames for several read chunks: cancelling acts between chunks.
    data = np.ones((40, 40, 40, 120), dtype=np.float32)
    src = open_volume(_save(tmp_path, "c.nii.gz", data))
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(StreamCancelled):
        src.stream(cancel=cancel)
    assert not src.fully_loaded


def test_voxel_series_reads_loaded_frames(tmp_path: Path) -> None:
    data = np.random.default_rng(2).random((5, 5, 5, 12)).astype(np.float32)
    src = open_volume(_save(tmp_path, "t.nii.gz", data))
    src.stream()
    np.testing.assert_allclose(src.voxel_series((1, 2, 3)), data[1, 2, 3, :])


def test_a_2d_image_is_one_slice(tmp_path: Path) -> None:
    src = open_volume(_save(tmp_path, "flat.nii.gz", np.ones((8, 9), dtype=np.float32)))
    assert src.spatial == (8, 9, 1) and src.n_frames == 1
    src.stream()
    assert src.frame(0).shape == (8, 9, 1)


def test_five_d_frames_flatten_time_first(tmp_path: Path) -> None:
    data = np.random.default_rng(4).random((3, 3, 3, 4, 2)).astype(np.float32)
    src = open_volume(_save(tmp_path, "five.nii.gz", data))
    assert src.n_frames == 8 and not src.is_rgb
    src.stream()
    np.testing.assert_allclose(src.frame(5), data[..., 1, 1])


def test_structured_rgb_is_one_colour_frame(tmp_path: Path) -> None:
    rgb = np.zeros((4, 4, 3), dtype=[("R", "u1"), ("G", "u1"), ("B", "u1")])
    rgb["R"] = 200
    rgb["B"] = 30
    src = open_volume(_save(tmp_path, "fa.nii.gz", rgb))
    assert src.is_rgb and src.channels == 3 and src.n_frames == 1
    src.stream()
    frame = src.frame(0)
    assert frame.shape == (4, 4, 3, 3)
    assert frame[0, 0, 0, 0] == 200 and frame[0, 0, 0, 2] == 30


def test_intent_coded_rgb_vector_is_colour(tmp_path: Path) -> None:
    data = np.zeros((3, 3, 3, 1, 3), dtype=np.float32)
    data[..., 0, 1] = 0.5
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_intent("vector")
    path = tmp_path / "v1.nii.gz"
    nib.save(img, str(path))
    src = open_volume(path)
    assert src.is_rgb and src.rgb_layout == "planes" and src.n_frames == 1
    src.stream()
    frame = src.frame(0)
    assert frame.shape == (3, 3, 3, 3)
    assert frame[1, 1, 1].tolist() == pytest.approx([0.0, 0.5, 0.0])


def test_header_facts_and_tr(tmp_path: Path) -> None:
    img = nib.Nifti1Image(np.zeros((4, 4, 4, 3), dtype=np.float32), np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header.set_zooms((1.0, 1.0, 1.0, 2.5))
    path = tmp_path / "f.nii.gz"
    nib.save(img, str(path))
    src = open_volume(path)
    assert src.header_tr == pytest.approx(2.5)
    assert src.facts.space_unit == "mm" and src.facts.time_unit == "sec"
    assert src.facts.shape == (4, 4, 4, 3)


def test_robust_range_ignores_the_tails(tmp_path: Path) -> None:
    data = np.random.default_rng(5).normal(100, 10, (20, 20, 20)).astype(np.float32)
    data[0, 0, 0] = 1e9
    src = open_volume(_save(tmp_path, "r.nii.gz", data))
    src.stream()
    lo, hi = src.robust_range(0)
    assert 50 < lo < 100 < hi < 200


def test_series_range_samples_several_frames(tmp_path: Path) -> None:
    """PET's first frames hold almost no counts; the window must not come
    from them alone."""
    data = np.zeros((10, 10, 10, 6), dtype=np.float32)
    data[..., 0] = np.random.default_rng(6).random((10, 10, 10))
    for t in range(1, 6):
        data[..., t] = np.random.default_rng(t).random((10, 10, 10)) * 100
    src = open_volume(_save(tmp_path, "pet.nii.gz", data))
    src.stream()
    assert src.series_range()[1] > 10 * src.robust_range(0)[1]


def test_release_drops_the_voxels(tmp_path: Path) -> None:
    src = open_volume(_save(tmp_path, "x.nii.gz", np.ones((4, 4, 4), np.float32)))
    src.stream()
    src.release()
    assert src.frame(0) is None and src.loaded_frames == 0


def test_no_file_handle_is_kept(tmp_path: Path) -> None:
    """An open handle on Windows blocks the Editor's rename and delete."""
    path = _save(tmp_path, "h.nii", np.ones((4, 4, 4), np.float32))
    src = open_volume(path)
    src.stream()
    renamed = path.with_name("h2.nii")
    path.rename(renamed)       # would fail on Windows with a handle held
    renamed.unlink()
    assert src.frame(0) is not None


def test_a_non_nifti_format_loads_in_one_call(tmp_path: Path) -> None:
    data = np.random.default_rng(7).random((5, 6, 7)).astype(np.float32)
    path = tmp_path / "t1.mgz"
    nib.save(nib.MGHImage(data, np.eye(4)), str(path))
    src = open_volume(path)
    assert not src.streamable
    src.stream()
    np.testing.assert_allclose(src.frame(0), data)


def test_a_series_larger_than_the_budget_reads_its_start(tmp_path: Path) -> None:
    """Better a usable start of the run than a machine swapping to a halt."""
    data = np.arange(4 * 4 * 4 * 10, dtype=np.int16).reshape(4, 4, 4, 10)
    src = open_volume(_save(tmp_path, "big.nii.gz", data))
    frame_bytes = 4 * 4 * 4 * 2
    src.stream(budget_bytes=3 * frame_bytes + 10)
    assert src.truncated and src.n_frames == 3 and src.n_frames_total == 10
    assert src.fully_loaded and src.resident_bytes == 3 * frame_bytes
    np.testing.assert_array_equal(src.frame(2), data[..., 2])
    assert not src.frame_ready(3)


def test_a_series_within_the_budget_is_read_whole(tmp_path: Path) -> None:
    data = np.ones((4, 4, 4, 5), dtype=np.float32)
    src = open_volume(_save(tmp_path, "ok.nii.gz", data))
    src.stream(budget_bytes=10 ** 9)
    assert not src.truncated and src.n_frames == src.n_frames_total == 5


def test_the_default_budget_is_half_the_free_memory() -> None:
    from bidsmgr.viz.data.volume import memory_budget_bytes

    assert memory_budget_bytes(512) == 512 * 1024 * 1024
    assert memory_budget_bytes(0) > 0

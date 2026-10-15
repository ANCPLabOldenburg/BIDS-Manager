"""The 2-D geometry: what each plane draws, and where every pixel is.

The reference for "draws the same as before" is the old viewer's own recipe,
re-stated here because the old module is gone: reorient the whole volume with
``nib.as_closest_canonical``, take ``vol[i, :, :]`` / ``vol[:, j, :]`` /
``vol[:, :, k]``, ``np.rot90`` it, then mirror for the RAS-off and
radiological toggles. The new code never reorients the volume (it slices the
file's own grid and turns the plane), so this is the test that says the two
are the same picture, for every storage order, both toggles, every slice.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bidsmgr.viz.compute import geometry as G
from bidsmgr.viz.data.volume import open_volume

PLANES = {0: "sagittal", 1: "coronal", 2: "axial"}
PLANE_HV = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def _rotation(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def _affines() -> dict[str, np.ndarray]:
    oblique = np.eye(4)
    oblique[:3, :3] = _rotation(0.3) @ np.diag([2.0, 3.0, 4.0])
    oblique[:3, 3] = [1.0, 2.0, 3.0]
    return {
        "RAS": np.diag([2.0, 3.0, 4.0, 1.0]),
        "LAS": np.diag([-2.0, 3.0, 4.0, 1.0]),
        "LPS": np.diag([-2.0, -3.0, 4.0, 1.0]),
        "LPI": np.diag([-2.0, -3.0, -4.0, 1.0]),
        "permuted": np.array([[0, 0, 4.0, 0], [2.0, 0, 0, 0],
                              [0, -3.0, 0, 0], [0, 0, 0, 1]]),
        "oblique": oblique,
    }


def _old_view(canon: np.ndarray, axis: int, index: int, flips) -> np.ndarray:
    plane = (canon[index, :, :] if axis == 0 else
             canon[:, index, :] if axis == 1 else canon[:, :, index])
    out = np.rot90(plane)
    h, v = PLANE_HV[axis]
    if flips[h] < 0:
        out = out[:, ::-1]
    if flips[v] < 0:
        out = out[::-1, :]
    return out


@pytest.mark.parametrize("name", list(_affines()))
@pytest.mark.parametrize("ras", [True, False])
@pytest.mark.parametrize("radiological", [False, True])
def test_every_slice_matches_the_old_viewer(tmp_path: Path, name: str, ras: bool,
                                            radiological: bool) -> None:
    affine = _affines()[name]
    data = np.random.default_rng(3).random((7, 9, 5)).astype(np.float32)
    path = tmp_path / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    canon_img = nib.as_closest_canonical(nib.load(str(path)))
    canon = canon_img.get_fdata()
    native_flip = [1.0, 1.0, 1.0]
    if not ras:
        for ras_axis, sign in nib.orientations.io_orientation(affine):
            native_flip[int(ras_axis)] = sign
    flips = (native_flip[0] * (-1.0 if radiological else 1.0), native_flip[1], native_flip[2])

    src = open_volume(path)
    src.stream()
    frame = src.frame(0)
    ornt = G.orientation_of(src.affine)
    for axis, plane in PLANES.items():
        for k in range(canon.shape[axis]):
            idx = [s // 2 for s in canon.shape]
            idx[axis] = k
            world = (canon_img.affine @ np.array([*idx, 1.0]))[:3]
            grid = G.grid_for(plane, affine=src.affine, spatial=src.spatial,
                              zooms=src.zooms3, orientation=ornt, cursor_world=world,
                              space="voxel", ras=ras, radiological=radiological)
            new = G.slice_voxel_grid(frame, grid)
            old = _old_view(canon, axis, k, flips)
            assert new.shape == old.shape, (plane, k)
            np.testing.assert_allclose(new, old, err_msg=f"{name} {plane} slice {k}")


def test_pixel_world_round_trip() -> None:
    affine = _affines()["oblique"]
    ornt = G.orientation_of(affine)
    grid = G.voxel_grid("coronal", affine, (7, 9, 5), ornt, 4)
    for c, r in ((0, 0), (3.5, 2), (6, 4)):
        w = grid.pixel_to_world(c, r)
        c2, r2, d = grid.world_to_pixel(w)
        assert c2 == pytest.approx(c) and r2 == pytest.approx(r) and d == pytest.approx(0)


def test_pixel_size_is_the_voxel_size() -> None:
    """A 1 x 1 x 5 mm scan: coronal pixels are 5 times taller than wide."""
    affine = np.diag([1.0, 1.0, 5.0, 1.0])
    grid = G.voxel_grid("coronal", affine, (10, 10, 4), G.orientation_of(affine), 5)
    assert grid.pixel_mm == pytest.approx((1.0, 5.0))
    assert grid.extent_mm == pytest.approx((10.0, 20.0))


@pytest.mark.parametrize("radiological, expected", [
    (False, {"left": "L", "right": "R", "top": "A", "bottom": "P"}),
    (True, {"left": "R", "right": "L", "top": "A", "bottom": "P"}),
])
def test_letters_are_read_off_the_grid(radiological: bool, expected: dict) -> None:
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])   # stored left-to-right reversed
    ornt = G.orientation_of(affine)
    grid = G.grid_for("axial", affine=affine, spatial=(8, 8, 8), zooms=(1, 1, 1),
                      orientation=ornt, cursor_world=(0, 4, 4), space="voxel",
                      ras=True, radiological=radiological)
    assert G.plane_labels("axial", grid) == expected


def test_sagittal_shows_the_face_on_the_right() -> None:
    affine = np.eye(4)
    grid = G.grid_for("sagittal", affine=affine, spatial=(8, 8, 8), zooms=(1, 1, 1),
                      orientation=G.orientation_of(affine), cursor_world=(4, 4, 4),
                      space="voxel", ras=True, radiological=False)
    assert G.plane_labels("sagittal", grid) == {"left": "P", "right": "A", "top": "S", "bottom": "I"}


def test_world_grid_samples_the_same_values_on_an_aligned_volume() -> None:
    """With no obliquity, the resampled world plane equals the voxel plane."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    data = np.random.default_rng(1).random((6, 7, 8)).astype(np.float32)
    ornt = G.orientation_of(affine)
    world = affine @ np.array([3, 3, 4, 1.0])
    vgrid = G.grid_for("axial", affine=affine, spatial=data.shape, zooms=(2, 2, 2),
                       orientation=ornt, cursor_world=world[:3], space="voxel",
                       ras=True, radiological=False)
    wgrid = G.grid_for("axial", affine=affine, spatial=data.shape, zooms=(2, 2, 2),
                       orientation=ornt, cursor_world=world[:3], space="world",
                       ras=True, radiological=False)
    voxel_plane = G.slice_voxel_grid(data, vgrid)
    world_plane = G.sample_grid(data, np.linalg.inv(affine), wgrid, order=0)
    assert voxel_plane.shape == world_plane.shape
    np.testing.assert_allclose(world_plane, voxel_plane)


def test_world_grid_draws_a_tilted_volume_upright() -> None:
    """An oblique volume's world plane is aligned to the scanner axes."""
    affine = np.eye(4)
    affine[:3, :3] = _rotation(0.4)
    grid = G.world_grid("axial", affine, (20, 20, 10), (1, 1, 1), 5.0)
    assert grid.col_vec == pytest.approx([1.0, 0.0, 0.0])
    assert grid.row_vec == pytest.approx([0.0, -1.0, 0.0])
    assert G.obliquity_deg(affine) == pytest.approx(np.degrees(0.4), abs=0.01)


def test_samples_outside_the_volume_are_empty() -> None:
    data = np.ones((4, 4, 4), dtype=np.float32)
    affine = np.eye(4)
    grid = G.world_grid("axial", np.diag([3.0, 3.0, 3.0, 1.0]), (4, 4, 4), (3, 3, 3), 1.0)
    out = G.sample_grid(data, np.linalg.inv(affine), grid, order=0)
    assert np.isnan(out).any() and np.isfinite(out).any()


def test_snap_to_voxel_centre() -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    snapped = G.snap_to_voxel(affine, (5, 5, 5), (2.9, 4.2, 100.0))
    assert snapped == pytest.approx([2.0, 4.0, 8.0])


def test_shear_is_measured() -> None:
    affine = np.eye(4)
    affine[0, 2] = 0.5   # x leans with z
    assert G.shear_deg(affine) > 20
    assert G.shear_deg(np.eye(4)) == pytest.approx(0.0)


def test_degenerate_axis_still_gives_a_permutation() -> None:
    affine = np.diag([1.0, 1.0, 0.0, 1.0])
    ornt = G.orientation_of(affine)
    assert sorted(ornt.ras_of) == [0, 1, 2]

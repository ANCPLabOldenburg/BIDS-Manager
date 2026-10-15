"""Shapes drawn as geometry: the MRS voxel's box, exactly.

``bidsmgr.viz.compute.shapes`` and the views that use it. A box sampled onto
an anatomy's pixels was only as precise as those pixels (a 20 mm voxel came
out 22 mm tall on 2 mm rows, with stepped edges) and, in 3-D, sampled at its
own 20 mm, a blur up to 10 mm off. These tests pin the exact versions.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bidsmgr.viz import colorbar, render2d, render3d, views
from bidsmgr.viz.compute.shapes import GridBox, outline_mask, section


def _rot(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def _box(size=20.0, centre=(10.0, -5.0, 30.0), deg=0.0, shape=(1, 1, 1)) -> GridBox:
    a = np.eye(4)
    a[:3, :3] = _rot(deg) * size
    n = np.asarray(shape, dtype=float)
    a[:3, 3] = np.asarray(centre) - a[:3, :3] @ ((n - 1) / 2.0)
    return GridBox(a, tuple(shape))


class TestTheBox:
    def test_corners_are_the_faces_not_the_centres(self):
        b = _box()
        c = b.corners()
        assert np.allclose(c.min(axis=0), [0.0, -15.0, 20.0])
        assert np.allclose(c.max(axis=0), [20.0, 5.0, 40.0])
        assert np.allclose(b.centre(), [10.0, -5.0, 30.0])
        assert np.allclose(b.size_mm(), [20.0, 20.0, 20.0])

    def test_a_grid_spans_every_voxel(self):
        b = _box(size=10.0, shape=(4, 2, 1))
        assert np.allclose(b.size_mm(), [40.0, 20.0, 10.0])
        assert np.allclose(b.centre(), [10.0, -5.0, 30.0])

    def test_contains_to_the_face(self):
        b = _box(deg=30.0)
        c = b.centre()
        axis = _rot(30.0)[:, 0]
        pts = np.array([c, c + 9.99 * axis, c + 10.01 * axis])
        assert list(b.contains(pts)) == [True, True, False]


class TestTheCut:
    def test_a_square_through_the_middle(self):
        b = _box()
        poly = section(b, b.centre(), [0, 0, 1])
        assert len(poly) == 4
        assert np.allclose(sorted(poly[:, 0]), [0, 0, 20, 20])

    def test_a_diagonal_cut_of_a_cube_is_a_hexagon(self):
        b = _box()
        poly = section(b, b.centre(), [1, 1, 1])
        assert len(poly) == 6
        side = np.linalg.norm(np.roll(poly, 1, axis=0) - poly, axis=1)
        assert np.allclose(side, side[0]), "regular: ordered around the plane"

    def test_a_plane_that_misses_is_nothing(self):
        assert section(_box(), [0, 0, 100.0], [0, 0, 1]).shape == (0, 3)

    def test_a_rotated_voxel_cut_is_rotated(self):
        b = _box(deg=25.0)
        poly = section(b, b.centre(), [0, 0, 1])
        edge = poly[1] - poly[0]
        angle = np.degrees(np.arctan2(edge[1], edge[0])) % 90.0
        assert angle == pytest.approx(25.0, abs=1e-6)


def test_the_rim_is_inside_the_region():
    region = np.zeros((10, 10), dtype=bool)
    region[2:8, 3:9] = True
    rim = outline_mask(region, 1)
    assert rim.sum() == 2 * 6 + 2 * 4
    assert not (rim & ~region).any()


@pytest.fixture()
def anatomy_and_voxel(tmp_path: Path):
    """A 1 x 1 x 2 mm anatomy, and an MRS file with a 20 mm voxel turned 10
    degrees, as dcm2niix writes one (complex FID along the 4th axis)."""
    anat = np.random.default_rng(0).random((80, 80, 40)).astype(np.float32)
    a = np.diag([1.0, 1.0, 2.0, 1.0])
    a[:3, 3] = [-40.0, -40.0, -40.0]
    anat_path = tmp_path / "sub-01" / "anat" / "sub-01_T2w.nii.gz"
    anat_path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(anat, a), str(anat_path))
    m = np.eye(4)
    m[:3, :3] = _rot(10.0) * 20.0
    m[:3, 3] = [5.0, -3.0, 4.0]
    fid = np.ones((1, 1, 1, 64), dtype=np.complex64)
    mrs_path = tmp_path / "sub-01" / "mrs" / "sub-01_svs.nii.gz"
    mrs_path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(fid, m), str(mrs_path))
    return anat_path, mrs_path


class TestTheMrsVoxel:
    def test_it_is_a_shape(self, anatomy_and_voxel):
        anat, mrs = anatomy_and_voxel
        store = render2d.open_store(anat, overlays=[mrs])
        found = views.shape_layers(store)
        assert len(found) == 1
        layer, src = found[0]
        assert np.allclose(src.box.centre(), [5.0, -3.0, 4.0])
        assert colorbar.bars_for(store)[0].title.endswith("T2w.nii.gz"), "no bar for a box"

    def test_drawn_exactly_whatever_the_anatomy_pixels(self, anatomy_and_voxel):
        anat, mrs = anatomy_and_voxel
        store = render2d.open_store(anat, overlays=[mrs])
        store.run("cursor.set_world", x=5.0, y=-3.0, z=4.0, snap=False)
        bare = render2d.open_store(anat)
        bare.run("cursor.set_world", x=5.0, y=-3.0, z=4.0, snap=False)
        for plane, ppm in (("axial", 4.0), ("coronal", 4.0)):
            drawn = render2d.panel(store, plane, px_per_mm=ppm)
            plain = render2d.panel(bare, plane, px_per_mm=ppm)
            changed = np.any(drawn != plain, axis=-1)
            area_mm2 = changed.sum() / ppm ** 2
            assert area_mm2 == pytest.approx(400.0, rel=0.03), plane

    def test_the_slices_leave_it_to_the_geometry(self, anatomy_and_voxel):
        anat, mrs = anatomy_and_voxel
        store = render2d.open_store(anat, overlays=[mrs])
        bare = render2d.open_store(anat)
        for s in (store, bare):
            s.run("cursor.set_world", x=5.0, y=-3.0, z=4.0, snap=False)
        assert np.array_equal(render2d.slice_rgba(store, "axial").rgba,
                              render2d.slice_rgba(bare, "axial").rgba)


def test_3d_uniforms_take_a_ray_into_the_box():
    canon = np.diag([2.0, 2.0, 2.0, 1.0])
    canon[:3, 3] = [-50.0, -60.0, -40.0]
    dims, half = (50, 60, 40), np.array([0.5, 0.6, 0.4])
    box = _box(size=20.0, centre=(0.0, 0.0, 0.0))
    mats, lens = render3d.shape_uniforms([box], canon, dims, half)
    assert mats.shape == (render3d.MAX_SHAPES, 4, 4)
    for world, unit in (((10.0, 0.0, 0.0), (1.0, 0.5, 0.5)), ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))):
        tex = render3d.texcoord_of_world(np.linalg.inv(canon), dims, np.asarray(world))
        ray = tex * 2.0 * half - half
        assert np.allclose((mats[0] @ np.append(ray, 1.0))[:3], unit)
    assert np.allclose(lens[0], 20.0 / (2.0 * np.asarray(dims)) * 2.0 * half)

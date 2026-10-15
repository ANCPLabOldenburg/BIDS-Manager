"""Overlays, atlases, quality maps and the MRS voxel: the Qt-free half.

``bidsmgr.viz.overlays``, ``viz.data.labels``, ``viz.compute.qc``, the layer
commands, outline drawing and the BIDS lookups they rely on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from bidsmgr.viz import views  # noqa: E402
from bidsmgr.viz.bids import anatomical_for, bids_context  # noqa: E402
from bidsmgr.viz.compute import intensity, qc  # noqa: E402
from bidsmgr.viz.data import labels as L  # noqa: E402
from bidsmgr.viz.data.volume import array_volume, open_volume  # noqa: E402
from bidsmgr.viz.overlays import classify, display_for, mrs_voxel, open_overlay  # noqa: E402
from bidsmgr.viz.scene import LabelTable, Scene, SourceRef, VolumeDisplay, VolumeLayer  # noqa: E402
from bidsmgr.viz.store import SceneStore  # noqa: E402
from tests.fixtures.signals import fid_at, write_mrs  # noqa: E402


def _save(path: Path, arr: np.ndarray, affine=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, np.eye(4) if affine is None else affine), str(path))
    return path


def _loaded(path: Path):
    src = open_volume(path)
    src.stream()
    return src


def _atlas(shape=(20, 20, 10)) -> np.ndarray:
    """Three blocks of one value each, as a segmentation looks."""
    a = np.zeros(shape, dtype=np.int16)
    a[2:8, 2:18, :] = 17
    a[10:18, 2:10, :] = 53
    a[10:18, 10:18, :] = 2
    return a


# ---------------------------------------------------------------------------
# Label tables
# ---------------------------------------------------------------------------


class TestLabelTables:
    def test_a_bids_table_is_read_with_its_colours(self, tmp_path):
        tsv = tmp_path / "x_dseg.tsv"
        tsv.write_text("index\tname\tcolor\n0\tBackground\t#000000\n"
                       "17\tLeft-Hippocampus\t#dcd814\n53\tRight-Hippocampus\t\n")
        table = L.read_label_tsv(tsv)
        assert table.labels[17] == "Left-Hippocampus"
        assert table.colors[17] == (0xdc, 0xd8, 0x14)
        assert 53 in table.labels and 53 not in table.colors

    def test_the_table_beside_the_image_is_found(self, tmp_path):
        img = _save(tmp_path / "sub-01_desc-aseg_dseg.nii.gz", _atlas())
        beside = tmp_path / "sub-01_desc-aseg_dseg.tsv"
        beside.write_text("index\tname\n17\tA\n")
        assert L.table_path_for(img) == beside

    def test_a_table_shared_at_the_derivative_root_is_found(self, tmp_path):
        """BIDS lets a pipeline name one table for every subject."""
        root = tmp_path / "derivatives" / "fs"
        (root / "dataset_description.json").parent.mkdir(parents=True)
        (root / "dataset_description.json").write_text("{}")
        shared = root / "desc-aseg_dseg.tsv"
        shared.write_text("index\tname\n17\tA\n")
        img = _save(root / "sub-01" / "anat" / "sub-01_desc-aseg_dseg.nii.gz", _atlas())
        assert L.table_path_for(img) == shared

    def test_distinct_values(self):
        assert list(L.distinct_values(np.array([0, 2, 2, 5]))) == [0, 2, 5]
        assert list(L.distinct_values(np.array([0.0, 3.0]))) == [0.0, 3.0]
        assert L.distinct_values(np.array([0.5, 1.0])) is None, "not integers"
        assert L.distinct_values(np.arange(5000)) is None, "a measurement, not labels"


# ---------------------------------------------------------------------------
# What an overlay is, and how it opens
# ---------------------------------------------------------------------------


class TestClassify:
    def test_a_segmentation_is_labels_named_from_its_table(self, tmp_path):
        img = _save(tmp_path / "sub-01_dseg.nii.gz", _atlas())
        (tmp_path / "sub-01_dseg.tsv").write_text(
            "index\tname\n2\tCortex\n17\tHippocampus\n53\tAmygdala\n")
        ov = open_overlay(img)
        assert ov.kind == "labels"
        assert ov.display.label_table.labels[17] == "Hippocampus"
        assert ov.display.interpolation == "nearest"
        assert "named from sub-01_dseg.tsv" in ov.note

    def test_an_unnamed_atlas_is_still_labels(self, tmp_path):
        img = _save(tmp_path / "atlas.nii.gz", _atlas())
        ov = open_overlay(img)
        assert ov.kind == "labels"
        assert set(ov.display.label_table.labels) == {2, 17, 53}
        assert "numbered" in ov.note

    def test_an_8_bit_anatomical_is_not_an_atlas(self, tmp_path):
        """256 distinct values like an atlas, but no regions of one value."""
        rng = np.random.default_rng(0)
        img = _save(tmp_path / "sub-01_T1w.nii.gz",
                    rng.integers(0, 256, (20, 20, 10)).astype(np.uint8))
        kind, _ = classify(_loaded(img), img)
        assert kind == "image"

    def test_a_mask(self, tmp_path):
        m = np.zeros((10, 10, 5), dtype=np.uint8)
        m[3:7, 3:7, :] = 1
        ov = open_overlay(_save(tmp_path / "sub-01_mask.nii.gz", m))
        assert ov.kind == "mask"
        assert ov.display.threshold_mode == "hide_below"

    def test_a_probability_map(self, tmp_path):
        p = np.random.default_rng(1).random((10, 10, 5)).astype(np.float32)
        ov = open_overlay(_save(tmp_path / "sub-01_label-GM_probseg.nii.gz", p))
        assert ov.kind == "probability"
        assert ov.display.window == (0.2, 1.0)

    def test_a_statistical_map_is_two_tailed(self, tmp_path):
        z = np.random.default_rng(2).normal(0, 1.5, (12, 12, 6)).astype(np.float32)
        ov = open_overlay(_save(tmp_path / "zstat.nii.gz", z))
        assert ov.kind == "two_tailed"
        d = ov.display
        assert (d.colormap, d.colormap_negative) == ("warm", "cool")
        assert d.window[0] > 0 and d.window_negative == d.window

    def test_a_positive_map_with_a_rounding_error_below_zero_is_not(self, tmp_path):
        v = np.random.default_rng(3).random((12, 12, 6)).astype(np.float32) * 100 + 1
        v[0, 0, 0] = -1e-4
        kind, _ = classify(_loaded(_save(tmp_path / "mean.nii.gz", v)), Path("mean.nii.gz"))
        assert kind == "image"

    def test_another_image_is_windowed_inside_the_head(self, tmp_path):
        """Over every non-zero voxel the background noise would pull the low
        end down and every tissue would saturate."""
        v = np.random.default_rng(4).random((20, 20, 10)).astype(np.float32) * 5
        v[5:15, 5:15, 2:8] = 1000 + np.random.default_rng(5).random((10, 10, 6)) * 200
        display, kind, _note = display_for(_loaded(_save(tmp_path / "x.nii.gz", v)),
                                           Path("x.nii.gz"))
        assert kind == "image"
        assert display.window[0] > 900, "the noise floor does not set the window"


# ---------------------------------------------------------------------------
# In-memory sources and the MRS voxel
# ---------------------------------------------------------------------------


class TestArrayVolume:
    def test_it_is_loaded_and_has_nothing_to_stream(self, tmp_path):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        src = array_volume(data, np.eye(4), path=tmp_path / "x.nii.gz", name="X")
        assert src.fully_loaded and src.in_memory
        src.stream()                      # a no-op, not a file read
        assert np.array_equal(src.frame(0), data)
        assert src.value_at((1, 2, 3), 0) == pytest.approx(23.0)

    def test_a_series_has_its_frames(self, tmp_path):
        data = np.random.default_rng(0).random((3, 3, 3, 5)).astype(np.float32)
        src = array_volume(data, np.eye(4) * 2, path=tmp_path / "x", name="X")
        assert src.n_frames == 5 and src.zooms3 == (2.0, 2.0, 2.0)
        assert np.allclose(src.frame(4), data[..., 4])


class TestMrsVoxel:
    def test_the_box_covers_exactly_the_voxel(self, tmp_path):
        """20 mm voxel centred at (10, -5, 30): inside within 10 mm of the
        centre, outside beyond it, whatever the grid it is drawn on."""
        affine = np.diag([20.0, 20.0, 20.0, 1.0])
        affine[:3, 3] = (10.0, -5.0, 30.0)
        fid = fid_at(2.01, 512)
        path = write_mrs(tmp_path / "sub-01" / "mrs", "sub-01_svs.nii.gz", fid)
        img = nib.load(str(path))
        nib.save(nib.Nifti2Image(np.asarray(img.dataobj), affine, img.header), str(path))
        ov = mrs_voxel(path)
        src = ov.source
        assert ov.display.outline_px > 0 and ov.display.interpolation == "nearest"

        def value(world):
            vox = views.voxel_in(src, np.asarray(world, dtype=float))
            return 0.0 if vox is None else float(src.value_at(vox, 0))

        assert value((10, -5, 30)) == 1.0
        assert value((19, 4, 39)) == 1.0, "just inside a corner"
        assert value((21, -5, 30)) == 0.0, "just outside a face"
        assert value((10, -5, 41)) == 0.0

    def test_it_is_what_an_mrs_file_becomes_as_an_overlay(self, tmp_path):
        path = write_mrs(tmp_path / "sub-01" / "mrs", "sub-01_svs.nii.gz", fid_at(2.01, 256))
        ov = open_overlay(path)
        assert ov.name == "Voxel of sub-01_svs.nii.gz"
        assert ov.note == "the spectroscopy voxel"


# ---------------------------------------------------------------------------
# The layer commands
# ---------------------------------------------------------------------------


def _store(tmp_path) -> SceneStore:
    base = array_volume(np.zeros((4, 4, 4), np.float32), np.eye(4), path=tmp_path / "b",
                        name="b")
    over = array_volume(np.ones((4, 4, 4), np.float32), np.eye(4), path=tmp_path / "o",
                        name="o")
    store = SceneStore()
    scene = Scene()
    scene.sources = {"vol0": SourceRef(id="vol0", path="b", kind="volume")}
    scene.layers = [VolumeLayer(id="base", source="vol0")]
    store.replace_scene(scene)
    store.sources = {"vol0": base, "ovl1": over, "ovl2": over}
    return store


class TestLayerCommands:
    def test_add_puts_it_on_top_and_records_where_it_came_from(self, tmp_path):
        store = _store(tmp_path)
        store.run("layer.add", id="overlay1", source="ovl1", name="o",
                  display={"colormap": "hot"}, path="/data/o.nii.gz")
        assert [lay.id for lay in store.scene.layers] == ["base", "overlay1"]
        assert store.scene.layers[1].display.colormap == "hot"
        assert store.scene.sources["ovl1"].path == "/data/o.nii.gz"

    def test_add_needs_a_loaded_source_and_a_new_id(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="no source"):
            store.run("layer.add", id="x", source="nope")
        store.run("layer.add", id="x", source="ovl1")
        with pytest.raises(ValueError, match="already"):
            store.run("layer.add", id="x", source="ovl1")

    def test_add_is_undone(self, tmp_path):
        store = _store(tmp_path)
        store.run("layer.add", id="x", source="ovl1")
        store.undo()
        assert [lay.id for lay in store.scene.layers] == ["base"]
        store.redo()
        assert [lay.id for lay in store.scene.layers] == ["base", "x"]

    def test_the_base_cannot_be_removed_or_moved(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="base image"):
            store.run("layer.remove", layer="base")
        with pytest.raises(ValueError, match="base image"):
            store.run("layer.move", layer="base", by=1)

    def test_move_stays_above_the_base(self, tmp_path):
        store = _store(tmp_path)
        store.run("layer.add", id="a", source="ovl1")
        store.run("layer.add", id="b", source="ovl2")
        store.run("layer.move", layer="b", by=-5)
        assert [lay.id for lay in store.scene.layers] == ["base", "b", "a"]
        assert store.run("layer.move", layer="b", by=-1) == frozenset(), "already lowest"
        store.run("layer.remove", layer="a")
        assert [lay.id for lay in store.scene.layers] == ["base", "b"]


# ---------------------------------------------------------------------------
# Outlines
# ---------------------------------------------------------------------------


class TestOutlines:
    def test_edges_of_a_square(self):
        m = np.zeros((7, 7), dtype=np.int64)
        m[1:6, 1:6] = 1
        e = intensity.edges(m)
        assert e[1, 1] and e[1, 3] and e[5, 5]
        assert not e[3, 3], "the inside is not an edge"
        assert not e[0, 0], "nor is the outside"

    def test_a_wider_edge_goes_inward(self):
        m = np.zeros((9, 9), dtype=np.int64)
        m[1:8, 1:8] = 1
        assert intensity.edges(m, 2)[2, 4] and not intensity.edges(m, 1)[2, 4]

    def test_between_two_labels_both_sides_are_edges(self):
        m = np.zeros((4, 6), dtype=np.int64)
        m[:, :3], m[:, 3:] = 2, 5
        e = intensity.edges(m)
        assert e[:, 2].all() and e[:, 3].all()

    def test_an_outlined_overlay_keeps_only_its_edge(self):
        vals = np.zeros((9, 9), dtype=np.float32)
        vals[2:7, 2:7] = 1.0
        disp = VolumeDisplay(colormap="red", window=(0.5, 1.0), threshold_mode="hide_below",
                             outline_px=1.0)
        rgba = intensity.colorize(vals, disp, is_base=False)
        assert rgba[2, 4, 3] > 0 and rgba[4, 4, 3] == 0 and rgba[0, 0, 3] == 0

    def test_an_outlined_atlas_keeps_only_its_borders(self):
        vals = np.zeros((6, 6), dtype=np.float32)
        vals[:, :3], vals[:, 3:] = 2, 5
        disp = VolumeDisplay(label_table=LabelTable(labels={2: "a", 5: "b"}), outline_px=1.0)
        rgba = intensity.colorize(vals, disp, is_base=False)
        assert rgba[3, 2, 3] > 0 and rgba[3, 0, 3] == 0


# ---------------------------------------------------------------------------
# Quality maps
# ---------------------------------------------------------------------------


class TestQualityMaps:
    def _series(self, tmp_path):
        rng = np.random.default_rng(7)
        data = np.zeros((8, 8, 4, 30), dtype=np.float32)
        data[2:6, 2:6, :, :] = 1000 + rng.normal(0, 10, (4, 4, 4, 30))
        return data, array_volume(data, np.eye(4), path=tmp_path / "bold", name="bold")

    def test_the_moments_match_numpy(self, tmp_path):
        data, src = self._series(tmp_path)
        mean, sd = qc.series_moments(src, detrend=0)
        assert np.allclose(mean, data.mean(axis=-1), atol=1e-3)
        assert np.allclose(sd, data.std(axis=-1, ddof=1), atol=1e-2)

    def test_drift_is_not_instability(self, tmp_path):
        """A slow scanner drift is removed before the SD is taken: it would
        otherwise lower every temporal SNR."""
        rng = np.random.default_rng(3)
        t = np.arange(120)
        noise = rng.normal(0, 5, 120)
        drift = 0.4 * t + 0.002 * (t - 60) ** 2
        data = np.zeros((4, 4, 2, 120), dtype=np.float32)
        data[1:3, 1:3, :, :] = 1000 + drift + noise
        src = array_volume(data, np.eye(4), path=tmp_path / "drift", name="drift")
        _mean, raw_sd = qc.series_moments(src, detrend=0)
        _mean, sd = qc.series_moments(src, detrend=2)
        resid = noise - np.polyval(np.polyfit(t, noise + drift, 2), t) + drift
        assert sd[1, 1, 0] == pytest.approx(np.sqrt((resid ** 2).sum() / (120 - 3)), rel=1e-3)
        assert raw_sd[1, 1, 0] > 2 * sd[1, 1, 0]

    def test_bright_first_volumes_are_skipped(self, tmp_path):
        data, _src = self._series(tmp_path)
        data = np.concatenate([data[..., :3] * 1.6, data], axis=-1)   # three dummies
        src = array_volume(data, np.eye(4), path=tmp_path / "dummy", name="dummy")
        assert qc.non_steady_state(src) == 3
        _values, facts = qc.quality_map(src, "tsnr")
        assert facts["skipped"] == 3 and facts["volumes"] == 30

    def test_tsnr_is_mean_over_sd_inside_the_head_and_nothing_outside(self, tmp_path):
        data, src = self._series(tmp_path)
        tsnr, facts = qc.quality_map(src, "tsnr", detrend=0)
        inside = tsnr[3, 3, 1]
        assert inside == pytest.approx(data[3, 3, 1].mean() / data[3, 3, 1].std(ddof=1), rel=1e-3)
        assert np.isnan(tsnr[0, 0, 0]), "outside the head is drawn as nothing"
        assert facts == {"volumes": 30, "skipped": 0, "detrend": 0, "map": "tsnr"}

    def test_one_volume_is_refused(self, tmp_path):
        src = array_volume(np.ones((3, 3, 3)), np.eye(4), path=tmp_path / "x", name="x")
        with pytest.raises(ValueError, match="at least two"):
            qc.quality_map(src, "mean")

    def test_the_overlay_shows_the_low_end_and_carries_its_reading(self, tmp_path):
        _data, src = self._series(tmp_path)
        ov = qc.quality_overlay(src, "tsnr")
        assert ov.name == "Temporal SNR of bold"
        assert ov.display.window[0] == 0.0, "the low end is what a QC map is for"
        assert ov.display.threshold_mode == "range"
        assert ov.source.in_memory and ov.source.quantity.startswith("tSNR")
        notes = ov.source.notes
        assert notes["qc"] == "tsnr" and notes["stats"]["voxels"] == 64
        assert "median" in notes["summary"] and "SAME protocol" in notes["help"]

    def test_the_summary(self):
        values = np.full((10, 10), np.nan)
        values[:5, :] = np.linspace(5, 95, 50).reshape(5, 10)
        stats = qc.summary(values, "tsnr")
        assert stats["voxels"] == 50 and stats["median"] == pytest.approx(50.0)
        assert stats["below"] == pytest.approx(np.mean(np.linspace(5, 95, 50) < 20))
        assert "%" in qc.describe_summary(stats, {"volumes": 100}, "tsnr")

    def test_dvars_flags_the_volume_that_jumps(self, tmp_path):
        data, _src = self._series(tmp_path)
        data[2:6, 2:6, :, 17] += 200.0                      # one bad volume
        src = array_volume(data, np.eye(4), path=tmp_path / "spike", name="spike")
        pv = qc.per_volume(src)
        assert np.isnan(pv["dvars"][0])
        assert set(pv["flagged"]) == {17, 18}, "into the spike and out of it"
        assert pv["global"].shape == (30,)


# ---------------------------------------------------------------------------
# The dataset around an image
# ---------------------------------------------------------------------------


class TestBidsLookups:
    def test_run_1_does_not_take_run_10s_physio(self, tmp_path):
        """A glob on the name matched run-10 when looking for run-1."""
        func = tmp_path / "sub-01" / "func"
        func.mkdir(parents=True)
        bold = _save(func / "sub-01_task-x_run-1_bold.nii.gz", np.zeros((2, 2, 2, 2)))
        for run in ("1", "10"):
            (func / f"sub-01_task-x_run-{run}_recording-cardiac_physio.tsv.gz").write_bytes(b"")
        ctx = bids_context(bold, tmp_path)
        assert [p.name for p in ctx.physio_paths] == [
            "sub-01_task-x_run-1_recording-cardiac_physio.tsv.gz"]

    def test_the_anatomy_of_the_same_session_comes_first(self, tmp_path):
        for ses in ("01", "02"):
            _save(tmp_path / "sub-01" / f"ses-{ses}" / "anat" / f"sub-01_ses-{ses}_T2w.nii.gz",
                  np.zeros((2, 2, 2)))
        _save(tmp_path / "sub-01" / "ses-02" / "anat" / "sub-01_ses-02_T1w.nii.gz",
              np.zeros((2, 2, 2)))
        mrs = tmp_path / "sub-01" / "ses-01" / "mrs" / "sub-01_ses-01_svs.nii.gz"
        mrs.parent.mkdir(parents=True)
        assert anatomical_for(mrs).name == "sub-01_ses-01_T2w.nii.gz"
        mrs2 = tmp_path / "sub-01" / "ses-02" / "mrs" / "sub-01_ses-02_svs.nii.gz"
        assert anatomical_for(mrs2).name == "sub-01_ses-02_T1w.nii.gz", "T1w before T2w"

    def test_no_subject_no_anatomy(self, tmp_path):
        assert anatomical_for(tmp_path / "x.nii.gz") is None

    def test_a_computed_overlay_reads_out_without_indices(self, tmp_path):
        store = _store(tmp_path)
        store.run("layer.add", id="x", source="ovl1", name="tSNR")
        store.scene.cursor.world = (1.0, 1.0, 1.0)
        text = views.readout(store)
        assert "tSNR = 1" in text
        assert "tSNR (" not in text

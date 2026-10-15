"""Overlays in the viewer, the Layers panel, quality maps, the MRS voxel on
its anatomy, and "Draw over" from the Editor's tree.

The Qt-free half (what an overlay is, its look, the commands, outlines, the
maps) is in ``tests/unit/test_viz_overlays.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import QPoint  # noqa: E402
from PyQt6.QtWidgets import QMenu  # noqa: E402

from bidsmgr.gui.viz import Viewer  # noqa: E402
from tests.fixtures.signals import fid_at, write_mrs  # noqa: E402

pytestmark = pytest.mark.gui


def _save(path: Path, arr: np.ndarray, affine=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, np.eye(4) if affine is None else affine), str(path))
    return path


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    root = tmp_path / "Study"
    anat = root / "sub-01" / "anat"
    t1 = np.zeros((20, 20, 10), dtype=np.float32)
    t1[2:18, 2:18, 1:9] = 500.0
    _save(anat / "sub-01_T1w.nii.gz", t1)
    atlas = np.zeros((20, 20, 10), dtype=np.int16)
    atlas[4:10, 4:16, :] = 17
    atlas[11:16, 4:16, :] = 53
    _save(anat / "sub-01_dseg.nii.gz", atlas)
    (anat / "sub-01_dseg.tsv").write_text("index\tname\n17\tHippocampus\n53\tAmygdala\n")
    rng = np.random.default_rng(3)
    bold = np.zeros((20, 20, 10, 12), dtype=np.float32)
    bold[2:18, 2:18, 1:9, :] = 800 + rng.normal(0, 8, (16, 16, 8, 12))
    _save(root / "sub-01" / "func" / "sub-01_task-x_bold.nii.gz", bold)
    return root


def _t1(root):
    return root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"


def _atlas(root):
    return root / "sub-01" / "anat" / "sub-01_dseg.nii.gz"


def _bold(root):
    return root / "sub-01" / "func" / "sub-01_task-x_bold.nii.gz"


def _viewer(qtbot, kind="volume") -> Viewer:
    v = Viewer(kind=kind)
    qtbot.addWidget(v)
    v.resize(1000, 640)
    v.show()
    qtbot.waitExposed(v)
    return v


def _open(qtbot, v: Viewer, path: Path, root) -> Viewer:
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(path, root)
    v.qstore.flush()
    return v


def _add(qtbot, v: Viewer, path: Path) -> str:
    with qtbot.waitSignal(v.overlay_added, timeout=20_000) as added:
        assert v.add_overlay(path)
    v.qstore.flush()
    return added.args[0]


class TestAddingAnOverlay:
    def test_an_atlas_goes_over_the_image_and_names_its_regions(self, qtbot, ds):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        layer_id = _add(qtbot, v, _atlas(ds))
        assert [lay.id for lay in v.scene.layers] == ["base", layer_id]
        v.run("cursor.set_world", x=6.0, y=8.0, z=4.0)
        v.qstore.flush()
        assert "Hippocampus" in v.readout_text()

    def test_it_is_drawn(self, qtbot, ds):
        """The atlas region shows in its own colour, not the T1's grey."""
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        v.trigger("view.axial")
        before = v.canvases("slice")[0].grab_image()
        _add(qtbot, v, _atlas(ds))
        canvas = v.canvases("slice")[0]
        canvas.repaint()
        after = canvas.grab_image()
        assert after is not None and before is not None
        changed = sum(
            before.pixelColor(x, y) != after.pixelColor(x, y)
            for x in range(0, after.width(), 7) for y in range(0, after.height(), 7))
        assert changed > 10

    def test_one_asked_for_while_the_image_opens_follows_it(self, qtbot, ds):
        """What "Draw over" does: re-open the image, then add the overlay."""
        v = _viewer(qtbot)
        v.set_file(_t1(ds), ds)
        with qtbot.waitSignal(v.overlay_added, timeout=20_000):
            assert v.add_overlay(_atlas(ds))
        assert len(v.scene.layers) == 2

    def test_nothing_open_nothing_added(self, qtbot, ds):
        v = _viewer(qtbot)
        with qtbot.waitSignal(v.status_message, timeout=2000) as said:
            assert v.add_overlay(_atlas(ds)) is False
        assert "Open an image first" in said.args[0]

    def test_opening_another_image_drops_them(self, qtbot, ds):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        _add(qtbot, v, _atlas(ds))
        _open(qtbot, v, _bold(ds), ds)
        assert [lay.id for lay in v.scene.layers] == ["base"]

    def test_a_signal_viewer_takes_none(self, qtbot, ds):
        assert Viewer(kind="signal").add_overlay(_atlas(ds)) is False


class TestTheLayersSection:
    """The layer list and the selected layer's look, in the controls column."""

    def _panel(self, qtbot, ds):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        if not v.presenter.inspector_open():
            v.trigger("view.inspector")
        insp = v.presenter.inspector
        assert insp is not None
        return v, insp

    @staticmethod
    def _control(insp, key):
        for name in ("display", "overlay"):
            found = insp.section(name).control(key)
            if found is not None:
                return found
        raise KeyError(key)

    def test_the_top_layer_is_listed_first(self, qtbot, ds):
        v, insp = self._panel(qtbot, ds)
        _add(qtbot, v, _atlas(ds))
        layers = insp.section("layers")
        assert layers.list.item(0).text() == "sub-01_dseg.nii.gz"
        assert layers.list.item(1).text() == "sub-01_T1w.nii.gz"

    def test_the_base_cannot_be_removed_and_an_overlay_can(self, qtbot, ds):
        v, insp = self._panel(qtbot, ds)
        lid = _add(qtbot, v, _atlas(ds))
        layers = insp.section("layers")
        layers.list.setCurrentRow(1)              # the base
        assert not layers.remove_button.isEnabled()
        layers.list.setCurrentRow(0)
        assert layers.remove_button.isEnabled()
        layers.remove_button.click()
        v.qstore.flush()
        assert lid not in [lay.id for lay in v.scene.layers]

    def test_up_and_down(self, qtbot, ds):
        v, insp = self._panel(qtbot, ds)
        a = _add(qtbot, v, _atlas(ds))
        b = _add(qtbot, v, _bold(ds))
        layers = insp.section("layers")
        layers.list.setCurrentRow(0)              # b, on top
        assert not layers.up_button.isEnabled() and layers.down_button.isEnabled()
        layers.down_button.click()
        v.qstore.flush()
        assert [lay.id for lay in v.scene.layers] == ["base", b, a]

    def test_a_series_is_labelled_with_its_volumes(self, qtbot, ds):
        v, insp = self._panel(qtbot, ds)
        _add(qtbot, v, _bold(ds))
        assert insp.section("layers").list.item(0).text().endswith("12 vol")

    def test_an_atlas_is_not_offered_a_colour_map_or_a_window(self, qtbot, ds):
        v, insp = self._panel(qtbot, ds)
        _add(qtbot, v, _atlas(ds))
        insp.section("layers").list.setCurrentRow(0)
        assert not self._control(insp, "colormap").isVisibleTo(insp)
        assert not self._control(insp, "window").isVisibleTo(insp)
        assert self._control(insp, "outline_px").isVisibleTo(insp)
        assert self._control(insp, "opacity").isVisibleTo(insp)

    def test_the_base_has_no_overlay_section(self, qtbot, ds):
        v, insp = self._panel(qtbot, ds)
        _add(qtbot, v, _atlas(ds))
        insp.section("layers").list.setCurrentRow(1)   # the base
        assert not insp.section("overlay").isVisibleTo(insp)
        insp.section("layers").list.setCurrentRow(0)
        assert insp.section("overlay").isVisibleTo(insp)

    def test_a_number_and_its_slider_change_the_look_once(self, qtbot, ds):
        v, insp = self._panel(qtbot, ds)
        opacity = self._control(insp, "opacity")
        opacity.type_value(0.4)
        v.qstore.flush()
        assert v.scene.base_layer().display.opacity == pytest.approx(0.4)
        assert opacity.slider.value() == 400, "the slider follows the number"
        v.store.undo()
        assert v.scene.base_layer().display.opacity == pytest.approx(1.0)

    def test_the_window_bar_shows_the_histogram_and_sets_the_window(self, qtbot, ds):
        v, insp = self._panel(qtbot, ds)
        rc = self._control(insp, "window").range
        assert rc.hist is not None and rc.hist[0].sum() > 0
        rc.hi_spin.setValue(300.0)
        v.qstore.flush()
        assert v.scene.base_layer().display.window[1] == pytest.approx(300.0)

    def test_add_opens_the_dataset_picker(self, qtbot, ds, monkeypatch):
        from bidsmgr.gui.viz.panels import overlay_picker

        v, insp = self._panel(qtbot, ds)
        asked = {}

        def fake(parent, root, base, **kwargs):
            asked.update(root=root, base=base)
            return _atlas(ds)

        monkeypatch.setattr(overlay_picker, "ask_for_overlay", fake)
        with qtbot.waitSignal(v.overlay_added, timeout=20_000):
            insp.section("layers").add_button.click()
        assert asked == {"root": ds, "base": _t1(ds)}


class TestTheOverlayPicker:
    @pytest.fixture
    def two_subjects(self, ds):
        other = np.zeros((20, 20, 10), dtype=np.float32)
        _save(ds / "sub-02" / "anat" / "sub-02_T1w.nii.gz", other)
        _save(ds / "derivatives" / "seg" / "sub-01" / "anat" / "sub-01_desc-brain_mask.nii.gz",
              (other > -1).astype(np.uint8))
        return ds

    def _dialog(self, qtbot, root):
        from bidsmgr.gui.viz.panels.overlay_picker import OverlayPickerDialog

        base = _t1(root)
        img = nib.load(str(base))
        dlg = OverlayPickerDialog(root, base, base_affine=img.affine, base_shape=img.shape)
        qtbot.addWidget(dlg)
        return dlg

    @staticmethod
    def _shown(dlg) -> list[str]:
        from PyQt6.QtCore import Qt

        return sorted(Path(leaf.data(0, Qt.ItemDataRole.UserRole)).name
                      for leaf in dlg._leaves() if not leaf.isHidden())

    def test_this_subject_first_without_the_open_image(self, qtbot, two_subjects):
        dlg = self._dialog(qtbot, two_subjects)
        assert dlg.only_subject.isVisibleTo(dlg) and dlg.only_subject.isChecked()
        assert self._shown(dlg) == ["sub-01_desc-brain_mask.nii.gz", "sub-01_dseg.nii.gz",
                                    "sub-01_task-x_bold.nii.gz"]
        dlg.only_subject.setChecked(False)
        assert "sub-02_T1w.nii.gz" in self._shown(dlg)
        assert "sub-01_T1w.nii.gz" not in self._shown(dlg)

    def test_it_says_what_a_choice_would_be(self, qtbot, two_subjects):
        dlg = self._dialog(qtbot, two_subjects)
        dlg._select(_atlas(two_subjects))
        assert "Segmentation" in dlg._status.text()
        assert "Same voxel grid" in dlg._status.text()
        dlg._select(_bold(two_subjects))
        assert "Series of 12 volumes" in dlg._status.text()
        assert dlg._ok.text() == "Add"

    def test_browse_reaches_outside_the_dataset(self, qtbot, two_subjects, tmp_path,
                                                monkeypatch):
        from PyQt6.QtWidgets import QFileDialog

        outside = _save(tmp_path / "elsewhere" / "map.mgz.nii.gz",
                        np.zeros((4, 4, 4), np.float32))
        seen = {}

        def fake(parent, title, start, filters):
            seen["filters"] = filters
            return str(outside), ""

        monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(fake))
        dlg = self._dialog(qtbot, two_subjects)
        dlg._on_browse()
        assert dlg.chosen() == outside
        assert "*.mgz" in seen["filters"]


class TestQualityMaps:
    def test_the_quality_menu_offers_them(self, qtbot, ds):
        v = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
        tools = [a.text() for a in v.presenter.tools_button.menu().actions() if a.text()]
        assert "Add an overlay..." in tools
        menu = v.presenter.quality_button.menu()
        menu.aboutToShow.emit()
        quality = [a.text() for a in menu.actions() if a.text() and a.isVisible()]
        # A BOLD is not an anatomical or diffusion image: no Check quality.
        assert quality[:4] == ["Show the noise", "Temporal SNR map", "Standard deviation map",
                               "Mean image of the series"]
        assert "QC plots under the time course" in quality
        assert "Run QC when a file opens" in quality
        assert v.presenter.quality_button.isVisibleTo(v)

    def test_an_anatomical_image_is_offered_its_check(self, qtbot, ds):
        """A 3-D T1w has no series maps, but the quality check is for it."""
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        assert v.presenter.quality_button.isVisibleTo(v)
        menu = v.presenter.quality_button.menu()
        menu.aboutToShow.emit()
        quality = [a.text() for a in menu.actions() if a.text() and a.isVisible()]
        # Running the check when a file opens applies to it too (while the
        # quality panel is open); the series plots do not.
        assert quality == ["Check image quality", "Show the noise", "Run QC when a file opens"]

    def test_quality_is_not_offered_for_a_label_image(self, qtbot, ds):
        v = _open(qtbot, _viewer(qtbot), _atlas(ds), ds)
        assert not v.presenter.quality_button.isVisibleTo(v)

    def test_the_quality_rows_open_the_time_course(self, qtbot, ds):
        v = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
        v.run("view.graph", value=False)
        v.qstore.flush()
        v.presenter._qc_plots_action.trigger()
        v.qstore.flush()
        assert v.scene.graph_visible and v.scene.graph.qc

    def test_tsnr_is_added_over_the_series(self, qtbot, ds):
        v = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
        assert v.action("qc.tsnr").isEnabled()
        with qtbot.waitSignal(v.overlay_added, timeout=20_000):
            v.trigger("qc.tsnr")
        v.qstore.flush()
        top = v.scene.layers[-1]
        assert top.name == "Temporal SNR of sub-01_task-x_bold.nii.gz"
        v.run("cursor.set_world", x=10.0, y=10.0, z=5.0)
        v.qstore.flush()
        assert "Temporal SNR of sub-01_task-x_bold.nii.gz = " in v.readout_text()

    def test_a_single_volume_has_none(self, qtbot, ds):
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        assert not v.action("qc.tsnr").isEnabled()


class TestTheMrsVoxel:
    def test_the_spectrum_viewer_shows_it_on_the_anatomy(self, qtbot, ds):
        mrs = write_mrs(ds / "sub-01" / "mrs", "sub-01_svs.nii.gz", fid_at(2.01, 256))
        m = _viewer(qtbot, "spectrum")
        _open(qtbot, m, mrs, ds)
        assert m.action("spectrum.anatomy").isEnabled()
        m.trigger("spectrum.anatomy")
        anat = m.presenter.anatomy_viewer
        qtbot.addWidget(m.presenter.anatomy_dialog)
        qtbot.waitUntil(lambda: len(anat.scene.layers) == 2, timeout=20_000)
        assert anat.current_file() == _t1(ds)
        assert anat.scene.layers[1].name == "Voxel of sub-01_svs.nii.gz"

    def test_no_anatomy_is_said(self, qtbot, tmp_path):
        mrs = write_mrs(tmp_path / "sub-02" / "mrs", "sub-02_svs.nii.gz", fid_at(2.01, 256))
        m = _viewer(qtbot, "spectrum")
        _open(qtbot, m, mrs, tmp_path)
        with qtbot.waitSignal(m.status_message, timeout=2000) as said:
            m.trigger("spectrum.anatomy")
        assert "No anatomical image" in said.args[0]


class TestDrawOverFromTheTree:
    def _editor(self, qtbot, ds):
        from bidsmgr.gui.editor_panel import EditorPanel

        ep = EditorPanel()
        qtbot.addWidget(ep)
        ep._set_root(ds, persist=False)
        return ep

    def test_the_tree_offers_the_last_image_and_draws_over_it(self, qtbot, ds, monkeypatch):
        ep = self._editor(qtbot, ds)
        with qtbot.waitSignal(ep._nifti_viewer.loaded, timeout=20_000):
            ep.select_file_in_tree(_t1(ds))
        # The right-click selects (and so shows) the clicked file first.
        ep.select_file_in_tree(_atlas(ds))
        pane = ep._tree_pane
        item = pane.reveal(_atlas(ds))
        monkeypatch.setattr(pane._tree, "itemAt", lambda _pos: item)
        captured = []
        monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: captured.extend(
            act for act in self.actions() if act.text()))
        pane._on_show_context_menu(QPoint(1, 1))
        draw = [a for a in captured if a.text() == "Draw over sub-01_T1w.nii.gz"]
        assert draw, [a.text() for a in captured]
        with qtbot.waitSignal(ep._nifti_viewer.overlay_added, timeout=20_000):
            draw[0].trigger()
        viewer = ep._nifti_viewer
        assert viewer.current_file() == _t1(ds)
        assert [lay.name for lay in viewer.scene.layers] == [
            "sub-01_T1w.nii.gz", "sub-01_dseg.nii.gz"]

    def test_nothing_shown_before_means_no_entry(self, qtbot, ds):
        ep = self._editor(qtbot, ds)
        assert ep._overlay_base(_atlas(ds)) is None


class TestDefacingPreview:
    def test_it_is_drawn_over_the_untouched_image(self, qtbot, ds, monkeypatch):
        from bidsmgr.deface import preview
        from bidsmgr.deface.run import DefaceResult
        from bidsmgr.gui.viz.presenters.volume import VolumePresenter

        monkeypatch.setattr(VolumePresenter, "_deface_ok", True)

        def fake(source, **_kw):
            img = nib.load(str(source))
            data = np.asarray(img.dataobj, dtype=np.float32).copy()
            data[:, :8, :] = 0
            out = Path(source).with_name("fake_defaced.nii.gz")
            nib.save(nib.Nifti1Image(data, img.affine), str(out))
            return DefaceResult(source=Path(source), output=out, engine_id="fake", seconds=0.0)

        monkeypatch.setattr(preview, "deface_to_temp", fake)
        before = _t1(ds).read_bytes()
        v = _open(qtbot, _viewer(qtbot), _t1(ds), ds)
        assert v.action("deface.preview").isEnabled()
        with qtbot.waitSignal(v.overlay_added, timeout=20_000):
            v.trigger("deface.preview")
        assert v.scene.layers[-1].name == "What defacing would remove"
        assert v.scene.layers[-1].origin == "deface:"
        assert _t1(ds).read_bytes() == before, "nothing was changed"

    def test_a_series_is_not_offered_it(self, qtbot, ds, monkeypatch):
        from bidsmgr.gui.viz.presenters.volume import VolumePresenter

        monkeypatch.setattr(VolumePresenter, "_deface_ok", True)
        v = _open(qtbot, _viewer(qtbot), _bold(ds), ds)
        assert not v.action("deface.preview").isEnabled()

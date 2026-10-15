"""The BOLD graph with the run around it: events and physio on its time axis."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("pyqtgraph")

from bidsmgr.gui.viz import Viewer  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """A 20-volume run at TR 2 s (40 s), two events and a cardiac trace
    that starts 3 s before the first volume."""
    root = tmp_path / "Study"
    func = root / "sub-01" / "func"
    func.mkdir(parents=True)
    rng = np.random.default_rng(0)
    data = (1000 + rng.normal(0, 5, (6, 6, 4, 20))).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header.set_zooms((1.0, 1.0, 1.0, 2.0))
    nib.save(img, str(func / "sub-01_task-x_run-1_bold.nii.gz"))
    (func / "sub-01_task-x_run-1_bold.json").write_text(json.dumps({"RepetitionTime": 2.0}))
    (func / "sub-01_task-x_run-1_events.tsv").write_text(
        "onset\tduration\ttrial_type\n4\t6\tfaces\n20\t0\tbutton\n")
    with gzip.open(func / "sub-01_task-x_run-1_recording-cardiac_physio.tsv.gz", "wt") as fh:
        for i in range(4600):           # 46 s at 100 Hz
            fh.write(f"{math.sin(i / 7.0):g}\n")
    (func / "sub-01_task-x_run-1_recording-cardiac_physio.json").write_text(json.dumps(
        {"Columns": ["cardiac"], "SamplingFrequency": 100.0, "StartTime": -3.0}))
    return root


def _bold(root: Path) -> Path:
    return root / "sub-01" / "func" / "sub-01_task-x_run-1_bold.nii.gz"


def _graph(qtbot, root: Path, path: Path):
    v = Viewer(kind="volume")
    qtbot.addWidget(v)
    v.resize(1000, 700)
    v.show()
    qtbot.waitExposed(v)
    with qtbot.waitSignal(v.loaded, timeout=20_000):
        v.set_file(path, root)
    v.run("view.graph", value=True)
    v.run("graph.set", scope=1, x_axis="seconds")
    v.qstore.flush()
    qtbot.waitUntil(lambda: v.presenter.graph is not None, timeout=5000)
    return v, v.presenter.graph


class TestEvents:
    def test_they_are_drawn_on_the_time_axis(self, qtbot, run):
        _v, g = _graph(qtbot, run, _bold(run))
        bands = g.event_bands()
        assert [(round(a, 3), lab) for a, _b, lab in bands] == [(4.0, "faces"), (20.0, "button")]
        assert bands[0][1] == pytest.approx(10.0), "a span for an event that lasts"
        assert bands[1][1] > 20.0, "an instant still a visible band"

    def test_against_volumes_they_are_placed_by_the_repetition_time(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", x_axis="frames")
        v.qstore.flush()
        assert g.event_bands()[0][0] == pytest.approx(2.0), "4 s at TR 2 s is volume 2"

    def test_they_can_be_turned_off(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        assert g.events_box.isVisibleTo(g)
        v.run("graph.set", events=False)
        v.qstore.flush()
        assert g.event_bands() == []
        assert not g.events_box.isChecked()

    def test_a_neighbourhood_carries_them_in_every_cell(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", scope=2)
        v.qstore.flush()
        assert g.cell_count() == 9
        assert len(g.event_bands()) == 2

    def test_a_run_without_events_offers_none(self, qtbot, run):
        (run / "sub-01" / "func" / "sub-01_task-x_run-1_events.tsv").unlink()
        _v, g = _graph(qtbot, run, _bold(run))
        assert not g.events_box.isVisibleTo(g)
        assert g.event_bands() == []


class TestPhysio:
    def test_it_is_drawn_under_the_graph_in_run_time(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", physio=True)
        v.qstore.flush()
        qtbot.waitUntil(lambda: g.physio_channels() == ["cardiac"], timeout=20_000)
        card = g.tracks.card("physio:cardiac")
        x, _y = card._curves[0].getData()
        assert x.min() >= 0.0 - 1e-6, "cut to the run (it started 3 s before it)"
        assert x.max() <= 38.0 + 0.02
        # Its own values on its own axis, its name in the header, not on the trace.
        assert card.title.text() == "cardiac"
        assert not card.up.isVisibleTo(card) and not card.hide_button.isVisibleTo(card)

    def test_it_is_shown_for_a_neighbourhood_too(self, qtbot, run):
        """It used to vanish at any scope but one voxel, with its box still
        ticked: the strip keeps the run's own time axis."""
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", physio=True, scope=2)
        v.qstore.flush()
        qtbot.waitUntil(lambda: g.physio_channels() == ["cardiac"], timeout=20_000)

    def test_a_trigger_log_is_drawn_as_ticks_with_its_count(self, qtbot, run):
        """A scanner log writes the trigger's value where it fired and
        nothing in between: a lane of ticks, not a line of dots."""
        import gzip
        import json

        func = run / "sub-01" / "func"
        rows = ["n/a"] * 400
        for k in range(0, 400, 40):
            rows[k] = "5"
        with gzip.open(func / "sub-01_task-x_run-1_recording-trigger_physio.tsv.gz", "wt") as fh:
            fh.write("\n".join(rows) + "\n")
        (func / "sub-01_task-x_run-1_recording-trigger_physio.json").write_text(json.dumps(
            {"Columns": ["trigger"], "SamplingFrequency": 10.0, "StartTime": 0.0}))
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", physio=True)
        v.qstore.flush()
        qtbot.waitUntil(lambda: len(g.shown_tracks()) == 2, timeout=20_000)
        tracks = {t["title"]: t for t in g.shown_tracks()}
        assert tracks["cardiac"]["kind"] == "line"
        assert tracks["trigger"]["kind"] == "events"
        assert tracks["trigger"]["ticks"] == 10
        assert "10 marks" in tracks["trigger"]["summary"]

    def test_moving_the_crosshair_does_not_read_it_again(self, qtbot, run, monkeypatch):
        from bidsmgr.gui.viz.canvases import graph as G

        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", physio=True)
        v.qstore.flush()
        qtbot.waitUntil(lambda: g.physio_channels() == ["cardiac"], timeout=20_000)
        reads = []
        real = G.physio_for_graph
        monkeypatch.setattr(G, "physio_for_graph", lambda *a, **k: (reads.append(1), real(*a, **k))[1])
        for _ in range(5):
            v.run("cursor.step", plane="axial", n=1)
            v.qstore.flush()
        assert reads == []


class TestDiffusion:
    @pytest.fixture
    def dwi(self, tmp_path: Path) -> Path:
        root = tmp_path / "Study"
        d = root / "sub-01" / "dwi"
        d.mkdir(parents=True)
        data = np.random.default_rng(1).random((4, 4, 3, 7)).astype(np.float32) * 100
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(d / "sub-01_dwi.nii.gz"))
        (d / "sub-01_dwi.bval").write_text("5 995 1000 1005 0 2000 1995\n")
        return root

    def _open(self, qtbot, root):
        path = root / "sub-01" / "dwi" / "sub-01_dwi.nii.gz"
        v = Viewer(kind="volume")
        qtbot.addWidget(v)
        v.resize(1000, 700)
        v.show()
        qtbot.waitExposed(v)
        with qtbot.waitSignal(v.loaded, timeout=20_000):
            v.set_file(path, root)
        v.qstore.flush()
        return v

    def test_the_volume_label_says_its_b_value(self, qtbot, dwi):
        v = self._open(qtbot, dwi)
        v.run("frame.set", frame=2)
        v.qstore.flush()
        assert v.presenter.frame_control.value() == 2
        assert v.presenter.frame_control.spin.suffix() == " / 6"
        assert v.presenter._frame_hdr.text() == "Volume  b=1000"

    def test_the_shell_keys_step_within_a_shell(self, qtbot, dwi):
        v = self._open(qtbot, dwi)
        assert v.action("frame.same_shell").isEnabled()
        assert v.action("frame.same_shell").shortcut().toString() == "]"
        v.run("frame.set", frame=1)
        v.trigger("frame.same_shell")
        v.qstore.flush()
        assert v.scene.layers[0].frame == 2
        v.trigger("frame.shell")
        v.qstore.flush()
        assert v.scene.layers[0].frame == 5

    def test_the_graph_marks_the_b0_volumes(self, qtbot, dwi):
        v = self._open(qtbot, dwi)
        v.run("view.graph", value=True)
        v.run("graph.set", scope=1, x_axis="frames")
        v.qstore.flush()
        qtbot.waitUntil(lambda: v.presenter.graph is not None, timeout=5000)
        bands = v.presenter.graph.event_bands()
        assert [(round(a + 0.5), lab) for a, _b, lab in bands] == [(0, "b=0"), (4, "b=0")]

    def test_a_series_without_b_values_has_no_shell_actions(self, qtbot, run):
        v, _g = _graph(qtbot, run, _bold(run))
        assert not v.action("frame.same_shell").isEnabled()


class TestQcTracesAndZoom:
    def test_qc_traces_flag_the_volume_that_jumps(self, qtbot, run):
        import numpy as _np

        bold = _bold(run)
        img = nib.load(str(bold))
        data = _np.asarray(img.dataobj).astype("float32")
        data[..., 11] += 400.0
        nib.save(nib.Nifti1Image(data, img.affine), str(bold))
        v, g = _graph(qtbot, run, bold)
        qtbot.waitUntil(lambda: g.qc_box.isVisibleTo(g), timeout=20_000)
        v.run("graph.set", qc=True)
        v.qstore.flush()
        qtbot.waitUntil(lambda: "dvars" in [t["id"] for t in g.shown_tracks()],
                        timeout=20_000)
        dvars = [t for t in g.shown_tracks() if t["id"] == "dvars"][0]
        assert dvars["ticks"] >= 1 and "box-plot" in dvars["note"]
        qtbot.waitUntil(lambda: [t["id"] for t in g.shown_tracks()] == [
            "fd", "translation", "rotation", "dvars", "outliers"], timeout=20_000)
        titles = [t["title"] for t in g.shown_tracks()]
        assert titles[:3] == ["Framewise displacement", "Translation", "Rotation"], \
            "framewise displacement is its own plot, apart from the parameters"
        assert g.tracks.card("rotation").track["legend"] == ["pitch", "roll", "yaw"]
        assert g.tracks.card("translation").track["legend"] == ["x", "y", "z"]

    def test_plots_being_computed_show_a_spinner(self, qtbot, run, monkeypatch):
        """While the worker computes, each plot asked for is a card with a
        spinner and its progress, where the plot will be."""
        import threading

        from bidsmgr.gui.viz.canvases import graph as G

        gate = threading.Event()
        real = G.qc_for_graph

        def slow(*args, **kwargs):
            gate.wait(10)
            return real(*args, **kwargs)

        monkeypatch.setattr(G, "qc_for_graph", slow)
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", qc=True, qc_rows=["global", "dvars"])
        v.qstore.flush()
        qtbot.waitUntil(lambda: g.pending_tracks() == ["global", "dvars"], timeout=5000)
        card = g.tracks.card("global")
        assert card.spinner.is_busy() and "Computing" in card.readout.text()
        assert g.shown_tracks() == [], "a card being computed is not a plot yet"
        gate.set()
        qtbot.waitUntil(lambda: [t["id"] for t in g.shown_tracks()] == ["global", "dvars"],
                        timeout=20_000)
        assert g.pending_tracks() == [] and not g.tracks.card("global").spinner.is_busy()

    def test_every_plot_explains_itself(self, qtbot, run):
        from bidsmgr.gui.viz.panels import explain_popup

        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", qc=True, qc_rows=["fd", "dvars"])
        v.qstore.flush()
        qtbot.waitUntil(lambda: [t["id"] for t in g.shown_tracks()] == ["fd", "dvars"],
                        timeout=20_000)
        card = g.tracks.card("fd")
        assert card.track["explain"] == "plot.bold.fd" and card.info.isVisibleTo(card)
        assert "info icon" in card.toolTip()
        card.info.click()
        menu, label = explain_popup.last_shown()
        text = label.text()
        assert "How to read it" in text and "In this image" in text
        menu.close()

    def test_a_changed_threshold_redraws_the_plots(self, qtbot, run):
        from bidsmgr.gui.viz.bridge import SettingsHub

        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", qc=True, qc_rows=["fd"])
        v.qstore.flush()
        qtbot.waitUntil(lambda: [t["id"] for t in g.shown_tracks()] == ["fd"], timeout=20_000)
        assert g.tracks.card("fd").track["rule"] == 0.5
        hub = SettingsHub.instance()
        try:
            hub.update(lambda s: setattr(s.qc_bold, "fd_threshold_mm", 0.2))
            qtbot.waitUntil(lambda: g.tracks.card("fd") is not None
                            and g.tracks.card("fd").track.get("rule") == 0.2, timeout=20_000)
            assert "above 0.2 mm" in g.tracks.card("fd").track["summary"]
        finally:
            hub.update(lambda s: setattr(s.qc_bold, "fd_threshold_mm", 0.5))

    def test_rows_are_chosen_and_computed_only_when_shown(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", qc=True, qc_rows=["global"])
        v.qstore.flush()
        qtbot.waitUntil(lambda: [t["id"] for t in g.shown_tracks()] == ["global"],
                        timeout=20_000)
        assert set(g._qc_src["rows"]) == {"global"}, "unasked rows were computed"
        g._qc_plot_actions["carpet"].trigger()
        v.qstore.flush()
        assert v.scene.graph.qc_rows == ["global", "carpet"]
        qtbot.waitUntil(lambda: any(t["kind"] == "image" for t in g.shown_tracks()),
                        timeout=20_000)
        carpet = g.tracks.card("carpet")
        assert carpet._image is not None and carpet._image.isVisible()
        g._qc_plot_actions["carpet"].trigger()
        v.qstore.flush()
        assert g.tracks.card("carpet") is None and carpet.isHidden(), \
            "the card is kept for later, hidden, never destroyed"
        with pytest.raises(ValueError):
            v.run("graph.set", qc_rows=["nonsense"])

    def test_motion_is_read_from_the_runs_confounds(self, qtbot, run):
        func = _bold(run).parent
        fd = ["n/a"] + ["0.1"] * 18 + ["0.9"]
        lines = ["trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\tframewise_displacement"]
        lines += [f"0\t0\t{i * 0.01:g}\t0\t0\t0\t{fd[i]}" for i in range(20)]
        (func / "sub-01_task-x_run-1_desc-confounds_timeseries.tsv").write_text(
            "\n".join(lines) + "\n")
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", qc=True, qc_rows=["fd"])
        v.qstore.flush()
        qtbot.waitUntil(lambda: [t["id"] for t in g.shown_tracks()] == ["fd"], timeout=20_000)
        fd_track = g.shown_tracks()[0]
        assert "desc-confounds_timeseries.tsv" in fd_track["note"]
        assert fd_track["ticks"] == 1, "one volume above 0.5 mm"
        assert g.tracks.card("fd").track["rule"] == 0.5


class TestThePanel:
    """The time course can sit beside the views, take the whole viewer, or
    go to a window of its own."""

    def test_beside_and_below(self, qtbot, run):
        from PyQt6.QtCore import Qt

        v, _g = _graph(qtbot, run, _bold(run))
        v.trigger("graph.beside")
        v.qstore.flush()
        assert v.scene.layout.graph == "right"
        assert v.presenter.vsplit.orientation() == Qt.Orientation.Horizontal
        assert v.action("graph.beside").isChecked()
        v.trigger("graph.beside")
        v.qstore.flush()
        assert v.presenter.vsplit.orientation() == Qt.Orientation.Vertical

    def test_maximised_it_has_the_viewer(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        v.trigger("graph.maximize")
        assert v.presenter.pages.isHidden() and g.isVisible()
        assert v.action("graph.maximize").isChecked()
        v.trigger("graph.maximize")
        assert not v.presenter.pages.isHidden()

    def test_its_own_window_and_back(self, qtbot, run):
        v, g = _graph(qtbot, run, _bold(run))
        v.trigger("graph.detach")
        win = v.presenter._graph_window
        assert win is not None and win.isVisible() and win.isAncestorOf(g)
        assert not v.presenter.graph_host.isVisible()
        assert not v.action("graph.maximize").isEnabled(), "nothing to maximise into"
        win.close()
        assert v.presenter._graph_window is None
        assert v.presenter.graph_host.isAncestorOf(g) and g.isVisible()
        # Turning the graph off while it is out brings it home too.
        v.trigger("graph.detach")
        v.run("view.graph", value=False)
        v.qstore.flush()
        assert v.presenter._graph_window is None

    def test_the_panel_buttons_are_in_the_graph_corner(self, qtbot, run):
        from PyQt6.QtWidgets import QPushButton

        _v, g = _graph(qtbot, run, _bold(run))
        corner = g.header.corner_buttons()
        ids = [w.property("viz_action") for w in corner]
        assert {"graph.beside", "graph.maximize", "graph.detach"} <= set(ids)
        # Icons in the corner, More last; the controls keep the left side.
        assert all(w.text() == "" for w in corner)
        assert all(g.header.corner.rect().contains(w.geometry()) for w in corner), \
            "an icon button larger than its row is clipped"
        assert corner[-1] is g.more_button
        assert not g.controls.findChildren(QPushButton, "viz-corner-btn")

    def test_the_qc_plots_menu_is_named_as_the_signal_one(self, qtbot, run):
        _v, g = _graph(qtbot, run, _bold(run))
        assert g.qc_plots_button.text() == "Plots"
        assert g.qc_plots_button.menu().actions()[-1].text() == "Run QC when a file opens"

    def test_more_holds_the_marker_and_the_export(self, qtbot, run):
        _v, g = _graph(qtbot, run, _bold(run))
        texts = [a.text() for a in g.more_button.menu().actions()]
        assert "Marker size" in texts
        assert "Marker on every voxel" in texts
        assert "Export CSV..." in texts

    def test_ctrl_wheel_zooms_time_and_reads_the_physio_for_it(self, qtbot, run):
        from PyQt6.QtCore import QPoint, QPointF, Qt
        from PyQt6.QtGui import QWheelEvent

        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", physio=True)
        v.qstore.flush()
        qtbot.waitUntil(lambda: g.physio_channels() == ["cardiac"], timeout=20_000)
        g.set_time_view((10.0, 20.0))
        assert g.time_view() == (10.0, 20.0)
        assert g.plot.getPlotItem().getViewBox().viewRange()[0] == pytest.approx([10.0, 20.0])
        qtbot.waitUntil(lambda: g._physio_src is not None
                        and float(g._physio_src["tracks"][0]["x"].min()) >= 9.9, timeout=20_000)
        qtbot.waitUntil(lambda: g.tracks.card("physio:cardiac") is not None, timeout=20_000)
        card = g.tracks.card("physio:cardiac")
        assert card.plot.getPlotItem().getViewBox().viewRange()[0] == pytest.approx(
            [10.0, 20.0]), "the plots under the graph follow its zoom"
        ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.ShiftModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        g._on_wheel(ev)
        lo, hi = g.time_view()
        assert hi - lo == pytest.approx(10.0) and lo < 10.0, "Shift+wheel pans"
        g.set_time_view(None)
        assert g.time_view() is None



class TestTracks:
    """Each plot under the graph is its own: ordered, hidden, read and
    clicked independently, fitted or scrolled, in the theme's colours."""

    def _qc(self, qtbot, run, rows):
        v, g = _graph(qtbot, run, _bold(run))
        v.run("graph.set", qc=True, qc_rows=rows)
        v.qstore.flush()
        qtbot.waitUntil(lambda: [t["id"] for t in g.shown_tracks()] == rows, timeout=20_000)
        return v, g

    def test_the_user_orders_and_hides_them(self, qtbot, run):
        v, g = self._qc(qtbot, run, ["fd", "dvars", "global"])
        g.tracks.card("global").up.click()
        v.qstore.flush()
        assert v.scene.graph.qc_rows == ["fd", "global", "dvars"]
        qtbot.waitUntil(lambda: g.tracks.track_ids() == ["fd", "global", "dvars"],
                        timeout=5000)
        assert not g.tracks.card("fd").up.isEnabled(), "the first cannot go higher"
        g.tracks.card("dvars").hide_button.click()
        v.qstore.flush()
        assert v.scene.graph.qc_rows == ["fd", "global"]
        assert not g._qc_plot_actions["dvars"].isChecked()

    def test_the_title_is_never_on_the_signal(self, qtbot, run):
        _v, g = self._qc(qtbot, run, ["dvars"])
        card = g.tracks.card("dvars")
        assert card.title.geometry().bottom() < card.plot.geometry().top(), \
            "the header sits above the plot"
        assert card.title.text() == "DVARS (% of mean)"

    def test_hover_reads_every_track_at_one_moment_and_click_goes_there(self, qtbot, run):
        v, g = self._qc(qtbot, run, ["dvars", "global"])
        v.run("graph.set", x_axis="frames")
        v.qstore.flush()
        g.tracks._hover_at(7.2, g.tracks.card("dvars"), None)
        for card in g.tracks.cards():
            assert card.readout.text().startswith("volume 7"), card.readout.text()
        assert "%" in g.tracks.card("dvars").readout.text()
        g.tracks._hover_at(None, None)
        assert g.tracks.card("dvars").readout.text().startswith("median")
        g.tracks.clicked.emit(12.0)
        assert v.scene.base_layer().frame == 12

    def test_scroll_mode_gives_each_a_readable_height(self, qtbot, run):
        from bidsmgr.gui.viz.canvases.tracks import SCROLL_PX

        v, g = self._qc(qtbot, run, ["fd", "dvars", "outliers"])
        g.expand_button.click()
        v.qstore.flush()
        assert v.scene.graph.tracks_mode == "scroll"
        assert g.tracks.mode() == "scroll"
        assert g.tracks.card("fd").height() >= SCROLL_PX

    def test_they_follow_the_theme(self, qtbot, run):
        from PyQt6.QtGui import QColor

        from bidsmgr.gui.viz.bridge import ThemeHub
        from bidsmgr.gui.theme_manager import DARK, LIGHT

        _v, g = self._qc(qtbot, run, ["dvars"])
        card = g.tracks.card("dvars")
        ThemeHub.instance().publish(LIGHT)
        light = card.plot.backgroundBrush().color().name()
        ThemeHub.instance().publish(DARK)
        dark = card.plot.backgroundBrush().color().name()
        assert light != dark
        assert QColor(dark).lightness() < QColor(light).lightness()

    def test_qc_is_off_on_opening_unless_asked(self, qtbot, run):
        from bidsmgr.gui.viz.bridge import SettingsHub

        v, g = self._qc(qtbot, run, ["global"])
        assert not v.ctx.settings.qc.on_open
        _open_again = Viewer(kind="volume")
        qtbot.addWidget(_open_again)
        with qtbot.waitSignal(_open_again.loaded, timeout=20_000):
            _open_again.set_file(_bold(run), run)
        assert not _open_again.scene.graph.qc, "QC ran again without being asked"
        SettingsHub.instance().update(lambda st: setattr(st.qc, "on_open", True))
        third = Viewer(kind="volume")
        qtbot.addWidget(third)
        with qtbot.waitSignal(third.loaded, timeout=20_000):
            third.set_file(_bold(run), run)
        assert third.scene.graph.qc, "asked to run QC on opening"

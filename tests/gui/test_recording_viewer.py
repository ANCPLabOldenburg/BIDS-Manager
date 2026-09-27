"""GUI tests for the EEG/MEG recording viewer pane + Editor routing."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from bidsmgr.gui.editor_panel import EditorPanel  # noqa: E402
from bidsmgr.gui.theme_manager import DARK, LIGHT  # noqa: E402
from bidsmgr.gui.widgets import RecordingViewerPane  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def rec(tmp_path: Path) -> Path:
    sfreq = 200.0
    n = int(sfreq * 5)
    data = np.random.default_rng(3).standard_normal((6, n)) * 1e-5
    data[5, :] = 0
    data[5, [40, 120, 300]] = 1  # stim pulses
    info = mne.create_info(
        ["Fp1", "Fp2", "Cz", "Pz", "EOG", "STI"],
        sfreq,
        ["eeg", "eeg", "eeg", "eeg", "eog", "stim"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.RawArray(data, info, verbose=False)
        path = tmp_path / "sub-01_task-rest_eeg.fif"
        raw.save(path, overwrite=True, verbose=False)
    (tmp_path / "sub-01_task-rest_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0.5\t0\tgo\n2.5\t0\tstop\n"
    )
    return path


def test_pane_starts_with_hint(qapp) -> None:
    pane = RecordingViewerPane()
    assert pane._stack.currentWidget() is pane._hint


def test_recording_tree_icons(qapp) -> None:
    """File and directory-shaped recordings get the brain+signal icon;
    plain folders keep the folder icon."""
    from bidsmgr.gui import icons

    assert not icons.icon_for_path("sub-01_task-x_eeg.edf").isNull()
    assert not icons.icon_for_path("sub-01_task-x_eeg.vhdr").isNull()
    ds = icons.icon_for_path("sub-01_task-x_meg.ds", is_dir=True)
    mff = icons.icon_for_path("sub-01_eeg.mff", is_dir=True)
    folder = icons.icon_for_path("anat", is_dir=True)
    assert not ds.isNull() and not mff.isNull()
    # CTF .ds / EGI .mff dirs must NOT use the plain folder icon.
    assert ds.cacheKey() != folder.cacheKey()
    assert mff.cacheKey() != folder.cacheKey()


def test_metadata_loads_then_signal(qapp, qtbot, rec: Path) -> None:
    pane = RecordingViewerPane()
    qtbot.addWidget(pane)
    pane.set_file(rec, rec.parent)
    qtbot.waitUntil(
        lambda: pane._stack.currentWidget() is pane._meta_page, timeout=20000
    )
    assert pane._meta is not None
    assert pane._meta["n_channels"] == 6

    pane._load_signal()
    qtbot.waitUntil(
        lambda: pane._stack.currentWidget() is pane._view, timeout=20000
    )
    view = pane._view
    assert view._raw is not None
    items = [view.cmb_ch_type.itemText(i) for i in range(view.cmb_ch_type.count())]
    assert "all" in items and "eeg" in items
    # Events from the sibling events.tsv + stim channel -> control enabled.
    assert view.chk_events.isEnabled()

    # Exercise the controls (must not raise).
    view.chk_events.setChecked(True)
    view._on_ch_type_changed("eeg")
    view._on_n_changed(3)
    view._on_scale_changed(2.0)
    view._on_window_changed(3.0)
    view._navigate("next")
    view._navigate("end")
    view._navigate("start")
    view._on_norm_toggled(True)
    view._on_norm_toggled(False)
    view._on_dark_toggled(True)
    view._on_dark_toggled(False)
    view.spn_hp.setValue(1.0)
    view.spn_lp.setValue(40.0)
    view._apply_filters()
    view._reset_filters()
    view._reset_view()

    # Theme swap + unload.
    pane.repaint_for_palette(LIGHT)
    pane.repaint_for_palette(DARK)
    view.unload()
    assert view._raw is None


def test_set_file_none_unloads(qapp, qtbot, rec: Path) -> None:
    pane = RecordingViewerPane()
    qtbot.addWidget(pane)
    pane.set_file(rec, rec.parent)
    qtbot.waitUntil(
        lambda: pane._stack.currentWidget() is pane._meta_page, timeout=20000
    )
    pane.set_file(None, None)
    assert pane._stack.currentWidget() is pane._hint


def test_event_jump_to_first_when_outside_window(qapp) -> None:
    """Enabling events when none fall in the current window scrolls the
    view to the first event (the MEG sample's first trigger is ~102 s in)."""
    from bidsmgr.gui.widgets.time_series_view import TimeSeriesView

    v = TimeSeriesView()
    v._duration = 100.0
    v._time_window = 5.0
    v._time_start = 0.0
    v._tsv_events = [(50.0, "x")]
    v._event_source = "auto"
    v._jump_to_first_event_if_needed()
    assert 44.0 < v._time_start < 50.0
    v.unload()


def test_psd_dialog_handles_channel_subset(qapp) -> None:
    """The PSD dialog must not crash when the caller passes more channel
    names/types than PSD data rows (compute_psd drops non-data channels)."""
    from bidsmgr.gui.widgets.psd_dialog import PsdDialog

    res = {
        "freqs": np.linspace(1.0, 40.0, 40),
        "data": np.random.default_rng(0).random((3, 40)) * 1e-10,
        "ch_names": ["a", "b", "c", "d", "e"],  # longer than data on purpose
        "ch_types": ["eeg", "eeg", "eeg", "mag", "mag"],
    }
    dlg = PsdDialog(res)
    assert dlg._tabs.count() == 2
    dlg._on_db_toggled(False)
    dlg._on_db_toggled(True)
    dlg._on_type_changed()


def test_editor_routes_recording_to_pane(qapp, qtbot, rec: Path) -> None:
    ep = EditorPanel()
    qtbot.addWidget(ep)
    ep._set_root(rec.parent, persist=False)
    ep._on_file_selected(rec)
    assert ep._center_stack.currentWidget() is ep._recording_viewer
    qtbot.waitUntil(
        lambda: ep._recording_viewer._stack.currentWidget()
        is ep._recording_viewer._meta_page,
        timeout=20000,
    )

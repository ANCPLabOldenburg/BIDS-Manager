"""The 4-D graph plots PET against real seconds, not volume index.

A PET series is 4-D in the same shape as a BOLD run, but its fourth axis is
nothing like one. BOLD frames are evenly spaced; PET frames get progressively
longer as the tracer decays, so a 10 s frame and a 300 s frame sit side by side
in one series. Drawing that against volume index makes every frame the same
width, which flattens the early fast-changing part of the curve, precisely
where the kinetics live.

These tests pin the behaviour and, just as importantly, pin that non-PET data
keeps the index axis it always had.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gui

nibabel = pytest.importorskip("nibabel")
pytest.importorskip("pyqtgraph")

from bidsmgr.gui.widgets.nifti_viewer_pane import (  # noqa: E402
    NiftiViewerPane,
    read_pet_frame_times,
)


# Realistic PET framing: short frames early while the tracer arrives, long ones
# late once the signal is slow. This unevenness is the whole point.
DURATIONS = [10] * 6 + [30] * 4 + [60] * 5 + [300] * 5


def _starts(durations=DURATIONS) -> list[float]:
    out, t = [], 0.0
    for d in durations:
        out.append(t)
        t += d
    return out


def _write_pet(tmp_path: Path, *, sidecar: dict | None = None,
               name: str = "sub-001_pet") -> Path:
    d = tmp_path / "sub-001" / "pet"
    d.mkdir(parents=True, exist_ok=True)
    n = len(DURATIONS)
    data = np.zeros((6, 6, 3, n), dtype=np.float32)
    mid = np.asarray(_starts()) + np.asarray(DURATIONS) / 2
    # A plausible tracer curve: fast uptake, slow washout.
    data[:] = (40 * np.exp(-mid / 200) + 12 * (1 - np.exp(-mid / 60)))[None, None, None, :]
    img = d / f"{name}.nii.gz"
    nibabel.save(nibabel.Nifti1Image(data, np.eye(4)), img)
    payload = {"FrameTimesStart": _starts(), "FrameDuration": DURATIONS}
    if sidecar is not None:
        payload = sidecar
    (d / f"{name}.json").write_text(json.dumps(payload))
    return img


# ---------------------------------------------------------------------------
# Reading the frame times
# ---------------------------------------------------------------------------


def test_frame_times_are_read_from_the_sidecar(tmp_path) -> None:
    img = _write_pet(tmp_path)
    times = read_pet_frame_times(img)
    assert times is not None
    assert len(times) == len(DURATIONS)
    assert times[0] == 0.0
    # The spacing must stay uneven; that is the information being preserved.
    assert times[1] - times[0] == 10
    assert times[-1] - times[-2] == 300


def test_a_non_pet_image_keeps_the_index_axis(tmp_path) -> None:
    """REGRESSION: a BOLD run must be unaffected."""
    img = _write_pet(tmp_path, name="sub-001_task-rest_bold")
    assert read_pet_frame_times(img) is None


def test_a_missing_sidecar_is_not_an_error(tmp_path) -> None:
    img = _write_pet(tmp_path)
    img.with_name(img.name.replace(".nii.gz", ".json")).unlink()
    assert read_pet_frame_times(img) is None


def test_a_single_frame_is_not_a_curve(tmp_path) -> None:
    """One frame has nothing to plot against time."""
    img = _write_pet(tmp_path, sidecar={"FrameTimesStart": [0], "FrameDuration": [300]})
    assert read_pet_frame_times(img) is None


def test_unusable_frame_times_fall_back(tmp_path) -> None:
    img = _write_pet(tmp_path, sidecar={"FrameTimesStart": ["early", "late"]})
    assert read_pet_frame_times(img) is None


def test_malformed_json_falls_back(tmp_path) -> None:
    img = _write_pet(tmp_path)
    img.with_name(img.name.replace(".nii.gz", ".json")).write_text("{ not json")
    assert read_pet_frame_times(img) is None


# ---------------------------------------------------------------------------
# The graph itself
# ---------------------------------------------------------------------------


def _loaded_pane(qtbot, img: Path, root: Path) -> NiftiViewerPane:
    pane = NiftiViewerPane()
    qtbot.addWidget(pane)
    pane.resize(700, 520)
    with qtbot.waitSignal(pane.loaded, timeout=20_000):
        pane.set_file(img, root)
    return pane


def test_graph_uses_seconds_for_pet(qtbot, tmp_path) -> None:
    pane = _loaded_pane(qtbot, _write_pet(tmp_path), tmp_path)
    pane._toggle_graph()
    qtbot.wait(50)

    assert pane._graph_x_is_time
    assert len(pane._graph_x) == len(DURATIONS)
    assert pane._graph_x[-1] == sum(DURATIONS) - DURATIONS[-1]
    # Uneven spacing survives into the plotted x values.
    assert pane._graph_x[1] - pane._graph_x[0] == 10
    assert pane._graph_x[-1] - pane._graph_x[-2] == 300
    assert pane._grid_cells


def test_graph_keeps_volume_index_for_non_pet(qtbot, tmp_path) -> None:
    """REGRESSION: the BOLD graph is exactly what it was."""
    img = _write_pet(tmp_path, name="sub-001_task-rest_bold")
    pane = _loaded_pane(qtbot, img, tmp_path)
    pane._toggle_graph()
    qtbot.wait(50)

    assert not pane._graph_x_is_time
    assert list(pane._graph_x) == list(range(len(DURATIONS)))


def test_marker_follows_the_time_axis(qtbot, tmp_path) -> None:
    """The volume marker has to land on the frame's real time, not its index."""
    pane = _loaded_pane(qtbot, _write_pet(tmp_path), tmp_path)
    pane._toggle_graph()
    qtbot.wait(50)

    pane._vol_slider.setValue(len(DURATIONS) - 1)
    pane._update_graph_marker()
    qtbot.wait(20)

    centre = next(c for c in pane._grid_cells if c["is_center"])
    marker = centre["marker"]
    assert marker is not None
    x_at = marker.getData()[0][0]
    assert x_at == pytest.approx(pane._graph_x[-1])
    assert x_at > len(DURATIONS), "a time axis, not an index"

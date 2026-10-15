"""From values to colours: windows, colour maps, layers, the mosaic line."""

from __future__ import annotations

import numpy as np
import pytest

from bidsmgr.viz.compute import colormaps, intensity, mosaic
from bidsmgr.viz.scene import LabelTable, VolumeDisplay


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_normalise_maps_the_window_to_zero_one() -> None:
    out = intensity.normalise(np.array([-10.0, 0.0, 50.0, 100.0, 300.0]), 0.0, 100.0)
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


def test_gamma_lifts_the_mid_tones() -> None:
    mid = intensity.normalise(np.array([50.0]), 0.0, 100.0, gamma=2.0)[0]
    assert mid == pytest.approx(0.5 ** 0.5)
    # The ends do not move.
    ends = intensity.normalise(np.array([0.0, 100.0]), 0.0, 100.0, gamma=2.0)
    assert ends.tolist() == pytest.approx([0.0, 1.0])


def test_a_zero_width_window_does_not_divide_by_zero() -> None:
    out = intensity.normalise(np.array([5.0, 6.0]), 5.0, 5.0)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Colouring one layer
# ---------------------------------------------------------------------------


def _gray(**kw) -> VolumeDisplay:
    return VolumeDisplay(colormap="gray", gamma=1.0, **kw)


def test_gray_runs_black_to_white_through_the_window() -> None:
    vals = np.array([[0.0, 100.0]])
    rgba = intensity.colorize(vals, _gray(), is_base=True, window=(0.0, 100.0))
    assert rgba.shape == (1, 2, 4) and rgba.dtype == np.uint8
    assert rgba[0, 0, :3].tolist() == [0, 0, 0]
    assert rgba[0, 1, :3].tolist() == [255, 255, 255]
    assert rgba[..., 3].tolist() == [[255, 255]]


def test_invert_reverses_the_map() -> None:
    vals = np.array([[0.0]])
    rgba = intensity.colorize(vals, _gray(invert=True), is_base=True, window=(0.0, 1.0))
    assert rgba[0, 0, :3].tolist() == [255, 255, 255]


def test_no_data_is_transparent() -> None:
    vals = np.array([[np.nan, 1.0]])
    rgba = intensity.colorize(vals, _gray(), is_base=True, window=(0.0, 1.0))
    assert rgba[0, 0, 3] == 0 and rgba[0, 1, 3] == 255


def test_an_overlay_hides_what_is_below_its_threshold() -> None:
    vals = np.array([[1.0, 5.0]])
    disp = _gray(threshold_mode="hide_below")
    rgba = intensity.colorize(vals, disp, is_base=False, window=(3.0, 6.0))
    assert rgba[0, 0, 3] == 0 and rgba[0, 1, 3] == 255


def test_translucent_below_fades_toward_the_threshold() -> None:
    """Faint near the threshold, gone one window-width below it."""
    vals = np.array([[2.9, 1.0, -5.0, 5.0]])
    disp = _gray(threshold_mode="translucent_below")
    rgba = intensity.colorize(vals, disp, is_base=False, window=(3.0, 6.0))
    near, far, gone, above = rgba[0, :, 3]
    assert 0 < far < near < above == 255
    assert gone == 0


def test_a_base_layer_in_range_mode_shows_everything() -> None:
    vals = np.array([[-5.0]])
    rgba = intensity.colorize(vals, _gray(), is_base=True, window=(0.0, 1.0))
    assert rgba[0, 0, 3] == 255


def test_opacity_scales_alpha() -> None:
    vals = np.array([[1.0]])
    rgba = intensity.colorize(vals, _gray(opacity=0.5), is_base=True, window=(0.0, 1.0))
    assert rgba[0, 0, 3] == 128


def test_the_negative_tail_takes_its_own_map() -> None:
    """A two-tailed statistic: positive in red, negative in blue."""
    vals = np.array([[4.0, -4.0, 0.5]])
    disp = VolumeDisplay(colormap="red", colormap_negative="blue", gamma=1.0,
                         threshold_mode="hide_below", window=(2.0, 4.0),
                         window_negative=(2.0, 4.0))
    rgba = intensity.colorize(vals, disp, is_base=False, window=(2.0, 4.0))
    pos, neg, weak = rgba[0]
    assert pos[0] > 200 and pos[2] < 50
    assert neg[2] > 200 and neg[0] < 50
    assert weak[3] == 0


def test_colour_images_keep_their_colours() -> None:
    rgb = np.zeros((1, 2, 3), dtype=np.float32)
    rgb[0, 0] = (200, 100, 0)
    rgb[0, 1] = (0, 0, 255)
    rgba = intensity.colorize(rgb, _gray(), is_base=True)
    assert rgba[0, 0, :3].tolist() == [200, 100, 0]
    assert rgba[0, 1, :3].tolist() == [0, 0, 255]
    # Colour already in 0-1 is stretched to bytes, not left nearly black.
    small = intensity.colorize(rgb / 255.0, _gray(), is_base=True)
    assert small[0, 0, 0] == 200


def test_labels_get_their_colours_and_zero_is_empty() -> None:
    vals = np.array([[0.0, 1.0, 2.0, 7.0]])
    table = LabelTable(labels={1: "left", 2: "right", 7: "seven"},
                       colors={1: (255, 0, 0), 2: (0, 255, 0)})
    rgba = intensity.colorize(vals, _gray(label_table=table), is_base=False)
    assert rgba[0, 0, 3] == 0
    assert rgba[0, 1, :3].tolist() == [255, 0, 0]
    assert rgba[0, 2, :3].tolist() == [0, 255, 0]
    # A named label with no colour still shows, in a stable colour.
    assert rgba[0, 3, 3] == 255
    again = intensity.colorize(vals, _gray(label_table=table), is_base=False)
    assert again[0, 3].tolist() == rgba[0, 3].tolist()


# ---------------------------------------------------------------------------
# Layers together
# ---------------------------------------------------------------------------


def test_composite_blends_bottom_to_top() -> None:
    base = np.zeros((1, 1, 4), dtype=np.uint8)
    base[..., :3] = 100
    base[..., 3] = 255
    top = np.zeros((1, 1, 4), dtype=np.uint8)
    top[..., 0] = 255
    top[..., 3] = 128
    out = intensity.composite([base, top])
    r, g, _b, a = out[0, 0].tolist()
    assert 170 <= r <= 180 and 45 <= g <= 55 and a == 255


def test_composite_needs_something() -> None:
    with pytest.raises(ValueError):
        intensity.composite([])


def test_outline_keeps_only_the_edge() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    edge = intensity.outline(mask)
    assert edge[1, 1] and edge[1, 2] and not edge[2, 2]
    assert not edge[0, 0]


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------


def test_every_vendored_map_loads() -> None:
    names = colormaps.names()
    assert len(names) >= 70
    for name in names:
        table = colormaps.lut(name)
        assert table.shape == (256, 4) and table.dtype == np.uint8


def test_favourites_come_first() -> None:
    names = colormaps.names()
    assert names[0] == "gray"
    assert names.index("viridis") < names.index("actc")


def test_an_unknown_map_falls_back_to_gray() -> None:
    """A scene saved with a map a later version renamed still opens."""
    assert np.array_equal(colormaps.lut("no_such_map"), colormaps.lut("gray"))


def test_tables_are_read_only_and_cached() -> None:
    table = colormaps.lut("hot")
    assert table is colormaps.lut("hot")
    with pytest.raises(ValueError):
        table[0, 0] = 1


def test_ct_maps_suggest_a_window_and_placeholders_do_not() -> None:
    assert colormaps.suggested_window("ct_bones") == (180.0, 600.0)
    assert colormaps.suggested_window("afni_blues_inv") is None
    assert colormaps.suggested_window("gray") is None


def test_swatch() -> None:
    assert colormaps.swatch("viridis", 64).shape == (1, 64, 4)


# ---------------------------------------------------------------------------
# The mosaic line
# ---------------------------------------------------------------------------


def test_mosaic_rows_planes_and_positions() -> None:
    m = mosaic.parse("A -20 0 20 ; C 0 S X 10")
    assert [[t.plane for t in row] for row in m.rows] == [
        ["axial", "axial", "axial"], ["coronal", "sagittal"],
    ]
    assert [t.mm for t in m.rows[0]] == [-20.0, 0.0, 20.0]
    assert m.rows[1][1].cross and not m.rows[1][0].cross
    assert m.labels and m.overlap == 0.0 and not m.errors


def test_mosaic_options() -> None:
    m = mosaic.parse("L- H 0.3 A 0 10")
    assert not m.labels and m.overlap == pytest.approx(0.3)
    assert len(m.tiles) == 2


def test_mosaic_render_tiles_are_skipped_and_counted() -> None:
    m = mosaic.parse("A 0 R 0 10")
    assert len(m.tiles) == 1 and m.skipped_renders == 2


def test_mosaic_reports_what_it_did_not_understand() -> None:
    m = mosaic.parse("A 0 banana 5")
    assert len(m.tiles) == 2 and m.errors == ["not understood: banana"]
    assert mosaic.parse("").tiles == []

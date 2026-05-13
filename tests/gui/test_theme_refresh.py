"""Tests for the theme refresh cascade.

Toggling the theme must propagate through every panel that holds
palette-baked styling so the user sees an immediate update without
restarting the app. Three checks:

* The brand logo's inline gradient stylesheet differs between dark
  and light.
* ``ConverterPanel.repaint_for_palette`` is invoked by the
  ``ThemeManager`` listener cascade (verified by spying on the panel
  method).
* The placeholder labels that used to be hardcoded grey are now
  driven by the ``pane-hint`` object name so QSS handles them.
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QLabel

from bidsmgr.gui.main_window import MainWindow
from bidsmgr.gui.theme_manager import ThemeManager


pytestmark = pytest.mark.gui


def test_logo_loads_bundled_pixmap(qapp) -> None:
    """The brand logo now uses the bundled PNG asset; the label should
    carry a non-null QPixmap (not an inline-gradient stylesheet).
    """
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    pix = win._header._logo.pixmap()
    assert pix is not None and not pix.isNull(), (
        "expected the bundled logo.png to load; got an empty pixmap"
    )
    # The logo survives a theme toggle (it's reloaded from the same
    # asset; we only check that the pixmap stays valid, not identical).
    theme.toggle()
    assert not win._header._logo.pixmap().isNull()


def test_logo_inverts_in_dark_theme(qapp) -> None:
    """Dark theme inverts the bundled PNG so the dark-on-transparent
    artwork reads as light-on-transparent against the dark surface.
    Sampled by hashing a center scanline of pixel RGB.
    """
    theme = ThemeManager(qapp)
    win = MainWindow(theme)

    def _sample_pixel(pix) -> tuple[int, int, int, int]:
        # Convert to image and read a pixel that's likely opaque
        # (centre of the logo, which on the source PNG is ink, not
        # background).
        img = pix.toImage()
        cx, cy = img.width() // 2, img.height() // 2
        c = img.pixelColor(cx, cy)
        return (c.red(), c.green(), c.blue(), c.alpha())

    theme.apply("light")
    qapp.processEvents()
    light_pixel = _sample_pixel(win._header._logo.pixmap())

    theme.apply("dark")
    qapp.processEvents()
    dark_pixel = _sample_pixel(win._header._logo.pixmap())

    # The two pixels should differ: ``invertPixels(InvertRgb)`` flips
    # RGB while preserving alpha, so r/g/b differ but alpha stays.
    assert light_pixel != dark_pixel
    assert light_pixel[3] == dark_pixel[3]  # alpha preserved
    # If the source was an ink pixel (say RGB ~(20,20,20,255)) it
    # inverts to ~(235,235,235,255). Sum of channels should rise.
    if light_pixel[3] > 0:
        light_sum = sum(light_pixel[:3])
        dark_sum = sum(dark_pixel[:3])
        # Inversion preserves total: r+g+b in light + r+g+b in dark
        # ≈ 3 * 255 = 765 (per channel: x + (255-x) = 255).
        assert abs((light_sum + dark_sum) - 3 * 255) < 6


def test_converter_panel_repaint_listener_fires(qapp, monkeypatch) -> None:
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    calls: list = []
    original = win.converter.repaint_for_palette
    def _spy(pal):
        calls.append(pal)
        return original(pal)
    monkeypatch.setattr(win.converter, "repaint_for_palette", _spy)

    theme.toggle()  # dark → light
    theme.toggle()  # light → dark

    assert len(calls) == 2
    # Distinct palettes per call.
    assert calls[0]["bg"] != calls[1]["bg"]


def test_placeholder_labels_use_pane_hint_object_name(qapp) -> None:
    """The empty-state hints in panes are no longer hardcoded grey;
    they live under ``#pane-hint`` so QSS handles light/dark refresh.
    """
    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qapp.processEvents()

    hints = [
        lbl for lbl in win.findChildren(QLabel)
        if lbl.objectName() == "pane-hint"
    ]
    # At least one hint per: raw FS pane, filter pane, conflicts tab,
    # stats tab, inspection-stack empty, plus the "(coming in a later
    # milestone)" placeholders. We don't pin an exact count — just that
    # the namespace is in use.
    assert len(hints) >= 3

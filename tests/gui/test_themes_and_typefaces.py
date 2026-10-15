"""The themes and the two bundled typefaces.

Every theme must give the stylesheet every token it names (a missing one
leaves ``$name`` in the text and Qt drops the whole stylesheet), read
clearly (contrast measured, not eyeballed), say whether it is dark, and be
pickable from the header and from Settings. The interface is drawn in
Inter and code in JetBrains Mono, bundled, so no rule may name a font only
macOS has.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bidsmgr.gui import theme_assets, typefaces
from bidsmgr.gui.theme_manager import PALETTES, THEMES, ThemeManager, theme_info
from bidsmgr.viz.theme import VizTheme, luminance

pytestmark = pytest.mark.gui

QSS = (Path(typefaces.__file__).parent / "theme.qss").read_text(encoding="utf-8")
GUI = Path(typefaces.__file__).parent


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _oklab(rgb: list[float]) -> tuple[float, float, float]:
    r, g, b = rgb
    lms = [0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
           0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
           0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b]
    lc, mc, sc = (v ** (1 / 3) if v >= 0 else -((-v) ** (1 / 3)) for v in lms)
    return (0.2104542553 * lc + 0.7936177850 * mc - 0.0040720468 * sc,
            1.9779984951 * lc - 2.4285922050 * mc + 0.4505937099 * sc,
            0.0259040371 * lc + 0.7827717662 * mc - 0.8086757660 * sc)


#: Machado et al. 2009, full-severity deuteranopia and protanopia (linear RGB).
_CVD = (((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413),
         (-0.011820, 0.042940, 0.968881)),
        ((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216),
         (-0.003882, -0.048116, 1.051998)))


def _separation(a: str, b: str) -> tuple[float, float]:
    """OKLab distance x100: for normal vision, and the worse of deutan and
    protan simulation."""
    import math

    from bidsmgr.viz.theme import parse_colour

    def rgb(h):
        return [_lin(c / 255) for c in parse_colour(h)[:3]]

    def sim(v, m):
        return [min(1.0, max(0.0, sum(m[i][j] * v[j] for j in range(3)))) for i in range(3)]

    va, vb = rgb(a), rgb(b)
    normal = 100 * math.dist(_oklab(va), _oklab(vb))
    cvd = min(100 * math.dist(_oklab(sim(va, m)), _oklab(sim(vb, m))) for m in _CVD)
    return normal, cvd


@pytest.fixture
def manager(qapp):
    m = ThemeManager(qapp)
    yield m
    m.apply("dark")


def test_every_theme_is_whole_and_grouped() -> None:
    ids = [t.id for t in THEMES]
    assert len(ids) == len(set(ids)) == len(PALETTES) == 19
    assert "monokai" in ids and theme_info("monokai").dark
    assert theme_info("original-dark").partner == "original-light"
    # Neutral greys beside the blue-cast Dark (user request, 2026-10-09).
    assert {"graphite", "carbon", "ash", "porcelain"} <= set(ids)
    groups = [t.group for t in THEMES]
    assert groups == sorted(groups, key=["Neutral", "Classic", "Colour",
                                         "High contrast"].index), "listed group by group"
    keys = set(PALETTES["dark"])
    for t in THEMES:
        assert set(PALETTES[t.id]) == keys, t.id
        assert theme_info(t.partner).dark != t.dark, t.id


@pytest.mark.parametrize("theme_id", [t.id for t in THEMES])
def test_plot_colours_stay_apart_in_every_theme(theme_id) -> None:
    """The plots' own series colours: neighbours apart for every reader
    (OKLab distance 15 for normal vision, 6 under red-green colour
    blindness), and 3:1 against the plot's background."""
    p = PALETTES[theme_id]
    series = [p[f"series{i}"] for i in range(1, 7)]
    for a, b in zip(series, series[1:]):
        normal, cvd = _separation(a, b)
        assert normal >= 15 and cvd >= 6, (a, b, normal, cvd)
    for colour in series:
        for surface in ("bg", "surface"):
            assert _contrast(colour, p[surface]) >= 3, (colour, surface)
    assert VizTheme.from_palette(p, theme_id).series(0) == series[0]


@pytest.mark.parametrize("theme_id", [t.id for t in THEMES])
def test_file_names_follow_the_theme_s_palette(theme_id) -> None:
    """Images, sidecars, tables and recordings are ONE family per theme:
    hues on an arc around the accent (within 80 degrees), at one lightness
    (within 0.03 OKLab), gently coloured. Round 4's hand-picked sets were
    vivid, of every lightness and of hues the theme did not have, and read as
    heterogeneous. They still read on every surface, differ from the text
    and from each other, and their icons differ too."""
    import itertools

    from bidsmgr.gui import tones

    p = PALETTES[theme_id]
    strong = theme_id.startswith("hc-")
    kinds = ("image", "sidecar", "table", "recording")
    colours = {k: p[f"tree_{k}"] for k in kinds}
    assert p["tree_folder"] == p["accent"] and p["tree_other"] == p["dim"]
    lightness = [tones.oklch(c)[0] for c in colours.values()]
    assert max(lightness) - min(lightness) <= 0.03, (theme_id, lightness)
    accent_hue = tones.oklch(p["accent"])[2]
    for kind, colour in colours.items():
        _l, chroma, hue = tones.oklch(colour)
        gap = abs(hue - accent_hue) % 360
        assert min(gap, 360 - gap) <= 80, (kind, colour)
        assert chroma <= 0.14, (kind, colour, "gently coloured")
        assert _separation(colour, p["text"])[0] >= 10, kind
        for surface in ("bg", "surface", "surface2", "surface3"):
            floor = 7 if strong else (4.0 if surface == "surface3" else 4.5)
            assert _contrast(colour, p[surface]) >= floor, (kind, surface)
    for a, b in itertools.combinations(("folder",) + kinds, 2):
        normal = _separation(p[f"tree_{a}"], p[f"tree_{b}"])[0]
        assert normal >= 4, (a, b, round(normal, 1))


def test_the_original_themes_are_the_first_ones() -> None:
    """Dark and Light as they were before the 2026-10-09 restyle (only the
    input outline moved, to reach 3:1 on the card)."""
    dark, light = PALETTES["original-dark"], PALETTES["original-light"]
    assert (dark["bg"], dark["surface"], dark["text"], dark["accent"]) == (
        "#0a0e13", "#11161d", "#e6edf3", "#58a6ff")
    assert (light["bg"], light["surface"], light["surface2"], light["text"], light["accent"]) \
        == ("#ffffff", "#f6f8fa", "#ffffff", "#1f2328", "#0969da")
    assert not theme_info("original-light").dark and theme_info("original-dark").dark


def test_every_token_the_stylesheet_names_is_given() -> None:
    used = set(re.findall(r"\$\{?([A-Za-z_]\w*)\}?", QSS))
    images = set(theme_assets.images(PALETTES["dark"]))
    for t in THEMES:
        assert not used - set(PALETTES[t.id]) - images, t.id


@pytest.mark.parametrize("theme_id", [t.id for t in THEMES])
def test_text_reads_clearly_in_every_theme(theme_id) -> None:
    """WCAG contrast: text at 7:1 (AAA) on every surface, secondary text and
    every status colour at 4.5:1 (4.0 on the button surface), the outline
    of an input at 3:1; stricter for the high-contrast pair."""
    p = PALETTES[theme_id]
    strong = theme_id.startswith("hc-")
    for surface in ("bg", "surface", "surface2", "surface3"):
        assert _contrast(p["text"], p[surface]) >= (15 if strong else 7), surface
        assert _contrast(p["dim"], p[surface]) >= (7 if strong else 4.5), surface
        for colour in ("accent", "success", "warning", "error", "purple", "teal"):
            floor = 7 if strong else (4.0 if surface == "surface3" else 4.5)
            assert _contrast(p[colour], p[surface]) >= floor, (colour, surface)
    assert _contrast(p["primary_btn_text"], p["accent"]) >= (7 if strong else 4.5)
    for surface in ("bg", "surface"):
        assert _contrast(p["input_border"], p[surface]) >= (4.5 if strong else 3.0), surface
    # The states a status chip shows side by side read apart in colour too.
    for a, b in (("success", "error"), ("warning", "error"), ("success", "warning"),
                 ("accent", "error")):
        assert _separation(p[a], p[b])[0] >= 12, (a, b)


def test_each_theme_says_whether_it_is_dark() -> None:
    for t in THEMES:
        assert VizTheme.from_palette(PALETTES[t.id], t.id).dark == t.dark, t.id


def test_applying_a_theme_leaves_no_token_unfilled(manager, qapp) -> None:
    for t in THEMES:
        manager.apply(t.id)
        assert manager.name == t.id and manager.is_dark == t.dark
        sheet = qapp.styleSheet()
        assert not re.search(r"\$[A-Za-z_]", sheet), t.id
        assert PALETTES[t.id]["accent"] in sheet


def test_toggle_goes_to_the_partner(manager) -> None:
    manager.apply("dim")
    assert manager.toggle() == "paper"
    manager.apply("hc-light")
    assert manager.toggle() == "hc-dark"


def test_the_stylesheet_images_are_drawn_in_the_theme(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(theme_assets, "_cache_dir", lambda: tmp_path / "with space")
    tokens = theme_assets.write(PALETTES["paper"], "paper")
    assert set(tokens) == set(theme_assets.images(PALETTES["paper"]))
    for token in tokens.values():
        assert token.startswith('url("') and "\\" not in token
        path = Path(token[5:-2])
        assert path.is_file() and path.read_text(encoding="utf-8").startswith("<svg")
    tick = Path(tokens["icon_check"][5:-2]).read_text(encoding="utf-8")
    assert PALETTES["paper"]["primary_btn_text"] in tick


def test_the_typefaces_are_bundled_and_used(qapp) -> None:
    from PyQt6.QtGui import QFontInfo

    assert typefaces.load()
    assert QFontInfo(typefaces.ui_font(13)).family() == typefaces.UI_FAMILY
    from bidsmgr.gui.viz import fonts

    assert QFontInfo(fonts.font(12, mono=True)).family() == typefaces.MONO_FAMILY
    for name in ("Inter-OFL.txt", "JetBrainsMono-OFL.txt"):
        assert (typefaces.FONT_DIR / name).is_file(), "a licence travels with its fonts"


def test_no_rule_or_widget_names_a_font_only_macos_has() -> None:
    mac = re.compile(r"SF Mono|Menlo|Monaco|AppleSystemUIFont|Helvetica Neue|SFMono")
    assert not mac.search(QSS)
    families = set(re.findall(r'font-family:\s*([^;]+);', QSS))
    assert families == {'"JetBrains Mono"'}, families
    for path in GUI.rglob("*.py"):
        if path.name == "typefaces.py":
            continue
        assert not mac.search(path.read_text(encoding="utf-8")), path


def test_the_header_lists_every_theme_and_applies_the_pick(manager, qtbot) -> None:
    from bidsmgr.gui.app_settings import AppSettings
    from bidsmgr.gui.main_window import MainWindow

    manager.apply("dark")
    win = MainWindow(manager)
    qtbot.addWidget(win)
    header = win._header
    header._open_theme_menu()
    menu = header._theme_menu
    acts = [a for a in menu.actions() if a.data()]
    assert [a.data() for a in acts] == [t.id for t in THEMES]
    from PyQt6.QtWidgets import QLabel

    headings = [lb.text() for lb in menu.findChildren(QLabel) if lb.objectName() == "menu-section"]
    assert headings == ["NEUTRAL", "CLASSIC", "COLOUR", "HIGH CONTRAST"]
    current = next(a for a in acts if a.isChecked())
    assert current.data() == "dark" and current.font().weight() > 500
    next(a for a in acts if a.data() == "nord").trigger()
    menu.close()
    assert manager.name == "nord"
    assert AppSettings.load().theme == "nord"
    assert "Nord" in header._theme_btn.toolTip()


def test_settings_lists_the_themes_by_name(qtbot) -> None:
    from bidsmgr.gui.app_settings import AppSettings
    from bidsmgr.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    combo = dlg._theme_combo
    listed = [combo.itemData(i) for i in range(combo.count())]
    assert [d for d in listed if d] == [t.id for t in THEMES]
    assert listed.count(None) == 3, "a separator between the four groups"
    assert combo.itemText(combo.findData("hc-light")) == "High contrast light"
    assert not combo.itemIcon(0).isNull(), "each theme shows its swatch"


def test_welcome_links_follow_a_theme_change(manager, qtbot) -> None:
    from PyQt6.QtWidgets import QLabel

    from bidsmgr.gui.welcome_panel import WelcomePanel

    manager.apply("nord")
    panel = WelcomePanel()
    qtbot.addWidget(panel)
    link = next(lb for lb in panel.findChildren(QLabel) if lb.objectName() == "welcome-link")
    assert PALETTES["nord"]["accent"] in link.text()
    manager.apply("paper")
    panel.repaint_for_palette(PALETTES["paper"])
    assert PALETTES["paper"]["accent"] in link.text()
    assert PALETTES["nord"]["accent"] not in link.text()


def test_the_cli_accepts_every_theme() -> None:
    from bidsmgr.main import _theme_ids

    assert _theme_ids() == [t.id for t in THEMES]

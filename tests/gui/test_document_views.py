"""The Editor's views of a README and of a gradient table: what they show,
that they follow a theme switch and the font size, and that their parts
stay linked.

The Markdown view gives Qt's parsed document a repository page's
typography and paints what rich text cannot (code boxes, quote bars,
rules); the gradient pane draws each shell's directions with an equal-area
projection and links its two plots to its table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bidsmgr.gui.viz import fonts
from bidsmgr.gui.viz.bridge import ThemeHub
from bidsmgr.gui.widgets.gradient_pane import (
    GradientPane, hemisphere, ring_radius, shell_rows, summary_lines,
)
from bidsmgr.gui.widgets.markdown_view import HEADING_PX, MarkdownView

pytestmark = pytest.mark.gui

README = """# Study

Some **words** with `code` and a [link](https://example.org).

## Contents

- one
- two
  - nested

> A quoted remark
> on two lines.

```bash
bidsmgr-validate study
```

| A | B |
|---|---|
| 1 | 2 |

---

See [CHANGES](CHANGES).
"""


@pytest.fixture
def themes():
    """Publish a palette, and put the dark one back afterwards."""
    from bidsmgr.gui.theme_manager import PALETTES

    def publish(name: str) -> None:
        ThemeHub.instance().publish(PALETTES[name], name)

    yield publish
    publish("dark")


def _blocks(doc):
    block = doc.begin()
    while block.isValid():
        yield block
        block = block.next()


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


@pytest.fixture
def view(qtbot, tmp_path) -> MarkdownView:
    (tmp_path / "CHANGES").write_text("1.0.0\n", encoding="utf-8")
    v = MarkdownView()
    qtbot.addWidget(v)
    v.resize(900, 700)
    v.set_markdown(README, tmp_path)
    v.show()
    qtbot.waitExposed(v)
    return v


def test_headings_have_their_size_room_and_rule(view) -> None:
    doc = view.document()
    heads = {b.text(): b for b in _blocks(doc) if b.blockFormat().headingLevel()}
    assert set(heads) == {"Study", "Contents"}
    first = heads["Study"].begin().fragment().charFormat().font()
    second = heads["Contents"].begin().fragment().charFormat().font()
    # Qt's size adjustment would otherwise override the size (and did).
    assert first.pixelSize() == fonts.px(HEADING_PX[1])
    assert second.pixelSize() == fonts.px(HEADING_PX[2])
    assert heads["Contents"].blockFormat().topMargin() > 0
    assert len(view._underlined) == 2


def test_code_quotes_rules_and_tables_are_drawn(view) -> None:
    assert len(view._codes) == 1 and len(view._quotes) == 1 and len(view._rules) == 1
    first, last = view._codes[0]
    code = view.document().findBlockByNumber(first)
    assert code.text() == "bidsmgr-validate study"
    from bidsmgr.gui.typefaces import MONO_FAMILY

    assert MONO_FAMILY in code.begin().fragment().charFormat().fontFamilies()
    # The rule is painted in the border colour, never Qt's own in the text
    # colour on top of it.
    from PyQt6.QtGui import QTextFormat

    rule = view.document().findBlockByNumber(view._rules[0])
    assert not rule.blockFormat().hasProperty(
        QTextFormat.Property.BlockTrailingHorizontalRulerWidth)
    image = view.viewport().grab().toImage()
    assert not image.isNull()


def test_the_text_sits_in_a_centred_column(view, qtbot) -> None:
    view.resize(1800, 700)
    qtbot.wait(20)
    left, right = view.column()
    assert right - left <= fonts.px(900)
    assert abs(left - (view.viewport().width() - right)) <= 2


def test_a_theme_switch_re_renders_in_the_new_colours(view, themes) -> None:
    def link_colour():
        for b in _blocks(view.document()):
            for frag in (b.begin().fragment(),):
                it = b.begin()
                while not it.atEnd():
                    cf = it.fragment().charFormat()
                    if cf.isAnchor():
                        return cf.foreground().color().name()
                    it += 1
        return None

    themes("dark")
    dark = link_colour()
    themes("light")
    light = link_colour()
    assert dark and light and dark != light
    assert light == ThemeHub.instance().theme.accent.lower()


def test_a_link_to_a_file_beside_it_is_announced(view, qtbot, tmp_path) -> None:
    from PyQt6.QtCore import QUrl

    with qtbot.waitSignal(view.file_requested, timeout=1000) as got:
        view._on_link(QUrl("CHANGES"))
    assert got.args[0] == (tmp_path / "CHANGES").resolve()


# ---------------------------------------------------------------------------
# Gradient table
# ---------------------------------------------------------------------------


def test_the_projection_keeps_an_even_scheme_even() -> None:
    # Opposite directions are one point; the axis is the centre; a direction
    # across the axis lies on the circle.
    v = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 0, 0]], float)
    xy = hemisphere(v, "z")
    assert np.allclose(xy[0], [0, 0]) and np.allclose(xy[1], [0, 0])
    assert np.allclose(np.hypot(*xy[2]), 1.0)
    assert np.all(np.isnan(xy[3]))
    # Equal area: uniformly random directions fill equal areas of the disc
    # equally (the inner half of the area holds about half the points).
    rng = np.random.default_rng(0)
    d = rng.normal(size=(20000, 3))
    r = np.hypot(*hemisphere(d, "y").T)
    assert abs(np.mean(r < np.sqrt(0.5)) - 0.5) < 0.02
    assert np.isclose(ring_radius(90), 1.0) and ring_radius(30) < ring_radius(60) < 1


def _dwi(tmp_path: Path, bvals, n_image=None) -> Path:
    import nibabel as nib

    dwi = tmp_path / "ds" / "sub-01" / "dwi"
    dwi.mkdir(parents=True, exist_ok=True)
    bvals = np.asarray(bvals)
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(len(bvals), 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs[bvals == 0] = 0
    np.savetxt(dwi / "sub-01_dwi.bval", bvals[None], fmt="%d")
    np.savetxt(dwi / "sub-01_dwi.bvec", vecs.T, fmt="%.6f")
    n = len(bvals) if n_image is None else n_image
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, n), np.int16), np.eye(4)),
             str(dwi / "sub-01_dwi.nii.gz"))
    return dwi / "sub-01_dwi.bval"


@pytest.fixture
def pane(qtbot, tmp_path) -> GradientPane:
    p = GradientPane()
    qtbot.addWidget(p)
    p.resize(1100, 640)
    p.set_file(_dwi(tmp_path, [0] + [1000] * 8 + [0] + [2000] * 10), None)
    p.show()
    qtbot.waitExposed(p)
    return p


def test_the_facts_and_the_shells(pane) -> None:
    assert pane.facts() == ["20 volumes", "2 at b=0", "2 shells",
                            "Matches sub-01_dwi.nii.gz (20 volumes)"]
    assert sorted(pane._shell_boxes) == [1000, 2000]
    assert pane._shell_boxes[1000].text().startswith("b=1000 · 8 directions · gap ")
    rows = shell_rows(pane._table_data)
    assert [r[:2] for r in rows] == [(1000, 8), (2000, 10)]
    assert not pane._findings.isVisible()


def test_a_table_that_does_not_match_its_image_says_so(qtbot, tmp_path) -> None:
    p = GradientPane()
    qtbot.addWidget(p)
    p.set_file(_dwi(tmp_path, [0, 1000, 1000, 1000, 1000, 1000, 1000], n_image=9), None)
    assert p._match_chip.text() == "sub-01_dwi.nii.gz has 9 volumes, the table 7"
    assert p._findings.isVisibleTo(p)
    assert any("9" in line for line in summary_lines(p._table_data))


def test_selection_is_shared_by_both_plots_and_the_table(pane) -> None:
    pane.table.selectRow(12)
    assert pane.selected_volume() == 12
    assert len(pane.sphere.ring.data) == 1 and len(pane.bplot.ring.data) == 1
    x, y = pane.bplot.ring.data[0][0], pane.bplot.ring.data[0][1]
    assert (x, y) == (12.0, 2000.0)
    assert pane._dir_card.readout.text().startswith("Volume 12 (")
    assert pane._b_card.readout.text() == "Volume 12: b=2000"
    # A b=0 volume has no direction: marked in the b-values only.
    pane.select_volume(9)
    assert len(pane.sphere.ring.data) == 0 and len(pane.bplot.ring.data) == 1
    assert pane.table.currentRow() == 9


def test_pointing_at_a_direction_reads_it_out(pane, qtbot) -> None:
    xy = pane._sphere_xy[3]
    scene = pane.sphere.vb.mapViewToScene(__import__("PyQt6.QtCore", fromlist=["QPointF"])
                                          .QPointF(*xy))
    pane._on_sphere_moved(scene)
    assert pane._hovered == 3
    assert pane._b_card.readout.text().startswith("Volume 3: b=1000")
    pane._set_hover(None)
    assert pane._dir_card.readout.text() == ""


def test_a_shell_can_be_hidden_in_both_plots(pane) -> None:
    pane._shell_boxes[2000].setChecked(False)
    hidden = [s for s in pane.sphere.scatters if len(s.data) and not s.isVisible()]
    assert len(hidden) == 1
    assert sum(1 for s in pane.bplot.scatters if len(s.data) and not s.isVisible()) == 1
    pane.select_volume(15)
    assert len(pane.sphere.ring.data) == 0, "a hidden shell's volume is not marked"
    pane.set_shell_shown(2000, True)
    assert pane._shell_boxes[2000].isChecked()


def test_the_sphere_is_seen_along_each_axis(pane) -> None:
    before = pane._sphere_xy.copy()
    pane._view_buttons["x"].click()
    assert pane.view() == "x"
    assert not np.allclose(np.nan_to_num(before), np.nan_to_num(pane._sphere_xy))
    assert [t.toPlainText() for t in pane._axis_labels] == ["−y", "+y", "+z", "−z"]


def test_a_theme_switch_recolours_everything(pane, themes) -> None:
    themes("dark")
    dark_bg = pane.sphere.widget.backgroundBrush().color().name()
    dark_dot = pane.sphere.scatters[0].opts["brush"].color().name()
    themes("light")
    theme = ThemeHub.instance().theme
    assert pane.sphere.widget.backgroundBrush().color().name() == theme.plot_background.lower()
    assert pane.bplot.widget.backgroundBrush().color().name() == theme.plot_background.lower()
    assert pane.sphere.scatters[0].opts["brush"].color().name() == theme.series(0).lower()
    assert dark_bg != theme.plot_background.lower() and dark_dot != theme.series(0).lower()
    # The disc stays whole and round: the shorter side keeps the set range
    # (a redraw must not autorange it; the aspect lock widens the other).
    (x0, x1), (y0, y1) = pane.sphere.vb.viewRange()
    assert x0 < -1.05 and x1 > 1.05 and y0 < -1.05 and y1 > 1.05
    assert min(x1 - x0, y1 - y0) < 4.5


def test_the_disc_keeps_room_for_its_letters_when_small(pane, qtbot) -> None:
    pane.resize(700, 330)
    qtbot.wait(30)
    vb = pane.sphere.vb
    (x0, x1), (y0, y1) = vb.viewRange()
    short = min(vb.width(), vb.height())
    # A letter's height fits between the circle's letters and the edge.
    per_px = min(x1 - x0, y1 - y0) / short
    assert (min(x1, y1) - 1.05) / per_px >= fonts.px(fonts.LABEL_PX)


def test_the_pane_shrinks_and_the_table_keeps_its_columns(pane, qtbot) -> None:
    assert pane._split.sizes()[1] >= pane._table_width()
    assert pane.minimumSizeHint().width() < 420

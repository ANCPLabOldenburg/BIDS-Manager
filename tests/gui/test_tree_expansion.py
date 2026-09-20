"""Getting back to the top of a tree, and going down one level at a time.

Expand-all on a real dataset produces thousands of rows and is almost never
what anybody wants. The useful pair, which BIDSvue gets right, is: fold
everything, and open one more level per click.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QSizePolicy

from bidsmgr.gui.widgets.bids_tree_pane import BidsTreePane

pytestmark = pytest.mark.gui


@pytest.fixture
def deep(tmp_path: Path) -> Path:
    """root / sub / ses / datatype / file: four levels below the root row."""
    root = tmp_path / "ds"
    for sub in ("sub-01", "sub-02"):
        for ses in ("ses-pre", "ses-post"):
            for datatype in ("anat", "func"):
                folder = root / sub / ses / datatype
                folder.mkdir(parents=True)
                (folder / f"{sub}_{ses}_T1w.json").write_text("{}")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


def _pane(qtbot, root: Path) -> BidsTreePane:
    pane = BidsTreePane()
    qtbot.addWidget(pane)
    pane.set_root(root)
    return pane


def _expanded_depths(pane: BidsTreePane) -> set[int]:
    out: set[int] = set()

    def visit(item, depth: int) -> None:
        if item.childCount() and item.isExpanded():
            out.add(depth)
        for i in range(item.childCount()):
            visit(item.child(i), depth + 1)

    for i in range(pane._tree.topLevelItemCount()):
        visit(pane._tree.topLevelItem(i), 0)
    return out


def test_collapse_all_leaves_the_dataset_row_open(qtbot, deep: Path) -> None:
    """Folding the root as well is not "back to the top", it is "gone"."""
    pane = _pane(qtbot, deep)
    pane.collapse_all()
    root = pane._tree.topLevelItem(0)
    assert root.isExpanded()
    assert _expanded_depths(pane) == {0}


def test_each_click_opens_exactly_one_more_level(qtbot, deep: Path) -> None:
    pane = _pane(qtbot, deep)
    pane.collapse_all()
    seen = [max(_expanded_depths(pane))]
    for _ in range(3):
        pane.expand_next_level()
        seen.append(max(_expanded_depths(pane)))
    assert seen == [0, 1, 2, 3], seen


def test_expanding_stops_when_everything_is_open(qtbot, deep: Path) -> None:
    pane = _pane(qtbot, deep)
    pane.collapse_all()
    for _ in range(12):
        pane.expand_next_level()
    before = _expanded_depths(pane)
    pane.expand_next_level()
    assert _expanded_depths(pane) == before


def test_the_expand_button_greys_out_when_there_is_nothing_left(
    qtbot, deep: Path,
) -> None:
    """A button that stays enabled and does nothing reads as broken."""
    pane = _pane(qtbot, deep)
    pane.collapse_all()
    assert pane._expand_btn.isEnabled()
    for _ in range(12):
        pane.expand_next_level()
    assert not pane._expand_btn.isEnabled()


def test_collapsing_re_enables_it(qtbot, deep: Path) -> None:
    pane = _pane(qtbot, deep)
    for _ in range(12):
        pane.expand_next_level()
    pane.collapse_all()
    assert pane._expand_btn.isEnabled()


def test_the_next_level_is_the_shallowest_folded_one(
    qtbot, deep: Path,
) -> None:
    """Not "one deeper than the deepest open node". A user who opened one
    branch by hand should still get the next LEVEL, not a jump past it."""
    pane = _pane(qtbot, deep)
    pane.collapse_all()
    root = pane._tree.topLevelItem(0)
    # Open one branch all the way down by hand.
    node = root
    while node.childCount():
        node.setExpanded(True)
        node = node.child(0)
    assert pane._next_folded_depth() == 1, "the sibling branches are still shut"


def test_a_tree_with_no_root_does_not_crash(qtbot, tmp_path: Path) -> None:
    pane = BidsTreePane()
    qtbot.addWidget(pane)
    pane.collapse_all()
    pane.expand_next_level()
    assert pane._next_folded_depth() is None


# ---------------------------------------------------------------------------
# What a freshly opened dataset looks like
# ---------------------------------------------------------------------------


def test_opening_a_dataset_starts_folded(qtbot, deep: Path) -> None:
    """Guessing at two levels opened a many-subject dataset as a wall of rows
    the user then had to close, and going down is the unfold button's job."""
    pane = _pane(qtbot, deep)
    assert _expanded_depths(pane) == {0}, "only the dataset row itself"


def test_the_subject_rows_are_still_visible(qtbot, deep: Path) -> None:
    """"Closed" means nothing below the dataset is open, not that the dataset
    is hidden behind one chevron."""
    pane = _pane(qtbot, deep)
    root = pane._tree.topLevelItem(0)
    assert root.isExpanded()
    labels = {root.child(i).text(0) for i in range(root.childCount())}
    assert {"sub-01", "sub-02"} <= labels


def test_the_unfold_button_is_live_on_open(qtbot, deep: Path) -> None:
    pane = _pane(qtbot, deep)
    assert pane._expand_btn.isEnabled()


# ---------------------------------------------------------------------------
# Where the two buttons are drawn
# ---------------------------------------------------------------------------


def test_the_pair_is_a_fixed_size_control(qtbot, deep: Path) -> None:
    """A button allowed to grow WILL grow. Wrapped in a PanelFrame, which
    hides the pane's own header, these two inherited the whole pane width and
    rendered as two buttons the width of the column."""
    pane = _pane(qtbot, deep)
    for button in (pane._collapse_btn, pane._expand_btn):
        # ``setFixedSize`` pins min to max, which is what actually stops the
        # growth; it deliberately leaves the policy alone.
        assert button.minimumWidth() == button.maximumWidth()
        assert button.minimumHeight() == button.maximumHeight()
    # And the group hugs them rather than claiming the row.
    group = pane._collapse_btn.parentWidget()
    assert group.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed


def test_a_wrapped_pane_puts_them_in_the_frame_title_bar(
    qtbot, deep: Path,
) -> None:
    """Rather than on a row of their own under it."""
    from bidsmgr.gui.widgets.panel_frame import HEADER_EXTRAS, PanelFrame

    pane = _pane(qtbot, deep)
    frame = PanelFrame(pane, "BIDS tree", edge="left")
    qtbot.addWidget(frame)

    group = frame._extras
    assert group is not None
    assert group.objectName() == HEADER_EXTRAS
    assert frame._bar.isAncestorOf(group)
    assert not pane.isAncestorOf(group), "lifted out of the pane's own row"


def test_detaching_hands_the_buttons_back_to_the_pane(
    qtbot, deep: Path,
) -> None:
    """The floating window must not arrive without the controls it owns."""
    from bidsmgr.gui.widgets.panel_frame import PanelFrame

    pane = _pane(qtbot, deep)
    frame = PanelFrame(pane, "BIDS tree", edge="left")
    qtbot.addWidget(frame)
    group = frame._extras

    frame.detach()
    assert pane.isAncestorOf(group), "buttons should travel with the pane"
    frame.reattach()
    assert frame._bar.isAncestorOf(group), "and come back with it"


# ---------------------------------------------------------------------------
# Hidden folders
# ---------------------------------------------------------------------------


@pytest.fixture
def with_machinery(deep: Path) -> Path:
    """The same tree plus ``.bidsmgr/``, deeper than anything in the dataset."""
    backups = deep / ".bidsmgr" / "operations" / "0001" / "backup" / "sub-01"
    backups.mkdir(parents=True)
    (backups / "sub-01_ses-pre_T1w.json").write_text("{}")
    return deep


def _pane_showing_hidden(qtbot, root: Path) -> BidsTreePane:
    """The tree reads the preference at build time, so set it first."""
    from bidsmgr.gui.app_settings import AppSettings

    before = AppSettings.load().editor_show_hidden
    AppSettings.remember_editor_show_hidden(True)
    try:
        return _pane(qtbot, root)
    finally:
        AppSettings.remember_editor_show_hidden(before)


def _hidden_root(pane: BidsTreePane):
    root = pane._tree.topLevelItem(0)
    for i in range(root.childCount()):
        if root.child(i).text(0) == ".bidsmgr":
            return root.child(i)
    return None


def test_unfolding_leaves_the_machinery_folder_shut(
    qtbot, with_machinery: Path,
) -> None:
    """``.bidsmgr/`` is the operation log and every backup it has taken.

    Unfolding the dataset one level at a time must not drag hundreds of rows
    nobody asked for along with it.
    """
    pane = _pane_showing_hidden(qtbot, with_machinery)
    pane.collapse_all()
    for _ in range(12):
        pane.expand_next_level()
    hidden = _hidden_root(pane)
    assert hidden is not None, "the fixture's .bidsmgr row should be visible"
    assert not hidden.isExpanded()


def test_the_machinery_folder_does_not_hold_the_button_open(
    qtbot, with_machinery: Path,
) -> None:
    """It is deeper than the dataset. Counting it would leave the unfold
    button enabled after everything a user cares about is already open."""
    pane = _pane_showing_hidden(qtbot, with_machinery)
    pane.collapse_all()
    for _ in range(12):
        pane.expand_next_level()
    assert not pane._expand_btn.isEnabled()


def test_folding_closes_it_too(qtbot, with_machinery: Path) -> None:
    """The asymmetry with unfolding is deliberate and is about rows.

    Opening ``.bidsmgr/`` produces hundreds nobody asked for; closing it
    produces none, and a "collapse all" that leaves something open is a lie.
    """
    pane = _pane_showing_hidden(qtbot, with_machinery)
    hidden = _hidden_root(pane)
    assert hidden is not None
    hidden.setExpanded(True)
    pane.collapse_all()
    assert not hidden.isExpanded()


def test_unfolding_never_re_opens_it(qtbot, with_machinery: Path) -> None:
    """Which is the half that matters: folded, it stays folded."""
    pane = _pane_showing_hidden(qtbot, with_machinery)
    pane.collapse_all()
    hidden = _hidden_root(pane)
    assert hidden is not None
    for _ in range(12):
        pane.expand_next_level()
    assert not hidden.isExpanded()

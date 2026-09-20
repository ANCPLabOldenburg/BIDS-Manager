"""What the dataset tree says to the right of each row.

Two things replaced a coloured dot: what a folder holds, and how many findings
are in it. A dot says "something is wrong in here" and stops, which on a
folder of four hundred files is the start of a search rather than an answer.

Both are painted by the delegate rather than written into the item text,
because several things look items up by name, and both have to scale with the
UI font: the previous version asked for ``pointSizeF()`` on a font set in
pixels, got ``-1``, and clamped to a fixed size that ignored the setting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtCore import QModelIndex, QRect, Qt
from PyQt6.QtWidgets import QStyleOptionViewItem

from bidsmgr.gui.delegates.bids_tree import (
    BADGE_ROLE,
    COUNT_ROLE,
    ISSUE_ROLE,
    BidsTreeDelegate,
)
from bidsmgr.gui.theme_manager import ThemeManager
from bidsmgr.gui.widgets.bids_tree_pane import BidsTreePane

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for sub in ("sub-01", "sub-02"):
        anat = root / sub / "ses-pre" / "anat"
        anat.mkdir(parents=True)
        (anat / f"{sub}_ses-pre_T1w.nii.gz").write_bytes(b"")
        (anat / f"{sub}_ses-pre_T1w.json").write_text("{}")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    return root


def _find(pane: BidsTreePane, name: str):
    found = pane._tree.findItems(
        name, Qt.MatchFlag.MatchRecursive | Qt.MatchFlag.MatchExactly, 0,
    )
    return found[0] if found else None


def _pane(qtbot, dataset: Path) -> BidsTreePane:
    pane = BidsTreePane()
    qtbot.addWidget(pane)
    pane.set_root(dataset)
    pane._tree.expandAll()
    return pane


# ---------------------------------------------------------------------------
# Counts, and how they roll up
# ---------------------------------------------------------------------------


def test_a_file_carries_its_own_finding_counts(qtbot, dataset: Path) -> None:
    pane = _pane(qtbot, dataset)
    target = dataset / "sub-01" / "ses-pre" / "anat" / "sub-01_ses-pre_T1w.json"
    pane.set_badges({target: "err"}, {target: (2, 5)})
    assert _find(pane, target.name).data(0, ISSUE_ROLE) == (2, 5)


def test_a_folder_sums_its_descendants_rather_than_taking_the_worst(
    qtbot, dataset: Path,
) -> None:
    """The point of a number is to say how much work is in there. A subject
    with one missing recommended field must not look like one with ninety."""
    pane = _pane(qtbot, dataset)
    anat = dataset / "sub-01" / "ses-pre" / "anat"
    pane.set_badges(
        {anat / "sub-01_ses-pre_T1w.json": "err",
         anat / "sub-01_ses-pre_T1w.nii.gz": "warn"},
        {anat / "sub-01_ses-pre_T1w.json": (2, 5),
         anat / "sub-01_ses-pre_T1w.nii.gz": (0, 3)},
    )
    assert _find(pane, "anat").data(0, ISSUE_ROLE) == (2, 8)
    assert _find(pane, "sub-01").data(0, ISSUE_ROLE) == (2, 8)
    # The other subject is untouched, not given the dataset total.
    assert _find(pane, "sub-02").data(0, ISSUE_ROLE) is None


def test_a_clean_file_gets_no_count_and_keeps_its_tick(
    qtbot, dataset: Path,
) -> None:
    pane = _pane(qtbot, dataset)
    target = dataset / "sub-01" / "ses-pre" / "anat" / "sub-01_ses-pre_T1w.json"
    pane.set_badges({target: "ok"}, {})
    item = _find(pane, target.name)
    assert item.data(0, ISSUE_ROLE) is None
    assert item.data(0, BADGE_ROLE) == "ok"


def test_clearing_removes_the_counts_too(qtbot, dataset: Path) -> None:
    pane = _pane(qtbot, dataset)
    target = dataset / "sub-01" / "ses-pre" / "anat" / "sub-01_ses-pre_T1w.json"
    pane.set_badges({target: "err"}, {target: (1, 0)})
    pane.clear_badges()
    item = _find(pane, target.name)
    assert item.data(0, ISSUE_ROLE) is None
    assert item.data(0, BADGE_ROLE) is None


def test_counts_survive_a_live_refresh(qtbot, dataset: Path) -> None:
    """A file watcher rebuilding the tree must not silently drop them."""
    pane = _pane(qtbot, dataset)
    target = dataset / "sub-01" / "ses-pre" / "anat" / "sub-01_ses-pre_T1w.json"
    pane.set_badges({target: "err"}, {target: (3, 1)})
    pane._populate(dataset)
    pane._tree.expandAll()
    if pane._last_badges:
        pane._apply_badge_map(pane._last_badges, pane._last_counts)
    assert _find(pane, target.name).data(0, ISSUE_ROLE) == (3, 1)


def test_a_mirrored_finding_is_not_counted_twice(qtbot, dataset: Path) -> None:
    """bidsval attaches a metadata finding to the data file; the adapter
    mirrors it onto the editable sidecar. Counting both doubles every number
    in the tree."""
    from bidsmgr.editor.types import (
        FileVerdict,
        Issue,
        Severity,
        ValidationReport,
    )
    from bidsmgr.gui.editor_panel import EditorPanel

    panel = EditorPanel()
    qtbot.addWidget(panel)
    panel._set_root(dataset, persist=False)
    rel = Path("sub-01/ses-pre/anat/sub-01_ses-pre_T1w.nii.gz")
    real = Issue(
        severity=Severity.WARN, rule_id="SIDECAR_KEY_RECOMMENDED",
        message="missing", field="InstitutionName",
    )
    mirrored = Issue(
        severity=Severity.WARN, rule_id="SIDECAR_KEY_RECOMMENDED",
        message="missing", field="InstitutionName", mirrored=True,
    )
    panel._stamp_tree_badges(ValidationReport(
        bids_root=dataset,
        files=[FileVerdict(
            path=rel, severity=Severity.WARN, issues=[real, mirrored],
        )],
    ))
    item = _find(panel._tree_pane, rel.name)
    assert item is not None
    assert item.data(0, ISSUE_ROLE) == (0, 1)


# ---------------------------------------------------------------------------
# Font scaling, which is what was actually broken
# ---------------------------------------------------------------------------


def _option(width: int = 320) -> QStyleOptionViewItem:
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, width, 20)
    return option


class _FakeIndex:
    """Just enough of a model index for the delegate's measuring code."""

    def __init__(self, data: dict[int, object]) -> None:
        self._data = data

    def data(self, role):
        return self._data.get(role)


def test_the_reserved_width_grows_with_the_font_scale(qtbot) -> None:
    """The regression: sizes were read from ``pointSizeF()`` on a font set in
    pixels, which is -1, so everything clamped to a fixed size and the
    setting did nothing."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    manager = ThemeManager(app)
    delegate = BidsTreeDelegate()
    index = _FakeIndex({ISSUE_ROLE: (12, 34), COUNT_ROLE: "3 ses, 47 files"})

    manager.set_font_scale(1.0)
    manager.apply("dark")
    small = delegate._reserved_width(_option(), index)

    manager.set_font_scale(1.8)
    manager.apply("dark")
    large = delegate._reserved_width(_option(), index)

    manager.set_font_scale(1.0)
    manager.apply("dark")

    assert small > 0
    assert large > small, "the chips must grow with the UI font"


def test_a_row_with_nothing_to_show_reserves_nothing(qtbot) -> None:
    delegate = BidsTreeDelegate()
    assert delegate._reserved_width(_option(), _FakeIndex({})) == 0


def test_the_name_is_elided_before_it_reaches_the_counts(
    qtbot, dataset: Path,
) -> None:
    """Qt elides to the rect it is given, so without reserving the chips'
    width a long filename is drawn straight through them."""
    delegate = BidsTreeDelegate()
    pane = _pane(qtbot, dataset)
    target = dataset / "sub-01" / "ses-pre" / "anat" / "sub-01_ses-pre_T1w.json"
    pane.set_badges({target: "err"}, {target: (2, 5)})
    item = _find(pane, target.name)
    index = pane._tree.indexFromItem(item, 0)
    assert index.isValid()

    option = QStyleOptionViewItem()
    # Narrow enough that the name plus the two count pills cannot both fit.
    option.rect = QRect(0, 0, 170, 20)
    delegate.initStyleOption(option, index)
    assert option.text != target.name, "a long name must be shortened"
    assert option.text.endswith("…")

    reserved = delegate._reserved_width(option, index)
    assert option.fontMetrics.horizontalAdvance(option.text) <= (
        delegate._text_width(option) - reserved
    )


def test_a_short_name_is_left_alone(qtbot, dataset: Path) -> None:
    delegate = BidsTreeDelegate()
    pane = _pane(qtbot, dataset)
    item = _find(pane, "anat")
    index = pane._tree.indexFromItem(item, 0)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 600, 20)
    delegate.initStyleOption(option, index)
    assert option.text == "anat"


def test_the_row_is_tall_enough_for_the_pills(qtbot, dataset: Path) -> None:
    delegate = BidsTreeDelegate()
    pane = _pane(qtbot, dataset)
    item = _find(pane, "anat")
    index = pane._tree.indexFromItem(item, 0)
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 320, 4)
    assert delegate.sizeHint(option, index).height() >= 20


def test_the_delegate_survives_a_malformed_count(qtbot) -> None:
    """Nothing should be able to make the tree fail to paint."""
    delegate = BidsTreeDelegate()
    for bad in ("nonsense", (1,), (None, None), []):
        assert delegate._reserved_width(
            _option(), _FakeIndex({ISSUE_ROLE: bad}),
        ) >= 0


def test_an_invalid_index_does_not_crash_the_option(qtbot) -> None:
    delegate = BidsTreeDelegate()
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 320, 20)
    delegate.initStyleOption(option, QModelIndex())

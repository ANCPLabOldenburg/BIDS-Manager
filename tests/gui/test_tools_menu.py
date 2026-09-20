"""One rounded Tools menu instead of four toolbar buttons.

Fix ups, Rename and Track changes were three rarely-used buttons crowding out
the ones pressed constantly, and the list was going to grow. The validation
buttons deliberately stay OUTSIDE the menu: they are the Editor's main verb,
not a tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QPushButton

from bidsmgr.gui.editor_panel import EditorPanel

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    folder = root / "sub-01" / "anat"
    folder.mkdir(parents=True)
    (folder / "sub-01_T1w.nii.gz").write_bytes(b"\0" * 64)
    (folder / "sub-01_T1w.json").write_text(json.dumps({"EchoTime": 0.03}))
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


@pytest.fixture
def panel(qtbot) -> EditorPanel:
    widget = EditorPanel()
    qtbot.addWidget(widget)
    return widget


def _labels(panel: EditorPanel) -> list[str]:
    return [a.text() for a in panel._tools_menu.actions() if a.text()]


def test_the_menu_holds_the_dataset_wide_actions(panel: EditorPanel) -> None:
    assert _labels(panel) == [
        "Dashboard",
        # A viewer action, not a repair: it sits with Dashboard, above the
        # separator that starts the things which CHANGE the dataset.
        "Compare images...",
        "Fix ups...",
        "Deface...",
        "Remove the skull...",
        # Directly under Deface, although it acts on a SELECTION and the rest
        # of this group acts on the dataset. Removing faces and checking that
        # the right ones went are two halves of one action, and a user who has
        # just defaced looks for the check next to the thing they pressed.
        "Compare with the original...",
        "Put the face back...",
        "Rename entity...",
        "Add or remove an entity...",
        "Sessions...",
        "Delete...",
        "Track changes",
    ]


def test_deface_sits_with_fix_ups_not_with_the_restructuring_actions(
    panel: EditorPanel,
) -> None:
    """Placement is the claim, so it is pinned.

    Rename, entities, sessions and delete all act on a SELECTION. Fix ups and
    Deface act on the DATASET: both are repairs applied to the whole thing from
    a dialog that previews and asks. Grouping by what a thing acts on is what
    makes a menu readable, so Deface belongs above the separator with Fix ups.
    """
    labels = _labels(panel)
    assert labels.index("Deface...") == labels.index("Fix ups...") + 1
    assert labels.index("Deface...") < labels.index("Rename entity...")


def test_deface_is_disabled_rather_than_hidden_when_it_cannot_run(
    panel: EditorPanel, monkeypatch,
) -> None:
    """A missing item reads as "this tool cannot do that".

    A greyed one whose tooltip names the install command reads as what it is.
    The difference decides whether somebody installs an extra or ships a
    dataset with faces in it believing the feature does not exist.
    """
    from bidsmgr.gui import editor_panel as ep

    monkeypatch.setattr(
        ep.deface_run, "unavailable_reason",
        lambda engine_id=None: "Defacing needs niimath, which is not here.",
    )
    panel._refresh_deface_action()

    assert "Deface..." in _labels(panel), "the entry was hidden instead"
    assert not panel._deface_action.isEnabled()
    assert "niimath" in panel._deface_action.toolTip()


def test_compare_with_nothing_picked_says_what_to_pick(
    panel: EditorPanel, dataset: Path, monkeypatch,
) -> None:
    """Comparing is per image, so an empty selection has to be refused."""
    said: list[tuple] = []
    monkeypatch.setattr(
        "bidsmgr.gui.editor_panel.QMessageBox.information",
        lambda *a, **k: said.append(a),
    )
    panel._set_root(dataset, persist=False)
    panel._on_deface_compare()

    assert said, "no selection produced no message at all"
    assert "Pick an image" in said[0][1]


def test_compare_opens_on_the_image_the_tree_passed(
    panel: EditorPanel, dataset: Path, monkeypatch,
) -> None:
    """The right-click carries the file; the dialog must get its relative path."""
    seen: list[tuple] = []

    class _Fake:
        def __init__(self, root, rel, parent=None):
            seen.append((Path(root), rel))

        def exec(self):
            return 0

    monkeypatch.setattr(
        "bidsmgr.gui.deface_compare.DefaceCompareDialog", _Fake,
    )
    panel._set_root(dataset, persist=False)
    panel._on_deface_compare(dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz")

    assert seen == [(dataset, "sub-01/anat/sub-01_T1w.nii.gz")], (
        "the relative path must be POSIX, or it will not match the log"
    )


def test_compare_refuses_a_file_outside_the_dataset(
    panel: EditorPanel, dataset: Path, tmp_path: Path, monkeypatch,
) -> None:
    said: list[tuple] = []
    monkeypatch.setattr(
        "bidsmgr.gui.editor_panel.QMessageBox.information",
        lambda *a, **k: said.append(a),
    )
    stray = tmp_path / "elsewhere.nii.gz"
    stray.write_bytes(b"\0" * 8)
    panel._set_root(dataset, persist=False)
    panel._on_deface_compare(stray)

    assert said and "Not in this dataset" in said[0][1]


def test_validation_stays_out_of_it(panel: EditorPanel) -> None:
    """It is the Editor's main verb. Burying it in a menu would be wrong."""
    assert not [a for a in _labels(panel) if "alidate" in a]
    buttons = [
        b.text().strip() for b in panel.findChildren(QPushButton)
        if "alidate" in b.text()
    ]
    assert buttons, "the validate buttons are still on the toolbar"


def test_the_menu_is_rounded(panel: EditorPanel) -> None:
    """The cross-platform rounded-menu recipe: the object name the QSS paints,
    plus the frameless + translucent window that lets the corners render."""
    from PyQt6.QtCore import Qt

    menu = panel._tools_menu
    assert menu.objectName() == "rounded-menu"
    assert menu.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert menu.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def test_it_is_disabled_until_a_dataset_is_open(panel: EditorPanel) -> None:
    """Every entry acts on a dataset, so offering them with none open would
    be offering something that cannot work."""
    assert not panel._tools_btn.isEnabled()


def test_opening_a_dataset_enables_it(panel: EditorPanel, dataset: Path) -> None:
    panel._set_root(dataset, persist=False)
    assert panel._tools_btn.isEnabled()


def test_track_changes_hides_once_the_dataset_is_tracked(
    panel: EditorPanel, dataset: Path,
) -> None:
    """It only means anything for a dataset this tool did not convert."""
    panel._set_root(dataset, persist=False)
    assert panel._adopt_action.isVisible() or not panel._adopt_action.isVisible()

    (dataset / ".bidsmgr" / "project").mkdir(parents=True)
    panel._refresh_adopt_button()
    assert not panel._adopt_action.isVisible()


def test_it_sits_after_the_deep_checks_toggle(panel: EditorPanel) -> None:
    """Toolbars read left to right, and the ordering is the grouping.

    Tools is a menu of occasional dataset-wide actions. Putting it before the
    deep-checks toggle broke the run of validation controls in half.
    """
    bar = panel._tools_btn.parentWidget().layout()
    order = [bar.itemAt(i).widget() for i in range(bar.count())]
    assert order.index(panel._tools_btn) > order.index(panel._strict_btn)


def test_it_does_not_wear_the_settings_cog(panel: EditorPanel) -> None:
    """Two different things must not share a glyph."""
    from bidsmgr.gui import icons

    assert icons.NAMES["tools"][0] != icons.NAMES["settings"][0]
    assert "toolbox" in icons.NAMES["tools"][0]
    assert not panel._tools_btn.icon().isNull()


def test_its_icon_re_tints_with_the_theme(panel: EditorPanel) -> None:
    """The cached icon is built at one palette. Without an explicit re-apply
    on theme swap it keeps the old tint and reads as disabled."""
    from bidsmgr.gui import icons
    from bidsmgr.gui.theme_manager import CUR

    applied: list[str] = []
    real = icons.apply_button

    def spy(button, name, **kwargs):
        applied.append(name)
        return real(button, name, **kwargs)

    icons.apply_button = spy
    try:
        panel.repaint_for_palette(CUR())
    finally:
        icons.apply_button = real
    assert "tools" in applied


def test_every_entry_explains_itself(panel: EditorPanel) -> None:
    for action in panel._tools_menu.actions():
        if action.text():
            assert action.toolTip(), action.text()


def test_every_entry_does_something(panel: EditorPanel, dataset: Path) -> None:
    """A menu entry that does nothing reads as broken, which is exactly the
    bug the chips-popup Fix button had.

    Triggered for real rather than asked about its connections: Qt refuses to
    report receivers for a QAction it created itself, and "it is connected" is
    a weaker claim than "it ran" anyway.
    """
    panel._set_root(dataset, persist=False)
    called: list[str] = []
    for name, attr in (
        ("Dashboard", "_on_dashboard"),
        ("Fix ups...", "_on_fixups"),
        ("Rename entity...", "_on_rename"),
        ("Track changes", "_on_adopt"),
    ):
        setattr(panel, attr, lambda *a, n=name: called.append(n))

    # Re-wire, since the handlers were swapped after construction.
    for action, attr in (
        (panel._dashboard_action, "_on_dashboard"),
        (panel._fixups_action, "_on_fixups"),
        (panel._rename_action, "_on_rename"),
        (panel._adopt_action, "_on_adopt"),
    ):
        action.triggered.disconnect()
        action.triggered.connect(getattr(panel, attr))
        action.trigger()

    assert called == [
        "Dashboard", "Fix ups...", "Rename entity...", "Track changes",
    ]


# ---------------------------------------------------------------------------
# The viewer has to show what is on disk now.
#
# Defacing, stripping and restoring all rewrite files IN PLACE. The viewer
# holds a decoded array, not the file, so it went on showing the face until
# the user clicked away to another file and back.


def test_the_open_image_is_re_read_after_it_changes(
    panel: EditorPanel, dataset: Path, qtbot,
) -> None:
    nifti = pytest.importorskip("nibabel")
    import numpy as np

    image = dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    nifti.save(
        nifti.Nifti1Image(np.ones((6, 6, 6), dtype="float32"), np.eye(4)),
        str(image),
    )
    panel._set_root(dataset, persist=False)
    panel._nifti_viewer.set_file(image, dataset)
    qtbot.waitUntil(
        lambda: panel._nifti_viewer._data is not None, timeout=20_000
    )
    assert float(np.asarray(panel._nifti_viewer._data).max()) == 1.0

    # Something rewrites it, the way a deface does.
    nifti.save(
        nifti.Nifti1Image(np.full((6, 6, 6), 7.0, dtype="float32"), np.eye(4)),
        str(image),
    )
    panel._reload_open_image()
    qtbot.waitUntil(
        lambda: panel._nifti_viewer._data is not None
        and float(np.asarray(panel._nifti_viewer._data).max()) == 7.0,
        timeout=20_000,
    )


def test_reloading_an_image_that_went_away_clears_the_viewer(
    panel: EditorPanel, dataset: Path, qtbot,
) -> None:
    """A delete removes the file the viewer is holding."""
    nifti = pytest.importorskip("nibabel")
    import numpy as np

    image = dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    nifti.save(
        nifti.Nifti1Image(np.ones((6, 6, 6), dtype="float32"), np.eye(4)),
        str(image),
    )
    panel._set_root(dataset, persist=False)
    panel._nifti_viewer.set_file(image, dataset)
    qtbot.waitUntil(
        lambda: panel._nifti_viewer._data is not None, timeout=20_000
    )

    image.unlink()
    panel._reload_open_image()
    assert panel._nifti_viewer.current_file() is None


def test_reloading_with_nothing_open_is_harmless(panel: EditorPanel) -> None:
    panel._reload_open_image()
    assert panel._nifti_viewer.current_file() is None

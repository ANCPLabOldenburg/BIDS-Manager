"""Checkboxes have to follow the palette, in both themes.

Unstyled, Qt falls back to the platform control, and the platform does not
know about our palette: on macOS a plain unchecked QCheckBox paints a light
grey block on the dark background, which reads as disabled, or as already
ticked. Two checkable trees were styled by object name and the rest were
not, so the ones that were missed stood out beside the ones that were.

This is a stylesheet test rather than a pixel test on purpose. Rendering and
comparing images would pin the exact colours, which are allowed to change;
what must not change is that a rule EXISTS for every state a user can put a
checkbox in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

pytestmark = pytest.mark.gui

QSS = Path(__file__).resolve().parents[2] / "bidsmgr" / "gui" / "theme.qss"


def _qss() -> str:
    return QSS.read_text(encoding="utf-8")


@pytest.mark.parametrize("selector", [
    "QCheckBox::indicator",
    "QCheckBox::indicator:checked",
    "QCheckBox::indicator:indeterminate",
    "QCheckBox::indicator:disabled",
    "QCheckBox::indicator:hover",
])
def test_every_checkbox_state_is_styled(selector):
    assert selector in _qss(), (
        f"{selector} has no rule, so that state falls back to the platform "
        "control and stops following the theme"
    )


def test_radio_buttons_are_styled_too_and_are_round():
    text = _qss()
    assert "QRadioButton::indicator" in text
    assert "QRadioButton::indicator { border-radius: 7px; }" in text, (
        "a radio button drawn as a square is a checkbox as far as the user "
        "is concerned"
    )


def test_the_indicator_colours_come_from_the_palette():
    """Hard-coded hex would be right in one theme and wrong in the other."""
    text = _qss()
    start = text.index("QCheckBox::indicator")
    block = text[start:text.index("/* ---------- Dropdowns", start)]
    assert "$accent" in block and "$border" in block and "$bg" in block
    # A literal colour in here is the bug this whole rule exists to fix.
    assert "#" not in block.replace("/*", "").split("*/")[-1], (
        "a literal colour in the indicator rules will be wrong in one theme"
    )


@pytest.mark.parametrize("module,attr", [
    ("bidsmgr.gui.widgets.move_preview", "MovePreviewTree"),
])
def test_the_shared_preview_tree_keeps_the_styled_name(module, attr):
    import importlib

    source = Path(
        importlib.import_module(module).__file__
    ).read_text(encoding="utf-8")
    assert 'setObjectName("check-tree")' in source


def test_the_deface_previews_use_the_same_name_as_the_others(qtbot, tmp_path):
    """They are the same kind of widget and were the odd ones out."""
    import json
    import shutil

    from bidsmgr.deface.engines import TEMPLATE
    from bidsmgr.gui.deface_dialog import DefaceDialog
    from bidsmgr.gui.deface_revert_dialog import DefaceRevertDialog

    root = tmp_path / "ds"
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "t", "BIDSVersion": "1.11.1"})
    )
    shutil.copyfile(TEMPLATE, root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")

    for dialog in (DefaceDialog(root), DefaceRevertDialog(root)):
        qtbot.addWidget(dialog)
        assert dialog._preview.objectName() == "check-tree"
        dialog.close()


def test_the_old_name_is_gone_rather_than_aliased():
    """One name for one thing. An alias is a second thing to keep in step."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    found = subprocess.run(
        ["grep", "-rn", "rename-preview", str(root / "bidsmgr")],
        capture_output=True, text=True,
    )
    assert not found.stdout.strip(), found.stdout

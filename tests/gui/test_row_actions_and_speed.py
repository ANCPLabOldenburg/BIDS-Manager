"""What a row offers from the table's right-click menu (the Properties
actions, one set of handlers), how quickly Properties follows the
selection, and the native colour scheme following the theme."""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

pytestmark = pytest.mark.gui


def _row(uid: str, datatype: str, suffix: str, name: str, **extra) -> dict:
    base = {"participant_id": "sub-001", "session": "", "include": 1, "modality": "mri",
            "datatype": datatype, "bids_guess_suffix": suffix, "bids_name": name,
            "bids_guess_confidence": "0.9", "bids_guess_skip": False, "issues": "",
            "sequence": name, "series_uid": uid, "dataset": "ds", "source_file": "",
            "source_folder": "s1", "acq_time": "10:00:00", "fieldmap_targets": "",
            "entities": json.dumps({"subject": "001"})}
    base.update(extra)
    return base


def _panel(qtbot, rows: list[dict]):
    from bidsmgr.gui.converter_panel import ConverterPanel

    panel = ConverterPanel()
    qtbot.addWidget(panel)
    panel.load_inventory(pd.DataFrame(rows))
    return panel


def _texts(menu) -> list[str]:
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def _submenu(menu, title: str):
    return next(a.menu() for a in menu.actions() if a.menu() is not None and a.text() == title)


# ---------------------------------------------------------------------------
# The table's right-click menu
# ---------------------------------------------------------------------------


def test_a_fieldmap_row_offers_its_runs(qtbot) -> None:
    from bidsmgr.gui.viz.menus import popup_menu

    panel = _panel(qtbot, [
        _row("f1", "fmap", "phasediff", "sub-001_acq-a_phasediff", acq_time="10:00:00"),
        _row("b1", "func", "bold", "sub-001_task-rest_bold", acq_time="10:05:00",
             entities=json.dumps({"subject": "001", "task": "rest"})),
        _row("b2", "func", "bold", "sub-001_task-two_bold", acq_time="10:15:00",
             entities=json.dumps({"subject": "001", "task": "two"})),
    ])
    menu = popup_menu(panel)
    qtbot.addWidget(menu)
    panel._properties.add_row_actions(menu, 0)
    assert "Preview the image..." in _texts(menu)
    runs = _submenu(menu, "Corrects these runs")
    checks = {a.text(): a.isChecked() for a in runs.actions() if a.isCheckable()}
    assert checks == {"sub-001_task-rest_bold": True, "sub-001_task-two_bold": True}
    next(a for a in runs.actions() if a.text() == "sub-001_task-two_bold").setChecked(False)
    qtbot.wait(20)
    model = panel._model
    assert json.loads(model.dataframe().at[0, "fieldmap_targets"]) == ["b1"]
    # The same handler as the Properties checkboxes (one Undo there, with a
    # project: test_undo_groups_and_rules).


def test_a_classified_row_offers_other_names(qtbot) -> None:
    from bidsmgr.gui.viz.menus import popup_menu

    evidence = {"chosen": "fmap/epi", "other": "func/bold", "decided_by": "parameters",
                "check": True, "note": "", "for": [], "runner_up": [], "entities": {}}
    panel = _panel(qtbot, [_row(
        "r1", "fmap", "epi", "sub-001_dir-AP_epi",
        classification_confidence="0.70",
        classification_alternatives=json.dumps([{"name": "func/sbref", "p": 0.2}]),
        classification_evidence=json.dumps(evidence))])
    menu = popup_menu(panel)
    qtbot.addWidget(menu)
    panel._properties.add_row_actions(menu, 0)
    names = _submenu(menu, "Name it instead")
    labels = [a.text() for a in names.actions()]
    assert labels == ["func/sbref   (0.20)", "func/bold   (dcm2niix)"]
    names.actions()[1].trigger()
    qtbot.wait(20)
    assert panel._model.datatype_suffix(0) == ("func", "bold")


def test_an_eye_tracker_row_offers_runs_to_pair_with(qtbot) -> None:
    from bidsmgr.gui.viz.menus import popup_menu

    panel = _panel(qtbot, [
        _row("", "beh", "physio", "sub-002_task-x_recording-eye1_physio", modality="eyetrack",
             participant_id="sub-002", source_file="subj/rec.edf",
             entities=json.dumps({"subject": "002", "task": "x"})),
        _row("b1", "func", "bold", "sub-001_task-rest_bold",
             entities=json.dumps({"subject": "001", "task": "rest"})),
    ])
    menu = popup_menu(panel)
    qtbot.addWidget(menu)
    panel._properties.add_row_actions(menu, 0)
    assert "Preview the recording..." in _texts(menu)
    pair = _submenu(menu, "Pair with a run")
    (choice,) = pair.actions()
    assert "task-rest" in choice.text()
    choice.trigger()
    qtbot.wait(20)
    assert panel._model.entities(0)["subject"] == "001"
    assert panel._model.entities(0)["task"] == "rest"


def test_an_eeg_row_offers_its_spectrum_and_says_when_it_cannot(qtbot) -> None:
    from bidsmgr.gui.viz.menus import popup_menu

    panel = _panel(qtbot, [_row("", "eeg", "eeg", "sub-001_task-rest_eeg", modality="eeg",
                                source_file="nowhere/rec.edf",
                                entities=json.dumps({"subject": "001", "task": "rest"}))])
    menu = popup_menu(panel)
    qtbot.addWidget(menu)
    panel._properties.add_row_actions(menu, 0)
    psd = next(a for a in menu.actions() if a.text() == "Compute the power spectrum")
    assert not psd.isEnabled() and "not found" in psd.toolTip()


# ---------------------------------------------------------------------------
# Properties keeps up with the table
# ---------------------------------------------------------------------------


def test_a_burst_of_selections_renders_once(qtbot, monkeypatch) -> None:
    panel = _panel(qtbot, [_row(f"u{i}", "anat", "T1w", f"sub-001_run-{i}_T1w")
                           for i in range(1, 6)])
    props = panel._properties
    rendered = []
    original = props._render_for_row
    monkeypatch.setattr(props, "_render_for_row",
                        lambda row: (rendered.append(row), original(row)))
    for row in range(5):
        panel._jump_to_row(row)
    assert rendered == [], "nothing rendered while the selection moves"
    assert props._header.text().lower().endswith("row 5 selected"), "the header follows"
    qtbot.waitUntil(lambda: rendered == [4], timeout=1000)


def test_the_sidecar_form_is_built_when_opened(qtbot) -> None:
    from PyQt6.QtWidgets import QLineEdit

    from bidsmgr.gui.widgets.template_form import CollapsibleSection

    panel = _panel(qtbot, [_row("u1", "anat", "T1w", "sub-001_T1w")])
    props = panel._properties
    props.set_selected_row(0)
    box = next(b for b in props._body.findChildren(CollapsibleSection)
               if b._title_text == "Sidecar fields")
    assert not box.is_realised()
    before = len(props._body.findChildren(QLineEdit))
    box.set_expanded(True)
    assert box.is_realised() and len(props._body.findChildren(QLineEdit)) > before


def test_the_links_plan_is_computed_once_per_table_state(qtbot, monkeypatch) -> None:
    from bidsmgr.fixups import intended_for

    rows = [_row("f1", "fmap", "phasediff", "sub-001_phasediff")]
    rows += [_row(f"b{i}", "func", "bold", f"sub-001_task-t{i}_bold",
                  entities=json.dumps({"subject": "001", "task": f"t{i}"}))
             for i in range(200)]
    panel = _panel(qtbot, rows)
    calls = []
    real = intended_for.plan_links
    monkeypatch.setattr(intended_for, "plan_links", lambda *a, **k: (calls.append(1),
                                                                     real(*a, **k))[1])
    props = panel._properties
    for _ in range(3):
        props.set_selected_row(0)
    assert len(calls) == 1, "cached until the table changes"
    panel._model.set_fieldmap_targets(0, ["b1"])
    props.set_selected_row(0)
    assert len(calls) == 2


def test_showing_a_row_does_not_import_the_scanner(tmp_path) -> None:
    """The first click imported the scan command (mne, joblib, pydicom) for
    one constant: 180 of its 250 ms."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import os, sys\n"
        "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        "from PyQt6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "import pandas as pd\n"
        "from bidsmgr.gui.models.inventory import InventoryTableModel\n"
        "from bidsmgr.gui.properties_panel import PropertiesPanel\n"
        "row = {'participant_id': 'sub-001', 'include': 1, 'modality': 'mri',\n"
        "       'datatype': 'anat', 'bids_guess_suffix': 'T1w', 'bids_name': 'sub-001_T1w',\n"
        "       'series_uid': 'u1', 'source_folder': 's', 'issues': '', 'session': ''}\n"
        "panel = PropertiesPanel(); model = InventoryTableModel(pd.DataFrame([row]))\n"
        "panel.bind_model(model); panel.set_selected_row(0)\n"
        "print('bidsmgr.cli.scan' in sys.modules, 'mne' in sys.modules)\n"
        "sys.stdout.flush(); os._exit(0)\n")
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                         timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "False False", out.stderr[-500:]


# ---------------------------------------------------------------------------
# The native colour scheme
# ---------------------------------------------------------------------------


def test_the_native_scheme_follows_the_theme(qapp, monkeypatch) -> None:
    """On a Mac in dark mode every window of a light theme had a black title
    bar and frame: the platform was never told the theme's scheme."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QStyleHints

    from bidsmgr.gui.theme_manager import ThemeManager

    seen = []
    monkeypatch.setattr(QStyleHints, "setColorScheme",
                        lambda self, scheme: seen.append(scheme), raising=False)
    manager = ThemeManager(qapp)
    manager.apply("porcelain")
    manager.apply("graphite")
    assert seen[-2:] == [Qt.ColorScheme.Light, Qt.ColorScheme.Dark]


def test_the_parameters_own_name_is_offered_when_another_decided(qtbot) -> None:
    from bidsmgr.gui.viz.menus import popup_menu

    evidence = {"chosen": "func/sbref", "ours": "fmap/epi", "other": "",
                "decided_by": "names", "check": True, "note": "", "for": [],
                "runner_up": [], "entities": {}}
    panel = _panel(qtbot, [_row(
        "r1", "func", "sbref", "sub-001_task-x_sbref",
        classification_confidence="0.78",
        classification_alternatives=json.dumps([{"name": "func/sbref", "p": 0.2}]),
        classification_evidence=json.dumps(evidence))])
    menu = popup_menu(panel)
    qtbot.addWidget(menu)
    panel._properties.add_row_actions(menu, 0)
    names = _submenu(menu, "Name it instead")
    assert [a.text() for a in names.actions()] == ["fmap/epi   (0.78)"]

"""The parameter classifier in the GUI: the confidence column, Why this name
in Properties, the Settings page, and naming the open table again."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bidsmgr.classifier.acquisition import cache
from bidsmgr.classifier.acquisition.features import SeriesFeatures
from bidsmgr.gui.app_settings import AppSettings

pytestmark = pytest.mark.gui

_EVIDENCE = {"chosen": "func/bold", "ours": "func/bold", "other": "func/bold",
             "decided_by": "agreement", "check": False, "note": "",
             "for": [{"for": "func/bold", "source": "parameters",
                      "text": "a time series of 300 volumes", "points": 3.0},
                     {"for": "func/bold", "source": "names", "text": "the name says BOLD",
                      "points": 2.0}],
             "runner_up": []}


def _rows() -> list[dict]:
    common = {"participant_id": "sub-001", "session": "", "include": 1, "issues": "",
              "modality": "mri", "bids_guess_confidence": "0.85", "bids_guess_skip": False,
              "dataset": "ds", "source_file": "", "source_folder": "s"}
    return [
        {**common, "series_uid": "1.1", "sequence": "bold_rest", "acq_time": "100000",
         "datatype": "func", "bids_guess_suffix": "bold", "task": "rest", "run": "",
         "entities": json.dumps({"subject": "001", "task": "rest"}),
         "bids_name": "sub-001_task-rest_bold",
         "classification_confidence": "0.970",
         "classification_alternatives": json.dumps([{"name": "fmap/epi", "p": 0.12}]),
         "classification_evidence": json.dumps(_EVIDENCE)},
        {**common, "series_uid": "1.2", "sequence": "bold_rest_rev", "acq_time": "100500",
         "datatype": "func", "bids_guess_suffix": "bold", "task": "rest", "run": "2",
         "entities": json.dumps({"subject": "001", "task": "rest", "run": "2"}),
         "bids_name": "sub-001_task-rest_run-2_bold",
         "classification_confidence": "", "classification_alternatives": "",
         "classification_evidence": ""},
    ]


def _panel(qtbot, tmp_path: Path):
    from bidsmgr.cli.create import open_or_create_workspace
    from bidsmgr.gui.converter_panel import ConverterPanel
    from bidsmgr.project import Project, ScanImported, workspace

    root = tmp_path / "ds"
    proj = open_or_create_workspace(root)
    vdir = workspace.allocate_version_dir(root, "raw")
    vproj = Project.create(vdir, name=root.name)
    tsv = workspace.version_inventory(vdir)
    pd.DataFrame(_rows()).to_csv(tsv, sep="\t", index=False)
    base = dict(rows=64.0, columns=64.0, n_slices=30, n_orientations=1, acquisition_type="2D",
                is_epi=True, is_gre=True, echo_times=[30.0], repetition_time=0.8, multiband=4)
    cache.write(tsv, {"1.1": SeriesFeatures(description="bold_rest", n_volumes=300,
                                            pe_dir="PA", **base),
                      "1.2": SeriesFeatures(description="bold_rest_rev", n_volumes=4,
                                            pe_dir="AP", **base)})
    workspace.write_version_meta(vdir, source_label="raw", raw_root=str(tmp_path / "raw"),
                                 status="curating")
    vproj.append(ScanImported(inventory_tsv=str(tsv), row_ids=("1.1", "1.2"), raw_root=None))
    panel = ConverterPanel(project=proj)
    qtbot.addWidget(panel)
    panel.set_project(proj, root)
    return panel


def _conf_column(model) -> int:
    return next(i for i, c in enumerate(model.COLUMNS) if c.key == "conf")


def test_the_confidence_column_says_who_named_it(qtbot, tmp_path) -> None:
    from PyQt6.QtCore import Qt

    panel = _panel(qtbot, tmp_path)
    model = panel._model
    col = _conf_column(model)
    assert model.data(model.index(0, col)) == ".97", "the parameters' confidence"
    assert model.data(model.index(1, col)) == ".85", "else the rule that named it"
    tip = model.data(model.index(0, col), Qt.ItemDataRole.ToolTipRole)
    assert "func/bold (0.97)" in tip and "agree" in tip


def test_why_this_name_lists_the_evidence_and_offers_the_alternatives(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    props = panel._properties
    props.set_selected_row(0)
    assert props.why_lines == ["a time series of 300 volumes", "the name says BOLD"]
    assert "fmap/epi" in props.why_alternatives
    props.why_alternatives["fmap/epi"].click()
    qtbot.wait(20)
    model = panel._model
    assert model.datatype_suffix(0) == ("fmap", "epi")
    panel._on_undo()
    assert panel._model.datatype_suffix(0) == ("func", "bold"), "one Undo"
    # A row the classifier did not score has no such section.
    props.why_box = None
    props.set_selected_row(1)
    assert props.why_box is None


def test_the_open_table_is_named_again_as_one_undo(qtbot, tmp_path) -> None:
    panel = _panel(qtbot, tmp_path)
    changes = panel.reclassify_changes()
    assert {c.index: c.new for c in changes if c.renamed} == {1: "fmap/epi"}
    assert panel.offer_reclassify(ask=False) == 1
    model = panel._model
    assert model.dataframe().iloc[1]["bids_name"] == "sub-001_dir-AP_run-2_epi"
    assert float(model.dataframe().iloc[1]["classification_confidence"]) > 0.8
    panel._on_undo()
    assert panel._model.dataframe().iloc[1]["bids_name"] == "sub-001_task-rest_run-2_bold"


def test_shadow_mode_reports_without_renaming(qtbot, tmp_path) -> None:
    s = AppSettings.load()
    config = s.classifier()
    config.general.mode = "shadow"
    s.classifier_config = config.model_dump_json()
    s.save()
    panel = _panel(qtbot, tmp_path)
    assert panel.offer_reclassify(ask=False) == 0
    assert panel._model.dataframe().iloc[1]["datatype"] == "func"
    assert panel._model.dataframe().iloc[1]["classification_confidence"]


def test_the_settings_page_saves_the_classifier_settings(qtbot) -> None:
    from bidsmgr.gui.delegates.inspection_cells import CONFIDENCE_BANDS
    from bidsmgr.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    page = dlg._classifier_page
    page.control("general.apply_at").setValue(0.9)
    page.control("series.run_min_volumes").setValue(20)
    dlg._on_save()
    config = AppSettings.load().classifier()
    assert config.general.apply_at == pytest.approx(0.9)
    assert config.series.run_min_volumes == 20
    from bidsmgr.gui.converter_panel import ConverterPanel

    panel = ConverterPanel()
    qtbot.addWidget(panel)
    assert CONFIDENCE_BANDS[0] == pytest.approx(0.9), "the column's colours follow"


def test_every_threshold_and_the_rules_are_in_settings(qtbot, tmp_path) -> None:
    """The physical and contrast ranges each have a control, and the
    evidence rules export, load back (only the change is kept) and reset."""
    from bidsmgr.classifier.acquisition import engine
    from bidsmgr.classifier.acquisition.config import ClassifierConfig
    from bidsmgr.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(AppSettings.load())
    qtbot.addWidget(dlg)
    page = dlg._classifier_page
    config = ClassifierConfig()
    for section in ("anatomy", "gradient_echo", "echoes", "series", "contrast"):
        for name in type(getattr(config, section)).model_fields:
            assert page.control(f"{section}.{name}") is not None, f"{section}.{name}"
    page.control("gradient_echo.spoiled_3d_flip_max").setValue(25)
    page.control("echoes.multi_echo_min").setValue(4)
    dsc = page.control("contrast.dsc")
    dsc.setCurrentIndex(dsc.findData("cbv"))
    page.control("contrast.fixed_label").setText("gadovist")
    assert "As shipped" in page.rules_state.text()
    assert not page.reset_button.isEnabled()

    exported = tmp_path / "rules.json"
    page.export_rules(exported)
    book = json.loads(exported.read_text())
    assert book == engine.rulebook()
    book["candidates"]["anat/PDw"]["evidence"][0]["weight"] = 0.5
    exported.write_text(json.dumps(book))
    assert page.load_rules(exported) == []
    assert "1 name changed" in page.rules_state.text()
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"candidates": {"anat/T1w": {"evidence": [
        {"source": "parameters", "feature": "nope", "weight": 1}]}}}))
    assert page.load_rules(broken), "refused, with the reason"
    assert "1 name changed" in page.rules_state.text(), "and nothing was loaded"

    dlg._on_save()
    saved = AppSettings.load().classifier()
    assert saved.gradient_echo.spoiled_3d_flip_max == pytest.approx(25)
    assert saved.echoes.multi_echo_min == 4
    assert (saved.contrast.dsc, saved.contrast.fixed_label) == ("cbv", "gadovist")
    assert set(saved.rulebook["candidates"]) == {"anat/PDw"}

    page.reset_button.click()
    assert "As shipped" in page.rules_state.text()

"""Custom schemas can be selected, persisted and adopted by the editor."""

import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QDialog, QFileDialog, QMessageBox

from bidsmgr import schema
from bidsmgr.editor import restructure
from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.settings_dialog import SettingsDialog
from bidsmgr.vendor.bidsval import schema as bidsval_schema

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def restore_schema():
    previous = schema.active_version()
    yield
    schema.set_active_version(previous)


@pytest.fixture
def custom_schema(tmp_path):
    bundled = Path(bidsval_schema.__file__).parent / "bundled" / "1.11.1.json"
    data = json.loads(bundled.read_text())
    data["objects"]["entities"]["customentity"] = {
        "name": "custom", "display_name": "Custom", "format": "label",
    }
    data["rules"]["entities"].append("customentity")
    path = tmp_path / "custom schema.json"
    path.write_text(json.dumps(data))
    return path


def test_custom_schema_save_reopen_and_editor_switch(qtbot, custom_schema):
    schema.set_active_version("1.11.1")
    assert "custom" not in restructure._short_to_long()
    dlg = SettingsDialog(AppSettings())
    qtbot.addWidget(dlg)
    dlg._validate_schema.setCurrentIndex(dlg._validate_schema.findData("custom"))
    dlg._custom_schema_path.setText(str(custom_schema))
    dlg._describe_bids_version()
    assert not dlg._custom_schema_row.isHidden()
    assert "Cannot load" not in dlg._version_summary.text()
    # Previewing must not activate the schema before Save.
    assert schema.active_version() == "1.11.1"
    dlg._on_save()
    assert dlg.result() == QDialog.DialogCode.Accepted
    assert AppSettings.load().validate_schema_version == str(custom_schema)
    assert schema.active_version() == str(custom_schema)
    assert restructure._short_to_long()["custom"] == "customentity"

    reopened = SettingsDialog(AppSettings.load())
    qtbot.addWidget(reopened)
    assert reopened._validate_schema.currentData() == "custom"
    assert reopened._custom_schema_path.text() == str(custom_schema)
    reopened._validate_schema.setCurrentIndex(0)
    assert f"{len(schema.get_schema('').rules.entities)} entities" in reopened._version_summary.text()
    reopened._on_save()
    assert AppSettings.load().validate_schema_version == ""
    assert "custom" not in restructure._short_to_long()


@pytest.mark.parametrize("kind", ["empty", "missing", "invalid_json", "incomplete"])
def test_invalid_custom_schema_does_not_save(qtbot, tmp_path, monkeypatch, kind):
    settings = AppSettings()
    dlg = SettingsDialog(settings)
    qtbot.addWidget(dlg)
    dlg._validate_schema.setCurrentIndex(dlg._validate_schema.findData("custom"))
    path = tmp_path / "schema.json"
    if kind in {"invalid_json", "incomplete"}:
        path.write_text("broken" if kind == "invalid_json" else "{}")
    dlg._custom_schema_path.setText("" if kind == "empty" else str(path))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    previous = schema.active_version()
    dlg._on_save()
    assert warnings
    assert dlg.result() != QDialog.DialogCode.Accepted
    assert settings.validate_schema_version == ""
    assert schema.active_version() == previous


@pytest.mark.parametrize("directory", [False, True])
def test_schema_browse(qtbot, custom_schema, monkeypatch, directory):
    dlg = SettingsDialog(AppSettings())
    qtbot.addWidget(dlg)
    dlg._validate_schema.setCurrentIndex(dlg._validate_schema.findData("custom"))
    # The picker hands a local path to the same field for files and folders.
    selected = str(custom_schema.parent if directory else custom_schema)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: selected)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (selected, ""))
    dlg._browse_custom_schema(directory)
    assert dlg._custom_schema_path.text() == selected


def test_cancel_does_not_activate_schema(qtbot, custom_schema):
    settings = AppSettings()
    dlg = SettingsDialog(settings)
    qtbot.addWidget(dlg)
    previous = schema.active_version()
    dlg._validate_schema.setCurrentIndex(dlg._validate_schema.findData("custom"))
    dlg._custom_schema_path.setText(str(custom_schema))
    dlg._describe_bids_version()
    dlg.reject()
    assert settings.validate_schema_version == ""
    assert schema.active_version() == previous

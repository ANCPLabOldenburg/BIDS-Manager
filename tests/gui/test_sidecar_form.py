"""Tests for the Editor's center pane — sidecar form (M6 Step 4).

The form loads JSON content **directly from disk** when a file is
clicked, so it works even without a validation report. The report
adds schema-level colour-coding (REQUIRED red, RECOMMENDED amber,
etc.) and surfaces missing required fields when available.

The pane shows the **single file** the user clicked — no peer
auto-discovery, no tab strip.

Theme refresh rebuilds the entire form from scratch so inline
``setStyleSheet`` styles (legend swatches, SidecarRow bars, footer)
always reflect the active palette.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bidsmgr.editor.types import (
    FieldLevel,
    FileVerdict,
    Severity,
    SidecarField,
    ValidationReport,
)
from PyQt6.QtWidgets import QFrame

from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.widgets.bids_tree_pane import PATH_ROLE
from bidsmgr.gui.widgets.sidecar_form_pane import (
    SidecarFormPane,
    find_peer_files,
)


pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bids_root(tmp_path: Path) -> Path:
    root = tmp_path / "Studyname"
    anat = root / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-01_T1w.nii.gz").write_bytes(b"")
    (anat / "sub-01_ses-01_T1w.json").write_text(
        '{"RepetitionTime": 2.0, "EchoTime": 0.03, "Manufacturer": "Siemens"}'
    )
    func = root / "sub-01" / "ses-01" / "func"
    func.mkdir(parents=True)
    (func / "sub-01_ses-01_task-rest_bold.nii.gz").write_bytes(b"")
    (func / "sub-01_ses-01_task-rest_bold.json").write_text(
        '{"RepetitionTime": 2.0, "TaskName": "TODO"}'
    )
    (func / "sub-01_ses-01_task-rest_events.tsv").write_text(
        "onset\tduration\n0\t1\n"
    )
    (root / "dataset_description.json").write_text(
        '{"Name": "Test", "BIDSVersion": "1.10.0"}'
    )
    return root


def _make_verdict(rel_path: Path) -> FileVerdict:
    return FileVerdict(
        path=rel_path,
        severity=Severity.WARN,
        datatype="anat",
        suffix="T1w",
        sidecar_fields=[
            SidecarField(
                level=FieldLevel.REQUIRED,
                name="MagneticFieldStrength",
                value=None,
                present=False,
                value_kind="missing",
            ),
            SidecarField(
                level=FieldLevel.RECOMMENDED,
                name="RepetitionTime",
                value=2.0,
                present=True,
                value_kind="number",
            ),
            SidecarField(
                level=FieldLevel.OPTIONAL,
                name="EchoTime",
                value=0.03,
                present=True,
                value_kind="number",
            ),
            SidecarField(
                level=FieldLevel.DEPRECATED,
                name="OldField",
                value="legacy",
                present=True,
                value_kind="string",
            ),
        ],
    )


def _report_with_t1w_verdict(bids_root: Path) -> ValidationReport:
    rel = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    ).relative_to(bids_root)
    return ValidationReport(
        bids_root=bids_root,
        bids_version="1.10.0",
        severity=Severity.WARN,
        counts={"ok": 0, "warn": 1, "err": 0},
        files=[_make_verdict(rel)],
    )


# ---------------------------------------------------------------------------
# find_peer_files (utility, still exported for future features)
# ---------------------------------------------------------------------------


def test_find_peer_files_groups_same_stem(bids_root: Path) -> None:
    t1w_json = bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    peers = find_peer_files(t1w_json)
    names = [p.name for p in peers]
    assert names == [
        "sub-01_ses-01_T1w.json",
        "sub-01_ses-01_T1w.nii.gz",
    ]


# ---------------------------------------------------------------------------
# SidecarFormPane: validation-independent load
# ---------------------------------------------------------------------------


def test_sidecar_form_starts_empty(qapp) -> None:
    pane = SidecarFormPane()
    assert pane.current_file() is None
    assert pane._rows == []


def test_set_file_renders_json_content_without_report(
    qapp, bids_root: Path,
) -> None:
    """JSON content must load even when no validation report is bound."""
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )

    pane.set_file(json_path, bids_root, None)

    assert pane.current_file() == json_path
    # Three keys in the test JSON.
    assert len(pane._rows) == 3
    # Without a report, every field is OPTIONAL.
    assert all(row._level == "opt" for row in pane._rows)


def test_set_file_detects_todo_placeholders_without_report(
    qapp, bids_root: Path,
) -> None:
    pane = SidecarFormPane()
    bold_json = (
        bids_root / "sub-01" / "ses-01" / "func"
        / "sub-01_ses-01_task-rest_bold.json"
    )
    pane.set_file(bold_json, bids_root, None)
    # Two keys: RepetitionTime (number) and TaskName (TODO placeholder).
    assert len(pane._rows) == 2


def test_set_file_with_verdict_uses_schema_levels(
    qapp, bids_root: Path,
) -> None:
    """When a report is bound the form picks up schema-level coloring."""
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    report = _report_with_t1w_verdict(bids_root)

    pane.set_file(json_path, bids_root, report)

    # Four rows from the verdict, in level order.
    assert len(pane._rows) == 4
    levels = [row._level for row in pane._rows]
    assert levels == ["req", "rec", "opt", "dep"]


def test_set_file_non_json_shows_hint(qapp, bids_root: Path) -> None:
    """Non-JSON files don't get a form — show the helpful empty hint."""
    pane = SidecarFormPane()
    nii_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
    )

    pane.set_file(nii_path, bids_root, None)

    assert pane.current_file() == nii_path
    assert pane._rows == []
    assert pane._empty_hint.isVisibleTo(pane)
    # Hint mentions the JSON sibling rather than a generic
    # "not yet validated" message.
    assert ".json" in pane._empty_hint.text()


def test_set_file_none_clears_form(qapp, bids_root: Path) -> None:
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)
    assert pane._rows

    pane.set_file(None, None, None)
    assert pane.current_file() is None
    assert pane._rows == []


def test_footer_summary_without_report_shows_field_count(
    qapp, bids_root: Path,
) -> None:
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)
    # 3 keys in the JSON.
    text = pane._footer_summary.text()
    assert "3 fields" in text
    assert "not yet validated" in text


def test_footer_summary_with_report_counts_missing(
    qapp, bids_root: Path,
) -> None:
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    report = _report_with_t1w_verdict(bids_root)
    pane.set_file(json_path, bids_root, report)
    text = pane._footer_summary.text()
    assert "4 fields" in text
    assert "1 missing" in text


# ---------------------------------------------------------------------------
# Theme refresh
# ---------------------------------------------------------------------------


def test_sidecar_row_bars_use_qss_object_names(qapp, bids_root: Path) -> None:
    """Theme refresh works because each row's level bar carries a
    QSS-only object name (``sc-bar-req`` / ``sc-bar-rec`` / etc.). The
    theme manager re-applies the global QSS on every toggle and Qt
    restyles automatically — same pattern as ``QPlainTextEdit#dock-log``
    in the converter's log view.
    """
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)

    assert pane._rows
    # Without a report every field is OPTIONAL, so every bar should
    # carry the OPT object name.
    for row in pane._rows:
        assert row._bar.objectName() == "sc-bar-opt"


def test_legend_swatches_use_qss_object_names(qapp) -> None:
    """The legend's four swatches each carry a per-level object name so
    their colors come from the global QSS, not an inline stylesheet."""
    pane = SidecarFormPane()
    swatches = pane._legend.findChildren(QFrame)
    object_names = {sw.objectName() for sw in swatches}
    assert "legend-swatch-req" in object_names
    assert "legend-swatch-rec" in object_names
    assert "legend-swatch-opt" in object_names
    assert "legend-swatch-dep" in object_names


def test_footer_widgets_use_qss_object_names(qapp) -> None:
    """Footer + its two labels are QSS-styled."""
    pane = SidecarFormPane()
    assert pane._footer.objectName() == "sidecar-footer"
    assert pane._footer_path.objectName() == "sidecar-footer-path"
    assert pane._footer_summary.objectName() == "sidecar-footer-summary"


def test_repaint_for_palette_forces_qss_re_evaluation(
    qapp, bids_root: Path, monkeypatch,
) -> None:
    """``repaint_for_palette`` must trigger Qt's unpolish/polish cycle
    on every descendant widget so the global QSS swap actually flows
    through to the pane (Qt sometimes caches per-widget styles and
    doesn't invalidate them when only the token-values inside the
    same rules change). Same fix the converter view relies on for
    custom QSS-driven widgets.

    Bound rows survive the cycle — no widget rebuild needed because the
    QSS-only design carries every palette colour itself.
    """
    pane = SidecarFormPane()
    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    pane.set_file(json_path, bids_root, None)
    rows_before = list(pane._rows)

    # Spy on QStyle.unpolish / polish to confirm the cycle runs.
    unpolished: list = []
    polished: list = []
    real_unpolish = pane.style().unpolish
    real_polish = pane.style().polish

    def _spy_unpolish(w):
        unpolished.append(w)
        return real_unpolish(w)

    def _spy_polish(w):
        polished.append(w)
        return real_polish(w)

    monkeypatch.setattr(pane.style(), "unpolish", _spy_unpolish)
    monkeypatch.setattr(pane.style(), "polish", _spy_polish)

    from bidsmgr.gui.theme_manager import LIGHT
    pane.repaint_for_palette(LIGHT)

    # The pane and at least every SidecarRow (+ their bars) got the cycle.
    assert pane in unpolished and pane in polished
    assert any(w in unpolished for w in rows_before)
    # Same row instances — QSS-only design means no rebuild.
    assert pane._rows == rows_before


# ---------------------------------------------------------------------------
# EditorPanel integration
# ---------------------------------------------------------------------------


def test_tree_click_loads_form_without_validation(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    """Clicking a JSON file in the tree fills the form even when no
    validation has been run yet."""
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)

    tree = panel.tree_pane()._tree
    json_path_str = str(
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    target = None

    def visit(item) -> None:
        nonlocal target
        if item.data(0, PATH_ROLE) == json_path_str:
            target = item
        for i in range(item.childCount()):
            visit(item.child(i))

    for i in range(tree.topLevelItemCount()):
        visit(tree.topLevelItem(i))
    assert target is not None

    tree.setCurrentItem(target)
    qapp.processEvents()

    assert panel._sidecar_form.current_file() == Path(json_path_str)
    # Three rows from the JSON content directly — no validation needed.
    assert len(panel._sidecar_form._rows) == 3


def test_tree_click_only_opens_clicked_file(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    """The center pane shows only the file the user clicked — no
    auto-opened peers (no tab strip, no peer signals).

    Since the NIfTI viewer landed, clicking a ``.nii.gz`` routes to
    :class:`NiftiViewerPane` instead of binding the sidecar form to
    the volume. The "no peer auto-loaded" invariant still holds — the
    sibling JSON stays untouched.
    """
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)

    nii_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
    )
    panel._on_file_selected(nii_path)

    # NIfTI clicks land on the NIfTI viewer; the sidecar form is
    # cleared so its previous binding doesn't leak.
    assert panel._center_stack.currentWidget() is panel._nifti_viewer
    assert panel._sidecar_form.current_file() is None
    # The sibling .json is NOT auto-loaded into the form.
    assert panel._sidecar_form._rows == []
    # And no SidecarFormPane peer-tab API survives.
    assert not hasattr(panel._sidecar_form, "_tab_buttons")


def test_directory_click_clears_form(
    qapp, isolated_settings, bids_root: Path,
) -> None:
    panel = EditorPanel()
    panel._set_root(bids_root, persist=False)

    json_path = (
        bids_root / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
    )
    panel._on_file_selected(json_path)
    assert panel._sidecar_form._rows

    anat_path = bids_root / "sub-01" / "ses-01" / "anat"
    panel._on_file_selected(anat_path)
    assert panel._sidecar_form.current_file() is None
    assert panel._sidecar_form._rows == []

"""The validation display on a large dataset: what made it freeze, fixed.

Measured on 21,663 files and 287,000 findings (1.5.1): the chip dialogs built
a dozen widgets per finding (more than 45 minutes for the warnings), every
lookup of a file's verdict walked the report resolving each path (3.4 s per
click), the badges resolved every path again (4.4 s per report), a one-file
re-validation recounted everything (8.3 s), the whole-dataset view regrouped
on every click. Each fix is pinned here by what it guarantees, not by a time.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt

from bidsmgr.editor import report_index
from bidsmgr.editor.types import FileVerdict, Issue, Severity, ValidationReport

pytestmark = pytest.mark.gui


def _issue(sev: Severity, n: int = 0, *, mirrored: bool = False) -> Issue:
    return Issue(severity=sev, rule_id=f"RULE_{n}", message=f"finding {n}",
                 field=f"Field{n}", mirrored=mirrored)


def _report(root: Path, files: int = 6) -> ValidationReport:
    verdicts = []
    for i in range(files):
        sev = Severity.ERR if i % 3 == 0 else Severity.WARN if i % 3 == 1 else Severity.OK
        issues = [] if sev is Severity.OK else [_issue(sev, i), _issue(Severity.WARN, 100 + i)]
        verdicts.append(FileVerdict(path=Path(f"sub-{i:02d}/anat/sub-{i:02d}_T1w.json"),
                                    severity=sev, issues=issues))
    report = ValidationReport(bids_root=root, files=verdicts)
    errs = sum(1 for v in verdicts for x in v.issues if x.severity is Severity.ERR)
    warns = sum(1 for v in verdicts for x in v.issues if x.severity is Severity.WARN)
    report.counts = {"ok": sum(1 for v in verdicts if v.severity is Severity.OK),
                     "warn": warns, "err": errs}
    report.severity = Severity.ERR
    return report


def _dataset(tmp_path: Path, files: int = 6) -> Path:
    root = tmp_path / "ds"
    for i in range(files):
        f = root / f"sub-{i:02d}" / "anat" / f"sub-{i:02d}_T1w.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}")
    (root / "dataset_description.json").write_text('{"Name": "d", "BIDSVersion": "1.10.0"}')
    return root


# -- one lookup, not a walk ---------------------------------------------------


def test_a_verdict_is_found_under_either_spelling_of_the_root(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    report = _report(root)
    link = tmp_path / "link"
    try:
        link.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here")
    target = report.files[4]
    for base in (root, link):
        assert report_index.find_verdict(report, base, base / target.path) is target
        assert report_index.find_verdict(report, base, link / target.path) is target
    assert report_index.find_verdict(report, root, root / "sub-00") is None, "a folder"
    assert report_index.find_verdict(report, root, tmp_path / "elsewhere.json") is None


def test_a_changed_report_is_looked_up_afresh(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    report = _report(root)
    path = root / report.files[0].path
    assert report_index.find_verdict(report, root, path) is report.files[0]
    replacement = FileVerdict(path=report.files[0].path, severity=Severity.OK)
    report.files[0] = replacement
    report_index.forget(report)
    assert report_index.find_verdict(report, root, path) is replacement


# -- the Editor ---------------------------------------------------------------


@pytest.fixture
def panel(qtbot, tmp_path: Path):
    from bidsmgr.gui.editor_panel import EditorPanel

    root = _dataset(tmp_path)
    ed = EditorPanel()
    qtbot.addWidget(ed)
    ed._set_root(root, persist=False, view_only=False)
    report = _report(root)
    ed._on_report_ready(report, root)
    return ed, root, report


def test_a_merged_file_adjusts_the_counts_as_a_recount_would(panel) -> None:
    ed, root, report = panel
    fresh = FileVerdict(path=report.files[0].path, severity=Severity.OK)
    ed._on_partial_ready([fresh], root, root / fresh.path)
    files = report.files
    assert report.counts == {
        "ok": sum(1 for v in files if v.severity is Severity.OK),
        "warn": sum(1 for v in files for i in v.issues
                    if i.severity is Severity.WARN and not i.mirrored),
        "err": sum(1 for v in files for i in v.issues
                   if i.severity is Severity.ERR and not i.mirrored)}
    assert ed._verdict_for(root / fresh.path) is fresh


def test_a_merged_file_restamps_its_badge_and_its_folders(panel) -> None:
    ed, root, report = panel
    tree = ed._tree_pane
    key = tree._key(str(root / report.files[0].path))
    assert tree._last_badges[key] == "err"
    subject = tree._key(str(root / "sub-00"))
    assert tree._dir_badges[subject] == "err"
    fresh = FileVerdict(path=report.files[0].path, severity=Severity.OK)
    ed._on_partial_ready([fresh], root, root / fresh.path)
    assert tree._last_badges[key] == "ok"
    assert key not in tree._last_counts
    assert tree._dir_badges[subject] == "ok", "the folder follows its only file"
    assert tree._dir_badges[tree._key(str(root))] == "err", "the dataset still has errors"


def test_badges_need_no_path_resolved_per_file(panel, monkeypatch) -> None:
    ed, root, report = panel
    calls = []
    real = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda self, *a, **k: calls.append(1) or real(self))
    ed._stamp_tree_badges(report)
    assert len(calls) <= 2, "the root at most, not every file"


def test_folders_roll_up_the_worst_and_the_sum() -> None:
    from bidsmgr.gui.widgets.bids_tree_pane import _rollup

    sep = os.sep
    root = f"{sep}ds"
    leaves = {f"{root}{sep}sub-01{sep}anat{sep}a.json": "warn",
              f"{root}{sep}sub-01{sep}func{sep}b.json": "err",
              f"{root}{sep}sub-02{sep}anat{sep}c.json": "ok"}
    counts = {f"{root}{sep}sub-01{sep}anat{sep}a.json": (0, 2),
              f"{root}{sep}sub-01{sep}func{sep}b.json": (1, 1)}
    badges, totals = _rollup(leaves, counts)
    assert badges[f"{root}{sep}sub-01"] == "err" and badges[f"{root}{sep}sub-02"] == "ok"
    assert badges[f"{root}{sep}sub-01{sep}anat"] == "warn"
    assert totals[f"{root}{sep}sub-01"] == (1, 3) and totals[root] == (1, 3)


# -- the validation pane ------------------------------------------------------


def test_a_section_lists_fifty_then_offers_the_rest(qtbot, tmp_path: Path) -> None:
    from bidsmgr.gui.widgets.validation_pane import ValidationPane
    from bidsmgr.gui.widgets.val_message import ValMessage

    root = _dataset(tmp_path)
    report = _report(root)
    report.dataset_issues = [_issue(Severity.WARN, n) for n in range(120)]
    pane = ValidationPane()
    qtbot.addWidget(pane)
    pane.set_report_and_file(report, None, root)
    from PyQt6.QtWidgets import QPushButton

    assert len(pane.findChildren(ValMessage)) == 50
    more = next(b for b in pane.findChildren(QPushButton) if b.text() == "Show all 120")
    more.click()
    assert len(pane.findChildren(ValMessage)) == 120


def test_the_whole_dataset_view_is_not_rebuilt_by_a_click(qtbot, tmp_path: Path,
                                                         monkeypatch) -> None:
    from bidsmgr.gui.widgets import validation_pane as vp_mod

    root = _dataset(tmp_path)
    report = _report(root)
    pane = vp_mod.ValidationPane()
    qtbot.addWidget(pane)
    pane.set_report_and_file(report, None, root)
    grouped = []
    real = vp_mod.group_report
    monkeypatch.setattr(vp_mod, "group_report",
                        lambda *a, **k: grouped.append(1) or real(*a, **k))
    pane._on_mode_clicked(1)
    renders = []
    monkeypatch.setattr(pane, "_render", lambda: renders.append(1))
    pane.set_current_file(root / report.files[1].path, root)
    pane.set_current_file(root / report.files[2].path, root)
    assert renders == [] and grouped == [1]


# -- the chip dialogs ---------------------------------------------------------


def test_the_chip_dialog_holds_twenty_thousand_findings_in_one_view(qtbot,
                                                                   tmp_path: Path) -> None:
    from bidsmgr.gui.editor_issues_dialog import EditorIssuesDialog
    from bidsmgr.gui.widgets.val_message import ValMessage

    root = tmp_path / "ds"
    root.mkdir()
    verdicts = [FileVerdict(path=Path(f"sub-{i:04d}/anat/sub-{i:04d}_T1w.json"),
                            severity=Severity.WARN,
                            issues=[_issue(Severity.WARN, n) for n in range(10)])
                for i in range(2000)]
    report = ValidationReport(bids_root=root, files=verdicts)
    start = time.perf_counter()
    dlg = EditorIssuesDialog(report, "warn", root)
    qtbot.addWidget(dlg)
    took = time.perf_counter() - start
    assert dlg._list.view.findings_model().total() == (2000, 20000)
    assert dlg.findChildren(ValMessage) == [], "no widget per finding"
    assert took < 5.0, f"{took:.1f} s"
    dlg._list.set_search("finding 3")
    assert dlg._list.view.findings_model().counts() == (2000, 2000)
    dlg._list.set_search("sub-0007")
    assert dlg._list.view.findings_model().counts() == (1, 10)


def test_a_click_on_a_file_folds_it_and_a_double_click_opens_it(qtbot, tmp_path: Path) -> None:
    from PyQt6.QtTest import QTest

    from bidsmgr.gui.editor_issues_dialog import EditorIssuesDialog

    root = tmp_path / "ds"
    root.mkdir()
    report = ValidationReport(bids_root=root, files=[FileVerdict(
        path=Path("sub-01/anat/sub-01_T1w.json"), severity=Severity.WARN,
        issues=[_issue(Severity.WARN, 1)])])
    dlg = EditorIssuesDialog(report, "warn", root)
    qtbot.addWidget(dlg)
    dlg.show()
    view = dlg._list.view
    group = view.findings_model().index(0, 0)
    assert view.isExpanded(group), "few findings: open"
    rect = view.visualRect(group)
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert not view.isExpanded(group)
    with qtbot.waitSignal(dlg.file_selected, timeout=1000) as sig:
        QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert sig.args[0] == root / "sub-01/anat/sub-01_T1w.json"

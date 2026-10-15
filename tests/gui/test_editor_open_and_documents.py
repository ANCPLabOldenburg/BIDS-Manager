"""The Editor's one Open folder (a project, or a folder open for viewing
that is never written into), the Welcome tab's Open and Recent projects,
switching projects without leaving the tab, and the views of the files
that are not images, recordings or tables: documents, the gradient table,
pictures, and anything else."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bidsmgr.cli.create import open_or_create_workspace
from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.editor_panel import EditorPanel
from bidsmgr.gui.welcome_panel import WelcomePanel
from bidsmgr.viz.data.formats import document_kind

pytestmark = pytest.mark.gui


def _bids(root: Path, name: str = "Study") -> Path:
    (root / "sub-01" / "anat").mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": name, "BIDSVersion": "1.10.0"}), encoding="utf-8")
    (root / "sub-01" / "anat" / "sub-01_T1w.json").write_text(
        json.dumps({"RepetitionTime": 2.3}), encoding="utf-8")
    return root


def _plain(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "notes.txt").write_text("measurements\n", encoding="utf-8")
    (root / "scan.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    return root


def _listing(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


@pytest.fixture
def panel(qtbot) -> EditorPanel:
    p = EditorPanel()
    qtbot.addWidget(p)
    return p


# ---------------------------------------------------------------------------
# One Open folder: the project, or a folder open for viewing
# ---------------------------------------------------------------------------


def test_there_is_one_way_to_open_a_folder(panel) -> None:
    assert panel._open_btn.text().strip() == "Open folder…"
    assert panel._path_bar.change_button.isHidden()


def test_a_folder_that_is_not_bids_opens_for_viewing_and_stays_untouched(
        panel, qtbot, tmp_path) -> None:
    folder = _plain(tmp_path / "scans")
    before = _listing(folder)
    assert panel.open_folder(folder) == "view"
    assert panel.is_view_only() and panel._path_bar.chip_text() == "view only"
    assert not panel._tools_btn.isEnabled()
    assert not panel._validate_dataset_btn.isEnabled(), "nothing to validate"
    # A validation that ends after the switch does not turn it back on.
    panel._set_busy(False)
    panel._on_worker_finished()
    assert not panel._validate_dataset_btn.isEnabled()
    assert panel._tree_pane.read_only
    assert panel._nifti_viewer.read_only and panel._signal_viewer.read_only
    # Its JSON shows, but cannot be edited; its text shows, but cannot be saved.
    panel._on_file_selected(folder / "scan.json")
    assert not panel._sidecar_form._edit_toolbar.isVisibleTo(panel._sidecar_form)
    panel._on_file_selected(folder / "notes.txt")
    assert panel._center_stack.currentWidget() is panel._text_pane
    assert panel._text_pane._editor.isReadOnly()
    assert _listing(folder) == before, "a folder open for viewing was written into"


def test_a_bids_dataset_asks_whether_to_open_it_as_the_project(panel, qtbot, tmp_path) -> None:
    ds = _bids(tmp_path / "ds")
    panel.ask_bids_folder = lambda _p: "project"
    with qtbot.waitSignal(panel.open_project_requested) as got:
        assert panel.open_folder(ds) == "project"
    assert got.args == [ds]
    panel.ask_bids_folder = lambda _p: "view"
    assert panel.open_folder(ds) == "view" and panel.is_view_only()
    # Viewing a BIDS dataset still validates (it only reads) and offers to
    # become the project.
    assert panel._validate_dataset_btn.isEnabled()
    assert panel._path_bar.action_button.text() == "Open as project"
    assert not (ds / ".bidsmgr").exists()
    panel.ask_bids_folder = lambda _p: ""
    assert panel.open_folder(ds) == ""


def test_a_folder_inside_the_project_opens_the_project(panel, tmp_path) -> None:
    project = _bids(tmp_path / "project")
    panel.set_project_root(project)
    assert not panel.is_view_only() and panel._path_bar.chip_text() == "project"
    assert panel.open_folder(project / "sub-01") == "project"
    assert panel.current_root() == project and not panel.is_view_only()


def test_back_to_the_project_from_a_folder_open_for_viewing(panel, tmp_path) -> None:
    project = _bids(tmp_path / "project")
    panel.set_project_root(project)
    panel.view_folder(_plain(tmp_path / "elsewhere"))
    assert panel._path_bar.action_button.text() == "Back to project"
    panel._path_bar.action_button.click()
    assert panel.current_root() == project and not panel.is_view_only()
    assert panel._tools_btn.isEnabled()


# ---------------------------------------------------------------------------
# The window: open as project from the Editor, and switching keeps the tab
# ---------------------------------------------------------------------------


@pytest.fixture
def window(qtbot, qapp):
    from bidsmgr.gui.main_window import MainWindow
    from bidsmgr.gui.theme_manager import ThemeManager

    theme = ThemeManager(qapp)
    theme.apply("dark")
    win = MainWindow(theme)
    qtbot.addWidget(win)
    return win


def test_switching_projects_keeps_the_tab(window, tmp_path) -> None:
    a = _bids(tmp_path / "a", "A")
    b = _bids(tmp_path / "b", "B")
    window.welcome.open_project(a)
    window._apply_active_view("editor", persist=True)
    window._on_switch_project(b)
    assert window._project_root == b and window.stack.currentIndex() == 1
    assert window.editor.current_root() == b and not window.editor.is_view_only()
    window._apply_active_view("converter", persist=True)
    window._on_switch_project(a)
    assert window._project_root == a and window.stack.currentIndex() == 0


def test_the_editor_opens_a_dataset_as_the_project(window, tmp_path) -> None:
    a = _bids(tmp_path / "a", "A")
    b = _bids(tmp_path / "b", "B")
    window.welcome.open_project(a)
    window._apply_active_view("editor", persist=True)
    window.editor.ask_bids_folder = lambda _p: "project"
    window.editor.open_folder(b)
    assert window._project_root == b and window.converter._bids_root == b
    assert window.stack.currentIndex() == 1, "opening it moved the user off the Editor"
    assert (b / ".bidsmgr" / "project").is_dir()


def test_welcome_opens_a_folder_that_is_not_bids_for_viewing(window, tmp_path) -> None:
    folder = _plain(tmp_path / "scans")
    before = _listing(folder)
    window.welcome.ask_not_bids = lambda _p: "view"
    assert window.welcome.open_folder(folder) == "view"
    assert window.stack.currentIndex() == 1 and window.editor.is_view_only()
    assert window.editor.current_root() == folder
    assert _listing(folder) == before
    assert window._project_root is None
    # Asked to, it becomes a project (and is written into).
    window.welcome.ask_not_bids = lambda _p: "project"
    assert window.welcome.open_folder(folder) == "project"
    assert (folder / "dataset_description.json").is_file()


def test_an_empty_folder_opens_as_a_project_without_asking(qtbot, tmp_path) -> None:
    w = WelcomePanel()
    qtbot.addWidget(w)
    w.ask_not_bids = lambda _p: pytest.fail("an empty folder needs no question")
    empty = tmp_path / "new"
    empty.mkdir()
    assert w.open_folder(empty) == "project"


# ---------------------------------------------------------------------------
# Recent projects: several at once
# ---------------------------------------------------------------------------


def test_recent_projects_are_removed_and_deleted_several_at_once(qtbot, tmp_path) -> None:
    roots = [_bids(tmp_path / n, n) for n in ("one", "two", "three")]
    for r in roots:
        open_or_create_workspace(r)
        AppSettings.remember_recent_project(r)
    w = WelcomePanel()
    qtbot.addWidget(w)
    w.refresh_recent()
    assert w._recent.count() == 3
    for i in range(2):
        w._recent.item(i).setSelected(True)
    assert len(w.selected_recent()) == 2
    assert w._recent_forget_btn.text() == "Remove 2 from list"
    assert not w._recent_open_btn.isEnabled()
    w._forget_selected()
    assert w._recent.count() == 1
    assert all(r.exists() for r in roots), "remove from list deleted a folder"
    for r in roots:
        AppSettings.remember_recent_project(r)
    w.refresh_recent()
    # The project open now is never deleted, the rest are.
    w.current_project = lambda: roots[0]
    deleted, kept = w.delete_projects(roots[:2])
    assert deleted == [roots[1]] and kept[0][0] == roots[0]
    assert roots[0].exists() and not roots[1].exists()
    listed = [w._recent.item(i).data(0x0100) for i in range(w._recent.count())]
    assert str(roots[1]) not in listed


def test_a_missing_project_can_be_selected_and_removed(qtbot, tmp_path) -> None:
    gone = tmp_path / "gone"
    AppSettings.remember_recent_project(gone)
    w = WelcomePanel()
    qtbot.addWidget(w)
    w.refresh_recent()
    w._recent.item(0).setSelected(True)
    assert w.selected_recent() == [gone]
    w._forget_selected()
    assert w._recent.count() == 0


# ---------------------------------------------------------------------------
# Documents, the gradient table, pictures, anything else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,kind", [
    ("README", "text"), ("README.md", "markdown"), ("CHANGES", "text"), ("LICENSE", "text"),
    ("sub-01_dwi.bval", "gradients"), ("sub-01_dwi.bvec", "gradients"),
    ("sub-01_photo.jpg", "picture"), ("validation_report.html", "html"),
    ("code/run.py", "text"), ("sub-01_task-x_eeg.eeg", "")])
def test_files_are_routed_by_name(name, kind) -> None:
    assert document_kind(Path(name)) == kind


def test_markdown_renders_and_text_is_saved_as_an_undoable_step(panel, tmp_path) -> None:
    project = _bids(tmp_path / "project")
    open_or_create_workspace(project)
    (project / "README.md").write_text("# Study\n\nSome *words*.\n", encoding="utf-8")
    panel.set_project_root(project)
    panel._on_file_selected(project / "README.md")
    pane = panel._text_pane
    assert panel._center_stack.currentWidget() is pane and pane.is_rendered()
    assert "Study" in pane._browser.toPlainText() and "#" not in pane._browser.toPlainText()
    pane._editor.setPlainText("# Study\n\nWords.\n\n## Contents\n\nMore.\n")
    pane._on_view(True)
    doc = pane._browser.document()
    heading = next(b for b in (doc.findBlockByNumber(i) for i in range(doc.blockCount()))
                   if b.text() == "Contents")
    assert heading.blockFormat().topMargin() > 0, "a heading sits on the paragraph above"
    pane.revert()
    pane._text_btn.click()
    assert not pane.is_rendered()
    pane._editor.setPlainText("# Study\n\nOther words.\n")
    assert pane.is_dirty() and pane._save_btn.isEnabled()
    assert pane.save()
    assert (project / "README.md").read_text(encoding="utf-8") == "# Study\n\nOther words.\n"
    from bidsmgr.project.operations import read_log

    assert any("README.md" in str(op.get("label", "")) for op in read_log(project))


def test_the_gradient_table_is_summarised_drawn_and_listed(panel, tmp_path) -> None:
    import nibabel as nib

    dwi = tmp_path / "ds" / "sub-01" / "dwi"
    dwi.mkdir(parents=True)
    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000, 0, 2000, 2000])
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(10, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs[bvals == 0] = 0
    np.savetxt(dwi / "sub-01_dwi.bval", bvals[None], fmt="%d")
    np.savetxt(dwi / "sub-01_dwi.bvec", vecs.T, fmt="%.6f")
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 10), np.int16), np.eye(4)),
             str(dwi / "sub-01_dwi.nii.gz"))
    panel._on_file_selected(dwi / "sub-01_dwi.bvec")
    pane = panel._gradient_pane
    assert panel._center_stack.currentWidget() is pane
    facts = pane.facts()
    assert facts[:3] == ["10 volumes", "2 at b=0", "2 shells"]
    assert facts[3] == "Matches sub-01_dwi.nii.gz (10 volumes)"
    assert pane.table.rowCount() == 10 and pane.table.item(1, 1).text() == "1000"
    drawn = [s for s in pane.sphere.scatters if len(s.data)]
    assert len(drawn) == 2, "one colour per shell"


def test_a_picture_is_shown_fitted(panel, tmp_path) -> None:
    from PyQt6.QtGui import QColor, QImage

    path = tmp_path / "sub-01_photo.png"
    img = QImage(64, 32, QImage.Format.Format_RGB32)
    img.fill(QColor("#336699"))
    img.save(str(path))
    panel._on_file_selected(path)
    pane = panel._picture_pane
    assert panel._center_stack.currentWidget() is pane and pane.is_fitted()
    assert "64 x 32" in pane._status.text()


def test_anything_else_gets_its_facts_and_the_system_app(panel, tmp_path) -> None:
    path = tmp_path / "sub-01_task-x_eeg.eeg"
    path.write_bytes(b"\0" * 2048)
    panel._on_file_selected(path)
    pane = panel._file_info_pane
    assert panel._center_stack.currentWidget() is pane
    assert "BrainVision" in pane._kind.text() and "2.0 KB" in pane._size.text()
    assert pane.open_button.isEnabled()

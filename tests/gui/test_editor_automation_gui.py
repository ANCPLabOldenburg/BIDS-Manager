"""The Editor's two views must hold the same fields, and the tree must be
able to show the files a dataset hides.

The first is the defect this work started from: with a validation report bound,
the BIDS form offered every field the schema declares for the datatype and the
Tree view rendered the file, so the same sidecar showed 130 rows in one and 90
in the other with nothing to say why.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.editor.types import (
    FieldLevel,
    FileVerdict,
    Severity,
    SidecarField,
    ValidationReport,
)
from .conftest import open_every_folder
from bidsmgr.gui.app_settings import AppSettings
from bidsmgr.gui.widgets.bids_tree_pane import BidsTreePane, is_hidden_name
from bidsmgr.gui.widgets.sidecar_form_pane import (
    SCOPE_ABSENT,
    SCOPE_ALL,
    SCOPE_PRESENT,
    SidecarFormPane,
)
from bidsmgr.gui.widgets.sidecar_row import SidecarRow

pytestmark = pytest.mark.gui


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    anat = root / "sub-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_T1w.json").write_text(
        json.dumps({"EchoTime": 0.03, "Manufacturer": "Siemens"})
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    (root / ".bidsignore").write_text("derivatives/\n")
    (root / ".bidsmgr").mkdir()
    return root


def _report(root: Path) -> ValidationReport:
    """A verdict declaring one field the file does not carry."""
    return ValidationReport(
        bids_root=root,
        files=[FileVerdict(
            path=Path("sub-01/anat/sub-01_T1w.json"),
            severity=Severity.WARN,
            datatype="anat",
            suffix="T1w",
            sidecar_fields=[
                SidecarField(
                    level=FieldLevel.REQUIRED, name="MagneticFieldStrength",
                    value=None, present=False, value_kind="missing",
                ),
                SidecarField(
                    level=FieldLevel.OPTIONAL, name="EchoTime",
                    value=0.03, present=True, value_kind="number",
                ),
            ],
        )],
    )


def _form_keys(pane: SidecarFormPane) -> list[str]:
    return [r.key for r in pane.findChildren(SidecarRow)]


def _tree_keys(pane: SidecarFormPane) -> list[str]:
    tree = pane._tree_view
    return [
        tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())
    ]


# ---------------------------------------------------------------------------
# The two views agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", [SCOPE_ALL, SCOPE_PRESENT, SCOPE_ABSENT])
def test_both_views_hold_the_same_fields(qtbot, dataset: Path, scope) -> None:
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_field_scope(scope)
    pane.set_file(
        dataset / "sub-01" / "anat" / "sub-01_T1w.json", dataset,
        _report(dataset),
    )
    form = sorted(_form_keys(pane))
    pane._apply_view_mode("tree", persist=False)
    tree = sorted(_tree_keys(pane))
    assert form == tree, f"views disagree in scope {scope}"


def test_all_scope_shows_more_than_the_file_carries(
    qtbot, dataset: Path,
) -> None:
    """The point of the BIDS form: it offers what the standard declares."""
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_field_scope(SCOPE_ALL)
    pane.set_file(
        dataset / "sub-01" / "anat" / "sub-01_T1w.json", dataset,
        _report(dataset),
    )
    keys = _form_keys(pane)
    assert "MagneticFieldStrength" in keys   # declared, absent
    assert "Manufacturer" in keys            # in the file, not in the verdict


def test_present_scope_is_exactly_the_file(qtbot, dataset: Path) -> None:
    fp = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    on_disk = set(json.loads(fp.read_text()))
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_field_scope(SCOPE_PRESENT)
    pane.set_file(fp, dataset, _report(dataset))
    assert set(_form_keys(pane)) == on_disk


def test_a_declared_but_absent_field_is_not_written_back(
    qtbot, dataset: Path,
) -> None:
    """The tree shows absent fields so the views match. Showing them must not
    add them to the file."""
    fp = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_field_scope(SCOPE_ALL)
    pane.set_file(fp, dataset, _report(dataset))
    pane._apply_view_mode("tree", persist=False)
    assert "MagneticFieldStrength" in _tree_keys(pane)
    written = pane._tree_view.to_dict()
    assert "MagneticFieldStrength" not in written
    assert set(written) == set(json.loads(fp.read_text()))


# ---------------------------------------------------------------------------
# Hidden files
# ---------------------------------------------------------------------------


def test_hidden_names_are_one_rule() -> None:
    assert is_hidden_name(".bidsignore")
    assert is_hidden_name(".bidsmgr")
    assert is_hidden_name(".git")
    assert not is_hidden_name("sub-01")
    assert not is_hidden_name("dataset_description.json")


def test_the_tree_can_show_and_hide_dotfiles(qtbot, dataset: Path) -> None:
    def _top_level_names() -> list[str]:
        pane = BidsTreePane()
        qtbot.addWidget(pane)
        pane.set_root(dataset)
        top = pane._tree.topLevelItem(0)
        return [top.child(i).text(0) for i in range(top.childCount())]

    previous = AppSettings.load().editor_show_hidden
    try:
        AppSettings.remember_editor_show_hidden(False)
        assert not [n for n in _top_level_names() if n.startswith(".")]

        AppSettings.remember_editor_show_hidden(True)
        shown = _top_level_names()
        assert ".bidsignore" in shown
        assert ".bidsmgr" in shown
    finally:
        AppSettings.remember_editor_show_hidden(previous)


# ---------------------------------------------------------------------------
# The save-defect reproductions the plan asked for before the save path was
# touched. Each is a way an edit could go missing.
# ---------------------------------------------------------------------------


def _edit(pane: SidecarFormPane, key: str, text: str) -> None:
    row = next(r for r in pane._rows if r.key == key)
    row.editor().setText(text)
    row.editor().editingFinished.emit()


def test_switching_view_keeps_the_edit(qtbot, dataset: Path) -> None:
    """BIDS form to Tree view and back. Both read one cache, so an edit made
    in either must survive the trip."""
    fp = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_autosave(False)
    pane.set_file(fp, dataset, None)
    _edit(pane, "EchoTime", "0.09")

    pane._apply_view_mode("tree", persist=False)
    assert pane._tree_view.to_dict()["EchoTime"] == 0.09
    pane._apply_view_mode("bids", persist=False)
    assert pane._json_cache["EchoTime"] == 0.09
    assert pane.is_dirty()


def test_editing_back_to_the_original_is_not_dirty(
    qtbot, dataset: Path,
) -> None:
    """A dirty flag that stays on after the value is back where it started
    would make the count meaningless."""
    fp = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_autosave(False)
    pane.set_file(fp, dataset, None)
    _edit(pane, "EchoTime", "0.09")
    assert pane.is_dirty()
    _edit(pane, "EchoTime", "0.03")
    assert not pane.is_dirty()


def test_a_nested_object_survives_a_save(qtbot, dataset: Path) -> None:
    """Round trip through the tree editor must not flatten a container or
    retype its values."""
    fp = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    original = {
        "EchoTime": 0.03,
        "GeneratedBy": [{"Name": "bidsmgr", "Version": "1.2.6"}],
        "Nested": {"a": 1, "b": [1, 2, 3]},
    }
    fp.write_text(json.dumps(original))
    pane = SidecarFormPane()
    qtbot.addWidget(pane)
    pane.set_autosave(False)
    pane.set_file(fp, dataset, None)
    _edit(pane, "EchoTime", "0.05")
    assert pane.save()

    on_disk = json.loads(fp.read_text())
    assert on_disk["EchoTime"] == 0.05
    assert on_disk["GeneratedBy"] == original["GeneratedBy"]
    assert on_disk["Nested"] == original["Nested"]
    assert list(on_disk) == list(original), "key order must be preserved"


def test_two_panes_on_one_file_do_not_clobber_each_other(
    qtbot, dataset: Path,
) -> None:
    """A detached panel means two panes can hold the same file. The second to
    save must not silently drop the first one's field."""
    fp = dataset / "sub-01" / "anat" / "sub-01_T1w.json"
    first, second = SidecarFormPane(), SidecarFormPane()
    for pane in (first, second):
        qtbot.addWidget(pane)
        pane.set_autosave(False)
        pane.set_file(fp, dataset, None)

    _edit(first, "EchoTime", "0.07")
    assert first.save()

    # The second pane still holds the pre-save state. Re-binding is what the
    # Editor does when a file it is showing changes on disk.
    second.set_file(fp, dataset, None)
    assert second._json_cache["EchoTime"] == 0.07


# ---------------------------------------------------------------------------
# A tree you scan rather than excavate
# ---------------------------------------------------------------------------


def _find(pane: BidsTreePane, name: str):
    from PyQt6.QtCore import Qt as _Qt

    it = pane._tree.findItems(
        name, _Qt.MatchFlag.MatchRecursive | _Qt.MatchFlag.MatchExactly, 0,
    )
    return it[0] if it else None


def test_a_folder_says_what_is_inside_it(qtbot, dataset: Path) -> None:
    from bidsmgr.gui.widgets.bids_tree_pane import COUNT_ROLE

    ses = dataset / "sub-02" / "ses-pre" / "anat"
    ses.mkdir(parents=True)
    (ses / "sub-02_ses-pre_T1w.json").write_text("{}")
    (ses / "sub-02_ses-pre_T2w.json").write_text("{}")

    pane = BidsTreePane()
    qtbot.addWidget(pane)
    pane.set_root(dataset)
    open_every_folder(pane._tree)

    subject = _find(pane, "sub-02")
    assert subject is not None
    badge = subject.data(0, COUNT_ROLE)
    assert "1 ses" in badge and "2 files" in badge


def test_the_count_is_a_role_not_the_name(qtbot, dataset: Path) -> None:
    """Several things look items up by their text. The badge is painted by the
    delegate so the name stays the name."""
    pane = BidsTreePane()
    qtbot.addWidget(pane)
    pane.set_root(dataset)
    open_every_folder(pane._tree)
    assert _find(pane, "sub-01") is not None
    assert _find(pane, "anat") is not None


def test_the_tree_reports_every_selected_file(qtbot, dataset: Path) -> None:
    """Bulk actions need more than one file, so the tree must hand over more
    than one."""
    from PyQt6.QtCore import QItemSelectionModel

    second = dataset / "sub-01" / "anat" / "sub-01_T2w.json"
    second.write_text("{}")

    pane = BidsTreePane()
    qtbot.addWidget(pane)
    pane.set_root(dataset)
    open_every_folder(pane._tree)

    seen: list[list[Path]] = []
    pane.selection_changed.connect(seen.append)
    for name in ("sub-01_T1w.json", "sub-01_T2w.json"):
        item = _find(pane, name)
        assert item is not None
        item.setSelected(True)
    pane._tree.setCurrentItem(
        _find(pane, "sub-01_T2w.json"),
        0,
        QItemSelectionModel.SelectionFlag.NoUpdate,
    )

    picked = {p.name for p in pane.selected_paths()}
    assert picked == {"sub-01_T1w.json", "sub-01_T2w.json"}
    assert seen, "selecting must announce itself"


# ---------------------------------------------------------------------------
# Accepting a warning
# ---------------------------------------------------------------------------


def _warn_report(root: Path) -> ValidationReport:
    from bidsmgr.editor.types import Issue

    return ValidationReport(
        bids_root=root,
        files=[FileVerdict(
            path=Path("sub-01/anat/sub-01_T1w.json"),
            severity=Severity.WARN,
            issues=[
                Issue(
                    severity=Severity.WARN,
                    rule_id="SIDECAR_KEY_RECOMMENDED",
                    message="InstitutionName is recommended",
                    field="InstitutionName",
                ),
                Issue(
                    severity=Severity.ERR, rule_id="SIDECAR_KEY_REQUIRED",
                    message="MagneticFieldStrength is required",
                    field="MagneticFieldStrength",
                ),
            ],
        )],
    )


def _messages(pane):
    from bidsmgr.gui.widgets.val_message import ValMessage

    return pane.findChildren(ValMessage)


def test_only_warnings_offer_acceptance(qtbot, dataset: Path) -> None:
    """A tool that lets you dismiss an error produces broken datasets."""
    from PyQt6.QtCore import Qt as _Qt
    from bidsmgr.gui.widgets.validation_pane import ValidationPane

    pane = ValidationPane()
    qtbot.addWidget(pane)
    pane.set_report(_warn_report(dataset))
    pane.set_current_file(
        dataset / "sub-01" / "anat" / "sub-01_T1w.json", dataset,
    )

    policies = {
        m.objectName(): m.contextMenuPolicy() for m in _messages(pane)
    }
    assert policies.get("val-msg-warn") == _Qt.ContextMenuPolicy.CustomContextMenu
    assert policies.get("val-msg-err") == _Qt.ContextMenuPolicy.DefaultContextMenu


def test_an_accepted_warning_is_dimmed_not_hidden(qtbot, dataset: Path) -> None:
    """The decision is part of the record. Hiding it would make the next
    reviewer rediscover the same finding."""
    from bidsmgr.editor import review
    from bidsmgr.gui.widgets.validation_pane import ValidationPane

    pane = ValidationPane()
    qtbot.addWidget(pane)
    pane.set_report(_warn_report(dataset))
    pane.set_current_file(
        dataset / "sub-01" / "anat" / "sub-01_T1w.json", dataset,
    )
    warn = next(m for m in _messages(pane) if m.objectName() == "val-msg-warn")
    assert warn.isEnabled()

    review.accept(
        dataset, file="sub-01/anat/sub-01_T1w.json",
        rule_id="SIDECAR_KEY_RECOMMENDED", field="InstitutionName",
        note="Resting state, deliberate.",
    )
    pane.reload_acceptances()

    warn = next(m for m in _messages(pane) if m.objectName() == "val-msg-warn")
    assert not warn.isEnabled()
    assert "Resting state" in warn.toolTip()
    # The error beside it is untouched.
    err = next(m for m in _messages(pane) if m.objectName() == "val-msg-err")
    assert err.isEnabled()


def test_the_pane_forwards_the_file_with_the_acceptance(
    qtbot, dataset: Path,
) -> None:
    from bidsmgr.gui.widgets.validation_pane import ValidationPane

    pane = ValidationPane()
    qtbot.addWidget(pane)
    pane.set_report(_warn_report(dataset))
    pane.set_current_file(
        dataset / "sub-01" / "anat" / "sub-01_T1w.json", dataset,
    )
    seen: list[tuple] = []
    pane.accept_requested.connect(lambda *a: seen.append(a))
    warn = next(m for m in _messages(pane) if m.objectName() == "val-msg-warn")
    warn.accept_requested.emit("SIDECAR_KEY_RECOMMENDED", "InstitutionName")

    assert seen == [(
        dataset / "sub-01" / "anat" / "sub-01_T1w.json",
        "SIDECAR_KEY_RECOMMENDED", "InstitutionName",
    )]


def test_a_finding_explains_what_its_rule_is_for(qtbot, dataset: Path) -> None:
    """The code lets a user check the claim; the prose says why it matters."""
    from PyQt6.QtWidgets import QLabel
    from bidsmgr.gui.widgets.validation_pane import ValidationPane

    pane = ValidationPane()
    qtbot.addWidget(pane)
    pane.set_report(_warn_report(dataset))
    pane.set_current_file(
        dataset / "sub-01" / "anat" / "sub-01_T1w.json", dataset,
    )
    warn = next(m for m in _messages(pane) if m.objectName() == "val-msg-warn")
    notes = [
        w for w in warn.findChildren(QLabel)
        if w.objectName() == "val-explanation"
    ]
    assert notes and "recommends" in notes[0].text()


# ---------------------------------------------------------------------------
# Renaming from where you noticed the problem
#
# You spot a wrong subject label while looking at the subject. Making the user
# go to a toolbar and re-pick it from a dropdown is the difference between an
# action that gets used and one that does not.
# ---------------------------------------------------------------------------


def test_the_tree_offers_the_entities_the_clicked_row_carries() -> None:
    from bidsmgr.gui.widgets.bids_tree_pane import _renameable_entities

    keys = [e for e, _v, _l in _renameable_entities(Path("sub-01"))]
    assert keys == ["sub"]

    got = {
        e: v for e, v, _l in _renameable_entities(
            Path("sub-01_ses-pre_task-rest_run-02_bold.nii.gz")
        )
    }
    assert got == {
        "sub": "01", "ses": "pre", "task": "rest", "run": "02",
    }


def test_a_plain_file_offers_nothing_to_rename() -> None:
    from bidsmgr.gui.widgets.bids_tree_pane import _renameable_entities

    assert _renameable_entities(Path("README")) == []
    assert _renameable_entities(Path("CHANGES")) == []


def test_entities_are_labelled_in_words() -> None:
    """`sub-` tells a first-time user nothing."""
    from bidsmgr.gui.widgets.bids_tree_pane import _renameable_entities

    labels = {e: lb for e, _v, lb in _renameable_entities(
        Path("sub-01_ses-pre_acq-highres_T1w.json")
    )}
    assert labels["sub"] == "subject"
    assert labels["ses"] == "session"
    assert labels["acq"] == "acquisition"


def test_the_tree_announces_a_rename_rather_than_doing_one(
    qtbot, dataset: Path,
) -> None:
    """The tree knows what was clicked; the panel knows the root and what to
    refresh. Keeping the dialog out of the tree is what makes both testable."""
    pane = BidsTreePane()
    qtbot.addWidget(pane)
    pane.set_root(dataset)
    seen: list[tuple] = []
    pane.rename_requested.connect(lambda *a: seen.append(a))
    pane.rename_requested.emit("sub", "01", str(dataset / "sub-01"))
    assert seen == [("sub", "01", str(dataset / "sub-01"))], (
        "the clicked PATH travels too, so the dialog can open on that "
        "recording rather than on every file sharing its subject"
    )


def test_the_tree_announces_an_entity_edit_the_same_way(
    qtbot, dataset: Path,
) -> None:
    """Same division of labour for adding and removing entities, and for the
    sessions made out of them: the tree says what was clicked and what was
    asked for, the panel owns the root, the dialog and the refresh."""
    pane = BidsTreePane()
    qtbot.addWidget(pane)
    pane.set_root(dataset)
    seen: list[tuple] = []
    pane.entities_requested.connect(lambda *a: seen.append(a))

    target = dataset / "sub-01"
    pane.entities_requested.emit([target], "add", True)
    assert seen == [([target], "add", True)]


def test_restructuring_is_offered_only_at_or_under_a_subject() -> None:
    """Above a subject there is no recording to speak of, and the tool's own
    ``.bidsmgr/`` holds the operation log, which is the one thing a
    restructuring must not restructure."""
    from bidsmgr.gui.widgets.bids_tree_pane import (
        _can_restructure,
        _in_session,
    )

    assert _can_restructure(Path("ds/sub-01/anat/sub-01_T1w.nii.gz"))
    assert _can_restructure(Path("ds/sub-01"))
    assert not _can_restructure(Path("ds/participants.tsv"))
    assert not _can_restructure(Path("ds/.bidsmgr/sub-01/anything"))

    assert _in_session(Path("ds/sub-01/ses-pre/anat/sub-01_ses-pre_T1w.nii.gz"))
    assert not _in_session(Path("ds/sub-01/anat/sub-01_T1w.nii.gz"))


# ---------------------------------------------------------------------------
# The rename dialog, including the merge it offers instead of a refusal
# ---------------------------------------------------------------------------


@pytest.fixture
def two_subjects(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    for sub, ses in (("sub-01", "ses-pre"), ("sub-02", "ses-post")):
        anat = root / sub / ses / "anat"
        anat.mkdir(parents=True)
        (anat / f"{sub}_{ses}_T1w.nii.gz").write_text("i")
        (anat / f"{sub}_{ses}_T1w.json").write_text(json.dumps({"EchoTime": 0.03}))
    (root / "participants.tsv").write_text(
        "participant_id\tage\nsub-01\t31\nsub-02\tn/a\n"
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    return root


def _preview_text(dlg) -> str:
    """Every row of the dry-run tree, flattened, for asserting on."""
    lines: list[str] = []
    tree = dlg._preview
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        lines.append(f"{top.text(0)}  {top.text(1)}")
        for j in range(top.childCount()):
            child = top.child(j)
            lines.append(f"{child.text(0)}  {child.text(1)}")
    return "\n".join(lines)


def _dialog(qtbot, root: Path, entity: str = "", value: str = ""):
    from bidsmgr.gui.rename_entity_dialog import RenameEntityDialog

    dlg = RenameEntityDialog(root, entity=entity, value=value)
    qtbot.addWidget(dlg)
    return dlg


def _type(dlg, text: str) -> None:
    """Type a new label and let the plan settle.

    Planning is debounced, because walking the dataset on every keystroke is
    what froze the window on a real one. A test that sets the text and reads
    the plan in the same breath is not testing what a user does; it is racing
    a timer that exists on purpose.
    """
    dlg._new.setText(text)
    dlg.plan_now()


def test_the_dialog_opens_on_what_the_user_clicked(
    qtbot, two_subjects: Path,
) -> None:
    dlg = _dialog(qtbot, two_subjects, "sub", "02")
    assert dlg._entity.currentData() == "sub"
    assert dlg._old.currentText() == "02"


def test_a_free_name_is_an_ordinary_rename(qtbot, two_subjects: Path) -> None:
    dlg = _dialog(qtbot, two_subjects, "sub", "02")
    _type(dlg, "99")
    # isHidden, not isVisible: an unshown dialog's children are all invisible,
    # so isVisible would pass whatever the code did.
    assert dlg._fuse.isHidden()
    assert dlg._ok.text() == "Rename"
    assert dlg._ok.isEnabled()
    assert not dlg._plan.fusion


def test_a_taken_name_offers_the_merge_instead_of_only_refusing(
    qtbot, two_subjects: Path,
) -> None:
    dlg = _dialog(qtbot, two_subjects, "sub", "02")
    _type(dlg, "01")
    assert not dlg._fuse.isHidden(), "the option must be offered where it bites"
    assert not dlg._ok.isEnabled(), "not until the user says merge"
    assert "Merge them into one subject" in dlg._summary.text()

    dlg._fuse.setChecked(True)
    assert dlg._ok.isEnabled()
    assert dlg._ok.text() == "Merge", "the button must not still say Rename"
    assert dlg._plan.fusion
    assert "cannot be undone by renaming back" in dlg._summary.text()


def test_the_merge_option_disappears_when_the_name_frees_up(
    qtbot, two_subjects: Path,
) -> None:
    """A ticked checkbox that no longer applies would silently change the
    meaning of the next Rename."""
    dlg = _dialog(qtbot, two_subjects, "sub", "02")
    _type(dlg, "01")
    dlg._fuse.setChecked(True)
    _type(dlg, "99")
    assert dlg._fuse.isHidden()
    assert not dlg._fuse.isChecked()
    assert dlg._ok.text() == "Rename"


def test_the_preview_names_the_tables_a_merge_would_touch(
    qtbot, two_subjects: Path,
) -> None:
    """A merge does things a rename does not, so the dry run has to show them."""
    (two_subjects / "sub-02" / "sub-02_scans.tsv").write_text(
        "filename\nses-post/anat/sub-02_ses-post_T1w.nii.gz\n"
    )
    (two_subjects / "sub-01" / "sub-01_scans.tsv").write_text(
        "filename\nses-pre/anat/sub-01_ses-pre_T1w.nii.gz\n"
    )
    dlg = _dialog(qtbot, two_subjects, "sub", "02")
    _type(dlg, "01")
    dlg._fuse.setChecked(True)
    text = _preview_text(dlg)
    assert "merged into the existing sub-01" in text
    assert "appended to sub-01_scans.tsv" in text
    assert "two participant_id rows folded into one" in text


# ---------------------------------------------------------------------------
# Dialog chrome
# ---------------------------------------------------------------------------


def test_a_wrapped_label_takes_the_height_its_text_needs(qtbot) -> None:
    """#pane-hint carries 24px of padding for the empty-pane case it was
    written for. A dialog of them drifts apart; that is what #dlg-hint is."""
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    from bidsmgr.gui.dialog_chrome import hint

    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    label = hint("One short line.")
    layout.addWidget(label)
    layout.addStretch(1)
    host.resize(700, 400)
    host.show()
    qtbot.waitExposed(host)

    assert label.objectName() == "dlg-hint"
    one_line = label.fontMetrics().height()
    assert label.height() <= one_line * 2


def test_the_new_dialogs_wear_the_app_chrome(qtbot, two_subjects: Path) -> None:
    """A dialog that does not set these object names is painted by Qt's
    defaults and looks like it came from a different program."""
    from PyQt6.QtWidgets import QFrame

    from bidsmgr.gui.fixups_dialog import FixupsDialog

    for dlg in (_dialog(qtbot, two_subjects), FixupsDialog(two_subjects)):
        qtbot.addWidget(dlg)
        names = {
            f.objectName() for f in dlg.findChildren(QFrame)
        }
        assert "issue-dialog-header" in names, type(dlg).__name__
        assert "issue-dialog-footer" in names, type(dlg).__name__


# ---------------------------------------------------------------------------
# Choosing which files to rename
# ---------------------------------------------------------------------------


@pytest.fixture
def three_runs(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    func = root / "sub-01" / "func"
    func.mkdir(parents=True)
    for run in ("1", "2", "3"):
        (func / f"sub-01_task-rest_run-{run}_bold.nii.gz").write_bytes(b"")
        (func / f"sub-01_task-rest_run-{run}_bold.json").write_text(
            json.dumps({"TaskName": "rest"})
        )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds", "BIDSVersion": "1.10.0"})
    )
    return root


def _file_rows(dlg):
    """Every checkable FILE row, at whatever depth it sits.

    Recursive, because the preview is a dataset tree: a file lives under its
    datatype, which lives under its session, which lives under its subject.
    Walking exactly two levels found the rows only while the tree was a flat
    list of folder groups.
    """
    out = []
    tree = dlg._preview
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        item = stack.pop()
        stack.extend(item.child(j) for j in range(item.childCount()))
        if item.data(0, _key_role()):
            out.append(item)
    return out


def _key_role():
    from bidsmgr.gui.rename_entity_dialog import _KEY_ROLE

    return _KEY_ROLE


def test_every_file_starts_selected(qtbot, three_runs: Path) -> None:
    """The common case is renaming all of them, so that is the default."""
    from PyQt6.QtCore import Qt as _Qt

    dlg = _dialog(qtbot, three_runs, "task", "rest")
    _type(dlg, "resting")
    rows = _file_rows(dlg)
    assert len(rows) == 6
    assert all(r.checkState(0) == _Qt.CheckState.Checked for r in rows)
    assert len(dlg.selected_keys()) == 6


def test_unticking_a_file_takes_it_out_of_the_rename(
    qtbot, three_runs: Path,
) -> None:
    from PyQt6.QtCore import Qt as _Qt

    dlg = _dialog(qtbot, three_runs, "task", "rest")
    _type(dlg, "resting")
    for row in _file_rows(dlg):
        if "run-3" in row.text(0):
            row.setCheckState(0, _Qt.CheckState.Unchecked)

    chosen = dlg.selected_keys()
    assert len(chosen) == 4
    assert not any("run-3" in k for k in chosen)


def test_the_button_says_what_a_partial_rename_is(
    qtbot, three_runs: Path,
) -> None:
    """Calling it Rename when it renames four of six files is a lie the user
    only discovers afterwards."""
    from PyQt6.QtCore import Qt as _Qt

    dlg = _dialog(qtbot, three_runs, "task", "rest")
    _type(dlg, "resting")
    assert dlg._ok.text() == "Rename"

    _file_rows(dlg)[0].setCheckState(0, _Qt.CheckState.Unchecked)
    assert dlg._ok.text() == "Rename selected"
    assert "5 of 6 file(s) selected" in dlg._status.text()


def test_selecting_none_disables_the_button(qtbot, three_runs: Path) -> None:
    from PyQt6.QtCore import Qt as _Qt

    dlg = _dialog(qtbot, three_runs, "task", "rest")
    _type(dlg, "resting")
    dlg._set_all(_Qt.CheckState.Unchecked)
    assert not dlg.selected_keys()
    assert not dlg._ok.isEnabled()

    dlg._set_all(_Qt.CheckState.Checked)
    assert len(dlg.selected_keys()) == 6
    assert dlg._ok.isEnabled()


def test_what_cannot_be_chosen_separately_is_not_offered_as_a_choice(
    qtbot, two_subjects: Path,
) -> None:
    """A cross-reference follows the file it names. Letting a user tick it
    independently would produce a pointer to a name nothing has."""
    from PyQt6.QtCore import Qt as _Qt

    (two_subjects / "sub-02" / "sub-02_scans.tsv").write_text(
        "filename\nses-post/anat/sub-02_ses-post_T1w.nii.gz\n"
    )
    dlg = _dialog(qtbot, two_subjects, "sub", "02")
    _type(dlg, "99")

    tree = dlg._preview
    follows = [
        tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
        if tree.topLevelItem(i).text(0) == "Follows automatically"
    ]
    assert follows, "the plan must still show them"
    for j in range(follows[0].childCount()):
        child = follows[0].child(j)
        assert not (child.flags() & _Qt.ItemFlag.ItemIsUserCheckable)


def test_the_rows_are_nested_like_the_dataset(qtbot, two_subjects: Path) -> None:
    """"These three runs" and "that whole session" are how a user thinks, and
    thinking that way needs a session to untick.

    The preview used to key its groups on the whole folder PATH, so
    ``sub-02/ses-post/anat`` and ``sub-02/ses-post/func`` came out as two
    unrelated top-level rows that happened to start with the same text. There
    was no subject to collapse and no session to untick.
    """
    dlg = _dialog(qtbot, two_subjects, "sub", "02")
    _type(dlg, "99")
    tree = dlg._preview

    tops = {
        tree.topLevelItem(i).text(0): tree.topLevelItem(i)
        for i in range(tree.topLevelItemCount())
    }
    assert "sub-02" in tops, "the subject should be the top-level row"
    assert not [name for name in tops if "/" in name], (
        "a folder path as a row label is the flat grouping this replaced"
    )

    sessions = {
        tops["sub-02"].child(i).text(0)
        for i in range(tops["sub-02"].childCount())
    }
    assert "ses-post" in sessions

    post = [tops["sub-02"].child(i)
            for i in range(tops["sub-02"].childCount())
            if tops["sub-02"].child(i).text(0) == "ses-post"][0]
    assert "anat" in {post.child(i).text(0) for i in range(post.childCount())}

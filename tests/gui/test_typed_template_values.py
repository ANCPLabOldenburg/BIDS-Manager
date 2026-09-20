"""A value the user TYPED must reach the sidecar, not only one they picked.

The report: "in the templates some fields are not being annotated after
conversion if the user writes something instead of selecting a fixed option.
For example the licence field: if you select one, all good, but if you write
it yourself, it won't be annotated."

Typing and selecting take different routes through a Qt combo. Selecting emits
``activated``; typing only emits ``editingFinished``, and only on focus-out or
Enter. Anything that reads a form on a button press rather than on those
signals has to read the widget itself, and anything in the chain below that
has to carry a value the schema's vocabulary does not contain.

These tests walk the whole chain for every kind of control the form builds:
tree -> values_by_key -> spec -> scaffold -> metadata run -> file on disk. They
do not reproduce the reported loss on the current code, which is why they
exist: so it cannot come back unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PyQt6.QtWidgets import QComboBox, QLineEdit

from bidsmgr.fixups.sidecar_schema import apply_stated_metadata
from bidsmgr.gui.widgets.template_form import TemplateTree
from bidsmgr.metadata.engine import run_metadata
from bidsmgr.metadata.template_plan import (
    TemplateNode,
    build_template_tree,
    dataset_description_section,
)
from bidsmgr.recording_meta import (
    CURATED_SUGGESTIONS,
    RecordingMetaSpec,
    dump_spec,
    scaffold_sidecar_path,
)
from bidsmgr.recording_meta.chain import dataset_description_from_bids

pytestmark = pytest.mark.gui

TYPED = "Something The Vocabulary Does Not Contain"


def _type_into(widget, value: str):
    """Put ``value`` in as TYPING does, not as selecting does.

    ``setEditText`` is what a keystroke ends up doing to an editable combo,
    and deliberately does not emit ``activated``. Using ``setCurrentText``
    here would test the selecting path and miss the reported bug entirely.
    """
    if isinstance(widget, QComboBox) and widget.isEditable():
        widget.setEditText(value)
        return value
    if isinstance(widget, QLineEdit):
        widget.setText(value)
        return value
    if hasattr(widget, "set_value"):
        widget.set_value([value])
        return [value]
    return None


def _expand_everything(tree: TemplateTree, nodes) -> None:
    """Sections build lazily, so nothing can be typed into a folded one."""
    for root in nodes:
        for node in root.walk():
            section = tree.section_widget(node.key)
            if section is not None:
                section.set_expanded(True)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    folder = root / "sub-01" / "eeg"
    folder.mkdir(parents=True)
    (folder / "sub-01_task-rest_eeg.edf").write_bytes(b"\0" * 16)
    (folder / "sub-01_task-rest_eeg.json").write_text(
        json.dumps({"TaskName": "rest"})
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "BIDSVersion": "1.10.0"})
    )
    return root


# ---------------------------------------------------------------------------
# The widget layer
# ---------------------------------------------------------------------------


def test_typing_into_a_vocabulary_field_is_read_back(qtbot) -> None:
    """The reported field. A curated vocabulary makes License an editable
    combo, and a value outside the list must still count."""
    node = TemplateNode(
        key="dd", label="Dataset", kind="file",
        section=dataset_description_section(),
    )
    tree = TemplateTree([node], values={}, suggestions=dict(CURATED_SUGGESTIONS))
    qtbot.addWidget(tree)
    _expand_everything(tree, [node])

    widget = tree._widgets["dd"]["License"]
    assert isinstance(widget, QComboBox) and widget.isEditable()
    assert TYPED not in [
        widget.itemText(i) for i in range(widget.count())
    ], "the point is a value the list does not offer"

    _type_into(widget, TYPED)
    assert tree.values_by_key()["dd"]["License"] == TYPED


def test_every_kind_of_control_carries_what_was_typed(qtbot) -> None:
    node = TemplateNode(
        key="dd", label="Dataset", kind="file",
        section=dataset_description_section(),
    )
    tree = TemplateTree([node], values={}, suggestions=dict(CURATED_SUGGESTIONS))
    qtbot.addWidget(tree)
    _expand_everything(tree, [node])

    wrote = {}
    for name, widget in tree._widgets["dd"].items():
        written = _type_into(widget, TYPED)
        if written is not None:
            wrote[name] = written
    assert len(wrote) >= 8, "the form should offer more than a handful"

    read_back = tree.values_by_key()["dd"]
    assert {
        name: read_back.get(name) for name in wrote
    } == wrote


# ---------------------------------------------------------------------------
# All the way to the file
# ---------------------------------------------------------------------------


def test_a_typed_dataset_field_reaches_dataset_description(
    qtbot, dataset: Path, tmp_path: Path,
) -> None:
    node = TemplateNode(
        key="dataset_description", label="Dataset", kind="file",
        section=dataset_description_section(),
    )
    tree = TemplateTree([node], values={}, suggestions=dict(CURATED_SUGGESTIONS))
    qtbot.addWidget(tree)
    _expand_everything(tree, [node])

    widget = tree._widgets["dataset_description"]["License"]
    _type_into(widget, TYPED)

    inventory = tmp_path / "inv.tsv"
    inventory.write_text("source_file\n")
    spec = RecordingMetaSpec()
    dataset_description_from_bids(
        spec.dataset_description,
        tree.values_by_key().get("dataset_description", {}),
    )
    scaffold_sidecar_path(inventory).write_text(dump_spec(spec))

    run_metadata(dataset, inventory_tsv=inventory, write_report=False)
    on_disk = json.loads(
        (dataset / "dataset_description.json").read_text()
    )
    assert on_disk["License"] == TYPED


def test_a_typed_sequence_template_field_reaches_the_sidecar(
    qtbot, dataset: Path, tmp_path: Path,
) -> None:
    """Including a field the schema gives a controlled vocabulary:
    RecordingType admits three values, and a user who types a fourth has
    still stated something, which the file must carry for the validator to
    be able to object to it."""
    nodes = build_template_tree([("eeg", "eeg")], bids_root=dataset)
    tree = TemplateTree(nodes, values={}, suggestions={})
    qtbot.addWidget(tree)
    _expand_everything(tree, nodes)

    wrote = {}
    for widgets in tree._widgets.values():
        for name in ("RecordingType", "InstitutionName", "EEGReference"):
            widget = widgets.get(name)
            if widget is None:
                continue
            written = _type_into(widget, f"typed-{name}")
            if written is not None:
                wrote[name] = written
    assert "RecordingType" in wrote, "the vocabulary field must be offered"

    inventory_frame = pd.DataFrame([{
        "source_file": "raw/a.edf", "modality": "eeg", "datatype": "eeg",
        "suffix": "eeg", "task": "rest", "include": "1",
        "proposed_basename": "sub-01_task-rest_eeg",
    }])
    spec = RecordingMetaSpec()
    spec.sequence_templates = dict(tree.values_by_key())
    apply_stated_metadata(dataset, spec, inventory_frame)

    on_disk = json.loads(
        (dataset / "sub-01" / "eeg" / "sub-01_task-rest_eeg.json").read_text()
    )
    for name, want in wrote.items():
        assert on_disk.get(name) == want, name


def test_selecting_and_typing_the_same_value_agree(
    qtbot, dataset: Path, tmp_path: Path,
) -> None:
    """The comparison the report is really about. Two runs, identical
    answers, one picked from the list and one typed by hand."""
    def run(pick: bool) -> dict:
        node = TemplateNode(
            key="dd", label="Dataset", kind="file",
            section=dataset_description_section(),
        )
        tree = TemplateTree(
            [node], values={}, suggestions=dict(CURATED_SUGGESTIONS),
        )
        qtbot.addWidget(tree)
        _expand_everything(tree, [node])
        widget = tree._widgets["dd"]["License"]
        offered = [widget.itemText(i) for i in range(widget.count()) if
                   widget.itemText(i)]
        value = offered[0]
        if pick:
            widget.setCurrentIndex(widget.findText(value))
            widget.activated.emit(widget.currentIndex())
        else:
            widget.setEditText(value)
        return tree.values_by_key().get("dd", {})

    assert run(pick=True).get("License") == run(pick=False).get("License")

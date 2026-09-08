"""Per-sequence templates and the VARIES sentinel.

Two additions to the metadata model, both about scope. A template says what a
CLASS of recordings shares, which is where most of what a study knows actually
lives. VARIES says a field's answer differs per recording, which the model
previously had no way to express at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidsmgr.fixups.sidecar_schema import apply_sequence_template, apply_stated_metadata
from bidsmgr.recording_meta import (
    AcquisitionSpec,
    VARIES,
    RecordingMetaSpec,
    is_varies,
    parse_template_key,
    resolve_sequence_template,
    template_key,
    validate_sequence_templates,
)


def _spec(**templates) -> RecordingMetaSpec:
    spec = RecordingMetaSpec()
    spec.sequence_templates = dict(templates)
    return spec


# ---------------------------------------------------------------------------
# Keys and resolution
# ---------------------------------------------------------------------------


def test_keys_round_trip() -> None:
    assert template_key("func", "bold") == "func/bold"
    assert template_key("func", "bold", "rest") == "func/bold@rest"
    assert parse_template_key("func/bold") == ("func", "bold", None)
    assert parse_template_key("func/bold@rest") == ("func", "bold", "rest")


def test_a_task_template_refines_the_general_one() -> None:
    """Least specific first: a study says what every bold run shares, then what
    is different about one task, rather than restating the whole thing."""
    spec = _spec(**{
        "func/bold": {"TaskDescription": "general", "Instructions": "Lie still"},
        "func/bold@rest": {"TaskDescription": "eyes open"},
    })
    general = resolve_sequence_template(spec, "func", "bold")
    assert general == {"TaskDescription": "general", "Instructions": "Lie still"}

    rest = resolve_sequence_template(spec, "func", "bold", "rest")
    assert rest["TaskDescription"] == "eyes open"   # refined
    assert rest["Instructions"] == "Lie still"      # inherited


def test_no_templates_resolves_to_nothing() -> None:
    assert resolve_sequence_template(None, "func", "bold") == {}
    assert resolve_sequence_template(RecordingMetaSpec(), "func", "bold") == {}


# ---------------------------------------------------------------------------
# Validation: a template cannot carry a field its datatype rejects
# ---------------------------------------------------------------------------


def test_a_field_the_datatype_rejects_is_reported() -> None:
    """The check that stops this layer becoming a second place to put a field
    where it does not belong."""
    spec = _spec(**{"anat/T2w": {"EEGReference": "Cz"}})
    problems = validate_sequence_templates(spec)
    assert len(problems) == 1
    assert "EEGReference" in problems[0] and "anat/T2w" in problems[0]


def test_a_malformed_key_is_reported_not_raised() -> None:
    """The scaffold is a file a user may hand-edit; one bad line must not stop
    the rest of it loading."""
    spec = _spec(**{"nonsense": {"TaskName": "x"}})
    problems = validate_sequence_templates(spec)
    assert problems and "nonsense" in problems[0]


def test_a_good_template_reports_nothing() -> None:
    spec = _spec(**{"func/bold": {"TaskDescription": "resting state"}})
    assert validate_sequence_templates(spec) == []


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------


def test_an_inapplicable_field_is_refused_however_plainly_asked_for() -> None:
    spec = _spec(**{"anat/T1w": {"EEGReference": "Cz", "PulseSequenceType": "MPRAGE"}})
    data: dict = {}
    assert apply_sequence_template(data, "anat", "T1w", None, spec) == 1
    assert data == {"PulseSequenceType": "MPRAGE"}


def test_a_stated_answer_replaces_what_the_converter_wrote(tmp_path: Path) -> None:
    """The user wins. This used to be the other way round.

    The reasoning was that a file's own header beats a statement about a class
    of files. That is wrong about who is talking: nothing in the chain is a
    guess, every layer of it is somebody having typed an answer, and the form
    only offers a field when it is worth asking about. A user who opens "already
    answered by the conversion" and corrects the manufacturer has said the
    header is wrong, and their correction used to vanish on the next run.
    """
    data = {"Manufacturer": "read from the header", "EchoTime": 0.03}
    spec = RecordingMetaSpec()
    spec.sequence_templates["anat/T1w"] = {"Manufacturer": "what the user says"}

    apply_sequence_template(data, "anat", "T1w", None, spec)
    assert data["Manufacturer"] == "what the user says"
    # And what nobody stated is left exactly as the converter wrote it.
    assert data["EchoTime"] == 0.03

def test_a_template_reaches_its_own_scope_and_nothing_else(tmp_path: Path) -> None:
    """The gate for this feature: a bold template reaches every bold run, the
    task-qualified one reaches only that task, and neither touches anat."""
    spec = _spec(**{
        "func/bold": {"Instructions": "Lie still"},
        "func/bold@rest": {"TaskDescription": "eyes open"},
        "anat/T1w": {"PulseSequenceType": "MPRAGE"},
    })
    root = tmp_path / "staging"
    for rel in (
        "sub-001/func/sub-001_task-rest_bold.json",
        "sub-001/func/sub-001_task-nback_bold.json",
        "sub-001/anat/sub-001_T1w.json",
    ):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    apply_stated_metadata(root, spec)

    rest = json.loads((root / "sub-001/func/sub-001_task-rest_bold.json").read_text())
    nback = json.loads((root / "sub-001/func/sub-001_task-nback_bold.json").read_text())
    anat = json.loads((root / "sub-001/anat/sub-001_T1w.json").read_text())

    assert rest["Instructions"] == "Lie still"
    assert rest["TaskDescription"] == "eyes open"
    assert nback["Instructions"] == "Lie still"
    assert "TaskDescription" not in nback
    assert anat == {"PulseSequenceType": "MPRAGE"}


# ---------------------------------------------------------------------------
# VARIES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    ("VARIES", True), ("varies", True), (" Varies ", True),
    ("Cz", False), ("", False), (None, False), (50, False),
])
def test_is_varies(value, expected) -> None:
    assert is_varies(value) is expected


def test_varies_is_never_written_to_a_sidecar(tmp_path: Path) -> None:
    """The entire point of the sentinel. It says where the answer lives; it is
    not the answer, and writing the word into a published sidecar would be
    worse than leaving the field out."""
    spec = _spec(**{"func/bold": {"TaskDescription": VARIES, "Instructions": "Lie still"}})
    spec.defaults.institution_name = VARIES

    root = tmp_path / "staging"
    p = root / "sub-001/func/sub-001_task-rest_bold.json"
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    apply_stated_metadata(root, spec)

    data = json.loads(p.read_text())
    assert data == {"Instructions": "Lie still"}
    assert "VARIES" not in json.dumps(data)



# ---------------------------------------------------------------------------
# EEG and MEG are different instruments
# ---------------------------------------------------------------------------


def test_each_modality_keeps_its_own_hardware() -> None:
    """REGRESSION: one shared block served eeg, meg, ieeg and nirs, so a study
    running a Brain Products amplifier AND an Elekta dewar could state only one
    manufacturer, and whichever it stated was claimed by both."""
    from bidsmgr.recording_meta import AcquisitionSpec, resolve_effective

    spec = RecordingMetaSpec()
    spec.defaults.institution_name = "Uni Oldenburg"   # the building: shared
    spec.defaults.power_line_freq = 50                 # the mains: shared
    spec.modality_defaults["eeg"] = AcquisitionSpec(
        manufacturer="Brain Products", cap_manufacturer="EasyCap",
    )
    spec.modality_defaults["meg"] = AcquisitionSpec(
        manufacturer="Elekta", dewar_position="upright",
    )

    eeg = resolve_effective(spec, "row", None, "eeg").acquisition
    meg = resolve_effective(spec, "row", None, "meg").acquisition

    assert eeg.manufacturer == "Brain Products"
    assert meg.manufacturer == "Elekta"
    assert eeg.cap_manufacturer == "EasyCap" and meg.cap_manufacturer is None
    assert meg.dewar_position == "upright" and eeg.dewar_position is None
    # What genuinely belongs to the study still reaches both.
    for acq in (eeg, meg):
        assert acq.institution_name == "Uni Oldenburg"
        assert acq.power_line_freq == 50


def test_a_scaffold_with_no_per_modality_block_behaves_as_before() -> None:
    """Every scaffold ever written has only the shared block, and must keep
    resolving exactly as it did."""
    from bidsmgr.recording_meta import resolve_effective

    spec = RecordingMetaSpec()
    spec.defaults.manufacturer = "Brain Products"
    for datatype in ("eeg", "meg", None):
        acq = resolve_effective(spec, "row", None, datatype).acquisition
        assert acq.manufacturer == "Brain Products"


# ---------------------------------------------------------------------------
# One chain, one answer
# ---------------------------------------------------------------------------


def test_the_same_field_stated_twice_has_a_defined_winner() -> None:
    """REGRESSION: the same fact could be stated as a spec attribute or as a
    BIDS field in a template, and two independent write passes applied them, so
    which one won depended on the order the fixups happened to run in. Nothing
    declared a precedence; swapping the passes flipped the answer."""
    from bidsmgr.recording_meta import AcquisitionSpec, resolve_sidecar_fields

    spec = RecordingMetaSpec()
    spec.modality_defaults["eeg"] = AcquisitionSpec(manufacturer="Brain Products")
    spec.sequence_templates = {"eeg/eeg": {"Manufacturer": "Elekta"}}

    resolved = resolve_sidecar_fields(spec, "eeg", "eeg")
    # More specific wins, and says so.
    assert resolved["Manufacturer"].value == "Elekta"
    assert resolved["Manufacturer"].origin == "template"


def test_every_layer_beats_the_one_above_it() -> None:
    from bidsmgr.recording_meta import AcquisitionSpec, resolve_sidecar_fields

    spec = RecordingMetaSpec()
    spec.defaults.manufacturer = "from dataset"
    spec.modality_defaults["eeg"] = AcquisitionSpec(manufacturer="from modality")
    spec.sequence_templates = {
        "eeg/eeg": {"Manufacturer": "from template"},
        "eeg/eeg@rest": {"Manufacturer": "from task template"},
    }
    spec.overrides["r1"] = AcquisitionSpec(manufacturer="from row")

    def winner(**kw):
        return resolve_sidecar_fields(spec, "eeg", "eeg", **kw)["Manufacturer"]

    assert winner().value == "from template"
    assert winner(task="rest").value == "from task template"
    assert winner(task="rest", row_id="r1").value == "from row"
    # The inventory cell is what the user can see while typing, so it wins.
    cell = winner(task="rest", row_id="r1", row_values={"manufacturer": "from cell"})
    assert cell.value == "from cell" and cell.origin == "cell"





# ---------------------------------------------------------------------------
# The cells a user types in the table
#
# These used to be applied by the EEG fixup during conversion, which walked the
# same chain as the template pass, so one field could be written twice and the
# winner depended on the order. There is one pass now, in the metadata step.
# ---------------------------------------------------------------------------


def test_a_row_cell_beats_a_dataset_default(tmp_path: Path) -> None:
    import pandas as pd
    from bidsmgr.fixups.sidecar_schema import apply_stated_metadata

    root = tmp_path / "ds"
    eeg = root / "sub-001" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "sub-001_task-rest_eeg.json").write_text(json.dumps({}))

    spec = RecordingMetaSpec()
    spec.defaults = AcquisitionSpec(eeg_reference="Cz", power_line_freq=60.0)
    inventory = pd.DataFrame([{
        "proposed_basename": "sub-001_task-rest_eeg",
        "source_file": "/raw/one.edf",
        "eeg_reference": "FCz",
        "line_freq": "50",
    }])

    apply_stated_metadata(root, spec, inventory)
    data = json.loads((eeg / "sub-001_task-rest_eeg.json").read_text())
    assert data["EEGReference"] == "FCz", "the cell the user typed has to win"
    # The table is text, and BIDS declares this field a number. Writing "50"
    # would fail validation for a field answered correctly.
    assert data["PowerLineFrequency"] == 50


def test_varies_is_not_written_from_a_cell_either(tmp_path: Path) -> None:
    import pandas as pd
    from bidsmgr.fixups.sidecar_schema import apply_stated_metadata

    root = tmp_path / "ds"
    eeg = root / "sub-001" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "sub-001_task-rest_eeg.json").write_text(json.dumps({}))

    spec = RecordingMetaSpec()
    spec.defaults = AcquisitionSpec(eeg_reference=VARIES)
    apply_stated_metadata(root, spec, pd.DataFrame([{
        "proposed_basename": "sub-001_task-rest_eeg", "source_file": "/raw/one.edf",
    }]))
    assert "EEGReference" not in json.loads(
        (eeg / "sub-001_task-rest_eeg.json").read_text()
    )

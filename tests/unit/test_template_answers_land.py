"""Every answer the metadata template accepts must reach a file.

The template asks its questions from the schema, but what it can STORE and what
gets WRITTEN were each decided elsewhere, so a field could be offered, typed
into, saved, and quietly dropped. Two ways that happened, both real:

* ``DatasetDescriptionSpec`` named nine fields because the CLI has nine flags.
  The schema declares twelve, the form offered twelve, and HEDVersion,
  DatasetLinks and Keywords went in and never came out.
* The agnostic answers reached ``dataset_description.json`` only through a
  separate ``bidsmgr-metadata`` run. Convert alone wrote none of them, so a user
  who filled in the form and pressed convert saw an empty file.

These compare the two lists rather than checking a handful of names, so a field
added to a future schema is covered without anyone remembering to add a case.
"""

from __future__ import annotations

import json
from pathlib import Path

from bidsmgr.cli._scaffold import ensure_dataset_description
from bidsmgr.metadata.template_plan import (
    build_template_tree,
    dataset_description_section,
    sidecar_section,
)
from bidsmgr.recording_meta import (
    DatasetDescriptionSpec,
    dataset_description_as_bids,
    dataset_description_from_bids,
)
from bidsmgr import schema as schema_mod


def _sample(field):
    """A value of the shape the schema declares for this field."""
    if field.enum:
        return field.enum[0]
    if field.type == "array":
        return [0] if field.item_type in ("number", "integer") else ["x"]
    if field.type in ("number", "integer"):
        return 1
    if field.type == "boolean":
        return True
    if field.type == "object":
        return {"k": "v"}
    return "stated"


def test_every_agnostic_answer_survives_a_save() -> None:
    offered = {f.name: _sample(f) for f in dataset_description_section().fields}
    spec = DatasetDescriptionSpec()
    dataset_description_from_bids(spec, offered)
    stored = dataset_description_as_bids(spec)
    assert set(stored) == set(offered), (
        f"discarded on save: {sorted(set(offered) - set(stored))}"
    )


def test_every_agnostic_answer_reaches_the_file_at_conversion(tmp_path: Path) -> None:
    """Convert alone, with no metadata run afterwards."""
    offered = {f.name: _sample(f) for f in dataset_description_section().fields}
    spec = DatasetDescriptionSpec()
    dataset_description_from_bids(spec, offered)

    root = tmp_path / "ds"
    root.mkdir()
    ensure_dataset_description(root, fields=dataset_description_as_bids(spec))

    written = json.loads((root / "dataset_description.json").read_text())
    missing = [name for name in offered if name not in written]
    assert not missing, f"never written: {missing}"


def test_a_stated_name_beats_the_folder_name(tmp_path: Path) -> None:
    """The form is the only place to say what a dataset is called, so saying it
    there has to win over the default taken from the directory."""
    root = tmp_path / "ds"
    root.mkdir()
    ensure_dataset_description(root)
    assert json.loads((root / "dataset_description.json").read_text())["Name"] == "ds"

    ensure_dataset_description(root, fields={"Name": "The Real Study"})
    assert (
        json.loads((root / "dataset_description.json").read_text())["Name"]
        == "The Real Study"
    )


def test_the_chain_accepts_every_field_the_form_offers() -> None:
    """The form asks bidsval what a file may carry; the chain asks it again per
    field before writing. If those two disagreed, an answer would be accepted by
    the form and silently refused by the writer."""
    pairs = [
        ("anat", "T1w"), ("func", "bold"), ("dwi", "dwi"), ("fmap", "epi"),
        ("eeg", "eeg"), ("meg", "meg"), ("ieeg", "ieeg"), ("pet", "pet"),
    ]
    refused: dict[str, list[str]] = {}
    for datatype, suffix in pairs:
        section = sidecar_section(datatype, suffix, include_derived=True)
        bad = [
            f.name for f in section.fields
            if not schema_mod.field_applies(f.name, datatype, suffix)
        ]
        if bad:
            refused[f"{datatype}/{suffix}"] = bad
    assert not refused, f"offered but unwritable: {refused}"


def test_every_file_node_stores_under_the_key_the_writer_reads() -> None:
    """A section's answers go to ``sequence_templates[storage_key]`` and the
    writer looks them up as ``<datatype>/<suffix>``. One typo apart and every
    per-file answer disappears."""
    from bidsmgr.recording_meta.templates import template_key

    tree = build_template_tree([("eeg", "eeg"), ("anat", "T1w"), ("pet", "pet")])
    for root in tree:
        for node in root.walk():
            if not node.section or node.section.datatype == "":
                continue
            assert node.key == template_key(
                node.section.datatype, node.section.suffix
            )


def test_a_correction_to_what_the_converter_wrote_survives(tmp_path: Path) -> None:
    """The whole reason "already answered by the conversion" is editable.

    A user who corrects the manufacturer there has said the header is wrong.
    Their answer used to be skipped, on the rule that a file's own header beats
    a statement about a class of files, so the correction vanished with no
    explanation on the next run.
    """
    import pandas as pd
    from bidsmgr.fixups.sidecar_schema import apply_stated_metadata
    from bidsmgr.recording_meta import RecordingMetaSpec

    root = tmp_path / "ds"
    eeg = root / "sub-001" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "sub-001_task-rest_eeg.json").write_text(json.dumps({
        "Manufacturer": "read from the header",
        "SamplingFrequency": 500.0,
    }))

    spec = RecordingMetaSpec()
    spec.sequence_templates["eeg/eeg"] = {"Manufacturer": "what the user says"}
    apply_stated_metadata(root, spec, pd.DataFrame([{
        "proposed_basename": "sub-001_task-rest_eeg", "source_file": "/raw/a.edf",
    }]))

    got = json.loads((eeg / "sub-001_task-rest_eeg.json").read_text())
    assert got["Manufacturer"] == "what the user says"
    # What nobody stated is left exactly as the converter wrote it.
    assert got["SamplingFrequency"] == 500.0


# ---------------------------------------------------------------------------
# Types
#
# Everything arrives as text somewhere: the inventory is a TSV and a combo box
# hands back whatever was typed. BIDS declares PowerLineFrequency a number, and
# the string "60" fails validation for a field answered correctly.
# ---------------------------------------------------------------------------


def test_a_value_takes_the_shape_the_standard_declares() -> None:
    from bidsmgr import schema as schema_mod

    cases = [
        ("PowerLineFrequency", "60", "eeg", "eeg", 60),
        ("HeadCircumference", "58.5", "eeg", "eeg", 58.5),
        ("EEGChannelCount", "64", "eeg", "eeg", 64),
        ("InjectedMass", "10", "pet", "pet", 10),
        ("ImageDecayCorrected", "true", "pet", "pet", True),
    ]
    for name, typed, datatype, suffix, want in cases:
        got = schema_mod.coerce(name, typed, datatype, suffix)
        assert got == want and type(got) is type(want), f"{name}: {got!r}"


def test_a_word_the_field_accepts_is_left_alone() -> None:
    """Several of these fields take a number OR "n/a"."""
    from bidsmgr import schema as schema_mod

    assert schema_mod.coerce("PowerLineFrequency", "n/a", "eeg", "eeg") == "n/a"
    assert schema_mod.coerce("EEGReference", "average", "eeg", "eeg") == "average"


def test_a_string_field_that_looks_numeric_stays_a_string() -> None:
    """A run label of "01" is not the number 1."""
    from bidsmgr import schema as schema_mod

    assert schema_mod.coerce("TaskName", "01", "eeg", "eeg") == "01"


def test_a_cell_typed_in_the_table_reaches_the_sidecar_as_a_number(tmp_path) -> None:
    import pandas as pd
    from bidsmgr.fixups.sidecar_schema import apply_stated_metadata
    from bidsmgr.recording_meta import RecordingMetaSpec

    root = tmp_path / "ds"
    eeg = root / "sub-001" / "eeg"
    eeg.mkdir(parents=True)
    (eeg / "sub-001_task-rest_eeg.json").write_text(json.dumps({}))

    apply_stated_metadata(root, RecordingMetaSpec(), pd.DataFrame([{
        "proposed_basename": "sub-001_task-rest_eeg",
        "source_file": "/raw/a.edf",
        "line_freq": "50",
    }]))
    written = json.loads((eeg / "sub-001_task-rest_eeg.json").read_text())
    assert written["PowerLineFrequency"] == 50
    assert not isinstance(written["PowerLineFrequency"], str)


def test_a_row_resolves_its_own_cells_with_no_dataset_metadata_at_all() -> None:
    """A user who typed a line frequency into the table and never opened the
    dataset dialog used to get nothing: the chain bailed out on a missing spec
    before it ever looked at the row."""
    from bidsmgr.recording_meta import resolve_sidecar_fields

    resolved = resolve_sidecar_fields(
        None, "eeg", "eeg", row_values={"power_line_freq": "50"},
    )
    assert resolved["PowerLineFrequency"].value == 50


def test_a_numeric_array_stored_as_text_is_repaired_when_it_is_applied(
    tmp_path: Path,
) -> None:
    """Answers are PERSISTED, so a bad one keeps arriving until it is repaired.

    A template saved before the form could read numeric arrays holds ``["0"]``
    on disk, and that file is applied again on every later run. Five PET fields
    reached a validated dataset as arrays of decimal STRINGS this way, three of
    them values the conversion had written correctly and the form only echoed
    back. Repairing on apply means the stored answer does not have to be typed
    in again to become valid, and it covers a hand-edited template too.
    """
    import pandas as pd
    from bidsmgr.fixups.sidecar_schema import apply_stated_metadata
    from bidsmgr.recording_meta import RecordingMetaSpec

    root = tmp_path / "ds"
    pet = root / "sub-001" / "pet"
    pet.mkdir(parents=True)
    (pet / "sub-001_pet.json").write_text(json.dumps({"Units": "Bq/mL"}))

    spec = RecordingMetaSpec()
    spec.sequence_templates["pet/pet"] = {
        "FrameTimesStart": ["0"],
        "FrameDuration": ["14400"],
        "ScatterFraction": ["1.90358e-08"],
        "ReconMethodParameterValues": ["0"],
        # An array of strings is a different thing and must survive untouched.
        "ReconMethodParameterLabels": ["none"],
    }
    apply_stated_metadata(root, spec, pd.DataFrame([{
        "proposed_basename": "sub-001_pet", "source_file": "/raw/a.dcm",
    }]))

    got = json.loads((pet / "sub-001_pet.json").read_text())
    assert got["FrameTimesStart"] == [0]
    assert got["FrameDuration"] == [14400]
    assert got["ScatterFraction"] == [1.90358e-08]
    assert got["ReconMethodParameterValues"] == [0]
    assert got["ReconMethodParameterLabels"] == ["none"]
    assert all(isinstance(x, (int, float)) for x in got["FrameTimesStart"])


def test_a_word_in_a_numeric_array_is_left_alone(tmp_path: Path) -> None:
    """Coercion is only for items that spell a number. Anything else is an
    answer the user meant, and mangling it would be worse than leaving it."""
    import pandas as pd
    from bidsmgr.fixups.sidecar_schema import apply_stated_metadata
    from bidsmgr.recording_meta import RecordingMetaSpec

    root = tmp_path / "ds"
    pet = root / "sub-001" / "pet"
    pet.mkdir(parents=True)
    (pet / "sub-001_pet.json").write_text(json.dumps({"Units": "Bq/mL"}))

    spec = RecordingMetaSpec()
    spec.sequence_templates["pet/pet"] = {"FrameTimesStart": ["n/a"]}
    apply_stated_metadata(root, spec, pd.DataFrame([{
        "proposed_basename": "sub-001_pet", "source_file": "/raw/a.dcm",
    }]))

    assert json.loads((pet / "sub-001_pet.json").read_text())["FrameTimesStart"] == ["n/a"]


def test_the_schema_says_what_an_array_holds() -> None:
    """``field_metadata`` reported only "array", which cannot tell a list of
    frame times from a list of parameter names, so nothing downstream could
    repair one without guessing."""
    assert schema_mod.field_metadata("FrameTimesStart").item_type == "number"
    assert schema_mod.field_metadata("ReconMethodParameterValues").item_type == "number"
    assert schema_mod.field_metadata("ReconMethodParameterLabels").item_type == "string"
    assert schema_mod.field_metadata("TaskName").item_type == ""

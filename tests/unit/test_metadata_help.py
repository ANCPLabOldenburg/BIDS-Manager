"""The recording-metadata tooltips are sourced from the live BIDS schema,
which also confirms every exposed field is a genuine schema key."""

from __future__ import annotations

import pytest

from bidsmgr.gui.metadata_help import bids_tooltip, tooltip_for


@pytest.mark.parametrize(
    "ui_key, must_contain",
    [
        ("manufacturer", "Manufacturer"),
        ("line_freq", "Power"),               # PowerLineFrequency
        ("eeg_reference", "reference"),
        ("eeg_ground", "ground"),
        ("cap_manufacturer", "cap"),
        ("dewar_position", "dewar"),
        ("associated_empty_room", "empty"),
        ("subject_artefact_description", "artifact"),
    ],
)
def test_tooltip_from_schema(ui_key, must_contain):
    tip = tooltip_for(ui_key)
    assert tip, f"no tooltip for {ui_key}"
    assert must_contain.lower() in tip.lower()


def test_tooltip_markdown_stripped():
    # CapManufacturer's schema description has a backticked example.
    tip = bids_tooltip("CapManufacturer")
    assert "`" not in tip and tip


def test_fallback_for_non_metadata_fields():
    # montage is a convention (electrodes.tsv), demographics are participant cols.
    assert "electrodes.tsv" in tooltip_for("montage")
    assert "participants.tsv" in tooltip_for("PatientSex")
    assert "participants.tsv" in tooltip_for("Handedness")


def test_unknown_field_returns_empty():
    assert tooltip_for("definitely_not_a_field") == ""


# -- schema_summary: a schema description as a tooltip shows it -----------

from bidsmgr import schema as _schema  # noqa: E402
from bidsmgr.gui.metadata_help import schema_summary  # noqa: E402


def _entity_descriptions() -> list[tuple[str, str]]:
    out = []
    for key in _schema.entity_keys():
        try:
            out.append((key, _schema.entity_key_info(key).description))
        except KeyError:
            continue
    return out


def test_summary_keeps_whole_sentences_within_the_limit():
    text = ("First sentence here. Second sentence, a little longer than the first. "
            "Third sentence that will not fit.")
    s = schema_summary(text, limit=80)
    assert s.replace("\n", " ") == "First sentence here. Second sentence, a little longer than the first."


def test_summary_always_keeps_the_first_sentence():
    long = "A first sentence " + "that goes on " * 40 + "and ends here. Then more."
    s = schema_summary(long, limit=50).replace("\n", " ")
    assert s.endswith("and ends here.")
    assert "Then more" not in s


def test_summary_strips_markdown_and_wraps():
    text = ("They **MUST** be indexed with the [`run-<index>`](SPEC_ROOT/appendices/entities.md#run) "
            "entity: `_run-1`, `_run-2`, and so on.")
    s = schema_summary(text)
    assert "`" not in s and "**" not in s and "](" not in s
    assert "run-<index>" in s
    assert all(len(line) <= 72 for line in s.splitlines())


@pytest.mark.parametrize("key, description", _entity_descriptions())
def test_every_entity_summary_reads_as_finished_text(key, description):
    """The Settings > Scan tooltips cut at 300 characters ended mid-word
    ("MUST be indexed w") with their Markdown showing; every entity's
    summary now ends a sentence and carries no Markdown."""
    s = schema_summary(description)
    assert s, key
    flat = s.replace("\n", " ")
    assert flat[-1] in '.!?)"', (key, flat[-40:])
    assert "`" not in flat and "](" not in flat and "**" not in flat, key
    assert all(len(line) <= 72 for line in s.splitlines()), key

"""What a validator finding means, in a sentence a person can act on.

A finding already carries the rule it came from, which lets a user check the
claim. It does not say what the rule is FOR, and the codes are written for
implementers: ``TSV_ADDITIONAL_COLUMNS_UNDEFINED`` is precise and tells a
first-time user nothing.

This is BIDS Manager's own prose, not the standard's, and it is deliberately
short: what the rule is protecting against, and what to do. Anything longer
belongs in the specification, which every entry can be checked against.

A code with no entry gets nothing rather than a guess. Silence is better than
a plausible sentence that turns out to describe a different rule.

Qt-free.
"""

from __future__ import annotations

from typing import Optional

# Keyed by bidsval's issue code. Each entry is (what it means, what to do).
_HELP: dict[str, tuple[str, str]] = {
    "INVALID_LOCATION": (
        "The file is named correctly but sits in a folder that is not a BIDS "
        "datatype, so tools that read inside datatype folders will never see "
        "it.",
        "Move it into the datatype folder it belongs to, or add it to "
        ".bidsignore if it is not part of the dataset.",
    ),
    "DATATYPE_MISMATCH": (
        "The suffix says one datatype and the folder says another.",
        "Either the file is in the wrong folder or its suffix is wrong. The "
        "suffix is usually the thing to trust.",
    ),
    "ENTITY_NOT_IN_RULE": (
        "The filename carries an entity the standard does not allow for this "
        "kind of file. An entity that is not allowed is not read by anything, "
        "so the information in it is lost.",
        "Remove the entity, or use one the standard defines for this "
        "datatype and suffix.",
    ),
    "MISSING_REQUIRED_ENTITY": (
        "An entity the standard requires for this kind of file is absent, so "
        "the file cannot be matched to the thing it belongs to.",
        "Rename the file to include it. The Rename button can do this across "
        "the dataset.",
    ),
    "ENTITY_VALUE_INVALID": (
        "An entity's value is not a valid BIDS label. Labels are letters and "
        "digits only.",
        "Rename it without spaces, hyphens or underscores.",
    ),
    "SIDECAR_KEY_REQUIRED": (
        "The standard requires this field for this kind of file, and nothing "
        "can derive it from the data.",
        "State it in the sidecar, or in a shared sidecar higher up if it is "
        "the same for every recording.",
    ),
    "SIDECAR_KEY_RECOMMENDED": (
        "Not an error. The standard recommends this field because analyses "
        "commonly need it and nobody can recover it later.",
        "Fill it in while you still know the answer.",
    ),
    "JSON_SCHEMA_VALIDATION_ERROR": (
        "A field holds a value of the wrong type: a number written as text, "
        "or a single value where a list is expected.",
        "The grouped view can repair the shape across every file at once "
        "without changing what the value says.",
    ),
    "TSV_COLUMN_MISSING": (
        "A column the standard requires for this table is not there.",
        "Add it. The grouped view can add it to every affected table, filled "
        "with n/a.",
    ),
    "TSV_COLUMN_ORDER_INCORRECT": (
        "The columns the standard knows about are not in the order it "
        "defines. Nothing is lost, but tools that read by position will "
        "misread the table.",
        "Reordering moves no data and is safe to apply everywhere.",
    ),
    "TSV_VALUE_INCORRECT_TYPE": (
        "A cell holds a value that is not the type its column declares, most "
        "often text in a numeric column.",
        "Repair the cells that can be converted; the ones that cannot need a "
        "human.",
    ),
    "TSV_ADDITIONAL_COLUMNS_UNDEFINED": (
        "The table has a column the standard does not define and the "
        "accompanying .json does not describe, so a reader has no way to know "
        "what it holds.",
        "Describe the column in the sidecar beside the table.",
    ),
    "TSV_INDEX_VALUE_NOT_UNIQUE": (
        "Two rows claim the same identifier, so anything joining on it will "
        "silently pick one.",
        "Make the identifiers unique, or remove the duplicate row.",
    ),
    "SIDECAR_WITHOUT_DATAFILE": (
        "A sidecar describes a data file that is not there. It is invisible "
        "to every tool, and usually means the data file was renamed or "
        "deleted and its sidecar was left behind.",
        "Delete it, or restore the file it describes.",
    ),
    "CASE_COLLISION": (
        "Two paths differ only by letter case. On macOS and Windows they are "
        "the same file, so the dataset cannot be copied there intact.",
        "Rename one of them.",
    ),
    "UNUSED_STIMULUS": (
        "A file in stimuli/ is referenced by no events table, so nothing in "
        "the dataset says when it was shown.",
        "Reference it from the events table, or remove it.",
    ),
    "EMPTY_FILE": (
        "The file is zero bytes. It passes a name check and holds nothing.",
        "Reconvert it or remove it.",
    ),
    "NIFTI_HEADER_UNREADABLE": (
        "The image header could not be read, which usually means the file is "
        "truncated or was not fully written.",
        "Reconvert from the source.",
    ),
    "AUTHORS_AND_CITATION_FILE_MUTUALLY_EXCLUSIVE": (
        "BIDS treats CITATION.cff as the single source of authorship. Having "
        "Authors in dataset_description.json as well means the two can "
        "disagree about who made the dataset.",
        "Fix ups can move the fields into the citation file for you.",
    ),
    "bidsmgr.todo_placeholder": (
        "BIDS Manager wrote TODO here because nothing could answer the field "
        "automatically. It is a note to yourself, not a value.",
        "Replace it, or turn the TODO warnings off in Settings if you are not "
        "ready to.",
    ),
}


def explain(rule_id: Optional[str]) -> Optional[tuple[str, str]]:
    """``(meaning, what to do)`` for a rule, or ``None`` if we have no entry."""
    if not rule_id:
        return None
    return _HELP.get(rule_id) or _HELP.get(rule_id.upper())


def as_text(rule_id: Optional[str]) -> str:
    """The explanation as one block, for a tooltip. Empty when unknown."""
    found = explain(rule_id)
    if not found:
        return ""
    meaning, action = found
    return f"{meaning}\n\n{action}"


def known_rules() -> tuple[str, ...]:
    return tuple(sorted(_HELP))


__all__ = ["as_text", "explain", "known_rules"]

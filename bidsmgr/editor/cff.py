"""The Citation File Format, as fields the Editor can render and check.

``CITATION.cff`` is not part of the BIDS schema, so guard 8 does not apply to
it: there is no authority to adapt, and the fields below are BIDS Manager's own
curated view of CFF 1.2.0. What IS schema-driven is where the values come from.
Everything BIDS already knows about a dataset is read from
``dataset_description.json``, whose fields the schema defines, so the citation
file restates the standard rather than asking the user again.

The field list is deliberately a useful subset, not the whole format. CFF 1.2.0
has around forty keys and most of them describe software releases; a dataset
citation needs a dozen. The rest of the file is preserved untouched when the
Editor saves, so a user who hand-writes a key this module has never heard of
does not lose it.

Parsing and writing go through PyYAML rather than string handling, because a
title with a colon in it is not a corner case.

Qt-free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

CFF_VERSION = "1.2.0"
FILENAME = "CITATION.cff"

# What the format always requires. bidsval checks exactly these three, so a
# file missing one is an error every validator will agree on.
REQUIRED = ("cff-version", "message", "title")


@dataclass(frozen=True)
class CffField:
    """One key the Editor offers, and what it is for."""

    name: str
    level: str              # "required" | "recommended" | "optional"
    kind: str               # "string" | "array" | "people" | "date"
    description: str
    # The ``dataset_description.json`` key this is derived from, when it is.
    # Shown in the form so a user can see why a value is already filled in.
    from_bids: Optional[str] = None
    example: str = ""


# Ordered as the form shows them: what the format demands, then what makes a
# citation actually usable, then the rest.
FIELDS: tuple[CffField, ...] = (
    CffField(
        "cff-version", "required", "string",
        "Which version of the Citation File Format this file follows.",
        example=CFF_VERSION,
    ),
    CffField(
        "message", "required", "string",
        "What a reader should do with this file. Shown by tools that "
        "surface citations.",
        example="If you use this dataset, please cite it as below.",
    ),
    CffField(
        "title", "required", "string",
        "The title of the dataset, as it should appear in a citation.",
        from_bids="Name",
    ),
    CffField(
        "authors", "required", "people",
        "Who should be credited. One entry per person, family name and "
        "given names apart.",
        from_bids="Authors",
    ),
    CffField(
        "type", "recommended", "string",
        "What is being cited. For a BIDS dataset this is 'dataset'.",
        example="dataset",
    ),
    CffField(
        "license", "recommended", "string",
        "SPDX identifier of the licence the data is released under.",
        from_bids="License", example="CC0-1.0",
    ),
    CffField(
        "doi", "recommended", "string",
        "The dataset's DOI, without the resolver prefix.",
        from_bids="DatasetDOI", example="10.18112/openneuro.ds000000.v1.0.0",
    ),
    CffField(
        "version", "recommended", "string",
        "The version of the dataset this citation describes.",
        example="1.0.0",
    ),
    CffField(
        "date-released", "recommended", "date",
        "When this version was released, as YYYY-MM-DD.",
    ),
    CffField(
        "url", "optional", "string",
        "A landing page for the dataset.",
        from_bids="ReferencesAndLinks",
    ),
    CffField(
        "repository-code", "optional", "string",
        "Where the code that produced or analyses the dataset lives.",
    ),
    CffField(
        "abstract", "optional", "string",
        "A short description of the dataset.",
        from_bids="HowToAcknowledge",
    ),
    CffField(
        "keywords", "optional", "array",
        "Terms a reader might search for. One per entry.",
    ),
    CffField(
        "contact", "optional", "people",
        "Who to approach with questions, if not the authors.",
    ),
    CffField(
        "identifiers", "optional", "array",
        "Other identifiers for this dataset, such as an accession number.",
    ),
)

FIELDS_BY_NAME: dict[str, CffField] = {f.name: f for f in FIELDS}

# The only BIDS key the citation file takes OWNERSHIP of.
#
# Measured against the validator, one field at a time, with a CITATION.cff
# present:
#
#   Authors             -> AUTHORS_AND_CITATION_FILE_MUTUALLY_EXCLUSIVE
#   License             -> no finding
#   HowToAcknowledge    -> no finding
#   ReferencesAndLinks  -> no finding
#
# This used to move all four. That was wrong, and it was destructive in a way
# a user could not see: somebody typed MIT into the licence field, the
# conversion wrote it into dataset_description.json correctly, and writing the
# citation file then deleted it from there. The standard forbids duplicating
# ONE field, so exactly one is moved. The other three are COPIED into the
# citation file, which is what makes it a complete citation, and left where
# they were.
MOVED_FIELDS = ("Authors",)

# Copied into the citation file and kept in dataset_description.json too. The
# citation file is a rendering of them, not their new home.
COPIED_FIELDS = ("HowToAcknowledge", "License", "ReferencesAndLinks")


def split_person(name: str) -> dict[str, str]:
    """``"Lopez Vilaret, Karel"`` or ``"Karel Lopez Vilaret"`` into CFF parts.

    BIDS stores an author as free text and CFF wants the parts apart. A comma
    is the reliable separator; without one the last word is taken as the family
    name, which is right far more often than not and is visible in the file for
    anyone it is wrong for.
    """
    name = str(name).strip()
    if "," in name:
        family, _, given = name.partition(",")
        out = {"family-names": family.strip()}
        if given.strip():
            out["given-names"] = given.strip()
        return out
    parts = name.split()
    if len(parts) <= 1:
        return {"family-names": name}
    return {"family-names": parts[-1], "given-names": " ".join(parts[:-1])}


def join_person(entry: Any) -> str:
    """A CFF person back to the single string BIDS uses."""
    if not isinstance(entry, dict):
        return str(entry)
    family = str(entry.get("family-names", "")).strip()
    given = str(entry.get("given-names", "")).strip()
    if family and given:
        return f"{family}, {given}"
    return family or given or str(entry)


def _strip_doi(value: str) -> str:
    text = str(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def from_dataset_description(description: dict[str, Any]) -> dict[str, Any]:
    """Build the citation mapping from what the dataset already states.

    Only the fields the standard defines are read, and each is read by its
    BIDS name, so a change in the schema shows up here rather than being
    silently ignored.
    """
    out: dict[str, Any] = {
        "cff-version": CFF_VERSION,
        "message": "If you use this dataset, please cite it as below.",
        "type": "dataset",
        "title": description.get("Name") or "TODO",
    }

    authors = description.get("Authors") or []
    if isinstance(authors, str):
        authors = [authors]
    if authors:
        out["authors"] = [split_person(a) for a in authors]
    else:
        # No authors is the commonest defect in the datasets this tool
        # produces, and a citation file is not the place to paper over it.
        out["authors"] = [{"family-names": "TODO", "given-names": "TODO"}]

    if description.get("License"):
        out["license"] = str(description["License"])
    if description.get("DatasetDOI"):
        out["doi"] = _strip_doi(description["DatasetDOI"])
    if description.get("HowToAcknowledge"):
        out["abstract"] = str(description["HowToAcknowledge"])

    refs = description.get("ReferencesAndLinks") or []
    if isinstance(refs, str):
        refs = [refs]
    urls = [str(r) for r in refs if str(r).startswith("http")]
    if urls:
        out["url"] = urls[0]
    return out


# --------------------------------------------------------------------------
# Reading and writing


def load(path: Path) -> Optional[dict[str, Any]]:
    """Parse a ``.cff`` file. ``None`` when it is unreadable or not a mapping."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - declared dependency
        return None
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        log.debug("cannot read %s: %s", path, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - yaml raises its own family
        log.debug("cannot parse %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def dumps(data: dict[str, Any], *, header: bool = True) -> str:
    """Serialise a citation mapping to YAML, in the field order the form uses.

    Keys this module does not know about are kept, at the end, so a
    hand-written extra survives a round trip through the Editor.
    """
    import yaml

    ordered: dict[str, Any] = {}
    for field in FIELDS:
        if field.name in data:
            ordered[field.name] = data[field.name]
    for key in data:
        if key not in ordered:
            ordered[key] = data[key]

    body = yaml.safe_dump(
        ordered, sort_keys=False, allow_unicode=True, default_flow_style=False,
    )
    if not header:
        return body
    return (
        "# Written by BIDS Manager from dataset_description.json.\n"
        "# Edit here or in the Editor; both write the same file.\n"
    ) + body


def missing_required(data: dict[str, Any]) -> list[str]:
    """Which of the three keys every CFF file must carry are absent."""
    return [k for k in REQUIRED if not data.get(k)]


__all__ = [
    "CFF_VERSION",
    "FIELDS",
    "FIELDS_BY_NAME",
    "FILENAME",
    "COPIED_FIELDS",
    "MOVED_FIELDS",
    "REQUIRED",
    "CffField",
    "dumps",
    "from_dataset_description",
    "join_person",
    "load",
    "missing_required",
    "split_person",
]

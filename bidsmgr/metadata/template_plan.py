"""What the metadata template should ask, arranged as scope, file, fields.

The dialog used to be hand-built groups: one "Acquisition system" box covering
EEG and MEG together with a single manufacturer between them, and targets
written as ``sub-..._<datatype>.json``. That is wrong in three ways at once. EEG
and MEG are different instruments and a study may run both. A user answering a
form is really answering "what will be in the file that comes out", so the file
is the unit, not the setting. And which fields are mandatory is the standard's
answer at the version in use, not something to assert in code.

This module builds that structure from three inputs and no hand-written lists:

* the **scan**, which says which files a conversion will actually produce, so
  an EEG-only study is never asked about dewar positions;
* the **schema**, which says which fields each file may carry, at what level,
  of what type, with which vocabulary, for the selected BIDS version;
* the **measured** set of what the converters fill by themselves, so the form
  asks only for what the data cannot answer.

Qt-free on purpose: the dialog renders this, the CLI can print it, and it can
be tested without a screen.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from .. import schema as schema_mod
from .derivable import CONVERTER_PRIVATE, derived_fields, determined_fields

# Where a section's answers are kept. The scaffold already has these blocks;
# the template is a view onto them rather than a new file.
STORAGE_DATASET_DESCRIPTION = "dataset_description"
STORAGE_SEQUENCE_TEMPLATE = "sequence_template"

_LEVEL_ORDER = {"required": 0, "recommended": 1, "optional": 2, "deprecated": 3}


@dataclass(frozen=True)
class TemplateField:
    """One question the template asks, and everything needed to render it."""

    name: str
    level: str
    type: str = ""
    item_type: str = ""
    enum: tuple[Any, ...] = ()
    unit: str = ""
    description: str = ""
    conditional: bool = False
    # Why this field is being asked about, which is not the same question as
    # what it means. Either no backend ever supplies it, or one usually does
    # and this dataset's files do not carry it. The second case is worth saying
    # out loud: it often means the answer exists at the scanner, or that an
    # anonymiser stripped it, and the user can go and get it.
    origin: str = "not in the data"

    @property
    def is_required(self) -> bool:
        return self.level == "required"

    @property
    def placeholder(self) -> str:
        """An example of the shape an answer takes, not a value to accept.

        A user faced with an empty box cannot tell whether a field wants one
        string, a list of them, or a number in particular units. The schema
        knows, so the box can say.
        """
        if self.enum:
            return " / ".join(str(v) for v in self.enum[:4]) + (
                " ..." if len(self.enum) > 4 else ""
            )
        if self.type == "array":
            inner = {"string": '"text"', "number": "0", "integer": "0"}.get(
                self.item_type, "...",
            )
            return f"[{inner}, {inner}]"
        if self.type in ("number", "integer"):
            return f"a number{f' in {self.unit}' if self.unit else ''}"
        if self.type == "boolean":
            return "true / false"
        if self.type == "object":
            return '{"key": "value"}'
        return "text"


@dataclass(frozen=True)
class TemplateSection:
    """One file's worth of questions."""

    scope: str            # "agnostic", or the datatype
    datatype: str         # "" for agnostic sections
    suffix: str
    title: str
    target: str           # one real path, as an example of what this covers
    storage: str
    storage_key: str = ""
    fields: tuple[TemplateField, ...] = field(default_factory=tuple)
    # How many files of this kind the scan found. A section is a statement
    # about a CLASS of files, and saying how many it covers is the difference
    # between "here is a form for one file" and "answer this once for all 61".
    n_files: int = 0
    # Fields NOT asked about because the conversion supplies them. Carried so
    # the form can still show them, folded away and editable, rather than
    # dropping them: a field that simply vanishes leaves a user who wants to
    # correct it with nowhere to go, and no way to tell it was considered.
    supplied: tuple[str, ...] = ()
    # EVERY field this file may carry, asked or not, with its level.
    #
    # The folded block used to look its fields up in ``fields``, which holds
    # only the ones being ASKED, so it never found them and fell back to a
    # stand-in with no level at all. A user could not see that the conversion
    # had answered something BIDS requires.
    declared: tuple[TemplateField, ...] = field(default_factory=tuple)

    @property
    def required_fields(self) -> tuple[TemplateField, ...]:
        return tuple(f for f in self.fields if f.is_required)


# How a field's absence reads to a user.
ORIGIN_ABSENT = "not in the data"
ORIGIN_USUALLY_DERIVED = "usually read from the data, but missing here"


def _as_template_field(spec, origin: str = ORIGIN_ABSENT) -> TemplateField:
    return TemplateField(
        name=spec.name,
        level=spec.level,
        type=spec.type,
        item_type=getattr(spec, "item_type", ""),
        enum=tuple(spec.enum),
        unit=spec.unit,
        description=spec.description,
        conditional=spec.conditional,
        origin=origin,
    )


def _sorted_fields(fields: list[TemplateField]) -> tuple[TemplateField, ...]:
    return tuple(sorted(fields, key=lambda f: (_LEVEL_ORDER.get(f.level, 9), f.name)))


def dataset_description_section(bids_root=None) -> TemplateSection:
    """The agnostic section: who made the dataset and under what terms.

    Nothing here can be read from a recording. It is also the section whose
    absence produced NO_AUTHORS on every dataset the tool has ever written.
    """
    # Only a schema that cannot be read is tolerated here. Swallowing
    # everything hid a missing function once and produced a silently empty
    # form, which looks exactly like "this dataset needs nothing".
    try:
        specs = schema_mod.dataset_description_fields(bids_root)
    except (KeyError, ValueError, OSError):
        specs = []
    fields = [
        _as_template_field(s)
        for s in specs
        # Written by the tool itself: the version it validates against and the
        # provenance chain. Asking would invite a wrong answer.
        if s.name not in ("BIDSVersion", "GeneratedBy", "DatasetType", "SourceDatasets")
        # Speculative here means the rule hinges on another file: Genetics is
        # required only of a dataset shipping genetic_info.json. Demanding it
        # of everyone sends the user hunting for a field they must not add.
        and not s.speculative
    ]
    return TemplateSection(
        scope="agnostic",
        datatype="",
        suffix="description",
        title="Dataset description",
        target="dataset_description.json",
        storage=STORAGE_DATASET_DESCRIPTION,
        fields=_sorted_fields(fields),
    )


def sidecar_section(
    datatype: str,
    suffix: str,
    example_path: str = "",
    bids_root=None,
    include_derived: bool = False,
    answered: Optional[dict] = None,
    n_files: int = 0,
    also_ask: Sequence[str] = (),
) -> TemplateSection:
    """The questions for one kind of sidecar, e.g. every ``*_eeg.json``.

    ``include_derived`` keeps the fields a conversion fills by itself, which is
    useful for showing a user the whole picture but is not what the form asks
    for by default.

    ``answered`` is what the scan observed the conversion actually producing FOR
    THIS DATASET, and it is the better authority: ``derivable`` was measured on
    one tree with one scanner, so a field a user's scanner supplies but ours did
    not was being asked for AND reported as already filled in, in the same form.
    """
    try:
        specs = schema_mod.sidecar_fields(datatype, suffix, bids_root)
    except (KeyError, ValueError, OSError):
        specs = []

    # What the conversion answers, and so what this form must not ask for.
    #
    # When the scan measured it on THIS dataset, that measurement is the whole
    # authority. The built-in list is only for a scan that could not measure,
    # and it is wrong in both directions by nature: it was taken once, from one
    # tree, with one scanner and one set of file formats. It claimed mne-bids
    # answers EEGReference, which mne-bids writes as "n/a" because BIDS demands
    # the key, and it missed five fields per MRI kind that dcm2niix does fill.
    # Never asked, whatever else is true: we write these from what was actually
    # attached, so a stated answer could only contradict the file beside it.
    determined = determined_fields(datatype, suffix)

    if include_derived:
        skip: set = set(determined)
    elif answered:
        skip = {
            name for name, value in answered.items()
            if value not in (None, "", [], {})
        } | CONVERTER_PRIVATE | determined
    else:
        skip = derived_fields(datatype) | CONVERTER_PRIVATE | determined
    # A field a converter USUALLY supplies but this dataset's files do not
    # carry. Worth saying so: it usually means the value exists at the scanner,
    # or that an anonymiser removed it, either of which the user can act on.
    usually = derived_fields(datatype)
    fields = [
        _as_template_field(
            s,
            ORIGIN_USUALLY_DERIVED if s.name in usually else ORIGIN_ABSENT,
        )
        for s in specs
        # A speculative requirement is one the schema imposes only in a
        # scenario this file may not be in; the form must not demand it.
        if s.name not in skip
        and (not s.speculative or s.name in also_ask)
        and s.level != "prohibited"
    ]
    declared_names = {spec.name for spec in specs}
    return TemplateSection(
        scope=datatype,
        datatype=datatype,
        suffix=suffix,
        title=f"{datatype.upper()}  ·  {suffix}",
        target=example_path or _pattern_for(datatype, suffix),
        storage=STORAGE_SEQUENCE_TEMPLATE,
        storage_key=f"{datatype}/{suffix}",
        fields=_sorted_fields(fields),
        n_files=n_files,
        supplied=tuple(sorted((skip & declared_names) - CONVERTER_PRIVATE)),
        declared=tuple(
            _as_template_field(spec)
            for spec in specs
            if spec.level != "prohibited"
        ),
    )


def _pattern_for(datatype: str, suffix: str) -> str:
    """The name this file takes, spelled out, when no scan row supplies one.

    Built from the entities the schema REQUIRES for the pair, so a functional
    run reads ``sub-<label>_task-<label>_bold.json`` rather than
    ``sub-<label>_bold.json``, which is not a name BIDS accepts. Half a name is
    worse than none: it looks like a real answer.
    """
    try:
        entities = schema_mod.required_entities(datatype, suffix)
    except (KeyError, ValueError, OSError):
        entities = ["subject"]
    tokens = []
    for entity in entities:
        try:
            key = schema_mod.entity_info(entity).name
        except (KeyError, ValueError, OSError):
            key = entity
        tokens.append(f"{key}-<label>")
    stem = "_".join(tokens) or "sub-<label>"
    return f"{tokens[0] if tokens else 'sub-<label>'}/{datatype}/{stem}_{suffix}.json"


def build_template_plan(
    present: list[tuple[str, str]],
    example_paths: Optional[dict[tuple[str, str], str]] = None,
    bids_root=None,
    include_derived: bool = False,
    also_ask: Optional[dict[tuple[str, str], Sequence[str]]] = None,
) -> list[TemplateSection]:
    """The whole template for a dataset: the agnostic file, then one per kind.

    ``present`` is the ``(datatype, suffix)`` pairs the scan found, so the form
    describes the dataset in front of the user rather than BIDS in general.
    Sections are ordered agnostic first, then by datatype, so the questions that
    apply to everything come before the ones that apply to one instrument.

    ``also_ask`` names, per kind, fields the caller knows apply after all. A
    conditional requirement is normally left out, because demanding it of every
    file sends the user hunting for something they must not add; when the
    caller can see the condition holds, the question is a real one.
    """
    example_paths = example_paths or {}
    sections = [dataset_description_section(bids_root)]

    seen: set[tuple[str, str]] = set()
    for datatype, suffix in sorted(present):
        if not datatype or not suffix or (datatype, suffix) in seen:
            continue
        seen.add((datatype, suffix))
        section = sidecar_section(
            datatype, suffix,
            example_paths.get((datatype, suffix), ""),
            bids_root, include_derived,
            also_ask=(also_ask or {}).get((datatype, suffix), ()),
        )
        # A file with nothing left to ask is not worth a section.
        if section.fields:
            sections.append(section)
    return sections


def _blood_rows(df) -> int:
    """How many included runs carry blood curves.

    A blood file is not a scanned row, so nothing in the inventory's own
    datatype and suffix columns announces it. It exists because the user
    attached curves to a PET run, and it needs its own questions asked: BIDS
    REQUIRES DispersionCorrected in a blood sidecar, and without this the form
    would never ask for it.
    """
    if df is None or not len(df) or "companion_files" not in df.columns:
        return 0
    from ..fixups.blood import is_blood_role

    found = 0
    for _, row in df.iterrows():
        if str(row.get("include", "1")).strip() in ("0", "False", "false"):
            continue
        raw = str(row.get("companion_files", "") or "").strip()
        if not raw:
            continue
        try:
            items = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(items, list) and any(
            isinstance(it, dict) and is_blood_role(str(it.get("suffix", "")))
            for it in items
        ):
            found += 1
    return found


def blood_conditions(df) -> tuple[str, ...]:
    """Fields a blood attachment makes applicable, that are otherwise skipped.

    The schema marks MetaboliteMethod required only when metabolite data is
    available, so the form leaves it out by default: a requirement that depends
    on a scenario a file may not be in must not be demanded of everybody.
    Attaching a parent-fraction curve IS that scenario, and then the question is
    a real one rather than a guess.
    """
    if df is None or not len(df) or "companion_files" not in df.columns:
        return ()
    from ..fixups.blood import parse_blood_role

    for _, row in df.iterrows():
        if str(row.get("include", "1")).strip() in ("0", "False", "false"):
            continue
        try:
            items = json.loads(str(row.get("companion_files", "") or "") or "[]")
        except (ValueError, TypeError):
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            parsed = parse_blood_role(str(it.get("suffix", "")))
            if parsed is not None and parsed[0] == "parentfraction":
                return ("MetaboliteMethod", "MetaboliteRecoveryCorrectionApplied")
    return ()


def present_pairs(df) -> list[tuple[str, str]]:
    """The ``(datatype, suffix)`` pairs an inventory contains, included rows only."""
    if df is None or not len(df):
        return []
    out: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        if str(row.get("include", "1")).strip() in ("0", "False", "false"):
            continue
        datatype = str(row.get("proposed_datatype", "") or "").strip()
        if not datatype:
            datatype = str(row.get("bids_guess_datatype", "") or "").strip()
        suffix = str(row.get("bids_guess_suffix", "") or "").strip()
        if datatype and suffix:
            out.add((datatype, suffix))
    if _blood_rows(df):
        out.add(("pet", "blood"))
    return sorted(out)


def pair_counts(df) -> dict[tuple[str, str], int]:
    """How many files of each kind the scan found, included rows only.

    So a section can say it speaks for 61 files rather than looking like a form
    about the one whose name it borrowed.
    """
    out: dict[tuple[str, str], int] = {}
    if df is None or not len(df):
        return out
    for _, row in df.iterrows():
        if str(row.get("include", "1")).strip() in ("0", "False", "false"):
            continue
        datatype = str(row.get("proposed_datatype", "") or "").strip()
        suffix = str(row.get("bids_guess_suffix", "") or "").strip()
        if datatype and suffix:
            out[(datatype, suffix)] = out.get((datatype, suffix), 0) + 1
    n_blood = _blood_rows(df)
    if n_blood:
        out[("pet", "blood")] = n_blood
    return out


def example_paths_for(df) -> dict[tuple[str, str], str]:
    """One real relative path per kind, so a section can name where it lands."""
    if df is None or not len(df):
        return {}
    out: dict[tuple[str, str], str] = {}
    for _, row in df.iterrows():
        datatype = str(row.get("proposed_datatype", "") or "").strip()
        suffix = str(row.get("bids_guess_suffix", "") or "").strip()
        basename = str(row.get("proposed_basename", "") or "").strip()
        if not (datatype and suffix and basename):
            continue
        key = (datatype, suffix)
        if key not in out:
            subject = str(row.get("BIDS_name", "") or "sub-<label>").strip()
            out[key] = f"{subject}/{datatype}/{basename}.json"
    return out


# ---------------------------------------------------------------------------
# The tree the form renders: region -> modality -> file
# ---------------------------------------------------------------------------

# Region keys. Agnostic first, always: what applies to the whole dataset is
# asked before what applies to one instrument.
REGION_AGNOSTIC = "agnostic"
REGION_MODALITY = "modality-specific"

# How a datatype reads as a heading, and the order regions list them in. A
# datatype absent from the scan never appears, so this is an ordering, not a
# list of what exists.
_MODALITY_ORDER = ("mri", "anat", "func", "dwi", "fmap", "perf",
                   "eeg", "meg", "ieeg", "nirs", "pet", "micr", "beh", "motion")


@dataclass(frozen=True)
class TemplateNode:
    """One node of the template tree.

    ``kind`` is ``region``, ``modality`` or ``file``. Only a file node carries a
    section; the others exist to group and to collapse.
    """

    key: str
    label: str
    kind: str
    children: tuple["TemplateNode", ...] = ()
    section: Optional[TemplateSection] = None
    subtitle: str = ""

    @property
    def is_leaf(self) -> bool:
        return self.section is not None

    def walk(self):
        """This node and every node beneath it, depth first."""
        yield self
        for child in self.children:
            yield from child.walk()

    @property
    def field_count(self) -> int:
        return sum(len(n.section.fields) for n in self.walk() if n.section)

    @property
    def required_count(self) -> int:
        return sum(len(n.section.required_fields) for n in self.walk() if n.section)


def _is_real_pair(datatype: str, suffix: str) -> bool:
    """Does BIDS declare this suffix for this datatype?

    The last line of defence for the form. A caller that guesses, or a stale
    inventory naming a suffix a newer schema dropped, would otherwise produce a
    section for a file that cannot exist, filed under a key nothing reads.
    """
    if not datatype or not suffix:
        return False
    try:
        return suffix in set(schema_mod.list_suffixes(datatype))
    except (KeyError, ValueError, OSError):
        return False


def _file_label(section: TemplateSection) -> tuple[str, str]:
    """``(what this section covers, the detail under it)``.

    A section is a class of files, not one file. Titling it with a real
    filename, which is what it used to do, reads as a list of individual files
    and hides the whole point: answer it once and every file of that kind gets
    it.

    So the title is the pattern, and the real name goes underneath as an
    example, with how many files it covers.
    """
    if not section.datatype:
        return section.target, ""

    title = f"every *_{section.suffix}.json"
    detail = f"in {section.datatype}/"
    if section.n_files:
        detail += f"  ·  {section.n_files} file{'s' if section.n_files != 1 else ''}"
    example = section.target.rpartition("/")[2]
    if example and not example.startswith("sub-<label>"):
        detail += f"  ·  e.g. {example}"
    return title, detail


def build_template_tree(
    present: list[tuple[str, str]],
    example_paths: Optional[dict[tuple[str, str], str]] = None,
    bids_root=None,
    include_derived: bool = False,
    answered: Optional[dict] = None,
    counts: Optional[dict[tuple[str, str], int]] = None,
    also_ask: Optional[dict[tuple[str, str], Sequence[str]]] = None,
) -> list[TemplateNode]:
    """The whole template as a tree, agnostic region first.

    Ordering lives here rather than in the widget code, so every surface that
    renders the tree agrees about what comes first.
    """
    example_paths = example_paths or {}
    present = [
        (datatype, suffix)
        for datatype, suffix in present
        if _is_real_pair(datatype, suffix)
    ]

    agnostic_section = dataset_description_section(bids_root)
    agnostic_name, agnostic_dir = _file_label(agnostic_section)
    agnostic = TemplateNode(
        key=REGION_AGNOSTIC,
        label="Modality-agnostic",
        kind="region",
        children=(
            TemplateNode(
                key=agnostic_section.storage_key or "dataset_description",
                label=agnostic_name,
                kind="file",
                section=agnostic_section,
                subtitle=agnostic_dir,
            ),
        ),
    )

    by_modality: dict[str, list[TemplateNode]] = {}
    seen: set[tuple[str, str]] = set()
    for datatype, suffix in sorted(present):
        if not datatype or not suffix or (datatype, suffix) in seen:
            continue
        seen.add((datatype, suffix))
        section = sidecar_section(
            datatype, suffix, example_paths.get((datatype, suffix), ""),
            bids_root, include_derived,
            answered=(answered or {}).get(f"{datatype}/{suffix}"),
            n_files=(counts or {}).get((datatype, suffix), 0),
            also_ask=(also_ask or {}).get((datatype, suffix), ()),
        )
        if not section.fields:
            # Nothing left to ask about this file: the converter answers it all.
            continue
        filename, directory = _file_label(section)
        by_modality.setdefault(datatype, []).append(
            TemplateNode(
                key=section.storage_key,
                label=filename,
                kind="file",
                section=section,
                subtitle=directory,
            )
        )

    order = {name: i for i, name in enumerate(_MODALITY_ORDER)}
    modalities = tuple(
        TemplateNode(
            key=datatype,
            label=datatype.upper(),
            kind="modality",
            children=tuple(files),
        )
        for datatype, files in sorted(
            by_modality.items(), key=lambda kv: (order.get(kv[0], 99), kv[0])
        )
    )

    tree = [agnostic]
    if modalities:
        tree.append(
            TemplateNode(
                key=REGION_MODALITY,
                label="Modality-specific",
                kind="region",
                children=modalities,
            )
        )
    return tree


__all__ = [
    "REGION_AGNOSTIC",
    "REGION_MODALITY",
    "STORAGE_DATASET_DESCRIPTION",
    "STORAGE_SEQUENCE_TEMPLATE",
    "ORIGIN_ABSENT",
    "ORIGIN_USUALLY_DERIVED",
    "TemplateField",
    "TemplateNode",
    "TemplateSection",
    "build_template_tree",
    "build_template_plan",
    "dataset_description_section",
    "example_paths_for",
    "pair_counts",
    "present_pairs",
    "sidecar_section",
]

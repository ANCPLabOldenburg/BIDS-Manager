from __future__ import annotations

import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union, Set

import ancpbids.model_v1_10_0 as ancp_schema  # noqa: F401 - imported for typing context
from ancpbids.model_v1_10_0 import DatatypeEnum, EntityEnum, ModalityEnum, SuffixEnum


# ---------------------------------------------------------------------------
# Feature flags and configuration shared with GUI/CLI entry points
# ---------------------------------------------------------------------------

# Historically the renamer shipped a copy of the official BIDS schema and the
# GUI exposed its location via ``DEFAULT_SCHEMA_DIR``.  The modern
# implementation derives all metadata directly from the ``ancpbids`` package, so
# the directory is no longer required.  Keep the constant for backwards
# compatibility but point it to ``None`` to signal that callers do not need to
# interact with on-disk files anymore.
DEFAULT_SCHEMA_DIR: Optional[Path] = None

# Feature toggles exposed to the GUI and CLI entry points.  Keeping them next to
# the implementation avoids additional indirection and mirrors the layout of
# the previous helper modules while still presenting a single public surface.
ENABLE_SCHEMA_RENAMER = True
ENABLE_FIELDMap_NORMALIZATION = True
ENABLE_DWI_DERIVATIVES_MOVE = True
DERIVATIVES_PIPELINE_NAME = "dcm2niix"  # or "BIDS-Manager" if you prefer


# ----------------------------- Utilities -----------------------------

_BIDS_EXTS = (".nii.gz", ".nii", ".json", ".bval", ".bvec", ".tsv")

# Canonical labels extracted from the official BIDS schema enumerations.  The
# ``ancpbids`` package exposes the model for BIDS 1.10.0 in a machine-readable
# form which we use to keep our internal hints aligned with the specification.
SUBJECT_ENTITY = EntityEnum.subject
TASK_ENTITY = EntityEnum.task

# Regular expression that captures alphanumeric "words" inside a study name.
_WORD_RE = re.compile(r"[0-9A-Za-z]+")


def _enum_lookup(enum_cls: type[Enum], target: str) -> Optional[Enum]:
    """Return the enum member that best matches ``target`` (case insensitive)."""

    if not target:
        return None
    target_lower = target.lower()
    for member in enum_cls:  # type: ignore[assignment]
        if member.name.lower() == target_lower:
            return member
        value = getattr(member, "value", None)
        if isinstance(value, dict):
            alias = value.get("value")
            if isinstance(alias, str) and alias.lower() == target_lower:
                return member
        elif isinstance(value, str) and value.lower() == target_lower:
            return member
    return None


def _enum_to_name(member: Optional[Enum]) -> str:
    """Return the canonical string representation for ``member``."""

    if member is None:
        return ""
    value = getattr(member, "value", None)
    if isinstance(value, dict):
        alias = value.get("value")
        if isinstance(alias, str):
            return alias
    if isinstance(value, str):
        return value
    return member.name


@dataclass
class SequenceHint:
    """Describe how to classify a raw sequence into BIDS terms."""

    label: str
    suffix: Optional[SuffixEnum]
    datatype: Optional[DatatypeEnum]
    modality: Optional[ModalityEnum]
    required_entities: Tuple[EntityEnum, ...]
    default_patterns: Tuple[str, ...]
    custom_patterns: Tuple[str, ...] = ()
    container_override: Optional[str] = None

    def copy(self) -> "SequenceHint":
        return replace(self)

    def get_patterns(self, source: str = "active") -> Tuple[str, ...]:
        """Return pattern hints for the requested ``source``."""

        if source == "default":
            return self.default_patterns
        if source == "custom":
            return self.custom_patterns
        return self.custom_patterns or self.default_patterns

    @property
    def patterns(self) -> Tuple[str, ...]:
        """Expose the currently active pattern hints."""

        return self.get_patterns("active")


def _build_default_sequence_hints() -> "OrderedDict[str, SequenceHint]":
    """Return the built-in sequence hint configuration."""

    mri_modality = _enum_lookup(ModalityEnum, "mri")
    subject_entity = _enum_lookup(EntityEnum, "subject") or SUBJECT_ENTITY
    task_entity = _enum_lookup(EntityEnum, "task") or TASK_ENTITY
    anat_dt = _enum_lookup(DatatypeEnum, "anat")
    func_dt = _enum_lookup(DatatypeEnum, "func")
    dwi_dt = _enum_lookup(DatatypeEnum, "dwi")
    fmap_dt = _enum_lookup(DatatypeEnum, "fmap")

    hints: "OrderedDict[str, SequenceHint]" = OrderedDict()

    hints["dwi_derivative"] = SequenceHint(
        label="dwi_derivative",
        suffix=None,
        datatype=dwi_dt,
        modality=mri_modality,
        required_entities=(subject_entity,),
        default_patterns=(
            "_adc",
            "_fa",
            "_tracew",
            "_colfa",
            "_expadc",
            " adc",
            " fa",
            " tracew",
            " colfa",
            " expadc",
            "adc_",
            "adc ",
            "trace",
        ),
        container_override="derivatives",
    )

    hints["T1w"] = SequenceHint(
        label="T1w",
        suffix=_enum_lookup(SuffixEnum, "T1w"),
        datatype=anat_dt,
        modality=mri_modality,
        required_entities=(subject_entity,),
        default_patterns=(
            "t1w",
            "t1-weight",
            "t1_",
            "t1 ",
            "mprage",
            "tfl3d",
            "fspgr",
        ),
    )

    hints["T2w"] = SequenceHint(
        label="T2w",
        suffix=_enum_lookup(SuffixEnum, "T2w"),
        datatype=anat_dt,
        modality=mri_modality,
        required_entities=(subject_entity,),
        default_patterns=("t2w", "space", "tse"),
    )

    hints["FLAIR"] = SequenceHint(
        label="FLAIR",
        suffix=_enum_lookup(SuffixEnum, "FLAIR"),
        datatype=anat_dt,
        modality=mri_modality,
        required_entities=(subject_entity,),
        default_patterns=("flair",),
    )

    hints["MTw"] = SequenceHint(
        label="MTw",
        suffix=_enum_lookup(SuffixEnum, "MTw"),
        datatype=anat_dt,
        modality=mri_modality,
        required_entities=(subject_entity,),
        default_patterns=("gre-mt", "gre_mt", "mt"),
    )

    hints["PDw"] = SequenceHint(
        label="PDw",
        suffix=_enum_lookup(SuffixEnum, "PDw"),
        datatype=anat_dt,
        modality=mri_modality,
        required_entities=(subject_entity,),
        default_patterns=("gre-nm", "gre_nm"),
    )

    hints["scout"] = SequenceHint(
        label="scout",
        suffix=_enum_lookup(SuffixEnum, "localizer"),
        datatype=anat_dt,
        modality=mri_modality,
        required_entities=(subject_entity,),
        default_patterns=("localizer", "scout"),
    )

    hints["report"] = SequenceHint(
        label="report",
        suffix=_enum_lookup(SuffixEnum, "reports"),
        datatype=anat_dt,
        modality=mri_modality,
        required_entities=(subject_entity,),
        default_patterns=("phoenixzipreport", "phoenix document", ".pdf", "report"),
    )

    hints["SBRef"] = SequenceHint(
        label="SBRef",
        suffix=_enum_lookup(SuffixEnum, "sbref"),
        datatype=func_dt,
        modality=mri_modality,
        required_entities=(subject_entity, task_entity),
        default_patterns=(
            "sbref",
            "type-ref",
            "reference",
            "refscan",
            " ref",
            "_ref",
        ),
    )

    hints["physio"] = SequenceHint(
        label="physio",
        suffix=_enum_lookup(SuffixEnum, "physio"),
        datatype=func_dt,
        modality=mri_modality,
        required_entities=(subject_entity, task_entity),
        default_patterns=("physiolog", "physio", "pulse", "resp"),
    )

    hints["bold"] = SequenceHint(
        label="bold",
        suffix=_enum_lookup(SuffixEnum, "bold"),
        datatype=func_dt,
        modality=mri_modality,
        required_entities=(subject_entity, task_entity),
        default_patterns=("fmri", "bold", "task-"),
    )

    hints["dwi"] = SequenceHint(
        label="dwi",
        suffix=_enum_lookup(SuffixEnum, "dwi"),
        datatype=dwi_dt,
        modality=mri_modality,
        required_entities=(subject_entity,),
        default_patterns=("dti", "dwi", "diff"),
    )

    hints["fmap"] = SequenceHint(
        label="fmap",
        suffix=_enum_lookup(SuffixEnum, "phasediff"),
        datatype=fmap_dt,
        modality=mri_modality,
        required_entities=(subject_entity,),
        default_patterns=(
            "gre_field",
            "fieldmapping",
            "_fmap",
            "fmap",
            "phase",
            "magnitude",
            "b0rf",
            "b0_map",
            "b0map",
        ),
    )

    return hints


DEFAULT_SEQUENCE_HINTS: "OrderedDict[str, SequenceHint]" = _build_default_sequence_hints()

SEQUENCE_HINTS: "OrderedDict[str, SequenceHint]" = OrderedDict(
    (label, hint.copy()) for label, hint in DEFAULT_SEQUENCE_HINTS.items()
)


def get_sequence_hint_patterns(source: str = "active") -> Dict[str, Tuple[str, ...]]:
    """Return the mapping of labels to pattern hints from ``source``."""

    return {label: hint.get_patterns(source) for label, hint in SEQUENCE_HINTS.items()}


def set_sequence_hint_patterns(patterns: Mapping[str, Iterable[str]]) -> None:
    """Override the substring hints used to classify raw sequences."""

    for label, pats in patterns.items():
        seen: Set[str] = set()
        cleaned: list[str] = []
        for pat in pats:
            text = str(pat).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(text)

        normalized = tuple(cleaned)
        hint = SEQUENCE_HINTS.get(label)
        if hint is None:
            SEQUENCE_HINTS[label] = SequenceHint(
                label=label,
                suffix=_enum_lookup(SuffixEnum, label),
                datatype=_enum_lookup(DatatypeEnum, label),
                modality=_enum_lookup(ModalityEnum, "mri"),
                required_entities=(SUBJECT_ENTITY,),
                default_patterns=(),
                custom_patterns=normalized,
            )
        else:
            hint.custom_patterns = normalized


def restore_default_sequence_hints() -> None:
    """Reset :data:`SEQUENCE_HINTS` to the built-in configuration."""

    SEQUENCE_HINTS.clear()
    for label, hint in DEFAULT_SEQUENCE_HINTS.items():
        clone = hint.copy()
        clone.custom_patterns = ()
        SEQUENCE_HINTS[label] = clone


# Modalities that should be skipped by default when auto-populating TSVs or
# heuristics.  Scouts remain separate so the GUI can keep listing them while
# marking them as excluded.
SKIP_MODALITIES: Set[str] = {"scout", "report"}


_SANITIZE_TOKEN = re.compile(r"[^a-zA-Z0-9]+")


@dataclass(frozen=True)
class _NormalizedSeries:
    """Lower-cased representation of a sequence name with token metadata."""

    raw: str
    lower: str
    tokens: Tuple[str, ...]
    token_set: Set[str]


def _normalize_series(text: Optional[str]) -> _NormalizedSeries:
    raw = text or ""
    lower = raw.lower()
    tokenized = _SANITIZE_TOKEN.sub(" ", lower)
    tokens = tuple(filter(None, tokenized.split()))
    return _NormalizedSeries(raw=raw, lower=lower, tokens=tokens, token_set=set(tokens))


def _score_patterns(patterns: Tuple[str, ...], normalized: _NormalizedSeries) -> Optional[Tuple[int, int]]:
    """Return a ranking describing how well ``patterns`` match ``normalized``."""

    best: Optional[Tuple[int, int]] = None
    for pat in patterns:
        pat_lower = pat.lower()
        if not pat_lower:
            continue

        token_candidate = _SANITIZE_TOKEN.sub("", pat_lower)
        score: Optional[Tuple[int, int]] = None
        if token_candidate and token_candidate in normalized.token_set:
            score = (3, len(token_candidate))
        elif (
            token_candidate
            and len(token_candidate) >= 3
            and any(tok.startswith(token_candidate) for tok in normalized.tokens)
        ):
            score = (2, len(token_candidate))
        elif pat_lower in normalized.lower:
            score = (1, len(pat_lower))

        if score and (best is None or score > best):
            best = score

    return best


def guess_modality(series: str) -> str:
    """Return the modality label whose patterns best describe ``series``."""

    normalized = _normalize_series(series)
    best_label = "unknown"
    best_score: Optional[Tuple[int, int, int]] = None

    for idx, (label, hint) in enumerate(SEQUENCE_HINTS.items()):
        patterns = hint.patterns
        if not patterns:
            continue
        score = _score_patterns(patterns, normalized)
        if score is None:
            continue
        ranked = (score[0], score[1], -idx)
        if best_score is None or ranked > best_score:
            best_label = label
            best_score = ranked

    return best_label


def modality_to_container(mod: str) -> str:
    """Translate fine modality labels into top-level BIDS folders."""

    hint = SEQUENCE_HINTS.get(mod)
    if hint is None:
        return ""
    if hint.container_override:
        return hint.container_override
    return _enum_to_name(hint.datatype)

_TASK_TOKEN = re.compile(r"(?:^|[_-])task-([a-zA-Z0-9]+)", re.IGNORECASE)
_ACQ_TOKEN = re.compile(r"(?:^|[_-])acq-([a-zA-Z0-9]+)", re.IGNORECASE)


def _sanitize_token(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    return _SANITIZE_TOKEN.sub("", x).strip()


def _extract_acq_token(sequence: Optional[str]) -> Optional[str]:
    """Return the most descriptive ``acq-`` label embedded in ``sequence``.

    Some scanners embed several ``acq-`` hints in the raw sequence text (for
    example ``"acq-15_acq-15b0"``).  The historic implementation would return
    the *first* match which often meant a richer discriminator was dropped.  To
    preserve the most useful hint we inspect every match and keep the longest
    sanitized label, breaking ties by favouring the right-most occurrence so the
    selection remains deterministic.
    """

    if not sequence:
        return None

    candidates = []
    for match in _ACQ_TOKEN.finditer(sequence):
        token = _sanitize_token(match.group(1))
        if token:
            candidates.append((token, match.start()))

    if not candidates:
        return None

    # ``max`` with a custom key favours the longer sanitized token which tends
    # to carry the most context (e.g. ``acq-15b0`` vs ``acq-15``).  When two
    # tokens share the same length we keep the right-most one so that repeated
    # hints resolve to the most specific entry.
    best_token, _ = max(candidates, key=lambda item: (len(item[0]), item[1]))
    return best_token


def _strip_run_tokens(sequence: str) -> str:
    """Remove any run-N tokens from sequence name before processing.
    
    This ensures we don't have run- artifacts in the original sequence names
    that could cause confusion with our repetition handling.
    """
    if not sequence:
        return sequence
    # Remove run-N patterns (run-01, run-1, etc.)
    sequence = re.sub(r"_?run-\d+_?", "_", sequence, flags=re.IGNORECASE)
    sequence = re.sub(r"^run-\d+_", "", sequence, flags=re.IGNORECASE)
    sequence = re.sub(r"_run-\d+$", "", sequence, flags=re.IGNORECASE)
    # Clean up multiple underscores
    sequence = re.sub(r"_+", "_", sequence).strip("_")
    return sequence


def _extract_run_number(sequence: Optional[str]) -> Optional[int]:
    """Return run number encoded in a sequence name if present.

    Fieldmaps and some other acquisitions may encode ``run`` information in
    the original ``SeriesDescription``.  Previously this information was
    stripped which caused different runs to collide under the same BIDS name
    (e.g. two fieldmaps both renamed to ``run-1``).  This helper captures the
    run number before any cleaning takes place so the caller can explicitly
    emit a ``run-<N>`` entity, preserving uniqueness.
    """
    if not sequence:
        return None
    m = re.search(r"run-?0*(\d+)", sequence, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _guess_task_from_text(*candidates: Optional[str]) -> Optional[str]:
    """Extract task name from text candidates.
    
    This function tries to find meaningful task hints, but if none are found,
    it will return None so the caller can use the full sequence as a fallback
    to ensure uniqueness.
    """
    # First strip run tokens from all candidates
    clean_candidates = [_strip_run_tokens(c) if c else c for c in candidates]
    
    for c in clean_candidates:
        if not c:
            continue
        m = _TASK_TOKEN.search(c)
        if m:
            return _sanitize_token(m.group(1))
    
    # Check for resting state patterns first
    resting_patterns = ("rs", "_rs", "rs_", "rest", "resting")
    for c in clean_candidates:
        if not c:
            continue
        low = c.lower()
        for pattern in resting_patterns:
            if pattern in low:
                return "rest"
    
    # Check for other common task patterns
    task_hints = (
        # Common explicit task labels
        "movie", "nback", "flanker", "stroop", "motor", "checker", "checkerboard",
        # Atypical labels observed in some centers
        "exec", "paradigma", "paradigm", "sparse", "mb",
        # Generic fallbacks
        "task", "activation",
    )
    for c in clean_candidates:
        if not c:
            continue
        low = c.lower()
        for hint in task_hints:
            if hint in low:
                return _sanitize_token(hint)
    
    # DON'T try to extract parts here - return None so caller can use full sequence
    # This ensures different sequences never get the same task name
    return None


def _resolve_ext(name: str) -> str:
    for ext in _BIDS_EXTS:
        if name.endswith(ext):
            return ext
    return Path(name).suffix


def _detect_dwi_derivative(sequence: str) -> Optional[str]:
    """Return standardized map name if ``sequence`` looks like a DWI derivative.

    Only a small, explicit set is considered a derivative according to the
    user's policy. Anything else is treated as raw DWI.
    """
    s = (sequence or "").lower()
    if "colfa" in s:
        return "ColFA"
    if "fa" in s and "colfa" not in s:
        return "FA"
    if "tensor" in s:
        return "TENSOR"
    if "adc" in s:
        return "ADC"
    if "trace" in s:
        return "TRACE"
    if "tracew" in s:
        return "TRACE"
    return None


def _infer_dwi_acq_dir(sequence: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer acquisition and direction tokens from a DWI sequence name.

    Parameters
    ----------
    sequence:
        Original sequence description (e.g. ``"DTI_LR"``).

    Returns
    -------
    acq, dir:
        Potential ``acq`` and ``dir`` entities extracted from the sequence. If no
        recognizable pattern is found, both are ``None``.

    Notes
    -----
    This helper looks for common diffusion naming conventions such as
    ``LR``/``RL`` and ``AP``/``PA`` for phase‑encoding direction as well as
    tokens like ``b0`` or ``15b0`` which typically indicate a specific
    acquisition. The goal is to keep these informative hints in the proposed
    BIDS name without requiring the user to supply explicit ``acq``/``dir``
    values.
    """

    if not sequence:
        return None, None

    # Replace non alphanumeric characters with spaces to simplify pattern
    # detection while preserving word boundaries.
    clean = _SANITIZE_TOKEN.sub(" ", sequence).lower()

    # Direction tokens are usually short (AP/PA/LR/RL). Use word boundaries to
    # avoid picking up unintended substrings.
    m_dir = re.search(r"(?<![a-z0-9])(lr|rl|ap|pa)(?![a-z0-9])", clean)
    direction = m_dir.group(1) if m_dir else None

    acq: Optional[str] = None
    # ``15b0`` or ``b0`` patterns take precedence.
    m_b0 = re.search(r"(?<![a-z0-9])(\d*b0)(?![a-z0-9])", clean)
    if m_b0:
        acq = m_b0.group(1)
    else:
        # If no ``b0`` token is present but a direction exists, capture a plain
        # numeric token such as ``15`` to distinguish acquisitions with different
        # numbers of directions.
        if direction:
            m_num = re.search(r"(?<![a-z0-9])(\d+)(?![a-z0-9])", clean)
            if m_num:
                acq = m_num.group(1)

    return acq, direction


def _replace_stem_keep_ext(src: Path, new_basename: str) -> Path:
    ext = _resolve_ext(src.name)
    return src.with_name(f"{new_basename}{ext}")


# --------------------------- Schema parsing ---------------------------

@dataclass
class SchemaInfo:
    suffix_requirements: Dict[str, List[str]]
    suffix_to_datatypes: Dict[str, List[str]]


def load_bids_schema(schema_dir: Union[str, Path, None] = None) -> SchemaInfo:
    """Return schema hints derived from the bundled ``ancpbids`` model.

    The previous implementation parsed the YAML documents shipped with
    ``bids-manager``.  Now that ``ancpbids`` exposes the same information as a
    Python module we no longer need to ship or read those files at runtime.  The
    *schema_dir* argument is accepted for backwards compatibility but ignored.
    """

    suffix_to_datatypes: Dict[str, Set[str]] = {}
    suffix_requirements: Dict[str, Set[str]] = {}

    def _register(suffix: str, datatype: str) -> None:
        if not suffix or not datatype:
            return
        suffix_to_datatypes.setdefault(suffix, set()).add(datatype)

    for hint in SEQUENCE_HINTS.values():
        suffix = _enum_to_name(hint.suffix)
        datatype = _enum_to_name(hint.datatype)
        _register(suffix, datatype)
        if suffix:
            requirements = suffix_requirements.setdefault(suffix, set())
            for entity in hint.required_entities:
                requirements.add(_enum_to_name(entity))

    anat_name = _enum_to_name(_enum_lookup(DatatypeEnum, "anat"))
    fmap_name = _enum_to_name(_enum_lookup(DatatypeEnum, "fmap"))

    additional_suffixes = {
        "T2star": anat_name,
        "PD": anat_name,
        "phasediff": fmap_name,
        "magnitude1": fmap_name,
        "magnitude2": fmap_name,
        "magnitude": fmap_name,
        "phase1": fmap_name,
        "phase2": fmap_name,
        "fieldmap": fmap_name,
        "epi": fmap_name,
    }
    for suffix, datatype in additional_suffixes.items():
        _register(suffix, datatype)
        suffix_requirements.setdefault(suffix, set()).add(_enum_to_name(SUBJECT_ENTITY))

    known_suffixes = {member.value["value"] for member in SuffixEnum}
    known_suffixes.update(suffix_to_datatypes.keys())

    subject_name = _enum_to_name(SUBJECT_ENTITY)
    task_name = _enum_to_name(TASK_ENTITY)

    for suffix in known_suffixes:
        suffix_requirements.setdefault(suffix, set()).add(subject_name)

    for suffix in ("bold", "sbref", "physio"):
        if suffix in known_suffixes:
            suffix_requirements.setdefault(suffix, set()).add(task_name)

    return SchemaInfo(
        suffix_requirements={k: sorted(v) for k, v in suffix_requirements.items()},
        suffix_to_datatypes={k: sorted(v) for k, v in suffix_to_datatypes.items()},
    )


# --------------------------- Core Proposer ----------------------------

@dataclass
class SeriesInfo:
    subject: str
    session: Optional[str]
    modality: str
    sequence: str
    rep: Optional[int]
    extra: Dict[str, str]


def _normalize_session_label(session: Optional[str]) -> Optional[str]:
    """Return a sanitised session label compatible with ``ses-<label>``."""

    if not session:
        return None

    raw = str(session)
    cleaned = _sanitize_token(raw)
    if not cleaned:
        return None

    if raw.lower().startswith(("ses-", "ses_")) and cleaned.lower().startswith("ses"):
        cleaned = cleaned[3:]

    return cleaned or None


def _normalize_suffix(modality: str) -> str:
    """Return a canonical BIDS suffix for a given modality string.

    The DICOM ``SeriesDescription`` or user‑supplied modality labels can use a
    wide range of capitalisation patterns or legacy terms (e.g. ``DTI`` instead
    of ``dwi``).  This helper performs a case‑insensitive normalisation so later
    logic only needs to work with the official BIDS suffixes.
    """

    m = modality.strip()
    alias = {
        "sbref": "sbref",
        "t2*": "T2star",
        "t2star": "T2star",
        "dti": "dwi",
    }
    return alias.get(m.lower(), m)


def _choose_datatype(suffix: str, schema: SchemaInfo) -> str:
    # Handle DWI derivatives first
    if suffix.lower() in ("adc", "fa", "tracew", "colfa", "expadc"):
        return "derivatives"
    
    dts = schema.suffix_to_datatypes.get(suffix)
    if dts:
        pref = ("anat", "func", "dwi", "fmap", "perf", "pet", "meg", "eeg", "ieeg")
        for p in pref:
            if p in dts:
                return p
        return dts[0]
    return {
        "T1w": "anat", "T2w": "anat", "FLAIR": "anat", "T2star": "anat", "PD": "anat",
        "bold": "func", "sbref": "func", "physio": "func", "dwi": "dwi",
        "phasediff": "fmap",
        "fieldmap": "fmap",
        "magnitude": "fmap",
        "magnitude1": "fmap",
        "magnitude2": "fmap",
        "phase1": "fmap",
        "phase2": "fmap",
        "fmap": "fmap",
    }.get(suffix, "misc")


def propose_bids_basename(series: SeriesInfo, schema: SchemaInfo) -> Tuple[str, str]:
    """Propose a BIDS basename for a series.
    
    This function guarantees that different sequence names will never produce
    identical BIDS names by using the full sanitized sequence as a fallback
    when no specific task hint is found.
    """
    suffix = _normalize_suffix(series.modality)
    # Update the series object so any external tables reflect the normalized
    # modality used for the proposed BIDS name (prevents "bold" vs "func" mismatches
    # in the GUI).
    series.modality = suffix
    datatype = _choose_datatype(suffix, schema)
    required = set(schema.suffix_requirements.get(suffix, []))

    parts: List[str] = []
    sub = _sanitize_token(series.subject)
    if not sub:
        raise ValueError("SeriesInfo.subject is required and must be alphanumeric")
    parts.append(f"sub-{sub}")

    ses = _normalize_session_label(series.session) or ""
    if ses:
        parts.append(f"ses-{ses}")

    # Strip run tokens from sequence before any processing
    # Capture run number before sanitising the sequence.  We keep the original
    # run information so that different fieldmaps (or any other modality using
    # runs) do not collide in their proposed BIDS names.
    run_number = _extract_run_number(series.sequence)
    clean_sequence = _strip_run_tokens(series.sequence)

    # ``task`` labels are mandatory for some suffixes (as per the schema
    # requirements) and also desirable for functional reference scans and
    # physiological recordings so that supporting files share the same base
    # name as their associated BOLD runs.  Treat ``physio`` like ``bold`` and
    # ``sbref`` so that it inherits task/run context from the sequence text.
    if "task" in required or suffix in ("bold", "sbref", "physio"):
        task_hint = series.extra.get("task") if series.extra else None

        # ``task_hits`` is an optional list of keywords extracted by the GUI
        # or TSV importer.  If provided we try to match any of those hits
        # against the sequence name.  The first match becomes the task label.
        if not task_hint and series.extra:
            hits = series.extra.get("task_hits")
            if hits:
                for token in re.split(r"[;,\s]+", str(hits)):
                    token_s = _sanitize_token(token)
                    if not token_s:
                        continue
                    if token_s.lower() in clean_sequence.lower():
                        task_hint = token_s
                        break

        # First try explicit or hit-based task hints, then try to guess from
        # the sequence text itself.
        task = _sanitize_token(task_hint) or _guess_task_from_text(clean_sequence)
        
        # If no specific task hint found, use the full sanitized sequence
        # This ensures different sequences NEVER get the same BIDS name
        if not task:
            task = _sanitize_token(clean_sequence)[:48]  # Truncate for safety
            if not task:
                task = "unknown"
        
        parts.append(f"task-{task}")

    # Detect DWI derivative from the sequence text itself and adjust datatype
    # and suffix accordingly so Preview/Table point to the derivatives tree and
    # not raw dwi/.
    direction: Optional[str] = None
    sequence_acq = _extract_acq_token(series.sequence)
    map_name = _detect_dwi_derivative(clean_sequence)
    if map_name:
        datatype = "derivatives"
        # Keep an acquisition discriminator for uniqueness between different
        # series descriptions.
        acq_token = sequence_acq or _sanitize_token(clean_sequence)[:32]
        if acq_token:
            parts.append(f"acq-{acq_token}")
        parts.append(f"desc-{map_name}")
        # Force suffix used later to dwi
        suffix = "dwi"
        direction = series.extra.get("dir") if series.extra else None
    else:
        # For non-derivatives, only include an acquisition label if one was
        # explicitly provided. Previously, the sequence text was used as a
        # fallback to guarantee uniqueness, which produced long
        # ``acq-<pattern>`` tokens. These were confusing and not BIDS
        # recommended, so we now omit ``acq`` unless the caller supplies it.
        explicit_acq = series.extra.get("acq") if series.extra else None
        direction = series.extra.get("dir") if series.extra else None
        inf_acq: Optional[str] = None

        if suffix == "dwi":
            # Infer missing ``acq``/``dir`` tokens from the sequence itself so
            # that common diffusion naming conventions (e.g. ``DTI_LR``) remain
            # distinguishable without manual intervention.
            inf_acq, inf_dir = _infer_dwi_acq_dir(clean_sequence)
            if not direction:
                direction = inf_dir

        # Determine the final acquisition label prioritising explicit input,
        # then the sequence token (to preserve existing ``acq-`` hints) and
        # finally any heuristically inferred value for DWI series.
        candidates = (
            _sanitize_token(explicit_acq) if explicit_acq else None,
            sequence_acq,
            _sanitize_token(inf_acq) if inf_acq else None,
        )
        for acq in candidates:
            if acq:
                parts.append(f"acq-{acq}")
                break

    echo = series.extra.get("echo") if series.extra else None
    if echo:
        echo = _sanitize_token(str(echo))
        if echo:
            parts.append(f"echo-{echo}")

    if direction:
        direction = _sanitize_token(direction)
        if direction:
            parts.append(f"dir-{direction}")

    # Preserve run information if we detected it earlier.  BIDS recommends
    # zero padding the run number to two digits for consistency.
    if run_number is not None:
        parts.append(f"run-{run_number:02d}")

    # Repetition numbering (same acquisition repeated) is handled by
    # ``build_preview_names`` using the ``_rep-<N>`` suffix.  Here we only add
    # the BIDS ``run-`` entity when it is explicitly encoded in the original
    # sequence name.
    parts.append(suffix)
    return datatype, "_".join(parts)


# -------------------------- Post-conv renaming ------------------------

def _glob_candidates(dt_dir: Path, subject: str, original_seq: str) -> List[Path]:
    """Return files in ``dt_dir`` whose names loosely match ``original_seq``.

    ``heudiconv`` preserves the letter case of the DICOM ``SeriesDescription``
    when generating filenames.  Earlier versions of this helper lower‑cased the
    search pattern which broke matching on case‑sensitive filesystems whenever a
    sequence contained uppercase characters.  To make the lookup robust we build
    glob patterns using both the original and a lower‑cased variant of the
    sequence name and collect the unique matches.
    """

    seq = original_seq or ""
    seq_clean = _SANITIZE_TOKEN.sub("", seq)
    seq_wc = _SANITIZE_TOKEN.sub("*", seq)

    patterns: List[str] = []
    for token in {seq_clean, seq_clean.lower()}:
        if token:
            patterns.extend([
                f"sub-*{token}*.*",
                f"sub-*{token}*.nii.gz",
                f"sub-*{token}*.json",
                f"*{token}*.*",  # Also try without sub- prefix
                f"*{token}*.nii.gz",
                f"*{token}*.json",
            ])
    for token in {seq_wc, seq_wc.lower()}:
        if token:
            patterns.extend([
                f"sub-*{token}*.*",
                f"sub-*{token}*.nii.gz",
                f"sub-*{token}*.json",
                f"*{token}*.*",  # Also try without sub- prefix
                f"*{token}*.nii.gz",
                f"*{token}*.json",
            ])

    out: List[Path] = []
    seen = set()
    for g in patterns:
        for p in dt_dir.glob(g):
            if not p.is_file() or p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def _normalize_fieldmaps(dt_dir: Path, rename_map: Dict[Path, Path]) -> None:
    for p in dt_dir.glob("*_echo-1.*"):
        newb = p.name.replace("_echo-1", "_magnitude1")
        rename_map[p] = p.with_name(newb)
    for p in dt_dir.glob("*_echo-2.*"):
        newb = p.name.replace("_echo-2", "_magnitude2")
        rename_map[p] = p.with_name(newb)
    for p in dt_dir.glob("*_fmap.*"):
        newb = p.name.replace("_fmap", "_phasediff")
        rename_map[p] = p.with_name(newb)


def _move_dwi_derivatives(bids_root: Path, pipeline_name: str, rename_map: Dict[Path, Path]) -> None:
    """
    Move vendor-derived DWI maps from raw dwi/ to derivatives/<pipeline>/dwi/
    and rename as sub-XXX_desc-<MAP>_dwi.<ext>.
    Maps handled: ADC, FA, TRACEW, ColFA
    """
    for sub_dir in bids_root.glob("sub-*"):
        dwi_dir = sub_dir / "dwi"
        if not dwi_dir.exists():
            continue
        # detect maps on disk
        for p in dwi_dir.glob("*_*"):
            stem = p.name
            # skip non-files and .bval/.bvec of raw runs
            if not p.is_file():
                continue
            if stem.endswith(".bval") or stem.endswith(".bvec"):
                continue

            # map suffix detection
            for tag in ("_ADC", "_FA", "_TRACEW", "_ColFA"):
                if tag in stem:
                    desc = tag[1:]  # remove leading underscore
                    # new location under derivatives
                    new_dir = bids_root / "derivatives" / pipeline_name / sub_dir.name / "dwi"
                    new_dir.mkdir(parents=True, exist_ok=True)
                    # build new basename: keep leading sub-XXX[_ses-YYY] if present, then desc-<MAP>_dwi
                    # try to extract sub- and ses- tokens
                    tokens = [t for t in stem.split("_") if t.startswith(("sub-", "ses-"))]
                    prefix = "_".join(tokens) if tokens else sub_dir.name
                    new_basename = f"{prefix}_desc-{desc}_dwi"
                    newp = _replace_stem_keep_ext(p, new_basename)
                    rename_map[p] = new_dir / newp.name
                    break


@dataclass(frozen=True)
class _SeriesPaths:
    """Pre-computed filesystem locations for a :class:`SeriesInfo`."""

    bids_root: Path
    subject_label: str
    session_label: Optional[str]

    @classmethod
    def from_series(cls, bids_root: Path, series: SeriesInfo) -> "_SeriesPaths":
        subject = _sanitize_token(series.subject)
        if not subject:
            raise ValueError("SeriesInfo.subject must contain an alphanumeric label")
        return cls(
            bids_root=bids_root,
            subject_label=subject,
            session_label=_normalize_session_label(series.session),
        )

    @property
    def subject_dir(self) -> Path:
        base = self.bids_root / f"sub-{self.subject_label}"
        if self.session_label:
            base = base / f"ses-{self.session_label}"
        return base

    def datatype_dir(self, datatype: str) -> Path:
        return self.subject_dir / datatype

    def derivatives_dir(self, pipeline_name: str, datatype: str = "dwi") -> Path:
        base = self.bids_root / "derivatives" / pipeline_name / f"sub-{self.subject_label}"
        if self.session_label:
            base = base / f"ses-{self.session_label}"
        return base / datatype

    def candidate_directories(
        self, datatype: str, pipeline_name: str, is_derivative: bool
    ) -> List[Path]:
        dirs: List[Path] = []
        if is_derivative:
            dirs.extend([self.datatype_dir("dwi"), self.datatype_dir("misc")])
            dirs.append(self.derivatives_dir(pipeline_name, "dwi"))
        else:
            dirs.append(self.datatype_dir(datatype))

        seen: Set[Path] = set()
        ordered: List[Path] = []
        for path in dirs:
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)
        return ordered


def _is_supported_bids_file(path: Path) -> bool:
    return any(path.name.endswith(ext) for ext in _BIDS_EXTS)


def _discover_series_files(series: SeriesInfo, directories: Iterable[Path]) -> List[Path]:
    """Return files associated with ``series`` found under ``directories``."""

    found: List[Path] = []
    seen: Set[Path] = set()

    for dt_dir in directories:
        if not dt_dir.exists():
            continue
        for candidate in _glob_candidates(dt_dir, series.subject, series.sequence):
            if not candidate.is_file() or candidate in seen:
                continue
            seen.add(candidate)
            found.append(candidate)

    if found:
        return found

    extra = series.extra or {}
    current = extra.get("current_bids")
    if not current:
        return found

    for dt_dir in directories:
        if not dt_dir.exists():
            continue
        for ext in _BIDS_EXTS:
            candidate = dt_dir / f"{current}{ext}"
            if candidate.exists() and candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                found.append(candidate)

    return found


def _log_missing_candidates(series: SeriesInfo, directories: Iterable[Path]) -> None:
    dirs = list(directories)
    if not dirs:
        return

    print(f"No candidates found for '{series.sequence}'")
    for directory in dirs:
        if not directory.exists():
            print(f"  - {directory} (missing)")
            continue

        print(f"  - {directory}")
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name)
        except Exception:
            entries = []

        preview = entries[:5]
        for entry in preview:
            print(f"      * {entry.name}")
        if len(entries) > len(preview):
            print("      …")


def _determine_destination(
    paths: _SeriesPaths,
    source: Path,
    new_base: str,
    pipeline_name: str,
    is_derivative: bool,
) -> Path:
    if is_derivative:
        dest_dir = paths.derivatives_dir(pipeline_name, "dwi")
        new_name = _replace_stem_keep_ext(source, new_base).name
        return dest_dir / new_name
    return _replace_stem_keep_ext(source, new_base)


def _finalize_rename_operations(rename_map: Dict[Path, Path]) -> Dict[Path, Path]:
    for old, new in sorted(rename_map.items(), key=lambda kv: len(str(kv[0])), reverse=True):
        if new.exists():
            if old.resolve() == new.resolve():
                continue
            stem, ext = new.stem, new.suffix
            if new.name.endswith(".nii.gz"):
                stem = new.name[:-7]
                ext = ".nii.gz"
            k = 2
            cand = new
            while cand.exists():
                cand = new.with_name(f"{stem}({k}){ext}")
                k += 1
            rename_map[old] = cand
            new = cand
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))

    return rename_map


def build_preview_names(
    inventory_rows: Iterable[SeriesInfo], schema: SchemaInfo
) -> List[Tuple[SeriesInfo, str, str]]:
    """Build mapping tuples for a set of series.

    Unlike the previous implementation, repetitions are determined *solely* by
    the ``rep`` field supplied with each :class:`SeriesInfo`.  If ``rep`` is
    ``None`` or ``1`` the base name is used as-is.  Any value greater than ``1``
    results in a ``(N)`` suffix being appended, which keeps the first occurrence
    untouched and numbers subsequent ones.  This respects the repeat detection
    logic computed during inventory generation and avoids inferring repeats from
    matching sequence patterns alone.
    """

    out: List[Tuple[SeriesInfo, str, str]] = []
    for s in inventory_rows:
        dt, base = propose_bids_basename(s, schema)
        # Reflect repeats using `_rep-<N>` instead of `(N)` so Preview/Table
        # match the final conversion/heuristic behavior and never introduce
        # parentheses which break some shell invocations.
        if s.rep and s.rep > 1:
            final_base = f"{base}_rep-{s.rep}"
        else:
            final_base = base

        # Clean up any redundant underscores that may have appeared after
        # assembling entities.  We intentionally preserve explicit ``run-``
        # numbers which are now part of ``base`` to avoid collisions between
        # different fieldmap runs.
        final_base = re.sub(r"_+", "_", final_base).strip("_")
        out.append((s, dt, final_base))
    return out


def apply_post_conversion_rename(
    bids_root: Union[str, Path],
    proposals: Iterable[Tuple[SeriesInfo, str, str]],
    also_normalize_fieldmaps: bool = True,
    handle_dwi_derivatives: bool = True,
    derivatives_pipeline_name: str = "dcm2niix",
) -> Dict[Path, Path]:
    bids_root = Path(bids_root)
    rename_map: Dict[Path, Path] = {}

    proposals = list(proposals)

    # main renaming based on proposals
    print(f"Processing {len(proposals)} proposals for renaming:")
    for idx, (series, datatype, new_base) in enumerate(proposals, start=1):
        print(f"  {idx}. {series.sequence} -> {datatype}/{new_base}")

    for series, datatype, new_base in proposals:
        try:
            paths = _SeriesPaths.from_series(bids_root, series)
        except ValueError as exc:
            print(f"  ! Skipping series {series.sequence!r}: {exc}")
            continue

        is_derivative = datatype == "derivatives"
        candidate_dirs = paths.candidate_directories(datatype, derivatives_pipeline_name, is_derivative)
        files = _discover_series_files(series, candidate_dirs)
        if not files:
            _log_missing_candidates(series, candidate_dirs)
            continue

        for src in files:
            if not _is_supported_bids_file(src):
                continue
            dest = _determine_destination(paths, src, new_base, derivatives_pipeline_name, is_derivative)
            if dest == src:
                continue
            rename_map[src] = dest

    # fieldmaps normalization
    if also_normalize_fieldmaps:
        top_fmap = bids_root / "fmap"
        if top_fmap.exists():
            _normalize_fieldmaps(top_fmap, rename_map)
        for fmap_dir in bids_root.glob("sub-*/fmap"):
            _normalize_fieldmaps(fmap_dir, rename_map)
        for fmap_dir in bids_root.glob("sub-*/ses-*/fmap"):
            _normalize_fieldmaps(fmap_dir, rename_map)

    # DWI derivative maps → derivatives/...
    if handle_dwi_derivatives:
        _move_dwi_derivatives(bids_root, derivatives_pipeline_name, rename_map)

    # Execute rename ops
    return _finalize_rename_operations(rename_map)


def normalize_study_name(raw: str) -> str:
    """Return ``raw`` with consecutive duplicate words removed."""

    text = "" if raw is None else str(raw)
    text = text.strip()
    if not text:
        return ""

    pieces: list[str] = []
    last_word: str | None = None
    cursor = 0

    for match in _WORD_RE.finditer(text):
        start, end = match.span()

        if start > cursor:
            separator = text[cursor:start]
            if separator:
                pieces.append(separator)

        word = match.group(0)
        lowered = word.lower()

        if lowered != last_word:
            pieces.append(word)
            last_word = lowered
        else:
            if start > cursor and pieces:
                separator = text[cursor:start]
                if pieces[-1] == separator:
                    pieces.pop()

        cursor = end

    if cursor < len(text):
        pieces.append(text[cursor:])

    return "".join(pieces).strip()


__all__ = [
    "DEFAULT_SCHEMA_DIR",
    "ENABLE_SCHEMA_RENAMER",
    "ENABLE_FIELDMap_NORMALIZATION",
    "ENABLE_DWI_DERIVATIVES_MOVE",
    "DERIVATIVES_PIPELINE_NAME",
    "SequenceHint",
    "DEFAULT_SEQUENCE_HINTS",
    "SEQUENCE_HINTS",
    "SKIP_MODALITIES",
    "SchemaInfo",
    "SeriesInfo",
    "load_bids_schema",
    "build_preview_names",
    "propose_bids_basename",
    "apply_post_conversion_rename",
    "guess_modality",
    "modality_to_container",
    "get_sequence_hint_patterns",
    "set_sequence_hint_patterns",
    "restore_default_sequence_hints",
    "normalize_study_name",
]

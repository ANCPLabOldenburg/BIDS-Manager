"""Legacy regex-dictionary classifier — architecture.md §4.2 layer 3.

This is the fallback classifier ported from BIDS-Manager v0.2.5
(``bids_manager/schema_renamer.py``). It maps DICOM ``SeriesDescription``
strings to a *fine* modality label (``"T1w"``, ``"bold"``, ``"dwi"``, …)
which then resolves to a top-level BIDS datatype via
:func:`modality_to_container`.

It runs after the higher-confidence classifiers (BidsGuess, mne channel
types). It is also exported here so the MRI inventory scanner can fill
the legacy ``modality`` / ``modality_bids`` columns in the v0.2.5 22-col
TSV contract (improvement_plan.md §4).

User-pref persistence (``user_preferences/sequence_dictionary.tsv`` in
v0.2.5) is intentionally not ported here — the GUI will provide a
schema-aware editor for it later, and the regex dictionary itself is
considered legacy.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional

from ..inventory.types import InventoryRow
from .types import Classification


_WORD_RE = re.compile(r"[0-9A-Za-z]+")
_SANITIZE_TOKEN = re.compile(r"[^a-zA-Z0-9]+")

# Explicit ``task-X`` / ``acq-X`` markers that operators sometimes embed
# directly into ``SeriesDescription``. Used by :func:`guess_task_from_text`
# and :func:`extract_acq_token`.
_TASK_TOKEN = re.compile(r"(?:^|[_-])task-([a-zA-Z0-9]+)", re.IGNORECASE)
_ACQ_TOKEN = re.compile(r"(?:^|[_-])acq-([a-zA-Z0-9]+)", re.IGNORECASE)
_RUN_TOKEN = re.compile(r"(?:^|[_-])run-?0*(\d+)", re.IGNORECASE)
_DIR_TOKEN = re.compile(r"(?<![a-z0-9])(ap|pa|lr|rl)(?![a-z0-9])", re.IGNORECASE)

# Curated task-label hints — small enough to avoid over-eager assignments
# on functional scans that don't explicitly encode paradigm information
# (port of v0.2.5 ``schema_renamer.TASK_HINT_PATTERNS``).
TASK_HINT_PATTERNS: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    (
        ("rest", ("rs", "_rs", "rs_", "rest", "resting")),
        ("movie", ("movie",)),
        ("nback", ("nback", "n-back")),
        ("flanker", ("flanker",)),
        ("stroop", ("stroop",)),
        ("motor", ("motor",)),
        ("checkerboard", ("checker", "checkerboard")),
        ("exec", ("exec",)),
        ("paradigm", ("paradigm", "paradigma")),
        ("sparse", ("sparse",)),
        ("activation", ("activation",)),
        ("task", ("task",)),
    )
)

SKIP_MODALITIES: frozenset[str] = frozenset({"scout", "report"})


@dataclass(frozen=True)
class SequenceHint:
    """One regex-dictionary entry: a fine modality label + match patterns."""

    label: str
    suffix: Optional[str]      # BIDS suffix produced if matched
    datatype: Optional[str]    # Top-level BIDS datatype (anat/func/dwi/fmap/...)
    patterns: tuple[str, ...]
    container_override: Optional[str] = None  # e.g. "derivatives" for DWI maps


# Patterns ported verbatim from the v0.2.5 trunk (schema_renamer.DEFAULT_SEQUENCE_HINTS).
SEQUENCE_HINTS: OrderedDict[str, SequenceHint] = OrderedDict(
    (
        (
            "dwi_derivative",
            SequenceHint(
                "dwi_derivative", None, "dwi",
                ("_adc", "_fa", "_tracew", "_colfa", "_expadc",
                 " adc", " fa", " tracew", " colfa", " expadc",
                 "adc_", "adc ", "trace"),
                container_override="derivatives",
            ),
        ),
        ("T1w", SequenceHint("T1w", "T1w", "anat",
            ("t1w", "t1-weight", "t1_", "t1 ", "mprage", "mp2rage", "tfl3d", "fspgr"))),
        ("T2w", SequenceHint("T2w", "T2w", "anat",
            ("t2w", "space", "tse", "t2", "hires"))),
        ("FLAIR", SequenceHint("FLAIR", "FLAIR", "anat", ("flair",))),
        ("MTw", SequenceHint("MTw", "MTw", "anat", ("gre-mt", "gre_mt", "mt"))),
        ("PDw", SequenceHint("PDw", "PDw", "anat", ("gre-nm", "gre_nm"))),
        ("scout", SequenceHint("scout", "localizer", "anat", ("localizer", "scout"))),
        ("report", SequenceHint("report", "reports", "anat",
            ("phoenixzipreport", "phoenix document", ".pdf", "report"))),
        ("SBRef", SequenceHint("SBRef", "sbref", "func",
            ("sbref", "type-ref", "reference", "refscan", " ref", "_ref"))),
        ("physio", SequenceHint("physio", "physio", "func",
            ("physiolog", "physio", "pulse", "resp"))),
        ("bold", SequenceHint("bold", "bold", "func",
            ("fmri", "bold", "task-", "study", "test", "epi", "cmrr"))),
        ("dwi", SequenceHint("dwi", "dwi", "dwi", ("dti", "dwi", "diff"))),
        ("fmap", SequenceHint("fmap", "phasediff", "fmap",
            ("gre_field", "fieldmapping", "_fmap", "fmap", "phase",
             "magnitude", "b0rf", "b0_map", "b0map", "b0"))),
    )
)


def _normalize_series(text: Optional[str]) -> tuple[str, set[str], list[str]]:
    raw = text or ""
    lower = raw.lower()
    tokens = list(filter(None, _SANITIZE_TOKEN.sub(" ", lower).split()))
    return lower, set(tokens), tokens


def _score_patterns(patterns: tuple[str, ...], lowered: str, token_set: set[str], tokens: list[str]) -> Optional[tuple[int, int]]:
    """Return a (kind, length) ranking describing how well ``patterns`` match ``series``."""
    best: Optional[tuple[int, int]] = None
    for pat in patterns:
        pat_lower = pat.lower()
        if not pat_lower:
            continue
        token_candidate = _SANITIZE_TOKEN.sub("", pat_lower)
        score: Optional[tuple[int, int]] = None
        if token_candidate and token_candidate in token_set:
            score = (3, len(token_candidate))
        elif (
            token_candidate
            and len(token_candidate) >= 3
            and any(tok.startswith(token_candidate) for tok in tokens)
        ):
            score = (2, len(token_candidate))
        elif pat_lower in lowered:
            score = (1, len(pat_lower))
        if score and (best is None or score > best):
            best = score
    return best


def guess_modality(series: str) -> str:
    """Return the fine modality label whose patterns best describe ``series``."""

    lowered, token_set, tokens = _normalize_series(series)
    best_label = "unknown"
    best_score: Optional[tuple[int, int, int]] = None
    for idx, (label, hint) in enumerate(SEQUENCE_HINTS.items()):
        score = _score_patterns(hint.patterns, lowered, token_set, tokens)
        if score is None:
            continue
        ranked = (score[0], score[1], -idx)
        if best_score is None or ranked > best_score:
            best_label = label
            best_score = ranked
    return best_label


def modality_to_container(modality: str) -> str:
    """Translate a fine modality label into its top-level BIDS folder."""

    hint = SEQUENCE_HINTS.get(modality)
    if hint is None:
        return ""
    if hint.container_override:
        return hint.container_override
    return hint.datatype or ""


def normalize_study_name(raw: str) -> str:
    """Collapse consecutive duplicate words in a study name."""

    text = "" if raw is None else str(raw).strip()
    if not text:
        return ""
    pieces: list[str] = []
    last_word: Optional[str] = None
    cursor = 0
    for match in _WORD_RE.finditer(text):
        start, end = match.span()
        if start > cursor:
            sep = text[cursor:start]
            if sep:
                pieces.append(sep)
        word = match.group(0)
        lowered = word.lower()
        if lowered != last_word:
            pieces.append(word)
            last_word = lowered
        else:
            if start > cursor and pieces:
                sep = text[cursor:start]
                if pieces[-1] == sep:
                    pieces.pop()
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:])
    return "".join(pieces).strip()


def _sanitize_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = _SANITIZE_TOKEN.sub("", str(value)).strip()
    return cleaned or None


def _strip_run_tokens(text: str) -> str:
    text = re.sub(r"_?run-\d+_?", "_", text, flags=re.IGNORECASE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _match_task_hint_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    explicit = _TASK_TOKEN.search(text)
    if explicit:
        token = _sanitize_token(explicit.group(1))
        if token:
            return token
    low = text.lower()
    for label, patterns in TASK_HINT_PATTERNS.items():
        for pat in patterns:
            if pat and pat.lower() in low:
                return _sanitize_token(label)
    return None


def guess_task_from_text(*candidates: Optional[str]) -> Optional[str]:
    """Return a task label inferred from any of ``candidates``, or ``None``."""
    cleaned = [_strip_run_tokens(c) if c else c for c in candidates]
    for c in cleaned:
        task = _match_task_hint_from_text(c) or _match_task_hint_from_text(_sanitize_token(c))
        if task:
            return task
    return None


def extract_acq_token(text: Optional[str]) -> Optional[str]:
    """Return the most descriptive ``acq-...`` label embedded in ``text``."""
    if not text:
        return None
    candidates = []
    for match in _ACQ_TOKEN.finditer(text):
        token = _sanitize_token(match.group(1))
        if token:
            candidates.append((token, match.start()))
    if not candidates:
        return None
    best, _ = max(candidates, key=lambda item: (len(item[0]), item[1]))
    return best


def extract_direction_token(text: Optional[str]) -> Optional[str]:
    """Return AP/PA/LR/RL if present in ``text``, else ``None``."""
    if not text:
        return None
    cleaned = _SANITIZE_TOKEN.sub(" ", text).lower()
    m = _DIR_TOKEN.search(cleaned)
    return m.group(1) if m else None


# Mapping from fine modality label → BIDS suffix the legacy classifier proposes.
# Keys not present here have no suffix proposal.
_MODALITY_TO_SUFFIX: dict[str, str] = {
    "T1w": "T1w",
    "T2w": "T2w",
    "FLAIR": "FLAIR",
    "MTw": "MTw",
    "PDw": "PDw",
    "SBRef": "sbref",
    "physio": "physio",
    "bold": "bold",
    "dwi": "dwi",
    "fmap": "phasediff",
    "scout": "localizer",
    "report": "reports",
}


# DWI scanner-derivative detection (architecture.md §4.2; BIDS 1.11+ rule
# group ``dwi.ScannerDerivatives``). When ``SeriesDescription`` ends in one
# of these tokens, the DICOM series is a scanner-computed derivative of a
# DWI acquisition and should be emitted with its own BIDS suffix in
# ``dwi/``, not as a raw ``_dwi`` file. ``TENSOR`` has no canonical raw
# suffix; we route it to ``derivatives/`` per the v0.2.5 convention.
_DWI_DERIVATIVE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    # (regex, BIDS suffix, target datatype)
    (r"(?:^|[_-])(?:colfa|col[_-]?fa)(?=$|[_-])", "colFA", "dwi"),
    (r"(?:^|[_-])(?:expadc|exp[_-]?adc)(?=$|[_-])", "expADC", "dwi"),
    (r"(?:^|[_-])tracew?(?=$|[_-])", "trace", "dwi"),
    (r"(?:^|[_-])tensor(?=$|[_-])", "TENSOR", "derivatives"),
    (r"(?:^|[_-])fa(?=$|[_-])", "FA", "dwi"),
    (r"(?:^|[_-])adc(?=$|[_-])", "ADC", "dwi"),
    (r"(?:^|[_-])s0[_-]?map(?=$|[_-])", "S0map", "dwi"),
)


def detect_dwi_derivative(sequence: Optional[str]) -> Optional[tuple[str, str]]:
    """Return ``(bids_suffix, target_datatype)`` if ``sequence`` is a known
    scanner-computed DWI derivative, else ``None``.

    Only fires when the marker token is at a word boundary so generic
    sequences like ``MPRAGE_FATsat`` are not mis-detected as ``FA``.
    """

    if not sequence:
        return None
    low = sequence.lower()
    for pattern, suffix, datatype in _DWI_DERIVATIVE_PATTERNS:
        if re.search(pattern, low):
            return suffix, datatype
    return None


# B0-reference detection. Matches sequences whose name contains a clear
# ``b0`` marker indicating a single-volume (or short) phase-encoding
# reference scan acquired alongside DWI/func runs for distortion
# correction. BIDS treats these as PEpolar fmaps (``fmap/_epi``); the
# user can re-route to ``dwi/_dwi`` via the GUI if it's a real b=0 DWI
# acquisition rather than a fmap reference.
_B0_REFERENCE_RE = re.compile(
    r"(?:^|[_-])(?:"
    r"b0[_-]?map|"          # b0_map / b0map
    r"\d*b0(?=$|[_-])|"     # 1b0 / 15b0 / b0 (boundary)
    r"b0[_-]?ref|"          # b0_ref
    r"b0rf"                 # Siemens specific
    r")",
    re.IGNORECASE,
)


def looks_like_b0_reference(sequence: Optional[str]) -> bool:
    """True when ``sequence`` contains a recognisable B0 / PEpolar marker."""
    if not sequence:
        return False
    return bool(_B0_REFERENCE_RE.search(sequence))


def classify(rows: Iterable[InventoryRow]) -> list[Classification]:
    """Legacy regex/sequence-dictionary classifier — architecture.md §4.2 layer 3.

    For each :class:`InventoryRow`, produces at most one
    :class:`Classification` derived from:

    * The fine modality label that the inventory scanner pre-computed
      (``row.fine_modality``) or, if absent, :func:`guess_modality` on the
      ``series_description``.
    * Best-effort task / direction / acquisition tokens extracted from the
      sequence text.

    Confidence is fixed at ``0.4`` (lower than the BidsGuess classifier so
    the planner prefers BidsGuess when both fire).
    """

    out: list[Classification] = []
    for row in rows:
        sequence = row.series_description or ""

        # 1. DWI scanner-derivative detection (FA / ADC / trace / colFA /
        #    expADC / S0map / TENSOR). Runs *before* the legacy modality
        #    map so a sequence like ``..._dwi_FA`` is correctly emitted as
        #    suffix ``FA`` rather than ``dwi``.
        dwi_deriv = detect_dwi_derivative(sequence)
        if dwi_deriv:
            suffix, datatype = dwi_deriv
            entities: dict[str, str] = {}
            direction = extract_direction_token(sequence)
            if direction:
                entities["direction"] = direction.upper()
            acq = extract_acq_token(sequence)
            if acq:
                entities["acquisition"] = acq
            rationale = (
                f"sequence_dict: DWI scanner-derivative "
                f"(suffix={suffix!r}, datatype={datatype!r})"
            )
            out.append(
                Classification(
                    row_id=row.row_id,
                    classifier="sequence_dict",
                    datatype=datatype,
                    suffix=suffix,
                    candidate_entities=entities,
                    confidence=0.45,  # higher than generic fallback
                    rationale=rationale,
                    skip=False,
                )
            )
            continue

        modality = row.fine_modality
        if not modality:
            modality = guess_modality(sequence)
        suffix = _MODALITY_TO_SUFFIX.get(modality)
        if not suffix:
            continue

        skip = modality in SKIP_MODALITIES
        datatype = modality_to_container(modality) or ""
        if not datatype:
            continue

        entities = {}

        # Direction is helpful for fmap epi / dwi / func.
        direction = extract_direction_token(sequence)
        if direction:
            entities["direction"] = direction.upper()

        # Acquisition label, if explicitly encoded in the series text.
        acq = extract_acq_token(sequence)
        if acq:
            entities["acquisition"] = acq

        # Task label for func/sbref/physio. The planner / GUI may override.
        if suffix in ("bold", "sbref", "physio"):
            task = guess_task_from_text(sequence)
            if task:
                entities["task"] = task

        rationale = f"sequence_dict regex matched modality={modality!r}"

        out.append(
            Classification(
                row_id=row.row_id,
                classifier="sequence_dict",
                datatype=datatype,
                suffix=suffix,
                candidate_entities=entities,
                confidence=0.0 if skip else 0.4,
                rationale=rationale,
                skip=skip,
            )
        )
    return out


__all__ = [
    "SequenceHint",
    "SEQUENCE_HINTS",
    "SKIP_MODALITIES",
    "TASK_HINT_PATTERNS",
    "guess_modality",
    "modality_to_container",
    "normalize_study_name",
    "guess_task_from_text",
    "extract_acq_token",
    "extract_direction_token",
    "detect_dwi_derivative",
    "looks_like_b0_reference",
    "classify",
]

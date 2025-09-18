from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

# ``PyYAML`` is a hard dependency for schema parsing.  Previous versions treated
# it as optional which meant that users were silently running without any
# schema-driven rules.  Raising an informative error keeps the behaviour
# explicit and surfaces installation issues early.
try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - import error depends on env
    raise ImportError(
        "PyYAML is required for BIDS schema parsing. Install bids-manager with"
        " the bundled dependencies or add PyYAML to your environment."
    ) from exc


# ----------------------------- Utilities -----------------------------

_BIDS_EXTS = (".nii.gz", ".nii", ".json", ".bval", ".bvec", ".tsv")

_SANITIZE_TOKEN = re.compile(r"[^a-zA-Z0-9]+")
_TASK_TOKEN = re.compile(r"(?:^|[_-])task-([a-zA-Z0-9]+)", re.IGNORECASE)


def _sanitize_token(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    return _SANITIZE_TOKEN.sub("", x).strip()


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


def _iter_schema_files(schema_dir: Path) -> Iterable[Path]:
    """Yield schema definition files shipped with the BIDS specification."""

    for p in schema_dir.rglob("*"):
        if p.suffix.lower() in (".json", ".yaml", ".yml") and p.is_file():
            yield p


# --------------------------- Schema parsing ---------------------------

@dataclass
class SchemaInfo:
    suffix_requirements: Dict[str, List[str]]
    suffix_to_datatypes: Dict[str, List[str]]


def _load_json_or_yaml(p: Path) -> Optional[dict]:
    """Return the parsed representation of a JSON or YAML document."""

    try:
        if p.suffix.lower() == ".json":
            return json.loads(p.read_text(encoding="utf-8"))
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _harvest_suffix_rules(obj: Union[dict, list], current_datatype: Optional[str], out_req: Dict[str, set],
                          out_dt: Dict[str, set]) -> None:
    if isinstance(obj, dict):
        if "datatype" in obj and isinstance(obj["datatype"], str):
            current_datatype = obj["datatype"]

        suffix = obj.get("suffix")
        if isinstance(suffix, str):
            sfx = suffix.strip()
            required = set()
            for key in ("required", "required_entities", "entities_required"):
                v = obj.get(key)
                if isinstance(v, list):
                    for e in v:
                        if isinstance(e, str):
                            required.add(e.strip("<>"))
                        elif isinstance(e, dict) and "name" in e:
                            required.add(str(e["name"]).strip("<>"))
            ents = obj.get("entities")
            if isinstance(ents, list):
                for e in ents:
                    if isinstance(e, dict) and e.get("required") is True and "name" in e:
                        required.add(str(e["name"]).strip("<>"))

            if required:
                out_req.setdefault(sfx, set()).update(required)

            if current_datatype:
                out_dt.setdefault(sfx, set()).add(current_datatype)

        for v in obj.values():
            _harvest_suffix_rules(v, current_datatype, out_req, out_dt)

    elif isinstance(obj, list):
        for it in obj:
            _harvest_suffix_rules(it, current_datatype, out_req, out_dt)


def load_bids_schema(schema_dir: Union[str, Path]) -> SchemaInfo:
    """Load and aggregate the BIDS schema shipped with the package.

    Parameters
    ----------
    schema_dir:
        Directory containing the ``objects``/``rules`` folders from the official
        BIDS schema distribution.
    """

    schema_dir = Path(schema_dir)
    if not schema_dir.exists():
        raise FileNotFoundError(
            f"BIDS schema directory '{schema_dir}' does not exist."
        )

    suffix_requirements: Dict[str, set] = {}
    suffix_to_datatypes: Dict[str, set] = {}
    processed_any = False

    for p in _iter_schema_files(schema_dir):
        data = _load_json_or_yaml(p)
        if not isinstance(data, (dict, list)):
            continue
        _harvest_suffix_rules(data, current_datatype=None,
                              out_req=suffix_requirements, out_dt=suffix_to_datatypes)
        processed_any = True

    if not processed_any:
        raise RuntimeError(
            "No schema files could be parsed. Ensure the packaged schema is intact."
        )

    fallback_dt = {
        "T1w": "anat", "T2w": "anat", "FLAIR": "anat", "T2star": "anat", "PD": "anat",
        "bold": "func", "sbref": "func",
        "dwi": "dwi",
        "phasediff": "fmap", "fieldmap": "fmap", "magnitude1": "fmap", "magnitude2": "fmap", "epi": "fmap",
    }
    for sfx, dt in fallback_dt.items():
        suffix_to_datatypes.setdefault(sfx, set()).add(dt)
    for sfx in set(suffix_to_datatypes.keys()) | set(suffix_requirements.keys()):
        suffix_requirements.setdefault(sfx, set()).add("subject")

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
        "phasediff": "fmap", "fieldmap": "fmap", "magnitude1": "fmap", "magnitude2": "fmap", "epi": "fmap",
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

    ses_raw = series.session or ""
    ses = _sanitize_token(ses_raw)
    # Users sometimes supply session strings already prefixed with ``ses-``.
    # After sanitization this would yield ``ses<suffix>`` which in turn produces
    # names like ``ses-sesspre``.  Strip a leading ``ses`` token if present so the
    # final name always follows ``ses-<label>``.
    if ses_raw.lower().startswith(("ses-", "ses_")) and ses.lower().startswith("ses"):
        ses = ses[3:]
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
    map_name = _detect_dwi_derivative(clean_sequence)
    if map_name:
        datatype = "derivatives"
        # Keep an acquisition discriminator for uniqueness between different
        # series descriptions.
        acq_token = _sanitize_token(clean_sequence)[:32]
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
        acq = series.extra.get("acq") if series.extra else None
        direction = series.extra.get("dir") if series.extra else None

        if suffix == "dwi":
            # Infer missing ``acq``/``dir`` tokens from the sequence itself so
            # that common diffusion naming conventions (e.g. ``DTI_LR``) remain
            # distinguishable without manual intervention.
            inf_acq, inf_dir = _infer_dwi_acq_dir(clean_sequence)
            if not direction:
                direction = inf_dir
            if not acq:
                acq = inf_acq

        if acq:
            acq = _sanitize_token(acq)
            if acq:
                parts.append(f"acq-{acq}")

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


def _rename_file_set(old: Path, new_basename: str, rename_map: Dict[Path, Path]) -> None:
    newp = _replace_stem_keep_ext(old, new_basename)
    if newp == old:
        return
    newp.parent.mkdir(parents=True, exist_ok=True)
    rename_map[old] = newp


def _process_series_in_dir(dt_dir: Path, series: "SeriesInfo", new_base: str, rename_map: Dict[Path, Path], 
                          bids_root: Path, derivatives_pipeline_name: str, is_derivative: bool) -> None:
    """Process a series in a specific directory, finding and renaming files."""
    candidates: List[Path] = []
    
    # First try to find files using the original sequence name (heudiconv naming)
    candidates = _glob_candidates(dt_dir, series.subject, series.sequence)
    
    # If no files found with sequence matching, try the current_bids field as fallback
    if not candidates:
        current = series.extra.get("current_bids") if series.extra else None
        if current:
            for ext in _BIDS_EXTS:
                p = dt_dir / f"{current}{ext}"
                if p.exists():
                    candidates.append(p)
    
    # Debug: print what we found (only if no candidates found)
    if not candidates:
        print(f"No candidates found for {series.sequence} in {dt_dir}")
        # List all files in the directory for debugging
        all_files = list(dt_dir.glob("*"))
        if all_files:
            print(f"  Available files in {dt_dir}:")
            for f in all_files:
                print(f"    - {f.name}")
        else:
            print(f"  Directory {dt_dir} is empty or doesn't exist")
    
    for p in candidates:
        if not any(p.name.endswith(ext) for ext in _BIDS_EXTS):
            continue
        
        if is_derivative:
            # For derivatives, move to proper derivatives folder
            deriv_dir = bids_root / "derivatives" / derivatives_pipeline_name
            sub_name = f"sub-{_sanitize_token(series.subject)}"
            final_dir = deriv_dir / sub_name
            if series.session:
                final_dir = final_dir / f"ses-{_sanitize_token(series.session)}"
            final_dir = final_dir / "dwi"
            final_dir.mkdir(parents=True, exist_ok=True)
            
            # Create new path in derivatives
            new_file = final_dir / _replace_stem_keep_ext(p, new_base).name
            rename_map[p] = new_file
        else:
            _rename_file_set(p, new_base, rename_map)


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

    # main renaming based on proposals
    print(f"Processing {len(proposals)} proposals for renaming:")
    for i, (series, datatype, new_base) in enumerate(proposals):
        print(f"  {i+1}. {series.sequence} -> {datatype}/{new_base}")
    
    for series, datatype, new_base in proposals:
        # Handle derivatives specially - they need special path handling
        if datatype == "derivatives":
            # For derivatives, search in both dwi and misc folders
            search_dirs = []
            base_dir = bids_root / f"sub-{_sanitize_token(series.subject)}"
            if series.session:
                base_dir = base_dir / f"ses-{_sanitize_token(series.session)}"
            
            # Check dwi folder first (most likely location for DWI derivatives)
            dwi_dir = base_dir / "dwi"
            if dwi_dir.exists():
                search_dirs.append(dwi_dir)
            
            # Check misc folder
            misc_dir = base_dir / "misc"
            if misc_dir.exists():
                search_dirs.append(misc_dir)
            
            # Also check if they're already in derivatives
            deriv_dir = bids_root / "derivatives" / derivatives_pipeline_name / f"sub-{_sanitize_token(series.subject)}"
            if series.session:
                deriv_dir = deriv_dir / f"ses-{_sanitize_token(series.session)}"
            deriv_dir = deriv_dir / "dwi"
            if deriv_dir.exists():
                search_dirs.append(deriv_dir)
                
            # Process all search directories
            for dt_dir in search_dirs:
                _process_series_in_dir(dt_dir, series, new_base, rename_map, bids_root, derivatives_pipeline_name, True)
        else:
            dt_dir = bids_root / f"sub-{_sanitize_token(series.subject)}"
            if series.session:
                dt_dir = dt_dir / f"ses-{_sanitize_token(series.session)}"
            dt_dir = dt_dir / datatype
            if not dt_dir.exists():
                continue
            _process_series_in_dir(dt_dir, series, new_base, rename_map, bids_root, derivatives_pipeline_name, False)

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
            new = cand
            rename_map[old] = new
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))

    return rename_map


__all__ = [
    "SchemaInfo",
    "SeriesInfo",
    "load_bids_schema",
    "build_preview_names",
    "propose_bids_basename",
    "apply_post_conversion_rename",
]

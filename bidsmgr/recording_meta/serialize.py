"""Load and build :class:`RecordingMetaSpec` objects.

The spec is a plain JSON document (our own layout, not tied to any external
tool). The convert and metadata verbs read it via ``--recording-meta``; when no
spec is supplied they fall back to :func:`default_spec`, which is EMPTY.

It did not used to be. It carried ``PowerLineFrequency = 50``, inherited from
the old ``--line-freq`` flag, and that number reached the sidecar of every
recording whose header did not state one. Fifty hertz is Europe. A recording
made in the US, Canada, Japan or Brazil is sixty, and the tool asserted
otherwise in a BIDS-REQUIRED field, in a way indistinguishable from a
measurement somebody took. Notch-filtering at 50 on 60 Hz data leaves the
artefact exactly where it was.

Nothing replaces it, because nothing needs to: mne-bids writes
``PowerLineFrequency: "n/a"`` of its own accord when ``raw.info["line_freq"]``
is unset, which is valid BIDS (the schema types the field ``anyOf`` number or
the literal ``"n/a"``) and is the true statement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .models import AcquisitionSpec, RecordingMetaSpec

# Suffix for the scaffold the scan verb writes next to an inventory TSV
# (``<inventory>.tsv.recording_meta.json``). The convert/metadata verbs
# auto-discover it when ``--recording-meta`` is not given, mirroring the
# ``files_by_uid`` sidecar convention.
RECORDING_META_SIDECAR = ".recording_meta.json"


def scaffold_sidecar_path(tsv_path) -> Path:
    """Return the recording-metadata scaffold path beside an inventory TSV."""
    p = Path(tsv_path)
    return p.with_name(p.name + RECORDING_META_SIDECAR)


def default_spec(power_line_freq: Optional[float] = None) -> RecordingMetaSpec:
    """An EMPTY spec: no enrichment, and no power-line frequency.

    ``power_line_freq`` stays a parameter because a caller that genuinely
    knows the answer should be able to say so. What changed is the DEFAULT:
    it is ``None``, so a value reaches the sidecar only when a header, an
    inventory cell or a template supplied one.
    """
    return RecordingMetaSpec(defaults=AcquisitionSpec(power_line_freq=power_line_freq))


def load_spec(path: Path) -> RecordingMetaSpec:
    """Read and validate a recording-metadata JSON document.

    Raises the underlying ``OSError`` / JSON / Pydantic error so the caller can
    report a precise message; callers that want a forgiving load should catch.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    return RecordingMetaSpec.model_validate(data)


def dump_spec(spec: RecordingMetaSpec) -> str:
    """Serialise a spec to pretty JSON (drops unset leaves for readability)."""
    return json.dumps(spec.model_dump(exclude_none=True), indent=2) + "\n"


__all__ = ["default_spec", "load_spec", "dump_spec"]

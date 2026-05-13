"""EEG / MEG / iEEG converter backend — wraps ``mne_bids.write_raw_bids``.

The orchestrator (``cli/convert.py``) drives parallelism per task; this
backend converts one recording per call. mne-bids does the heavy lifting:
``write_raw_bids(format="auto")`` writes the data file plus
``*_channels.tsv``, the datatype JSON sidecar, ``*_events.tsv``, and (when
channel positions are present) ``*_electrodes.tsv`` + ``*_coordsystem.json``.
Dataset-level metadata (``dataset_description.json``, ``participants.tsv``,
README, CHANGES) is filled afterwards by ``bidsmgr-metadata`` — same flow
as the MRI side.

Reference: ported from
``BIDS-Manager/bids_manager/run_mne_bids.py`` (v0.2.5), refactored
to fit bidsmgr's per-task ``ConverterBackend`` Protocol.
"""

from __future__ import annotations

import contextlib
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Optional

from ..types import ConvertResult, ConvertTask


@contextlib.contextmanager
def _noop():
    yield

log = logging.getLogger(__name__)


# Datatypes this backend claims. mne-bids supports these directly.
_SUPPORTED_DATATYPES: frozenset[str] = frozenset({"eeg", "meg", "ieeg", "nirs"})


class MneBidsBackend:
    """Convert raw EEG/MEG/iEEG/NIRS recordings via mne-bids.

    Per-row knobs come from the :class:`ConvertTask` (set by the CLI
    from the inventory TSV's ``line_freq`` / ``montage`` columns). The
    constructor takes only fallback defaults used when those columns
    are blank — typical case is the user supplies a single dataset-
    wide default at scan time and the per-row column carries it
    forward.

    What the backend writes (via mne-bids):

    * The recording in its native format where mne-bids supports BIDS
      I/O (EDF/BDF/BrainVision/EEGLAB/FIF/CTF/KIT), or a re-encoded
      sibling otherwise.
    * ``*_channels.tsv`` (always — one row per recording channel).
    * The datatype JSON sidecar with ``PowerLineFrequency``
      (BIDS-required for EEG/MEG/iEEG, populated from the row's
      ``line_freq``).
    * ``*_events.tsv`` from ``raw.annotations`` (auto for EDF+).
    * ``*_electrodes.tsv`` + ``*_coordsystem.json`` when a montage is
      applied (per-row ``montage`` column, fallback to constructor).
    """

    name = "mne_bids"

    def __init__(
        self,
        *,
        line_freq: Optional[float] = 50.0,
        montage: Optional[str] = None,
    ) -> None:
        # Fallback defaults used when the per-task value is None/blank.
        self.default_line_freq = line_freq
        self.default_montage = montage

    def can_handle(self, task: ConvertTask) -> bool:
        if task.datatype not in _SUPPORTED_DATATYPES:
            return False
        if not task.source_files or not task.basename:
            return False
        return True

    def convert(self, task: ConvertTask, staging_dir: Path) -> ConvertResult:
        t0 = time.monotonic()
        try:
            return self._convert_inner(task, staging_dir, t0)
        except Exception as exc:
            log.exception("mne_bids: unexpected error for %s", task.basename)
            return ConvertResult(
                task=task, success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=time.monotonic() - t0,
            )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _convert_inner(
        self, task: ConvertTask, staging_dir: Path, t0: float,
    ) -> ConvertResult:
        # Lazy imports — mne is heavy. The pkg_resources warning from
        # legacy mne / mne_bids versions is silenced for tidiness.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pkg_resources.*")
            try:
                import mne
                from mne_bids import BIDSPath, write_raw_bids
            except ImportError as exc:
                return ConvertResult(
                    task=task, success=False,
                    error=(
                        "mne / mne-bids not installed; cannot convert "
                        f"{task.datatype} row {task.basename!r}: {exc}"
                    ),
                    duration_s=time.monotonic() - t0,
                )

            # EEG/MEG tasks carry exactly one source file (or one folder
            # for .ds/.mff). Pick the first; ignore extras defensively.
            source = task.source_files[0]
            if not source.exists():
                return ConvertResult(
                    task=task, success=False,
                    error=f"source not found: {source}",
                    duration_s=time.monotonic() - t0,
                )

            try:
                raw = mne.io.read_raw(str(source), preload=False, verbose="ERROR")
            except Exception as exc:
                return ConvertResult(
                    task=task, success=False,
                    error=f"mne.io.read_raw failed: {type(exc).__name__}: {exc}",
                    duration_s=time.monotonic() - t0,
                )

            # Per-row line_freq (TSV) wins; constructor default is the
            # dataset-wide fallback.
            line_freq = task.line_freq
            if line_freq is None:
                line_freq = self.default_line_freq
            if line_freq is not None and not raw.info.get("line_freq"):
                with raw.info._unlock() if hasattr(raw.info, "_unlock") else _noop():
                    raw.info["line_freq"] = float(line_freq)

            # Per-row montage (TSV) wins; constructor default is the
            # fallback. Standard 10-20-style montages are an EEG / iEEG /
            # NIRS concept — MEG sensors have intrinsic positions in
            # scanner space, and applying an EEG montage's channel rename
            # to MEG-Neuromag data routinely collides (e.g. ``MEG 0113``
            # + ``EEG 001`` channel sets after non-alphanumeric stripping).
            # Skip silently for MEG so the user's dataset-wide setting
            # doesn't crash MEG rows.
            montage_name = task.montage or self.default_montage
            if montage_name and task.datatype != "meg":
                _apply_standard_montage(raw, montage_name, source)

            # mne-bids writes ``sub-XXX/[ses-Y/]<datatype>/...`` *inside*
            # its ``root``. The orchestrator hands us a per-subject (or
            # per-session) staging dir; we need to walk up to the BIDS
            # root so mne-bids doesn't produce ``sub-XXX/sub-XXX/...``.
            bids_root_for_mne = _find_bids_root(staging_dir)
            bids_root_for_mne.mkdir(parents=True, exist_ok=True)

            bids_path = BIDSPath(
                subject=task.subject,
                session=task.session,
                task=task.entities.get("task") or None,
                run=_coerce_run(task.entities.get("run")),
                datatype=task.datatype,
                root=str(bids_root_for_mne),
            )

            try:
                # ``overwrite=True`` lets sibling tasks for the same
                # subject share metadata files (coordsystem.json,
                # electrodes.tsv) without mne-bids' default "refuse to
                # overwrite" guard rejecting later writes. The staging
                # tree is fresh per convert run so we never clobber
                # already-committed BIDS data — atomic-rename only
                # promotes the staged sub-XXX into the live tree on
                # success.
                write_raw_bids(
                    raw,
                    bids_path,
                    overwrite=True,
                    format="auto",
                    verbose="ERROR",
                )
            except Exception as exc:
                return ConvertResult(
                    task=task, success=False,
                    error=(
                        f"mne_bids.write_raw_bids failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    duration_s=time.monotonic() - t0,
                )

        # Collect outputs. mne-bids may emit:
        #   <bids_root>/sub-X/[ses-Y/]<datatype>/sub-X..._<datatype>.<ext>
        #   plus channels.tsv, datatype JSON, events.tsv, electrodes.tsv,
        #   coordsystem.json.
        sub_dir = bids_root_for_mne / f"sub-{task.subject}"
        if task.session:
            sub_dir = sub_dir / f"ses-{task.session}"
        datatype_dir = sub_dir / task.datatype
        if not datatype_dir.is_dir():
            return ConvertResult(
                task=task, success=False,
                error=(
                    f"mne_bids reported success but no {task.datatype} "
                    f"output dir found at {datatype_dir}"
                ),
                duration_s=time.monotonic() - t0,
            )
        staged = sorted(datatype_dir.glob(f"sub-{task.subject}_*"))
        if not staged:
            return ConvertResult(
                task=task, success=False,
                error="mne_bids produced no output files",
                duration_s=time.monotonic() - t0,
            )

        return ConvertResult(
            task=task, staged_files=tuple(staged), success=True,
            duration_s=time.monotonic() - t0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_standard_montage(raw, montage_name: str, source: Path) -> None:
    """Apply a built-in mne montage to ``raw``, in place.

    Workaround for EDF padding artefacts: PhysioNet pads channel
    names to 16 ASCII chars with trailing dots (``Fc5.``, ``C5..``).
    Some Siemens/Brain Products vendors pad with spaces. Strip both
    before matching against the montage's canonical names so
    ``electrodes.tsv`` ends up with real coordinates instead of
    ``n/a``. Originals are preserved in ``raw.annotations`` /
    ``channels.tsv``; only the in-memory ``ch_names`` is normalised.
    """
    import re

    import mne  # local import — caller already triggered the heavy import

    try:
        montage = mne.channels.make_standard_montage(montage_name)
    except (ValueError, KeyError) as exc:
        log.warning(
            "unknown montage name %r for %s: %s",
            montage_name, source.name, exc,
        )
        return

    # Build a normalisation map: original ch_name → cleaned ch_name.
    # ``re.sub('[^A-Za-z0-9]', '', name)`` drops dots and whitespace
    # while preserving alphanumerics; mne's set_montage with
    # ``match_case=False`` then matches ``Fc5`` → montage's ``FC5``.
    #
    # Pre-seed the seen-names set with channels that are ALREADY in
    # alphanumeric form (``MEG0113`` without space). If a different
    # channel like ``MEG 0113`` would strip to the same name, skip the
    # rename so we don't introduce a duplicate.
    original_names = list(raw.ch_names)
    seen_cleaned: set[str] = {
        ch for ch in original_names
        if re.sub(r"[^A-Za-z0-9]", "", ch) == ch  # already clean
    }

    rename: dict[str, str] = {}
    for ch in original_names:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", ch)
        if not cleaned or cleaned == ch:
            continue  # already in seen_cleaned via the pre-seed above
        if cleaned in seen_cleaned:
            log.warning(
                "skipping channel rename %r → %r for %s "
                "(would collide with an existing channel)",
                ch, cleaned, source.name,
            )
            continue
        rename[ch] = cleaned
        seen_cleaned.add(cleaned)
    if rename:
        try:
            raw.rename_channels(rename, allow_duplicates=False, verbose="ERROR")
        except ValueError as exc:
            # Last-resort guard: if mne still rejects (unlikely after
            # the pre-flight collision check above), warn and continue
            # without the rename rather than crashing the row.
            log.warning(
                "channel rename failed for %s: %s — proceeding without rename",
                source.name, exc,
            )

    try:
        raw.set_montage(
            montage,
            match_case=False,
            match_alias=False,
            on_missing="ignore",
            verbose="ERROR",
        )
    except Exception as exc:
        log.warning(
            "could not apply montage %r to %s: %s",
            montage_name, source.name, exc,
        )


def _find_bids_root(staging_dir: Path) -> Path:
    """Walk up to the BIDS-root parent that mne-bids should populate.

    The orchestrator passes a per-subject staging dir like
    ``<...>/.tmp_bidsmgr/sub-001/`` or ``<...>/.tmp_bidsmgr/sub-001/ses-pre/``.
    mne-bids writes ``sub-001/[ses-pre/]<datatype>/...`` inside its
    ``root`` — so we need to feed it the parent of the ``sub-XXX``
    component, not the ``sub-XXX`` directory itself.
    """
    p = Path(staging_dir)
    if p.name.startswith("ses-") and p.parent.name.startswith("sub-"):
        return p.parent.parent
    if p.name.startswith("sub-"):
        return p.parent
    return p


def _coerce_run(value: Any) -> Optional[int]:
    """Best-effort int coercion for the ``run`` BIDS entity.

    BIDS run is always an integer; a string ``"1"`` from the TSV becomes
    ``1``, anything else returns ``None`` so mne-bids leaves the
    ``_run-`` token off the BIDS path.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "n/a", "none"}:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


__all__ = ["MneBidsBackend"]

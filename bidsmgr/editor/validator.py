"""Schema-driven BIDS validation, delegated to the standalone ``bidsval`` engine.

BIDS Manager used to carry its own two-layer validator (an in-house schema audit
plus a ``bidsschematools.validator`` pass behind ``--strict``). Validation is now
delegated to :mod:`bidsval` - the schema-driven, pydantic-typed, false-positive-
verified BIDS validator - and this module is a thin adapter that:

* runs bidsval and maps its results into the GUI's
  :class:`bidsmgr.editor.types` shapes (see :mod:`bidsmgr.editor._bidsval_adapter`);
* adds the two BIDS Manager-native supplements bidsval does not produce - the
  sidecar-editing form data (``sidecar_fields``) and TODO-placeholder findings
  (see :mod:`bidsmgr.editor.bidsmgr_checks`).

The public signatures are unchanged, so the workers, the Editor panes, the HTML
report, and ``bidsmgr-validate`` keep working without changes:

* :func:`validate` - a whole dataset.
* :func:`validate_file` - one file (the rest of the dataset is indexed for
  inheritance / existence checks).
* :func:`validate_folder` - every file under a folder.

``strict`` is mapped to bidsval's ``read_headers`` ("deep checks": read NIfTI
headers and file contents; slower). ``strict=False`` is the fast structural pass
used for live revalidation in the Editor. ``schema`` selects the BIDS schema
version (``None`` = bidsval's bundled default); ``max_rows`` bounds how many TSV
rows are scanned. Both are threaded from the GUI / CLI settings (the engine never
reads settings itself - architectural rule).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import bidsval

import bidsmgr

from . import _bidsval_adapter as adapter
from .types import FileVerdict, ValidationReport

log = logging.getLogger(__name__)


def _bidsmgr_version() -> str:
    return str(getattr(bidsmgr, "__version__", "0.0.0"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _version(schema: Optional[str]) -> Optional[str]:
    """The version to validate against: the caller's, else the session's.

    Without this a bare ``validate()`` judged a dataset by bidsval's default
    while every form beside it described whichever version the user chose, and
    nothing on screen said the two disagreed.
    """
    if schema:
        return schema
    from .. import schema as schema_mod

    return schema_mod.active_version()


def validate(
    bids_root: Path,
    *,
    strict: bool = False,
    schema: Optional[str] = None,
    max_rows: int = 1000,
    flag_todos: bool = True,
) -> ValidationReport:
    """Validate the BIDS dataset at ``bids_root``.

    ``strict=True`` enables "deep checks" (bidsval ``read_headers=True``: open
    NIfTI image headers; slower). ``flag_todos`` adds the BIDS Manager TODO-
    placeholder warnings (``False`` gives exact bidsval parity). The function
    never raises on an invalid dataset - problems land in
    ``report.files[*].issues`` and ``report.dataset_issues``; ``report.severity``
    rolls up to the worst.
    """
    bids_root = Path(bids_root).resolve()
    if not bids_root.is_dir():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    bv_report = bidsval.validate(
        bids_root,
        schema=_version(schema),
        read_headers=strict,
        max_rows=max_rows,
    )
    return adapter.to_bm_report(
        bv_report,
        bids_root,
        bidsmgr_version=_bidsmgr_version(),
        generated_at=_now_iso(),
        flag_todos=flag_todos,
    )


def validate_file(
    bids_root: Path,
    file_path: Path,
    *,
    schema: Optional[str] = None,
    max_rows: int = 1000,
    flag_todos: bool = True,
) -> FileVerdict:
    """Validate a single file; the rest of the dataset is indexed for
    inheritance / existence checks. ``file_path`` must live under ``bids_root``
    (raises ``ValueError`` otherwise). The returned ``FileVerdict.path`` is
    relative to ``bids_root`` so it merges cleanly into a dataset-wide report.

    Uses the fast structural pass (``read_headers=False``) since this runs live
    as the user clicks around the Editor tree.
    """
    bids_root = Path(bids_root).resolve()
    file_path = Path(file_path).resolve()
    rel = file_path.relative_to(bids_root)  # ValueError if outside the root

    bv_verdict = bidsval.validate_file(
        bids_root,
        rel.as_posix(),
        schema=_version(schema),
        read_headers=False,
        max_rows=max_rows,
    )
    # bidsval indexes only files it recognises as BIDS data. If the user
    # selected an existing file bidsval doesn't index (e.g. one nested inside a
    # directory-recording), it reports FILE_NOT_FOUND - which would be a false
    # error here. Fall back to a clean verdict so the BM-native supplements
    # (sidecar_fields + TODO) still render without a spurious finding.
    if file_path.exists() and any(
        i.code == "FILE_NOT_FOUND" for i in bv_verdict.issues
    ):
        from bidsval import FileVerdict as _BvFileVerdict
        bv_verdict = _BvFileVerdict(path=rel)

    return adapter.to_bm_file_verdict(bv_verdict, bids_root, flag_todos=flag_todos)


def validate_folder(
    bids_root: Path,
    folder_path: Path,
    *,
    schema: Optional[str] = None,
    max_rows: int = 1000,
    flag_todos: bool = True,
) -> list[FileVerdict]:
    """Validate every file under ``folder_path`` and return one verdict each.

    Runs one dataset-wide bidsval pass (so each file is validated with full
    inheritance context) and filters to the files under ``folder_path`` - faster
    and more correct than rebuilding the file tree once per file. Like
    :func:`validate_file`, no dataset-level findings are returned (those belong
    to the full ``validate`` pass).
    """
    bids_root = Path(bids_root).resolve()
    folder_path = Path(folder_path).resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(folder_path)

    bv_report = bidsval.validate(
        bids_root,
        schema=_version(schema),
        read_headers=False,
        max_rows=max_rows,
    )
    out: list[FileVerdict] = []
    for fv in bv_report.files:
        abs_path = (bids_root / Path(fv.path)).resolve()
        try:
            abs_path.relative_to(folder_path)
        except ValueError:
            continue
        out.append(adapter.to_bm_file_verdict(fv, bids_root, flag_todos=flag_todos))
    return out


__all__ = ["validate", "validate_file", "validate_folder"]

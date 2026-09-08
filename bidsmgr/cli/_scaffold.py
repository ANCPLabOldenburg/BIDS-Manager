"""Shared BIDS dataset scaffolding (cli layer, no Qt).

Single home for creating the minimal valid BIDS skeleton and the files that keep
a BIDS-Manager dataset clean under the official bids-validator. Used by the
convert verb today; reused by the project "create dataset" flow later, so the
BIDS-version string, the ``GeneratedBy`` de-dup, and ``.bidsignore`` all live in
ONE place instead of being hardcoded per call site.

``.bidsmgr/`` layout convention (the single source of truth for the hidden
working dir, resolving the historical overlap of the name):

    <bids_root>/
    |-- dataset_description.json      # ensure_dataset_description()
    |-- .bidsignore                   # ensure_bidsignore() -> exempts the below
    |-- .bidsmgr/                     # BM working dir (NOT valid BIDS; bids-ignored)
    |   |-- errors/                   #   convert per-subject failure logs
    |   |-- backup/                   #   --overwrite subject backups
    |   `-- project/                  #   (reserved) event-sourced project bundle:
    |       |-- events.jsonl          #     meta.json + events.jsonl + scans/<version>/
    |       `-- meta.json
    |-- sub-XXX/.bidsmgr/provenance.json   # per-subject convert provenance
    `-- .tmp_bidsmgr/                 # per-run staging (wiped on success)

The event-sourced project bundle (``bidsmgr.project``) will live under
``<bids_root>/.bidsmgr/project/`` rather than as a sibling ``<name>.bidsmgr/``
directory, so it cannot collide with convert's ``errors/`` + ``backup/`` + the
per-subject ``provenance.json`` that already live in ``.bidsmgr/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..schema import bids_version as schema_bids_version

_SLUG_BAD_RE = re.compile(r"[^A-Za-z0-9_-]+")
_SLUG_DASH_RE = re.compile(r"-{2,}")


def slugify_name(name: str) -> str:
    """Turn a human dataset name into a safe folder/dataset slug.

    Mirrors the scan verb's slug rules (non-alphanumerics -> ``-``, repeated
    dashes collapsed, trimmed) so the folder a project is created in matches the
    ``dataset`` slug scan/convert use. Falls back to ``"dataset"`` when empty.
    """
    slug = _SLUG_BAD_RE.sub("-", (name or "").strip())
    slug = _SLUG_DASH_RE.sub("-", slug).strip("-_")
    return slug or "dataset"

# gitignore-style globs written to ``.bidsignore`` so the official
# bids-validator skips BM's hidden working dirs. A pattern without a leading
# slash matches at any depth, so ``.bidsmgr/`` also covers the per-subject
# ``sub-XXX/.bidsmgr/`` provenance dirs.
BIDSIGNORE_LINES: tuple[str, ...] = (".bidsmgr/", ".tmp_bidsmgr/")


def project_bundle_dir(bids_root: Path) -> Path:
    """Return the event-sourced project bundle dir for a dataset.

    ``<bids_root>/.bidsmgr/project`` (see the module docstring): nested under the
    hidden working dir so it cannot collide with convert's ``errors/`` /
    ``backup/`` / per-subject ``provenance.json``.
    """
    return Path(bids_root) / ".bidsmgr" / "project"


def ensure_dataset_description(
    bids_root: Path,
    *,
    name: Optional[str] = None,
    generated_by: Optional[dict] = None,
    fields: Optional[dict] = None,
) -> None:
    """Create or update ``<bids_root>/dataset_description.json``.

    Sets the minimal valid fields (``Name`` + ``BIDSVersion`` sourced from the
    bundled schema + ``DatasetType``) without overwriting existing values, and
    appends one ``GeneratedBy`` entry idempotently (matched on
    Name/Version/Description, so re-running never piles up duplicates). ``name``
    seeds ``Name`` on first write (defaults to the folder name); an existing
    ``Name`` is never overwritten. ``generated_by`` is the backend stamp to
    record; pass ``None`` to scaffold a dataset without adding a generator entry.

    ``fields`` are what the user STATED about this dataset, and they win over
    what is on disk, ``Name`` included. Everything else set here is a default
    filled in for want of an answer; these are the answer. Until this existed
    the agnostic half of the metadata template reached the file only if a
    separate metadata run happened afterwards, so a user who filled in the form
    and converted saw none of it.
    """
    p = bids_root / "dataset_description.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
    else:
        data = {}

    data.setdefault("Name", name or bids_root.name)
    data.setdefault("BIDSVersion", schema_bids_version())
    data.setdefault("DatasetType", "raw")

    for key, value in (fields or {}).items():
        if value not in (None, "", [], {}):
            data[key] = value

    if generated_by is not None:
        existing = data.get("GeneratedBy")
        if not isinstance(existing, list):
            existing = []
        already = any(
            isinstance(g, dict)
            and g.get("Name") == generated_by.get("Name")
            and g.get("Version") == generated_by.get("Version")
            and g.get("Description") == generated_by.get("Description")
            for g in existing
        )
        if not already:
            existing.append(generated_by)
        data["GeneratedBy"] = existing

    p.write_text(json.dumps(data, indent=2) + "\n")


def ensure_bidsignore(bids_root: Path) -> None:
    """Ensure ``<bids_root>/.bidsignore`` exempts BM's hidden working dirs.

    BM keeps non-BIDS state inside the dataset (``.bidsmgr/`` for convert error
    logs, per-subject provenance, overwrite backups, and later the project
    bundle; ``.tmp_bidsmgr/`` for per-run staging). The official bids-validator
    would flag these unless they are listed in ``.bidsignore``. Existing user
    lines are preserved; only the missing BM lines are appended (idempotent).
    """
    p = bids_root / ".bidsignore"
    existing_lines: list[str] = []
    if p.exists():
        try:
            existing_lines = p.read_text().splitlines()
        except OSError:
            existing_lines = []
    have = {ln.strip() for ln in existing_lines}
    missing = [ln for ln in BIDSIGNORE_LINES if ln not in have]
    if not missing:
        return
    out = list(existing_lines) + missing
    p.write_text("\n".join(out) + "\n")


def ensure_readme(bids_root: Path, name: str) -> None:
    """Write a minimal README scaffold. Never overwrites a hand-edited file.

    Matches the metadata engine's README byte-for-byte so a workspace scaffolded
    by ``bidsmgr create`` and one later touched by ``bidsmgr-metadata`` look
    identical (and metadata's never-overwrite leaves this one in place).
    """
    out = bids_root / "README"
    if out.exists():
        return
    out.write_text(
        f"# {name}\n\n"
        "This BIDS dataset was generated by bidsmgr.\n\n"
        "Edit this README to describe acquisition details, study design,\n"
        "and any deviations from the standard BIDS layout.\n",
        encoding="utf-8",
    )


__all__ = [
    "BIDSIGNORE_LINES",
    "slugify_name",
    "project_bundle_dir",
    "ensure_dataset_description",
    "ensure_bidsignore",
    "ensure_readme",
]

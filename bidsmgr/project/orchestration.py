"""Qt-free project orchestration shared by the GUI and the CLI.

The project-first model (event-sourced ``.bidsmgr/project/`` bundle, versioned
scans under ``scans/<NNNN>__<slug>/``, resume by replaying edits, output locked
to the dataset root) used to live only in the GUI's converter panel. This
module extracts the pure-data pieces so both the GUI worker layer and the new
``--project`` CLI flags drive the exact same logic:

* :func:`import_scan_as_version` promotes a freshly-staged scan into a new
  version bundle (the body of the GUI's ``_promote_scan_to_version``).
* :func:`apply_project_state` replays a project's curation edits onto an
  inventory DataFrame (the body of the model's ``_apply_project_overlay``).
* :func:`row_id` is the stable per-row identity used to key those edits.
* :func:`latest_version` / :func:`version_dataframe` resolve and load the
  active scan version (with edits applied) for convert / metadata.

Depends only on ``inventory`` + ``recording_meta`` primitives (project ->
inventory is the safe direction; inventory never imports project).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from ..inventory.rebuild import rebuild_from_columns, rebuild_from_entities
from . import workspace
from .project import Project
from .types import ProjectState, ScanImported, StageCompleted
from .workspace import ScanVersion


# ---------------------------------------------------------------------------
# Row identity + edit replay (mirror of InventoryTableModel)
# ---------------------------------------------------------------------------


def row_id(df: pd.DataFrame, i: int) -> str:
    """Stable per-row id used to key project edits.

    Prefers ``series_uid`` (MRI) then ``source_file`` (EEG/MEG), else a
    positional ``row-<i>`` fallback. Identical to
    ``InventoryTableModel.row_id`` so GUI-recorded edits replay in the CLI.
    """
    for col in ("series_uid", "source_file"):
        if col in df.columns:
            v = df.at[i, col]
            if isinstance(v, str) and v.strip():
                return v.strip()
    return f"row-{i}"


def _row_entities(df: pd.DataFrame, r: int) -> dict:
    if "entities" not in df.columns:
        return {}
    raw = df.at[r, "entities"]
    if pd.isna(raw):
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _rebuild_one_row(df: pd.DataFrame, r: int, *, direction: str = "columns") -> None:
    sub = df.iloc[[r]].copy()
    if direction == "entities":
        new_sub, _ = rebuild_from_entities(sub)
    else:
        new_sub, _ = rebuild_from_columns(sub)
    for col in new_sub.columns:
        df.at[r, col] = new_sub.iloc[0][col]


def apply_project_state(df: pd.DataFrame, state: ProjectState) -> None:
    """Replay a project's curation edits onto ``df`` in place.

    Mirrors ``InventoryTableModel._apply_project_overlay``: entity edits first
    (subject renames keep ``participant_id`` in sync, then rebuild from entities),
    then cell overrides (rebuild from columns), then include toggles. Unknown
    row ids are ignored (the inventory may have been re-scanned).
    """
    if not (
        state.cell_overrides or state.include_overrides or state.entity_overrides
    ):
        return

    index_for_rid = {row_id(df, i): i for i in range(len(df))}

    if "entities" in df.columns:
        for rid, ents in state.entity_overrides.items():
            r = index_for_rid.get(rid)
            if r is None:
                continue
            current = _row_entities(df, r)
            for ent, val in ents.items():
                if val:
                    current[ent] = val
                else:
                    current.pop(ent, None)
                if ent == "subject" and "participant_id" in df.columns:
                    df.at[r, "participant_id"] = f"sub-{val}" if val else ""
            df.at[r, "entities"] = json.dumps(current, sort_keys=True)
            _rebuild_one_row(df, r, direction="entities")

    for rid, cells in state.cell_overrides.items():
        r = index_for_rid.get(rid)
        if r is None:
            continue
        for col, val in cells.items():
            if col in df.columns:
                df.at[r, col] = val
        _rebuild_one_row(df, r)

    if "include" in df.columns:
        for rid, inc in state.include_overrides.items():
            r = index_for_rid.get(rid)
            if r is None:
                continue
            df.at[r, "include"] = 1 if inc else 0


# ---------------------------------------------------------------------------
# Scan versions
# ---------------------------------------------------------------------------


def _move_if_exists(src: Path, dst: Path) -> None:
    src, dst = Path(src), Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def import_scan_as_version(
    bids_root: Path,
    staged_inventory: Path,
    *,
    raw_root: Optional[Path] = None,
    source_label: Optional[str] = None,
    row_ids: tuple[str, ...] = (),
    n_rows: int = 0,
) -> ScanVersion:
    """Promote a freshly-staged scan into a new project version.

    Creates ``.bidsmgr/project/scans/<NNNN>__<slug>/`` (its own Project bundle),
    moves the inventory snapshot (+ recording-metadata scaffold + files_by_uid
    sidecar) into it, writes the version descriptor, and appends
    ``ScanImported`` + ``StageCompleted`` events. Returns the new
    :class:`ScanVersion`. This is the engine behind both the GUI scan-finish
    handler and ``bidsmgr-scan --project``.
    """
    from ..recording_meta import scaffold_sidecar_path

    bids_root = Path(bids_root)
    staged = Path(staged_inventory)
    label = source_label or (Path(raw_root).name if raw_root else "scan")

    version_dir = workspace.allocate_version_dir(bids_root, label)
    proj = Project.create(version_dir, name=bids_root.name)

    dest = workspace.version_inventory(version_dir)
    _move_if_exists(staged, dest)
    _move_if_exists(scaffold_sidecar_path(staged), scaffold_sidecar_path(dest))
    _move_if_exists(
        Path(str(staged) + ".files_by_uid.json.gz"),
        Path(str(dest) + ".files_by_uid.json.gz"),
    )
    workspace.write_version_meta(
        version_dir,
        source_label=label,
        raw_root=str(raw_root) if raw_root else None,
        status="curating",
    )
    proj.append(ScanImported(
        inventory_tsv=str(dest),
        row_ids=tuple(row_ids),
        raw_root=str(raw_root) if raw_root else None,
    ))
    proj.append(StageCompleted(
        stage="scan",
        success=True,
        summary={
            "raw_root": str(raw_root) if raw_root else "",
            "inventory_tsv": str(dest),
            "rows": int(n_rows),
        },
    ))

    for v in workspace.list_versions(bids_root):
        if Path(v.dir) == version_dir:
            return v
    # Fallback (should not happen now the inventory exists).
    return ScanVersion(
        dir=version_dir, version_id=version_dir.name, index=0,
        source_label=label, raw_root=str(raw_root) if raw_root else None,
        status="curating", inventory=dest,
    )


def latest_version(bids_root: Path) -> Optional[ScanVersion]:
    """Return the most recent scan version, or ``None`` if there are none."""
    versions = workspace.list_versions(Path(bids_root))
    return versions[-1] if versions else None


def find_version(bids_root: Path, selector: str) -> Optional[ScanVersion]:
    """Resolve a scan version by a lenient ``selector``.

    Matches (in order): exact ``version_id`` (e.g. ``"0001__neuro2"``), the
    numeric index (``"1"`` or ``"0001"``), or an ``id__``-prefix. Returns
    ``None`` when nothing matches.
    """
    sel = str(selector).strip()
    versions = workspace.list_versions(Path(bids_root))
    if not sel:
        return None
    for v in versions:
        if v.version_id == sel:
            return v
    if sel.isdigit():
        idx = int(sel)
        for v in versions:
            if v.index == idx:
                return v
    for v in versions:
        if v.version_id.startswith(f"{sel}__") or v.version_id.startswith(sel):
            return v
    return None


def version_dataframe(
    version: ScanVersion, *, apply_edits: bool = True,
) -> pd.DataFrame:
    """Load a version's inventory DataFrame, replaying its curation edits.

    With ``apply_edits`` (the default) the version's project events are replayed
    onto the table so the CLI converts exactly what the user curated in the GUI.
    """
    df = pd.read_csv(version.inventory, sep="\t", dtype=str).fillna("")
    if apply_edits:
        try:
            proj = Project.open(version.dir)
            apply_project_state(df, proj.state())
        except Exception:
            pass  # a missing/corrupt bundle just means no edits to replay
    return df


__all__ = [
    "row_id",
    "apply_project_state",
    "import_scan_as_version",
    "latest_version",
    "find_version",
    "version_dataframe",
]

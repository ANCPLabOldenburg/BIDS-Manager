"""Top-level :class:`Project` API — what the GUI and (later) CLI verbs hold.

A project is a bundle directory ending in ``.bidsmgr``. It looks like a
single file in a file manager (mac-style "package") but is a directory
internally so individual artifacts can be rewritten atomically without
serialising the whole project on every write.

Bundle layout::

    <name>.bidsmgr/
    ├── meta.json          # name, created_at, bidsmgr/schema versions
    ├── events.jsonl       # append-only event log (the source of truth)
    └── provenance.json    # (row_id, field) → source side-table

``meta.json`` duplicates a subset of the ``ProjectCreated`` event's
fields. It exists so a project picker can list bundles without parsing
every log; the event log remains authoritative on conflict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import bidsmgr
from .. import schema as schema_module
from .log import EventLog
from .provenance import ProvenanceMap
from .replay import replay
from .types import Event, ProjectCreated, ProjectState, utc_now

log = logging.getLogger(__name__)


EVENTS_FILENAME = "events.jsonl"
PROVENANCE_FILENAME = "provenance.json"
META_FILENAME = "meta.json"


class ProjectError(RuntimeError):
    """Raised on bundle-shape problems (missing files, unknown layout)."""


class Project:
    """Open or create a ``*.bidsmgr`` bundle.

    Typical lifecycle::

        proj = Project.create(Path("/path/to/myStudy.bidsmgr"), name="myStudy")
        proj.append(ScanImported(inventory_tsv="...", row_ids=("r1", "r2")))
        proj.append(UserSetEntity(row_id="r1", entity="task", value="rest"))
        state = proj.state()              # replays the log
        prov = proj.provenance()          # reads provenance.json

        # Later session:
        proj = Project.open(Path("/path/to/myStudy.bidsmgr"))
        state = proj.state()
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.log = EventLog(self.root / EVENTS_FILENAME)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        bids_schema_version: Optional[str] = None,
    ) -> "Project":
        """Create a fresh bundle at ``root``.

        Raises :class:`ProjectError` if ``root`` already exists. The
        creator records exactly one ``ProjectCreated`` event so every
        log starts identically.
        """
        root = Path(root)
        if root.exists():
            raise ProjectError(f"{root} already exists; refusing to clobber")
        root.mkdir(parents=True)

        schema_version = bids_schema_version or _detect_schema_version()
        proj = cls(root)
        proj.append(
            ProjectCreated(
                bidsmgr_version=bidsmgr.__version__,
                bids_schema_version=schema_version,
                name=name,
                description=description,
            )
        )
        proj._write_meta(
            name=name,
            description=description,
            bids_schema_version=schema_version,
        )
        return proj

    @classmethod
    def open(cls, root: Path) -> "Project":
        """Open an existing bundle. Raises :class:`ProjectError` if the
        directory is missing or has no event log.
        """
        root = Path(root)
        if not root.is_dir():
            raise ProjectError(f"{root} is not a project bundle directory")
        proj = cls(root)
        if not proj.log.path.exists():
            raise ProjectError(
                f"{root} has no {EVENTS_FILENAME}; not a bidsmgr project"
            )
        return proj

    # ------------------------------------------------------------------
    # Reading / writing
    # ------------------------------------------------------------------

    def append(self, event: Event) -> None:
        """Append one event to the log."""
        self.log.append(event)

    def state(self) -> ProjectState:
        """Replay the log and return the derived state."""
        return replay(iter(self.log))

    def provenance(self) -> ProvenanceMap:
        """Load the on-disk provenance map (or an empty one if absent)."""
        return ProvenanceMap.load(self.root / PROVENANCE_FILENAME)

    def save_provenance(self, provenance: ProvenanceMap) -> None:
        """Persist ``provenance`` to disk, atomically replacing the file."""
        provenance.save(self.root / PROVENANCE_FILENAME)

    def undo_last(self) -> bool:
        """Drop the last recorded event. Returns ``True`` if something was dropped.

        The GUI's "undo" button rides this primitive. Note: the cached
        provenance is NOT rolled back automatically; the GUI must
        re-derive the affected provenance entries if it wants the
        side-table to stay in sync. For v1 the GUI calls
        :meth:`save_provenance` after re-deriving from the new state.
        """
        return self.log.truncate_last()

    # ------------------------------------------------------------------
    # Bundle housekeeping
    # ------------------------------------------------------------------

    def _write_meta(
        self,
        *,
        name: Optional[str],
        description: Optional[str],
        bids_schema_version: str,
    ) -> None:
        meta = {
            "name": name,
            "description": description,
            "created_at": utc_now(),
            "bidsmgr_version": bidsmgr.__version__,
            "bids_schema_version": bids_schema_version,
        }
        (self.root / META_FILENAME).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def meta(self) -> dict:
        """Return the contents of ``meta.json`` (empty dict if missing).

        Convenient for a project picker that wants to list bundles
        without replaying their logs.
        """
        path = self.root / META_FILENAME
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


def _detect_schema_version() -> str:
    """Best-effort BIDS schema version readout from the schema engine.

    Falls back to an empty string if the wrapper does not expose a
    version; project creation does not block on this — the log records
    whatever we knew at write time.
    """
    for attr in ("bids_version", "schema_version"):
        v = getattr(schema_module, attr, None)
        if callable(v):
            try:
                v = v()
            except Exception:
                v = None
        if isinstance(v, str) and v:
            return v
    return ""


__all__ = [
    "EVENTS_FILENAME",
    "META_FILENAME",
    "PROVENANCE_FILENAME",
    "Project",
    "ProjectError",
]

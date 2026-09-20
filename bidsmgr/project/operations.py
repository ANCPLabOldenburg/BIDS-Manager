"""Every file the Editor writes, written reversibly.

The Editor can now change hundreds of files from one click: fix a finding in
every file it applies to, write a field across a selection, generate the
companion files a recording is missing. A mistake at that scale is not
something a user can undo by hand, so nothing here writes without first
recording what was there.

Three guarantees, in the order they matter:

**Nothing is half-written.** Every write goes to a temporary file in the same
directory and is moved into place with :func:`os.replace`, which is atomic on
POSIX and on Windows. A crash leaves either the old file or the new one.

**Nothing is lost.** The first time an operation touches a file, the original
bytes are copied under ``.bidsmgr/editor/originals/<op_id>/``. A file that did
not exist is recorded as such, so undo deletes it rather than restoring a
mystery.

**Nothing is half-applied.** A failure part way through rolls the whole
operation back, last step first.

State lives inside the dataset, under ``.bidsmgr/``, which is the choice BIDS
Manager already makes for curation history: the history travels with the data.
The cost is a dataset root that must be writable, so :func:`begin_operation`
checks and says so rather than failing on the first write.

Qt-free. The GUI, the CLI and the tests all drive the same code.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# Where the editor keeps its history, relative to the dataset root.
EDITOR_STATE_DIR = Path(".bidsmgr") / "editor"
ORIGINALS_DIR = EDITOR_STATE_DIR / "originals"
OPERATIONS_LOG = EDITOR_STATE_DIR / "operations.log"


class OperationError(RuntimeError):
    """An operation could not be started or could not be completed."""


@dataclass
class ChildStep:
    """One file-level change inside an operation."""

    kind: str            # "write" | "create" | "delete" | "rename"
    path: str            # relative to the dataset root
    existed: bool        # was there a file here before?
    # Only for "rename": where it came from, so undo can move it back.
    origin: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = {"kind": self.kind, "path": self.path, "existed": self.existed}
        if self.origin:
            out["origin"] = self.origin
        return out


@dataclass
class Operation:
    """A group of writes that succeed or fail together.

    Use through :func:`begin_operation`; the context manager commits on a
    clean exit and rolls back on an exception.
    """

    root: Path
    op_id: str
    label: str
    children: list[ChildStep] = dc_field(default_factory=list)
    _committed: bool = False

    # -- paths ---------------------------------------------------------

    @property
    def originals_dir(self) -> Path:
        return self.root / ORIGINALS_DIR / self.op_id

    def _rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            # Outside the dataset. Recorded by absolute path so undo can still
            # find it, but it will not be inside originals/.
            return str(Path(path).resolve())

    # -- writing -------------------------------------------------------

    def _backup(self, path: Path) -> ChildStep:
        """Copy the current bytes aside, once per path per operation."""
        rel = self._rel(path)
        for step in self.children:
            if step.path == rel:
                return step
        existed = path.exists()
        if existed:
            dest = self.originals_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        step = ChildStep(
            kind="write" if existed else "create", path=rel, existed=existed,
        )
        self.children.append(step)
        return step

    def write_text(self, path: Path, text: str) -> None:
        """Replace ``path`` with ``text``, atomically, after backing it up."""
        path = Path(path)
        self._backup(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{self.op_id}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise

    def write_bytes(self, path: Path, data: bytes) -> None:
        """Replace ``path`` with ``data``, atomically, after backing it up.

        The binary twin of :meth:`write_text`. Not every file the Editor writes
        is text: defacing replaces a ``.nii.gz`` with new image bytes, and
        routing that through ``write_text`` would corrupt it.
        """
        path = Path(path)
        self._backup(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{self.op_id}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise

    def replace_from(self, path: Path, source: Path) -> None:
        """Replace ``path`` with the contents of ``source``, reversibly.

        For a file another program just produced. ``write_bytes`` would hold
        the whole thing in memory first, and a 3-D anatomical image is tens of
        megabytes; this streams it instead.

        ``source`` is expected to be a temporary the caller owns, outside the
        dataset. It is left alone, so a failed replace does not destroy the one
        copy of the new bytes.
        """
        path = Path(path)
        source = Path(source)
        if not source.is_file():
            raise OperationError(f"{source} is not a file")
        self._backup(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The temp lands beside the target so the replace is a rename within
        # one filesystem. A temp directory can be on another device, where
        # os.replace fails with EXDEV.
        tmp = path.with_name(f".{path.name}.{self.op_id}.tmp")
        try:
            shutil.copyfile(source, tmp)
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise

    def write_json(self, path: Path, data: Any, *, indent: int = 4) -> None:
        """Write ``data`` as JSON. Trailing newline, so the file is diffable."""
        self.write_text(
            path,
            json.dumps(data, indent=indent, ensure_ascii=False) + "\n",
        )

    def delete(self, path: Path) -> None:
        """Remove ``path``, keeping a copy so undo can put it back."""
        path = Path(path)
        if not path.exists():
            return
        rel = self._rel(path)
        dest = self.originals_dir / rel
        # Only if nothing in this operation has saved it yet. An operation that
        # edits a file and then deletes it (fusing two subjects rewrites the
        # source's scans table before merging it away) must restore what was
        # there when the operation STARTED, not the intermediate state it
        # wrote a moment ago.
        if not any(step.path == rel for step in self.children):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        self.children.append(ChildStep(kind="delete", path=rel, existed=True))
        path.unlink()

    def rename(self, src: Path, dst: Path) -> None:
        """Move ``src`` to ``dst``, recording where it came from.

        A rename is not a write: copying the bytes aside would double the disk
        cost of renaming a subject, and there is nothing to restore from
        because the content never changes. Undo moves it back instead, which
        is why the origin is recorded rather than the contents.
        """
        src, dst = Path(src), Path(dst)
        if not src.exists():
            return
        if dst.exists():
            raise OperationError(f"{dst} already exists")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        self.children.append(ChildStep(
            kind="rename", path=self._rel(dst), existed=False,
            origin=self._rel(src),
        ))

    # -- finishing -----------------------------------------------------

    def commit(self) -> None:
        """Append the operation to the log. Nothing to move; writes are done."""
        if self._committed:
            return
        self._committed = True
        if not self.children:
            # An operation that changed nothing does not deserve a log line.
            shutil.rmtree(self.originals_dir, ignore_errors=True)
            return
        record = {
            "op_id": self.op_id,
            "label": self.label,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "children": [c.as_dict() for c in self.children],
        }
        log_path = self.root / OPERATIONS_LOG
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def rollback(self) -> None:
        """Undo this operation's writes, last first."""
        _restore(self.root, self.op_id, list(reversed(self.children)))
        shutil.rmtree(self.originals_dir, ignore_errors=True)
        self.children.clear()
        self._committed = True


class _OperationContext:
    """Context manager wrapper so a caller cannot forget to finish."""

    def __init__(self, op: Operation) -> None:
        self.op = op

    def __enter__(self) -> Operation:
        return self.op

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.op.commit()
            return False
        log.warning(
            "operation %s failed (%s); rolling back %d step(s)",
            self.op.op_id, exc, len(self.op.children),
        )
        try:
            self.op.rollback()
        except OSError:
            log.exception("rollback of %s failed", self.op.op_id)
        return False


def begin_operation(root: Path, label: str) -> _OperationContext:
    """Start a reversible group of writes on the dataset at ``root``.

    ``label`` is what the user will see in a history list, so write it as a
    sentence: "Fill Manufacturer in 12 files".
    """
    root = Path(root)
    if not root.is_dir():
        raise OperationError(f"{root} is not a directory")
    if not os.access(root, os.W_OK):
        raise OperationError(
            f"{root} is not writable, so changes could not be made "
            "reversible. Editing is refused rather than done unsafely."
        )
    op = Operation(root=root, op_id=uuid.uuid4().hex[:12], label=label)
    op.originals_dir.mkdir(parents=True, exist_ok=True)
    return _OperationContext(op)


# --------------------------------------------------------------------------
# Reading the history back


def read_log(root: Path) -> list[dict[str, Any]]:
    """Every recorded operation, oldest first. Unreadable lines are skipped."""
    path = Path(root) / OPERATIONS_LOG
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            log.debug("skipping unreadable operations.log line")
    return out


def _restore(root: Path, op_id: str, children: list[ChildStep]) -> int:
    """Put ``children`` back the way they were. Returns files restored."""
    originals = Path(root) / ORIGINALS_DIR / op_id
    n = 0
    for step in children:
        target = Path(root) / step.path
        if step.kind == "rename":
            origin = Path(root) / step.origin
            if target.exists() and not origin.exists():
                origin.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, origin)
                n += 1
            continue
        if not step.existed:
            # We created it; undo means it should not be there.
            target.unlink(missing_ok=True)
            n += 1
            continue
        src = originals / step.path
        if not src.exists():
            log.warning("no original kept for %s in %s", step.path, op_id)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        n += 1
    return n


def undo_last(root: Path) -> Optional[dict[str, Any]]:
    """Reverse the most recent operation. Returns its record, or ``None``.

    The record is removed from the log, so undo can be pressed repeatedly to
    walk back through a session.
    """
    root = Path(root)
    records = read_log(root)
    if not records:
        return None
    last = records[-1]
    children = [
        ChildStep(
            kind=c["kind"], path=c["path"], existed=c["existed"],
            origin=c.get("origin", ""),
        )
        for c in last.get("children", [])
    ]
    _restore(root, last["op_id"], list(reversed(children)))
    remaining = records[:-1]
    log_path = root / OPERATIONS_LOG
    log_path.write_text(
        "".join(json.dumps(r) + "\n" for r in remaining), encoding="utf-8",
    )
    shutil.rmtree(root / ORIGINALS_DIR / last["op_id"], ignore_errors=True)
    return last


__all__ = [
    "ChildStep",
    "EDITOR_STATE_DIR",
    "Operation",
    "OperationError",
    "begin_operation",
    "read_log",
    "undo_last",
]

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

**Folders too.** Every folder an operation creates is recorded, and so is
every folder it removes, so an undo takes away the folders it no longer needs
(a renamed subject's new folder) and puts back the ones it emptied, and a
redo does both again.

**Every operation is one step of undo, and an undo can be redone.** Undo
puts back every file of the newest operation together, however many there
were, and does it AS an operation of its own, so the state it replaced is
kept too and Redo can bring it back. A new change ends what can be redone.
Each operation records the size and modification time of the files it left,
so an undo can say which of them were changed since (by hand, by another
program) before it replaces them.

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
# Operations undone and not yet redone, newest last. Cleared by a new change.
REDO_LOG = EDITOR_STATE_DIR / "redo.log"

# Files an operating system drops into any folder a person has looked at
# (``Icon\r`` is the macOS custom folder icon; ``._name`` files are the
# resource forks macOS writes on USB and network drives). They carry no
# information and nobody put them there on purpose, so a folder holding only
# these counts as empty, and they go with it: the only moment the tool ever
# deletes one. The one definition; the scans tables and renames use it too.
OS_JUNK = frozenset({".DS_Store", "Thumbs.db", "desktop.ini", "Icon\r", ".directory"})


def is_os_junk(name: str) -> bool:
    return name in OS_JUNK or name.startswith("._")


def is_hollow(folder: Path) -> bool:
    """A folder holding nothing but the operating system's droppings."""
    try:
        return all(not p.is_dir() and is_os_junk(p.name) for p in Path(folder).iterdir())
    except OSError:
        return False


class OperationError(RuntimeError):
    """An operation could not be started or could not be completed."""


@dataclass
class ChildStep:
    """One file-level change inside an operation."""

    kind: str            # "write" | "create" | "delete" | "rename" | "mkdir" | "rmdir"
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
        return relative(self.root, path)

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

    def _ensure_dir(self, directory: Path) -> None:
        """Create ``directory`` and any missing parent, recording each one, so
        undo removes the folders it created rather than leaving them empty."""
        directory = Path(directory)
        missing: list[Path] = []
        current = directory
        while not current.exists() and current.parent != current:
            missing.append(current)
            current = current.parent
        for folder in reversed(missing):
            folder.mkdir(exist_ok=True)
            self.children.append(ChildStep(kind="mkdir", path=self._rel(folder), existed=False))

    def rmdir(self, path: Path) -> None:
        """Remove an EMPTY folder, recording it so undo puts it back. Raises
        ``OSError`` when it is not empty, which is the caller's test."""
        path = Path(path)
        os.rmdir(path)
        self.children.append(ChildStep(kind="rmdir", path=self._rel(path), existed=True))

    def remove_if_empty(self, folder: Path) -> bool:
        """Remove ``folder`` when it holds nothing but the operating system's
        droppings, those included, all recorded. True when it went."""
        folder = Path(folder)
        if not folder.is_dir() or not is_hollow(folder):
            return False
        try:
            for junk in list(folder.iterdir()):
                self.delete(junk)
            self.rmdir(folder)
        except OSError:
            return False
        return True

    def write_text(self, path: Path, text: str) -> None:
        """Replace ``path`` with ``text``, atomically, after backing it up."""
        path = Path(path)
        self._backup(path)
        self._ensure_dir(path.parent)
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
        self._ensure_dir(path.parent)
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
        self._ensure_dir(path.parent)
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
        self._ensure_dir(dst.parent)
        os.replace(src, dst)
        self.children.append(ChildStep(
            kind="rename", path=self._rel(dst), existed=False,
            origin=self._rel(src),
        ))

    # -- finishing -----------------------------------------------------

    def record(self) -> dict[str, Any]:
        """What the log keeps of this operation, with the state it left.

        ``dirs`` says the folders it created and removed are in ``children``;
        a record from before that is undone with a careful clean-up instead
        (:func:`_reverse`)."""
        paths = []
        for c in self.children:
            if c.kind in _FOLDER_KINDS:
                continue
            for rel in (c.path, c.origin):
                if rel and rel not in paths:
                    paths.append(rel)
        return {
            "op_id": self.op_id,
            "label": self.label,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dirs": True,
            "children": [c.as_dict() for c in self.children],
            "after": {rel: _stat(self.root / rel) for rel in paths},
        }

    def commit(self) -> None:
        """Append the operation to the log. Nothing to move; writes are done.

        A new change ends what can be redone: the timeline has diverged."""
        if self._committed:
            return
        self._committed = True
        if not self.children:
            # An operation that changed nothing does not deserve a log line.
            shutil.rmtree(self.originals_dir, ignore_errors=True)
            return
        _append(self.root, OPERATIONS_LOG, self.record())
        _clear_redo(self.root)

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
# A step that writes for itself, made one undoable operation

#: What :func:`run_as_operation` watches: the text files a metadata step
#: writes. Images and recordings are never among them, so the snapshot is
#: small however large the dataset.
_TEXT_SUFFIXES = (".json", ".tsv", ".txt", ".md", ".cff", ".bidsignore")
_TEXT_NAMES = frozenset({"README", "CHANGES", "LICENSE"})
_TEXT_CAP = 8 * 1024 * 1024


def _text_files(root: Path) -> tuple[dict[Path, bytes], set[Path]]:
    """``({path: bytes}, folders)`` for the dataset's small text files, the
    tool's own folders and every hidden one left out."""
    files: dict[Path, bytes] = {}
    folders: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        base = Path(dirpath)
        folders.add(base)
        for name in filenames:
            if not (name in _TEXT_NAMES or name.startswith("README")
                    or name.endswith(_TEXT_SUFFIXES)):
                continue
            path = base / name
            try:
                if path.stat().st_size <= _TEXT_CAP:
                    files[path] = path.read_bytes()
            except OSError:
                continue
    return files, folders


def run_as_operation(root: Path, label: str, fn):
    """Run ``fn()``, which writes text files in ``root`` by itself (the
    metadata step), and make what it changed ONE undoable operation.

    The step cannot be handed an operation: it writes in many places on its
    own. So the small text files are read before, the step runs, the dataset
    is put back as it was, and each change is replayed through an operation
    (``replace_from`` for a changed or new file, ``delete`` for one it
    removed, the folders it made recorded too). If the step fails, what it
    wrote is taken back and the error raised. Returns what ``fn`` returned.
    """
    import tempfile

    root = Path(root)
    before, folders_before = _text_files(root)

    def restore(after_files, after_folders) -> None:
        for path in after_files:
            if path not in before:
                path.unlink(missing_ok=True)
        for path, data in before.items():
            if after_files.get(path) != data:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        for folder in sorted(after_folders - folders_before,
                             key=lambda p: len(p.parts), reverse=True):
            try:
                folder.rmdir()
            except OSError:
                pass

    try:
        result = fn()
    except BaseException:
        restore(*_text_files(root))
        raise
    after, folders_after = _text_files(root)
    written = [p for p, data in after.items() if before.get(p) != data]
    removed = [p for p in before if p not in after and not p.exists()]
    if not written and not removed:
        return result
    with tempfile.TemporaryDirectory(prefix="bidsmgr-op-") as tmp:
        staged = {}
        for i, path in enumerate(written):
            staged[path] = Path(tmp) / str(i)
            staged[path].write_bytes(after[path])
        restore(after, folders_after)
        with begin_operation(root, label) as op:
            for path in written:
                op.replace_from(path, staged[path])
            for path in removed:
                op.delete(path)
    return result


def forget_history(root: Path) -> int:
    """Delete the Editor's history of ``root``: every copy it keeps to undo
    and redo, and the logs. Returns the bytes freed.

    For after an anonymisation: those copies ARE the originals, names and
    dates and all, so a dataset shared with ``.bidsmgr/`` in it, or a backup
    of it, still holds them. Irreversible: nothing done before can be undone
    afterwards, which is why it is never done without being asked.
    """
    root = Path(root)
    freed = 0
    originals = root / ORIGINALS_DIR
    if originals.is_dir():
        for path in originals.rglob("*"):
            if path.is_file():
                try:
                    freed += path.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(originals, ignore_errors=True)
    for log in (OPERATIONS_LOG, REDO_LOG):
        target = root / log
        if target.is_file():
            freed += target.stat().st_size
            target.unlink()
    return freed


# --------------------------------------------------------------------------
# Reading the history back


def relative(root: Path, path: Path) -> str:
    """How the log names ``path``: relative to the dataset root.

    Outside the dataset it is recorded by absolute path, so undo can still
    find it, but it will not be inside originals/."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


#: Steps about folders rather than files: not counted as files, never checked
#: for changes (a folder's own time moves whenever its contents do).
_FOLDER_KINDS = frozenset({"mkdir", "rmdir"})


def _stat(path: Path) -> Optional[list[int]]:
    """``[size, mtime_ns]`` of a file, or ``None`` when there is none (or it
    is a folder: a folder renamed as a whole changes whenever its contents
    do, which is not a change to what the step did)."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    if Path(path).is_dir():
        return None
    return [st.st_size, st.st_mtime_ns]


def _append(root: Path, name: Path, record: dict[str, Any]) -> None:
    log_path = Path(root) / name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _rewrite(root: Path, name: Path, records: list[dict[str, Any]]) -> None:
    log_path = Path(root) / name
    if not records:
        log_path.unlink(missing_ok=True)
        return
    log_path.write_text("".join(json.dumps(r) + "\n" for r in records),
                        encoding="utf-8")


def _clear_redo(root: Path) -> None:
    """Forget what was undone, with the copies kept to redo it."""
    for record in _read(Path(root) / REDO_LOG):
        shutil.rmtree(Path(root) / ORIGINALS_DIR / record["op_id"], ignore_errors=True)
    (Path(root) / REDO_LOG).unlink(missing_ok=True)


def _read(path: Path) -> list[dict[str, Any]]:
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
            log.debug("skipping unreadable %s line", path.name)
    return out


def read_log(root: Path) -> list[dict[str, Any]]:
    """Every recorded operation, oldest first. Unreadable lines are skipped."""
    return _read(Path(root) / OPERATIONS_LOG)


def _last(path: Path) -> Optional[dict[str, Any]]:
    """The newest record of a log, read from its end: a long history must
    not be parsed whole every time a toolbar asks what Undo would do."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            chunk, pos, data = 65536, end, b""
            while pos > 0:
                step = min(chunk, pos)
                pos -= step
                fh.seek(pos)
                data = fh.read(step) + data
                lines = [ln for ln in data.split(b"\n") if ln.strip()]
                if len(lines) > 1 or (lines and pos == 0):
                    return json.loads(lines[-1])
    except (OSError, ValueError):
        return None
    return None


def _restore(root: Path, op_id: str, children: list[ChildStep]) -> int:
    """Put ``children`` back the way they were. Returns files restored.

    For rolling back an operation that failed part way: it never reached the
    log, so nothing needs to be kept for a redo."""
    originals = Path(root) / ORIGINALS_DIR / op_id
    n = 0
    for step in children:
        target = Path(root) / step.path
        if step.kind == "mkdir":
            try:
                target.rmdir()          # only if it is empty again
            except OSError:
                pass
            continue
        if step.kind == "rmdir":
            target.mkdir(parents=True, exist_ok=True)
            continue
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


def _children(record: dict[str, Any]) -> list[ChildStep]:
    return [
        ChildStep(kind=c["kind"], path=c["path"], existed=c["existed"],
                  origin=c.get("origin", ""))
        for c in record.get("children", [])
    ]


def _reverse(root: Path, record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Put back what ``record`` changed, as an operation of its own, and
    return that operation's record (``None`` if nothing needed changing).

    Going through :class:`Operation` keeps the state being replaced, which is
    what makes the undo itself undoable (a redo), and makes the reversal all
    or nothing like any other operation. Restored files get back their old
    modification time, so the step before reads them as untouched."""
    root = Path(root)
    originals = root / ORIGINALS_DIR / record["op_id"]
    op = Operation(root=root, op_id=uuid.uuid4().hex[:12], label=record["label"])
    op.originals_dir.mkdir(parents=True, exist_ok=True)
    vacated: list[Path] = []
    try:
        for step in reversed(_children(record)):
            target = root / step.path
            if step.kind == "mkdir":
                # Created by the step: it goes if nothing has been put in it
                # since. Folders were added deepest last, so they come back
                # here deepest first.
                op.remove_if_empty(target)
            elif step.kind == "rmdir":
                op._ensure_dir(target)
            elif step.kind == "rename":
                origin = root / step.origin
                if target.exists() and not origin.exists():
                    op.rename(target, origin)
                    vacated.append(target.parent)
            elif not step.existed:
                op.delete(target)
                vacated.append(target.parent)
            else:
                src = originals / step.path
                if not src.is_file():
                    log.warning("no original kept for %s in %s", step.path, record["op_id"])
                    continue
                op.replace_from(target, src)
                shutil.copystat(src, target)
        if not record.get("dirs"):
            # Recorded before folders were: the folders this reversal emptied
            # are the ones the step had created, so they go, recorded, as a
            # newer record's would.
            _drop_vacated(op, root, vacated)
    except Exception:
        op.rollback()
        raise
    op._committed = True
    if not op.children:
        shutil.rmtree(op.originals_dir, ignore_errors=True)
        return None
    return op.record()


def _drop_vacated(op: Operation, root: Path, folders: list[Path]) -> None:
    """Remove each folder left empty, and each parent it empties, inside the
    dataset and outside the tool's own state."""
    stop = {Path(root), Path(root) / EDITOR_STATE_DIR.parts[0]}
    for folder in sorted(dict.fromkeys(folders), key=lambda p: len(p.parts), reverse=True):
        current = folder
        while current not in stop and Path(root) in current.parents:
            if not op.remove_if_empty(current):
                break
            current = current.parent


@dataclass(frozen=True)
class HistoryStep:
    """One step of undo or redo, as the user is shown it."""

    label: str
    at: str
    #: Every file the step touched, relative to the dataset root.
    files: tuple[str, ...]
    #: Of those, the ones changed since the step (by hand, by another
    #: program): undoing it replaces those changes too. Filled only when
    #: asked for, because it reads every file's state.
    changed_since: tuple[str, ...] = ()


def _step(root: Path, record: Optional[dict[str, Any]], check: bool) -> Optional[HistoryStep]:
    if record is None:
        return None
    files: list[str] = []
    for c in record.get("children", []):
        if c.get("kind") in _FOLDER_KINDS:
            continue
        for rel in (c.get("path", ""), c.get("origin", "")):
            if rel and rel not in files:
                files.append(rel)
    changed: tuple[str, ...] = ()
    if check and isinstance(record.get("after"), dict):
        changed = tuple(rel for rel, state in record["after"].items()
                        if _stat(Path(root) / rel) != state)
    return HistoryStep(label=record.get("label", ""), at=record.get("at", ""),
                       files=tuple(files), changed_since=changed)


def peek_undo(root: Path, *, check: bool = False) -> Optional[HistoryStep]:
    """What Undo would put back, or ``None``. ``check`` also looks for files
    changed since."""
    root = Path(root)
    return _step(root, _last(root / OPERATIONS_LOG), check)


def peek_redo(root: Path, *, check: bool = False) -> Optional[HistoryStep]:
    """What Redo would bring back, or ``None``."""
    root = Path(root)
    return _step(root, _last(root / REDO_LOG), check)


def history(root: Path, which: str = "undo") -> list[HistoryStep]:
    """Every step Undo (or Redo) would take, newest first: what a
    right-click on the button lists."""
    root = Path(root)
    records = _read(root / (OPERATIONS_LOG if which == "undo" else REDO_LOG))
    return [s for s in (_step(root, r, False) for r in reversed(records)) if s is not None]


def only_touches(step: Optional[HistoryStep], root: Path, path: Path) -> bool:
    """True when ``step`` changed ``path`` and nothing else: a pane showing
    that file may undo or redo it as its own."""
    return step is not None and step.files == (relative(root, path),)


def undo_last(root: Path) -> Optional[dict[str, Any]]:
    """Reverse the most recent operation. Returns its record, or ``None``.

    Every file it touched goes back together: one step, however many files.
    The record leaves the log, so undo can be pressed repeatedly to walk back
    through a session, and what the undo replaced is kept for
    :func:`redo_last` until a new change is made.
    """
    root = Path(root)
    records = read_log(root)
    if not records:
        return None
    last = records[-1]
    reversal = _reverse(root, last)
    _rewrite(root, OPERATIONS_LOG, records[:-1])
    shutil.rmtree(root / ORIGINALS_DIR / last["op_id"], ignore_errors=True)
    if reversal is not None:
        _append(root, REDO_LOG, reversal)
    return last


def redo_last(root: Path) -> Optional[dict[str, Any]]:
    """Bring back the most recently undone operation. Returns its record, or
    ``None``. It goes back on the log, so it can be undone again."""
    root = Path(root)
    records = _read(root / REDO_LOG)
    if not records:
        return None
    last = records[-1]
    again = _reverse(root, last)
    _rewrite(root, REDO_LOG, records[:-1])
    shutil.rmtree(root / ORIGINALS_DIR / last["op_id"], ignore_errors=True)
    if again is not None:
        _append(root, OPERATIONS_LOG, again)
    return last


__all__ = [
    "ChildStep",
    "EDITOR_STATE_DIR",
    "HistoryStep",
    "OS_JUNK",
    "Operation",
    "OperationError",
    "begin_operation",
    "forget_history",
    "history",
    "is_hollow",
    "is_os_junk",
    "only_touches",
    "peek_redo",
    "peek_undo",
    "read_log",
    "redo_last",
    "relative",
    "run_as_operation",
    "undo_last",
]

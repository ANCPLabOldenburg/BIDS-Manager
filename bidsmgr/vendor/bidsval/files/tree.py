"""A lazy, indexed view of a dataset's files.

The tree is built once by scanning the dataset root. Hidden files and
directories (anything whose name starts with ``.`` - ``.git``, ``.bidsval``,
``.DS_Store`` ...) are skipped: they are bookkeeping, not BIDS content. File
contents are read on demand, never eagerly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Reserved top-level directories that are not part of the validated dataset:
# raw source data, analysis code, and derivative datasets (each validated on its
# own, not as part of the parent). Matches the reference validator.
_RESERVED_TOP_DIRS = frozenset({"sourcedata", "code", "derivatives"})


@dataclass(frozen=True)
class BIDSFile:
    """One file in the dataset, addressed by its path relative to the root."""

    relpath: str  # POSIX, relative to the dataset root, no leading slash
    abspath: Path

    @property
    def name(self) -> str:
        return self.abspath.name

    @property
    def parent(self) -> str:
        """The POSIX relative path of the containing directory (``""`` at root)."""
        return self.relpath.rsplit("/", 1)[0] if "/" in self.relpath else ""

    @property
    def is_symlink(self) -> bool:
        """Whether this entry is a symlink (e.g. an unfetched git-annex file)."""
        return self.abspath.is_symlink()

    def size(self) -> int:
        try:
            return self.abspath.stat().st_size
        except OSError:
            return 0

    def read_text(self) -> str:
        return self.abspath.read_text(encoding="utf-8")

    def read_bytes(self, length: int | None = None) -> bytes:
        with self.abspath.open("rb") as handle:
            return handle.read() if length is None else handle.read(length)


class FileTree:
    """An indexed view of every (non-hidden) file under a dataset root."""

    def __init__(self, root: str | Path, directory_recordings: tuple[str, ...] = ()) -> None:
        self.root = Path(root)
        # Extensions of directory-based recordings (``.ds``, ``.mefd`` ...). Such a
        # directory is indexed as ONE entry; its internal files are not validated.
        self._recording_exts = tuple(e for e in directory_recordings if e)
        self._index: dict[str, BIDSFile] = {}
        # Directory paths, so a reference to a directory recording resolves as existing.
        self._dirs: set[str] = set()
        # Files grouped by parent directory, built once for O(1) per-dir lookups.
        self._by_dir: dict[str, list[BIDSFile]] = {}
        if self.root.is_dir():
            for path in self.root.rglob("*"):
                # Include regular files and symlinks: an unfetched git-annex file
                # (a symlink) still exists as part of the dataset structure.
                if not (path.is_file() or path.is_symlink()):
                    continue
                relpath = path.relative_to(self.root).as_posix()
                parts = relpath.split("/")
                if any(part.startswith(".") for part in parts):
                    continue  # skip hidden files / anything under a hidden dir
                if parts[0] in _RESERVED_TOP_DIRS:
                    continue  # not validated as part of the dataset
                recording = self._enclosing_recording(parts)
                if recording is not None:
                    self._add(recording)  # index the recording dir once; skip its contents
                    continue
                self._add(relpath)

    def _enclosing_recording(self, parts: list[str]) -> str | None:
        """The relpath of the nearest ancestor that is a directory recording, if any."""
        for depth, name in enumerate(parts):
            if any(name.endswith(ext) for ext in self._recording_exts):
                return "/".join(parts[: depth + 1])
        return None

    def _add(self, relpath: str) -> None:
        if relpath in self._index:
            return
        bids_file = BIDSFile(relpath, self.root / relpath)
        self._index[relpath] = bids_file
        self._by_dir.setdefault(bids_file.parent, []).append(bids_file)
        parts = relpath.split("/")
        for depth in range(1, len(parts)):
            self._dirs.add("/".join(parts[:depth]))

    def files(self) -> list[BIDSFile]:
        """Every indexed file, sorted by path for stable output."""
        return [self._index[key] for key in sorted(self._index)]

    def get(self, relpath: str) -> BIDSFile | None:
        return self._index.get(relpath)

    def exists(self, relpath: str) -> bool:
        # Resolve purely from the index (like the reference's tree-only lookup):
        # no filesystem fallback, so hidden/ignored files and ``..`` traversal
        # cannot spuriously resolve. Directories (e.g. ``.ds`` recordings) count.
        return relpath in self._index or relpath in self._dirs

    def subjects(self) -> list[str]:
        """Top-level ``sub-*`` directory names, sorted."""
        names = set()
        for relpath in self._index:
            head = relpath.split("/", 1)[0]
            if head.startswith("sub-"):
                names.add(head)
        return sorted(names)

    def files_in(self, dir_relpath: str) -> list[BIDSFile]:
        """Files directly inside ``dir_relpath`` (``""`` is the root), sorted."""
        return sorted(self._by_dir.get(dir_relpath, []), key=lambda f: f.relpath)

    def json_sidecars_in(self, dir_relpath: str) -> list[BIDSFile]:
        """The ``.json`` files directly inside ``dir_relpath`` (``""`` is root)."""
        return [f for f in self.files_in(dir_relpath) if f.name.endswith(".json")]

    def ancestor_dirs(self, relpath: str) -> list[str]:
        """Directory paths from a file's own directory up to the root.

        Ordered most-specific first (the file's directory) to least-specific
        (the root, ``""``). Used to apply the BIDS inheritance principle.
        """
        dirs: list[str] = []
        current = relpath.rsplit("/", 1)[0] if "/" in relpath else ""
        while True:
            dirs.append(current)
            if current == "":
                break
            current = current.rsplit("/", 1)[0] if "/" in current else ""
        return dirs

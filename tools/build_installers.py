#!/usr/bin/env python3
"""Build ``Installers/Installers.zip`` from the installer sources.

The scripts under ``Installers/`` are the source and are tracked as they
are, so every change to them is a readable diff. The zip is what the website
links to, and it is rebuilt from them after every change.

It is built the same way every time: entries in sorted order, one fixed
timestamp, Unix permissions so the scripts are still executable once
unzipped (``.command`` files are double-clicked), and no macOS metadata
(``__MACOSX/``, ``._*``, ``.DS_Store``). ``--check`` compares contents rather
than bytes, because two zlib builds can compress the same file differently.

    python tools/build_installers.py           write the zip
    python tools/build_installers.py --check   exit 1 if the zip is out of date
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Installers"
ZIP = SOURCE / "Installers.zip"
STAMP = (2026, 1, 1, 0, 0, 0)
EXECUTABLE = {".command", ".sh", ".bat"}


def members(source: Path = SOURCE) -> list[Path]:
    """The files that go into the zip, in the order they go in."""
    return sorted(
        path for path in source.rglob("*")
        if path.is_file()
        and path.name not in {".DS_Store", ZIP.name}
        and not path.name.startswith("._")
    )


def mode_of(path: Path) -> int:
    return 0o755 if path.suffix in EXECUTABLE else 0o644


def expected(source: Path = SOURCE) -> dict[str, tuple[bytes, int]]:
    """Name in the zip -> (contents, permissions), from the sources."""
    return {
        "Installers/" + path.relative_to(source).as_posix(): (path.read_bytes(), mode_of(path))
        for path in members(source)
    }


def build(source: Path = SOURCE) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, (data, mode) in expected(source).items():
            info = zipfile.ZipInfo(name, date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # Unix, so the permissions below are honoured
            info.external_attr = (0o100000 | mode) << 16
            archive.writestr(info, data, compresslevel=9)
    return buffer.getvalue()


def contents(zip_path: Path = ZIP) -> dict[str, tuple[bytes, int]]:
    """Name -> (contents, permissions), from a built zip."""
    with zipfile.ZipFile(zip_path) as archive:
        return {
            info.filename: (archive.read(info), (info.external_attr >> 16) & 0o777)
            for info in archive.infolist()
            if not info.is_dir()
        }


def main(argv: list[str]) -> int:
    if "--check" in argv:
        if not ZIP.exists() or contents(ZIP) != expected():
            print(f"{ZIP.relative_to(ROOT)} is out of date: run python tools/build_installers.py")
            return 1
        print(f"{ZIP.relative_to(ROOT)} matches the sources")
        return 0
    ZIP.write_bytes(build())
    for name in expected():
        print(f"  {name}")
    print(f"wrote {ZIP.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

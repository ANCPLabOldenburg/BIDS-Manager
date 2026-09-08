"""Cached loader for the BIDS schema, at whichever version is in use.

Reference: architecture.md §3.

``version`` used to be a parameter that was accepted and ignored: the loader
always returned whatever ``bidsschematools`` shipped with the install. So the
setting that lets a user validate against a different BIDS version changed the
validator's answers and nothing else, and every form, table and fixup went on
describing the installed version. A dataset could be validated against 1.9 while
being filled in against 1.11.

It is honoured now, through ``bidsval``, which bundles several versions and is
already the one place BIDS is interpreted. The namespace it returns is the same
``bidsschematools`` type this module always returned, so nothing downstream
changes shape.

:func:`set_active_version` is how a process says which version it means. One
process, one answer: the alternative is a version parameter on every schema
question, which drifts the moment a caller forgets it, and drift between what a
tool fills and what it checks is exactly what these modules exist to prevent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from bidsval import schema as bidsval_schema
from bidsschematools.types.namespace import Namespace

_BUNDLED_DIR = Path(__file__).parent / "bundled"

# The version every schema question in this process is asked of. ``None`` means
# bidsval's own default, which is the newest it bundles.
_active_version: Optional[str] = None


def set_active_version(version: Optional[str]) -> None:
    """Choose the BIDS version this process speaks. Empty means the default.

    Every cache keyed on schema content is dropped, because they were answers
    about the old version. Callers are the GUI (from its Validation setting) and
    the CLI verbs (from their own flag).
    """
    global _active_version
    version = (version or "").strip() or None
    if version == _active_version:
        return
    _active_version = version
    invalidate_caches()


def active_version() -> Optional[str]:
    """The version in force, or ``None`` for the default."""
    return _active_version


def available_versions() -> list[str]:
    """Every BIDS version that can be selected."""
    try:
        return sorted(bidsval_schema.available_versions())
    except Exception:  # noqa: BLE001 - an odd install is not worth a crash here
        return []


@lru_cache(maxsize=8)
def _load(version: Optional[str]) -> Namespace:
    return bidsval_schema.resolve(version)


def get_schema(version: Optional[str] = None) -> Namespace:
    """The BIDS schema namespace, at ``version`` or the active one."""
    return _load(version if version is not None else _active_version)


def invalidate_caches() -> None:
    """Drop every memoised schema answer in the package.

    Lives here rather than in the engine so that switching version is one call
    and cannot half-happen: the loader knows nothing about which caches exist,
    so the engine registers them.
    """
    _load.cache_clear()
    for clear in _CACHE_CLEARERS:
        clear()
    try:
        bidsval_schema.cache.clear()
    except Exception:  # noqa: BLE001
        pass


_CACHE_CLEARERS: list = []


def register_cache(cached) -> None:
    """Have ``cached``'s memoisation dropped whenever the version changes."""
    _CACHE_CLEARERS.append(cached.cache_clear)


def schema_version(schema: Optional[Namespace] = None) -> str:
    schema = schema or get_schema()
    return str(schema.schema_version)


def bids_version(schema: Optional[Namespace] = None) -> str:
    schema = schema or get_schema()
    return str(schema.bids_version)


__all__ = [
    "active_version",
    "available_versions",
    "bids_version",
    "get_schema",
    "invalidate_caches",
    "register_cache",
    "schema_version",
    "set_active_version",
]

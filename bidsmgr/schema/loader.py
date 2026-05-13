"""Cached loader for the BIDS schema (``bidsschematools``).

Reference: architecture.md §3.

The schema is loaded once per process and cached. Schema versions are pinned
per project (architecture.md §3, §15.4); the loader currently uses whatever
``bidsschematools`` ships with the install. A bundled snapshot directory exists
at ``bundled/`` for offline/version-pinned use; populate it later when we wire
up per-project schema pinning.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from bidsschematools.schema import load_schema as _load_schema
from bidsschematools.types.namespace import Namespace

_BUNDLED_DIR = Path(__file__).parent / "bundled"


@lru_cache(maxsize=4)
def get_schema(version: Optional[str] = None) -> Namespace:
    """Return the cached BIDS schema namespace.

    Parameters
    ----------
    version
        Schema version to load (e.g. ``"1.10.0"``). Currently ignored; we
        always load the version that ships with the installed
        ``bidsschematools``. Will be honoured against ``bundled/`` once the
        per-project pinning policy lands.
    """

    return _load_schema()


def schema_version(schema: Optional[Namespace] = None) -> str:
    schema = schema or get_schema()
    return str(schema.schema_version)


def bids_version(schema: Optional[Namespace] = None) -> str:
    schema = schema or get_schema()
    return str(schema.bids_version)


__all__ = ["get_schema", "schema_version", "bids_version"]

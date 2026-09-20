"""Resolve a schema *selector* to one in-memory schema object.

Every other module receives a resolved schema and reads vocabulary and rules from
it, so nothing else branches on the BIDS version. This module is the only one
that decides *which* schema that is.

A selector is one of:

* ``None`` - the schema bundled in this package (the default, :data:`DEFAULT_VERSION`).
  Bundling the schema pins it to a known BIDS version regardless of which
  ``bidsschematools`` is installed; only the parser comes from ``bidsschematools``.
* a bundled version tag (a BIDS version such as ``"1.11.1"``) - loaded from the
  package, offline. :func:`available_versions` lists what is bundled.
* a local path - a dereferenced ``schema.json`` file or a schema source
  directory, for a custom or forked schema.
* a version tag that is not bundled, ``"latest"``, or a URL - fetched and cached.
  This lands behind the same interface (see the project plan); until then it
  raises :class:`SchemaNotAvailable` with a clear message.

The returned object is a ``bidsschematools`` ``Namespace``. Loads are cached, so
resolving the same selector repeatedly is free.
"""

from __future__ import annotations

import importlib.resources as resources
from functools import lru_cache
from pathlib import Path

from bidsschematools.schema import load_schema
from bidsschematools.types.namespace import Namespace

from . import cache

SchemaSelector = str | Path | None

#: The BIDS version used when no selector is given. Must match a bundled file.
DEFAULT_VERSION = "1.11.1"


class SchemaNotAvailable(Exception):
    """Raised when a requested schema cannot be located or loaded."""


def resolve(selector: SchemaSelector = None) -> Namespace:
    """Return the schema named by ``selector`` as a ``Namespace``.

    See the module docstring for the accepted selector forms.
    """
    if selector is None:
        selector = DEFAULT_VERSION
    sel = str(selector)

    # A URL: fetch and cache.
    if cache.is_url(sel):
        return _load(_fetched(sel))

    # A local schema.json file or source directory.
    path = Path(selector)
    if path.exists():
        # Normalise so a str and an equivalent Path share one cached load.
        return _load(str(path.resolve()))

    # A version tag bundled in the package (offline, no network).
    bundled = _bundled_file(sel)
    if bundled is not None:
        return _load(str(bundled))

    # A published version ("latest" or "X.Y.Z" not bundled): fetch and cache.
    url = cache.published_url(sel)
    if url is not None:
        return _load(_fetched(url))

    raise SchemaNotAvailable(
        f"schema {selector!r} is not a bundled version, a local path, a published "
        f"version, or a URL. Bundled versions: {', '.join(available_versions()) or 'none'}."
    )


def _fetched(url: str) -> str:
    try:
        return str(cache.fetch(url))
    except Exception as error:  # network / TLS / HTTP
        raise SchemaNotAvailable(f"could not fetch schema from {url}: {error}") from error


def available_versions() -> list[str]:
    """The BIDS versions bundled in the package, sorted."""
    return sorted(p.stem for p in _bundled_dir().glob("*.json"))


@lru_cache(maxsize=16)
def _load(path_key: str) -> Namespace:
    """Load and cache a schema from a filesystem path.

    ``load_schema`` accepts both a dereferenced ``schema.json`` file and a schema
    source directory, picking the loader by inspecting the path, so this single
    call covers bundled, local, and forked schemas.
    """
    return load_schema(Path(path_key))


def _bundled_dir() -> Path:
    """Filesystem directory holding the bundled schema files.

    VENDORED CHANGE. Upstream names the package absolutely, as
    ``resources.files("bidsval.schema")``. Inside this tree the package is
    ``bidsmgr.vendor.bidsval.schema``, so that spelling imports the SEPARATELY
    INSTALLED bidsval if there is one and raises ``ModuleNotFoundError`` if
    there is not. Both outcomes are wrong, and the first hides the second:
    on a developer machine with bidsval pip-installed the vendored copy
    silently read the installed package's schemas and looked fine, while a
    clean Linux or Windows machine could not import BIDS Manager at all.

    ``__package__`` is this module's own package whatever the tree is called,
    which is the whole point of vendoring.
    """
    return Path(str(resources.files(__package__))) / "bundled"


def _bundled_file(version: str) -> Path | None:
    """The bundled schema file for ``version``, or ``None`` if not bundled."""
    candidate = _bundled_dir() / f"{version}.json"
    return candidate if candidate.is_file() else None


def schema_version(schema: Namespace | None = None) -> str:
    """The schema's own structural version (the ``SCHEMA_VERSION`` axis)."""
    return str((schema or resolve()).schema_version)


def bids_version(schema: Namespace | None = None) -> str:
    """The BIDS specification version the schema describes (the ``BIDS_VERSION`` axis)."""
    return str((schema or resolve()).bids_version)


__all__ = [
    "resolve",
    "available_versions",
    "schema_version",
    "bids_version",
    "DEFAULT_VERSION",
    "SchemaNotAvailable",
    "SchemaSelector",
]

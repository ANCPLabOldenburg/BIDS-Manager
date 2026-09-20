"""Fetch and cache schema.json from a URL or a published BIDS version.

Lets the validator use a schema that is not bundled - a specific published
version (``--schema 1.7.0``), the development tip (``--schema latest``), or any
URL (``--schema https://.../schema.json``) - while caching each fetch so repeated
runs and offline reuse are free.
"""

from __future__ import annotations

import hashlib

# Where fetched schemas are cached. Kept simple (no extra dependency); honours
# XDG_CACHE_HOME when set.
import os
import re
import ssl
import urllib.request
from pathlib import Path

_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "bidsval" / "schemas"

# The BIDS specification publishes a dereferenced schema.json per version.
_PUBLISHED = "https://bids-specification.readthedocs.io/en/{ref}/schema.json"
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def is_url(selector: str) -> bool:
    return selector.startswith(("http://", "https://"))


def published_url(selector: str) -> str | None:
    """The canonical schema.json URL for a version tag or ``"latest"``."""
    if selector == "latest":
        return _PUBLISHED.format(ref="latest")
    if _VERSION_RE.match(selector):
        return _PUBLISHED.format(ref=f"v{selector}")
    return None


def fetch(url: str) -> Path:
    """Download ``url`` to the cache (once) and return the local file path.

    Raises the underlying exception (network / TLS / HTTP error) on failure, so
    callers can turn it into a clear message.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    dest = _CACHE_DIR / f"{key}.json"
    if dest.is_file():
        return dest
    request = urllib.request.Request(url, headers={"User-Agent": "bidsval"})
    with urllib.request.urlopen(request, timeout=30, context=_ssl_context()) as response:
        data = response.read()
    dest.write_bytes(data)
    return dest


def _ssl_context() -> ssl.SSLContext:
    # Use certifi's CA bundle when available (avoids the common macOS "unable to
    # get local issuer certificate" failure), else the system default.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover - certifi normally present
        return ssl.create_default_context()

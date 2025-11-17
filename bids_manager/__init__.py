"""BIDS Manager package."""

from importlib import metadata

__all__ = ["__version__"]

try:  # pragma: no cover - version resolution
    __version__ = metadata.version("bids-manager")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


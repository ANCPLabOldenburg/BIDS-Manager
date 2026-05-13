"""Pluggable converter backends. Each consumes a ``ConvertTask`` and
produces files in a staging directory.

Reference: architecture.md §7. The schema engine builds the basename;
backends just produce the file at that path. **Backends never decide
BIDS names.**

Default backend: ``dcm2niix_direct`` (MRI). dcm2bids and heudiconv ship
as optional plugins later.
"""

from .registry import ConverterBackend, default_backends, dispatch, select_backend
from .types import ConvertResult, ConvertTask

__all__ = [
    "ConverterBackend", "ConvertResult", "ConvertTask",
    "default_backends", "dispatch", "select_backend",
]

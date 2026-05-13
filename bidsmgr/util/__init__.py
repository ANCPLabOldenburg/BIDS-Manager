"""Small cross-cutting utilities.

Keep this subpackage minimal — only things shared by ``cli``,
``converter``, ``gui``, ``workers``. Domain logic belongs in the
modality-specific packages, not here.
"""

from .paths import (
    WINDOWS_RESERVED_CHARS,
    WINDOWS_RESERVED_NAMES,
    long_path,
    safe_path_component,
)

__all__ = [
    "WINDOWS_RESERVED_CHARS",
    "WINDOWS_RESERVED_NAMES",
    "long_path",
    "safe_path_component",
]

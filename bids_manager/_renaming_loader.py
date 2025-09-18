"""Dynamic loader for the bundled renaming helpers.

This project keeps a ``bids_manager/renaming`` folder in the source tree for
organizational purposes—the raw helper modules live there alongside the
packaged schema files.  The folder intentionally lacks an ``__init__`` module
so it is *not* treated as a standalone Python package; this avoids having to
list additional packages in ``pyproject.toml`` while still keeping the code
grouped together on disk.

To make the helpers importable we manufacture a lightweight module object at
runtime and register the implementation modules under the historical import
paths (``bids_manager.renaming.config`` and
``bids_manager.renaming.schema_renamer``).  The helpers then behave exactly
like regular modules even though the on-disk layout is just a plain folder.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

_PACKAGE_NAME = "bids_manager.renaming"
_RENAMING_DIR = Path(__file__).resolve().parent / "renaming"

# Cache loaded modules so repeated imports share the same module instances.
_MODULE_CACHE: Dict[str, ModuleType] = {}


def _load_from_disk(module_name: str) -> ModuleType:
    """Load and return the renaming helper ``module_name``.

    Parameters
    ----------
    module_name:
        Stem of the ``.py`` file inside :mod:`bids_manager.renaming` that should
        be loaded, e.g. ``"config"`` or ``"schema_renamer"``.
    """

    if module_name in _MODULE_CACHE:
        return _MODULE_CACHE[module_name]

    module_path = _RENAMING_DIR / f"{module_name}.py"
    if not module_path.exists():  # pragma: no cover - defensive safety check
        raise ImportError(
            f"Cannot import {module_name!r} from {_PACKAGE_NAME}:"
            f" expected file {module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{module_name}", module_path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        raise ImportError(f"Unable to load module specification for {module_path}")

    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None

    # Register the module before executing it so decorators such as
    # :func:`dataclasses.dataclass` can resolve ``cls.__module__`` during class
    # creation.
    sys.modules[f"{_PACKAGE_NAME}.{module_name}"] = module
    loader.exec_module(module)
    _MODULE_CACHE[module_name] = module

    # Attach the module as an attribute of the synthetic ``bids_manager.renaming``
    # module so ``from bids_manager.renaming import schema_renamer`` works.
    package = ensure_package()
    setattr(package, module_name, module)

    return module


def ensure_package() -> ModuleType:
    """Return the synthetic :mod:`bids_manager.renaming` module.

    The object behaves like a package for import purposes but is created
    dynamically so the on-disk ``renaming`` folder can remain a plain
    directory.
    """

    package = sys.modules.get(_PACKAGE_NAME)
    if package is None:
        package = ModuleType(_PACKAGE_NAME)
        package.__file__ = str(_RENAMING_DIR)
        # Setting ``__path__`` allows ``importlib`` to treat the module like a
        # package when resolving dotted imports without requiring an
        # ``__init__`` file in the folder.
        package.__path__ = [str(_RENAMING_DIR)]  # type: ignore[attr-defined]
        package.__package__ = _PACKAGE_NAME

        def _lazy_attr(name: str) -> ModuleType:
            return _load_from_disk(name)

        package.__getattr__ = _lazy_attr  # type: ignore[attr-defined]
        # ``__all__`` advertises the known helper modules, aiding tools like
        # auto-completion without eagerly importing everything.
        package.__all__ = tuple(p.stem for p in _RENAMING_DIR.glob("*.py"))

        sys.modules[_PACKAGE_NAME] = package

    return package


def load_module(module_name: str) -> ModuleType:
    """Public helper used by wrappers to load renaming modules."""

    ensure_package()
    return _load_from_disk(module_name)


# Eagerly register the synthetic package so simple ``import bids_manager.renaming``
# works even before any helper module is accessed.
ensure_package()


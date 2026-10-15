"""NiiVue's colour maps, as data.

Copied verbatim from ``niivue/packages/niivue/src/cmaps/*.json`` (commit
``ee9aefc``, 2026-08-18) under NiiVue's BSD 2-Clause licence (``LICENSE``
beside this file). Each file holds control points: ``R``, ``G``, ``B``,
``A`` (0-255) at intensity indices ``I`` (0-255). Some CT maps also carry a
suggested window as ``min`` / ``max`` in the data's own units (Hounsfield),
and ``_slicer3d.json`` carries ``labels``.

Nothing here is code: :mod:`bidsmgr.viz.compute.colormaps` reads the files
through ``importlib.resources``, relative to this package, so the copy that
is read is always the one that shipped.
"""

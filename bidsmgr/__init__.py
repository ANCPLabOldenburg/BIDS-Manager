"""bidsmgr — schema-driven BIDS converter, curator, and editor.

The v1.0.0 re-imagination of BIDS-Manager. Distributed on PyPI as
``bids-manager`` (``pip install bids-manager``), imported as
``bidsmgr``. The 0.x line lived under the ``bids_manager`` import
name; that package is gone — every workflow now goes through this
schema-driven engine.

The single design bet: **the BIDS schema is the engine**, not a
check at the end. Every layer — classification, naming, GUI form
generation, validation, sidecar generation, post-conversion auditing
— reads from the same machine-readable BIDS schema.

Layout:

* ``schema``     — keystone rules engine (bidsschematools wrapper)
* ``inventory``  — per-modality scanners (DICOM, EEG/MEG raw, physio)
* ``classifier`` — chained classifiers (BidsGuess, mne channel types,
  sequence dict)
* ``planner``    — entity plans + user edits, schema-validated
* ``converter``  — pluggable backends (dcm2niix-direct default)
* ``metadata``   — post-conversion sidecar engine
* ``fixups``     — fieldmap renaming, IntendedFor, derivatives moves
* ``project``    — event-sourced project files + provenance
* ``editor``     — post-conv BIDS browser + viewers (logic, no Qt)
* ``gui``        — the only Qt-coupled subtree (PyQt6, Inspector layout)
* ``workers``    — QThread bridges (no Qt logic in core modules)
* ``cli``        — CLI dispatch entry points

Nothing imports ``gui``; ``gui`` imports everything else.
"""

__version__ = "1.0.1"
__all__ = ["__version__"]
